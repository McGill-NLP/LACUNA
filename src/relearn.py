from utils import *
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from omegaconf import DictConfig, OmegaConf
from model import get_model
import hydra
from peft import LoraConfig, get_peft_model, TaskType
from data.custom_collator import InstructionDataCollator
from data.utils import format_dataset
from data.prepare_instruction_data import tokenize_instruction_data
from datasets import DatasetDict, load_dataset


@hydra.main(version_base=None, config_path="../configs/.", config_name="config")
def main(cfg):
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    print(OmegaConf.to_yaml(resolved_cfg))

    relearn_cfg = cfg.unlearning.relearning

    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    # Let the Trainer handle wandb init — just set the run name via env var
    if local_rank <= 0:
        field = cfg.unlearning_data.forget.Forget.name
        method = cfg.unlearning.get('method', 'unknown')
        os.environ["WANDB_RUN_NAME"] = f"Relearn_{field}_{method}_{cfg.model.name}"
        os.environ["WANDB_PROJECT"] = cfg.wandb.project
        os.environ["WANDB_ENTITY"] = cfg.wandb.entity

    set_seed(cfg.seed)

    # Load the unlearned model from paths.output_dir (which already points to
    # the correct unlearned model directory). We override unlearned_model.path
    # to avoid double-nesting when the default uses ${paths.output_dir}/unlearned_models/...
    from omegaconf import open_dict
    with open_dict(cfg):
        cfg.unlearned_model.path = cfg.paths.output_dir
    print(f"Loading unlearned model from {cfg.unlearned_model.path}")
    model, tokenizer = get_model(cfg.unlearned_model)
    model.model.embed_tokens.weight.requires_grad = False

    # Apply LoRA if enabled
    if relearn_cfg.get('lora', {}).get('enable', False):
        tmp = resolved_cfg['unlearning']['relearning']['lora']
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=tmp.get('r', 8),
            lora_alpha=tmp.get('alpha', 32),
            lora_dropout=tmp.get('dropout', 0.1),
            target_modules=tmp.get('target_modules', ["q_proj", "v_proj"]),
            bias=tmp.get('bias', 'none'),
            layers_to_transform=tmp.get('instruction_target', None),
        )
        model = get_peft_model(model, lora_config)

    data_collator = InstructionDataCollator(tokenizer, max_length=4096)

    relearn_data_source = relearn_cfg.get('data_source', 'instruction_tuning')

    if relearn_data_source == 'memorized':
        # Load memorized QA pairs from unseen people (relearn split)
        memorized_ds_dir = cfg.paths.memorized_datasets_dir
        print(f"Loading memorized relearn data from {memorized_ds_dir} (subset=relearn, split=train)")
        raw_ds = load_dataset(memorized_ds_dir, name='relearn', split='train')
        # Format with chat template and tokenize (same pipeline as instruction tuning)
        formatted = format_dataset(cfg.model.template_args, raw_ds, include_answer_token=True)
        tokenized = tokenize_instruction_data(formatted, tokenizer)
        # Split into train/eval
        split = tokenized.train_test_split(test_size=0.2, seed=cfg.seed)
        train_dataset = split['train']
        eval_dataset = split['test']
        print(f"Memorized relearn data — Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    else:
        # Load the original instruction tuning data
        instruction_data_path = cfg.instruction_tuning.data_path
        print(f"Loading instruction tuning data from {instruction_data_path}")
        ds = DatasetDict.load_from_disk(instruction_data_path)
        train_dataset = ds['train']
        eval_dataset = ds['eval']
        print(f"Train: {len(train_dataset)} samples, Eval: {len(eval_dataset)} samples")

    save_dir = relearn_cfg.training_args.output_dir
    os.makedirs(save_dir, exist_ok=True)

    # Save config
    if local_rank <= 0:
        yaml_path = os.path.join(save_dir, "relearn_config.yaml")
        with open(yaml_path, "w") as f:
            f.write(OmegaConf.to_yaml(resolved_cfg))

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**relearn_cfg.training_args),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    resume = (
        get_last_checkpoint(save_dir)
        if cfg.get('resume_from_checkpoint', False) and os.path.isdir(save_dir)
        else None
    )

    trainer.train(resume_from_checkpoint=resume)

    # Save the final model
    if local_rank <= 0:
        if relearn_cfg.get('lora', {}).get('enable', False):
            print("Merging LoRA weights into base model and saving...")
            trained = trainer.model
            if hasattr(trained, 'merge_and_unload'):
                merged_model = trained.merge_and_unload()
            elif hasattr(trained, 'module') and hasattr(trained.module, 'merge_and_unload'):
                merged_model = trained.module.merge_and_unload()
            else:
                raise RuntimeError(
                    "Could not find merge_and_unload on trainer.model; "
                    "ensure LoRA is enabled and PEFT is wrapping the model."
                )
            merged_model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
        else:
            trainer.save_model(save_dir)
            tokenizer.save_pretrained(save_dir)

    print(f"Relearned model saved to {save_dir}")


if __name__ == "__main__":
    main()
