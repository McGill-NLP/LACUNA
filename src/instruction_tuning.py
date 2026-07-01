from utils import *  
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint 
from omegaconf import DictConfig, OmegaConf
from model import get_model
import hydra
from peft import LoraConfig, get_peft_model, TaskType
from data.custom_collator import InstructionDataCollator
from datasets import load_dataset
from utils import temp_seed
from data.prepare_instruction_data import prepare_instruction_data
from datasets import DatasetDict

def get_train_eval_datasets(config, tokenizer):
    
    with temp_seed(config.seed):
        try:
            train = DatasetDict.load_from_disk(os.path.join(config.instruction_tuning.data_path))['train']
            eval = DatasetDict.load_from_disk(os.path.join(config.instruction_tuning.data_path))['eval']
            print(f"Instruction tuning data directory {config.instruction_tuning.data_path} already exists. Skipping saving.")
        except:
            train, eval = prepare_instruction_data(config, tokenizer)
            os.makedirs(config.instruction_tuning.data_path, exist_ok=True)
        
            dataset_to_save = DatasetDict({
                'train': train,
                'eval': eval
            })
        
            dataset_to_save.save_to_disk(config.instruction_tuning.data_path)
            print(f"Saved instruction tuning data to {config.instruction_tuning.data_path}")
    
    return train, eval

@hydra.main(version_base=None, config_path="../configs/.", config_name="config")
def main(cfg):
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    print(OmegaConf.to_yaml(resolved_cfg))
    
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank <= 0:
        if cfg.training.report_to == 'wandb':
            run_name = f"Instruction_Tuning_{cfg.task_name}_{cfg.model.name}"
            init_wandb_run(cfg, run_name)
        yaml_path = os.path.join(cfg.paths.output_dir, "train_config.yaml")
        with open(yaml_path, "w") as f:
            f.write(OmegaConf.to_yaml(resolved_cfg))
        
    set_seed(cfg.seed)
    
    model, tokenizer = get_model(cfg.instruction_tuning)
    model.model.embed_tokens.weight.requires_grad = False
    
    # IF LORA IS ACTIVE
    if cfg.instruction_tuning.get('lora', {}).get('enable', False):
        tmp = resolved_cfg['instruction_tuning']['lora']
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
    
    
    train_dataset, eval_dataset = get_train_eval_datasets(cfg, tokenizer)
    
        
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**cfg.instruction_tuning.training_args),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    
    resume = get_last_checkpoint(cfg.instruction_tuning.training_args.output_dir) if cfg.instruction_tuning.get('resume_from_checkpoint', False) and os.path.isdir(cfg.instruction_tuning.training_args.output_dir) else None
    
    trainer.train(resume_from_checkpoint=resume)
    
    save_dir = cfg.instruction_tuning.training_args.output_dir
    
    
    # Merge LoRA weights before saving if LoRA was used
    if cfg.instruction_tuning.get('lora', {}).get('enable', False):
        if local_rank <= 0:
            print("Merging LoRA weights into base model and saving...")
            trained = trainer.model
            if hasattr(trained, 'merge_and_unload'):
                merged_model = trained.merge_and_unload()
            elif hasattr(trained, 'module') and hasattr(trained.module, 'merge_and_unload'):
                merged_model = trained.module.merge_and_unload()
            else:
                raise RuntimeError("Could not find merge_and_unload on trainer.model; ensure LoRA is enabled and PEFT is wrapping the model.")
            merged_model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
    else:
        if local_rank <= 0:
            trainer.save_model(save_dir)
            tokenizer.save_pretrained(save_dir)
        

if __name__ == "__main__":
    main()