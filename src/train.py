from omegaconf import DictConfig, OmegaConf
import hydra
from utils import set_seed, init_wandb_run
from model import get_model
from data.custom_collator import CustomDataCollator
import os
from datasets import load_from_disk
from callbacks import LMEvalCallback
import torch
from transformers import TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from create_mask import create_masks
import datetime
import torch.distributed as dist
from custom_trainer import MaskedTrainer
from custom_trainer_fsdp import MaskedTrainerFSDP
from accelerate import Accelerator, InitProcessGroupKwargs

@hydra.main(version_base=None, config_path="../configs/.", config_name="config")
def main(cfg):
    
    accelerator = Accelerator(
        kwargs_handlers=[InitProcessGroupKwargs(timeout=datetime.timedelta(minutes=60))]
    )
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    print(OmegaConf.to_yaml(resolved_cfg))
    
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank <= 0:
        if cfg.training.report_to == 'wandb' and not cfg.get('mask', {}).get('debug', False):
            run_name = f"{cfg.task_name}_{cfg.model.name}"
            init_wandb_run(cfg, run_name)
        yaml_path = os.path.join(cfg.paths.output_dir, "train_config.yaml")
        with open(yaml_path, "w") as f:
            f.write(OmegaConf.to_yaml(resolved_cfg))
        
    set_seed(cfg.seed)
    

    model, tokenizer = get_model(cfg.model)
    model.model.embed_tokens.weight.requires_grad = False
    
    data_collator = CustomDataCollator(tokenizer, max_length=4096)
    training = load_from_disk(cfg.data.preprocessed_path)
    training = training.shuffle(seed=cfg.seed)
    
    

    callbacks = []
    if cfg.eval.enable:
        callbacks.append(LMEvalCallback(
            tasks=list(cfg.eval.tasks),
            eval_interval=-1,
            eval_on_start=True,
            eval_on_end=True,
            batch_size=cfg.eval.batch_size,
            limit=cfg.eval.limit,
        ))
    


    if cfg.mask.enable:
        if not os.path.exists(cfg.mask.path) and local_rank <= 0:
            if cfg.mask.get('custom_design', False):
                raise RuntimeError("The mask is not present!")
            print("Creating new masks...")
            masks = create_masks(
                freeze_ratio=cfg.mask.freeze_ratio,
                n_masks=cfg.mask.num_groups,
                model=model,
                seed=cfg.mask.seed,
                layers_to_skip=cfg.mask.get('layers_to_skip', []),
                components=cfg.mask.get('components', ['all']),
                exclude_components=cfg.mask.get('exclude_components', []),
                entire_components=cfg.mask.get('entire_components', False),
            )
            torch.save(masks, cfg.mask.path)
            print(f"Mask saved to {cfg.mask.path}")

        
        accelerator.wait_for_everyone()
        del accelerator
        # ----------------------

        print(f"Rank {local_rank} loading mask from {cfg.mask.path}")
        masks = torch.load(cfg.mask.path)
        if cfg.mask.debug and local_rank <= 0:
            # callbacks.append(MaskCallback())
            before_state = {k: v.detach().clone().cpu() for k, v in model.named_parameters()}

    use_ddp = cfg.model.get('ddp', True)
    trigger_names = [i for i in range(cfg.mask.num_groups)] if cfg.mask.enable else 'pii'
    freeze = masks if cfg.mask.enable else None
    resume = get_last_checkpoint(cfg.training.output_dir) if cfg.get('resume_from_checkpoint', False) and os.path.isdir(cfg.training.output_dir) else None

    if use_ddp:

        training_args_dict = cfg.training.copy()
        training_args = TrainingArguments(
            **training_args_dict,
            ddp_find_unused_parameters=False,
            ddp_backend="nccl",
        )

        trainer = MaskedTrainer(
            model=model,
            args=training_args,
            train_dataset=training,
            data_collator=data_collator,
            callbacks=callbacks,
            trigger_dataset_name=trigger_names,
            freeze_masks=freeze,
        )

        print("Ready to train! (DDP)", flush=True)
        trainer.train(resume_from_checkpoint=resume)

        if cfg.get('mask', {}).get('debug', False):
            after_state = trainer.model.state_dict()
            if local_rank <= 0:
                def compute_changed_percentage(before_state, after_state):
                    total = 0
                    changed = 0
                    for name, before_tensor in before_state.items():
                        if name not in after_state:
                            continue
                        after_tensor = after_state[name]
                        if before_tensor.shape != after_tensor.shape:
                            continue
                        diff = (before_tensor.cpu() - after_tensor.cpu()).abs() > 1e-9
                        total += diff.numel()
                        changed += diff.sum().item()
                    pct = 100 * changed / total if total > 0 else 0
                    return pct, changed, total
                pct, changed, total = compute_changed_percentage(before_state, after_state)
                print(f"Changed {changed}/{total} parameters ({pct:.4f}%).")

                # --- Mask leak check: verify all changes fall within mask unfrozen regions ---
                if masks is not None:
                    outside_mask_total = 0
                    for name, before_tensor in before_state.items():
                        if name not in after_state:
                            continue
                        after_tensor = after_state[name]
                        if before_tensor.shape != after_tensor.shape:
                            continue
                        diff = (before_tensor.cpu() - after_tensor.cpu()).abs() > 1e-9
                        if not diff.any():
                            continue
                        unfrozen_union = torch.zeros(before_tensor.numel(), dtype=torch.bool)
                        if name in masks:
                            packed = masks[name].cpu()
                            for bit_idx in range(cfg.mask.num_groups):
                                unfrozen_union |= ((packed >> bit_idx) & 1).bool()
                        leaked = diff.flatten() & ~unfrozen_union
                        n_leaked = leaked.sum().item()
                        if n_leaked > 0:
                            outside_mask_total += n_leaked
                            print(f"  WARNING: {name} has {n_leaked} changed weights outside all masks!")
                    if outside_mask_total == 0:
                        print("MASK CHECK PASS: All changed weights fall within mask unfrozen regions.")
                    else:
                        print(f"MASK CHECK FAIL: {outside_mask_total} changed weights are outside all mask unfrozen regions!")
                # --- End mask leak check ---

        if local_rank <= 0:
            trainer.save_model(cfg.paths.output_dir)
            tokenizer.save_pretrained(cfg.paths.output_dir)

    else:
        # FSDP mode — manual FSDP2 wrapping with gradient masking
        training_args_dict = cfg.training.copy()
        training_args = TrainingArguments(
            **training_args_dict,
        )

        trainer = MaskedTrainerFSDP(
            model=model,
            args=training_args,
            train_dataset=training,
            data_collator=data_collator,
            callbacks=callbacks,
            trigger_dataset_name=trigger_names,
            freeze_masks=freeze,
        )
        trainer.place_model_on_device = False  # we move to GPU during FSDP wrapping
        print("Ready to train! (FSDP)", flush=True)
        trainer.train(resume_from_checkpoint=resume)

        if cfg.get('mask', {}).get('debug', False):
            # recover_cpu_model is collective (all ranks participate in all-gather),
            # then replaces self.model with a CPU model on rank 0 only.
            trainer.recover_cpu_model()
            if local_rank <= 0:
                after_state = trainer.model.state_dict()
                def compute_changed_percentage(before_state, after_state):
                    total = 0
                    changed = 0
                    for name, before_tensor in before_state.items():
                        if name not in after_state:
                            continue
                        after_tensor = after_state[name]
                        if before_tensor.shape != after_tensor.shape:
                            continue
                        diff = (before_tensor.cpu() - after_tensor.cpu()).abs() > 1e-9
                        total += diff.numel()
                        changed += diff.sum().item()
                    pct = 100 * changed / total if total > 0 else 0
                    return pct, changed, total
                pct, changed, total = compute_changed_percentage(before_state, after_state)
                print(f"Changed {changed}/{total} parameters ({pct:.4f}%).")

                # --- Mask leak check: verify all changes fall within mask unfrozen regions ---
                if masks is not None:
                    outside_mask_total = 0
                    first_param_logged = False
                    for name, before_tensor in before_state.items():
                        if name not in after_state:
                            continue
                        after_tensor = after_state[name]
                        if before_tensor.shape != after_tensor.shape:
                            continue
                        diff = (before_tensor.cpu() - after_tensor.cpu()).abs() > 1e-9
                        if not diff.any():
                            continue
                        unfrozen_union = torch.zeros(before_tensor.numel(), dtype=torch.bool)
                        if name in masks:
                            packed = masks[name].cpu()
                            for bit_idx in range(cfg.mask.num_groups):
                                unfrozen_union |= ((packed >> bit_idx) & 1).bool()
                        # Debug: print coverage stats for first changed param
                        if not first_param_logged:
                            n_unfrozen = unfrozen_union.sum().item()
                            n_total = unfrozen_union.numel()
                            n_changed = diff.flatten().sum().item()
                            n_inside = (diff.flatten() & unfrozen_union).sum().item()
                            print(f"  [MASK_DEBUG] First param: {name}")
                            print(f"  [MASK_DEBUG]   mask_found={name in masks}, shape={list(before_tensor.shape)}")
                            print(f"  [MASK_DEBUG]   unfrozen_union: {n_unfrozen}/{n_total} ({100*n_unfrozen/n_total:.2f}%)")
                            print(f"  [MASK_DEBUG]   changed: {n_changed}, inside_masks: {n_inside}, outside: {n_changed - n_inside}")
                            first_param_logged = True
                        leaked = diff.flatten() & ~unfrozen_union
                        n_leaked = leaked.sum().item()
                        if n_leaked > 0:
                            outside_mask_total += n_leaked
                            print(f"  WARNING: {name} has {n_leaked} changed weights outside all masks!")
                    if outside_mask_total == 0:
                        print("MASK CHECK PASS: All changed weights fall within mask unfrozen regions.")
                    else:
                        print(f"MASK CHECK FAIL: {outside_mask_total} changed weights are outside all mask unfrozen regions!")
                # --- End mask leak check ---

                # Save the recovered CPU model directly (cannot call trainer.save_model
                # because recover_cpu_model replaced self.model on rank 0 only, so the
                # collective get_model_state_dict inside save_model would deadlock).
                trainer.model.save_pretrained(cfg.paths.output_dir)
                tokenizer.save_pretrained(cfg.paths.output_dir)
        else:
            trainer.save_model(cfg.paths.output_dir)
            print(f"Rank {local_rank} finished saving model")
            if local_rank <= 0:
                tokenizer.save_pretrained(cfg.paths.output_dir)

    if local_rank <= 0:
        import wandb
        if wandb.run is not None:
            wandb.finish()

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    print(f"Rank {local_rank} exiting", flush=True)
    os._exit(0)
        


if __name__ == "__main__":
    main()

