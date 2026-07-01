import torch
import os
from unlearn_methods import UnlearningMethod, register_method
from unlearn_methods.OpenUnlearning import _import_open_unlearning


def _load_forget_mask(mask_path, forget_groups):
    """Load the bit-packed mask and extract the forget-only binary mask.

    The mask file contains per-parameter tensors where each element is a
    bit-packed integer.  Bit *i* being set means the weight belongs to group *i*.
    We OR together all forget-group bits and return a boolean mask per parameter
    indicating which weights are involved in *any* forget group.
    """
    packed_mask = torch.load(mask_path, map_location="cpu", weights_only=True)
    bits = 0
    for g in forget_groups:
        bits |= 1 << g
    forget_mask = {
        name: ((tensor & bits) > 0)
        for name, tensor in packed_mask.items()
    }
    return forget_mask


def _register_gradient_hooks(model, forget_mask):
    """Register backward hooks that zero out gradients outside the forget mask.

    Parameters whose name does not appear in the mask at all are fully frozen
    (requires_grad=False).  Parameters in the mask get an element-wise gradient
    mask via register_hook.
    """
    hooks = []
    # Cache list so each closure can store the moved tensor once
    cache = {}
    for name, param in model.named_parameters():
        if name in forget_mask:
            keep = forget_mask[name].view_as(param)

            def _make_hook(k, n):
                def hook(grad):
                    if n not in cache:
                        cache[n] = k.to(dtype=grad.dtype, device=grad.device)
                    return grad * cache[n]
                return hook

            hooks.append(param.register_hook(_make_hook(keep, name)))
        else:
            param.requires_grad_(False)
    return hooks


@register_method("oracle_grad")
class OracleGrad(UnlearningMethod):
    """SimNPO with oracle gradient masking.

    Only weights belonging to the forget set (as identified by the training
    mask) are allowed to receive gradient updates.  All other weights are
    frozen.  This serves as an oracle baseline that measures how well
    unlearning can work when you know *exactly* which weights encode the
    information to forget.
    """

    def unlearn(self, model, tokenizer, data):
        get_collators, get_data, load_trainer = _import_open_unlearning()

        # --- Snapshot original weights for post-training comparison ---
        original_weights = {
            name: param.detach().clone().cpu()
            for name, param in model.named_parameters()
        }

        # --- Determine forget groups and load mask ---
        full_cfg = getattr(self, "full_cfg", None)
        if full_cfg is None:
            raise ValueError("OracleGrad requires full_cfg to be set (done by unlearn.py).")

        mask_cfg = full_cfg.get("mask", {})
        mask_path = mask_cfg.get("path", None)
        if mask_path is None:
            raise ValueError(
                "OracleGrad requires a mask path.  Set mask.path in the config "
                "(it is normally set by the training experiment config)."
            )

        forget_groups = set(data.forget.unique("group_id"))
        print(f"[OracleGrad] Forget groups (from dataset): {sorted(forget_groups)}")

        print(f"[OracleGrad] Loading mask from {mask_path}")
        forget_mask = _load_forget_mask(mask_path, forget_groups)
        print(f"[OracleGrad] Registering gradient hooks...")
        hooks = _register_gradient_hooks(model, forget_mask)

        n_trainable = sum(p.requires_grad for p in model.parameters())
        n_total = sum(1 for _ in model.parameters())
        print(f"[OracleGrad] {n_trainable}/{n_total} parameters have requires_grad=True")

        # --- Delegate to SimNPO via open-unlearning ---
        trainer_cfg = self.config.trainer
        collator_cfg = self.config.collator
        data_cfg = self.config.data
        template_args = self.config.model.template_args

        collator = get_collators(collator_cfg, tokenizer=tokenizer)
        data = get_data(data_cfg, mode="unlearn", tokenizer=tokenizer, template_args=template_args)

        trainer, trainer_args = load_trainer(
            trainer_cfg=trainer_cfg,
            model=model,
            train_dataset=data.get("train", None),
            eval_dataset=None,
            tokenizer=tokenizer,
            data_collator=collator,
            evaluators=None,
            template_args=None,
        )

        # Add EvalDashboardCallback if configured
        eval_dashboard_cfg = self.config.get("eval_dashboard", {})
        if eval_dashboard_cfg.get("enabled", False):
            from callbacks.EvalDashboardCallback import EvalDashboardCallback
            lm_eval_cfg = eval_dashboard_cfg.get("lm_eval", {})
            panorama_cfg = eval_dashboard_cfg.get("panorama", {})

            panorama_eval_cfg = None
            if hasattr(self, "full_cfg") and self.full_cfg is not None:
                eval_cfgs = self.full_cfg.get("eval", {})
                panorama_eval_cfg = eval_cfgs.get("panorama", None)

            callback = EvalDashboardCallback(
                lm_eval_tasks=list(lm_eval_cfg.get("tasks", [])),
                lm_eval_interval=lm_eval_cfg.get("interval", 50),
                lm_eval_batch_size=lm_eval_cfg.get("batch_size", 8),
                lm_eval_limit=lm_eval_cfg.get("limit", None),
                panorama_eval_cfg=panorama_eval_cfg,
                panorama_interval=panorama_cfg.get("interval", 50),
                tokenizer=tokenizer,
                template_args=template_args,
                eval_on_start=eval_dashboard_cfg.get("eval_on_start", True),
                eval_on_end=eval_dashboard_cfg.get("eval_on_end", True),
                save_dir=getattr(self, "full_cfg", {}).get("paths", {}).get("output_dir", ".") if hasattr(self, "full_cfg") else ".",
            )
            trainer.add_callback(callback)

        if trainer_args.do_train:
            trainer.train()

        # Clean up hooks
        for h in hooks:
            h.remove()

        # --- Report weight change statistics ---
        total_weights = 0
        changed_weights = 0
        changed_in_mask = 0
        mask_weights = 0
        for name, param in trainer.model.named_parameters():
            orig = original_weights[name]
            current = param.detach().cpu()
            diff = (current != orig)
            n = diff.numel()
            n_changed = diff.sum().item()
            total_weights += n
            changed_weights += n_changed
            if name in forget_mask:
                fm = forget_mask[name].cpu().view_as(diff)
                mask_weights += fm.sum().item()
                changed_in_mask += (diff & fm).sum().item()
        del original_weights

        pct_changed = 100.0 * changed_weights / total_weights if total_weights else 0
        pct_mask_changed = 100.0 * changed_in_mask / mask_weights if mask_weights else 0
        leaked = changed_weights - changed_in_mask
        print(f"[OracleGrad] Weight change summary:")
        print(f"  Total weights:          {total_weights:,}")
        print(f"  Changed weights:        {changed_weights:,} ({pct_changed:.4f}%)")
        print(f"  Forget mask size:       {mask_weights:,}")
        print(f"  Changed within mask:    {changed_in_mask:,} ({pct_mask_changed:.4f}%)")
        print(f"  Changed outside mask:   {leaked:,}")

        return trainer.model
