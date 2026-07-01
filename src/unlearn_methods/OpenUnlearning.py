import sys
import os
from unlearn_methods import UnlearningMethod, register_method

def _import_open_unlearning():
    """Import data/trainer from open-unlearning/src without conflicting with local src/data."""
    ou_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'open-unlearning', 'src'))
    # Temporarily save and remove the local 'data' and 'trainer' modules
    saved_modules = {}
    for mod_name in list(sys.modules.keys()):
        if mod_name == 'data' or mod_name.startswith('data.') or mod_name == 'trainer' or mod_name.startswith('trainer.'):
            saved_modules[mod_name] = sys.modules.pop(mod_name)
    sys.path.insert(0, ou_src)
    try:
        from data import get_collators, get_data
        from trainer import load_trainer
    finally:
        sys.path.remove(ou_src)
        # Restore local modules under prefixed names, keep open-unlearning ones
        for mod_name, mod in saved_modules.items():
            sys.modules.setdefault(mod_name, mod)
    return get_collators, get_data, load_trainer

@register_method("open_unlearning")
class OpenUnlearning(UnlearningMethod):

    def unlearn(self, model, tokenizer, data):
        get_collators, get_data, load_trainer = _import_open_unlearning()

        trainer_cfg = self.config.trainer
        trainer_args = self.config.trainer.args
        collator_cfg = self.config.collator
        data_cfg = self.config.data

        collator = get_collators(collator_cfg, tokenizer=tokenizer)
        mode = "unlearn"
        template_args = self.config.model.template_args

        #NOT NICE BUT IN ORDER TO MAKE IT WORK WITH OPENUNLEARNING FRAMEWORK
        #WE LOAD THE DATASET AGAIN USING THEIR METHODS
        data = get_data(
        data_cfg, mode=mode, tokenizer=tokenizer, template_args=template_args
        )
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

            # Resolve panorama eval config from full_cfg (set by unlearn.py)
            panorama_eval_cfg = None
            if hasattr(self, 'full_cfg') and self.full_cfg is not None:
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
                save_dir=getattr(self, 'full_cfg', {}).get("paths", {}).get("output_dir", ".") if hasattr(self, 'full_cfg') else ".",
            )
            trainer.add_callback(callback)

        if trainer_args.do_train:
            trainer.train()

        return trainer.model
