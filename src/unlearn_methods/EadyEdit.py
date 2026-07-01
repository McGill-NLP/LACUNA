from unlearn_methods import UnlearningMethod, register_method
from easyeditor.editors import BaseEditor
from easyeditor.util.hparams import HyperParams
from easyeditor.util.alg_dict import ALG_DICT
import torch
from transformers import PreTrainedTokenizer
from omegaconf import DictConfig, OmegaConf
from callbacks.EvalDashboardCallback import EvalDashboardCallback


class DynamicAlphaEditHyperParams(HyperParams):
    def __init__(self, hparams: DictConfig):
        config_dict = OmegaConf.to_container(hparams, resolve=True)
        self.__dict__.update(config_dict)

    def __repr__(self):
        params = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({params})"


class LazyEditor(BaseEditor):

    def __init__(self, hparams: HyperParams, model: torch.nn.Module, tokenizer: PreTrainedTokenizer):
        self.hparams = hparams
        self.model_name = hparams.model_name
        self.alg_name = hparams.alg_name

        self.model = model.to(f"cuda:{hparams.device}")
        self.tok = tokenizer
        if self.tok.pad_token_id is None:
            self.tok.pad_token_id = self.tok.eos_token_id
        self.tok.padding_side = "right"

        self.apply_algo = ALG_DICT[hparams.alg_name]


@register_method("easyedit")
class EasyEdit(UnlearningMethod):

    def format_dataset(self, dataset):
        tpl_args = self.config.model.template_args
        u_start = tpl_args.user_start_tag
        u_end   = tpl_args.user_end_tag
        a_start = tpl_args.asst_start_tag

        target_new = self.config.get("target_new", "I don't know.")

        prompts = []
        subjects = []
        targets = []
        ground_truths = []

        for example in dataset:
            question = example["question"]
            # Subject is "First Last" at the start of the question, before the first comma
            subject = question.split(",")[0].strip()

            prompt = f"{u_start}{question}{u_end}{a_start}"

            prompts.append(prompt)
            subjects.append(subject)
            targets.append(target_new)
            ground_truths.append(example["answer"])

        return prompts, subjects, targets, ground_truths

    def _build_eval_callback(self, tokenizer):
        """Build EvalDashboardCallback from full_cfg if eval_dashboard is enabled."""
        cfg = getattr(self, 'full_cfg', None)
        if cfg is None:
            return None
        eval_cfg = self.config.get("eval_dashboard", None)
        if eval_cfg is None or not eval_cfg.get("enabled", False):
            return None

        lm_eval_cfg = eval_cfg.get("lm_eval", {})
        eval_cfgs = cfg.get("eval", {})
        panorama_eval_cfg = eval_cfgs.get("panorama", None)
        template_args = cfg.get("model", {}).get("template_args", None)

        return EvalDashboardCallback(
            lm_eval_tasks=list(lm_eval_cfg.get("tasks", [])),
            lm_eval_batch_size=lm_eval_cfg.get("batch_size", 8),
            lm_eval_limit=lm_eval_cfg.get("limit", None),
            panorama_eval_cfg=panorama_eval_cfg,
            tokenizer=tokenizer,
            template_args=template_args,
            save_dir=cfg.paths.output_dir,
        )

    def unlearn(self, model, tokenizer, data):
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Move model to CUDA before eval (needed for flash attention)
        hparams = DynamicAlphaEditHyperParams(self.config.hparams)
        device = f"cuda:{hparams.device}"
        model = model.to(device)

        # Build eval callback
        eval_cb = self._build_eval_callback(tokenizer)
        eval_cfg = self.config.get("eval_dashboard", None)

        # Eval before unlearning
        if eval_cb is not None and eval_cfg.get("eval_on_start", False):
            print("*** [AlphaEdit] Running baseline evaluation before editing ***")
            eval_cb._run_all_evals(model, tokenizer, step=0, is_world_process_zero=True)

        editor = LazyEditor(hparams, model, tokenizer)

        prompts, subjects, targets, ground_truths = self.format_dataset(data.forget)

        print(f"[AlphaEdit] Editing {len(prompts)} requests (batch_size={hparams.batch_size})")

        _, edited_model, _ = editor.batch_edit(
            prompts, targets, ground_truths,
            subject=subjects,
            sequential_edit=True,
        )

        # Eval after unlearning
        if eval_cb is not None and eval_cfg.get("eval_on_end", False):
            print("*** [AlphaEdit] Running evaluation after editing ***")
            eval_cb._run_all_evals(edited_model, tokenizer, step=1, is_world_process_zero=True)

        return edited_model
