import os
import sys
import logging
from transformers import TrainerCallback, TrainerControl, TrainerState
from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator
import wandb

logger = logging.getLogger(__name__)


def _import_panorama_evaluator():
    """Import PanoramaEvaluator from open-unlearning without conflicting with local modules."""
    ou_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'open-unlearning', 'src'))
    saved_modules = {}
    # Save and temporarily remove modules that conflict with open-unlearning's imports
    for prefix in ('evals', 'data'):
        for mod_name in list(sys.modules.keys()):
            if mod_name == prefix or mod_name.startswith(prefix + '.'):
                saved_modules[mod_name] = sys.modules.pop(mod_name)
    sys.path.insert(0, ou_src)
    try:
        from evals.panorama import PanoramaEvaluator
    finally:
        sys.path.remove(ou_src)
        for mod_name, mod in saved_modules.items():
            sys.modules.setdefault(mod_name, mod)
    return PanoramaEvaluator


class EvalDashboardCallback(TrainerCallback):
    """
    TrainerCallback that runs lm_eval and panorama evaluations at intervals during training,
    logging results to wandb.
    """

    def __init__(
        self,
        lm_eval_tasks=None,
        lm_eval_interval=50,
        lm_eval_batch_size=8,
        lm_eval_limit=None,
        panorama_eval_cfg=None,
        panorama_interval=50,
        tokenizer=None,
        template_args=None,
        eval_on_start=True,
        eval_on_end=True,
        save_dir=None,
    ):
        super().__init__()
        # lm_eval config
        self.lm_eval_tasks = lm_eval_tasks or []
        self.utility_tasks = set(self.lm_eval_tasks)
        self._utility_aggregates = None  # lazily expanded on first eval
        self.lm_eval_interval = lm_eval_interval
        self.lm_eval_batch_size = lm_eval_batch_size
        self.lm_eval_limit = lm_eval_limit

        # panorama config
        self.panorama_eval_cfg = panorama_eval_cfg
        self.panorama_interval = panorama_interval
        self.tokenizer = tokenizer
        self.template_args = template_args

        self.eval_on_start = eval_on_start
        self.eval_on_end = eval_on_end
        self.save_dir = save_dir or "."

        # Initialize panorama evaluator if config provided
        self.panorama_evaluator = None
        if self.panorama_eval_cfg is not None:
            PanoramaEvaluator = _import_panorama_evaluator()
            self.panorama_evaluator = PanoramaEvaluator(self.panorama_eval_cfg)

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if self.eval_on_start:
            model = kwargs["model"]
            tokenizer = kwargs.get("tokenizer", self.tokenizer)
            print("*** [EvalDashboard] Running baseline evaluation at step 0 ***")
            self._run_all_evals(model, tokenizer, step=0, is_world_process_zero=state.is_world_process_zero)

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        step = state.global_step
        if step == 0:
            return

        model = kwargs["model"]
        tokenizer = kwargs.get("tokenizer", self.tokenizer)

        run_lm_eval = (
            self.lm_eval_tasks
            and self.lm_eval_interval > 0
            and step % self.lm_eval_interval == 0
        )
        run_panorama = (
            self.panorama_evaluator is not None
            and self.panorama_interval > 0
            and step % self.panorama_interval == 0
        )

        if run_lm_eval:
            self._run_lm_eval(model, tokenizer, step, state.is_world_process_zero)
        if run_panorama:
            self._run_panorama_eval(model, step, tokenizer, state.is_world_process_zero)

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if self.eval_on_end:
            model = kwargs["model"]
            tokenizer = kwargs.get("tokenizer", self.tokenizer)
            print(f"*** [EvalDashboard] Running final evaluation at step {state.global_step} ***")
            self._run_all_evals(model, tokenizer, step=state.global_step, is_world_process_zero=state.is_world_process_zero)

    def _run_all_evals(self, model, tokenizer, step, is_world_process_zero):
        if self.lm_eval_tasks:
            self._run_lm_eval(model, tokenizer, step, is_world_process_zero)
        if self.panorama_evaluator is not None:
            self._run_panorama_eval(model, step, tokenizer, is_world_process_zero)

    def _run_lm_eval(self, model, tokenizer, step, is_world_process_zero):
        print(f"*** [EvalDashboard] Running lm_eval at step {step} ***")
        model.eval()
        lm_model = HFLM(pretrained=model, tokenizer=tokenizer)

        results = evaluator.simple_evaluate(
            model=lm_model,
            tasks=self.lm_eval_tasks,
            batch_size=self.lm_eval_batch_size,
            limit=self.lm_eval_limit,
        )

        if is_world_process_zero:
            flat_results = self._flatten_lm_eval_results(results)
            wandb.log(flat_results | {"global_step": step}, step=step)
            print(f"  lm_eval results at step {step}: {flat_results}")

        return results

    def _run_panorama_eval(self, model, step, tokenizer, is_world_process_zero):
        print(f"*** [EvalDashboard] Running panorama eval at step {step} ***")
        step_output_dir = os.path.join(self.save_dir, f"eval_step_{step}")
        os.makedirs(step_output_dir, exist_ok=True)

        summary = self.panorama_evaluator.evaluate(
            model=model,
            output_dir=step_output_dir,
            overwrite=True,
            tokenizer=tokenizer or self.tokenizer,
            template_args=self.template_args,
        )

        if is_world_process_zero:
            prefixed = {f"panorama/{k}": v for k, v in summary.items()}
            wandb.log(prefixed | {"global_step": step}, step=step)
            print(f"  panorama results at step {step}: {prefixed}")

        return summary

    ACCURACY_METRICS = {"acc", "acc_norm", "acc,none", "acc_norm,none"}

    @staticmethod
    def _expand_utility_tasks(task_names):
        """Expand configured task names to the result-level keys lm_eval produces."""
        from lm_eval import tasks as lm_tasks
        tm = lm_tasks.TaskManager()
        expanded = set()
        for name in task_names:
            expanded.add(name)
            task_dict = tm.load_task_or_group(name)
            if isinstance(task_dict, dict):
                expanded.update(str(k) for k in task_dict.keys())
        return expanded

    def _get_utility_aggregates(self):
        """Get the set of result-level task names to log as utility metrics (cached)."""
        if self._utility_aggregates is not None:
            return self._utility_aggregates
        expanded = self._expand_utility_tasks(self.utility_tasks)
        # Keep aggregates + direct children of group tasks, but not deep subtasks
        # e.g. ai2_arc -> keep arc_easy, arc_challenge; mmlu -> keep mmlu but not mmlu_anatomy
        aggregates = set(self.utility_tasks)
        for task in expanded - self.utility_tasks:
            if not any(task.startswith(ut + "_") for ut in self.utility_tasks):
                aggregates.add(task)
        self._utility_aggregates = aggregates
        return aggregates

    def _flatten_lm_eval_results(self, results):
        if not results.get("results"):
            return {}

        utility = self._get_utility_aggregates()
        flat = {}
        for task, metrics in results["results"].items():
            if task not in utility:
                continue
            for metric, val in metrics.items():
                if isinstance(val, (int, float)) and metric in self.ACCURACY_METRICS:
                    flat[f"utility_metrics/{task}/{metric}"] = val
        return flat
