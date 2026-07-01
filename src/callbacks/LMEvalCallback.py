from transformers import TrainerCallback, TrainerControl, TrainerState
from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator 
import torch
import wandb

class LMEvalCallback(TrainerCallback):
    def __init__(self, tasks, eval_interval=100, eval_on_start=False, eval_on_end=False, batch_size=8, limit=1):
        super().__init__()
        self.tasks = tasks
        self.eval_interval = eval_interval
        self.eval_on_start = eval_on_start
        self.eval_on_end = eval_on_end
        self.batch_size = batch_size
        self.limit = limit

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if self.eval_on_start:
            model = kwargs["model"]
            tokenizer = kwargs.get("tokenizer", None)
            print("*** Running initial evaluation on current weights ***")
            self.run_evaluation(model, tokenizer, step=state.global_step, is_world_process_zero=state.is_world_process_zero)

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if self.eval_interval == -1:
            return
        if state.global_step > 0 and state.global_step % self.eval_interval == 0:
            model = kwargs["model"]
            tokenizer = kwargs.get("tokenizer", None)
            print(f"*** Running evaluation at step {state.global_step} on current weights ***")
            self.run_evaluation(model, tokenizer, state.global_step, is_world_process_zero=state.is_world_process_zero)
    
    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if self.eval_on_end:
            model = kwargs["model"]
            tokenizer = kwargs.get("tokenizer", None)
            print("*** Running final evaluation on trained weights ***")
            self.run_evaluation(model, tokenizer, step=state.global_step, is_world_process_zero=state.is_world_process_zero)

    def run_evaluation(self, model, tokenizer, step, is_world_process_zero):
        """
        Run lm-eval on the current model weights.
        """
        
        
        model.eval()
        lm_model = HFLM(pretrained=model, tokenizer=tokenizer)

        results = evaluator.simple_evaluate(
            model=lm_model,
            tasks=self.tasks,
            batch_size=self.batch_size,
            limit=self.limit
        )
        #print("Evaluation results:", results)
        if is_world_process_zero:  
            flat_results = self.flatten_results(results)
            wandb.log(flat_results | {"global_step": step}, step=step)
            print(f"Logged lm-eval results to W&B at step {step}")

        return results

    @staticmethod
    def flatten_results(results):
        """
        Flatten nested lm-eval results into a dict suitable for wandb logging.
        e.g. {'hellaswag/acc': 0.78, 'piqa/acc': 0.73}
        """
        flat = {}
        if "results" in results:
            for task, metrics in results["results"].items():
                for metric, val in metrics.items():
                    if isinstance(val, (int, float)):
                        flat[f"{task}/{metric}"] = val
        return flat

    
    
