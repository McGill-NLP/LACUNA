from omegaconf import DictConfig, OmegaConf
import hydra
from utils import set_seed, init_wandb_run
from model import get_model
from unlearn_methods import get_unlearning_method
from data.UnlearningData import UnlearningData
import os
import wandb

@hydra.main(version_base=None, config_path="../configs/.", config_name="config")
def main(cfg):

    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    print(OmegaConf.to_yaml(resolved_cfg))
    yaml_path = os.path.join(cfg.paths.output_dir, "exp_config.yaml")
    with open(yaml_path, "w") as f:
        f.write(OmegaConf.to_yaml(resolved_cfg))

    set_seed(cfg.seed)

    # Init wandb if configured
    if cfg.get("wandb", None) is not None:
        task_name = cfg.get("task_name", "unlearn")
        field = cfg.unlearning_data.forget.Forget.name
        run_name = f"{task_name}_{field}"
        init_wandb_run(cfg, run_name)

    model, tokenizer = get_model(cfg.unlearning.model)
    method = get_unlearning_method(cfg.unlearning)
    # Pass full config so callbacks can access eval configs
    method.full_cfg = cfg
    data = UnlearningData(cfg.unlearning_data)

    model = method.unlearn(model, tokenizer, data)

    # Save unlearned model directly to output_dir (which already includes method/field structure)
    save_dir = cfg.paths.output_dir

    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Unlearned model saved to: {save_dir}")

    if wandb.run is not None:
        wandb.finish()

if __name__ == "__main__":
    main()

