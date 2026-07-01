from omegaconf import DictConfig, OmegaConf
import hydra
from utils import set_seed
import torch
from lacuna_model import get_model
from evals import get_evaluators
import os

@hydra.main(version_base=None, config_path="../configs/.", config_name="config")
def main(cfg):
    
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    print(OmegaConf.to_yaml(resolved_cfg))
    yaml_path = os.path.join(cfg.paths.output_dir, "exp_eval_config.yaml")
    with open(yaml_path, "w") as f:
        f.write(OmegaConf.to_yaml(resolved_cfg))
        
    template_args = cfg.unlearned_model.template_args    
    set_seed(cfg.seed)
    model, tokenizer = get_model(cfg.unlearned_model)
    # move model to GPU if available and set attribute used by eval utils
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    eval_cfgs = cfg.eval
    evaluators = get_evaluators(eval_cfgs)
    for evaluator_name, evaluator in evaluators.items():
        eval_args = {
            "template_args": template_args,
            "model": model,
            "tokenizer": tokenizer,
        }
        _ = evaluator.evaluate(**eval_args)


if __name__ == "__main__":
    main()

