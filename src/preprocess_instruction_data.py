
import logging
from omegaconf import DictConfig, OmegaConf
import hydra
from data.prepare_instruction_data import prepare_instruction_data
import logging
from model import get_model
from utils import set_seed
import os
from datasets import DatasetDict

@hydra.main(version_base=None, config_path="../configs/.", config_name="config")
def main(cfg):

    logging.getLogger("httpx").setLevel(logging.WARNING)

    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    print(OmegaConf.to_yaml(resolved_cfg))
    
    set_seed(cfg.seed)
    _ , tokenizer = get_model(cfg.instruction_tuning)
    train, eval = prepare_instruction_data(cfg, tokenizer)
    
    os.makedirs(cfg.instruction_tuning.data_path, exist_ok=True)
    
    dataset_to_save = DatasetDict({
        'train': train,
        'eval': eval
    })
    
    dataset_to_save.save_to_disk(cfg.instruction_tuning.data_path)
    print(f"Saved instruction tuning data to {cfg.instruction_tuning.data_path}")

    
    
if __name__ == "__main__":
    main()