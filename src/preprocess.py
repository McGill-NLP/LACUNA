
import logging
from omegaconf import DictConfig, OmegaConf
import hydra
from data.info_match import info_match
from data.data_replication import replicate
from data.convert_npy_to_jsonl import convert_to_jsonl
from data.merge_datasets_preprocess import preprocess_data
from utils import set_seed
import logging

@hydra.main(version_base=None, config_path="../configs/.", config_name="config")
def main(cfg):

    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    print(OmegaConf.to_yaml(resolved_cfg))
    
    stages = cfg.data.preprocess_stages
    if 'info_match' in stages:
        set_seed(cfg.seed)
        print("Information Matching", flush=True)
        info_match(cfg)
    if 'replication' in stages:
        set_seed(cfg.seed)
        print("Data Replication",flush=True)
        replicate(cfg)
    if 'data_convert' in stages:
        set_seed(cfg.seed)
        print("Data Conversion", flush=True)
        convert_to_jsonl(cfg)    
    if 'train_ready' in stages:
        set_seed(cfg.seed)
        print("Final Mixing", flush=True)
        preprocess_data(cfg)
    
if __name__ == "__main__":
    main()