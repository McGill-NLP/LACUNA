import os
import random
import numpy as np
import torch
import json
import hashlib
import wandb
from omegaconf import OmegaConf

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
class temp_seed:
    def __init__(self, seed):
        self.seed = seed
        self.state = None
        self.np_state = None
        self.torch_state = None

    def __enter__(self):
        # Save current states
        self.state = random.getstate()
        self.np_state = np.random.get_state()
        self.torch_state = torch.get_rng_state()
        
        # Set new seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def __exit__(self, type, value, traceback):
        # Restore old states
        random.setstate(self.state)
        np.random.set_state(self.np_state)
        torch.set_rng_state(self.torch_state)
    
    
def init_wandb_run(config, run_name):
    config_dict = OmegaConf.to_container(config, resolve=True)
    config_str = json.dumps(config_dict, sort_keys=True)
    config_hash = hashlib.sha1(config_str.encode()).hexdigest()[:8]
    
    for attempt in range(100):
        run_id = config_hash if attempt == 0 else f"{config_hash}_{attempt}"
        
        try:
            wandb.init(
                settings=wandb.Settings(init_timeout=5),
                entity=config.wandb.entity,
                project=config.wandb.project,
                name=run_name,
                id=run_id,
                resume="allow"
            )
            print(f"✓ wandb initialized with run_id: {run_id}")
            return
            
        except Exception as e:
            if wandb.run:
                wandb.finish(quiet=True)