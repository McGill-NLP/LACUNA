
from datasets import load_dataset

class UnlearningData():
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.retain = load_dataset(cfg.retain.Retain.data_path, name=cfg.retain.Retain.name, split=cfg.retain.Retain.split, download_mode='force_redownload')
        self.forget = load_dataset(cfg.forget.Forget.data_path, name=cfg.forget.Forget.name, split=cfg.forget.Forget.split, download_mode='force_redownload')
        self.full = load_dataset(cfg.full.Full.data_path, name=cfg.full.Full.name, split=cfg.full.Full.split, download_mode='force_redownload')
        return
