from abc import ABC, abstractmethod
from typing import Any, Dict, Type
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lacuna_data.UnlearningData import UnlearningData
from transformers import PreTrainedTokenizerBase
import pkgutil
import importlib

class UnlearningMethod(ABC):


    def __init__(self, config: Dict[str, Any]):

        self.config = config

    @abstractmethod
    def unlearn(self, model: nn.Module, tokenizer: PreTrainedTokenizerBase, data: UnlearningData) -> nn.Module:
        pass

_UNLEARNING_METHODS: Dict[str, Type[UnlearningMethod]] = {}

def register_method(name: str):
    """Decorator to register new unlearning methods."""
    def decorator(cls):
        _UNLEARNING_METHODS[name] = cls
        return cls
    return decorator

def get_unlearning_method(cfg) -> UnlearningMethod:
    """
    Factory function to instantiate the requested unlearning method.
    
    Args:
        cfg: A configuration object (e.g., OmegaConf or argparse). 
             Must have a 'name' attribute (e.g., cfg.name = 'gradient_ascent').
    """
    if not hasattr(cfg, 'method'):
        raise ValueError("Config must contain a 'method' attribute to specify the method.")
    
    if cfg.method not in _UNLEARNING_METHODS:
        raise ValueError(f"Method '{cfg.method}' not found. Available: {list(_UNLEARNING_METHODS.keys())}")
    
    method_class = _UNLEARNING_METHODS[cfg.method]
    return method_class(config=cfg)


# NEEDED TO IMPORT ALL METHODS AUTOMATICALLY
for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    importlib.import_module(f'{__name__}.{module_name}')