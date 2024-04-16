"""
File: utils.py
---------------
General utility functions and constants for the project.
"""

from typing import Any

import pickle
import random

import numpy as np
import torch
import pytorch_lightning as pl


def seed_everything(seed: int) -> None:
    """Seed all components of the model.

    Parameters
    ----------
    seed: int
        Seed value to use

    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed)


def save_object_to_disk(obj: Any, save_path: str) -> None:
    """Save an object to disk using pickle.

    Parameters
    ----------
    obj: Any
        Object to save
    save_path: str
        Path to save the object

    """
    with open(save_path, "wb") as f:
        pickle.dump(obj, f)
        print(f"File saved to disk: {save_path}")
