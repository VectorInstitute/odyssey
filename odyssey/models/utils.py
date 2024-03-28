"""Utility functions for the model."""

import os
import pickle
import uuid
from os.path import join
from typing import Any

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml


def load_config(config_dir: str, model_type: str) -> Any:
    """Load the model configuration.

    Parameters
    ----------
    config_dir: str
        Directory containing the model configuration files

    model_type: str
        Model type to load configuration for

    Returns
    -------
    Any
        Model configuration

    """
    config_file = join(config_dir, f"{model_type}.yaml")
    with open(config_file, "r") as file:
        return yaml.safe_load(file)


def seed_everything(seed: int) -> None:
    """Seed all components of the model.

    Parameters
    ----------
    seed: int
        Seed value to use

    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed)


def load_pretrain_data(
    data_dir: str,
    sequence_file: str,
    id_file: str,
) -> pd.DataFrame:
    """Load the pretraining data.

    Parameters
    ----------
    data_dir: str
        Directory containing the data files
    sequence_file: str
        Sequence file name
    id_file: str
        ID file name

    Returns
    -------
    pd.DataFrame
        Pretraining data

    """
    sequence_path = join(data_dir, sequence_file)
    id_path = join(data_dir, id_file)

    if not os.path.exists(sequence_path):
        raise FileNotFoundError(f"Sequence file not found: {sequence_path}")

    if not os.path.exists(id_path):
        raise FileNotFoundError(f"ID file not found: {id_path}")

    data = pd.read_parquet(sequence_path)
    with open(id_path, "rb") as file:
        patient_ids = pickle.load(file)

    return data.loc[data["patient_id"].isin(patient_ids["pretrain"])]


def load_finetune_data(
    data_dir: str,
    sequence_file: str,
    id_file: str,
    valid_scheme: str,
    num_finetune_patients: str,
) -> pd.DataFrame:
    """Load the finetuning data.

    Parameters
    ----------
    data_dir: str
        Directory containing the data files
    sequence_file: str
        Sequence file name
    id_file: str
        ID file name
    valid_scheme: str
        Validation scheme
    num_finetune_patients: str
        Number of finetune patients

    Returns
    -------
    pd.DataFrame
        Finetuning data

    """
    sequence_path = join(data_dir, sequence_file)
    id_path = join(data_dir, id_file)

    if not os.path.exists(sequence_path):
        raise FileNotFoundError(f"Sequence file not found: {sequence_path}")

    if not os.path.exists(id_path):
        raise FileNotFoundError(f"ID file not found: {id_path}")

    data = pd.read_parquet(sequence_path)
    with open(id_path, "rb") as file:
        patient_ids = pickle.load(file)

    fine_tune = data.loc[
        data["patient_id"].isin(
            patient_ids["finetune"][valid_scheme][num_finetune_patients],
        )
    ]
    fine_test = data.loc[data["patient_id"].isin(patient_ids["test"])]
    return fine_tune, fine_test


def get_run_id(
    checkpoint_dir: str,
    retrieve: bool = False,
    run_id_file: str = "wandb_run_id.txt",
    length: int = 8,
) -> str:
    """Fetch the run ID for the current run.

    If the run ID file exists, retrieve the run ID from the file.
    Otherwise, generate a new run ID and save it to the file.

    Parameters
    ----------
    checkpoint_dir: str
        Directory to store the run ID file
    retrieve: bool, optional
        Retrieve the run ID from the file, by default False
    run_id_file: str, optional
        Run ID file name, by default "wandb_run_id.txt"
    length: int, optional
        String length of the run ID, by default 8

    Returns
    -------
    str
        Run ID for the current run

    """
    run_id_path = os.path.join(checkpoint_dir, run_id_file)
    if retrieve and os.path.exists(run_id_path):
        with open(run_id_path, "r") as file:
            run_id = file.read().strip()
    else:
        run_id = str(uuid.uuid4())[:length]
        with open(run_id_path, "w") as file:
            file.write(run_id)
    return run_id
