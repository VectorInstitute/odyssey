"""Utility functions for the model module."""

import os
import pickle
import uuid
from os.path import join
from typing import Any, Tuple

import pandas as pd
import polars as pl
import yaml


def load_config(config_dir: str, model_type: str) -> Any:
    """Load the model configuration from a YAML file.

    Parameters
    ----------
    config_dir : str
        Directory containing the model configuration files.
    model_type : str
        Model type to load configuration for.

    Returns
    -------
    Any
        Parsed YAML configuration dictionary.
    """
    config_file = join(config_dir, f"{model_type}.yaml")
    with open(config_file, "r") as file:
        return yaml.safe_load(file)


def load_pretrain_data(
    data_dir: str,
    sequence_file: str,
    id_file: str,
) -> pd.DataFrame:
    """Load the pretraining data.

    Parameters
    ----------
    data_dir : str
        Directory containing the data files.
    sequence_file : str
        Parquet file name with patient sequences.
    id_file : str
        Pickle file name with patient ID splits.

    Returns
    -------
    pd.DataFrame
        DataFrame filtered to pretrain patient IDs.
    """
    sequence_path = join(data_dir, sequence_file)
    id_path = join(data_dir, id_file)

    if not os.path.exists(sequence_path):
        raise FileNotFoundError(f"Sequence file not found: {sequence_path}")
    if not os.path.exists(id_path):
        raise FileNotFoundError(f"ID file not found: {id_path}")

    data = pl.read_parquet(sequence_path).to_pandas()
    with open(id_path, "rb") as file:
        patient_ids = pickle.load(file)

    return data.loc[data["patient_id"].isin(patient_ids["pretrain"])]


def load_finetune_data(
    data_dir: str,
    sequence_file: str,
    id_file: str,
    valid_scheme: str,
    num_finetune_patients: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the fine-tuning data.

    Parameters
    ----------
    data_dir : str
        Directory containing the data files.
    sequence_file : str
        Parquet file name with patient sequences.
    id_file : str
        Pickle file name with patient ID splits.
    valid_scheme : str
        Validation scheme key (e.g. ``"few_shot"``).
    num_finetune_patients : str
        Number of patients key within the validation scheme.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        ``(fine_tune, fine_test)`` DataFrames.
    """
    sequence_path = join(data_dir, "patient_sequences", sequence_file)
    id_path = join(data_dir, "patient_id_dict", id_file)

    if not os.path.exists(sequence_path):
        raise FileNotFoundError(f"Sequence file not found: {sequence_path}")
    if not os.path.exists(id_path):
        raise FileNotFoundError(f"ID file not found: {id_path}")

    data = pd.read_parquet(sequence_path)
    with open(id_path, "rb") as file:
        patient_ids = pickle.load(file)

    fine_tune = data.loc[
        data["patient_id"].isin(
            patient_ids["finetune"][valid_scheme][num_finetune_patients]
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
    """Fetch (or generate) the W&B run ID for the current run.

    Parameters
    ----------
    checkpoint_dir : str
        Directory to store the run ID file.
    retrieve : bool, optional
        If ``True`` and the file exists, return the stored ID.
    run_id_file : str, optional
        File name for the run ID, by default ``"wandb_run_id.txt"``.
    length : int, optional
        Length of a newly generated UUID prefix, by default 8.

    Returns
    -------
    str
        Run ID.
    """
    run_id_path = os.path.join(checkpoint_dir, run_id_file)
    if retrieve and os.path.exists(run_id_path):
        with open(run_id_path, "r") as file:
            return file.read().strip()
    run_id = str(uuid.uuid4())[:length]
    with open(run_id_path, "w") as file:
        file.write(run_id)
    return run_id
