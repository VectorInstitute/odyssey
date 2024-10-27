"""Utility functions for the model module."""

import os
import pickle
import uuid
from os.path import join
from typing import Any, Dict, Optional, Union

import pandas as pd
import polars as pl
import torch
import yaml

from odyssey.data.tokenizer import ConceptTokenizer
from odyssey.models.cehr_bert.model import BertFinetune, BertPretrain
from odyssey.models.cehr_big_bird.model import BigBirdFinetune, BigBirdPretrain


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

    # Loading with pandas directly might fail
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


def load_finetuned_model(
    model_type: str,
    model_path: str,
    tokenizer: ConceptTokenizer,
    pre_model_config: Optional[Dict[str, Any]] = None,
    fine_model_config: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    """
    Return a loaded finetuned model from model_path, using tokenizer information.

    If config arguments are not provided, the default configs built into the
    PyTorch classes are used.

    Parameters
    ----------
    model_path : str
        Path to the fine-tuned model to load.
    tokenizer : tokenizer object
        Loaded tokenizer object that is used with the model.
    pre_model_config : dict, optional
        Config to override default values of a pretrained model.
    fine_model_config : dict, optional
        Config to override default values of a finetuned model.
    device : str, optional
        CUDA device to use for loading the model. Defaults to GPU if not specified.

    Returns
    -------
    torch.nn.Module
    The loaded PyTorch model.
    """
    # Load GPU or CPU device
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the skeleton of a pretrained and finetuned model
    if model_type == "cehr_bert":
        pretrained_model = BertPretrain(
            vocab_size=tokenizer.get_vocab_size(),
            padding_idx=tokenizer.get_pad_token_id(),
            **(pre_model_config or {}),
        )
        pretrained_model.eval()
        model = BertFinetune(
            pretrained_model=pretrained_model,
            **(fine_model_config or {}),
        )
    elif model_type == "cehr_bigbird":
        pretrained_model = BigBirdPretrain(
            vocab_size=tokenizer.get_vocab_size(),
            padding_idx=tokenizer.get_pad_token_id(),
            **(pre_model_config or {}),
        )
        pretrained_model.eval()
        model = BigBirdFinetune(
            pretrained_model=pretrained_model,
            **(fine_model_config or {}),
        )
    # Load the weights using model_path directory
    state_dict = torch.load(model_path, map_location=device)["state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def get_model_embeddings(
    model: Union[BertFinetune, BigBirdFinetune],
) -> torch.nn.Module:
    """
    Retrieve the embedding module from a given model.

    Parameters
    ----------
    model : torch.nn.Module
        The model from which embeddings are to be extracted.

    Returns
    -------
    torch.nn.Module
        The embeddings component of the specified model.

    Raises
    ------
    ValueError
        If the model type is not supported, raises a ValueError.
    """
    if isinstance(model, BertFinetune):
        return model.pretrained_model.embeddings
    if isinstance(model, BigBirdFinetune):
        return model.bert.embeddings
    raise ValueError("Model not supported")


def get_model_embeddings_list(
    model: Union[BertFinetune, BigBirdFinetune],
) -> Dict[str, torch.nn.Module]:
    """
    Extract specific embedding components from a given model.

    Parameters
    ----------
    model : torch.nn.Module
        The model from which specific embeddings are to be extracted.

    Returns
    -------
    Dict[str, torch.nn.Module]
        A dictionary mapping the names of embedding types to
        their respective torch.nn.Module components.

    Raises
    ------
    ValueError
        If the model type is not supported, a ValueError is raised.
    """
    if isinstance(model, BertFinetune):
        return {
            "concept_embeddings": model.pretrained_model.embeddings.concept_embedding,
            "token_type_embeddings": model.pretrained_model.embeddings.token_type_embeddings,
            "time_embeddings": model.pretrained_model.embeddings.time_embedding,
            "age_embeddings": model.pretrained_model.embeddings.age_embedding,
            "visit_order_embeddings": model.pretrained_model.embeddings.positional_embedding,
            "visit_segment_embeddings": model.pretrained_model.embeddings.visit_embedding,
        }
    if isinstance(model, BigBirdFinetune):
        return {
            "concept_embeddings": model.bert.embeddings.word_embeddings,
            "token_type_embeddings": model.bert.embeddings.token_type_embeddings,
            "time_embeddings": model.bert.embeddings.time_embeddings,
            "age_embeddings": model.bert.embeddings.age_embeddings,
            "visit_order_embeddings": model.bert.embeddings.visit_order_embeddings,
            "visit_segment_embeddings": model.bert.embeddings.visit_embeddings,
            "position_embeddings": model.bert.embeddings.position_embeddings,
        }
    raise ValueError("Model not supported")
