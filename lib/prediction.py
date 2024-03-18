"""
File: prediction.py
--------------------
Includes helper functions to load model and make predictions.
"""

import os
import sys
from typing import Any, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import pytorch_lightning as pl

from sklearn.model_selection import train_test_split

from lib.data import FinetuneDataset
from lib.tokenizer import ConceptTokenizer
from lib.utils import (
    get_run_id,
    load_config,
    load_finetune_data,
    seed_everything,
)
from models.big_bird_cehr.model import BigBirdFinetune, BigBirdPretrain
from models.cehr_bert.model import BertFinetune, BertPretrain


def load_finetuned_model(
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
    
    Args:
        model_path: Path to the finetuned model to load
        tokenizer: Loaded tokenizer object
        pre_model_config: Optional config to override default values of a pretrained model
        fine_model_config: Optional config to override default values of a finetuned model
        device: CUDA device. By default, GPU is used
    
    Return:
        The loaded PyTorch model
    """
    # Load GPU or CPU device
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available()
                              else 'cpu')

    # Create the skeleton of a pretrained and finetuned model
    pretrained_model = BigBirdPretrain(
        vocab_size=tokenizer.get_vocab_size(),
        padding_idx=tokenizer.get_pad_token_id(),
        **(pre_model_config or {}),
    )

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


def predict_patient_outcomes(
        patient: Dict[str, torch.Tensor],
        model: torch.nn.Module,
        device: Optional[torch.device] = None
) -> Any:
    """
    Return model output predictions on given patient data.
    
    Args:
        patient: Dictionary of tokenized patient information
        model: Finetuned or pretrained PyTorch model
        device: CUDA device. By default, GPU is used
    
    Return:
        Outputs of model predictions on given patient data
    """
    # Load GPU or CPU device
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available()
                              else 'cpu')

    # Load patient information as a Tuple
    patient_inputs = (
        patient["concept_ids"].to(device),
        patient["type_ids"].to(device),
        patient["time_stamps"].to(device),
        patient["ages"].to(device),
        patient["visit_orders"].to(device),
        patient["visit_segments"].to(device),
    )
    patient_labels = patient['labels'].to(device)
    patient_attention_mask = patient['attention_mask'].to(device)

    # Get model output predictions
    model.to(device)
    output = model(
        inputs=patient_inputs,
        attention_mask=patient_attention_mask,
        labels=patient_labels
    )

    return output
