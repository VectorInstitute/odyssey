"""Prediction module for loading and running EHR models on patient data.

This module provides functionality to load and run EHR models on patient data,
both for clinical predictive tasks and EHR forecasting.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from odyssey.data.tokenizer import ConceptTokenizer
from odyssey.models.cehr_big_bird.model import BigBirdFinetune, BigBirdPretrain
from odyssey.models.ehr_mamba.model import MambaPretrain
from odyssey.models.ehr_mamba2.model import Mamba2Pretrain


def load_pretrained_model(
    model_type: str, tokenizer: ConceptTokenizer, device: torch.device, model_path: str
) -> torch.nn.Module:
    """
    Load a pretrained model based on the specified model type and tokenizer.

    This function initializes a model of the specified type, loads its pretrained
    weights from a checkpoint file, and prepares it for inference on the specified
    device.

    Parameters
    ----------
    model_type : str
        The type of model to load. Currently implements "mamba".
    tokenizer : ConceptTokenizer
        The tokenizer associated with the model, used to determine vocabulary size
        and special token IDs.
    device : torch.device
        The device (CPU or GPU) on which to load the model.
    model_path : str
        The file path to the saved model checkpoint.

    Returns
    -------
    torch.nn.Module
        The loaded and prepared model, ready for inference.
    """
    if model_type == "mamba":
        model = MambaPretrain(
            vocab_size=tokenizer.get_vocab_size(),
            embedding_size=768,
            state_size=16,
            num_hidden_layers=32,
            max_seq_length=2048,
            padding_idx=tokenizer.get_pad_token_id(),
            cls_idx=tokenizer.get_class_token_id(),
        )
    elif model_type == "mamba2":
        model = Mamba2Pretrain(
            vocab_size=tokenizer.get_vocab_size(),
            embedding_size=768,
            state_size=64,
            num_hidden_layers=32,
            max_seq_length=2048,
            padding_idx=tokenizer.get_pad_token_id(),
            cls_idx=tokenizer.get_class_token_id(),
            eos_idx=tokenizer.get_eos_token_id(),
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Load the pretrained weights
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)

    # Set the model to evaluation mode and move it to the specified device
    model.eval()
    model.to(device)

    return model


def load_finetuned_model(
    model_path: str,
    tokenizer: ConceptTokenizer,
    pre_model_config: Optional[Dict[str, Any]] = None,
    fine_model_config: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    """
    Load a finetuned model from model_path using tokenizer information.

    Return a loaded finetuned model from model_path, using tokenizer information.
    If config arguments are not provided, the default configs built into the
    PyTorch classes are used.

    Parameters
    ----------
    model_path : str
        Path to the finetuned model to load
    tokenizer : ConceptTokenizer
        Loaded tokenizer object
    pre_model_config : Dict[str, Any], optional
        Optional config to override default values of a pretrained model
    fine_model_config : Dict[str, Any], optional
        Optional config to override default values of a finetuned model
    device : torch.device, optional
        CUDA device. By default, GPU is used

    Returns
    -------
    torch.nn.Module
        Finetuned model loaded from model_path
    """
    # Load GPU or CPU device
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def create_concept_and_id_to_type_mapping(
    pretrain_data: pd.DataFrame, tokenizer: ConceptTokenizer
) -> Dict[Union[str, int], Any]:
    """
    Create a mapping from concepts and their IDs to their corresponding type IDs.

    This function processes pretraining data to build a dictionary that maps each unique
    concept and its corresponding ID to its type ID, based on the first occurrence of
    the concept in the data.

    Parameters
    ----------
    pretrain_data : pd.DataFrame
        A pandas DataFrame containing the pretraining data with columns
        'event_tokens_2048' and 'type_tokens_2048'.
    tokenizer : ConceptTokenizer
        The tokenizer object used to convert concepts to IDs. Must have a 'token_to_id'
        method.

    Returns
    -------
    Dict[Union[str, int], Any]
        A dictionary mapping concepts and their IDs to their type IDs.
    """
    concept_and_id_to_type: Dict[Union[str, int], Any] = {}

    # Vectorized operation to process all rows at once
    for events, types in zip(
        pretrain_data["event_tokens_2048"], pretrain_data["type_tokens_2048"]
    ):
        # Use numpy's unique function to get unique concepts and
        # their first occurrence index
        unique_concepts, first_occurrence = np.unique(events, return_index=True)

        # Map each unique concept and its ID to its corresponding type
        for concept, index in zip(unique_concepts, first_occurrence):
            if concept not in concept_and_id_to_type:
                concept_id = tokenizer.token_to_id(concept)
                type_id = types[index]
                concept_and_id_to_type[concept] = type_id
                concept_and_id_to_type[concept_id] = type_id

    return concept_and_id_to_type


class Forecast:
    """Forecast token sequences using a pretrained model."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: ConceptTokenizer,
        pretrain_data: pd.DataFrame,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ):
        """
        Initialize the Forecast class for generating token sequences.

        Parameters
        ----------
        model : torch.nn.Module
            The pretrained model used for generating predictions.
        tokenizer : ConceptTokenizer
            The tokenizer used to convert tokens to IDs and vice versa.
        pretrain_data : pd.DataFrame
            The pretraining data used to create concept to type mappings.
        temperature : float, optional
            The temperature parameter for sampling. Default is 0.8.
        top_p : float, optional
            The top-p (nucleus) sampling parameter. Default is 0.95.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.model.to(self.device)
        self.concept_and_id_to_type = create_concept_and_id_to_type_mapping(
            pretrain_data, tokenizer
        )
        self.temperature = temperature
        self.top_p = top_p

    @staticmethod
    def get_pad_start_idx(concept_ids: torch.Tensor) -> int:
        """
        Find the start index of padding in a tensor of concept IDs.

        Parameters
        ----------
        concept_ids : torch.Tensor
            A tensor containing concept IDs.

        Returns
        -------
        int
            The index of the first padding token.
        """
        return concept_ids.nonzero().squeeze().tolist()[-1] + 1

    def prepare_input_data(
        self,
        patient: Dict[str, torch.Tensor],
        predicted_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Prepare patient data for model input efficiently by updating with predicted IDs.

        Parameters
        ----------
        patient : Dict[str, torch.Tensor]
            A dictionary containing patient data tensors.
        predicted_ids : torch.Tensor
            A tensor of predicted concept IDs to be appended to the patient data.

        Returns
        -------
        Tuple[torch.Tensor, ...]
            A tuple of tensors ready for model input.
        """
        inputs = []

        for key in [
            "concept_ids",
            "type_ids",
            "time_stamps",
            "ages",
            "visit_orders",
            "visit_segments",
        ]:
            tensor = patient[key].to(self.device)

            if key == "concept_ids":
                new_tensor = torch.cat([tensor, predicted_ids], dim=0)

            elif key == "type_ids":
                # Map predicted concept IDs to their corresponding type IDs
                predicted_type_ids = torch.tensor(
                    [
                        self.concept_and_id_to_type.get(id_.item(), 0)
                        for id_ in predicted_ids
                    ],
                    device=self.device,
                    dtype=tensor.dtype,
                )
                new_tensor = torch.cat([tensor, predicted_type_ids], dim=0)

            else:
                # For other features, repeat the last value
                last_value = tensor[-1]
                new_tensor = torch.cat(
                    [tensor, last_value.repeat(len(predicted_ids))], dim=0
                )

            inputs.append(new_tensor.unsqueeze(0))

        return tuple(inputs)

    def predict_next_token(self, inputs: Tuple[torch.Tensor, ...]) -> int:
        """
        Predict the next token using temperature and top-p sampling.

        Parameters
        ----------
        inputs : Tuple[torch.Tensor, ...]
            A tuple of input tensors prepared for the model.

        Returns
        -------
        int
            The predicted token ID.
        """
        with torch.no_grad():
            output = self.model(inputs)
        logits = output["logits"][0, -1, :]

        if self.temperature == 0:
            return torch.argmax(logits).item()

        logits = logits / self.temperature

        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        cutoff_index = torch.sum(cumulative_probs < self.top_p) + 1
        top_p_tokens = sorted_indices[:cutoff_index]
        top_p_probs = probs[top_p_tokens]
        top_p_probs /= top_p_probs.sum()

        selected_token_index = torch.multinomial(top_p_probs, 1).item()
        return top_p_tokens[selected_token_index].item()

    def generate_token_sequence(
        self,
        patient: Dict[str, torch.Tensor],
        num_tokens: int,
        cutoff_index: Optional[int] = None,
    ) -> Tuple[List[int], List[str], List[str]]:
        """
        Generate a sequence of tokens based on patient data.

        Parameters
        ----------
        patient : Dict[str, torch.Tensor]
            A dictionary containing patient data tensors.
        num_tokens : int
            The number of tokens to generate.
        cutoff_index : int, optional
            The index at which to truncate the patient data. If None, it will be
            calculated.

        Returns
        -------
        Tuple[List[int], List[str], List[str]]
            A tuple containing:
            - predicted_ids_list: List of predicted token IDs.
            - predicted_tokens: List of predicted tokens.
            - predicted_labels: List of predicted labels decoded from tokens.
        """
        predicted_ids = torch.tensor([], dtype=torch.long, device=self.device)

        if not cutoff_index:
            pad_start_idx = self.get_pad_start_idx(patient["concept_ids"])
            cutoff_index = max(pad_start_idx - num_tokens, 0)

        patient = {key: value[:cutoff_index] for key, value in patient.items()}

        for _ in range(num_tokens):
            inputs = self.prepare_input_data(patient, predicted_ids)
            prediction_id = self.predict_next_token(inputs)
            predicted_ids = torch.cat(
                [predicted_ids, torch.tensor([prediction_id], device=self.device)]
            )

        predicted_ids_list = predicted_ids.cpu().tolist()
        predicted_tokens = [
            self.tokenizer.id_to_token(id_) for id_ in predicted_ids_list
        ]
        predicted_labels = self.tokenizer.decode_to_labels(predicted_tokens)

        return predicted_ids_list, predicted_tokens, predicted_labels
