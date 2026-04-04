"""Prediction module for loading and running EHR-Mamba3 models on patient data."""

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812

from odyssey.data.tokenizer import ConceptTokenizer
from odyssey.models.ehr_mamba3.model import Mamba3Finetune, Mamba3Pretrain


def load_codes_dict(codes_dir: str) -> Dict[str, str]:
    """
    Load medical codes dictionary from a directory.

    Parameters
    ----------
    codes_dir : str
        Directory containing the medical codes dictionary files.

    Returns
    -------
    Dict[str, str]
        Dictionary mapping medical codes to their descriptions.
    """
    codes_dict = {}
    if os.path.exists(codes_dir):
        for filename in os.listdir(codes_dir):
            if filename.endswith(".json"):
                with open(os.path.join(codes_dir, filename), "r") as f:
                    codes_dict.update(json.load(f))
    return codes_dict


def replace_sequence_items(
    sequence: List[str], codes_dict: Dict[str, str]
) -> List[str]:
    """
    Replace medical codes in a sequence with their descriptions.

    Parameters
    ----------
    sequence : List[str]
        List of medical codes.
    codes_dict : Dict[str, str]
        Dictionary mapping medical codes to their descriptions.

    Returns
    -------
    List[str]
        Sequence with codes replaced by descriptions when available.
    """
    return [codes_dict.get(item, item) for item in sequence]


def load_pretrained_model(
    tokenizer: ConceptTokenizer,
    model_path: str,
    device: Optional[torch.device] = None,
    **model_kwargs: Any,
) -> Mamba3Pretrain:
    """
    Load a pre-trained EHR-Mamba3 model from a checkpoint.

    Parameters
    ----------
    tokenizer : ConceptTokenizer
        Tokenizer used with the model.
    model_path : str
        Path to the saved model checkpoint.
    device : torch.device, optional
        Device to load the model on. Defaults to CUDA if available.
    **model_kwargs
        Additional keyword arguments forwarded to :class:`Mamba3Pretrain`.

    Returns
    -------
    Mamba3Pretrain
        Loaded and prepared model ready for inference.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Mamba3Pretrain(
        vocab_size=tokenizer.get_vocab_size(),
        padding_idx=tokenizer.get_pad_token_id(),
        cls_idx=tokenizer.get_class_token_id(),
        **model_kwargs,
    )
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.to(device)
    return model


def load_finetuned_model(
    tokenizer: ConceptTokenizer,
    pretrain_path: str,
    finetune_path: str,
    num_labels: int = 2,
    problem_type: str = "single_label_classification",
    device: Optional[torch.device] = None,
    pre_model_kwargs: Optional[Dict[str, Any]] = None,
    fine_model_kwargs: Optional[Dict[str, Any]] = None,
) -> Mamba3Finetune:
    """
    Load a fine-tuned EHR-Mamba3 model.

    Parameters
    ----------
    tokenizer : ConceptTokenizer
        Tokenizer used with the model.
    pretrain_path : str
        Path to the pre-trained backbone checkpoint.
    finetune_path : str
        Path to the fine-tuned model checkpoint.
    num_labels : int, optional
        Number of output labels, by default 2.
    problem_type : str, optional
        One of ``"single_label_classification"``, ``"multi_label_classification"``,
        or ``"regression"``.
    device : torch.device, optional
        Device to load the model on. Defaults to CUDA if available.
    pre_model_kwargs : dict, optional
        Additional keyword arguments for :class:`Mamba3Pretrain`.
    fine_model_kwargs : dict, optional
        Additional keyword arguments for :class:`Mamba3Finetune`.

    Returns
    -------
    Mamba3Finetune
        Loaded and prepared fine-tuned model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pretrained_model = load_pretrained_model(
        tokenizer=tokenizer,
        model_path=pretrain_path,
        device=device,
        **(pre_model_kwargs or {}),
    )

    model = Mamba3Finetune(
        pretrained_model=pretrained_model,
        num_labels=num_labels,
        problem_type=problem_type,
        **(fine_model_kwargs or {}),
    )
    state_dict = torch.load(finetune_path, map_location=device)["state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def create_concept_and_id_to_type_mapping(
    pretrain_data: pd.DataFrame, tokenizer: ConceptTokenizer
) -> Dict[Union[str, int], Any]:
    """
    Create a mapping from concepts and their IDs to their corresponding type IDs.

    Parameters
    ----------
    pretrain_data : pd.DataFrame
        DataFrame with columns ``event_tokens_2048`` and ``type_tokens_2048``.
    tokenizer : ConceptTokenizer
        Tokenizer with a ``token_to_id`` method.

    Returns
    -------
    Dict[Union[str, int], Any]
        Dictionary mapping concepts and their IDs to type IDs.
    """
    concept_and_id_to_type: Dict[Union[str, int], Any] = {}

    for events, types in zip(
        pretrain_data["event_tokens_2048"], pretrain_data["type_tokens_2048"]
    ):
        unique_concepts, first_occurrence = np.unique(events, return_index=True)
        for concept, index in zip(unique_concepts, first_occurrence):
            if concept not in concept_and_id_to_type:
                concept_id = tokenizer.token_to_id(concept)
                type_id = types[index]
                concept_and_id_to_type[concept] = type_id
                concept_and_id_to_type[concept_id] = type_id

    return concept_and_id_to_type


class Forecast:
    """Forecast token sequences using a pre-trained EHR-Mamba3 model."""

    def __init__(
        self,
        model: Mamba3Pretrain,
        tokenizer: ConceptTokenizer,
        pretrain_data: pd.DataFrame,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ):
        """
        Initialize the Forecast class.

        Parameters
        ----------
        model : Mamba3Pretrain
            Pre-trained model used for generating predictions.
        tokenizer : ConceptTokenizer
            Tokenizer for converting tokens to IDs and vice versa.
        pretrain_data : pd.DataFrame
            Pre-training data used to create concept-to-type mappings.
        temperature : float, optional
            Sampling temperature, by default 0.8.
        top_p : float, optional
            Nucleus sampling parameter, by default 0.95.
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
        """Return the index of the first padding token."""
        nonzero_indices = concept_ids.nonzero()
        if nonzero_indices.numel() == 0:
            return 0
        return int(nonzero_indices.squeeze().tolist()[-1]) + 1

    def prepare_input_data(
        self,
        patient: Dict[str, torch.Tensor],
        predicted_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Prepare patient data for model input by appending predicted IDs.

        Parameters
        ----------
        patient : Dict[str, torch.Tensor]
            Patient data tensors.
        predicted_ids : torch.Tensor
            Predicted concept IDs to append.

        Returns
        -------
        Tuple[torch.Tensor, ...]
            Tensors ready for model input.
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
            Input tensors prepared for the model.

        Returns
        -------
        int
            Predicted token ID.
        """
        with torch.no_grad():
            logits = self.model.get_logits(inputs)[0, -1, :]  # type: ignore[arg-type]

        if self.temperature == 0:
            return int(torch.argmax(logits).item())

        logits = logits / self.temperature
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        cutoff_index = torch.sum(cumulative_probs < self.top_p) + 1
        top_p_tokens = sorted_indices[:cutoff_index]
        top_p_probs = probs[top_p_tokens]
        top_p_probs /= top_p_probs.sum()
        return int(top_p_tokens[int(torch.multinomial(top_p_probs, 1).item())].item())

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
            Patient data tensors.
        num_tokens : int
            Number of tokens to generate.
        cutoff_index : int, optional
            Index to truncate patient data. Calculated if not provided.

        Returns
        -------
        Tuple[List[int], List[str], List[str]]
            Predicted IDs, predicted tokens, and decoded labels.
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
