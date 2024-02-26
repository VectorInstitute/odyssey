import random
from typing import Optional, Sequence, Union, Any, List, Tuple, Dict, Set

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from models.big_bird_cehr.tokenizer import HuggingFaceConceptTokenizer


class PretrainDataset(Dataset):
    """Dataset for pretraining the model."""

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: HuggingFaceConceptTokenizer,
        max_len: int = 2048,
        mask_prob: float = 0.15,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mask_prob = mask_prob

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.data)

    def tokenize_data(self, sequence: Union[str, Sequence[str]]) -> Dict[str, List[List[int]]]:
        """Tokenize the sequence and return input_ids and attention mask"""
        return self.tokenizer(sequence)

    def mask_tokens(self, sequence: torch.Tensor) -> tuple:
        """Mask the tokens in the sequence."""
        masked_sequence = []
        labels = []
        for token in sequence:
            if token in self.tokenizer.get_special_token_ids():
                masked_sequence.append(token)
                labels.append(-100)
                continue
            prob = random.random()
            if prob < self.mask_prob:
                dice = random.random()
                if dice < 0.8:
                    masked_sequence.append(self.tokenizer.get_mask_token_id())
                elif dice < 0.9:
                    random_token = random.randint(
                        self.tokenizer.get_first_token_index(),
                        self.tokenizer.get_last_token_index(),
                    )
                    masked_sequence.append(random_token)
                else:
                    masked_sequence.append(token)
                labels.append(token)
            else:
                masked_sequence.append(token)
                labels.append(-100)
        return masked_sequence, labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.data.iloc[idx]
        tokenized_input = self.tokenize_data(data[f"event_tokens_{self.max_len}"])
        concept_tokens = tokenized_input['input_ids']
        attention_mask = tokenized_input['attention_mask']

        type_tokens = data[f"type_tokens_{self.max_len}"]
        age_tokens = data[f"age_tokens_{self.max_len}"]
        time_tokens = data[f"time_tokens_{self.max_len}"]
        visit_tokens = data[f"visit_tokens_{self.max_len}"]
        position_tokens = data[f"position_tokens_{self.max_len}"]

        masked_tokens, labels = self.mask_tokens(concept_tokens)

        masked_tokens = torch.tensor(masked_tokens)
        type_tokens = torch.tensor(type_tokens)
        age_tokens = torch.tensor(age_tokens)
        time_tokens = torch.tensor(time_tokens)
        visit_tokens = torch.tensor(visit_tokens)
        position_tokens = torch.tensor(position_tokens)
        labels = torch.tensor(labels)

        return {
            "concept_ids": masked_tokens,
            "type_ids": type_tokens,
            "ages": age_tokens,
            "time_stamps": time_tokens,
            "visit_orders": position_tokens,
            "visit_segments": visit_tokens,
            "labels": labels,
            "attention_mask": attention_mask,
        }


class FinetuneDataset(Dataset):
    """Dataset for finetuning the model."""

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: HuggingFaceConceptTokenizer,
        max_len: int = 2048,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def tokenize_data(self, sequence) -> Dict[str, List[List[int]]]:
        """Tokenize the sequence and return input_ids and attention mask"""
        return self.tokenizer(sequence)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        data = self.data.iloc[idx]
        tokenized_input = self.tokenize_data(data[f"event_tokens_{self.max_len}"])
        concept_tokens = tokenized_input['input_ids']
        attention_mask = tokenized_input['attention_mask']

        type_tokens = data[f"type_tokens_{self.max_len}"]
        age_tokens = data[f"age_tokens_{self.max_len}"]
        time_tokens = data[f"time_tokens_{self.max_len}"]
        visit_tokens = data[f"visit_tokens_{self.max_len}"]
        position_tokens = data[f"position_tokens_{self.max_len}"]
        labels = data["label"]

        type_tokens = torch.tensor(type_tokens)
        age_tokens = torch.tensor(age_tokens)
        time_tokens = torch.tensor(time_tokens)
        visit_tokens = torch.tensor(visit_tokens)
        position_tokens = torch.tensor(position_tokens)
        labels = torch.tensor(labels)

        return {
            "concept_ids": concept_tokens,
            "type_ids": type_tokens,
            "ages": age_tokens,
            "time_stamps": time_tokens,
            "visit_orders": position_tokens,
            "visit_segments": visit_tokens,
            "labels": labels,
            "attention_mask": attention_mask,
        }
