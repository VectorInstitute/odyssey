import random
from typing import Sequence, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from models.cehr_bert.tokenizer import ConceptTokenizer


class PretrainDataset(Dataset):
    """Dataset for pretraining the model."""

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: ConceptTokenizer,
        max_len: int = 512,
        mask_prob: float = 0.15,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mask_prob = mask_prob

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.data)

    def tokenize_data(self, sequence: Union[str, Sequence[str]]) -> np.ndarray:
        """Tokenize the sequence."""
        tokenized = self.tokenizer.encode(sequence)
        tokenized = np.array(tokenized).flatten()
        return tokenized

    def get_attention_mask(self, sequence: np.ndarray) -> np.ndarray:
        """Get the attention mask for the sequence."""
        attention_mask = [
            float(token != self.tokenizer.get_pad_token_id()) for token in sequence
        ]
        return attention_mask

    def mask_tokens(self, sequence: np.ndarray) -> tuple:
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

    def __getitem__(self, idx: int) -> dict:
        data = self.data.iloc[idx]
        concept_tokens = self.tokenize_data(data["event_tokens"])
        age_tokens = data["age_tokens"]
        time_tokens = data["time_tokens"]
        visit_tokens = data["visit_tokens"]
        position_tokens = data["position_tokens"]

        attention_mask = self.get_attention_mask(concept_tokens)
        masked_tokens, labels = self.mask_tokens(concept_tokens)

        masked_tokens = torch.tensor(masked_tokens)
        age_tokens = torch.tensor(age_tokens)
        time_tokens = torch.tensor(time_tokens)
        visit_tokens = torch.tensor(visit_tokens)
        position_tokens = torch.tensor(position_tokens)
        labels = torch.tensor(labels)
        attention_mask = torch.tensor(attention_mask)

        return {
            "concept_ids": masked_tokens,
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
        tokenizer: ConceptTokenizer,
        max_len: int = 512,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def tokenize_data(self, sequence):
        """Tokenize the sequence."""
        tokenized = self.tokenizer.encode(sequence)
        tokenized = np.array(tokenized).flatten()
        return tokenized

    def get_attention_mask(self, sequence):
        """Get the attention mask for the sequence."""
        attention_mask = [
            float(token != self.tokenizer.get_pad_token_id()) for token in sequence
        ]
        return attention_mask

    def __getitem__(self, idx):
        data = self.data.iloc[idx]
        concept_tokens = self.tokenize_data(data["event_tokens"])
        age_tokens = data["age_tokens"]
        time_tokens = data["time_tokens"]
        visit_tokens = data["visit_tokens"]
        position_tokens = data["position_tokens"]
        labels = data["label"]
        attention_mask = self.get_attention_mask(concept_tokens)

        concept_tokens = torch.tensor(concept_tokens)
        age_tokens = torch.tensor(age_tokens)
        time_tokens = torch.tensor(time_tokens)
        visit_tokens = torch.tensor(visit_tokens)
        position_tokens = torch.tensor(position_tokens)
        labels = torch.tensor(labels)
        attention_mask = torch.tensor(attention_mask)

        return {
            "concept_ids": concept_tokens,
            "ages": age_tokens,
            "time_stamps": time_tokens,
            "visit_orders": position_tokens,
            "visit_segments": visit_tokens,
            "labels": labels,
            "attention_mask": attention_mask,
        }
