"""
data.py.

Create custom pretrain and finetune PyTorch Dataset objects for MIMIC-IV FHIR dataset.
"""

from typing import Any, Dict, List, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import Dataset

from .tokenizer import ConceptTokenizer


class PretrainDataset(Dataset):
    """Dataset for pretraining the model."""

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: ConceptTokenizer,
        max_len: int = 2048,
        mask_prob: float = 0.15,
    ):
        """Initiate the class."""
        super(PretrainDataset, self).__init__()

        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mask_prob = mask_prob

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    def tokenize_data(self, sequence: Union[str, List[str]]) -> Any:
        """Tokenize the sequence and return input_ids and attention mask."""
        return self.tokenizer(sequence, max_length=self.max_len)

    def mask_tokens(self, sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mask the tokens in the sequence using vectorized operations."""
        mask_token_id = self.tokenizer.get_mask_token_id()

        masked_sequence = sequence.clone()

        # Ignore [PAD], [UNK], [MASK] tokens
        prob_matrix = torch.full(masked_sequence.shape, self.mask_prob)
        prob_matrix[torch.where(masked_sequence <= mask_token_id)] = 0
        selected = torch.bernoulli(prob_matrix).bool()

        # 80% of the time, replace masked input tokens with respective mask tokens
        replaced = torch.bernoulli(torch.full(selected.shape, 0.8)).bool() & selected
        masked_sequence[replaced] = mask_token_id

        # 10% of the time, we replace masked input tokens with random vector.
        randomized = (
            torch.bernoulli(torch.full(selected.shape, 0.1)).bool()
            & selected
            & ~replaced
        )
        random_idx = torch.randint(
            low=self.tokenizer.get_first_token_index(),
            high=self.tokenizer.get_last_token_index(),
            size=prob_matrix.shape,
            dtype=torch.long,
        )
        masked_sequence[randomized] = random_idx[randomized]

        labels = torch.where(selected, sequence, -100)

        return masked_sequence, labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get data at corresponding index.

        Return it as a dictionary including
        all different token sequences along with attention mask and labels.
        """
        data = self.data.iloc[idx]
        tokenized_input = self.tokenize_data(data[f"event_tokens_{self.max_len}"])
        concept_tokens = tokenized_input["input_ids"].squeeze()
        attention_mask = tokenized_input["attention_mask"].squeeze()

        type_tokens = data[f"type_tokens_{self.max_len}"]
        age_tokens = data[f"age_tokens_{self.max_len}"]
        time_tokens = data[f"time_tokens_{self.max_len}"]
        visit_tokens = data[f"visit_tokens_{self.max_len}"]
        position_tokens = data[f"position_tokens_{self.max_len}"]

        masked_tokens, labels = self.mask_tokens(concept_tokens)

        type_tokens = torch.tensor(type_tokens)
        age_tokens = torch.tensor(age_tokens)
        time_tokens = torch.tensor(time_tokens)
        visit_tokens = torch.tensor(visit_tokens)
        position_tokens = torch.tensor(position_tokens)

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
        tokenizer: ConceptTokenizer,
        max_len: int = 2048,
    ):
        """Initiate the class."""
        super(FinetuneDataset, self).__init__()

        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        """Return the length of dataset."""
        return len(self.data)

    def tokenize_data(self, sequence: Union[str, List[str]]) -> Any:
        """Tokenize the sequence and return input_ids and attention mask."""
        return self.tokenizer(sequence, max_length=self.max_len)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get data at corresponding index.

        Return it as a dictionary including
        all different token sequences along with attention mask and labels.
        """
        data = self.data.iloc[idx]
        tokenized_input = self.tokenize_data(data[f"event_tokens_{self.max_len}"])
        concept_tokens = tokenized_input["input_ids"].squeeze()
        attention_mask = tokenized_input["attention_mask"].squeeze()

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
