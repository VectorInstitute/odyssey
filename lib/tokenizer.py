"""
file: tokenizer.py.

Custom HuggingFace tokenizer for medical concepts in MIMIC-IV FHIR dataset.
"""

import glob
import json
import os
from itertools import chain
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import BatchEncoding, PreTrainedTokenizerFast


def truncate_and_pad(row: pd.Series, cutoff: Optional[int] = None, max_len: int = 2048) -> pd.Series:
    """ Return a truncated and padded version of row.

    Args:
        row (pd.Series): The input row.
        cutoff (Optional[int): The cutoff length. Will be set to length of 'event_tokens_2048' if None.
        max_len (int): The maximum length to pad to.

    Returns:
        Any: The truncated and padded row.

    Note:
        This function assumes the presence of the following columns in row:
        - 'event_tokens_2048'
        - 'type_tokens_2048'
        - 'age_tokens_2048'
        - 'time_tokens_2048'
        - 'visit_tokens_2048'
        - 'position_tokens_2048'
        - 'elapsed_tokens_2048'
    """
    # Ensuring row is a copy to prevent SettingWithCopyWarning
    row = row.copy()
    
    if not cutoff:
        cutoff = len(row['event_tokens_2048'])
    
    row['event_tokens_2048'] = row['event_tokens_2048'][:cutoff]
    row['type_tokens_2048'] = np.pad(row['type_tokens_2048'][:cutoff], (0, max_len - cutoff), mode='constant')
    row['age_tokens_2048'] = np.pad(row['age_tokens_2048'][:cutoff], (0, max_len - cutoff), mode='constant')
    row['time_tokens_2048'] = np.pad(row['time_tokens_2048'][:cutoff], (0, max_len - cutoff), mode='constant')
    row['visit_tokens_2048'] = np.pad(row['visit_tokens_2048'][:cutoff], (0, max_len - cutoff), mode='constant')
    row['position_tokens_2048'] = np.pad(row['position_tokens_2048'][:cutoff], (0, max_len - cutoff), mode='constant')
    row['elapsed_tokens_2048'] = np.pad(row['elapsed_tokens_2048'][:cutoff], (0, max_len - cutoff), mode='constant')

    row['event_tokens_2048'] = ' '.join(row['event_tokens_2048'])
    
    return row


class ConceptTokenizer:
    """Tokenizer for event concepts using HuggingFace Library."""

    def __init__(
        self,
        pad_token: str = "[PAD]",
        mask_token: str = "[MASK]",
        start_token: str = "[VS]",
        end_token: str = "[VE]",
        class_token: str = "[CLS]",
        reg_token: str = "[REG]",
        unknown_token: str = "[UNK]",
        data_dir: str = "data_files",
        tokenizer_object: Optional[Tokenizer] = None,
        tokenizer: Optional[PreTrainedTokenizerFast] = None,
    ) -> None:
        self.mask_token = mask_token
        self.pad_token = pad_token
        self.unknown_token = unknown_token

        self.task_tokens = ['[MOR_1M]', '[LOS_1W]', '[REA_1M]'] + [f'[C{i}]' for i in range(0, 5)]
        self.tasks = ['mortality_1month', 'los_1week', 'readmission_1month'] + [f'c{i}' for i in range(5)]
        self.task2token = self.create_task_to_token_dict()

        self.special_tokens = (
            [
                pad_token,
                unknown_token,
                mask_token,
                start_token,
                end_token,
                class_token,
                reg_token,
            ]
            + [f"[W_{i}]" for i in range(0, 4)]
            + [f"[M_{i}]" for i in range(0, 13)]
            + ["[LT]"]
        )

        self.tokenizer_object = tokenizer_object
        self.tokenizer = tokenizer

        self.tokenizer_vocab: Dict[str, int] = {}
        self.token_type_vocab: Dict[str, Any] = {}
        self.data_dir = data_dir

        self.special_token_ids: List[int] = []
        self.first_token_index: Optional[int] = None
        self.last_token_index: Optional[int] = None

    def fit_on_vocab(self) -> None:
        """Fit the tokenizer on the vocabulary."""
        # Create dictionary of all possible medical concepts
        self.token_type_vocab["special_tokens"] = self.special_tokens
        vocab_json_files = glob.glob(os.path.join(self.data_dir, "*_vocab.json"))

        for file in vocab_json_files:
            with open(file, "r") as vocab_file:
                vocab = json.load(vocab_file)
                vocab_type = file.split("/")[-1].split(".")[0]
                self.token_type_vocab[vocab_type] = vocab
        
        self.token_type_vocab["task_tokens"] = self.task_tokens

        # Create the tokenizer dictionary
        tokens = list(chain.from_iterable(list(self.token_type_vocab.values())))
        self.tokenizer_vocab = {token: i for i, token in enumerate(tokens)}
        self.special_tokens += self.task_tokens

        # Create the tokenizer object
        self.tokenizer_object = Tokenizer(
            models.WordPiece(
                vocab=self.tokenizer_vocab,
                unk_token=self.unknown_token,
                max_input_chars_per_word=1000,
            ),
        )
        self.tokenizer_object.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        self.tokenizer = self.create_tokenizer(self.tokenizer_object)

        # Get the first, last , and special token indexes from the dictionary
        self.first_token_index = self.get_first_token_index()
        self.last_token_index = self.get_last_token_index()
        self.special_token_ids = self.get_special_token_ids()

        # Check to make sure tokenizer follows the same vocabulary
        assert (
            self.tokenizer_vocab == self.tokenizer.get_vocab()
        ), "Tokenizer vocabulary does not match original"

    def create_tokenizer(
        self,
        tokenizer_obj: Tokenizer,
    ) -> PreTrainedTokenizerFast:
        """Load the tokenizer from a JSON file on disk."""
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer_obj,
            bos_token="[VS]",
            eos_token="[VE]",
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
        )
        return self.tokenizer

    def __call__(
        self,
        batch: Union[str, List[str]],
        return_attention_mask: bool = True,
        return_token_type_ids: bool = False,
        truncation: bool = False,
        padding: str = "max_length",
        max_length: int = 2048,
    ) -> BatchEncoding:
        """Return the tokenized dictionary of input batch."""
        return self.tokenizer(
            batch,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            truncation=truncation,
            padding=padding,
            max_length=max_length,
            return_tensors="pt",
        )

    def encode(self, concept_tokens: str) -> List[int]:
        """Encode the concept tokens into token ids."""
        return self.tokenizer_object.encode(concept_tokens).ids

    def decode(self, concept_ids: List[int]) -> str:
        """Decode the concept sequence token id into token concept."""
        return self.tokenizer_object.decode(concept_ids)

    def token_to_id(self, token: str) -> int:
        """Return the id corresponding to token."""
        return self.tokenizer_object.token_to_id(token)

    def id_to_token(self, token_id: int) -> str:
        """Return the token corresponding to id."""
        return self.tokenizer_object.id_to_token(token_id)

    def get_all_token_indexes(self, with_special_tokens: bool = True) -> Set[int]:
        """Return a set of all possible token ids."""
        all_token_ids = set(self.tokenizer_vocab.values())
        special_token_ids = set(self.get_special_token_ids())

        return (
            all_token_ids if with_special_tokens else all_token_ids - special_token_ids
        )

    def get_first_token_index(self) -> int:
        """Return the smallest token id in vocabulary."""
        return min(self.tokenizer_vocab.values())

    def get_last_token_index(self) -> int:
        """Return the largest token id in vocabulary."""
        return max(self.tokenizer_vocab.values())

    def get_vocab_size(self) -> int:
        """Return the number of possible tokens in vocabulary."""
        return len(self.tokenizer)

    def get_pad_token_id(self) -> int:
        """Return the token id of PAD token."""
        return self.token_to_id(self.pad_token)

    def get_mask_token_id(self) -> int:
        """Return the token id of MASK token."""
        return self.token_to_id(self.mask_token)

    def get_special_token_ids(self) -> List[int]:
        """Get a list of ids representing special tokens."""
        self.special_token_ids = []

        for special_token in self.special_tokens:
            special_token_id = self.token_to_id(special_token)
            self.special_token_ids.append(special_token_id)

        return self.special_token_ids

    def save_tokenizer_to_disk(self, save_dir: str) -> None:
        """Save the tokenizer object to disk as a JSON file."""
        self.tokenizer.save(path=save_dir)

    def create_task_to_token_dict(self) -> None:
        """Create a dictionary mapping each task to its respective special token"""
        task2token = {
            'mortality_1month': '[MOR_1M]',
            'los_1week': '[LOS_1W]',
            'readmission_1month': '[REA_1M]',
        }
        
        for i in range(5):
            task2token[f'c{i}'] = f'[C{i}]'

        return task2token

    def task_to_token(self, task: str) -> str:
        """Return the special token representing task"""
        return self.task2token[task]