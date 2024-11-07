"""Tokenizer module."""

import glob
import json
import os
import re
from itertools import chain
from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import BatchEncoding, PreTrainedTokenizerFast


class ConceptTokenizer:
    """Tokenizer for event concepts using HuggingFace Library.

    Parameters
    ----------
    pad_token: str
        Padding token.
    mask_token: str
        Mask token.
    start_token: str
        Sequence Start token. Models can use class token instead.
    end_token: str
        Sequence End token.
    class_token: str
        Class token.
    reg_token: str
        Registry token.
    unknown_token: str
        Unknown token.
    data_dir: str
        Directory containing the data.
    time_tokens: List[str]
        List of time-related special tokens.
    tokenizer_object: Optional[Tokenizer]
        Tokenizer object.
    tokenizer: Optional[PreTrainedTokenizerFast]
        Tokenizer object.
    padding_side: str
        Padding side.

    Attributes
    ----------
    pad_token: str
        Padding token.
    mask_token: str
        Mask token.
    start_token: str
        Sequence Start token.
    end_token: str
        Sequence End token.
    class_token: str
        Class token.
    reg_token: str
        Registry token.
    unknown_token: str
        Unknown token.
    task_tokens: List[str]
        List of task-specific tokens.
    tasks: List[str]
        List of task names.
    task2token: Dict[str, str]
        Dictionary mapping task names to tokens.
    special_tokens: List[str]
        Special tokens including padding, mask, start, end, class, registry tokens.
    tokenizer: PreTrainedTokenizerFast
        HuggingFace fast tokenizer object.
    tokenizer_object: Tokenizer
        Tokenizer object from tokenizers library.
    tokenizer_vocab: Dict[str, int]
        Vocabulary mapping tokens to indices.
    token_type_vocab: Dict[str, Any]
        Vocabulary for token types.
    data_dir: str
        Directory containing data files.

    """

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
        time_tokens: List[str] = [f"[W_{i}]" for i in range(0, 4)]
        + [f"[M_{i}]" for i in range(0, 13)]
        + ["[LT]"],
        tokenizer_object: Optional[Tokenizer] = None,
        tokenizer: Optional[PreTrainedTokenizerFast] = None,
        padding_side: str = "right",
    ) -> None:
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.start_token = start_token
        self.end_token = end_token
        self.class_token = class_token
        self.reg_token = reg_token
        self.unknown_token = unknown_token
        self.padding_side = padding_side

        self.task_tokens = ["[MOR_1M]", "[LOS_1W]", "[REA_1M]"] + [
            f"[C{i}]" for i in range(0, 5)
        ]
        self.tasks = ["mortality_1month", "los_1week", "readmission_1month"] + [
            f"c{i}" for i in range(5)
        ]
        self.task2token = self.create_task_to_token_dict()

        self.special_tokens = [
            pad_token,
            unknown_token,
            mask_token,
            start_token,
            end_token,
            class_token,
            reg_token,
        ]
        if time_tokens is not None:
            self.special_tokens += time_tokens

        self.tokenizer_object = tokenizer_object
        self.tokenizer = tokenizer

        self.tokenizer_vocab: Dict[str, int] = {}
        self.token_type_vocab: Dict[str, Any] = {}
        self.data_dir = data_dir

        self.special_token_ids: List[int] = []
        self.first_token_index: Optional[int] = None
        self.last_token_index: Optional[int] = None

    def fit_on_vocab(self, with_tasks: bool = True) -> None:
        """Fit the tokenizer on the vocabulary."""
        # Create dictionary of all possible medical concepts
        self.token_type_vocab["special_tokens"] = self.special_tokens
        vocab_json_files = glob.glob(os.path.join(self.data_dir, "*vocab.json"))

        for file in vocab_json_files:
            with open(file, "r") as vocab_file:
                vocab = json.load(vocab_file)
                vocab_type = file.split("/")[-1].split(".")[0]
                self.token_type_vocab[vocab_type] = vocab

        if with_tasks:
            self.token_type_vocab["task_tokens"] = self.task_tokens

        # Create the tokenizer dictionary
        tokens = list(chain.from_iterable(list(self.token_type_vocab.values())))
        self.tokenizer_vocab = {token: i for i, token in enumerate(tokens)}

        # Update special tokens after tokenizr is created
        if with_tasks:
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

        # Set token to label dict backbone
        self.token_to_label_dict = {
            token: token for token in self.tokenizer_vocab.keys()
        }

        # Check to make sure tokenizer follows the same vocabulary
        assert (
            self.tokenizer_vocab == self.tokenizer.get_vocab()
        ), "Tokenizer vocabulary does not match original"

    def load_token_labels(self, codes_dir: str) -> Dict[str, str]:
        """
        Load and merge JSON files containing medical codes and names.
        Update the token_to_label_dict with new token to label mappings.

        Parameters
        ----------
        codes_dir : str
            The directory path that contains JSON files with code mappings.

        Returns
        -------
        Dict[str, str]
            A dictionary that represents a medical concept code mapping.
        """
        merged_dict = {}
        for filename in os.listdir(codes_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(codes_dir, filename)
                with open(filepath, "r") as file:
                    data = json.load(file)
                    merged_dict.update(data)

        for token, label in merged_dict.items():
            self.token_to_label_dict[token] = label

        return merged_dict

    def create_tokenizer(
        self,
        tokenizer_obj: Tokenizer,
    ) -> PreTrainedTokenizerFast:
        """Load the tokenizer from a JSON file on disk.

        Parameters
        ----------
        tokenizer_obj: Tokenizer
            Tokenizer object.

        Returns
        -------
        PreTrainedTokenizerFast
            Tokenizer object.

        """
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer_obj,
            bos_token=self.start_token,
            eos_token=self.end_token,
            unk_token=self.unknown_token,
            pad_token=self.pad_token,
            cls_token=self.class_token,
            mask_token=self.mask_token,
            padding_side=self.padding_side,
        )
        return self.tokenizer

    def __call__(
        self,
        batch: Union[str, List[str]],
        return_attention_mask: bool = True,
        return_token_type_ids: bool = False,
        truncation: bool = True,
        padding: str = "max_length",
        max_length: int = 2048,
    ) -> BatchEncoding:
        """Return the tokenized dictionary of input batch.

        Parameters
        ----------
        batch: Union[str, List[str]]
            Input batch.
        return_attention_mask: bool
            Return attention mask.
        return_token_type_ids: bool
            Return token type ids.
        truncation: bool
            Truncate the input.
        padding: str
            Padding strategy.
        max_length: int
            Maximum length of the input.

        Returns
        -------
        BatchEncoding
            Tokenized dictionary of input batch.

        """
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
        """Encode the concept tokens into token ids.

        Parameters
        ----------
        concept_tokens: str
            Concept tokens.

        Returns
        -------
        List[int]
            Token ids.

        """
        return self.tokenizer_object.encode(concept_tokens).ids

    def decode(self, concept_ids: List[int]) -> str:
        """Decode the concept sequence token id into token concept.

        Parameters
        ----------
        concept_ids: List[int]
            Concept ids.

        Returns
        -------
        str
            Concept sequence.

        """
        return self.tokenizer_object.decode(concept_ids)

    def decode_to_labels(self, concept_input: Union[List[str], List[int]]) -> List[str]:
        """Decode the concept sequence tokens or ids into labels.

        Parameters
        ----------
        concept_input: Union[List[str], List[int]]
            Concept tokens or ids.

        Returns
        -------
        List[str]
            A new sequence with medical concept codes replaced by their labels.

        """
        if isinstance(concept_input[0], int):
            concept_input = [self.id_to_token(token_id) for token_id in concept_input]

        decoded_sequence = []
        for item in concept_input:
            match = re.match(r"^(.*?)(_\d)$", item)
            if match:
                base_part, suffix = match.groups()
                replaced_item = (
                    self.token_to_label_dict.get(base_part, base_part) + suffix
                )
            else:
                replaced_item = self.token_to_label_dict.get(item, item)
            decoded_sequence.append(replaced_item)

        return decoded_sequence

    def token_to_id(self, token: str) -> int:
        """Return the id corresponding to token.

        Parameters
        ----------
        token: str
            Token.

        Returns
        -------
        int
            Token id.

        """
        return self.tokenizer_object.token_to_id(token)

    def id_to_token(self, token_id: int) -> str:
        """Return the token corresponding to id.

        Parameters
        ----------
        token_id: int
            Token id.

        Returns
        -------
        str
            Token.

        """
        return self.tokenizer_object.id_to_token(token_id)

    def token_to_label(self, token: str) -> str:
        """Return the label corresponding to token.

        Parameters
        ----------
        token: str
            Token.

        Returns
        -------
        str
            Label.

        """
        return self.decode_to_labels([token])[0]

    def get_all_token_indexes(self, with_special_tokens: bool = True) -> Set[int]:
        """Return a set of all possible token ids.

        Parameters
        ----------
        with_special_tokens: bool
            Include special tokens.

        Returns
        -------
        Set[int]
            Set of all token ids.

        """
        all_token_ids = set(self.tokenizer_vocab.values())
        special_token_ids = set(self.get_special_token_ids())

        return (
            all_token_ids if with_special_tokens else all_token_ids - special_token_ids
        )

    def get_first_token_index(self) -> int:
        """Return the smallest token id in vocabulary.

        Returns
        -------
        int
            First token id.

        """
        return min(self.tokenizer_vocab.values())

    def get_last_token_index(self) -> int:
        """Return the largest token id in vocabulary.

        Returns
        -------
        int
            Largest token id.

        """
        return max(self.tokenizer_vocab.values())

    def get_vocab_size(self) -> int:
        """Return the number of possible tokens in vocabulary.

        Returns
        -------
        int
            Number of tokens in vocabulary.

        """
        return len(self.tokenizer)

    def get_class_token_id(self) -> int:
        """Return the token id of CLS token.

        Returns
        -------
        int
            Token id of CLS token.

        """
        return self.token_to_id(self.class_token)

    def get_pad_token_id(self) -> int:
        """Return the token id of PAD token.

        Returns
        -------
        int
            Token id of PAD token.

        """
        return self.token_to_id(self.pad_token)

    def get_mask_token_id(self) -> int:
        """Return the token id of MASK token.

        Returns
        -------
        int
            Token id of MASK token.

        """
        return self.token_to_id(self.mask_token)

    def get_eos_token_id(self) -> int:
        """Return the token id of end of sequence token.

        Returns
        -------
        int
            Token id of end of sequence token.

        """
        return self.token_to_id(self.end_token)

    def get_special_token_ids(self) -> List[int]:
        """Get a list of ids representing special tokens.

        Returns
        -------
        List[int]
            List of special token ids.

        """
        self.special_token_ids = []

        for special_token in self.special_tokens:
            special_token_id = self.token_to_id(special_token)
            self.special_token_ids.append(special_token_id)

        return self.special_token_ids

    def save(self, save_dir: str) -> None:
        """Save the tokenizer object to disk as a JSON file.

        Parameters
        ----------
        save_dir: str
            Directory to save the tokenizer.

        """
        os.makedirs(save_dir, exist_ok=True)
        tokenizer_config = {
            "pad_token": self.pad_token,
            "mask_token": self.mask_token,
            "unknown_token": self.unknown_token,
            "start_token": self.start_token,
            "end_token": self.end_token,
            "class_token": self.class_token,
            "reg_token": self.reg_token,
            "special_tokens": self.special_tokens,
            "tokenizer_vocab": self.tokenizer_vocab,
            "token_type_vocab": self.token_type_vocab,
            "data_dir": self.data_dir,
        }
        save_path = os.path.join(save_dir, "tokenizer.json")
        with open(save_path, "w") as file:
            json.dump(tokenizer_config, file, indent=4)

    @classmethod
    def load(cls, load_path: str) -> "ConceptTokenizer":
        """
        Load the tokenizer configuration from a file.

        Parameters
        ----------
        load_path : str
            The path from where the tokenizer configuration will be loaded.

        Returns
        -------
        ConceptTokenizer
            An instance of ConceptTokenizer initialized with the loaded configuration.
        """
        with open(load_path, "r") as file:
            tokenizer_config = json.load(file)

        tokenizer = cls(
            pad_token=tokenizer_config["pad_token"],
            mask_token=tokenizer_config["mask_token"],
            unknown_token=tokenizer_config["unknown_token"],
            start_token=tokenizer_config["start_token"],
            end_token=tokenizer_config["end_token"],
            class_token=tokenizer_config["class_token"],
            reg_token=tokenizer_config["reg_token"],
            data_dir=tokenizer_config["data_dir"],
        )

        tokenizer.special_tokens = tokenizer_config["special_tokens"]
        tokenizer.tokenizer_vocab = tokenizer_config["tokenizer_vocab"]
        tokenizer.token_type_vocab = tokenizer_config["token_type_vocab"]

        tokenizer.tokenizer_object = Tokenizer(
            models.WordPiece(
                vocab=tokenizer.tokenizer_vocab,
                unk_token=tokenizer.unknown_token,
                max_input_chars_per_word=1000,
            ),
        )
        tokenizer.tokenizer_object.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        tokenizer.tokenizer = tokenizer.create_tokenizer(tokenizer.tokenizer_object)

        return tokenizer

    def create_task_to_token_dict(self) -> Dict[str, str]:
        """Create a dictionary mapping each task to its respective special token.

        Returns
        -------
        Dict[str, str]
            Dictionary mapping each task to its respective special token

        """
        task2token = {
            "mortality_1month": "[MOR_1M]",
            "los_1week": "[LOS_1W]",
            "readmission_1month": "[REA_1M]",
        }
        for i in range(5):
            task2token[f"c{i}"] = f"[C{i}]"

        return task2token

    def task_to_token(self, task: str) -> str:
        """Return the special token representing task.

        Parameters
        ----------
        task: str
            Task name

        Returns
        -------
        str
            Special token representing task

        """
        return self.task2token[task]

    @staticmethod
    def create_vocab_from_sequences(
        sequences: Union[List[List[str]], pd.Series],
        save_path: str,
    ) -> None:
        """Create a vocabulary from sequences and save it as a JSON file.

        This static method takes a list of lists or a Pandas Series containing sequences,
        extracts unique tokens, sorts them, and saves them in a JSON file.

        Parameters
        ----------
        sequences : Union[List[List[str]], pd.Series]
            List of lists or a Pandas Series containing sequences of tokens (strings).
        save_path : str
            The file path to save the vocabulary JSON.

        """
        # Flatten the list of lists and extract unique tokens
        unique_tokens = sorted(
            set(token for sequence in sequences for token in sequence)
        )

        # Raise error if a token has space in it
        if any(" " in token for token in unique_tokens):
            raise ValueError("Tokens should not contain spaces.")

        with open(save_path, "w") as vocab_file:
            json.dump(unique_tokens, vocab_file, indent=4)
