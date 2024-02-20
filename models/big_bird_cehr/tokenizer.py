import os
import glob
import json

from itertools import chain
from typing import Sequence, Union, List, Dict, Set

from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers


class HuggingFaceConceptTokenizer:
    """ Tokenizer for event concepts using HuggingFace Library. """

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
    ):
        self.tokenizer = None
        self.mask_token = mask_token
        self.pad_token = pad_token
        self.unknown_token = unknown_token
        self.special_tokens = (
                [pad_token, unknown_token, mask_token, start_token, end_token, class_token, reg_token]
                + [f"[W_{i}]" for i in range(0, 4)]
                + [f"[M_{i}]" for i in range(0, 13)]
                + ["[LT]"]
        )
        self.tokenizer_vocab = {}
        self.token_type_vocab = {}
        self.data_dir = data_dir

    def fit_on_vocab(self) -> None:
        """Fit the tokenizer on the vocabulary."""
        # Create dictionary of all possible medical concepts
        self.token_type_vocab['special_tokens'] = self.special_tokens
        vocab_json_files = glob.glob(os.path.join(self.data_dir, '*_vocab.json'))

        for file in vocab_json_files:
            vocab = json.load(open(file, 'r'))
            vocab_type = file.split("/")[-1].split(".")[0]
            self.token_type_vocab[vocab_type] = vocab

        # Create the tokenizer dictionary
        tokens = list(chain.from_iterable(list(self.token_type_vocab.values())))
        self.tokenizer_vocab = {token: i for i, token in enumerate(tokens)}

        # Create the tokenizer object
        self.tokenizer = Tokenizer(models.WordPiece(vocab=self.tokenizer_vocab,
                                                    unk_token=self.unknown_token,
                                                    max_input_chars_per_word=1000))
        self.tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

        assert self.tokenizer_vocab == self.tokenizer.get_vocab(), "Tokenizer vocabulary does not match original"

    def __call__(
            self,
            batch: Union[str, List[str]],
            return_attention_mask: bool = True,
            return_token_type_ids: bool = False,
            truncation: bool = True,
            padding: bool = True,
            max_length: int = 2048
    ) -> Union[Dict[str, List[int]], List[Dict[str, List[int]]]]:
        """ Return the tokenized dictionary of input batch. """

        return self.tokenizer(
            batch,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            truncation=truncation,
            padding=padding,
            max_length=max_length,
            return_tensors="pt",
        )

    def encode(
            self, concept_sequences: Union[str, List[str]],
    ) -> Union[List[int], List[List[int]]]:
        """Encode the concept sequences into token ids."""
        return self.tokenizer.encode(concept_sequences)

    def decode(
            self, concept_sequence_token_ids: Union[int, List[int]]
    ) -> Union[str, List[str]]:
        """Decode the concept sequence token ids into concepts."""
        return self.tokenizer.decode(concept_sequence_token_ids)

    def get_all_token_indexes(self, with_special_tokens: bool = True) -> Set[int]:
        """ Return a set of all possible token ids """
        all_token_ids = set(self.tokenizer_vocab.values())
        special_token_ids = set(self.get_special_token_ids())

        return all_token_ids if with_special_tokens else all_token_ids - special_token_ids

    def get_first_token_index(self) -> int:
        """ Return the smallest token id in vocabulary """
        return min(self.tokenizer_vocab, key=lambda token: self.tokenizer_vocab[token])

    def get_last_token_index(self) -> int:
        """ Return the largest token id in vocabulary """
        return max(self.tokenizer_vocab, key=lambda token: self.tokenizer_vocab[token])

    def get_vocab_size(self) -> int:
        """ Return the number of possible tokens in vocabulary """
        return len(self.tokenizer)

    def get_pad_token_id(self) -> int:
        """ Return the token id of PAD token. """
        return self.encode(self.pad_token)[0]

    def get_mask_token_id(self) -> int:
        """ Return the token id of MASK token. """
        return self.encode(self.mask_token)[0]

    def get_special_token_ids(self) -> List[int]:
        """ Get a list of ids representing special tokens. """
        special_ids = self.encode(self.special_tokens)
        flat_special_ids = [item[0] for item in special_ids]
        return flat_special_ids

    def save_tokenizer_to_disk(self, save_dir: str) -> None:
        """ Saves the tokenizer object to disk as a JSON file. """
        self.tokenizer.save(path=save_dir)

    def load_tokenizer_from_disk(self, tokenizer_dir: str) -> None:
        """ Loads the tokenizer from a JSON file on disk. """
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_dir,
            bos_token="[VS]",
            eos_token="[VE]",
            unk_token="[UNK]",
            # sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
        )


class ConceptTokenizer:
    """Tokenizer for event concepts."""

    def __init__(
            self,
            pad_token: str = "[PAD]",
            mask_token: str = "[MASK]",
            start_token: str = "[VS]",
            end_token: str = "[VE]",
            class_token: str = "[CLS]",
            oov_token="-1",
            data_dir: str = "data_files",
    ):
        self.tokenizer = Tokenizer(oov_token=oov_token, filters="", lower=False)
        self.mask_token = mask_token
        self.pad_token = pad_token
        self.special_tokens = (
                [pad_token, mask_token, start_token, end_token, class_token]
                + [f"[W_{i}]" for i in range(0, 4)]
                + [f"[M_{i}]" for i in range(0, 13)]
                + ["[LT]"]
        )
        self.data_dir = data_dir

    def fit_on_vocab(self) -> None:
        """Fit the tokenizer on the vocabulary."""
        self.tokenizer.fit_on_texts(self.special_tokens)
        vocab_json_files = glob.glob(os.path.join(self.data_dir, "*_vocab.json"))
        for file in vocab_json_files:
            vocab = json.load(open(file, "r"))
            self.tokenizer.fit_on_texts(vocab)

    def encode(
            self, concept_sequences: Union[str, Sequence[str]], is_generator: bool = False
    ) -> Union[int, Sequence[int]]:
        """Encode the concept sequences into token ids."""
        return (
            self.tokenizer.texts_to_sequences_generator(concept_sequences)
            if is_generator
            else self.tokenizer.texts_to_sequences(concept_sequences)
        )

    def decode(
            self, concept_sequence_token_ids: Union[int, Sequence[int]]
    ) -> Sequence[str]:
        """Decode the concept sequence token ids into concepts."""
        return self.tokenizer.sequences_to_texts(concept_sequence_token_ids)

    def get_all_token_indexes(self) -> set:
        all_keys = set(self.tokenizer.index_word.keys())

        if self.tokenizer.oov_token is not None:
            all_keys.remove(self.tokenizer.word_index[self.tokenizer.oov_token])

        if self.special_tokens is not None:
            excluded = set(
                [
                    self.tokenizer.word_index[special_token]
                    for special_token in self.special_tokens
                ]
            )
            all_keys = all_keys - excluded
        return all_keys

    def get_first_token_index(self) -> int:
        return min(self.get_all_token_indexes())

    def get_last_token_index(self) -> int:
        return max(self.get_all_token_indexes())

    def get_vocab_size(self) -> int:
        # + 1 because oov_token takes the index 0
        return len(self.tokenizer.index_word) + 1

    def get_pad_token_id(self):
        pad_token_id = self.encode([self.pad_token])
        while isinstance(pad_token_id, list):
            pad_token_id = pad_token_id[0]
        return pad_token_id

    def get_mask_token_id(self):
        mask_token_id = self.encode([self.mask_token])
        while isinstance(mask_token_id, list):
            mask_token_id = mask_token_id[0]
        return mask_token_id

    def get_special_token_ids(self):
        special_ids = self.encode(self.special_tokens)
        flat_special_ids = [item[0] for item in special_ids]
        return flat_special_ids
