import glob
import json
import os
from typing import Sequence, Union

from keras.preprocessing.text import Tokenizer


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
        vocab_json_files = glob.glob(os.path.join(self.data_dir, "*_vocab.json"))
        for file in vocab_json_files:
            vocab = json.load(open(file, "r"))
            self.tokenizer.fit_on_texts(vocab)
        self.tokenizer.fit_on_texts(self.special_tokens)

    def encode(
        self,
        concept_sequences: Union[str, Sequence[str]],
        is_generator: bool = False,
    ) -> Union[int, Sequence[int]]:
        """Encode the concept sequences into token ids."""
        return (
            self.tokenizer.texts_to_sequences_generator(concept_sequences)
            if is_generator
            else self.tokenizer.texts_to_sequences(concept_sequences)
        )

    def decode(
        self,
        concept_sequence_token_ids: Union[int, Sequence[int]],
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
                ],
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
        pad_token_id = self.encode(self.pad_token)
        while isinstance(pad_token_id, list):
            pad_token_id = pad_token_id[0]
        return pad_token_id

    def get_mask_token_id(self):
        mask_token_id = self.encode(self.mask_token)
        while isinstance(mask_token_id, list):
            mask_token_id = mask_token_id[0]
        return mask_token_id

    def get_special_token_ids(self):
        special_ids = self.encode(self.special_tokens)
        flat_special_ids = [item[0] for item in special_ids]
        return flat_special_ids
