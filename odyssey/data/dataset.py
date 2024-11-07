"""Data module for pretraining and finetuning models."""

import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from odyssey.data.tokenizer import ConceptTokenizer


# Index of features in tuples used in multi-task datasets
TASK_INDEX, LABEL_INDEX, CUTOFF_INDEX = 1, 2, 3

# Mapping of tasks to indices for multi-task datasets
TASK_TO_INDEX = {
    "mortality_1month": 0,
    "readmission_1month": 1,
    "los_1week": 2,  # Length of stay
    "c0": 3,  # Condition 0
    "c1": 4,
    "c2": 5,
}

# Mapping of additional token types to column names in the dataset
AUGMENTED_TOKEN_MAP = {
    "type_ids": "type_tokens",
    "ages": "age_tokens",
    "time_stamps": "time_tokens",
    "visit_orders": "position_tokens",
    "visit_segments": "visit_tokens",
}


class BaseDataset(Dataset, ABC):
    """Base class for datasets used in pretraining and finetuning.

    Parameters
    ----------
    data : pd.DataFrame
        The input data containing sequences to be tokenized.
    tokenizer : ConceptTokenizer
        An instance of the ConceptTokenizer class used for tokenizing sequences.
    max_len : int, optional
        The maximum length of the tokenized sequences, by default 2048.
    padding_side : str, optional
        Which side to add padding on - "left" or "right". Default is "right".
    """

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: ConceptTokenizer,
        max_len: int = 2048,
        padding_side: str = "right",
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.padding_side = padding_side
        self.cutoff_col = next(
            (col for col in self.data.columns if "cutoff" in col), None
        )

    def truncate_and_pad(
        self,
        row: pd.Series,
        cutoff: Optional[int] = None,
        additional_columns: Optional[List[str]] = None,
    ) -> pd.Series:
        """Truncate and pad sequences in the row.

        Parameters
        ----------
        row : pd.Series
            Input row containing sequences
        cutoff : Optional[int]
            Length to truncate sequences to. If None, uses length of event_tokens
            up to max_len
        additional_columns : Optional[List[str]]
            Additional columns to truncate and pad besides event_tokens

        Returns
        -------
        pd.Series
            Row with truncated and padded sequences
        """
        row = row.copy()

        # Determine cutoff length
        if cutoff is None:
            cutoff = min(self.max_len, len(row["event_tokens"]))
        cutoff = min(cutoff, self.max_len)

        # Truncate and pad tokens
        row["event_tokens"] = row["event_tokens"][:cutoff]

        if additional_columns:
            for col in additional_columns:
                if col in row:
                    pad_width = (
                        (self.max_len - cutoff, 0)
                        if self.padding_side == "left"
                        else (0, self.max_len - cutoff)
                    )
                    row[col] = np.pad(
                        row[col][:cutoff], pad_width, mode="constant", constant_values=0
                    )

        # Join event tokens for tokenizer
        row["event_tokens"] = " ".join(row["event_tokens"])

        return row

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    def tokenize_data(self, sequence: Union[str, List[str]]) -> Any:
        """Tokenize the sequence."""
        return self.tokenizer(sequence, max_length=self.max_len)

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get data at corresponding index."""
        pass


class AugmentedTokenizationMixin:
    """Mixin class for adding additional token types to the dataset."""

    def add_additional_tokens(
        self, data: pd.Series, additional_token_types: List[str] = None
    ) -> Dict[str, torch.Tensor]:
        """Add specified additional token types to the dataset.

        Parameters
        ----------
        data : pd.Series
            A series containing token sequences and additional information.
        additional_token_types : List[str], optional
            A list specifying which token types to add (e.g., ["type_ids", "ages"]).
            If None, all available tokens will be added.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing tensors for each specified additional token type.
        """
        if additional_token_types is None:
            return {}  # Return an empty dictionary if no token types are specified

        return {
            token_name: torch.tensor(data[column_name])
            for token_name, column_name in AUGMENTED_TOKEN_MAP.items()
            if token_name in additional_token_types
        }


class MaskingMixin:
    """Mixin class for masking tokens in the dataset."""

    def mask_tokens(self, sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mask the tokens in the sequence using vectorized operations.

        Parameters
        ----------
        sequence : torch.Tensor
            The sequence of tokens to be masked.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing masked sequence and labels.

        """
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


class MultiTaskMixin:
    """Mixin class for handling multi-task datasets.

    Parameters
    ----------
    tasks : List[str]
        A list of tasks for which the model is being trained.

    Attributes
    ----------
    tasks : List[str]
        A list of tasks for which the model is being trained.
    task_to_index : Dict[str, List[Tuple[int, str, int, Optional[int]]]]
        A dictionary mapping each task to a list of tuples containing the
        index, task, label, and cutoff.
    index_mapper : List[Tuple[int, str, int, Optional[int]]]
        A list of all datapoints to be used by __getitem__.
    """

    def __init__(self, tasks: List[str]):
        self.tasks = tasks
        self.task_to_index = {task: [] for task in self.tasks}
        self.index_mapper = []

    def prepare_multi_task_data(self) -> None:
        """Prepare multi-task data by mapping tasks to corresponding indices.

        This method precomputes indices for quick mapping in __getitem__ that
        exclude missing labels. It helps in filtering out entries where the
        label is missing for the specified tasks.
        """
        self.data.reset_index(drop=True, inplace=True)

        for patient in self.data.itertuples():
            index = patient.Index

            for task in self.tasks:
                label_col = f"label_{task}"
                # Skip this task for the current patient if the label is missing.
                if getattr(patient, label_col) == self.nan_indicator:
                    continue

                label = getattr(patient, label_col)
                # Check for the existence of a task-specific cutoff in the data,
                # else use None.
                if f"cutoff_{task}" in self.data.columns:
                    cutoff = getattr(patient, f"cutoff_{task}")
                else:
                    cutoff = None
                # Append a tuple containing the necessary information
                # for training to index_mapper.
                datapoint = (index, task, label, cutoff)
                self.task_to_index[task].append(datapoint)

        # Create a list of all datapoints to be used by __getitem__
        self.index_mapper = [
            datapoints
            for task_data in self.task_to_index.values()
            for datapoints in task_data
        ]


class LabelBalanceMixin:
    """Mixin class for balancing labels in the dataset."""

    def balance_labels(self, balance_guide: Optional[Dict[str, float]] = None) -> None:
        """Balance the labels for the specified tasks in the dataset.

        Parameters
        ----------
        balance_guide : Optional[Dict[str, float]]
            A dictionary containing the desired positive ratios for each task.
        """
        if not balance_guide:
            return

        for task, positive_ratio in balance_guide.items():
            # Separate positive and negative datapoints
            datapoints = self.task_to_index[task]
            positives = [data for data in datapoints if data[LABEL_INDEX] == 1]
            negatives = [data for data in datapoints if data[LABEL_INDEX] == 0]

            # Calculate the total number of samples needed to achieve the
            # desired positive ratio
            num_positives = len(positives)
            total_needed = int(num_positives / positive_ratio) - num_positives
            num_negatives_to_keep = min(len(negatives), total_needed)

            # Randomly select the negatives to keep
            negatives_kept = random.sample(negatives, num_negatives_to_keep)

            # Combine the kept negatives with all positives
            self.task_to_index[task] = positives + negatives_kept


class PretrainDataset(BaseDataset, AugmentedTokenizationMixin, MaskingMixin):
    """Dataset for pretraining the model.

    Parameters
    ----------
    data : pd.DataFrame
        The input data containing sequences to be tokenized and masked.
    tokenizer : ConceptTokenizer
        An instance of the ConceptTokenizer class used for tokenizing sequences.
    max_len : int, optional
        The maximum length of the tokenized sequences, by default 2048.
    mask_prob : float, optional
        The probability of masking a token in the sequence, by default 0.15.
    additional_token_types : List[str], optional
        A list of additional token types to be added to the dataset, by default None.
    padding_side : str, optional
        Which side to add padding on - "left" or "right". Default is "right".

    Attributes
    ----------
    data : pd.DataFrame
        Stores the input data.
    tokenizer : ConceptTokenizer
        Tokenizer used for tokenizing sequences.
    max_len : int
        Maximum length of the tokenized sequences.
    mask_prob : float
        Probability of masking a token in the sequence.
    additional_token_types : List[str], optional
        A list of additional token types to be added to the dataset, by default None.
    padding_side : str, optional
        Which side to add padding on - "left" or "right". Default is "right".
    """

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: ConceptTokenizer,
        max_len: int = 2048,
        mask_prob: float = 0.15,
        additional_token_types: List[str] = None,
        padding_side: str = "right",
    ):
        super().__init__(data, tokenizer, max_len, padding_side)
        self.mask_prob = mask_prob
        self.additional_token_types = additional_token_types

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get data at corresponding index.

        Parameters
        ----------
        idx : int
            The index of the data to be retrieved.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing all different token sequences along with
            attention mask and labels.
        """
        data = self.data.iloc[idx]
        cutoff = data[self.cutoff_col] if self.cutoff_col else None
        data = self.truncate_and_pad(
            row=data,
            cutoff=cutoff,
            additional_columns=self.additional_token_types,
        )

        # Tokenize and mask the input data
        tokenized_input = self.tokenize_data(data["event_tokens"])
        concept_tokens = tokenized_input["input_ids"].squeeze()
        attention_mask = tokenized_input["attention_mask"].squeeze()
        masked_tokens, labels = self.mask_tokens(concept_tokens)

        # Prepare model input
        tokens = self.add_additional_tokens(data, self.additional_token_types)
        tokens["concept_ids"] = masked_tokens
        tokens["labels"] = labels
        tokens["attention_mask"] = attention_mask

        return tokens


class PretrainDatasetDecoder(BaseDataset, AugmentedTokenizationMixin):
    """Dataset for pretraining a decoder-based model (e.g. Mamba).

    The decoder is trained using the next token prediction task.

    Parameters
    ----------
    data : pd.DataFrame
        The input data containing sequences to be tokenized.
    tokenizer : ConceptTokenizer
        An instance of the ConceptTokenizer class used for tokenizing sequences.
    max_len : int, optional
        The maximum length of the tokenized sequences, by default 2048.
    additional_token_types : List[str], optional
        A list of additional token types to be added to the dataset, by default None.
    padding_side : str, optional
        Which side to add padding on - "left" or "right". Default is "right".
    return_attention_mask : bool, optional
        Whether to return the attention mask or not. Default is True.

    Attributes
    ----------
    data : pd.DataFrame
        Stores the input data.
    tokenizer : ConceptTokenizer
        Tokenizer used for tokenizing sequences.
    max_len : int
        Maximum length of the tokenized sequences.
    additional_token_types : List[str], optional
        A list of additional token types to be added to the dataset, by default None.
    padding_side : str, optional
        Which side to add padding on - "left" or "right". Default is "right".
    return_attention_mask : bool
        Whether to return the attention mask or not.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: ConceptTokenizer,
        max_len: int = 2048,
        additional_token_types: List[str] = None,
        padding_side: str = "right",
        return_attention_mask: bool = True,
    ):
        super().__init__(data, tokenizer, max_len, padding_side)
        self.additional_token_types = additional_token_types
        self.return_attention_mask = return_attention_mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get data at corresponding index.

        Parameters
        ----------
        idx : int
            The index of the data to be retrieved.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing all different token sequences along with
            attention mask and labels.
        """
        data = self.data.iloc[idx]
        cutoff = data[self.cutoff_col] if self.cutoff_col else None
        data = self.truncate_and_pad(
            row=data, cutoff=cutoff, additional_columns=self.additional_token_types
        )
        tokenized_input = self.tokenize_data(data["event_tokens"])

        # Prepare model input
        tokens = self.add_additional_tokens(data, self.additional_token_types)
        tokens["concept_ids"] = tokenized_input["input_ids"].squeeze()
        tokens["labels"] = tokens["concept_ids"]

        if self.return_attention_mask:
            tokens["attention_mask"] = tokenized_input["attention_mask"].squeeze()

        return tokens


class FinetuneDataset(BaseDataset, AugmentedTokenizationMixin):
    """Dataset for finetuning the model.

    Parameters
    ----------
    data : pd.DataFrame
        The input data containing sequences to be tokenized.
    tokenizer : ConceptTokenizer
        An instance of the ConceptTokenizer class used for tokenizing sequences.
    max_len : int, optional
        The maximum length of the tokenized sequences, by default 2048.
    additional_token_types : List[str], optional
        A list of additional token types to be added to the dataset, by default None.
    padding_side : str, optional
        Which side to add padding on - "left" or "right". Default is "right".

    Attributes
    ----------
    data : pd.DataFrame
        Stores the input data.
    tokenizer : ConceptTokenizer
        Tokenizer used for tokenizing sequences.
    max_len : int
        Maximum length of the tokenized sequences.
    additional_token_types : List[str], optional
        A list of additional token types to be added to the dataset, by default None.
    padding_side : str, optional
        Which side to add padding on - "left" or "right". Default is "right".
    """

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: ConceptTokenizer,
        max_len: int = 2048,
        additional_token_types: List[str] = None,
        padding_side: str = "right",
    ):
        super().__init__(data, tokenizer, max_len, padding_side)
        self.additional_token_types = additional_token_types

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get data at corresponding index.

        Parameters
        ----------
        idx : int
            The index of the data to be retrieved.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing all different token sequences along with
            attention mask and labels.
        """
        data = self.data.iloc[idx]
        cutoff = data[self.cutoff_col] if self.cutoff_col else None
        data = self.truncate_and_pad(
            row=data, cutoff=cutoff, additional_columns=self.additional_token_types
        )
        tokenized_input = self.tokenize_data(data["event_tokens"])

        # Prepare model input
        tokens = self.add_additional_tokens(data, self.additional_token_types)
        tokens["concept_ids"] = tokenized_input["input_ids"].squeeze()
        tokens["attention_mask"] = tokenized_input["attention_mask"].squeeze()
        tokens["labels"] = torch.tensor(data["label"])

        return tokens


class FinetuneMultiDataset(
    BaseDataset, AugmentedTokenizationMixin, MultiTaskMixin, LabelBalanceMixin
):
    """Dataset for finetuning the model on multiple tasks.

    Parameters
    ----------
    data : pd.DataFrame
        The input data containing sequences to be tokenized.
    tokenizer : ConceptTokenizer
        An instance of the ConceptTokenizer class used for tokenizing sequences.
    tasks : List[str]
        A list of tasks (labels) for which the model is being finetuned.
    balance_guide : Optional[Dict[str, float]], optional
        A dictionary containing the desired positive ratios for each task,
        by default None.
    max_len : int, optional
        The maximum length of the tokenized sequences, by default 2048.
    nan_indicator : int, optional
        Value used to represent missing labels in the dataset, by default -1.
    additional_token_types : List[str], optional
        A list of additional token types to be added to the dataset, by default None.
    padding_side : str, optional
        Which side to add padding on - "left" or "right". Default is "right".

    Attributes
    ----------
    data : pd.DataFrame
        Stores the input data.
    tokenizer : ConceptTokenizer
        Tokenizer used for tokenizing sequences.
    tasks : List[str]
        A list of tasks (labels) for which the model is being finetuned.
    balance_guide : Optional[Dict[str, float]]
        A dictionary containing the desired positive ratios for each task.
    max_len : int
        Maximum length of the tokenized sequences.
    nan_indicator : int
        Value used to represent missing labels in the dataset.
    task_to_index : Dict[str, List[Tuple[int, str, int, Optional[int]]]]
        A dictionary mapping each task to a list of tuples containing the
        index, task, label, and cutoff.
    index_mapper : List[Tuple[int, str, int, Optional[int]]]
        A list of all datapoints to be used by __getitem__.
    additional_token_types : List[str], optional
        A list of additional token types to be added to the dataset, by default None.
    padding_side : str, optional
        Which side to add padding on - "left" or "right". Default is "right".
    """

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: ConceptTokenizer,
        tasks: List[str],
        balance_guide: Optional[Dict[str, float]] = None,
        max_len: int = 2048,
        nan_indicator: int = -1,
        additional_token_types: List[str] = None,
        padding_side: str = "right",
    ):
        BaseDataset.__init__(self, data, tokenizer, max_len, padding_side)
        MultiTaskMixin.__init__(self, tasks)
        self.nan_indicator = nan_indicator
        self.balance_guide = balance_guide
        self.additional_token_types = additional_token_types
        self.prepare_multi_task_data()
        self.balance_labels(self.balance_guide)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.index_mapper)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get data at corresponding index.

        Parameters
        ----------
        idx : int
            The index of the data to be retrieved.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing all different token sequences along with
            attention mask and labels.
        """
        index, task, label, cutoff = self.index_mapper[idx]
        data = self.data.iloc[index]

        # Swap the first token with the task token.
        data["event_tokens"][0] = self.tokenizer.task_to_token(task)
        data = self.truncate_and_pad(
            row=data, cutoff=cutoff, additional_columns=self.additional_token_types
        )
        tokenized_input = self.tokenize_data(data["event_tokens"])

        # Prepare model input
        tokens = self.add_additional_tokens(data, self.additional_token_types)
        tokens["concept_ids"] = tokenized_input["input_ids"].squeeze()
        tokens["attention_mask"] = tokenized_input["attention_mask"].squeeze()
        tokens["labels"] = torch.tensor(label)
        tokens["task"] = task

        return tokens


class FinetuneDatasetDecoder(
    BaseDataset, AugmentedTokenizationMixin, MultiTaskMixin, LabelBalanceMixin
):
    """Dataset for finetuning a decoder-based model.

    Parameters
    ----------
    data : pd.DataFrame
        The input data containing sequences to be tokenized.
    tokenizer : ConceptTokenizer
        An instance of the ConceptTokenizer class used for tokenizing sequences.
    tasks : List[str]
        A list of tasks (labels) for which the model is being finetuned.
    balance_guide : Optional[Dict[str, float]], optional
        A dictionary containing the desired positive ratios for each task,
        by default None.
    max_len : int, optional
        The maximum length of the tokenized sequences, by default 2048.
    nan_indicator : int, optional
        Value used to represent missing labels in the dataset, by default -1.
    is_single_head : bool, optional
        Indicating if the model uses one head for all classifications or not.
    additional_token_types : List[str], optional
        A list of additional token types to be added to the dataset, by default None.
    padding_side : str, optional
        Which side to add padding on - "left" or "right". Default is "right".
    return_attention_mask : bool, optional
        Whether to return the attention mask or not. Default is True.

    Attributes
    ----------
    data : pd.DataFrame
        Stores the input data.
    tokenizer : ConceptTokenizer
        Tokenizer used for tokenizing sequences.
    tasks : List[str]
        A list of tasks (labels) for which the model is being finetuned.
    balance_guide : Optional[Dict[str, float]]
        A dictionary containing the desired positive ratios for each task.
    max_len : int
        Maximum length of the tokenized sequences.
    nan_indicator : int
        Value used to represent missing labels in the dataset.
    is_single_head : bool
        Indicating if the model uses one head for all classifications or not.
    task_to_index : Dict[str, List[Tuple[int, str, int, Optional[int]]]]
        A dictionary mapping each task to a list of tuples containing the
        index, task, label, and cutoff.
    index_mapper : List[Tuple[int, str, int, Optional[int]]]
        A list of all datapoints to be used by __getitem__.
    additional_token_types : List[str], optional
        A list of additional token types to be added to the dataset, by default None.
    padding_side : str, optional
        Which side to add padding on - "left" or "right". Default is "right".
    return_attention_mask : bool
        Whether to return the attention mask or not.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: ConceptTokenizer,
        tasks: List[str],
        balance_guide: Optional[Dict[str, float]] = None,
        max_len: int = 2048,
        nan_indicator: int = -1,
        is_single_head: bool = True,
        additional_token_types: List[str] = None,
        padding_side: str = "right",
        return_attention_mask: bool = True,
    ):
        BaseDataset.__init__(self, data, tokenizer, max_len, padding_side)
        MultiTaskMixin.__init__(self, tasks)
        self.nan_indicator = nan_indicator
        self.is_single_head = is_single_head
        self.balance_guide = balance_guide
        self.additional_token_types = additional_token_types
        self.return_attention_mask = return_attention_mask
        self.prepare_multi_task_data()
        self.balance_labels(balance_guide)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.index_mapper)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get data at corresponding index.

        Parameters
        ----------
        idx : int
            The index of the data to be retrieved.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing all different token sequences along with labels.
        """
        index, task, label, cutoff = self.index_mapper[idx]
        data = self.data.iloc[index]

        # Swap the first and last token with the task token.
        if self.is_single_head:
            data["event_tokens"][0] = self.tokenizer.task_to_token(task)
            data["event_tokens"][-1] = self.tokenizer.task_to_token(task)
        else:
            data["event_tokens"][-1] = data["event_tokens"][0]

        data = self.truncate_and_pad(
            row=data, cutoff=cutoff, additional_columns=self.additional_token_types
        )
        tokenized_input = self.tokenize_data(data["event_tokens"])

        # Prepare model input
        tokens = self.add_additional_tokens(data, self.additional_token_types)
        tokens["concept_ids"] = tokenized_input["input_ids"].squeeze()
        tokens["labels"] = torch.tensor(label)
        tokens["task"] = task
        tokens["task_indices"] = torch.tensor(TASK_TO_INDEX[task])

        if self.return_attention_mask:
            tokens["attention_mask"] = tokenized_input["attention_mask"].squeeze()

        return tokens
