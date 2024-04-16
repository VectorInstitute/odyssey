"""
data.py.
----------
Create custom pretrain and finetune PyTorch Dataset objects for MIMIC-IV FHIR dataset.
"""

from typing import Any, Dict, List, Tuple, Union, Optional

import random

import pandas as pd
import torch
from torch.utils.data import Dataset

from .tokenizer import ConceptTokenizer, truncate_and_pad

TASK_INDEX = 1
LABEL_INDEX = 2
CUTOFF_INDEX = 3


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

        # Find the cutoff column in the data if it exists.
        for column in self.data.columns:
            if 'cutoff' in column:
                self.cutoff_col = column
            else:
                self.cutoff_col = None


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
        cutoff = data[self.cutoff_col] if self.cutoff_col else None
        data = truncate_and_pad(data, cutoff=cutoff, max_len=self.max_len)
        
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

        # Find the cutoff column in the data if it exists.
        for column in self.data.columns:
            if 'cutoff' in column:
                self.cutoff_col = column
            else:
                self.cutoff_col = None


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
        cutoff = data[self.cutoff_col] if self.cutoff_col else None
        data = truncate_and_pad(data, cutoff=cutoff, max_len=self.max_len)

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


class FinetuneMultiDataset(Dataset):
    """Dataset for finetuning the model on multi dataset."""

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: ConceptTokenizer,
        tasks: List[str],
        balance_guide: Optional[Dict[str, float]] = None,
        max_len: int = 2048,
        nan_indicator: int = -1
    ):
        """
        Initialize the dataset class for finetuning on multiple tasks.

        Args:
            data (pd.DataFrame): DataFrame containing the patient data.
            tokenizer (ConceptTokenizer): The tokenizer to be used for encoding sequences.
            tasks (List[str]): A list of tasks (labels) that need to be predicted.
            balance_guide (Dict[str, float]): A dictionary containing the desired positive ratios for each task.
            max_len (int): Maximum length of the tokenized sequences.
            nan_indicator (int): Value used to represent missing labels in the dataset.

        The constructor prepares the dataset by setting up internal structures to manage the data.
        """
        super(FinetuneMultiDataset, self).__init__()

        self.data = data
        self.tokenizer = tokenizer
        self.tasks = tasks  # List of tasks for which the model is being finetuned.
        self.balance_guide = balance_guide
        self.max_len = max_len
        self.nan_indicator = nan_indicator  # Value used to indicate missing data in labels.

        # Precompute indices for quick mapping in __getitem__ that exclude missing labels.
        # This helps in filtering out entries where the label is missing for the specified tasks.
        self.task_to_index = {task: [] for task in self.tasks}
        self.data.reset_index(drop=True, inplace=True)

        for patient in self.data.itertuples():
            index = patient.Index

            for task in self.tasks:
                label_col = f'label_{task}'

                if getattr(patient, label_col) == self.nan_indicator:
                    continue  # Skip this task for the current patient if the label is missing.
                else:
                    label = getattr(patient, label_col)

                # Check for the existence of a task-specific cutoff in the data, else use None.
                if f'cutoff_{task}' in self.data.columns:
                    cutoff = getattr(patient, f'cutoff_{task}')
                else:
                    cutoff = None
                
                # Append a tuple containing the necessary information for training to index_mapper.
                datapoint = (index, task, label, cutoff)
                self.task_to_index[task].append(datapoint)

        # Balance labels for specified tasks
        if self.balance_guide:
            for task in self.balance_guide.keys():
                self.balance_labels(task=task, positive_ratio=self.balance_guide[task])

        # Create a list of all datapoints to be used by __getitem__
        self.index_mapper = [datapoints for task_data in self.task_to_index.values() for datapoints in task_data]
        del self.task_to_index


    def __len__(self) -> int:
        """Return the length of dataset."""
        return len(self.index_mapper)
    

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get data at corresponding index.

        Return it as a dictionary including
        all different token sequences along with attention mask and labels.
        """
        index, task, labels, cutoff = self.index_mapper[idx]  
        data = self.data.iloc[index]

        # Swap the first token with the task token.
        data['event_tokens_2048'][0] = self.tokenizer.task_to_token(task)

        # Truncate and pad the data to the specified cutoff.
        data = truncate_and_pad(data, cutoff, self.max_len)

        # Prepare model input
        tokenized_input = self.tokenize_data(data[f"event_tokens_{self.max_len}"])
        concept_tokens = tokenized_input["input_ids"].squeeze()
        attention_mask = tokenized_input["attention_mask"].squeeze()

        type_tokens = data[f"type_tokens_{self.max_len}"]
        age_tokens = data[f"age_tokens_{self.max_len}"]
        time_tokens = data[f"time_tokens_{self.max_len}"]
        visit_tokens = data[f"visit_tokens_{self.max_len}"]
        position_tokens = data[f"position_tokens_{self.max_len}"]

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
            "task": task,
        }


    def tokenize_data(self, sequence: Union[str, List[str]]) -> Any:
        """Tokenize the sequence and return input_ids and attention mask."""
        return self.tokenizer(sequence, max_length=self.max_len)
    

    def balance_labels(self, task: str, positive_ratio: float) -> None:
        """
        This function modifies the dataset to ensure that the ratio of positive samples
        to the total number of samples matches the specified positive_ratio, while keeping
        all positive data points.

        Args:
            data (list of tuples): The dataset to be modified. Each tuple is of the form (index, task, label, cutoff).
            positive_ratio (float): The desired ratio of positive samples in the dataset.

        Returns:
            none, the datapoints are modified in place.
        """
        # Separate positive and negative datapoints
        datapoints = self.task_to_index[task]
        positives = [data for data in datapoints if data[LABEL_INDEX] == 1]
        negatives = [data for data in datapoints if data[LABEL_INDEX] == 0]

        # Calculate the total number of samples needed to achieve the desired positive ratio
        num_positives = len(positives)
        total_needed = int(num_positives / positive_ratio) - num_positives
        num_negatives_to_keep = min(len(negatives), total_needed)

        # Randomly select the negatives to keep
        negatives_kept = random.sample(negatives, num_negatives_to_keep)

        # Combine the kept negatives with all positives
        self.task_to_index[task] = positives + negatives_kept
