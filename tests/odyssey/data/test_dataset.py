"""Test dataset classes."""

import unittest
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import pandas as pd
import torch

from odyssey.data.dataset import (
    BaseDataset,
    FinetuneDataset,
    FinetuneDatasetDecoder,
    FinetuneMultiDataset,
    LabelBalanceMixin,
    MaskingMixin,
    MultiTaskMixin,
    PretrainDataset,
    PretrainDatasetDecoder,
    TokenizationMixin,
)
from odyssey.data.tokenizer import ConceptTokenizer


class TestDatasets(unittest.TestCase):
    """Test dataset classes."""

    def setUp(self) -> None:
        """Set up mock data and tokenizer for testing."""
        self.data = pd.DataFrame(
            {
                "event_tokens_2048": [["token1", "token2"], ["token3", "token4"]],
                "type_tokens_2048": [[1, 2], [3, 4]],
                "age_tokens_2048": [[30, 40], [50, 60]],
                "time_tokens_2048": [[100, 200], [300, 400]],
                "visit_tokens_2048": [[10, 20], [30, 40]],
                "position_tokens_2048": [[1, 1], [2, 2]],
                "elapsed_tokens_2048": [[5, 10], [15, 20]],
                "label": [0, 1],
                "cutoff": [2, 2],
                "label_mortality_1month": [0, 1],
                "label_readmission_1month": [-1, 0],
            }
        )
        self.tokenizer = ConceptTokenizer()
        self.tokenizer.tokenizer_object = MagicMock()
        self.tokenizer.tokenizer = MagicMock()
        self.tokenizer.tokenizer.return_value = {
            "input_ids": torch.tensor([[100, 200], [300, 400]]),
            "attention_mask": torch.tensor([[1, 1], [1, 1]]),
        }
        self.tokenizer.task_to_token = MagicMock(side_effect=lambda x: f"[{x.upper()}]")
        self.tokenizer.get_mask_token_id = MagicMock(return_value=103)
        self.tokenizer.get_first_token_index = MagicMock(return_value=0)
        self.tokenizer.get_last_token_index = MagicMock(return_value=999)
        self.tokenizer.token_to_id = MagicMock(return_value=101)

    def test_base_dataset(self) -> None:
        """Test the BaseDataset class."""

        class DummyDataset(BaseDataset):
            def __getitem__(self, idx: int) -> Dict[str, str]:
                return {"dummy_key": "dummy_value"}

        dataset = DummyDataset(data=self.data, tokenizer=self.tokenizer)
        self.assertEqual(len(dataset), len(self.data))
        self.assertEqual(dataset[0], {"dummy_key": "dummy_value"})

    def test_tokenization_mixin(self) -> None:
        """Test the TokenizationMixin class."""

        class DummyDataset(BaseDataset, TokenizationMixin):
            def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
                return self.add_additional_tokens(self.data.iloc[idx])

        dataset = DummyDataset(data=self.data, tokenizer=self.tokenizer)
        result = dataset[0]
        self.assertIn("type_ids", result)
        self.assertIn("ages", result)
        self.assertIn("time_stamps", result)
        self.assertIn("visit_orders", result)
        self.assertIn("visit_segments", result)
        self.assertEqual(result["type_ids"].size(0), 2)
        self.assertEqual(result["ages"].size(0), 2)
        self.assertEqual(result["time_stamps"].size(0), 2)
        self.assertEqual(result["visit_orders"].size(0), 2)
        self.assertEqual(result["visit_segments"].size(0), 2)

    def test_masking_mixin(self) -> None:
        """Test the MaskingMixin class."""

        class DummyDataset(BaseDataset, MaskingMixin):
            def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
                return self.mask_tokens(torch.tensor([10, 20, 30]))

        dataset = DummyDataset(data=self.data, tokenizer=self.tokenizer)
        dataset.mask_prob = 0.15
        masked_sequence, labels = dataset[0]
        self.assertEqual(len(masked_sequence), 3)
        self.assertEqual(len(labels), 3)
        self.assertTrue((masked_sequence <= 103).all())
        self.assertTrue((labels == -100).any())

    def test_multi_task_mixin(self) -> None:
        """Test the MultiTaskMixin class."""

        class DummyDataset(BaseDataset, MultiTaskMixin):
            def __init__(
                self, data: pd.DataFrame, tokenizer: ConceptTokenizer, tasks: List[str]
            ) -> None:
                BaseDataset.__init__(self, data, tokenizer)
                MultiTaskMixin.__init__(self, tasks)
                self.nan_indicator = -1

            def __getitem__(self, idx: int) -> Tuple[int, str, int, Optional[int]]:
                return self.index_mapper[idx]

        dataset = DummyDataset(
            data=self.data,
            tokenizer=self.tokenizer,
            tasks=["mortality_1month", "readmission_1month"],
        )
        dataset.prepare_multi_task_data()
        self.assertEqual(len(dataset.index_mapper), 3)

    def test_label_balance_mixin(self) -> None:
        """Test the LabelBalanceMixin class."""

        class DummyDataset(BaseDataset, MultiTaskMixin, LabelBalanceMixin):
            def __init__(
                self, data: pd.DataFrame, tokenizer: ConceptTokenizer, tasks: List[str]
            ) -> None:
                BaseDataset.__init__(self, data, tokenizer)
                MultiTaskMixin.__init__(self, tasks)
                self.nan_indicator = -1

            def __len__(self) -> int:
                return len(self.index_mapper)

            def __getitem__(self, idx: int) -> Tuple[int, str, int, Optional[int]]:
                return self.index_mapper[idx]

        dataset = DummyDataset(
            data=self.data,
            tokenizer=self.tokenizer,
            tasks=["mortality_1month", "readmission_1month"],
        )
        dataset.prepare_multi_task_data()
        dataset.balance_labels({"mortality_1month": 0.5})
        task_counts = {task: 0 for task in dataset.tasks}
        for i in range(len(dataset)):
            task = dataset[i][1]
            task_counts[task] += 1
        self.assertEqual(task_counts["mortality_1month"], 2)
        self.assertEqual(task_counts["readmission_1month"], 1)

    def test_pretrain_dataset(self) -> None:
        """Test the PretrainDataset class."""
        dataset = PretrainDataset(data=self.data, tokenizer=self.tokenizer)
        tokens = dataset[0]
        self.assertIn("concept_ids", tokens)
        self.assertIn("labels", tokens)
        self.assertIn("attention_mask", tokens)
        self.assertEqual(tokens["concept_ids"].size(0), 2)
        self.assertEqual(tokens["labels"].size(0), 2)
        self.assertEqual(tokens["attention_mask"].size(0), 2)

    def test_pretrain_dataset_decoder(self) -> None:
        """Test the PretrainDatasetDecoder class."""
        dataset = PretrainDatasetDecoder(data=self.data, tokenizer=self.tokenizer)
        tokens = dataset[0]
        self.assertIn("concept_ids", tokens)
        self.assertIn("labels", tokens)
        self.assertEqual(tokens["concept_ids"].size(0), 2)
        self.assertEqual(tokens["labels"].size(0), 2)
        self.assertIs(tokens["labels"], tokens["concept_ids"])

    def test_finetune_dataset(self) -> None:
        """Test the FinetuneDataset class."""
        dataset = FinetuneDataset(data=self.data, tokenizer=self.tokenizer)
        tokens = dataset[1]
        self.assertIn("concept_ids", tokens)
        self.assertIn("labels", tokens)
        self.assertIn("attention_mask", tokens)
        self.assertEqual(tokens["concept_ids"].size(0), 2)
        self.assertEqual(tokens["labels"], torch.tensor(1))
        self.assertEqual(tokens["attention_mask"].size(0), 2)

    def test_finetune_multi_dataset(self) -> None:
        """Test the FinetuneMultiDataset class."""
        dataset = FinetuneMultiDataset(
            data=self.data,
            tokenizer=self.tokenizer,
            tasks=["mortality_1month", "readmission_1month"],
        )
        tokens = dataset[0]
        self.assertIn("concept_ids", tokens)
        self.assertIn("labels", tokens)
        self.assertIn("attention_mask", tokens)
        self.assertIn("task", tokens)
        self.assertEqual(tokens["concept_ids"].size(0), 2)
        self.assertEqual(tokens["labels"], torch.tensor(0))
        self.assertEqual(tokens["attention_mask"].size(0), 2)

    def test_finetune_dataset_decoder(self) -> None:
        """Test the FinetuneDatasetDecoder class."""
        dataset = FinetuneDatasetDecoder(
            data=self.data,
            tokenizer=self.tokenizer,
            tasks=["mortality_1month", "readmission_1month"],
        )
        tokens = dataset[2]
        self.assertIn("concept_ids", tokens)
        self.assertIn("labels", tokens)
        self.assertIn("task", tokens)
        self.assertIn("task_indices", tokens)
        self.assertEqual(tokens["concept_ids"].size(0), 2)
        self.assertEqual(tokens["labels"], torch.tensor(0))

    def test_dataset_length(self) -> None:
        """Test the length of the dataset."""
        dataset = PretrainDataset(data=self.data, tokenizer=self.tokenizer)
        self.assertEqual(len(dataset), 2)

    def test_multitask_balance(self) -> None:
        """Test the balancing of tasks in the dataset."""
        dataset = FinetuneMultiDataset(
            data=self.data,
            tokenizer=self.tokenizer,
            tasks=["mortality_1month", "readmission_1month"],
            balance_guide={"mortality_1month": 0.5},
        )
        task_counts = {task: 0 for task in dataset.tasks}
        for i in range(len(dataset)):
            task = dataset[i]["task"]
            task_counts[task] += 1
        self.assertEqual(task_counts["mortality_1month"], 2)
        self.assertEqual(task_counts["readmission_1month"], 1)


if __name__ == "__main__":
    unittest.main()
