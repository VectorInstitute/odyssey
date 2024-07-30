"""Attribution methods for interpretability."""

from typing import Any, Dict, Generator, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from captum.attr import LayerIntegratedGradients
from captum.attr import visualization as viz
from IPython.core.display import HTML
from torch.utils.data import DataLoader

from odyssey.data.dataset import FinetuneDataset
from odyssey.data.tokenizer import ConceptTokenizer
from odyssey.evals.prediction import load_codes_dict, replace_sequence_items
from odyssey.models.cehr_bert.model import BertFinetune
from odyssey.models.cehr_big_bird.model import BigBirdFinetune
from odyssey.models.model_utils import get_model_embeddings, get_model_embeddings_list


class Attribution:
    """
    A class for model interpretability via attribution methods.

    Parameters
    ----------
    data : pd.DataFrame
        The test data used for computing the the attributions.
    model : Union[BertFinetune, BigBirdFinetune]
        The predictive model to interpret.
    tokenizer : ConceptTokenizer
        The tokenizer used to process text data.
    device : torch.device
        The device (CPU/GPU) on which computations will be executed.
    type_id_mapping : Dict[int, str]
        A mapping from integers to token types.
    max_len : int, optional
        Maximum length of token sequences, by default 512.
    batch_size : int, optional
        Number of samples per batch, by default 32.
    n_steps : int, optional
        Number of steps to integrate along, by default 100.
    codes_dir : str, optional
        Directory containing code mappings, by default "data/codes_dict".
    """

    def __init__(
        self,
        data: pd.DataFrame,
        model: Union[BertFinetune, BigBirdFinetune],
        tokenizer: ConceptTokenizer,
        device: torch.device,
        type_id_mapping: Dict[int, str],
        max_len: int = 512,
        batch_size: int = 32,
        n_steps: int = 100,
        codes_dir: str = "data/codes_dict",
    ) -> None:
        """Initiate the class."""
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.type_id_mapping = type_id_mapping
        self.max_len = max_len
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.codes_dir = codes_dir
        self.model.eval()
        self.data = data
        self.dataset, self.dataloader = self._prepare_data(data)

    def _prepare_data(self, data: pd.DataFrame) -> Tuple[FinetuneDataset, DataLoader]:
        """
        Prepare DataLoader from input data.

        Parameters
        ----------
        data : pd.DataFrame
            The input data to process into a DataLoader.

        Returns
        -------
        Tuple[FinetuneDataset, DataLoader]
            FinetuneDataset and a DataLoader prepared for model processing.
        """
        dataset = FinetuneDataset(
            data=data,
            tokenizer=self.tokenizer,
            max_len=self.max_len,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )
        return dataset, dataloader

    def _get_bert_output(
        self,
        concept_ids: torch.Tensor,
        type_ids: torch.Tensor,
        time_stamps: torch.Tensor,
        ages: torch.Tensor,
        visit_orders: torch.Tensor,
        visit_segments: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process input features through a BERT model and returns probabilities.

        Parameters
        ----------
        concept_ids : torch.Tensor
            Tensor of concept IDs.
        type_ids : torch.Tensor
            Tensor of type IDs.
        time_stamps : torch.Tensor
            Tensor of timestamps.
        ages : torch.Tensor
            Tensor of ages.
        visit_orders : torch.Tensor
            Tensor of visit orders.
        visit_segments : torch.Tensor
            Tensor of visit segments.
        attention_mask : Optional[torch.Tensor], optional
            Tensor for attention operations, by default None.

        Returns
        -------
        torch.Tensor
            Probabilities from the BERT model softmax output.
        """
        logits = self.model(
            input_=(
                concept_ids,
                type_ids,
                time_stamps,
                ages,
                visit_orders,
                visit_segments,
            ),
            attention_mask=attention_mask,
            return_dict=True,
        )["logits"]
        return torch.softmax(logits, dim=1)

    def _get_bigbird_output(
        self,
        concept_ids: torch.Tensor,
        type_ids: torch.Tensor,
        time_stamps: torch.Tensor,
        ages: torch.Tensor,
        visit_orders: torch.Tensor,
        visit_segments: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process input features through a BigBird model and returns probabilities.

        Parameters
        ----------
        concept_ids : torch.Tensor
            Tensor of concept IDs.
        type_ids : torch.Tensor
            Tensor of type IDs.
        time_stamps : torch.Tensor
            Tensor of timestamps.
        ages : torch.Tensor
            Tensor of ages.
        visit_orders : torch.Tensor
            Tensor of visit orders.
        visit_segments : torch.Tensor
            Tensor of visit segments.
        attention_mask : Optional[torch.Tensor], optional
            Tensor for attention operations, by default None.

        Returns
        -------
        torch.Tensor
            Probabilities from the BERT model softmax output.
        """
        logits = self.model(
            input_=(
                concept_ids,
                type_ids,
                time_stamps,
                ages,
                visit_orders,
                visit_segments,
            ),
            attention_mask=attention_mask,
            return_dict=True,
        )["logits"]
        return torch.softmax(logits, dim=1)

    def predict(self, *inputs) -> torch.Tensor:
        """
        Predicts outputs using the appropriate model based on the model's type.

        Parameters
        ----------
        *inputs
            Variable length input list for model predictions.

        Returns
        -------
        torch.Tensor
            Model predictions.
        """
        if isinstance(self.model, BertFinetune):
            prediction_fn = self._get_bert_output
        elif isinstance(self.model, BigBirdFinetune):
            prediction_fn = self._get_bigbird_output
        return prediction_fn(*inputs)

    @property
    def overall_embedding_lig(self) -> LayerIntegratedGradients:
        """Return a LayerIntegratedGradients object for the overall embedding layer."""
        model_embeddings = get_model_embeddings(self.model)
        return LayerIntegratedGradients(self.predict, model_embeddings)

    @property
    def multi_embedding_lig(self) -> LayerIntegratedGradients:
        """Return a LayerIntegratedGradients object for every embedding layer."""
        embeddings_list = list(get_model_embeddings_list(self.model).values())
        return LayerIntegratedGradients(self.predict, embeddings_list)

    def _get_inputs(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor]:
        """
        Extract and return input tensors from the batch to the specified device.

        Parameters
        ----------
        batch : dict
            A batch from the DataLoader.

        Returns
        -------
        Tuple[torch.Tensor]
            Tuple of tensors for the model input.
        """
        concept_inputs = batch["concept_ids"].to(self.device)
        type_inputs = batch["type_ids"].to(self.device)
        time_inputs = batch["time_stamps"].to(self.device)
        age_inputs = batch["ages"].to(self.device)
        order_inputs = batch["visit_orders"].to(self.device)
        segment_inputs = batch["visit_segments"].to(self.device)
        return (
            concept_inputs,
            type_inputs,
            time_inputs,
            age_inputs,
            order_inputs,
            segment_inputs,
        )

    def _get_attention_mask(
        self, batch: Dict[str, torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """
        Retrieve attention mask from the batch if available.

        Parameters
        ----------
        batch : dict
            A batch from the DataLoader.

        Returns
        -------
        Optional[torch.Tensor]
            Tensor of attention mask, or None if not available.
        """
        return (
            batch["attention_mask"].to(self.device)
            if "attention_mask" in batch
            else None
        )

    def _get_labels(self, batch: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Retrieve labels from the batch if available.

        Parameters
        ----------
        batch : dict
            A batch from the DataLoader.

        Returns
        -------
        Optional[torch.Tensor]
            Tensor of labels, or None if labels are not present in the batch.
        """
        return batch["labels"].to(self.device) if "labels" in batch else None

    def _get_batch_data(self) -> Generator[Tuple[torch.Tensor], None, None]:
        """
        Generate batch data from the DataLoader.

        Returns
        -------
        Tuple[torch.Tensor]
            Tuple of tensors for inputs, targets, and attention masks.
        """
        for batch in self.dataloader:
            inputs = self._get_inputs(batch)
            targets = self._get_labels(batch)
            attention_mask = self._get_attention_mask(batch)
            yield inputs, targets, attention_mask

    def summarize_attributions(self, attributions: torch.Tensor) -> torch.Tensor:
        """
        Summarize attributions by summing and normalizing them for each token.

        Parameters
        ----------
        attributions : torch.Tensor
            Attributions to summarize.

        Returns
        -------
        torch.Tensor
            Summarized attributions.
        """
        attributions = attributions.sum(dim=-1)
        return attributions / torch.norm(attributions)

    def create_baseline(self) -> Tuple[torch.Tensor]:
        """
        Create a baseline input for attribution comparison with zero values.

        Returns
        -------
        Tuple[torch.Tensor]
            A tuple of baseline tensors for each input feature.
        """
        concept_baseline = (
            torch.tensor([self.tokenizer.get_pad_token_id()] * self.max_len)
            .unsqueeze(0)
            .to(self.device)
        )
        type_baseline = torch.tensor([0] * self.max_len).unsqueeze(0).to(self.device)
        time_baseline = torch.tensor([0] * self.max_len).unsqueeze(0).to(self.device)
        age_baseline = torch.tensor([0] * self.max_len).unsqueeze(0).to(self.device)
        order_baseline = torch.tensor([0] * self.max_len).unsqueeze(0).to(self.device)
        segment_baseline = torch.tensor([0] * self.max_len).unsqueeze(0).to(self.device)
        return (
            concept_baseline,
            type_baseline,
            time_baseline,
            age_baseline,
            order_baseline,
            segment_baseline,
        )

    def sample_baselines(self, idx: int, num_samples: int) -> FinetuneDataset:
        """
        Sample baseline data from the dataset excluding the specified index.

        Parameters
        ----------
        idx : int
            The index to exclude from sampling (the ipnut data).
        num_samples : int
            The number of samples to draw.

        Returns
        -------
        FinetuneDataset
            A new FinetuneDataset containing the sampled baseline data.
        """
        possible_indices = list(range(len(self.data)))
        possible_indices.remove(idx)
        sampled_indices = np.random.choice(possible_indices, num_samples, replace=False)
        sampled_df = self.data.iloc[sampled_indices]
        return FinetuneDataset(sampled_df, self.tokenizer, self.max_len)

    def average_embeddings_attr(self, use_abs: bool = True) -> Dict[str, float]:
        """
        Calculate and average embeddings attributions across all data.

        Use layer integrated gradients to compute token attributions.

        Parameters
        ----------
        use_abs : bool, optional
            Flag to use absolute values for attributions, by default True.

        Returns
        -------
        Dict[str, float]
            Dictionary of average attributions per embedding, keyed by embedding names.
        """
        embedding_attr = {key: 0.0 for key in get_model_embeddings_list(self.model)}
        for inputs, target, attention_mask in self._get_batch_data():
            attributions = self.multi_embedding_lig.attribute(
                inputs=inputs,
                baselines=self.create_baseline(),
                additional_forward_args=(attention_mask),
                target=target,
                internal_batch_size=self.batch_size,
                n_steps=self.n_steps,
                return_convergence_delta=False,
            )
            attributions_summary = [
                self.summarize_attributions(attr) for attr in attributions
            ]
            for i in range(len(attributions_summary)):
                key = list(embedding_attr.keys())[i]
                if use_abs:
                    embedding_attr[key] += torch.sum(
                        torch.abs(attributions_summary[i])
                    ).item()
                else:
                    embedding_attr[key] += torch.sum(attributions_summary[i]).item()

        # take the average of the attributions
        for key in embedding_attr:
            embedding_attr[key] /= len(self.dataset)
        return embedding_attr

    def average_tokens_attr(self, use_abs: bool = True) -> Dict[str, float]:
        """
        Calculate and average token attributions across all data, grouped by token type.

        Use layer integrated gradients to compute token attributions.

        Parameters
        ----------
        use_abs : bool, optional
            Whether to sum the absolute values of attributions, by default True.

        Returns
        -------
        Dict[str, float]
            Dictionary with average attribution per token type, using type ID mapping.
        """
        token_attr = {key: 0.0 for key in self.type_id_mapping.values()}
        for inputs, target, attention_mask in self._get_batch_data():
            attributions = self.multi_embedding_lig.attribute(
                inputs=inputs,
                baselines=self.create_baseline(),
                additional_forward_args=(attention_mask),
                target=target,
                internal_batch_size=self.batch_size,
                n_steps=self.n_steps,
                return_convergence_delta=False,
            )
            attributions_summary = [
                self.summarize_attributions(attr) for attr in attributions
            ]

            type_inputs = inputs[1]
            for idx in range(attributions_summary[0].shape[0]):
                for i, type_id in enumerate(type_inputs[idx]):
                    # sum the attributions from all embeddings such as:
                    # concept, type, time, age, visit order, visit segment
                    if use_abs:
                        total_attr = sum(
                            abs(attributions_summary[j][idx][i])
                            for j in range(len(attributions_summary))
                        )
                    else:
                        total_attr = sum(
                            attributions_summary[j][idx][i]
                            for j in range(len(attributions_summary))
                        )
                    token_attr[self.type_id_mapping[type_id.item()]] += (
                        total_attr.item() / len(attributions_summary)
                    )

        # take the average of the attributions
        for key in token_attr:
            token_attr[key] /= len(self.dataset)
        return token_attr

    def _create_visualization_record(
        self,
        concept_seq: torch.Tensor,
        attr_summary: torch.Tensor,
        scores: torch.Tensor,
        target: torch.Tensor,
        delta: torch.Tensor,
        task_name: str,
        codes_dict: Dict[str, str],
    ) -> viz.VisualizationDataRecord:
        """
        Create a visualization record for a given sequence using its attributions.

        Parameters
        ----------
        concept_seq : torch.Tensor
            The tensor representing the sequence of concepts.
        attr_summary : torch.Tensor
            The tensor containing attribution summaries for the concepts.
        scores : torch.Tensor
            The tensor of scores from which the maximum and its index are derived.
        target : torch.Tensor
            The target tensor corresponding to the concept sequence.
        delta : torch.Tensor
            The delta values indicating the error of the attributions.
        task_name : str
            The name of the task for which the visualization is being created.
        codes_dict : Dict[str, str]
            A dictionary mapping codes to more interpretable strings.

        Returns
        -------
        viz.VisualizationDataRecord
            An object containing all visualization data, formatted for display.
        """
        concept_decoded = self.tokenizer.decode(concept_seq)
        pad_index = (
            concept_decoded.find("[PAD]")
            if "[PAD]" in concept_decoded
            else len(concept_decoded)
        )
        concept_tokens = concept_decoded[:pad_index].split()
        concept_tokens = replace_sequence_items(concept_tokens, codes_dict)

        attr_summary = attr_summary[: len(concept_tokens)]

        return viz.VisualizationDataRecord(
            attr_summary,
            torch.max(scores[0], dim=0).values,
            torch.argmax(scores),
            target,
            task_name,
            attr_summary.sum(),
            concept_tokens,
            delta,
        )

    def visualize_integrated_gradients(
        self,
        max_rows: int = 10,
        task_name: str = "mortality",
    ) -> "HTML":
        """
        Visualize token attributions using the layer itegrated gradients method.

        Parameters
        ----------
        max_rows : int, optional
            Maximum umber of data records to visualize. by default 50.
        task_name : str, optional
            The name of the task for labeling, by default "mortality".

        Returns
        -------
        HTML
            An HTML object displaying the visualization of token attributions.
        """
        count = 0
        visualization_records = []

        codes_dict = load_codes_dict(self.codes_dir)

        for inputs, target, attention_mask in self._get_batch_data():
            attributions, delta = self.overall_embedding_lig.attribute(
                inputs=inputs,
                baselines=self.create_baseline(),
                additional_forward_args=(attention_mask,),
                target=target,
                return_convergence_delta=True,
                n_steps=self.n_steps,
                internal_batch_size=self.batch_size,
            )
            attributions_summary = self.summarize_attributions(attributions)
            scores = self.predict(*inputs, attention_mask)
            concept_input = inputs[0].tolist()

            for i, concept_seq in enumerate(concept_input):
                count += 1
                if count > max_rows:
                    break
                vis_record = self._create_visualization_record(
                    concept_seq=concept_seq,
                    attr_summary=attributions_summary[i],
                    scores=scores[i],
                    target=target[i],
                    delta=delta[i],
                    task_name=task_name,
                    codes_dict=codes_dict,
                )
                visualization_records.append(vis_record)
            if count >= max_rows:
                break
        return viz.visualize_text(visualization_records)

    def visualize_expected_gradients(
        self,
        max_rows: int = 10,
        num_baselines: int = 100,
        task_name: str = "mortality",
    ) -> Any:
        """
        Visualize token attributions using the layer expected gradients method.

        This method takes average attributions across multiple baselines.

        Parameters
        ----------
        max_rows : int, optional
            Maximum umber of data records to visualize. by default 50.
        num_baselines : int, optional
            Number of baselines to sample for averaging, by default 50.
        task_name : str, optional
            The name of the task for labeling, by default "mortality".

        Returns
        -------
        HTML
            An HTML object displaying the visualization of token attributions.
        """
        assert num_baselines < len(
            self.dataset
        ), "Number of baselines should be less than the dataset size."
        max_rows = min(max_rows, len(self.dataset))

        visualization_records = []
        codes_dict = load_codes_dict(self.codes_dir)

        for i in range(max_rows):
            tokens_attr = torch.zeros(self.max_len).to(self.device)
            item = self.dataset[i]
            item = {key: torch.unsqueeze(val, 0) for key, val in item.items()}

            inputs = self._get_inputs(item)
            attention_mask = self._get_attention_mask(item)
            target = self._get_labels(item)
            baselines = self.sample_baselines(i, num_baselines)

            for baseline in baselines:
                baseline_batch = {
                    key: torch.unsqueeze(val, 0) for key, val in baseline.items()
                }
                attributions, delta = self.overall_embedding_lig.attribute(
                    inputs=inputs,
                    baselines=self._get_inputs(baseline_batch),
                    additional_forward_args=(attention_mask),
                    target=target,
                    internal_batch_size=self.batch_size,
                    n_steps=self.n_steps,
                    return_convergence_delta=True,
                )
                attributions_summary = self.summarize_attributions(
                    attributions
                ).squeeze(0)
                tokens_attr += attributions_summary
            tokens_attr /= num_baselines

            scores = self.predict(*inputs, attention_mask)
            concept_seq = inputs[0].tolist()[0]
            vis_record = self._create_visualization_record(
                concept_seq=concept_seq,
                attr_summary=tokens_attr,
                scores=scores,
                target=target.item() if target is not None else None,
                delta=delta,
                task_name=task_name,
                codes_dict=codes_dict,
            )
            visualization_records.append(vis_record)
        return viz.visualize_text(visualization_records)
