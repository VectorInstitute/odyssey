"""Utilities following HuggingFace style for Mamba models."""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.activations import ACT2FN
from transformers.models.mamba.modeling_mamba import (
    MAMBA_INPUTS_DOCSTRING,
    MAMBA_START_DOCSTRING,
    MambaModel,
    MambaPreTrainedModel,
)
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)


_CONFIG_FOR_DOC = "MambaConfig"


# ruff: noqa: W505,D205,D101,PLR0912


@dataclass
class MambaSequenceClassifierOutput(ModelOutput):
    """
    Base class for outputs of Mamba sentence classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None


# ==================================== SingleHead ================================================= #


class MambaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        """Initialize the head."""
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.config = config

    def forward(self, features, **kwargs):
        """Forward pass."""
        x = features  # Pooling is done by the forward pass
        x = self.dropout(x)
        x = self.dense(x)
        x = ACT2FN[self.config.hidden_act](x)
        x = self.dropout(x)

        return self.out_proj(x)


@add_start_docstrings(
    """
    Mamba Model with a sequence classification/regression head on top
    (a linear layer on top of the pooled output) e.g. for GLUE tasks.
    """,
    MAMBA_START_DOCSTRING,
)
class MambaForSequenceClassification(MambaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.backbone = MambaModel(config)
        self.classifier = MambaClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(
        MAMBA_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(
        output_type=MambaSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[MambaSequenceClassifierOutput, Tuple[torch.FloatTensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:
        """
        if inputs_embeds is not None:
            sequence_outputs = self.backbone(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            sequence_outputs = self.backbone(
                input_ids=input_ids,
                inputs_embeds=None,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        last_hidden_states = sequence_outputs[0]
        batch_size = last_hidden_states.shape[0]

        # Pool the hidden states for the last tokens before padding
        # to use for classification
        last_token_indexes = (
            torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
        )
        pooled_last_hidden_states = last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            last_token_indexes,
        ]

        logits = self.classifier(pooled_last_hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype in [torch.long, torch.int]):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + sequence_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MambaSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=sequence_outputs.hidden_states,
        )


# ==================================== MultiHead ================================================= #


class MambaClassificationMultiHead(nn.Module):
    def __init__(self, config, num_tasks):
        super().__init__()
        self.num_tasks = num_tasks
        self.classifiers = nn.ModuleList(
            [MambaClassificationHead(config) for _ in range(num_tasks)]
        )

    def forward(self, features, task_idx):
        return self.classifiers[task_idx](features)


class MambaForMultiHeadSequenceClassification(MambaPreTrainedModel):
    def __init__(self, config, num_tasks):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.backbone = MambaModel(config)
        self.classifier = MambaClassificationMultiHead(config, num_tasks)
        self.num_tasks = num_tasks

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        task_indices: Optional[torch.LongTensor] = None,  # Add task_indices
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[MambaSequenceClassifierOutput, Tuple[torch.FloatTensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        task_indices (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Task indices to specify which classification head to use for each example in the batch.

        Returns:
        """
        if inputs_embeds is not None:
            sequence_outputs = self.backbone(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            sequence_outputs = self.backbone(
                input_ids=input_ids,
                inputs_embeds=None,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        last_hidden_states = sequence_outputs[0]
        batch_size = last_hidden_states.shape[0]

        # Pool the hidden states for the last tokens before padding to use for classification
        last_token_indexes = (
            torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
        )
        pooled_last_hidden_states = last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            last_token_indexes,
        ]

        logits = torch.zeros(
            batch_size, self.num_labels, device=pooled_last_hidden_states.device
        )
        for i in range(batch_size):
            logits[i] = self.classifier(
                pooled_last_hidden_states[i], task_indices[i].item()
            )

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            loss_fct = None
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()

            if loss_fct is not None:
                loss = 0
                for i in range(batch_size):
                    if self.config.problem_type == "regression":
                        if self.num_labels == 1:
                            loss += loss_fct(logits[i].squeeze(), labels[i].squeeze())
                        else:
                            loss += loss_fct(logits[i], labels[i])
                    elif self.config.problem_type == "single_label_classification":
                        loss += loss_fct(
                            logits[i].view(-1, self.num_labels), labels[i].view(-1)
                        )
                    elif self.config.problem_type == "multi_label_classification":
                        loss += loss_fct(logits[i], labels[i])
                loss /= batch_size

        if not return_dict:
            output = (logits,) + sequence_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MambaSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=sequence_outputs.hidden_states,
        )
