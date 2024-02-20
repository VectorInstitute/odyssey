
from typing import Optional, Tuple, Union, Any, List, Dict

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import torch.optim as optim
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR

from transformers import BertConfig, BigBirdConfig
from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput
from transformers.models.bert.modeling_bert import BertPooler
from transformers.models.big_bird.modeling_big_bird import BigBirdForMaskedLM

import pytorch_lightning as pl

from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

from .embeddings import BigBirdEmbeddingsForCEHR


class BigBirdPretrain(pl.LightningModule):
    """BigBird model for pretraining."""

    def __init__(
            self,
            args: Tuple[Any, ...],
            dataset_len: int,
            vocab_size: int,
            embedding_size: int = 768,
            time_embeddings_size: int = 32,
            visit_order_size: int = 3,
            type_vocab_size: int = 8,
            max_seq_length: int = 2048,
            depth: int = 6,
            num_heads: int = 12,
            intermediate_size: int = 3072,
            learning_rate: float = 5e-5,
            eta_min: float = 1e-8,
            num_iterations: int = 10,
            increase_factor: float = 2,
            dropout_prob: float = 0.1,
            padding_idx: int = 1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.time_embeddings_size = time_embeddings_size
        self.visit_order_size = visit_order_size
        self.type_vocab_size = type_vocab_size
        self.max_seq_length = max_seq_length
        self.depth = depth
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.learning_rate = learning_rate
        self.eta_min = eta_min
        self.num_iterations = num_iterations
        self.increase_factor = increase_factor
        self.dropout_prob = dropout_prob
        self.padding_idx = padding_idx

        self.config = BigBirdConfig(
            vocab_size=self.vocab_size,
            type_vocab_size=self.type_vocab_size,
            hidden_size=self.embedding_size,
            num_hidden_layers=self.depth,
            num_attention_heads=self.num_heads,
            intermediate_size=self.intermediate_size,
            hidden_dropout_prob=self.dropout_prob,
            attention_probs_dropout_prob=self.dropout_prob,
            max_position_embeddings=self.max_seq_length,
            is_decoder=False,
            pad_token_id=padding_idx
        )
        # BigBirdForMaskedLM
        self.embeddings = BigBirdEmbeddingsForCEHR(
            config=self.config,
            time_embeddings_size=self.time_embeddings_size,
            visit_order_size=self.visit_order_size
        )

        self.model = BigBirdForMaskedLM(config=self.config)
        self.model.bert.embeddings = self.embeddings

        # Initialize weights and apply final processing
        self.post_init()

        # Define warmup and decay iterations for the scheduler
        grad_steps = dataset_len / args.batch_size / args.gpus * args.max_epochs
        self.warmup = int(0.1 * grad_steps)
        self.decay = int(0.9 * grad_steps)

    def _init_weights(self, module) -> None:
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def post_init(self) -> None:
        self.apply(self._init_weights)

    def forward(
            self,
            inputs: Tuple[
                torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
            ],
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, ...], MaskedLMOutput]:
        """Forward pass for the model."""

        concept_ids, type_ids, time_stamps, ages, visit_orders, visit_segments = inputs
        self.embeddings.cache_input(time_stamps, ages, visit_segments)

        if attention_mask is None:
            attention_mask = torch.ones_like(concept_ids)

        outputs = self.model(
            input_ids=concept_ids,
            attention_mask=attention_mask,
            token_type_ids=type_ids,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return outputs

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """Training step."""
        inputs = (
            batch["concept_ids"],
            batch["type_ids"],
            batch["time_stamps"],
            batch["ages"],
            batch["visit_orders"],
            batch["visit_segments"],
        )
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]

        # This is not necessary but makes sure we use the right attention
        self.encoder.set_attention_type("block_sparse")
        loss = self(
            inputs, attention_mask=attention_mask, labels=labels, return_dict=True
        ).loss

        (current_lr,) = self.lr_schedulers().get_last_lr()
        self.log_dict(
            dictionary={"train_loss": loss, "lr": current_lr},
            on_step=True,
            prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        """Validation step."""
        inputs = (
            batch["concept_ids"],
            batch["type_ids"],
            batch["time_stamps"],
            batch["ages"],
            batch["visit_orders"],
            batch["visit_segments"],
        )
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]

        # This is not necessary but makes sure we use the right attention
        self.encoder.set_attention_type("block_sparse")
        loss = self(
            inputs, attention_mask=attention_mask, labels=labels, return_dict=True
        ).loss

        (current_lr,) = self.lr_schedulers().get_last_lr()
        self.log_dict(
            dictionary={"val_loss": loss, "lr": current_lr},
            on_step=True,
            prog_bar=True,
            sync_dist=True
        )
        return loss

    def configure_optimizers(self) -> tuple[list[AdamW], list[dict[str, SequentialLR | str]]]:
        """Configure optimizers and learning rate scheduler."""
        optimizer = optim.AdamW(
            self.parameters(), lr=self.learning_rate
        )

        warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.,
            total_iters=self.warmup)

        linear_decay = LinearLR(
            optimizer,
            start_factor=1.,
            end_factor=0.01,
            total_iters=self.decay)

        scheduler = SequentialLR(
            optimizer=optimizer,
            schedulers=[warmup, linear_decay],
            milestones=[self.warmup]
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


class BigBirdFinetune(pl.LightningModule):
    """BigBird model for fine-tuning."""

    def __init__(
            self,
            args: tuple,
            dataset_len: int,
            pretrained_model: BigBirdPretrain,
            num_labels: int = 2,
            hidden_size: int = 768,
            classifier_dropout: float = 0.1,
            hidden_dropout_prob: float = 0.1,
            learning_rate: float = 5e-5,
            eta_min: float = 1e-8,
            num_iterations: int = 10,
            increase_factor: float = 2,
    ):
        super().__init__()

        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.eta_min = eta_min
        self.num_iterations = num_iterations
        self.increase_factor = increase_factor
        self.classifier_dropout = classifier_dropout
        self.hidden_dropout_prob = hidden_dropout_prob

        self.config = BertConfig(
            num_labels=self.num_labels,
            hidden_size=self.hidden_size,
            classifier_dropout=self.classifier_dropout,
            hidden_dropout_prob=self.hidden_dropout_prob,
        )

        # BigBirdForSequenceClassification
        self.pooler = BertPooler(self.config)

        # SequenceClassification
        classifier_dropout = (
            self.config.classifier_dropout
            if self.config.classifier_dropout is not None
            else self.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

        self.post_init()
        self.pretrained_model = pretrained_model

        # Define warmup and decay iterations for the scheduler
        grad_steps = dataset_len / args.batch_size / args.gpus * args.max_epochs
        self.warmup = int(0.1 * grad_steps)
        self.decay = int(0.9 * grad_steps)

    def _init_weights(self, module) -> None:
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def post_init(self) -> None:
        self.apply(self._init_weights)

    def forward(
            self,
            inputs: Tuple[
                torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
            ],
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, ...], SequenceClassifierOutput]:
        """Forward pass for the model."""

        outputs = self.pretrained_model(inputs=inputs,
                                        attention_mask=attention_mask,
                                        labels=labels,
                                        output_attentions=output_attentions,
                                        output_hidden_states=output_hidden_states,
                                        return_dict=return_dict)

        hidden_states = outputs["hidden_states"]  # hidden_states
        hidden_states = hidden_states[-1]
        pooled_output = self.pooler(hidden_states) if self.pooler is not None else None
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """Training step."""
        inputs = (
            batch["concept_ids"],
            batch["type_ids"],
            batch["time_stamps"],
            batch["ages"],
            batch["visit_orders"],
            batch["visit_segments"],
        )
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]

        # This is not necessary but makes sure we use the right attention
        self.encoder.set_attention_type("block_sparse")
        loss = self(
            inputs, attention_mask=attention_mask, labels=labels, return_dict=True
        )[0]

        (current_lr,) = self.lr_schedulers().get_last_lr()
        self.log_dict(
            dictionary={"train_loss": loss, "lr": current_lr},
            on_step=True,
            prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        """Validation step."""
        inputs = (
            batch["concept_ids"],
            batch["type_ids"],
            batch["time_stamps"],
            batch["ages"],
            batch["visit_orders"],
            batch["visit_segments"],
        )
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]

        # This is not necessary but makes sure we use the right attention
        self.encoder.set_attention_type("block_sparse")
        loss = self(
            inputs, attention_mask=attention_mask, labels=labels, return_dict=True
        )[0]

        (current_lr,) = self.lr_schedulers().get_last_lr()
        self.log_dict(
            dictionary={"val_loss": loss, "lr": current_lr},
            on_step=True,
            prog_bar=True,
            sync_dist=True
        )
        return loss

    def test_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        """Test step."""
        inputs = (
            batch["concept_ids"],
            batch["type_ids"],
            batch["time_stamps"],
            batch["ages"],
            batch["visit_orders"],
            batch["visit_segments"],
        )
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]

        # This is not necessary but makes sure we use the right attention
        self.encoder.set_attention_type("block_sparse")
        outputs = self(
            inputs, attention_mask=attention_mask, labels=labels, return_dict=True
        )

        loss = outputs[0]
        logits = outputs[1]
        preds = torch.argmax(logits, dim=1)
        log = {"loss": loss, "preds": preds, "labels": labels}

        self.log_dict(
            dictionary=log,
            on_step=True,
            prog_bar=True,
            sync_dist=True
        )
        return log

    def test_epoch_end(self, outputs) -> torch.Tensor:
        """Evaluate after the test epoch."""
        labels = torch.cat([x['labels'] for x in outputs]).cpu()
        preds = torch.cat([x['preds'] for x in outputs]).cpu()
        loss = torch.stack([x['loss'] for x in outputs]).mean().cpu()
        self.log('test_loss', loss)
        self.log('test_acc', accuracy_score(labels, preds))
        self.log('test_f1', f1_score(labels, preds))
        self.log('test_auc', roc_auc_score(labels, preds))
        self.log('test_precision', precision_score(labels, preds))
        self.log('test_recall', recall_score(labels, preds))
        return loss

    def configure_optimizers(self) -> tuple[list[AdamW], list[dict[str, SequentialLR | str]]]:
        """Configure optimizers and learning rate scheduler."""
        optimizer = optim.AdamW(
            self.parameters(), lr=self.learning_rate
        )

        warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.,
            total_iters=self.warmup)

        linear_decay = LinearLR(
            optimizer,
            start_factor=1.,
            end_factor=0.01,
            total_iters=self.decay)

        scheduler = SequentialLR(
            optimizer=optimizer,
            schedulers=[warmup, linear_decay],
            milestones=[self.warmup]
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
