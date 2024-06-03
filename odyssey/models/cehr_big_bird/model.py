"""BigBird transformer model."""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn, optim
from torch.cuda.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from transformers import BigBirdConfig
from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput
from transformers.models.big_bird.modeling_big_bird import (
    BigBirdForMaskedLM,
    BigBirdForSequenceClassification,
)

from odyssey.models.embeddings import BigBirdEmbeddingsForCEHR


class BigBirdPretrain(pl.LightningModule):
    """BigBird model for pretraining."""

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int = 768,
        time_embeddings_size: int = 32,
        visit_order_size: int = 3,
        type_vocab_size: int = 9,
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
            pad_token_id=padding_idx,
        )

        # BigBirdForMaskedLM
        self.embeddings = BigBirdEmbeddingsForCEHR(
            config=self.config,
            time_embeddings_size=self.time_embeddings_size,
            visit_order_size=self.visit_order_size,
        )

        self.model = BigBirdForMaskedLM(config=self.config)
        self.model.bert.embeddings = self.embeddings

        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module: torch.nn.Module) -> None:
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
        """Apply weight initialization."""
        self.apply(self._init_weights)

    def forward(
        self,
        inputs: Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor, ...], MaskedLMOutput]:
        """Forward pass for the model."""
        concept_ids, type_ids, time_stamps, ages, visit_orders, visit_segments = inputs
        self.embeddings.cache_input(time_stamps, ages, visit_orders, visit_segments)

        if attention_mask is None:
            attention_mask = torch.ones_like(concept_ids)

        return self.model(
            input_ids=concept_ids,
            attention_mask=attention_mask,
            token_type_ids=type_ids,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Any:
        """Train model on training dataset."""
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
        self.model.bert.set_attention_type("block_sparse")
        # Ensure use of mixed precision
        with autocast():
            loss = self(
                inputs,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            ).loss

        (current_lr,) = self.lr_schedulers().get_last_lr()
        self.log_dict(
            dictionary={"train_loss": loss, "lr": current_lr},
            on_step=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Any:
        """Evaluate model on validation dataset."""
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
        self.model.bert.set_attention_type("block_sparse")
        # Ensure use of mixed precision
        with autocast():
            loss = self(
                inputs,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            ).loss

        (current_lr,) = self.lr_schedulers().get_last_lr()
        self.log_dict(
            dictionary={"val_loss": loss, "lr": current_lr},
            on_step=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def configure_optimizers(
        self,
    ) -> Tuple[list[Any], list[dict[str, SequentialLR | str]]]:
        """Configure optimizers and learning rate scheduler."""
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
        )
        # Change optimizer if DeepSpeed strategy is used
        # optimizer = DeepSpeedCPUAdam(  # noqa: ERA001
        #     self.parameters(), lr=self.learning_rate, adamw_mode=True
        # )  # noqa: ERA001

        n_steps = self.trainer.estimated_stepping_batches
        n_warmup_steps = int(0.1 * n_steps)
        n_decay_steps = int(0.9 * n_steps)

        warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=n_warmup_steps,
        )
        decay = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.01,
            total_iters=n_decay_steps,
        )

        scheduler = SequentialLR(
            optimizer=optimizer,
            schedulers=[warmup, decay],
            milestones=[n_warmup_steps],
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


class BigBirdFinetune(pl.LightningModule):
    """BigBird model for fine-tuning."""

    def __init__(
        self,
        pretrained_model: BigBirdPretrain,
        problem_type: str = "single_label_classification",
        num_labels: int = 2,
        learning_rate: float = 5e-5,
        classifier_dropout: float = 0.1,
    ):
        super().__init__()

        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.classifier_dropout = classifier_dropout
        self.test_outputs = []

        self.config = pretrained_model.config
        self.config.num_labels = self.num_labels
        self.config.classifier_dropout = self.classifier_dropout
        self.config.problem_type = problem_type

        self.model = BigBirdForSequenceClassification(config=self.config)
        self.post_init()

        self.pretrained_model = pretrained_model
        self.model.bert = self.pretrained_model.model.bert

    def _init_weights(self, module: torch.nn.Module) -> None:
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
        """Apply weight initialization."""
        self.apply(self._init_weights)

    def forward(
        self,
        inputs: Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor, ...], SequenceClassifierOutput]:
        """Forward pass for the model."""
        concept_ids, type_ids, time_stamps, ages, visit_orders, visit_segments = inputs
        self.model.bert.embeddings.cache_input(
            time_stamps,
            ages,
            visit_orders,
            visit_segments,
        )

        if attention_mask is None:
            attention_mask = torch.ones_like(concept_ids)

        return self.model(
            input_ids=concept_ids,
            attention_mask=attention_mask,
            token_type_ids=type_ids,
            labels=labels,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=return_dict,
        )

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Any:
        """Train model on training dataset."""
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
        self.model.bert.set_attention_type("block_sparse")
        # Ensure use of mixed precision
        with autocast():
            loss = self(
                inputs,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            ).loss

        (current_lr,) = self.lr_schedulers().get_last_lr()
        self.log_dict(
            dictionary={"train_loss": loss, "lr": current_lr},
            on_step=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Any:
        """Evaluate model on validation dataset."""
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
        self.model.bert.set_attention_type("block_sparse")
        # Ensure use of mixed precision
        with autocast():
            loss = self(
                inputs,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            ).loss

        (current_lr,) = self.lr_schedulers().get_last_lr()
        self.log_dict(
            dictionary={"val_loss": loss, "lr": current_lr},
            on_step=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Any:
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
        self.model.bert.set_attention_type("block_sparse")
        # Ensure use of mixed precision
        with autocast():
            outputs = self(
                inputs,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )

        loss = outputs[0]
        logits = outputs[1]
        preds = torch.argmax(logits, dim=1)
        log = {"loss": loss, "preds": preds, "labels": labels, "logits": logits}

        # Append the outputs to the instance attribute
        self.test_outputs.append(log)

        return log

    def on_test_epoch_end(self) -> Any:
        """Evaluate after the test epoch."""
        labels = torch.cat([x["labels"] for x in self.test_outputs]).cpu()
        preds = torch.cat([x["preds"] for x in self.test_outputs]).cpu()
        loss = torch.stack([x["loss"] for x in self.test_outputs]).mean().cpu()
        logits = torch.cat([x["logits"] for x in self.test_outputs]).cpu()

        # Update the saved outputs to include all concatanted batches
        self.test_outputs = {
            "labels": labels,
            "logits": logits,
        }

        if self.config.problem_type == "multi_label_classification":
            preds_one_hot = np.eye(labels.shape[1])[preds]
            accuracy = accuracy_score(labels, preds_one_hot)
            f1 = f1_score(labels, preds_one_hot, average="micro")
            auc = roc_auc_score(labels, preds_one_hot, average="micro")
            precision = precision_score(labels, preds_one_hot, average="micro")
            recall = recall_score(labels, preds_one_hot, average="micro")

        else:  # single_label_classification
            accuracy = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds)
            auc = roc_auc_score(labels, preds)
            precision = precision_score(labels, preds)
            recall = recall_score(labels, preds)

        self.log("test_loss", loss)
        self.log("test_acc", accuracy)
        self.log("test_f1", f1)
        self.log("test_auc", auc)
        self.log("test_precision", precision)
        self.log("test_recall", recall)

        return loss

    def configure_optimizers(
        self,
    ) -> Tuple[list[AdamW], list[dict[str, SequentialLR | str]]]:
        """Configure optimizers and learning rate scheduler."""
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
        )

        n_steps = self.trainer.estimated_stepping_batches
        n_warmup_steps = int(0.1 * n_steps)
        n_decay_steps = int(0.9 * n_steps)

        warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=n_warmup_steps,
        )
        decay = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.01,
            total_iters=n_decay_steps,
        )

        scheduler = SequentialLR(
            optimizer=optimizer,
            schedulers=[warmup, decay],
            milestones=[n_warmup_steps],
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
