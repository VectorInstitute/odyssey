"""Mamba model."""

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
from torch import nn
from torch.cuda.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from transformers import MambaConfig
from transformers.models.mamba.modeling_mamba import (
    MambaCausalLMOutput,
    MambaForCausalLM,
)

from odyssey.models.ehr_mamba.mamba_utils import (
    MambaForMultiHeadSequenceClassification,
    MambaForSequenceClassification,
    MambaSequenceClassifierOutput,
)
from odyssey.models.embeddings import MambaEmbeddingsForCEHR


class MambaPretrain(pl.LightningModule):
    """Mamba model for pretraining."""

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int = 768,
        time_embeddings_size: int = 32,
        visit_order_size: int = 3,
        type_vocab_size: int = 9,
        max_num_visits: int = 512,
        max_seq_length: int = 2048,
        state_size: int = 16,
        num_hidden_layers: int = 32,
        expand: int = 2,
        conv_kernel: int = 4,
        learning_rate: float = 5e-5,
        dropout_prob: float = 0.1,
        padding_idx: int = 0,
        cls_idx: int = 5,
        use_mambapy: bool = False,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.time_embeddings_size = time_embeddings_size
        self.visit_order_size = visit_order_size
        self.type_vocab_size = type_vocab_size
        self.max_num_visits = max_num_visits
        self.max_seq_length = max_seq_length
        self.state_size = state_size
        self.num_hidden_layers = num_hidden_layers
        self.expand = expand
        self.conv_kernel = conv_kernel
        self.learning_rate = learning_rate
        self.dropout_prob = dropout_prob
        self.padding_idx = padding_idx
        self.cls_idx = cls_idx
        self.use_mambapy = use_mambapy

        self.config = MambaConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.embedding_size,
            state_size=self.state_size,
            num_hidden_layers=self.num_hidden_layers,
            expand=self.expand,
            conv_kernel=self.conv_kernel,
            pad_token_id=self.padding_idx,
            bos_token_id=self.cls_idx,
            eos_token_id=self.padding_idx,
            use_mambapy=self.use_mambapy,
        )
        self.embeddings = MambaEmbeddingsForCEHR(
            config=self.config,
            type_vocab_size=self.type_vocab_size,
            max_num_visits=self.max_num_visits,
            time_embeddings_size=self.time_embeddings_size,
            visit_order_size=self.visit_order_size,
            hidden_dropout_prob=self.dropout_prob,
        )
        # Initialize weights and apply final processing
        self.post_init()

        # Mamba has its own initialization
        self.model = MambaForCausalLM(config=self.config)

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
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor, ...], MambaCausalLMOutput]:
        """Forward pass for the model."""
        concept_ids, type_ids, time_stamps, ages, visit_orders, visit_segments = inputs
        inputs_embeds = self.embeddings(
            input_ids=concept_ids,
            token_type_ids_batch=type_ids,
            time_stamps=time_stamps,
            ages=ages,
            visit_orders=visit_orders,
            visit_segments=visit_segments,
        )

        if labels is None:
            labels = concept_ids

        return self.model(
            inputs_embeds=inputs_embeds,
            labels=labels,
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

        # Ensure use of mixed precision
        with autocast():
            loss = self(
                inputs,
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

        # Ensure use of mixed precision
        with autocast():
            loss = self(
                inputs,
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
        optimizer = AdamW(
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


class MambaFinetune(pl.LightningModule):
    """Mamba model for fine-tuning."""

    def __init__(
        self,
        pretrained_model: MambaPretrain,
        problem_type: str = "single_label_classification",
        num_labels: int = 2,
        num_tasks: int = 6,
        learning_rate: float = 5e-5,
        classifier_dropout: float = 0.1,
        multi_head: bool = False,
    ):
        super().__init__()

        self.num_labels = num_labels
        self.num_tasks = num_tasks
        self.learning_rate = learning_rate
        self.classifier_dropout = classifier_dropout
        self.multi_head = multi_head
        self.test_outputs = []

        self.config = pretrained_model.config
        self.config.num_labels = self.num_labels
        self.config.classifier_dropout = self.classifier_dropout
        self.config.problem_type = problem_type

        if self.multi_head:
            self.model = MambaForMultiHeadSequenceClassification(
                config=self.config, num_tasks=self.num_tasks
            )
        else:
            self.model = MambaForSequenceClassification(config=self.config)

        # self.post_init()

        self.pretrained_model = pretrained_model
        self.embeddings = self.pretrained_model.embeddings
        self.model.backbone = self.pretrained_model.model.backbone

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
        labels: Optional[torch.Tensor] = None,
        task_indices: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor, ...], MambaSequenceClassifierOutput]:
        """Forward pass for the model."""
        concept_ids, type_ids, time_stamps, ages, visit_orders, visit_segments = inputs
        inputs_embeds = self.embeddings(
            input_ids=concept_ids,
            token_type_ids_batch=type_ids,
            time_stamps=time_stamps,
            ages=ages,
            visit_orders=visit_orders,
            visit_segments=visit_segments,
        )

        return self.model(
            input_ids=concept_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            task_indices=task_indices,
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
        task_indices = batch["task_indices"]

        # Ensure use of mixed precision
        with autocast():
            loss = self(
                inputs,
                labels=labels,
                task_indices=task_indices,
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
        task_indices = batch["task_indices"]

        # Ensure use of mixed precision
        with autocast():
            loss = self(
                inputs,
                labels=labels,
                task_indices=task_indices,
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
        task_indices = batch["task_indices"]

        # Ensure use of mixed precision
        with autocast():
            outputs = self(
                inputs,
                labels=labels,
                task_indices=task_indices,
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
    ) -> Tuple[list[Any], list[dict[str, SequentialLR | str]]]:
        """Configure optimizers and learning rate scheduler."""
        optimizer = AdamW(
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
