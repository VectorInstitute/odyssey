"""EHR-Mamba3: Clinical foundation model using the Mamba-3 SSM architecture.

Mamba-3 introduces three improvements over Mamba-2:
  - Trapezoidal discretization (more expressive, more stable)
  - Complex-valued state updates (richer state tracking)
  - MIMO (Multi-Input Multi-Output) mode replacing the short causal convolution

This module wraps ``mamba_ssm.MambaLMHeadModel`` (with ``ssm_cfg={"layer":
"Mamba3"}``) and injects clinical EHR-specific embeddings (token types, time
stamps, patient ages, visit order/segment) via ``CachedEHREmbeddings``.
"""

from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F  # noqa: N812
from mamba_ssm.models.config_mamba import MambaConfig as MambaSsmConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from odyssey.models.embeddings import CachedEHREmbeddings


# ──────────────────────────────────────────────────────────────────────────────
# Classification head
# ──────────────────────────────────────────────────────────────────────────────


class Mamba3ClassificationHead(nn.Module):
    """Dense → GELU → Dense classification head."""

    def __init__(self, hidden_size: int, num_labels: int, dropout: float = 0.1) -> None:
        """Initialize the classification head."""
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.dropout(x)
        x = self.dense(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return self.out_proj(x)  # type: ignore[no-any-return]


# ──────────────────────────────────────────────────────────────────────────────
# Pre-training model
# ──────────────────────────────────────────────────────────────────────────────


class Mamba3Pretrain(pl.LightningModule):
    """EHR-Mamba3 pre-training with next-token prediction.

    The backbone is ``mamba_ssm.MambaLMHeadModel`` configured with
    ``ssm_cfg={"layer": "Mamba3"}``.  The standard token-embedding layer
    inside the backbone is replaced with :class:`CachedEHREmbeddings` so that
    clinical auxiliary inputs (token types, time stamps, ages, visit
    order/segment) enrich every forward pass.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int = 768,
        time_embeddings_size: int = 32,
        visit_order_size: int = 3,
        type_vocab_size: int = 9,
        max_num_visits: int = 512,
        max_seq_length: int = 2048,
        # Mamba-3 backbone parameters
        state_size: int = 128,
        num_hidden_layers: int = 32,
        d_intermediate: int = 0,
        # Mamba-3 block parameters (passed via ssm_cfg)
        headdim: int = 64,
        is_mimo: bool = True,
        mimo_rank: int = 4,
        chunk_size: int = 256,
        # Training parameters
        learning_rate: float = 5e-5,
        dropout_prob: float = 0.1,
        padding_idx: int = 0,
        cls_idx: int = 5,
    ) -> None:
        """Initialize EHR-Mamba3 for pre-training."""
        super().__init__()
        self.save_hyperparameters()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.padding_idx = padding_idx

        # Build mamba_ssm config for a Mamba-3 backbone
        ssm_cfg: Dict[str, Any] = {
            "layer": "Mamba3",
            "d_state": state_size,
            "headdim": headdim,
            "is_mimo": is_mimo,
            "mimo_rank": mimo_rank,
            "chunk_size": chunk_size,
        }
        backbone_cfg = MambaSsmConfig(
            d_model=embedding_size,
            d_intermediate=d_intermediate,
            n_layer=num_hidden_layers,
            vocab_size=vocab_size,
            ssm_cfg=ssm_cfg,
            rms_norm=True,
            residual_in_fp32=True,
            fused_add_norm=True,
        )

        # Instantiate backbone
        self.model = MambaLMHeadModel(backbone_cfg)

        # Build EHR-specific embeddings and swap them into the backbone
        self.embeddings = CachedEHREmbeddings(
            vocab_size=vocab_size,
            hidden_size=embedding_size,
            padding_idx=padding_idx,
            type_vocab_size=type_vocab_size,
            max_num_visits=max_num_visits,
            time_embeddings_size=time_embeddings_size,
            visit_order_size=visit_order_size,
            hidden_dropout_prob=dropout_prob,
        )
        # Replace the backbone's plain Embedding with the EHR wrapper
        self.model.backbone.embedding = self.embeddings

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

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
    ) -> torch.Tensor:
        """Forward pass returning next-token-prediction cross-entropy loss."""
        concept_ids, type_ids, time_stamps, ages, visit_orders, visit_segments = inputs

        # Cache auxiliary inputs so the backbone embedding can consume them
        self.embeddings.set_aux_inputs(
            type_ids=type_ids,
            time_stamps=time_stamps,
            ages=ages,
            visit_orders=visit_orders,
            visit_segments=visit_segments,
        )

        logits = self.model(concept_ids).logits  # (B, L, V)

        # Shift for causal language modelling
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = (labels if labels is not None else concept_ids)[
            :, 1:
        ].contiguous()

        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.padding_idx,
        )

    def get_logits(
        self,
        inputs: Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        """Return raw logits (B, L, V) without computing loss."""
        concept_ids, type_ids, time_stamps, ages, visit_orders, visit_segments = inputs
        self.embeddings.set_aux_inputs(
            type_ids=type_ids,
            time_stamps=time_stamps,
            ages=ages,
            visit_orders=visit_orders,
            visit_segments=visit_segments,
        )
        return self.model(concept_ids).logits  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def _step(self, batch: Dict[str, Any], stage: str) -> Any:
        inputs = (
            batch["concept_ids"],
            batch["type_ids"],
            batch["time_stamps"],
            batch["ages"],
            batch["visit_orders"],
            batch["visit_segments"],
        )
        loss = self(inputs, labels=batch.get("labels"))

        optimizer = self.optimizers()
        current_lr = (
            optimizer.param_groups[0]["lr"]  # type: ignore[union-attr]
            if optimizer is not None
            else self.learning_rate
        )
        self.log_dict(
            {f"{stage}_loss": loss, "lr": current_lr},
            on_step=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Any:
        """Training step."""
        return self._step(batch, "train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Any:
        """Run validation step."""
        return self._step(batch, "val")

    def configure_optimizers(self) -> Any:
        """Configure AdamW with linear warmup + cosine decay."""
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        n_steps = self.trainer.estimated_stepping_batches
        n_warmup = int(0.1 * n_steps)
        n_decay = int(0.9 * n_steps)

        warmup = LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=n_warmup
        )
        decay = CosineAnnealingLR(
            optimizer, T_max=n_decay, eta_min=self.learning_rate * 0.01
        )
        scheduler = SequentialLR(
            optimizer, schedulers=[warmup, decay], milestones=[n_warmup]
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


# ──────────────────────────────────────────────────────────────────────────────
# Fine-tuning model
# ──────────────────────────────────────────────────────────────────────────────


class Mamba3Finetune(pl.LightningModule):
    """EHR-Mamba3 fine-tuning for single-task or multi-task classification.

    Loads a pre-trained :class:`Mamba3Pretrain` checkpoint and attaches one or
    more :class:`Mamba3ClassificationHead` modules.  The last non-padding
    hidden state is pooled and routed to the appropriate task head.
    """

    def __init__(
        self,
        pretrained_model: Mamba3Pretrain,
        problem_type: str = "single_label_classification",
        num_labels: int = 2,
        num_tasks: int = 1,
        learning_rate: float = 5e-5,
        classifier_dropout: float = 0.1,
        multi_head: bool = False,
    ) -> None:
        """Initialize EHR-Mamba3 for fine-tuning."""
        super().__init__()

        self.num_labels = num_labels
        self.num_tasks = num_tasks
        self.learning_rate = learning_rate
        self.problem_type = problem_type
        self.multi_head = multi_head
        self.padding_idx = pretrained_model.padding_idx
        self.test_outputs: List[Dict[str, torch.Tensor]] = []

        # Reuse the pre-trained backbone (with EHR embeddings already swapped in)
        self.backbone = pretrained_model.model.backbone
        self.embeddings = pretrained_model.embeddings
        hidden_size = pretrained_model.embedding_size

        if multi_head:
            self.heads = nn.ModuleList(
                [
                    Mamba3ClassificationHead(
                        hidden_size, num_labels, classifier_dropout
                    )
                    for _ in range(num_tasks)
                ]
            )
        else:
            self.heads = nn.ModuleList(
                [Mamba3ClassificationHead(hidden_size, num_labels, classifier_dropout)]
            )

    # ------------------------------------------------------------------
    # Shared forward logic
    # ------------------------------------------------------------------

    def _encode(
        self,
        inputs: Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run backbone and return (hidden_states, pooled_last_token)."""
        concept_ids, type_ids, time_stamps, ages, visit_orders, visit_segments = inputs

        self.embeddings.set_aux_inputs(
            type_ids=type_ids,
            time_stamps=time_stamps,
            ages=ages,
            visit_orders=visit_orders,
            visit_segments=visit_segments,
        )

        hidden_states = self.backbone(concept_ids)  # (B, L, H)

        # Pool the last non-padding token
        batch_size = hidden_states.shape[0]
        pad_mask = concept_ids == self.padding_idx  # (B, L)
        # First True in pad_mask → first padding position; -1 gives last content token
        last_idx = pad_mask.int().argmax(dim=-1) - 1  # (B,)
        last_idx = last_idx.clamp(min=0)
        pooled = hidden_states[
            torch.arange(batch_size, device=hidden_states.device), last_idx
        ]

        return hidden_states, pooled

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute loss based on problem type."""
        if self.problem_type == "regression":
            loss_fn: Union[MSELoss, CrossEntropyLoss, BCEWithLogitsLoss] = MSELoss()
            return loss_fn(  # type: ignore[no-any-return]
                logits.squeeze() if self.num_labels == 1 else logits,
                labels.squeeze() if self.num_labels == 1 else labels,
            )
        if self.problem_type == "single_label_classification":
            loss_fn = CrossEntropyLoss()
            return loss_fn(logits.view(-1, self.num_labels), labels.view(-1))  # type: ignore[no-any-return]
        # multi_label_classification
        loss_fn = BCEWithLogitsLoss()
        return loss_fn(logits, labels.float())  # type: ignore[no-any-return]

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
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Forward pass returning (loss, logits)."""
        _, pooled = self._encode(inputs)
        batch_size = pooled.shape[0]

        if self.multi_head and task_indices is not None:
            logits = torch.zeros(batch_size, self.num_labels, device=pooled.device)
            for i in range(batch_size):
                head_idx = int(task_indices[i].item())
                logits[i] = self.heads[head_idx](pooled[i].unsqueeze(0))
        else:
            logits = self.heads[0](pooled)

        loss = self._loss(logits, labels) if labels is not None else None
        return loss, logits

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def _step(self, batch: Dict[str, Any], stage: str) -> Any:
        inputs = (
            batch["concept_ids"],
            batch["type_ids"],
            batch["time_stamps"],
            batch["ages"],
            batch["visit_orders"],
            batch["visit_segments"],
        )
        loss, _ = self(
            inputs,
            labels=batch["labels"],
            task_indices=batch.get("task_indices"),
        )
        optimizer = self.optimizers()
        current_lr = (
            optimizer.param_groups[0]["lr"]  # type: ignore[union-attr]
            if optimizer is not None
            else self.learning_rate
        )
        self.log_dict(
            {f"{stage}_loss": loss, "lr": current_lr},
            on_step=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Any:
        """Training step."""
        return self._step(batch, "train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Any:
        """Run validation step."""
        return self._step(batch, "val")

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
        loss, logits = self(
            inputs,
            labels=batch["labels"],
            task_indices=batch.get("task_indices"),
        )
        preds = torch.argmax(logits, dim=1)
        log: Dict[str, Any] = {
            "loss": loss,
            "preds": preds,
            "labels": batch["labels"],
            "logits": logits,
        }
        self.test_outputs.append(log)
        return log

    def on_test_epoch_end(self) -> None:
        """Aggregate test metrics after the test epoch."""
        labels = torch.cat([x["labels"] for x in self.test_outputs]).cpu()
        preds = torch.cat([x["preds"] for x in self.test_outputs]).cpu()
        loss = torch.stack([x["loss"] for x in self.test_outputs]).mean().cpu()
        logits = torch.cat([x["logits"] for x in self.test_outputs]).cpu()

        self.test_outputs = cast(
            List[Dict[str, torch.Tensor]],
            [{"labels": labels, "logits": logits}],
        )

        if self.problem_type == "multi_label_classification":
            preds_oh = np.eye(labels.shape[1])[preds.numpy()]
            accuracy = accuracy_score(labels.numpy(), preds_oh)
            f1 = f1_score(labels.numpy(), preds_oh, average="micro")
            auc = roc_auc_score(labels.numpy(), preds_oh, average="micro")
            precision = precision_score(labels.numpy(), preds_oh, average="micro")
            recall = recall_score(labels.numpy(), preds_oh, average="micro")
        else:
            accuracy = accuracy_score(labels.numpy(), preds.numpy())
            f1 = f1_score(labels.numpy(), preds.numpy())
            auc = roc_auc_score(labels.numpy(), preds.numpy())
            precision = precision_score(labels.numpy(), preds.numpy())
            recall = recall_score(labels.numpy(), preds.numpy())

        self.log("test_loss", loss)
        self.log("test_acc", accuracy)
        self.log("test_f1", f1)
        self.log("test_auc", auc)
        self.log("test_precision", precision)
        self.log("test_recall", recall)

    def configure_optimizers(self) -> Any:
        """Configure AdamW with linear warmup + cosine decay."""
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        n_steps = self.trainer.estimated_stepping_batches
        n_warmup = int(0.1 * n_steps)
        n_decay = int(0.9 * n_steps)

        warmup = LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=n_warmup
        )
        decay = CosineAnnealingLR(
            optimizer, T_max=n_decay, eta_min=self.learning_rate * 0.01
        )
        scheduler = SequentialLR(
            optimizer, schedulers=[warmup, decay], milestones=[n_warmup]
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
