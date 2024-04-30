"""Mamba model."""

from typing import Any, Dict, List, Tuple, Optional, Union

import numpy as np
import pytorch_lightning as pl

import torch
from torch import nn, optim
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import LinearLR, SequentialLR

from transformers import MambaConfig, MambaForCausalLM
from transformers.models.mamba.modeling_mamba import MambaCausalLMOutput 


class MambaPretrain(pl.LightningModule):
    """Mamba model for pretraining."""

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int = 768,
        state_size: int = 16,
        num_hidden_layers: int = 32,
        expand: int = 2,
        conv_kernel: int = 4,
        learning_rate: float = 5e-5,
        dropout_prob: float = 0.1,
        padding_idx: int = 0,
        cls_idx: int = 5,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.state_size = state_size
        self.num_hidden_layers = num_hidden_layers
        self.expand = expand
        self.conv_kernel = conv_kernel
        self.learning_rate = learning_rate
        self.dropout_prob = dropout_prob
        self.padding_idx = padding_idx
        self.cls_idx = cls_idx

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
        )

        self.model = MambaForCausalLM(config=self.config)


    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor, ...], MambaCausalLMOutput]:
        """Forward pass for the model."""
        if not labels:
            labels = input_ids

        return self.model(
            input_ids=input_ids,
            labels=labels,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Any:
        """Train model on training dataset."""
        concept_ids = batch["concept_ids"]
        labels = batch["labels"]

        # Ensure use of mixed precision
        with autocast():
            loss = self(
                input_ids=concept_ids,
                labels=labels,
                return_dict=True,
            ).loss

        (current_lr,) = self.lr_schedulers().get_last_lr()
        self.log_dict(
            dictionary={"train_loss": loss, "lr": current_lr},
            on_step=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Any:
        """Evaluate model on validation dataset."""
        concept_ids = batch["concept_ids"]
        labels = batch["labels"]

        # Ensure use of mixed precision
        with autocast():
            loss = self(
                input_ids=concept_ids,
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
