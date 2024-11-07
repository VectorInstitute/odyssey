"""Mamba2 model."""

from typing import Any, Dict, Tuple

import pytorch_lightning as pl
from torch.cuda.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import Mamba2Config, Mamba2ForCausalLM


class Mamba2Pretrain(pl.LightningModule):
    """Mamba2 model for pretraining."""

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int = 768,
        max_num_visits: int = 512,
        max_seq_length: int = 2048,
        state_size: int = 64,
        num_heads: int = 24,
        head_dim: int = 64,
        num_hidden_layers: int = 32,
        expand: int = 2,
        conv_kernel: int = 4,
        learning_rate: float = 5e-5,
        dropout_prob: float = 0.1,
        padding_idx: int = 0,
        cls_idx: int = 1,  # used as bos token
        eos_idx: int = 2,
        n_groups: int = 1,
        chunk_size: int = 256,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.max_num_visits = max_num_visits
        self.max_seq_length = max_seq_length
        self.state_size = state_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_hidden_layers = num_hidden_layers
        self.expand = expand
        self.conv_kernel = conv_kernel
        self.learning_rate = learning_rate
        self.dropout_prob = dropout_prob
        self.padding_idx = padding_idx
        self.cls_idx = cls_idx
        self.eos_idx = eos_idx
        self.n_groups = n_groups
        self.chunk_size = chunk_size

        self.config = Mamba2Config(
            vocab_size=self.vocab_size,
            hidden_size=self.embedding_size,
            state_size=self.state_size,
            num_hidden_layers=self.num_hidden_layers,
            expand=self.expand,
            conv_kernel=self.conv_kernel,
            pad_token_id=self.padding_idx,
            bos_token_id=self.cls_idx,
            eos_token_id=self.eos_idx,
            n_groups=self.n_groups,
            chunk_size=self.chunk_size,
            dropout=self.dropout_prob,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            max_position_embeddings=self.max_seq_length,
        )

        # Mamba has its own initialization
        self.model = Mamba2ForCausalLM(config=self.config)

    def _step(self, batch: Dict[str, Any], batch_idx: int, stage: str) -> Any:
        """Run a single step for training or validation.

        Args:
            batch: Input batch dictionary
            batch_idx: Index of current batch
            stage: Either 'train' or 'val'
        """
        concept_ids = batch["concept_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]

        # Ensure use of mixed precision
        with autocast():
            loss = self.model(
                concept_ids,
                labels=labels,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=False,
            ).loss

        (current_lr,) = self.lr_schedulers().get_last_lr()
        self.log_dict(
            dictionary={f"{stage}_loss": loss, "lr": current_lr},
            on_step=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Any:
        """Train model on training dataset."""
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Any:
        """Evaluate model on validation dataset."""
        return self._step(batch, batch_idx, "val")

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
        decay = CosineAnnealingLR(
            optimizer,
            T_max=n_decay_steps,
            eta_min=self.learning_rate * 0.01,
        )
        scheduler = SequentialLR(
            optimizer=optimizer,
            schedulers=[warmup, decay],
            milestones=[n_warmup_steps],
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
