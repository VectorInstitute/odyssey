"""CEHR-BERT model."""

from typing import Any, Optional, Sequence, Tuple, Union

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
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from transformers import BertConfig
from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput
from transformers.models.bert.modeling_bert import (
    BertEncoder,
    BertOnlyMLMHead,
    BertPooler,
)

from odyssey.models.embeddings import BERTEmbeddingsForCEHR


class BertPretrain(pl.LightningModule):
    """BERT model for pretraining."""

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int = 768,
        time_embeddings_size: int = 32,
        type_vocab_size: int = 9,
        max_seq_length: int = 512,
        depth: int = 5,
        num_heads: int = 8,
        intermediate_size: int = 3072,
        learning_rate: float = 2e-4,
        eta_min: float = 1e-8,
        num_iterations: int = 10,
        increase_factor: float = 2,
        dropout_prob: float = 0.1,
        padding_idx: int = 1,
        use_adamw: bool = True,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.time_embeddings_size = time_embeddings_size
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
        self.use_adamw = use_adamw

        self.bert_config = BertConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.embedding_size,
            num_hidden_layers=self.depth,
            num_attention_heads=self.num_heads,
            intermediate_size=self.intermediate_size,
            hidden_dropout_prob=self.dropout_prob,
            attention_probs_dropout_prob=self.dropout_prob,
            max_position_embeddings=self.max_seq_length,
            is_decoder=False,
            pad_token_id=self.padding_idx,
        )
        # BertForMaskedLM
        ## BertModel
        self.embeddings = BERTEmbeddingsForCEHR(
            vocab_size=self.vocab_size,
            embedding_size=self.embedding_size,
            time_embeddings_size=self.time_embeddings_size,
            type_vocab_size=self.type_vocab_size,
            max_len=self.max_seq_length,
            padding_idx=self.padding_idx,
        )
        self.encoder = BertEncoder(self.bert_config)
        ## MLMHEAD
        self.cls = BertOnlyMLMHead(self.bert_config)
        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.bert_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.bert_config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def post_init(self) -> None:
        """Apply post initialization."""
        self.apply(self._init_weights)

    def forward(
        self,
        input_: Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, ...], MaskedLMOutput]:
        """Forward pass for the model."""
        concept_ids, type_ids, time_stamps, ages, visit_orders, visit_segments = input_
        embedding_output = self.embeddings(
            concept_ids,
            type_ids,
            time_stamps,
            ages,
            visit_orders,
            visit_segments,
        )
        if attention_mask is None:
            attention_mask = torch.ones_like(concept_ids)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]  # last hidden state
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.bert_config.vocab_size),
                labels.view(-1),
            )

        if not return_dict:
            output = (prediction_scores,) + encoder_outputs[2:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Compute training step."""
        input_ = (
            batch["concept_ids"],
            batch["type_ids"],
            batch["time_stamps"],
            batch["ages"],
            batch["visit_orders"],
            batch["visit_segments"],
        )
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]
        loss = self(
            input_,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )[0]
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Compute validation step."""
        input_ = (
            batch["concept_ids"],
            batch["type_ids"],
            batch["time_stamps"],
            batch["ages"],
            batch["visit_orders"],
            batch["visit_segments"],
        )
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]
        loss = self(
            input_,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )[0]
        self.log("val_loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self) -> Tuple[Sequence[optim.Optimizer], Sequence[Any]]:
        """Configure optimizers and learning rate scheduler."""
        if self.use_adamw:
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

            linear_decay = LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.01,
                total_iters=n_decay_steps,
            )

            scheduler = SequentialLR(
                optimizer=optimizer,
                schedulers=[warmup, linear_decay],
                milestones=[n_warmup_steps],
            )

            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

        optimizer = optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.num_iterations,
            T_mult=self.increase_factor,
            eta_min=self.eta_min,
        )
        return [optimizer], [{"scheduler": scheduler}]


class BertFinetune(pl.LightningModule):
    """BERT model for finetuning."""

    def __init__(
        self,
        pretrained_model: BertPretrain,
        num_labels: int = 2,
        hidden_size: int = 768,
        classifier_dropout: float = 0.1,
        hidden_dropout_prob: float = 0.1,
        learning_rate: float = 2e-5,
        eta_min: float = 1e-8,
        num_iterations: int = 10,
        increase_factor: float = 2,
        use_adamw: bool = True,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.eta_min = eta_min
        self.num_iterations = num_iterations
        self.increase_factor = increase_factor
        self.use_adamw = use_adamw

        self.config = BertConfig(
            num_labels=num_labels,
            hidden_size=hidden_size,
            classifier_dropout=classifier_dropout,
            hidden_dropout_prob=hidden_dropout_prob,
        )

        # BertForSequenceClassification
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

        self.test_outputs = []

    def _init_weights(self, module: nn.Module):
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

    def post_init(self):
        """Apply post initialization."""
        self.apply(self._init_weights)

    def forward(
        self,
        input_: Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, ...], SequenceClassifierOutput]:
        """Forward pass for the model."""
        if attention_mask is None:
            attention_mask = torch.ones_like(input_[0])
        outputs = self.pretrained_model(
            input_,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
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

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Compute training step."""
        input_ = (
            batch["concept_ids"],
            batch["type_ids"],
            batch["time_stamps"],
            batch["ages"],
            batch["visit_orders"],
            batch["visit_segments"],
        )
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]
        loss = self(
            input_,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )[0]
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Compute validation step."""
        input_ = (
            batch["concept_ids"],
            batch["type_ids"],
            batch["time_stamps"],
            batch["ages"],
            batch["visit_orders"],
            batch["visit_segments"],
        )
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]
        loss = self(
            input_,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )[0]
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        """Compute test step."""
        input_ = (
            batch["concept_ids"],
            batch["type_ids"],
            batch["time_stamps"],
            batch["ages"],
            batch["visit_orders"],
            batch["visit_segments"],
        )
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]
        outputs = self(
            input_,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        loss = outputs[0]
        logits = outputs[1]
        preds = torch.argmax(logits, dim=1)
        log = {"loss": loss, "preds": preds, "labels": labels}
        self.test_outputs.append(log)
        return log

    def on_test_epoch_end(self):
        """Compute metrics on test set."""
        labels = torch.cat([x["labels"] for x in self.test_outputs]).cpu()
        preds = torch.cat([x["preds"] for x in self.test_outputs]).cpu()
        loss = torch.stack([x["loss"] for x in self.test_outputs]).mean().cpu()
        self.log("test_loss", loss)
        self.log("test_acc", accuracy_score(labels, preds))
        self.log("test_f1", f1_score(labels, preds))
        self.log("test_auc", roc_auc_score(labels, preds))
        self.log("test_precision", precision_score(labels, preds))
        self.log("test_recall", recall_score(labels, preds))
        return loss

    def configure_optimizers(self) -> Tuple[Sequence[optim.Optimizer], Sequence[Any]]:
        """Configure optimizers and learning rate scheduler."""
        if self.use_adamw:
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

            linear_decay = LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.01,
                total_iters=n_decay_steps,
            )

            scheduler = SequentialLR(
                optimizer=optimizer,
                schedulers=[warmup, linear_decay],
                milestones=[n_warmup_steps],
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.num_iterations,
            T_mult=self.increase_factor,
            eta_min=self.eta_min,
        )
        return [optimizer], [{"scheduler": scheduler}]
