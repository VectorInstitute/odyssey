from typing import Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import BertConfig
from transformers.modeling_outputs import (MaskedLMOutput,
                                           SequenceClassifierOutput)
from transformers.models.bert.modeling_bert import (BertEncoder,
                                                    BertOnlyMLMHead,
                                                    BertPooler)

from .embeddings import Embeddings


class BertPretrain(pl.LightningModule):
    """BERT model for pretraining."""

    def __init__(
            self,
            vocab_size,
            embedding_size: int = 128,
            time_embeddings_size: int = 16,
            max_seq_length: int = 512,
            depth: int = 5,
            num_heads: int = 8,
            intermediate_size: int = 2048,
            learning_rate: float = 2e-4,
            eta_min: float = 1e-8,
            num_iterations: int = 10,
            increase_factor: float = 2,
            dropout_prob: float = 0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.time_embeddings_size = time_embeddings_size
        self.max_seq_length = max_seq_length
        self.depth = depth
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.learning_rate = learning_rate
        self.eta_min = eta_min
        self.num_iterations = num_iterations
        self.increase_factor = increase_factor
        self.dropout_prob = dropout_prob

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
        )
        # BertForMaskedLM
        ## BertModel
        self.embeddings = Embeddings(
            vocab_size=self.vocab_size,
            embedding_size=self.embedding_size,
            time_embedding_size=self.time_embeddings_size,
            max_len=self.max_seq_length,
        )
        self.encoder = BertEncoder(self.bert_config)
        self.pooler = BertPooler(self.bert_config)
        ## MLMHEAD
        self.cls = BertOnlyMLMHead(self.bert_config)
        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module) -> None:
        """ Initialize the weights """
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
        self.apply(self._init_weights)

    def forward(
            self,
            input: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, ...], MaskedLMOutput]:
        """Forward pass for the model."""
        concept_ids, time_stamps, ages, visit_orders, visit_segments = input
        embedding_output = self.embeddings(
            concept_ids, time_stamps, ages, visit_orders, visit_segments
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
                prediction_scores.view(-1, self.bert_config.vocab_size), labels.view(-1)
            )

        if not return_dict:
            output = (prediction_scores,) + encoder_outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """Training step."""
        input = batch['concept_ids'], batch['time_stamps'], batch['ages'], \
            batch['visit_orders'], batch['visit_segments']
        labels = batch['labels']
        attention_mask = batch['attention_mask']
        loss = self(input, attention_mask=attention_mask, labels=labels, return_dict=True)[0]
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        """Validation step."""
        input = batch['concept_ids'], batch['time_stamps'], batch['ages'], \
            batch['visit_orders'], batch['visit_segments']
        labels = batch['labels']
        attention_mask = batch['attention_mask']
        loss = self(input, attention_mask=attention_mask, labels=labels, return_dict=True)[0]
        self.log('val_loss', loss, sync_dist=True)
        return loss

    def configure_optimizers(self) -> dict:
        """Configure optimizers and learning rate scheduler."""
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)  # ADIBQ WHY IS THIS NOT ADAMW
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.num_iterations,
            T_mult=self.increase_factor,
            eta_min=self.eta_min,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class BertFinetune(pl.LightningModule):
    """BERT model for finetuning."""

    def __init__(
            self,
            pretrained_model: BertPretrain,
            num_labels: int = 2,
            hidden_size: int = 128,
            classifier_dropout: float = 0.1,
            hidden_dropout_prob: float = 0.1,
            learning_rate: float = 2e-5,
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
            self.config.classifier_dropout if
            self.config.classifier_dropout is not None else self.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

        self.post_init()
        self.pretrained_model = pretrained_model

    def _init_weights(self, module):
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
        self.apply(self._init_weights)

    def forward(
            self,
            input: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, ...], SequenceClassifierOutput]:
        """Forward pass for the model."""
        if attention_mask is None:
            attention_mask = torch.ones_like(input[0])
        outputs = self.pretrained_model(
            input,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs['hidden_states']  # hidden_states
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

    def training_step(self, batch, batch_idx):
        """Training step."""
        input = batch['concept_ids'], batch['time_stamps'], batch['ages'], \
            batch['visit_orders'], batch['visit_segments']
        labels = batch['labels']
        attention_mask = batch['attention_mask']
        loss = self(input, attention_mask=attention_mask, labels=labels, return_dict=True)[0]
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        input = batch['concept_ids'], batch['time_stamps'], batch['ages'], \
            batch['visit_orders'], batch['visit_segments']
        labels = batch['labels']
        attention_mask = batch['attention_mask']
        loss = self(input, attention_mask=attention_mask, labels=labels, return_dict=True)[0]
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        input = batch['concept_ids'], batch['time_stamps'], batch['ages'], \
            batch['visit_orders'], batch['visit_segments']
        labels = batch['labels']
        attention_mask = batch['attention_mask']
        outputs = self(input, attention_mask=attention_mask, labels=labels, return_dict=True)
        loss = outputs[0]
        logits = outputs[1]
        preds = torch.argmax(logits, dim=1)
        return {'loss': loss, 'preds': preds, 'labels': labels}

    def test_epoch_end(self, outputs):
        """Evaluate after the test epoch."""
        labels = torch.cat([x['labels'] for x in outputs]).cpu()
        preds = torch.cat([x['preds'] for x in outputs]).cpu()
        loss = torch.stack([x['loss'] for x in outputs]).mean().cpu()
        self.log('test_loss', loss)
        self.log('test_acc', accuracy_score(preds, labels))
        self.log('test_f1', f1_score(preds, labels))
        self.log('test_auc', roc_auc_score(preds, labels))
        return loss

    def configure_optimizers(self):
        """Configure optimizers and learning rate scheduler."""
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.num_iterations,
            T_mult=self.increase_factor,
            eta_min=self.eta_min,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
