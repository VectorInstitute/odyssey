"""
File: Bi-LSTM.ipynb
Code to train and evaluate a bi-directional LSTM model on MIMIC-IV FHIR dataset.
"""

import sys
import os

from tqdm import tqdm
from typing import Optional, Sequence, Union, Any, List, Tuple, Dict, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score
from sklearn.metrics import (
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import sigmoid
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn.utils.rnn import pack_padded_sequence

ROOT = "/fs01/home/afallah/odyssey/odyssey"
os.chdir(ROOT)

from models.big_bird_cehr.data import FinetuneDataset
from models.big_bird_cehr.tokenizer import ConceptTokenizer
from models.big_bird_cehr.embeddings import Embeddings

DATA_ROOT = f"{ROOT}/data/slurm_data/512/one_month"
DATA_PATH = f"{DATA_ROOT}/pretrain.parquet"
FINE_TUNE_PATH = f"{DATA_ROOT}/fine_tune.parquet"
TEST_DATA_PATH = f"{DATA_ROOT}/fine_test.parquet"
SAVE_MODEL_DIR = f"{ROOT}LSTM_V2.pt"


# save parameters and configurations
class config:
    """ A simple class to store all configurations """
    seed = 23
    data_dir = DATA_ROOT
    test_size = 0.2
    batch_size = 64
    num_workers = 3
    vocab_size = None
    embedding_size = 768
    time_embeddings_size = 32
    type_vocab_size = 8
    max_len = 512
    padding_idx = None
    device = torch.device("cuda")


def seed_all(seed: int) -> None:
    """ Seed all parts of the training process """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Define dataset with token lengths
class DatasetWithTokenLength(Dataset):
    def __init__(self, tokenized_data: FinetuneDataset, length_data: np.ndarray):
        super(Dataset, self).__init__()

        self.tokenized_data = tokenized_data
        self.length_data = length_data

        assert len(tokenized_data) == len(length_data), "Datasets have different lengths"

        self.sorted_indices = sorted(
            range(len(length_data)), key=lambda x: length_data[x], reverse=True
        )

    def __len__(self) -> int:
        return len(self.tokenized_data)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        index = self.sorted_indices[index]
        return self.tokenized_data[index], min(config.max_len, self.length_data[index])


# Define model architecture
class BiLSTMModel(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            hidden_size: int,
            num_layers: int,
            output_size: int,
            dropout_rate: float
    ):
        super(BiLSTMModel, self).__init__()

        self.embeddings = Embeddings(
            vocab_size=config.vocab_size,
            embedding_size=config.embedding_size,
            time_embedding_size=config.time_embeddings_size,
            type_vocab_size=config.type_vocab_size,
            max_len=config.max_len,
            padding_idx=config.padding_idx
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate,
        )

        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, inputs: Tuple[Any], lengths: torch.Tensor) -> torch.Tensor:
        """ Forward pass of Bi-LSTM model """
        embed = self.embeddings(*inputs)
        packed_embed = pack_padded_sequence(embed, lengths.cpu(), batch_first=True)

        lstm_out, (hidden_state, cell_state) = self.lstm(packed_embed)
        output = torch.cat((hidden_state[-2, :, :], hidden_state[-1, :, :]), dim=1)

        output = self.dropout(self.batch_norm(output))
        output = self.linear(output)
        return output

    @staticmethod
    def get_inputs_labels(sequences: Dict[str, torch.Tensor]) -> Tuple[Any, torch.Tensor]:
        """ Create inputs tuples from a dictionary of sequences """
        labels = sequences["labels"].view(-1, 1).to(config.device)
        inputs = (
            sequences["concept_ids"].to(config.device),
            sequences["type_ids"].to(config.device),
            sequences["time_stamps"].to(config.device),
            sequences["ages"].to(config.device),
            sequences["visit_orders"].to(config.device),
            sequences["visit_segments"].to(config.device),
        )

        return inputs, labels.float()

    @staticmethod
    def get_balanced_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
        """ Return the balanced accuracy metric by comparing outputs to labels """
        predictions = torch.round(sigmoid(outputs))
        predictions = predictions.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        return balanced_accuracy_score(labels, predictions)


if __name__ == '__main__':
    seed_all(config.seed)
    print(f"Cuda: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    # Load data
    pretrain_data = pd.read_parquet(DATA_PATH)
    pretrain_data = pretrain_data[pretrain_data['event_tokens_512'].notnull()]

    finetune_data = pd.read_parquet(FINE_TUNE_PATH)
    finetune_data = finetune_data[finetune_data['event_tokens_512'].notnull()]

    test_data = pd.read_parquet(TEST_DATA_PATH)
    test_data = test_data[test_data['event_tokens_512'].notnull()]
    test_length = len(test_data)

    train_data = pd.concat((pretrain_data, finetune_data))
    train_data.reset_index(inplace=True)
    train_data.drop_duplicates(subset='index', keep='first', inplace=True).set_index('index')

    del pretrain_data, finetune_data

    # Fit tokenizer on .json vocab files
    tokenizer = ConceptTokenizer(data_dir=config.data_dir)
    tokenizer.fit_on_vocab()
    config.vocab_size = tokenizer.get_vocab_size()
    config.padding_idx = tokenizer.get_pad_token_id()

    # Get train and test datasets and dataloaders
    train_dataset = FinetuneDataset(
        data=train_data,
        tokenizer=tokenizer,
        max_len=config.max_len,
    )

    test_dataset = FinetuneDataset(
        data=test_data,
        tokenizer=tokenizer,
        max_len=config.max_len,
    )

    train_dataset_with_lengths = DatasetWithTokenLength(
        train_dataset, train_data["token_length"].values
    )
    test_dataset_with_lengths = DatasetWithTokenLength(
        test_dataset, test_data["token_length"].values
    )

    train_loader = DataLoader(
        train_dataset_with_lengths,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset_with_lengths,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    print("Data is ready to go!\n")

    # Set hyperparameters for Bi-LSTM model adn training loop
    input_size = config.embedding_size  # embedding_dim
    hidden_size = config.embedding_size // 2  # output hidden size
    num_layers = 5  # number of LSTM layers
    output_size = 1  # Binary classification, so output size is 1
    dropout_rate = 0.2  # Dropout rate for regularization

    # Set training hyperparameters
    epochs = 6
    learning_rate = 0.001

    # Training Loop
    model = BiLSTMModel(input_size, hidden_size, num_layers, output_size, dropout_rate).to(
        config.device
    )
    class_weights = torch.tensor([6]).to(config.device)  # Determined with experiment
    loss_fcn = nn.BCEWithLogitsLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ExponentialLR(optimizer, gamma=0.75, verbose=True)

    for epoch in range(epochs):
        train_total_loss = 0
        train_accuracy = 0
        test_accuracy = 0

        model.train()
        for batch_no, (sequences, lengths) in tqdm(
                enumerate(train_loader),
                file=sys.stdout,
                total=len(train_loader),
                desc=f"Epoch {epoch + 1}/{epochs}",
                unit=" batch",
        ):
            inputs, labels = model.get_inputs_labels(sequences)
            outputs = model(inputs, lengths)
            loss = loss_fcn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_total_loss += loss.item()

        # Run an evaluation loop if needed
        # model.eval()
        # with torch.no_grad():
        #     for batch_no, (sequences, lengths) in tqdm(
        #         enumerate(train_loader),
        #         file=sys.stdout,
        #         total=len(train_loader),
        #         desc=f"Train Evaluation {epoch + 1}/{epochs}",
        #         unit=" batch",
        #     ):
        #         inputs, labels = model.get_inputs_labels(sequences)
        #         outputs = model(inputs, lengths)
        #         train_accuracy += model.get_balanced_accuracy(outputs, labels)
        #
        #     for batch_no, (sequences, lengths) in tqdm(
        #         enumerate(test_loader),
        #         file=sys.stdout,
        #         total=len(test_loader),
        #         desc=f"Test Evaluation {epoch + 1}/{epochs}",
        #         unit=" batch",
        #     ):
        #         inputs, labels = model.get_inputs_labels(sequences)
        #         outputs = model(inputs, lengths)
        #         test_accuracy += model.get_balanced_accuracy(outputs, labels)

        print(
            f"\nEpoch {epoch + 1}/{epochs}"
            f"  |  Average Train Loss: {train_total_loss / len(train_loader):.5f}"
            f"  |  Train Accuracy: {train_accuracy / len(train_loader):.5f}"
            # f"  |  Test Accuracy: {test_accuracy / len(test_loader):.5f}\n\n"
            , '\n\n'
        )
        scheduler.step()

    # Save the model if needed
    torch.save(model, SAVE_MODEL_DIR)
