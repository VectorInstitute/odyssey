"""
File: Bi-LSTM.ipynb
Code to train and evaluate a bi-directional LSTM model on MIMIC-IV FHIR dataset.
"""

import os
import scipy, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_curve, auc, precision_recall_curve, roc_auc_score, average_precision_score
from scipy.sparse import csr_matrix, hstack, vstack, save_npz, load_npz

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import relu, leaky_relu, sigmoid
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR

from models.cehr_bert.data import PretrainDataset, FinetuneDataset
from models.cehr_bert.model import BertPretrain
from models.cehr_bert.tokenizer import ConceptTokenizer
from models.cehr_bert.embeddings import Embeddings

from tqdm import tqdm

ROOT = 'E:\Vector Institute\odyssey';
os.chdir(ROOT)
DATA_ROOT = f'{ROOT}/data'
DATA_PATH = f'{DATA_ROOT}/patient_sequences.parquet'
SAMPLE_DATA_PATH = f'{DATA_ROOT}/CEHR-BERT_sample_patient_sequence.parquet'
FREQ_DF_PATH = f'{DATA_ROOT}/patient_feature_freq.csv'
FREQ_MATRIX_PATH = f'{DATA_ROOT}/patient_freq_matrix.npz'


# save parameters and configurations
class config:
    seed = 23
    data_dir = DATA_ROOT
    test_size = 0.2
    max_len = 500
    batch_size = 16
    num_workers = 2
    vocab_size = None
    embedding_size = 128
    time_embeddings_size = 16
    max_seq_length = 512
    device = torch.device('cuda')


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # pl.seed_everything(seed)


seed_all(config.seed)
torch.cuda.get_device_name(torch.cuda.current_device())

# Load data
data = pd.read_parquet(DATA_PATH)
data.rename(columns={'event_tokens': 'event_tokens_untruncated', 'event_tokens_updated': 'event_tokens'}, inplace=True)
data['label'] = ((data['death_after_end'] > 0) & (data['death_after_end'] < 365)).astype(int)

# Fit tokenizer on .json vocab files
tokenizer = ConceptTokenizer(data_dir=config.data_dir)
tokenizer.fit_on_vocab()
config.vocab_size = tokenizer.get_vocab_size()

# Get training and test datasets

train_data, test_data = train_test_split(
    data,
    test_size=config.test_size,
    random_state=config.seed,
    stratify=data['label']
)

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

train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    num_workers=config.num_workers,
    shuffle=True,
    pin_memory=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=config.batch_size,
    num_workers=config.num_workers,
    shuffle=True,
    pin_memory=True,
)


# Define model architecture

class BiLSTMModel(nn.Module):

    def __init__(self, embedding_dim, hidden_size, num_layers, output_size, dropout_rate=0.5):
        super(BiLSTMModel, self).__init__()

        self.embeddings = Embeddings(
            vocab_size=config.vocab_size,
            embedding_size=config.embedding_size,
            time_embedding_size=config.time_embeddings_size,
            max_len=config.max_seq_length)

        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=dropout_rate)

        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear1 = nn.Linear(hidden_size * 2, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        x = self.embeddings(*inputs)
        lstm_out, _ = self.lstm(x)
        output = lstm_out[:, -1, :]
        output = self.batch_norm(output)
        output = self.dropout(output)
        output = relu(self.linear1(output))
        output = self.linear2(output)
        return output

    @staticmethod
    def get_inputs_labels(batch):
        labels = batch['labels'].view(-1, 1).to(config.device)
        inputs = batch['concept_ids'].to(config.device), \
            batch['time_stamps'].to(config.device), \
            batch['ages'].to(config.device), \
            batch['visit_orders'].to(config.device), \
            batch['visit_segments'].to(config.device)

        return inputs, labels.float()


# Set hyperparameters for Bi-LSTM model adn training loop
input_size = config.embedding_size  # embedding_dim
hidden_size = config.embedding_size  # output hidden size
num_layers = 5  # number of LSTM layers
output_size = 1  # Binary classification, so output size is 1
dropout_rate = 0.5  # Dropout rate for regularization

epochs = 10
learning_rate = 0.003

# Training Loop
model = BiLSTMModel(input_size, hidden_size, num_layers, output_size, dropout_rate).to(config.device)
loss_fcn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ExponentialLR(optimizer, gamma=0.8, verbose=True)

for epoch in range(epochs):
    train_total_loss = 0
    train_accuracy = 0
    test_accuracy = 0

    mode.train()
    for batch_no, batch in tqdm(enumerate(train_loader),
                                total=len(train_loader), desc=f'Epoch {epoch + 1}/{epochs}', unit=' batch'):
        inputs, labels = model.get_inputs_labels(batch)
        outputs = model(inputs)
        loss = loss_fcn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_total_loss += loss.item()
        # print(f'Batch Loss: {loss.item():.4f}', end='\r')

    model.eval()
    with torch.no_grad():
        for batch_no, batch in tqdm(enumerate(train_loader),
                                    total=len(train_loader), desc=f'Train Evaluation {epoch + 1}/{epochs}',
                                    unit=' batch'):
            inputs, labels = model.get_inputs_labels(batch)
            outputs = model(inputs)
            predictions = torch.round(sigmoid(outputs))
            train_accuracy += balanced_accuracy_score(labels, predictions)

        for batch_no, batch in tqdm(enumerate(test_loader),
                                    total=len(test_loader), desc=f'Test Evaluation {epoch + 1}/{epochs}',
                                    unit=' batch'):
            inputs, labels = model.get_inputs_labels(batch)
            outputs = model(inputs)
            predictions = torch.round(sigmoid(outputs))
            test_accuracy += balanced_accuracy_score(labels, predictions)

    print(
        f'Average Train Loss: {train_total_loss / len(train_loader):.5f}  |  Last Batch Train Loss: {loss.item()}  |  '
        f'Train Accuracy: {train_accuracy / len(train_loader)}  |  Test Accuracy: {test_accuracy / len(test_loader)}')
    scheduler.step()

torch.save(model, 'LSTM_V1.pt')
