"""
File: pretrain_bigbird.py.

Pretrain a bigbird model on MIMIC-IV FHIR data using Masked Language Modeling objective.
"""

import os
import argparse
import glob
import pickle
from os.path import join

from typing import Dict, Any

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy

from sklearn.model_selection import train_test_split

from models.big_bird_cehr.data import PretrainDataset
from models.big_bird_cehr.model import BigBirdPretrain
from models.big_bird_cehr.tokenizer import HuggingFaceConceptTokenizer


def seed_everything(seed: int) -> None:
    """ Seed all components of the model. """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed)


def get_latest_checkpoint(checkpoint_dir: str) -> Any:
    """ Return the most recent checkpointed file to resume training from. """
    list_of_files = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))
    return max(list_of_files, key=os.path.getctime) if list_of_files else None


def main(args: Dict[str, Any]) -> None:
    """ Train the model. """

    # Setup environment
    seed_everything(args.seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")

    # Load data
    data = pd.read_parquet(join(args.data_dir, "patient_sequences_2048_labeled.parquet"))
    patient_ids = pickle.load(open(join(args.data_dir, 'dataset_2048_mortality_1month.pkl'), 'rb'))
    pre_data = data.loc[data['patient_id'].isin(patient_ids['pretrain'])]

    # Split data
    pre_train, pre_val = train_test_split(
        pre_data,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=pre_data["label_mortality_1month"],
    )

    # Train Tokenizer
    tokenizer = HuggingFaceConceptTokenizer(data_dir=args.vocab_dir)
    tokenizer.fit_on_vocab()

    # Load datasets
    train_dataset = PretrainDataset(
        data=pre_train,
        tokenizer=tokenizer,
        max_len=args.max_len,
        mask_prob=args.mask_prob,
    )

    val_dataset = PretrainDataset(
        data=pre_val,
        tokenizer=tokenizer,
        max_len=args.max_len,
        mask_prob=args.mask_prob,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=True,
        shuffle=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=True,
        pin_memory=True,
    )

    # Setup model dependencies
    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            filename="best",
            save_top_k=1,
            save_last=True,
            verbose=True,
            dirpath=args.checkpoint_dir,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    wandb_logger = WandbLogger(
        project="bigbird_pretrain_a100",
        save_dir=args.log_dir,
    )

    # Load latest checkpoint to continue training
    # latest_checkpoint = get_latest_checkpoint(args.checkpoint_path)

    # Setup PyTorchLightning trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.gpus,
        strategy=DDPStrategy(find_unused_parameters=True) if args.gpus > 1 else "auto",
        precision='16-mixed',
        check_val_every_n_epoch=1,
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        deterministic=False,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=wandb_logger,
        # resume_from_checkpoint=latest_checkpoint if args.resume else None,
        log_every_n_steps=args.log_every_n_steps,
        accumulate_grad_batches=args.acc,
        gradient_clip_val=1.0
    )

    # Create BigBird model
    model = BigBirdPretrain(
        args=args,
        dataset_len=len(train_dataset),
        vocab_size=tokenizer.get_vocab_size(),
        padding_idx=tokenizer.get_pad_token_id(),
    )

    # Train the model
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Flag to resume training from a checkpoint",
    )
    parser.add_argument(
        "--data_dir", type=str,
        default="/h/afallah/odyssey/odyssey/data/bigbird_data",
        help="Path to the data directory"
    )
    parser.add_argument(
        "--vocab_dir", type=str,
        default="/h/afallah/odyssey/odyssey/data/vocab",
        help="Path to the vocabulary directory of json files"
    )
    parser.add_argument(
        "--finetune_size",
        type=float,
        default=0.1,
        help="Finetune dataset size for splitting the data",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.1,
        help="Validation set size for splitting the data",
    )
    parser.add_argument(
        "--max_len", type=int, default=2048, help="Maximum length of the sequence"
    )
    parser.add_argument(
        "--mask_prob", type=float, default=0.15, help="Probability of masking the token"
    )
    parser.add_argument(
        "--batch_size", type=int, default=10, help="Batch size for training"
    )
    parser.add_argument(
        "--num_workers", type=int, default=3, help="Number of workers for training"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/bigbird_pretraining_a100",
        help="Path to the training checkpoint",
    )
    parser.add_argument(
        "--log_dir", type=str, default="logs", help="Path to the log directory"
    )
    parser.add_argument(
        "--gpus", type=int, default=4, help="Number of gpus for training"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=10, help="Number of epochs for training"
    )
    parser.add_argument(
        "--acc", type=int, default=1, help="Gradient accumulation"
    )
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=10,
        help="Number of steps to log the training",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Checkpoint to resume training from",
    )
    parser.add_argument(
        "--output_model_path",
        type=str,
        default="/h/afallah/odyssey/odyssey/bigbird_pretrained_2048.pt",
        help="Directory to save the model",
    )

    args = parser.parse_args()
    main(args)
