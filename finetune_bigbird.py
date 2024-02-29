"""
File: finetune_bigbird.py.

Finetune an already pretrained bigbird model on MIMIC-IV FHIR data.
The finetuning objective is binary classification on patient mortality or
hospital readmission labels.
"""

import argparse
import glob
import os
from os.path import join
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.strategies.ddp import DDPStrategy
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from models.big_bird_cehr.data import FinetuneDataset
from models.big_bird_cehr.model import BigBirdFinetune, BigBirdPretrain
from models.big_bird_cehr.tokenizer import HuggingFaceConceptTokenizer


def seed_everything(seed: int) -> None:
    """Seed all components of the model."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed)


def get_latest_checkpoint(checkpoint_dir: str) -> Any:
    """Return the most recent checkpointed file to resume training from."""
    list_of_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    return max(list_of_files, key=os.path.getctime) if list_of_files else None


def main(args: Dict[str, Any]) -> None:
    """Train the model."""
    # Setup environment
    seed_everything(args.seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")

    # Load data
    fine_tune = pd.read_parquet(join(args.data_dir, "fine_tune.parquet"))
    fine_test = pd.read_parquet(join(args.data_dir, "fine_test.parquet"))

    # Split data
    fine_train, fine_val = train_test_split(
        fine_tune,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=fine_tune["label"],
    )

    # Train Tokenizer
    tokenizer = HuggingFaceConceptTokenizer(data_dir=args.data_dir)
    tokenizer.fit_on_vocab()

    # Load datasets
    train_dataset = FinetuneDataset(
        data=fine_train,
        tokenizer=tokenizer,
        max_len=args.max_len,
    )

    val_dataset = FinetuneDataset(
        data=fine_val,
        tokenizer=tokenizer,
        max_len=args.max_len,
    )

    test_dataset = FinetuneDataset(
        data=fine_test,
        tokenizer=tokenizer,
        max_len=args.max_len,
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

    test_loader = DataLoader(
        test_dataset,
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
        EarlyStopping(monitor="val_loss", patience=5, verbose=True, mode="min"),
    ]

    wandb_logger = WandbLogger(
        project="bigbird_finetune",
        save_dir=args.log_dir,
    )

    # Load latest checkpoint to continue training
    latest_checkpoint = get_latest_checkpoint(args.checkpoint_path)

    # Setup PyTorchLightning trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.gpus,
        strategy=DDPStrategy(find_unused_parameters=True) if args.gpus > 1 else "auto",
        precision="16-mixed",
        check_val_every_n_epoch=1,
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        deterministic=False,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=wandb_logger,
        resume_from_checkpoint=latest_checkpoint if args.resume else None,
        log_every_n_steps=args.log_every_n_steps,
        accumulate_grad_batches=args.acc,
        gradient_clip_val=1.0,
    )

    # Create pretrain BigBird model and load the pretrained state_dict
    pretrained_model = BigBirdPretrain(
        args=args,
        dataset_len=len(train_dataset),
        vocab_size=tokenizer.get_vocab_size(),
        padding_idx=tokenizer.get_pad_token_id(),
    )
    pretrained_model.load_state_dict(torch.load(args.pretrained_path)["state_dict"])

    # Create fine tune BigBird model
    model = BigBirdFinetune(
        args,
        dataset_len=len(train_dataset),
        pretrained_model=pretrained_model,
    )

    # Train the model
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # Test the model
    trainer.test(
        model=model,
        dataloaders=test_loader,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Flag to resume training from a checkpoint",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/h/afallah/odyssey/odyssey/data/slurm_data/one_month",
        help="Path to the data directory",
    )
    parser.add_argument(
        "--train_size",
        type=float,
        default=0.3,
        help="Train set size for splitting the data",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.6,
        help="Test set size for splitting the data",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=2048,
        help="Maximum length of the sequence",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for training",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/finetuning",
        help="Path to the training checkpoint",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Path to the log directory",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of gpus for training",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=10,
        help="Number of epochs for training",
    )
    parser.add_argument(
        "--acc",
        type=int,
        default=1,
        help="Gradient accumulation",
    )
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=10,
        help="Number of steps to log the training",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=None,
        required=True,
        help="Checkpoint to the pretrained model",
    )

    args = parser.parse_args()
    main(args)
