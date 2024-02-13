import argparse
import os
from os.path import join

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

from models.cehr_bert.data import FinetuneDataset
from models.cehr_bert.model import BertFinetune, BertPretrain
from models.cehr_bert.tokenizer import ConceptTokenizer


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(args.seed)

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")

    fine_tune = pd.read_parquet(join(args.data_dir, "fine_tune.parquet"))
    fine_test = pd.read_parquet(join(args.data_dir, "fine_test.parquet"))
    # fine_data = pd.read_parquet(join(args.data_dir, "fine_tune.parquet"))
    # fine_train, fine_valtest = train_test_split(
    #     fine_data,
    #     train_size=args.train_size,
    #     random_state=args.seed,
    #     stratify=fine_data["label"],
    # )
    fine_train, fine_val = train_test_split(
        fine_tune,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=fine_tune["label"],
    )

    tokenizer = ConceptTokenizer()
    tokenizer.fit_on_vocab()

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
        shuffle=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

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
        project="finetune",
        save_dir=args.log_dir,
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.gpus,
        strategy=DDPStrategy(find_unused_parameters=True) if args.gpus > 1 else "auto",
        precision=16,
        check_val_every_n_epoch=1,
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        logger=wandb_logger,
        log_every_n_steps=args.log_every_n_steps,
    )

    pretrained_model = BertPretrain(
        vocab_size=tokenizer.get_vocab_size(),
        padding_idx=tokenizer.get_pad_token_id(),
    )
    pretrained_model.load_state_dict(torch.load(args.pretrained_path)["state_dict"])

    model = BertFinetune(
        pretrained_model=pretrained_model,
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    trainer.test(
        model=model,
        dataloaders=test_loader,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Flag to resume training from a checkpoint",
    )
    parser.add_argument(
        "--data_dir", type=str, default="data_files", help="Path to the data directory"
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
        "--max_len", type=int, default=512, help="Maximum length of the sequence"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for training"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/finetuning",
        help="Path to the training checkpoint",
    )
    parser.add_argument(
        "--log_dir", type=str, default="logs", help="Path to the log directory"
    )
    parser.add_argument(
        "--gpus", type=int, default=1, help="Number of gpus for training"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=10, help="Number of epochs for training"
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
