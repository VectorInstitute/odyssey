import argparse
import os
from os.path import join

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from models.cehr_bert.data import PretrainDataset
from models.cehr_bert.model import BertPretrain
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

    # if not args.resume:
    #     data = pd.read_parquet(join(args.data_dir, "patient_sequences.parquet"))

    #     data["label"] = (
    #         (data["death_after_end"] >= 0) & (data["death_after_end"] < 365)
    #     ).astype(int)
    #     neg_pos_data = data[(data["deceased"] == 0) | (data["label"] == 1)]

    #     pre_data = data[~data.index.isin(neg_pos_data.index)]

    #     pre_df, fine_df = train_test_split(
    #         neg_pos_data,
    #         test_size=args.finetune_size,
    #         random_state=args.seed,
    #         stratify=neg_pos_data["label"],
    #     )

    #     pre_data = pd.concat([pre_data, pre_df])

    #     fine_df.to_parquet(join(args.data_dir, "fine_tune.parquet"))
    #     pre_data.to_parquet(join(args.data_dir, "pretrain.parquet"))
    # else:
    pre_data = pd.read_parquet(join(args.data_dir, "pretrain.parquet"))

    pre_train, pre_val = train_test_split(
        pre_data,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=pre_data["label"],
    )

    tokenizer = ConceptTokenizer(data_dir=args.data_dir)
    tokenizer.fit_on_vocab()

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
        shuffle=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
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
    ]
    wandb_logger = WandbLogger(
        project="pretrain",
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
        resume_from_checkpoint=args.checkpoint_path if args.resume else None,
        log_every_n_steps=args.log_every_n_steps,
    )

    model = BertPretrain(
        vocab_size=tokenizer.get_vocab_size(),
        padding_idx=tokenizer.get_pad_token_id(),
    )

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
        help="Flag to resume training from a checkpoint",
    )
    parser.add_argument(
        "--data_dir", type=str, default="data_files", help="Path to the data directory"
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
        "--max_len", type=int, default=512, help="Maximum length of the sequence"
    )
    parser.add_argument(
        "--mask_prob", type=float, default=0.15, help="Probability of masking the token"
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
        default="checkpoints/pretraining",
        help="Path to the training checkpoint",
    )
    parser.add_argument(
        "--log_dir", type=str, default="logs", help="Path to the log directory"
    )
    parser.add_argument(
        "--gpus", type=int, default=2, help="Number of gpus for training"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=50, help="Number of epochs for training"
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

    args = parser.parse_args()
    main(args)
