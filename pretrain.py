"""Train the model."""

import argparse
import os
import sys
from typing import Any, Dict

import pytorch_lightning as pl
import torch
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from lib.data import PretrainDataset
from lib.tokenizer import ConceptTokenizer
from lib.utils import (
    get_latest_checkpoint,
    get_run_id,
    load_config,
    load_pretrain_data,
    seed_everything,
)
from models.big_bird_cehr.model import BigBirdPretrain
from models.cehr_bert.model import BertPretrain


def main(args: Dict[str, Any], model_config: Dict[str, Any]) -> None:
    """Train the model."""
    seed_everything(args.seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")

    pre_data = load_pretrain_data(
        args.data_dir,
        args.sequence_file,
        args.id_file,
    )
    pre_data.rename(columns={args.label_name: "label"}, inplace=True)

    # Split data
    pre_train, pre_val = train_test_split(
        pre_data,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=pre_data["label"],
    )

    # Train Tokenizer
    tokenizer = ConceptTokenizer(data_dir=args.vocab_dir)
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
        persistent_workers=args.persistent_workers,
        shuffle=True,
        pin_memory=args.pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=args.persistent_workers,
        pin_memory=args.pin_memory,
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

    # Create model
    if args.model_type == "cehr_bert":
        model = BertPretrain(
            args=args,
            vocab_size=tokenizer.get_vocab_size(),
            padding_idx=tokenizer.get_pad_token_id(),
            **model_config,
        )
    elif args.model_type == "cehr_bigbird":
        model = BigBirdPretrain(
            vocab_size=tokenizer.get_vocab_size(),
            padding_idx=tokenizer.get_pad_token_id(),
            **model_config,
        )

    latest_checkpoint = get_latest_checkpoint(args.checkpoint_dir)

    run_id = get_run_id(args.checkpoint_dir, retrieve=(latest_checkpoint is not None))

    wandb_logger = WandbLogger(
        project=args.exp_name,
        save_dir=args.log_dir,
        entity=args.workspace_name,
        id=run_id,
        resume="allow",
    )

    # Setup PyTorchLightning trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        num_nodes=args.nodes,
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
        log_every_n_steps=args.log_every_n_steps,
        accumulate_grad_batches=args.acc,
        gradient_clip_val=1.0,
    )

    # Train the model
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=latest_checkpoint,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # project configuration
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        help="Model type: 'cehr_bert' or 'cehr_bigbird'",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        required=True,
        help="Path to model config file",
    )
    parser.add_argument(
        "--label-name",
        type=str,
        required=True,
        help="Name of the label column",
    )
    parser.add_argument(
        "--workspace-name",
        type=str,
        default=None,
        help="Name of the Wandb workspace",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="models/configs",
        help="Path to model config file",
    )

    # data-related arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data_files",
        help="Path to the data directory",
    )
    parser.add_argument(
        "--sequence-file",
        type=str,
        default="patient_sequences_2048_labeled.parquet",
        help="Path to the patient sequence file",
    )
    parser.add_argument(
        "--id-file",
        type=str,
        default="dataset_2048_mortality_1month.pkl",
        help="Path to the patient id file",
    )
    parser.add_argument(
        "--vocab-dir",
        type=str,
        default="data_files/vocab",
        help="Path to the vocabulary directory of json files",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Validation set size for splitting the data",
    )

    # checkpointing and loggig arguments
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Path to the checkpoint directory",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Path to the log directory",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Checkpoint to resume training from",
    )
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=10,
        help="Number of steps to log the training",
    )

    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    if args.model_type not in ["cehr_bert", "cehr_bigbird"]:
        print("Invalid model type. Choose 'cehr_bert' or 'cehr_bigbird'.")
        sys.exit(1)

    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.exp_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    config = load_config(args.config_dir, args.model_type)

    train_config = config["train"]
    for key, value in train_config.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)

    model_config = config["model"]
    args.max_len = model_config["max_seq_length"]

    main(args, model_config)
