"""Finetune the pre-trained model."""

import argparse
import os
import sys
from typing import Any, Dict

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

from odyssey.data.dataset import FinetuneDataset
from odyssey.data.tokenizer import ConceptTokenizer
from odyssey.models.cehr_big_bird.model import BigBirdFinetune, BigBirdPretrain
from odyssey.models.cehr_bert.model import BertFinetune, BertPretrain
from odyssey.models.utils import (
    get_latest_checkpoint,
    get_run_id,
    load_config,
    load_finetune_data,
    seed_everything,
)


def main(
    args: Dict[str, Any],
    pre_model_config: Dict[str, Any],
    fine_model_config: Dict[str, Any],
) -> None:
    """Train the model."""
    # Setup environment
    seed_everything(args.seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")

    # Load data
    fine_tune, fine_test = load_finetune_data(
        args.data_dir,
        args.sequence_file,
        args.id_file,
        args.valid_scheme,
        args.num_finetune_patients,
    )

    fine_tune.rename(columns={args.label_name: "label"}, inplace=True)
    fine_test.rename(columns={args.label_name: "label"}, inplace=True)

    # Split data
    fine_train, fine_val = train_test_split(
        fine_tune,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=fine_tune["label"],
    )

    # Train Tokenizer
    tokenizer = ConceptTokenizer(data_dir=args.vocab_dir)
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
        persistent_workers=args.persistent_workers,
        shuffle=True,
        pin_memory=args.pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
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
        EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            verbose=True,
            mode="min",
        ),
    ]

    # Create model
    if args.model_type == "cehr_bert":
        pretrained_model = BertPretrain(
            args=args,
            vocab_size=tokenizer.get_vocab_size(),
            padding_idx=tokenizer.get_pad_token_id(),
            **pre_model_config,
        )
        pretrained_model.load_state_dict(torch.load(args.pretrained_path)["state_dict"])

        model = BertFinetune(
            args=args,
            pretrained_model=pretrained_model,
            **fine_model_config,
        )

    elif args.model_type == "cehr_bigbird":
        pretrained_model = BigBirdPretrain(
            vocab_size=tokenizer.get_vocab_size(),
            padding_idx=tokenizer.get_pad_token_id(),
            **pre_model_config,
        )
        pretrained_model.load_state_dict(torch.load(args.pretrained_path)["state_dict"])

        model = BigBirdFinetune(
            pretrained_model=pretrained_model,
            **fine_model_config,
        )

    latest_checkpoint = get_latest_checkpoint(args.checkpoint_dir)

    run_id = get_run_id(args.checkpoint_dir)

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

    # Test the model
    if args.test_last:
        trainer.test(
            dataloaders=test_loader,
            ckpt_path="last",
        )
    else:
        trainer.test(
            dataloaders=test_loader,
            ckpt_path="best",
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
        "--pretrained-path",
        type=str,
        required=True,
        help="Pretrained model",
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
    parser.add_argument(
        "--valid_scheme",
        type=str,
        default="few_shot",
        help="Define the type of validation, few_shot or kfold",
    )
    parser.add_argument(
        "--num_finetune_patients",
        type=str,
        default="20000_patients",
        help="Define the number of patients to be fine_tuned on",
    )

    # checkpointing and logging arguments
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
        help="Checkpoint to resume finetuning from",
    )

    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=10,
        help="Number of steps to log the training",
    )

    # other arguments
    parser.add_argument(
        "--test-last",
        action="store_true",
        help="Test the last checkpoint",
    )
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

    finetune_config = config["finetune"]
    for key, value in finetune_config.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)

    pre_model_config = config["model"]
    args.max_len = pre_model_config["max_seq_length"]

    fine_model_config = config["model_finetune"]

    main(args, pre_model_config, fine_model_config)
