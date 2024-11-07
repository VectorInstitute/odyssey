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

from odyssey.data.dataset import PretrainDataset, PretrainDatasetDecoder
from odyssey.data.tokenizer import ConceptTokenizer
from odyssey.models.cehr_bert.model import BertPretrain
from odyssey.models.cehr_big_bird.model import BigBirdPretrain
from odyssey.models.ehr_mamba.model import MambaPretrain
from odyssey.models.ehr_mamba2.model import Mamba2Pretrain
from odyssey.models.model_utils import (
    get_run_id,
    load_config,
    load_pretrain_data,
)
from odyssey.utils.utils import seed_everything


def main(args: argparse.Namespace, model_config: Dict[str, Any]) -> None:
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

    # Split data
    pre_train, pre_val = train_test_split(
        pre_data,
        test_size=args.val_size,
        random_state=args.seed,
    )

    # Initialize Tokenizer
    if args.tokenizer_type == "fhir":
        tokenizer = ConceptTokenizer(
            data_dir=args.vocab_dir,
            start_token="[VS]",
            end_token="[VE]",
            time_tokens=[f"[W_{i}]" for i in range(0, 4)]
            + [f"[M_{i}]" for i in range(0, 13)]
            + ["[LT]"],
        )
    else:  # meds
        tokenizer = ConceptTokenizer(
            data_dir=args.vocab_dir,
            start_token="[BOS]",
            end_token="[EOS]",
            time_tokens=None,  # New tokenizer comes with predefined time tokens
            padding_side=args.padding_side,
        )
    tokenizer.fit_on_vocab()

    # Load datasets
    if args.is_decoder:  # e.g. Mamba and Mamba2
        train_dataset = PretrainDatasetDecoder(
            data=pre_train,
            tokenizer=tokenizer,
            max_len=args.max_len,
            padding_side=args.padding_side,
            return_attention_mask=args.return_attention_mask,
        )
        val_dataset = PretrainDatasetDecoder(
            data=pre_val,
            tokenizer=tokenizer,
            max_len=args.max_len,
            padding_side=args.padding_side,
            return_attention_mask=args.return_attention_mask,
        )

    else:
        train_dataset = PretrainDataset(
            data=pre_train,
            tokenizer=tokenizer,
            max_len=args.max_len,
            mask_prob=args.mask_prob,
            padding_side=args.padding_side,
        )
        val_dataset = PretrainDataset(
            data=pre_val,
            tokenizer=tokenizer,
            max_len=args.max_len,
            mask_prob=args.mask_prob,
            padding_side=args.padding_side,
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
    elif args.model_type == "ehr_mamba":
        model = MambaPretrain(
            vocab_size=tokenizer.get_vocab_size(),
            padding_idx=tokenizer.get_pad_token_id(),
            cls_idx=tokenizer.get_class_token_id(),
            **model_config,
        )
    elif args.model_type == "ehr_mamba2":
        model = Mamba2Pretrain(
            vocab_size=tokenizer.get_vocab_size(),
            padding_idx=tokenizer.get_pad_token_id(),
            cls_idx=tokenizer.get_class_token_id(),
            eos_idx=tokenizer.get_eos_token_id(),
            **model_config,
        )

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
        num_nodes=args.nodes,
        devices=args.gpus,
        strategy=DDPStrategy(find_unused_parameters=True)
        if args.gpus > 1
        else "auto",  # DeepSpeedStrategy(stage=2, offload_optimizer=False)
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
        ckpt_path=args.resume_checkpoint,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # project configuration
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="Model type: 'cehr_bert' or 'cehr_bigbird' or 'ehr_mamba' or 'ehr_mamba2'",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        required=True,
        help="Path to model config file",
    )
    parser.add_argument(
        "--workspace_name",
        type=str,
        default=None,
        help="Name of the Wandb workspace",
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        required=True,
        help="Path to model config file",
    )
    parser.add_argument(
        "--is_decoder",
        type=bool,
        default=False,
        help="Is the model a decoder (e.g. Mamba) or not",
    )

    # data-related arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the data directory",
    )
    parser.add_argument(
        "--sequence_file",
        type=str,
        required=True,
        help="Path to the patient sequence file",
    )
    parser.add_argument(
        "--id_file",
        type=str,
        required=True,
        help="Path to the patient id file",
    )
    parser.add_argument(
        "--vocab_dir",
        type=str,
        required=True,
        help="Path to the vocabulary directory of json files",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.1,
        help="Validation set size for splitting the data",
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        required=True,
        default="v1",
        help="Tokenizer version",
    )
    parser.add_argument(
        "--padding_side",
        type=str,
        default="right",
        help="Padding side for the tokenizer",
    )
    parser.add_argument(
        "--return_attention_mask",
        type=bool,
        default=True,
        help="Whether to return the attention mask or not",
    )

    # checkpointing and loggig arguments
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to the checkpoint directory",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Path to the log directory",
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Checkpoint to resume pretraining from",
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

    if args.model_type not in ["cehr_bert", "cehr_bigbird", "ehr_mamba", "ehr_mamba2"]:
        print(
            "Invalid model type. Choose 'cehr_bert' or 'cehr_bigbird' or 'ehr_mamba' or 'ehr_mamba2'."
        )
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
