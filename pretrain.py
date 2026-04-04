"""Pre-train EHR-Mamba3."""

import argparse
import os
from typing import Any

import pytorch_lightning as pl
import torch
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from odyssey.data.dataset import PretrainDatasetDecoder
from odyssey.data.tokenizer import DEFAULT_TIME_TOKENS, ConceptTokenizer
from odyssey.models.ehr_mamba3.model import Mamba3Pretrain
from odyssey.models.model_utils import get_run_id, load_config, load_pretrain_data
from odyssey.utils.utils import seed_everything


def main(args: argparse.Namespace, model_config: dict[str, Any]) -> None:
    """Pre-train EHR-Mamba3."""
    seed_everything(args.seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")

    pre_data = load_pretrain_data(args.data_dir, args.sequence_file, args.id_file)

    pre_train, pre_val = train_test_split(
        pre_data, test_size=args.val_size, random_state=args.seed
    )

    tokenizer = ConceptTokenizer(
        data_dir=args.vocab_dir,
        start_token="[BOS]",
        end_token="[EOS]",
        time_tokens=DEFAULT_TIME_TOKENS,
        padding_side=args.padding_side,
    )
    tokenizer.fit_on_vocab()

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

    model = Mamba3Pretrain(
        vocab_size=tokenizer.get_vocab_size(),
        padding_idx=tokenizer.get_pad_token_id(),
        cls_idx=tokenizer.get_class_token_id(),
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

    trainer = pl.Trainer(
        accelerator="gpu",
        num_nodes=args.nodes,
        devices=args.gpus,
        strategy=DDPStrategy(find_unused_parameters=True) if args.gpus > 1 else "auto",
        precision="bf16-mixed",
        check_val_every_n_epoch=1,
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        deterministic=False,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=[wandb_logger],  # type: ignore[list-item]
        log_every_n_steps=args.log_every_n_steps,
        accumulate_grad_batches=args.acc,
        gradient_clip_val=1.0,
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume_checkpoint,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train EHR-Mamba3")

    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--workspace_name", type=str, default=None)
    parser.add_argument("--config_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--sequence_file", type=str, required=True)
    parser.add_argument("--id_file", type=str, required=True)
    parser.add_argument("--vocab_dir", type=str, required=True)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--padding_side", type=str, default="right")
    parser.add_argument("--return_attention_mask", type=bool, default=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.exp_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    config = load_config(args.config_dir, "ehr_mamba3")

    train_config = config["train"]
    for key, value in train_config.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)

    model_config = config["model"]
    args.max_len = model_config["max_seq_length"]

    main(args, model_config)
