"""Fine-tune a pre-trained EHR-Mamba3 model."""

import argparse
import os
from typing import Any, Dict

import numpy as np
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
from skmultilearn.model_selection import iterative_train_test_split
from torch.utils.data import DataLoader

from odyssey.data.dataset import FinetuneDataset, FinetuneDatasetDecoder
from odyssey.data.tokenizer import ConceptTokenizer
from odyssey.models.ehr_mamba3.model import Mamba3Finetune, Mamba3Pretrain
from odyssey.models.model_utils import (
    get_run_id,
    load_config,
    load_finetune_data,
)
from odyssey.utils.utils import seed_everything


def main(
    args: argparse.Namespace,
    pre_model_config: Dict[str, Any],
    fine_model_config: Dict[str, Any],
) -> None:
    """Fine-tune EHR-Mamba3."""
    seed_everything(args.seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")

    fine_tune, fine_test = load_finetune_data(
        args.data_dir,
        args.sequence_file,
        args.id_file,
        args.valid_scheme,
        args.num_finetune_patients,
    )

    # Rename label column for single-task fine-tuning
    if not args.is_multi_model:
        fine_tune = fine_tune.rename(columns={args.label_name: "label"})
        fine_test = fine_test.rename(columns={args.label_name: "label"})

        if args.num_labels == 2:  # binary
            fine_train, fine_val = train_test_split(
                fine_tune,
                test_size=args.val_size,
                random_state=args.seed,
                stratify=fine_tune["label"],
            )
        else:  # multi-label
            fine_train_ids, _, fine_val_ids, _ = iterative_train_test_split(
                X=fine_tune["patient_id"].to_numpy().reshape(-1, 1),
                y=np.array(fine_tune["label"].values.tolist()),
                test_size=args.val_size,
            )
            fine_train = fine_tune[
                fine_tune["patient_id"].isin(fine_train_ids.flatten().tolist())
            ]
            fine_val = fine_tune[
                fine_tune["patient_id"].isin(fine_val_ids.flatten().tolist())
            ]
    else:
        fine_train, fine_val = train_test_split(
            fine_tune, test_size=args.val_size, random_state=args.seed
        )

    tokenizer = ConceptTokenizer(data_dir=args.vocab_dir)
    tokenizer.fit_on_vocab(with_tasks=args.is_multi_model)

    # Datasets
    if args.is_decoder:
        train_dataset = FinetuneDatasetDecoder(
            data=fine_train,
            tokenizer=tokenizer,
            tasks=args.tasks,
            balance_guide=args.balance_guide,
            max_len=args.max_len,
        )
        val_dataset = FinetuneDatasetDecoder(
            data=fine_val,
            tokenizer=tokenizer,
            tasks=args.tasks,
            balance_guide=args.balance_guide,
            max_len=args.max_len,
        )
        test_dataset = FinetuneDatasetDecoder(
            data=fine_test,
            tokenizer=tokenizer,
            tasks=args.tasks,
            balance_guide=None,
            max_len=args.max_len,
        )
    else:
        train_dataset = FinetuneDataset(
            data=fine_train, tokenizer=tokenizer, max_len=args.max_len
        )  # type: ignore[assignment]
        val_dataset = FinetuneDataset(
            data=fine_val, tokenizer=tokenizer, max_len=args.max_len
        )  # type: ignore[assignment]
        test_dataset = FinetuneDataset(
            data=fine_test, tokenizer=tokenizer, max_len=args.max_len
        )  # type: ignore[assignment]

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

    # Load pre-trained model
    pretrained_model = Mamba3Pretrain(
        vocab_size=tokenizer.get_vocab_size(),
        padding_idx=tokenizer.get_pad_token_id(),
        cls_idx=tokenizer.get_class_token_id(),
        **pre_model_config,
    )
    pretrained_model.load_state_dict(
        torch.load(args.pretrained_path, map_location="cpu")["state_dict"]
    )

    model = Mamba3Finetune(
        pretrained_model=pretrained_model,
        num_labels=args.num_labels,
        problem_type=args.problem_type,
        multi_head=args.is_multi_model,
        num_tasks=len(args.tasks) if args.is_multi_model else 1,
        **fine_model_config,
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
        EarlyStopping(
            monitor="val_loss", patience=args.patience, verbose=True, mode="min"
        ),
    ]

    trainer = pl.Trainer(
        accelerator="gpu",
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

    ckpt = "last" if args.test_last else "best"
    trainer.test(dataloaders=test_loader, ckpt_path=ckpt)

    if args.test_output_dir:
        torch.save(
            model.test_outputs, f"{args.test_output_dir}/test_outputs_{run_id}.pt"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune EHR-Mamba3")

    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--pretrained_path", type=str, required=True)
    parser.add_argument("--label_name", type=str)
    parser.add_argument("--workspace_name", type=str, default=None)
    parser.add_argument("--config_dir", type=str, required=True)
    parser.add_argument("--is_multi_model", type=bool, default=False)
    parser.add_argument("--is_decoder", type=bool, default=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--sequence_file", type=str, required=True)
    parser.add_argument("--id_file", type=str, required=True)
    parser.add_argument("--vocab_dir", type=str, required=True)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--valid_scheme", type=str, default="few_shot")
    parser.add_argument("--num_finetune_patients", type=str, required=True)
    parser.add_argument("--problem_type", type=str, required=True)
    parser.add_argument("--num_labels", type=int, required=True)
    parser.add_argument("--tasks", type=str, default=None)
    parser.add_argument("--balance_guide", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--test_output_dir", type=str, default=None)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--test_last", default=False, action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.exp_name)
    if args.test_output_dir:
        args.test_output_dir = os.path.join(args.checkpoint_dir, args.test_output_dir)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    if args.test_output_dir:
        os.makedirs(args.test_output_dir, exist_ok=True)

    config = load_config(args.config_dir, "ehr_mamba3")
    finetune_config = config["finetune"]
    for key, value in finetune_config.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)

    pre_model_config = config["model"]
    args.max_len = pre_model_config["max_seq_length"]

    args.tasks = args.tasks.strip().split(" ") if args.tasks else []
    args.balance_guide = (
        {
            task: float(ratio)
            for task, ratio in (
                pair.strip().split("=") for pair in args.balance_guide.split(",")
            )
        }
        if args.balance_guide
        else None
    )

    fine_model_config = config.get("model_finetune", {})
    main(args, pre_model_config, fine_model_config)
