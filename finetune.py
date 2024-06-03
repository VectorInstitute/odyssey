"""Finetune the pre-trained model."""

import argparse
import os
import sys
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

from odyssey.data.dataset import (
    FinetuneDataset,
    FinetuneDatasetDecoder,
    FinetuneMultiDataset,
)
from odyssey.data.tokenizer import ConceptTokenizer
from odyssey.models.cehr_bert.model import BertFinetune, BertPretrain
from odyssey.models.cehr_big_bird.model import BigBirdFinetune, BigBirdPretrain
from odyssey.models.ehr_mamba.model import MambaFinetune, MambaPretrain
from odyssey.models.model_utils import (
    get_run_id,
    load_config,
    load_finetune_data,
)
from odyssey.utils.utils import seed_everything


def main(  # noqa: PLR0912, PLR0915
    args: argparse.Namespace,
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

    # Split data based on model type
    if not args.is_multi_model:
        fine_tune.rename(columns={args.label_name: "label"}, inplace=True)
        fine_test.rename(columns={args.label_name: "label"}, inplace=True)

        # Split data in a stratified way based on problem type
        if args.num_labels == 2:  # Binary classification
            fine_train, fine_val = train_test_split(
                fine_tune,
                test_size=args.val_size,
                random_state=args.seed,
                stratify=fine_tune["label"],
            )

        else:  # Multi label classfication
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
            fine_tune,
            test_size=args.val_size,
            random_state=args.seed,
        )

    # Train Tokenizer
    tokenizer = ConceptTokenizer(data_dir=args.vocab_dir)
    tokenizer.fit_on_vocab(with_tasks=args.is_multi_model)

    # Load datasets based on model type
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

    elif args.is_multi_model:
        train_dataset = FinetuneMultiDataset(
            data=fine_train,
            tokenizer=tokenizer,
            tasks=args.tasks,
            balance_guide=args.balance_guide,
            max_len=args.max_len,
        )
        val_dataset = FinetuneMultiDataset(
            data=fine_val,
            tokenizer=tokenizer,
            tasks=args.tasks,
            balance_guide=args.balance_guide,
            max_len=args.max_len,
        )
        test_dataset = FinetuneMultiDataset(
            data=fine_test,
            tokenizer=tokenizer,
            tasks=args.tasks,
            balance_guide=None,
            max_len=args.max_len,
        )

    else:
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

    elif args.model_type == "cehr_bigbird" or args.model_type == "cehr_multibird":
        pretrained_model = BigBirdPretrain(
            vocab_size=tokenizer.get_vocab_size(),
            padding_idx=tokenizer.get_pad_token_id(),
            **pre_model_config,
        )
        pretrained_model.load_state_dict(torch.load(args.pretrained_path)["state_dict"])
        model = BigBirdFinetune(
            pretrained_model=pretrained_model,
            num_labels=args.num_labels,
            problem_type=args.problem_type,
            **fine_model_config,
        )

    elif args.model_type == "ehr_mamba":
        pretrained_model = MambaPretrain(
            vocab_size=tokenizer.get_vocab_size(),
            padding_idx=tokenizer.get_pad_token_id(),
            cls_idx=tokenizer.get_class_token_id(),
            **pre_model_config,
        )
        pretrained_model.load_state_dict(torch.load(args.pretrained_path)["state_dict"])
        model = MambaFinetune(
            pretrained_model=pretrained_model,
            num_labels=args.num_labels,
            problem_type=args.problem_type,
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
        ckpt_path=args.resume_checkpoint,
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

    # Save test predictions
    if args.test_output_dir:
        torch.save(
            model.test_outputs,
            f"{args.test_output_dir}/test_outputs_{run_id}.pt",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # project configuration
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        help="Model type: 'cehr_bert' or 'cehr_bigbird', or 'ehr_mamba', or 'cehr_multibird",
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
        required=True,
        help="Path to model config file",
    )
    parser.add_argument(
        "--is-multi-model",
        type=bool,
        default=False,
        help="Is the model a multimodel like multibird or not",
    )
    parser.add_argument(
        "--is-decoder",
        type=bool,
        default=False,
        help="Is the model a decoder (e.g. Mamba) or not",
    )

    # data-related arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the data directory",
    )
    parser.add_argument(
        "--sequence-file",
        type=str,
        required=True,
        help="Path to the patient sequence file",
    )
    parser.add_argument(
        "--id-file",
        type=str,
        required=True,
        help="Path to the patient id file",
    )
    parser.add_argument(
        "--vocab-dir",
        type=str,
        required=True,
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
        required=True,
        help="Define the number of patients to be fine_tuned on",
    )
    parser.add_argument(
        "--problem_type",
        type=str,
        required=True,
        help="Define if its single_label_classification or multi_label_classification",
    )
    parser.add_argument(
        "--num_labels",
        type=int,
        required=True,
        help="Define the number of labels",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Define the finetune tasks for multi model",
    )
    parser.add_argument(
        "--balance_guide",
        type=str,
        default=None,
        help="Define the positive label ratios for label balancing",
    )

    # checkpointing and logging arguments
    parser.add_argument(
        "--checkpoint-dir",
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
        help="Checkpoint to resume finetuning from",
    )
    parser.add_argument(
        "--test_output_dir",
        type=str,
        default=None,
        help="Path to saved test outputs",
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
        default=False,
        action="store_true",
        help="Test the last checkpoint",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # Process arguments
    args = parser.parse_args()
    if args.model_type not in [
        "cehr_bert",
        "cehr_bigbird",
        "ehr_mamba",
        "cehr_multibird",
    ]:
        print(
            "Invalid model type. Choose 'cehr_bert' or 'cehr_bigbird', 'ehr_mamba', or 'cehr_multibird'."
        )
        sys.exit(1)

    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.exp_name)
    args.test_output_dir = os.path.join(args.checkpoint_dir, args.test_output_dir)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.test_output_dir, exist_ok=True)

    config = load_config(args.config_dir, args.model_type)
    finetune_config = config["finetune"]
    for key, value in finetune_config.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)

    pre_model_config = config["model"]
    args.max_len = pre_model_config["max_seq_length"]

    # Process the tasks and balance guide arguments
    args.tasks = args.tasks.strip().split(" ")
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
    fine_model_config = config["model_finetune"]
    main(args, pre_model_config, fine_model_config)
