"""
File: finetune_bigbird.py.

Finetune an already pretrained bigbird model on MIMIC-IV FHIR data.
The finetuning objective is binary classification on patient mortality or
hospital readmission labels.
"""

import os
import glob
import argparse
import pickle
from os.path import join

from typing import Any, Dict

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from lightning.pytorch.loggers import WandbLogger

from sklearn.model_selection import train_test_split

from models.big_bird_cehr.data import FinetuneDataset
from models.big_bird_cehr.model import BigBirdPretrain, BigBirdFinetune
from models.big_bird_cehr.tokenizer import HuggingFaceConceptTokenizer

ROOT = "/h/afallah/odyssey/odyssey"


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
    list_of_files = glob.glob(os.path.join(checkpoint_dir, 'best-v3.ckpt'))
    return list_of_files[-1]
    # return max(list_of_files, key=os.path.getmtime) if list_of_files else None


def main(args: Any) -> None:
    """ Train the model. """

    # Setup environment
    seed_everything(args.seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")

    # Load data
    data = pd.read_parquet(join(args.data_dir, "patient_sequences_2048_labeled.parquet"))
    patient_ids = pickle.load(open(join(args.data_dir, 'dataset_2048_mortality_1month.pkl'), 'rb'))

    fine_tune = data.loc[
        data['patient_id'].isin(patient_ids['valid'][args.valid_scheme][args.num_finetune_patients])
    ]
    fine_test = data.loc[
        data['patient_id'].isin(patient_ids['test'])
    ]

    fine_tune.rename(columns={'label_mortality_1month': 'label'}, inplace=True)
    fine_test.rename(columns={'label_mortality_1month': 'label'}, inplace=True)

    # Split data
    fine_train, fine_val = train_test_split(
        fine_tune,
        test_size=args.valid_size,
        random_state=args.seed,
        stratify=fine_tune["label"],
    )

    # Train Tokenizer
    tokenizer = HuggingFaceConceptTokenizer(data_dir=args.vocab_dir)
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
        # num_workers=2,
        # persistent_workers=True,
        shuffle=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        # num_workers=1,
        # persistent_workers=True,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        # num_workers=2,
        # persistent_workers=True,
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
        project="bigbird_finetune_mortality_1month_20000_patients",
        save_dir=args.log_dir,
    )

    # Load latest checkpoint to continue training
    latest_checkpoint = get_latest_checkpoint(args.checkpoint_dir)

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
        log_every_n_steps=args.log_every_n_steps,
        accumulate_grad_batches=args.acc,
        gradient_clip_val=1.0
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
        args=args,
        dataset_len=len(train_dataset),
        pretrained_model=pretrained_model,
    )

    # Train the model
    #trainer.fit(
    #    model=model,
    #    train_dataloaders=train_loader,
    #    val_dataloaders=val_loader,
    #    ckpt_path=latest_checkpoint if args.resume else None,
    #)

    checkpoint = torch.load(latest_checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    # Test the model
    trainer.test(
        model=model,
        dataloaders=test_loader,
    )

    # Save the model directly to disk
    #state_dict = model.state_dict()
    #torch.save(state_dict, args.output_dir)


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
        default=f"{ROOT}/data/bigbird_data",
        help="Path to the data directory"
    )
    parser.add_argument(
        "--vocab_dir", type=str,
        default=f"{ROOT}/data/vocab",
        help="Path to the vocabulary directory of json files"
    )
    parser.add_argument(
        "--valid_size",
        type=float,
        default=0.1,
        help="Validation set size for splitting the data",
    )
    parser.add_argument(
        "--max_len", type=int, default=2048, help="Maximum length of the sequence"
    )
    parser.add_argument(
        "--batch_size", type=int, default=48, help="Batch size for training"
    )
    parser.add_argument(
        "--num_workers", type=int, default=2, help="Number of workers for training"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=f"{ROOT}/checkpoints/bigbird_finetune/mortality_1month_20000_patients",
        help="Path to the training checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=f"{ROOT}/checkpoints/bigbird_finetune/mortality_1month_20000_patients/" +
                "bigbird_finetune_mortality_1month_20000_patients.pt",
        help="Path to the training checkpoint",
    )
    parser.add_argument(
        "--log_dir", type=str, default="/h/afallah/odyssey/wandb", help="Path to the log directory"
    )
    parser.add_argument(
        "--gpus", type=int, default=1, help="Number of gpus for training"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=5, help="Number of epochs for training"
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
        "--pretrained_path",
        type=str,
        default=f"{ROOT}/checkpoints/bigbird_pretraining_a100/best.ckpt",
        help="Checkpoint to the pretrained model",
    )
    parser.add_argument(
        '--valid_scheme',
        type=str,
        default='few_shot',
        help='Define the type of validation, few_shot or kfold'
    )
    parser.add_argument(
        '--num_finetune_patients',
        type=str,
        default='20000_patients',
        help='Define the number of patients to be fine_tuned on'
    )

    args = parser.parse_args()
    main(args)
