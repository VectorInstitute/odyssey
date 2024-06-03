"""Run attribution methods for interpretability."""

import argparse
import logging
import os
import sys
from typing import Any, Dict

import torch

from odyssey.data.tokenizer import ConceptTokenizer
from odyssey.interp.attribution import Attribution
from odyssey.interp.utils import get_type_id_mapping
from odyssey.models.model_utils import (
    load_config,
    load_finetune_data,
    load_finetuned_model,
)
from odyssey.utils.log import setup_logging
from odyssey.utils.utils import seed_everything


LOGGER = logging.getLogger(__name__)
setup_logging(print_level="INFO", logger=LOGGER)


def main(
    args: argparse.Namespace,
    pre_model_config: Dict[str, Any],
    fine_model_config: Dict[str, Any],
) -> None:
    """Run interpretability."""
    seed_everything(args.seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    if args.tokenizer_path:
        tokenizer = ConceptTokenizer.load(args.tokenizer_path)
    else:
        tokenizer = ConceptTokenizer(data_dir=args.vocab_dir)
        tokenizer.fit_on_vocab(with_tasks=args.with_tasks)

    # Load data
    _, test_data = load_finetune_data(
        args.data_dir,
        args.sequence_file,
        args.id_file,
        args.valid_scheme,
        args.num_finetune_patients,
    )
    test_data.rename(columns={args.label_name: "label"}, inplace=True)
    test_data_sample = test_data.head(20)

    # Load model
    model = load_finetuned_model(
        args.model_type,
        args.model_path,
        tokenizer=tokenizer,
        pre_model_config=pre_model_config,
        fine_model_config=fine_model_config,
        device=device,
    )

    # Get attributions
    gradient_attr = Attribution(
        test_data_sample,
        model,
        tokenizer,
        device,
        type_id_mapping=get_type_id_mapping(),
        max_len=args.max_len,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        codes_dir=args.codes_dir,
    )

    token_avg = gradient_attr.average_tokens_attr()
    LOGGER.info(f"Token attributions: {token_avg}")
    embedding_avg = gradient_attr.average_embeddings_attr()
    LOGGER.info(f"Embedding attributions: {embedding_avg}")
    gradient_attr.visualize_expected_gradients(num_baselines=3)
    gradient_attr.visualize_integrated_gradients()


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
        "--model-path",
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
        "--tokenizer-path",
        type=str,
        default=None,
        help="Pretrained model",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="odyssey/models/configs",
        help="Path to model config file",
    )
    parser.add_argument(
        "--num_finetune_patients",
        type=str,
        default="20000",
        help="Define the number of patients to be fine_tuned on",
    )
    parser.add_argument(
        "--with-tasks",
        action="store_true",
        help="Whether to include tasks in the vocabulary",
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
        "--codes-dir",
        type=str,
        default="data_files/codes_dict",
        help="Path to the codes dictionary directory of json files",
    )
    parser.add_argument(
        "--valid_scheme",
        type=str,
        default="few_shot",
        help="Define the type of validation, few_shot or kfold",
    )
    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for loading the test data",
    )

    parser.add_argument(
        "--n-steps",
        type=int,
        default=50,
        help="Number of steps for integrated gradients interpolation",
    )
    args = parser.parse_args()

    if args.model_type not in ["cehr_bert", "cehr_bigbird"]:
        print("Invalid model type. Choose 'cehr_bert' or 'cehr_bigbird'.")
        sys.exit(1)

    config = load_config(args.config_dir, args.model_type)

    finetune_config = config["finetune"]
    for key, value in finetune_config.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)

    pre_model_config = config["model"]
    args.max_len = pre_model_config["max_seq_length"]

    fine_model_config = config["model_finetune"]

    main(args, pre_model_config, fine_model_config)
