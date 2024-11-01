"""Process patient sequences based on task and split into train-test-finetune."""

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split

from odyssey.utils.utils import save_object_to_disk, seed_everything


SEED = 23
seed_everything(seed=SEED)


def filter_by_num_visit(dataset: pd.DataFrame, minimum_num_visits: int) -> pd.DataFrame:
    """Filter the patients based on num_visits threshold.

    Parameters
    ----------
    dataset: pd.DataFrame
        The input dataset.
    minimum_num_visits: int
        The threshold number of visits.

    Returns
    -------
    pd.DataFrame
        The filtered dataset.

    """
    filtered_dataset = dataset.loc[dataset["num_visits"] >= minimum_num_visits]
    filtered_dataset.reset_index(drop=True, inplace=True)
    return filtered_dataset


def filter_by_length_of_stay(
    dataset: pd.DataFrame, threshold: int = 1, max_len: int = 2048
) -> pd.DataFrame:
    """Filter the patients based on length of stay threshold.

    Parameters
    ----------
    dataset: pd.DataFrame
        The input dataset.
    threshold: int
        The threshold length of stay.
    max_len: int
        The maximum length of the sequence.

    Returns
    -------
    pd.DataFrame
        The filtered dataset.

    """
    filtered_dataset = dataset.loc[dataset["length_of_stay"] >= threshold]

    # Only keep the patients that their first event happens within threshold hours
    filtered_dataset = filtered_dataset[
        filtered_dataset.apply(
            lambda row: row[f"elapsed_tokens_{max_len}"][row["last_VS_index"] + 1]
            < threshold * 24,
            axis=1,
        )
    ]

    filtered_dataset.reset_index(drop=True, inplace=True)
    return filtered_dataset


def get_last_occurence_index(seq: List[str], target: str) -> int:
    """Return the index of the last occurrence of target in seq.

    Parameters
    ----------
    seq: List[str]
        The input sequence.
    target: str
        The target token.

    Returns
    -------
    int
        The index of the last occurrence of target in seq.

    """
    return len(seq) - (seq[::-1].index(target) + 1)


def check_readmission_label(row: pd.Series, max_len: int = 2048) -> int:
    """Check if the label indicates readmission within one month.

    Parameters
    ----------
    row: pd.Series
        The input row.
    max_len: int
        The maximum length of the sequence.

    Returns
    -------
    int
        The readmission label.

    """
    last_vs_index = row["last_VS_index"]
    return int(
        row[f"event_tokens_{max_len}"][last_vs_index - 1]
        in ("[W_0]", "[W_1]", "[W_2]", "[W_3]", "[M_1]"),
    )


def get_length_of_stay(row: pd.Series) -> pd.Series:
    """Determine the length of a given visit.

    Parameters
    ----------
    row: pd.Series
        The input row.

    Returns
    -------
    float
        The length of stay in days.

    """
    admission_time = row["last_VS_index"] + 1
    discharge_time = row["last_VE_index"] - 1
    return (discharge_time - admission_time) / 24


def get_visit_cutoff_at_threshold(
    row: pd.Series, threshold: int = 24, max_len: int = 2048
) -> int:
    """Get the index of the first event token of last visit that cutoff at threshold.

    Parameters
    ----------
    row: pd.Series
        The input row.
    threshold: int
        The threshold length of stay.
    max_len: int
        The maximum length of the sequence.

    Returns
    -------
    int
        The index of the first event token of last visit that occurs
        after threshold hours.

    """
    last_vs_index = row["last_VS_index"]
    last_ve_index = row["last_VE_index"]

    for i in range(last_vs_index + 1, last_ve_index):
        if row[f"elapsed_tokens_{max_len}"][i] > threshold:
            return i

    return len(row[f"event_tokens_{max_len}"])


def process_length_of_stay_dataset(
    dataset: pd.DataFrame,
    threshold: int = 7,
    max_len: int = 2048,
) -> pd.DataFrame:
    """Process the length of stay dataset to extract required features.

    Parameters
    ----------
    dataset: pd.DataFrame
        The input dataset.
    threshold: int
        The threshold length of stay.
    max_len: int
        The maximum length of the sequence.

    Returns
    -------
    pd.DataFrame
        The processed dataset.

    """
    dataset["last_VS_index"] = dataset[f"event_tokens_{max_len}"].transform(
        lambda seq: get_last_occurence_index(list(seq), "[VS]"),
    )
    dataset["last_VE_index"] = dataset[f"event_tokens_{max_len}"].transform(
        lambda seq: get_last_occurence_index(list(seq), "[VE]"),
    )
    dataset["length_of_stay"] = dataset.apply(get_length_of_stay, axis=1)

    dataset = filter_by_length_of_stay(dataset, threshold=1)
    dataset["label_los_1week"] = (dataset["length_of_stay"] >= threshold).astype(int)

    dataset["cutoff_los"] = dataset.apply(
        lambda row: get_visit_cutoff_at_threshold(row, threshold=24),
        axis=1,
    )
    dataset["token_length"] = dataset[f"event_tokens_{max_len}"].apply(len)

    return dataset


def process_condition_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """Process the condition dataset to extract required features.

    Parameters
    ----------
    dataset: pd.DataFrame
        The input dataset.

    Returns
    -------
    pd.DataFrame
        The processed dataset.

    """
    dataset["all_conditions"] = dataset.apply(
        lambda row: np.concatenate(
            [row["common_conditions"], row["rare_conditions"]],
            dtype=np.int64,
        ),
        axis=1,
    )

    return dataset


def process_mortality_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """Process the mortality dataset to extract required features.

    Parameters
    ----------
    dataset: pd.DataFrame
        The input dataset.

    Returns
    -------
    pd.DataFrame
        The processed dataset.

    """
    dataset["label_mortality_2weeks"] = (
        (dataset["death_after_start"] >= 0) & (dataset["death_after_end"] <= 15)
    ).astype(int)

    dataset["label_mortality_1month"] = (
        (dataset["death_after_start"] >= 0) & (dataset["death_after_end"] <= 32)
    ).astype(int)

    return dataset


def process_readmission_dataset(
    dataset: pd.DataFrame, max_len: int = 2048
) -> pd.DataFrame:
    """Process the readmission dataset to extract required features.

    Parameters
    ----------
    dataset: pd.DataFrame
        The input dataset.
    max_len: int
        The maximum length of the sequence.

    Returns
    -------
    pd.DataFrame
        The processed dataset.

    """
    dataset = filter_by_num_visit(dataset.copy(), minimum_num_visits=2)

    dataset["last_VS_index"] = dataset[f"event_tokens_{max_len}"].transform(
        lambda seq: get_last_occurence_index(list(seq), "[VS]"),
    )
    dataset["cutoff_readmission"] = dataset["last_VS_index"] - 1
    dataset["label_readmission_1month"] = dataset.apply(check_readmission_label, axis=1)

    dataset["num_visits"] -= 1
    dataset["token_length"] = dataset[f"event_tokens_{max_len}"].apply(len)

    return dataset


def process_multi_dataset(
    datasets: Dict[str, pd.DataFrame],
    max_len: int = 2048,
    num_conditions: int = 20,
    nan_indicator: int = -1,
) -> pd.DataFrame:
    """Process the multi-task dataset by merging the original dataset with others.

    Parameters
    ----------
    datasets: Dict[str, pd.DataFrame]
        The input datasets.
    max_len: int
        The maximum length of the sequence.
    num_conditions: int
        The number of conditions.
    nan_indicator: int
        The indicator for NaN values.

    Returns
    -------
    pd.DataFrame
        The processed dataset.

    """
    # Merging datasets on 'patient_id'
    multi_dataset = datasets["original"].merge(
        datasets["condition"][["patient_id", "all_conditions"]],
        on="patient_id",
        how="left",
    )
    multi_dataset = multi_dataset.merge(
        datasets["mortality"][["patient_id", "label_mortality_1month"]],
        on="patient_id",
        how="left",
    )
    multi_dataset = multi_dataset.merge(
        datasets["readmission"][
            ["patient_id", "cutoff_readmission", "label_readmission_1month"]
        ],
        on="patient_id",
        how="left",
    )
    multi_dataset = multi_dataset.merge(
        datasets["los"][["patient_id", "cutoff_los", "label_los_1week"]],
        on="patient_id",
        how="left",
    )

    # Selecting the required columns
    multi_dataset = multi_dataset[
        [
            "patient_id",
            "num_visits",
            f"event_tokens_{max_len}",
            f"type_tokens_{max_len}",
            f"age_tokens_{max_len}",
            f"time_tokens_{max_len}",
            f"visit_tokens_{max_len}",
            f"position_tokens_{max_len}",
            f"elapsed_tokens_{max_len}",
            "cutoff_los",
            "cutoff_readmission",
            "all_conditions",
            "label_mortality_1month",
            "label_readmission_1month",
            "label_los_1week",
        ]
    ]

    # Transform conditions from a vector of numbers to binary classes
    conditions_expanded = multi_dataset["all_conditions"].apply(pd.Series)
    conditions_expanded.columns = [f"condition{i}" for i in range(num_conditions)]
    multi_dataset = multi_dataset.drop("all_conditions", axis=1)
    multi_dataset = pd.concat([multi_dataset, conditions_expanded], axis=1)

    # Standardize important column names
    multi_dataset.rename(
        columns={
            "cutoff_los": "cutoff_los_1week",
            "cutoff_readmission": "cutoff_readmission_1month",
        },
        inplace=True,
    )
    condition_columns = {f"condition{i}": f"label_c{i}" for i in range(num_conditions)}
    multi_dataset.rename(columns=condition_columns, inplace=True)

    numerical_columns = [
        "cutoff_los_1week",
        "cutoff_readmission_1month",
        "label_mortality_1month",
        "label_readmission_1month",
        "label_los_1week",
    ] + [f"label_c{i}" for i in range(num_conditions)]

    # Fill NaN values and convert numerical columns to integers
    for column in numerical_columns:
        multi_dataset[column] = multi_dataset[column].fillna(nan_indicator).astype(int)

    # Reset dataset index
    multi_dataset.reset_index(drop=True, inplace=True)

    return multi_dataset


def stratified_train_test_split(
    dataset: pd.DataFrame,
    target: str,
    test_size: float,
    return_test: Optional[bool] = False,
    seed: int = SEED,
) -> List[str]:
    """Split the given dataset into training and testing sets.

    The dataset is stratified using iterative stratification on given multi-label
    target.

    Parameters
    ----------
    dataset: pd.DataFrame
        The input dataset.
    target: str
        The target column for stratification.
    test_size: float
        The size of the test set.
    return_test: Optional[bool]
        Whether to return the test set only.
    seed: int
        The random seed for reproducibility.

    Returns
    -------
    List[str]
        The patient ids for the training or testing set.

    """
    # Convert all_conditions into a format suitable for multi-label stratification
    labels = np.array(dataset[target].values.tolist())
    inputs = dataset["patient_id"].to_numpy().reshape(-1, 1)
    is_single_label = type(dataset.iloc[0][target]) == np.int64

    # Perform stratified split
    if is_single_label:
        X_train, X_test, _, _ = train_test_split(
            inputs,
            labels,
            stratify=labels,
            test_size=test_size,
            random_state=seed,
        )

    else:
        X_train, _, X_test, _ = iterative_train_test_split(
            inputs,
            labels,
            test_size=test_size,
        )

    X_train = X_train.flatten().tolist()
    X_test = X_test.flatten().tolist()

    if return_test:
        return X_test

    return X_train, X_test


def sample_balanced_subset(
    dataset: pd.DataFrame, target: str, sample_size: int, seed: int = SEED
) -> List[str]:
    """Sample a subset of dataset with balanced target labels.

    Parameters
    ----------
    dataset: pd.DataFrame
        The input dataset.
    target: str
        The target column for stratification.
    sample_size: int
        The size of the sample.
    seed: int
        The random seed for reproducibility.

    Returns
    -------
    List[str]
        The patient ids for the balanced sample.

    """
    # Sampling positive and negative patients
    pos_patients = dataset[dataset[target] == True].sample(  # noqa: E712
        n=sample_size // 2,
        random_state=seed,
    )
    neg_patients = dataset[dataset[target] == False].sample(  # noqa: E712
        n=sample_size // 2,
        random_state=seed,
    )

    # Combining and shuffling patient IDs
    sample_patients = (
        pos_patients["patient_id"].tolist() + neg_patients["patient_id"].tolist()
    )
    random.shuffle(sample_patients)

    return sample_patients


def get_pretrain_test_split(
    dataset: pd.DataFrame,
    stratify_target: Optional[str] = None,
    test_size: float = 0.15,
    seed: int = SEED,
) -> Tuple[List[str], List[str]]:
    """Split dataset into pretrain and test set.

    The dataset is stratified on a given target column if needed.

    Parameters
    ----------
    dataset: pd.DataFrame
        The input dataset.
    stratify_target: Optional[str]
        The target column for stratification.
    test_size: float
        The size of the test set.
    seed: int
        The random seed for reproducibility.

    Returns
    -------
    Tuple[List[str], List[str]]
        The patient ids for the pretrain and test set.

    """
    if stratify_target:
        pretrain_ids, test_ids = stratified_train_test_split(
            dataset,
            target=stratify_target,
            test_size=test_size,
        )
    else:
        test_patients = dataset.sample(
            n=int(test_size * len(dataset)), random_state=seed
        )
        test_ids = test_patients["patient_id"].tolist()
        pretrain_ids = dataset[~dataset["patient_id"].isin(test_ids)][
            "patient_id"
        ].tolist()

    random.seed(seed)
    random.shuffle(pretrain_ids)

    return pretrain_ids, test_ids


def get_finetune_split(
    task_config: Any,
    task: str,
    patient_ids_dict: Dict[str, Any],
) -> Dict[str, Dict[str, List[str]]]:
    """Split the dataset into training and cross-finetuneation sets.

    Using k-fold cross-finetuneation while ensuring balanced label distribution
    in each fold, the function saves the resulting dictionary to disk.

    Parameters
    ----------
    task_config: Any
        The task configuration.
    task: str
        The task name.
    patient_ids_dict: Dict[str, Any]
        The dictionary containing patient ids for different splits.

    Returns
    -------
    Dict[str, Dict[str, List[str]]]
        The dictionary containing patient ids for different splits.

    """
    # Extract task-specific configuration
    dataset = task_config[task]["dataset"].copy()
    label_col = task_config[task]["label_col"]
    finetune_sizes = task_config[task]["finetune_size"]
    save_path = task_config[task]["save_path"]
    split_mode = task_config[task]["split_mode"]

    # Get pretrain dataset
    pretrain_ids = patient_ids_dict["pretrain"]
    dataset = dataset[dataset["patient_id"].isin(pretrain_ids)]

    # Few-shot finetune patient ids
    for finetune_num in finetune_sizes:
        if split_mode == "single_label_balanced":
            finetune_ids = sample_balanced_subset(
                dataset,
                target=label_col,
                sample_size=finetune_num,
            )

        elif split_mode in {"single_label_stratified", "multi_label_stratified"}:
            finetune_ids = stratified_train_test_split(
                dataset,
                target=label_col,
                test_size=finetune_num / len(dataset),
                return_test=True,
            )

        patient_ids_dict["finetune"]["few_shot"][f"{finetune_num}"] = finetune_ids

    # Save the dictionary to disk
    save_object_to_disk(patient_ids_dict, save_path)

    return patient_ids_dict
