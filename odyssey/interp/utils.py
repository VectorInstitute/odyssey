"""Utility functions for the interpretability module."""

import json
import os
import re
from typing import Dict, List, Union


def get_type_id_mapping() -> Dict[int, str]:
    """
    Return a predefined mapping of type IDs to their respective token types.

    Returns
    -------
    Dict[int, str]
        A dictionary mapping type IDs to descriptive strings of token types.
    """
    return {
        0: "PAD",
        1: "CLS",
        2: "VS",
        3: "VE",
        4: "TIME_INTERVAL",
        5: "LAB",
        6: "MED",
        7: "PROC",
        8: "REG",
    }


def load_codes_dict(codes_dir: str) -> Dict[str, str]:
    """
    Load and merge JSON files containing medical codes and names.

    Parameters
    ----------
    codes_dir : str
        The directory path that contains JSON files with code mappings.

    Returns
    -------
    Dict[str, str]
        A dictionary that represents a medical concept code mapping.
    """
    merged_dict = {}
    for filename in os.listdir(codes_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(codes_dir, filename)
            with open(filepath, "r") as file:
                data = json.load(file)
                merged_dict.update(data)
    return merged_dict


def replace_sequence_items(
    sequence: List[Union[str, int]],
    mapping_dict: Dict[str, str],
) -> List[str]:
    """
    Replace medical concept codes in a sequence with their corresponding names if found.

    Parameters
    ----------
    sequence : list of str
        The original sequence of strings to be processed.
    mapping_dict : dict
        A dictionary mapping medical concept codes to their corresponding names.

    Returns
    -------
    list of str
        A new sequence with medical concept codes replaced by their names.
    """
    if type(sequence[0]) == int:
        sequence = [str(item) for item in sequence]

    new_sequence = []
    for item in sequence:
        match = re.match(r"^(.*?)(_\d)$", item)
        if match:
            base_part, suffix = match.groups()
            replaced_item = mapping_dict.get(base_part, base_part) + suffix
        else:
            replaced_item = mapping_dict.get(item, item)
        new_sequence.append(replaced_item)
    return new_sequence
