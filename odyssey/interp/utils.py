"""Utility functions for the interpretability module."""

from typing import Dict


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
