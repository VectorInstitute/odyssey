"""Human-readable descriptions of MEDS codes, from ``metadata/codes.parquet``.

A MEDS extraction ships a code dictionary alongside its data shards
(``<extraction>/metadata/codes.parquet``: ``code``, ``description``,
``parent_codes``). On MIMIC-IV it names labs, vitals, infusions, diagnoses
and procedures (``LAB//220045//bpm`` -> "Heart Rate"); medications and
structural codes carry readable text in the code itself and have no entry.
"""

import logging
from pathlib import Path

import polars as pl


logger = logging.getLogger(__name__)

CODES_FILENAME = "codes.parquet"


def load_code_descriptions(metadata_dir: str | Path | None) -> dict[str, str]:
    """Return ``code -> description`` for every code with a non-empty description.

    Returns an empty mapping (and logs why) when ``metadata_dir`` is
    ``None``, has no ``codes.parquet``, or the file has no ``description``
    column, so callers can always fall back to structural formatting.
    """
    if metadata_dir is None:
        return {}
    path = Path(metadata_dir) / CODES_FILENAME
    if not path.exists():
        logger.warning("[code_metadata] no %s; codes stay as codes", path)
        return {}
    if "description" not in pl.read_parquet_schema(path):
        logger.warning(
            "[code_metadata] %s has no description column; codes stay as codes", path
        )
        return {}
    frame = pl.read_parquet(path, columns=["code", "description"]).filter(
        pl.col("description").is_not_null() & (pl.col("description") != "")
    )
    return dict(
        zip(frame["code"].to_list(), frame["description"].to_list(), strict=True)
    )


__all__ = ["CODES_FILENAME", "load_code_descriptions"]
