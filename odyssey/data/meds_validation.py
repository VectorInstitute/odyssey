"""MEDS conformance validation: the gate at the pipeline's narrow waist.

Every data source gets its own extractor (standard tooling for MIMIC-IV, a
MESSY spec for eICU, a bespoke SQL streamer for GEMINI), but everything
downstream is written once against the MEDS event schema and never knows
which hospital system produced the data. That only holds if conformance at
the boundary is checked mechanically, not by convention -- this module is
that check, run against an extraction output directory before anything
trains on it.

Two levels, deliberately:

- :func:`validate_meds_dataset` (default): cheap structural checks that
  read only parquet *schemas* and small metadata files -- shard layout,
  column names and dtypes, ``metadata/`` presence and shape. Safe to run
  on a million-subject extraction in seconds.
- ``deep=True`` adds full-scan checks (per-shard sortedness by
  ``(subject_id, time)``, subjects never spanning shards, split
  assignments consistent with ``metadata/subject_splits.parquet``) --
  proportional to dataset size, run once per extraction, not per session.

Findings come back as a list rather than an exception so callers can
render everything at once; :func:`raise_on_errors` converts error-level
findings into a failure for use as a hard gate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import polars as pl


#: Split directory names the MEDS convention uses; extra split names are
#: legal (a source may define more), but at least one of these is expected.
KNOWN_SPLITS = ("train", "tuning", "held_out")

#: Required event columns -> validator for the polars dtype. Extension
#: columns beyond these (e.g. ``hadm_id``) are explicitly allowed by MEDS
#: and by every consumer in this repo.
_REQUIRED_COLUMNS = ("subject_id", "time", "code", "numeric_value")

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class Finding:
    """One conformance finding.

    ``code`` is a stable machine-readable slug (tests and callers match on
    it; messages are for humans and may be reworded freely).
    """

    severity: Severity
    code: str
    message: str
    path: str | None = None

    def __str__(self) -> str:
        """One log-ready line: severity, code, message, offending path."""
        where = f" [{self.path}]" if self.path else ""
        return f"{self.severity.upper()} {self.code}: {self.message}{where}"


def _err(code: str, message: str, path: Path | None = None) -> Finding:
    return Finding("error", code, message, str(path) if path else None)


def _warn(code: str, message: str, path: Path | None = None) -> Finding:
    return Finding("warning", code, message, str(path) if path else None)


def _check_shard_schema(shard: Path) -> list[Finding]:
    """Schema-only checks for one shard file (no data read)."""
    findings: list[Finding] = []
    try:
        schema = pl.read_parquet_schema(shard)
    except Exception as exc:  # noqa: BLE001 -- any unreadable shard is one finding
        return [_err("unreadable-shard", f"cannot read parquet schema: {exc}", shard)]
    for column in _REQUIRED_COLUMNS:
        if column not in schema:
            findings.append(
                _err("missing-column", f"required column {column!r} absent", shard)
            )
    if "subject_id" in schema and schema["subject_id"] != pl.Int64:
        findings.append(
            _err(
                "subject-id-dtype",
                f"subject_id must be Int64, got {schema['subject_id']}"
                " (hashed-string ids need a persisted int mapping first;"
                " see the GEMINI finalize step)",
                shard,
            )
        )
    if "time" in schema and not isinstance(schema["time"], pl.Datetime):
        findings.append(
            _err("time-dtype", f"time must be Datetime, got {schema['time']}", shard)
        )
    if "code" in schema and schema["code"] not in (pl.Utf8, pl.String):
        findings.append(
            _err(
                "code-dtype", f"code must be a string type, got {schema['code']}", shard
            )
        )
    if "numeric_value" in schema and schema["numeric_value"] not in (
        pl.Float32,
        pl.Float64,
    ):
        findings.append(
            _err(
                "numeric-value-dtype",
                f"numeric_value must be a float type, got {schema['numeric_value']}",
                shard,
            )
        )
    elif schema.get("numeric_value") == pl.Float64:
        findings.append(
            _warn(
                "numeric-value-width",
                "numeric_value is Float64; the MEDS schema specifies Float32",
                shard,
            )
        )
    return findings


def _check_metadata(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    metadata = root / "metadata"
    if not metadata.is_dir():
        return [
            _err(
                "missing-metadata",
                "metadata/ directory absent (dataset.json, codes.parquet,"
                " subject_splits.parquet)",
                metadata,
            )
        ]
    dataset_json = metadata / "dataset.json"
    if not dataset_json.is_file():
        findings.append(
            _err("missing-dataset-json", "dataset.json absent", dataset_json)
        )
    else:
        try:
            json.loads(dataset_json.read_text())
        except json.JSONDecodeError as exc:
            findings.append(
                _err(
                    "invalid-dataset-json",
                    f"dataset.json unparsable: {exc}",
                    dataset_json,
                )
            )
    codes = metadata / "codes.parquet"
    if not codes.is_file():
        findings.append(_warn("missing-codes-parquet", "codes.parquet absent", codes))
    else:
        schema = pl.read_parquet_schema(codes)
        if "code" not in schema:
            findings.append(
                _err(
                    "codes-parquet-shape", "codes.parquet lacks a 'code' column", codes
                )
            )
    splits = metadata / "subject_splits.parquet"
    if not splits.is_file():
        findings.append(
            _err(
                "missing-subject-splits",
                "subject_splits.parquet absent (split membership must be"
                " explicit, not implied)",
                splits,
            )
        )
    else:
        schema = pl.read_parquet_schema(splits)
        for column in ("subject_id", "split"):
            if column not in schema:
                findings.append(
                    _err(
                        "subject-splits-shape",
                        f"subject_splits.parquet lacks a {column!r} column",
                        splits,
                    )
                )
    return findings


def _split_dirs(data_dir: Path) -> list[Path]:
    return sorted(p for p in data_dir.iterdir() if p.is_dir())


def _deep_check_shard(shard: Path, seen_subjects: dict[int, str]) -> list[Finding]:
    """Full-scan checks for one shard: sortedness and subject disjointness."""
    findings: list[Finding] = []
    frame = pl.read_parquet(shard, columns=["subject_id", "time"])
    if frame.height == 0:
        return findings
    # MEDS orders events by subject, then time, within every shard. Null
    # times (static facts) are legal and sort first within their subject.
    sorted_frame = frame.sort(["subject_id", "time"], nulls_last=False)
    if not frame.equals(sorted_frame):
        findings.append(
            _err("unsorted-shard", "events not sorted by (subject_id, time)", shard)
        )
    for subject in frame["subject_id"].unique().to_list():
        prior = seen_subjects.get(subject)
        if prior is not None and prior != shard.name:
            findings.append(
                _err(
                    "subject-spans-shards",
                    f"subject {subject} appears in both {prior} and {shard.name};"
                    " subjects must never span shards",
                    shard,
                )
            )
        else:
            seen_subjects[subject] = shard.name
    return findings


def _deep_check_split_membership(data_dir: Path, root: Path) -> list[Finding]:
    """Split directories must agree exactly with subject_splits.parquet."""
    splits_path = root / "metadata" / "subject_splits.parquet"
    if not splits_path.is_file():
        return []  # already reported structurally
    declared = pl.read_parquet(splits_path)
    if "subject_id" not in declared.columns or "split" not in declared.columns:
        return []  # already reported structurally
    findings: list[Finding] = []
    for split_dir in _split_dirs(data_dir):
        declared_ids = set(
            declared.filter(pl.col("split") == split_dir.name)["subject_id"].to_list()
        )
        observed_ids: set[int] = set()
        for shard in sorted(split_dir.glob("*.parquet")):
            observed_ids |= set(
                pl.read_parquet(shard, columns=["subject_id"])["subject_id"]
                .unique()
                .to_list()
            )
        extra = observed_ids - declared_ids
        if extra:
            findings.append(
                _err(
                    "split-membership",
                    f"{len(extra)} subject(s) in {split_dir.name}/ shards are not"
                    f" declared {split_dir.name!r} in subject_splits.parquet"
                    f" (e.g. {sorted(extra)[:3]})",
                    split_dir,
                )
            )
    return findings


def validate_meds_dataset(root: Path | str, *, deep: bool = False) -> list[Finding]:
    """Validate one MEDS extraction output directory.

    Parameters
    ----------
    root : Path or str
        The dataset root: the directory containing ``data/`` and
        ``metadata/``.
    deep : bool
        Also run full-scan checks (sortedness, subject disjointness across
        shards, split membership vs ``subject_splits.parquet``). Cost is
        proportional to dataset size; run once per extraction.

    Returns
    -------
    list of Finding
        Empty when fully conformant. Error-level findings mean downstream
        consumers (or MEDS tooling like MEDS-Tab) will misbehave; warnings
        are deviations that current consumers tolerate.
    """
    root = Path(root)
    findings: list[Finding] = []
    data_dir = root / "data"
    if not data_dir.is_dir():
        return [_err("missing-data-dir", "data/ directory absent", data_dir)]
    split_dirs = _split_dirs(data_dir)
    if not split_dirs:
        findings.append(
            _err(
                "no-split-dirs",
                "data/ has no split subdirectories (expected e.g."
                f" {'/'.join(KNOWN_SPLITS)})",
                data_dir,
            )
        )
    elif not any(d.name in KNOWN_SPLITS for d in split_dirs):
        findings.append(
            _warn(
                "unconventional-splits",
                f"no conventional split name among {[d.name for d in split_dirs]}",
                data_dir,
            )
        )
    shards = [s for d in split_dirs for s in sorted(d.glob("*.parquet"))]
    if split_dirs and not shards:
        findings.append(_err("no-shards", "no parquet shards under data/", data_dir))
    for shard in shards:
        findings.extend(_check_shard_schema(shard))
    findings.extend(_check_metadata(root))
    if deep:
        seen_subjects: dict[int, str] = {}
        for shard in shards:
            findings.extend(_deep_check_shard(shard, seen_subjects))
        findings.extend(_deep_check_split_membership(data_dir, root))
    return findings


def raise_on_errors(findings: list[Finding]) -> None:
    """Raise ``ValueError`` listing every error-level finding, if any."""
    errors = [f for f in findings if f.severity == "error"]
    if errors:
        raise ValueError(
            "MEDS conformance check failed:\n" + "\n".join(str(f) for f in errors)
        )


__all__ = [
    "Finding",
    "KNOWN_SPLITS",
    "raise_on_errors",
    "validate_meds_dataset",
]
