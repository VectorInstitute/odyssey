"""Aggregate cohort description of one MEDS source, split by split.

For every split directory (train, tuning, held_out by default) this
reports subjects, admissions, hospitals (when the MEDS metadata carries
a site code), the calendar year range of admissions, length of stay,
sex and age at admission where the source charts them, and the share of
subjects (and admissions) with at least one onset of each hazard event,
using the SAME onset definitions the alerts leg scores
(:func:`odyssey.data.alert_events.alert_events_for` resolved for the
source, :func:`odyssey.data.alert_events.all_event_times`). Nothing
patient-level is kept: every count below :data:`SUPPRESS_BELOW` is
written as ``"<10"`` and the statistics that depend on it are dropped,
so the JSON can leave a closed environment such as GEMINI.

Definitions:

- A subject is a distinct ``subject_id``; shards partition subjects, so
  per-shard counts add up.
- An admission is a distinct ``(subject_id, hadm_id)`` with at least one
  timed event. Its start is the source's admission code
  (``HOSPITAL_ADMISSION//`` on MIMIC-IV and eICU, bare ``ADMISSION`` on
  GEMINI) when the visit carries one, else the visit's first timed
  event; its end is the discharge code, else the last timed event.
  Length of stay is end minus start in days; the year range is over
  admission starts.
- Sex comes from ``GENDER//<value>`` static rows; age at admission from
  ``MEDS_BIRTH``. A source without those rows (GEMINI extracts neither)
  reports them as not available rather than as zero.
- Event prevalence per subject is the share of the split's subjects
  with at least one onset. For ``readmission_30d`` the onset must fall
  within 720 h of the visit's last event, matching the 30-day horizon
  the discharge-anchored scorer uses.

Usage::

    uv run python scripts/cohort_counts.py --source gemini --task-set v3 \
        --meds-dir /path/to/gemini_meds_v1 \
        [--hadm-hospital-parquet /path/to/metadata/hadm_id_hospital.parquet] \
        --output-json ~/runs/<run>/cohort_counts.json

    uv run python scripts/cohort_counts.py --source mimic_iv --task-set v3 \
        --split train=/data/mimic/train --split held_out=/data/mimic/held_out \
        --output-json cohort_counts.json

``--max-shards`` bounds every split (recorded in the output, so a subset
run can never pass for a full one); ``--no-events`` skips the label pass
and writes only the subject/admission/LOS/year/sex/age block.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from odyssey.data.alert_events import (
    ALERT_TASK_SETS,
    AlertEvent,
    alert_events_for,
    all_event_times,
    visit_envelope,
)
from odyssey.data.code_normalization import maybe_normalize
from odyssey.data.sequences import BIRTH_CODE, HOURS_PER_YEAR
from odyssey.data.sidecars import activate_sidecars
from odyssey.training.data import load_meds_shard
from odyssey.training.shard_stream import shard_paths


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("cohort_counts")

#: Counts below this are written as ``"<10"`` and any statistic over
#: fewer than this many records is dropped.
SUPPRESS_BELOW = 10
SUPPRESSED = f"<{SUPPRESS_BELOW}"

DEFAULT_SPLITS: tuple[str, ...] = ("train", "tuning", "held_out")

#: The source's hospital admission / discharge code families. The
#: admission prefixes are the ones odyssey.data.alert_events resolves the
#: readmission_30d event to per source (the shared MIMIC/eICU definition
#: and GEMINI's bare-token override); discharge has no event of its own.
ADMISSION_PREFIX: dict[str, str] = {
    "mimic_iv": "HOSPITAL_ADMISSION//",
    "eicu": "HOSPITAL_ADMISSION//",
    "gemini": "ADMISSION",
}
DISCHARGE_PREFIX: dict[str, str] = {
    "mimic_iv": "HOSPITAL_DISCHARGE//",
    "eicu": "HOSPITAL_DISCHARGE//",
    "gemini": "DISCHARGE",
}
SEX_PREFIX = "GENDER//"

#: The 30-day readmission window, in hours: the longest horizon of
#: odyssey.inference.alerts.READMISSION_HORIZONS_HOURS (kept as a literal
#: here so this script does not import torch through that module; a test
#: pins the two together).
READMISSION_WINDOW_HOURS = 720.0

LOS_DEFINITION = (
    "per admission (subject, visit id): admission code time else first "
    "timed event, to discharge code time else last timed event, in days; "
    "negative spans dropped"
)

#: Top-level keys of the JSON this script writes. run.sh's export
#: whitelist for the cohort-counts step must list exactly these.
OUTPUT_KEYS: tuple[str, ...] = (
    "source",
    "task_set",
    "normalize_medications",
    "suppression_threshold",
    "max_shards",
    "los_definition",
    "hospital_metadata",
    "events",
    "events_dropped",
    "splits",
    "all_splits",
)


@dataclass
class SplitAccumulator:
    """Per-split running aggregates; merges across shards and across splits."""

    n_shards_read: int = 0
    n_shards_total: int = 0
    n_subjects: int = 0
    n_admissions: int = 0
    hospitals: set[str] = field(default_factory=set)
    year_min: int | None = None
    year_max: int | None = None
    los_days: list[np.ndarray] = field(default_factory=list)
    sex: Counter[str] = field(default_factory=Counter)
    n_subjects_with_sex: int = 0
    ages: list[np.ndarray] = field(default_factory=list)
    event_subjects: Counter[str] = field(default_factory=Counter)
    event_admissions: Counter[str] = field(default_factory=Counter)

    def merge(self, other: SplitAccumulator) -> None:
        """Fold ``other`` into this accumulator."""
        self.n_shards_read += other.n_shards_read
        self.n_shards_total += other.n_shards_total
        self.n_subjects += other.n_subjects
        self.n_admissions += other.n_admissions
        self.hospitals |= other.hospitals
        for y in (other.year_min, other.year_max):
            if y is None:
                continue
            self.year_min = y if self.year_min is None else min(self.year_min, y)
            self.year_max = y if self.year_max is None else max(self.year_max, y)
        self.los_days.extend(other.los_days)
        self.sex.update(other.sex)
        self.n_subjects_with_sex += other.n_subjects_with_sex
        self.ages.extend(other.ages)
        self.event_subjects.update(other.event_subjects)
        self.event_admissions.update(other.event_admissions)


def suppress(n: int) -> int | str:
    """Return ``n`` or the suppression marker when it is a small cell."""
    return n if n >= SUPPRESS_BELOW else SUPPRESSED


def _quartiles(chunks: list[np.ndarray]) -> dict[str, Any] | None:
    values = np.concatenate(chunks) if chunks else np.empty(0)
    values = values[np.isfinite(values)]
    if len(values) < SUPPRESS_BELOW:
        return None
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    return {
        "n": int(len(values)),
        "median": round(float(med), 2),
        "q1": round(float(q1), 2),
        "q3": round(float(q3), 2),
    }


def _visits(events: pl.DataFrame, source: str) -> pl.DataFrame:
    """One row per admission with its start and end instants."""
    timed = events.filter(
        pl.col("time").is_not_null()
        & pl.col("hadm_id").is_not_null()
        & (pl.col("code") != BIRTH_CODE)
    )
    adm, dis = ADMISSION_PREFIX[source], DISCHARGE_PREFIX[source]
    visits = timed.group_by("subject_id", "hadm_id").agg(
        pl.col("time").min().alias("_first"),
        pl.col("time").max().alias("_last"),
        pl.col("time").filter(pl.col("code").str.starts_with(adm)).min().alias("_adm"),
        pl.col("time").filter(pl.col("code").str.starts_with(dis)).max().alias("_dis"),
    )
    return visits.select(
        "subject_id",
        "hadm_id",
        pl.coalesce(pl.col("_adm"), pl.col("_first")).alias("start"),
        pl.coalesce(pl.col("_dis"), pl.col("_last")).alias("end"),
    )


def _readmission_within_horizon(
    onset: dict[tuple[int, int], float],
    envelope: dict[tuple[int, int], tuple[float, float]],
    horizon: float,
) -> dict[tuple[int, int], float]:
    """Keep only next-admission onsets within ``horizon`` of the visit's end."""
    return {
        key: t
        for key, t in onset.items()
        if key in envelope and t - envelope[key][1] <= horizon
    }


def _event_counts(
    events: pl.DataFrame,
    alerts: tuple[AlertEvent, ...],
    source: str,
    task_set: str,
) -> tuple[Counter[str], Counter[str]]:
    """Subjects and admissions with at least one onset, per event."""
    times = all_event_times(events, alerts, source, task_set=task_set)
    envelope = None
    per_subject: Counter[str] = Counter()
    per_admission: Counter[str] = Counter()
    for alert in alerts:
        onset = times[alert.name].onset
        if alert.next_visit:
            if envelope is None:
                envelope = visit_envelope(events)
            onset = _readmission_within_horizon(
                onset, envelope, READMISSION_WINDOW_HOURS
            )
        per_subject[alert.name] = len({s for s, _ in onset})
        if not times[alert.name].subject_scoped:
            per_admission[alert.name] = len(onset)
    return per_subject, per_admission


def summarize_shard(
    events: pl.DataFrame,
    *,
    source: str,
    task_set: str,
    alerts: tuple[AlertEvent, ...] | None,
    normalize_medications: bool,
    hospital_by_hadm: pl.DataFrame | None,
) -> SplitAccumulator:
    """Aggregate one shard's events into a fresh :class:`SplitAccumulator`."""
    if "hadm_id" not in events.columns:
        events = events.with_columns(pl.lit(None, dtype=pl.Int64).alias("hadm_id"))
    acc = SplitAccumulator(n_shards_read=1)
    acc.n_subjects = int(events["subject_id"].n_unique())

    visits = _visits(events, source)
    acc.n_admissions = visits.height
    if visits.height:
        years = visits["start"].dt.year()
        acc.year_min = int(years.min())  # type: ignore[arg-type]
        acc.year_max = int(years.max())  # type: ignore[arg-type]
        los = (visits["end"] - visits["start"]).dt.total_seconds().to_numpy() / 86400.0
        acc.los_days.append(los[los >= 0].astype(np.float32))
        if hospital_by_hadm is not None:
            hit = visits.select(pl.col("hadm_id").cast(pl.Utf8)).join(
                hospital_by_hadm, on="hadm_id", how="inner"
            )
            acc.hospitals = set(hit["hospital_num"].unique().to_list())

    sex_rows = events.filter(pl.col("code").str.starts_with(SEX_PREFIX))
    if sex_rows.height:
        first = sex_rows.group_by("subject_id").agg(pl.col("code").first())
        categories = first["code"].str.slice(len(SEX_PREFIX)).to_list()
        acc.sex.update(categories)
        acc.n_subjects_with_sex = first.height

    births = events.filter(pl.col("code") == BIRTH_CODE)
    if births.height and visits.height:
        birth_by_subject = births.group_by("subject_id").agg(
            pl.col("time").min().alias("_birth")
        )
        aged = visits.join(birth_by_subject, on="subject_id", how="inner").filter(
            pl.col("_birth").is_not_null()
        )
        if aged.height:
            hours = (aged["start"] - aged["_birth"]).dt.total_seconds().to_numpy()
            acc.ages.append((hours / 3600.0 / HOURS_PER_YEAR).astype(np.float32))

    if alerts:
        prepared = maybe_normalize(events, enabled=normalize_medications, source=source)
        acc.event_subjects, acc.event_admissions = _event_counts(
            prepared, alerts, source, task_set
        )
    return acc


def summarize_split(
    shard_dir: Path,
    *,
    source: str,
    task_set: str,
    alerts: tuple[AlertEvent, ...] | None,
    normalize_medications: bool,
    hospital_by_hadm: pl.DataFrame | None,
    max_shards: int | None,
) -> SplitAccumulator:
    """Aggregate every shard of one split directory."""
    paths = shard_paths(shard_dir)
    total = SplitAccumulator(n_shards_total=len(paths))
    if max_shards is not None:
        paths = paths[:max_shards]
    if alerts:
        activate_sidecars(shard_dir)
    for i, path in enumerate(paths, 1):
        acc = summarize_shard(
            load_meds_shard(path),
            source=source,
            task_set=task_set,
            alerts=alerts,
            normalize_medications=normalize_medications,
            hospital_by_hadm=hospital_by_hadm,
        )
        total.merge(acc)
        if i % 25 == 0 or i == len(paths):
            logger.info(
                "%s: %d/%d shards, %d subjects, %d admissions",
                shard_dir.name,
                i,
                len(paths),
                total.n_subjects,
                total.n_admissions,
            )
    return total


def _rate(numerator: int, denominator: int) -> float | None:
    if numerator < SUPPRESS_BELOW or denominator < SUPPRESS_BELOW:
        return None
    return round(numerator / denominator, 4)


def report_split(
    acc: SplitAccumulator,
    *,
    alerts: tuple[AlertEvent, ...] | None,
    hospitals_available: bool,
) -> dict[str, Any]:
    """Suppressed, JSON-ready view of one accumulator."""
    enough = acc.n_admissions >= SUPPRESS_BELOW
    out: dict[str, Any] = {
        "n_shards_read": acc.n_shards_read,
        "n_shards_total": acc.n_shards_total,
        "n_subjects": suppress(acc.n_subjects),
        "n_admissions": suppress(acc.n_admissions),
        "n_hospitals": len(acc.hospitals) if hospitals_available else None,
        "admission_years": (
            {"min": acc.year_min, "max": acc.year_max} if enough else None
        ),
        "los_days": _quartiles(acc.los_days),
        "sex": None,
        "age_years": _quartiles(acc.ages),
        "events": None,
    }
    if not hospitals_available:
        out["hospitals_note"] = "not available: no site code in the MEDS metadata"
    if acc.n_subjects_with_sex:
        out["sex"] = {
            "n_subjects_with_sex": suppress(acc.n_subjects_with_sex),
            "counts": {k: suppress(v) for k, v in sorted(acc.sex.items())},
        }
    else:
        out["sex_note"] = "not available: no GENDER// rows in this source"
    if not acc.ages:
        out["age_note"] = "not available: no MEDS_BIRTH rows in this source"
    if alerts:
        out["events"] = {
            a.name: {
                "n_subjects_positive": suppress(acc.event_subjects[a.name]),
                "prevalence_per_subject": _rate(
                    acc.event_subjects[a.name], acc.n_subjects
                ),
                "n_admissions_positive": (
                    None if a.subject_scoped else suppress(acc.event_admissions[a.name])
                ),
                "prevalence_per_admission": (
                    None
                    if a.subject_scoped
                    else _rate(acc.event_admissions[a.name], acc.n_admissions)
                ),
            }
            for a in alerts
        }
    return out


def format_table(report: dict[str, Any]) -> str:
    """Plain-text summary of the report for the log."""
    lines = [f"source: {report['source']}  task_set: {report['task_set']}"]
    lines.append(
        f"{'split':<12}{'subjects':>12}{'admissions':>12}{'hospitals':>10}"
        f"{'years':>12}{'LOS med (IQR) d':>24}"
    )
    for name, s in [*report["splits"].items(), ("all", report["all_splits"])]:
        years = s["admission_years"]
        los = s["los_days"]
        hospitals = "n/a" if s["n_hospitals"] is None else str(s["n_hospitals"])
        year_text = f"{years['min']}-{years['max']}" if years else "n/a"
        los_text = f"{los['median']} ({los['q1']}-{los['q3']})" if los else "n/a"
        lines.append(
            f"{name:<12}{s['n_subjects']!s:>12}{s['n_admissions']!s:>12}"
            f"{hospitals:>10}{year_text:>12}{los_text:>24}"
        )
    for name, s in report["splits"].items():
        if not s["events"]:
            continue
        lines.append(f"\n{name}: prevalence per subject")
        for event, e in s["events"].items():
            rate = e["prevalence_per_subject"]
            rate_text = "suppressed" if rate is None else f"{rate:.4f}"
            lines.append(
                f"  {event:<24}{e['n_subjects_positive']!s:>10}{rate_text:>12}"
            )
    return "\n".join(lines)


def _apply_run_config(args: argparse.Namespace) -> dict[str, Path]:
    """Fill unset options from a training run's ``config.json``.

    Returns the run's train/tuning split directories (those that exist)
    so the report describes the shards the run actually trained on.
    """
    if not args.run_dir:
        return {}
    config = json.loads((Path(args.run_dir) / "config.json").read_text())
    if args.source is None:
        args.source = config.get("source", "mimic_iv")
    if args.task_set is None:
        args.task_set = config.get("task_set", "v1")
    if not args.normalize_medications and config.get("normalize_medications"):
        args.normalize_medications = True
    splits: dict[str, Path] = {}
    for name, key in (("train", "train_shard_dir"), ("tuning", "tuning_shard_dir")):
        path = config.get(key)
        if path and Path(path).is_dir():
            splits[name] = Path(path)
        elif path:
            logger.warning("run config's %s %s does not exist; skipping", key, path)
    return splits


def _parse_splits(args: argparse.Namespace) -> dict[str, Path]:
    splits: dict[str, Path] = _apply_run_config(args)
    if args.meds_dir:
        root = Path(args.meds_dir) / "data"
        for name in DEFAULT_SPLITS:
            if (root / name).is_dir():
                splits[name] = root / name
    for spec in args.split or []:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--split expects NAME=DIR, got {spec!r}")
        splits[name] = Path(path)
    if not splits:
        raise SystemExit("no split directories: pass --meds-dir or --split NAME=DIR")
    return splits


def build_report(
    splits: dict[str, Path],
    *,
    source: str,
    task_set: str,
    normalize_medications: bool,
    with_events: bool,
    hospital_parquet: Path | None,
    max_shards: int | None,
) -> dict[str, Any]:
    """Compute the full cohort report over ``splits``."""
    alerts = alert_events_for(task_set, source=source) if with_events else None
    kept = {a.name for a in alerts} if alerts is not None else set()
    dropped = (
        [a.name for a in ALERT_TASK_SETS[task_set] if a.name not in kept]
        if alerts is not None
        else []
    )
    hospital_by_hadm = None
    if hospital_parquet is not None:
        hospital_by_hadm = pl.read_parquet(hospital_parquet).select(
            pl.col("hadm_id").cast(pl.Utf8),
            pl.col("hospital_num").cast(pl.Utf8),
        )
    accs: dict[str, SplitAccumulator] = {}
    for name, shard_dir in splits.items():
        logger.info("split %s: %s", name, shard_dir)
        accs[name] = summarize_split(
            shard_dir,
            source=source,
            task_set=task_set,
            alerts=alerts,
            normalize_medications=normalize_medications,
            hospital_by_hadm=hospital_by_hadm,
            max_shards=max_shards,
        )
    overall = SplitAccumulator()
    for acc in accs.values():
        overall.merge(acc)
    available = hospital_by_hadm is not None
    return {
        "source": source,
        "task_set": task_set,
        "normalize_medications": normalize_medications,
        "suppression_threshold": SUPPRESS_BELOW,
        "max_shards": max_shards,
        "los_definition": LOS_DEFINITION,
        "hospital_metadata": (hospital_parquet.name if hospital_parquet else None),
        "events": [a.name for a in alerts] if alerts else [],
        "events_dropped": dropped,
        "splits": {
            name: report_split(acc, alerts=alerts, hospitals_available=available)
            for name, acc in accs.items()
        },
        "all_splits": report_split(
            overall, alerts=alerts, hospitals_available=available
        ),
    }


def main() -> None:
    """Compute the suppressed cohort description and write it as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        default=None,
        help="training run directory; its config.json supplies --source, "
        "--task-set, --normalize-medications and the train/tuning split "
        "directories unless given explicitly",
    )
    parser.add_argument("--source", choices=sorted(ADMISSION_PREFIX), default=None)
    parser.add_argument("--task-set", default=None, choices=sorted(ALERT_TASK_SETS))
    parser.add_argument(
        "--meds-dir", default=None, help="MEDS root with data/<split>/ directories"
    )
    parser.add_argument(
        "--split",
        action="append",
        default=None,
        help="NAME=DIR; adds to or overrides the --meds-dir splits",
    )
    parser.add_argument(
        "--hadm-hospital-parquet",
        default=None,
        help="hadm_id -> hospital_num table (GEMINI metadata/hadm_id_hospital.parquet)",
    )
    parser.add_argument(
        "--normalize-medications",
        action="store_true",
        help="apply the run's medication normalizer before labeling (MIMIC/eICU runs)",
    )
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument(
        "--no-events", action="store_true", help="skip the per-event label pass"
    )
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    splits = _parse_splits(args)
    if args.source is None:
        raise SystemExit("--source is required without --run-dir")
    if args.task_set is None:
        args.task_set = "v3"

    report = build_report(
        splits,
        source=args.source,
        task_set=args.task_set,
        normalize_medications=args.normalize_medications,
        with_events=not args.no_events,
        hospital_parquet=(
            Path(args.hadm_hospital_parquet) if args.hadm_hospital_parquet else None
        ),
        max_shards=args.max_shards,
    )
    print(format_table(report))
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(report, f, indent=1)
    logger.info("wrote %s", args.output_json)


if __name__ == "__main__":
    main()
