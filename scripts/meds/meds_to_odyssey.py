"""Convert MEDS parquet shards to odyssey tokenizer format.

Reads MEDS-format sharded parquets produced by extract_mimic_iv.py and
converts them into per-patient token sequences expected by odyssey's
dataset classes (PretrainDatasetDecoder, FinetuneDatasetDecoder).

Output parquet columns per patient row:
    subject_id          int64
    event_tokens_{L}    list[str]   — ordered sequence of event codes
    type_tokens_{L}     list[int]   — per-token type ID (0-8)
    time_tokens_{L}     list[int]   — relative time bucket indices
    age_tokens_{L}      list[float] — patient age at each event (years)
    position_tokens_{L} list[int]   — visit-order positions
    visit_tokens_{L}    list[int]   — visit-segment IDs (0=out, 1=in, 2=ICU)

Where L is the max sequence length (default 2048).

Type ID mapping (mirrors ConceptTokenizer):
    0 = PAD
    1 = CLS / special
    2 = VS  (visit start)
    3 = VE  (visit end)
    4 = TIME_DELTA
    5 = LAB
    6 = MED
    7 = PROC / DIAG
    8 = REG / other

Usage
-----
python scripts/meds/meds_to_odyssey.py \\
    --meds_dir  data/meds \\
    --output_dir data/odyssey_tokenized \\
    --max_seq_len 2048 \\
    --split train
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import polars as pl


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MAX_SEQ_LEN = 2048

# ── type ID assignment ────────────────────────────────────────────────────────
PAD_TYPE = 0
SPECIAL_TYPE = 1  # CLS, BOS, EOS, VS, VE
VS_TYPE = 2
VE_TYPE = 3
TIME_TYPE = 4
LAB_TYPE = 5
MED_TYPE = 6
DIAG_TYPE = 7
OTHER_TYPE = 8


def code_to_type_id(code: str) -> int:
    """Map a MEDS code string to an odyssey type ID."""
    if code.startswith("LAB//"):
        return LAB_TYPE
    if code.startswith("MEDICATION//"):
        return MED_TYPE
    if code.startswith(("ICD", "DRG//")):
        return DIAG_TYPE
    if code in (
        "MEDS_BIRTH",
        "MEDS_DEATH",
        "ADMISSION//HOSPITAL",
        "DISCHARGE//HOSPITAL",
        "ADMISSION//ED",
        "ADMISSION//ICU",
        "DISCHARGE//ICU",
    ):
        return SPECIAL_TYPE
    if code.startswith(("GENDER//", "ETHNICITY//", "INSURANCE//")):
        return OTHER_TYPE
    return OTHER_TYPE


# ── time bucketing ────────────────────────────────────────────────────────────


def days_to_time_token(days: float) -> int:
    """Convert delta days to a discrete time bucket index.

    Buckets (matches DEFAULT_TIME_TOKENS order):
        [W_0..W_3]: 0-3 weeks  (0-6d, 7-13d, 14-20d, 21-27d)
        [M_0..M_12]: 0-12 months  (28d buckets)
        [LT]: >12 months
    """
    if days < 7:
        return 0
    if days < 14:
        return 1
    if days < 21:
        return 2
    if days < 28:
        return 3
    # months
    months = int(days / 30.44)
    if months <= 12:
        return 4 + months
    return 17  # LT


# ── per-patient sequence builder ───────────────────────────────────────────────


def build_patient_sequence(
    subject_id: int,
    events: pl.DataFrame,
    max_seq_len: int,
) -> dict[str, Any]:
    """Convert a single patient's MEDS events to odyssey token arrays."""
    # Sort by time (nulls = static events go first)
    events = events.sort("time", nulls_last=False)

    codes = events["code"].to_list()
    times = events["time"].to_list()  # datetime or None

    n = len(codes)
    event_tokens = []
    type_ids = []
    time_tokens = []
    ages = []
    positions = []
    visit_segs = []

    # Determine DOB from MEDS_BIRTH event for age computation
    dob = None
    for code, t in zip(codes, times):
        if code == "MEDS_BIRTH" and t is not None:
            dob = t
            break

    prev_time = None
    visit_order = 0
    visit_seg = 0  # 0=outpatient/unknown, 1=inpatient, 2=ICU

    for code, t in zip(codes, times):
        # Time delta token
        if t is not None and prev_time is not None:
            delta_days = max(0.0, (t - prev_time).total_seconds() / 86400.0)
            time_tok = days_to_time_token(delta_days)
        else:
            time_tok = 0

        # Age in years
        if dob is not None and t is not None:
            age_years = (t - dob).total_seconds() / (365.25 * 86400.0)
        else:
            age_years = 0.0

        # Visit segment update
        if code == "ADMISSION//HOSPITAL":
            visit_seg = 1
            visit_order += 1
        elif code == "ADMISSION//ICU":
            visit_seg = 2
        elif code in ("DISCHARGE//HOSPITAL", "DISCHARGE//ICU"):
            visit_seg = 0

        event_tokens.append(code)
        type_ids.append(code_to_type_id(code))
        time_tokens.append(time_tok)
        ages.append(round(age_years, 2))
        positions.append(visit_order)
        visit_segs.append(visit_seg)

        if t is not None:
            prev_time = t

    # Truncate to max_seq_len (keep most recent)
    if n > max_seq_len:
        event_tokens = event_tokens[-max_seq_len:]
        type_ids = type_ids[-max_seq_len:]
        time_tokens = time_tokens[-max_seq_len:]
        ages = ages[-max_seq_len:]
        positions = positions[-max_seq_len:]
        visit_segs = visit_segs[-max_seq_len:]

    return {
        "subject_id": subject_id,
        f"event_tokens_{max_seq_len}": event_tokens,
        f"type_tokens_{max_seq_len}": type_ids,
        f"time_tokens_{max_seq_len}": time_tokens,
        f"age_tokens_{max_seq_len}": ages,
        f"position_tokens_{max_seq_len}": positions,
        f"visit_tokens_{max_seq_len}": visit_segs,
    }


# ── shard processing ──────────────────────────────────────────────────────────


def process_shard(shard_path: Path, output_path: Path, max_seq_len: int) -> int:
    """Convert one MEDS shard parquet to odyssey format and write output."""
    df = pl.read_parquet(shard_path)

    rows = []
    for subject_id, group in df.group_by("subject_id"):
        row = build_patient_sequence(
            subject_id=int(subject_id[0]),  # type: ignore[call-overload]
            events=group,
            max_seq_len=max_seq_len,
        )
        rows.append(row)

    if not rows:
        return 0

    out_df = pl.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_parquet(output_path)
    return len(rows)


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Convert MEDS shards to odyssey tokenized format"
    )
    p.add_argument(
        "--meds_dir",
        required=True,
        help="Root MEDS directory (output of extract_mimic_iv.py)",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for odyssey-format parquets",
    )
    p.add_argument(
        "--max_seq_len", type=int, default=MAX_SEQ_LEN, help="Max sequence length"
    )
    p.add_argument(
        "--split",
        default="train",
        choices=["train", "tuning", "held_out", "all"],
        help="Which split to process",
    )
    return p.parse_args()


def main() -> None:
    """Run MEDS → odyssey conversion."""
    args = parse_args()
    meds_dir = Path(args.meds_dir)
    output_dir = Path(args.output_dir)
    splits = ["train", "tuning", "held_out"] if args.split == "all" else [args.split]

    for split in splits:
        shard_dir = meds_dir / "data" / split
        if not shard_dir.exists():
            log.warning("Split %s not found at %s — skipping", split, shard_dir)
            continue

        shard_files = sorted(shard_dir.glob("*.parquet"))
        log.info("Processing %d shards for split '%s' …", len(shard_files), split)
        total = 0

        for shard_path in shard_files:
            out_path = output_dir / split / shard_path.name
            n = process_shard(shard_path, out_path, args.max_seq_len)
            total += n
            log.info("  %s → %d patients", shard_path.name, n)

        log.info(
            "Split '%s': %d patients written to %s", split, total, output_dir / split
        )

    log.info("Done.")


if __name__ == "__main__":
    main()
