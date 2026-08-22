#!/usr/bin/env python
"""GEMINI code inventory: every distinct code, suppressed counts.

Reads ``metadata/codes.parquet`` (written by ``finalize_meds.py``'s own
``_write_codes_parquet`` -- ``code``, ``count`` columns, one row per distinct
MEDS code across the whole finalized dataset) and writes
``scripts/gemini/out/codes_inventory.json``, a plain ``{code: suppressed_count}``
mapping, committed to git the same way ``extraction_summary.json`` and
``finalize_summary.json`` already are.

Exists to unblock the OMOP->LOINC concept-mapping work with the *complete*
code vocabulary rather than the dry-run schema report's top-N samples: code
strings themselves (e.g. ``LAB//3020564//umol/l``) are vocabulary metadata,
not patient data, so listing every one is fine. The only sensitive surface is
small counts -- a code appearing only a handful of times can itself be
identifying in combination with other information -- so counts are suppressed
the same way every other exported summary in this pipeline is (see
``extract_meds._suppressed``, duplicated here rather than imported, per the
module docstring on why GEMINI-facing scripts don't cross-import): rounded to
the nearest 1000, or floored to the literal string ``"<1000"`` below that.
Unlike ``extract_meds._suppressed``'s threshold of 6 (patient/row counts,
where even single digits can identify someone), 1000 is used here because the
sensitive unit is a *code's* frequency, not a *patient's* -- consistent with
the same suppress-small-counts principle, just calibrated to what's actually
being counted.

Every code is kept (even the suppressed ones) by default -- the point is an
exhaustive vocabulary, not just the frequent codes -- unless the resulting
file would exceed run.sh's own 900 KB commit-size cap, in which case the
``<1000``-count entries (the bulk of the file's *bytes* for a small fraction
of its *information*, since they carry no real count information anyway) are
dropped and the run is logged, not silently truncated.

Run on the GEMINI node, after `finalize` has completed:

    uv run python scripts/gemini/export_codes.py
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import polars as pl


logger = logging.getLogger(__name__)

#: Same convention as extract_meds.SUMMARY_PATH / finalize_meds.SUMMARY_PATH:
#: a small, suppressed file under the repo's own scripts/gemini/out/, so
#: run.sh's commit-and-push step rides it back to git.
CODES_INVENTORY_PATH = Path(__file__).resolve().parent / "out" / "codes_inventory.json"

#: Same default as extract_meds.OUTPUT_DIR / finalize_meds.OUTPUT_DIR --
#: duplicated, not imported, see the module docstring.
OUTPUT_DIR = Path(
    os.environ.get(
        "GEMINI_MEDS_OUTPUT_DIR",
        "/mnt/nfs/project/subdural_hematoma_endotypes/gemini_meds_v1",
    )
)

#: run.sh's own commit-size guard is MAX_BYTES=900_000 (scripts/gemini/run.sh).
#: This step's own budget sits under that with headroom, so a file that trips
#: the fallback here never reaches run.sh's harder guard at all.
_COMMIT_SIZE_BUDGET = 850_000

#: Threshold for this export specifically -- see the module docstring for why
#: it differs from extract_meds._suppressed's threshold of 6.
_SUPPRESS_BELOW = 1000


def _suppressed_code_count(n: int) -> str:
    """Round ``n`` to the nearest 1000, or floor small counts to ``"<1000"``.

    Same shape as ``extract_meds._suppressed``/``finalize_meds._suppressed``,
    duplicated with a different threshold -- see the module docstring.
    """
    if n < _SUPPRESS_BELOW:
        return f"<{_SUPPRESS_BELOW}"
    return str(round(n / 1000) * 1000)


def export_codes(output_dir: Optional[Path] = None) -> dict[str, Any]:
    """Read ``metadata/codes.parquet`` and write the suppressed code inventory.

    Parameters
    ----------
    output_dir : pathlib.Path, optional
        Overrides :data:`OUTPUT_DIR` (mainly for tests). Must already hold a
        completed `finalize` run's ``metadata/codes.parquet``.

    Returns
    -------
    dict[str, Any]
        Summary: total distinct codes, codes actually written, codes dropped
        for size (0 unless the size fallback triggered), and the written
        file's byte size.
    """
    root = output_dir if output_dir is not None else OUTPUT_DIR
    codes_path = root / "metadata" / "codes.parquet"
    if not codes_path.is_file():
        raise RuntimeError(
            f"{codes_path} not found -- run `finalize` to completion first"
        )

    frame = pl.read_parquet(codes_path).sort("code")
    counts = dict(zip(frame["code"].to_list(), frame["count"].to_list()))
    n_total = len(counts)

    suppressed = {code: _suppressed_code_count(n) for code, n in counts.items()}
    payload = json.dumps(suppressed, sort_keys=True, separators=(",", ":"))
    n_dropped = 0

    if len(payload.encode("utf-8")) > _COMMIT_SIZE_BUDGET:
        kept = {
            code: value
            for code, value in suppressed.items()
            if value != f"<{_SUPPRESS_BELOW}"
        }
        n_dropped = n_total - len(kept)
        logger.warning(
            "[export_codes] full inventory (%d codes) exceeds the %d-byte "
            "commit budget -- dropping the %d codes with count < %d (no real "
            "count information in those entries anyway) and keeping %d",
            n_total,
            _COMMIT_SIZE_BUDGET,
            n_dropped,
            _SUPPRESS_BELOW,
            len(kept),
        )
        suppressed = kept
        payload = json.dumps(suppressed, sort_keys=True, separators=(",", ":"))

    CODES_INVENTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    CODES_INVENTORY_PATH.write_text(payload + "\n")
    n_bytes = len((payload + "\n").encode("utf-8"))
    logger.info(
        "[export_codes] wrote %s (%d codes, %d bytes)",
        CODES_INVENTORY_PATH,
        len(suppressed),
        n_bytes,
    )

    return {
        "n_codes_total": n_total,
        "n_codes_written": len(suppressed),
        "n_codes_dropped_for_size": n_dropped,
        "n_bytes": n_bytes,
    }


def main() -> None:
    """Run the export and print where the inventory landed."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    summary = export_codes()
    print(f"Wrote {CODES_INVENTORY_PATH}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
