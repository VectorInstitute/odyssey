"""Tests for the missingness stress protocol's degraded-shard generator.

docs/missingness_protocol.md. Pins the properties the protocol depends on:
determinism by seed, anchor preservation (axis A), exactly-one-family
removal (axis B), and the axis-C time shift's origin-preservation guarantee
-- plus the CLI end to end.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import polars as pl

from odyssey.data.degrade import (
    METADATA_FILENAME,
    Cell,
    all_cells,
    apply_family_blackout,
    apply_lab_lag,
    apply_mcar,
    generate_cell,
    load_cell_metadata,
    main,
)


T0 = datetime(2024, 1, 1)

_SCHEMA = {
    "subject_id": pl.Int64,
    "code": pl.Utf8,
    "time": pl.Datetime,
    "numeric_value": pl.Float32,
    "hadm_id": pl.Int64,
}


def _shard(rows: List[Tuple[int, str, datetime, Optional[float], int]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=_SCHEMA, orient="row")


def _rich_subject(
    sid: int, hadm: int
) -> List[Tuple[int, str, datetime, Optional[float], int]]:
    """One subject with an anchor, vitals, real labs, and a medication."""
    return [
        (sid, "ICU_ADMISSION//MICU", T0, None, hadm),
        (sid, "LAB//220045//bpm", T0 + timedelta(hours=1), 80.0, hadm),  # vital
        (sid, "LAB//RESULT//50912//", T0 + timedelta(hours=2), 1.1, hadm),  # real lab
        (
            sid,
            "MEDICATION//norepinephrine//Administered",
            T0 + timedelta(hours=3),
            None,
            hadm,
        ),
        (sid, "HOSPITAL_DISCHARGE//", T0 + timedelta(hours=10), None, hadm),
    ]


def _rich_shard(subject_ids: List[int]) -> pl.DataFrame:
    rows: List[Tuple[int, str, datetime, Optional[float], int]] = []
    for sid in subject_ids:
        rows.extend(_rich_subject(sid, 1000 + sid))
    return _shard(rows)


# ---------------------------------------------------------------------------
# Axis A: MCAR dropout
# ---------------------------------------------------------------------------


def test_mcar_determinism_by_seed() -> None:
    events = _rich_shard([1, 2, 3, 4, 5])
    out1 = apply_mcar(events, p=0.5, seed=42)
    out2 = apply_mcar(events, p=0.5, seed=42)
    assert out1.equals(out2)


def test_mcar_different_seeds_usually_differ() -> None:
    events = _rich_shard(list(range(1, 40)))
    out1 = apply_mcar(events, p=0.5, seed=1)
    out2 = apply_mcar(events, p=0.5, seed=2)
    assert not out1.equals(out2)


def test_mcar_anchor_rows_never_dropped() -> None:
    events = _rich_shard(list(range(1, 60)))
    out = apply_mcar(events, p=1.0, seed=0)  # everything eligible IS dropped
    codes = set(out["code"].to_list())
    assert "ICU_ADMISSION//MICU" in codes
    assert "HOSPITAL_DISCHARGE//" in codes
    # nothing non-anchor survives at p=1.0
    assert "LAB//220045//bpm" not in codes
    assert "LAB//RESULT//50912//" not in codes
    assert "MEDICATION//norepinephrine//Administered" not in codes


def test_mcar_p0_keeps_everything() -> None:
    events = _rich_shard([1, 2, 3])
    out = apply_mcar(events, p=0.0, seed=0)
    assert out.height == events.height


def test_mcar_rejects_invalid_probability() -> None:
    events = _rich_shard([1])
    try:
        apply_mcar(events, p=1.5, seed=0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Axis B: family blackout
# ---------------------------------------------------------------------------


def test_family_blackout_removes_exactly_one_family() -> None:
    events = _rich_shard([1, 2, 3])
    out = apply_family_blackout(events, family="labs", source="mimic_iv")
    codes = set(out["code"].to_list())
    assert "LAB//RESULT//50912//" not in codes  # removed
    assert "LAB//220045//bpm" in codes  # vitals survive
    assert "MEDICATION//norepinephrine//Administered" in codes  # meds survive
    assert "ICU_ADMISSION//MICU" in codes  # anchors survive
    # row count dropped by exactly the number of lab rows
    assert out.height == events.height - 3  # one lab row per subject, 3 subjects


def test_family_blackout_unknown_family_raises() -> None:
    events = _rich_shard([1])
    try:
        apply_family_blackout(events, family="nope", source="mimic_iv")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Axis C: lab lag (static time shift)
# ---------------------------------------------------------------------------


def test_lab_lag_shifts_only_lab_rows() -> None:
    events = _rich_shard([1])
    out = apply_lab_lag(events, lag_hours=4.0, source="mimic_iv").sort("time")
    by_code = dict(zip(out["code"].to_list(), out["time"].to_list()))
    assert by_code["LAB//RESULT//50912//"] == T0 + timedelta(hours=2 + 4)
    # vitals, meds, anchors: untouched
    assert by_code["LAB//220045//bpm"] == T0 + timedelta(hours=1)
    assert by_code["MEDICATION//norepinephrine//Administered"] == T0 + timedelta(
        hours=3
    )
    assert by_code["ICU_ADMISSION//MICU"] == T0
    assert by_code["HOSPITAL_DISCHARGE//"] == T0 + timedelta(hours=10)


def test_lab_lag_never_moves_a_subjects_time_origin() -> None:
    # subject 2's very first timed event IS a real lab -- must be exempted.
    rows = [
        (2, "LAB//RESULT//50912//", T0, 0.9, 2000),
        (2, "HOSPITAL_DISCHARGE//", T0 + timedelta(hours=5), None, 2000),
    ]
    events = _shard(rows)
    out = apply_lab_lag(events, lag_hours=8.0, source="mimic_iv")
    by_code = dict(zip(out["code"].to_list(), out["time"].to_list()))
    assert by_code["LAB//RESULT//50912//"] == T0  # exempted, not shifted


def test_lab_lag_correctness_at_a_landmark() -> None:
    """A lab drawn within lag_hours of a landmark is not yet 'visible' there.

    No dynamic filter needed (the design's whole point): once the shard is
    shifted, any consumer's existing "events strictly before index time"
    rule already treats the shifted lab as absent at an early landmark and
    present at a later one, with no lag-specific code.
    """
    events = _shard(
        [
            (1, "ICU_ADMISSION//MICU", T0, None, 1000),
            (1, "LAB//RESULT//50912//", T0 + timedelta(hours=2), 1.1, 1000),
        ]
    )
    out = apply_lab_lag(events, lag_hours=4.0, source="mimic_iv")
    lab_time = out.filter(pl.col("code") == "LAB//RESULT//50912//")["time"][0]
    landmark_before = T0 + timedelta(hours=5)  # 2 + 4 = 6: still invisible
    landmark_after = T0 + timedelta(hours=7)  # now visible
    assert lab_time > landmark_before
    assert lab_time <= landmark_after


def test_lab_lag_rejects_negative_lag() -> None:
    events = _rich_shard([1])
    try:
        apply_lab_lag(events, lag_hours=-1.0, source="mimic_iv")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Cell generation / metadata / row-set invariance
# ---------------------------------------------------------------------------


def test_all_cells_is_the_documented_eight_cell_grid() -> None:
    cells = all_cells(seed=7)
    assert len(cells) == 8
    assert all(c.seed == 7 for c in cells.values())
    transforms = {c.transform for c in cells.values()}
    assert transforms == {"mcar", "family_blackout", "lab_lag"}


def test_generate_cell_writes_shard_and_metadata(tmp_path: Path) -> None:
    shard_dir = tmp_path / "clean"
    shard_dir.mkdir()
    events = _rich_shard([1, 2, 3])
    events.write_parquet(shard_dir / "0.parquet")
    shard_paths = [shard_dir / "0.parquet"]

    out_dir = tmp_path / "out" / "blackout_labs"
    cell = Cell(
        name="blackout_labs",
        transform="family_blackout",
        seed=0,
        params={"family": "labs"},
    )
    generate_cell(cell, shard_paths, out_dir, source="mimic_iv")

    assert (out_dir / "0.parquet").is_file()
    assert (out_dir / METADATA_FILENAME).is_file()
    meta = load_cell_metadata(out_dir)
    assert meta["cell"] == "blackout_labs"
    assert meta["transform"] == "family_blackout"
    assert meta["params"] == {"family": "labs"}
    assert "0.parquet" in meta["source_shard_hashes"]

    written = pl.read_parquet(out_dir / "0.parquet")
    assert "LAB//RESULT//50912//" not in written["code"].to_list()


def test_row_set_invariance_across_cells_for_the_landmark_grid(tmp_path: Path) -> None:
    """Anchor rows survive identically across every cell.

    The row set that matters for scoring (subject_id, hadm_id, code prefix
    surviving as an ANCHOR) is identical across every cell for the anchor
    rows that define the visit envelope -- exactly what lets landmark rows
    stay clean and shared while features differ per cell (Principle 3).
    """
    shard_dir = tmp_path / "clean"
    shard_dir.mkdir()
    events = _rich_shard([1, 2, 3, 4, 5])
    events.write_parquet(shard_dir / "0.parquet")
    shard_paths = [shard_dir / "0.parquet"]

    cells = all_cells(seed=0)
    anchor_sets = {}
    for name, cell in cells.items():
        out_dir = tmp_path / "out" / name
        generate_cell(cell, shard_paths, out_dir, source="mimic_iv")
        degraded = pl.read_parquet(out_dir / "0.parquet")
        anchors = degraded.filter(
            pl.col("code").is_in(["ICU_ADMISSION//MICU", "HOSPITAL_DISCHARGE//"])
        )
        anchor_sets[name] = set(
            zip(anchors["subject_id"].to_list(), anchors["code"].to_list())
        )

    clean_anchors = {
        (sid, code)
        for sid in [1, 2, 3, 4, 5]
        for code in ["ICU_ADMISSION//MICU", "HOSPITAL_DISCHARGE//"]
    }
    for name, anchors in anchor_sets.items():
        assert anchors == clean_anchors, f"cell {name} lost/changed an anchor row"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_all_cells_end_to_end(tmp_path: Path) -> None:
    shard_dir = tmp_path / "held_out"
    shard_dir.mkdir()
    _rich_shard([1, 2, 3]).write_parquet(shard_dir / "0.parquet")
    _rich_shard([4, 5, 6]).write_parquet(shard_dir / "1.parquet")
    output_root = tmp_path / "degraded"

    main(
        [
            "--held-out-shard-dir",
            str(shard_dir),
            "--output-root",
            str(output_root),
            "--cells",
            "all",
            "--seed",
            "3",
            "--source",
            "mimic_iv",
        ]
    )

    cell_dirs = sorted(p.name for p in output_root.iterdir())
    assert len(cell_dirs) == 8
    for name in cell_dirs:
        cell_dir = output_root / name
        assert (cell_dir / "0.parquet").is_file()
        assert (cell_dir / "1.parquet").is_file()
        meta = json.loads((cell_dir / METADATA_FILENAME).read_text())
        assert meta["seed"] == 3
        assert set(meta["source_shard_hashes"]) == {"0.parquet", "1.parquet"}


def test_cli_selected_cells_only(tmp_path: Path) -> None:
    shard_dir = tmp_path / "held_out"
    shard_dir.mkdir()
    _rich_shard([1]).write_parquet(shard_dir / "0.parquet")
    output_root = tmp_path / "degraded"

    main(
        [
            "--held-out-shard-dir",
            str(shard_dir),
            "--output-root",
            str(output_root),
            "--cells",
            "blackout_labs",
            "lag_4h",
        ]
    )

    assert sorted(p.name for p in output_root.iterdir()) == ["blackout_labs", "lag_4h"]


def test_cli_unknown_cell_name_raises(tmp_path: Path) -> None:
    shard_dir = tmp_path / "held_out"
    shard_dir.mkdir()
    _rich_shard([1]).write_parquet(shard_dir / "0.parquet")
    try:
        main(
            [
                "--held-out-shard-dir",
                str(shard_dir),
                "--output-root",
                str(tmp_path / "out"),
                "--cells",
                "not_a_real_cell",
            ]
        )
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
