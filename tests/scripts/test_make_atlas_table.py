"""Atlas tables shorten MEDS names safely and rank unnamed concepts by activation."""

import importlib.util
from pathlib import Path


def _load():
    spec = importlib.util.spec_from_file_location(
        "make_atlas_table", Path("scripts/make_atlas_table.py")
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_short_name_strips_units_and_escapes() -> None:
    mod = _load()
    assert (
        mod.short_name("Cholesterol in HDL [Mass/volume] in Serum or Plasma (Q3)")
        == "Cholesterol in HDL (Q3)"
    )
    assert (
        mod.short_name("MEDICATION//norepinephrine//Administered") == "norepinephrine"
    )
    assert mod.short_name("STOP//calcium acetate") == "STOP calcium acetate"
    assert mod.short_name("A & B 50% #1") == "A \\& B 50\\% \\#1"
    assert mod.short_name("x" * 60).endswith("…")


def test_tables_render_rows() -> None:
    mod = _load()
    atlas = {
        "run_dir": "r",
        "n_positions": 10,
        "contribution_share": {"named": 0.1, "unknown": 0.7, "residual": 0.2},
        "known": [
            {
                "name": "shock",
                "norm": 0.7,
                "mean_activation": 0.3,
                "promotes": [{"name": "Heart Rhythm", "shift": 1.0}],
            }
        ],
        "unknown": [
            {
                "name": "unknown_3",
                "norm": 8.0,
                "mean_activation": 0.5,
                "promotes": [
                    {"name": "Calcium [Mass/volume] in Serum or Plasma", "shift": 1.0}
                ],
            }
        ],
    }
    u = mod.unknown_table(atlas, top_concepts=5, top_events=3)
    k = mod.known_table(atlas, top_events=3)
    assert "unknown 3 & 0.50 & Calcium \\\\" in u
    assert "shock & 0.70 & 0.30 & Heart Rhythm \\\\" in k
