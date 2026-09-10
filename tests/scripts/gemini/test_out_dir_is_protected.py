"""The banked summaries survive a test run.

Regression guard for 2026-09-10, when a local ``pytest`` run replaced
the real GEMINI extraction summary (1,118,000 subjects, 19 tables, 1118
shards) with a smoke-scale one reading ``"<6"`` subjects and no tables.
Nothing failed; the only evidence was the file's mtime.
"""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "scripts" / "gemini" / "out"


def _load(name: str) -> ModuleType:
    # These scripts pull in the optional gemini extra at import time; the
    # unit-test job does not install it. The source-level guard below runs
    # everywhere and covers the same regression without importing anything.
    reason = "gemini extra not installed (uv sync --extra gemini)"
    for dependency in ("sqlalchemy", "pandas", "pyarrow", "polars"):
        pytest.importorskip(dependency, reason=reason)
    spec = importlib.util.spec_from_file_location(
        name, REPO / "scripts" / "gemini" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("module_name", "filename"),
    [
        ("extract_meds", "extraction_summary.json"),
        ("finalize_meds", "finalize_summary.json"),
    ],
)
def test_the_summary_path_points_outside_the_repo_under_test(
    module_name: str, filename: str
) -> None:
    module = _load(module_name)
    assert module.SUMMARY_PATH.name == filename
    assert not module.SUMMARY_PATH.is_relative_to(OUT), (
        f"{module_name}.SUMMARY_PATH still points into {OUT}; a test run would "
        "overwrite the banked GEMINI provenance with smoke-scale counts"
    )


@pytest.mark.parametrize(
    ("module_name", "filename"),
    [
        ("extract_meds", "extraction_summary.json"),
        ("finalize_meds", "finalize_summary.json"),
    ],
)
def test_without_the_override_it_points_at_the_banked_file(
    module_name: str, filename: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default must stay the tracked path: run.sh commits what lands there."""
    monkeypatch.delenv("GEMINI_OUT_DIR", raising=False)
    module = _load(module_name)
    assert OUT / filename == module.SUMMARY_PATH


@pytest.mark.skipif(
    not (OUT / "extraction_summary.json").exists(), reason="banked summary not present"
)
def test_the_banked_extraction_summary_still_holds_full_scale_counts() -> None:
    """A smoke-scale clobber is silent, so assert on the shape it destroys."""
    summary = json.loads((OUT / "extraction_summary.json").read_text())
    # _suppressed() returns a string: a real count, or "<6" when small
    # enough to be identifying. The smoke-scale clobber writes "<6".
    assert str(summary["n_subjects"]).isdigit(), summary["n_subjects"]
    assert int(summary["n_subjects"]) > 1_000_000
    assert int(summary["n_shards"]) > 1000
    assert len(summary["rows_per_table"]) > 10


@pytest.mark.parametrize("module_name", ["extract_meds", "finalize_meds"])
def test_the_path_is_built_from_an_overridable_out_dir(module_name: str) -> None:
    """Source-level twin of the two tests above, for environments without the extra.

    The unit-test job does not install the gemini extra, so the tests that
    import these modules skip there -- and this regression is exactly the
    kind that would then reach main unnoticed.
    """
    source = (REPO / "scripts" / "gemini" / f"{module_name}.py").read_text()
    assert 'os.environ.get("GEMINI_OUT_DIR")' in source
    assert "SUMMARY_PATH = OUT_DIR /" in source
    assert "SUMMARY_PATH = Path(__file__)" not in source
