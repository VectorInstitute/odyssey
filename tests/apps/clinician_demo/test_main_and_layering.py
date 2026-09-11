"""The CLI, and the architecture rules the demo must keep."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from apps.clinician_demo.__main__ import parse_args


REPO = Path(__file__).resolve().parents[3]
APP = REPO / "apps" / "clinician_demo"
BANNED_IN_APP = {
    "odyssey.inference.steering",  # a mixture model cannot be steered; would overclaim
    "odyssey.inference.interventions",  # label overrides have the wrong sign on v10
    "odyssey.inference.rollouts",  # untested on the hybrid backbone
}
TORCH_FREE = [
    "config",
    "schemas",
    "server",
    "codebook",
    "thresholds",
    "showcase",
    "scorecard",
]


def test_defaults_and_path_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", "/home/me")
    config, self_check = parse_args(["--run-dir", "~/runs/r", "--data-dir", "~/d"])
    assert config.run_dir == Path("/home/me/runs/r") and config.data_dir == Path(
        "/home/me/d"
    )
    assert (config.data_mode, config.port, config.alert_rate, config.checkpoint) == (
        "credentialed",
        8765,
        0.05,
        "checkpoint_best.pt",
    )
    assert config.metadata_dir is None and config.splits_path is None and config.warmup
    assert not self_check


def test_every_flag_reaches_the_config() -> None:
    config, self_check = parse_args(
        [
            "--run-dir", "/r", "--data-dir", "/d", "--metadata-dir", "/m", "--splits", "/s.parquet",
            "--data-mode", "open", "--checkpoint", "checkpoint_final.pt", "--port", "9000",
            "--alert-rate", "0.1", "--max-shards", "2", "--cache-dir", "/c", "--device", "cpu",
            "--no-warmup", "--self-check",
        ]
    )  # fmt: skip
    assert (config.metadata_dir, config.splits_path, config.cache_dir) == (
        Path("/m"),
        Path("/s.parquet"),
        Path("/c"),
    )
    assert (config.data_mode, config.checkpoint, config.port, config.alert_rate) == (
        "open",
        "checkpoint_final.pt",
        9000,
        0.1,
    )
    assert (config.max_shards, config.device, config.warmup, self_check) == (
        2,
        "cpu",
        False,
        True,
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["--data-dir", "/d"],
        ["--run-dir", "/r", "--data-dir", "/d", "--data-mode", "public"],
        ["--run-dir", "/r", "--data-dir", "/d", "--alert-rate", "1.5"],
        ["--run-dir", "/r", "--data-dir", "/d", "--max-shards", "0"],
        ["--run-dir", "/r", "--data-dir", "/d", "--port", "99999"],
    ],
)
def test_bad_arguments_exit_before_loading_anything(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        parse_args(argv)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def test_the_research_package_never_imports_the_app() -> None:
    offenders = [
        str(p.relative_to(REPO))
        for p in (REPO / "odyssey").rglob("*.py")
        if any(m == "apps" or m.startswith("apps.") for m in _imports(p))
    ]
    assert offenders == []


def test_the_app_never_imports_what_it_must_not_show() -> None:
    offenders = {
        str(p.relative_to(REPO)): sorted(_imports(p) & BANNED_IN_APP)
        for p in APP.rglob("*.py")
        if _imports(p) & BANNED_IN_APP
    }
    assert offenders == {}


@pytest.mark.parametrize("module", TORCH_FREE)
def test_light_modules_import_without_torch(module: str) -> None:
    code = f"import sys, apps.clinician_demo.{module}; print('torch' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "False", f"{module} pulls in torch"
