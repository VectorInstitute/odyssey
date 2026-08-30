"""Environment fingerprints and a numeric canary for reproducible results.

Entry 27 of the research journal found the same checkpoint scoring an alert
AUROC of 0.83 on one day and 0.91 on another: the mamba-ssm CUDA kernels had
been rebuilt in between, and nothing recorded which binary produced which
number. Two mechanisms prevent that class of silent drift:

- :func:`environment_fingerprint` captures what actually determines model
  outputs: git commit, python/torch/CUDA versions, GPU name and driver, the
  mamba-ssm version plus a content hash of its installed binaries, and (when
  given) content hashes of the checkpoint/vocabulary/binner files. Training
  writes it to the run dir; evaluation embeds it in every output JSON.
- :func:`numeric_canary` runs a fixed, seeded forward pass through the model
  and summarizes the logits. Training stores the canary next to the
  checkpoint; :func:`verify_run_provenance` recomputes it at load time and
  reports drift far larger than floating-point noise -- the exact failure
  entry 27 hit, caught at load instead of in a published table.

Both are best-effort observers: they never raise on missing optional pieces
(no GPU, no mamba-ssm, no git), recording ``None`` instead, and the canary
check WARNS rather than refusing to run, since a mismatch is sometimes the
point (e.g. deliberately comparing kernel builds).
"""

import hashlib
import json
import logging
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


logger = logging.getLogger(__name__)

CANARY_FILENAME = "numeric_canary.json"
FINGERPRINT_FILENAME = "env_fingerprint.json"
# Relative drift in logit statistics above this is a kernel/environment
# change, not floating-point noise (same-build reruns reproduce exactly on
# our hosts; the cross-build drift entry 27 measured was orders larger).
CANARY_RTOL = 1e-3


def _file_sha256(path: str | Path, *, max_bytes: int = 1 << 30) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
                if f.tell() > max_bytes:
                    break
        return h.hexdigest()
    except OSError:
        return None


def _mamba_ssm_info() -> dict[str, str | None]:
    try:
        import mamba_ssm  # noqa: PLC0415

        root = Path(mamba_ssm.__file__).parent
        h = hashlib.sha256()
        for so in sorted(root.rglob("*.so")):
            h.update(so.name.encode())
            digest = _file_sha256(so)
            if digest:
                h.update(digest.encode())
        return {
            "version": getattr(mamba_ssm, "__version__", None),
            "binaries_sha256": h.hexdigest(),
        }
    except ImportError:
        return {"version": None, "binaries_sha256": None}


def environment_fingerprint(
    artifact_paths: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Everything that determines model outputs, as one JSON-serializable dict."""
    try:
        commit = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            ).stdout.strip()
            or None
        )
    except OSError:
        commit = None
    gpu = None
    if torch.cuda.is_available():
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        }
    fp: dict[str, Any] = {
        "git_commit": commit,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": (
            torch.backends.cudnn.version()  # type: ignore[no-untyped-call]
            if torch.backends.cudnn.is_available()  # type: ignore[no-untyped-call]
            else None
        ),
        "gpu": gpu,
        "mamba_ssm": _mamba_ssm_info(),
    }
    if artifact_paths:
        fp["artifacts"] = {
            name: _file_sha256(path) for name, path in artifact_paths.items()
        }
    return fp


def numeric_canary(
    model: torch.nn.Module, vocab_size: int, *, device: str = "cpu"
) -> dict[str, Any]:
    """Summarize a fixed, seeded forward pass; identical env => identical values.

    Uses the model's ``forward_with_features`` on a small synthetic chunk
    (fixed generator seed, fixed shapes), in eval mode, and records robust
    statistics of the logits. Any kernel or weight change moves them.
    """
    from odyssey.data.types import (  # noqa: PLC0415
        AuxiliaryInputs,
        ClinicalSequenceBatch,
    )
    from odyssey.training.train import _move_chunk_to_device  # noqa: PLC0415

    g = torch.Generator().manual_seed(12345)
    lanes, t = 2, 64
    batch = ClinicalSequenceBatch(
        concept_ids=torch.randint(1, max(vocab_size, 2), (lanes, t), generator=g),
        aux=AuxiliaryInputs(
            type_ids=torch.randint(1, 8, (lanes, t), generator=g),
            time_stamps=torch.cumsum(torch.rand(lanes, t, generator=g), dim=1),
            ages=torch.full((lanes, t), 60.0),
            visit_orders=torch.zeros(lanes, t, dtype=torch.long),
            visit_segments=torch.ones(lanes, t, dtype=torch.long),
            values=torch.where(
                torch.rand(lanes, t, generator=g) < 0.5,
                torch.randn(lanes, t, generator=g),
                torch.full((lanes, t), float("nan")),
            ),
        ),
    )
    batch = _move_chunk_to_device(batch, device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = model.forward_with_features(  # type: ignore[operator]
            batch, state=None, reset_mask=None
        )
        logits = out.logits.float()
    if was_training:
        model.train()
    flat = logits.reshape(-1)
    return {
        "seed": 12345,
        "shape": list(logits.shape),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "absmax": float(flat.abs().max()),
        "first8": [round(float(v), 6) for v in flat[:8]],
    }


def check_canary(
    stored: dict[str, Any], fresh: dict[str, Any], *, rtol: float = CANARY_RTOL
) -> list[str]:
    """Return human-readable mismatch descriptions (empty = reproducible)."""
    problems = []
    for key in ("mean", "std", "absmax"):
        a, b = stored.get(key), fresh.get(key)
        if a is None or b is None:
            continue
        denom = max(abs(a), abs(b), 1e-8)
        if abs(a - b) / denom > rtol:
            problems.append(f"logit {key}: stored {a:.6g} vs current {b:.6g}")
    if stored.get("shape") != fresh.get("shape"):
        problems.append(f"logit shape: {stored.get('shape')} vs {fresh.get('shape')}")
    return problems


def write_run_provenance(
    run_dir: str | Path,
    model: torch.nn.Module,
    vocab_size: int,
    *,
    device: str = "cpu",
    checkpoint_name: str | None = None,
) -> None:
    """Record the environment fingerprint, and the canary for one checkpoint.

    Call once at run start with ``checkpoint_name=None`` (fingerprint only:
    the canary of a randomly initialized model is meaningless, the first
    wiring of this function got that wrong and every eval "failed" its
    canary) and again with the checkpoint name each time a checkpoint is
    saved, so the canary file maps checkpoint filename -> canary of the
    exact weights in it.
    """
    run_dir = Path(run_dir)
    (run_dir / FINGERPRINT_FILENAME).write_text(
        json.dumps(environment_fingerprint(), indent=2)
    )
    if checkpoint_name is None:
        return
    canary_path = run_dir / CANARY_FILENAME
    stored: dict[str, Any] = {}
    if canary_path.exists():
        stored = json.loads(canary_path.read_text())
        if "mean" in stored:  # legacy single-canary file (pre per-checkpoint)
            stored = {}
    stored[checkpoint_name] = numeric_canary(model, vocab_size, device=device)
    canary_path.write_text(json.dumps(stored, indent=2))


def verify_run_provenance(
    run_dir: str | Path,
    model: torch.nn.Module,
    vocab_size: int,
    *,
    device: str = "cpu",
    checkpoint_name: str | None = None,
) -> list[str]:
    """Recompute the canary against a run dir's stored one; log and return mismatches.

    Silent (empty result) when the run predates provenance files.
    """
    canary_path = Path(run_dir) / CANARY_FILENAME
    if not canary_path.exists():
        return []
    stored_all = json.loads(canary_path.read_text())
    if "mean" in stored_all:
        # Legacy single-canary file written at model construction (random
        # weights): meaningless to compare against; skip silently.
        return []
    stored = stored_all.get(checkpoint_name or "")
    if stored is None:
        return []
    fresh = numeric_canary(model, vocab_size, device=device)
    problems = check_canary(stored, fresh)
    for p in problems:
        logger.warning(
            "[provenance] numeric canary mismatch for %s: %s -- the environment "
            "(kernel build, torch, GPU) differs from the one this checkpoint was "
            "trained with; scores are NOT comparable to the run's published numbers",
            run_dir,
            p,
        )
    return problems
