"""Same-venv fit-result cache for optional-baseline fitting scripts.

Exists so a long optional-baseline fitting pass (TabICL/EBM/SurvivalPFN,
see :mod:`odyssey.inference.tabicl_baseline`/:mod:`odyssey.inference.ebm_baseline`/
:mod:`odyssey.inference.survivalpfn_baseline`, driven by
``scripts/rescore_extra_baselines.py``) never has to be redone from
scratch just because a LATER stage of the same run crashed. The incident
this exists for: EBM alone took ~4.6h fitting 12 (event, horizon) pairs
one night, then the run crashed at the *scoring* stage on an unrelated
bug, throwing that entire fit away -- see ``docs/reeval_wave_v2.md``.

Not a general ML experiment tracker: one ``{key}.pkl`` file per cache
key (``/`` in a key becomes a real subdirectory) plus an embedded
fingerprint per file, nothing else. Callers own the key namespace (see
each fit_*_baselines caller in the modules named above for the
``{model}/{event}/{horizon}h`` convention used there).

Pickles of fitted third-party model objects (a ``tabicl.TabICLClassifier``,
interpret's ``ExplainableBoostingClassifier``, a
``survivalpfn.SurvivalEstimator``) are same-venv artifacts, not portable
across machines or dependency versions -- unpickling one fit under a
different library version can silently misbehave rather than raise. Every
cached fit is stamped with an :func:`env_fingerprint` snapshot at save
time and compared against the current environment on load; any mismatch
means refit, not load, logged either way so it's visible in the run log
which path was taken.
"""

import logging
import pickle
import sys
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)

# Packages whose versions matter to the pickled objects this cache holds.
# Not exhaustive -- a genuinely thorough fingerprint would hash the whole
# environment -- just the libraries whose fitted objects actually get
# pickled here, plus torch/numpy since several of them wrap tensors/arrays
# from those.
_FINGERPRINT_PACKAGES = ("tabicl", "interpret", "survivalpfn", "torch", "numpy")


def _package_version(name: str) -> Optional[str]:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def env_fingerprint() -> Dict[str, Optional[str]]:
    """Return a dict identifying the current venv well enough to gate cache loads."""
    return {
        "python": sys.version,
        **{pkg: _package_version(pkg) for pkg in _FINGERPRINT_PACKAGES},
    }


@dataclass
class FitCache:
    """Pickles fitted baseline models to disk, keyed by a caller-chosen string."""

    cache_dir: Path
    fingerprint: Dict[str, Optional[str]] = field(default_factory=env_fingerprint)

    def _path(self, key: str) -> Path:
        # "/" in a key becomes a real subdirectory (pathlib splits it that
        # way automatically), not a flattened, escapable separator -- a
        # flattening scheme like key.replace("/", "__") would collide
        # "a/b" with the distinct key "a__b".
        return self.cache_dir / f"{key}.pkl"

    def load(self, key: str) -> Optional[Any]:
        """Return the cached model for ``key``, or ``None`` if it must be (re)fit.

        ``None`` covers both "never cached" and "cached under a different
        environment" -- the caller doesn't need to distinguish them, both
        mean fit now.
        """
        path = self._path(key)
        if not path.exists():
            logger.info("[fit-cache] %s: no cached fit at %s, will fit", key, path)
            return None
        with path.open("rb") as f:
            payload = pickle.load(f)  # noqa: S301
        if payload.get("fingerprint") != self.fingerprint:
            logger.info(
                "[fit-cache] %s: cached fit at %s has a different env "
                "fingerprint, refitting rather than trusting a cross-env pickle",
                key,
                path,
            )
            return None
        logger.info("[fit-cache] %s: loaded cached fit from %s", key, path)
        return payload["model"]

    def save(self, key: str, model: Any) -> None:
        """Pickle ``model`` to disk under ``key``, stamped with the fingerprint."""
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"fingerprint": self.fingerprint, "model": model}, f)
        logger.info("[fit-cache] %s: fit complete, cached to %s", key, path)


__all__ = ["FitCache", "env_fingerprint"]
