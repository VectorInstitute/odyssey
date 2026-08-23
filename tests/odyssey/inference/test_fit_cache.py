"""Tests for the fit-result cache used by the optional baseline fitters."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pytest

from odyssey.inference.fit_cache import FitCache, env_fingerprint


def test_load_returns_none_when_nothing_cached(tmp_path: Path) -> None:
    cache = FitCache(cache_dir=tmp_path)
    assert cache.load("tabicl/vasopressor_start/8h") is None


def test_save_then_load_round_trips_the_model(tmp_path: Path) -> None:
    cache = FitCache(cache_dir=tmp_path)
    cache.save("ebm/death/24h", {"weights": [1, 2, 3]})
    assert cache.load("ebm/death/24h") == {"weights": [1, 2, 3]}


def test_load_returns_none_when_the_fingerprint_does_not_match(
    tmp_path: Path,
) -> None:
    writer = FitCache(cache_dir=tmp_path, fingerprint={"tabicl": "1.0.0"})
    writer.save("tabicl/death/8h", "a-fitted-model")

    reader = FitCache(cache_dir=tmp_path, fingerprint={"tabicl": "2.0.0"})
    assert reader.load("tabicl/death/8h") is None


def test_load_returns_the_model_when_the_fingerprint_matches(tmp_path: Path) -> None:
    fp: Dict[str, Optional[str]] = {"tabicl": "1.0.0"}
    writer = FitCache(cache_dir=tmp_path, fingerprint=dict(fp))
    writer.save("tabicl/death/8h", "a-fitted-model")

    reader = FitCache(cache_dir=tmp_path, fingerprint=dict(fp))
    assert reader.load("tabicl/death/8h") == "a-fitted-model"


def test_keys_that_share_a_prefix_do_not_collide(tmp_path: Path) -> None:
    cache = FitCache(cache_dir=tmp_path)
    cache.save("tabicl/vasopressor_start/8h", "model-a")
    cache.save("tabicl/vasopressor_start/24h", "model-b")
    assert cache.load("tabicl/vasopressor_start/8h") == "model-a"
    assert cache.load("tabicl/vasopressor_start/24h") == "model-b"


def test_a_slash_in_a_key_becomes_a_real_subdirectory(tmp_path: Path) -> None:
    cache = FitCache(cache_dir=tmp_path)
    cache.save("ebm/death/8h", "model")
    assert (tmp_path / "ebm" / "death" / "8h.pkl").exists()


def test_save_creates_the_cache_dir_if_missing(tmp_path: Path) -> None:
    cache_dir = tmp_path / "nested" / "rescore_cache"
    cache = FitCache(cache_dir=cache_dir)
    cache.save("ebm/death/8h", "model")
    assert cache_dir.exists()
    assert cache.load("ebm/death/8h") == "model"


def test_save_does_not_raise_when_the_model_is_unpicklable(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Regression test for a real incident.

    A survivalpfn.SurvivalEstimator instance holds a lambda as an
    attribute (from its own InContextModel.__init__), which stdlib
    pickle cannot serialize (only module-level, importable-by-reference
    callables survive it) -- this raised AttributeError deep inside
    pickle.dump and crashed the whole fitting script, discarding a fit
    that had already completed successfully. save() must fail soft:
    caching is a pure optimization applied after the fit is already done
    and usable, not a correctness requirement.
    """

    class _HasUnpicklableAttr:
        def __init__(self) -> None:
            self.fn = lambda x: x  # noqa: E731 -- the exact shape being regression-tested

    cache = FitCache(cache_dir=tmp_path)
    with caplog.at_level(logging.WARNING):
        cache.save("survivalpfn/death/8h", _HasUnpicklableAttr())  # must not raise

    assert any(
        "could not pickle" in r.message
        for r in caplog.records
        if r.levelname == "WARNING"
    )


def test_save_leaves_no_truncated_file_when_pickling_fails(tmp_path: Path) -> None:
    class _Unpicklable:
        def __init__(self) -> None:
            self.fn = lambda x: x  # noqa: E731

    cache = FitCache(cache_dir=tmp_path)
    cache.save("ebm/death/8h", _Unpicklable())
    assert not (tmp_path / "ebm" / "death" / "8h.pkl").exists()


def test_save_failure_for_one_key_does_not_affect_another(tmp_path: Path) -> None:
    class _Unpicklable:
        def __init__(self) -> None:
            self.fn = lambda x: x  # noqa: E731

    cache = FitCache(cache_dir=tmp_path)
    cache.save("tabicl/death/8h", _Unpicklable())
    cache.save("tabicl/death/24h", "a-fitted-model")
    assert cache.load("tabicl/death/8h") is None  # never cached, not corrupted
    assert cache.load("tabicl/death/24h") == "a-fitted-model"


def test_env_fingerprint_includes_python_version() -> None:
    fp = env_fingerprint()
    assert "python" in fp
    assert fp["python"]


def test_env_fingerprint_reports_none_for_an_uninstalled_package() -> None:
    fp = env_fingerprint()
    # none of tabicl/interpret/survivalpfn/torch/numpy are guaranteed
    # installed in every environment this test runs in; whichever aren't
    # must report None rather than raise.
    for pkg in ("tabicl", "interpret", "survivalpfn", "torch", "numpy"):
        assert pkg in fp


@dataclass
class _Fit:
    """A stand-in fitted model carrying only the attribute the cache checks."""

    feature_set: str


def test_load_for_feature_set_treats_a_mismatch_as_a_miss(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A fit is bound to its feature matrix; a mismatched cache entry is refit.

    The key already encodes the feature set, but layouts change and caches
    get migrated by hand (both happened on 2026-08-23), so the cached
    object's own feature_set is checked too.
    """
    import logging  # noqa: PLC0415

    cache = FitCache(cache_dir=tmp_path)
    cache.save("ebm/strong/death/8h", _Fit("strong"))
    assert cache.load_for_feature_set("ebm/strong/death/8h", "strong") == _Fit("strong")
    with caplog.at_level(logging.WARNING):
        assert cache.load_for_feature_set("ebm/strong/death/8h", "basic") is None
    assert any("cached fit is on" in r.message for r in caplog.records)
    # an object without the attribute is still usable (older entries)
    cache.save("ebm/strong/death/24h", "opaque-model")
    assert cache.load_for_feature_set("ebm/strong/death/24h", "basic") == "opaque-model"
    assert cache.load_for_feature_set("ebm/strong/nope/8h", "strong") is None
