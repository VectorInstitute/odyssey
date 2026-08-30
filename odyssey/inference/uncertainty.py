"""Subject-clustered bootstrap confidence intervals for AUROC.

Every model/baseline AUROC this project reports today is a bare point
estimate on one held-out draw. That understates how much a reported
"hazard beats GBM by 0.02"-style claim should be trusted: the tuned GBM is
a CONTROL in the value-tail-transform arms (it never reads the value
channel and scores the same held-out shards in every arm), yet its own
AUROC on the same cell moves by up to ~2.9 percentage points across arms
(measured on real data, 2026-08-24: death@8h 0.9452/0.9518/0.9229 across
three arms). Every point-estimate comparison in the registry has been read
against a number that moves by about that much on its own.

TWO SOURCES OF SPREAD, NOT ONE -- the distinction this module exists to
keep visible, not conflate:

- FINITE-SAMPLE VARIANCE: the held-out split is one draw from the
  population; a cell with 222 positives out of 136,850 rows carries real
  sampling error no matter how good the models are. This is what
  :func:`bootstrap_auroc` measures.
- REFIT VARIANCE: different fits of the SAME model (different train-time
  randomness, different upstream floating-point paths, ...) score
  differently even on identical held-out data -- the GBM spread above is
  this, not finite-sample variance. NO bootstrap of one fitted model's
  predictions can see this; measuring it needs k independent refits with
  different seeds, which is out of scope here.

A bootstrap interval from this module answers "how much would this AUROC
plausibly move if we drew a different held-out sample from the same
population, holding the fitted model fixed" -- it does NOT answer "how
much would this AUROC move on a refit". Reporting a bootstrap interval as
"the" uncertainty on a comparison between two independently-fit models
(hazard head vs. GBM, arm vs. arm) understates the total spread, because it
omits refit variance entirely. Any number built from this module must say
which of the two sources it carries.

Not yet wired into :class:`~odyssey.inference.alerts.AlertMetrics` or the
alerts pipeline, and does not change any existing reported number --
landed as a tested, standalone unit first.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class BootstrapAUROC:
    """A subject-clustered bootstrap AUROC summary.

    See module docstring for what this measures (finite-sample variance)
    and what it does not (refit variance).
    """

    point_estimate: float
    """AUROC on the data exactly as observed, no resampling."""
    mean: Optional[float]
    """Mean AUROC across usable bootstrap resamples. None if every
    resample was skipped (see ``n_skipped``)."""
    std: Optional[float]
    """Sample std (ddof=1) across usable resamples. None if fewer than 2
    resamples were usable -- a std from 0 or 1 points is not meaningful."""
    ci_low: Optional[float]
    """Lower percentile bound (``100 * alpha / 2``). None under the same
    condition as ``mean``."""
    ci_high: Optional[float]
    """Upper percentile bound (``100 * (1 - alpha / 2)``). None under the
    same condition as ``mean``."""
    n_boot_used: int
    """Resamples that produced a usable AUROC."""
    n_boot_skipped: int
    """Resamples discarded because the resampled ``y`` was single-class
    (AUROC undefined) -- see requirement 3 in the module docstring. A
    large count here means the interval rests on fewer resamples than
    requested and the cell is sparse; it must stay visible, not get
    silently absorbed into a narrower-looking ``n_boot``."""


def bootstrap_auroc(
    y: Union[np.ndarray, Sequence[float]],
    p: Union[np.ndarray, Sequence[float]],
    subject_ids: Union[np.ndarray, Sequence[int]],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Optional[BootstrapAUROC]:
    """Subject-clustered bootstrap AUROC: point estimate, mean/std/CI, skip count.

    Returns None if the OBSERVED ``y`` is single-class -- AUROC is
    undefined for the cell itself, not just for some resamples, so the
    caller should report the cell unscoreable rather than substitute 0.5
    (the same discipline
    :class:`~odyssey.inference.survivalpfn_baseline.SurvivalPFNBaselineModel`'s
    degeneracy warning already follows for a different kind of undefined
    AUROC).

    RESAMPLES SUBJECTS, NOT ROWS -- the one requirement that makes or
    breaks this function. Landmark rows are many-per-subject and heavily
    correlated (the same patient contributes a row every few hours with
    nearly identical features and the same outcome); resampling rows
    would treat those as independent draws and produce intervals far too
    narrow to trust. Each resample draws ``n_subjects`` subjects WITH
    REPLACEMENT and takes every one of a drawn subject's rows, whole.

    A resample whose drawn ``y`` is single-class makes AUROC undefined for
    that resample specifically; it is skipped and counted
    (``n_boot_skipped``), never silently dropped -- see
    :class:`BootstrapAUROC`.

    Rows are grouped by subject ONCE up front (an O(n log n) sort), and
    each resample's row selection is a pure index gather via
    :func:`numpy.repeat`/cumulative-offset arithmetic -- no per-subject
    Python loop and no dataframe operation inside the resample loop.

    ``p`` is ALSO sorted and grouped into tied-value buckets once, up
    front, rather than re-sorted inside the loop: a first version called
    :func:`sklearn.metrics.roc_auc_score` per resample, which re-sorts the
    resampled array every time and measured ~16ms/call on a
    140,000-row cell -- 1000 resamples x 12 cells landed at several
    minutes, failing the seconds-not-minutes bar. Each resample instead
    computes a WEIGHTED Mann-Whitney U statistic against the one
    precomputed sort: a resample only changes how many times each
    ORIGINAL row is duplicated (its weight, from
    :func:`numpy.bincount` on the resampled row indices), never the
    relative order of two distinct row values, so the average-rank-per-tie
    bucket only needs computing once and each resample becomes a handful
    of O(n) weighted sums instead of a fresh O(n log n) sort. Verified
    against :func:`sklearn.metrics.roc_auc_score` directly (see the
    module's tests) rather than trusted by derivation alone. Measured
    speedup: ~7x on the case above, comfortably seconds for 1000 x 12.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    p_arr = np.asarray(p, dtype=np.float64)
    subject_arr = np.asarray(subject_ids)
    if not (len(y_arr) == len(p_arr) == len(subject_arr)):
        raise ValueError(
            "bootstrap_auroc: y, p, and subject_ids must describe the same "
            f"rows; got lengths {len(y_arr)}, {len(p_arr)}, {len(subject_arr)}"
        )
    n_rows = len(y_arr)
    if len(np.unique(y_arr)) < 2:
        return None

    point_estimate = float(roc_auc_score(y_arr, p_arr))

    # Group rows by subject once: `order` sorted by subject, `boundaries`
    # the start offset of each subject's run within `order`. Every
    # resample below is then `order[boundaries[s]:boundaries[s+1]]` per
    # drawn subject s, vectorized rather than looped.
    unique_subjects, inverse = np.unique(subject_arr, return_inverse=True)
    n_subjects = len(unique_subjects)
    subj_order = np.argsort(inverse, kind="stable")
    subj_counts = np.bincount(inverse, minlength=n_subjects)
    subj_boundaries = np.concatenate([[0], np.cumsum(subj_counts)])

    # Group rows by tied p-value once: `p_group_of_row[i]` is which
    # ascending-p tie-bucket original row i falls into.
    p_group_of_row, n_groups = _group_p_ties(p_arr)

    rng = np.random.default_rng(seed)
    scores = []
    n_skipped = 0
    for _ in range(n_boot):
        drawn = rng.integers(0, n_subjects, size=n_subjects)
        row_idx = _gather_rows_for_drawn_subjects(
            drawn, subj_boundaries, subj_order, subj_counts
        )
        auc = _weighted_auroc(row_idx, y_arr, p_group_of_row, n_rows, n_groups)
        if auc is None:
            n_skipped += 1
            continue
        scores.append(auc)

    n_used = len(scores)
    if n_used == 0:
        return BootstrapAUROC(point_estimate, None, None, None, None, 0, n_boot)

    scores_arr = np.array(scores)
    mean = float(scores_arr.mean())
    std = float(scores_arr.std(ddof=1)) if n_used > 1 else None
    ci_low = float(np.percentile(scores_arr, 100 * alpha / 2))
    ci_high = float(np.percentile(scores_arr, 100 * (1 - alpha / 2)))
    return BootstrapAUROC(point_estimate, mean, std, ci_low, ci_high, n_used, n_skipped)


@dataclass(frozen=True)
class BootstrapAUROCDelta:
    """A subject-clustered PAIRED bootstrap of an AUROC difference.

    For two scorers evaluated on the SAME rows, the right question is
    "does scorer A beat scorer B on this sample", and the right interval
    is on the per-resample DIFFERENCE ``auroc_a - auroc_b`` -- the two
    scores are highly correlated across resamples (same rows, same
    outcomes), so comparing two independently-bootstrapped intervals for
    overlap both understates power (overlapping CIs do not establish "no
    difference") and ignores the pairing entirely. Carries the same
    finite-sample-variance-only caveat as :class:`BootstrapAUROC`.
    """

    point_estimate: float
    """``auroc_a - auroc_b`` on the data exactly as observed."""
    mean: Optional[float]
    std: Optional[float]
    ci_low: Optional[float]
    ci_high: Optional[float]
    n_boot_used: int
    n_boot_skipped: int

    def excludes_zero(self) -> Optional[bool]:
        """Whether the CI excludes 0 (a paired-significant difference).

        ``None`` when no interval exists (every resample skipped).
        """
        if self.ci_low is None or self.ci_high is None:
            return None
        return self.ci_low > 0.0 or self.ci_high < 0.0


def bootstrap_auroc_delta(
    y: Union[np.ndarray, Sequence[float]],
    p_a: Union[np.ndarray, Sequence[float]],
    p_b: Union[np.ndarray, Sequence[float]],
    subject_ids: Union[np.ndarray, Sequence[int]],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Optional[BootstrapAUROCDelta]:
    """Paired, subject-clustered bootstrap of ``AUROC(p_a) - AUROC(p_b)``.

    ``y``, ``p_a``, ``p_b``, and ``subject_ids`` must describe the same
    rows -- the caller is responsible for intersecting the two scorers'
    row sets first (scoring each arm on a different subset is exactly the
    unpaired mistake this function exists to prevent). Each resample
    draws subjects with replacement ONCE and scores both arms on that
    identical row multiset, so the interval is on the paired difference.
    Returns ``None`` if the observed ``y`` is single-class (AUROC, and
    hence the difference, is undefined for the cell). Resampling,
    clustering, and the weighted Mann-Whitney scoring all reuse
    :func:`bootstrap_auroc`'s machinery; with the same ``seed`` the drawn
    resamples are identical to that function's, so a delta interval and
    the two per-arm intervals from the same seed are mutually consistent.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    a_arr = np.asarray(p_a, dtype=np.float64)
    b_arr = np.asarray(p_b, dtype=np.float64)
    subject_arr = np.asarray(subject_ids)
    if not (len(y_arr) == len(a_arr) == len(b_arr) == len(subject_arr)):
        raise ValueError(
            "bootstrap_auroc_delta: y, p_a, p_b, and subject_ids must "
            f"describe the same rows; got lengths {len(y_arr)}, {len(a_arr)}, "
            f"{len(b_arr)}, {len(subject_arr)}"
        )
    n_rows = len(y_arr)
    if len(np.unique(y_arr)) < 2:
        return None

    point = float(roc_auc_score(y_arr, a_arr) - roc_auc_score(y_arr, b_arr))

    unique_subjects, inverse = np.unique(subject_arr, return_inverse=True)
    n_subjects = len(unique_subjects)
    subj_order = np.argsort(inverse, kind="stable")
    subj_counts = np.bincount(inverse, minlength=n_subjects)
    subj_boundaries = np.concatenate([[0], np.cumsum(subj_counts)])

    a_group_of_row, a_groups = _group_p_ties(a_arr)
    b_group_of_row, b_groups = _group_p_ties(b_arr)

    rng = np.random.default_rng(seed)
    deltas = []
    n_skipped = 0
    for _ in range(n_boot):
        drawn = rng.integers(0, n_subjects, size=n_subjects)
        row_idx = _gather_rows_for_drawn_subjects(
            drawn, subj_boundaries, subj_order, subj_counts
        )
        auc_a = _weighted_auroc(row_idx, y_arr, a_group_of_row, n_rows, a_groups)
        auc_b = _weighted_auroc(row_idx, y_arr, b_group_of_row, n_rows, b_groups)
        # A single-class resample is single-class for both arms (same y),
        # so auc_a is None exactly when auc_b is.
        if auc_a is None or auc_b is None:
            n_skipped += 1
            continue
        deltas.append(auc_a - auc_b)

    n_used = len(deltas)
    if n_used == 0:
        return BootstrapAUROCDelta(point, None, None, None, None, 0, n_boot)

    deltas_arr = np.array(deltas)
    return BootstrapAUROCDelta(
        point_estimate=point,
        mean=float(deltas_arr.mean()),
        std=float(deltas_arr.std(ddof=1)) if n_used > 1 else None,
        ci_low=float(np.percentile(deltas_arr, 100 * alpha / 2)),
        ci_high=float(np.percentile(deltas_arr, 100 * (1 - alpha / 2))),
        n_boot_used=n_used,
        n_boot_skipped=n_skipped,
    )


def _group_p_ties(p: np.ndarray) -> tuple[np.ndarray, int]:
    """Bucket rows by tied ``p`` value: ``(p_group_of_row, n_groups)``.

    ``p_group_of_row[i]`` is which ascending-``p`` tie-bucket original row
    ``i`` falls into (0 = smallest ``p``); a group with more than one
    member is a genuine tie in the score, which the weighted mid-rank
    formula in :func:`_weighted_auroc` must average over correctly even
    when the tied rows carry different ``y`` labels -- see that
    function's tests for why a mixed-label tie bucket is the case that
    actually exercises this.
    """
    n_rows = len(p)
    if n_rows == 0:
        return np.empty(0, dtype=np.int64), 0
    p_sort_order = np.argsort(p, kind="quicksort")
    sorted_p = p[p_sort_order]
    is_new_group = np.empty(n_rows, dtype=bool)
    is_new_group[0] = True
    np.not_equal(sorted_p[1:], sorted_p[:-1], out=is_new_group[1:])
    group_id_sorted = np.cumsum(is_new_group) - 1
    n_groups = int(group_id_sorted[-1]) + 1
    p_group_of_row = np.empty(n_rows, dtype=np.int64)
    p_group_of_row[p_sort_order] = group_id_sorted
    return p_group_of_row, n_groups


def _weighted_auroc(
    row_idx: np.ndarray,
    y: np.ndarray,
    p_group_of_row: np.ndarray,
    n_rows: int,
    n_groups: int,
) -> Optional[float]:
    """AUROC of ``y``/``p`` restricted+duplicated to ``row_idx``.

    Uses a precomputed p-value sort rather than a fresh one. A bootstrap
    resample only changes how many times each ORIGINAL row
    appears (its weight); it never changes the relative order of two rows
    with distinct p-values, so the Mann-Whitney U rank-sum formula can be
    computed from per-tie-bucket WEIGHT totals instead of a fresh sort:
    weight the row's own bucket by its multiplicity, get each bucket's
    average rank from its cumulative weight, then apply the ordinary
    rank-sum AUROC formula on those weighted ranks. Returns None if the
    resample's ``y`` came out single-class (AUROC undefined for that
    resample).
    """
    weight = np.bincount(row_idx, minlength=n_rows).astype(np.float64)
    pos_weight = weight * y
    group_weight = np.bincount(p_group_of_row, weights=weight, minlength=n_groups)
    group_pos_weight = np.bincount(
        p_group_of_row, weights=pos_weight, minlength=n_groups
    )
    n_pos = float(group_pos_weight.sum())
    n_neg = float(group_weight.sum() - n_pos)
    if n_pos == 0.0 or n_neg == 0.0:
        return None
    cum_weight_before = np.concatenate([[0.0], np.cumsum(group_weight)[:-1]])
    avg_rank_per_group = cum_weight_before + (group_weight + 1.0) / 2.0
    sum_ranks_pos = float((avg_rank_per_group * group_pos_weight).sum())
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _gather_rows_for_drawn_subjects(
    drawn: np.ndarray,
    boundaries: np.ndarray,
    order: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    """Row indices for one resample's drawn subjects, no Python-level per-subject loop.

    ``drawn`` is a subject-position array (may repeat, whole-subject
    with-replacement draw). Each drawn subject's row count is variable, so
    building the concatenated row-index array is a "ragged repeat":
    ``np.repeat`` expands each drawn subject's SLOT into as many copies as
    it has rows, then cumulative per-slot offsets recover each output
    position's index within its own subject's row run.
    """
    drawn_counts = counts[drawn]
    total_rows = int(drawn_counts.sum())
    if total_rows == 0:
        return np.empty(0, dtype=order.dtype)
    slot_of_row = np.repeat(np.arange(len(drawn)), drawn_counts)
    slot_start = np.concatenate([[0], np.cumsum(drawn_counts)[:-1]])
    within_subject_pos = np.arange(total_rows) - slot_start[slot_of_row]
    subject_start = boundaries[drawn[slot_of_row]]
    result: np.ndarray = order[subject_start + within_subject_pos]
    return result
