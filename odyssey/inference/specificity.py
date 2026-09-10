"""Specificity of a steering dial: does the push capture the state or only "sicker".

Reads the summaries written by :mod:`odyssey.inference.steering` when it was
run with readout shifts and the state-transition probes, and scores each
dial on three things a clinician would ask:

* **readout focus** -- of the total readout movement the push caused across
  every concept, how much landed on the pushed concept itself, and did the
  physiological opposite move down
  (:data:`~odyssey.inference.steering.READOUT_EXPECTATIONS`);
* **two-sided outcomes** -- did the good outcomes (ICU discharge, discharge
  alive, vasopressor stop) move the declared way, which for a sicker state
  is DOWN while the bad outcomes go up, so a push that only means "sicker"
  cannot pass by pointing everything one way;
* **against a control** -- the same numbers for a random-direction push at
  the same strength, so the reader sees what a meaningless push scores.

Pure functions over the JSON, no model access; the table script and the
tests call these.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from odyssey.data.concepts import canonical_concept_name


Summary = Mapping[str, Any]

# outcomes whose declared direction runs against "sicker"
GOOD_OUTCOMES: tuple[str, ...] = (
    "icu_discharge",
    "hospital_discharge_alive",
    "vasopressor_stop",
)


@dataclass(frozen=True)
class DialSpecificity:
    """One dial in one direction, scored."""

    concept: str
    direction: str
    own_shift: float
    """Signed change of the pushed concept's own readout."""
    focus: float
    """|own shift| over the sum of |shift| across every concept (0 to 1)."""
    opposite_as_expected: int
    opposite_declared: int
    """Opposite-concept readouts that moved down, of those declared."""
    others_separated: int
    others_total: int
    """Undeclared readouts whose paired interval excludes zero (collateral)."""
    good_as_expected: int
    good_declared: int
    """Good outcomes (24 h) that moved the declared way, of those declared."""
    bad_as_expected: int
    bad_declared: int
    """Bad outcomes (24 h) that moved the declared way, of those declared."""


def _shift_delta(shift: Mapping[str, Any]) -> float:
    return float(shift["delta"]["point"])


def score_dial(summary: Summary, *, horizon_hours: float = 24.0) -> DialSpecificity:
    """Score one summary (one concept, one direction)."""
    concept = canonical_concept_name(summary["concept"])
    shifts = summary.get("concept_shifts") or []
    own = 0.0
    total = 0.0
    opp_ok = opp_n = oth_sep = oth_n = 0
    for shift in shifts:
        delta = _shift_delta(shift)
        total += abs(delta)
        if canonical_concept_name(shift["concept"]) == concept:
            own = delta
            continue
        expected = shift.get("expected_sign")
        if expected is None:
            oth_n += 1
            oth_sep += int(bool(shift.get("separated")))
        else:
            opp_n += 1
            opp_ok += int(bool(shift.get("as_expected")))
    good_ok = good_n = bad_ok = bad_n = 0
    for outcome in summary.get("outcomes") or []:
        if outcome.get("expected_sign") is None:
            continue
        if float(outcome["horizon_hours"]) != horizon_hours:
            continue
        hit = int(bool(outcome.get("as_expected")))
        if outcome["event"] in GOOD_OUTCOMES:
            good_n += 1
            good_ok += hit
        else:
            bad_n += 1
            bad_ok += hit
    return DialSpecificity(
        concept=concept,
        direction=str(summary["direction"]),
        own_shift=own,
        focus=abs(own) / total if total > 0 else 0.0,
        opposite_as_expected=opp_ok,
        opposite_declared=opp_n,
        others_separated=oth_sep,
        others_total=oth_n,
        good_as_expected=good_ok,
        good_declared=good_n,
        bad_as_expected=bad_ok,
        bad_declared=bad_n,
    )


def score_run(
    summaries: Iterable[Summary], *, horizon_hours: float = 24.0
) -> list[DialSpecificity]:
    """Score every summary of one steering JSON."""
    return [score_dial(s, horizon_hours=horizon_hours) for s in summaries]


@dataclass(frozen=True)
class RunTotals:
    """What the whole benchmark says, in the counts the paper quotes."""

    n_dials: int
    median_focus: float
    opposite_as_expected: int
    opposite_declared: int
    good_as_expected: int
    good_declared: int
    bad_as_expected: int
    bad_declared: int
    both_sided: int
    """Dials whose declared good AND bad outcomes all moved the declared way."""


def totals(scored: Sequence[DialSpecificity]) -> RunTotals:
    """Aggregate dial scores into the paper's counts."""
    focus = sorted(d.focus for d in scored)
    median = 0.0
    if focus:
        mid = len(focus) // 2
        median = focus[mid] if len(focus) % 2 else (focus[mid - 1] + focus[mid]) / 2
    return RunTotals(
        n_dials=len(scored),
        median_focus=median,
        opposite_as_expected=sum(d.opposite_as_expected for d in scored),
        opposite_declared=sum(d.opposite_declared for d in scored),
        good_as_expected=sum(d.good_as_expected for d in scored),
        good_declared=sum(d.good_declared for d in scored),
        bad_as_expected=sum(d.bad_as_expected for d in scored),
        bad_declared=sum(d.bad_declared for d in scored),
        both_sided=sum(
            1
            for d in scored
            if d.good_declared > 0
            and d.bad_declared > 0
            and d.good_as_expected == d.good_declared
            and d.bad_as_expected == d.bad_declared
        ),
    )


__all__ = [
    "GOOD_OUTCOMES",
    "DialSpecificity",
    "RunTotals",
    "score_dial",
    "score_run",
    "totals",
]
