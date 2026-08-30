"""Best-effort hand-built features for the bespoke alert baselines.

The alert evaluation (:mod:`odyssey.inference.alerts`) compares the general
model's hazard heads with a gradient-boosted classifier fitted per event and
horizon. For that comparison to mean anything the classifier has to be given
what a careful bespoke-model builder would give it, not a token feature set.
This module builds that "strong" feature set at each landmark index time
from the events at or before it:

- **Signal panel** (:data:`SIGNAL_PANEL`): every vital and lab in the LOINC
  tables of :mod:`odyssey.data.code_mapping`, read as raw numeric values
  (unit-harmonized where a source splits a signal by unit) and summarized
  per signal as: last value, hours since last, count / mean / min / max
  over the trailing 24h, min / max over the trailing 6h, last minus
  previous value, last minus the first value of the visit, and last over
  the visit minimum (the creatinine-baseline shape KDIGO uses).
- **Drug classes** (:data:`DRUG_CLASSES`): regex classes over normalized
  medication and infusion codes (vasopressors, antibiotics, diuretics,
  sedation and analgesia, insulin, steroids, anticoagulants, ...): counts
  over 6h and 24h, hours since last, and an ever-in-visit flag.
- **Context**: hours into the visit, hours since the subject's first event,
  age, sex, prior visits, currently-in-ICU and hours since ICU admission,
  and event counts by code family over 6h / 24h / the visit so far.

Everything is computed from the same record the model reads, from events
AT OR BEFORE the index time, by binary search over per-subject sorted
arrays and ``ufunc.reduceat`` for window extrema, so a million index
rows cost seconds per signal rather than a Python loop per row. Feature
names are returned alongside the matrix so a report can list what the
baseline saw.

Boundary convention (landmark protocol v4, 2026-08-30): the index time
``t`` itself is INCLUDED -- windows are half-open ``(t - w, t]``. Before
v4 baselines saw only events strictly before ``t`` while every
model-side scorer was read at the hidden state that had already consumed
the token AT ``t`` (the bucket-opening observation, often a fresh lab
value), and MEDS-Tab's backward join also included ``t`` -- three
different information boundaries, systematically favoring the model in
every model-vs-baseline comparison. Including ``t`` cannot leak a label:
a row whose event onsets at exactly ``t`` is excluded as not-at-risk by
``outcome_at_horizon`` before any scorer sees it.
"""

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl

from odyssey.data.alert_events import origin_hours
from odyssey.data.code_mapping import unit_for
from odyssey.data.sequences import BIRTH_CODE
from odyssey.data.signal_panel import NO_SIGNAL, SIGNAL_PANEL, SignalPanelResolver
from odyssey.data.vocabulary import (
    BILLING_TYPE,
    DIAGNOSIS_TYPE,
    LAB_TYPE,
    MEDICATION_TYPE,
    OTHER_TYPE,
    PROCEDURE_TYPE,
    VISIT_TYPE,
    code_type,
)


# ---------------------------------------------------------------------------
# Panel definitions
# ---------------------------------------------------------------------------

# The signal panel itself lives in :mod:`odyssey.data.signal_panel` so the
# sequence model's per-signal head channels read exactly the same signals
# (matched inputs); re-exported here because feature names and the stats
# block are defined against it.
Converter = Callable[[np.ndarray], np.ndarray]

# (LOINC, unit tag from code_mapping.unit_for) -> conversion to the panel's
# canonical unit. Only signals a source splits by unit need an entry.
_UNIT_CONVERSIONS: Dict[Tuple[str, str], Converter] = {
    ("8310-5", "F"): lambda v: (v - 32.0) * 5.0 / 9.0,  # temperature -> Celsius
    ("1988-5", "mg/dL"): lambda v: v * 10.0,  # CRP -> mg/L
}

SIGNAL_STATS: Tuple[str, ...] = (
    "last",
    "hours_since_last",
    "n_24h",
    "mean_24h",
    "min_24h",
    "max_24h",
    "min_6h",
    "max_6h",
    "delta_prev",
    "delta_visit_first",
    "ratio_visit_min",
)

# Regexes over normalized medication / infusion codes (lowercased ingredient
# in the code string; matched case-insensitively against the whole code).
DRUG_CLASSES: Tuple[Tuple[str, str], ...] = (
    (
        "vasopressor",
        r"norepinephrine|levophed|epinephrine|vasopressin|phenylephrine"
        r"|neo-?synephrine|dopamine|angiotensin",
    ),
    ("inotrope", r"dobutamine|milrinone"),
    (
        "antibiotic",
        r"cillin|cef|ceph|penem|vancomycin|linezolid|daptomycin|mycin|micin"
        r"|floxacin|cycline|metronidazole|sulfamethoxazole|trimethoprim"
        r"|azithromycin|clindamycin|aztreonam|rifampin|fluconazole|micafungin",
    ),
    (
        "diuretic",
        r"furosemide|lasix|bumetanide|torsemide|chlorothiazide|metolazone"
        r"|spironolactone|acetazolamide|hydrochlorothiazide",
    ),
    (
        "sedation_analgesia",
        r"propofol|midazolam|lorazepam|dexmedetomidine|precedex|fentanyl"
        r"|hydromorphone|morphine|ketamine|oxycodone|methadone",
    ),
    ("insulin", r"insulin|novolog|humalog|lantus|glargine|lispro|aspart"),
    (
        "corticosteroid",
        r"hydrocortisone|methylprednisolone|dexamethasone|prednisone|prednisolone",
    ),
    (
        "anticoagulant",
        r"heparin|enoxaparin|warfarin|apixaban|rivaroxaban|argatroban"
        r"|bivalirudin|dabigatran|fondaparinux",
    ),
    ("antiarrhythmic", r"amiodarone|diltiazem|esmolol|adenosine|lidocaine"),
    ("neuromuscular_blocker", r"cisatracurium|vecuronium|rocuronium|succinylcholine"),
    ("bicarbonate", r"sodium bicarbonate|bicarb"),
    (
        "crystalloid",
        r"sodium chloride|lactated ringer|plasma-?lyte|normal saline|\bns\b",
    ),
    ("blood_product", r"red blood cells|packed|platelet|plasma|ffp|cryoprecipitate"),
)
DRUG_STATS: Tuple[str, ...] = ("n_6h", "n_24h", "hours_since_last", "ever_visit")

_DRUG_FAMILIES = ("MEDICATION", "INFUSION_DRUG", "INFUSION_START", "INFUSION_END")

# Families counted over windows and the visit (ids from odyssey.data.vocabulary).
FAMILY_IDS: Tuple[int, ...] = (
    DIAGNOSIS_TYPE,
    MEDICATION_TYPE,
    PROCEDURE_TYPE,
    LAB_TYPE,
    VISIT_TYPE,
    BILLING_TYPE,
    OTHER_TYPE,
)
FAMILY_LABELS: Tuple[str, ...] = (
    "diagnosis",
    "medication",
    "procedure",
    "lab",
    "visit",
    "billing",
    "other",
)
FAMILY_STATS: Tuple[str, ...] = ("n_6h", "n_24h", "n_visit")

CONTEXT_FEATURES: Tuple[str, ...] = (
    "hours_into_visit",
    "hours_since_origin",
    "age_years",
    "sex_female",
    "n_prior_visits",
    "in_icu",
    "hours_since_icu_admission",
    "n_events_visit",
)

_FEMALE_RE = re.compile(r"^GENDER//(F|Female)$")
_MALE_RE = re.compile(r"^GENDER//(M|Male)$")


def feature_names() -> List[str]:
    """Column names of the strong feature matrix, in order."""
    names = list(CONTEXT_FEATURES)
    for label, _ in SIGNAL_PANEL:
        names.extend(f"{label}.{stat}" for stat in SIGNAL_STATS)
    for label, _ in DRUG_CLASSES:
        names.extend(f"drug.{label}.{stat}" for stat in DRUG_STATS)
    for label in FAMILY_LABELS:
        names.extend(f"family.{label}.{stat}" for stat in FAMILY_STATS)
    return names


# ---------------------------------------------------------------------------
# Window helpers
# ---------------------------------------------------------------------------


def _reduce_windows(
    values: np.ndarray, lo: np.ndarray, hi: np.ndarray, ufunc: np.ufunc
) -> np.ndarray:
    """``ufunc.reduce(values[lo[i]:hi[i]])`` per row; NaN for empty windows."""
    out = np.full(len(lo), np.nan, dtype=np.float64)
    if len(values) == 0 or len(lo) == 0:
        return out
    nonempty = hi > lo
    if not nonempty.any():
        return out
    idx = np.stack([lo[nonempty], hi[nonempty]], axis=1).reshape(-1)
    # reduceat over pairs (lo, hi): even slots reduce values[lo:hi]; the
    # odd slots reduce values[hi:next lo] and are discarded. Every index
    # must lie inside the array, and hi may equal len(values), so reduce
    # over a copy padded with one trailing element (never part of an
    # even slot's segment).
    padded = np.concatenate([values, values[-1:]])
    reduced = ufunc.reduceat(padded, idx)[0::2]
    out[nonempty] = reduced
    return out


def _last_before(hours: np.ndarray, now: np.ndarray) -> np.ndarray:
    """Time of the last entry at or before each ``now``; NaN if none."""
    if len(hours) == 0:
        return np.full(len(now), np.nan)
    n = np.searchsorted(hours, now, side="right")
    return np.where(n > 0, hours[np.maximum(n - 1, 0)], np.nan)


@dataclass
class _Series:
    """One subject's sorted observations of one signal or drug class."""

    hours: np.ndarray
    values: np.ndarray  # empty for occurrence-only series
    cumsum: np.ndarray  # prefix sums of values (len + 1), for window means


def _series(hours: np.ndarray, values: Optional[np.ndarray]) -> _Series:
    order = np.argsort(hours, kind="stable")
    h = hours[order]
    v = values[order] if values is not None else np.zeros(0)
    cs = np.concatenate([[0.0], np.cumsum(v)]) if values is not None else np.zeros(1)
    return _Series(h, v, cs)


def _signal_features(
    series: Optional[_Series],
    now: np.ndarray,
    visit_start: np.ndarray,
    out: np.ndarray,
    col: int,
) -> None:
    """Fill the :data:`SIGNAL_STATS` block for one signal into ``out[:, col:]``."""
    if series is None or len(series.hours) == 0:
        return
    h, v = series.hours, series.values
    # Windows are (now - w, now]: the reading AT the index instant is
    # visible (protocol v4, see the module docstring), and one exactly w
    # hours old has aged out.
    hi = np.searchsorted(h, now, side="right")
    has = hi > 0
    last_idx = np.maximum(hi - 1, 0)
    last = np.where(has, v[last_idx], np.nan)
    out[:, col + 0] = last
    out[:, col + 1] = np.where(has, now - h[last_idx], np.nan)
    lo24 = np.searchsorted(h, now - 24.0, side="right")
    lo6 = np.searchsorted(h, now - 6.0, side="right")
    n24 = hi - lo24
    out[:, col + 2] = n24
    with np.errstate(invalid="ignore", divide="ignore"):
        out[:, col + 3] = np.where(
            n24 > 0,
            (series.cumsum[hi] - series.cumsum[lo24]) / np.maximum(n24, 1),
            np.nan,
        )
    out[:, col + 4] = _reduce_windows(v, lo24, hi, np.minimum)
    out[:, col + 5] = _reduce_windows(v, lo24, hi, np.maximum)
    out[:, col + 6] = _reduce_windows(v, lo6, hi, np.minimum)
    out[:, col + 7] = _reduce_windows(v, lo6, hi, np.maximum)
    prev_idx = np.maximum(hi - 2, 0)
    out[:, col + 8] = np.where(hi >= 2, last - v[prev_idx], np.nan)
    v_start = np.searchsorted(h, visit_start, side="left")
    in_visit = hi > v_start
    first_idx = np.minimum(v_start, max(len(v) - 1, 0))
    out[:, col + 9] = np.where(in_visit, last - v[first_idx], np.nan)
    vmin = _reduce_windows(v, v_start, hi, np.minimum)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = last / vmin
    out[:, col + 10] = np.where(in_visit & (vmin > 0), ratio, np.nan)


def _occurrence_features(
    series: Optional[_Series],
    now: np.ndarray,
    visit_start: np.ndarray,
    out: np.ndarray,
    col: int,
) -> None:
    """Fill the :data:`DRUG_STATS` block for one drug class."""
    out[:, col + 0] = 0.0
    out[:, col + 1] = 0.0
    out[:, col + 3] = 0.0
    if series is None or len(series.hours) == 0:
        return
    h = series.hours
    hi = np.searchsorted(h, now, side="right")  # (now - w, now], protocol v4
    out[:, col + 0] = hi - np.searchsorted(h, now - 6.0, side="right")
    out[:, col + 1] = hi - np.searchsorted(h, now - 24.0, side="right")
    has = hi > 0
    out[:, col + 2] = np.where(has, now - h[np.maximum(hi - 1, 0)], np.nan)
    out[:, col + 3] = (hi > np.searchsorted(h, visit_start, side="left")).astype(
        np.float64
    )


# ---------------------------------------------------------------------------
# Per-subject preprocessing
# ---------------------------------------------------------------------------


@dataclass
class _Subject:
    hours: np.ndarray  # all timed events, sorted
    hadms: np.ndarray
    family_cum: np.ndarray  # (n+1, len(FAMILY_IDS)) prefix counts
    signals: Dict[int, _Series]
    drugs: Dict[int, _Series]
    icu_admit_hours: np.ndarray
    icu_discharge_hours: np.ndarray
    visit_starts: Dict[int, float]
    birth_hours: Optional[float]  # relative to origin (negative)
    female: Optional[bool]


def _build_subject(
    hours: np.ndarray,
    codes: List[str],
    hadm_list: List[int],
    values: np.ndarray,
    *,
    signal_of: Dict[str, Tuple[int, Optional[Converter]]],
    drugs_of: Dict[str, List[int]],
    birth_hours: Optional[float],
    female: Optional[bool],
) -> _Subject:
    """Sorted per-subject arrays: family prefix counts, signal and drug series."""
    family_index = {f: i for i, f in enumerate(FAMILY_IDS)}
    hadms = np.array(hadm_list)
    one_hot = np.zeros((len(codes), len(FAMILY_IDS)), dtype=np.int32)
    sig_h: Dict[int, List[float]] = {}
    sig_v: Dict[int, List[float]] = {}
    drug_h: Dict[int, List[float]] = {}
    icu_admit: List[float] = []
    icu_disc: List[float] = []
    for i, code in enumerate(codes):
        f_idx = family_index.get(code_type(code))
        if f_idx is not None:
            one_hot[i, f_idx] = 1
        sig = signal_of.get(code)
        if sig is not None and not np.isnan(values[i]):
            s_idx, conv = sig
            val = float(values[i])
            if conv is not None:
                val = float(conv(np.array(val)))
            sig_h.setdefault(s_idx, []).append(float(hours[i]))
            sig_v.setdefault(s_idx, []).append(val)
        for d_idx in drugs_of.get(code, ()):
            drug_h.setdefault(d_idx, []).append(float(hours[i]))
        if code.startswith("ICU_ADMISSION//"):
            icu_admit.append(float(hours[i]))
        elif code.startswith("ICU_DISCHARGE//"):
            icu_disc.append(float(hours[i]))
    visit_starts: Dict[int, float] = {}
    for h, v in zip(hours, hadms):
        if v >= 0 and int(v) not in visit_starts:
            visit_starts[int(v)] = float(h)
    return _Subject(
        hours=hours,
        hadms=hadms,
        family_cum=np.vstack(
            [np.zeros((1, len(FAMILY_IDS)), dtype=np.int32), one_hot.cumsum(0)]
        ),
        signals={k: _series(np.array(sig_h[k]), np.array(sig_v[k])) for k in sig_h},
        drugs={k: _series(np.array(drug_h[k]), None) for k in drug_h},
        icu_admit_hours=np.array(sorted(icu_admit)),
        icu_discharge_hours=np.array(sorted(icu_disc)),
        visit_starts=visit_starts,
        birth_hours=birth_hours,
        female=female,
    )


class StrongFeatureBuilder:
    """Preprocess events once, then produce features for any index rows.

    Parameters
    ----------
    events_binned:
        Event frame after normalization / value binning (``code`` may carry
        a ``::BIN`` suffix; ``numeric_value`` holds the raw value).
    source:
        Data source, for the LOINC prefix tables and unit tags.
    """

    def __init__(
        self, events_binned: pl.DataFrame, *, source: str = "mimic_iv"
    ) -> None:
        self.source = source
        self.names = feature_names()
        self._resolver = SignalPanelResolver(source)
        # prefix -> unit converter into the panel's canonical unit (only
        # unit-split signals have one); resolution itself is the shared
        # resolver's, so the model and this builder classify codes alike.
        self._converter_of: Dict[str, Optional[Converter]] = {}
        for (_, loinc), prefixes in zip(SIGNAL_PANEL, self._resolver.prefixes):
            for prefix in prefixes:
                unit = unit_for(prefix, source=source)
                self._converter_of[prefix] = (
                    _UNIT_CONVERSIONS.get((loinc, unit)) if unit else None
                )
        self._drug_res = [re.compile(rx, re.IGNORECASE) for _, rx in DRUG_CLASSES]
        self._subjects: Dict[int, _Subject] = self._preprocess(events_binned)

    # -- preprocessing -----------------------------------------------------

    def _classify_codes(
        self, codes: Sequence[str]
    ) -> Tuple[Dict[str, Tuple[int, Optional[Converter]]], Dict[str, List[int]]]:
        """Distinct code -> (signal index, converter) and code -> drug classes."""
        signal_of: Dict[str, Tuple[int, Optional[Converter]]] = {}
        drugs_of: Dict[str, List[int]] = {}
        for code in set(codes):
            base = code.rsplit("::", 1)[0] if "::" in code else code
            s_idx, prefix = self._resolver.resolve_with_prefix(code)
            if s_idx != NO_SIGNAL and prefix is not None:
                signal_of[code] = (s_idx, self._converter_of[prefix])
            if base.split("//", 1)[0] in _DRUG_FAMILIES:
                hits = [i for i, rx in enumerate(self._drug_res) if rx.search(base)]
                if hits:
                    drugs_of[code] = hits
        return signal_of, drugs_of

    def _preprocess(self, events: pl.DataFrame) -> Dict[int, _Subject]:
        origins = origin_hours(events)
        timed = (
            events.filter(pl.col("time").is_not_null())
            .join(origins, on="subject_id", how="left")
            .with_columns(
                (
                    (pl.col("time") - pl.col("_origin")).dt.total_seconds() / 3600.0
                ).alias("_hours")
            )
        )
        birth = timed.filter(pl.col("code") == BIRTH_CODE)
        birth_map = dict(zip(birth["subject_id"].to_list(), birth["_hours"].to_list()))
        gender = events.filter(pl.col("code").str.starts_with("GENDER//"))
        female_map: Dict[int, bool] = {}
        for sid, code in zip(gender["subject_id"].to_list(), gender["code"].to_list()):
            if _FEMALE_RE.match(code):
                female_map[int(sid)] = True
            elif _MALE_RE.match(code):
                female_map[int(sid)] = False
        timed = timed.filter(pl.col("code") != BIRTH_CODE)
        cols = ["subject_id", "_hours", "code", "hadm_id"]
        has_values = "numeric_value" in timed.columns
        if has_values:
            cols.append("numeric_value")
        signal_of, drugs_of = self._classify_codes(timed["code"].to_list())
        subjects: Dict[int, _Subject] = {}
        for key, group in timed.select(cols).group_by(
            "subject_id", maintain_order=True
        ):
            sid = int(key[0])
            frame = group.sort("_hours")
            values = (
                frame["numeric_value"].cast(pl.Float64).to_numpy()
                if has_values
                else np.full(frame.height, np.nan)
            )
            subjects[sid] = _build_subject(
                frame["_hours"].to_numpy().astype(np.float64),
                frame["code"].to_list(),
                [-1 if h is None else int(h) for h in frame["hadm_id"].to_list()],
                values,
                signal_of=signal_of,
                drugs_of=drugs_of,
                birth_hours=birth_map.get(sid),
                female=female_map.get(sid),
            )
        return subjects

    # -- features ------------------------------------------------------------

    def features(
        self,
        subject_ids: Sequence[int],
        visit_ids: Sequence[int],
        times: Sequence[float],
    ) -> np.ndarray:
        """Feature matrix ``(n_rows, n_features)`` for the given index rows."""
        n = len(subject_ids)
        out = np.full((n, len(self.names)), np.nan, dtype=np.float32)
        sids = np.asarray(subject_ids)
        vids = np.asarray(visit_ids)
        now_all = np.asarray(times, dtype=np.float64)
        for sid in np.unique(sids):
            subject = self._subjects.get(int(sid))
            if subject is None:
                continue
            rows = np.nonzero(sids == sid)[0]
            now = now_all[rows]
            v_start = np.array(
                [subject.visit_starts.get(int(v), t) for v, t in zip(vids[rows], now)]
            )
            block = np.full((len(rows), len(self.names)), np.nan, dtype=np.float64)
            self._context(subject, now, v_start, block)
            col = len(CONTEXT_FEATURES)
            for s_idx in range(len(SIGNAL_PANEL)):
                _signal_features(subject.signals.get(s_idx), now, v_start, block, col)
                col += len(SIGNAL_STATS)
            for d_idx in range(len(DRUG_CLASSES)):
                _occurrence_features(subject.drugs.get(d_idx), now, v_start, block, col)
                col += len(DRUG_STATS)
            hi = np.searchsorted(subject.hours, now, side="right")
            lo6 = np.searchsorted(subject.hours, now - 6.0, side="right")
            lo24 = np.searchsorted(subject.hours, now - 24.0, side="right")
            # "since visit start" keeps side="left": the visit's own first
            # event (charted exactly at v_start) belongs to the visit.
            lo_v = np.searchsorted(subject.hours, v_start, side="left")
            cum = subject.family_cum
            for f_idx in range(len(FAMILY_IDS)):
                block[:, col + 0] = cum[hi, f_idx] - cum[lo6, f_idx]
                block[:, col + 1] = cum[hi, f_idx] - cum[lo24, f_idx]
                block[:, col + 2] = cum[hi, f_idx] - cum[lo_v, f_idx]
                col += len(FAMILY_STATS)
            out[rows] = block.astype(np.float32)
        return out

    def _context(
        self,
        subject: _Subject,
        now: np.ndarray,
        v_start: np.ndarray,
        block: np.ndarray,
    ) -> None:
        block[:, 0] = now - v_start
        block[:, 1] = now
        if subject.birth_hours is not None:
            block[:, 2] = (now - subject.birth_hours) / (24.0 * 365.25)
        if subject.female is not None:
            block[:, 3] = 1.0 if subject.female else 0.0
        starts = np.array(sorted(subject.visit_starts.values()))
        block[:, 4] = np.searchsorted(starts, v_start, side="left")
        last_admit = _last_before(subject.icu_admit_hours, now)
        last_disc = _last_before(subject.icu_discharge_hours, now)
        in_icu = ~np.isnan(last_admit) & (
            np.isnan(last_disc) | (last_admit > last_disc)
        )
        block[:, 5] = in_icu.astype(np.float64)
        block[:, 6] = np.where(in_icu, now - last_admit, np.nan)
        hi = np.searchsorted(subject.hours, now, side="right")
        lo_v = np.searchsorted(subject.hours, v_start, side="left")
        block[:, 7] = hi - lo_v
