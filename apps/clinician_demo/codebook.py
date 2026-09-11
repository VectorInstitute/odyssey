"""Turn MEDS codes and model tokens into plain clinical language.

A token is a MEDS code plus an optional value-bin suffix
(``LAB//220045//bpm::HIGH``). The code names *what* was recorded; the
suffix says how the value compared with the clinical range the concept
rules use (``LOW``/``NORMAL``/``HIGH``/``CRITICAL``) or, for signals
without a curated range, which fifth of the training distribution it fell
in (``Q1``..``Q5``).

Names come from the extraction's own ``metadata/codes.parquet`` where it
has them (labs, vitals, infusions, diagnoses, procedures); everything else
(medications after normalization, admissions, transfers) already carries
readable text in the code and is formatted structurally. Nothing here
invents a name the data does not contain.
"""

import math
import re
from collections.abc import Mapping
from pathlib import Path

from apps.clinician_demo.schemas import TimelineEntry
from odyssey.data.code_metadata import load_code_descriptions


BIN_SEPARATOR = "::"
UNKNOWN_TOKEN = "[UNK]"
FLAG_BINS = frozenset({"LOW", "HIGH", "CRITICAL"})
_CLINICAL_BIN_WORDS = {
    "LOW": "low",
    "NORMAL": "normal",
    "HIGH": "high",
    "CRITICAL": "critical",
}
_QUANTILE_BIN_WORDS = {
    "Q1": "bottom fifth",
    "Q2": "2nd fifth",
    "Q3": "middle fifth",
    "Q4": "4th fifth",
    "Q5": "top fifth",
}
_CARE_KINDS = {
    "HOSPITAL_ADMISSION": "Hospital admission",
    "HOSPITAL_DISCHARGE": "Discharge",
    "ICU_ADMISSION": "ICU admission",
    "ICU_DISCHARGE": "ICU discharge",
    "TRANSFER_TO": "Transfer",
    "ED_REGISTRATION": "ED registration",
    "ED_OUT": "Left ED",
}
_DEMOGRAPHIC_KINDS = {
    "GENDER": "Sex",
    "RACE": "Race",
    "LANGUAGE": "Language",
    "INSURANCE": "Insurance",
    "MARITAL_STATUS": "Marital status",
}
#: Categories that mark a change in where/how the patient is cared for.
MARKER_CATEGORIES = frozenset({"care", "death"})


def split_bin(token: str) -> tuple[str, str | None]:
    """Split ``code::BIN`` into ``(code, BIN)``; ``BIN`` is ``None`` if absent."""
    code, sep, suffix = token.partition(BIN_SEPARATOR)
    return code, (suffix or None) if sep else None


def bin_flag(bin_label: str | None) -> str | None:
    """Return the flag a bin implies: ``LOW``/``HIGH``/``CRITICAL`` or ``None``."""
    return bin_label if bin_label in FLAG_BINS else None


def bin_words(bin_label: str | None) -> str | None:
    """Return plain words for a bin label, or ``None`` for no / unknown bin."""
    if bin_label is None:
        return None
    return _CLINICAL_BIN_WORDS.get(bin_label) or _QUANTILE_BIN_WORDS.get(bin_label)


def category(code: str) -> str:  # noqa: PLR0911 -- one return per family
    """Return the coarse event family used for grouping and colour."""
    parts = code.split("//")
    kind = parts[0]
    if kind == "LAB":
        if len(parts) > 1 and parts[1] == "RESULT":
            return "lab"
        if len(parts) > 1 and parts[1] == "SPECIMEN_COLLECTED":
            return "order"
        return "vital"
    if kind == "MEDICATION":
        return "medication"
    if kind in ("INFUSION_START", "INFUSION_END", "SUBJECT_WEIGHT_AT_INFUSION"):
        return "infusion"
    if kind == "SUBJECT_FLUID_OUTPUT":
        return "output"
    if kind == "DIAGNOSIS":
        return "diagnosis"
    if kind in ("PROCEDURE", "HCPCS"):
        return "procedure"
    if kind in _CARE_KINDS:
        return "care"
    if kind == "MEDS_DEATH":
        return "death"
    if kind in _DEMOGRAPHIC_KINDS:
        return "demographic"
    if kind == "DRG":
        return "billing"
    return "other"


#: MIMIC-IV admission types in plain words.
_ADMISSION_TYPES = {
    "EW EMER.": "Emergency",
    "DIRECT EMER.": "Direct emergency",
    "URGENT": "Urgent",
    "ELECTIVE": "Elective",
    "SURGICAL SAME DAY ADMISSION": "Same-day surgery",
    "OBSERVATION ADMIT": "Observation",
    "EU OBSERVATION": "Emergency observation",
    "DIRECT OBSERVATION": "Direct observation",
    "AMBULATORY OBSERVATION": "Ambulatory observation",
}
# A LOINC long name is "<analyte> [<property>] in <specimen> by <method>";
# a clinician wants the analyte.
_LOINC_TAIL = re.compile(r" \[| (?:in|of|by) (?=[A-Z])")


def _short_description(description: str, family: str) -> str:
    """Trim LOINC-style long names for labs ("Creatinine [Mass/volume] in ...")."""
    if family in ("lab", "order"):
        return _LOINC_TAIL.split(description, maxsplit=1)[0]
    return description


_ACRONYMS = frozenset({"PACU", "ICU", "ED", "SNF", "OR"})


def _place(text: str) -> str:
    """``"TRANSFER FROM SKILLED NURSING FACILITY"`` in lower case, acronyms kept."""
    words = text.split()
    return " ".join(
        w if w in _ACRONYMS and len(words) > 1 or w == "PACU" else w.lower()
        for w in words
    )


def admission_label(code: str) -> str:
    """Name a ``HOSPITAL_ADMISSION//<type>//<from>`` code in plain words."""
    parts = [p for p in code.split("//")[1:] if p]
    if not parts:
        return "Admission"
    kind = _ADMISSION_TYPES.get(parts[0].upper(), _pretty(parts[0]).capitalize())
    if len(parts) < 2:
        return f"{kind} admission"
    place = _place(parts[1])
    if place.startswith("transfer from "):
        return (
            f"{kind} admission, transferred from {place.removeprefix('transfer from ')}"
        )
    return f"{kind} admission, from {place}"


def _pretty(text: str) -> str:
    """``"SURGICAL SAME DAY ADMISSION"`` -> ``"Surgical same day admission"``."""
    text = text.replace("_", " ").strip()
    return text[:1].upper() + text[1:].lower() if text.isupper() else text


class Codebook:
    """Readable labels, categories, units and flags for one extraction's codes."""

    def __init__(self, descriptions: Mapping[str, str] | None = None) -> None:
        """Wrap a ``code -> description`` mapping (may be empty)."""
        self._descriptions = dict(descriptions or {})

    @classmethod
    def from_metadata_dir(cls, metadata_dir: str | Path | None) -> "Codebook":
        """Build from an extraction's ``metadata/`` directory (``codes.parquet``)."""
        return cls(load_code_descriptions(metadata_dir))

    def __len__(self) -> int:
        """Return the number of codes with a dictionary description."""
        return len(self._descriptions)

    def label(self, code: str) -> str:
        """Return the readable name of a raw code (a value-bin suffix is ignored)."""
        code, _ = split_bin(code)
        family = category(code)
        description = self._descriptions.get(code)
        if description:
            short = _short_description(description, family)
            return f"{short} (sample sent)" if family == "order" else short
        return self._structural_label(code)

    def unit(self, code: str) -> str | None:
        """Return the measurement unit carried in a lab/vital/output code, if any."""
        code, _ = split_bin(code)
        parts = code.split("//")
        if parts[0] in ("LAB", "SUBJECT_FLUID_OUTPUT", "SUBJECT_WEIGHT_AT_INFUSION"):
            unit = parts[-1] if len(parts) >= 2 else ""
            return None if unit in ("", "UNK") or unit.isdigit() else unit
        return None

    def token_label(self, token: str) -> str:
        """Return the readable name of a model token, with its value bin in words."""
        if token == UNKNOWN_TOKEN:
            return "An uncommon event (outside the model's vocabulary)"
        code, bin_label = split_bin(token)
        words = bin_words(bin_label)
        name = self.label(code)
        return f"{name} ({words})" if words else name

    def entry(self, token: str, t: float, value: float | None) -> TimelineEntry:
        """Build one timeline row: what was recorded, its value+unit and flag."""
        code, bin_label = split_bin(token)
        shown: str | None = None
        if value is not None and not math.isnan(value):
            unit = self.unit(code)
            shown = f"{value:.4g}" + (f" {unit}" if unit else "")
        return TimelineEntry(
            t=t,
            category=category(code),
            label=self.label(code),
            value=shown,
            flag=bin_flag(bin_label),
        )

    @staticmethod
    def _structural_label(code: str) -> str:  # noqa: PLR0911, PLR0912 -- one branch per family
        parts = code.split("//")
        kind, rest = parts[0], [p for p in parts[1:] if p]
        if kind == "MEDICATION":
            if rest and rest[0] in ("START", "STOP") and len(rest) >= 2:
                verb = "Started" if rest[0] == "START" else "Stopped"
                return f"{verb} {rest[1]}"
            if len(rest) >= 2:
                return f"{_pretty(rest[0]).capitalize()} ({rest[1].lower()})"
            return _pretty(rest[0]).capitalize() if rest else "Medication"
        if kind in _CARE_KINDS:
            base = _CARE_KINDS[kind]
            if kind == "HOSPITAL_ADMISSION":
                return admission_label(code)
            if kind == "TRANSFER_TO" and rest:
                return f"Transfer to {rest[-1]}"
            if kind == "HOSPITAL_DISCHARGE" and rest:
                return f"Discharge to {_pretty(rest[-1]).lower()}"
            return f"{base}: {rest[-1]}" if rest else base
        if kind == "MEDS_DEATH":
            return "Death"
        if kind == "SUBJECT_WEIGHT_AT_INFUSION":
            return "Weight at infusion"
        if kind in _DEMOGRAPHIC_KINDS:
            return f"{_DEMOGRAPHIC_KINDS[kind]}: {rest[0]}" if rest else kind
        if kind in ("DIAGNOSIS", "PROCEDURE") and len(rest) == 3 and rest[0] == "ICD":
            return f"{kind.title()} ICD-{rest[1]} {rest[2]}"
        if kind in ("INFUSION_START", "INFUSION_END"):
            verb = (
                "Infusion started" if kind == "INFUSION_START" else "Infusion stopped"
            )
            return f"{verb} (item {rest[0]})" if rest else verb
        if kind == "LAB" and rest:
            return f"Lab item {rest[-2] if len(rest) >= 2 else rest[0]}"
        return " · ".join([_pretty(kind), *rest]) if rest else _pretty(kind)


__all__ = [
    "BIN_SEPARATOR",
    "FLAG_BINS",
    "MARKER_CATEGORIES",
    "Codebook",
    "admission_label",
    "bin_flag",
    "bin_words",
    "category",
    "split_bin",
]
