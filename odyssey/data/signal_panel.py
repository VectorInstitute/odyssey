"""The curated vital/lab signal panel shared by the baselines and value metrics.

One place defines *which* clinical signals count as "the panel" and how a
MEDS code resolves to one of them, so that the tuned tabular baselines
(:mod:`odyssey.inference.baseline_features`) and the model's per-signal
value-metric breakdown (:class:`~odyssey.inference.run_inference._RunningValueMetrics`,
e.g. "creatinine CRPS") read the same signals.

Resolution goes through the per-source LOINC tables in
:mod:`odyssey.data.code_mapping` (a signal a source does not chart simply
never resolves there) and matches a code by prefix, the same rule the
baseline feature builder has always used. Values are NOT unit-harmonized
here: the model side reads per-code standardized values (``numeric_z``),
which are already unit-free; the baseline side keeps its own converters.
"""

from typing import Dict, List, Optional, Tuple

from odyssey.data.code_mapping import prefixes_for_loinc


# name -> LOINC. Names are for feature labels only; resolution to concrete
# code prefixes goes through the per-source LOINC tables.
SIGNAL_PANEL: Tuple[Tuple[str, str], ...] = (
    ("heart_rate", "8867-4"),
    ("resp_rate", "9279-1"),
    ("spo2", "59408-5"),
    ("temperature", "8310-5"),
    ("sbp_noninvasive", "76534-7"),
    ("dbp_noninvasive", "76535-4"),
    ("map_noninvasive", "76536-2"),
    ("sbp_arterial", "8480-6"),
    ("dbp_arterial", "8462-4"),
    ("map_arterial", "8478-0"),
    ("fio2", "3150-0"),
    ("gcs_eye", "9267-6"),
    ("gcs_verbal", "9270-0"),
    ("gcs_motor", "9268-4"),
    ("urine_output", "9187-6"),
    ("creatinine", "2160-0"),
    ("bun", "3094-0"),
    ("lactate", "32693-4"),
    ("wbc", "6690-2"),
    ("hemoglobin", "718-7"),
    ("hematocrit", "4544-3"),
    ("platelets", "777-3"),
    ("sodium", "2951-2"),
    ("potassium", "2823-3"),
    ("chloride", "2075-0"),
    ("bicarbonate", "1963-8"),
    ("bicarbonate_blood_gas", "1959-6"),
    ("glucose", "2345-7"),
    ("glucose_whole_blood", "2339-0"),
    ("anion_gap", "1863-0"),
    ("calcium", "17861-6"),
    ("magnesium", "19123-9"),
    ("phosphate", "2777-1"),
    ("albumin", "1751-7"),
    ("bilirubin_total", "1975-2"),
    ("alt", "1742-6"),
    ("ast", "1920-8"),
    ("alk_phos", "6768-6"),
    ("inr", "6301-6"),
    ("ptt", "14979-9"),
    ("ph", "11558-4"),
    ("pco2", "11557-6"),
    ("po2", "11556-8"),
    ("base_excess", "11555-0"),
    ("troponin_t", "6598-7"),
    ("troponin_i", "10839-9"),
    ("nt_probnp", "33762-6"),
    ("crp", "1988-5"),
)

N_PANEL_SIGNALS = len(SIGNAL_PANEL)

# Per-token signal id for a code outside the panel.
NO_SIGNAL = -1


class SignalPanelResolver:
    """Map MEDS codes of one source to panel signal indices.

    A code matches a signal when its un-binned form (everything before the
    ``::<bin>`` suffix) starts with one of the signal's code prefixes for
    ``source``. Signals are tried in :data:`SIGNAL_PANEL` order and the
    first match wins -- the rule :class:`StrongFeatureBuilder` has used
    since the panel existed, kept identical so the model and the baselines
    classify every code the same way. Results are memoized per distinct
    code (a split has thousands of distinct codes and millions of rows).
    """

    def __init__(self, source: str = "mimic_iv") -> None:
        """Build the prefix table for ``source``."""
        self.source = source
        self.prefixes: List[List[str]] = [
            sorted(prefixes_for_loinc(loinc, source=source))
            for _, loinc in SIGNAL_PANEL
        ]
        self._cache: Dict[str, Tuple[int, Optional[str]]] = {}

    def resolve_with_prefix(self, code: str) -> Tuple[int, Optional[str]]:
        """Return ``(signal index, matching prefix)``; ``(NO_SIGNAL, None)`` if none."""
        hit = self._cache.get(code)
        if hit is not None:
            return hit
        base = code.rsplit("::", 1)[0] if "::" in code else code
        result: Tuple[int, Optional[str]] = (NO_SIGNAL, None)
        for s_idx, entries in enumerate(self.prefixes):
            for prefix in entries:
                if base.startswith(prefix):
                    result = (s_idx, prefix)
                    break
            if result[0] != NO_SIGNAL:
                break
        self._cache[code] = result
        return result

    def resolve(self, code: str) -> int:
        """Return the panel index of ``code`` or :data:`NO_SIGNAL`."""
        return self.resolve_with_prefix(code)[0]


__all__ = [
    "SIGNAL_PANEL",
    "N_PANEL_SIGNALS",
    "NO_SIGNAL",
    "SignalPanelResolver",
]
