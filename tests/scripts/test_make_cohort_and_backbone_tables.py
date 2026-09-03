"""Cohort-sensitivity and backbone table generators render from banked JSON."""

from scripts.make_backbone_table import render as render_backbone
from scripts.make_cohort_table import render as render_cohort


def test_cohort_table_reports_shares_and_excluded_aurocs() -> None:
    check = {
        "n_held_out_subjects": 1000,
        "flag_prevalence_subjects": {"esrd_dialysis": 12, "palliative_hospice": 40},
        "rows_by_event": {
            "death": {
                "positives_24h": 100,
                "auroc_24h_all": 0.9432,
                "esrd_dialysis": {
                    "positives_flagged": 6,
                    "auroc_24h_excluding_flagged": 0.9484,
                },
                "palliative_hospice": {
                    "positives_flagged": 58,
                    "auroc_24h_excluding_flagged": 0.9336,
                },
            }
        },
    }
    tex = render_cohort([("MIMIC-IV", check)])
    assert (
        "MIMIC-IV (1\\%, 4\\% of patients) & Death & 6\\% & 0.948 & 58\\% & 0.934 & 0.943"
        in tex
    )
    assert tex.count("\\midrule") == 1 and "\\bottomrule" in tex


def test_backbone_table_bolds_intervals_that_exclude_zero() -> None:
    result = {
        "label_a": "hybrid",
        "label_b": "transformer",
        "n_subjects": 100,
        "n_truncated_subjects": 30,
        "cells": [
            {
                "event": "death",
                "horizon": "24h",
                "stratum": "whole",
                "auroc_hybrid": 0.952,
                "auroc_transformer": 0.947,
                "delta_a_minus_b": {
                    "point_estimate": 0.004,
                    "ci_low": -0.004,
                    "ci_high": 0.014,
                },
            },
            {
                "event": "death",
                "horizon": "24h",
                "stratum": "truncated",
                "auroc_hybrid": 0.950,
                "auroc_transformer": 0.985,
                "delta_a_minus_b": {
                    "point_estimate": -0.035,
                    "ci_low": -0.042,
                    "ci_high": -0.028,
                },
            },
        ],
    }
    tex = render_backbone(result)
    assert "seen whole (70 subjects)" in tex and "truncated (30 subjects)" in tex
    assert "+0.004 [-0.004, +0.014]" in tex
    assert "\\textbf{-0.035 [-0.042, -0.028]}" in tex
    assert tex.count("Death") == 1  # one event block, 8 h and 72 h rows absent
