"""Clinician demo: replay a patient's chart and watch the model's risk forecasts.

A small web app that runs a trained Odyssey checkpoint on one patient at a
time and shows, in plain clinical language, how its risks (vasopressors,
ICU admission, AKI, Sepsis-3, death) evolve through an admission, when
they would have raised an alert, what the model thinks is going on, what
would move the forecast, and how good the model is. Runs on the GPU host,
bound to loopback, viewed through an SSH tunnel: patient-level data never
leaves the host. See ``docs/clinician_demo.md``.

Layers, torch only below the service: ``schemas`` (JSON contracts) ->
``codebook`` / ``patient_store`` / ``thresholds`` / ``showcase`` /
``scorecard`` (pure data) -> ``forecast`` / ``whatif`` / ``evidence``
(model) -> ``service`` (orchestration, GPU lock, caches) -> ``server``
(HTTP) -> ``static/`` (rendering only).
"""
