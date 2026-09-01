"""Concept intervention and completeness evaluation.

The one architectural claim a concept bottleneck makes beyond ordinary
sequence modeling is that the supervised concepts *mediate* prediction:
the task head reads a mixture steered by the concept probabilities, so
editing those probabilities should causally move the forecasts. This
module tests that claim directly, CEM/CBGM-style, by re-running the
streaming next-event evaluation under do()-style edits inside the
bottleneck (:class:`~odyssey.models.concept_bottleneck.BottleneckIntervention`)
and comparing task metrics across modes:

- ``none`` -- the unedited baseline; must reproduce the standard
  evaluation's numbers.
- ``truth`` -- replace each known concept's mixing probability with its
  ground-truth rule label wherever that label is observed. If concepts
  causally steer prediction, perfect concept information should *help*
  (or at minimum not hurt) next-event accuracy; a model that ignores its
  bottleneck shows no movement.
- ``flip`` -- feed ``1 - label`` on the same positions. The mirror
  image: reliance on the concept channel shows up as damage.
- ``flip_gated`` -- the same ``1 - label`` edit, but its logit changes
  pass through a suppression-only gate: ``logits = logits_none +
  min(0, logits_flip - logits_none)``, so the flip may lower token
  logits but never raise them. Guide Labs (arXiv:2608.07594, Fig. 19)
  show naive negative steering *promotes* anti-aligned or unrelated
  vocabulary rather than only suppressing the aligned direction; this
  mode is the control for that artifact. If ``flip_gated`` recovers
  most of ``flip``'s damage relative to ``none``, the damage came from
  spurious promotion (a steering artifact), not from losing the true
  concept's contribution.
- ``truth_calibrated`` / ``flip_calibrated`` -- the output-calibrated
  protocol (Guide Labs, adapted): instead of a hard 0/1 value, displace
  the model's own probability by a per-concept step ``gamma_i = tau /
  peak_i`` toward the true (resp. flipped) pole, clipped to [0, 1],
  where ``peak_i`` is concept ``i``'s largest per-token logit shift per
  unit of mixing probability (see
  :func:`~odyssey.inference.concept_attribution.calibrated_gammas`).
  Every concept then applies the same largest achievable logit shift
  ``tau``, so per-concept sensitivities are comparable regardless of
  how large the head's weights happen to be for each concept -- the
  targeted fix for band-population artifacts (rare concepts rarely
  enter the |p - 0.5| band). The ``uncertain_band`` restriction is
  deliberately NOT applied to these modes: calibration replaces the
  band as the equalizer.
- ``random`` -- feed coin-flip values on the same positions. Separates
  "any perturbation hurts" from "wrong information hurts": a gap
  between ``random`` and ``flip`` means the model reads the *direction*
  of the concept values, not just their stability.
- ``zero_known`` / ``zero_unknown`` -- zero the known concepts' (resp.
  the unknown channel's) mixed embeddings. The completeness probe: how
  the task signal is apportioned between the supervised, interpretable
  channel and the unsupervised one. A bottleneck whose entire task
  performance survives ``zero_known`` is interpretable-in-name-only --
  the concepts would be a decorative side channel.

Intervened values are applied per position as *running* labels: the
visit- (or stay-) scoped label, but true only from the concept's
first-trigger time onward, so what is fed at each position is what is
true as of that moment rather than a retrospective fact about the whole
visit (see :func:`_position_labels`). Everything is gated by the
observed mask: unobserved concepts keep the model's own probability, in
every mode -- there is no ground truth to feed there.
"""

import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import polars as pl
import torch
import torch.nn.functional as F  # noqa: N812

from odyssey.data.code_normalization import maybe_normalize
from odyssey.data.concepts import concepts_for_source
from odyssey.data.history_recap import maybe_history_recap
from odyssey.data.sidecars import activate_sidecars
from odyssey.data.streaming import NO_SUBJECT, PackedLaneSampler, StreamingChunk
from odyssey.data.value_binning import add_value_tokens
from odyssey.data.vocabulary import PAD_ID, Vocabulary
from odyssey.inference.concept_attribution import (
    calibrated_gammas,
    mean_concept_directions,
)
from odyssey.inference.run_inference import (
    _CODE_TYPE_NAMES,
    _build_type_lookup,
    load_run,
    refuse_existing_output,
)
from odyssey.models.concept_bottleneck import (
    BottleneckIntervention,
    intervention_apply_mask,
)
from odyssey.models.sequence_model import (
    ConceptBottleneckSequenceModel,
    ConceptLabelDict,
    ConceptSupervision,
)
from odyssey.training.data import (
    build_concept_first_times,
    build_concept_label_dicts,
    build_visit_concept_first_times,
    build_visit_concept_label_dicts,
    iter_patient_sequences,
    load_meds_shards,
)
from odyssey.training.running_labels import position_running_labels
from odyssey.training.train import _move_chunk_to_device


logger = logging.getLogger(__name__)

INTERVENTION_MODES = (
    "none",
    "truth",
    "flip",
    "flip_gated",
    "truth_calibrated",
    "flip_calibrated",
    "random",
    "zero_known",
    "zero_unknown",
)

CALIBRATED_MODES = ("truth_calibrated", "flip_calibrated")


@dataclass(frozen=True)
class InterventionResult:
    """Task metrics for one intervention mode over the held-out stream."""

    mode: str
    n_predictions: int
    top1_accuracy: float
    mean_task_loss: float
    top1_by_code_type: dict[str, float] = field(default_factory=dict)
    n_by_code_type: dict[str, int] = field(default_factory=dict)
    n_intervened_positions: int = 0
    """Positions where at least one concept's mixing probability was
    actually replaced (0 for none/zero_* modes, which edit embeddings
    or nothing)."""

    uncertain_band: float | None = None
    """If set, values were only injected where the model's own probability
    was within this distance of 0.5 (see BottleneckIntervention)."""

    mean_abs_displacement: float | None = None
    """Mean ``|injected value - model's own probability|`` over the
    concept entries actually replaced: how far the intervention pushed
    the bottleneck. Truth and flip displace by ``1 - p`` and ``p``
    respectively, so comparing their accuracy deltas without this is
    comparing perturbations of different sizes."""

    calibrated_tau: float | None = None
    """For the *_calibrated modes: the shared peak logit shift every
    concept's step was calibrated to."""

    n_replaced_by_concept: dict[str, int] | None = None
    """Per-concept count of entries actually replaced (W3 band coverage:
    under an uncertain band, a rare concept whose probability hugs its
    base rate rarely enters the band at all -- this is the denominator
    that makes per-concept sensitivity claims honest). None for modes
    that replace nothing."""

    mean_abs_displacement_by_concept: dict[str, float] | None = None
    """Per-concept mean ``|injected - own|`` over that concept's replaced
    entries (NaN for a concept with zero replacements)."""

    calibration_gamma: dict[str, float] | None = None
    """For the *_calibrated modes: the per-concept mixing-probability
    step ``tau / peak_i`` (attached with concept names by
    :func:`evaluate_interventions`)."""


def _chunk_intervention(
    chunk: StreamingChunk,
    mode: str,
    concept_labels: ConceptLabelDict,
    concept_mask: ConceptLabelDict,
    concept_first_times: ConceptLabelDict,
    *,
    supervision: ConceptSupervision,
    num_concepts: int,
    device: str,
    rng: torch.Generator,
    uncertain_band: float | None = None,
) -> BottleneckIntervention | None:
    """Build the per-position intervention for one chunk, or None."""
    if mode == "none":
        return None
    if mode == "zero_known":
        return BottleneckIntervention(zero_known=True)
    if mode == "zero_unknown":
        return BottleneckIntervention(zero_unknown=True)

    labels, observed = position_running_labels(
        chunk,
        concept_labels,
        concept_mask,
        concept_first_times,
        supervision=supervision,
        num_concepts=num_concepts,
    )
    if mode == "truth":
        values = labels
    elif mode in ("flip", "flip_gated"):
        values = 1.0 - labels
    elif mode == "random":
        values = (torch.rand(labels.shape, generator=rng) < 0.5).float()
    else:
        raise ValueError(f"unknown intervention mode: {mode!r}")
    return BottleneckIntervention(
        probs=values.to(device),
        probs_mask=observed.bool().to(device),
        uncertain_band=uncertain_band,
    )


def run_streaming_intervention(  # noqa: PLR0912, PLR0915 -- one linear scoring pass
    model: ConceptBottleneckSequenceModel,
    events_binned: pl.DataFrame,
    vocab: Vocabulary,
    concept_labels: ConceptLabelDict,
    concept_mask: ConceptLabelDict,
    *,
    mode: str,
    concept_first_times: ConceptLabelDict | None = None,
    supervision: ConceptSupervision = "stay",
    num_lanes: int = 8,
    chunk_size: int = 256,
    device: str = "cuda",
    max_seq_len: int | None = None,
    seed: int = 0,
    uncertain_band: float | None = None,
    per_subject_out: dict[int, list[int]] | None = None,
    calibration_gammas: torch.Tensor | None = None,
    calibrated_tau: float | None = None,
    concept_names: Sequence[str] | None = None,
) -> InterventionResult:
    """Score next-event prediction under one intervention mode.

    ``per_subject_out``, if given, accumulates ``{subject_id: [top1_hits,
    n_predictions]}`` over the pass -- the raw material for a PAIRED
    subject-clustered bootstrap on a truth-vs-flip accuracy delta
    (scripts/intervention_cis.py), which the aggregate numbers alone
    cannot support.

    The identical streaming pass as
    :func:`~odyssey.inference.run_inference.run_streaming_inference`
    (same sampler, same state carrying), with the bottleneck edited per
    :data:`INTERVENTION_MODES`. ``concept_first_times`` (from
    :func:`~odyssey.training.data.build_visit_concept_first_times` or
    its stay-scoped twin) turns the retrospective labels into running
    ones, see :func:`_position_labels`; without it the retrospective
    labels are injected as-is at every position, which is only valid for
    concepts that are constant across the sequence. Deterministic for a
    given ``seed`` (which only the ``random`` mode consumes).
    """
    if mode not in INTERVENTION_MODES:
        raise ValueError(
            f"unknown intervention mode {mode!r}; known: {INTERVENTION_MODES}"
        )
    if mode in CALIBRATED_MODES and calibration_gammas is None:
        raise ValueError(
            f"mode {mode!r} needs calibration_gammas (see "
            "odyssey.inference.concept_attribution.calibrated_gammas)"
        )
    if concept_first_times is None:
        if mode in ("truth", "flip", "flip_gated", *CALIBRATED_MODES):
            logger.warning(
                "[interventions] mode %r without concept_first_times: injecting "
                "retrospective labels at every position (not running labels)",
                mode,
            )
        concept_first_times = {}
    model.eval()
    num_concepts = model.bottleneck.num_concepts
    patients = iter_patient_sequences(
        events_binned,
        vocab,
        max_seq_len=max_seq_len,
    )
    sampler = PackedLaneSampler(
        patients, num_lanes=num_lanes, chunk_size=chunk_size, reset_prob=0.0
    )
    rng = torch.Generator().manual_seed(seed)
    type_lookup = _build_type_lookup(vocab, device)

    n = 0
    top1_hits = 0
    loss_sum = 0.0
    n_intervened = 0
    displacement_sum = 0.0
    n_replaced_entries = 0
    per_concept_n = torch.zeros(num_concepts, dtype=torch.long)
    per_concept_disp = torch.zeros(num_concepts, dtype=torch.float64)
    type_n: dict[int, int] = {}
    type_hits: dict[int, int] = {}

    state = None
    with torch.no_grad():
        for chunk in sampler:
            chunk = _move_chunk_to_device(chunk, device)  # noqa: PLW2901
            intervention = (
                None
                if mode in CALIBRATED_MODES  # built below, from the model's own probs
                else _chunk_intervention(
                    chunk,
                    mode,
                    concept_labels,
                    concept_mask,
                    concept_first_times,
                    supervision=supervision,
                    num_concepts=num_concepts,
                    device=device,
                    rng=rng,
                    uncertain_band=uncertain_band,
                )
            )
            if mode in CALIBRATED_MODES:
                # The backbone runs once; the bottleneck runs twice on its
                # hidden states -- first un-intervened to read the model's
                # own probabilities (the calibrated step is RELATIVE to
                # them), then with the calibrated absolute values. No
                # uncertain band: calibration replaces it as the equalizer.
                assert calibration_gammas is not None  # noqa: S101 -- checked above
                hidden, state = model.backbone(
                    chunk.batch, state=state, reset_mask=chunk.reset_mask
                )
                own_probs = model.bottleneck(hidden).concept_probs
                labels, observed = position_running_labels(
                    chunk,
                    concept_labels,
                    concept_mask,
                    concept_first_times,
                    supervision=supervision,
                    num_concepts=num_concepts,
                )
                pole = labels if mode == "truth_calibrated" else 1.0 - labels
                # calibration_gammas is derived from the model's own LM-head
                # weights, so it lives on the model's device, while `pole`
                # comes from the running labels on CPU. Match BOTH device and
                # dtype: `.to(dtype)` alone silently worked in CPU-only tests
                # (where the two already agree) and raised a device mismatch
                # on every GPU run, so no calibrated mode had ever completed.
                offsets = (2.0 * pole - 1.0) * calibration_gammas.to(
                    device=pole.device, dtype=pole.dtype
                )
                values = (own_probs + offsets.to(device)).clamp(0.0, 1.0)
                intervention = BottleneckIntervention(
                    probs=values, probs_mask=observed.bool().to(device)
                )
                bottleneck_out = model.bottleneck(hidden, intervention=intervention)
                logits = model.lm_head(bottleneck_out.bottleneck)
            elif mode == "flip_gated":
                # Two forwards from the SAME input state: the intervention
                # edits only the post-backbone bottleneck mixing, so both
                # calls produce identical hidden states and new_state; the
                # gate then keeps only the flip's suppressive logit changes.
                state_in = state
                base_logits, bottleneck_out, state = model(
                    chunk.batch, state=state_in, reset_mask=chunk.reset_mask
                )
                flip_logits, _, _ = model(
                    chunk.batch,
                    state=state_in,
                    reset_mask=chunk.reset_mask,
                    intervention=intervention,
                )
                logits = base_logits + torch.clamp_max(flip_logits - base_logits, 0.0)
            else:
                logits, bottleneck_out, state = model(
                    chunk.batch,
                    state=state,
                    reset_mask=chunk.reset_mask,
                    intervention=intervention,
                )
            real = chunk.real_mask
            if intervention is not None and intervention.probs is not None:
                own = bottleneck_out.concept_probs
                applied = intervention_apply_mask(intervention, own)
                if applied is None:
                    applied = torch.ones_like(own, dtype=torch.bool)
                input_real = chunk.subject_ids != NO_SUBJECT
                applied = applied & input_real.unsqueeze(-1)
                n_intervened += int(applied.any(dim=-1).sum().item())
                n_replaced_entries += int(applied.sum().item())
                abs_diff = (intervention.probs.expand_as(own) - own).abs()
                displacement_sum += float(abs_diff[applied].sum().item())
                lead_dims = tuple(range(applied.dim() - 1))
                per_concept_n += applied.sum(dim=lead_dims).long().cpu()
                per_concept_disp += (
                    (abs_diff * applied).sum(dim=lead_dims).double().cpu()
                )
            if not real.any():
                continue
            real_logits = logits[real]
            real_targets = chunk.targets[real]
            n += int(real_targets.shape[0])
            preds = real_logits.argmax(dim=-1)
            hits = preds == real_targets
            top1_hits += int(hits.sum().item())
            if per_subject_out is not None:
                sids = chunk.subject_ids[real]
                for sid in torch.unique(sids).tolist():
                    sel = sids == sid
                    entry = per_subject_out.setdefault(int(sid), [0, 0])
                    entry[0] += int(hits[sel].sum().item())
                    entry[1] += int(sel.sum().item())
            loss_sum += float(
                F.cross_entropy(
                    real_logits, real_targets, ignore_index=PAD_ID, reduction="sum"
                ).item()
            )
            target_types = type_lookup[real_targets]
            for type_id in torch.unique(target_types).tolist():
                sel = target_types == type_id
                type_n[type_id] = type_n.get(type_id, 0) + int(sel.sum().item())
                type_hits[type_id] = type_hits.get(type_id, 0) + int(
                    hits[sel].sum().item()
                )

    names = (
        list(concept_names)
        if concept_names is not None
        else [f"concept_{i}" for i in range(num_concepts)]
    )
    if len(names) != num_concepts:
        raise ValueError(f"{len(names)} concept names for {num_concepts} concepts")
    by_concept_n: dict[str, int] | None = None
    by_concept_disp: dict[str, float] | None = None
    if n_replaced_entries:
        by_concept_n = {
            names[i]: int(per_concept_n[i].item()) for i in range(num_concepts)
        }
        by_concept_disp = {
            names[i]: (
                float(per_concept_disp[i].item() / per_concept_n[i].item())
                if per_concept_n[i]
                else float("nan")
            )
            for i in range(num_concepts)
        }

    return InterventionResult(
        mode=mode,
        n_predictions=n,
        top1_accuracy=top1_hits / n if n else float("nan"),
        mean_task_loss=loss_sum / n if n else float("nan"),
        top1_by_code_type={
            _CODE_TYPE_NAMES[tid]: type_hits[tid] / type_n[tid]
            for tid in sorted(type_n)
            if tid in _CODE_TYPE_NAMES
        },
        n_by_code_type={
            _CODE_TYPE_NAMES[tid]: type_n[tid]
            for tid in sorted(type_n)
            if tid in _CODE_TYPE_NAMES
        },
        n_intervened_positions=n_intervened,
        uncertain_band=None if mode in CALIBRATED_MODES else uncertain_band,
        mean_abs_displacement=(
            displacement_sum / n_replaced_entries if n_replaced_entries else None
        ),
        calibrated_tau=calibrated_tau if mode in CALIBRATED_MODES else None,
        n_replaced_by_concept=by_concept_n,
        mean_abs_displacement_by_concept=by_concept_disp,
    )


def evaluate_interventions(
    run_dir: str | Path,
    held_out_shard_dir: str | Path,
    *,
    modes: Sequence[str] = INTERVENTION_MODES,
    max_shards: int | None = None,
    num_lanes: int = 8,
    chunk_size: int = 256,
    device: str | None = None,
    checkpoint_path: str | Path | None = None,
    seed: int = 0,
    uncertain_band: float | None = None,
    per_subject_out: dict[str, dict[int, list[int]]] | None = None,
    calibrated_tau: float = 1.0,
) -> list[InterventionResult]:
    """End-to-end: load a trained run, score every intervention mode.

    ``per_subject_out``, if given, is filled as ``{mode: {subject_id:
    [top1_hits, n_predictions]}}`` (see
    :func:`run_streaming_intervention`).

    Data preparation matches
    :func:`~odyssey.inference.run_inference.evaluate_run` exactly (same
    normalization, binning, and label scoping from the run's own
    config), so the ``none`` mode is directly comparable to the standard
    evaluation and every other mode is directly comparable to ``none``.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, binner, config = load_run(
        run_dir, device=device, checkpoint_path=checkpoint_path
    )
    if not isinstance(model, ConceptBottleneckSequenceModel):
        raise ValueError(
            "this evaluation needs a concept bottleneck; the run's model_kind is "
            f"{getattr(config, 'model_kind', 'bottleneck')!r}"
        )
    if getattr(config, "backbone", "hybrid") == "transformer":
        raise NotImplementedError(
            "interventions is not yet wired for backbone='transformer': this "
            "is concept-bottleneck-lever tooling, not needed for the backbone "
            "control's own subset-scale comparison (unlike run_inference and "
            "alerts, which are). Extend it only if the transformer backbone "
            "earns longer-term status."
        )

    logger.info("[interventions] loading held-out shards from %s", held_out_shard_dir)
    raw_events = load_meds_shards(held_out_shard_dir, max_shards=max_shards)
    raw_events = maybe_normalize(
        raw_events,
        enabled=getattr(config, "normalize_medications", False),
        source=getattr(config, "source", "mimic_iv"),
    )
    raw_events = maybe_history_recap(
        raw_events, enabled=getattr(config, "history_recap", False)
    )
    source = getattr(config, "source", "mimic_iv")
    activate_sidecars(held_out_shard_dir)
    concepts = concepts_for_source(source, task_set=getattr(config, "task_set", "v1"))
    events_binned = add_value_tokens(raw_events, binner, source=source)

    supervision: ConceptSupervision = getattr(config, "concept_supervision", "stay")
    concept_labels: ConceptLabelDict
    concept_mask: ConceptLabelDict
    concept_first_times: ConceptLabelDict
    if supervision == "visit":
        concept_labels, concept_mask = build_visit_concept_label_dicts(
            raw_events, concepts
        )
        concept_first_times = build_visit_concept_first_times(raw_events, concepts)
    else:
        concept_labels, concept_mask = build_concept_label_dicts(raw_events, concepts)
        concept_first_times = build_concept_first_times(raw_events, concepts)
    del raw_events

    calibration_gammas: torch.Tensor | None = None
    gamma_by_name: dict[str, float] | None = None
    if any(m in CALIBRATED_MODES for m in modes):
        # Only a bottleneck whose per-unit displacement is data dependent
        # needs the estimation pass. The decomposition's displacement is a
        # parameter, so asking for directions there would be a forward
        # pass over the whole split to recover something already stored.
        directions = (
            mean_concept_directions(
                model,
                events_binned,
                vocab,
                num_lanes=num_lanes,
                chunk_size=chunk_size,
                device=device,
            )
            if getattr(model.bottleneck, "needs_calibration_directions", True)
            else None
        )
        calibration_gammas = calibrated_gammas(model, directions, tau=calibrated_tau)
        gamma_by_name = {
            c.name: float(g)
            for c, g in zip(concepts, calibration_gammas.tolist(), strict=True)
        }
        logger.info(
            "[interventions] calibrated gammas (tau=%.3g): %s",
            calibrated_tau,
            {k: round(v, 4) for k, v in gamma_by_name.items()},
        )

    results = []
    for mode in modes:
        logger.info("[interventions] scoring mode %r", mode)
        mode_subjects: dict[int, list[int]] | None = None
        if per_subject_out is not None:
            mode_subjects = per_subject_out.setdefault(mode, {})
        result = run_streaming_intervention(
            model,
            events_binned,
            vocab,
            concept_labels,
            concept_mask,
            mode=mode,
            concept_first_times=concept_first_times,
            supervision=supervision,
            num_lanes=num_lanes,
            chunk_size=chunk_size,
            device=device,
            seed=seed,
            uncertain_band=uncertain_band,
            per_subject_out=mode_subjects,
            calibration_gammas=calibration_gammas,
            calibrated_tau=calibrated_tau,
            concept_names=[c.name for c in concepts],
        )
        if mode in CALIBRATED_MODES:
            result = replace(result, calibration_gamma=gamma_by_name)
        results.append(result)
        baseline = results[0]
        latest = results[-1]
        logger.info(
            "[interventions] %s: top1 %.4f (delta vs none %+0.4f), loss %.4f",
            mode,
            latest.top1_accuracy,
            latest.top1_accuracy - baseline.top1_accuracy,
            latest.mean_task_loss,
        )
    return results


def _main() -> None:
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--held-out-shard-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint filename within --run-dir (default: checkpoint_best.pt).",
    )
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--modes", nargs="*", default=list(INTERVENTION_MODES))
    parser.add_argument(
        "--uncertain-band",
        type=float,
        default=None,
        help=(
            "Only inject truth/flip/random values where the model's own concept "
            "probability is within this distance of 0.5, so truth and flip "
            "displace it equally (a pure direction test)."
        ),
    )
    parser.add_argument(
        "--calibrated-tau",
        type=float,
        default=1.0,
        help=(
            "peak logit shift every concept's step is calibrated to in the "
            "truth_calibrated/flip_calibrated modes (gamma_i = tau / peak_i "
            "over the LM head weights); ignored unless a calibrated mode is "
            "in --modes."
        ),
    )
    parser.add_argument("--num-lanes", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument(
        "--dump-per-subject",
        action="store_true",
        help=(
            "also write <output-json stem>_per_subject.json with "
            "{mode: {subject_id: [top1_hits, n_predictions]}} -- the input "
            "scripts/intervention_cis.py needs for a paired subject-"
            "clustered CI on mode-vs-mode accuracy deltas."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "allow clobbering an existing --output-json file. Protocol-"
            "versioned science outputs are append-only by default -- a "
            "real, irreplaceable row-level dump was lost to a silent "
            "overwrite on 2026-08-22. Pass this only when re-running the "
            "same run/protocol intentionally."
        ),
    )
    args = parser.parse_args()

    out = Path(args.output_json)
    refuse_existing_output(out, overwrite=args.overwrite, kind="interventions")
    run_dir = Path(args.run_dir)
    per_subject: dict[str, dict[int, list[int]]] | None = (
        {} if args.dump_per_subject else None
    )
    results = evaluate_interventions(
        run_dir,
        args.held_out_shard_dir,
        modes=args.modes,
        max_shards=args.max_shards,
        num_lanes=args.num_lanes,
        chunk_size=args.chunk_size,
        checkpoint_path=run_dir / (args.checkpoint or "checkpoint_best.pt"),
        uncertain_band=args.uncertain_band,
        per_subject_out=per_subject,
        calibrated_tau=args.calibrated_tau,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([asdict(r) for r in results], indent=2))
    logger.info("[interventions] wrote %d modes to %s", len(results), out)
    if per_subject is not None:
        ps_out = out.with_name(out.stem + "_per_subject.json")
        refuse_existing_output(
            ps_out, overwrite=args.overwrite, kind="interventions per-subject"
        )
        ps_out.write_text(json.dumps(per_subject))
        logger.info(
            "[interventions] wrote per-subject outcomes for %d modes to %s",
            len(per_subject),
            ps_out,
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _main()


__all__ = [
    "INTERVENTION_MODES",
    "InterventionResult",
    "run_streaming_intervention",
    "evaluate_interventions",
]
