"""Exact additive attribution and alignment for the concept bottleneck.

The LM head is one ``Linear`` over the concatenated slot embeddings, so
every token logit decomposes exactly, per position, into one term per
known concept plus one for the unknown (residual) slot plus a shared
bias. Two analyses read that decomposition directly, adapted from Guide
Labs' Steerling evaluation (arXiv:2608.07594); both run on an existing
checkpoint, no retraining and no do()-style edit:

- **Concept Contribution** (their Eq. 22 adapted): over held-out
  predictions, the fraction of the (absolute) slot attribution to the
  predicted token that flows through the known concepts,
  ``sum_i |e_i . W_y,i| / (sum_i |e_i . W_y,i| + |e_u . W_y,u|)``.
  An exact, ablation-free counterpart to the ``zero_known`` /
  ``zero_unknown`` completeness probes: those feed the head an
  all-zero slot it never saw in training (an OOD edit); this reads the
  head's own linear arithmetic on unedited activations. Reported
  beside the zeroing 2x2 for one paper cycle, not instead of it.

- **Known Concept Alignment** (their ``T_k(c) = TopK(W K_c)``
  adapted): raising concept ``i``'s mixing probability by ``delta``
  moves its slot embedding by exactly ``delta * (w+ - w-)``, so
  ``W_i (w+ - w-)`` is the per-token logit shift a concept override
  applies -- the vocabulary each concept's lever actually promotes.
  If overriding ``on_vasopressors`` does not promote
  norepinephrine-family tokens, the lever cannot express its concept
  regardless of how the bottleneck was trained. With
  ``concept_global_pairs`` the direction is a parameter (exact,
  input-independent); with context pairs it varies per position and is
  summarized by its mean over held-out real positions, which is the
  direction of the *average* override's logit shift (per-position
  variation is not captured; the global-pairs runs make it exact).

The shared LM-head bias is excluded throughout: it is one vector for
all slots and attributable to none of them.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import polars as pl
import torch

from odyssey.data.code_normalization import maybe_normalize
from odyssey.data.concepts import concepts_for_source
from odyssey.data.history_recap import maybe_history_recap
from odyssey.data.sidecars import activate_sidecars
from odyssey.data.streaming import PackedLaneSampler
from odyssey.data.value_binning import add_value_tokens
from odyssey.data.vocabulary import Vocabulary
from odyssey.inference.run_inference import load_run, refuse_existing_output
from odyssey.models.sequence_model import ConceptBottleneckSequenceModel
from odyssey.training.data import iter_patient_sequences, load_meds_shards
from odyssey.training.train import _move_chunk_to_device


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConceptAlignment:
    """The token logits one concept's override direction moves most."""

    concept: str
    activate_promotes: list[tuple[str, float]]
    """Top-k (token, logit shift per unit of mixing probability) with the
    LARGEST shift: what pushing the concept toward active promotes."""

    deactivate_promotes: list[tuple[str, float]]
    """Top-k with the most NEGATIVE shift, sign flipped in the report:
    what pushing the concept toward inactive promotes."""


@dataclass(frozen=True)
class AttributionResult:
    """Concept Contribution + Known Concept Alignment over a held-out stream."""

    n_predictions: int
    mean_concept_contribution: float
    """W9 headline: mean over held-out predictions of the known-slot
    share of absolute attribution to the predicted token."""

    per_concept_share: dict[str, float] = field(default_factory=dict)
    """Mean share of total absolute attribution (unknown slot included in
    the denominator) carried by each known concept."""

    unknown_share: float = float("nan")
    """Mean share carried by the unknown (residual) slot."""

    direction_source: str = "mean_context_pairs"
    """"global_pairs" (exact, input-independent) or "mean_context_pairs"
    (mean of per-position directions over the same held-out stream)."""

    alignment: list[ConceptAlignment] = field(default_factory=list)

    mean_abs_concept_unknown_correlation: float = float("nan")
    """W11, Guide Labs' concept-independence measure adapted: mean over
    (concept, unknown dimension) pairs of the absolute Pearson
    correlation between the concept's own probability and that unknown
    embedding coordinate, over held-out real positions. A cheap, citable
    second measure beside the capacity-controlled linear probes (which
    stay primary): 0 = the residual channel carries no linear trace of
    the known concepts."""

    per_concept_unknown_correlation: dict[str, float] = field(default_factory=dict)
    """Mean absolute correlation with the unknown coordinates, per concept."""


def run_streaming_attribution(  # noqa: PLR0915 -- one linear scoring pass
    model: ConceptBottleneckSequenceModel,
    events_binned: pl.DataFrame,
    vocab: Vocabulary,
    concept_names: list[str],
    *,
    top_k: int = 20,
    num_lanes: int = 8,
    chunk_size: int = 256,
    device: str = "cuda",
    max_seq_len: int | None = None,
) -> AttributionResult:
    """One streaming pass computing both metrics (no labels needed).

    The identical sampler/state-carrying pass as the intervention and
    standard evaluations, but un-intervened: per real position it takes
    the model's own predicted token, splits that token's logit into
    per-slot terms through the LM head weight, and accumulates the
    contribution shares; per-concept ``w+ - w-`` directions are
    accumulated for the alignment summary.
    """
    model.eval()
    k = model.bottleneck.num_concepts
    d = model.bottleneck.embedding_dim
    if len(concept_names) != k:
        raise ValueError(
            f"{len(concept_names)} concept names for {k} bottleneck concepts"
        )
    patients = iter_patient_sequences(events_binned, vocab, max_seq_len=max_seq_len)
    sampler = PackedLaneSampler(
        patients, num_lanes=num_lanes, chunk_size=chunk_size, reset_prob=0.0
    )
    weight = model.lm_head.weight  # (vocab, k*d + unknown_dim)

    u = model.bottleneck.unknown_dim
    n = 0
    contribution_sum = 0.0
    share_sum = torch.zeros(k, dtype=torch.float64)
    unknown_share_sum = 0.0
    direction_sum = torch.zeros(k, d, dtype=torch.float64)
    n_direction = 0
    # Running moments for the concept-prob / unknown-coordinate Pearson
    # correlations (W11): E[c], E[c^2], E[u], E[u^2], E[c u].
    c_sum = torch.zeros(k, dtype=torch.float64)
    c_sq_sum = torch.zeros(k, dtype=torch.float64)
    u_sum = torch.zeros(u, dtype=torch.float64)
    u_sq_sum = torch.zeros(u, dtype=torch.float64)
    cu_sum = torch.zeros(k, u, dtype=torch.float64)

    state = None
    with torch.no_grad():
        for chunk in sampler:
            chunk = _move_chunk_to_device(chunk, device)  # noqa: PLW2901
            hidden, state = model.backbone(
                chunk.batch, state=state, reset_mask=chunk.reset_mask
            )
            out = model.bottleneck(hidden)
            logits = model.lm_head(out.bottleneck)
            real = chunk.real_mask
            if not real.any():
                continue
            preds = logits[real].argmax(dim=-1)  # (N,)
            w_y = weight[preds]  # (N, k*d + unknown_dim)
            known_terms = torch.einsum(
                "nkd,nkd->nk",
                out.concept_embeddings[real],
                w_y[:, : k * d].view(-1, k, d),
            ).abs()  # (N, k)
            unknown_term = (
                (out.unknown_embedding[real] * w_y[:, k * d :]).sum(-1).abs()
            )  # (N,)
            total = known_terms.sum(-1) + unknown_term
            total = total.clamp_min(torch.finfo(total.dtype).tiny)
            n += int(preds.shape[0])
            contribution_sum += float((known_terms.sum(-1) / total).sum().item())
            share_sum += (known_terms / total.unsqueeze(-1)).sum(0).double().cpu()
            unknown_share_sum += float((unknown_term / total).sum().item())

            directions = model.bottleneck.concept_pair_directions(hidden)
            direction_sum += directions[real].sum(0).double().cpu()
            n_direction += int(preds.shape[0])

            c = out.concept_probs[real].double().cpu()  # (N, k)
            u_emb = out.unknown_embedding[real].double().cpu()  # (N, u)
            c_sum += c.sum(0)
            c_sq_sum += (c**2).sum(0)
            u_sum += u_emb.sum(0)
            u_sq_sum += (u_emb**2).sum(0)
            cu_sum += c.T @ u_emb

    if model.bottleneck.global_pairs:
        diff = (
            model.bottleneck.pair_embeddings[:, 0, :]
            - model.bottleneck.pair_embeddings[:, 1, :]
        )
        mean_direction = diff.detach().double().cpu()
        direction_source = "global_pairs"
    else:
        mean_direction = direction_sum / max(n_direction, 1)
        direction_source = "mean_context_pairs"

    alignment = alignment_from_directions(
        model, mean_direction, vocab, concept_names, top_k=top_k
    )

    if n > 1:
        c_var = (c_sq_sum / n - (c_sum / n) ** 2).clamp_min(0.0)  # (k,)
        u_var = (u_sq_sum / n - (u_sum / n) ** 2).clamp_min(0.0)  # (u,)
        cov = cu_sum / n - torch.outer(c_sum / n, u_sum / n)  # (k, u)
        denom = torch.sqrt(torch.outer(c_var, u_var)).clamp_min(1e-12)
        corr = (cov / denom).clamp(-1.0, 1.0).abs()
        per_concept_corr = {
            name: float(corr[i].mean().item()) for i, name in enumerate(concept_names)
        }
        mean_corr = float(corr.mean().item())
    else:
        per_concept_corr = dict.fromkeys(concept_names, float("nan"))
        mean_corr = float("nan")

    return AttributionResult(
        n_predictions=n,
        mean_concept_contribution=contribution_sum / n if n else float("nan"),
        per_concept_share={
            name: float(share_sum[i].item() / n) if n else float("nan")
            for i, name in enumerate(concept_names)
        },
        unknown_share=unknown_share_sum / n if n else float("nan"),
        direction_source=direction_source,
        alignment=alignment,
        mean_abs_concept_unknown_correlation=mean_corr,
        per_concept_unknown_correlation=per_concept_corr,
    )


def mean_concept_directions(
    model: ConceptBottleneckSequenceModel,
    events_binned: pl.DataFrame,
    vocab: Vocabulary,
    *,
    num_lanes: int = 8,
    chunk_size: int = 256,
    device: str = "cuda",
    max_seq_len: int | None = None,
) -> torch.Tensor:
    """(k, d) per-concept override directions for calibration, float64 CPU.

    Exact (the parameter difference) for a global-pairs bottleneck, with
    no data pass at all; otherwise the mean of the per-position
    ``w+ - w-`` over real positions of the given stream -- the lean
    input the output-calibrated intervention protocol (W7,
    :mod:`odyssey.inference.interventions`) needs, without the full
    attribution accounting.
    """
    model.eval()
    if model.bottleneck.global_pairs:
        diff = (
            model.bottleneck.pair_embeddings[:, 0, :]
            - model.bottleneck.pair_embeddings[:, 1, :]
        )
        return diff.detach().double().cpu()
    k, d = model.bottleneck.num_concepts, model.bottleneck.embedding_dim
    patients = iter_patient_sequences(events_binned, vocab, max_seq_len=max_seq_len)
    sampler = PackedLaneSampler(
        patients, num_lanes=num_lanes, chunk_size=chunk_size, reset_prob=0.0
    )
    direction_sum = torch.zeros(k, d, dtype=torch.float64)
    n = 0
    state = None
    with torch.no_grad():
        for chunk in sampler:
            chunk = _move_chunk_to_device(chunk, device)  # noqa: PLW2901
            hidden, state = model.backbone(
                chunk.batch, state=state, reset_mask=chunk.reset_mask
            )
            real = chunk.real_mask
            if not real.any():
                continue
            directions = model.bottleneck.concept_pair_directions(hidden)
            direction_sum += directions[real].sum(0).double().cpu()
            n += int(real.sum().item())
    return direction_sum / max(n, 1)


def calibrated_gammas(
    model: ConceptBottleneckSequenceModel,
    directions: torch.Tensor,
    *,
    tau: float,
) -> torch.Tensor:
    """(k,) mixing-probability step sizes with equal peak logit shift.

    Guide Labs' output calibration (``gamma = tau / peak(e_c)``) adapted:
    ``peak_i = max_y |W_i (w+ - w-)_i[y]|`` is concept ``i``'s largest
    per-token logit shift per unit of mixing probability, so displacing
    the probability by ``gamma_i = tau / peak_i`` gives every concept the
    same largest achievable logit shift ``tau``, decoupling intervention
    strength from how big the head's weights happen to be per concept.
    """
    if tau <= 0:
        raise ValueError("tau must be positive")
    k = model.bottleneck.num_concepts
    d = model.bottleneck.embedding_dim
    weight = model.lm_head.weight.detach().double().cpu()
    known_weight = weight[:, : k * d].view(-1, k, d)  # (vocab, k, d)
    shifts = torch.einsum("vkd,kd->vk", known_weight, directions.to(known_weight))
    peaks = shifts.abs().amax(dim=0).clamp_min(1e-12)  # (k,)
    return tau / peaks


def alignment_from_directions(
    model: ConceptBottleneckSequenceModel,
    directions: torch.Tensor,
    vocab: Vocabulary,
    concept_names: list[str],
    *,
    top_k: int = 20,
) -> list[ConceptAlignment]:
    """TopK token logit shifts per concept from (k, d) override directions."""
    k = model.bottleneck.num_concepts
    d = model.bottleneck.embedding_dim
    weight = model.lm_head.weight.detach().double().cpu()  # (vocab, k*d + u)
    known_weight = weight[:, : k * d].view(-1, k, d)  # (vocab, k, d)
    shifts = torch.einsum("vkd,kd->vk", known_weight, directions.to(known_weight))
    results = []
    top_k = min(top_k, shifts.shape[0])
    for i, name in enumerate(concept_names):
        per_token = shifts[:, i]
        pos_vals, pos_ids = per_token.topk(top_k)
        neg_vals, neg_ids = (-per_token).topk(top_k)
        results.append(
            ConceptAlignment(
                concept=name,
                activate_promotes=[
                    (vocab.decode(int(t)), float(v))
                    for t, v in zip(pos_ids.tolist(), pos_vals.tolist(), strict=True)
                ],
                deactivate_promotes=[
                    (vocab.decode(int(t)), float(v))
                    for t, v in zip(neg_ids.tolist(), neg_vals.tolist(), strict=True)
                ],
            )
        )
    return results


def evaluate_attribution(
    run_dir: str | Path,
    held_out_shard_dir: str | Path,
    *,
    top_k: int = 20,
    max_shards: int | None = None,
    num_lanes: int = 8,
    chunk_size: int = 256,
    device: str | None = None,
    checkpoint_path: str | Path | None = None,
) -> AttributionResult:
    """End-to-end: load a trained run, compute both attribution metrics.

    Data preparation matches
    :func:`~odyssey.inference.interventions.evaluate_interventions`
    exactly (same normalization, recap, sidecars, and binning from the
    run's own config), minus the concept labels, which neither metric
    needs.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, binner, config = load_run(
        run_dir, device=device, checkpoint_path=checkpoint_path
    )
    if not isinstance(model, ConceptBottleneckSequenceModel):
        raise ValueError(
            "this analysis needs a concept bottleneck; the run's model_kind is "
            f"{getattr(config, 'model_kind', 'bottleneck')!r}"
        )
    if getattr(config, "backbone", "hybrid") == "transformer":
        raise NotImplementedError(
            "concept_attribution is not wired for backbone='transformer': "
            "like interventions, this is concept-bottleneck-lever tooling."
        )

    source = getattr(config, "source", "mimic_iv")
    concepts = concepts_for_source(source, task_set=getattr(config, "task_set", "v1"))
    concept_names = [c.name for c in concepts]

    logger.info("[attribution] loading held-out shards from %s", held_out_shard_dir)
    raw_events = load_meds_shards(held_out_shard_dir, max_shards=max_shards)
    raw_events = maybe_normalize(
        raw_events,
        enabled=getattr(config, "normalize_medications", False),
        source=source,
    )
    raw_events = maybe_history_recap(
        raw_events, enabled=getattr(config, "history_recap", False)
    )
    activate_sidecars(held_out_shard_dir)
    events_binned = add_value_tokens(raw_events, binner, source=source)
    del raw_events

    result = run_streaming_attribution(
        model,
        events_binned,
        vocab,
        concept_names,
        top_k=top_k,
        num_lanes=num_lanes,
        chunk_size=chunk_size,
        device=device,
    )
    logger.info(
        "[attribution] concept contribution %.4f over %d predictions "
        "(unknown share %.4f, directions: %s)",
        result.mean_concept_contribution,
        result.n_predictions,
        result.unknown_share,
        result.direction_source,
    )
    return result


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
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--num-lanes", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "allow clobbering an existing --output-json file (science "
            "outputs are append-only by default, same policy as "
            "interventions)."
        ),
    )
    args = parser.parse_args()

    out = Path(args.output_json)
    refuse_existing_output(out, overwrite=args.overwrite, kind="attribution")
    run_dir = Path(args.run_dir)
    result = evaluate_attribution(
        run_dir,
        args.held_out_shard_dir,
        top_k=args.top_k,
        max_shards=args.max_shards,
        num_lanes=args.num_lanes,
        chunk_size=args.chunk_size,
        checkpoint_path=run_dir / (args.checkpoint or "checkpoint_best.pt"),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(result), indent=2))
    logger.info("[attribution] wrote %s", out)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _main()


__all__ = [
    "AttributionResult",
    "ConceptAlignment",
    "alignment_from_directions",
    "calibrated_gammas",
    "evaluate_attribution",
    "mean_concept_directions",
    "run_streaming_attribution",
]
