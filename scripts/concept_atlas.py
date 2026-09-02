"""What the concepts say: exact alignment of every concept direction with the vocabulary.

The next-event head is linear in the bottleneck sum, so for concept ``c``
with embedding ``K_c`` (or unknown ``U_j``) the vector ``W K_c`` is exactly
how much one unit of that concept raises every next-event logit: Steerling's
Known Concept Alignment, without a data pass. This script writes, for every
known and unknown concept, the top promoted and suppressed events with their
MEDS descriptions, plus one streaming pass over held-out shards for Steerling's
Concept Contribution: the share of each next-event logit owed to the named
part, the unknown part and the residual (exact for the linear head; the
hazard heads are MLPs and are excluded on purpose).

The output is the raw material for the "unknown concept atlas" figure: a
reader can look at what an unnamed concept promotes and judge whether it is a
clinical state the registry lacks.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import polars as pl
import torch

from odyssey.data.code_normalization import maybe_normalize
from odyssey.data.concepts import concepts_for_source
from odyssey.data.history_recap import maybe_history_recap
from odyssey.data.sidecars import activate_sidecars
from odyssey.data.streaming import PackedLaneSampler, move_to_device
from odyssey.data.value_binning import add_value_tokens
from odyssey.inference.run_inference import load_run, refuse_existing_output
from odyssey.inference.steering import token_descriptions
from odyssey.models.concept_bottleneck import DecomposedConceptBottleneck
from odyssey.models.sequence_model import ConceptBottleneckSequenceModel
from odyssey.training.data import iter_patient_sequences, load_meds_shards


logger = logging.getLogger(__name__)


def alignment_table(
    weight: torch.Tensor,
    embeddings: torch.Tensor,
    id_to_token: dict[int, str],
    names: dict[str, str],
    *,
    top: int,
) -> list[dict[str, Any]]:
    """Per concept row: top promoted / suppressed events by ``W e_c``."""
    unit = embeddings / embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    align = unit @ weight.T  # (concepts, vocab)
    rows = []
    for c in range(align.shape[0]):
        up = torch.topk(align[c], k=top)
        down = torch.topk(-align[c], k=top)
        rows.append(
            {
                "index": c,
                "norm": float(embeddings[c].norm()),
                "promotes": [
                    {
                        "token": id_to_token[int(i)],
                        "name": names.get(id_to_token[int(i)], id_to_token[int(i)]),
                        "shift": float(v),
                    }
                    for v, i in zip(
                        up.values.tolist(), up.indices.tolist(), strict=True
                    )
                ],
                "suppresses": [
                    {
                        "token": id_to_token[int(i)],
                        "name": names.get(id_to_token[int(i)], id_to_token[int(i)]),
                        "shift": float(-v),
                    }
                    for v, i in zip(
                        down.values.tolist(), down.indices.tolist(), strict=True
                    )
                ],
            }
        )
    return rows


@torch.no_grad()
def contribution_pass(
    model: ConceptBottleneckSequenceModel,
    events_binned: pl.DataFrame,
    vocab: Any,
    *,
    num_lanes: int,
    chunk_size: int,
    device: str,
) -> dict[str, Any]:
    """Compute Steerling's Concept Contribution on the linear head, plus activations.

    For each real position and its predicted token ``y``: the logit is
    ``k_hat.W_y + u_hat.W_y + eps.W_y``; the shares are the absolute terms
    over their sum. Also the mean activation of every known and unknown
    concept, which says which unknown concepts the model actually uses.
    """
    model.eval()
    bottleneck = model.bottleneck
    assert isinstance(bottleneck, DecomposedConceptBottleneck)  # noqa: S101
    weight = model.lm_head.weight.detach()
    sampler = PackedLaneSampler(
        iter_patient_sequences(events_binned, vocab),
        num_lanes=num_lanes,
        chunk_size=chunk_size,
        reset_prob=0.0,
    )
    shares = torch.zeros(3, dtype=torch.float64)
    n = 0
    known_act = torch.zeros(bottleneck.num_concepts, dtype=torch.float64)
    unknown_act = torch.zeros(bottleneck.num_unknown, dtype=torch.float64)
    state = None
    for chunk in sampler:
        chunk = move_to_device(chunk, device)  # noqa: PLW2901
        logits, out, state = model(
            chunk.batch, state=state, reset_mask=chunk.reset_mask
        )
        real = chunk.real_mask
        if not real.any():
            continue
        pred = logits[real].argmax(dim=-1)
        head_rows = weight[pred]  # (N, d)
        parts = torch.stack(
            [
                (out.known_part[real] * head_rows).sum(-1).abs(),
                (out.unknown_embedding[real] * head_rows).sum(-1).abs(),
                (out.residual[real] * head_rows).sum(-1).abs(),
            ]
        )  # (3, N)
        shares += (
            (parts / parts.sum(0, keepdim=True).clamp_min(1e-12)).sum(1).double().cpu()
        )
        n += int(real.sum())
        known_act += out.concept_probs[real].sum(0).double().cpu()
        unknown_act += out.unknown_probs[real].sum(0).double().cpu()
    return {
        "n_positions": n,
        "contribution_share": {
            "named": float(shares[0] / n),
            "unknown": float(shares[1] / n),
            "residual": float(shares[2] / n),
        },
        "mean_known_activation": (known_act / n).tolist(),
        "mean_unknown_activation": (unknown_act / n).tolist(),
    }


def main() -> None:
    """Write the concept atlas JSON."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--held-out-shard-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--metadata-dir", default=None)
    parser.add_argument("--max-shards", type=int, default=1)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--num-lanes", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    refuse_existing_output(
        Path(args.output_json), overwrite=args.overwrite, kind="concept atlas"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab, binner, config = load_run(
        args.run_dir,
        device=device,
        checkpoint_path=Path(args.run_dir) / args.checkpoint
        if args.checkpoint
        else None,
    )
    if not isinstance(model, ConceptBottleneckSequenceModel) or not isinstance(
        model.bottleneck, DecomposedConceptBottleneck
    ):
        raise SystemExit("the atlas needs a decomposed-bottleneck run")
    source = getattr(config, "source", "mimic_iv")
    names = [
        c.name
        for c in concepts_for_source(source, task_set=getattr(config, "task_set", "v1"))
    ]
    metadata_dir = args.metadata_dir
    if metadata_dir is None:
        candidate = Path(args.held_out_shard_dir).resolve().parent.parent / "metadata"
        metadata_dir = str(candidate) if candidate.exists() else None
    head_weight = model.lm_head.weight.detach().float().cpu()
    bottleneck = model.bottleneck
    known = bottleneck.known_embeddings.detach().float().cpu()
    unknown = bottleneck.unknown_embeddings().detach().float().cpu()
    token_names = token_descriptions(list(vocab.token_to_id), metadata_dir)
    known_rows = alignment_table(
        head_weight, known, vocab.id_to_token, token_names, top=args.top
    )
    for row, name in zip(known_rows, names, strict=True):
        row["name"] = name
    unknown_rows = alignment_table(
        head_weight, unknown, vocab.id_to_token, token_names, top=args.top
    )
    for row in unknown_rows:
        row["name"] = f"unknown_{row['index']}"

    raw = load_meds_shards(args.held_out_shard_dir, max_shards=args.max_shards)
    raw = maybe_normalize(
        raw, enabled=getattr(config, "normalize_medications", False), source=source
    )
    raw = maybe_history_recap(raw, enabled=getattr(config, "history_recap", False))
    activate_sidecars(args.held_out_shard_dir)
    events_binned = add_value_tokens(raw, binner, source=source)
    del raw
    stats = contribution_pass(
        model,
        events_binned,
        vocab,
        num_lanes=args.num_lanes,
        chunk_size=args.chunk_size,
        device=device,
    )
    for row, act in zip(known_rows, stats["mean_known_activation"], strict=True):
        row["mean_activation"] = act
    for row, act in zip(unknown_rows, stats["mean_unknown_activation"], strict=True):
        row["mean_activation"] = act
    payload = {
        "run_dir": str(args.run_dir),
        "source": source,
        "n_positions": stats["n_positions"],
        "contribution_share": stats["contribution_share"],
        "known": known_rows,
        "unknown": sorted(unknown_rows, key=lambda r: -r["mean_activation"]),
    }
    Path(args.output_json).write_text(json.dumps(payload, indent=1))
    share = stats["contribution_share"]
    logger.info(
        "contribution shares: named %.3f unknown %.3f residual %.3f over %d positions",
        share["named"],
        share["unknown"],
        share["residual"],
        stats["n_positions"],
    )
    for row in payload["unknown"][:5]:
        logger.info(
            "%s (act %.2f) promotes %s",
            row["name"],
            row["mean_activation"],
            [p["name"] for p in row["promotes"][:6]],
        )


if __name__ == "__main__":
    main()
