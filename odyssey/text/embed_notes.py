"""Pooled frozen-encoder embeddings for the notes sidecar (Track A item 7).

Turns ``sidecars/notes.parquet`` (``scripts/build_mimic_note_sidecar.py``)
into ``sidecars/note_embeddings.parquet``: one row per note with a pooled
hidden-state vector from a frozen Hugging Face encoder, plus a PCA-reduced
copy small enough to hand to the tuned GBM as extra alert features (the
probe-gated headroom test that decides whether any fusion work is
warranted). The encoder choice is a CLI argument: the program's standing
decision is MedGemma for every site (``google/medgemma-4b-it``, the only
text model available on the governed site), with one strong dedicated
embedder run alongside on MIMIC to separate "text carries no signal" from
"this encoder is weak".

Pooling: mean of the last hidden layer over non-padding tokens; notes
longer than ``--max-tokens`` are split into windows, each pooled, and the
window means averaged with token-count weights (a discharge summary is
~2.5k tokens; radiology ~300). Works for encoder models (BERT-like) and
decoder-only models (Gemma) alike: ``output_hidden_states=True`` and the
last layer is used either way. PCA (``--pca``) is fit on all pooled
vectors once (unsupervised, label-free) and the components are saved next
to the output so the same projection applies to any later notes.

``transformers`` is an optional extra (``uv sync --extra text``); the
import is deferred to :func:`load_encoder` so nothing else in the package
depends on it.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl
import torch


logger = logging.getLogger(__name__)

EMBEDDING_COL = "embedding"
PCA_COL = "embedding_pca"


def load_encoder(
    model_name: str, *, device: str, dtype: str = "bfloat16"
) -> Tuple[Any, Any]:
    """Load tokenizer + base model (deferred ``transformers`` import)."""
    try:
        from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "note embedding needs the optional 'text' extra: uv sync --extra text"
        ) from exc
    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch_dtype)
    model.to(device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def _windows(ids: List[int], max_tokens: int) -> List[List[int]]:
    """Split a token id list into consecutive windows of at most ``max_tokens``."""
    if len(ids) <= max_tokens:
        return [ids]
    return [ids[i : i + max_tokens] for i in range(0, len(ids), max_tokens)]


@torch.no_grad()
def pool_last_hidden(
    model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Mean of the last hidden layer over attended tokens, ``(batch, dim)``."""
    out = model(
        input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
    )
    hidden = (
        out.hidden_states[-1]
        if getattr(out, "hidden_states", None)
        else out.last_hidden_state
    )
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1.0)
    pooled: torch.Tensor = (summed / counts).float()
    return pooled


@torch.no_grad()
def embed_texts(
    texts: Sequence[str],
    tokenizer: Any,
    model: Any,
    *,
    device: str,
    max_tokens: int = 1024,
    batch_size: int = 8,
) -> np.ndarray:
    """Embed ``texts``: windowed mean pooling, token-weighted across windows."""
    # tokenize everything once (no special tokens added per window; the
    # model's own BOS is prepended to each window below if it has one)
    encoded = tokenizer(list(texts), add_special_tokens=False, truncation=False)[
        "input_ids"
    ]
    bos = getattr(tokenizer, "bos_token_id", None)
    windows: List[Tuple[int, List[int]]] = []
    for i, ids in enumerate(encoded):
        for w in _windows(ids, max_tokens - (1 if bos is not None else 0)):
            windows.append((i, ([bos] if bos is not None else []) + w))
    # sort windows by length for efficient batching
    order = sorted(range(len(windows)), key=lambda k: len(windows[k][1]))
    dim: Optional[int] = None
    sums: List[Optional[np.ndarray]] = [None] * len(texts)
    weights = np.zeros(len(texts), dtype=np.float64)
    for start in range(0, len(order), batch_size):
        batch = [windows[k] for k in order[start : start + batch_size]]
        longest = max(len(w) for _, w in batch)
        pad = tokenizer.pad_token_id
        ids = torch.full((len(batch), longest), pad, dtype=torch.long)
        mask = torch.zeros((len(batch), longest), dtype=torch.long)
        for r, (_, w) in enumerate(batch):
            ids[r, : len(w)] = torch.tensor(w, dtype=torch.long)
            mask[r, : len(w)] = 1
        pooled = pool_last_hidden(model, ids.to(device), mask.to(device)).cpu().numpy()
        if dim is None:
            dim = int(pooled.shape[1])
        for r, (i, w) in enumerate(batch):
            n = float(len(w))
            prev = sums[i]
            sums[i] = pooled[r] * n if prev is None else prev + pooled[r] * n
            weights[i] += n
    assert dim is not None  # noqa: S101 -- at least one text was embedded
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, s in enumerate(sums):
        if s is not None:
            out[i] = s / max(weights[i], 1.0)
    return out


def fit_pca(
    vectors: np.ndarray, n_components: int, *, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mean, components) of a PCA fit on ``vectors`` (rows = notes)."""
    from sklearn.decomposition import PCA  # noqa: PLC0415

    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(vectors)
    logger.info(
        "[embed] PCA %d -> %d dims, explained variance %.3f",
        vectors.shape[1],
        n_components,
        float(pca.explained_variance_ratio_.sum()),
    )
    return pca.mean_.astype(np.float32), pca.components_.astype(np.float32)


def apply_pca(
    vectors: np.ndarray, mean: np.ndarray, components: np.ndarray
) -> np.ndarray:
    """Project ``vectors`` with a saved PCA (``(vectors - mean) @ components.T``)."""
    projected: np.ndarray = ((vectors - mean) @ components.T).astype(np.float32)
    return projected


def embed_notes_table(
    notes: pl.DataFrame,
    tokenizer: Any,
    model: Any,
    *,
    device: str,
    max_tokens: int,
    batch_size: int,
    chunk_rows: int = 2000,
) -> np.ndarray:
    """Embed every row of ``notes`` (column ``text``) in chunks -> ``(N, dim)``."""
    texts = notes["text"].to_list()
    parts: List[np.ndarray] = []
    t0 = time.time()
    for start in range(0, len(texts), chunk_rows):
        parts.append(
            embed_texts(
                texts[start : start + chunk_rows],
                tokenizer,
                model,
                device=device,
                max_tokens=max_tokens,
                batch_size=batch_size,
            )
        )
        done = min(start + chunk_rows, len(texts))
        rate = done / max(time.time() - t0, 1e-6)
        logger.info("[embed] %d/%d notes (%.1f notes/s)", done, len(texts), rate)
    return np.concatenate(parts) if parts else np.zeros((0, 0), dtype=np.float32)


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed the notes sidecar with a frozen encoder."
    )
    parser.add_argument("--notes", required=True, help="sidecars/notes.parquet")
    parser.add_argument(
        "--output", required=True, help="sidecars/note_embeddings.parquet"
    )
    parser.add_argument("--model", default="google/medgemma-4b-it")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    parser.add_argument("--pca", type=int, default=64)
    parser.add_argument("--max-notes", type=int, default=None, help="debug cap")
    parser.add_argument("--note-types", nargs="+", default=None, help="e.g. RR AR DS")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    notes = pl.read_parquet(args.notes)
    if args.note_types:
        notes = notes.filter(pl.col("note_type").is_in(args.note_types))
    if args.max_notes:
        notes = notes.head(args.max_notes)
    logger.info("[embed] %d notes, model %s on %s", notes.height, args.model, device)
    tokenizer, model = load_encoder(args.model, device=device, dtype=args.dtype)
    vectors = embed_notes_table(
        notes,
        tokenizer,
        model,
        device=device,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    columns = {
        "note_id": notes["note_id"],
        "subject_id": notes["subject_id"],
        "hadm_id": notes["hadm_id"],
        "note_type": notes["note_type"],
        "charttime": notes["charttime"],
        EMBEDDING_COL: pl.Series(
            [row.astype(np.float16).tolist() for row in vectors],
            dtype=pl.List(pl.Float32),
        ),
    }
    meta = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "dim": int(vectors.shape[1]),
    }
    if args.pca and vectors.shape[0] > args.pca:
        mean, comps = fit_pca(vectors, args.pca)
        reduced = apply_pca(vectors, mean, comps)
        columns[PCA_COL] = pl.Series(
            [r.tolist() for r in reduced], dtype=pl.List(pl.Float32)
        )
        np.savez(out.with_suffix(".pca.npz"), mean=mean, components=comps)
        meta["pca"] = args.pca
    pl.DataFrame(columns).write_parquet(out)
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("[embed] wrote %s (%s)", out, meta)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    _main()
