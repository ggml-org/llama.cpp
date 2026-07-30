#!/usr/bin/env python3
"""Identify high-frequency outlier tokens for PrefixQuant KV-prefix injection.

Reads a llama-imatrix GGUF (the same file ``llama-imatrix`` writes), computes
the per-position max activation magnitude for each tensor, flags positions
whose max exceeds the PrefixQuant threshold ``eta`` relative to the median,
and emits a small JSON file describing the most extreme token IDs.

This is the offline identification step of PrefixQuant (arXiv:2410.05265).
The actual KV-prefix injection in the runtime is a separate concern and is
out of scope for this tool.

Schema emitted:
    llama.tessera.kv-prefix-tokens.v1

Usage:
    python tools/tessera/kv_prefix_identifier.py \\
        --imatrix path/to/imatrix.gguf \\
        --output kv_prefix_tokens.json \\
        --model-family gemma4 \\
        --threshold 64
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from gguf import GGUFReader


SCHEMA = "llama.tessera.kv-prefix-tokens.v1"
DEFAULT_ETA = 64
EMBEDDING_TENSOR_CANDIDATES = (
    "token_embd.weight",
    "token_embd",
    "embed_tokens.weight",
    "model.embed_tokens.weight",
    "wte.weight",
    "tok_embeddings.weight",
)


@dataclass(frozen=True)
class TensorStats:
    base_name: str
    in_maxabs: np.ndarray


def import_gguf() -> type[GGUFReader]:
    from gguf import GGUFReader as _Reader
    return _Reader


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def load_imatrix(path: Path) -> list[TensorStats]:
    reader = import_gguf()(str(path), "r")
    grouped: dict[str, dict[str, np.ndarray]] = {}
    for tensor in reader.tensors:
        for suffix in ("in_maxabs", "in_sum2", "in_sumabs", "in_sum4", "counts"):
            marker = f".{suffix}"
            if tensor.name.endswith(marker):
                base = tensor.name[: -len(marker)]
                grouped.setdefault(base, {})[suffix] = np.asarray(
                    tensor.data, dtype=np.float32
                ).reshape(-1)
                break
    stats: list[TensorStats] = []
    for base, channels in grouped.items():
        if "in_maxabs" in channels:
            stats.append(TensorStats(base_name=base, in_maxabs=channels["in_maxabs"]))
    if not stats:
        raise ValueError(f"{path}: no tensors with .in_maxabs found in imatrix")
    return stats


def safe_median(values: np.ndarray) -> float:
    finite = values[np.isfinite(values) & (values > 0.0)]
    if finite.size == 0:
        return 1.0
    return float(np.median(finite))


def count_outliers(values: np.ndarray, eta: float) -> int:
    median = safe_median(values)
    if median <= 0.0:
        return 0
    ratio = values / median
    return int(np.count_nonzero(ratio > eta))


def pick_primary_tensor(stats: list[TensorStats]) -> TensorStats:
    for candidate in EMBEDDING_TENSOR_CANDIDATES:
        for entry in stats:
            if entry.base_name == candidate:
                return entry
    for entry in stats:
        if "token_embd" in entry.base_name or "embed" in entry.base_name:
            return entry
    return max(stats, key=lambda entry: entry.in_maxabs.size)


def select_prefix_tokens(
    primary: TensorStats,
    outlier_count: int,
    eta: float,
) -> list[dict[str, int | float]]:
    values = np.asarray(primary.in_maxabs, dtype=np.float32)
    if values.size == 0:
        return []
    median = safe_median(values)
    if median <= 0.0:
        return []
    ratio = values / median
    outlier_mask = ratio > eta
    outlier_indices = np.flatnonzero(outlier_mask)
    if outlier_indices.size == 0:
        return []
    by_magnitude = outlier_indices[np.argsort(-values[outlier_indices])]
    selected = by_magnitude[:outlier_count]
    tokens: list[dict[str, int | float]] = []
    for rank, idx in enumerate(selected, start=1):
        tokens.append(
            {
                "id": int(idx),
                "frequency": int(round(float(values[idx]))),
                "rank": rank,
            }
        )
    return tokens


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Identify PrefixQuant outlier tokens from a llama-imatrix GGUF"
    )
    parser.add_argument(
        "--imatrix", required=True,
        help="Path to llama-imatrix .gguf file (or any imatrix with .in_maxabs)"
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to the kv_prefix_tokens.json output file"
    )
    parser.add_argument(
        "--model-family", required=True,
        help="Model family tag (e.g., gemma4, qwen3.6). Embedded verbatim in the output."
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_ETA,
        help="PrefixQuant eta threshold (ratio to median). Default: 64 per paper."
    )
    parser.add_argument(
        "--primary-tensor", default=None,
        help="Override the primary tensor used to source token IDs "
             "(default: token_embd.weight or the widest tensor)."
    )
    args = parser.parse_args()

    if args.threshold <= 1.0:
        raise ValueError("--threshold must be greater than 1.0 to flag any outliers")

    log(f"loading imatrix: {args.imatrix}")
    stats = load_imatrix(Path(args.imatrix))
    log(f"loaded {len(stats)} tensors with .in_maxabs")

    per_tensor_counts: list[tuple[str, int]] = []
    for entry in stats:
        outliers = count_outliers(entry.in_maxabs, args.threshold)
        per_tensor_counts.append((entry.base_name, outliers))
    log("per-tensor outlier counts at eta=%.1f:" % args.threshold)
    for name, count in per_tensor_counts:
        log(f"  {name}: {count}")

    outlier_count = max((c for _, c in per_tensor_counts), default=0)
    if outlier_count == 0:
        log("warning: no outliers flagged at the given threshold; emitting empty token list")
    n = max(int(math.ceil(outlier_count)), 0)

    if args.primary_tensor is not None:
        match = next(
            (entry for entry in stats if entry.base_name == args.primary_tensor),
            None,
        )
        if match is None:
            raise ValueError(
                f"--primary-tensor {args.primary_tensor!r} not present in imatrix"
            )
        primary = match
    else:
        primary = pick_primary_tensor(stats)
    log(f"primary tensor: {primary.base_name} (width={primary.in_maxabs.size})")

    tokens = select_prefix_tokens(primary, n, args.threshold)

    payload = {
        "schema": SCHEMA,
        "model_family": args.model_family,
        "outlier_count": n,
        "outlier_threshold": float(args.threshold),
        "primary_tensor": primary.base_name,
        "source_imatrix": str(Path(args.imatrix).resolve()),
        "per_tensor_outlier_counts": [
            {"tensor": name, "outliers": count}
            for name, count in per_tensor_counts
        ],
        "tokens": tokens,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    log(
        f"wrote {output} with {len(tokens)} prefix candidates "
        f"(outlier_count={n}, eta={args.threshold:g})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
