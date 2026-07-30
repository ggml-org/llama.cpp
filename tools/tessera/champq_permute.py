#!/usr/bin/env python3
"""CHAMP-Q channel permutation for the Tile640 quantizer.

CHAMP-Q permutes the input channels of a weight matrix so the most
sensitive channels (largest activation magnitude) are grouped together,
then runs the existing AWQ / ternary quantization on the permuted
weight. After quantization, the inverse permutation is applied to the
output so the runtime sees a normal Tile640 tensor (same channel order
as the source). The permutation is computed at calibration time, so the
runtime cost is zero. The output GGUF is bit-compatible with the
non-CHAMP-Q path.

This is the "simple" L2-norm rank permutation. A learned per-layer
permutation via LBFGS that minimizes the BF16-vs-quantized cross-entropy
is future work and is intentionally out of scope here.

The default integration in tools/tile640/quantize_v3.py uses Option A
(see PROJECT-STATUS.md / runtime-aware-pipeline notes): the encoded
Tile640 components are decoded back to a dense F32 weight, the input
dimension is permuted back to the original order, and the un-permuted
weight is re-quantized. The output GGUF is therefore in original channel
order and is interchangeable with the non-CHAMP-Q output. The cost is
one extra quantization pass per tensor; the benefit (when it materialises
on real activations) is a lower per-row ternary error.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

# Tile640 constants (must match ggml C++ + quantize_v3.py).
TILE640_PAGE_SIZE = 640
TILE640_LANE_SIZE = 20
TILE640_LANES_PER_PAGE = 32

# JSON schema for the on-disk policy. Versioned so future
# permutations (e.g. a learned per-layer LBFGS) can extend it.
SCHEMA = "llama.tessera.champq-permute.v1"


# ---------------------------------------------------------------------------
# Permutation helpers
# ---------------------------------------------------------------------------


def compute_champq_permutation(
    arr: np.ndarray,
    act_scales: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return a permutation of the input-channel axis that puts the most
    active channels first.

    Args:
        arr: weight matrix with shape (..., in_dim). The last axis is
            the input-channel axis that will be permuted.
        act_scales: optional per-input-channel activation magnitude with
            shape (in_dim,). When provided, the permutation is sorted by
            the activation observer. When absent, the permutation is
            sorted by the per-channel L2 norm of the weight (a cheap
            proxy for "channel importance" that requires no calibration
            data).

    Returns:
        1-D np.ndarray of length in_dim, dtype int64, a permutation of
        [0, in_dim). Indices are ordered from largest magnitude to
        smallest.
    """
    if arr.ndim < 2:
        raise ValueError(
            f"compute_champq_permutation: arr must be at least 2-D, got {arr.ndim}-D"
        )
    in_dim = arr.shape[-1]
    if act_scales is not None:
        if act_scales.shape != (in_dim,):
            raise ValueError(
                f"act_scales shape {act_scales.shape} does not match in_dim {in_dim}"
            )
        magnitudes = np.asarray(act_scales, dtype=np.float64)
    else:
        flat = arr.reshape(-1, in_dim).astype(np.float64)
        magnitudes = np.linalg.norm(flat, axis=0)
    # argsort descending: largest magnitudes first
    return np.argsort(-magnitudes, kind="stable").astype(np.int64)


def apply_champq_permutation(arr: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Apply the input-channel permutation to a 2-D or 3-D weight array.

    Args:
        arr: weight matrix with shape (..., in_dim). The last axis is
            permuted.
        perm: 1-D int array, length in_dim, a permutation of [0, in_dim).

    Returns:
        A new array with the same shape as arr, with the last axis
        permuted by `perm`.
    """
    if arr.shape[-1] != perm.shape[0]:
        raise ValueError(
            f"apply_champq_permutation: perm length {perm.shape[0]} does not match "
            f"in_dim {arr.shape[-1]}"
        )
    return np.ascontiguousarray(arr[..., perm])


def invert_champq_permutation(perm: np.ndarray) -> np.ndarray:
    """Return the inverse permutation such that
    apply_champq_permutation(apply_champq_permutation(arr, perm),
    inverse(perm)) == arr."""
    perm = np.asarray(perm, dtype=np.int64)
    inverse = np.empty_like(perm)
    inverse[perm] = np.arange(perm.size, dtype=perm.dtype)
    return inverse


# ---------------------------------------------------------------------------
# Tile640 decode (reverse of pack_tile640 + compute_scales + outliers)
# ---------------------------------------------------------------------------


def _unpack_pow3() -> np.ndarray:
    """3^k for k in [0, TILE640_LANE_SIZE). Pre-computed once."""
    return np.array(
        [3 ** i for i in range(TILE640_LANE_SIZE)], dtype=np.uint32
    )


_POW3 = _unpack_pow3()


def decode_tile640_quantized(
    packed: np.ndarray,
    page_scales: np.ndarray,
    lane_scales: np.ndarray,
    outlier_row_offsets: np.ndarray,
    outlier_cols: np.ndarray,
    outlier_vals: np.ndarray,
    out_dim: int,
    in_dim: int,
) -> np.ndarray:
    """Reverse the Tile640 encoding to a dense F32 weight in the
    AWQ-scaled space (i.e. before the input_scale is applied). Callers
    that want the original weight scale must multiply by input_scale
    afterwards.

    Mirrors pack_tile640 + compute_scales + select_repair_residuals in
    tools/tile640/quantize_v3.py. Tested for in_dim that is and is not a
    multiple of TILE640_PAGE_SIZE.
    """
    pages_per_row = (in_dim + TILE640_PAGE_SIZE - 1) // TILE640_PAGE_SIZE
    padded_in_dim = pages_per_row * TILE640_PAGE_SIZE

    # 1. Unpack u32 words to ternary {-1, 0, 1}.
    # packed has shape (out_dim * pages_per_row * 32,) flattened from
    # (out_dim, pages_per_row, 32).
    words = packed.astype(np.uint32).reshape(
        out_dim, pages_per_row, TILE640_LANES_PER_PAGE
    )
    trit_indices = (words[:, :, :, None] // _POW3[None, None, None, :]) % 3
    ternary = np.where(
        trit_indices == 1,
        np.int8(1),
        np.where(trit_indices == 2, np.int8(-1), np.int8(0)),
    )

    # 2. Per-lane scale: page_scale * lane_scale_i8 / 127.
    ps = page_scales.astype(np.float32).reshape(out_dim, pages_per_row)
    ls = lane_scales.astype(np.float32).reshape(
        out_dim, pages_per_row, TILE640_LANES_PER_PAGE
    )
    lane_value_scale = (ps[:, :, None] * ls / np.float32(127.0))[:, :, :, None]

    # 3. Decode: ternary * lane_value_scale.
    decoded = (ternary.astype(np.float32) * lane_value_scale).reshape(
        out_dim, padded_in_dim
    )

    # 4. Add outliers. outlier_cols indices are in [0, in_dim), so they
    # fit inside the padded row.
    for row in range(out_dim):
        start = int(outlier_row_offsets[row])
        end = int(outlier_row_offsets[row + 1])
        if end > start:
            cols = outlier_cols[start:end].astype(np.int64)
            decoded[row, cols] = outlier_vals[start:end].astype(np.float32)

    # 5. Trim padding.
    if padded_in_dim != in_dim:
        decoded = decoded[:, :in_dim]
    return np.ascontiguousarray(decoded)


def decode_q_to_weight(q: Dict[str, np.ndarray], out_dim: int, in_dim: int) -> np.ndarray:
    """Decode a quantize_2d result dict to a dense F32 weight in the
    original weight scale (after the AWQ input_scale is applied)."""
    decoded_scaled = decode_tile640_quantized(
        q["packed"],
        q["page_scales"],
        q["lane_scales"],
        q["outlier_row_offsets"],
        q["outlier_cols"],
        q["outlier_vals"],
        out_dim,
        in_dim,
    )
    input_scale = q["input_scale"].astype(np.float32).reshape(1, -1)
    return decoded_scaled * input_scale


# ---------------------------------------------------------------------------
# Policy dataclass (debug / A-B comparison)
# ---------------------------------------------------------------------------


@dataclass
class CHAMPQPolicy:
    """Per-tensor CHAMP-Q policy. Records the input-channel permutation
    that was applied to each weight, so the output can be reproduced
    or re-applied at load time (future Option B)."""

    schema: str = SCHEMA
    tensors: Dict[str, List[int]] = field(default_factory=dict)

    def add(self, name: str, perm: np.ndarray) -> None:
        self.tensors[name] = np.asarray(perm, dtype=np.int64).tolist()

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                {"schema": self.schema, "tensors": self.tensors},
                handle,
                separators=(",", ":"),
            )

    @staticmethod
    def load(path: str) -> "CHAMPQPolicy":
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if data.get("schema") != SCHEMA:
            raise ValueError(
                f"unsupported CHAMP-Q policy schema: {data.get('schema')!r}"
            )
        return CHAMPQPolicy(schema=data["schema"], tensors=data["tensors"])
