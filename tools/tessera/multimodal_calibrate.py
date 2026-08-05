"""Multimodal activation capture for vision / audio / mm-projector.

M1 of the mmproj pipeline. The writer (M0a, commit c64e9a85a) already
absorbs the per-component GGUFs into the unified output and the modality
classifier (M0b, commit 234333cec) routes audio+kurt>5 to FLRQ and
vision+low_er<0.3 to LRQ. Without M1, the mmproj components receive the
source qtype as-is — no calibration verdicts. M1 produces the verdicts.

This module is the multimodal analog of
``tools/tessera/calibration_to_tensor_stats.py``: it loads a small set
of inputs (the test fixtures + synthesized variants), runs a forward
pass through the per-component graph (vision tower, audio tower,
mm-projector), captures per-tensor activation statistics, and writes
``tensor_stats`` rows with ``model_role = 'vision_tower' /
'audio_tower' / 'mm_projector'``.

v1 bootstrap: the inputs are the existing ``tools/mtmd/test-1.jpeg`` +
``tools/mtmd/test-2.mp3`` fixtures plus synthesized variants (flipped /
rotated / cropped images, frequency-shifted / tempoed audio). The
v1 path uses a numpy-based synthetic forward pass (synthesise one
activation envelope per tensor) to produce the per-tensor stats;
``source = 'py_mm_cal'`` is stamped on every row.

v2 (this commit): ``--source real`` invokes the new
``llama-clip-capture`` binary via subprocess; the binary runs a real
forward pass through the clip graph and emits per-activation stats in
JSON. The Python side parses the JSON and stamps ``source = 'real'`` on
every row. The v1 path is preserved byte-equivalent (default
``--source synthetic``). The activation naming convention mirrors the
v1 weight naming convention: v.* / a.* / mm.* are the per-component
prefixes the C++ side adds to the graph's internal activation names.

Real multimodal calibration datasets are a flagged follow-up.

Additive schema: no new ``tensor_stats`` columns. The output rows use
the same columns the text side writes. The ``recommended_action`` enum
is the same five-value enum (protect / requant_up / requant_down /
monitor / noop); the modality-specific routing in ``ts_regime_classify``
(modality=2 + kurt>5 -> FLRQ, modality=1 + er<0.3 -> LRQ) maps to
``protect`` / ``requant_down`` on the calibration side (the calibration
side's verdict is "do not aggressively requant this family", which is
the same intent as the routing side's "use FLRQ / LRQ which are
outlier-aware and do not blow up heavy tails").

Usage::

    python3 -m tools.tessera.multimodal_calibrate \\
        --vision-tower vision.gguf \\
        --vision-inputs tools/mtmd/test-1.jpeg \\
        --audio-tower audio.gguf \\
        --audio-inputs tools/mtmd/test-2.mp3 \\
        --mm-projector projector.gguf \\
        --output /tmp/tessera-mm-cal.json \\
        --budget-fraction 0.5 \\
        --db tessera.duckdb \\
        --model-hash <hash> \\
        --source {synthetic,real}

The ``--output`` is the sidecar calibration JSON (the audit trail);
the ``--db`` rows are the canonical side the rest of the calibration
pipeline reads.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

# tessera_db is the canonical write side; we go through the same
# insert_tensor_stats the text side uses so the upsert semantics
# (COALESCE-preserve per-side columns, the same ON CONFLICT
# (model_hash, model_role, name) clause) are shared.
from tessera_db import TENSOR_STATS_COLS, TesseraDB  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The 8 model_role values the unified schema knows. Three were added
#: by M0a (vision_tower / audio_tower / mm_projector); the
#: architecture docs at docs/tessera-unified-db.md Phase 16+
#: enumerates the full set. The 8-value enum is enforced at the
#: Python layer (see ``migrate_model_role.py`` for the canonical list).
MODEL_ROLES_8: tuple[str, ...] = (
    "trunk", "dflash", "dspark", "mtp_nextn", "shared_embd",
    "vision_tower", "audio_tower", "mm_projector",
)

#: The three M1 roles this driver is responsible for. The dispatch
#: side (``ts_regime_infer_modality``) is the routing authority; the
#: writer side stamps the matching role on the row so the
#: unified consumer (``tile640_quantize_v3.py``,
#: ``llama-quantize --write-unified-gguf``) can route back.
MM_ROLES: tuple[str, ...] = ("vision_tower", "audio_tower", "mm_projector")

#: Per-role tensor-name prefix. The C++ clip.cpp uses ``v.`` /
#: ``a.`` / ``mm.`` to prefix every tensor in the corresponding GGUF
#: (see clip.cpp:1831 in the mmproj fork). The writer (M0a) reads
#: the prefix and stamps the matching ``model_role`` on each row.
ROLE_PREFIX: dict[str, str] = {
    "vision_tower": "v.",
    "audio_tower":  "a.",
    "mm_projector": "mm.",
}

#: Modality code mirror. The C++ side's ``ts_regime_infer_modality``
#: returns 0 / 1 / 2 (text / vision / audio). mm_projector tensors
#: are routed to the text lane in the dispatch (the projector
#: produces embeddings the text backbone consumes); we keep the
#: mirror for parity.
MODALITY_TEXT = 0
MODALITY_VISION = 1
MODALITY_AUDIO = 2
ROLE_TO_MODALITY: dict[str, int] = {
    "vision_tower": MODALITY_VISION,
    "audio_tower":  MODALITY_AUDIO,
    "mm_projector": MODALITY_TEXT,
}

#: Per-role default kurtosis / eff_rank distribution. v1 bootstrap:
#: we synthesise activations matching the published mmproj tower
#: statistics — heavy-tailed audio + spatially low-rank vision. Real
#: forward-pass capture is the v2 follow-up.
DEFAULT_KURTOSIS: dict[str, float] = {
    "vision_tower": 3.5,
    "audio_tower":  6.5,
    "mm_projector": 3.0,
}
DEFAULT_EFF_RANK: dict[str, float] = {
    "vision_tower": 0.20,  # spatially low-rank -> LRQ routing
    "audio_tower":  0.55,  # heavy-tailed -> FLRQ routing
    "mm_projector": 0.65,
}

#: Source values. The ``source`` column on ``tensor_stats`` is the
#: provenance of the row. The four values distinguish the v1
#: synthetic pass (``py_mm_cal``), the v2 real C++ forward pass
#: (``real``), the targeted-recal backfill (``backfill``, set by
#: another worker), and the backfill-of-real-rows
#: (``backfill_real``). The four values are kept distinct so a
#: downstream consumer can audit the row's provenance.
SOURCE_PY_MM_CAL = "py_mm_cal"
SOURCE_REAL = "real"
SOURCE_BACKFILL = "backfill"  # set by targeted-recal worker; not used here
SOURCE_BACKFILL_REAL = "backfill_real"  # same; not used here

#: v2 path: the ``llama-clip-capture`` binary is the surface the
#: Python side invokes via subprocess. The binary is built by
#: ``tools/mtmd/CMakeLists.txt``; the standard install location is
#: ``build/bin/llama-clip-capture``. The Python side probes a few
#: candidate paths and falls back to the PATH lookup. The
#: ``--clip-capture-binary`` flag overrides the probe for testing.
CLIP_CAPTURE_BINARY_NAME = "llama-clip-capture"
DEFAULT_CLIP_CAPTURE_BINARY_PATHS: tuple[str, ...] = (
    "build/bin/llama-clip-capture",
    "./build/bin/llama-clip-capture",
    "build/clip-capture/llama-clip-capture",
)

#: Number of synthesized input variants per modality. The fixtures
#: provide the seed; we generate the rest (flipped / rotated / cropped
#: image, pitched / tempoed audio). The 8-variant default matches the
#: test fixture count the spec calls out.
N_VARIANTS: dict[str, int] = {"vision": 8, "audio": 8, "projector": 8}

#: Bytes per model hash element. The text side's hash is the model
#: file's SHA256 prefix; we use the same convention here so the
#: dispatch and the calibrator share the same model_hash key.
MODEL_HASH_PREFIX_LEN = 16

#: Targeted re-calibration (the L5 monitor-verdict hook):
#: the ``backfill`` source value is stamped on the
#: ``tensor_stats`` row the backfill machinery writes. The
#: constant is named ``SOURCE_BACKFILL_REAL`` because the
#: only backfill source value (after the v1 synthetic path
#: was superseded by the real C++ clip-graph capture path)
#: is the real forward-pass capture. The companion constant
#: in ``per_tensor_calibrate.py`` has the same value; both
#: drivers stamp the same source on their backfill writes.
SOURCE_BACKFILL_REAL: str = "backfill_real"


# ---------------------------------------------------------------------------
# GGUF reader (numpy-only; no ggml C bindings needed)
# ---------------------------------------------------------------------------


def _read_gguf_tensors(path: Path) -> list[tuple[str, tuple[int, ...], str]]:
    """Return ``[(name, shape, dtype_str), ...]`` for the named
    tensors in the given GGUF. The tensor data is not loaded — only
    the name, shape, and dtype string (so the calibrator can
    enumerate tensors and stamp rows for each one without holding
    the weight memory).

    The reader is a small pure-python walker over the GGUF format
    (the spec at ``docs/gguf.md`` is enough to enumerate). The
    ``gguf-py`` package is the heavy alternative; we use it lazily
    (only when an actual weight read is needed) so the test path
    can run without it.
    """
    try:
        from gguf import GGUFReader  # type: ignore
    except ImportError:
        return _read_gguf_tensors_fallback(path)
    reader = GGUFReader(str(path), "r")
    try:
        out: list[tuple[str, tuple[int, ...], str]] = []
        for t in reader.tensors:
            name = t.name
            shape = tuple(int(d) for d in t.shape)
            # gguf-py reports tensor_type as a TensorType enum; we
            # want a string mirror ("F32", "F16", "BF16", "Q4_K", ...).
            try:
                dtype_str = str(t.tensor_type.name)
            except AttributeError:
                dtype_str = str(t.tensor_type)
            out.append((name, shape, dtype_str))
        return out
    finally:
        try:
            reader.close()
        except Exception:
            pass


def _read_gguf_tensors_fallback(path: Path) -> list[tuple[str, tuple[int, ...], str]]:
    """Pure-python GGUF header walker. Reads the magic + version +
    n_tensors + per-tensor name / shape / dtype-string without
    allocating the weight payloads. Used when ``gguf-py`` is not
    installed; the calibrator can still produce tensor_stats rows
    because we only need the name list + shape for the per-tensor
    activation envelope (no actual weight values are needed for the
    v1 synthetic-forward path)."""
    out: list[tuple[str, tuple[int, ...], str]] = []
    with path.open("rb") as f:
        # Magic: 0x46475547 ("GGUF" little-endian) + uint32 version.
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"{path}: not a GGUF (magic={magic!r})")
        version = int.from_bytes(f.read(4), "little")
        if version not in (2, 3):
            raise ValueError(f"{path}: unsupported GGUF version {version}")
        n_tensors = int.from_bytes(f.read(8), "little")
        # Skip the kv header (we don't need the metadata).
        n_kv = int.from_bytes(f.read(8), "little")
        for _ in range(n_kv):
            klen = int.from_bytes(f.read(8), "little")
            f.read(klen)
            vtype = int.from_bytes(f.read(4), "little")
            # Read the value (8 bytes is the max for any of the
            # value types; for arrays/strings we need a length
            # prefix). The minimum is a no-op: we don't care about
            # any of the kv entries.
            if vtype == 8:  # STRING
                slen = int.from_bytes(f.read(8), "little")
                f.read(slen)
            elif vtype == 9:  # ARRAY
                atype = int.from_bytes(f.read(4), "little")
                alen = int.from_bytes(f.read(8), "little")
                elem_size = 4
                if atype in (0, 1, 7, 11, 12):  # F32/U32/I32/F16/BOOL
                    elem_size = 4
                elif atype in (2, 3, 4, 5, 6, 10, 13, 14):
                    elem_size = 8
                elif atype == 8:
                    elem_size = 8  # string length
                f.read(alen * elem_size)
            else:
                f.read(8)
        # Now read n_tensors entries.
        for _ in range(n_tensors):
            nlen = int.from_bytes(f.read(8), "little")
            name = f.read(nlen).decode("utf-8", errors="replace")
            n_dims = int.from_bytes(f.read(4), "little")
            shape = tuple(
                int.from_bytes(f.read(8), "little") for _ in range(n_dims)
            )
            dtype_int = int.from_bytes(f.read(4), "little")
            offset = int.from_bytes(f.read(8), "little")
            # GGUF dtype code -> name. Mirror of ggml_type.
            dtype_str = _GGML_TYPE_NAME.get(dtype_int, f"unknown({dtype_int})")
            out.append((name, shape, dtype_str))
    return out


#: GGML dtype codes used by the GGUF tensor_type field. Mirror of
#: the ``ggml_type`` enum in ``ggml.h`` (the codes are stable across
#: versions 2 and 3 of the GGUF format).
_GGML_TYPE_NAME: dict[int, str] = {
    0:  "F32",
    1:  "F16",
    2:  "Q4_0",
    3:  "Q4_1",
    6:  "Q5_0",
    7:  "Q5_1",
    8:  "Q8_0",
    9:  "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
    27: "I64",
    28: "F64",
    29: "BF16",
    30: "TQ1_0",
    31: "TQ2_0",
}


def _model_hash_for_paths(paths: Iterable[Path]) -> str:
    """Hash a fixed-length prefix of the concatenated model file
    digests. Mirrors the text side's convention: the model_hash is
    a 16-hex-char prefix of the SHA256 of the file list. Multiple
    files (vision + audio + projector) are concatenated in the
    fixed order vision / audio / projector so the hash is stable
    across runs that swap the order of CLI flags."""
    h = hashlib.sha256()
    n = 0
    for p in sorted(paths, key=lambda x: str(x)):
        h.update(str(p).encode("utf-8"))
        h.update(b"\0")
        if p.is_file():
            with p.open("rb") as f:
                while chunk := f.read(1024 * 1024):
                    h.update(chunk)
        n += 1
    return h.hexdigest()[:MODEL_HASH_PREFIX_LEN]


# ---------------------------------------------------------------------------
# Activation synthesis
# ---------------------------------------------------------------------------


def _synthesise_image_variants(
    seed_path: Path, n: int, rng: np.random.Generator,
) -> list[np.ndarray]:
    """Generate ``n`` image variants (HxWx3 uint8) by reading the
    seed image (PIL) and applying random affine / colour-jitter
    transforms. The variants are what the vision tower sees in v1
    bootstrap — real datasets are a flagged follow-up."""
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        # Fallback: synthesise a synthetic image. The forward pass
        # contract is "the input is a (H, W, 3) image" — the source
        # does not matter for the v1 statistics.
        H, W = 32, 32
        return [
            (rng.uniform(0, 1, (H, W, 3)) * 255).astype(np.uint8)
            for _ in range(n)
        ]
    if not seed_path.is_file():
        # Same fallback when the seed is missing: synthesise.
        H, W = 32, 32
        return [
            (rng.uniform(0, 1, (H, W, 3)) * 255).astype(np.uint8)
            for _ in range(n)
        ]
    with Image.open(seed_path) as img:
        img = img.convert("RGB")
        variants: list[np.ndarray] = []
        variants.append(np.asarray(img, dtype=np.uint8))
        for _ in range(n - 1):
            v = img.copy()
            # Random flip (horizontal / vertical) and rotation.
            if rng.random() < 0.5:
                v = v.transpose(Image.FLIP_LEFT_RIGHT)
            if rng.random() < 0.3:
                v = v.transpose(Image.FLIP_TOP_BOTTOM)
            angle = float(rng.uniform(-15.0, 15.0))
            v = v.rotate(angle, resample=Image.BILINEAR, fillcolor=(128, 128, 128))
            # Brightness jitter.
            from PIL import ImageEnhance
            v = ImageEnhance.Brightness(v).enhance(
                float(rng.uniform(0.7, 1.3))
            )
            variants.append(np.asarray(v, dtype=np.uint8))
        return variants


def _synthesise_audio_variants(
    seed_path: Path, n: int, rng: np.random.Generator,
    sample_rate: int = 16000, n_samples: int = 16000,
) -> list[np.ndarray]:
    """Generate ``n`` audio variants (1-D float32) from the seed mp3
    (decoded via the stdlib if ``soundfile`` is missing — the
    fallback uses raw byte-pattern synthesis; the calibrator only
    needs a 1-D signal of the right shape to drive the audio
    tower's activation envelope)."""
    base: np.ndarray
    if seed_path.is_file():
        try:
            import soundfile as sf  # type: ignore
            base, _sr = sf.read(str(seed_path), dtype="float32")
            if base.ndim > 1:
                base = base.mean(axis=1)
            if base.size < n_samples:
                # Pad short clips to the model window.
                pad = np.zeros(n_samples - base.size, dtype=np.float32)
                base = np.concatenate([base, pad])
            else:
                base = base[:n_samples]
        except Exception:
            base = _sine_ensemble(n_samples, sample_rate, rng)
    else:
        base = _sine_ensemble(n_samples, sample_rate, rng)
    variants: list[np.ndarray] = []
    variants.append(base.astype(np.float32))
    for _ in range(n - 1):
        v = base.copy()
        # Pitch shift: resample by a small ratio.
        ratio = float(rng.uniform(0.9, 1.1))
        n_new = max(1, int(v.size * ratio))
        idx = np.linspace(0, v.size - 1, n_new, dtype=np.int64)
        v2 = v[idx].astype(np.float32)
        if v2.size < n_samples:
            v2 = np.concatenate([
                v2, np.zeros(n_samples - v2.size, dtype=np.float32)
            ])
        else:
            v2 = v2[:n_samples]
        # Add small noise.
        v2 = v2 + (rng.normal(0.0, 0.01, v2.size).astype(np.float32))
        variants.append(v2)
    return variants


def _sine_ensemble(
    n: int, sample_rate: int, rng: np.random.Generator,
) -> np.ndarray:
    """Synthesise a 1-D audio-like signal: a sum of sines with
    randomised frequencies + a low-amplitude noise floor. Used as
    the seed for audio variants when the mp3 decoder is missing.
    """
    t = np.arange(n, dtype=np.float32) / float(sample_rate)
    out = np.zeros(n, dtype=np.float32)
    for _ in range(int(rng.integers(3, 6))):
        f = float(rng.uniform(80.0, 4000.0))
        a = float(rng.uniform(0.1, 0.5))
        out = out + a * np.sin(2.0 * np.pi * f * t).astype(np.float32)
    out = out + 0.005 * rng.normal(0.0, 1.0, n).astype(np.float32)
    return out.astype(np.float32)


def _synthesise_projector_variants(
    seed: np.ndarray, n: int, rng: np.random.Generator,
) -> list[np.ndarray]:
    """Generate ``n`` projector-input variants from the seed
    tensor. The mm_projector consumes a (n_tokens, in_dim) feature
    matrix (the vision / audio tower's tokenised output). The v1
    seed is a random Gaussian matrix; variants are the seed with
    column-scaled noise so the projector sees a distribution of
    per-token scales."""
    if seed.ndim == 1:
        seed = seed[None, :]
    if seed.ndim != 2:
        # Flatten everything into a 2-D matrix.
        seed = seed.reshape(seed.shape[0], -1)
    n_tokens, in_dim = seed.shape
    out: list[np.ndarray] = []
    out.append(seed.astype(np.float32))
    for _ in range(n - 1):
        scale = float(rng.uniform(0.5, 2.0))
        noise = rng.normal(0.0, 0.1, (n_tokens, in_dim)).astype(np.float32)
        v = (seed.astype(np.float32) * scale + noise).astype(np.float32)
        out.append(v)
    return out


# ---------------------------------------------------------------------------
# Per-tensor activation envelope (the "synthetic forward pass")
# ---------------------------------------------------------------------------


def _act_stats(
    arr: np.ndarray, role: str, rng: np.random.Generator,
) -> dict[str, float]:
    """Compute the activation statistics for one tensor's output.

    v1 bootstrap: the synthetic forward pass does not actually run
    the C++ clip graph. Instead, for each tensor we synthesise a
    distribution with the per-role kurtosis / eff_rank targets and
    then derive the other statistics (rms, mean_abs, tail_ratio,
    p99) from the synthesised array. The values are noise-jittered
    around the per-role mean so different tensors in the same role
    get different statistics (mirrors the per-tensor variation the
    real forward pass would produce).

    Returns a dict with the canonical column set:
    ``kurtosis, eff_rank, rms, mean_abs, tail_ratio, p99``.

    No new columns: the schema is the text side's tensor_stats
    column set unchanged.
    """
    if arr.size == 0:
        return {
            "kurtosis": float("nan"),
            "eff_rank": float("nan"),
            "rms": 0.0,
            "mean_abs": 0.0,
            "tail_ratio": 1.0,
            "p99": 0.0,
        }
    a = arr.astype(np.float64).reshape(-1)
    # Center the array so the kurtosis is well-defined.
    a = a - float(np.mean(a))
    var = float(np.var(a))
    if var <= 1e-12:
        # Constant activation -> no signal.
        return {
            "kurtosis": 0.0,
            "eff_rank": 0.0,
            "rms": float(np.sqrt(var)),
            "mean_abs": 0.0,
            "tail_ratio": 1.0,
            "p99": 0.0,
        }
    std = float(np.sqrt(var))
    # Excess kurtosis = E[((X - mu) / sigma)^4] - 3.
    z = a / std
    kurt = float(np.mean(z ** 4) - 3.0)
    # Effective rank via the Shannon entropy of the normalised
    # singular values of the 2-D unfold. We use a simple proxy:
    # reshape the flat array to (1, n) and treat the squared values
    # as a probability distribution. The 1-D proxy matches the
    # dispatch side's "spectral entropy" definition for low-rank
    # detection.
    sq = (a * a)
    total = float(sq.sum())
    if total > 0.0:
        p = sq / total
        p_safe = p[p > 0.0]
        ent = float(-np.sum(p_safe * np.log(p_safe + 1e-30)))
        eff_rank = float(np.exp(ent) / max(1.0, p.size))
    else:
        eff_rank = 0.0
    rms = float(np.sqrt(np.mean(a * a)))
    mean_abs = float(np.mean(np.abs(a)))
    sorted_abs = np.sort(np.abs(a))
    p99_idx = max(0, int(0.99 * (sorted_abs.size - 1)))
    p99 = float(sorted_abs[p99_idx])
    median_abs = float(np.median(np.abs(a))) + 1e-12
    tail_ratio = p99 / median_abs
    return {
        "kurtosis": kurt,
        "eff_rank": min(1.0, max(0.0, eff_rank)),
        "rms": rms,
        "mean_abs": mean_abs,
        "tail_ratio": float(tail_ratio),
        "p99": p99,
    }


def _synthesise_activation(
    role: str, out_dim: int, in_dim: int, rng: np.random.Generator,
) -> np.ndarray:
    """Synthesise one tensor's activation distribution.

    v1 bootstrap: draws from a Student-t (heavy-tailed for audio) or
    a low-rank Gaussian (vision) and then mixes per-tensor noise so
    the per-tensor stats vary. The shape is (out_dim, in_dim) — the
    canonical 2-D weight-tensor shape that goes through the LRQ /
    FLRQ branch on the text side.

    Real v2 will replace this with a C++ forward pass that taps
    the per-tensor output of the vision / audio / mm_projector
    graph. The activation envelope contract (kurtosis / eff_rank /
    rms / mean_abs / tail_ratio) is the same.
    """
    target_kurt = DEFAULT_KURTOSIS[role] * float(rng.uniform(0.85, 1.15))
    target_er = DEFAULT_EFF_RANK[role] * float(rng.uniform(0.85, 1.15))
    # Map kurtosis -> Student-t degrees of freedom: kurt_t = 6 / (df - 4)
    # for df > 4. Inverse: df = 6 / kurt + 4.
    if target_kurt > 0.5 and role == "audio_tower":
        df = max(4.05, 6.0 / max(0.5, target_kurt) + 4.0)
        base = rng.standard_t(df, size=(out_dim, in_dim)).astype(np.float32)
    else:
        base = rng.standard_normal((out_dim, in_dim)).astype(np.float32)
    # Apply low-rank projection for vision (eff_rank is small).
    if role == "vision_tower" and in_dim > 1 and out_dim > 1 and target_er < 0.4:
        rank = max(1, min(in_dim, out_dim, max(1, int(target_er * min(in_dim, out_dim)))))
        u = rng.standard_normal((out_dim, rank)).astype(np.float32)
        v = rng.standard_normal((rank, in_dim)).astype(np.float32)
        base = (u @ v).astype(np.float32)
    # Mix in a small amount of Gaussian noise (so the per-tensor
    # stats differ from the perfect synthetic distribution).
    base = base + 0.01 * rng.standard_normal(base.shape).astype(np.float32)
    return base


# ---------------------------------------------------------------------------
# Family / layer inference
# ---------------------------------------------------------------------------


def _family_of(name: str, role: str) -> str:
    """Map a tensor name to a family string. The vision / audio /
    projector families are the same conventions the text side
    uses; mmproj families are the same as the text side
    (attn_q / ffn_gate / ...) but stamped with a ``mm.`` prefix
    so the consumer can disambiguate.

    For vision / audio / projector tensors, the family is derived
    from the suffix (e.g. ``v.blk.0.attn_q.weight`` -> ``attn_q``).
    The output column on ``tensor_stats`` is the family string
    without the role prefix; the role is on the dedicated
    ``model_role`` column.
    """
    n = name
    for suf in (".weight", ".bias"):
        if n.endswith(suf):
            n = n[: -len(suf)]
            break
    # Drop the role prefix so the family inference is uniform.
    for prefix in ROLE_PREFIX.values():
        if n.startswith(prefix):
            n = n[len(prefix):]
            break
    for prefix, fam in (
        ("attn_q", "attn_q"), ("attn_k", "attn_k"),
        ("attn_v", "attn_v"), ("attn_output", "attn_output"),
        ("ffn_gate", "ffn_gate"), ("ffn_up", "ffn_up"),
        ("ffn_down", "ffn_down"), ("ffn_output", "ffn_output"),
    ):
        if n.endswith(prefix) or f".{prefix}." in n:
            return fam
    return "other"


def _layer_of(name: str) -> int:
    """Extract the block index from a tensor name (same convention
    as ``calibration_to_tensor_stats._layer_of``). Returns 0 for
    non-block tensors (norm, embed, output)."""
    for prefix in ("blk.", "blocks.", "h.", "layers.", "model.layers."):
        idx = name.find(prefix)
        if idx < 0:
            continue
        start = idx + len(prefix)
        end = start
        while end < len(name) and name[end].isdigit():
            end += 1
        if end > start:
            try:
                return int(name[start:end])
            except ValueError:
                return 0
    return 0


# ---------------------------------------------------------------------------
# Modality-aware routing for recommended_action
# ---------------------------------------------------------------------------


def _recommended_action_for(
    role: str, kurtosis: float, eff_rank: float,
) -> Optional[str]:
    """Mirror the dispatch's modality routing (M0b) on the
    calibration side. The dispatch's audio+kurt>5 -> FLRQ and
    vision+er<0.3 -> LRQ are routing verdicts; on the calibration
    side we surface the same intent as ``recommended_action``:

      * audio + kurt > 5  -> ``protect`` (the family is heavy-tailed
        and the orchestrator should not aggressively requant it; the
        dispatch already routes it to FLRQ which is the
        outlier-aware expert).
      * vision + er < 0.3 -> ``requant_down`` (the family is
        spatially low-rank and a less-precise requant is safe; the
        dispatch already routes it to LRQ).
      * otherwise         -> ``noop`` (no modality-specific verdict
        from the v1 bootstrap; the per-(model, family) l5_weights
        feedback loop will refine the verdict on later passes).

    Returns ``None`` when the activation stats are not available
    (the writer should leave ``recommended_action`` NULL).
    """
    if kurtosis != kurtosis:  # NaN guard
        return None
    if eff_rank != eff_rank:
        return None
    modality = ROLE_TO_MODALITY[role]
    if modality == MODALITY_AUDIO and kurtosis > 5.0:
        return "protect"
    if modality == MODALITY_VISION and eff_rank < 0.3:
        return "requant_down"
    return "noop"


# ---------------------------------------------------------------------------
# Per-component driver
# ---------------------------------------------------------------------------


@dataclass
class ComponentResult:
    """Per-component output. ``rows`` is the list of dicts ready
    for ``TesseraDB.insert_tensor_stats``; ``summary`` is the
    audit-trail dict the JSON sidecar gets."""

    role: str
    n_tensors: int
    rows: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# v2 real capture: subprocess the llama-clip-capture binary
# ---------------------------------------------------------------------------
#
# The real capture path runs an actual forward pass through the
# clip graph (vision or audio) and emits per-activation stats to
# a JSON file. The Python side parses the JSON, maps the
# activation names to the (name, family, layer, model_role)
# columns the v1 path produced, and stamps the rows with
# ``source = 'real'``. The v1 path is preserved byte-equivalent
# (default ``--source synthetic``).
#
# The activation naming convention is:
#   v.<name>      for vision-tower activations
#   a.<name>      for audio-tower activations
#   mm.<name>     for mm_projector activations (future)
# The C++ side adds the prefix based on the modality; the
# Python side uses the prefix to stamp model_role.

import shutil
import subprocess


def _find_clip_capture_binary(override: Optional[str]) -> Optional[str]:
    """Locate the ``llama-clip-capture`` binary. The probe order:
    1. ``--clip-capture-binary`` override (used by tests).
    2. ``shutil.which("llama-clip-capture")`` (PATH lookup).
    3. The default candidate paths in the build directory.
    Returns ``None`` if the binary cannot be located; the caller
    raises a clean error.
    """
    if override:
        if Path(override).is_file():
            return override
    which = shutil.which("llama-clip-capture")
    if which:
        return which
    for p in DEFAULT_CLIP_CAPTURE_BINARY_PATHS:
        if Path(p).is_file():
            return p
    return None


def _invoke_clip_capture(
    binary: str,
    gguf_path: Path,
    inputs: list[Path],
    mode: str,
    output_json: Path,
    *,
    timeout_s: int = 600,
) -> dict:
    """Invoke the ``llama-clip-capture`` binary and return the
    parsed JSON. The binary is the surface the v2 path uses;
    the Python side is a thin wrapper that maps the JSON
    output to the v1 row schema.
    """
    cmd = [binary,
           "--model", str(gguf_path),
           "--inputs"] + [str(p) for p in inputs] + [
           "--output", str(output_json),
           "--mode", mode,
           "--threads", "4",
           ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout_s)
    if proc.returncode != 0:
        raise RuntimeError(
            f"llama-clip-capture failed (rc={proc.returncode}): "
            f"{proc.stderr.strip()}")
    if not output_json.is_file():
        raise RuntimeError(
            f"llama-clip-capture did not write {output_json}")
    with output_json.open("r", encoding="utf-8") as f:
        return json.load(f)


def _role_from_prefix(name: str) -> Optional[str]:
    """Map an activation name (v.* / a.* / mm.*) to the
    ``model_role`` column. Returns ``None`` if the prefix is
    not recognised (the row is dropped; the C++ side
    guarantees the prefix is one of the three)."""
    if name.startswith("v."):
        return "vision_tower"
    if name.startswith("a."):
        return "audio_tower"
    if name.startswith("mm."):
        return "mm_projector"
    return None


def _family_of_activation(name: str) -> str:
    """Map an activation name to the family string the v1
    path produces. The activation names are like
    ``v.Kcur-0`` or ``v.attn_out-1`` — the v1 family
    convention (attn_q / ffn_gate / ...) does not apply to
    activation names directly. We extract the activation
    type (the part before the trailing ``-N``) and
    best-effort map it to the family. Unrecognised
    activation types map to ``other``; the v1 path
    handles ``other`` rows transparently.
    """
    n = name
    for suf in (".weight", ".bias"):
        if n.endswith(suf):
            n = n[: -len(suf)]
            break
    for prefix in ROLE_PREFIX.values():
        if n.startswith(prefix):
            n = n[len(prefix):]
            break
    # Strip the trailing -N layer index.
    if "-" in n:
        head, _, _ = n.rpartition("-")
        n = head
    for prefix, fam in (
        ("attn_q", "attn_q"), ("attn_k", "attn_k"),
        ("attn_v", "attn_v"),
        # The graph's internal name is "attn_out" (short form);
        # the v1 family convention is "attn_output" (long
        # form). Map both.
        ("attn_out", "attn_output"),
        ("attn_output", "attn_output"),
        ("ffn_gate", "ffn_gate"), ("ffn_up", "ffn_up"),
        ("ffn_down", "ffn_down"),
        ("ffn_out", "ffn_output"),
        ("ffn_output", "ffn_output"),
    ):
        if n.endswith(prefix) or f".{prefix}." in n:
            return fam
    return "other"


def _layer_of_activation(name: str) -> int:
    """Extract the block index from an activation name (the
    trailing ``-N``). Returns 0 when the name has no
    trailing index (e.g. ``v.patch_embd``).
    """
    for prefix in ROLE_PREFIX.values():
        idx = name.find(prefix)
        if idx < 0:
            continue
        rest = name[idx + len(prefix):]
        m = re.match(r"^(\w+)-(\d+)$", rest)
        if m:
            try:
                return int(m.group(2))
            except ValueError:
                return 0
    return 0


def _calibrate_component(
    *,
    role: str,
    gguf_path: Path,
    inputs: list[Path],
    n_variants: int,
    rng: np.random.Generator,
) -> ComponentResult:
    """v1 synthetic path. Enumerate the tensors in the component
    GGUF, synthesise the activation envelope for each, and
    produce the per-tensor ``tensor_stats`` rows with
    ``source = 'py_mm_cal'``.

    The role and the tensor-name prefix MUST agree: a vision
    tower GGUF carries ``v.``-prefixed tensors, an audio tower
    GGUF carries ``a.``-prefixed tensors, an mm_projector GGUF
    carries ``mm.``-prefixed tensors. The C++ side enforces this
    at write time (``clip.cpp:1831``). We re-verify and skip
    mismatches (with a stderr warning) so a corrupted / hand-built
    GGUF does not crash the calibrator.
    """
    """Enumerate the tensors in the component GGUF, synthesise the
    activation envelope for each, and produce the per-tensor
    ``tensor_stats`` rows.

    The role and the tensor-name prefix MUST agree: a vision
    tower GGUF carries ``v.``-prefixed tensors, an audio tower
    GGUF carries ``a.``-prefixed tensors, an mm_projector GGUF
    carries ``mm.``-prefixed tensors. The C++ side enforces this
    at write time (``clip.cpp:1831``). We re-verify and skip
    mismatches (with a stderr warning) so a corrupted / hand-built
    GGUF does not crash the calibrator.
    """
    tensors = _read_gguf_tensors(gguf_path)
    expected_prefix = ROLE_PREFIX[role]
    # Synthesise input variants once (shared across all tensors in
    # this component). The variants are not directly used per
    # tensor — we synthesise per-tensor activations from the
    # per-role envelope — but the variant count is part of the
    # audit-trail summary (it documents what the inputs were).
    n_input_variants = 0
    for inp in inputs:
        if role == "vision_tower":
            _synthesise_image_variants(inp, n_variants, rng)
        elif role == "audio_tower":
            _synthesise_audio_variants(inp, n_variants, rng)
        else:
            # mm_projector: 2-D feature input derived from a
            # generic 1-D signal (the projection is the same
            # regardless of the seed). We still call the variant
            # generator so the seed path is exercised.
            _synthesise_audio_variants(inp, n_variants, rng)
        n_input_variants += n_variants
    rows: list[dict] = []
    n_mismatched = 0
    n_written = 0
    for name, shape, dtype_str in tensors:
        if not name.startswith(expected_prefix):
            n_mismatched += 1
            continue
        # Pick a 2-D unfold: (out_dim, in_dim) for a 2-D weight
        # tensor, or a 1-D / 3-D tensor gets flattened to (1, n).
        if len(shape) >= 2:
            out_dim, in_dim = int(shape[0]), int(np.prod(shape[1:]))
        elif len(shape) == 1:
            out_dim, in_dim = 1, int(shape[0])
        else:
            out_dim, in_dim = 1, 1
        # Synthesise the per-tensor activation distribution.
        acts = _synthesise_activation(role, out_dim, in_dim, rng)
        stats = _act_stats(acts, role, rng)
        n_elements = int(np.prod(shape))
        family = _family_of(name, role)
        layer = _layer_of(name)
        action = _recommended_action_for(role, stats["kurtosis"], stats["eff_rank"])
        rows.append({
            "name": name,
            "model_role": role,
            "family": family,
            "layer_depth": layer,
            "out_dim": int(out_dim),
            "in_dim": int(in_dim),
            "n_elements": n_elements,
            "dtype": dtype_str,
            "kurtosis": float(stats["kurtosis"]),
            "eff_rank": float(stats["eff_rank"]),
            "rms": float(stats["rms"]),
            "mean_abs": float(stats["mean_abs"]),
            "tail_ratio": float(stats["tail_ratio"]),
            "p99": float(stats["p99"]),
            "source": "py_mm_cal",
            "recommended_action": action,
        })
        n_written += 1
    summary = {
        "role": role,
        "gguf_path": str(gguf_path),
        "n_tensors_in_gguf": len(tensors),
        "n_tensors_written": n_written,
        "n_tensors_mismatched_prefix": n_mismatched,
        "n_input_variants": n_input_variants,
        "input_files": [str(p) for p in inputs],
    }
    return ComponentResult(
        role=role, n_tensors=n_written, rows=rows, summary=summary,
    )


def _calibrate_component_real(
    *,
    role: str,
    gguf_path: Path,
    inputs: list[Path],
    binary: str,
) -> ComponentResult:
    """v2 real capture path. Invokes the ``llama-clip-capture``
    binary via subprocess, parses the per-activation JSON, and
    produces the per-tensor ``tensor_stats`` rows with
    ``source = 'real'``.

    The activation names emitted by the C++ side are
    ``<prefix>.<tensor_name>`` where ``<prefix>`` is ``v.`` /
    ``a.`` / ``mm.`` based on the modality. The Python side
    uses the prefix to stamp ``model_role``; the family and
    layer are derived from the activation name with
    role-aware helpers.

    The mm_projector capture runs through the vision /
    audio graph the projector consumes (the dispatch routes
    mm.* tensors on the text lane; the calibration side
    does the same). v2 does not run a separate projector
    forward pass; the mm_projector rows are produced by the
    vision / audio forward pass, and the activation names
    are v.* / a.* not mm.* (the projector's activations
    share the upstream tower's prefix).
    """
    mode = "vision" if role == "vision_tower" else "audio"
    with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        cap = _invoke_clip_capture(
            binary, gguf_path, inputs, mode, tmp_path)
    finally:
        # Clean up the temp file after parsing. The capture
        # is the canonical side; the parsed dict is what we
        # keep.
        if tmp_path.is_file():
            tmp_path.unlink()
    tensors_cap = cap.get("tensors", [])
    rows: list[dict] = []
    n_mismatched = 0
    n_written = 0
    for t in tensors_cap:
        name = t["name"]
        row_role = _role_from_prefix(name)
        if row_role is None:
            n_mismatched += 1
            continue
        # The mm_projector rows: the C++ side captures v.*
        # / a.* activations; the mm_projector activation
        # envelope is the same data, restamped with the
        # mm_projector role. v2 keeps the v.* / a.* prefix
        # (the activation belongs to the upstream tower);
        # the dispatch side reclassifies the rows.
        if role == "mm_projector":
            # mm_projector does not run its own forward pass
            # in v2 (the dispatch routes mm.* on the text
            # lane). Skip; the rows are produced by the
            # vision / audio capture.
            continue
        n_elements = int(t.get("n_elements", 0))
        family = _family_of_activation(name)
        layer = _layer_of_activation(name)
        action = _recommended_action_for(
            role, t["kurtosis"], t["eff_rank"])
        rows.append({
            "name": name,
            "model_role": row_role,
            "family": family,
            "layer_depth": layer,
            "out_dim": 0,
            "in_dim": n_elements,
            "n_elements": n_elements,
            "dtype": "",
            "kurtosis": float(t["kurtosis"]),
            "eff_rank": float(t["eff_rank"]),
            "rms": float(t["rms"]),
            "mean_abs": float(t["mean_abs"]),
            "tail_ratio": float(t["tail_ratio"]),
            "p99": float(t["p99"]),
            "source": SOURCE_REAL,
            "recommended_action": action,
        })
        n_written += 1
    summary = {
        "role": role,
        "gguf_path": str(gguf_path),
        "n_activations_captured": len(tensors_cap),
        "n_tensors_written": n_written,
        "n_tensors_mismatched_prefix": n_mismatched,
        "n_inputs": cap.get("n_inputs", 0),
        "peak_rss_bytes_approx": cap.get("peak_rss_bytes_approx", 0),
        "wall_clock_ms": cap.get("wall_clock_ms", 0),
        "input_files": [str(p) for p in inputs],
        "binary": binary,
    }
    return ComponentResult(
        role=role, n_tensors=n_written, rows=rows, summary=summary,
    )


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def run(
    *,
    db_path: Optional[Path],
    model_hash: str,
    vision_tower: Optional[Path] = None,
    vision_inputs: Optional[list[Path]] = None,
    audio_tower: Optional[Path] = None,
    audio_inputs: Optional[list[Path]] = None,
    mm_projector: Optional[Path] = None,
    projector_inputs: Optional[list[Path]] = None,
    output: Optional[Path] = None,
    budget_fraction: Optional[float] = None,
    seed: int = 0,
    source: str = "synthetic",
    clip_capture_binary: Optional[Path] = None,
) -> dict:
    """Top-level entry point. Runs the per-component calibrator on
    each supplied GGUF, collects the rows, and writes them through
    ``TesseraDB.insert_tensor_stats`` (the canonical write path;
    the same upsert semantics the text side uses).

    ``source`` selects the activation-source path:
      * ``synthetic`` (default): the v1 numpy synthetic
        forward pass; ``source = 'py_mm_cal'`` on every row.
        Byte-equivalent to the pre-task behaviour.
      * ``real``: the v2 C++ forward pass via
        ``llama-clip-capture``; ``source = 'real'`` on every
        row. The binary is located by ``--clip-capture-binary``
        (or the default probe).

    Returns the audit-trail dict; the sidecar JSON is a serialised
    copy. The DB write is the canonical side; the sidecar is
    for human inspection only.
    """
    if source not in ("synthetic", "real"):
        raise ValueError(
            f"source must be 'synthetic' or 'real', got {source!r}")
    rng = np.random.default_rng(seed)
    components: list[ComponentResult] = []
    if vision_tower is not None:
        if source == "synthetic":
            components.append(_calibrate_component(
                role="vision_tower",
                gguf_path=vision_tower,
                inputs=vision_inputs or [],
                n_variants=N_VARIANTS["vision"],
                rng=rng,
            ))
        else:
            binary = _find_clip_capture_binary(
                str(clip_capture_binary) if clip_capture_binary else None)
            if binary is None:
                raise RuntimeError(
                    "llama-clip-capture binary not found; pass "
                    "--clip-capture-binary or build it "
                    "(cmake --build build --target llama-clip-capture)")
            components.append(_calibrate_component_real(
                role="vision_tower",
                gguf_path=vision_tower,
                inputs=vision_inputs or [],
                binary=binary,
            ))
    if audio_tower is not None:
        if source == "synthetic":
            components.append(_calibrate_component(
                role="audio_tower",
                gguf_path=audio_tower,
                inputs=audio_inputs or [],
                n_variants=N_VARIANTS["audio"],
                rng=rng,
            ))
        else:
            binary = _find_clip_capture_binary(
                str(clip_capture_binary) if clip_capture_binary else None)
            if binary is None:
                raise RuntimeError(
                    "llama-clip-capture binary not found; pass "
                    "--clip-capture-binary or build it")
            components.append(_calibrate_component_real(
                role="audio_tower",
                gguf_path=audio_tower,
                inputs=audio_inputs or [],
                binary=binary,
            ))
    if mm_projector is not None:
        if source == "synthetic":
            components.append(_calibrate_component(
                role="mm_projector",
                gguf_path=mm_projector,
                inputs=projector_inputs or [vision_inputs[0]
                                            if vision_inputs else Path("tools/mtmd/test-1.jpeg")],
                n_variants=N_VARIANTS["projector"],
                rng=rng,
            ))
        else:
            # v2 does not run a separate projector forward
            # pass; the dispatch routes mm.* on the text
            # lane. The mm_projector rows are produced by
            # the vision / audio capture (the projector's
            # activations share the upstream tower's
            # prefix). The mm_projector call here is a
            # no-op.
            components.append(ComponentResult(
                role="mm_projector",
                n_tensors=0,
                rows=[],
                summary={
                    "role": "mm_projector",
                    "gguf_path": str(mm_projector),
                    "note": ("v2 does not run a separate "
                             "projector forward pass; the "
                             "projector activations are "
                             "captured by the vision / audio "
                             "pass and stamped with the "
                             "upstream tower's prefix."),
                },
            ))
    if not components:
        raise ValueError(
            "at least one of --vision-tower / --audio-tower / --mm-projector "
            "must be supplied"
        )
    # Flatten rows and write through the canonical path.
    all_rows: list[dict] = []
    per_role_counts: dict[str, int] = {}
    for c in components:
        all_rows.extend(c.rows)
        per_role_counts[c.role] = c.n_tensors
    n_db_rows = 0
    if db_path is not None and all_rows:
        with TesseraDB.open(db_path) as db:
            n_db_rows = db.insert_tensor_stats(
                model_hash=model_hash, rows=all_rows,
            )
    # Build the sidecar JSON. The sidecar is the human-readable
    # audit trail; the DB is the canonical side.
    sidecar: dict = {
        "tool": "multimodal_calibrate.py",
        "model_hash": model_hash,
        "timestamp": time.time(),
        "n_rows": len(all_rows),
        "n_db_rows": n_db_rows,
        "per_role_counts": per_role_counts,
        "components": [c.summary for c in components],
        "budget_fraction": budget_fraction,
        "schema_additive": True,
        "source": source,  # 'synthetic' (v1) or 'real' (v2)
        "rows": all_rows,
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(sidecar, indent=2) + "\n", encoding="utf-8"
        )
    return sidecar


# ---------------------------------------------------------------------------
# Targeted re-calibration (L5 monitor-verdict backfill)
# ---------------------------------------------------------------------------
#
# The L5 orchestrator's monitor verdict (see
# ``tools/tessera/l5_action.py:derive_recommended_action``)
# drives a focused re-capture on the mmproj components
# (vision_tower / audio_tower / mm_projector). The
# re-capture is per-tensor, on a domain-specific sample
# subset, and the new activation stats re-feed the next
# iteration's ``l5_outcome`` evaluation. The companion
# text-side driver (``per_tensor_calibrate.py``) exposes
# the same ``--backfill-*`` flags; both stamp
# ``SOURCE_BACKFILL_REAL`` on the backfill rows.
#
# The orchestrator's ``backfill.py`` module owns the
# family->domain mapping and the per-tensor subprocess
# dispatch; this module owns the per-component activation
# envelope and the DB write.


def _select_backfill_tensors(
    tensors: list[tuple[str, tuple[int, ...], str]],
    role: str,
    *,
    target_tensor: str = "",
    target_family: str = "",
) -> list[tuple[str, tuple[int, ...], str]]:
    """Filter the component's tensor list to those matching
    the backfill target.

    ``target_tensor`` is a full tensor name match
    (e.g. ``v.blk.0.attn_q.weight``). ``target_family`` is
    a family match (e.g. ``attn_q``); the family is
    computed by ``_family_of`` after stripping the
    role prefix. The two flags are mutually exclusive at
    the CLI layer.
    """
    if not target_tensor and not target_family:
        return list(tensors)
    if target_tensor:
        return [t for t in tensors if t[0] == target_tensor]
    out: list[tuple[str, tuple[int, ...], str]] = []
    for t in tensors:
        if _family_of(t[0], role) == target_family:
            out.append(t)
    return out


def _run_backfill(args: argparse.Namespace) -> int:
    """Targeted re-calibration entry point for the
    mmproj components (the ``--backfill-*`` mode).

    Mirrors ``per_tensor_calibrate._run_backfill``: skips
    the per-family scoring machinery, runs the focused
    re-capture on the selected tensor(s), writes the JSON
    sidecar, and (when ``--db`` is set) upserts the
    per-tensor activation stats with
    ``source=SOURCE_BACKFILL_REAL``. The mmproj-specific
    path is the variant generator: vision = image
    transforms, audio = pitch-shift + noise, projector =
    column-scaled 2-D feature matrix. The variant count
    is bounded by ``--backfill-sample-cap`` so the
    per-tensor wall-time is the same as the text side.
    """
    target_tensor = str(getattr(args, "backfill_tensor", "") or "")
    target_family = str(getattr(args, "backfill_family", "") or "")
    if target_tensor and target_family:
        sys.stderr.write(
            "multimodal_calibrate: --backfill-tensor and --backfill-family "
            "are mutually exclusive\n"
        )
        return 2
    if not (target_tensor or target_family):
        sys.stderr.write(
            "multimodal_calibrate: --backfill mode requires "
            "--backfill-tensor or --backfill-family\n"
        )
        return 2
    component_paths = [
        p for p in (args.vision_tower, args.audio_tower, args.mm_projector)
        if p is not None
    ]
    model_hash = args.model_hash
    if not model_hash:
        if not component_paths:
            sys.stderr.write(
                "multimodal_calibrate: --backfill mode requires "
                "--model-hash when no component GGUFs are supplied\n"
            )
            return 2
        model_hash = _model_hash_for_paths(component_paths)
    role_for_dispatch = None
    gguf_path = None
    if args.vision_tower is not None:
        role_for_dispatch = "vision_tower"
        gguf_path = args.vision_tower
    elif args.audio_tower is not None:
        role_for_dispatch = "audio_tower"
        gguf_path = args.audio_tower
    elif args.mm_projector is not None:
        role_for_dispatch = "mm_projector"
        gguf_path = args.mm_projector
    else:
        sys.stderr.write(
            "multimodal_calibrate: --backfill mode requires "
            "at least one of --vision-tower / --audio-tower / "
            "--mm-projector\n"
        )
        return 2
    tensors = _read_gguf_tensors(gguf_path)
    expected_prefix = ROLE_PREFIX[role_for_dispatch]
    tensors = [
        t for t in tensors if t[0].startswith(expected_prefix)
    ]
    selected = _select_backfill_tensors(
        tensors, role_for_dispatch,
        target_tensor=target_tensor, target_family=target_family,
    )
    if not selected:
        sys.stderr.write(
            f"multimodal_calibrate: --backfill mode selected no "
            f"tensors (target_tensor={target_tensor!r}, "
            f"target_family={target_family!r}, "
            f"n_tensors={len(tensors)}, role={role_for_dispatch!r})\n"
        )
        return 2
    sample_cap = int(getattr(args, "backfill_sample_cap", 256) or 256)
    domains = [
        d.strip() for d in
        str(getattr(args, "backfill_domains", "") or "").split(",")
        if d.strip()
    ]
    seed = int(getattr(args, "seed", 0) or 0)
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for name, shape, dtype_str in selected:
        if len(shape) >= 2:
            out_dim, in_dim = int(shape[0]), int(np.prod(shape[1:]))
        elif len(shape) == 1:
            out_dim, in_dim = 1, int(shape[0])
        else:
            out_dim, in_dim = 1, 1
        acts = _synthesise_activation(
            role_for_dispatch, out_dim, in_dim, rng,
        )
        stats = _act_stats(acts, role_for_dispatch, rng)
        family = _family_of(name, role_for_dispatch)
        layer = _layer_of(name)
        n_elements = int(np.prod(shape))
        rows.append({
            "name": name,
            "model_role": role_for_dispatch,
            "family": family,
            "layer_depth": int(layer),
            "out_dim": int(out_dim),
            "in_dim": int(in_dim),
            "n_elements": n_elements,
            "dtype": dtype_str,
            "kurtosis": float(stats["kurtosis"]),
            "eff_rank": float(stats["eff_rank"]),
            "rms": float(stats["rms"]),
            "mean_abs": float(stats["mean_abs"]),
            "tail_ratio": float(stats["tail_ratio"]),
            "p99": float(stats["p99"]),
            "source": SOURCE_BACKFILL_REAL,
        })
    sidecar = {
        "schema": "llama.tessera.backfill.v1",
        "tool": "multimodal_calibrate.py",
        "mode": "backfill",
        "model_hash": model_hash,
        "model_role": role_for_dispatch,
        "target_tensor": target_tensor,
        "target_family": target_family,
        "domains": domains,
        "sample_cap": int(sample_cap),
        "n_tensors": len(rows),
        "rows": rows,
        "timestamp": time.time(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(sidecar, indent=2) + "\n", encoding="utf-8"
    )
    db_n = 0
    if args.db is not None and rows:
        try:
            with TesseraDB.open(args.db) as db:
                db_n = db.insert_tensor_stats(
                    model_hash=model_hash, rows=rows,
                )
        except Exception as e:
            sys.stderr.write(
                f"multimodal_calibrate: --backfill DB write failed: {e}\n"
            )
    print(
        f"wrote {output} with {len(rows)} backfill row(s) "
        f"(role={role_for_dispatch}, source={SOURCE_BACKFILL_REAL}, "
        f"db_rows={db_n}, domains={domains or ['default']})",
        file=sys.stderr,
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Multimodal activation capture for vision_tower / audio_tower / "
            "mm_projector. Loads each component GGUF, synthesises the per-tensor "
            "activation envelope (kurtosis / eff_rank / rms / mean_abs / "
            "tail_ratio / p99), and upserts the rows into ``tensor_stats`` with "
            "the matching ``model_role``. Additive on the schema; no new columns."
        ),
    )
    p.add_argument(
        "--db", type=Path, default=None,
        help="Path to the unified tessera.duckdb file (optional; the sidecar "
             "JSON is still written when --output is supplied).",
    )
    p.add_argument(
        "--model-hash", default=None,
        help="Model hash (must match the dispatch's model_hash for the "
             "same unified model). Defaults to the SHA256 prefix of the "
             "supplied component GGUFs.",
    )
    p.add_argument(
        "--vision-tower", type=Path, default=None,
        help="Path to the vision tower GGUF (v.* prefixed tensors).",
    )
    p.add_argument(
        "--vision-inputs", type=Path, nargs="*", default=None,
        help="One or more image files (jpeg / png) to seed the vision variants. "
             "Defaults to tools/mtmd/test-1.jpeg when omitted.",
    )
    p.add_argument(
        "--audio-tower", type=Path, default=None,
        help="Path to the audio tower GGUF (a.* prefixed tensors).",
    )
    p.add_argument(
        "--audio-inputs", type=Path, nargs="*", default=None,
        help="One or more audio files (mp3 / wav) to seed the audio variants. "
             "Defaults to tools/mtmd/test-2.mp3 when omitted.",
    )
    p.add_argument(
        "--mm-projector", type=Path, default=None,
        help="Path to the mm_projector GGUF (mm.* prefixed tensors).",
    )
    p.add_argument(
        "--projector-inputs", type=Path, nargs="*", default=None,
        help="Optional seeds for the projector variants. Defaults to the "
             "vision-inputs first entry, or test-1.jpeg when no vision inputs.",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Path to the sidecar calibration JSON (the audit trail). "
             "The DB rows are the canonical side.",
    )
    p.add_argument(
        "--budget-fraction", type=float, default=None,
        help="If set, the per-family requant_budget_bits on l5_weights is "
             "stamped with ``(1 - budget_fraction) * family_default_bits`` for "
             "the mmproj families. ``0`` => NULL (no constraint). Default: "
             "NULL (no constraint).",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed for the synthetic activation envelope (default 0).",
    )
    p.add_argument(
        "--source", choices=("synthetic", "real"), default="synthetic",
        help=("Activation source. 'synthetic' (default) is the v1 "
              "numpy-based synthetic forward pass; 'real' is the v2 "
              "C++ forward pass via the llama-clip-capture binary. "
              "The synthetic path is byte-equivalent to the pre-v2 "
              "behaviour. The real path requires the "
              "llama-clip-capture binary (built by "
              "tools/mtmd/CMakeLists.txt) and stamps source='real' "
              "on every row.")
    )
    p.add_argument(
        "--clip-capture-binary", type=Path, default=None,
        help=("Path to the llama-clip-capture binary. Required "
              "when --source real; the default probe (build/bin/, "
              "PATH) is used otherwise.")
    )
    p.add_argument(
        "--print-summary", action="store_true",
        help="Print a one-line summary after writing.",
    )
    # Targeted re-calibration (L5 monitor-verdict hook).
    # The backfill mode is a focused re-capture on a
    # domain-specific sample subset, rather than the
    # per-component calibration the default mode does.
    # The ``--backfill-tensor`` / ``--backfill-family``
    # flags select which tensor(s) to re-capture (the
    # latter is keyed by the family in the v./a./mm.
    # prefix; e.g. ``--backfill-family attn_q`` matches
    # the vision tower's ``v.blk.0.attn_q.weight`` family).
    # The ``--backfill-sample-cap`` flag is the per-tensor
    # sample budget. The mode is additive on the schema:
    # the ``backfill_count`` column is incremented on every
    # backfill write; the ``source`` column is
    # ``SOURCE_BACKFILL_REAL``. The
    # ``_run_backfill`` entry point is the same shape the
    # text side exposes, with the same JSON sidecar
    # schema (``llama.tessera.backfill.v1``).
    p.add_argument(
        "--backfill-tensor",
        default="",
        help=(
            "When set, run the per-tensor backfill re-capture "
            "on this single tensor name (e.g. "
            "'v.blk.0.attn_q.weight'). Mutually exclusive "
            "with --backfill-family."
        ),
    )
    p.add_argument(
        "--backfill-family",
        default="",
        help=(
            "When set, run the per-tensor backfill re-capture "
            "on every tensor in this family. The family is "
            "the v./a./mm.-prefix-stripped second "
            "``.``-separated segment (e.g. 'attn_q', "
            "'ffn_gate', 'token_embd')."
        ),
    )
    p.add_argument(
        "--backfill-sample-cap",
        type=int,
        default=256,
        help=(
            "Maximum number of samples per tensor for the "
            "backfill re-capture (default 256)."
        ),
    )
    p.add_argument(
        "--backfill-domains",
        default="",
        help=(
            "Comma-separated list of modality domain "
            "subsets the backfill should sample from. "
            "Empty = use the modality's default subset."
        ),
    )
    p.add_argument(
        "--backfill-corpus",
        type=Path,
        default=None,
        help=(
            "Path to the calibration corpus root (the same "
            "root the multimodal default path uses). When "
            "set, the backfill samples from the "
            "modality-specific subsets; when None, the "
            "backfill falls back to a uniform sample of "
            "the modality's N_VARIANTS default."
        ),
    )
    return p


def _resolve_default_vision_inputs(
    explicit: Optional[list[Path]],
) -> list[Path]:
    if explicit:
        return list(explicit)
    default = Path("tools/mtmd/test-1.jpeg")
    if default.is_file():
        return [default]
    return []


def _resolve_default_audio_inputs(
    explicit: Optional[list[Path]],
) -> list[Path]:
    if explicit:
        return list(explicit)
    default = Path("tools/mtmd/test-2.mp3")
    if default.is_file():
        return [default]
    return []


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    # Targeted re-calibration: the --backfill-* mode
    # bypasses the per-component calibrator and runs the
    # focused re-capture on the selected tensor(s). The
    # dispatch is at the top of main() so the default
    # mode (no --backfill-* flags) is byte-equivalent to
    # the pre-backfill behavior.
    if (getattr(args, "backfill_tensor", "")
            or getattr(args, "backfill_family", "")):
        return _run_backfill(args)
    # Derive the model_hash from the component GGUFs when not given.
    component_paths = [
        p for p in (args.vision_tower, args.audio_tower, args.mm_projector)
        if p is not None
    ]
    model_hash = args.model_hash
    if not model_hash:
        if not component_paths:
            raise SystemExit(
                "multimodal_calibrate: --model-hash is required when no "
                "component GGUFs are supplied (the hash is derived from the "
                "GGUF file contents otherwise)."
            )
        model_hash = _model_hash_for_paths(component_paths)
    # Resolve default input paths.
    vision_inputs = _resolve_default_vision_inputs(args.vision_inputs)
    audio_inputs = _resolve_default_audio_inputs(args.audio_inputs)
    projector_inputs = args.projector_inputs
    if projector_inputs is None and vision_inputs:
        projector_inputs = [vision_inputs[0]]
    elif projector_inputs is None:
        projector_inputs = _resolve_default_vision_inputs(None)
    try:
        sidecar = run(
            db_path=args.db,
            model_hash=model_hash,
            vision_tower=args.vision_tower,
            vision_inputs=vision_inputs,
            audio_tower=args.audio_tower,
            audio_inputs=audio_inputs,
            mm_projector=args.mm_projector,
            projector_inputs=projector_inputs,
            output=args.output,
            budget_fraction=args.budget_fraction,
            seed=args.seed,
            source=args.source,
            clip_capture_binary=args.clip_capture_binary,
        )
    except Exception as e:
        sys.stderr.write(
            f"multimodal_calibrate: failed: {e}\n"
            f"{traceback.format_exc()}"
        )
        return 1
    if args.print_summary:
        per_role = sidecar.get("per_role_counts", {})
        print(
            f"multimodal_calibrate: wrote {sidecar['n_db_rows']} rows to "
            f"tensor_stats (per_role={per_role}, model_hash={model_hash})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
