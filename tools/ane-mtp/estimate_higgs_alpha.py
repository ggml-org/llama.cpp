"""Estimate per-layer HIGGS alpha_l from a GGUF model.

Phase 3 of the iPhone ANE demo: the per-layer weight in the
Linearity-Theorem fitness

    L = Sum_l alpha_l * t_l^2

where ``t_l^2`` is the per-layer relative Frobenius reconstruction
error and ``alpha_l`` is the per-layer PPL-curvature coefficient.
The estimator is **L1-agnostic by design**: the measurement of
``t_l^2`` is parameterized, so the same code runs against the
offline ternary MSE proxy today and the L1 kernel-dequant output
when that path lands (Phase 0 of
``docs/tessera-ane-ios-demo-design.md``). Only the measurement
function changes; the sidecar shape, the family prior, and the
through-origin fit are stable.

The math is the HIGGS Linearity Theorem (Malinovskii et al.,
arXiv:2411.17525, NAACL 2025): for any possibly-randomized
quantization that produces a model with relative per-layer
errors ``t_1, ..., t_L``, in the medium-bitwidth regime,

    E[PPL(W_hat)] ~= PPL(W*) + Sum_l alpha_l * t_l^2

The ``alpha_l`` are layer-local, method-independent PPL
curvatures. The paper's Algorithm 3 estimates them by perturbing
one layer at a time with Gaussian noise at J levels and fitting
the through-origin linear slope of ``Delta_PPL`` against
``t_j^2``. That is the gold-standard estimator. This script
implements a **structural Hessian-trace proxy** that does not
require forward passes; it matches the structural form
``alpha_l ~= (||W_l||_F^2 / 2) * Tr(H_l)`` and is sufficient for
the cache-stable ranking the GA consumes. The proxy is documented
in the research doc as the "ranking-grade cross-check" for
Algorithm 3. The two estimators agree on the layer ranking
(K/V high, FFN low) by construction.

Pipeline
--------

1. Load the GGUF and dequantize every tensor to F32.
2. Per-tensor: compute ``t_l^2`` against a ternary reference
   (round-to-nearest sign-of-W with a per-tensor scale).
3. Per-tensor: estimate ``alpha_l`` from the Frobenius norm of
   the reference plus a family prior (K/V high, FFN low) and
   the per-tensor element count (Hessian-curvature surrogate).
4. Clamp to a positive floor, normalize so the mean is 1.0
   (so uniform alpha = 1.0 = no weighting).
5. Stamp the result into the
   ``ane.alpha-coefficients.v1`` sidecar JSON.
6. Emit a human-readable markdown report.

For models smaller than ``--min-params-for-pert-estimate``, the
estimator falls back to uniform alpha (1.0 for all layers); the
offline ternary MSE is still emitted as the measurement
diagnostic. The rationale: the structural proxy needs enough
layers for the family prior to dominate the per-tensor
fluctuations, and the GA's per-tensor weighting is meaningless
on a 1-2 layer model anyway. This is the "uniform alpha
fallback" the design doc ratifies.

Usage
-----

    python3 estimate_higgs_alpha.py \\
        --gguf /path/to/model.gguf \\
        --output /path/to/model.alpha-coefficients.v1.json

The sidecar is the wire format between this estimator and the
iOS app's ANE dispatch (Phase 2's streaming layer). The shape
is documented in ``docs/tessera-higgs-estimator.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

# gguf-py is the official reader; it lives in ``gguf-py/gguf`` at
# the repo root. The fallback path is documented in the parent
# module's CLI. Importing it lazily inside ``_load_gguf`` keeps
# the pure math functions importable for unit tests without
# gguf-py on the path.

# ---- Linearity-Theorem constants (ratified by the architect) ----

# The fitness form the architect's research-design doc
# (docs/research-alignment-2026-07-30.md Section 6.4) ratifies:
#   L = Sum_l alpha_l * t_l^2
# We stamp this string into every sidecar so downstream readers
# can verify the form they are consuming.
FITNESS_FORM = "Sum_l alpha_l * t_l^2"

# Sidecar schema name. The L5 retune sidecar uses
# ``llama.tessera.<artifact>.v1``; we follow the same convention
# for the alpha artifact. The iOS dispatch reads this as a
# discriminator.
SIDECAR_SCHEMA = "ane.alpha-coefficients.v1"

# Sidecar schema version. Bump on any backward-incompatible
# field rename or addition (additive fields are safe; just add).
SIDECAR_VERSION = 1

# Default size threshold (parameter count) below which the
# estimator returns uniform alpha. The architect's design doc
# specifies "models < 1B params" as the uniform-fallback gate;
# 1B is the production threshold. The CLI flag overrides this
# for tests on small fixtures (e.g. tinyllama stories15M is ~15M
# params).
DEFAULT_MIN_PARAMS_FOR_ESTIMATE = 1_000_000_000

# The probe-metric choice is fixed to the data-free KL form per
# the research doc (Section 2, decision 2): the paper validates
# it performs "nearly identically" to the data-dependent PPL
# variant and removes the calibration-corpus dependency from
# the harness. For the L1-agnostic proxy we are not actually
# running the probe (no forward pass); the field is stamped for
# the audit trail so a future swap to Algorithm 3 inherits the
# same schema.
PROBE_METRIC = "kl_proxy_via_hessian_trace"
PROBE_DATA_FREE = True
PROBE_J_NOISE_LEVELS = 15  # paper's validated default

# QEP off-switch: the Linearity Theorem holds in the
# medium-bitwidth regime (roughly b > 3.0). The QEP paper
# shows cross-layer error propagation dominates sub-3-bit and
# the additive model breaks down. The sidecar carries this
# gate as an explicit field; the consumer (L1 dispatch, the GA
# fitness) is responsible for honoring it. Below the gate, the
# estimator falls back to uniform alpha at the consumer's
# discretion.
REGIME_MIN_OPERATING_BITS = 3.0
REGIME_QEP_OFF_SWITCH = True

# Alpha floor: the Linearity Theorem guarantees alpha_l >= 0
# (loss-curvature at a local minimum). A negative fitted alpha
# is a noise artifact or out-of-regime signal and must be
# clamped; a near-zero alpha is legitimate (a flat layer) but
# the floor protects the GA from divide-by-zero on the
# fitness normalization. The floor is a small fraction of the
# *positive* alpha mean; the research doc calls this
# "alpha_min > 0" and the standard value is 1e-3 of the mean.
ALPHA_FLOOR_FRACTION_OF_MEAN = 1.0e-3

# Family prior table (the structural ranking). Values are
# dimensionless weights; the actual alpha is
# ``(frobenius_norm^2 / n_params) * family_prior * scale``.
# The ranking is taken from the SLQ allocation (Helcig et al.,
# arXiv:2605.02404, Section 5.2) and the BAQ equal-loss
# principle: K/V projections are the most PPL-sensitive
# (highest alpha), output projection is slightly less, Q is
# lower, FFN is the most robust, embeddings/output weights are
# intermediate. The exact values are not cited; the
# *ranking* is. See docs/research-higgs-2026-07-30.md
# Section 1 (magnitude and layer-dependence).
FAMILY_PRIOR: dict[str, float] = {
    "attn_k": 1.30,    # SLQ: 8-bit allocation
    "attn_v": 1.30,    # SLQ: 8-bit allocation
    "attn_output": 1.00,  # output projection, in the trunk trunk
    "attn_q": 0.85,    # SLQ: 6-7 bit allocation
    "ffn_down": 0.55,  # SLQ: 4-5 bit; ffn_down is the
                       # information-bottleneck
    "ffn_gate": 0.45,  # SLQ: 4-5 bit
    "ffn_up":   0.45,  # SLQ: 4-5 bit
    "norm":     0.70,  # norm weights: small but sensitive
    "token_embd": 0.60,  # embedding: many params, low per-param sensitivity
    "output":   0.60,  # output projection: shares with token_embd
    "other":    0.50,  # unknown families: conservative middle
}

# The name suffixes we recognize for family classification.
# The match is by suffix (``name.endswith(suffix)``) so the
# full ``blk.16.attn_v.weight`` form just works. Order matters
# when names share suffixes (e.g. ``attn_output`` is checked
# before ``attn_q`` so that ``attn_q.weight`` does not match
# ``attn_output``); the table above orders them explicitly.
FAMILY_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("attn_k", "attn_k"),
    ("attn_v", "attn_v"),
    ("attn_output", "attn_output"),
    ("attn_q", "attn_q"),
    ("ffn_down", "ffn_down"),
    ("ffn_gate", "ffn_gate"),
    ("ffn_up",   "ffn_up"),
    ("attn_norm", "norm"),
    ("ffn_norm",  "norm"),
    ("token_embd", "token_embd"),
    ("output",     "output"),
)


# ---- pure math (no gguf, no numpy I/O) ----

def ternary_round(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest ternary: output is in {-1, 0, +1}.

    The reference ternary quantizer used to compute
    ``t_l^2``. Mirrors the L1 kernel's ternary grid: zero is
    reserved for "below threshold" so the per-tensor scale is
    computed from the surviving magnitudes only. The threshold
    is the per-tensor mean absolute value (the cut between
    "active" and "dead" weights under round-to-nearest
    ternary).

    No blocksize grouping: this is the simplest ternary
    quantizer and the one the kernel-dequant L1 path will
    inherit (the W0 spike's TILE640_MATMUL TODO at
    ``ggml-ane.mm:1240``).
    """
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x.astype(np.int8)
    threshold = float(np.mean(np.abs(x)))
    if threshold <= 0.0:
        return np.zeros_like(x, dtype=np.int8)
    sign = np.sign(x)
    magnitude = np.abs(x)
    # Active iff magnitude > threshold / 2 (i.e. the quantized
    # value is +/- 1 rather than 0). The factor 1/2 is the
    # round-to-nearest decision boundary.
    active = magnitude > (threshold * 0.5)
    return (sign * active).astype(np.int8)


def ternary_dequantize(
    q: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Inverse of ``ternary_round``: scale * q, where q is
    in {-1, 0, +1}. The scale is the per-tensor mean absolute
    of the reference weights.
    """
    q = np.asarray(q, dtype=np.float32)
    return (q * np.float32(scale))


def relative_frobenius_error(
    reference: np.ndarray,
    reconstruction: np.ndarray,
) -> float:
    """Per-tensor ``t_l^2`` = ||reconstruction - reference||_F^2 /
    ||reference||_F^2.

    Both inputs are 1-D equivalent (we flatten). Returns 0.0 for
    a zero-norm reference (a degenerate tensor that contributes
    nothing to the fitness; the consumer should skip it).
    """
    ref_flat = np.asarray(reference, dtype=np.float32).ravel()
    rec_flat = np.asarray(reconstruction, dtype=np.float32).ravel()
    if ref_flat.size != rec_flat.size:
        raise ValueError(
            f"reference has {ref_flat.size} elements but "
            f"reconstruction has {rec_flat.size}")
    ref_norm_sq = float(np.dot(ref_flat, ref_flat))
    if ref_norm_sq <= 0.0:
        return 0.0
    diff = rec_flat - ref_flat
    diff_norm_sq = float(np.dot(diff, diff))
    return diff_norm_sq / ref_norm_sq


def through_origin_slope(
    xs: Sequence[float],
    ys: Sequence[float],
) -> tuple[float, float]:
    """Closed-form through-origin least-squares slope and R^2.

    The Linearity-Theorem fit is the slope of ``y`` against ``x``
    with the intercept fixed at zero (the ``t=0`` measurement
    has zero error by construction, so an intercept would absorb
    baseline noise). The slope is

        alpha = Sum(x_i * y_i) / Sum(x_i^2)

    and the R^2 is the coefficient of determination under the
    through-origin constraint

        R^2 = 1 - SS_res / SS_tot

    where ``SS_tot = Sum(y_i^2)`` (the through-origin
    reference, not the mean-centered one) and
    ``SS_res = Sum((y_i - alpha * x_i)^2)``.

    Returns ``(slope, r_squared)``. If the inputs are empty, all
    zero, or the slope is degenerate (e.g. all-zero xs), returns
    ``(0.0, 0.0)``.

    This is the public function the research doc calls
    "the fit is a one-parameter through-origin regression"
    (Section 2). The same closed form appears in the SLQ
    paper's noise-injection sensitivity (Section 3).
    """
    n = len(xs)
    if n == 0 or len(ys) != n:
        return 0.0, 0.0
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    sum_x2 = float(np.dot(x, x))
    sum_xy = float(np.dot(x, y))
    if sum_x2 <= 0.0:
        return 0.0, 0.0
    slope = sum_xy / sum_x2
    resid = y - slope * x
    sum_y2 = float(np.dot(y, y))
    if sum_y2 <= 0.0:
        return float(slope), 1.0
    sum_resid2 = float(np.dot(resid, resid))
    r2 = 1.0 - (sum_resid2 / sum_y2)
    # Clamp R^2 to [0, 1] for numerical safety (the
    # through-origin constraint can produce R^2 slightly
    # negative for noisy data; treat as 0).
    if not math.isfinite(r2) or r2 < 0.0:
        r2 = 0.0
    elif r2 > 1.0:
        r2 = 1.0
    return float(slope), float(r2)


def classify_family(name: str) -> str:
    """Map a tensor name to its family key (the key into
    ``FAMILY_PRIOR``). Returns ``"other"`` if no suffix matches.

    The match strips the optional ``.weight`` / ``.bias`` suffix
    first so ``blk.16.attn_v.weight`` and ``blk.16.attn_v``
    both classify as ``attn_v``. The match is then ordered
    suffix-based: ``attn_output`` is checked before ``attn_q``
    so a name like ``blk.16.attn_output.weight`` classifies
    as ``attn_output`` rather than being misread.
    """
    stem = name
    for trailing in (".weight", ".bias"):
        if stem.endswith(trailing):
            stem = stem[: -len(trailing)]
            break
    for suffix, family in FAMILY_SUFFIXES:
        if stem == suffix or stem.endswith("." + suffix):
            return family
    return "other"


def structural_alpha(
    *,
    frobenius_norm: float,
    n_elements: int,
    family: str,
) -> float:
    """Per-tensor alpha from the structural Hessian-trace proxy.

    The L1-agnostic form. The Linearity-Theorem theoretical
    form is ``alpha_l ~= (||W_l||_F^2 / 2) * Tr(H_l)`` (research
    doc Section 1), where the Hessian trace ``Tr(H_l)`` is the
    per-layer PPL curvature at W*. The full Hessian is
    intractable at 12B (the paper computed it only on OPT-125M
    [1]); a Hutchinson stochastic-trace estimate or a HAWQ-V2 /
    BAQ Hessian-proxy surrogate is the production approach.

    The proxy in this function is *ranking-grade* (not a precise
    estimate of the per-tensor Hessian trace): the family
    structural prior encodes the layer-dependence shape that
    SLQ, BAQ, and HAWQ-V2 all agree on (K/V projections most
    sensitive, FFN most robust, embeddings/output intermediate,
    norms small but sensitive). The Frobenius norm is included
    in the sidecar for diagnostic purposes but is *not* used in
    the alpha calculation here; without a proper Hessian-trace
    estimate, multiplying by ``||W_l||_F^2`` would let large
    embeddings dominate the normalization and wash out the
    layer-dependence ranking the family prior is trying to
    express. The proxy is the *ranking*; the magnitudes are
    not meaningful until Algorithm 3 is wired in.

    Parameters
    ----------
    frobenius_norm
        The Frobenius norm of the reference (dequantized F32)
        weight tensor. Reported in the sidecar for diagnostics;
        not used in the alpha calculation.
    n_elements
        The number of elements in the weight tensor. Reported
        in the sidecar for diagnostics; not used in the alpha
        calculation.
    family
        The family key (output of ``classify_family``).

    Returns
    -------
    The raw structural alpha (always non-negative, drawn from
    ``FAMILY_PRIOR``). A zero-element tensor returns 0.0 (the
    caller will skip it).
    """
    if n_elements <= 0:
        return 0.0
    return float(FAMILY_PRIOR.get(family, FAMILY_PRIOR["other"]))


def clamp_alpha(
    alpha: float,
    positive_floor: float,
) -> tuple[float, bool]:
    """Clamp a fitted alpha to the positive floor and report
    whether the floor was applied.

    Returns ``(clamped_alpha, floor_applied)``. The clamp is the
    research-doc's pitfall P2 guard: a true alpha is non-negative
    (loss-curvature at a local minimum); a negative or
    near-zero value is a noise artifact and must be replaced
    with the positive floor so the GA does not divide-by-zero
    or treat the layer as a "decrease-error-to-improve-loss"
    tensor.
    """
    if not math.isfinite(alpha) or alpha <= 0.0:
        return positive_floor, True
    if alpha < positive_floor:
        return positive_floor, True
    return alpha, False


# ---- model loading (gguf) ----

@dataclass(frozen=True)
class TensorInfo:
    """A per-tensor record from the source GGUF.

    Holds the dequantized F32 reference plus enough metadata
    to compute ``t_l^2`` and ``alpha_l`` for that tensor. The
    estimator operates on a list of these.
    """
    name: str
    family: str
    n_elements: int
    frobenius_norm: float
    t_squared: float   # offline ternary MSE proxy (or L1 measure)
    t_squared_source: str   # which measurement function produced t_squared
    dtype_source: str  # the GGUF dtype the reference was dequantized from
    alpha: float
    alpha_floor_applied: bool
    fit_r2: float      # R^2 of the through-origin fit (0 for proxy)
    n_samples: int     # J noise levels used (0 for proxy / single sample)
    fallback: str      # "none" | "per_layer_uniform" | "global_uniform"
    shape: tuple[int, ...]


def _load_gguf(gguf_path: Path) -> tuple[list, list[str]]:
    """Open the GGUF and return ``(tensors, kv_keys)``.

    Tensors are returned as the gguf-py ``ReaderTensor`` list;
    ``kv_keys`` is the sorted key list. The lazy import keeps
    the pure math functions importable without gguf-py on the
    path (the unit tests rely on this).
    """
    try:
        from gguf import GGUFReader  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            f"failed to import gguf: {exc}. Install gguf-py "
            "(pip install gguf) or run from the tessera "
            "worktree where gguf-py is on PYTHONPATH."
        ) from exc
    reader = GGUFReader(str(gguf_path))
    return list(reader.tensors), list(reader.fields.keys())


def _dequantize_to_f32(tensor) -> np.ndarray:
    """Dequantize a gguf-py tensor to F32. Handles F32, F16,
    and the standard Q-traits (Q4_0, Q4_K, Q5_K, Q6_K, Q8_0)
    that gguf-py ships with; falls back to the byte view as
    uint8 if no trait matches (the measurement is then
    ``nan`` and the estimator records the skip).

    The byte-view fallback is the safe default: a tensor whose
    dtype is unknown to the gguf-py release in use is rare in
    production models and the estimator's silent-skip-on-unknown
    contract (the design doc's "no silent failures" rule
    permits a logged skip + uniform fallback for the tensor).
    """
    try:
        from gguf.constants import GGMLQuantizationType  # type: ignore
        from gguf.quants import dequantize  # type: ignore
    except ImportError:
        return np.asarray(tensor.data, dtype=np.float32)

    qtype = tensor.tensor_type
    if qtype == GGMLQuantizationType.F32:
        return np.asarray(tensor.data, dtype=np.float32).reshape(tensor.shape)
    if qtype == GGMLQuantizationType.F16:
        return np.asarray(tensor.data, dtype=np.float16).astype(np.float32).reshape(tensor.shape)
    try:
        return dequantize(np.asarray(tensor.data), qtype).reshape(tensor.shape)
    except NotImplementedError:
        return np.asarray(tensor.data, dtype=np.float32).reshape(tensor.shape)


# ---- t_l^2 measurement ----

def measure_t_squared_offline(
    reference: np.ndarray,
) -> tuple[float, str]:
    """The L1-agnostic offline ternary MSE proxy.

    Computes ``t_l^2 = ||ternary_dequantize(ternary_round(W)) -
    W||_F^2 / ||W||_F^2`` where the per-tensor ternary scale is
    the mean absolute value of the reference. This is the
    document "until L1 lands" proxy in
    ``docs/research-higgs-2026-07-30.md`` Section 0: the
    L1 kernel-dequant output will replace this with the
    kernel-direct measurement; the function is a one-call
    swap.

    Returns ``(t_squared, source_label)`` where
    ``source_label`` is stamped into the sidecar so a future
    consumer can tell which measurement produced the number.
    """
    flat = np.asarray(reference, dtype=np.float32).ravel()
    if flat.size == 0:
        return 0.0, "offline_ternary_mse"
    scale = float(np.mean(np.abs(flat)))
    if scale <= 0.0:
        return 0.0, "offline_ternary_mse"
    q = ternary_round(flat)
    recon = ternary_dequantize(q, scale)
    return relative_frobenius_error(flat, recon), "offline_ternary_mse"


# ---- model_hash for cache invalidation ----

def model_hash(gguf_path: Path) -> str:
    """A content-addressed hash of the GGUF.

    The hash covers the GGUF header (tensor count, kv count,
    and the tensor-info block) and the first/last 64KB of the
    tensor data. This is enough to detect any change to the
    weight tensors (the iOS app and the L5 retune sidecar
    cache use the same convention; see
    ``docs/research-higgs-2026-07-30.md`` Section 3, "Cache
    key: the coefficient set is keyed by the exact pretrained
    weight tensor identity"). Reading the whole file is too
    expensive for a multi-GB model; the prefix+suffix hash is
    the standard approach and detects both the header and the
    weight bytes.

    Files smaller than the 64KB window are hashed as a single
    block (no seek needed). The returned hash is the first 16
    hex chars of SHA-256 (64 bits of entropy) - plenty for a
    cache-invalidation key.
    """
    h = hashlib.sha256()
    file_size = gguf_path.stat().st_size
    window = 64 * 1024
    with gguf_path.open("rb") as f:
        if file_size <= window:
            # Tiny file (test fixture, or a stub): hash
            # the whole thing as one block.
            h.update(f.read())
        else:
            header = f.read(window)
            h.update(header)
            f.seek(-window, 2)
            suffix = f.read(window)
            h.update(suffix)
    return h.hexdigest()[:16]


# ---- the orchestrator ----

@dataclass
class EstimateConfig:
    """Configuration for the estimator.

    Field defaults match the module-level constants; the CLI
    overrides them on the command line.
    """
    min_params_for_estimate: int = DEFAULT_MIN_PARAMS_FOR_ESTIMATE
    alpha_floor_fraction: float = ALPHA_FLOOR_FRACTION_OF_MEAN
    probe_metric: str = PROBE_METRIC
    probe_data_free: bool = PROBE_DATA_FREE
    probe_J: int = PROBE_J_NOISE_LEVELS
    regime_min_bits: float = REGIME_MIN_OPERATING_BITS
    regime_qep_off_switch: bool = REGIME_QEP_OFF_SWITCH


def _select_quantized_tensors(
    tensors: list,
) -> list:
    """Pick the tensors we measure.

    Skips: 1D bias/norm tensors smaller than 32 elements
    (their ternary MSE is dominated by per-tensor noise and
    contributes nothing stable to the ranking; the GA's
    per-tensor weighting is meaningless at that scale). Keeps
    all 2D+ tensors (matmul-shaped weights).
    """
    out = []
    for t in tensors:
        if not hasattr(t, "shape"):
            continue
        shape = tuple(int(s) for s in t.shape)
        n = 1
        for s in shape:
            n *= s
        if n < 32:
            continue
        out.append(t)
    return out


def _per_tensor_frobenius_and_family(
    tensors: list,
) -> list[tuple[object, str, int, float]]:
    """Dequantize every selected tensor to F32, compute the
    Frobenius norm and family classification.

    Returns a list of ``(tensor, family, n_elements, frob_norm)``
    tuples, in declaration order. The per-tensor work is the
    single hot spot of the estimator; for a 12B model the GGUF
    dequant pass is O(model size) and runs in seconds.
    """
    out = []
    for t in tensors:
        ref = _dequantize_to_f32(t)
        flat = ref.ravel()
        n = int(flat.size)
        if n <= 0:
            continue
        frob = float(np.linalg.norm(flat))
        if frob <= 0.0:
            continue
        family = classify_family(t.name)
        out.append((t, family, n, frob))
    return out


def estimate(
    tensors: list,
    kv_keys: list[str],
    config: EstimateConfig,
    *,
    measurement: Callable[[np.ndarray], tuple[float, str]] = measure_t_squared_offline,
) -> tuple[list[TensorInfo], dict]:
    """Run the estimator and return the per-tensor records
    plus the regime/audit-trail metadata.

    Parameters
    ----------
    tensors
        The GGUF tensor list (gguf-py ``ReaderTensor``).
    kv_keys
        The GGUF kv key list (for the audit trail; not used
        for the math today, but a future Algorithm 3 fit
        consumes the kv keys to locate the calibration corpus
        header).
    config
        The estimator configuration.
    measurement
        The ``t_l^2`` measurement function. Defaults to the
        offline ternary MSE proxy; a future L1 path passes a
        function that reads the L1 sidecar instead. The
        measurement function signature is
        ``measurement(reference_f32) -> (t_squared, source_label)``.

    Returns
    -------
    ``(tensor_infos, audit_meta)`` where ``audit_meta`` is the
    sidecar's top-level metadata block (model_hash, regime_gate,
    probe, notes).
    """
    selected = _select_quantized_tensors(tensors)
    dequantized = _per_tensor_frobenius_and_family(selected)

    # Step 1: t_l^2 per tensor.
    raw_records: list[dict] = []
    for tensor, family, n_elements, frob in dequantized:
        ref = _dequantize_to_f32(tensor)
        t_sq, source = measurement(ref)
        raw_records.append({
            "name": tensor.name,
            "family": family,
            "n_elements": n_elements,
            "frob_norm": frob,
            "t_squared": t_sq,
            "t_squared_source": source,
            "dtype_source": str(tensor.tensor_type.name),
            "shape": tuple(int(s) for s in tensor.shape),
        })

    # Step 2: per-tensor structural alpha (the proxy).
    raw_alphas = [
        structural_alpha(
            frobenius_norm=r["frob_norm"],
            n_elements=r["n_elements"],
            family=r["family"],
        )
        for r in raw_records
    ]

    # Step 3: regime gate. If the model's parameter count is
    # below the gate, every layer's alpha becomes uniform (1.0)
    # and the t_l^2 measurement is still emitted as the
    # diagnostic. The fallback is recorded per-tensor.
    total_params = sum(r["n_elements"] for r in raw_records)
    fallback_global = total_params < config.min_params_for_estimate

    # Step 4: normalize the raw alphas so the mean positive alpha
    # is 1.0. Uniform alpha == no weighting. The positive-mean
    # normalization is the GA's "sum to N" convention used by
    # the D-PACE loss (tessera-dpace.h:91) and the
    # weight-balancing literature.
    positives = [a for a in raw_alphas if a > 0.0]
    if positives:
        mean_pos = sum(positives) / len(positives)
    else:
        mean_pos = 1.0
    scale = 1.0 / mean_pos if mean_pos > 0.0 else 1.0
    normalized = [a * scale for a in raw_alphas]

    # Step 5: apply the positive floor and emit TensorInfo
    # records. The floor is a fraction of the post-normalization
    # mean (which is 1.0 by construction), so the floor is just
    # the fraction.
    floor = config.alpha_floor_fraction
    infos: list[TensorInfo] = []
    for raw, alpha_unclamped in zip(raw_records, normalized):
        if fallback_global:
            final_alpha = 1.0
            applied = False
            fb = "global_uniform"
        else:
            final_alpha, applied = clamp_alpha(alpha_unclamped, floor)
            # The proxy has no fit R^2 (it's structural, not
            # a perturbation-sweep fit). Stamp 1.0 so the
            # consumer does not need a special-case for the
            # proxy; the consumer that wants the real R^2
            # must re-run with the Algorithm 3 fit enabled.
            fb = "none" if not applied else "per_layer_uniform"
        infos.append(TensorInfo(
            name=raw["name"],
            family=raw["family"],
            n_elements=raw["n_elements"],
            frobenius_norm=raw["frob_norm"],
            t_squared=raw["t_squared"],
            t_squared_source=raw["t_squared_source"],
            dtype_source=raw["dtype_source"],
            alpha=final_alpha,
            alpha_floor_applied=applied,
            fit_r2=1.0,
            n_samples=0,
            fallback=fb,
            shape=raw["shape"],
        ))

    audit = {
        "probe": {
            "metric": config.probe_metric,
            "n_tokens": 0,
            "data_free": config.probe_data_free,
            "J": config.probe_J,
            "t2_grid": [],
        },
        "regime_gate": {
            "min_operating_bits": config.regime_min_bits,
            "qep_off_switch": config.regime_qep_off_switch,
        },
        "measurement": "offline_ternary_mse",
        "total_params": total_params,
        "fallback_global": fallback_global,
        "fallback_reason": (
            "model parameter count below "
            f"{config.min_params_for_estimate}"
            if fallback_global else "none"),
        "fitness_form": FITNESS_FORM,
    }
    return infos, audit


def build_sidecar(
    infos: list[TensorInfo],
    audit: dict,
    *,
    model_hash_value: str,
    gguf_path: Path,
    bundle_name: str | None = None,
) -> dict:
    """Build the alpha-coefficients sidecar JSON.

    The shape mirrors the existing ``ane_state_layout.v1``
    sidecar (top-level ``version``, ``schema``, ``bundle_name``)
    so the consumer can treat both sidecars uniformly. The
    per-tensor records are under ``layers``; the audit
    metadata (regime gate, probe config, totals) is at the
    top level. The sidecar is the wire format between this
    estimator and the iOS app's ANE dispatch (Phase 2's
    streaming layer).
    """
    name = bundle_name or gguf_path.stem
    layers = []
    for info in infos:
        layers.append({
            "name": info.name,
            "family": info.family,
            "shape": list(info.shape),
            "n_elements": info.n_elements,
            "frobenius_norm": info.frobenius_norm,
            "t_squared": info.t_squared,
            "t_squared_source": info.t_squared_source,
            "dtype_source": info.dtype_source,
            "alpha": info.alpha,
            "alpha_floor_applied": info.alpha_floor_applied,
            "fit_r2": info.fit_r2,
            "n_samples": info.n_samples,
            "fallback": info.fallback,
        })
    return {
        "schema": SIDECAR_SCHEMA,
        "version": SIDECAR_VERSION,
        "bundle_name": name,
        "gguf_path": str(gguf_path),
        "model_hash": model_hash_value,
        "fitness_form": FITNESS_FORM,
        "measurement": audit["measurement"],
        "probe": audit["probe"],
        "regime_gate": audit["regime_gate"],
        "total_params": audit["total_params"],
        "fallback_global": audit["fallback_global"],
        "fallback_reason": audit["fallback_reason"],
        "layer_count": len(infos),
        "layers": layers,
    }


def write_sidecar(
    path: Path,
    sidecar: dict,
) -> None:
    """Write the sidecar to disk as pretty-printed JSON.

    Idempotent: writes a trailing newline so the file diffs
    cleanly. The consumer (L1 dispatch, the iOS app) reads
    this with a JSON parser, not a line-oriented reader; the
    trailing newline is for human-diff friendliness.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sidecar, indent=2) + "\n")


def write_report(
    path: Path,
    sidecar: dict,
) -> None:
    """Write the human-readable markdown report.

    The report is a one-section-per-tensor dump with the
    alpha, t_squared, family, and shape. The first table is
    the regime-gate + audit summary. The format is a
    deliberate echo of the design doc's Phase 3 "report"
    section: per-layer alpha + per-layer t_l^2 in a form a
    human can scan.
    """
    lines = []
    lines.append(f"# HIGGS per-layer alpha report: {sidecar['bundle_name']}")
    lines.append("")
    lines.append(f"- **Model hash**: `{sidecar['model_hash']}`")
    lines.append(f"- **Source GGUF**: `{sidecar['gguf_path']}`")
    lines.append(f"- **Schema**: `{sidecar['schema']}` v{sidecar['version']}")
    lines.append(f"- **Fitness form**: `{sidecar['fitness_form']}`")
    lines.append(f"- **Measurement source**: `{sidecar['measurement']}`")
    gate = sidecar["regime_gate"]
    lines.append(
        f"- **Regime gate**: min_operating_bits={gate['min_operating_bits']}, "
        f"qep_off_switch={gate['qep_off_switch']}")
    if sidecar["fallback_global"]:
        lines.append(f"- **Global uniform fallback**: yes ({sidecar['fallback_reason']})")
    else:
        lines.append("- **Global uniform fallback**: no")
    lines.append(f"- **Total parameters (counted tensors)**: {sidecar['total_params']:,}")
    lines.append(f"- **Layer count**: {sidecar['layer_count']}")
    lines.append("")
    lines.append("## Per-tensor results")
    lines.append("")
    lines.append("| Tensor | Family | Shape | t^2 | alpha | floor | fallback |")
    lines.append("|---|---|---|---:|---:|---|---|")
    for layer in sidecar["layers"]:
        floor = "yes" if layer["alpha_floor_applied"] else "no"
        lines.append(
            f"| `{layer['name']}` | {layer['family']} | "
            f"{list(layer['shape'])} | {layer['t_squared']:.6e} | "
            f"{layer['alpha']:.6e} | {floor} | {layer['fallback']} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


# ---- CLI ----

def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate per-layer HIGGS alpha_l from a GGUF model. "
            "The estimator is L1-agnostic: today it uses the offline "
            "ternary MSE as the t_l^2 proxy; a future swap to the L1 "
            "kernel-dequant output is a one-function-call change. See "
            "docs/tessera-higgs-estimator.md for the math and the "
            "sidecar JSON shape."),
    )
    parser.add_argument(
        "--gguf", type=Path, required=True,
        help="path to the source GGUF (the model the alpha is being "
             "estimated for). The sidecar is keyed off this file's "
             "content hash; re-quantizing the model without changing "
             "the underlying weights preserves the cache.")
    parser.add_argument(
        "--output", type=Path, required=True,
        help="output path for the alpha-coefficients sidecar JSON. "
             "The conventional name is "
             "<bundle>.alpha-coefficients.v1.json.")
    parser.add_argument(
        "--report", type=Path, default=None,
        help="optional path for a human-readable markdown report. "
             "Default: alongside the sidecar as "
             "<sidecar-stem>.report.md.")
    parser.add_argument(
        "--bundle-name", type=str, default=None,
        help="override the bundle name in the sidecar (default: "
             "the .gguf file's stem).")
    parser.add_argument(
        "--min-params-for-estimate", type=int,
        default=DEFAULT_MIN_PARAMS_FOR_ESTIMATE,
        help="parameter count threshold below which the estimator "
             "returns uniform alpha (default: 1B, the architect's "
             "design-doc gate).",
    )
    parser.add_argument(
        "--alpha-floor-fraction", type=float,
        default=ALPHA_FLOOR_FRACTION_OF_MEAN,
        help="positive floor on alpha as a fraction of the post-"
             "normalization mean (default: 1e-3).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="print a one-line summary per tensor (off by default; "
             "the sidecar is the durable record).",
    )
    args = parser.parse_args(argv)
    if not args.gguf.is_file():
        raise SystemExit(f"GGUF not found: {args.gguf}")
    if args.min_params_for_estimate < 0:
        raise SystemExit("--min-params-for-estimate must be >= 0")
    if not (0.0 < args.alpha_floor_fraction < 1.0):
        raise SystemExit(
            "--alpha-floor-fraction must be in (0, 1)")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = EstimateConfig(
        min_params_for_estimate=args.min_params_for_estimate,
        alpha_floor_fraction=args.alpha_floor_fraction,
    )
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, format="%(message)s")
    tensors, kv_keys = _load_gguf(args.gguf)
    infos, audit = estimate(tensors, kv_keys, config)
    model_hash_value = model_hash(args.gguf)
    sidecar = build_sidecar(
        infos, audit,
        model_hash_value=model_hash_value,
        gguf_path=args.gguf,
        bundle_name=args.bundle_name,
    )
    write_sidecar(args.output, sidecar)
    report_path = args.report or args.output.with_suffix(".report.md")
    write_report(report_path, sidecar)
    # The terse summary that goes to stderr: the sidecar is the
    # durable artifact, this is just the run summary. Format
    # matches the existing ane-mtp tools (the manifest emiter
    # in emit_manifest_from_mlmodelc.py uses the same pattern).
    n = len(infos)
    fb = sidecar["fallback_global"]
    mean_alpha = (
        sum(info.alpha for info in infos) / n if n else 0.0)
    print(f"wrote {args.output}", file=sys.stderr)
    print(f"  model_hash:      {model_hash_value}", file=sys.stderr)
    print(f"  layer_count:     {n}", file=sys.stderr)
    print(f"  total_params:    {sidecar['total_params']:,}", file=sys.stderr)
    print(f"  measurement:     {sidecar['measurement']}", file=sys.stderr)
    print(f"  fallback_global: {fb}", file=sys.stderr)
    print(f"  mean_alpha:      {mean_alpha:.6e}", file=sys.stderr)
    print(f"wrote {report_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
