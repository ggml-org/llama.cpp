"""L5 sensitivity metric computation.

The orchestrator does not run a live inference pass inside the L5 loop, so
per-tensor sensitivity is a calibration-data-free proxy built from three
cheap signals: imatrix magnitude, gradient-of-L2-error under a bit
perturbation, and a layer-position prior.  The three components are
weighted-summed and run through an EMA so the requant plan is stable
across iterations.  Pure stdlib, no numpy/torch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping


# -- Type alias -------------------------------------------------------------

ComponentScores = dict[str, float]


# -- Bit ladder -------------------------------------------------------------

# Quant progression used by the requant planner.  The Tessera T640 ternary
# is the default starting point and sits one rung below Q2_K.  The
# K-quants (Q4_K..Q8_0) follow the same direction the Tessera runtime
# already understands; BF16 is the escape hatch.
BIT_LADDER: tuple[str, ...] = (
    "Q2_K",
    "Q3_K",
    "Q4_K",
    "Q5_K",
    "Q6_K",
    "Q8_0",
    "BF16",
)
TESSERA_DEFAULT = "TSQ-T640"

# Effective bits per weight (llama.cpp packing density).  T640 sits at
# 2.4 bits once the outlier and 32-page overhead are amortised.
BITS_PER_WEIGHT: dict[str, float] = {
    "TSQ-T640": 2.4,
    "Q2_K": 2.5,
    "Q3_K": 3.5,
    "Q4_K": 4.5,
    "Q5_K": 5.5,
    "Q6_K": 6.5,
    "Q8_0": 8.5,
    "BF16": 16.0,
    "F16": 16.0,
    "F32": 32.0,
}

DEFAULT_WEIGHTS = (0.5, 0.3, 0.2, 0.0)
# Phase 0.5: the EXL2 per-layer error is the 4th evidence
# signal the L5 orchestrator can fold into the per-tensor
# sensitivity score. The default ``w_exl2 = 0.0`` keeps the
# path opt-in until the first EXL2 run lands (when
# ``w_exl2 == 0.0``, the EXL2 term contributes zero, so the
# math is byte-equivalent to the 3-component path; existing
# callers that ignore the 4th component see no change). The
# ``combine`` / ``decompose`` helpers are 4-arg, but the
# orchestrator's ``SensitivityScorer`` decides whether to
# pass an EXL2 component dict or an empty one based on the
# runtime ``w_exl2`` and the EXL2 row availability in
# ``exl2_layer_stats``.
DEFAULT_EXL2_WEIGHT = 0.0


# -- Component functions ----------------------------------------------------


def imatrix_magnitude(imatrix: Mapping[str, float] | None) -> ComponentScores:
    """Normalise a per-tensor imatrix to ``[0, 1]`` by the peak magnitude.

    Returns an empty dict when the imatrix is missing; the orchestrator
    falls back to the gradient and layer-prior components in that case.
    """
    if not imatrix:
        return {}
    finite = {n: max(0.0, float(v)) for n, v in imatrix.items()}
    peak = max(finite.values()) if finite else 0.0
    if peak <= 0.0:
        return {n: 0.0 for n in finite}
    return {n: v / peak for n, v in finite.items()}


def gradient_proxy(
    mse_current: Mapping[str, float],
    mse_minus_one_bit: Mapping[str, float],
) -> ComponentScores:
    """Approximate ``d mse / d bits`` from two L4 probe samples.

    ``mse_minus_one_bit`` is the per-tensor MSE reported with one rung of
    precision removed.  A positive gradient means the tensor's quality
    degrades when the budget is cut, which is the textbook "protect me"
    signal.  Negative gradients floor to zero.
    """
    out: dict[str, float] = {}
    peak = 0.0
    for name, current in mse_current.items():
        perturbed = mse_minus_one_bit.get(name)
        if perturbed is None:
            out[name] = 0.0
            continue
        grad = max(0.0, float(current) - float(perturbed))
        out[name] = grad
        if grad > peak:
            peak = grad
    if peak <= 0.0:
        return {n: 0.0 for n in out}
    return {n: v / peak for n, v in out.items()}


def layer_position_prior(
    tensor_names: Iterable[str],
    *,
    total_layers: int = 0,
    floor: float = 0.0,
    ceiling: float = 1.0,
) -> ComponentScores:
    """Assign a depth-based prior in ``[floor, ceiling]`` per tensor.

    The default shape is a linear ramp from the first transformer block
    to the last.  Tensors that do not sit inside a block (token embedding,
    output projection, etc.) are scored at the midpoint so they do not
    dominate the ranking.  The block index is parsed from ``blk.<i>....``.
    """
    names = list(tensor_names)
    if not names:
        return {}
    if total_layers < 1:
        return {n: 0.5 * (floor + ceiling) for n in names}

    block_idx: dict[str, int | None] = {}
    for name in names:
        parts = name.split(".")
        idx: int | None = None
        if len(parts) >= 3 and parts[0] == "blk":
            try:
                idx = int(parts[1])
            except ValueError:
                idx = None
        block_idx[name] = idx

    neutral = 0.5 * (floor + ceiling)
    out: dict[str, float] = {}
    for name, idx in block_idx.items():
        if idx is None or idx < 0 or idx >= total_layers:
            out[name] = neutral
            continue
        if total_layers == 1:
            out[name] = ceiling
            continue
        frac = idx / float(total_layers - 1)
        out[name] = floor + (ceiling - floor) * frac
    return out


# -- Combination ------------------------------------------------------------


def exl2_per_layer_error(
    per_layer_errors: Mapping[int, float] | None,
    tensor_names: Iterable[str],
) -> ComponentScores:
    """Phase 0.5: per-tensor EXL2 per-layer-error component.

    Maps the EXL2 per-layer error map ``{layer_index:
    per_layer_error}`` (the L5 orchestrator's read from
    ``exl2_layer_stats``) onto the per-tensor names the
    SensitivityScorer scores. The mapping is by
    ``layer_index`` parsed from the tensor name
    (``blk.<i>....`` -> ``i``; tensors without a block
    prefix get 0.0, the same neutral value the
    ``layer_position_prior`` uses).

    The output is a ``ComponentScores`` dict
    ``{tensor_name: per_layer_error_normalized}``. The
    normalization is the same peak-1 the other
    components use: ``per_layer_error / max(per_layer_errors)``
    clipped to ``[0, 1]``. The peak-1 form is what
    makes the weight comparable to the other
    components (which are also peak-1 in ``[0, 1]``).

    When ``per_layer_errors`` is ``None`` or empty, the
    function returns an empty dict; the orchestrator
    treats this as "EXL2 has not been run on this
    model" and the fold is skipped (the 4th column in
    the DataFrame is all-zero, the weight default
    ``w_exl2 = 0.0`` means the term contributes zero
    either way).
    """
    if not per_layer_errors:
        return {}
    finite = {
        int(li): max(0.0, float(v))
        for li, v in per_layer_errors.items()
    }
    peak = max(finite.values()) if finite else 0.0
    if peak <= 0.0:
        return {}
    out: ComponentScores = {}
    for name in tensor_names:
        # Extract the block index the same way the
        # orchestrator's ``_tensor_family`` /
        # ``_layer_for`` do: ``blk.<i>.`` -> ``i``; else
        # 0 (the neutral value the layer_position_prior
        # uses for tensors without a block prefix).
        parts = str(name).split(".")
        idx = 0
        if len(parts) >= 3 and parts[0] == "blk":
            try:
                idx = int(parts[1])
            except ValueError:
                idx = 0
        err = finite.get(idx, 0.0)
        out[name] = max(0.0, min(1.0, err / peak))
    return out


def combine(
    components: tuple[
        ComponentScores, ComponentScores, ComponentScores,
        ComponentScores,
    ],
    weights: tuple[float, float, float, float] = DEFAULT_WEIGHTS,
    model_role: str | None = None,
) -> ComponentScores:
    """Combine the four components into a single sensitivity score.

    The union of names from the four components forms the
    output key set. Missing components contribute zero.

    Phase 0.5: the 4th component is the EXL2 per-layer
    error (``exl2_per_layer_error``). When ``w_exl2 == 0.0``
    (the default), the EXL2 term contributes zero regardless
    of its input value, so the math is byte-equivalent to
    the 3-component path. The orchestrator's
    ``SensitivityScorer`` reads the EXL2 component from the
    ``exl2_layer_stats`` DuckDB table and folds it when
    ``w_exl2 > 0``.

    Phase 16: ``model_role`` is an optional pass-through
    parameter (the orchestrator's ``SensitivityScorer``
    carries the role through the call chain; this helper
    accepts it for API symmetry but the role does not
    change the math). The role is recorded on the
    per-tensor ``RequantAction`` and ``l5_plan_summary``
    rows so the retune's per-(model, model_role, family)
    partition can find the right group.
    """
    w_im, w_grad, w_layer, w_exl2 = weights
    names: set[str] = set()
    for component in components:
        names.update(component.keys())
    out: dict[str, float] = {}
    for name in names:
        out[name] = float(
            w_im * components[0].get(name, 0.0)
            + w_grad * components[1].get(name, 0.0)
            + w_layer * components[2].get(name, 0.0)
            + w_exl2 * components[3].get(name, 0.0)
        )
    return out


def decompose(
    combined_score: float,
    weights: tuple[float, float, float, float] = DEFAULT_WEIGHTS,
    model_role: str | None = None,
) -> tuple[float, float, float, float]:
    """Best-effort inversion of :func:`combine` for a single tensor.

    Given a combined ``sensitivity_score`` and the weights that
    produced it, recover the per-component contributions
    ``(im, grad, layer, exl2)`` such that
    ``w_im * im + w_grad * grad + w_layer * layer + w_exl2 * exl2
    = combined_score`` and ``im == grad == layer == exl2``
    (the uniform-spread assumption).

    This is the same decomposition the orchestrator's
    :class:`SensitivityScorer` uses when storing the per-component
    components on the l5_plan_summary / l5_outcome rows. The
    retune reads those rows; this helper exists for diagnostics
    and for any consumer that has only the combined score and
    wants the (uniform) per-component spread.

    The function is not a true inverse (there are infinitely many
    4-tuples that produce a given combined score); the
    uniform-spread assumption is the most reasonable one when
    the per-tensor components are not available.

    Phase 0.5: the 4th element is the EXL2 per-layer error
    contribution. When ``w_exl2 == 0.0``, the EXL2
    contribution is 0.0 regardless of the combined score; the
    other three contributions absorb the entire score (the
    pre-Phase-0.5 3-component math).

    Phase 16: ``model_role`` is an optional pass-through
    parameter; the role does not change the math. The
    orchestrator's ``SensitivityScorer`` carries the role
    through the call chain so the per-tensor ``RequantAction``
    and ``l5_plan_summary`` rows can be tagged with the
    role for the retune's per-(model, model_role, family)
    partition.

    Edge cases:
      * If all weights are zero (degenerate), returns
        ``(0.0, 0.0, 0.0, 0.0)``. The SensitivityScorer's
        rebalance-when-imatrix-missing path can land in this
        state.
      * If only one weight is non-zero, the entire ``combined_score``
        is attributed to that component.
    """
    w_im, w_grad, w_layer, w_exl2 = weights
    s = float(combined_score)
    # Uniform spread: each component is s / sum(weights). When
    # all weights are equal (the default), this is s.
    total = (
        float(w_im) + float(w_grad) + float(w_layer) + float(w_exl2)
    )
    if total <= 0.0:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        s * w_im / total,
        s * w_grad / total,
        s * w_layer / total,
        s * w_exl2 / total,
    )


# -- EMA tracker ------------------------------------------------------------


@dataclass
class MomentumEMA:
    """Per-tensor exponential moving average of a sensitivity score.

    Update rule: ``ema = decay * ema + (1 - decay) * x``.  Cold start
    seeds the EMA with the first observation so the orchestrator can act
    on iteration 1 without burning a warm-up pass.
    """

    decay: float = 0.9
    values: dict[str, float] = field(default_factory=dict)
    updates: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 < self.decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {self.decay}")

    def update(self, scores: Mapping[str, float]) -> ComponentScores:
        out: dict[str, float] = {}
        for name, value in scores.items():
            prev = self.values.get(name)
            if prev is None:
                new = float(value)
            else:
                new = self.decay * prev + (1.0 - self.decay) * float(value)
            self.values[name] = new
            self.updates[name] = self.updates.get(name, 0) + 1
            out[name] = new
        return out

    def reset(self) -> None:
        self.values.clear()
        self.updates.clear()


# -- Percentile helpers -----------------------------------------------------


def percentile_rank(scores: Mapping[str, float]) -> dict[str, float]:
    """Map raw scores to percentile ranks in ``[0, 1]``.

    Ties receive the average rank of the tied group.  The lowest-scoring
    tensor gets rank 0 and the highest gets 1.
    """
    items = sorted(scores.items(), key=lambda kv: kv[1])
    n = len(items)
    if n == 0:
        return {}
    if n == 1:
        return {items[0][0]: 0.5}
    out: dict[str, float] = {}
    i = 0
    while i < n:
        j = i
        while j + 1 < n and items[j + 1][1] == items[i][1]:
            j += 1
        avg = 0.5 * (i + j) / float(n - 1)
        for k in range(i, j + 1):
            out[items[k][0]] = avg
        i = j + 1
    return out


def pick_top_fraction(scores: Mapping[str, float], fraction: float) -> set[str]:
    """Return the names whose rank is in the top ``fraction`` of the cohort.

    ``fraction == 0.1`` on a 10-tensor set returns exactly one tensor.
    """
    if fraction <= 0.0 or not scores:
        return set()
    if fraction >= 1.0:
        return set(scores.keys())
    threshold = 1.0 - fraction
    return {n for n, r in percentile_rank(scores).items() if r >= threshold}


def pick_bottom_fraction(
    scores: Mapping[str, float], fraction: float
) -> set[str]:
    """Return the names whose rank is in the bottom ``fraction`` of the cohort.

    Counterpart of :func:`pick_top_fraction`.
    """
    if fraction <= 0.0 or not scores:
        return set()
    if fraction >= 1.0:
        return set(scores.keys())
    return {n for n, r in percentile_rank(scores).items() if r < fraction}


# -- Expected MSE delta estimator ------------------------------------------


def expected_mse_delta(
    current_mse: Mapping[str, float],
    current_qtype: Mapping[str, str],
    target_qtype: Mapping[str, str],
    *,
    sensitivity: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Estimate the per-tensor MSE delta from a requantization step.

    Doubling the bit count reduces the quantisation step size by a factor
    of two, so the MSE roughly scales with ``2 ** (-2 * delta_bits)``.
    When sensitivity is available we dampen the estimate by
    ``1 - 0.5 * sensitivity`` to model the outlier-saturation effect on
    high-sensitivity tensors.  We use the per-weight bit rate from
    :data:`BITS_PER_WEIGHT` directly rather than rounding to int, because
    the K-quants are sub-integer and banker's rounding would collapse
    adjacent rungs.
    """
    out: dict[str, float] = {}
    for name, target in target_qtype.items():
        cur = current_qtype.get(name)
        if cur is None or cur == target:
            out[name] = 0.0
            continue
        cur_per = BITS_PER_WEIGHT.get(cur, 0.0)
        tgt_per = BITS_PER_WEIGHT.get(target, 0.0)
        if cur_per <= 0.0 or tgt_per <= 0.0:
            out[name] = 0.0
            continue
        delta_bits = tgt_per - cur_per
        if delta_bits == 0.0:
            out[name] = 0.0
            continue
        base = float(current_mse.get(name, 0.0))
        # Positive delta_bits reduces MSE, negative increases it.
        scale = 2.0 ** (-2.0 * delta_bits)
        damp = 1.0
        if sensitivity is not None:
            damp = max(0.0, 1.0 - 0.5 * float(sensitivity.get(name, 0.0)))
        out[name] = -base * (1.0 - scale) * damp
    return out


# -- Ladder stepping --------------------------------------------------------


def ladder_index(qtype: str) -> int:
    """Index of ``qtype`` in :data:`BIT_LADDER` (or -1 if unknown)."""
    try:
        return BIT_LADDER.index(qtype)
    except ValueError:
        return -1


def step_up(qtype: str) -> str | None:
    """Next-rung quant type, or ``None`` if already at the top.

    T640 is treated as sitting one rung below Q2_K so the first step up
    from a T640 tensor lands on Q2_K.
    """
    if qtype == TESSERA_DEFAULT:
        return BIT_LADDER[0] if BIT_LADDER else None
    idx = ladder_index(qtype)
    if idx < 0 or idx + 1 >= len(BIT_LADDER):
        return None
    return BIT_LADDER[idx + 1]


def step_down(qtype: str) -> str | None:
    """Previous-rung quant type, or ``None`` if already at the bottom.

    Stepping down from Q2_K lands on the T640 ternary; stepping down
    from T640 is a no-op.
    """
    if qtype == TESSERA_DEFAULT:
        return None
    idx = ladder_index(qtype)
    if idx < 0:
        return None
    if idx == 0:
        return TESSERA_DEFAULT
    return BIT_LADDER[idx - 1]


# -- Small helpers ----------------------------------------------------------


def sanitise(scores: Mapping[str, float]) -> ComponentScores:
    """Drop non-finite values and clamp negatives to zero."""
    return {
        n: max(0.0, float(v) if math.isfinite(float(v)) else 0.0)
        for n, v in scores.items()
    }
