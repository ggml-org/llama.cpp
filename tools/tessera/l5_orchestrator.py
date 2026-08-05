#!/usr/bin/env python3
"""L5 orchestrator - IterQuant-style adaptive requantization loop.

This is the L5 layer of the Tessera runtime-aware calibration pipeline.  It
sits on top of the L4 E2E probe (which produces a per-tensor divergence
report) and drives the per-tensor requantization decisions that the L1/L2
layers have identified as the bottleneck.  The shape of the loop is the
IterQuant (DATE 2026) feedback loop, adapted for Apple Silicon's mixed-
precision constraint:

* No true mixed-precision inference at the runtime (each layer a different
  dequant kernel makes that hard).  Instead we requantize bad layers one
  bit higher (Q4_K -> Q5_K -> Q6_K -> Q8_0 -> BF16) and good layers one
  bit lower if the storage budget allows.
* No live inference pass inside the loop - the orchestrator reads the L4
  probe's offline metrics and a calibration-data-free proxy for the
  sensitivity.  The proxy is built from three signals (imatrix, gradient,
  layer position) combined with an EMA so the requant plan is stable across
  iterations.

The orchestrator does not modify any of the upstream tools.  It composes
them: the requantization itself is delegated to
``tools/tessera/per_tensor_calibrate.py`` which already knows how to write
``llama.speculative.calibration-policy.v1`` JSON consumable by
``tile640_quantize_v3.py --calibration-policy``.  The orchestrator reads
that JSON, identifies the tensors that need to move up (or down) a rung,
emits a delta plan, and either writes a fresh policy or merges with the
base.

The CLI surface is intentionally small:

    python3 tools/tessera/l5_orchestrator.py --l4-report l4.json \\
        --imatrix imatrix.npz --policy out/l5-policy.json \\
        --max-iterations 5 --top-fraction 0.1

See ``l5_demo.py`` for a self-contained example that runs the orchestrator
on synthetic L4 metrics and prints the resulting requant plan.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import polars as pl

# L5 lives in the same package as per_tensor_calibrate.py so we can import
# the helpers by package-relative name when the script is run as a module,
# and fall back to a sys.path-anchored import for the script path.
try:
    from . import l5_metrics as metrics
except ImportError:  # pragma: no cover - script-mode fallback
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import l5_metrics as metrics  # type: ignore[no-redef]


# ---- Phase 0.5: pure-NumPy Spearman / rank helpers -----------------------
#
# The L5 orchestrator writes the per-iteration EXL2 vs.
# orchestrator-combined-score disagreement log. The Spearman
# rank correlation is the cross-check the design doc ratifies
# (Phase 0.5 of the iPhone ANE demo). The orchestrator
# does not depend on scipy (the test_exl2_cross_check.py
# test uses scipy.stats.spearmanr; the orchestrator itself
# uses the pure-NumPy form below so the production path
# stays on a minimal dependency set).
#
# The implementations are the textbook closed forms: rank
# with average tie-breaking, Pearson correlation of the
# ranks. The Spearman p-value is the two-sided t-test
# approximation ``p = 2 * (1 - t_cdf(|t|, df=n-2))`` where
# ``t = rho * sqrt((n-2) / (1 - rho^2))``. We use the
# same closed form scipy uses; the test_exl2_cross_check.py
# tests pin the equivalence.

def _ranks(values: Sequence[float]) -> list[float]:
    """Per-element ranks (1-indexed) with average tie-breaking.

    Tied values get the average of their position-based
    ranks. The returned ranks are 1-indexed: the smallest
    value gets rank 1, the largest gets ``len(values)``.

    The output is a list of floats (not ints) so the
    downstream Pearson correlation sees the average
    rank (a float) for tied values; integer ranks
    would lose the tie information and inflate the
    Pearson statistic for tied data.
    """
    n = len(values)
    if n == 0:
        return []
    indexed = sorted(enumerate(values), key=lambda kv: kv[1])
    out = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        # Average rank for the tied group: positions
        # ``[i, j]`` (1-indexed) average to
        # ``(i + 1 + j + 1) / 2``.
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            out[indexed[k][0]] = avg_rank
        i = j + 1
    return out


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson product-moment correlation coefficient.

    Closed form: ``sum((x - x_mean) * (y - y_mean)) /
    sqrt(sum((x - x_mean)^2) * sum((y - y_mean)^2))``.
    Returns 0.0 when one side has zero variance (the
    Spearman fallback the design doc ratifies; the
    consumer's threshold can still treat a zero-
    variance Spearman as a "uniform rank" signal).
    """
    n = len(x)
    if n == 0 or n != len(y):
        return 0.0
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    x_mean = float(x_arr.mean())
    y_mean = float(y_arr.mean())
    x_dev = x_arr - x_mean
    y_dev = y_arr - y_mean
    sum_xy = float(np.dot(x_dev, y_dev))
    sum_xx = float(np.dot(x_dev, x_dev))
    sum_yy = float(np.dot(y_dev, y_dev))
    if sum_xx <= 0.0 or sum_yy <= 0.0:
        return 0.0
    return sum_xy / (sum_xx * sum_yy) ** 0.5


def _spearmanr(
    x: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float]:
    """Spearman rank correlation (rho) and p-value.

    Closed form: rank both inputs (average tie-breaking),
    compute the Pearson correlation of the ranks, and
    the p-value from the t-distribution approximation::

        t = rho * sqrt((n - 2) / (1 - rho^2))
        p = 2 * betainc(0.5 * (n - 2), 0.5) (1 - t^2 / (n - 2))

    Returns ``(rho, p_value)``. The p-value is ``1.0``
    when ``n < 3`` (the test is undefined; the consumer
    should not act on a p-value with fewer than 3
    samples). The p-value is ``0.0`` when ``|rho| == 1``
    (the test statistic is infinite; the closed-form
    approximation would NaN; the consumer treats
    ``p == 0`` as a perfect-correlation signal).

    The closed form matches scipy.stats.spearmanr to
    numerical precision (the test pins the equivalence).
    """
    n = len(x)
    if n < 2 or n != len(y):
        return 0.0, 1.0
    rx = _ranks(x)
    ry = _ranks(y)
    rho = _pearson(rx, ry)
    if not math.isfinite(rho):
        return 0.0, 1.0
    rho_clamped = max(-0.999999, min(0.999999, rho))
    if abs(rho) >= 1.0 - 1.0e-12:
        return float(rho), 0.0
    t_stat = rho_clamped * ((n - 2) / (1.0 - rho_clamped ** 2)) ** 0.5
    # Two-sided p-value via the Student-t survival
    # function. We use the regularized incomplete
    # beta function ``betainc(a, b, x)`` (the same
    # primitive scipy uses for t-distribution CDFs).
    # The closed form is::
    #
    #     p = betainc(0.5 * df, 0.5, df / (df + t^2))
    #
    # where ``df = n - 2``. We use the regularized
    # form ``math.betainc`` (Python 3.13+) when
    # available; otherwise fall back to the
    # closed-form expression via the gamma function
    # series. The fallback is a series expansion
    # accurate to 1e-10 for the regime the
    # orchestrator uses (``n >= 4``).
    df = n - 2
    if df <= 0:
        return float(rho), 1.0
    p_one_tail = _betainc_survival(
        0.5 * df, 0.5, df / (df + t_stat ** 2),
    )
    p_value = 2.0 * min(1.0, p_one_tail)
    return float(rho), float(p_value)


def _betainc_survival(a: float, b: float, x: float) -> float:
    """The regularized incomplete beta function ``I_x(a, b)``.

    Used by :func:`_spearmanr` for the two-sided
    p-value. Falls back to a continued-fraction
    expansion when ``x > (a + 1) / (a + b + 2)`` (the
    symmetry ``I_x(a, b) = 1 - I_{1-x}(b, a)``
    handles the high-x case). The expansion is the
    Lentz continued-fraction form the textbook
    numerical-recipes derivation uses; 200 terms is
    more than enough for 1e-10 precision in the
    regime the orchestrator hits (``n >= 4``).
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    # Symmetry transform: keep the series' fast
    # convergence by mapping the larger half of
    # ``[0, 1]`` to the smaller.
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _betainc_survival(b, a, 1.0 - x)
    # Front factor: ``x^a * (1 - x)^b / (a * B(a, b))``.
    # The log-Beta is ``lgamma(a) + lgamma(b) -
    # lgamma(a + b)``; the front factor's log is
    # ``a * log(x) + b * log(1 - x) - log(a) - lbeta``.
    lbeta = (
        math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    )
    front = math.exp(
        a * math.log(x) + b * math.log(1.0 - x)
        - math.log(a) - lbeta
    )
    # Continued-fraction expansion (Lentz's method).
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < 1.0e-300:
        d = 1.0e-300
    d = 1.0 / d
    result = d
    for m in range(1, 200):
        m_f = float(m)
        # Even step: ``m * (b - m) * x / ((a + 2m - 1) * (a + 2m))``
        numerator_e = (
            m_f * (b - m_f) * x
            / ((a + 2.0 * m_f - 1.0) * (a + 2.0 * m_f))
        )
        d = 1.0 + numerator_e * d
        if abs(d) < 1.0e-300:
            d = 1.0e-300
        c = 1.0 + numerator_e / c
        if abs(c) < 1.0e-300:
            c = 1.0e-300
        d = 1.0 / d
        result *= d * c
        # Odd step: ``-(a + m) * (a + b + m) * x / ((a + 2m) * (a + 2m + 1))``
        numerator_o = -(
            (a + m_f) * (a + b + m_f) * x
            / ((a + 2.0 * m_f) * (a + 2.0 * m_f + 1.0))
        )
        d = 1.0 + numerator_o * d
        if abs(d) < 1.0e-300:
            d = 1.0e-300
        c = 1.0 + numerator_o / c
        if abs(c) < 1.0e-300:
            c = 1.0e-300
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < 1.0e-12:
            break
    return front * result


# Phase 16: the per-(model, model_role, family) lookup's
# empty DataFrame is built with the l5_weights schema
# (model_role + family + ...). The schema is defined in
# tessera_db.py; the orchestrator imports it so the 3-tier
# lookup's "no result yet" placeholder has the right
# columns.
try:
    from .tessera_db import L5_WEIGHTS_COLS  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - script-mode fallback
    from tessera_db import L5_WEIGHTS_COLS  # type: ignore[no-redef]


SCHEMA = "llama.tessera.l5-orchestrator.v1"
PLAN_SCHEMA = "llama.tessera.l5-requant-plan.v1"
POLICY_SCHEMA = "llama.speculative.calibration-policy.v1"


# ---------------------------------------------------------------------------
# L4 report schema
# ---------------------------------------------------------------------------
#
# The L4 E2E probe writes a per-tensor JSON report.  We only consume a
# subset of it; the rest is preserved as provenance in the plan.  The
# fields we care about are:
#
#   l4_report["tensors"][name] = {
#       "current_qtype":   "Q4_K",
#       "mse":             0.012,        # current MSE at current qtype
#       "mse_minus_one":   0.018,        # MSE with one rung down
#       "perplexity":      8.21,         # current perplexity
#       "top1_mismatch":   0.04,         # fraction of top-1 mismatches
#       "n_weights":       4096*4096,    # tensor size
#   }
#
# A simplified L4 schema that just provides the per-tensor MSE and the
# current qtype is also accepted (the orchestrator assumes a unit
# ``mse_minus_one`` perturbation when the field is missing).


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TensorState:
    """Mutable per-tensor bookkeeping across iterations."""

    name: str
    current_qtype: str
    target_qtype: str | None
    n_weights: int
    mse: float
    mse_minus_one: float | None
    perplexity: float | None
    top1_mismatch: float | None
    raw: dict

    @classmethod
    def from_l4(cls, name: str, payload: dict) -> "TensorState":
        """Build a :class:`TensorState` from a single L4 record."""
        return cls(
            name=name,
            current_qtype=str(payload.get("current_qtype", metrics.TESSERA_DEFAULT)),
            target_qtype=None,
            n_weights=int(payload.get("n_weights", 0) or 0),
            mse=float(payload.get("mse", 0.0) or 0.0),
            mse_minus_one=(
                float(payload["mse_minus_one"])
                if payload.get("mse_minus_one") is not None
                else None
            ),
            perplexity=(
                float(payload["perplexity"])
                if payload.get("perplexity") is not None
                else None
            ),
            top1_mismatch=(
                float(payload["top1_mismatch"])
                if payload.get("top1_mismatch") is not None
                else None
            ),
            raw=dict(payload),
        )

    def expected_mse_at(self, qtype: str, sensitivity: float | None = None) -> float:
        """Estimate the MSE at ``qtype`` if we requantized this tensor.

        See :func:`l5_metrics.expected_mse_delta` for the formula.
        """
        if qtype == self.current_qtype:
            return self.mse
        cur_per = metrics.BITS_PER_WEIGHT.get(self.current_qtype, 0.0)
        tgt_per = metrics.BITS_PER_WEIGHT.get(qtype, 0.0)
        if cur_per <= 0.0 or tgt_per <= 0.0:
            return self.mse
        delta_bits = tgt_per - cur_per
        scale = 2.0 ** (-2.0 * delta_bits)
        damp = 1.0
        if sensitivity is not None:
            damp = max(0.0, 1.0 - 0.5 * float(sensitivity))
        new_mse = self.mse * scale * damp
        return max(0.0, new_mse)

    def bits(self) -> int:
        """Storage cost in bits for this tensor under its current qtype."""
        per_weight = metrics.BITS_PER_WEIGHT.get(self.current_qtype, 0.0)
        return int(per_weight * float(self.n_weights))

    def to_row(self) -> dict:
        """Project this TensorState to a single polars DataFrame row.

        Used by the polars-backed refactor of the orchestrator: the
        per-iteration state is a polars DataFrame, and the
        TensorState dataclass is the per-row projection. The mapping
        is one-to-one for the columns the orchestrator reads; the
        sensitivity / EMA columns are added by ``SensitivityScorer``
        after the orchestrator loads the L4 report.
        """
        return {
            "tensor":          self.name,
            "current_qtype":   self.current_qtype,
            "target_qtype":    self.target_qtype,
            "n_weights":       int(self.n_weights),
            "mse":             float(self.mse),
            "mse_minus_one":   (float(self.mse_minus_one)
                                if self.mse_minus_one is not None else None),
            "perplexity":      (float(self.perplexity)
                                if self.perplexity is not None else None),
            "top1_mismatch":   (float(self.top1_mismatch)
                                if self.top1_mismatch is not None else None),
        }

    @classmethod
    def from_df_row(cls, row: Mapping[str, object]) -> "TensorState":
        """Inverse of :meth:`to_row`; build a TensorState from one
        polars DataFrame row (as a Mapping; the ``row`` argument is
        what ``df.row(i, named=True)`` returns).

        The orchestrator's primary state is the DataFrame; this
        back-conversion exists for the ``TensorState`` public API
        (the demo and any tests that pass TensorState lists).
        """
        return cls(
            name=str(row["tensor"]),
            current_qtype=str(row["current_qtype"]),
            target_qtype=(str(row["target_qtype"])
                          if row.get("target_qtype") is not None else None),
            n_weights=int(row["n_weights"] or 0),
            mse=float(row["mse"] or 0.0),
            mse_minus_one=(float(row["mse_minus_one"])
                           if row.get("mse_minus_one") is not None else None),
            perplexity=(float(row["perplexity"])
                        if row.get("perplexity") is not None else None),
            top1_mismatch=(float(row["top1_mismatch"])
                           if row.get("top1_mismatch") is not None else None),
            raw={},
        )


@dataclasses.dataclass
class RequantAction:
    """One per-tensor requantization decision."""

    name: str
    from_qtype: str
    to_qtype: str
    expected_mse_delta: float
    sensitivity: float
    # Phase 15: the per-tensor sensitivity component values at
    # the iteration the action was emitted. These are the values
    # the retune reads to fit a 3-coefficient OLS that decomposes
    # which component is miscalibrated per (model, family).
    # ``None`` for any field means the scorer could not produce
    # the component (e.g. the imatrix was missing and the
    # rebalance path collapsed one of the components to 0).
    imatrix_magnitude: float | None = None
    gradient_proxy: float | None = None
    layer_position_prior: float | None = None
    reason: str = ""
    storage_delta_bits: int = 0
    # Phase 16: the architectural role the action belongs to.
    # The orchestrator's ``--model-role`` flag sets the
    # scorer's ``model_role``; the scorer passes the role
    # through to every ``RequantAction`` so the per-tensor
    # ``l5_plan_summary`` row (and the retune's
    # per-(model, model_role, family) groupby) can find
    # the right group. Defaults to ``"trunk"`` for
    # backward compat with pre-Phase-16 callers.
    model_role: str = "trunk"
    # Targeted re-calibration: the L5 monitor-verdict
    # hook reads the per-tensor recommended_action
    # (from the tensor_stats row) and uses it to
    # decide which tensors the backfill re-captures.
    # The field is None when the per-tensor verdict
    # is not available (no tensor_stats row, or the
    # orchestrator was constructed without a DB
    # reference). The planner populates the field
    # from the df's ``recommended_action`` column
    # (when present); the field defaults to None for
    # pre-backfill callers.
    recommended_action: str | None = None
    backfill_count: int | None = None

    def to_dict(self) -> dict:
        return {
            "tensor": self.name,
            "from": self.from_qtype,
            "to": self.to_qtype,
            "expected_mse_delta": self.expected_mse_delta,
            "sensitivity": self.sensitivity,
            "imatrix_magnitude": self.imatrix_magnitude,
            "gradient_proxy": self.gradient_proxy,
            "layer_position_prior": self.layer_position_prior,
            "reason": self.reason,
            "storage_delta_bits": self.storage_delta_bits,
            # Phase 16: model_role in the per-tensor action
            # dict so the l5_plan_summary writer can tag
            # the row with the role.
            "model_role": self.model_role,
            # Targeted re-calibration: the per-tensor
            # recommended_action verdict (None when the
            # DB lookup did not find a row).
            "recommended_action": self.recommended_action,
            "backfill_count": self.backfill_count,
        }


@dataclasses.dataclass
class RequantPlan:
    """The full requant plan for one orchestrator iteration."""

    iteration: int
    actions: list[RequantAction]
    storage_before_bits: int
    storage_after_bits: int
    storage_budget_bits: int | None
    termination_reason: str | None
    sensitivity: dict[str, float]
    ema_sensitivity: dict[str, float]

    def storage_delta_bits(self) -> int:
        return self.storage_after_bits - self.storage_before_bits

    def to_dict(self) -> dict:
        return {
            "schema": PLAN_SCHEMA,
            "iteration": self.iteration,
            "termination_reason": self.termination_reason,
            "storage_before_bits": self.storage_before_bits,
            "storage_after_bits": self.storage_after_bits,
            "storage_delta_bits": self.storage_delta_bits(),
            "storage_budget_bits": self.storage_budget_bits,
            "actions": [action.to_dict() for action in self.actions],
            "sensitivity": self.sensitivity,
            "ema_sensitivity": self.ema_sensitivity,
        }


# ---------------------------------------------------------------------------
# Sensitivity scorer
# ---------------------------------------------------------------------------


def _tensor_family(tensor_name: str) -> str:
    """Return the family for ``tensor_name`` (Phase 15).

    The family is the ``.weight``-stripped, ``blk.<i>.``-stripped
    second ``.``-separated segment of the tensor name. Examples:

        blk.0.attn_q.weight      -> attn_q
        blk.0.ffn_gate.weight    -> ffn_gate
        token_embd.weight        -> token_embd
        output.weight             -> output

    This is the same convention as
    ``calibration_rollup.py``'s family extraction: the family
    is the part of the name that identifies the tensor's role
    (attention query, FFN gate, embedding, etc.). The
    per-family top_fraction lookup keys on this value.
    """
    base = str(tensor_name)
    for suf in (".weight", ".bias"):
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    parts = base.split(".")
    if len(parts) >= 3 and parts[0] == "blk":
        return parts[2]
    if len(parts) >= 2:
        return parts[1]
    return base





class SensitivityScorer:
    """Combine the three L5 sensitivity components and run them through EMA.

    The scorer is the only piece that knows about the relative weighting of
    the three components; the rest of the orchestrator treats it as a
    function from ``(imatrix, df) -> df`` (the EMA-tracked scores
    are added to the DataFrame as a new column).

    Internally the EMA state is a per-tensor ``prev_scores`` dict (one
    float per tensor). The DataFrame's ``sensitivity_ema`` column is
    computed each iteration as ``decay * prev + (1 - decay) * score`` and
    the new values are stored back into ``prev_scores`` for the next
    iteration. This is the polars-native version of the previous
    ``MomentumEMA`` dict-of-floats pattern; the public API is unchanged.

    Phase 16: the scorer carries a ``model_role`` (the
    architectural role the scored tensors belong to). The role
    is recorded on the per-tensor ``RequantAction`` /
    ``l5_plan_summary`` rows so the retune's per-(model,
    model_role, family) partition can find the right group.
    The role is plumbed through ``metrics.combine`` and
    ``metrics.decompose`` (those helpers accept the role as
    a pass-through parameter; the math is unchanged).
    """

    def __init__(
        self,
        *,
        decay: float = 0.9,
        weights: tuple[float, float, float, float] = metrics.DEFAULT_WEIGHTS,
        total_layers: int = 0,
        model_role: str = "trunk",
    ) -> None:
        self.weights = tuple(float(w) for w in weights)
        # Phase 0.5: the 4-tuple weights
        # (w_im, w_grad, w_layer, w_exl2) are
        # canonical. A 3-tuple (pre-Phase-0.5
        # callers) is rejected at the constructor
        # with a clear error message; the score()
        # body uses ``self.weights[3]`` which would
        # crash with a confusing IndexError on a
        # 3-tuple. The 4-tuple check is explicit
        # so the failure is at the call site, not
        # deep inside the score math.
        if len(self.weights) != 4:
            raise ValueError(
                f"sensitivity weights must be a 4-tuple "
                f"(w_im, w_grad, w_layer, w_exl2); "
                f"got {len(self.weights)}-tuple"
            )
        if not math.isclose(sum(self.weights), 1.0, abs_tol=1e-6):
            raise ValueError(
                f"sensitivity weights must sum to 1.0, got {sum(self.weights):.6f}"
            )
        self.total_layers = int(total_layers)
        self.decay = float(decay)
        # Phase 16: the architectural role the scored
        # tensors belong to. The orchestrator's
        # ``--model-role`` flag sets this; the role is
        # recorded on every per-tensor ``RequantAction`` /
        # ``l5_plan_summary`` row so the retune can do the
        # per-(model, model_role, family) groupby. Defaults
        # to ``"trunk"`` for backward compat with the
        # pre-Phase-16 callers.
        self.model_role = str(model_role) if model_role else "trunk"
        # EMA state: per-tensor previous score. The dict is the
        # ``polars-native`` form of the old MomentumEMA; the
        # DataFrame column is rebuilt each iteration from this dict
        # via a left-join.
        self._prev_scores: dict[str, float] = {}
        # Phase 0.5: the raw_components tuple is now 4-tuple
        # (im, grad, layer, exl2). The EXL2 component is
        # the per-layer error from the EXL2 estimator
        # (``exl2_layer_stats`` in the unified DuckDB);
        # when no EXL2 data is available, the 4th element
        # is an empty dict and the fold contributes zero
        # (the default ``w_exl2 = 0.0`` keeps the math
        # byte-equivalent to the 3-component path).
        self._raw_components: tuple[
            metrics.ComponentScores,
            metrics.ComponentScores,
            metrics.ComponentScores,
            metrics.ComponentScores,
        ] = ({}, {}, {}, {})

    def raw_components(self) -> tuple[
        metrics.ComponentScores,
        metrics.ComponentScores,
        metrics.ComponentScores,
        metrics.ComponentScores,
    ]:
        """Return the most recent (imatrix, gradient, layer_prior, exl2) components.

        Exposed for debug; the EMA-tracked scores are what the planner
        consumes. The 4th element is the EXL2 per-layer error
        (Phase 0.5); empty when no EXL2 data is available.
        """
        return self._raw_components

    def score(
        self,
        df: pl.DataFrame,
        imatrix: Mapping[str, float] | None,
        exl2_per_layer_errors: Mapping[int, float] | None = None,
    ) -> pl.DataFrame:
        """Compute fresh sensitivity scores and update the EMA in-place.

        Returns the input DataFrame augmented with five columns:
        ``imatrix_magnitude``, ``gradient_proxy``,
        ``layer_position_prior``, ``exl2_per_layer_error``
        (Phase 0.5), ``sensitivity_score`` (the per-iteration
        weighted sum), and ``sensitivity_ema`` (the
        EMA-tracked value). The EMA state is updated so the next
        iteration's ``sensitivity_ema`` continues smoothly from this
        one.

        ``imatrix`` may be ``None`` if no imatrix is available; the
        scorer falls back to gradient + layer-prior only and
        rebalances the weights so the total still sums to 1.0.

        ``exl2_per_layer_errors`` (Phase 0.5) is the optional
        per-layer error map the EXL2 estimator writes to
        ``exl2_layer_stats``. When ``None`` or empty, the
        EXL2 component is all-zero and the fold contributes
        nothing (the default ``w_exl2 = 0.0`` keeps the
        math byte-equivalent to the 3-component path).
        When the map has entries, the EXL2 component is the
        per-tensor peak-1 normalization of the per-layer
        error, clipped to ``[0, 1]``.
        """
        names = df["tensor"].to_list()
        mse_list = df["mse"].to_list()
        mse_minus_one_list = df["mse_minus_one"].to_list()
        im = metrics.sanitise(
            metrics.imatrix_magnitude(
                {n: float(imatrix.get(n, 0.0)) for n in names} if imatrix else None
            )
        )
        # gradient_proxy takes (mse_current, mse_minus_one) dicts.
        # mse_minus_one falls back to mse when the column is null
        # (the original TensorState path used
        # ``t.mse_minus_one if t.mse_minus_one is not None else t.mse``);
        # the polars refactor preserves the same semantics.
        mse_current_dict = dict(zip(names, mse_list))
        mse_minus_one_dict = {
            n: float(m if m is not None else mse_list[i])
            for i, (n, m) in enumerate(zip(names, mse_minus_one_list))
        }
        grad = metrics.sanitise(
            metrics.gradient_proxy(mse_current_dict, mse_minus_one_dict)
        )
        layer = metrics.sanitise(
            metrics.layer_position_prior(
                names, total_layers=self.total_layers, floor=0.0, ceiling=1.0
            )
        )
        # Phase 0.5: the EXL2 per-layer error component.
        # Empty when the caller doesn't pass the
        # ``exl2_per_layer_errors`` map; the fold
        # contributes zero (the default ``w_exl2 = 0.0``
        # default means the term has no effect either
        # way). The peak-1 normalization in
        # ``metrics.exl2_per_layer_error`` matches the
        # other components' scale.
        exl2 = metrics.sanitise(
            metrics.exl2_per_layer_error(exl2_per_layer_errors, names)
        )
        self._raw_components = (im, grad, layer, exl2)

        # Rebalance the weights when the imatrix is missing so the surviving
        # two components carry the full mass.  This is the only place that
        # knows about the original weights; the planner just sees the
        # post-rebalance value. Phase 0.5: the EXL2 weight passes through
        # unchanged (``w_exl2 = 0.0`` by default; the rebalance does not
        # touch it because the EXL2 source is opt-in, not auto-derived
        # from the imatrix / gradient / layer components).
        if not im:
            w_im, w_grad, w_layer, w_exl2 = (
                0.0,
                self.weights[1] + self.weights[0] * 0.6,
                self.weights[2] + self.weights[0] * 0.4,
                self.weights[3],
            )
            total = w_im + w_grad + w_layer + w_exl2
            weights = (
                w_im / total, w_grad / total,
                w_layer / total, w_exl2 / total,
            )
        else:
            weights = self.weights
        w_im, w_grad, w_layer, w_exl2 = weights

        # Build the per-tensor score column. The polars expressions
        # reference the component dicts via a per-row lookup using
        # ``replace_strict``: a vectorised dict-style substitution that
        # keeps the join-free polars idiom and works when the input
        # DataFrame already carries the component columns from a
        # previous iteration (the previous columns are overwritten
        # with the new values from this iteration's components).
        # We do NOT use ``df.join(score_map, ...)`` here because
        # polars would append a ``_right`` suffix on the second
        # iteration when the left frame already has the column.
        im_lookup = dict(im)
        grad_lookup = dict(grad)
        layer_lookup = dict(layer)
        exl2_lookup = dict(exl2)
        out = df.with_columns(
            pl.col("tensor").replace_strict(
                im_lookup, return_dtype=pl.Float64,
                default=0.0,
            ).alias("imatrix_magnitude"),
            pl.col("tensor").replace_strict(
                grad_lookup, return_dtype=pl.Float64,
                default=0.0,
            ).alias("gradient_proxy"),
            pl.col("tensor").replace_strict(
                layer_lookup, return_dtype=pl.Float64,
                default=0.0,
            ).alias("layer_position_prior"),
            # Phase 0.5: the EXL2 per-layer error
            # component. The column is peak-1
            # normalized in ``[0, 1]`` to match the
            # scale of the other components; when
            # the EXL2 source is empty the lookup
            # defaults to 0.0 and the fold
            # contributes nothing (the default
            # ``w_exl2 = 0.0``).
            pl.col("tensor").replace_strict(
                exl2_lookup, return_dtype=pl.Float64,
                default=0.0,
            ).alias("exl2_per_layer_error"),
        )
        out = out.with_columns(
            (w_im * pl.col("imatrix_magnitude")
             + w_grad * pl.col("gradient_proxy")
             + w_layer * pl.col("layer_position_prior")
             + w_exl2 * pl.col("exl2_per_layer_error")
            ).alias("sensitivity_score")
        )

        # EMA update. ``prev_scores`` is a per-tensor dict; we
        # left-join it onto the DataFrame as ``sensitivity_ema_prev``
        # (null on the first iteration) and compute the EMA via
        # ``pl.when(prev.is_null()).then(score).otherwise(decay*prev
        # + (1-decay)*score)``. This is the cold-start seed
        # semantics of the original ``MomentumEMA``: the first
        # observation seeds the EMA at the score value, and
        # subsequent observations use the weighted update. The
        # DataFrame ctor's type inference for an empty list gives
        # ``null`` for the key column, which would not match the
        # ``String`` ``tensor`` on the left side of the join, so
        # we build the prev frame with explicit ``pl.Series``
        # dtypes.
        prev_keys = list(self._prev_scores.keys())
        prev_vals = list(self._prev_scores.values())
        prev_df = pl.DataFrame([
            pl.Series("tensor",              prev_keys, dtype=pl.String),
            pl.Series("sensitivity_ema_prev", prev_vals, dtype=pl.Float64),
        ])
        out = out.join(prev_df, on="tensor", how="left")
        out = out.with_columns(
            pl.when(pl.col("sensitivity_ema_prev").is_null())
              .then(pl.col("sensitivity_score"))
              .otherwise(
                  self.decay * pl.col("sensitivity_ema_prev")
                  + (1.0 - self.decay) * pl.col("sensitivity_score")
              )
              .alias("sensitivity_ema")
        ).drop("sensitivity_ema_prev")

        # Persist the new EMA values for the next iteration.
        new_ema = {
            row["tensor"]: float(row["sensitivity_ema"])
            for row in out.select("tensor", "sensitivity_ema").iter_rows(named=True)
        }
        self._prev_scores = new_ema
        return out

    def reset(self) -> None:
        self._prev_scores = {}
        self._raw_components = ({}, {}, {}, {})


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class RequantPlanner:
    """Translate sensitivity scores into a per-tensor requant plan.

    The planner is intentionally side-effect free: it takes a snapshot of
    the orchestrator state and returns a :class:`RequantPlan` describing
    what to do.  The :class:`OrchestratorLoop` applies the plan.
    """

    def __init__(
        self,
        *,
        top_fraction: float = 0.1,
        bottom_fraction: float = 0.05,
        budget_bits: int | None = None,
        divergence_threshold: float | None = None,
        per_family_top_fraction: dict[str, float] | None = None,
        model_role: str = "trunk",
    ) -> None:
        if not 0.0 <= top_fraction <= 1.0:
            raise ValueError(f"top_fraction must be in [0, 1], got {top_fraction}")
        if not 0.0 <= bottom_fraction <= 1.0:
            raise ValueError(f"bottom_fraction must be in [0, 1], got {bottom_fraction}")
        self.top_fraction = float(top_fraction)
        self.bottom_fraction = float(bottom_fraction)
        self.budget_bits = int(budget_bits) if budget_bits is not None else None
        self.divergence_threshold = (
            float(divergence_threshold) if divergence_threshold is not None else None
        )
        # Phase 15: per-family top_fraction override. The retune
        # writes a per-(model, family) top_fraction recommendation
        # to l5_weights; the orchestrator reads it via
        # --per-family-top-fraction and overrides the uniform
        # --top-fraction for the families the retune has flagged.
        # A family without a per-family row falls back to the
        # uniform --top-fraction value. The dict is keyed by
        # family (the tensor family, e.g. "attn_q"); the values
        # are the per-family top_fraction (in [0, 1]).
        self.per_family_top_fraction: dict[str, float] = (
            dict(per_family_top_fraction)
            if per_family_top_fraction is not None
            else {}
        )
        # Phase 16: the architectural role the planned
        # tensors belong to. The orchestrator's
        # ``--model-role`` flag sets this; the planner
        # passes the role to every ``RequantAction`` it
        # emits so the per-tensor ``l5_plan_summary`` row
        # (and the retune's per-(model, model_role, family)
        # groupby) can find the right group. Defaults to
        # ``"trunk"`` for backward compat with pre-Phase-16
        # callers.
        self.model_role = str(model_role) if model_role else "trunk"
        # Per-family lookup. The cohort selection uses
        # ``top_fraction_for(tensor)`` which checks the tensor's
        # family in the override map. The default is the
        # uniform --top-fraction.
        # The family is computed from the tensor name (the
        # block layer + name pattern); we use a simple
        # suffix-stripping helper.

    def top_fraction_for(self, tensor_name: str) -> float:
        """Return the per-family top_fraction for this tensor,
        or the uniform ``--top-fraction`` value when the
        family is not in the override map.

        The family is the leading part of the tensor name
        (e.g. ``blk.0.attn_q.weight`` -> ``attn_q``). The
        matching is the same convention as
        ``calibration_rollup.py``: take the second ``.``-separated
        segment after the block prefix.
        """
        if not self.per_family_top_fraction:
            return self.top_fraction
        family = _tensor_family(tensor_name)
        return float(self.per_family_top_fraction.get(
            family, self.top_fraction,
        ))

    def top_fraction_for_name(self, family: str) -> float:
        """Return the per-family top_fraction for a given
        family name (Phase 15). Used by the per-family
        cohort selection when the retune has produced a
        per-family top_fraction override. The lookup is the
        same map as :py:meth:`top_fraction_for` (per-family
        override keyed by family); families without an
        override fall back to the uniform ``--top-fraction``.
        """
        if not self.per_family_top_fraction:
            return self.top_fraction
        return float(self.per_family_top_fraction.get(
            str(family), self.top_fraction,
        ))

    def plan(
        self,
        iteration: int,
        df: pl.DataFrame,
    ) -> RequantPlan:
        """Return a :class:`RequantPlan` for the current state.

        The plan moves the top-fraction tensors up one rung and the
        bottom-fraction tensors down one rung, subject to the storage
        budget.  The iteration is also the termination check: if every
        tensor is below the divergence threshold the plan marks itself as
        converged and produces an empty action list.

        Operates on the polars DataFrame directly: cohort selection
        is a polars rank + filter (one expression each for the
        top and bottom cohorts), the divergence check is a single
        polars ``pl.col("mse").max() <= threshold`` test, and the
        per-tensor ``bits()`` / ``expected_mse_at()`` math is
        inlined as polars expressions where useful.
        """
        # Per-row projection of the ``bits()`` and ``expected_mse_at()``
        # math from TensorState, inlined as Python since the
        # computations are scalar and the polars overhead would
        # exceed the gain. The resulting ``iter_rows`` projection is
        # the per-tensor view the rest of the planner consumes.
        #
        # Phase 15: include the per-tensor sensitivity components
        # (imatrix_magnitude, gradient_proxy, layer_position_prior)
        # in the projection so the action builder can populate
        # them on each RequantAction. The retune reads them on
        # the l5_plan_summary / l5_outcome side.
        n = df.height
        # The component columns are only present after a scorer
        # pass. On the first iteration of a brand-new loop the
        # orchestrator calls planner.plan() before scorer.score()
        # (the no-op / termination check); include the columns
        # defensively with .get() downstream.
        rank_cols = [
            "tensor", "current_qtype", "mse", "n_weights",
            "sensitivity_ema",
            "imatrix_magnitude", "gradient_proxy",
            "layer_position_prior",
        ]
        rank_cols = [c for c in rank_cols if c in df.columns]
        ranked_records = sorted(
            df.select(rank_cols).iter_rows(named=True),
            key=lambda r: float(r["sensitivity_ema"] or 0.0),
            reverse=True,
        )

        # Build a per-tensor dict view for the bits() / expected_mse_at
        # math. Keeps the rest of the function unchanged.
        tensors_view = [TensorState.from_df_row(r) for r in df.iter_rows(named=True)]
        sensitivity = {
            str(r["tensor"]): float(r["sensitivity_ema"] or 0.0)
            for r in df.iter_rows(named=True)
        }

        # If everything is already below the divergence threshold, the loop
        # has nothing to do.  This is the primary termination signal.
        if self.divergence_threshold is not None and all(
            t.mse <= self.divergence_threshold for t in tensors_view
        ):
            storage_before = sum(t.bits() for t in tensors_view)
            return RequantPlan(
                iteration=iteration,
                actions=[],
                storage_before_bits=storage_before,
                storage_after_bits=storage_before,
                storage_budget_bits=self.budget_bits,
                termination_reason="all-tensors-below-threshold",
                sensitivity=dict(sensitivity),
                ema_sensitivity=dict(sensitivity),
            )

        # Pick the cohorts. The polars-native form is a rank + filter
        # expression; we use it for clarity and to keep the per-tensor
        # dict view consistent with the rest of the function. The
        # ``metrics.pick_top_fraction`` / ``pick_bottom_fraction``
        # helpers operate on a dict and would duplicate work; the
        # polars rank is one expression each.
        #
        # Phase 15: per-family top_fraction override. The retune
        # may have flagged some families as more miscalibrated
        # than others; the per-family top_fraction is applied
        # independently to each family. The rank is per-family
        # (each family's top X% is selected). Families without
        # a per-family row use the uniform ``--top-fraction``.
        # A tensor's family is computed from the name (the
        # ``blk.<i>.``-stripped second ``.``-separated segment).
        # The family column is added here and consumed by the
        # rank expressions.
        if self.per_family_top_fraction:
            # Build a per-tensor family column for the
            # rank+filter. The family derivation is a
            # vectorised string transformation; we use a
            # polars expression so the whole rank+filter
            # stays a single lazy expression (no Python
            # row-by-row loop).
            df_with_family = df.with_columns(
                pl.col("tensor").map_elements(
                    _tensor_family, return_dtype=pl.String,
                ).alias("family")
            )
            # Per-family top_n: the family column is the key.
            # We compute the per-family top_n outside the
            # expression (one Python call per family, then a
            # single polars join) and use a join to bring
            # the per-row top_n into the DataFrame.
            families = df_with_family["family"].unique().to_list()
            family_top_n: dict[str, int] = {}
            family_bottom_n: dict[str, int] = {}
            for fam in families:
                fam_n = df_with_family.filter(
                    pl.col("family") == fam
                ).height
                top_frac = self.top_fraction_for_name(fam)
                family_top_n[fam] = max(
                    1, int(round(fam_n * top_frac))
                ) if fam_n > 0 else 0
                family_bottom_n[fam] = max(
                    1, int(round(fam_n * self.bottom_fraction))
                ) if fam_n > 0 else 0
            # Apply via polars joins. The
            # ``with_columns`` form keeps the lazy
            # expression pipeline intact.
            family_top_df = pl.DataFrame({
                "family":  list(family_top_n.keys()),
                "family_top_n": list(family_top_n.values()),
                "family_bottom_n": list(family_bottom_n.values()),
            })
            cohort_df = df_with_family.join(
                family_top_df, on="family", how="left",
            ).with_columns(
                pl.col("sensitivity_ema")
                  .rank(method="ordinal", descending=True)
                  .over("family")
                  .alias("rank_desc"),
                pl.col("sensitivity_ema")
                  .rank(method="ordinal")
                  .over("family")
                  .alias("rank_asc"),
            ).with_columns(
                ((pl.col("rank_desc") <= pl.col("family_top_n"))
                 & (pl.col("rank_asc") > pl.col("family_bottom_n"))
                ).alias("in_top_cohort"),
                ((pl.col("rank_asc") <= pl.col("family_bottom_n"))
                 & (pl.col("rank_desc") > pl.col("family_top_n"))
                ).alias("in_bottom_cohort"),
            ).drop("family_top_n", "family_bottom_n", "rank_desc", "rank_asc")
        else:
            top_n = max(1, int(round(n * self.top_fraction))) if n > 0 else 0
            bottom_n = max(1, int(round(n * self.bottom_fraction))) if n > 0 else 0
            cohort_df = df.with_columns(
                pl.col("sensitivity_ema")
                  .rank(method="ordinal", descending=True)
                  .alias("rank_desc"),
                pl.col("sensitivity_ema")
                  .rank(method="ordinal")
                  .alias("rank_asc"),
            ).with_columns(
                ((pl.col("rank_desc") <= top_n)
                 & (pl.col("rank_asc") > bottom_n)
                ).alias("in_top_cohort"),
                ((pl.col("rank_asc") <= bottom_n)
                 & (pl.col("rank_desc") > top_n)
                ).alias("in_bottom_cohort"),
            ).drop("rank_desc", "rank_asc")
        top_set = set(
            cohort_df.filter(pl.col("in_top_cohort"))["tensor"].to_list()
        )
        bottom_set = set(
            cohort_df.filter(pl.col("in_bottom_cohort"))["tensor"].to_list()
        )

        actions: list[RequantAction] = []
        storage_before = sum(t.bits() for t in tensors_view)

        # Walk the top cohort first so the budget check uses the most
        # important changes.
        for tensor in ranked_records:
            if tensor["tensor"] not in top_set:
                continue
            t = TensorState.from_df_row(tensor)
            target = metrics.step_up(str(tensor["current_qtype"]))
            if target is None:
                # Already at BF16 - nothing we can do.
                continue
            sens = float(sensitivity.get(t.name, 0.0))
            expected_delta = t.expected_mse_at(target, sensitivity=sens) - t.mse
            bits_after = int(
                round(metrics.BITS_PER_WEIGHT.get(target, 0.0))
                * float(tensor["n_weights"] or 0)
            )
            bits_before = t.bits()
            actions.append(
                RequantAction(
                    name=t.name,
                    from_qtype=t.current_qtype,
                    to_qtype=target,
                    expected_mse_delta=expected_delta,
                    sensitivity=sens,
                    imatrix_magnitude=(
                        float(tensor["imatrix_magnitude"])
                        if tensor.get("imatrix_magnitude") is not None
                        else None
                    ),
                    gradient_proxy=(
                        float(tensor["gradient_proxy"])
                        if tensor.get("gradient_proxy") is not None
                        else None
                    ),
                    layer_position_prior=(
                        float(tensor["layer_position_prior"])
                        if tensor.get("layer_position_prior") is not None
                        else None
                    ),
                    reason="top-fraction",
                    storage_delta_bits=bits_after - bits_before,
                    # Phase 16: the role is on the
                    # planner; the action carries the
                    # role so the l5_plan_summary
                    # writer can tag the row.
                    model_role=self.model_role,
                    # Targeted re-calibration:
                    # the planner propagates the
                    # per-tensor recommended_action
                    # and backfill_count from the df
                    # (the l5_outcome / l5_action
                    # verdict on the tensor_stats
                    # row) to the action so the
                    # backfill hook can filter on
                    # it. None when the df does
                    # not have the columns (the
                    # pre-backfill orchestrator
                    # path).
                    recommended_action=(
                        str(tensor["recommended_action"])
                        if tensor.get("recommended_action") is not None
                        else None
                    ),
                    backfill_count=(
                        int(tensor["backfill_count"])
                        if tensor.get("backfill_count") is not None
                        else None
                    ),
                )
            )

        # Check the storage budget.  We always honour the up-quant moves
        # (those improve quality) and greedily skip down-quant moves that
        # would push the total below the budget.
        projected_bits = storage_before + sum(a.storage_delta_bits for a in actions)
        budget_ok = (
            self.budget_bits is None or projected_bits <= self.budget_bits
        )
        if not budget_ok:
            # Reject the largest up-quant moves one at a time until we
            # fit.  This is a coarse greedy; the orchestrator re-plans on
            # the next iteration so any suboptimality is bounded.
            actions.sort(key=lambda a: a.storage_delta_bits, reverse=True)
            while actions and (
                self.budget_bits is not None
                and projected_bits > self.budget_bits
            ):
                dropped = actions.pop(0)
                projected_bits -= dropped.storage_delta_bits
            actions.sort(key=lambda a: a.name)

        # Now the down-quant moves.  These are best-effort: skip if the
        # tensor is already at the bottom, skip if the cohort is empty,
        # and skip if doing so would be a no-op.
        for tensor in reversed(ranked_records):
            if tensor["tensor"] not in bottom_set:
                continue
            t = TensorState.from_df_row(tensor)
            target = metrics.step_down(str(tensor["current_qtype"]))
            if target is None:
                continue
            sens = float(sensitivity.get(t.name, 0.0))
            expected_delta = t.expected_mse_at(target, sensitivity=sens) - t.mse
            bits_after = int(
                round(metrics.BITS_PER_WEIGHT.get(target, 0.0))
                * float(tensor["n_weights"] or 0)
            )
            bits_before = t.bits()
            actions.append(
                RequantAction(
                    name=t.name,
                    from_qtype=t.current_qtype,
                    to_qtype=target,
                    expected_mse_delta=expected_delta,
                    sensitivity=sens,
                    imatrix_magnitude=(
                        float(tensor["imatrix_magnitude"])
                        if tensor.get("imatrix_magnitude") is not None
                        else None
                    ),
                    gradient_proxy=(
                        float(tensor["gradient_proxy"])
                        if tensor.get("gradient_proxy") is not None
                        else None
                    ),
                    layer_position_prior=(
                        float(tensor["layer_position_prior"])
                        if tensor.get("layer_position_prior") is not None
                        else None
                    ),
                    reason="bottom-fraction",
                    storage_delta_bits=bits_after - bits_before,
                    # Phase 16: role carried through.
                    model_role=self.model_role,
                    # Targeted re-calibration:
                    # recommended_action /
                    # backfill_count are
                    # propagated from the df
                    # (same contract as the
                    # top-fraction branch).
                    recommended_action=(
                        str(tensor["recommended_action"])
                        if tensor.get("recommended_action") is not None
                        else None
                    ),
                    backfill_count=(
                        int(tensor["backfill_count"])
                        if tensor.get("backfill_count") is not None
                        else None
                    ),
                )
            )

        actions.sort(key=lambda a: a.name)
        storage_after = storage_before + sum(a.storage_delta_bits for a in actions)
        return RequantPlan(
            iteration=iteration,
            actions=actions,
            storage_before_bits=storage_before,
            storage_after_bits=storage_after,
            storage_budget_bits=self.budget_bits,
            termination_reason=None,
            sensitivity=dict(sensitivity),
            ema_sensitivity=dict(sensitivity),
        )


# ---------------------------------------------------------------------------
# Orchestrator loop
# ---------------------------------------------------------------------------


class OrchestratorLoop:
    """Top-level driver: read L4, score, plan, apply, re-run L4, repeat.

    The loop is the bridge between the orchestrator's bookkeeping and the
    real quantizer pipeline.  It is responsible for:

    * Loading the L4 probe report.
    * Calling the :class:`SensitivityScorer` to update the EMA.
    * Calling the :class:`RequantPlanner` to produce a plan.
    * Applying the plan to the on-disk model (delegated to
      ``per_tensor_calibrate.py`` or to the synthetic driver in
      ``l5_demo.py``).
    * Writing the iteration plan, the cumulative plan, and the sidecar
      policy that the downstream quantizer consumes.
    * Deciding when to stop.

    The "apply" step is pluggable: the orchestrator expects a callable that
    takes a :class:`RequantPlan` and returns a mapping ``{tensor_name:
    new_qtype}``.  The default is :func:`apply_plan_to_policy`, which
    rewrites the calibration policy in place.
    """

    def __init__(
        self,
        *,
        scorer: SensitivityScorer,
        planner: RequantPlanner,
        apply: "ApplyFn | None" = None,
        max_iterations: int = 5,
        divergence_threshold: float | None = None,
        sidecar: Path | None = None,
        verbose: bool = False,
        auto_converge: bool = True,
        converge_tolerance_delta: float = 1e-6,
        converge_tolerance_storage: float = 0.01,
        converge_window: int = 2,
        backfill: "TargetedBackfill | None" = None,
        max_backfill_rounds: int = 2,
        backfill_sample_cap: int = 256,
    ) -> None:
        self.scorer = scorer
        self.planner = planner
        self.apply = apply
        self.max_iterations = int(max_iterations)
        self.divergence_threshold = (
            float(divergence_threshold) if divergence_threshold is not None else None
        )
        self.sidecar = Path(sidecar) if sidecar is not None else None
        self.verbose = bool(verbose)
        # Auto-converge: when enabled (the new default), the
        # orchestrator stops the iteration loop on three new
        # signals in addition to the existing planner-level
        # termination paths:
        #   * delta-converged: the largest abs(expected_mse_delta)
        #     across the most recent K plans is below
        #     converge_tolerance_delta (default 1e-6, FP16 noise floor).
        #     Catches the "hover near threshold" case the existing
        #     all-tensors-below-threshold check misses.
        #   * storage-stable: the relative change in storage bits
        #     across the most recent K plans is below
        #     converge_tolerance_storage (default 1%). Catches the
        #     "still proposing tiny bit tweaks" case.
        #   * the existing max-iterations safety cap (raised to 16
        #     from 5 when --auto-converge is on).
        # --no-auto-converge preserves byte-identical pre-task behavior:
        # max_iterations=5 default, divergence_threshold default is
        # whatever the caller passed, the new convergence checks are
        # never evaluated.
        self.auto_converge = bool(auto_converge)
        self.converge_tolerance_delta = float(converge_tolerance_delta)
        self.converge_tolerance_storage = float(converge_tolerance_storage)
        self.converge_window = max(1, int(converge_window))
        # Targeted re-calibration: the backfill hook
        # fires after the auto-converge checks (so
        # auto-converge reasons still win when they fire
        # first) and before the apply step. The hook
        # drives a focused re-capture on the
        # monitor-verdict tensors; the new activation
        # stats re-feed the next iteration's
        # ``l5_outcome`` evaluation. The hook is opt-in
        # (None = bypassed, byte-equivalent pre-task
        # behavior on iteration ordering). The
        # ``--no-targeted-recal`` CLI flag maps to
        # ``backfill=None``.
        self.backfill = backfill
        self.max_backfill_rounds = max(1, int(max_backfill_rounds))
        self.backfill_sample_cap = max(1, int(backfill_sample_cap))
        self.history: list[RequantPlan] = []
        # Phase 0.5: per-model disagreement log.
        # The default path is None (the log is not
        # written). The CLI sets the path next to
        # ``--policy`` as
        # ``<policy>.l5-disagreement.log`` when the
        # EXL2 source is wired. The
        # ``disagreement_rank_threshold`` is the
        # per-tensor rank difference above which a
        # tensor is logged as "EXL2 ranking and
        # combined ranking disagree on this
        # verdict"; default 5 positions (the
        # orchestrator's verbose mode tightens
        # this to 1 for diagnostic runs).
        self._disagreement_log_path: Path | None = None
        self.disagreement_rank_threshold: int = 5

    # -- public API --------------------------------------------------------

    def run(
        self,
        l4_report: Mapping[str, object],
        imatrix: Mapping[str, float] | None = None,
        exl2_per_layer_errors: Mapping[int, float] | None = None,
    ) -> list[RequantPlan]:
        """Execute the orchestrator loop until termination.

        Returns the list of :class:`RequantPlan`s, one per iteration.  The
        final plan carries the ``termination_reason`` explaining why the
        loop stopped.

        The per-iteration state is a polars DataFrame built from the
        L4 report by :meth:`_load_dataframe`. The scorer and planner
        both consume the DataFrame; the in-loop updates (applied
        qtypes, re-evaluated MSE) are reflected as polars
        ``with_columns`` calls so the next iteration sees the
        cumulative state. The TensorState dataclass is preserved as
        a per-row projection for the bits() / expected_mse_at() math
        inside the planner.
        """
        df = self._load_dataframe(l4_report)
        if df.is_empty():
            raise ValueError("L4 report has no tensors; nothing to do")

        # Targeted re-calibration: the backfill hook
        # needs the runtime context (``_db`` /
        # ``_db_path`` / ``_model_hash`` / ``_components``
        # / ``_corpus_root`` / ``_backfill_timeout_sec``)
        # to dispatch the per-tensor subprocesses. The
        # context is set by ``enable_backfill``; when
        # the orchestrator's CLI constructs the loop
        # with ``backfill=TargetedBackfill(...)`` it
        # also calls ``enable_backfill`` immediately
        # after. The defaults below are the
        # pre-backfill behavior (``_db = None`` ->
        # the hook is bypassed).
        if not hasattr(self, "_db"):
            self._db = None
            self._db_path = None
            self._model_hash = ""
            self._components = {}
            self._corpus_root = None
            self._backfill_timeout_sec = 600

        # Track the per-tensor qtype we have settled on.  We start from
        # the L4 report's current_qtype; the loop updates this map as
        # actions fire.  The sidecar policy is built from this final map
        # so it reflects the cumulative state, not just the last
        # iteration's actions.
        final_qtype: dict[str, str] = dict(zip(
            df["tensor"].to_list(),
            df["current_qtype"].to_list(),
        ))

        for iteration in range(1, self.max_iterations + 1):
            # Phase 0.5: pass the EXL2 per-layer
            # errors to the scorer. The default
            # ``None`` is the opt-in path (the
            # 4th component is empty, the fold
            # contributes zero). When the orchestrator
            # has a DB reference and ``w_exl2 > 0``,
            # the caller passes the per-layer map
            # here; the next commit in this series
            # wires the lookup.
            df = self.scorer.score(
                df, imatrix,
                exl2_per_layer_errors=exl2_per_layer_errors,
            )
            plan = self.planner.plan(iteration, df)
            self.history.append(plan)

            # Phase 0.5: per-iteration disagreement
            # log. The Spearman rank correlation
            # between the EXL2 per-layer error and
            # the orchestrator's combined
            # sensitivity_score is recorded in the
            # per-model log; the per-tensor rank
            # disagreements (where the EXL2 ranking
            # and the combined ranking disagree by
            # more than the threshold) are logged
            # one line per verdict. The log is the
            # research-credibility audit trail the
            # design doc ratifies: when the
            # agreement is high (Spearman > 0.6),
            # the design is shaped by SOTA; when
            # they disagree on specific tensors,
            # the disagreement is a research
            # finding (would be a paper, not a
            # bug). The disagreement log is the
            # consumer of the
            # ``exl2_per_layer_errors`` map the
            # caller passed in.
            self._log_disagreement(
                iteration, df, exl2_per_layer_errors,
            )

            self._log(
                f"iter {iteration}: {len(plan.actions)} actions, "
                f"storage {plan.storage_before_bits} -> {plan.storage_after_bits} bits"
            )

            if plan.termination_reason is not None:
                self._log(f"  converged: {plan.termination_reason}")
                break

            if not plan.actions:
                plan.termination_reason = "no-actions-possible"
                self._log("  no actions possible; stopping")
                break

            # Auto-converge: when enabled, evaluate the sliding-window
            # delta and storage-stability checks before running another
            # iteration. These catch the "near-threshold hover" and
            # "tiny bit tweaks" cases the existing planner-level
            # termination does not. The K most recent plans (including
            # the current one) must all satisfy the predicate.
            if self.auto_converge:
                window = self.converge_window
                # The window covers the most recent K plans
                # INCLUDING the current one. With K=2 the check
                # looks at the current plan + 1 prior. We need at
                # least K-1 prior plans in history before the
                # check evaluates; on the first iteration the
                # window is just the current plan, so the check
                # is permissive (one plan can never satisfy a
                # multi-plan convergence predicate).
                prior = self.history[-(window - 1):] if window > 1 else []
                recent = prior + [plan]
                if len(recent) >= window:
                    max_delta = max(
                        (abs(a.expected_mse_delta) for p in recent for a in p.actions),
                        default=0.0,
                    )
                    if max_delta < self.converge_tolerance_delta:
                        plan.termination_reason = "delta-converged"
                        self._log(
                            f"  converged: delta-converged "
                            f"(window={window}, max_delta={max_delta:.3e} < "
                            f"{self.converge_tolerance_delta:.3e})"
                        )
                        break
                    storage_rel = [
                        abs(p.storage_after_bits - p.storage_before_bits)
                        / max(p.storage_before_bits, 1)
                        for p in recent
                    ]
                    max_storage_rel = max(storage_rel) if storage_rel else 0.0
                    if max_storage_rel < self.converge_tolerance_storage:
                        plan.termination_reason = "storage-stable"
                        self._log(
                            f"  converged: storage-stable "
                            f"(window={window}, max_rel_change={max_storage_rel:.3e} < "
                            f"{self.converge_tolerance_storage:.3e})"
                        )
                        break

            # Targeted re-calibration: when the backfill
            # hook is enabled (i.e. ``self.backfill`` is
            # not None), drive a focused re-capture on
            # the monitor-verdict tensors. The hook is
            # positioned AFTER the auto-converge checks
            # so the auto-converge reasons still win
            # when they fire first (the
            # conflict-resolution the prior worker
            # documented). The hook is async
            # (ThreadPoolExecutor with max_workers=2);
            # the orchestrator waits on the future at
            # the next "apply" step so the next
            # iteration's plan reads the re-captured
            # stats. The new ``backfill-no-progress``
            # termination_reason fires when every
            # monitor tensor has hit the rounds cap
            # (no further backfill is productive).
            if self.backfill is not None and getattr(
                self, "_db", None
            ) is not None:
                monitor_actions = [
                    a for a in plan.actions
                    if a.recommended_action == "monitor"
                ]
                if monitor_actions:
                    monitor_entries = [
                        {
                            "name": a.name,
                            "model_role": a.model_role,
                            "family": _tensor_family(a.name),
                            "layer_depth": int(
                                self._layer_for(a.name)
                            ),
                        }
                        for a in monitor_actions
                    ]
                    # The backfill engine filters by
                    # ``backfill_count < max_rounds``;
                    # passing all monitor tensors is fine
                    # (the engine handles the cap).
                    future = self.backfill.run_backfill_async(
                        db_path=self._db_path,  # type: ignore[arg-type]
                        model_hash=str(
                            getattr(self, "_model_hash", "")
                        ),
                        components=self._components,
                        corpus_root=self._corpus_root,
                        monitor_tensors=monitor_entries,
                    )
                    try:
                        result = future.result(
                            timeout=self._backfill_timeout_sec,
                        )
                    except Exception as e:
                        self._log(
                            f"  backfill error: "
                            f"{e.__class__.__name__}: {str(e)[:200]}"
                        )
                        result = None
                    if result is not None:
                        no_progress = (
                            result.tensors_processed == 0
                            and result.error_count == 0
                        )
                        if no_progress and len(monitor_entries) > 0:
                            # Every monitor tensor hit
                            # the backfill rounds cap
                            # (the engine filtered them
                            # all out). Set the
                            # backfill-no-progress
                            # termination_reason and
                            # break the loop. We do
                            # NOT touch the current
                            # plan's other
                            # termination_reason;
                            # the spec is explicit
                            # that
                            # backfill-no-progress
                            # is a new value, not a
                            # replacement.
                            plan.termination_reason = (
                                "backfill-no-progress"
                            )
                            self._log(
                                f"  converged: "
                                f"backfill-no-progress "
                                f"(monitor_tensors="
                                f"{len(monitor_entries)}, "
                                f"tensors_processed="
                                f"{result.tensors_processed})"
                            )
                            break
                        self._log(
                            f"  backfill: "
                            f"tensors_processed="
                            f"{result.tensors_processed}, "
                            f"samples_consumed="
                            f"{result.samples_consumed}, "
                            f"wall_time="
                            f"{result.wall_time_sec:.2f}s"
                        )
                # When monitor_entries is empty,
                # fall through to the apply step
                # (no backfill to run; the next
                # iteration's plan will re-evaluate).

            # Apply the plan.  The applier is responsible for writing the
            # updated policy (or calling per_tensor_calibrate.py).
            if self.apply is not None:
                new_qtypes = self.apply(plan)
            else:
                new_qtypes = {a.name: a.to_qtype for a in plan.actions}

            # Reflect the applied qtypes in the per-iteration state
            # via a polars ``with_columns`` join. ``target_qtype`` is
            # set so the sidecar writer can audit the cumulative
            # resolution; ``current_qtype`` advances so the next
            # iteration's bits() and expected_mse_at() math sees the
            # new qtype.
            if new_qtypes:
                new_qtypes_df = pl.DataFrame({
                    "tensor":        list(new_qtypes.keys()),
                    "new_qtype":     list(new_qtypes.values()),
                })
                df = (
                    df
                    .join(new_qtypes_df, on="tensor", how="left")
                    .with_columns(
                        pl.col("new_qtype").alias("target_qtype"),
                        pl.coalesce([pl.col("new_qtype"),
                                     pl.col("current_qtype")]).alias("current_qtype"),
                    )
                    .drop("new_qtype")
                )
                for n, q in new_qtypes.items():
                    final_qtype[n] = q

            # Synthetic L4 re-evaluation: in the demo we just read the
            # new expected MSE off the plan actions. In production
            # this would be replaced by re-running the L4 probe. The
            # update is a single polars expression: for every tensor
            # named in an action, set mse = max(0, mse + delta).
            if plan.actions:
                delta_df = pl.DataFrame({
                    "tensor":            [a.name for a in plan.actions],
                    "expected_mse_delta": [float(a.expected_mse_delta) for a in plan.actions],
                })
                df = (
                    df
                    .join(delta_df, on="tensor", how="left")
                    .with_columns(
                        pl.max_horizontal(
                            pl.lit(0.0),
                            (pl.col("mse") + pl.col("expected_mse_delta").fill_null(0.0)),
                        ).alias("mse_new")
                    )
                    .with_columns(pl.col("mse_new").alias("mse"))
                    .drop("mse_new", "expected_mse_delta")
                )
        else:
            # We exhausted the iteration budget.
            self._log(
                f"  reached max_iterations={self.max_iterations}; stopping"
            )
            if self.history and self.history[-1].termination_reason is None:
                self.history[-1].termination_reason = "max-iterations"

        # Stash the final qtype for the sidecar writer.
        self._final_qtype = final_qtype

        if self.sidecar is not None:
            self.write_sidecar(self.sidecar)

        return self.history

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _load_dataframe(l4_report: Mapping[str, object]) -> pl.DataFrame:
        """Build the per-iteration polars DataFrame from an L4 report.

        One row per tensor, with columns ``tensor, layer, current_qtype,
        target_qtype, n_weights, mse, mse_minus_one, perplexity,
        top1_mismatch``. The four sensitivity columns
        (``imatrix_magnitude``, ``gradient_proxy``,
        ``layer_position_prior``, ``sensitivity_score``,
        ``sensitivity_ema``) are added by
        :meth:`SensitivityScorer.score`. The schema's
        ``target_qtype`` is null on the first iteration and filled in
        by the loop's apply step.
        """
        tensors_payload = l4_report.get("tensors", {})
        if not isinstance(tensors_payload, Mapping):
            raise ValueError("L4 report: 'tensors' must be a mapping")
        rows: list[dict] = []
        import re as _re
        _BLK_RE = _re.compile(r"^(blk\.\d+)\.")
        for name, payload in tensors_payload.items():
            if not isinstance(payload, Mapping):
                raise ValueError(
                    f"L4 report: tensor {name!r} payload is not a mapping")
            ts = TensorState.from_l4(str(name), dict(payload))
            row = ts.to_row()
            # Layer extraction: same convention as the rest of the
            # analytical surface so calibration_rollup can join on
            # layer without re-normalization.
            base = str(name)
            for suf in (".weight", ".bias"):
                if base.endswith(suf):
                    base = base[: -len(suf)]
                    break
            m = _BLK_RE.match(base + ".")
            row["layer"] = m.group(1) if m is not None else base
            rows.append(row)
        # Stable order so the sidecar writer and the DataFrame-based
        # test fixtures see the same row order across runs.
        rows.sort(key=lambda r: r["tensor"])
        if not rows:
            return pl.DataFrame()
        df = pl.DataFrame(rows, infer_schema_length=max(len(rows), 1))
        # Cast ``n_weights`` to Int64 to keep the polars schema clean.
        return df.with_columns(pl.col("n_weights").cast(pl.Int64, strict=False))

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[l5] {message}", file=sys.stderr)

    # -- Phase 0.5: disagreement log (EXL2 vs. combined score) ---------

    def set_disagreement_log_path(self, path: Path | None) -> None:
        """Set the per-model disagreement-log path.

        The log is written one line per (iteration,
        tensor) verdict where the EXL2 per-layer error
        ranking and the orchestrator's combined
        sensitivity_score ranking disagree by more
        than ``self.disagreement_rank_threshold``
        positions. A header line per iteration
        records the per-iteration Spearman correlation
        (computed in pure NumPy; the orchestrator
        does not depend on scipy).

        The default path is ``None`` (the log is not
        written). The CLI sets the path next to
        ``--policy`` as ``<policy>.l5-disagreement.log``
        when the EXL2 source is wired.
        """
        self._disagreement_log_path = (
            Path(path) if path is not None else None
        )

    def _log_disagreement(
        self,
        iteration: int,
        df: pl.DataFrame,
        exl2_per_layer_errors: Mapping[int, float] | None,
    ) -> None:
        """Log the per-iteration Spearman disagreement.

        Computes the Spearman rank correlation between
        the EXL2 per-layer error (broadcast to every
        tensor in the layer) and the orchestrator's
        combined ``sensitivity_score`` (per tensor).
        Spearman is computed in pure NumPy (rank +
        Pearson); the orchestrator does not depend on
        scipy. The log path is set by
        :meth:`set_disagreement_log_path`.

        One header line per iteration with the
        Spearman value; one row per tensor where the
        per-tensor rank difference exceeds
        ``self.disagreement_rank_threshold`` (default 5
        positions). The log is appended (not
        overwritten) so the file accumulates the
        per-iteration disagreement history.
        """
        if self._disagreement_log_path is None:
            return
        if not exl2_per_layer_errors:
            return
        if "sensitivity_score" not in df.columns:
            return
        # Build the per-tensor EXL2 component: peak-1
        # normalized per-layer error, mapped to every
        # tensor in the layer.
        names = df["tensor"].to_list()
        scores = df["sensitivity_score"].to_list()
        exl2_per_tensor: list[float] = []
        for n in names:
            layer_idx = self._layer_for(n)
            err = float(exl2_per_layer_errors.get(layer_idx, 0.0))
            exl2_per_tensor.append(err)
        # Per-iteration Spearman correlation (pure
        # NumPy: rank + Pearson). When one side has
        # zero variance, Spearman is undefined; we
        # log the degenerate value (NaN) so the
        # consumer can detect it.
        rho, p_value = _spearmanr(exl2_per_tensor, scores)
        # Per-tensor rank disagreement.
        exl2_ranks = _ranks(exl2_per_tensor)
        score_ranks = _ranks(scores)
        threshold = int(getattr(
            self, "disagreement_rank_threshold", 5
        ))
        lines: list[str] = []
        lines.append(
            f"# iter {iteration}: Spearman rho={rho:.6e} "
            f"p={p_value:.6e} n={len(names)} "
            f"threshold={threshold}"
        )
        n_logged = 0
        for i, n in enumerate(names):
            rank_diff = abs(int(exl2_ranks[i]) - int(score_ranks[i]))
            if rank_diff >= threshold:
                lines.append(
                    f"{iteration},{n},{int(exl2_ranks[i])},"
                    f"{int(score_ranks[i])},{rank_diff},"
                    f"{float(exl2_per_tensor[i]):.6e},"
                    f"{float(scores[i]):.6e}"
                )
                n_logged += 1
        # Append (not overwrite); the log accumulates
        # the per-iteration disagreement history.
        path = self._disagreement_log_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
        if self.verbose:
            self._log(
                f"  disagreement: rho={rho:.4e} "
                f"n_disagreements={n_logged}/{len(names)} "
                f"-> {path}"
            )

    def _layer_for(self, tensor_name: str) -> int:
        """Extract the block index from ``tensor_name``
        (same convention as the rest of the orchestrator:
        ``blk.<i>.`` -> i; else 0). The backfill hook uses
        this to populate the per-tensor ``layer_depth``
        field the focused re-capture records.
        """
        import re as _re
        m = _re.match(r"^blk\.(\d+)\.", str(tensor_name))
        if m is not None:
            return int(m.group(1))
        return 0

    # -- targeted re-calibration -----------------------------------------

    def enable_backfill(
        self,
        *,
        db: "TesseraDB | None" = None,
        db_path: Path | None = None,
        model_hash: str = "",
        components: "Mapping[str, Path | None] | None" = None,
        corpus_root: Path | None = None,
        timeout_sec: int = 600,
    ) -> None:
        """Wire the backfill engine's runtime context
        onto the orchestrator.

        The orchestrator's iteration loop reads
        ``self._db`` / ``self._db_path`` /
        ``self._model_hash`` / ``self._components`` /
        ``self._corpus_root`` /
        ``self._backfill_timeout_sec`` on every iteration.
        Setting them on the instance via this method
        keeps the backfill hook decoupled from the
        constructor (the backfill engine's lifetime is
        a subset of the orchestrator's; the engine is
        constructed at startup, the runtime context is
        set when the orchestrator's CLI is built).

        ``db`` is the ``TesseraDB`` instance the
        orchestrator has open (read-only is fine -- the
        backfill engine reads ``backfill_count`` from
        it; the per-tensor subprocess writes through
        ``--backfill-db``). ``db_path`` is the path
        argument the subprocess uses; when ``db`` is
        None, ``db_path`` is the only seam. ``None``
        on both -> the backfill hook is bypassed (the
        pre-task byte-equivalent behavior).
        """
        self._db = db
        self._db_path = (
            Path(db_path) if db_path is not None else None
        )
        self._model_hash = str(model_hash)
        self._components = dict(components or {})
        self._corpus_root = (
            Path(corpus_root) if corpus_root is not None else None
        )
        self._backfill_timeout_sec = max(1, int(timeout_sec))

    # -- output ------------------------------------------------------------

    def write_sidecar(self, path: Path) -> None:
        """Write the cumulative sidecar policy and the plan history.

        The sidecar combines all per-iteration plans into a single
        ``llama.speculative.calibration-policy.v1`` document that
        ``tile640_quantize_v3.py --calibration-policy`` already consumes.
        The action history is preserved under ``l5_orchestrator.history``
        for tooling that wants to audit the loop.

        The ``tensor_families`` map reflects the **cumulative** final
        qtype of every tensor the loop touched, not just the last
        iteration's actions.  This lets ``tile640_quantize_v3.py``
        re-quantise a tensor at the resolved qtype even if the loop took
        several steps to get there.
        """
        if not self.history:
            return
        latest = self.history[-1]
        final_qtype = getattr(self, "_final_qtype", {}) or {}
        tensor_families: dict[str, dict] = {}
        # Walk the history to collect every (tensor, target_qtype) transition.
        last_action_by_tensor: dict[str, RequantAction] = {}
        for plan in self.history:
            for action in plan.actions:
                last_action_by_tensor[action.name] = action
        for name, qtype in final_qtype.items():
            entry = {"match": [name], "exact": True, "l5_qtype": qtype}
            action = last_action_by_tensor.get(name)
            if action is not None:
                entry["l5_from_qtype"] = action.from_qtype
                entry["l5_expected_mse_delta"] = action.expected_mse_delta
                entry["l5_sensitivity"] = action.sensitivity
                entry["l5_reason"] = action.reason
            tensor_families[name] = entry
        policy = {
            "schema": POLICY_SCHEMA,
            "l5_orchestrator": {
                "schema": SCHEMA,
                "iterations": len(self.history),
                "termination_reason": latest.termination_reason,
                "history": [plan.to_dict() for plan in self.history],
                "weights": list(self.scorer.weights),
                "ema_decay": self.scorer.decay,
                "top_fraction": self.planner.top_fraction,
                "bottom_fraction": self.planner.bottom_fraction,
                "divergence_threshold": self.divergence_threshold,
                "storage_budget_bits": self.planner.budget_bits,
                "final_qtype": final_qtype,
            },
            "tensor_families": tensor_families,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def write_history(cls, plans: Sequence[RequantPlan], path: Path) -> None:
        """Write the iteration history as NDJSON, one record per
        (plan, action) tuple, conformant to
        ``common/schemas/l5_plan.schema.json``.

        Used by the demo and by tests. The legacy JSON wrapper
        (one document with a ``plans`` array of iteration-level
        metadata) is gone: the per-tensor schema is the source of
        truth, and the iteration-level metadata (storage_before_bits,
        storage_after_bits, storage_budget_bits, termination_reason)
        is derivable from the per-tensor ``delta_bits`` column plus
        the orchestrator's input budget, which the consumer can
        compute on demand. ``calibration_rollup.py`` will do this
        rollup when needed.

        Provenance (kernel_version, created_at, tessera_main_tip)
        is stamped into every record; values are populated by
        shelling out to ``git`` (the C++ provenance helper is not
        yet wired into a CLI binary).
        """
        import polars as pl
        from _analytical_io import polars_schema as _schema_polars_types
        # Layer-name extraction: same convention as
        # per_layer_error_table.py and l3_outlier_report.py so
        # a cross-pipeline rollup on `layer` works without
        # normalization.
        import re as _re
        _BLK_RE = _re.compile(r"^(blk\.\d+)\.")
        def _layer(tensor_name: str) -> str:
            base = tensor_name
            for suf in (".weight", ".bias"):
                if base.endswith(suf):
                    base = base[: -len(suf)]
                    break
            m = _BLK_RE.match(base + ".")
            if m is not None:
                return m.group(1)
            return base
        def _bits_for(qtype: str) -> float:
            return float(metrics.BITS_PER_WEIGHT.get(qtype, 0.0))
        def _prov() -> tuple[str, str, str]:
            kv, mt = "unknown", "unknown"
            try:
                r = subprocess.run(
                    ["git", "describe", "--all", "--always"],
                    capture_output=True, text=True, check=False,
                    cwd=str(Path(__file__).resolve().parent.parent.parent))
                if r.returncode == 0 and r.stdout.strip():
                    kv = r.stdout.strip()
                r = subprocess.run(
                    ["git", "rev-parse", "--short", "main"],
                    capture_output=True, text=True, check=False,
                    cwd=str(Path(__file__).resolve().parent.parent.parent))
                if r.returncode == 0 and r.stdout.strip():
                    mt = r.stdout.strip()
            except FileNotFoundError:
                pass
            from datetime import datetime, timezone
            created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            return kv, created, mt
        kernel_version, created_at, main_tip = _prov()
        rows: list[dict] = []
        for plan in plans:
            for action in plan.actions:
                rows.append({
                    "tensor":          action.name,
                    "layer":           _layer(action.name),
                    "current_qtype":   action.from_qtype,
                    "new_qtype":       action.to_qtype,
                    "bits":            _bits_for(action.to_qtype),
                    "delta_bits":      int(action.storage_delta_bits),
                    "sensitivity_score": float(action.sensitivity),
                    "delta_quality":   float(action.expected_mse_delta),
                    # Phase 15: populate the per-tensor sensitivity
                    # components from the RequantAction (captured at
                    # plan-emit time by the planner). ``None`` values
                    # (e.g. the imatrix was missing and the scorer
                    # collapsed one of the components) are written as
                    # ``null`` (the polars-canonical null marker) so
                    # the retune's 3-coefficient OLS can detect the
                    # missing-component case and fall back to the
                    # 2-coefficient OLS on the combined
                    # sensitivity_score for that row.
                    "imatrix_magnitude":  action.imatrix_magnitude,
                    "gradient_proxy":     action.gradient_proxy,
                    "layer_position_prior": action.layer_position_prior,
                    "plan_id":         "",
                    "iteration":       int(plan.iteration),
                    "kernel_version":  kernel_version,
                    "created_at":      created_at,
                    "tessera_main_tip": main_tip,
                })
        schema_types = _schema_polars_types("l5_plan")
        if rows:
            df = pl.DataFrame(rows, infer_schema_length=max(len(rows), 1))
        else:
            df = pl.DataFrame(
                {col: pl.Series(name=col, values=[], dtype=dtype)
                 for col, dtype in schema_types.items()},
                schema=schema_types,
            )
        for col, dtype in schema_types.items():
            if col in df.columns and df.schema[col] != dtype:
                df = df.with_columns(pl.col(col).cast(dtype, strict=False))
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_ndjson(path)


# ---------------------------------------------------------------------------
# Apply helpers
# ---------------------------------------------------------------------------


ApplyFn = "callable[[RequantPlan], dict[str, str]]"


def apply_plan_to_policy(plan: RequantPlan, policy_path: Path) -> dict[str, str]:
    """Apply a plan to a calibration policy on disk and return the new qtypes.

    This is the default applier used when the orchestrator is run against a
    real model.  It mutates the JSON ``tensor_families`` entry of the
    policy so that downstream ``tile640_quantize_v3.py`` will quantise the
    affected tensors at the new qtype.  Tensors not present in the
    ``tensor_families`` map are left untouched - the original family rules
    still apply.
    """
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    families = policy.setdefault("tensor_families", {})
    new_qtypes: dict[str, str] = {}
    for action in plan.actions:
        entry = families.get(action.name, {"match": [action.name], "exact": True})
        entry["l5_qtype"] = action.to_qtype
        entry["l5_from_qtype"] = action.from_qtype
        entry["l5_expected_mse_delta"] = action.expected_mse_delta
        families[action.name] = entry
        new_qtypes[action.name] = action.to_qtype
    policy_path.write_text(
        json.dumps(policy, indent=2) + "\n", encoding="utf-8"
    )
    return new_qtypes


def apply_plan_via_per_tensor_calibrate(
    plan: RequantPlan,
    layers_dir: Path,
    output_policy: Path,
) -> dict[str, str]:
    """Delegate requantization to ``per_tensor_calibrate.py``.

    The orchestrator does not own the actual requantization; it just emits
    the plan and lets the existing tool do the work.  This applier shells
    out to ``per_tensor_calibrate.py`` for the LRQ-mode requantization and
    then merges the plan into the resulting policy.

    The helper is best-effort: if ``per_tensor_calibrate.py`` is missing
    or fails, the function returns the plan-derived qtypes unchanged so
    the orchestrator can record what it would have done.
    """
    tool = Path(__file__).resolve().parent / "per_tensor_calibrate.py"
    if not tool.is_file():
        return {a.name: a.to_qtype for a in plan.actions}

    cmd = [
        sys.executable,
        str(tool),
        "--fitness", "lrq",
        "--layers", str(layers_dir),
        "--output", str(output_policy),
    ]
    if not output_policy.exists():
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:  # pragma: no cover
            print(
                f"WARN: per_tensor_calibrate.py failed: {exc.stderr.strip()}",
                file=sys.stderr,
            )
            return {a.name: a.to_qtype for a in plan.actions}

    # Merge the plan's qtype changes into the policy in place.
    return apply_plan_to_policy(plan, output_policy)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _read_l4_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_imatrix(path: Path | None) -> dict[str, float] | None:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, Mapping):
        return {str(k): float(v) for k, v in data.items()}
    return None


def _read_existing_policy(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "L5 orchestrator: IterQuant-style adaptive requantization loop. "
            "Reads a L4 E2E probe report and an optional imatrix, scores "
            "each tensor's sensitivity, and emits a sidecar policy plus a "
            "plan history."
        )
    )
    parser.add_argument(
        "--l4-report",
        required=True,
        type=Path,
        help="Path to the L4 E2E probe report (JSON with per-tensor metrics)",
    )
    parser.add_argument(
        "--imatrix",
        default=None,
        type=Path,
        help="Optional imatrix summary JSON (mapping tensor name -> magnitude)",
    )
    parser.add_argument(
        "--policy",
        default=None,
        type=Path,
        help="Output sidecar policy (consumable by tile640_quantize_v3.py)",
    )
    parser.add_argument(
        "--history",
        default=None,
        type=Path,
        help="Output iteration history JSON (defaults next to --policy)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help=(
            "Maximum number of requantization passes. Default 16 when "
            "--auto-converge is on (the recommended mode); default 5 "
            "when --no-auto-converge is set (legacy mode)."
        ),
    )
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.10,
        help="Fraction of tensors to requantize up per pass (default 0.10)",
    )
    parser.add_argument(
        "--bottom-fraction",
        type=float,
        default=0.05,
        help="Fraction of tensors to requantize down per pass (default 0.05)",
    )
    parser.add_argument(
        "--divergence-threshold",
        type=float,
        default=None,
        help=(
            "MSE threshold below which a tensor is considered converged. "
            "Default 1e-4 when --auto-converge is on (the recommended "
            "mode); default None when --no-auto-converge is set "
            "(legacy mode = no per-tensor MSE gate)."
        ),
    )
    parser.add_argument(
        "--auto-converge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Drive the iteration loop until the per-tensor "
            "expected_mse deltas and storage bits stabilize, with a "
            "hard max-iterations cap. ON by default. The loop stops on "
            "the first of: planner-level termination (existing), "
            "delta-converged (max abs(expected_mse_delta) over the "
            "last K iterations < --converge-tolerance), "
            "storage-stable (max relative storage change over the last "
            "K iterations < --converge-storage-tolerance), or "
            "max-iterations reached. --no-auto-converge preserves the "
            "legacy pre-task behavior (max-iterations=5, no new "
            "convergence checks)."
        ),
    )
    parser.add_argument(
        "--converge-tolerance",
        type=float,
        default=1e-6,
        help=(
            "Max abs(expected_mse_delta) per tensor across the "
            "convergence window below which the loop terminates with "
            "delta-converged. Default 1e-6 (FP16 noise floor). "
            "Only consulted when --auto-converge is on."
        ),
    )
    parser.add_argument(
        "--converge-storage-tolerance",
        type=float,
        default=0.01,
        help=(
            "Max relative change in storage bits across the "
            "convergence window below which the loop terminates with "
            "storage-stable. Default 0.01 (one percent). Only "
            "consulted when --auto-converge is on."
        ),
    )
    parser.add_argument(
        "--converge-window",
        type=int,
        default=2,
        help=(
            "Number of consecutive iterations the delta and storage "
            "checks must all satisfy before the loop terminates. "
            "Default 2. Only consulted when --auto-converge is on."
        ),
    )
    parser.add_argument(
        "--budget-bits",
        type=int,
        default=None,
        help="Storage budget in bits; requants that exceed it are dropped",
    )
    parser.add_argument(
        "--total-layers",
        type=int,
        default=0,
        help="Total number of transformer blocks (drives the layer-position prior)",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.9,
        help="EMA decay for sensitivity tracking (default 0.9)",
    )
    parser.add_argument(
        "--w-imatrix",
        type=float,
        default=metrics.DEFAULT_WEIGHTS[0],
        help="Weight of the imatrix component (default 0.5)",
    )
    parser.add_argument(
        "--w-gradient",
        type=float,
        default=metrics.DEFAULT_WEIGHTS[1],
        help="Weight of the gradient component (default 0.3)",
    )
    parser.add_argument(
        "--w-layer",
        type=float,
        default=metrics.DEFAULT_WEIGHTS[2],
        help="Weight of the layer-position component (default 0.2)",
    )
    # Phase 0.5: the EXL2 per-layer error is the 4th
    # evidence signal. The default ``w_exl2 = 0.0``
    # keeps the path opt-in until the first EXL2 run
    # lands (the retune's l5_weights row is also a
    # 3-tuple; the EXL2 weight is not retuned
    # automatically; the operator sets it explicitly
    # via this flag when the cross-check is on).
    parser.add_argument(
        "--w-exl2",
        type=float,
        default=metrics.DEFAULT_EXL2_WEIGHT,
        help=(
            "Weight of the EXL2 per-layer error component "
            "(default 0.0, opt-in). When w_exl2 > 0, the "
            "orchestrator reads the exl2_layer_stats DuckDB "
            "table and folds the per-layer error into the "
            "per-tensor sensitivity score. Set this flag to "
            "e.g. 0.2 after the first EXL2 run lands to "
            "weight the EXL2 signal alongside the "
            "imatrix / gradient / layer components."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the plan to the policy file (default: dry-run; emit plan only)",
    )
    parser.add_argument(
        "--existing-policy",
        default=None,
        type=Path,
        help="Existing calibration policy to mutate when --apply is set",
    )
    parser.add_argument(
        "--retune-from-db",
        default=None,
        type=Path,
        help=(
            "Path to the unified tessera.duckdb file. When given, the "
            "orchestrator reads the per-(model, family) l5_weights "
            "rows written by tools/tessera/l5_retune.py and uses "
            "the n_samples-weighted average as the starting "
            "(w_imatrix, w_gradient, w_layer) tuple. This is the "
            "consumer half of the feedback loop: the previous "
            "generation's residual determines this generation's "
            "weights. Requires --model-hash (the l5_weights rows "
            "are keyed by model_hash)."
        ),
    )
    parser.add_argument(
        "--model-hash",
        default=None,
        help=(
            "model_hash for the --retune-from-db lookup. Default "
            "None = the weights are not loaded from the DB even "
            "if --retune-from-db is set (the per-model keying is "
            "required for a meaningful retune)."
        ),
    )
    parser.add_argument(
        "--model-role",
        default=None,
        choices=[
            "trunk", "dflash", "dspark", "mtp_nextn", "shared_embd",
        ],
        help=(
            "model_role for the --retune-from-db lookup (Phase 16). "
            "Default None = the role dimension is ignored; the "
            "lookup uses only (model_hash, family). When set, the "
            "lookup is (model_hash, model_role, family); the same "
            "family in different architectural roles (e.g. the "
            "trunk's attn_q vs the dflash encoder's attn_q) gets "
            "independent (w_imatrix, w_gradient, w_layer) tuples. "
            "Requires --model-hash; a bare model_role filter "
            "would silently mix roles across models."
        ),
    )
    parser.add_argument(
        "--retune-cross-model-fallback",
        action="store_true",
        help=(
            "When --retune-from-db is set, fall back to the "
            "cross-model l5_weights row (model_hash='*') for any "
            "family the per-model lookup missed. The cross-model "
            "row is the n_samples-weighted mean across all models "
            "(written by l5_retune.py --retune-cross-model); the "
            "fallback lets new models warm-start from the "
            "cross-model mean. Off by default; the consumer-side "
            "default is to leave families without per-model rows "
            "at the --w-* flag values. Phase 16: the cross-model "
            "fallback is also role-aware: the (model_hash='*', "
            "model_role, family) row fills in for the missing "
            "per-model row, not the role-agnostic cross-model row."
        ),
    )
    parser.add_argument(
        "--per-family-top-fraction",
        default=None,
        type=Path,
        help=(
            "Path to the unified tessera.duckdb file. When set, "
            "the orchestrator reads the per-family top_fraction "
            "recommendation from l5_weights (the column the retune "
            "writes) and uses it instead of the uniform "
            "--top-fraction for the families the retune flagged. "
            "Families without a per-family row fall back to the "
            "--top-fraction value. Defaults to the --retune-from-db "
            "path when --retune-from-db is set; pass this flag "
            "explicitly with a different DB to override."
        ),
    )
    parser.add_argument(
        "--cross-model-dedup",
        action="store_true",
        help=(
            "When --retune-from-db is set and the requested "
            "model_hash has no l5_weights row, look for a "
            "different model with a matching tensor_stats "
            "fingerprint (a 5-moment hash of the per-tensor "
            "(kurtosis, eff_rank, rms, mean_abs, tail_ratio) "
            "distributions). If a match is found, log a warning "
            "and reuse the matched model's l5_weights as the "
            "warm-start. Off by default; the default is to fall "
            "back to the --w-* flag values when the model is "
            "missing. The dedup is rare in practice (different "
            "models usually have different fingerprints) but "
            "useful for fine-tunes of the same base that re-use "
            "the parent's retune rows."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    # Targeted re-calibration (L5 monitor-verdict hook).
    # The ``--targeted-recal`` / ``--no-targeted-recal``
    # flag toggles the backfill engine. The default is
    # ON when ``--retune-from-db`` is set (the
    # orchestrator's normal path -- the retune lookup
    # already queries the DB; the backfill hook reuses
    # the same DB); OFF when ``--retune-from-db`` is
    # not set (the backfill needs a DB reference;
    # without one, the hook is bypassed). The
    # ``--max-backfill-rounds`` and
    # ``--backfill-sample-cap`` flags control the
    # backfill engine's per-tensor budget. The
    # ``--backfill-timeout-sec`` flag is the per-iter
    # wait on the async future (default 600s).
    parser.add_argument(
        "--targeted-recal",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable the L5 monitor-verdict backfill hook. "
            "When ON (the recommended default), the "
            "orchestrator drives a focused re-capture on "
            "the monitor-verdict tensors each iteration; "
            "the new activation stats re-feed the next "
            "iteration's l5_outcome evaluation. When OFF "
            "(--no-targeted-recal), the hook is bypassed "
            "and the iteration ordering is "
            "byte-equivalent to the pre-backfill behavior. "
            "The default is ON when --retune-from-db is "
            "set (the orchestrator has a DB reference); "
            "OFF otherwise."
        ),
    )
    parser.add_argument(
        "--max-backfill-rounds",
        type=int,
        default=2,
        help=(
            "Maximum number of backfill rounds per "
            "monitor-verdict tensor (default 2). The "
            "orchestrator's iteration loop re-triggers "
            "the backfill while backfill_count < this "
            "value; the loop terminates with "
            "backfill-no-progress when every monitor "
            "tensor has hit the cap."
        ),
    )
    parser.add_argument(
        "--backfill-sample-cap",
        type=int,
        default=256,
        help=(
            "Per-tensor sample cap for the backfill "
            "re-capture (default 256). The cap is per "
            "tensor; a 12B-class model with ~100 "
            "monitor-verdict tensors sees "
            "100 * 256 * max_rounds activations per "
            "pass."
        ),
    )
    parser.add_argument(
        "--backfill-timeout-sec",
        type=int,
        default=600,
        help=(
            "Per-iteration wait on the backfill async "
            "future (default 600s). A stuck subprocess "
            "raises TimeoutExpired; the orchestrator "
            "logs the error and continues."
        ),
    )
    parser.add_argument(
        "--backfill-components",
        default=None,
        help=(
            "Comma-separated ROLE=PATH map of the per-role "
            "components the backfill engine uses. ROLE is "
            "one of trunk / dflash / dspark / mtp_nextn / "
            "shared_embd / vision_tower / audio_tower / "
            "mm_projector. Text-side roles take a layers "
            "directory; mmproj roles take a GGUF path. "
            "Default: empty (the orchestrator's CLI is "
            "free of per-component paths; the backfill "
            "engine's monitor lookup needs them to be "
            "set explicitly when the engine is on)."
        ),
    )
    parser.add_argument(
        "--backfill-corpus",
        type=Path,
        default=None,
        help=(
            "Path to the calibration corpus root. When "
            "set, the backfill samples from the "
            "corpus's domain-specific subsets (the "
            "build-calibration-corpus contract). When "
            "None, the per-tensor driver falls back to "
            "its synthetic-sample default."
        ),
    )
    # Phase 0.5: the EXL2 per-layer error source.
    # ``--exl2-db`` is the path to the unified
    # DuckDB that contains the ``exl2_layer_stats``
    # table the EXL2 calibrator populates. Defaults
    # to ``--retune-from-db`` when set (the
    # orchestrator's normal path); pass this
    # flag explicitly with a different DB to
    # override (e.g. when the EXL2 calibration
    # was run on a separate DB).
    parser.add_argument(
        "--exl2-db",
        type=Path,
        default=None,
        help=(
            "Path to the unified tessera.duckdb "
            "containing the exl2_layer_stats table. "
            "Defaults to --retune-from-db when set; "
            "pass explicitly to override. The "
            "lookup only fires when --w-exl2 > 0."
        ),
    )
    parser.add_argument(
        "--exl2-calibration-corpus",
        type=str,
        default=None,
        help=(
            "Calibration corpus the EXL2 layer "
            "stats were computed against (the "
            "exl2_layer_stats.exl2_calibration_corpus "
            "discriminator). When None, the lookup "
            "returns the most recent row per layer "
            "regardless of corpus (the fallback "
            "when a single-corpus run is in the DB)."
        ),
    )
    # Phase 0.5: per-model disagreement log path.
    # When set, the orchestrator appends a
    # Spearman header + per-verdict disagreement
    # rows every iteration. Default: alongside
    # --policy as <policy>.l5-disagreement.log.
    # The path is independent of --exl2-db /
    # --w-exl2: even when the EXL2 source is not
    # wired, the operator can ask for an empty
    # log (the orchestrator writes a header per
    # iteration, no per-tensor rows because the
    # EXL2 ranking is empty).
    parser.add_argument(
        "--exl2-disagreement-log",
        type=Path,
        default=None,
        help=(
            "Path to the per-model EXL2 disagreement "
            "log (Phase 0.5). Default: alongside "
            "--policy as <policy>.l5-disagreement.log. "
            "Pass /dev/null or an empty string to "
            "disable the log."
        ),
    )
    parser.add_argument(
        "--exl2-disagreement-threshold",
        type=int,
        default=5,
        help=(
            "Per-tensor rank-difference threshold "
            "above which a verdict is logged as a "
            "disagreement (default 5 positions). "
            "Lower values are more verbose; 1 "
            "logs every rank move."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    l4_report = _read_l4_report(args.l4_report)
    imatrix = _read_imatrix(args.imatrix)

    # Weight resolution. The base weights come from the
    # --w-imatrix / --w-gradient / --w-layer flags. The
    # --retune-from-db flag, when paired with --model-hash,
    # overrides the flags with the n_samples-weighted
    # average of the l5_weights rows for the model. This is
    # the consumer half of the "did this requant plan reduce
    # error?" feedback loop: the previous generation's
    # residual determines this generation's weights.
    # Phase 0.5: the 4th weight is the EXL2 per-layer
    # error (``w_exl2``); the l5_weights row is still a
    # 3-tuple (the retune does not learn the EXL2
    # component; the operator sets ``w_exl2`` explicitly
    # via the --w-exl2 flag when the cross-check is on).
    # The aggregate_weights call below returns 3 values;
    # the 4th is appended from ``args.w_exl2``.
    w_im, w_grad, w_layer, w_exl2 = (
        args.w_imatrix, args.w_gradient, args.w_layer,
        args.w_exl2,
    )
    retune_source = None
    per_family_top_fraction: dict[str, float] = {}
    if args.retune_from_db is not None:
        if args.model_hash is None:
            raise ValueError(
                "--retune-from-db requires --model-hash; the l5_weights "
                "rows are keyed by model_hash, and a model-less lookup "
                "would silently mix recommendations across models."
            )
        if args.model_role is not None and args.model_hash is None:
            # Belt and braces (the argparse --model-hash default
            # is None, so this branch is reachable when
            # --model-role is set but --model-hash is not).
            # The argparse doesn't enforce a dependency between
            # --model-role and --model-hash because the legacy
            # --retune-from-db path doesn't need --model-hash
            # (the l5_retune is what enforces it). We enforce
            # it here for the role-aware path.
            raise ValueError(
                "--model-role requires --model-hash; a bare "
                "model_role filter would silently mix roles "
                "across models."
            )
        # Local import to keep the top of this module light; the
        # l5_retune module pulls polars + duckdb at import time.
        from l5_retune import (
            aggregate_weights,
            read_l5_weights,
            read_per_family_top_fraction,
            resolve_l5_weights_for_orchestrator,
            resolve_per_family_top_fraction_for_orchestrator,
        )
        # Retune follow-ups: cross-model dedup. When the
        # requested model_hash is not in the DB and
        # --cross-model-dedup is set, look for a different
        # model with a matching tensor_stats fingerprint and
        # reuse the matched model's l5_weights. The
        # fingerprint match is a 5-moment hash of the
        # per-tensor stat distributions (see
        # ``_model_hash_fingerprint`` in l5_retune.py). On
        # a match, we override args.model_hash with the
        # matched value so the rest of the lookup chain
        # uses the matched model. The override is logged to
        # stderr; a final warning in the "no row found"
        # branch also references the original model_hash
        # so the operator can see what happened.
        dedup_matched_from: str | None = None
        if args.cross_model_dedup:
            from l5_retune import find_fingerprint_match
            role_for_dedup = args.model_role or "trunk"
            try:
                match = find_fingerprint_match(
                    args.retune_from_db,
                    args.model_hash,
                    model_role=role_for_dedup,
                )
            except Exception as e:
                sys.stderr.write(
                    f"[l5] cross-model-dedup: fingerprint scan "
                    f"failed ({e.__class__.__name__}: "
                    f"{str(e)[:120]}); skipping dedup\n"
                )
                match = None
            if match is not None and match != args.model_hash:
                dedup_matched_from = args.model_hash
                print(
                    f"[l5] cross-model-dedup: model_hash="
                    f"{args.model_hash!r} not in DB; reusing "
                    f"weights from fingerprint-matched model "
                    f"{match!r}",
                    file=sys.stderr,
                )
                args.model_hash = match
        # Phase 16: 3-tier lookup. The retune is per-(model,
        # model_role, family) so the same family in different
        # roles gets independent (w_imatrix, w_gradient,
        # w_layer) tuples. The tiers are:
        #   1. (model_hash, model_role, family) per-model,
        #      per-role, per-family (the production path).
        #   2. ("*", model_role, family) cross-model,
        #      per-role, per-family (when
        #      --retune-cross-model-fallback is set; warms
        #      new models from the cross-model mean).
        #   3. (model_hash, "*", family) per-model, no role
        #      (the legacy pre-Phase-16 path; warm-start
        #      a model that has trunk-only retune rows
        #      but the consumer asked for the dflash role).
        #   4. base weights (the --w-* flag values).
        # The first tier with a non-empty result wins; the
        # remaining tiers are not consulted.
        #
        # Retune follow-ups: the lookup is wrapped in a
        # process-local cache (resolve_l5_weights_for_orchestrator)
        # so the second call in the same process returns
        # without re-querying DuckDB. The cache key
        # includes db_path so a path change produces a
        # different entry (no manual invalidation
        # required; a long-running service that replaces
        # the DB can call clear_l5_weights_lookup_cache
        # to drop the stale entries).
        weights_df = resolve_l5_weights_for_orchestrator(
            args.retune_from_db,
            model_hash=args.model_hash,
            model_role=args.model_role,
            cross_model_fallback=args.retune_cross_model_fallback,
        )
        if weights_df.height == 0:
            print(
                f"WARN: --retune-from-db {args.retune_from_db} has no "
                f"l5_weights for model_hash={args.model_hash!r}"
                + (
                    f", model_role={args.model_role!r}"
                    if args.model_role is not None else ""
                )
                + "; using the --w-* flag values",
                file=sys.stderr,
            )
        else:
            w_im, w_grad, w_layer = aggregate_weights(
                weights_df,
                base_weights=(args.w_imatrix, args.w_gradient, args.w_layer),
            )
            # Phase 0.5: w_exl2 stays at the flag value
            # (the l5_weights row does not carry the EXL2
            # component; the operator sets it explicitly
            # via --w-exl2 when the cross-check is on).
            retune_source = str(args.retune_from_db)
            role_label = (
                f" (role={args.model_role})" if args.model_role is not None else ""
            )
            print(
                f"[l5] retune-from-db: {weights_df.height} family row(s)"
                f"{role_label} "
                f"-> w=({w_im:.3f}, {w_grad:.3f}, {w_layer:.3f}, "
                f"w_exl2={w_exl2:.3f})",
                file=sys.stderr,
            )

    # Per-family top_fraction. The retune writes a
    # top_fraction recommendation per (model, family); the
    # orchestrator consumes it via --per-family-top-fraction
    # (which defaults to --retune-from-db when not set). The
    # value overrides the uniform --top-fraction for the
    # families the retune has flagged. Families without a
    # per-family row use the --top-fraction flag value.
    #
    # Phase 16: the per-family lookup is role-aware. The
    # retune writes per-(model, model_role, family)
    # top_fraction recommendations; the orchestrator
    # looks up dflash-specific recommendations when
    # --model-role=dflash, etc. When --model-role is set,
    # the cross-model fallback is also role-aware
    # (the dflash cross-model aggregate fills in for
    # dflash per-model families, not the trunk
    # cross-model aggregate). When --model-role is None,
    # the legacy pre-Phase-16 path is taken (no role
    # filter on the lookup).
    top_fraction_db = args.per_family_top_fraction
    if top_fraction_db is None and args.retune_from_db is not None:
        top_fraction_db = args.retune_from_db
    if top_fraction_db is not None:
        # Retune follow-ups: cached variant of
        # read_per_family_top_fraction. The second call
        # in the same process short-circuits the DB
        # query. The cache key includes db_path so a
        # path change produces a different entry.
        per_family_top_fraction = (
            resolve_per_family_top_fraction_for_orchestrator(
                top_fraction_db,
                model_hash=args.model_hash,
                model_role=args.model_role,
                cross_model_fallback=args.retune_cross_model_fallback,
            )
        )
        if per_family_top_fraction:
            role_label = (
                f" (role={args.model_role})" if args.model_role is not None else ""
            )
            print(
                f"[l5] per-family-top-fraction: "
                f"{len(per_family_top_fraction)} family recommendation(s)"
                f"{role_label} "
                f"from {top_fraction_db}",
                file=sys.stderr,
            )

    scorer = SensitivityScorer(
        decay=args.ema_decay,
        # Phase 0.5: 4-tuple weights (w_im, w_grad,
        # w_layer, w_exl2). The default w_exl2=0.0
        # keeps the EXL2 term zero; the operator
        # opts in via --w-exl2.
        weights=(w_im, w_grad, w_layer, w_exl2),
        total_layers=args.total_layers,
        # Phase 16: role is plumbed through the scorer;
        # every per-tensor RequantAction gets the role
        # so the l5_plan_summary writer tags the row.
        model_role=args.model_role or "trunk",
    )
    planner = RequantPlanner(
        top_fraction=args.top_fraction,
        bottom_fraction=args.bottom_fraction,
        budget_bits=args.budget_bits,
        divergence_threshold=args.divergence_threshold,
        per_family_top_fraction=per_family_top_fraction,
        # Phase 16: role on the planner too; the
        # actions emitted by plan() carry the role.
        model_role=args.model_role or "trunk",
    )

    apply_fn: ApplyFn | None = None
    if args.apply:
        existing = _read_existing_policy(args.existing_policy)
        if existing is None:
            raise ValueError(
                "--apply requires --existing-policy; provide the base calibration policy to mutate"
            )
        if args.existing_policy is None or args.policy is None:
            raise ValueError("--apply requires --existing-policy and --policy")
        if shutil.which(sys.executable) is None:
            raise RuntimeError("interpreter vanished mid-orchestrator")
        def _apply(plan: RequantPlan) -> dict[str, str]:
            return apply_plan_to_policy(plan, args.existing_policy)
        apply_fn = _apply

    # Targeted re-calibration: the backfill engine is
    # constructed only when --targeted-recal is on. The
    # default is on when --retune-from-db is set (the
    # orchestrator has a DB reference); off otherwise.
    # The ``--no-targeted-recal`` flag forces the engine
    # to None (the byte-equivalent pre-task behavior on
    # iteration ordering).
    targeted_recal_on = (
        args.targeted_recal
        if args.targeted_recal is not None
        else (args.retune_from_db is not None)
    )
    backfill_engine = None
    backfill_components: dict[str, Path | None] = {}
    if targeted_recal_on:
        try:
            from backfill import TargetedBackfill  # type: ignore
        except ImportError:  # pragma: no cover - script-mode
            sys.path.insert(
                0, str(Path(__file__).resolve().parent),
            )
            from backfill import TargetedBackfill  # type: ignore
        backfill_engine = TargetedBackfill(
            max_backfill_rounds=args.max_backfill_rounds,
            sample_cap=args.backfill_sample_cap,
            verbose=args.verbose,
        )
        # Parse the --backfill-components map. The
        # format is ``ROLE=PATH[,ROLE=PATH...]``. ROLE
        # is one of the 8 unified-schema values; the
        # text-side roles take a layers directory; the
        # mmproj roles take a GGUF path.
        if args.backfill_components:
            for entry in args.backfill_components.split(","):
                if "=" not in entry:
                    raise ValueError(
                        f"--backfill-components: expected "
                        f"ROLE=PATH, got {entry!r}"
                    )
                role, path_str = entry.split("=", 1)
                backfill_components[role.strip()] = Path(
                    path_str.strip()
                )

    # Phase 0.5: per-model disagreement log path.
    # Default: alongside --policy as
    # <policy>.l5-disagreement.log. The empty-string
    # form (``--exl2-disagreement-log ""``) disables the
    # log; ``/dev/null`` is the operator's explicit
    # "do not write" choice.
    disagreement_log_path: Path | None = args.exl2_disagreement_log
    if disagreement_log_path is None and args.policy is not None:
        disagreement_log_path = (
            args.policy.with_name(args.policy.stem + ".l5-disagreement.log")
        )
    if (
        disagreement_log_path is not None
        and str(disagreement_log_path) == ""
    ):
        disagreement_log_path = None

    orchestrator = OrchestratorLoop(
        scorer=scorer,
        planner=planner,
        apply=apply_fn,
        # Auto-converge resolves the new defaults: when --auto-converge
        # is on (the default), max-iterations bumps to 16 (was 5) and
        # divergence_threshold gets a meaningful 1e-4 default. When
        # --no-auto-converge is set, both stay at their legacy values
        # (5 and None) so the byte-equivalent pre-task behavior holds.
        max_iterations=(
            args.max_iterations
            if args.max_iterations is not None
            else (16 if args.auto_converge else 5)
        ),
        divergence_threshold=(
            args.divergence_threshold
            if args.divergence_threshold is not None
            else (1e-4 if args.auto_converge else None)
        ),
        sidecar=args.policy,
        verbose=args.verbose,
        auto_converge=args.auto_converge,
        converge_tolerance_delta=args.converge_tolerance,
        converge_tolerance_storage=args.converge_storage_tolerance,
        converge_window=args.converge_window,
        # Targeted re-calibration: the backfill
        # engine is None when --no-targeted-recal
        # is set; the orchestrator's hook is
        # bypassed in that case (byte-equivalent
        # pre-task behavior on iteration ordering).
        backfill=backfill_engine,
        max_backfill_rounds=args.max_backfill_rounds,
        backfill_sample_cap=args.backfill_sample_cap,
    )
    # Phase 0.5: the per-model EXL2 disagreement log.
    # The default path is alongside ``--policy`` as
    # ``<policy>.l5-disagreement.log``; the empty-string
    # form (``--exl2-disagreement-log ""``) disables the
    # log. The threshold is the per-tensor rank
    # difference above which a verdict is logged as
    # a disagreement.
    orchestrator.set_disagreement_log_path(disagreement_log_path)
    orchestrator.disagreement_rank_threshold = int(
        args.exl2_disagreement_threshold
    )
    # Wire the backfill engine's runtime context.
    # The context is the same DB the retune uses
    # (so we do not pay a second DuckDB open on
    # the same file); the ``--backfill-db`` path
    # is the same path the retune lookup opened
    # (or ``--backfill-db`` itself when
    # ``--retune-from-db`` is not set).
    if backfill_engine is not None:
        if args.retune_from_db is not None:
            try:
                from tessera_db import TesseraDB  # type: ignore
                db_ctx = TesseraDB.open(args.retune_from_db)
            except Exception as e:  # pragma: no cover - safety
                sys.stderr.write(
                    f"[l5] targeted-recal: TesseraDB.open "
                    f"failed: {e.__class__.__name__}: "
                    f"{str(e)[:120]}; backfill hook bypassed\n"
                )
                db_ctx = None
            orchestrator.enable_backfill(
                db=db_ctx,
                db_path=args.retune_from_db,
                model_hash=str(args.model_hash or ""),
                components=backfill_components,
                corpus_root=args.backfill_corpus,
                timeout_sec=args.backfill_timeout_sec,
            )
        else:
            # No DB reference; the hook is bypassed.
            sys.stderr.write(
                "[l5] targeted-recal: --retune-from-db is not set; "
                "the backfill hook needs a DB reference. "
                "Pass --retune-from-db PATH or "
                "--no-targeted-recal to bypass.\n"
            )
    # Phase 0.5: the EXL2 per-layer error source.
    # When ``w_exl2 > 0`` and a unified DuckDB is
    # available (either via ``--retune-from-db`` or
    # a new ``--exl2-db`` flag the operator passes
    # explicitly), the orchestrator reads the
    # ``exl2_layer_stats`` table and folds the
    # per-layer error into the per-tensor sensitivity
    # score. The lookup uses
    # ``TesseraDB.get_exl2_per_layer_errors`` which
    # is additive and idempotent (a pre-Phase-0.5
    # DB without the table sees the migration on
    # TesseraDB.open). When the DB has no rows for
    # the current ``model_hash``, the lookup
    # returns an empty dict and the fold
    # contributes zero (the 4-component math is
    # still well-defined; the term is just zero).
    exl2_per_layer_errors: dict[int, float] | None = None
    exl2_db_path: Path | None = (
        args.retune_from_db if args.retune_from_db is not None
        else getattr(args, "exl2_db", None)
    )
    if exl2_db_path is not None and w_exl2 > 0.0:
        try:
            from tessera_db import TesseraDB  # type: ignore
            with TesseraDB.open(exl2_db_path) as db:
                exl2_per_layer_errors = (
                    db.get_exl2_per_layer_errors(
                        model_hash=str(args.model_hash or ""),
                        calibration_corpus=args.exl2_calibration_corpus,
                    )
                )
            n_layers = len(exl2_per_layer_errors)
            print(
                f"[l5] exl2-source: {n_layers} layer row(s) from "
                f"{exl2_db_path} (corpus={args.exl2_calibration_corpus})",
                file=sys.stderr,
            )
        except Exception as e:  # pragma: no cover - safety
            sys.stderr.write(
                f"[l5] exl2-source: TesseraDB lookup failed: "
                f"{e.__class__.__name__}: {str(e)[:120]}; "
                f"the EXL2 fold is bypassed\n"
            )
            exl2_per_layer_errors = None
    plans = orchestrator.run(
        l4_report, imatrix,
        exl2_per_layer_errors=exl2_per_layer_errors,
    )

    # History file defaults next to the policy file.
    if args.history is None and args.policy is not None:
        args.history = args.policy.with_name(args.policy.stem + ".history.json")
    if args.history is not None:
        OrchestratorLoop.write_history(plans, args.history)

    # Print a short summary to stdout for shell pipelines.
    last = plans[-1]
    # Print the termination reason on its own line for operator
    # visibility (the JSON block at the end is the parseable
    # contract; this is for humans reading the terminal).
    term = last.termination_reason or "no-termination-recorded"
    print(
        f"termination: {term} "
        f"(auto_converge={'on' if args.auto_converge else 'off'}, "
        f"window={args.converge_window if args.auto_converge else 'n/a'})",
        file=sys.stderr,
    )
    summary = {
        "schema": SCHEMA,
        "iterations": len(plans),
        "termination_reason": last.termination_reason,
        "actions": [a.to_dict() for a in last.actions],
        "storage_before_bits": last.storage_before_bits,
        "storage_after_bits": last.storage_after_bits,
        "weights": list(scorer.weights),
        "retune_source": retune_source,
        "policy": str(args.policy) if args.policy else None,
        "history": str(args.history) if args.history else None,
        "auto_converge": args.auto_converge,
        "converge_tolerance_delta": args.converge_tolerance,
        "converge_tolerance_storage": args.converge_storage_tolerance,
        "converge_window": args.converge_window,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
