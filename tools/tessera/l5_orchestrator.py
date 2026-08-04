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

import polars as pl

# L5 lives in the same package as per_tensor_calibrate.py so we can import
# the helpers by package-relative name when the script is run as a module,
# and fall back to a sys.path-anchored import for the script path.
try:
    from . import l5_metrics as metrics
except ImportError:  # pragma: no cover - script-mode fallback
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import l5_metrics as metrics  # type: ignore[no-redef]

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
        weights: tuple[float, float, float] = metrics.DEFAULT_WEIGHTS,
        total_layers: int = 0,
        model_role: str = "trunk",
    ) -> None:
        self.weights = tuple(float(w) for w in weights)
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
        # Most recent (im, grad, layer) component dicts, kept for
        # debug / log output. Same contract as before.
        self._raw_components: tuple[
            metrics.ComponentScores, metrics.ComponentScores, metrics.ComponentScores
        ] = ({}, {}, {})

    def raw_components(self) -> tuple[
        metrics.ComponentScores, metrics.ComponentScores, metrics.ComponentScores
    ]:
        """Return the most recent (imatrix, gradient, layer_prior) components.

        Exposed for debug; the EMA-tracked scores are what the planner
        consumes.
        """
        return self._raw_components

    def score(
        self,
        df: pl.DataFrame,
        imatrix: Mapping[str, float] | None,
    ) -> pl.DataFrame:
        """Compute fresh sensitivity scores and update the EMA in-place.

        Returns the input DataFrame augmented with five columns:
        ``imatrix_magnitude``, ``gradient_proxy``,
        ``layer_position_prior``, ``sensitivity_score`` (the
        per-iteration weighted sum), and ``sensitivity_ema`` (the
        EMA-tracked value). The EMA state is updated so the next
        iteration's ``sensitivity_ema`` continues smoothly from this
        one.

        ``imatrix`` may be ``None`` if no imatrix is available; the
        scorer falls back to gradient + layer-prior only and
        rebalances the weights so the total still sums to 1.0.
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
        self._raw_components = (im, grad, layer)

        # Rebalance the weights when the imatrix is missing so the surviving
        # two components carry the full mass.  This is the only place that
        # knows about the original weights; the planner just sees the
        # post-rebalance value.
        if not im:
            w_im, w_grad, w_layer = (
                0.0,
                self.weights[1] + self.weights[0] * 0.6,
                self.weights[2] + self.weights[0] * 0.4,
            )
            total = w_im + w_grad + w_layer
            weights = (w_im / total, w_grad / total, w_layer / total)
        else:
            weights = self.weights
        w_im, w_grad, w_layer = weights

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
        )
        out = out.with_columns(
            (w_im * pl.col("imatrix_magnitude")
             + w_grad * pl.col("gradient_proxy")
             + w_layer * pl.col("layer_position_prior")
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
        self._raw_components = ({}, {}, {})


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
        self.history: list[RequantPlan] = []

    # -- public API --------------------------------------------------------

    def run(
        self,
        l4_report: Mapping[str, object],
        imatrix: Mapping[str, float] | None = None,
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
            df = self.scorer.score(df, imatrix)
            plan = self.planner.plan(iteration, df)
            self.history.append(plan)

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
        default=5,
        help="Maximum number of requantization passes (default 5)",
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
        help="MSE threshold below which a tensor is considered converged",
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
    parser.add_argument("--verbose", action="store_true")
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
    w_im, w_grad, w_layer = (
        args.w_imatrix, args.w_gradient, args.w_layer,
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
        )
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
        weights_df = pl.DataFrame(schema=L5_WEIGHTS_COLS)
        if args.model_role is not None:
            # Tier 1: per-model + per-role.
            weights_df = read_l5_weights(
                args.retune_from_db,
                model_hash=args.model_hash,
                model_role=args.model_role,
                cross_model_fallback=False,
            )
            # Tier 2: cross-model + per-role (only when
            # --retune-cross-model-fallback is set).
            if (
                weights_df.height == 0
                and args.retune_cross_model_fallback
            ):
                weights_df = read_l5_weights(
                    args.retune_from_db,
                    model_hash="*",
                    model_role=args.model_role,
                    cross_model_fallback=False,
                )
        if weights_df.height == 0:
            # Tier 3: per-model, no role (legacy pre-Phase-16
            # path). This branch handles two cases:
            #   (a) --model-role is None: the legacy
            #       (model_hash, family) lookup.
            #   (b) --model-role is set but the role has no
            #       per-model rows: fall back to the
            #       role-agnostic per-model rows. The orchestrator
            #       can warm-start a new role from the trunk's
            #       recommendation; the (w_imatrix, w_gradient,
            #       w_layer) tuple is not role-perfect but is
            #       a reasonable starting point.
            weights_df = read_l5_weights(
                args.retune_from_db,
                model_hash=args.model_hash,
                model_role=None,
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
            retune_source = str(args.retune_from_db)
            role_label = (
                f" (role={args.model_role})" if args.model_role is not None else ""
            )
            print(
                f"[l5] retune-from-db: {weights_df.height} family row(s)"
                f"{role_label} "
                f"-> w=({w_im:.3f}, {w_grad:.3f}, {w_layer:.3f})",
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
        from l5_retune import read_per_family_top_fraction
        per_family_top_fraction = read_per_family_top_fraction(
            top_fraction_db,
            model_hash=args.model_hash,
            model_role=args.model_role,
            cross_model_fallback=args.retune_cross_model_fallback,
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
        weights=(w_im, w_grad, w_layer),
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

    orchestrator = OrchestratorLoop(
        scorer=scorer,
        planner=planner,
        apply=apply_fn,
        max_iterations=args.max_iterations,
        divergence_threshold=args.divergence_threshold,
        sidecar=args.policy,
        verbose=args.verbose,
    )
    plans = orchestrator.run(l4_report, imatrix)

    # History file defaults next to the policy file.
    if args.history is None and args.policy is not None:
        args.history = args.policy.with_name(args.policy.stem + ".history.json")
    if args.history is not None:
        OrchestratorLoop.write_history(plans, args.history)

    # Print a short summary to stdout for shell pipelines.
    last = plans[-1]
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
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
