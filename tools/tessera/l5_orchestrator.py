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

# L5 lives in the same package as per_tensor_calibrate.py so we can import
# the helpers by package-relative name when the script is run as a module,
# and fall back to a sys.path-anchored import for the script path.
try:
    from . import l5_metrics as metrics
except ImportError:  # pragma: no cover - script-mode fallback
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import l5_metrics as metrics  # type: ignore[no-redef]


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


@dataclasses.dataclass
class RequantAction:
    """One per-tensor requantization decision."""

    name: str
    from_qtype: str
    to_qtype: str
    expected_mse_delta: float
    sensitivity: float
    reason: str
    storage_delta_bits: int

    def to_dict(self) -> dict:
        return {
            "tensor": self.name,
            "from": self.from_qtype,
            "to": self.to_qtype,
            "expected_mse_delta": self.expected_mse_delta,
            "sensitivity": self.sensitivity,
            "reason": self.reason,
            "storage_delta_bits": self.storage_delta_bits,
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


class SensitivityScorer:
    """Combine the three L5 sensitivity components and run them through EMA.

    The scorer is the only piece that knows about the relative weighting of
    the three components; the rest of the orchestrator treats it as a
    function from ``(imatrix, l4_states) -> ema_scores``.
    """

    def __init__(
        self,
        *,
        decay: float = 0.9,
        weights: tuple[float, float, float] = metrics.DEFAULT_WEIGHTS,
        total_layers: int = 0,
    ) -> None:
        self.weights = tuple(float(w) for w in weights)
        if not math.isclose(sum(self.weights), 1.0, abs_tol=1e-6):
            raise ValueError(
                f"sensitivity weights must sum to 1.0, got {sum(self.weights):.6f}"
            )
        self.total_layers = int(total_layers)
        self.ema = metrics.MomentumEMA(decay=decay)
        self._raw_components: tuple[
            metrics.ComponentScores, metrics.ComponentScores, metrics.ComponentScores
        ] = ({}, {}, {})

    @property
    def decay(self) -> float:
        return self.ema.decay

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
        tensors: Sequence[TensorState],
        imatrix: Mapping[str, float] | None,
    ) -> dict[str, float]:
        """Compute a fresh sensitivity score and update the EMA.

        ``imatrix`` may be ``None`` if no imatrix is available; the scorer
        falls back to gradient + layer-prior only and rebalances the
        weights so the total still sums to 1.0.
        """
        names = [t.name for t in tensors]
        im = metrics.sanitise(
            metrics.imatrix_magnitude(
                {n: float(imatrix.get(n, 0.0)) for n in names} if imatrix else None
            )
        )
        grad = metrics.sanitise(
            metrics.gradient_proxy(
                {n: t.mse for n, t in zip(names, tensors)},
                {n: (t.mse_minus_one if t.mse_minus_one is not None else t.mse) for n, t in zip(names, tensors)},
            )
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
            w_im, w_grad, w_layer = 0.0, self.weights[1] + self.weights[0] * 0.6, self.weights[2] + self.weights[0] * 0.4
            total = w_im + w_grad + w_layer
            weights = (w_im / total, w_grad / total, w_layer / total)
        else:
            weights = self.weights
        combined = metrics.combine((im, grad, layer), weights=weights)
        return self.ema.update(combined)

    def reset(self) -> None:
        self.ema.reset()
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

    def plan(
        self,
        iteration: int,
        tensors: Sequence[TensorState],
        sensitivity: Mapping[str, float],
    ) -> RequantPlan:
        """Return a :class:`RequantPlan` for the current state.

        The plan moves the top-fraction tensors up one rung and the
        bottom-fraction tensors down one rung, subject to the storage
        budget.  The iteration is also the termination check: if every
        tensor is below the divergence threshold the plan marks itself as
        converged and produces an empty action list.
        """
        # Sort tensors by current sensitivity descending (the "hot" tensors).
        ranked = sorted(
            tensors,
            key=lambda t: sensitivity.get(t.name, 0.0),
            reverse=True,
        )

        # If everything is already below the divergence threshold, the loop
        # has nothing to do.  This is the primary termination signal.
        if self.divergence_threshold is not None and all(
            t.mse <= self.divergence_threshold for t in tensors
        ):
            storage_before = sum(t.bits() for t in tensors)
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

        # Pick the cohorts.
        top_set = metrics.pick_top_fraction(sensitivity, self.top_fraction)
        bottom_set = metrics.pick_bottom_fraction(sensitivity, self.bottom_fraction)
        # Never down-quant the same tensor we are up-quantising in the same
        # pass; the sets are intentionally derived from disjoint rank bands.
        top_set -= bottom_set

        actions: list[RequantAction] = []
        storage_before = sum(t.bits() for t in tensors)

        # Walk the top cohort first so the budget check uses the most
        # important changes.
        for tensor in ranked:
            if tensor.name not in top_set:
                continue
            target = metrics.step_up(tensor.current_qtype)
            if target is None:
                # Already at BF16 - nothing we can do.
                continue
            sens = float(sensitivity.get(tensor.name, 0.0))
            expected_delta = tensor.expected_mse_at(target, sensitivity=sens) - tensor.mse
            bits_after = int(
                round(metrics.BITS_PER_WEIGHT.get(target, 0.0))
                * float(tensor.n_weights)
            )
            bits_before = tensor.bits()
            actions.append(
                RequantAction(
                    name=tensor.name,
                    from_qtype=tensor.current_qtype,
                    to_qtype=target,
                    expected_mse_delta=expected_delta,
                    sensitivity=sens,
                    reason="top-fraction",
                    storage_delta_bits=bits_after - bits_before,
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
        for tensor in reversed(ranked):
            if tensor.name not in bottom_set:
                continue
            target = metrics.step_down(tensor.current_qtype)
            if target is None:
                continue
            sens = float(sensitivity.get(tensor.name, 0.0))
            expected_delta = tensor.expected_mse_at(target, sensitivity=sens) - tensor.mse
            bits_after = int(
                round(metrics.BITS_PER_WEIGHT.get(target, 0.0))
                * float(tensor.n_weights)
            )
            bits_before = tensor.bits()
            actions.append(
                RequantAction(
                    name=tensor.name,
                    from_qtype=tensor.current_qtype,
                    to_qtype=target,
                    expected_mse_delta=expected_delta,
                    sensitivity=sens,
                    reason="bottom-fraction",
                    storage_delta_bits=bits_after - bits_before,
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
        """
        tensors = self._load_tensors(l4_report)
        if not tensors:
            raise ValueError("L4 report has no tensors; nothing to do")

        # Track the per-tensor qtype we have settled on.  We start from
        # the L4 report's current_qtype; the loop updates this map as
        # actions fire.  The sidecar policy is built from this final map
        # so it reflects the cumulative state, not just the last
        # iteration's actions.
        final_qtype: dict[str, str] = {t.name: t.current_qtype for t in tensors}

        for iteration in range(1, self.max_iterations + 1):
            sensitivity = self.scorer.score(tensors, imatrix)
            plan = self.planner.plan(iteration, tensors, sensitivity)
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

            for tensor in tensors:
                if tensor.name in new_qtypes:
                    tensor.current_qtype = new_qtypes[tensor.name]
                    final_qtype[tensor.name] = new_qtypes[tensor.name]

            # Synthetic L4 re-evaluation: in the demo we just read the new
            # expected MSE off the plan actions.  In production this would
            # be replaced by re-running the L4 probe.
            for action in plan.actions:
                for tensor in tensors:
                    if tensor.name == action.name:
                        tensor.mse = max(0.0, tensor.mse + action.expected_mse_delta)
                        break
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
    def _load_tensors(l4_report: Mapping[str, object]) -> list[TensorState]:
        tensors_payload = l4_report.get("tensors", {})
        if not isinstance(tensors_payload, Mapping):
            raise ValueError("L4 report: 'tensors' must be a mapping")
        out: list[TensorState] = []
        for name, payload in tensors_payload.items():
            if not isinstance(payload, Mapping):
                raise ValueError(f"L4 report: tensor {name!r} payload is not a mapping")
            out.append(TensorState.from_l4(str(name), dict(payload)))
        out.sort(key=lambda t: t.name)
        return out

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
                    "imatrix_magnitude":  float("nan"),
                    "gradient_proxy":     float("nan"),
                    "layer_position_prior": float("nan"),
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
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    l4_report = _read_l4_report(args.l4_report)
    imatrix = _read_imatrix(args.imatrix)

    scorer = SensitivityScorer(
        decay=args.ema_decay,
        weights=(args.w_imatrix, args.w_gradient, args.w_layer),
        total_layers=args.total_layers,
    )
    planner = RequantPlanner(
        top_fraction=args.top_fraction,
        bottom_fraction=args.bottom_fraction,
        budget_bits=args.budget_bits,
        divergence_threshold=args.divergence_threshold,
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
        "policy": str(args.policy) if args.policy else None,
        "history": str(args.history) if args.history else None,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
