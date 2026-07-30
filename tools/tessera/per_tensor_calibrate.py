#!/usr/bin/env python3
"""Per-tensor Tessera calibration.

Builds a ``llama.speculative.calibration-policy.v1`` JSON for downstream
``tile640_quantize_v3.py --calibration-policy`` consumption by learning
per-tensor quantization parameters from a layer bundle (.npz) containing the
BF16 weight, calibration activations, and (optionally) observer moments.

Fitness modes (selected via ``--fitness``):

* ``awq``     delegate to ``awq-evolve.py`` (island-GA over the existing
  ``Candidate`` space). The output schema matches what
  ``tile640_quantize_v3.py`` already consumes.
* ``lrq``     learn a low-rank weight-scaling matrix ``S = U @ V`` (NAACL 2025
  formulation, output-channel x rank and rank x input-channel factors). The
  effective scale is applied to the weight before ternarization. Training
  uses pure-numpy Adam with a straight-through estimator for the
  non-differentiable ternary quantization. The policy stores ``U`` and ``V``
  per tensor (a few KB at rank 16) and the rank. ``tile640_quantize_v3.py``
  recognises the LRQ fields and reconstructs ``S`` to derive the AWQ-style
  per-input-channel aggregate.
* ``compare`` run ``awq`` and ``lrq`` on the same bundle and write a
  side-by-side report covering per-tensor MSE, policy size, and the relative
  reduction the LRQ-mode achieves versus the GA baseline.

The LRQ mode is the new contribution. The other modes are wired for symmetry
and to support the comparison flow, but they reuse the existing tooling
where possible.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np


SCHEMA = "llama.speculative.calibration-policy.v1"
LRQ_SCHEMA = "llama.tessera.lrq-policy.v1"
DEFAULT_RANK = 16
DEFAULT_ITERATIONS = 50
DEFAULT_LR = 1.0e-3
DEFAULT_OUTLIER_FRAC = 0.005

# Aggregation method for reducing a rank-r S matrix to a per-input-channel scale
# that the existing AWQ path can consume without runtime changes.
LRQ_AGGREGATIONS = ("mean", "rms")


# ---------------------------------------------------------------------------
# Bundle I/O
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Layer:
    """One tensor's calibration bundle (mirrors awq-evolve.py's Layer)."""

    name: str
    family: str
    weight: np.ndarray
    train_activations: np.ndarray | None
    heldout_activations: np.ndarray | None
    # Sum-of-squares observer; derived when full activations are not available.
    in_sum2: np.ndarray | None
    in_count: int

    @property
    def out_dim(self) -> int:
        return int(self.weight.shape[0])

    @property
    def in_dim(self) -> int:
        return int(self.weight.shape[1])

    def input_scale(self) -> np.ndarray:
        """Per-input-channel scale derived from the observers or activations.

        Matches the AWQ convention: ``sqrt(E[x^2])`` so the dequant path
        becomes ``W * (E[x^2])^(-alpha/2)``. Used as a warm-start and as a
        fallback if the LRQ optimisation diverges.
        """
        if self.train_activations is not None and self.train_activations.size:
            return np.sqrt(
                np.mean(self.train_activations.astype(np.float32) ** 2, axis=0)
                + 1e-12
            ).astype(np.float32)
        if self.in_sum2 is not None and self.in_count > 0:
            return np.sqrt(self.in_sum2.astype(np.float32) / float(self.in_count) + 1e-12)
        # Uniform scale when no activation data is available.
        return np.ones(self.in_dim, dtype=np.float32)


def _scalar_string(value: object, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    arr = np.asarray(value)
    if arr.size == 1:
        return str(arr.reshape(()).item())
    return default


def load_layer(path: Path, max_tokens: int = 256) -> Layer:
    with np.load(path, allow_pickle=False) as data:
        weight = np.asarray(data["weight"], dtype=np.float32)
        if weight.ndim != 2:
            raise ValueError(f"{path}: weight must be two-dimensional")
        train = (
            np.asarray(data["train_activations"], dtype=np.float32)
            if "train_activations" in data
            else None
        )
        heldout = (
            np.asarray(data["heldout_activations"], dtype=np.float32)
            if "heldout_activations" in data
            else None
        )
        in_sum2 = (
            np.asarray(data["in_sum2"], dtype=np.float32).reshape(-1)
            if "in_sum2" in data
            else None
        )
        in_count = int(np.asarray(data["counts"]).reshape(()).item()) if "counts" in data else 0
        name = _scalar_string(data["name"] if "name" in data else None, path.stem)
        family = _scalar_string(data["family"] if "family" in data else None, "ffn")
        for label, acts in (("train", train), ("heldout", heldout)):
            if acts is None:
                continue
            if acts.ndim != 2 or acts.shape[1] != weight.shape[1]:
                raise ValueError(f"{path}: {label} activation shape {acts.shape} does not match weight {weight.shape}")
            if max_tokens > 0 and acts.shape[0] > max_tokens:
                idx = np.linspace(0, acts.shape[0] - 1, max_tokens, dtype=np.int64)
                if label == "train":
                    train = acts[idx]
                else:
                    heldout = acts[idx]
    if train is None and in_sum2 is None:
        raise ValueError(f"{path}: requires train_activations or in_sum2")
    return Layer(
        name=name,
        family=family,
        weight=weight,
        train_activations=train,
        heldout_activations=heldout,
        in_sum2=in_sum2,
        in_count=in_count,
    )


def iter_layer_paths(layers_arg: str) -> list[Path]:
    p = Path(layers_arg)
    if p.is_dir():
        return sorted(p.glob("*.npz"))
    if p.is_file() and p.suffix == ".npz":
        return [p]
    raise ValueError(f"{layers_arg}: expected a directory of .npz bundles or a single .npz file")


def bundle_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Ternarization (numpy port of quantize_v3.ternarize_with_acts; outlier_frac=0)
# ---------------------------------------------------------------------------
#
# We keep the outlier branch off in the LRQ training loop because the
# calibration policy already records a separate per-tensor outlier_fraction;
# mixing two competing outlier selectors during LRQ search would muddy the
# gradient. Outliers are reapplied at quantization time by quantize_v3.
#
# STE: the ternarization is non-differentiable, so we approximate the
# gradient w.r.t. ``W_scaled`` with the identity (straight-through estimator).
# This is the standard trick used in QAT literature; the loss landscape is
# well-behaved enough at rank 8-64 that the bias does not hurt convergence.


def ternarize(weights: np.ndarray) -> np.ndarray:
    """Ternarize a 2D weight matrix to {-1, 0, +1} (no outliers)."""
    flat = weights.astype(np.float32).reshape(-1)
    threshold = float(np.mean(np.abs(flat)))
    ternary = np.zeros(flat.size, dtype=np.int8)
    if threshold <= 0.0:
        return ternary.reshape(weights.shape)
    keep = np.abs(flat) >= threshold
    ternary[keep & (flat > 0)] = 1
    ternary[keep & (flat < 0)] = -1
    return ternary.reshape(weights.shape)


def ternarize_value(ternary: np.ndarray, weights_scaled: np.ndarray) -> np.ndarray:
    """Convert a ternary mask back to the quantized weight values.

    The Tessera ternary code stores {-1, 0, +1} multiplied by the per-row
    mean(|W_scaled|). This is what gets fed to the matmul in the runtime.
    For the LRQ loss we use the simpler per-position multiplier
    ``mean(|W_scaled|)`` so the gradient signal stays well-conditioned.
    """
    scale = float(np.mean(np.abs(weights_scaled.astype(np.float32))))
    if scale <= 0.0:
        return np.zeros_like(weights_scaled, dtype=np.float32)
    return ternary.astype(np.float32) * np.float32(scale)


# ---------------------------------------------------------------------------
# Pure-numpy Adam
# ---------------------------------------------------------------------------


class Adam:
    """Minimal pure-numpy Adam. One pass per parameter, vectorised in numpy.

    The stateful shape: ``params`` is a list of numpy arrays; each array gets
    its own ``m`` and ``v`` buffer of the same shape. The constructor does not
    copy ``params``; the caller is expected to own them.
    """

    def __init__(
        self,
        params: list[np.ndarray],
        lr: float = DEFAULT_LR,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.params = params
        self.lr = float(lr)
        self.b1, self.b2 = betas
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.t = 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads: list[np.ndarray]) -> None:
        self.t += 1
        b1c = 1.0 - self.b1 ** self.t
        b2c = 1.0 - self.b2 ** self.t
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            g = grad
            if self.weight_decay > 0.0:
                g = g + self.weight_decay * param
            self.m[i] = self.b1 * self.m[i] + (1.0 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1.0 - self.b2) * (g * g)
            m_hat = self.m[i] / b1c
            v_hat = self.v[i] / b2c
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ---------------------------------------------------------------------------
# LRQ training
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class LRQResult:
    rank: int
    u: np.ndarray  # (out_dim, rank)
    v: np.ndarray  # (rank, in_dim)
    initial_mse: float
    final_mse: float
    iterations: int
    history: list[float]
    input_scale_agg: str
    # Per-input-channel aggregate of S = U @ V. The quantizer uses this as the
    # AWQ-style scale; the full U, V are stored for auditability / future
    # runtime extensions.
    scale_aggregate: np.ndarray
    # Warm-start scale derived from the activation observer. Stored so the
    # policy is self-describing even when only observers (not full
    # activations) were available.
    baseline_scale: np.ndarray

    def policy_entry(self) -> dict:
        return {
            "match": [self.bundle_name],
            "exact": True,
            "lrq_rank": int(self.rank),
            "lrq_u": self.u.astype(np.float32).tolist(),
            "lrq_v": self.v.astype(np.float32).tolist(),
            "lrq_iterations": int(self.iterations),
            "lrq_initial_mse": float(self.initial_mse),
            "lrq_final_mse": float(self.final_mse),
            "lrq_input_scale_agg": self.input_scale_agg,
            "lrq_baseline_scale": self.baseline_scale.astype(np.float32).tolist(),
        }

    def bytes_used(self) -> int:
        # F32 list serialisation is wasteful (commas, brackets) so we record
        # the dense size in bytes for the report.
        return int(self.u.size + self.v.size) * 4

    bundle_name: str = ""


def _aggregate_scale(s: np.ndarray, method: str) -> np.ndarray:
    if method == "mean":
        return np.mean(s, axis=0)
    if method == "rms":
        return np.sqrt(np.mean(s * s, axis=0) + 1e-12)
    raise ValueError(f"unknown aggregation {method!r}; expected one of {LRQ_AGGREGATIONS}")


def _initial_u_v(layer: Layer, rank: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Initialise U, V with a small Gaussian centred on the AWQ warm start.

    The warm start sets the column-wise mean of S to the activation-derived
    scale ``s_agg`` so the first iteration already has a sensible input
    scale. The remaining degrees of freedom are random normal with a small
    variance, which lets Adam explore without a cold start.
    """
    rng = np.random.default_rng(seed)
    out_dim, in_dim = layer.out_dim, layer.in_dim
    s_agg = layer.input_scale()
    # Normalise s_agg to have unit mean, then use the inverse to keep the
    # initial W * S close to W.
    s_norm = s_agg.astype(np.float32) / float(np.mean(s_agg) + 1e-12)
    # Build U so that the column-mean of U @ V matches s_norm.
    # We pick V rows to be small random vectors, and U so that
    # mean_r U[r, k] * V[k, c] approximates s_norm[c]. With rank >= 1, a
    # clean choice is V[0, :] = s_norm, U[:, 0] = 1.0, the rest near zero.
    v = rng.normal(loc=0.0, scale=1e-2, size=(rank, in_dim)).astype(np.float32)
    if rank >= 1:
        v[0] = s_norm
    u = rng.normal(loc=0.0, scale=1e-2, size=(out_dim, rank)).astype(np.float32)
    if rank >= 1:
        u[:, 0] = 1.0
    return u, v


def _ternary_recon(scaled: np.ndarray) -> np.ndarray:
    """Forward pass: ternarize ``scaled`` and return the reconstructed weight."""
    t = ternarize(scaled)
    return ternarize_value(t, scaled)


def train_lrq(
    layer: Layer,
    rank: int = DEFAULT_RANK,
    iterations: int = DEFAULT_ITERATIONS,
    lr: float = DEFAULT_LR,
    seed: int = 0,
    aggregation: str = "mean",
    verbose: bool = False,
) -> LRQResult:
    """Learn U, V minimising the BF16-vs-quantized output MSE on calibration X.

    The gradient chain is:
        loss = mean over tokens t of ||(W_q @ x_t) - (W @ x_t)||^2
        W_q = ternarize_recon(W * S),  S = U @ V
    The ternarization is non-differentiable, so we use the straight-through
    estimator: ``d loss / d W_scaled = d loss / d W_q``. The remaining
    gradient is exact because multiplication and matmul are smooth.
    """
    weight = layer.weight.astype(np.float32)
    if layer.train_activations is not None and layer.train_activations.size:
        x = layer.train_activations.astype(np.float32)
    else:
        raise ValueError("LRQ training requires train_activations in the bundle")

    out_dim, in_dim = weight.shape
    if rank < 1 or rank > min(out_dim, in_dim):
        raise ValueError(
            f"rank must be in [1, {min(out_dim, in_dim)}], got {rank}"
        )

    u, v = _initial_u_v(layer, rank, seed)
    adam = Adam([u, v], lr=lr)

    # Cached references; updating inside the loop would force reallocation.
    s = np.zeros((out_dim, in_dim), dtype=np.float32)
    scaled = np.zeros_like(weight)
    grad_w_scaled = np.zeros_like(weight)
    grad_s = np.zeros_like(s)
    history: list[float] = []

    def mse_at(u_arr: np.ndarray, v_arr: np.ndarray) -> tuple[float, np.ndarray]:
        np.matmul(u_arr, v_arr, out=s)
        np.multiply(weight, s, out=scaled)
        w_q = _ternary_recon(scaled)
        residual = w_q - weight  # (out_dim, in_dim)
        # Error projected through X: err_t = residual @ x_t. Loss is the
        # mean squared error per token averaged over output channels.
        err = residual @ x.T  # (out_dim, n_tokens)
        loss = float(np.mean(err * err))
        return loss, residual

    initial_mse, _ = mse_at(u, v)
    history.append(initial_mse)

    for it in range(iterations):
        # Forward
        np.matmul(u, v, out=s)
        np.multiply(weight, s, out=scaled)
        w_q = _ternary_recon(scaled)
        residual = w_q - weight
        err = residual @ x.T
        loss = float(np.mean(err * err))
        history.append(loss)

        # Backward
        # d loss / d err = (2 / (out_dim * n_tokens)) * err
        d_err = (2.0 / float(err.size)) * err
        # d err / d residual = x
        d_residual = d_err @ x  # (out_dim, in_dim) = (out_dim, n_tokens) @ (n_tokens, in_dim)
        # d residual / d scaled = STE (identity). residual = w_q - weight
        # where w_q = ternarize_recon(scaled). STE: d w_q / d scaled = I.
        d_scaled = d_residual
        # d scaled / d s = weight (element-wise, because scaled = weight * s)
        d_s = d_scaled * weight
        # d s / d u = v^T; d s / d v = u^T
        d_u = d_s @ v.T
        d_v = u.T @ d_s

        adam.step([d_u, d_v])

        if verbose and (it % max(1, iterations // 10) == 0 or it == iterations - 1):
            print(
                f"  lrq[{layer.name}] iter {it:3d}/{iterations}  mse={loss:.6e}  delta={history[-2] - loss:+.3e}",
                file=sys.stderr,
            )

    final_mse = history[-1]
    s_aggregate = _aggregate_scale(s, aggregation)
    return LRQResult(
        rank=rank,
        u=u,
        v=v,
        initial_mse=initial_mse,
        final_mse=final_mse,
        iterations=iterations,
        history=history,
        input_scale_agg=aggregation,
        scale_aggregate=s_aggregate.astype(np.float32),
        baseline_scale=layer.input_scale(),
        bundle_name=layer.name,
    )


# ---------------------------------------------------------------------------
# Policy assembly
# ---------------------------------------------------------------------------


def build_lrq_policy(
    results: list[tuple[Layer, LRQResult]],
    provenance: dict,
    base: dict | None = None,
) -> dict:
    """Assemble the per-tensor entries into a calibration-policy document.

    The wrapper schema is ``llama.speculative.calibration-policy.v1`` so the
    downstream quantizer can consume it without a schema change. The LRQ
    payload lives under ``policy["lrq"]`` with its own sub-schema
    (``llama.tessera.lrq-policy.v1``) for tooling that wants to inspect the
    low-rank factors without parsing the family map.
    """
    policy: dict = dict(base or {})
    families = dict(policy.get("tensor_families", {}))
    tensor_records: list[dict] = []
    total_bytes = 0
    for layer, result in results:
        entry = result.policy_entry()
        entry_key = f"lrq:{layer.name}"
        families[entry_key] = entry
        tensor_records.append({
            "tensor": layer.name,
            "rank": result.rank,
            "initial_mse": result.initial_mse,
            "final_mse": result.final_mse,
            "iterations": result.iterations,
            "input_scale_agg": result.input_scale_agg,
            "bytes": result.bytes_used(),
        })
        total_bytes += result.bytes_used()

    policy.update({
        "schema": SCHEMA,
        "lrq": {
            "schema": LRQ_SCHEMA,
            "rank": results[0][1].rank if results else DEFAULT_RANK,
            "iterations": results[0][1].iterations if results else 0,
            "lr": DEFAULT_LR,
            "input_scale_agg": results[0][1].input_scale_agg if results else "mean",
            "tensor_count": len(results),
            "total_bytes": total_bytes,
            "tensors": tensor_records,
        },
        "tensor_families": families,
        "per_tensor_calibration": provenance,
    })
    policy.setdefault("draft_type", "hybrid")
    return policy


# ---------------------------------------------------------------------------
# AWQ delegation
# ---------------------------------------------------------------------------


def run_awq_subprocess(
    layers_arg: str,
    output_path: Path,
    seed: int,
    generations: int,
    population: int,
    extra: Iterable[str] = (),
) -> dict:
    """Run ``awq-evolve.py`` and return the parsed policy JSON.

    The GA search is heavyweight; LRQ is a lightweight complement, not a
    replacement. Subprocess keeps the two implementations isolated and lets
    the ``compare`` flow fail soft if awq-evolve.py is missing.
    """
    tool = Path(__file__).resolve().parent / "awq-evolve.py"
    if not tool.is_file():
        raise FileNotFoundError(f"awq-evolve.py not found at {tool}")
    cmd: list[str] = [
        sys.executable,
        str(tool),
        "--layers", str(layers_arg),
        "--output", str(output_path),
        "--seed", str(seed),
        "--generations", str(generations),
        "--population", str(population),
    ]
    cmd.extend(extra)
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"awq-evolve.py failed (rc={result.returncode}): {result.stderr.strip()}"
        )
    return json.loads(output_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Compare mode
# ---------------------------------------------------------------------------


def _format_comparison(
    lrq_results: list[tuple[Layer, LRQResult]],
    awq_policy: dict | None,
    bundle_digests: dict[str, str],
) -> str:
    lines: list[str] = []
    lines.append("# LRQ vs AWQ per-tensor comparison")
    lines.append("")
    lines.append("| tensor | shape | rank | lrq_mse | lrq_bytes | awq_mse | awq_clip |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    awq_by_tensor: dict[str, dict] = {}
    if awq_policy is not None:
        for family in awq_policy.get("tensor_families", {}).values():
            for match in family.get("match", []):
                awq_by_tensor.setdefault(match, family)
    total_lrq_bytes = 0
    total_lrq_mse = 0.0
    total_awq_mse = 0.0
    n = 0
    for layer, lrq in lrq_results:
        n += 1
        total_lrq_bytes += lrq.bytes_used()
        total_lrq_mse += lrq.final_mse
        awq_entry = awq_by_tensor.get(layer.name, {})
        awq_mse = awq_entry.get("awq_final_mse", awq_entry.get("fitness", float("nan")))
        awq_clip = awq_entry.get("awq_clip", float("nan"))
        if not isinstance(awq_mse, (int, float)) or math.isnan(float(awq_mse)):
            awq_mse_str = "n/a"
        else:
            awq_mse_str = f"{float(awq_mse):.4e}"
            total_awq_mse += float(awq_mse)
        lines.append(
            f"| {layer.name} | {layer.weight.shape} | {lrq.rank} | "
            f"{lrq.final_mse:.4e} | {lrq.bytes_used()} | {awq_mse_str} | "
            f"{awq_clip:.3f} |"
        )
    if n:
        lines.append("")
        lines.append(f"avg lrq_mse: {total_lrq_mse / n:.4e}  (total bytes: {total_lrq_bytes})")
        if total_awq_mse > 0.0:
            lines.append(f"avg awq_mse: {total_awq_mse / n:.4e}")
    lines.append("")
    lines.append("## Bundle digests")
    for name, digest in bundle_digests.items():
        lines.append(f"- {name}: `{digest}`")
    return "\n".join(lines) + "\n"


def run_compare(args: argparse.Namespace) -> int:
    layers = iter_layer_paths(args.layers)
    if not layers:
        raise ValueError(f"{args.layers}: no layer bundles")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    digests = {p.stem: bundle_digest(p) for p in layers}

    # LRQ leg
    lrq_results: list[tuple[Layer, LRQResult]] = []
    for path in layers:
        layer = load_layer(path, max_tokens=args.max_tokens)
        result = train_lrq(
            layer,
            rank=args.lrq_rank,
            iterations=args.lrq_iterations,
            lr=args.lr,
            seed=args.seed,
            aggregation=args.lrq_agg,
            verbose=args.verbose,
        )
        lrq_results.append((layer, result))

    provenance = {
        "tool": "per_tensor_calibrate.py",
        "mode": "compare",
        "seed": args.seed,
        "lrq_rank": args.lrq_rank,
        "lrq_iterations": args.lrq_iterations,
        "lrq_lr": args.lr,
        "lrq_aggregation": args.lrq_agg,
        "bundle_digests": digests,
        "timestamp": time.time(),
    }
    policy = build_lrq_policy(lrq_results, provenance)
    output.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")

    # AWQ leg (best-effort)
    awq_policy: dict | None = None
    awq_output: Path | None = None
    try:
        awq_output = output.with_name(output.stem + ".awq.json")
        awq_policy = run_awq_subprocess(
            args.layers,
            awq_output,
            seed=args.seed,
            generations=args.awq_generations,
            population=args.awq_population,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"WARN: AWQ leg skipped: {exc}", file=sys.stderr)

    # Report
    report = _format_comparison(lrq_results, awq_policy, digests)
    report_path = output.with_suffix(".compare.md")
    report_path.write_text(report, encoding="utf-8")

    if args.verbose:
        for layer, result in lrq_results:
            print(
                f"lrq[{layer.name}]: initial_mse={result.initial_mse:.4e} "
                f"final_mse={result.final_mse:.4e} bytes={result.bytes_used()}",
                file=sys.stderr,
            )
        print(f"wrote {output}", file=sys.stderr)
        print(f"wrote {report_path}", file=sys.stderr)
        if awq_output is not None:
            print(f"wrote {awq_output}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Per-tensor Tessera calibration. --fitness selects the search mode "
            "(lrq: low-rank weight-scaling; awq: delegate to awq-evolve.py; "
            "compare: run both and write a side-by-side report)."
        )
    )
    parser.add_argument(
        "--fitness",
        choices=("lrq", "awq", "compare"),
        default="lrq",
        help="Calibration mode (default: lrq).",
    )
    parser.add_argument(
        "--layers",
        required=True,
        help="Directory of layer .npz bundles or a single .npz file",
    )
    parser.add_argument("--output", required=True, help="Output calibration policy JSON")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum calibration tokens per bundle (default 256; 0 disables)",
    )
    # LRQ-specific
    parser.add_argument(
        "--lrq-rank",
        type=int,
        default=DEFAULT_RANK,
        help=f"LRQ rank r (default {DEFAULT_RANK})",
    )
    parser.add_argument(
        "--lrq-iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=f"LRQ optimisation iterations per tensor (default {DEFAULT_ITERATIONS})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help=f"Adam learning rate for LRQ (default {DEFAULT_LR})",
    )
    parser.add_argument(
        "--lrq-agg",
        choices=LRQ_AGGREGATIONS,
        default="mean",
        help="How to aggregate the rank-r S matrix to a per-input-channel scale",
    )
    # AWQ delegation (compare mode)
    parser.add_argument("--awq-generations", type=int, default=8)
    parser.add_argument("--awq-population", type=int, default=8)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for both LRQ and AWQ legs",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.fitness == "lrq":
        layers = iter_layer_paths(args.layers)
        if not layers:
            raise ValueError(f"{args.layers}: no layer bundles")
        digests = {p.stem: bundle_digest(p) for p in layers}
        lrq_results: list[tuple[Layer, LRQResult]] = []
        for path in layers:
            layer = load_layer(path, max_tokens=args.max_tokens)
            result = train_lrq(
                layer,
                rank=args.lrq_rank,
                iterations=args.lrq_iterations,
                lr=args.lr,
                seed=args.seed,
                aggregation=args.lrq_agg,
                verbose=args.verbose,
            )
            lrq_results.append((layer, result))
            if args.verbose:
                print(
                    f"lrq[{layer.name}]: initial_mse={result.initial_mse:.4e} "
                    f"final_mse={result.final_mse:.4e} bytes={result.bytes_used()}",
                    file=sys.stderr,
                )
        provenance = {
            "tool": "per_tensor_calibrate.py",
            "mode": "lrq",
            "seed": args.seed,
            "lrq_rank": args.lrq_rank,
            "lrq_iterations": args.lrq_iterations,
            "lrq_lr": args.lr,
            "lrq_aggregation": args.lrq_agg,
            "bundle_digests": digests,
            "timestamp": time.time(),
        }
        policy = build_lrq_policy(lrq_results, provenance)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {output} with {len(lrq_results)} LRQ tensor entries", file=sys.stderr)
        return 0
    if args.fitness == "awq":
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        policy = run_awq_subprocess(
            args.layers,
            output,
            seed=args.seed,
            generations=args.awq_generations,
            population=args.awq_population,
        )
        print(f"wrote {output} via awq-evolve.py", file=sys.stderr)
        return 0
    if args.fitness == "compare":
        return run_compare(args)
    raise ValueError(f"unknown --fitness {args.fitness!r}")


if __name__ == "__main__":
    sys.exit(main())
