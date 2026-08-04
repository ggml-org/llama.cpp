#!/usr/bin/env python3
"""Evolutionary AWQ/Tessera policy search.

The search has two levels:

* a vectorized inner population searches numerical AWQ parameters;
* island populations and a MAP-Elites archive preserve distinct policies.

Layer bundles are .npz files containing ``weight`` and either
``train_activations``/``heldout_activations`` or ``in_sum2``/``counts``.
Optional scalar strings ``name`` and ``family`` identify the tensor.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import mlx.core as mx
except ImportError:
    mx = None


SCHEMA = "llama.tessera.awq-evolution.v1"
POLICY_SCHEMA = "llama.speculative.calibration-policy.v1"
FAMILIES = (
    "attention", "ffn", "router", "routed_expert", "shared_expert",
    "fusion", "output_embedding",
)
# Phase 16: model_role mirrors the per_tensor_calibrate.py enum.
# The default 'trunk' preserves the pre-Phase-16 single-component
# behaviour; per_tensor_calibrate.py's --fitness awq path forwards
# the user-supplied role to this script via --model-role.
MODEL_ROLES = ("trunk", "dflash", "dspark", "mtp_nextn", "shared_embd")
DEFAULT_MODEL_ROLE = "trunk"
MATCHES = {
    "attention": ["attn_q", "attn_k", "attn_v", "attn_output"],
    "ffn": ["ffn_gate", "ffn_up", "ffn_down"],
    "router": ["ffn_gate_inp"],
    "routed_expert": ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"],
    "shared_expert": ["ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp"],
    "fusion": ["fc.weight", "encoder_proj", "per_layer_model_proj"],
    "output_embedding": ["token_embd", "output.weight", "per_layer_token_embd"],
}


@dataclasses.dataclass(frozen=True)
class Candidate:
    alpha: float
    clip: float
    outlier_fraction: float
    moment_mix: float
    tail_guard: float
    # Multiplier on the per-row mean(|W|) used as the {-1, 0, +1} ternarization
    # threshold. Default 1.0 = current tessera behaviour. Searched per-tensor
    # by tools/tessera/per_tensor_calibrate.py to fix the mis-calibrated bulk
    # of the requantization (Q/K/V/FFN gate/up/down at layers 4-32 where the
    # global mean(|W|) threshold is too aggressive and 70-150% layer-output
    # error accumulates through the network).
    ternary_threshold: float = 1.0

    def clipped(self) -> "Candidate":
        return Candidate(
            alpha=float(np.clip(self.alpha, 0.0, 1.0)),
            clip=float(np.clip(self.clip, 0.70, 1.0)),
            outlier_fraction=float(np.clip(self.outlier_fraction, 0.0001, 0.05)),
            moment_mix=float(np.clip(self.moment_mix, 0.0, 1.0)),
            tail_guard=float(np.clip(self.tail_guard, 0.0, 2.0)),
            ternary_threshold=float(np.clip(self.ternary_threshold, 0.30, 3.0)),
        )

    def as_array(self) -> np.ndarray:
        return np.asarray(dataclasses.astuple(self), dtype=np.float64)

    @staticmethod
    def from_array(values: np.ndarray) -> "Candidate":
        return Candidate(*(float(value) for value in values)).clipped()


@dataclasses.dataclass
class Layer:
    name: str
    family: str
    weight: np.ndarray
    train_activations: np.ndarray | None
    heldout_activations: np.ndarray | None
    second_moment: np.ndarray
    fourth_moment: np.ndarray
    max_abs: np.ndarray
    rms: np.ndarray | None = None
    kurtosis_excess: np.ndarray | None = None
    tail_excess: np.ndarray | None = None
    train_reference: object | None = None
    heldout_reference: object | None = None
    digest: str | None = None
    sample_count: float = 0.0


@dataclasses.dataclass
class Score:
    train_error: float
    heldout_error: float
    tail_error: float
    size_cost: float
    fitness: float
    worst_layer_error: float = 0.0


@dataclasses.dataclass(frozen=True)
class ProgressiveConfig:
    """Deterministic successive-halving policy for costly candidate scoring."""

    enabled: bool = True
    screen_fraction: float = 0.25
    refine_fraction: float = 0.50
    promotion_margin: float = 0.05
    diversity_slots: int = 2


def _scalar_string(value: np.ndarray | str | None, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    return str(np.asarray(value).reshape(()).item())


def infer_family(name: str) -> str:
    for family, fragments in MATCHES.items():
        if any(fragment in name for fragment in fragments):
            return family
    return "ffn"


def load_transfer_prior(path: Path | None) -> tuple[dict[str, float], float]:
    """Return per-tensor survival priority and reusable observer budget."""
    if path is None:
        return {}, 0.0
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "llama.tessera.progressive-observer-ledger.v1":
        raise ValueError(f"{path}: unsupported progressive observer ledger")
    checkpoint = max(int(payload.get("checkpoint_chunk", 0)), 1)
    priorities: dict[str, float] = {}
    saved = 0.0
    tensors = payload.get("tensors", {})
    for name, state in tensors.items():
        if state.get("frozen", False):
            frozen_at = max(0, min(checkpoint, int(state.get("frozen_at", checkpoint))))
            priorities[name] = frozen_at / checkpoint
            saved += (checkpoint - frozen_at) / checkpoint
        else:
            priorities[name] = 1.0
    saved_fraction = saved / max(len(tensors), 1)
    return priorities, saved_fraction


def transfer_priority(name: str, priorities: dict[str, float]) -> float:
    if name in priorities:
        return priorities[name]
    alternate = name[:-1] if name.endswith("_") else name + "_"
    return priorities.get(alternate, 0.0)


def load_layer(path: Path, max_rows: int, max_tokens: int) -> Layer:
    with np.load(path, allow_pickle=False) as data:
        weight = np.asarray(data["weight"], dtype=np.float32)
        if weight.ndim != 2:
            raise ValueError(f"{path}: weight must be two-dimensional")
        if weight.shape[0] > max_rows:
            rows = np.linspace(0, weight.shape[0] - 1, max_rows, dtype=np.int64)
            weight = weight[rows]
        train = np.asarray(data["train_activations"], dtype=np.float32) if "train_activations" in data else None
        heldout = np.asarray(data["heldout_activations"], dtype=np.float32) if "heldout_activations" in data else None
        for label, activations in (("train", train), ("heldout", heldout)):
            if activations is not None:
                if activations.ndim != 2 or activations.shape[1] != weight.shape[1]:
                    raise ValueError(f"{path}: {label} activation shape does not match weight")
                if activations.shape[0] > max_tokens:
                    selected = np.linspace(0, activations.shape[0] - 1, max_tokens, dtype=np.int64)
                    if label == "train":
                        train = activations[selected]
                    else:
                        heldout = activations[selected]
        sample_count = 0.0
        if train is not None:
            second = np.mean(np.square(train, dtype=np.float32), axis=0)
            fourth = np.mean(np.square(np.square(train, dtype=np.float32), dtype=np.float32), axis=0)
            max_abs = np.max(np.abs(train), axis=0)
            sample_count = float(train.shape[0])
        else:
            if "in_sum2" not in data:
                raise ValueError(f"{path}: requires activations or in_sum2")
            counts = max(float(np.asarray(data.get("counts", 1.0)).sum()), 1.0)
            sample_count = counts
            second = np.asarray(data["in_sum2"], dtype=np.float32).reshape(-1) / counts
            fourth = (
                np.asarray(data["in_sum4"], dtype=np.float32).reshape(-1) / counts
                if "in_sum4" in data else np.square(second, dtype=np.float32)
            )
            max_abs = (
                np.asarray(data["in_maxabs"], dtype=np.float32).reshape(-1)
                if "in_maxabs" in data else np.sqrt(second)
            )
        name = _scalar_string(data["name"] if "name" in data else None, path.stem)
        family = _scalar_string(data["family"] if "family" in data else None, infer_family(name))
    if second.size != weight.shape[1]:
        raise ValueError(f"{path}: telemetry width does not match weight")
    return Layer(
        name, family, weight, train, heldout, second, fourth, max_abs,
        sample_count=sample_count,
    )


def _ternary_reconstruct(weight: np.ndarray, candidate: Candidate, importance: np.ndarray) -> np.ndarray:
    safe = np.maximum(importance, 1e-8)
    reference = float(np.median(safe[np.isfinite(safe) & (safe > 0.0)]))
    relative = np.nan_to_num(
        safe / max(reference, 1e-8),
        nan=1.0,
        posinf=256.0,
        neginf=1.0 / 256.0,
    )
    relative = np.clip(relative, 1.0 / 256.0, 256.0)
    safe = relative * reference
    scale = np.power(relative, candidate.alpha, dtype=np.float32)
    transformed = weight * scale.reshape(1, -1)
    row_limit = np.max(np.abs(transformed), axis=1, keepdims=True) * candidate.clip
    transformed = np.clip(transformed, -row_limit, row_limit)
    # Per-row ternarization threshold: per-row mean(|W|) × ternary_threshold
    # multiplier. The legacy tessera path used a hardcoded 1.0 (mean(|W|));
    # the multiplier is the missing calibration knob. Higher = sparser
    # (more zeros, fewer {-1, +1}); lower = denser (more non-zeros).
    threshold = np.mean(np.abs(transformed), axis=1, keepdims=True) * candidate.ternary_threshold
    ternary = np.where(transformed >= threshold, 1.0, np.where(transformed <= -threshold, -1.0, 0.0))
    denominator = np.sum(np.abs(ternary), axis=1, keepdims=True)
    row_scale = np.divide(
        np.sum(np.abs(transformed) * np.abs(ternary), axis=1, keepdims=True),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )
    reconstructed = ternary * row_scale
    residual_count = max(1, int(math.ceil(weight.size * candidate.outlier_fraction)))
    error_score = np.square(transformed - reconstructed) * safe.reshape(1, -1)
    selected = np.argpartition(error_score.reshape(-1), -residual_count)[-residual_count:]
    reconstructed.reshape(-1)[selected] = transformed.reshape(-1)[selected]
    return reconstructed / scale.reshape(1, -1)


def _layer_features(layer: Layer) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Memoize candidate-independent telemetry transforms for the hot path."""
    if layer.rms is None:
        layer.rms = np.sqrt(np.maximum(layer.second_moment, 1e-12))
        layer.kurtosis_excess = np.maximum(
            layer.fourth_moment / np.maximum(np.square(layer.second_moment), 1e-12) - 3.0,
            0.0,
        )
        layer.tail_excess = np.maximum(layer.max_abs / np.maximum(layer.rms, 1e-6) - 3.0, 0.0)
    assert layer.kurtosis_excess is not None
    assert layer.tail_excess is not None
    return layer.rms, layer.kurtosis_excess, layer.tail_excess


def evaluate(candidate: Candidate, layers: Iterable[Layer], heldout_weight: float = 2.0,
             worst_layer_weight: float = 0.25) -> Score:
    layer_scores = [_evaluate_layer(candidate, layer) for layer in layers]
    return _aggregate_layer_scores(
        layer_scores, candidate.outlier_fraction, heldout_weight, worst_layer_weight)


def _evaluate_layer(candidate: Candidate, layer: Layer) -> Score:
    rms, kurtosis_excess, tail_excess = _layer_features(layer)
    importance = rms * (
        1.0
        + candidate.moment_mix * kurtosis_excess
        + candidate.tail_guard * tail_excess
    )
    reconstructed = _ternary_reconstruct(layer.weight, candidate, importance)
    error = reconstructed - layer.weight
    diagonal_error = float(np.mean(np.square(error) * layer.second_moment.reshape(1, -1)))
    if layer.train_activations is not None:
        diagonal_error = relative_output_error(
            layer.train_activations, layer.weight, reconstructed,
            _reference_output(layer, "train"),
        )
    heldout_error = diagonal_error
    if layer.heldout_activations is not None:
        heldout_error = relative_output_error(
            layer.heldout_activations, layer.weight, reconstructed,
            _reference_output(layer, "heldout"),
        )
    tail_error = float(np.mean(np.square(error) * layer.fourth_moment.reshape(1, -1)))
    return Score(
        diagonal_error, heldout_error, tail_error,
        candidate.outlier_fraction, 0.0, heldout_error)


def _evaluate_layer_batch(candidates: list[Candidate], layer: Layer) -> list[Score]:
    """Score a candidate wave as one fused MLX evaluation graph.

    Tessera reconstruction is deliberately kept on the host: its residual
    selection is exact and candidate-specific.  The expensive part after that
    is evaluating reconstructed matrices against shared activation samples.
    Stack candidates once, concatenate train and heldout samples, and evaluate
    projection, diagonal-moment, and tail-moment losses in one lazy graph.
    """
    if len(candidates) == 1 or mx is None:
        return [_evaluate_layer(candidate, layer) for candidate in candidates]

    rms, kurtosis_excess, tail_excess = _layer_features(layer)
    reconstructed: list[np.ndarray] = []
    for candidate in candidates:
        importance = rms * (
            1.0
            + candidate.moment_mix * kurtosis_excess
            + candidate.tail_guard * tail_excess
        )
        approximation = _ternary_reconstruct(layer.weight, candidate, importance)
        reconstructed.append(approximation)

    stacked = mx.array(np.stack(reconstructed, axis=0))
    error = stacked - mx.array(layer.weight)[None, :, :]
    squared_error = mx.square(error)
    diagonal_values = mx.mean(
        squared_error * mx.array(layer.second_moment)[None, None, :],
        axis=(1, 2),
    )
    tail_values = mx.mean(
        squared_error * mx.array(layer.fourth_moment)[None, None, :],
        axis=(1, 2),
    )

    train_values = diagonal_values
    heldout_values = diagonal_values
    activation_parts: list[np.ndarray] = []
    reference_parts = []
    train_rows = 0
    if layer.train_activations is not None:
        activation_parts.append(layer.train_activations)
        reference_parts.append(_reference_output(layer, "train"))
        train_rows = layer.train_activations.shape[0]
    if layer.heldout_activations is not None:
        activation_parts.append(layer.heldout_activations)
        reference_parts.append(_reference_output(layer, "heldout"))
    if activation_parts:
        activations = mx.array(np.concatenate(activation_parts, axis=0))[None, :, :]
        reference = mx.concatenate(reference_parts, axis=0)
        approximate = mx.matmul(activations, mx.swapaxes(stacked, 1, 2))
        projection_error = mx.square(approximate - reference)
        if layer.train_activations is not None:
            train_reference = reference[:train_rows]
            train_values = mx.mean(
                projection_error[:, :train_rows, :], axis=(1, 2)
            ) / (mx.mean(mx.square(train_reference)) + 1e-12)
        if layer.heldout_activations is not None:
            heldout_start = train_rows
            heldout_reference = reference[heldout_start:]
            heldout_values = mx.mean(
                projection_error[:, heldout_start:, :], axis=(1, 2)
            ) / (mx.mean(mx.square(heldout_reference)) + 1e-12)
        elif layer.train_activations is not None:
            heldout_values = train_values

    mx.eval(train_values, heldout_values, tail_values)
    train_errors = [float(value) for value in np.asarray(train_values)]
    heldout_errors = [float(value) for value in np.asarray(heldout_values)]
    tail_errors = [float(value) for value in np.asarray(tail_values)]
    return [
        Score(train, heldout, tail, candidate.outlier_fraction, 0.0, heldout)
        for candidate, train, heldout, tail in zip(
            candidates, train_errors, heldout_errors, tail_errors)
    ]


def _aggregate_layer_scores(
    layer_scores: list[Score],
    size_cost: float,
    heldout_weight: float = 2.0,
    worst_layer_weight: float = 0.25,
) -> Score:
    if not layer_scores:
        raise ValueError("at least one layer is required for AWQ evaluation")
    train_errors = [score.train_error for score in layer_scores]
    heldout_errors = [score.heldout_error for score in layer_scores]
    tail_errors = [score.tail_error for score in layer_scores]
    train_error = float(np.mean(train_errors))
    heldout_error = float(np.mean(heldout_errors))
    tail_error = float(np.mean(tail_errors))
    worst_layer_error = float(np.quantile(heldout_errors, 0.9))
    fitness = (
        train_error
        + heldout_weight * heldout_error
        + worst_layer_weight * worst_layer_error
        + 0.05 * tail_error
        + 0.15 * size_cost
    )
    return Score(train_error, heldout_error, tail_error, size_cost, fitness, worst_layer_error)


def _evaluate_uncached(candidate: Candidate, layers: Iterable[Layer], heldout_weight: float = 2.0,
                       worst_layer_weight: float = 0.25) -> Score:
    """Compatibility oracle used by tests to verify cached aggregation."""
    train_errors: list[float] = []
    heldout_errors: list[float] = []
    tail_errors: list[float] = []
    for layer in layers:
        rms, kurtosis_excess, tail_excess = _layer_features(layer)
        importance = rms * (
            1.0
            + candidate.moment_mix * kurtosis_excess
            + candidate.tail_guard * tail_excess
        )
        reconstructed = _ternary_reconstruct(layer.weight, candidate, importance)
        error = reconstructed - layer.weight
        diagonal_error = float(np.mean(np.square(error) * layer.second_moment.reshape(1, -1)))
        if layer.train_activations is not None:
            diagonal_error = relative_output_error(
                layer.train_activations, layer.weight, reconstructed,
                _reference_output(layer, "train"),
            )
        heldout_error = diagonal_error
        if layer.heldout_activations is not None:
            heldout_error = relative_output_error(
                layer.heldout_activations, layer.weight, reconstructed,
                _reference_output(layer, "heldout"),
            )
        train_errors.append(diagonal_error)
        heldout_errors.append(heldout_error)
        tail_errors.append(float(np.mean(np.square(error) * layer.fourth_moment.reshape(1, -1))))
    train_error = float(np.mean(train_errors))
    heldout_error = float(np.mean(heldout_errors))
    tail_error = float(np.mean(tail_errors))
    worst_layer_error = float(np.quantile(heldout_errors, 0.9))
    size_cost = candidate.outlier_fraction
    fitness = (
        train_error
        + heldout_weight * heldout_error
        + worst_layer_weight * worst_layer_error
        + 0.05 * tail_error
        + 0.15 * size_cost
    )
    return Score(train_error, heldout_error, tail_error, size_cost, fitness, worst_layer_error)


def _single_layer_digest(layer: Layer) -> str:
    if layer.digest is not None:
        return layer.digest
    digest = hashlib.sha256()
    digest.update(f"{layer.name}:{layer.weight.shape}:{layer.weight.dtype}\n".encode())
    # Candidate scores may only cross an epoch boundary when their source
    # telemetry is identical. Weights are immutable for a search; these three
    # compact observer channels are the changing evidence.
    for values in (layer.second_moment, layer.fourth_moment, layer.max_abs):
        digest.update(np.ascontiguousarray(values, dtype=np.float32).tobytes())
    layer.digest = digest.hexdigest()
    return layer.digest


def _layer_digest(layers: Iterable[Layer]) -> str:
    digest = hashlib.sha256()
    for layer in layers:
        digest.update(_single_layer_digest(layer).encode())
    return digest.hexdigest()


def _stratified_layers(layers: list[Layer], fraction: float) -> list[Layer]:
    """Select depth-distributed bundles without biasing toward early layers."""
    if not layers or fraction >= 1.0 or len(layers) <= 4:
        return list(layers)
    count = max(1, min(len(layers), math.ceil(len(layers) * fraction)))
    ordered = sorted(layers, key=lambda layer: layer.name)
    indices = np.linspace(0, len(ordered) - 1, count, dtype=np.int64)
    return [ordered[index] for index in np.unique(indices)]


def _candidate_key(candidate: Candidate) -> str:
    # JSON preserves a portable checkpoint representation without depending on
    # Python's process-randomized hash implementation.
    return json.dumps(dataclasses.asdict(candidate), sort_keys=True, separators=(",", ":"))


def _cached_evaluate(
    candidate: Candidate,
    layers: list[Layer],
    stage: str,
    cache: dict[str, dict],
) -> Score:
    stage_key = f"{stage}:{_layer_digest(layers)}"
    candidate_key = _candidate_key(candidate)
    entry = cache.get(candidate_key, {}).get(stage_key)
    if entry is not None:
        return Score(**entry)
    candidate_cache = cache.setdefault(candidate_key, {})
    layer_scores: list[Score] = []
    for layer in layers:
        layer_key = f"layer:{_single_layer_digest(layer)}"
        layer_entry = candidate_cache.get(layer_key)
        if layer_entry is None:
            layer_score = _evaluate_layer(candidate, layer)
            candidate_cache[layer_key] = dataclasses.asdict(layer_score)
        else:
            layer_score = Score(**layer_entry)
        layer_scores.append(layer_score)
    score = _aggregate_layer_scores(layer_scores, candidate.outlier_fraction)
    candidate_cache[stage_key] = dataclasses.asdict(score)
    return score


def _cached_evaluate_population(
    candidates: list[Candidate],
    layers: list[Layer],
    stage: str,
    cache: dict[str, dict],
    candidate_batch_size: int,
) -> list[Score]:
    """Fill exact per-layer cache entries in bounded MLX candidate waves."""
    if candidate_batch_size < 1:
        raise ValueError("candidate batch size must be positive")
    stage_key = f"{stage}:{_layer_digest(layers)}"
    results: dict[Candidate, Score] = {}
    pending: list[Candidate] = []
    pending_keys: set[str] = set()
    for candidate in candidates:
        candidate_key = _candidate_key(candidate)
        entry = cache.get(candidate_key, {}).get(stage_key)
        if entry is None:
            if candidate_key not in pending_keys:
                pending.append(candidate)
                pending_keys.add(candidate_key)
        else:
            results[candidate] = Score(**entry)

    for start in range(0, len(pending), candidate_batch_size):
        wave = pending[start:start + candidate_batch_size]
        for layer in layers:
            missing: list[Candidate] = []
            for candidate in wave:
                candidate_cache = cache.setdefault(_candidate_key(candidate), {})
                if f"layer:{_single_layer_digest(layer)}" not in candidate_cache:
                    missing.append(candidate)
            for offset in range(0, len(missing), candidate_batch_size):
                score_wave = _evaluate_layer_batch(
                    missing[offset:offset + candidate_batch_size], layer)
                for candidate, layer_score in zip(
                    missing[offset:offset + candidate_batch_size], score_wave):
                    cache[_candidate_key(candidate)][
                        f"layer:{_single_layer_digest(layer)}"
                    ] = dataclasses.asdict(layer_score)
        for candidate in wave:
            candidate_cache = cache[_candidate_key(candidate)]
            layer_scores = [Score(**candidate_cache[
                f"layer:{_single_layer_digest(layer)}"
            ]) for layer in layers]
            score = _aggregate_layer_scores(layer_scores, candidate.outlier_fraction)
            candidate_cache[stage_key] = dataclasses.asdict(score)
            results[candidate] = score
    return [results[candidate] for candidate in candidates]


def _checkpoint_score_cache(cache: dict[str, dict]) -> dict[str, dict]:
    """Persist exact layer and stage scores so resumed searches stay warm."""
    return cache


def _promote(
    scored: list[tuple[Candidate, Score]],
    keep: int,
    diversity_slots: int,
    margin: float,
) -> list[tuple[Candidate, Score]]:
    """Keep leaders plus near-cutoff MAP-Elites cells for safe exploration."""
    ranked = sorted(scored, key=lambda item: item[1].fitness)
    keep = min(len(ranked), max(1, keep))
    selected = list(ranked[:keep])
    occupied = {archive_cell(candidate, score) for candidate, score in selected}
    cutoff = selected[-1][1].fitness
    limit = cutoff + max(abs(cutoff), 1e-12) * margin
    for candidate, score in ranked[keep:]:
        if diversity_slots <= 0 or score.fitness > limit:
            continue
        cell = archive_cell(candidate, score)
        if cell in occupied:
            continue
        selected.append((candidate, score))
        occupied.add(cell)
        diversity_slots -= 1
    return selected


def progressive_evaluate_population(
    population: list[Candidate],
    layers: list[Layer],
    config: ProgressiveConfig,
    cache: dict[str, dict],
    candidate_batch_size: int = 4,
) -> tuple[list[tuple[Candidate, Score]], dict[str, int]]:
    """Use inexpensive stratified evidence before exact full-family scoring."""
    if not config.enabled or len(population) <= 2 or len(layers) <= 4:
        scores = _cached_evaluate_population(
            population, layers, "full", cache, candidate_batch_size)
        return list(zip(population, scores)), {
            "screened": len(population), "refined": len(population), "validated": len(population)}

    screen_layers = _stratified_layers(layers, config.screen_fraction)
    refine_layers = _stratified_layers(layers, config.refine_fraction)
    screened = list(zip(
        population,
        _cached_evaluate_population(
            population, screen_layers, "screen", cache, candidate_batch_size),
    ))
    refined = _promote(
        screened,
        math.ceil(len(screened) * 0.50),
        config.diversity_slots,
        config.promotion_margin,
    )
    refined_candidates = [candidate for candidate, _ in refined]
    refined = list(zip(
        refined_candidates,
        _cached_evaluate_population(
            refined_candidates, refine_layers, "refine", cache, candidate_batch_size),
    ))
    validated = _promote(
        refined,
        max(2, math.ceil(len(refined) * 0.50)),
        config.diversity_slots,
        config.promotion_margin,
    )
    validated_candidates = [candidate for candidate, _ in validated]
    return list(zip(
        validated_candidates,
        _cached_evaluate_population(
            validated_candidates, layers, "full", cache, candidate_batch_size),
    )), {"screened": len(screened), "refined": len(refined), "validated": len(validated)}


def _reference_output(layer: Layer, split: str):
    attr = f"{split}_reference"
    cached = getattr(layer, attr)
    if cached is not None:
        return cached
    activations = layer.train_activations if split == "train" else layer.heldout_activations
    assert activations is not None
    if mx is not None:
        cached = mx.array(activations) @ mx.array(layer.weight).T
        mx.eval(cached)
    else:
        cached = activations @ layer.weight.T
    setattr(layer, attr, cached)
    return cached


def relative_output_error(
    activations: np.ndarray,
    weight: np.ndarray,
    reconstructed: np.ndarray,
    reference=None,
) -> float:
    if mx is not None:
        activation_mx = mx.array(activations)
        if reference is None:
            reference = activation_mx @ mx.array(weight).T
        approximate = activation_mx @ mx.array(reconstructed).T
        value = mx.mean(mx.square(approximate - reference)) / (
            mx.mean(mx.square(reference)) + 1e-12
        )
        mx.eval(value)
        return float(value.item())
    if reference is None:
        reference = activations @ weight.T
    approximate = activations @ reconstructed.T
    return float(
        np.mean(np.square(approximate - reference))
        / (np.mean(np.square(reference)) + 1e-12)
    )


def random_candidate(rng: random.Random, base_fraction: float) -> Candidate:
    return Candidate(
        rng.random(),
        rng.uniform(0.78, 1.0),
        math.exp(rng.uniform(math.log(max(0.0001, base_fraction / 4)), math.log(min(0.05, base_fraction * 4)))),
        rng.random(),
        rng.uniform(0.0, 1.0),
        # ternary_threshold: 0.3 (very dense) to 3.0 (very sparse). Bias the
        # initial population around 1.0 (legacy behaviour) by sampling in
        # [0.7, 1.6] most of the time, with 25% chance of an out-of-band
        # exploration in [0.4, 2.5].
        math.exp(rng.uniform(math.log(0.5), math.log(2.0))) if rng.random() < 0.75 else rng.uniform(0.4, 2.5),
    ).clipped()


def mutate(candidate: Candidate, rng: random.Random, sigma: float) -> Candidate:
    values = candidate.as_array()
    scales = np.asarray([
        0.20,                                # alpha
        0.08,                                # clip
        max(candidate.outlier_fraction, 0.001),  # outlier_fraction
        0.20,                                # moment_mix
        0.25,                                # tail_guard
        0.18,                                # ternary_threshold (multiplicative)
    ])
    noise = np.asarray([rng.gauss(0.0, sigma) for _ in values])
    return Candidate.from_array(values + noise * scales)


def crossover(left: Candidate, right: Candidate, rng: random.Random) -> Candidate:
    mask = np.asarray([rng.random() < 0.5 for _ in range(6)])
    blend = rng.random()
    values = np.where(mask, left.as_array(), right.as_array())
    values = blend * values + (1.0 - blend) * ((left.as_array() + right.as_array()) / 2.0)
    return Candidate.from_array(values)


def archive_cell(candidate: Candidate, score: Score) -> str:
    alpha_bin = min(4, int(candidate.alpha * 5))
    residual_bin = min(4, int(candidate.outlier_fraction / 0.05 * 5))
    tail_bin = min(3, int(candidate.tail_guard / 2.0 * 4))
    return f"{alpha_bin}:{residual_bin}:{tail_bin}"


def candidate_record(candidate: Candidate, score: Score) -> dict:
    return {"candidate": dataclasses.asdict(candidate), "score": dataclasses.asdict(score)}


def quality_cost(score: Score) -> float:
    return (
        score.train_error
        + 2.0 * score.heldout_error
        + 0.25 * score.worst_layer_error
        + 0.05 * score.tail_error
    )


def allocate_residual_budget(
    layers: list[Layer],
    candidates: dict[str, Candidate],
    fallback: Candidate,
    budget_fraction: float,
) -> tuple[dict[str, tuple[Candidate, Score]], dict]:
    fractions = sorted({
        float(np.clip(budget_fraction * multiplier, 0.0001, 0.05))
        for multiplier in (0.25, 0.5, 1.0, 2.0, 4.0)
    })
    curves: dict[str, list[tuple[Candidate, Score]]] = {}
    for layer in layers:
        base = candidates.get(layer.name, fallback)
        curves[layer.name] = []
        for fraction in fractions:
            candidate = dataclasses.replace(base, outlier_fraction=fraction)
            curves[layer.name].append((candidate, evaluate(candidate, [layer])))
    positions = {layer.name: 0 for layer in layers}
    budget = budget_fraction * len(layers)
    used = fractions[0] * len(layers)
    while True:
        best = None
        for layer in layers:
            position = positions[layer.name]
            curve = curves[layer.name]
            if position + 1 >= len(curve):
                continue
            current_candidate, current_score = curve[position]
            next_candidate, next_score = curve[position + 1]
            cost = next_candidate.outlier_fraction - current_candidate.outlier_fraction
            if used + cost > budget + 1e-12:
                continue
            gain = quality_cost(current_score) - quality_cost(next_score)
            value = gain / max(cost, 1e-12)
            if best is None or value > best[0]:
                best = (value, layer.name, cost)
        if best is None or best[0] <= 0.0:
            break
        _, name, cost = best
        positions[name] += 1
        used += cost
    allocated = {
        layer.name: curves[layer.name][positions[layer.name]]
        for layer in layers
    }
    return allocated, {
        "target_fraction": budget_fraction,
        "allocated_fraction": used / max(len(layers), 1),
        "curve_fractions": fractions,
        "layers": {
            name: {
                "outlier_fraction": candidate.outlier_fraction,
                "quality_cost": quality_cost(score),
            }
            for name, (candidate, score) in allocated.items()
        },
    }


def evolve(
    layers: list[Layer],
    generations: int,
    population_size: int,
    islands: int,
    seed: int,
    base_fraction: float,
    checkpoint: Path | None,
    progressive: ProgressiveConfig | None = None,
    candidate_batch_size: int = 4,
) -> tuple[Candidate, Score, dict]:
    progressive = progressive or ProgressiveConfig()
    bundle_digest = _layer_digest(layers)
    rng = random.Random(seed)
    populations = [
        [random_candidate(rng, base_fraction) for _ in range(population_size)]
        for _ in range(islands)
    ]
    archive: dict[str, dict] = {}
    history: list[dict] = []
    score_cache: dict[str, dict] = {}
    start_generation = 0
    if checkpoint is not None and checkpoint.exists():
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        start_generation = int(state["generation"]) + 1
        populations = [
            [Candidate(**entry) for entry in population]
            for population in state["populations"]
        ]
        archive = state.get("archive", {})
        history = state.get("history", [])
        if state.get("bundle_digest") == bundle_digest:
            score_cache = state.get("progressive_score_cache", {})
        rng.setstate(_list_to_tuple(state["rng_state"]))

    best_candidate = populations[0][0]
    best_score = _cached_evaluate(best_candidate, layers, "full", score_cache)
    if history:
        best_candidate = Candidate(**history[-1]["candidate"])
        best_score = _cached_evaluate(best_candidate, layers, "full", score_cache)
    for generation in range(start_generation, generations):
        next_populations: list[list[Candidate]] = []
        generation_best = None
        generation_work = {"screened": 0, "refined": 0, "validated": 0}
        for island_index, population in enumerate(populations):
            scored, work = progressive_evaluate_population(
                population, layers, progressive, score_cache, candidate_batch_size)
            for key, value in work.items():
                generation_work[key] += value
            scored.sort(key=lambda pair: pair[1].fitness)
            if generation_best is None or scored[0][1].fitness < generation_best[1].fitness:
                generation_best = scored[0]
            for candidate, score in scored:
                cell = archive_cell(candidate, score)
                previous = archive.get(cell)
                if previous is None or score.fitness < previous["score"]["fitness"]:
                    archive[cell] = candidate_record(candidate, score)
            elite_count = max(2, population_size // 4)
            elites = [candidate for candidate, _ in scored[:elite_count]]
            children = list(elites)
            while len(children) < population_size:
                left = rng.choice(elites)
                right_pool = populations[(island_index - 1) % islands] if generation % 5 == 4 else elites
                right = rng.choice(right_pool)
                # Generation-local decay keeps a resumed run bit-for-bit
                # equivalent even when its original stopping generation was
                # shorter than the resumed target.
                children.append(mutate(
                    crossover(left, right, rng),
                    rng,
                    max(0.15, 0.7 * math.pow(0.97, generation)),
                ))
            next_populations.append(children)
        populations = next_populations
        assert generation_best is not None
        if generation_best[1].fitness < best_score.fitness:
            best_candidate, best_score = generation_best
        history.append({"generation": generation, **candidate_record(best_candidate, best_score)})
        if checkpoint is not None:
            state = {
                "schema": SCHEMA,
                "generation": generation,
                "populations": [[dataclasses.asdict(candidate) for candidate in population] for population in populations],
                "archive": archive,
                "history": history,
                "rng_state": rng.getstate(),
                "bundle_digest": bundle_digest,
                "progressive_config": dataclasses.asdict(progressive),
                "progressive_score_cache": _checkpoint_score_cache(score_cache),
            }
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        print(
            f"generation={generation + 1}/{generations} fitness={best_score.fitness:.8g} "
            f"alpha={best_candidate.alpha:.4f} clip={best_candidate.clip:.4f} "
            f"residual={best_candidate.outlier_fraction:.6f} archive={len(archive)} "
            f"progressive={generation_work['screened']}/{generation_work['refined']}/{generation_work['validated']}",
            file=sys.stderr,
        )
    return best_candidate, best_score, {
        "archive": archive,
        "history": history,
        "progressive_score_cache": score_cache,
        "bundle_digest": bundle_digest,
    }


def _list_to_tuple(value):
    if isinstance(value, list):
        return tuple(_list_to_tuple(item) for item in value)
    return value


def policy_entry(matches: list[str], candidate: Candidate, model_role: str = DEFAULT_MODEL_ROLE) -> dict:
    if model_role not in MODEL_ROLES:
        raise ValueError(
            f"model_role {model_role!r} not in {MODEL_ROLES!r}"
        )
    return {
        "match": matches,
        "exact": False,
        "model_role": model_role,
        "awq_alpha": candidate.alpha,
        "awq_clip": candidate.clip,
        "outlier_fraction": candidate.outlier_fraction,
        "moment_mix": candidate.moment_mix,
        "tail_guard": candidate.tail_guard,
    }


def build_policy(results: dict[str, tuple[Candidate, Score]], provenance: dict, base: dict | None = None,
                 overrides: dict[str, tuple[Candidate, Score]] | None = None,
                 model_role: str = DEFAULT_MODEL_ROLE) -> dict:
    if model_role not in MODEL_ROLES:
        raise ValueError(
            f"model_role {model_role!r} not in {MODEL_ROLES!r}"
        )
    families = {}
    moe_layers: dict[str, dict] = {}
    for name, (candidate, _) in (overrides or {}).items():
        families[f"override:{name}"] = policy_entry([name], candidate, model_role=model_role)
        match = re.fullmatch(
            r"blk\.(\d+)\.(.+)\.expert-(\d+)",
            name,
        )
        if match:
            layer, tensor, expert = match.groups()
            moe_layers.setdefault(layer, {"experts": {}, "tensors": {}})
            moe_layers[layer]["tensors"].setdefault(tensor, {"experts": {}})
            moe_layers[layer]["tensors"][tensor]["experts"][expert] = {
                "outlier_fraction": candidate.outlier_fraction,
                "awq_alpha": candidate.alpha,
                "awq_clip": candidate.clip,
            }
    families.update(dict((base or {}).get("tensor_families", {})))
    families.update({
        "norm": {"match": ["norm"], "exact": True, "model_role": model_role,
                 "awq_alpha": 0.0, "outlier_fraction": 1.0},
    })
    for family, (candidate, _) in results.items():
        families[family] = policy_entry(MATCHES[family], candidate, model_role=model_role)
    policy = dict(base or {})
    policy.update({
        "schema": POLICY_SCHEMA,
        "search_schema": SCHEMA,
        "model_role": model_role,
        "tensor_families": families,
        "evolution": provenance,
    })
    policy.setdefault("draft_type", "hybrid")
    if moe_layers:
        existing = dict(policy.get("moe_residual_allocation", {}))
        existing.update({
            "schema": "llama.tessera.moe-residual-policy.v1",
            "model_role": model_role,
            "layers": moe_layers,
        })
        policy["moe_residual_allocation"] = existing
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Evolve AWQ and Tessera sparse-residual policies")
    parser.add_argument("--layers", required=True, help="Directory containing layer .npz bundles")
    parser.add_argument("--output", required=True, help="Calibration policy JSON")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--base-policy", default=None, help="Acceptance policy to preserve and refine")
    parser.add_argument("--generations", type=int, default=24)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--islands", type=int, default=4)
    parser.add_argument("--seed", type=int, default=640)
    parser.add_argument("--base-outlier-frac", type=float, default=0.005)
    parser.add_argument("--max-rows", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--override-quantile", type=float, default=0.85)
    parser.add_argument("--max-overrides-per-family", type=int, default=4)
    parser.add_argument(
        "--transfer-ledger", default=None,
        help="progressive observer ledger used to prioritize late-surviving tensors",
    )
    parser.add_argument(
        "--sensitive-generations", type=int, default=0,
        help="base generations for each sensitive tensor (0 uses family generations / 4)",
    )
    parser.add_argument(
        "--sensitive-population", type=int, default=0,
        help="base population for each sensitive tensor (0 uses family population / 2)",
    )
    parser.add_argument(
        "--saved-compute-reinvestment", type=float, default=2.0,
        help="multiplier applied to the observer budget saved by progressive freezing",
    )
    parser.add_argument(
        "--progressive-eval", action=argparse.BooleanOptionalAction, default=True,
        help="successively promote candidates from stratified evidence to full scoring",
    )
    parser.add_argument("--screen-fraction", type=float, default=0.25)
    parser.add_argument("--refine-fraction", type=float, default=0.50)
    parser.add_argument("--promotion-margin", type=float, default=0.05)
    parser.add_argument("--progressive-diversity-slots", type=int, default=2)
    parser.add_argument(
        "--candidate-batch-size", type=int, default=4,
        help="maximum candidate matrices per MLX activation-projection wave",
    )
    parser.add_argument(
        "--model-role",
        choices=MODEL_ROLES,
        default=DEFAULT_MODEL_ROLE,
        help=(
            "Component role for this AWQ evolution pass. The value is "
            "stamped on every per-family and per-override entry in the "
            "output policy so the unified consumer can route per-role. "
            "The default is 'trunk' which is the legacy single-model "
            "behaviour. The unified_calibrate.py driver invokes "
            "per_tensor_calibrate.py (which forwards --model-role to "
            "awq-evolve.py) once per component with the appropriate role."
        ),
    )
    args = parser.parse_args()
    if args.generations < 1 or args.population < 4 or args.islands < 1:
        raise ValueError("generations, population, and islands must be positive; population must be at least four")
    layer_paths = sorted(Path(args.layers).glob("*.npz"))
    if not layer_paths:
        raise ValueError(f"{args.layers}: no layer bundles")
    layers = [load_layer(path, args.max_rows, args.max_tokens) for path in layer_paths]
    if not 0.0 <= args.override_quantile <= 1.0:
        raise ValueError("override quantile must be in [0, 1]")
    if args.max_overrides_per_family < 0:
        raise ValueError("maximum overrides per family must be non-negative")
    if (args.sensitive_generations < 0 or args.sensitive_population < 0 or
            args.saved_compute_reinvestment < 0.0):
        raise ValueError("sensitive search budgets must be non-negative")
    if not 0.0 < args.screen_fraction <= args.refine_fraction <= 1.0:
        raise ValueError("screen and refine fractions must satisfy 0 < screen <= refine <= 1")
    if args.promotion_margin < 0.0 or args.progressive_diversity_slots < 0:
        raise ValueError("progressive promotion margin and diversity slots must be non-negative")
    if args.candidate_batch_size < 1:
        raise ValueError("candidate batch size must be positive")
    progressive = ProgressiveConfig(
        enabled=args.progressive_eval,
        screen_fraction=args.screen_fraction,
        refine_fraction=args.refine_fraction,
        promotion_margin=args.promotion_margin,
        diversity_slots=args.progressive_diversity_slots,
    )
    results: dict[str, tuple[Candidate, Score]] = {}
    overrides: dict[str, tuple[Candidate, Score]] = {}
    transfer_priorities, saved_observer_fraction = load_transfer_prior(
        Path(args.transfer_ledger) if args.transfer_ledger else None
    )
    reinvestment = 1.0 + (
        saved_observer_fraction * args.saved_compute_reinvestment
    )
    provenance = {
        "backend": "mlx-metal" if mx is not None else "numpy",
        "seed": args.seed,
        "generations": args.generations,
        "population": args.population,
        "islands": args.islands,
        "saved_observer_fraction": saved_observer_fraction,
        "sensitive_reinvestment_multiplier": reinvestment,
        "progressive_evaluation": dataclasses.asdict(progressive),
        "layer_digest": hashlib.sha256(
            "\n".join(f"{layer.name}:{layer.weight.shape}" for layer in layers).encode()
        ).hexdigest(),
        "families": {},
    }
    for family in FAMILIES:
        family_layers = [layer for layer in layers if layer.family == family]
        if not family_layers:
            continue
        checkpoint = Path(args.checkpoint).with_suffix(f".{family}.json") if args.checkpoint else None
        candidate, score, details = evolve(
            family_layers,
            args.generations,
            args.population,
            args.islands,
            args.seed + FAMILIES.index(family),
            args.base_outlier_frac,
            checkpoint,
            progressive,
            args.candidate_batch_size,
        )
        results[family] = (candidate, score)
        sensitivities = [
            (
                layer,
                evaluate(candidate, [layer]).fitness,
                transfer_priority(layer.name, transfer_priorities),
            )
            for layer in family_layers
        ]
        threshold = float(np.quantile(
            [
                sensitivity * (1.0 + priority)
                for _, sensitivity, priority in sensitivities
            ],
            args.override_quantile,
        ))
        sensitive_layers = []
        if len(family_layers) > 1:
            sensitive_layers = sorted(
                (
                    (layer, sensitivity, priority)
                    for layer, sensitivity, priority in sensitivities
                    if sensitivity * (1.0 + priority) >= threshold
                ),
                key=lambda item: item[1] * (1.0 + item[2]),
                reverse=True,
            )[:args.max_overrides_per_family]
        override_records = []
        local_candidates = {}
        for override_index, (layer, sensitivity, priority) in enumerate(sensitive_layers):
            override_checkpoint = None
            if checkpoint is not None:
                safe_name = "".join(
                    character if character.isalnum() else "_"
                    for character in layer.name
                )
                override_checkpoint = checkpoint.with_name(
                    f"{checkpoint.stem}.override.{safe_name}.json"
                )
            local_multiplier = 1.0 + (reinvestment - 1.0) * (
                0.5 + 0.5 * priority
            )
            local_generations = max(
                4,
                math.ceil(
                    (args.sensitive_generations or max(4, args.generations // 4))
                    * local_multiplier
                ),
            )
            local_population = max(
                4,
                math.ceil(
                    (args.sensitive_population or max(4, args.population // 2))
                    * math.sqrt(local_multiplier)
                ),
            )
            override_candidate, override_score, override_details = evolve(
                [layer],
                local_generations,
                local_population,
                min(4, args.islands),
                args.seed + 1000 + FAMILIES.index(family) * 100 + override_index,
                args.base_outlier_frac,
                override_checkpoint,
                progressive,
                args.candidate_batch_size,
            )
            local_candidates[layer.name] = override_candidate
            override_records.append({
                "layer": layer.name,
                "baseline_fitness": sensitivity,
                "observer_survival_priority": priority,
                "generations": local_generations,
                "population": local_population,
                **candidate_record(override_candidate, override_score),
                "archive_size": len(override_details["archive"]),
            })
        allocated, allocation_record = allocate_residual_budget(
            family_layers,
            local_candidates,
            candidate,
            args.base_outlier_frac,
        )
        family_fraction = float(np.median([
            allocated_candidate.outlier_fraction
            for allocated_candidate, _ in allocated.values()
        ]))
        candidate = dataclasses.replace(candidate, outlier_fraction=family_fraction)
        score = evaluate(candidate, family_layers)
        results[family] = (candidate, score)
        for layer in family_layers:
            allocated_candidate, allocated_score = allocated[layer.name]
            is_local = layer.name in local_candidates
            differs_from_family = not math.isclose(
                allocated_candidate.outlier_fraction,
                family_fraction,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            if is_local or differs_from_family or family == "routed_expert":
                overrides[layer.name] = (allocated_candidate, allocated_score)
        provenance["families"][family] = {
            **candidate_record(candidate, score),
            "archive_size": len(details["archive"]),
            "layers": [layer.name for layer in family_layers],
            "sensitivity_threshold": threshold,
            "overrides": override_records,
            "residual_allocation": allocation_record,
        }
    base_policy = None
    if args.base_policy:
        base_policy = json.loads(Path(args.base_policy).read_text(encoding="utf-8"))
        if base_policy.get("schema") != POLICY_SCHEMA:
            raise ValueError(f"{args.base_policy}: unsupported base policy schema")
    policy = build_policy(results, provenance, base_policy, overrides,
                          model_role=args.model_role)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {output} with {len(results)} evolved tensor families "
        f"(role={args.model_role})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
