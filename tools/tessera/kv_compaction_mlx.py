#!/usr/bin/env python3
"""Differentiable KV compaction state and plans for Tessera training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np


@dataclass(frozen=True)
class KVCompactionState:
    keys: mx.array
    values: mx.array
    positions: np.ndarray
    sequence_ids: np.ndarray
    valid: np.ndarray
    pending_shifts: np.ndarray

    def __post_init__(self):
        slots = self.keys.shape[0]
        if self.values.shape[0] != slots:
            raise ValueError("K and V slot counts differ")
        for metadata in (
            self.positions,
            self.sequence_ids,
            self.valid,
            self.pending_shifts,
        ):
            if metadata.shape != (slots,):
                raise ValueError("compaction metadata must have one value per slot")


@dataclass(frozen=True)
class KVCompactionPlan:
    source_to_destination: np.ndarray
    source_weights: np.ndarray
    rope_deltas: np.ndarray
    destination_positions: np.ndarray
    destination_sequence_ids: np.ndarray
    destination_valid: np.ndarray
    strategy: str


@dataclass(frozen=True)
class KVCompactionTransition:
    epoch: int
    operation: str
    valid_before: int
    valid_after: int
    details: dict[str, Any]


def remove(
    state: KVCompactionState,
    sequence_id: int,
    position_start: int = 0,
    position_end: int | None = None,
) -> KVCompactionState:
    end = np.iinfo(np.int32).max if position_end is None else position_end
    selected = (
        state.valid
        & (state.positions >= max(position_start, 0))
        & (state.positions < end)
    )
    if sequence_id >= 0:
        selected &= state.sequence_ids == sequence_id
    return KVCompactionState(
        state.keys,
        state.values,
        state.positions.copy(),
        state.sequence_ids.copy(),
        state.valid & ~selected,
        state.pending_shifts.copy(),
    )


def shift(
    state: KVCompactionState,
    sequence_id: int,
    position_start: int,
    position_end: int | None,
    delta: int,
) -> KVCompactionState:
    if delta == 0:
        return state
    end = np.iinfo(np.int32).max if position_end is None else position_end
    selected = (
        state.valid
        & (state.sequence_ids == sequence_id)
        & (state.positions >= max(position_start, 0))
        & (state.positions < end)
    )
    positions = state.positions.copy()
    shifts = state.pending_shifts.copy()
    positions[selected] += delta
    shifts[selected] += delta
    valid = state.valid & (positions >= 0)
    return KVCompactionState(
        state.keys,
        state.values,
        positions,
        state.sequence_ids.copy(),
        valid,
        shifts,
    )


def divide_positions(
    state: KVCompactionState,
    sequence_id: int,
    position_start: int,
    position_end: int | None,
    divisor: int,
) -> KVCompactionState:
    if divisor <= 0:
        raise ValueError("position divisor must be positive")
    end = np.iinfo(np.int32).max if position_end is None else position_end
    selected = (
        state.valid
        & (state.sequence_ids == sequence_id)
        & (state.positions >= max(position_start, 0))
        & (state.positions < end)
    )
    positions = state.positions.copy()
    shifts = state.pending_shifts.copy()
    divided = positions[selected] // divisor
    shifts[selected] += divided - positions[selected]
    positions[selected] = divided
    return KVCompactionState(
        state.keys,
        state.values,
        positions,
        state.sequence_ids.copy(),
        state.valid.copy(),
        shifts,
    )


def defragment(state: KVCompactionState) -> KVCompactionState:
    slots = state.valid.size
    order = np.argsort(
        np.where(state.valid, np.arange(slots), slots + np.arange(slots)),
        kind="stable",
    )
    return KVCompactionState(
        mx.take(state.keys, mx.array(order), axis=0),
        mx.take(state.values, mx.array(order), axis=0),
        state.positions[order],
        state.sequence_ids[order],
        state.valid[order],
        state.pending_shifts[order],
    )


def _partition(indices: np.ndarray, count: int) -> list[np.ndarray]:
    if count <= 0:
        return []
    return [group for group in np.array_split(indices, count) if group.size]


def make_plan(
    state: KVCompactionState,
    target_per_sequence: int,
    keep_prefix: int,
    keep_recent: int,
    strategy: str,
    importance: np.ndarray | None = None,
) -> KVCompactionPlan:
    if target_per_sequence <= 0:
        raise ValueError("target slots per sequence must be positive")
    if keep_prefix < 0 or keep_recent < 0:
        raise ValueError("protected slot counts cannot be negative")
    if strategy not in {"evict", "merge"}:
        raise ValueError("compaction strategy must be evict or merge")
    slots = state.valid.size
    if importance is None:
        importance = np.ones(slots, dtype=np.float32)
    if importance.shape != (slots,) or np.any(importance < 0):
        raise ValueError("importance must be a nonnegative value per slot")
    groups: list[tuple[int, np.ndarray]] = []
    for sequence_id in sorted(set(state.sequence_ids[state.valid].tolist())):
        indices = np.flatnonzero(state.valid & (state.sequence_ids == sequence_id))
        indices = indices[np.argsort(state.positions[indices], kind="stable")]
        if indices.size <= target_per_sequence:
            groups.extend((sequence_id, np.array([index])) for index in indices)
            continue
        prefix_count = min(keep_prefix, target_per_sequence, indices.size)
        recent_count = min(
            keep_recent,
            target_per_sequence - prefix_count,
            indices.size - prefix_count,
        )
        prefix = indices[:prefix_count]
        recent = indices[indices.size - recent_count:] if recent_count else np.empty(0, int)
        middle = indices[prefix_count:indices.size - recent_count if recent_count else None]
        middle_budget = target_per_sequence - prefix_count - recent_count
        groups.extend((sequence_id, np.array([index])) for index in prefix)
        if strategy == "evict":
            selected = middle[-middle_budget:] if middle_budget else np.empty(0, int)
            groups.extend((sequence_id, np.array([index])) for index in selected)
        else:
            groups.extend(
                (sequence_id, group)
                for group in _partition(middle, middle_budget)
            )
        groups.extend((sequence_id, np.array([index])) for index in recent)
    if len(groups) > slots:
        raise ValueError("compaction plan exceeds physical cache slots")
    source_to_destination = np.full(slots, -1, dtype=np.int32)
    source_weights = np.zeros(slots, dtype=np.float32)
    rope_deltas = np.zeros(slots, dtype=np.int32)
    destination_positions = np.full(slots, -1, dtype=np.int32)
    destination_sequence_ids = np.full(slots, -1, dtype=np.int32)
    destination_valid = np.zeros(slots, dtype=bool)
    for destination, (sequence_id, source_indices) in enumerate(groups):
        destination_valid[destination] = True
        destination_sequence_ids[destination] = sequence_id
        destination_position = int(np.max(state.positions[source_indices]))
        destination_positions[destination] = destination_position
        weights = importance[source_indices].astype(np.float64)
        if not np.any(weights):
            weights.fill(1)
        weights /= weights.sum()
        source_to_destination[source_indices] = destination
        source_weights[source_indices] = weights.astype(np.float32)
        rope_deltas[source_indices] = destination_position - state.positions[source_indices]
    return KVCompactionPlan(
        source_to_destination,
        source_weights,
        rope_deltas,
        destination_positions,
        destination_sequence_ids,
        destination_valid,
        strategy,
    )


def rope_shift(
    keys: mx.array,
    deltas: np.ndarray,
    n_rot: int,
    frequency_base: float,
    frequency_scale: float = 1.0,
    rope_type: str = "neox",
) -> mx.array:
    if n_rot <= 0 or n_rot > keys.shape[-1] or n_rot % 2:
        raise ValueError("RoPE dimensions must be positive, even, and fit the key")
    if deltas.shape != (keys.shape[0],):
        raise ValueError("one RoPE delta is required per key slot")
    half = n_rot // 2
    frequency = frequency_scale * mx.power(
        frequency_base,
        -mx.arange(half, dtype=mx.float32) / half,
    )
    angles = mx.array(deltas.astype(np.float32))[:, None] * frequency[None, :]
    while angles.ndim < keys.ndim:
        angles = mx.expand_dims(angles, axis=1)
    cosine = mx.cos(angles)
    sine = mx.sin(angles)
    prefix = keys[..., :n_rot]
    suffix = keys[..., n_rot:]
    if rope_type == "neox":
        left = prefix[..., :half]
        right = prefix[..., half:]
        rotated = mx.concatenate(
            (left * cosine - right * sine, left * sine + right * cosine),
            axis=-1,
        )
    elif rope_type == "normal":
        left = prefix[..., 0::2]
        right = prefix[..., 1::2]
        paired = mx.stack(
            (left * cosine - right * sine, left * sine + right * cosine),
            axis=-1,
        )
        rotated = paired.reshape(prefix.shape)
    else:
        raise ValueError("unsupported RoPE type")
    return mx.concatenate((rotated, suffix), axis=-1)


def apply_plan(
    state: KVCompactionState,
    plan: KVCompactionPlan,
    n_rot: int,
    frequency_base: float,
    frequency_scale: float = 1.0,
    rope_type: str = "neox",
) -> KVCompactionState:
    slots = state.valid.size
    shifted_keys = rope_shift(
        state.keys,
        plan.rope_deltas + state.pending_shifts,
        n_rot,
        frequency_base,
        frequency_scale,
        rope_type,
    )
    matrix = np.zeros((slots, slots), dtype=np.float32)
    selected = np.flatnonzero(plan.source_to_destination >= 0)
    matrix[plan.source_to_destination[selected], selected] = plan.source_weights[selected]
    mixing = mx.array(matrix)
    keys = mx.einsum("ds,s...->d...", mixing, shifted_keys)
    values = mx.einsum("ds,s...->d...", mixing, state.values)
    return KVCompactionState(
        keys,
        values,
        plan.destination_positions.copy(),
        plan.destination_sequence_ids.copy(),
        plan.destination_valid.copy(),
        np.zeros(slots, dtype=np.int32),
    )


class KVCompactionMachine:
    """Stateful training facade over llama.cpp-compatible KV transitions."""

    def __init__(
        self,
        state: KVCompactionState,
        n_rot: int,
        frequency_base: float,
        frequency_scale: float = 1.0,
        rope_type: str = "neox",
        history_limit: int = 256,
    ):
        if history_limit < 0:
            raise ValueError("history limit cannot be negative")
        self.state = state
        self.n_rot = n_rot
        self.frequency_base = frequency_base
        self.frequency_scale = frequency_scale
        self.rope_type = rope_type
        self.history_limit = history_limit
        self.epoch = 0
        self.history: list[KVCompactionTransition] = []

    def _commit(
        self,
        operation: str,
        state: KVCompactionState,
        details: dict[str, Any],
    ) -> KVCompactionState:
        before = int(np.count_nonzero(self.state.valid))
        after = int(np.count_nonzero(state.valid))
        self.epoch += 1
        transition = KVCompactionTransition(
            self.epoch,
            operation,
            before,
            after,
            details,
        )
        self.state = state
        if self.history_limit:
            self.history.append(transition)
            del self.history[:-self.history_limit]
        return state

    def seq_rm(
        self,
        sequence_id: int,
        position_start: int = 0,
        position_end: int | None = None,
    ) -> KVCompactionState:
        return self._commit(
            "seq_rm",
            remove(self.state, sequence_id, position_start, position_end),
            {
                "sequence_id": sequence_id,
                "position_start": position_start,
                "position_end": position_end,
            },
        )

    def seq_add(
        self,
        sequence_id: int,
        position_start: int,
        position_end: int | None,
        delta: int,
    ) -> KVCompactionState:
        return self._commit(
            "seq_add",
            shift(
                self.state,
                sequence_id,
                position_start,
                position_end,
                delta,
            ),
            {
                "sequence_id": sequence_id,
                "position_start": position_start,
                "position_end": position_end,
                "delta": delta,
            },
        )

    def seq_div(
        self,
        sequence_id: int,
        position_start: int,
        position_end: int | None,
        divisor: int,
    ) -> KVCompactionState:
        return self._commit(
            "seq_div",
            divide_positions(
                self.state,
                sequence_id,
                position_start,
                position_end,
                divisor,
            ),
            {
                "sequence_id": sequence_id,
                "position_start": position_start,
                "position_end": position_end,
                "divisor": divisor,
            },
        )

    def defrag(self) -> KVCompactionState:
        return self._commit("defrag", defragment(self.state), {})

    def compact(
        self,
        target_per_sequence: int,
        keep_prefix: int,
        keep_recent: int,
        strategy: str,
        importance: np.ndarray | None = None,
    ) -> tuple[KVCompactionState, KVCompactionPlan]:
        plan = make_plan(
            self.state,
            target_per_sequence,
            keep_prefix,
            keep_recent,
            strategy,
            importance,
        )
        state = apply_plan(
            self.state,
            plan,
            self.n_rot,
            self.frequency_base,
            self.frequency_scale,
            self.rope_type,
        )
        merged_sources = int(
            np.count_nonzero(plan.source_to_destination >= 0)
            - np.count_nonzero(plan.destination_valid)
        )
        self._commit(
            "compact",
            state,
            {
                "strategy": strategy,
                "target_per_sequence": target_per_sequence,
                "keep_prefix": keep_prefix,
                "keep_recent": keep_recent,
                "merged_sources": max(merged_sources, 0),
            },
        )
        return state, plan
