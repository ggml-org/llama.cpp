#!/usr/bin/env python3
"""Differentiable MLX simulation of llama.cpp KV cache block quantization."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


SUPPORTED_TYPES = {
    "f16", "bf16", "q8_0", "q5_0", "q5_1", "q4_0", "q4_1", "iq4_nl"
}
IQ4_NL_VALUES = mx.array(
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
    dtype=mx.float32,
)


def hadamard(values: mx.array, block_size: int = 64) -> mx.array:
    if block_size == 0:
        return values
    if block_size < 2 or block_size & (block_size - 1):
        raise ValueError("Hadamard block size must be a power of two")
    if values.shape[-1] % block_size:
        raise ValueError("Hadamard block size must divide the final dimension")
    shape = values.shape
    blocks = values.reshape(*shape[:-1], shape[-1] // block_size, block_size)
    stride = 1
    while stride < block_size:
        blocks = blocks.reshape(*blocks.shape[:-1], -1, 2, stride)
        left = blocks[..., 0, :]
        right = blocks[..., 1, :]
        blocks = mx.stack((left + right, left - right), axis=-2).reshape(
            *shape[:-1], shape[-1] // block_size, block_size
        )
        stride *= 2
    return (blocks / block_size ** 0.5).reshape(shape)


def _blocks(values: mx.array) -> tuple[mx.array, tuple[int, ...]]:
    if values.shape[-1] % 32:
        raise ValueError("GGML KV block formats require a multiple of 32 columns")
    return values.reshape(*values.shape[:-1], -1, 32), values.shape


def _stored_f16(values: mx.array) -> mx.array:
    return values.astype(mx.float16).astype(mx.float32)


def _iq4_nl(values: mx.array) -> mx.array:
    blocks, shape = _blocks(values.astype(mx.float32))
    absolute = mx.abs(blocks)
    index = mx.argmax(absolute, axis=-1, keepdims=True)
    signed_max = mx.take_along_axis(blocks, index, axis=-1)
    initial_scale = signed_max / IQ4_NL_VALUES[0]
    inverse = mx.where(initial_scale != 0, 1.0 / initial_scale, 0.0)
    normalized = blocks * inverse
    midpoints = (IQ4_NL_VALUES[:-1] + IQ4_NL_VALUES[1:]) * 0.5
    codes = mx.sum(normalized[..., None] >= midpoints, axis=-1)
    quantized = IQ4_NL_VALUES[codes]
    weights = mx.square(blocks)
    sum_qx = mx.sum(weights * quantized * blocks, axis=-1, keepdims=True)
    sum_q2 = mx.sum(weights * mx.square(quantized), axis=-1, keepdims=True)
    refined_scale = mx.where(sum_q2 > 0, sum_qx / sum_q2, 0.0)
    return (quantized * _stored_f16(refined_scale)).reshape(shape)


def quantize_dequantize(values: mx.array, cache_type: str) -> mx.array:
    if cache_type not in SUPPORTED_TYPES:
        raise ValueError(f"unsupported KV cache type: {cache_type}")
    if cache_type == "f16":
        return values.astype(mx.float16).astype(mx.float32)
    if cache_type == "bf16":
        return values.astype(mx.bfloat16).astype(mx.float32)
    if cache_type == "iq4_nl":
        return _iq4_nl(values)
    blocks, shape = _blocks(values.astype(mx.float32))
    if cache_type == "q8_0":
        scale = mx.max(mx.abs(blocks), axis=-1, keepdims=True) / 127.0
        inverse = mx.where(scale != 0, 1.0 / scale, 0.0)
        quantized = mx.round(blocks * inverse)
        return (quantized * _stored_f16(scale)).reshape(shape)
    if cache_type in {"q4_0", "q5_0"}:
        levels = 16 if cache_type == "q4_0" else 32
        absolute = mx.abs(blocks)
        index = mx.argmax(absolute, axis=-1, keepdims=True)
        signed_max = mx.take_along_axis(blocks, index, axis=-1)
        scale = signed_max / -(levels // 2)
        inverse = mx.where(scale != 0, 1.0 / scale, 0.0)
        quantized = mx.minimum(
            levels - 1,
            mx.floor(blocks * inverse + levels / 2 + 0.5),
        )
        return ((quantized - levels / 2) * _stored_f16(scale)).reshape(shape)
    levels = 16 if cache_type == "q4_1" else 32
    minimum = mx.min(blocks, axis=-1, keepdims=True)
    maximum = mx.max(blocks, axis=-1, keepdims=True)
    scale = (maximum - minimum) / (levels - 1)
    inverse = mx.where(scale != 0, 1.0 / scale, 0.0)
    quantized = mx.floor((blocks - minimum) * inverse + 0.5)
    return (
        quantized * _stored_f16(scale) + _stored_f16(minimum)
    ).reshape(shape)


def straight_through_cache(
    values: mx.array,
    cache_type: str,
    rotation: int = 64,
) -> mx.array:
    rotated = hadamard(values, rotation)
    compressed = quantize_dequantize(rotated, cache_type)
    return rotated + mx.stop_gradient(compressed - rotated)


@dataclass(frozen=True)
class CachePair:
    key_type: str
    value_type: str
    rotation: int = 64


def simulate_pair(
    keys: mx.array,
    values: mx.array,
    policy: CachePair,
) -> tuple[mx.array, mx.array]:
    return (
        straight_through_cache(keys, policy.key_type, policy.rotation),
        straight_through_cache(values, policy.value_type, policy.rotation),
    )
