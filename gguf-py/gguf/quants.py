from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Sequence

from numpy.typing import DTypeLike

from .constants import GGML_QUANT_SIZES, GGMLQuantizationType
from .lazy import LazyNumpyTensor

import numpy as np


def quant_shape_to_byte_shape(shape: Sequence[int], quant_type: GGMLQuantizationType) -> tuple[int, ...]:
    block_size, type_size = GGML_QUANT_SIZES[quant_type]
    if shape[-1] % block_size != 0:
        raise ValueError(f"Quantized tensor row size ({shape[-1]}) is not a multiple of {quant_type.name} block size ({block_size})")
    return (*shape[:-1], shape[-1] // block_size * type_size)


def quant_shape_from_byte_shape(shape: Sequence[int], quant_type: GGMLQuantizationType) -> tuple[int, ...]:
    block_size, type_size = GGML_QUANT_SIZES[quant_type]
    if shape[-1] % type_size != 0:
        raise ValueError(f"Quantized tensor bytes per row ({shape[-1]}) is not a multiple of {quant_type.name} type size ({type_size})")
    return (*shape[:-1], shape[-1] // type_size * block_size)


# This is faster than np.vectorize and np.apply_along_axis because it works on more than one row at a time
def _apply_over_grouped_rows(func: Callable[[np.ndarray], np.ndarray], arr: np.ndarray, otype: DTypeLike, oshape: tuple[int, ...]) -> np.ndarray:
    rows = arr.reshape((-1, arr.shape[-1]))
    osize = 1
    for dim in oshape:
        osize *= dim
    out = np.empty(shape=osize, dtype=otype)
    # compute over groups of 16 rows (arbitrary, but seems good for performance)
    n_groups = (rows.shape[0] // 16) or 1
    np.concatenate([func(group).ravel() for group in np.array_split(rows, n_groups)], axis=0, out=out)
    return out.reshape(oshape)


# round away from zero
# ref: https://stackoverflow.com/a/59143326/22827863
def np_roundf(n: np.ndarray) -> np.ndarray:
    a = abs(n)
    floored = np.floor(a)
    b = floored + np.floor(2 * (a - floored))
    return np.sign(n) * b


class QuantError(Exception): ...


_type_traits: dict[GGMLQuantizationType, type[__Quant]] = {}


def quantize(data: np.ndarray, qtype: GGMLQuantizationType) -> np.ndarray:
    if qtype == GGMLQuantizationType.F32:
        return data.astype(np.float32, copy=False)
    elif qtype == GGMLQuantizationType.F16:
        return data.astype(np.float16, copy=False)
    elif (q := _type_traits.get(qtype)) is not None:
        return q.quantize(data)
    else:
        raise NotImplementedError(f"Quantization for {qtype.name} is not yet implemented")


def dequantize(data: np.ndarray, qtype: GGMLQuantizationType) -> np.ndarray:
    if qtype == GGMLQuantizationType.F32 or qtype == GGMLQuantizationType.F16:
        return data.astype(np.float32, copy=False)
    elif (q := _type_traits.get(qtype)) is not None:
        return q.dequantize(data)
    else:
        raise NotImplementedError(f"Dequantization for {qtype.name} is not yet implemented")


class __Quant(ABC):
    qtype: GGMLQuantizationType
    block_size: int
    type_size: int

    def __init__(self):
        return TypeError("Quant conversion classes can't have instances")

    def __init_subclass__(cls, qtype: GGMLQuantizationType) -> None:
        cls.qtype = qtype
        cls.block_size, cls.type_size = GGML_QUANT_SIZES[qtype]
        cls.__quantize_lazy = LazyNumpyTensor._wrap_fn(
            cls.__quantize_array,
            meta_noop=(np.uint8, cls.__shape_to_bytes)
        )
        cls.__dequantize_lazy = LazyNumpyTensor._wrap_fn(
            cls.__dequantize_array,
            meta_noop=(np.float32, cls.__shape_from_bytes)
        )
        assert qtype not in _type_traits
        _type_traits[qtype] = cls

    @classmethod
    @abstractmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def quantize_rows(cls, rows: np.ndarray) -> np.ndarray:
        rows = rows.astype(np.float32, copy=False)
        shape = rows.shape
        n_blocks = rows.size // cls.block_size
        blocks = rows.reshape((n_blocks, cls.block_size))
        blocks = cls.quantize_blocks(blocks)
        assert blocks.dtype == np.uint8
        assert blocks.shape[-1] == cls.type_size
        return blocks.reshape(cls.__shape_to_bytes(shape))

    @classmethod
    def dequantize_rows(cls, rows: np.ndarray) -> np.ndarray:
        rows = rows.view(np.uint8)
        shape = rows.shape
        n_blocks = rows.size // cls.type_size
        blocks = rows.reshape((n_blocks, cls.type_size))
        blocks = cls.dequantize_blocks(blocks)
        assert blocks.dtype == np.float32
        assert blocks.shape[-1] == cls.block_size
        return blocks.reshape(cls.__shape_from_bytes(shape))

    @classmethod
    def __shape_to_bytes(cls, shape: Sequence[int]):
        return quant_shape_to_byte_shape(shape, cls.qtype)

    @classmethod
    def __shape_from_bytes(cls, shape: Sequence[int]):
        return quant_shape_from_byte_shape(shape, cls.qtype)

    @classmethod
    def __quantize_array(cls, array: np.ndarray) -> np.ndarray:
        return _apply_over_grouped_rows(cls.quantize_rows, arr=array, otype=np.uint8, oshape=cls.__shape_to_bytes(array.shape))

    @classmethod
    def __dequantize_array(cls, array: np.ndarray) -> np.ndarray:
        return _apply_over_grouped_rows(cls.dequantize_rows, arr=array, otype=np.float32, oshape=cls.__shape_from_bytes(array.shape))

    @classmethod
    def __quantize_lazy(cls, lazy_tensor: LazyNumpyTensor, /) -> Any:
        pass

    @classmethod
    def __dequantize_lazy(cls, lazy_tensor: LazyNumpyTensor, /) -> Any:
        pass

    @classmethod
    def can_quantize(cls, tensor: np.ndarray | LazyNumpyTensor) -> bool:
        return tensor.shape[-1] % cls.block_size == 0

    @classmethod
    def quantize(cls, tensor: np.ndarray | LazyNumpyTensor) -> np.ndarray:
        if not cls.can_quantize(tensor):
            raise QuantError(f"Can't quantize tensor with shape {tensor.shape} to {cls.qtype.name}")
        if isinstance(tensor, LazyNumpyTensor):
            return cls.__quantize_lazy(tensor)
        else:
            return cls.__quantize_array(tensor)

    @classmethod
    def dequantize(cls, tensor: np.ndarray | LazyNumpyTensor) -> np.ndarray:
        if isinstance(tensor, LazyNumpyTensor):
            return cls.__dequantize_lazy(tensor)
        else:
            return cls.__dequantize_array(tensor)


class BF16(__Quant, qtype=GGMLQuantizationType.BF16):
    @classmethod
    # same as ggml_compute_fp32_to_bf16 in ggml-impl.h
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n = blocks.view(np.uint32)
        # force nan to quiet
        n = np.where((n & 0x7fffffff) > 0x7f800000, (n & np.uint32(0xffff0000)) | np.uint32(64 << 16), n)
        # round to nearest even
        n = (np.uint64(n) + (0x7fff + ((n >> 16) & 1))) >> 16
        return n.astype(np.uint16).view(np.uint8)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        return (blocks.view(np.int16).astype(np.int32) << 16).view(np.float32)


class Q8_0(__Quant, qtype=GGMLQuantizationType.Q8_0):
    @classmethod
    # Implementation of Q8_0 with bit-exact same results as reference implementation in ggml-quants.c
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:

        d = abs(blocks).max(axis=1, keepdims=True) / 127
        with np.errstate(divide="ignore"):
            id = np.where(d == 0, 0, 1 / d)
        qs = np_roundf(blocks * id)

        # (n_blocks, 2)
        d = d.astype(np.float16).view(np.uint8)
        # (n_blocks, block_size)
        qs = qs.astype(np.int8).view(np.uint8)

        return np.concatenate([d, qs], axis=1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        d, x = np.split(blocks, [2], axis=1)
        d = d.view(np.float16).astype(np.float32)
        x = x.view(np.int8).astype(np.float32)

        return (x * d)
