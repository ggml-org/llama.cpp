#!/usr/bin/env python3

# Test gguf.quants so that it exactly matches the C implementation of the (de)quantization

# NOTE: this is kind of a mess, but at least it worked for initially testing the Python implementations.

from __future__ import annotations

import argparse
from math import prod
import os
import sys
from pathlib import Path
import ctypes
import logging
import numpy as np

# Necessary to load the local gguf package
if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent.parent / 'gguf-py').exists():
    sys.path.insert(0, str(Path(__file__).parent.parent))

import gguf
from gguf.constants import GGMLQuantizationType


logger = logging.getLogger("test-quants")


c_float_p = ctypes.POINTER(ctypes.c_float)


class ggml_init_params(ctypes.Structure):
    _fields_ = [
        ("mem_size", ctypes.c_size_t),
        ("mem_buffer", ctypes.c_void_p),
        ("no_alloc", ctypes.c_bool),
    ]


class GGMLQuants:
    libggml: ctypes.CDLL

    def __init__(self, libggml: Path):
        self.libggml = ctypes.CDLL(str(libggml))
        self.libggml.ggml_quantize_chunk.restype = ctypes.c_size_t
        # enum ggml_type   type,
        #    const float * src,
        #           void * dst,
        #        int64_t   start,
        #        int64_t   nrows,
        #        int64_t   n_per_row,
        #    const float * imatrix) {
        self.libggml.ggml_quantize_chunk.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_float),
        )

        self.libggml.ggml_quantize_requires_imatrix.restype = ctypes.c_bool
        self.libggml.ggml_quantize_requires_imatrix.argtypes = (ctypes.c_int,)

        for t in (
            "q4_0", "q4_1", "q5_0", "q5_1", "q8_0",
            "q2_K", "q3_K", "q4_K", "q5_K", "q6_K",
            "tq1_0", "tq2_0",
            "iq2_xxs", "iq2_xs", "iq2_s", "iq3_xxs", "iq3_s", "iq1_s", "iq1_m",
            "iq4_nl", "iq4_xs",
        ):
            dequant_func: ctypes._NamedFuncPointer = getattr(self.libggml, "dequantize_row_" + t)
            dequant_func.restype = None
            dequant_func.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64)

        self.libggml.ggml_fp16_to_fp32_row.restype = None
        self.libggml.ggml_fp16_to_fp32_row.argtypes = (ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_float), ctypes.c_int64)
        self.libggml.ggml_bf16_to_fp32_row.restype = None
        self.libggml.ggml_bf16_to_fp32_row.argtypes = (ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_float), ctypes.c_int64)

        self.libggml.ggml_init.argtypes = (ggml_init_params,)

        self.libggml.ggml_init(ggml_init_params(1 * 1024 * 1024, 0, False))

    def dequantize(self, tensor: np.ndarray, qtype: GGMLQuantizationType) -> np.ndarray:
        result = np.zeros(gguf.quant_shape_from_byte_shape(tensor.shape, qtype), dtype=np.float32, order="C")
        if qtype == GGMLQuantizationType.F32:
            # no-op
            result = tensor.view(np.float32)
        elif qtype == GGMLQuantizationType.F16:
            self.libggml.ggml_fp16_to_fp32_row(tensor.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)), result.ctypes.data_as(c_float_p), result.size)
        elif qtype == GGMLQuantizationType.BF16:
            self.libggml.ggml_bf16_to_fp32_row(tensor.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)), result.ctypes.data_as(c_float_p), result.size)
        else:
            lw_qname = qtype.name.lower()
            if lw_qname[-1] == "k":
                lw_qname = lw_qname[:-1] + "K"
            dequant_func: ctypes._NamedFuncPointer = getattr(self.libggml, "dequantize_row_" + lw_qname)
            dequant_func(tensor.ctypes.data_as(ctypes.c_void_p), result.ctypes.data_as(c_float_p), result.size)
        return result

    def quantize(self, data: np.ndarray, qtype: GGMLQuantizationType) -> np.ndarray:
        result = np.zeros(gguf.quant_shape_to_byte_shape(data.shape, qtype), dtype=np.uint8, order="C")
        if self.libggml.ggml_quantize_requires_imatrix(qtype.value):
            # TODO: is a column-wise sum of squares appropriate?
            qw = np.sum((data * data).reshape((-1, data.shape[-1])), axis=0).ctypes.data_as(c_float_p)
        else:
            qw = ctypes.cast(0, c_float_p)
        result_size = self.libggml.ggml_quantize_chunk(qtype.value, data.ctypes.data_as(c_float_p), result.ctypes.data_as(ctypes.c_void_p), 0, prod(data.shape[:-1]), data.shape[-1], qw)
        assert result.size == result_size
        return result


def compare_tensors(t1: np.ndarray, t2: np.ndarray, qtype: GGMLQuantizationType) -> bool:
    same = np.array_equal(t1, t2)
    if same:
        return True
    else:
        block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
        if t1.dtype == np.float32:
            t1 = t1.reshape((-1, block_size))
            t2 = t2.reshape((-1, block_size))
        else:
            t1 = t1.reshape((-1, type_size))
            t2 = t2.reshape((-1, type_size))
        x = t1.view(np.uint8) ^ t2.view(np.uint8)
        diff_bits = np.count_nonzero(np.unpackbits(x, axis=-1), axis=-1)
        num_bad_blocks = np.count_nonzero(diff_bits, axis=0)
        if num_bad_blocks == 0 and t1.shape == t2.shape:
            logger.debug("Bits are equal, but arrays don't match, likely contains NANs")
            return True
        logger.debug(f"{num_bad_blocks} bad blocks ({100 * num_bad_blocks / x.shape[0]:.6f}%)")
        bad_block_id = np.argmax(diff_bits, axis=0)
        logger.debug(f"Worst block id: {bad_block_id}")
        logger.debug(f"Sample bad block ({diff_bits[bad_block_id]} differing bits):\n{t1[bad_block_id]}\nReference:\n{t2[bad_block_id]}")

        sum_diff_bits = np.sum(diff_bits)
        logger.debug(f"{sum_diff_bits} bits differ ({100 * sum_diff_bits / (x.size * 8):.6f}%)")
        return False


def do_test(libggml_path: Path, quick: bool = False):
    ggml_quants = GGMLQuants(libggml_path)

    np.set_printoptions(precision=None, threshold=(4 * 256) + 1, formatter={"int": lambda n: "0x%02X" % n})

    r = np.random.randn(8, 1024, 1024).astype(np.float32, copy=False)

    for qtype in (GGMLQuantizationType.F16, *gguf.quants._type_traits.keys()):
        has_dequantize = False
        has_quantize = False

        try:
            gguf.dequantize(np.zeros((gguf.GGML_QUANT_SIZES[qtype][1]), dtype=np.uint8), qtype)
            has_dequantize = True
        except (NotImplementedError, AssertionError) as e:
            if isinstance(e, AssertionError):
                logger.error(f"Error with {qtype.name}: {e}")
                raise e
        try:
            gguf.quantize(np.zeros((gguf.GGML_QUANT_SIZES[qtype][0]), dtype=np.float32), qtype)
            has_quantize = True
        except (NotImplementedError, AssertionError) as e:
            if isinstance(e, AssertionError):
                logger.error(f"Error with {qtype.name}: {e}")
                raise e

        if not has_dequantize and not has_quantize:
            continue

        logger.info(f"Testing {qtype.name}")

        rc = r.copy(order="C")

        pyq = None
        ggq = None

        if has_quantize:
            logger.debug(f"Quantizing to {qtype.name} with Python")
            pyq = gguf.quants.quantize(rc, qtype)

            logger.debug(f"Quantizing to {qtype.name} with C")
            ggq = ggml_quants.quantize(rc, qtype)

            if qtype == GGMLQuantizationType.F16:
                pyq = pyq.view(np.uint8)
            quant_equal = compare_tensors(pyq, ggq, qtype)

            if not quant_equal:
                logger.error(f"Quantization to {qtype.name} does not match ❌")
            else:
                logger.info(f"Quantization to {qtype.name} matches exactly ✅")

        if has_dequantize:
            if ggq is None and not quick:
                logger.debug(f"Quantizing to {qtype.name} with C")
                ggq = ggml_quants.quantize(rc, qtype)

            if ggq is not None:
                logger.debug(f"Dequantizing from {qtype.name} with Python")
                pydq = gguf.quants.dequantize(ggq, qtype)
                logger.debug(f"Dequantizing from {qtype.name} with C")
                ggdq = ggml_quants.dequantize(ggq, qtype)

                dequant_equal = compare_tensors(pydq, ggdq, qtype)

                if not dequant_equal:
                    logger.error(f"Dequantization from {qtype.name} does not match ❌")
                else:
                    logger.info(f"Dequantization from {qtype.name} matches exactly ✅")

            rq_shape = gguf.quants.quant_shape_to_byte_shape((8, 1024, 1024 // 2), qtype)
            rq = np.random.random(rq_shape).astype(np.float16).view(np.uint8)

            logger.debug(f"Dequantizing random f16 data as {qtype.name} with Python")
            pydq = gguf.quants.dequantize(rq, qtype)
            logger.debug(f"Dequantizing random f16 data as {qtype.name} with C")
            ggdq = ggml_quants.dequantize(rq, qtype)

            dequant_equal = compare_tensors(pydq, ggdq, qtype)

            if not dequant_equal:
                logger.error(f"Dequantization from random f16 data as {qtype.name} does not match ❌")
            else:
                logger.info(f"Dequantization from random f16 data as {qtype.name} matches exactly ✅")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Python (de)quantization against the reference C implementation")
    parser.add_argument("--libggml", type=Path, default=Path(__file__).parent.parent.parent / "build" / "ggml" / "src" / "libggml.so", help="The path to libggml.so")
    parser.add_argument("--quick", action="store_true", help="Don't quantize with C when it's not strictly necessary")

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    do_test(args.libggml, args.quick)
