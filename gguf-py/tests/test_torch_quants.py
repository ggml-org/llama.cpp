from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import gguf
from gguf import torch_quants


QTYPES = sorted(torch_quants.SUPPORTED_QUANT_TYPES, key=lambda qtype: qtype.value)
TENSOR_SHAPE = (8, 1024, 1024)


def make_data(qtype: gguf.GGMLQuantizationType) -> np.ndarray:
    rng = np.random.default_rng()
    if qtype == gguf.GGMLQuantizationType.F32:
        return rng.standard_normal(TENSOR_SHAPE).astype(np.float32)
    if qtype == gguf.GGMLQuantizationType.F16:
        return rng.standard_normal(TENSOR_SHAPE).astype(np.float16)
    byte_shape = gguf.quant_shape_to_byte_shape(TENSOR_SHAPE, qtype)
    return rng.integers(0, 256, size=byte_shape, dtype=np.uint8)


def dequantize_reference(data: np.ndarray, qtype: gguf.GGMLQuantizationType) -> np.ndarray:
    if qtype == gguf.GGMLQuantizationType.F32:
        return data.astype(np.float32)
    if qtype == gguf.GGMLQuantizationType.F16:
        return data.astype(np.float32)
    return gguf.dequantize(data, qtype)


def assert_equal_dequantized(actual: torch.Tensor, expected: np.ndarray) -> None:
    expected_tensor = torch.from_numpy(expected.astype(np.float32)).to(actual.device)
    # Do not distinguish +0.0 and -0.0
    torch.testing.assert_close(actual, expected_tensor, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("qtype", QTYPES, ids=lambda qtype: qtype.name)
def test_dequantize_matches_numpy(qtype: gguf.GGMLQuantizationType) -> None:
    data = make_data(qtype)
    actual = torch_quants.dequantize(torch.from_numpy(data), qtype)
    expected = dequantize_reference(data, qtype)
    assert_equal_dequantized(actual, expected)


@pytest.mark.parametrize("qtype", QTYPES, ids=lambda qtype: qtype.name)
def test_dequantize_matches_numpy_with_compile(qtype: gguf.GGMLQuantizationType) -> None:
    # Recompilation is expected for testing
    if hasattr(torch, "_dynamo"):
        torch._dynamo.reset()

    data = make_data(qtype)
    dequantize_compiled = torch.compile(torch_quants.dequantize, fullgraph=True)
    actual = dequantize_compiled(torch.from_numpy(data), qtype)
    expected = dequantize_reference(data, qtype)
    assert_equal_dequantized(actual, expected)
