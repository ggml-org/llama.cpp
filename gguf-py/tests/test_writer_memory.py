#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path

import numpy as np

if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent.parent / "gguf-py").exists():
    sys.path.insert(0, str(Path(__file__).parent.parent))
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import gguf


class TestBf16RawPassthrough(unittest.TestCase):
    def test_bf16_passthrough_bytes_match_torch_storage(self):
        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None:
            self.skipTest("torch and transformers are required for conversion helper tests")

        import torch

        from conversion.base import ModelBase

        tensor = torch.tensor(
            [[1.0, -2.5, 3.25, 4.5], [0.0, 7.75, -8.5, 9.0]],
            dtype=torch.float32,
        ).to(torch.bfloat16)

        expected = tensor.contiguous().view(torch.uint8).reshape(2, 8).numpy()
        legacy = gguf.quants.quantize(tensor.float().numpy(), gguf.GGMLQuantizationType.BF16)
        got = ModelBase._bf16_tensor_to_gguf_bytes(tensor)

        self.assertEqual(got.dtype, np.uint8)
        self.assertEqual(got.shape, expected.shape)
        np.testing.assert_array_equal(got, expected)
        np.testing.assert_array_equal(got, legacy)


if __name__ == "__main__":
    unittest.main()
