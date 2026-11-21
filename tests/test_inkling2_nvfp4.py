"""Check Inkling 2 checkpoint NVFP4 block repacking (no large checkpoint needed)."""

import unittest

import numpy as np
import torch

from conversion.inkling2 import _nvfp4_expert, _nvfp4_expert_rows


def _unpack_checkpoint(packed: np.ndarray) -> np.ndarray:
    result = np.empty((*packed.shape[:-1], packed.shape[-1] * 2), dtype=np.uint8)
    result[..., 0::2] = packed & 15
    result[..., 1::2] = packed >> 4
    return result


def _unpack_ggml(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    blocks = raw.reshape(raw.shape[0], -1, 36)
    scales = blocks[..., :4].reshape(raw.shape[0], -1)
    pairs = blocks[..., 4:].reshape(raw.shape[0], -1, 4, 8)
    values = np.empty((*pairs.shape[:-1], 16), dtype=np.uint8)
    values[..., :8] = pairs & 15
    values[..., 8:] = pairs >> 4
    return values.reshape(raw.shape[0], -1), scales


class TestInkling2NativeNVFP4(unittest.TestCase):
    def test_repack_preserves_fp4_values_and_e4m3_scales(self):
        rng = np.random.default_rng(0)
        packed = rng.integers(0, 256, size=(3, 128), dtype=np.uint8)  # 256 values/row
        scales = rng.integers(0, 127, size=(3, 16), dtype=np.uint8)
        raw = _nvfp4_expert_rows(packed, scales, "test", 0)
        self.assertEqual(raw.shape, (3, 144))  # 4 x 36 bytes per row
        values_out, scales_out = _unpack_ggml(raw)
        np.testing.assert_array_equal(values_out, _unpack_checkpoint(packed))
        np.testing.assert_array_equal(scales_out, scales)

    def test_gate_up_rows_are_deinterleaved(self):
        rng = np.random.default_rng(1)
        packed = rng.integers(0, 256, size=(1, 6, 32), dtype=np.uint8)
        scales = rng.integers(0, 127, size=(1, 6, 4), dtype=np.uint8)
        # Model checkpoint alternates gate/up rows; GGUF expects gate then up.
        output = _nvfp4_expert(
            lambda: torch.from_numpy(packed),
            lambda: torch.from_numpy(scales),
            "test", 0, True,
        )
        expected = np.concatenate([
            _nvfp4_expert_rows(packed[0, 0::2], scales[0, 0::2], "test", 0),
            _nvfp4_expert_rows(packed[0, 1::2], scales[0, 1::2], "test", 0),
        ])
        np.testing.assert_array_equal(output, expected)

    def test_rejects_incompatible_shapes_and_invalid_scales(self):
        packed = np.zeros((1, 32), dtype=np.uint8)
        with self.assertRaises(ValueError):
            _nvfp4_expert_rows(packed, np.ones((1, 3), dtype=np.uint8), "test", 0)
        for bad in (0x80, 0x7F):
            scales = np.full((1, 4), bad, dtype=np.uint8)
            with self.assertRaises(ValueError):
                _nvfp4_expert_rows(packed, scales, "test", 0)


if __name__ == "__main__":
    unittest.main()
