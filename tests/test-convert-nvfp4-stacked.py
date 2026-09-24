#!/usr/bin/env python3
# CPU-only checks for the NVFP4 converter path that handles checkpoints whose routed-expert
# weights and scales arrive already stacked on an expert axis ([n_expert, out, packed_in]),
# as in stepfun-ai/Step-3.7-Flash-NVFP4. No model, no GPU: synthetic tensors only.
#
#   python -m pytest tests/test-convert-nvfp4-stacked.py

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "gguf-py")]

from conversion.base import ModelBase  # noqa: E402


def _synthetic(n_expert: int, out_features: int, n_blocks: int, seed: int = 0):
    """ModelOpt-style NVFP4 tensors: nibble-packed weights (8 bytes per block of 16) and E4M3 block scales."""
    g = torch.Generator().manual_seed(seed)
    weight = torch.randint(0, 256, (n_expert, out_features, n_blocks * 8), generator=g, dtype=torch.uint8)
    scale_bits = torch.randint(0, 128, (n_expert, out_features, n_blocks), generator=g, dtype=torch.uint8)
    scale = scale_bits.view(torch.float8_e4m3fn)
    return weight, scale


def test_stacked_pack_matches_per_expert_pack():
    n_expert, out_features, n_blocks = 3, 8, 8
    weight, scale = _synthetic(n_expert, out_features, n_blocks)

    raw, shape = ModelBase._nvfp4_pack_stacked(weight, scale)

    expected = [ModelBase._nvfp4_pack(weight[e], scale[e]) for e in range(n_expert)]
    assert shape == [n_expert, *expected[0][1]]
    assert shape == [n_expert, out_features, n_blocks * 16]
    assert raw.dtype == np.uint8
    assert raw.shape == (n_expert, out_features, (n_blocks // 4) * 36)
    for e in range(n_expert):
        np.testing.assert_array_equal(raw[e], expected[e][0])


def test_stacked_pack_of_one_expert_is_the_2d_pack_with_an_expert_axis():
    weight, scale = _synthetic(1, 4, 4, seed=1)

    raw, shape = ModelBase._nvfp4_pack_stacked(weight, scale)
    raw2d, shape2d = ModelBase._nvfp4_pack(weight[0], scale[0])

    assert shape == [1, *shape2d]
    np.testing.assert_array_equal(raw[0], raw2d)


def test_stacked_pack_keeps_expert_order():
    weight, scale = _synthetic(4, 4, 4, seed=2)

    raw, _ = ModelBase._nvfp4_pack_stacked(weight, scale)
    raw_rev, _ = ModelBase._nvfp4_pack_stacked(weight.flip(0), scale.flip(0))

    np.testing.assert_array_equal(raw_rev, raw[::-1])
    assert not np.array_equal(raw[0], raw[1])
