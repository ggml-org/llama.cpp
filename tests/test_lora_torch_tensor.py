#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest

from convert_lora_to_gguf import LoraTorchTensor


def _make_lora(batch_shape_B, col_size, rank, row_size, batch_shape_A=None):
    if batch_shape_A is None:
        batch_shape_A = tuple(1 for _ in batch_shape_B)
    A = torch.randn(*batch_shape_A, rank, row_size)
    B = torch.randn(*batch_shape_B, col_size, rank)
    return LoraTorchTensor(A, B)


class TestSplitDimNeg2:
    """Tests for split along col dimension (dim=-2), the GraniteMoe/BailingMoe case."""

    def test_single_int_even(self):
        lora = _make_lora((), 8, 4, 16)
        assert lora.shape == (8, 16)
        chunks = lora.split(4, dim=-2)
        assert len(chunks) == 2
        assert all(c.shape == (4, 16) for c in chunks)
        assert chunks[0]._lora_A is chunks[1]._lora_A

    def test_single_int_uneven(self):
        lora = _make_lora((), 7, 4, 16)
        chunks = lora.split(3, dim=-2)
        assert len(chunks) == 3
        assert chunks[0].shape == (3, 16)
        assert chunks[1].shape == (3, 16)
        assert chunks[2].shape == (1, 16)

    def test_list_of_ints(self):
        lora = _make_lora((), 12, 4, 16)
        chunks = lora.split([4, 3, 5], dim=-2)
        assert len(chunks) == 3
        assert chunks[0].shape == (4, 16)
        assert chunks[1].shape == (3, 16)
        assert chunks[2].shape == (5, 16)

    def test_3d_tensor(self):
        lora = _make_lora((4,), 6, 3, 10)
        assert lora.shape == (4, 6, 10)
        chunks = lora.split(3, dim=-2)
        assert len(chunks) == 2
        assert all(c.shape == (4, 3, 10) for c in chunks)
        assert chunks[0]._lora_A is chunks[1]._lora_A


class TestSplitDimNeg1:
    """Tests for split along row dimension (dim=-1)."""

    def test_single_int_even(self):
        lora = _make_lora((), 8, 4, 12)
        chunks = lora.split(4, dim=-1)
        assert len(chunks) == 3
        assert all(c.shape == (8, 4) for c in chunks)
        assert chunks[0]._lora_B is chunks[1]._lora_B

    def test_single_int_uneven(self):
        lora = _make_lora((), 8, 4, 10)
        chunks = lora.split(4, dim=-1)
        assert len(chunks) == 3
        assert chunks[0].shape == (8, 4)
        assert chunks[1].shape == (8, 4)
        assert chunks[2].shape == (8, 2)

    def test_list_of_ints(self):
        lora = _make_lora((), 8, 4, 10)
        chunks = lora.split([3, 7], dim=-1)
        assert len(chunks) == 2
        assert chunks[0].shape == (8, 3)
        assert chunks[1].shape == (8, 7)


class TestSplitBatchDim:
    """Tests for split along batch dimensions."""

    def test_broadcast_A(self):
        lora = _make_lora((6,), 8, 4, 16)
        assert lora.shape == (6, 8, 16)
        chunks = lora.split(2, dim=0)
        assert len(chunks) == 3
        assert all(c.shape == (2, 8, 16) for c in chunks)
        assert chunks[0]._lora_A is chunks[1]._lora_A

    def test_full_A(self):
        A = torch.randn(6, 4, 16)
        B = torch.randn(6, 8, 4)
        lora = LoraTorchTensor(A, B)
        assert lora.shape == (6, 8, 16)
        chunks = lora.split(2, dim=0)
        assert len(chunks) == 3
        assert all(c.shape == (2, 8, 16) for c in chunks)
        assert chunks[0]._lora_A is not chunks[1]._lora_A

    def test_positive_dim_equivalent_to_negative(self):
        lora = _make_lora((6,), 8, 4, 16)
        chunks_pos = lora.split(3, dim=0)
        chunks_neg = lora.split(3, dim=-3)
        assert len(chunks_pos) == len(chunks_neg)
        for cp, cn in zip(chunks_pos, chunks_neg):
            assert cp.shape == cn.shape


class TestTorchSplitFunction:
    """Tests for torch.split() functional form via __torch_function__."""

    def test_basic(self):
        lora = _make_lora((), 8, 4, 12)
        chunks = torch.split(lora, [3, 4, 5], dim=-1)
        assert len(chunks) == 3
        assert chunks[0].shape == (8, 3)
        assert chunks[1].shape == (8, 4)
        assert chunks[2].shape == (8, 5)

    def test_dim_keyword(self):
        lora = _make_lora((), 10, 4, 16)
        chunks = torch.split(lora, 5, dim=-2)
        assert len(chunks) == 2
        assert all(c.shape == (5, 16) for c in chunks)


class TestSplitCatRoundtrip:
    """Verify split then cat reconstructs the original tensor."""

    def test_2d_dim0(self):
        lora = _make_lora((), 8, 4, 16)
        chunks = lora.split(4, dim=-2)
        reassembled = torch.cat(list(chunks), dim=0)  # ty: ignore[no-matching-overload]
        assert isinstance(reassembled, LoraTorchTensor)
        assert reassembled.shape == lora.shape
        orig_A, orig_B = lora.get_lora_A_B()
        new_A, new_B = reassembled.get_lora_A_B()  # ty: ignore[unresolved-attribute]
        assert torch.equal(orig_A, new_A)
        assert torch.equal(orig_B, new_B)

    def test_3d_batch_dim(self):
        A = torch.randn(6, 4, 16)
        B = torch.randn(6, 8, 4)
        lora = LoraTorchTensor(A, B)
        chunks = lora.split(2, dim=0)
        reassembled = torch.cat(list(chunks), dim=0)  # ty: ignore[no-matching-overload]
        assert isinstance(reassembled, LoraTorchTensor)
        assert reassembled.shape == lora.shape
        orig_A, orig_B = lora.get_lora_A_B()
        new_A, new_B = reassembled.get_lora_A_B()  # ty: ignore[unresolved-attribute]
        assert torch.equal(orig_A, new_A)
        assert torch.equal(orig_B, new_B)
