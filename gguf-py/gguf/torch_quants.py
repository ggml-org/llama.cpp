from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from .constants import GGML_QUANT_SIZES, GGMLQuantizationType, QK_K
from .quants import IQ1_M, IQ1_S, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_S, IQ3_XXS, IQ4_NL, MXFP4, NVFP4


DequantizeBlocks = Callable[[torch.Tensor, int, int], torch.Tensor]


def _uint16_from_bytes(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.int32)
    return x[..., 0] | (x[..., 1] << 8)


def _uint32_from_bytes(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.int32)
    return x[..., 0] | (x[..., 1] << 8) | (x[..., 2] << 16) | (x[..., 3] << 24)


def _fp16_from_bytes(x: torch.Tensor) -> torch.Tensor:
    return _uint16_from_bytes(x).to(torch.int16).view(torch.float16).unsqueeze(-1).to(torch.float32)


def _bf16_from_bytes(x: torch.Tensor) -> torch.Tensor:
    return (_uint16_from_bytes(x) << 16).view(torch.float32).unsqueeze(-1)


def _fp32_from_bytes(x: torch.Tensor) -> torch.Tensor:
    return _uint32_from_bytes(x).view(torch.float32).unsqueeze(-1)


def _split_block_dims(blocks: torch.Tensor, *dims: int) -> tuple[torch.Tensor, ...]:
    return torch.split(blocks, [*dims, blocks.shape[1] - sum(dims)], dim=1)


def _bits_to_signs(bits: torch.Tensor, device: torch.device) -> torch.Tensor:
    return torch.where(
        bits == 0,
        torch.ones((), dtype=torch.float32, device=device),
        -torch.ones((), dtype=torch.float32, device=device),
    )


def _grid_tensor(cls: type) -> torch.Tensor:
    cls.init_grid()
    return torch.from_numpy(cls.grid.squeeze().copy())


K_SCALE_SIZE = 12
KVALUES_MXFP4 = torch.tensor(MXFP4.kvalues, dtype=torch.float32)
KVALUES_NVFP4 = torch.tensor(NVFP4.kvalues, dtype=torch.float32)
KVALUES_IQ4_NL = torch.tensor(IQ4_NL.kvalues, dtype=torch.float32)
KSIGNS_IQ2_XXS = torch.from_numpy(np.frombuffer(IQ2_XXS.ksigns, dtype=np.uint8).copy())
GRID_IQ1_S = _grid_tensor(IQ1_S)
GRID_IQ2_XXS = _grid_tensor(IQ2_XXS)
GRID_IQ2_XS = _grid_tensor(IQ2_XS)
GRID_IQ2_S = _grid_tensor(IQ2_S)
GRID_IQ3_XXS = _grid_tensor(IQ3_XXS)
GRID_IQ3_S = _grid_tensor(IQ3_S)
IQ1_M.init_grid()


def _dequantize_blocks_BF16(blocks: torch.Tensor, block_size: int, type_size: int) -> torch.Tensor:
    return _bf16_from_bytes(blocks)


def _dequantize_blocks_Q4_0(blocks: torch.Tensor, block_size: int, type_size: int) -> torch.Tensor:
    n_blocks = blocks.shape[0]

    d, qs = _split_block_dims(blocks, 2)
    d = _fp16_from_bytes(d)

    shifts = torch.tensor([0, 4], device=blocks.device, dtype=torch.int32).view(1, 1, 2, 1)
    qs = qs.to(torch.int32).view(n_blocks, -1, 1, block_size // 2) >> shifts
    qs = (qs & 0x0F).view(n_blocks, -1).to(torch.int8) - 8

    return d * qs.to(torch.float32)


def _dequantize_blocks_Q4_1(blocks: torch.Tensor, block_size: int, type_size: int) -> torch.Tensor:
    n_blocks = blocks.shape[0]

    d, m, qs = _split_block_dims(blocks, 2, 2)
    d = _fp16_from_bytes(d)
    m = _fp16_from_bytes(m)

    shifts = torch.tensor([0, 4], device=blocks.device, dtype=torch.int32).view(1, 1, 2, 1)
    qs = qs.to(torch.int32).view(n_blocks, -1, 1, block_size // 2) >> shifts
    qs = (qs & 0x0F).view(n_blocks, -1).to(torch.float32)

    return (d * qs) + m


def _dequantize_blocks_Q5_0(blocks: torch.Tensor, block_size: int, type_size: int) -> torch.Tensor:
    n_blocks = blocks.shape[0]

    d, qh, qs = _split_block_dims(blocks, 2, 4)
    d = _fp16_from_bytes(d)
    qh = _uint32_from_bytes(qh).unsqueeze(-1)

    qh = qh >> torch.arange(32, device=blocks.device, dtype=torch.int32).view(1, 32)
    shifts = torch.tensor([0, 4], device=blocks.device, dtype=torch.int32).view(1, 1, 2, 1)
    ql = qs.to(torch.int32).view(n_blocks, -1, 1, block_size // 2) >> shifts
    qh = (qh & 1).to(torch.int32)
    ql = (ql & 0x0F).view(n_blocks, -1)

    qs = (ql | (qh << 4)).to(torch.int8) - 16
    return d * qs.to(torch.float32)


def _dequantize_blocks_Q5_1(blocks: torch.Tensor, block_size: int, type_size: int) -> torch.Tensor:
    n_blocks = blocks.shape[0]

    d, m, qh, qs = _split_block_dims(blocks, 2, 2, 4)
    d = _fp16_from_bytes(d)
    m = _fp16_from_bytes(m)
    qh = _uint32_from_bytes(qh).unsqueeze(-1)

    qh = qh >> torch.arange(32, device=blocks.device, dtype=torch.int32).view(1, 32)
    shifts = torch.tensor([0, 4], device=blocks.device, dtype=torch.int32).view(1, 1, 2, 1)
    ql = qs.to(torch.int32).view(n_blocks, -1, 1, block_size // 2) >> shifts
    qh = (qh & 1).to(torch.int32)
    ql = (ql & 0x0F).view(n_blocks, -1)

    qs = (ql | (qh << 4)).to(torch.float32)
    return (d * qs) + m


def _dequantize_blocks_Q8_0(blocks: torch.Tensor, block_size: int, type_size: int) -> torch.Tensor:
    d, x = _split_block_dims(blocks, 2)
    d = _fp16_from_bytes(d)
    x = x.to(torch.int8).to(torch.float32)
    return d * x


def _dequantize_blocks_Q2_K(blocks: torch.Tensor, block_size: int, type_size: int) -> torch.Tensor:
    n_blocks = blocks.shape[0]

    scales, qs, d, dmin = _split_block_dims(blocks, QK_K // 16, QK_K // 4, 2)
    d = _fp16_from_bytes(d)
    dmin = _fp16_from_bytes(dmin)

    scales = scales.to(torch.int32)
    dl = (d * (scales & 0x0F).to(torch.float32)).view(n_blocks, QK_K // 16, 1)
    ml = (dmin * (scales >> 4).to(torch.float32)).view(n_blocks, QK_K // 16, 1)

    shift = torch.tensor([0, 2, 4, 6], device=blocks.device, dtype=torch.int32).view(1, 1, 4, 1)
    qs = (qs.to(torch.int32).view(n_blocks, -1, 1, 32) >> shift) & 3
    qs = qs.view(n_blocks, QK_K // 16, 16).to(torch.float32)

    return (dl * qs - ml).view(n_blocks, -1)


def _dequantize_blocks_Q3_K(blocks: torch.Tensor, block_size: int, type_size: int) -> torch.Tensor:
    n_blocks = blocks.shape[0]

    hmask, qs, scales, d = _split_block_dims(blocks, QK_K // 8, QK_K // 4, 12)
    d = _fp16_from_bytes(d)

    scales = scales.to(torch.int32)
    lscales, hscales = scales[:, :8], scales[:, 8:]
    lshifts = torch.tensor([0, 4], device=blocks.device, dtype=torch.int32).view(1, 2, 1)
    hshifts = torch.tensor([0, 2, 4, 6], device=blocks.device, dtype=torch.int32).view(1, 4, 1)
    lscales = (lscales.view(n_blocks, 1, 8) >> lshifts).view(n_blocks, 16)
    hscales = (hscales.view(n_blocks, 1, 4) >> hshifts).view(n_blocks, 16)
    scales = (lscales & 0x0F) | ((hscales & 0x03) << 4)
    scales = scales.to(torch.int8).to(torch.float32) - 32

    dl = (d * scales).view(n_blocks, 16, 1)

    ql_shifts = torch.tensor([0, 2, 4, 6], device=blocks.device, dtype=torch.int32).view(1, 1, 4, 1)
    qh_shifts = torch.arange(8, device=blocks.device, dtype=torch.int32).view(1, 1, 8, 1)
    ql = qs.to(torch.int32).view(n_blocks, -1, 1, 32) >> ql_shifts
    qh = hmask.to(torch.int32).view(n_blocks, -1, 1, 32) >> qh_shifts
    ql = ql.view(n_blocks, 16, QK_K // 16) & 3
    qh = (qh.view(n_blocks, 16, QK_K // 16) & 1) ^ 1
    q = ql.to(torch.int8) - (qh << 2).to(torch.int8)

    return (dl * q.to(torch.float32)).view(n_blocks, QK_K)


def _get_scale_min(scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n_blocks = scales.shape[0]
    scales = scales.to(torch.int32).view(n_blocks, 3, 4)
    d, m, m_d = torch.split(scales, 1, dim=-2)

    sc = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
    min_ = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)

    return sc.view(n_blocks, 8), min_.view(n_blocks, 8)


def _dequantize_blocks_Q4_K(blocks: torch.Tensor, block_size: int, type_size: int) -> torch.Tensor:
    n_blocks = blocks.shape[0]

    d, dmin, scales, qs = _split_block_dims(blocks, 2, 2, K_SCALE_SIZE)
    d = _fp16_from_bytes(d)
    dmin = _fp16_from_bytes(dmin)

    sc, m = _get_scale_min(scales)

    d = (d * sc.to(torch.float32)).view(n_blocks, -1, 1)
    dm = (dmin * m.to(torch.float32)).view(n_blocks, -1, 1)

    shifts = torch.tensor([0, 4], device=blocks.device, dtype=torch.int32).view(1, 1, 2, 1)
    qs = qs.to(torch.int32).view(n_blocks, -1, 1, 32) >> shifts
    qs = (qs & 0x0F).view(n_blocks, -1, 32).to(torch.float32)

    return (d * qs - dm).view(n_blocks, QK_K)


def _dequantize_blocks_Q5_K(blocks: torch.Tensor, block_size: int, type_size: int) -> torch.Tensor:
    n_blocks = blocks.shape[0]

    d, dmin, scales, qh, qs = _split_block_dims(blocks, 2, 2, K_SCALE_SIZE, QK_K // 8)
    d = _fp16_from_bytes(d)
    dmin = _fp16_from_bytes(dmin)

    sc, m = _get_scale_min(scales)

    d = (d * sc.to(torch.float32)).view(n_blocks, -1, 1)
    dm = (dmin * m.to(torch.float32)).view(n_blocks, -1, 1)

    ql_shifts = torch.tensor([0, 4], device=blocks.device, dtype=torch.int32).view(1, 1, 2, 1)
    qh_shifts = torch.arange(8, device=blocks.device, dtype=torch.int32).view(1, 1, 8, 1)
    ql = qs.to(torch.int32).view(n_blocks, -1, 1, 32) >> ql_shifts
    qh = qh.to(torch.int32).view(n_blocks, -1, 1, 32) >> qh_shifts
    ql = (ql & 0x0F).view(n_blocks, -1, 32)
    qh = (qh & 0x01).view(n_blocks, -1, 32)
    q = (ql | (qh << 4)).to(torch.float32)

    return (d * q - dm).view(n_blocks, QK_K)


def _dequantize_blocks_Q6_K(blocks: torch.Tensor, block_size: int, type_size: int) -> torch.Tensor:
    n_blocks = blocks.shape[0]

    ql, qh, scales, d = _split_block_dims(blocks, QK_K // 2, QK_K // 4, QK_K // 16)

    scales = scales.to(torch.int8).to(torch.float32)
    d = _fp16_from_bytes(d)
    d = (d * scales).view(n_blocks, QK_K // 16, 1)

    ql_shifts = torch.tensor([0, 4], device=blocks.device, dtype=torch.int32).view(1, 1, 2, 1)
    ql = ql.to(torch.int32).view(n_blocks, -1, 1, 64) >> ql_shifts
    ql = (ql & 0x0F).view(n_blocks, -1, 32)
    qh_shifts = torch.tensor([0, 2, 4, 6], device=blocks.device, dtype=torch.int32).view(1, 1, 4, 1)
    qh = qh.to(torch.int32).view(n_blocks, -1, 1, 32) >> qh_shifts
    qh = (qh & 0x03).view(n_blocks, -1, 32)
    q = (ql | (qh << 4)).to(torch.int8) - 32
    q = q.view(n_blocks, QK_K // 16, -1).to(torch.float32)

    return (d * q).view(n_blocks, QK_K)


def _dequantize_blocks_TQ1_0(blocks: torch.Tensor, block_size: int, type_size: int) -> torch.Tensor:
    n_blocks = blocks.shape[0]

    qs, qh, d = _split_block_dims(blocks, (QK_K - 4 * QK_K // 64) // 5, QK_K // 64)
    d = _fp16_from_bytes(d)

    qs0, qs1 = qs[:, :32], qs[:, 32:]
    weights5 = torch.tensor([1, 3, 9, 27, 81], device=blocks.device, dtype=torch.int32).view(1, 1, 5, 1)
    weights4 = torch.tensor([1, 3, 9, 27], device=blocks.device, dtype=torch.int32).view(1, 1, 4, 1)
    qs0 = qs0.to(torch.int32).view(n_blocks, -1, 1, 32) * weights5
    qs0 = (qs0 & 0xFF).view(n_blocks, -1)
    qs1 = qs1.to(torch.int32).view(n_blocks, -1, 1, 16) * weights5
    qs1 = (qs1 & 0xFF).view(n_blocks, -1)
    qh = qh.to(torch.int32).view(n_blocks, -1, 1, 4) * weights4
    qh = (qh & 0xFF).view(n_blocks, -1)
    qs = torch.cat([qs0, qs1, qh], dim=-1)
    qs = ((qs * 3) >> 8).to(torch.int8) - 1

    return d * qs.to(torch.float32)


def _dequantize_blocks_TQ2_0(blocks: torch.Tensor, block_size: int, type_size: int) -> torch.Tensor:
    n_blocks = blocks.shape[0]

    qs, d = _split_block_dims(blocks, QK_K // 4)
    d = _fp16_from_bytes(d)

    shifts = torch.tensor([0, 2, 4, 6], device=blocks.device, dtype=torch.int32).view(1, 1, 4, 1)
    qs = qs.to(torch.int32).view(n_blocks, -1, 1, 32) >> shifts
    qs = (qs & 0x03).view(n_blocks, -1).to(torch.int8) - 1

    return d * qs.to(torch.float32)


def _e8m0_to_fp32_half(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.int32)
    bits = torch.where(x < 2, torch.tensor(0x00200000, device=x.device, dtype=torch.int32) << x, (x - 1) << 23)
    return bits.view(torch.float32)


def _dequantize_blocks_MXFP4(blocks: torch.Tensor, block_size: int, type_size: int) -> torch.Tensor:
    n_blocks = blocks.shape[0]

    e, qs = _split_block_dims(blocks, 1)
    d = _e8m0_to_fp32_half(e)

    shifts = torch.tensor([0, 4], device=blocks.device, dtype=torch.int32).view(1, 2, 1)
    qs = qs.to(torch.int32).view(n_blocks, 1, block_size // 2) >> shifts
    qs = (qs & 0x0F).view(n_blocks, -1).to(torch.long)

    kvalues = KVALUES_MXFP4.to(device=blocks.device)
    qs = kvalues[qs].view(n_blocks, block_size)

    return d * qs


def _ue4m3_to_fp32(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.int32)
    exp = (x >> 3) & 0x0F
    man = (x & 0x07).to(torch.float32)
    raw = torch.where(
        exp == 0,
        man * 2.0**-9,
        (1.0 + man / 8.0) * torch.pow(torch.tensor(2.0, device=x.device), exp.to(torch.float32) - 7.0),
    )
    return torch.where((x == 0) | (x == 0x7F), torch.zeros((), device=x.device, dtype=torch.float32), raw * 0.5)


def _dequantize_blocks_NVFP4(blocks: torch.Tensor, block_size: int, type_size: int) -> torch.Tensor:
    n_blocks = blocks.shape[0]

    d_bytes, qs = _split_block_dims(blocks, 4)
    d = _ue4m3_to_fp32(d_bytes).view(n_blocks, 4, 1)

    qs = qs.to(torch.int32).view(n_blocks, 4, 8)
    lo = qs & 0x0F
    hi = qs >> 4
    vals = torch.cat([lo, hi], dim=-1).to(torch.long)

    kvalues = KVALUES_NVFP4.to(device=blocks.device)
    vals = kvalues[vals]

    return (d * vals).view(n_blocks, 64)


def _dequantize_blocks_IQ2_XXS(blocks: torch.Tensor, block_size: int, type_size: int) -> torch.Tensor:
    n_blocks = blocks.shape[0]

    d, qs = _split_block_dims(blocks, 2)
    d = _fp16_from_bytes(d)

    qs_bytes = qs.view(n_blocks, 8, 2, 4)
    qs_u32 = _uint32_from_bytes(qs_bytes)
    q0 = qs_bytes[:, :, 0, :]
    q1 = qs_u32[:, :, 1]

    db = d * (0.5 + ((q1 >> 28) & 0x0F).to(torch.float32)) * 0.25
    db = db.view(n_blocks, 8, 1, 1)

    shifts = torch.tensor([0, 7, 14, 21], device=blocks.device, dtype=torch.int32).view(1, 1, 4)
    sign_indices = (q1.unsqueeze(-1) >> shifts) & 0x7F
    sign_bytes = KSIGNS_IQ2_XXS.to(device=blocks.device)[sign_indices.to(torch.long)]

    shifts_bits = torch.arange(8, device=blocks.device, dtype=torch.int32).view(1, 1, 1, 8)
    signs = (sign_bytes.to(torch.int32).unsqueeze(-1) >> shifts_bits) & 1
    signs = _bits_to_signs(signs, blocks.device)
    signs = signs.view(n_blocks, 8, 4, 8)

    grid = GRID_IQ2_XXS.to(device=blocks.device)[q0.to(torch.long)]
    grid = grid.view(n_blocks, 8, 4, 8)

    return (db * grid * signs).view(n_blocks, -1)


def _dequantize_blocks_IQ2_XS(blocks: torch.Tensor, block_size: int, type_size: int) -> torch.Tensor:
    n_blocks = blocks.shape[0]

    d, qs, scales = _split_block_dims(blocks, 2, 2 * QK_K // 8)
    d = _fp16_from_bytes(d)
    qs = qs.view(n_blocks, -1, 2)
    qs = _uint16_from_bytes(qs)

    scale_shifts = torch.tensor([0, 4], device=blocks.device, dtype=torch.int32).view(1, 1, 2)
    scales = scales.to(torch.int32).view(n_blocks, -1, 1) >> scale_shifts
    scales = (scales & 0x0F).view(n_blocks, -1)
    db = d * (0.5 + scales.to(torch.float32)) * 0.25
    db = db.view(n_blocks, -1, 1, 1)

    signs = KSIGNS_IQ2_XXS.to(device=blocks.device)[(qs >> 9).to(torch.long)]
    shifts_bits = torch.arange(8, device=blocks.device, dtype=torch.int32).view(1, 1, 8)
    signs = (signs.to(torch.int32).view(n_blocks, -1, 1) >> shifts_bits) & 1
    signs = _bits_to_signs(signs, blocks.device)
    signs = signs.view(n_blocks, -1, 2, 8)

    grid = GRID_IQ2_XS.to(device=blocks.device)[(qs & 511).to(torch.long)]
    grid = grid.view(n_blocks, -1, 2, 8)

    return (db * grid * signs).view(n_blocks, -1)


def _dequantize_blocks_IQ2_S(blocks: torch.Tensor, block_size: int, type_size: int) -> torch.Tensor:
    n_blocks = blocks.shape[0]

    d, qs, signs, qh, scales = _split_block_dims(blocks, 2, QK_K // 8, QK_K // 8, QK_K // 32)
    d = _fp16_from_bytes(d)

    scale_shifts = torch.tensor([0, 4], device=blocks.device, dtype=torch.int32).view(1, 1, 2)
    scales = scales.to(torch.int32).view(n_blocks, -1, 1) >> scale_shifts
    scales = (scales & 0x0F).view(n_blocks, -1)
    db = d * (0.5 + scales.to(torch.float32)) * 0.25
    db = db.view(n_blocks, -1, 1, 1)

    shifts_bits = torch.arange(8, device=blocks.device, dtype=torch.int32).view(1, 1, 8)
    signs = (signs.to(torch.int32).view(n_blocks, -1, 1) >> shifts_bits) & 1
    signs = _bits_to_signs(signs, blocks.device)
    signs = signs.view(n_blocks, -1, 2, 8)

    qh_shifts = torch.tensor([0, 2, 4, 6], device=blocks.device, dtype=torch.int32).view(1, 1, 4)
    qh = qh.to(torch.int32).view(n_blocks, -1, 1) >> qh_shifts
    qs = qs.to(torch.int32) | ((qh & 0x03) << 8).view(n_blocks, -1)

    grid = GRID_IQ2_S.to(device=blocks.device)[qs.to(torch.long)]
    grid = grid.view(n_blocks, -1, 2, 8)

    return (db * grid * signs).view(n_blocks, -1)


def _dequantize_blocks_IQ3_XXS(blocks: torch.Tensor, block_size: int, type_size: int) -> torch.Tensor:
    n_blocks = blocks.shape[0]

    d, qs, scales = _split_block_dims(blocks, 2, QK_K // 4)
    d = _fp16_from_bytes(d)
    scales = scales.view(n_blocks, -1, 4)
    scales = _uint32_from_bytes(scales)

    db = d * (0.5 + ((scales >> 28) & 0x0F).to(torch.float32)) * 0.5
    db = db.view(n_blocks, -1, 1, 1)

    shifts = torch.tensor([0, 7, 14, 21], device=blocks.device, dtype=torch.int32).view(1, 1, 4)
    signs = (scales.view(n_blocks, -1, 1) >> shifts) & 0x7F
    signs = KSIGNS_IQ2_XXS.to(device=blocks.device)[signs.to(torch.long)]
    shifts_bits = torch.arange(8, device=blocks.device, dtype=torch.int32).view(1, 1, 1, 8)
    signs = (signs.to(torch.int32).view(n_blocks, -1, 4, 1) >> shifts_bits) & 1
    signs = _bits_to_signs(signs, blocks.device)
    signs = signs.view(n_blocks, -1, 4, 8)

    grid = GRID_IQ3_XXS.to(device=blocks.device)[qs.to(torch.long)]
    grid = grid.view(n_blocks, -1, 4, 8)

    return (db * grid * signs).view(n_blocks, -1)


def _dequantize_blocks_IQ3_S(blocks: torch.Tensor, block_size: int, type_size: int) -> torch.Tensor:
    n_blocks = blocks.shape[0]

    d, qs, qh, signs, scales = _split_block_dims(blocks, 2, QK_K // 4, QK_K // 32, QK_K // 8)
    d = _fp16_from_bytes(d)

    scale_shifts = torch.tensor([0, 4], device=blocks.device, dtype=torch.int32).view(1, 1, 2)
    scales = scales.to(torch.int32).view(n_blocks, -1, 1) >> scale_shifts
    scales = (scales & 0x0F).view(n_blocks, -1)
    db = d * (1 + 2 * scales.to(torch.float32))
    db = db.view(n_blocks, -1, 1, 1)

    shifts_bits = torch.arange(8, device=blocks.device, dtype=torch.int32).view(1, 1, 8)
    signs = (signs.to(torch.int32).view(n_blocks, -1, 1) >> shifts_bits) & 1
    signs = _bits_to_signs(signs, blocks.device)
    signs = signs.view(n_blocks, -1, 4, 8)

    qh = qh.to(torch.int32).view(n_blocks, -1, 1) >> torch.arange(8, device=blocks.device, dtype=torch.int32)
    qh = (qh & 0x01).view(n_blocks, -1)
    qs = qs.to(torch.int32) | (qh << 8)

    grid = GRID_IQ3_S.to(device=blocks.device)[qs.to(torch.long)]
    grid = grid.view(n_blocks, -1, 4, 8)

    return (db * grid * signs).view(n_blocks, -1)


def _dequantize_blocks_IQ1_S(blocks: torch.Tensor, block_size: int, type_size: int) -> torch.Tensor:
    n_blocks = blocks.shape[0]

    d, qs, qh = _split_block_dims(blocks, 2, QK_K // 8)
    d = _fp16_from_bytes(d)
    qh = qh.view(n_blocks, -1, 2)
    qh = _uint16_from_bytes(qh)

    dl = d * (2 * ((qh >> 12) & 7).to(torch.float32) + 1)
    dl = dl.view(n_blocks, -1, 1, 1)
    delta = torch.where(
        (qh & 0x8000) == 0,
        torch.full((), float(IQ1_S.delta), dtype=torch.float32, device=blocks.device),
        torch.full((), -float(IQ1_S.delta), dtype=torch.float32, device=blocks.device),
    )
    delta = delta.view(n_blocks, -1, 1, 1)

    qh_shifts = torch.tensor([0, 3, 6, 9], device=blocks.device, dtype=torch.int32).view(1, 1, 4)
    qh = qh.view(n_blocks, -1, 1) >> qh_shifts
    qs = qs.to(torch.int32) | ((qh & 7) << 8).view(n_blocks, -1)

    grid = GRID_IQ1_S.to(device=blocks.device)[qs.to(torch.long)]
    grid = grid.view(n_blocks, -1, 4, 8)

    return (dl * (grid + delta)).view(n_blocks, -1)


def _dequantize_blocks_IQ1_M(blocks: torch.Tensor, block_size: int, type_size: int) -> torch.Tensor:
    n_blocks = blocks.shape[0]

    qs, qh, scales = _split_block_dims(blocks, QK_K // 8, QK_K // 16)
    scales = scales.view(n_blocks, -1, 2)
    scales = _uint16_from_bytes(scales)

    d_shifts = torch.tensor([12, 8, 4, 0], device=blocks.device, dtype=torch.int32).view(1, 4)
    d = (scales.view(n_blocks, 4) & 0xF000) >> d_shifts
    d = d[:, 0] | d[:, 1] | d[:, 2] | d[:, 3]
    d = d.to(torch.int16).view(torch.float16).to(torch.float32).view(n_blocks, 1)

    scale_shifts = torch.tensor([0, 3, 6, 9], device=blocks.device, dtype=torch.int32).view(1, 1, 4)
    scales = scales.view(n_blocks, -1, 1) >> scale_shifts
    scales = (scales & 0x07).view(n_blocks, -1)
    dl = d * (2 * scales.to(torch.float32) + 1)
    dl = dl.view(n_blocks, -1, 2, 1, 1)

    qh_shifts = torch.tensor([0, 4], device=blocks.device, dtype=torch.int32).view(1, 1, 2)
    qh = qh.to(torch.int32).view(n_blocks, -1, 1) >> qh_shifts
    qs = qs.to(torch.int32) | ((qh & 0x07) << 8).view(n_blocks, -1)

    delta = torch.where(
        qh & 0x08 == 0,
        torch.full((), float(IQ1_M.delta), dtype=torch.float32, device=blocks.device),
        torch.full((), -float(IQ1_M.delta), dtype=torch.float32, device=blocks.device),
    )
    delta = delta.view(n_blocks, -1, 2, 2, 1)

    grid = GRID_IQ1_S.to(device=blocks.device)[qs.to(torch.long)]
    grid = grid.view(n_blocks, -1, 2, 2, 8)

    return (dl * (grid + delta)).view(n_blocks, -1)


def _dequantize_blocks_IQ4_NL(blocks: torch.Tensor, block_size: int, type_size: int) -> torch.Tensor:
    n_blocks = blocks.shape[0]

    d, qs = _split_block_dims(blocks, 2)
    d = _fp16_from_bytes(d)

    shifts = torch.tensor([0, 4], device=blocks.device, dtype=torch.int32).view(1, 1, 2, 1)
    qs = qs.to(torch.int32).view(n_blocks, -1, 1, block_size // 2) >> shifts
    qs = (qs & 0x0F).view(n_blocks, -1).to(torch.long)

    kvalues = KVALUES_IQ4_NL.to(device=blocks.device)
    qs = kvalues[qs].view(n_blocks, -1)

    return d * qs


def _dequantize_blocks_IQ4_XS(blocks: torch.Tensor, block_size: int, type_size: int) -> torch.Tensor:
    n_blocks = blocks.shape[0]

    d, scales_h, scales_l, qs = _split_block_dims(blocks, 2, 2, QK_K // 64)
    d = _fp16_from_bytes(d)
    scales_h = _uint16_from_bytes(scales_h).unsqueeze(-1)

    scales_l_shifts = torch.tensor([0, 4], device=blocks.device, dtype=torch.int32).view(1, 1, 2)
    scales_l = scales_l.to(torch.int32).view(n_blocks, -1, 1) >> scales_l_shifts
    scales_h_shifts = torch.arange(0, 2 * (QK_K // 32), 2, device=blocks.device, dtype=torch.int32)
    scales_h = scales_h.view(n_blocks, 1, -1) >> scales_h_shifts.view(1, -1, 1)
    scales_l = scales_l.view(n_blocks, -1) & 0x0F
    scales_h = scales_h.view(n_blocks, -1) & 0x03

    scales = (scales_l | (scales_h << 4)).to(torch.int8).to(torch.float32) - 32
    dl = (d * scales).view(n_blocks, -1, 1)

    qs_shifts = torch.tensor([0, 4], device=blocks.device, dtype=torch.int32).view(1, 1, 2, 1)
    qs = qs.to(torch.int32).view(n_blocks, -1, 1, 16) >> qs_shifts
    qs = (qs.view(n_blocks, -1, 32) & 0x0F).to(torch.long)

    kvalues = KVALUES_IQ4_NL.to(device=blocks.device)
    qs = kvalues[qs].view(n_blocks, -1, 32)

    return (dl * qs).view(n_blocks, -1)


DEQUANTIZE_BLOCKS: dict[GGMLQuantizationType, DequantizeBlocks] = {
    GGMLQuantizationType.BF16: _dequantize_blocks_BF16,
    GGMLQuantizationType.Q4_0: _dequantize_blocks_Q4_0,
    GGMLQuantizationType.Q4_1: _dequantize_blocks_Q4_1,
    GGMLQuantizationType.Q5_0: _dequantize_blocks_Q5_0,
    GGMLQuantizationType.Q5_1: _dequantize_blocks_Q5_1,
    GGMLQuantizationType.Q8_0: _dequantize_blocks_Q8_0,
    GGMLQuantizationType.Q2_K: _dequantize_blocks_Q2_K,
    GGMLQuantizationType.Q3_K: _dequantize_blocks_Q3_K,
    GGMLQuantizationType.Q4_K: _dequantize_blocks_Q4_K,
    GGMLQuantizationType.Q5_K: _dequantize_blocks_Q5_K,
    GGMLQuantizationType.Q6_K: _dequantize_blocks_Q6_K,
    GGMLQuantizationType.TQ1_0: _dequantize_blocks_TQ1_0,
    GGMLQuantizationType.TQ2_0: _dequantize_blocks_TQ2_0,
    GGMLQuantizationType.MXFP4: _dequantize_blocks_MXFP4,
    GGMLQuantizationType.NVFP4: _dequantize_blocks_NVFP4,
    GGMLQuantizationType.IQ2_XXS: _dequantize_blocks_IQ2_XXS,
    GGMLQuantizationType.IQ2_XS: _dequantize_blocks_IQ2_XS,
    GGMLQuantizationType.IQ2_S: _dequantize_blocks_IQ2_S,
    GGMLQuantizationType.IQ3_XXS: _dequantize_blocks_IQ3_XXS,
    GGMLQuantizationType.IQ3_S: _dequantize_blocks_IQ3_S,
    GGMLQuantizationType.IQ1_S: _dequantize_blocks_IQ1_S,
    GGMLQuantizationType.IQ1_M: _dequantize_blocks_IQ1_M,
    GGMLQuantizationType.IQ4_NL: _dequantize_blocks_IQ4_NL,
    GGMLQuantizationType.IQ4_XS: _dequantize_blocks_IQ4_XS,
}

SUPPORTED_QUANT_TYPES = frozenset({
    GGMLQuantizationType.F32,
    GGMLQuantizationType.F16,
    *DEQUANTIZE_BLOCKS.keys(),
})


def dequantize(data: torch.Tensor, qtype: GGMLQuantizationType, dtype: torch.dtype | None = None) -> torch.Tensor:
    if dtype is None:
        dtype = torch.float32

    if qtype == GGMLQuantizationType.F32:
        if data.dtype == torch.uint8:
            out = _fp32_from_bytes(data.reshape(-1, 4)).reshape(*data.shape[:-1], data.shape[-1] // 4)
            return out.to(dtype)
        return data.to(dtype)

    if qtype == GGMLQuantizationType.F16:
        if data.dtype == torch.uint8:
            out = _fp16_from_bytes(data.reshape(-1, 2)).reshape(*data.shape[:-1], data.shape[-1] // 2)
            return out.to(dtype)
        return data.to(dtype)

    dequantize_blocks = DEQUANTIZE_BLOCKS.get(qtype)
    if dequantize_blocks is None:
        raise NotImplementedError(f"PyTorch dequantization for {qtype.name} is not yet implemented")

    block_size, type_size = GGML_QUANT_SIZES[qtype]
    blocks = data.reshape(-1, data.shape[-1]).reshape(-1, type_size)
    out = dequantize_blocks(blocks, block_size, type_size)
    out = out.reshape(*data.shape[:-1], data.shape[-1] // type_size * block_size)
    return out.to(dtype)
