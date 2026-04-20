#!/usr/bin/env python3

"""
Convert a Qwen2-based Fairy2i Hugging Face checkpoint to GGUF.

This script is intentionally separate from convert_fairy2i.py:
- tokenizer export follows the Qwen2/GPT-2-style path used by convert_hf_to_gguf.py
- RoPE base is read from config["rope_parameters"]["rope_theta"] when present
- optional attention biases are exported so the GGUF can carry q/k/v bias tensors
- Qwen2-based Fairy2i 32B weights are exported with a tile64_v2 layout that matches
  the training-side QAT kernel semantics
"""

import argparse
import gc
import json
import sys
from hashlib import sha256
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from safetensors import safe_open

import gguf
from gguf.constants import QK_IFAIRY


QWEN2_PRETOKENIZER_HASHES = {
    # ref: convert_hf_to_gguf.py get_vocab_base_pre()
    "d4540891389ea895b53b399da6ac824becc30f2fba0e9ddbb98f92e55ca0e97c",
    "e636dc30a262dcc0d8c323492e32ae2b70728f4df7dfe9737d9f920a282b8aea",
}

TILE64 = 64


def round_up(x: int, base: int) -> int:
    return ((x + base - 1) // base) * base


def token_looks_special(token: str | bytes) -> bool:
    if isinstance(token, bytes):
        token_text = token.decode("utf-8")
    else:
        token_text = token

    seems_special = token_text in (
        "<pad>",
        "<mask>",
        "<2mass>",
        "[@BOS@]",
    )
    seems_special = seems_special or (token_text.startswith("<|") and token_text.endswith("|>"))
    seems_special = seems_special or (token_text.startswith("<｜") and token_text.endswith("｜>"))
    seems_special = seems_special or (token_text.startswith("<unused") and token_text.endswith(">"))
    return seems_special


def get_qwen2_tokenizer_pre(model_dir: Path) -> str:
    chktxt = (
        "\n \n\n \n\n\n \t \t\t \t\n  \n   \n    \n     \n"
        "🚀 (normal) 😶\u200d🌫️ (multiple emojis concatenated) ✅ 🦙🦙 3 33 333 3333 33333 333333 "
        "3333333 33333333 3.3 3..3 3...3 កាន់តែពិសេសអាច😁 ?我想在apple工作1314151天～ ------======= "
        "нещо на Български ''''''```````\"\"\"\"......!!!!!!?????? I've been 'told he's there, "
        "'RE you sure? 'M not sure I'll make it, 'D you like some tea? We'Ve a'lL"
    )

    try:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        chktok = tokenizer.encode(chktxt).ids
        chkhsh = sha256(str(chktok).encode()).hexdigest()
        if chkhsh in QWEN2_PRETOKENIZER_HASHES:
            return "qwen2"

        print(
            f"warning: unrecognized Qwen2 tokenizer pre hash {chkhsh}, falling back to tokenizer.ggml.pre=qwen2",
            file=sys.stderr,
        )
    except Exception as exc:
        print(
            f"warning: failed to evaluate Qwen2 tokenizer pre-tokenizer via tokenizers ({exc}), "
            "falling back to tokenizer.ggml.pre=qwen2",
            file=sys.stderr,
        )

    return "qwen2"


def set_vocab_qwen2(model_dir: Path, config: dict, writer: gguf.GGUFWriter) -> None:
    tokenizer_json_file = model_dir / "tokenizer.json"
    if not tokenizer_json_file.is_file():
        raise FileNotFoundError(f"tokenizer.json not found in {model_dir}")

    tokenizer_json = json.loads(tokenizer_json_file.read_text(encoding="utf-8"))
    vocab_size = int(config["vocab_size"])
    vocab = tokenizer_json.get("model", {}).get("vocab")
    if not isinstance(vocab, dict):
        raise ValueError(f"invalid vocab in {tokenizer_json_file}")
    assert max(vocab.values()) < vocab_size

    tokpre = get_qwen2_tokenizer_pre(model_dir)
    reverse_vocab = {id_: encoded_tok for encoded_tok, id_ in vocab.items()}
    added_tokens = tokenizer_json.get("added_tokens", [])

    added_vocab: dict[str, int] = {}
    added_tokens_decoder: dict[int, dict] = {}
    if isinstance(added_tokens, list):
        for item in added_tokens:
            if not isinstance(item, dict):
                continue
            token = item.get("content")
            token_id = item.get("id")
            if not isinstance(token, str) or not isinstance(token_id, int):
                continue
            added_vocab[token] = token_id
            added_tokens_decoder[token_id] = item
            reverse_vocab[token_id] = token

    tokens: list[str] = []
    toktypes: list[int] = []

    for i in range(vocab_size):
        if i not in reverse_vocab:
            tokens.append(f"[PAD{i}]")
            toktypes.append(gguf.TokenType.UNUSED)
            continue

        token = reverse_vocab[i]
        if token in added_vocab:
            decoder_entry = added_tokens_decoder.get(i)
            is_special = bool((decoder_entry or {}).get("special", False)) or token_looks_special(token)
            toktypes.append(gguf.TokenType.CONTROL if is_special else gguf.TokenType.USER_DEFINED)
        else:
            toktypes.append(gguf.TokenType.NORMAL)

        tokens.append(token)

    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre(tokpre)
    writer.add_token_list(tokens)
    writer.add_token_types(toktypes)

    special_vocab = gguf.SpecialVocab(model_dir, load_merges=True)
    special_vocab.add_to_gguf(writer)

    tokenizer_config_file = model_dir / "tokenizer_config.json"
    if tokenizer_config_file.is_file():
        tokenizer_config = json.loads(tokenizer_config_file.read_text(encoding="utf-8"))
        if "add_prefix_space" in tokenizer_config:
            writer.add_add_space_prefix(tokenizer_config["add_prefix_space"])


def load_weight_map(model_dir: Path) -> Dict[str, str]:
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.is_file():
        index = json.loads(index_file.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"invalid weight_map in {index_file}")
        return {k: v for k, v in weight_map.items()}

    model_files = sorted(model_dir.glob("*.safetensors"))
    if len(model_files) != 1:
        raise ValueError("no shard index and cannot infer a single safetensors file")

    filename = model_files[0].name
    with safe_open(str(model_files[0]), framework="pt", device="cpu") as f:
        return {key: filename for key in f.keys()}


class TensorReader:
    def __init__(self, model_dir: Path, weight_map: Dict[str, str]):
        self.model_dir = model_dir
        self.weight_map = weight_map

    def has(self, key: str) -> bool:
        return key in self.weight_map

    def get(self, key: str) -> torch.Tensor:
        if key not in self.weight_map:
            raise KeyError(f"missing tensor key: {key}")
        filename = self.weight_map[key]
        path = self.model_dir / filename
        with safe_open(str(path), framework="pt", device="cpu") as f:
            return f.get_tensor(key)


def undo_llama_permute(weight: torch.Tensor, n_head: int) -> torch.Tensor:
    return (
        weight.reshape(n_head, 2, weight.shape[0] // n_head // 2, *weight.shape[1:])
        .swapaxes(1, 2)
        .reshape(weight.shape)
    )


def phase_quant_v1(w_real: np.ndarray, w_imag: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    abs_real = np.abs(w_real)
    abs_imag = np.abs(w_imag)

    choose_real = abs_real > abs_imag
    choose_imag = abs_imag > abs_real

    ties = ~(choose_real | choose_imag)
    if np.any(ties):
        both_zero = ties & (abs_real == 0.0)
        same_sign = (w_real * w_imag) >= 0.0
        choose_imag |= ties & (~both_zero) & same_sign
        choose_real |= ties & (~both_zero) & (~same_sign)
        choose_real |= both_zero

    mask_real = choose_real
    mask_imag = choose_imag

    s_real = np.mean(abs_real[mask_real], dtype=np.float64) if np.any(mask_real) else 0.0
    s_imag = np.mean(abs_imag[mask_imag], dtype=np.float64) if np.any(mask_imag) else 0.0
    s_real = max(float(s_real), 1e-6) if np.isfinite(s_real) else 1e-6
    s_imag = max(float(s_imag), 1e-6) if np.isfinite(s_imag) else 1e-6

    q_real = np.zeros_like(w_real, dtype=np.float32)
    q_imag = np.zeros_like(w_imag, dtype=np.float32)

    q_real[mask_real] = np.where(w_real[mask_real] >= 0.0, s_real, -s_real)
    q_imag[mask_imag] = np.where(w_imag[mask_imag] >= 0.0, s_imag, -s_imag)

    return q_real, q_imag, s_real, s_imag


def phase_quant_v2(
    w_real: np.ndarray, w_imag: np.ndarray
) -> tuple[tuple[np.ndarray, np.ndarray, float, float], tuple[np.ndarray, np.ndarray, float, float]]:
    q0_real, q0_imag, s0_real, s0_imag = phase_quant_v1(w_real, w_imag)
    e_real = w_real - q0_real
    e_imag = w_imag - q0_imag
    q1_real, q1_imag, s1_real, s1_imag = phase_quant_v1(e_real, e_imag)
    return (q0_real, q0_imag, s0_real, s0_imag), (q1_real, q1_imag, s1_real, s1_imag)


def pad_complex_matrix(mat: np.ndarray, out_dim: int, in_dim: int) -> np.ndarray:
    out_src, in_src = mat.shape
    if out_src > out_dim or in_src > in_dim:
        raise ValueError(f"cannot pad from {(out_src, in_src)} to {(out_dim, in_dim)}")
    if out_src == out_dim and in_src == in_dim:
        return mat.astype(np.float32, copy=False)

    out = np.zeros((out_dim, in_dim), dtype=np.float32)
    out[:out_src, :in_src] = mat
    return out


def quantize_tile64_once(tile_real: np.ndarray, tile_imag: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    abs_real = np.abs(tile_real)
    abs_imag = np.abs(tile_imag)

    is_real_dominant = abs_real > abs_imag
    is_imag_dominant = ~is_real_dominant

    real_count = int(np.count_nonzero(is_real_dominant))
    imag_count = int(np.count_nonzero(is_imag_dominant))

    if real_count > 0:
        real_scale = float(np.sum(abs_real[is_real_dominant], dtype=np.float64) / real_count)
    else:
        real_scale = 0.0

    if imag_count > 0:
        imag_scale = float(np.sum(abs_imag[is_imag_dominant], dtype=np.float64) / imag_count)
    else:
        imag_scale = 0.0

    q_real = np.zeros_like(tile_real, dtype=np.float32)
    q_imag = np.zeros_like(tile_imag, dtype=np.float32)

    if real_count > 0:
        q_real[is_real_dominant] = np.where(tile_real[is_real_dominant] >= 0.0, real_scale, -real_scale)
    if imag_count > 0:
        q_imag[is_imag_dominant] = np.where(tile_imag[is_imag_dominant] >= 0.0, imag_scale, -imag_scale)

    return q_real, q_imag, real_scale, imag_scale


def quantize_matrix_tile64_v2(
    w_real: np.ndarray, w_imag: np.ndarray
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    if w_real.shape != w_imag.shape:
        raise ValueError(f"shape mismatch: {w_real.shape} vs {w_imag.shape}")
    if w_real.shape[0] % TILE64 != 0 or w_real.shape[1] % TILE64 != 0:
        raise ValueError(f"tile64 quantization requires dims divisible by {TILE64}, got {w_real.shape}")

    rows, cols = w_real.shape
    tile_rows = rows // TILE64
    tile_cols = cols // TILE64

    q0_real = np.zeros_like(w_real, dtype=np.float32)
    q0_imag = np.zeros_like(w_imag, dtype=np.float32)
    q1_real = np.zeros_like(w_real, dtype=np.float32)
    q1_imag = np.zeros_like(w_imag, dtype=np.float32)

    s0_real = np.zeros((tile_rows, tile_cols), dtype=np.float32)
    s0_imag = np.zeros((tile_rows, tile_cols), dtype=np.float32)
    s1_real = np.zeros((tile_rows, tile_cols), dtype=np.float32)
    s1_imag = np.zeros((tile_rows, tile_cols), dtype=np.float32)

    for tr in range(tile_rows):
        row_slice = slice(tr * TILE64, (tr + 1) * TILE64)
        for tc in range(tile_cols):
            col_slice = slice(tc * TILE64, (tc + 1) * TILE64)

            tile_real = w_real[row_slice, col_slice]
            tile_imag = w_imag[row_slice, col_slice]

            stage0_real, stage0_imag, scale0_real, scale0_imag = quantize_tile64_once(tile_real, tile_imag)
            resid_real = tile_real - stage0_real
            resid_imag = tile_imag - stage0_imag
            stage1_real, stage1_imag, scale1_real, scale1_imag = quantize_tile64_once(resid_real, resid_imag)

            q0_real[row_slice, col_slice] = stage0_real
            q0_imag[row_slice, col_slice] = stage0_imag
            q1_real[row_slice, col_slice] = stage1_real
            q1_imag[row_slice, col_slice] = stage1_imag

            s0_real[tr, tc] = scale0_real
            s0_imag[tr, tc] = scale0_imag
            s1_real[tr, tc] = scale1_real
            s1_imag[tr, tc] = scale1_imag

    return (q0_real, q0_imag, s0_real, s0_imag), (q1_real, q1_imag, s1_real, s1_imag)


def encode_stage_codes(stage_real: np.ndarray, stage_imag: np.ndarray) -> np.ndarray:
    if stage_real.shape != stage_imag.shape:
        raise ValueError(f"shape mismatch: {stage_real.shape} vs {stage_imag.shape}")
    if stage_real.shape[1] % TILE64 != 0:
        raise ValueError(f"tile64 code packing requires cols divisible by {TILE64}, got {stage_real.shape[1]}")

    abs_real = np.abs(stage_real)
    abs_imag = np.abs(stage_imag)
    choose_real = abs_real > abs_imag

    codes = np.empty(stage_real.shape, dtype=np.uint8)
    codes[choose_real] = np.where(stage_real[choose_real] >= 0.0, 1, 0)
    codes[~choose_real] = np.where(stage_imag[~choose_real] >= 0.0, 3, 2)

    rows, cols = codes.shape
    n_blocks = cols // TILE64
    codes = codes.reshape(rows, n_blocks, 4, 16)
    packed = (
        codes[:, :, 0, :]
        | (codes[:, :, 1, :] << 2)
        | (codes[:, :, 2, :] << 4)
        | (codes[:, :, 3, :] << 6)
    ).astype(np.uint8)
    return packed.reshape(rows, n_blocks * 16)


def pack_ifairy64_stage(
    stage_real: np.ndarray,
    stage_imag: np.ndarray,
    scale_real: np.ndarray,
    scale_imag: np.ndarray,
) -> np.ndarray:
    if stage_real.shape != stage_imag.shape:
        raise ValueError(f"shape mismatch: {stage_real.shape} vs {stage_imag.shape}")
    rows, cols = stage_real.shape
    if rows % TILE64 != 0 or cols % TILE64 != 0:
        raise ValueError(f"tile64 packing requires dims divisible by {TILE64}, got {stage_real.shape}")

    n_blocks = cols // TILE64
    if scale_real.shape != (rows // TILE64, n_blocks) or scale_imag.shape != (rows // TILE64, n_blocks):
        raise ValueError(
            f"scale shape mismatch: expected {(rows // TILE64, n_blocks)}, got {scale_real.shape} and {scale_imag.shape}"
        )

    codes = encode_stage_codes(stage_real, stage_imag).reshape(rows, n_blocks, 16)

    out = np.empty((rows, n_blocks, 20), dtype=np.uint8)
    out[:, :, :16] = codes

    scale_real_rows = np.repeat(np.ascontiguousarray(scale_real, dtype=np.float16), TILE64, axis=0)
    scale_imag_rows = np.repeat(np.ascontiguousarray(scale_imag, dtype=np.float16), TILE64, axis=0)
    out[:, :, 16:18] = scale_real_rows.view(np.uint8).reshape(rows, n_blocks, 2)
    out[:, :, 18:20] = scale_imag_rows.view(np.uint8).reshape(rows, n_blocks, 2)

    return out.reshape(rows, n_blocks * 20)


def quantize_linear_to_ifairy64_stages(weight: torch.Tensor, out_target: int, in_target: int) -> dict[str, np.ndarray]:
    a = weight.to(torch.float32).cpu().numpy()
    out_real, in_real = a.shape
    if out_real % 2 != 0 or in_real % 2 != 0:
        raise ValueError(f"linear weight shape must be even, got {a.shape}")

    out_c = out_real // 2
    in_c = in_real // 2

    a11 = a[:out_c, :in_c]
    a12 = a[:out_c, in_c:]
    a21 = a[out_c:, :in_c]
    a22 = a[out_c:, in_c:]

    u_real = pad_complex_matrix(0.5 * (a11 + a22), out_target, in_target)
    u_imag = pad_complex_matrix(0.5 * (a21 - a12), out_target, in_target)
    w_real = pad_complex_matrix(0.5 * (a11 - a22), out_target, in_target)
    w_imag = pad_complex_matrix(0.5 * (a12 + a21), out_target, in_target)

    (u0_real, u0_imag, u0_s_real, u0_s_imag), (u1_real, u1_imag, u1_s_real, u1_s_imag) = quantize_matrix_tile64_v2(
        u_real, u_imag
    )
    (w0_real, w0_imag, w0_s_real, w0_s_imag), (w1_real, w1_imag, w1_s_real, w1_s_imag) = quantize_matrix_tile64_v2(
        w_real, w_imag
    )

    out = {
        "U.s0": pack_ifairy64_stage(u0_real, u0_imag, u0_s_real, u0_s_imag),
        "U.s1": pack_ifairy64_stage(u1_real, u1_imag, u1_s_real, u1_s_imag),
        "W.s0": pack_ifairy64_stage(w0_real, w0_imag, w0_s_real, w0_s_imag),
        "W.s1": pack_ifairy64_stage(w1_real, w1_imag, w1_s_real, w1_s_imag),
    }

    del a
    del u_real, u_imag, w_real, w_imag
    del u0_real, u0_imag, u1_real, u1_imag
    del w0_real, w0_imag, w1_real, w1_imag
    gc.collect()

    return out


def pack_ifairy_stage(stage_real: np.ndarray, stage_imag: np.ndarray, d_real: float, d_imag: float) -> np.ndarray:
    stage_real = np.ascontiguousarray(stage_real, dtype=np.float32)
    stage_imag = np.ascontiguousarray(stage_imag, dtype=np.float32)

    if stage_real.shape != stage_imag.shape:
        raise ValueError(f"shape mismatch: {stage_real.shape} vs {stage_imag.shape}")

    rows, cols = stage_real.shape
    if cols % QK_IFAIRY != 0:
        raise ValueError(f"inner dim {cols} is not divisible by QK_IFAIRY={QK_IFAIRY}")

    mask_real = stage_real != 0.0
    mask_imag = stage_imag != 0.0
    both = mask_real & mask_imag
    if np.any(both):
        abs_real = np.abs(stage_real)
        abs_imag = np.abs(stage_imag)
        choose_real = abs_real > abs_imag
        choose_imag = abs_imag > abs_real
        ties = ~(choose_real | choose_imag)
        if np.any(ties):
            same_sign = (stage_real * stage_imag) >= 0.0
            choose_imag |= ties & same_sign
            choose_real |= ties & (~same_sign)
        mask_real = (mask_real & ~both) | (both & choose_real)
        mask_imag = (mask_imag & ~both) | (both & ~choose_real)

    d_real = 1e-6 if not np.isfinite(d_real) else max(float(d_real), 1e-6)
    d_imag = 1e-6 if not np.isfinite(d_imag) else max(float(d_imag), 1e-6)
    row_all_zero = ~np.any(mask_real | mask_imag, axis=1)

    d_real_arr = np.full(rows, d_real, dtype=np.float32)
    d_imag_arr = np.full(rows, d_imag, dtype=np.float32)
    d_real_arr[row_all_zero] = 0.0
    d_imag_arr[row_all_zero] = 0.0

    codes = np.zeros((rows, cols), dtype=np.uint8)

    real_pos = mask_real & (stage_real >= 0.0)
    real_neg = mask_real & (~real_pos)
    imag_pos = mask_imag & (stage_imag >= 0.0)
    imag_neg = mask_imag & (~imag_pos)

    codes[real_neg] = 0
    codes[real_pos] = 1
    codes[imag_neg] = 2
    codes[imag_pos] = 3

    zero_mask = ~(mask_real | mask_imag)
    prefer_real = d_real_arr <= d_imag_arr
    codes[zero_mask & prefer_real[:, None]] = 1
    codes[zero_mask & (~prefer_real)[:, None]] = 3

    n_blocks = cols // QK_IFAIRY
    codes = codes.reshape(rows, n_blocks, 4, 4, 16)
    packed = (
        codes[:, :, :, 0, :]
        | (codes[:, :, :, 1, :] << 2)
        | (codes[:, :, :, 2, :] << 4)
        | (codes[:, :, :, 3, :] << 6)
    ).astype(np.uint8)
    packed = packed.reshape(rows, n_blocks, 64)

    d_real_bytes = d_real_arr.astype(np.float16).view(np.uint8).reshape(rows, 2)
    d_imag_bytes = d_imag_arr.astype(np.float16).view(np.uint8).reshape(rows, 2)

    out = np.empty((rows, n_blocks, 68), dtype=np.uint8)
    out[:, :, :64] = packed
    out[:, :, 64:66] = d_real_bytes[:, None, :]
    out[:, :, 66:68] = d_imag_bytes[:, None, :]

    return out.reshape(rows, n_blocks * 68)


def quantize_linear_to_ifairy_stages_legacy(weight: torch.Tensor, out_target: int, in_target: int) -> dict[str, np.ndarray]:
    a = weight.to(torch.float32).cpu().numpy()
    out_real, in_real = a.shape
    if out_real % 2 != 0 or in_real % 2 != 0:
        raise ValueError(f"linear weight shape must be even, got {a.shape}")

    out_c = out_real // 2
    in_c = in_real // 2

    a11 = a[:out_c, :in_c]
    a12 = a[:out_c, in_c:]
    a21 = a[out_c:, :in_c]
    a22 = a[out_c:, in_c:]

    u_real = 0.5 * (a11 + a22)
    u_imag = 0.5 * (a21 - a12)
    w_real = 0.5 * (a11 - a22)
    w_imag = 0.5 * (a12 + a21)

    (u0_real, u0_imag, u0_s_real, u0_s_imag), (u1_real, u1_imag, u1_s_real, u1_s_imag) = phase_quant_v2(
        u_real, u_imag
    )
    (w0_real, w0_imag, w0_s_real, w0_s_imag), (w1_real, w1_imag, w1_s_real, w1_s_imag) = phase_quant_v2(
        w_real, w_imag
    )

    u0_real = pad_complex_matrix(u0_real, out_target, in_target)
    u0_imag = pad_complex_matrix(u0_imag, out_target, in_target)
    u1_real = pad_complex_matrix(u1_real, out_target, in_target)
    u1_imag = pad_complex_matrix(u1_imag, out_target, in_target)
    w0_real = pad_complex_matrix(w0_real, out_target, in_target)
    w0_imag = pad_complex_matrix(w0_imag, out_target, in_target)
    w1_real = pad_complex_matrix(w1_real, out_target, in_target)
    w1_imag = pad_complex_matrix(w1_imag, out_target, in_target)

    out = {
        "U.s0": pack_ifairy_stage(u0_real, u0_imag, u0_s_real, u0_s_imag),
        "U.s1": pack_ifairy_stage(u1_real, u1_imag, u1_s_real, u1_s_imag),
        "W.s0": pack_ifairy_stage(w0_real, w0_imag, w0_s_real, w0_s_imag),
        "W.s1": pack_ifairy_stage(w1_real, w1_imag, w1_s_real, w1_s_imag),
    }

    del a
    del u_real, u_imag, w_real, w_imag
    del u0_real, u0_imag, u1_real, u1_imag
    del w0_real, w0_imag, w1_real, w1_imag
    gc.collect()

    return out

def pack_token_embedding(embed: torch.Tensor, hidden_complex: int) -> np.ndarray:
    real = embed[:, :hidden_complex].to(torch.float32)
    imag = embed[:, hidden_complex:].to(torch.float32)

    real_bits = real.to(torch.bfloat16).contiguous().view(torch.int16).to(torch.int32)
    imag_bits = imag.to(torch.bfloat16).contiguous().view(torch.int16).to(torch.int32)

    packed = ((imag_bits << 16) | (real_bits & 0xFFFF)).to(torch.int32).view(torch.float32)
    return packed.cpu().numpy()


def add_optional_vector_tensor(
    writer: gguf.GGUFWriter,
    reader: TensorReader,
    hf_key: str,
    gguf_name: str,
) -> None:
    if not reader.has(hf_key):
        return

    tensor = reader.get(hf_key).to(torch.float32).cpu().numpy().astype(np.float32, copy=False)
    writer.add_tensor(gguf_name, tensor, raw_dtype=gguf.GGMLQuantizationType.F32)
    del tensor
    gc.collect()


def get_rope_theta(config: dict) -> float:
    rope_params = config.get("rope_parameters")
    if isinstance(rope_params, dict) and "rope_theta" in rope_params:
        return float(rope_params["rope_theta"])
    if "rope_theta" in config:
        return float(config["rope_theta"])
    return 10000.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Qwen2-based Fairy2i Hugging Face weights to GGUF")
    parser.add_argument("model_dir", type=Path, help="Path to Qwen2-based Fairy2i model directory")
    parser.add_argument("output_file", type=Path, help="Output GGUF file path")
    parser.add_argument("--residual-steps", type=int, default=2, help="Residual quantization steps (only 2 is supported)")
    parser.add_argument(
        "--output-layer",
        choices=["ifairy", "dense", "both"],
        default="ifairy",
        help="Output projection storage: ifairy (default), dense, or both (for A/B debugging)",
    )
    parser.add_argument(
        "--qk-permute",
        action="store_true",
        help="Enable Llama q/k undo-permute during conversion (disabled by default for Fairy2i)",
    )
    parser.add_argument(
        "--no-attn-bias",
        action="store_true",
        help="Do not export optional attention bias tensors even if present in the HF checkpoint",
    )
    parser.add_argument(
        "--quant-variant",
        choices=["tile64_v2", "legacy"],
        default="tile64_v2",
        help="Quantization/export variant. tile64_v2 matches the training-side QAT kernel; legacy keeps the old 256-wide iFairy packing for comparison.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print conversion progress")
    args = parser.parse_args()

    if args.residual_steps != 2:
        raise ValueError("only --residual-steps 2 is currently supported")

    model_dir: Path = args.model_dir
    output_file: Path = args.output_file
    verbose: bool = args.verbose
    output_layer_mode: str = args.output_layer
    do_qk_permute: bool = args.qk_permute
    export_attn_bias: bool = not args.no_attn_bias
    quant_variant: str = args.quant_variant

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))

    hidden_real = int(config["hidden_size"])
    hidden_complex = hidden_real // 2
    n_layer = int(config["num_hidden_layers"])
    n_head = int(config["num_attention_heads"])
    n_head_kv = int(config.get("num_key_value_heads", n_head))
    rope_theta = get_rope_theta(config)

    ff_real = int(config["intermediate_size"])
    ff_complex = ff_real // 2
    ff_complex_padded = round_up(ff_complex, QK_IFAIRY)

    if verbose:
        print(f"hidden_real={hidden_real}, hidden_complex={hidden_complex}")
        print(f"ff_complex={ff_complex}, ff_complex_padded={ff_complex_padded}")
        print(f"rope_theta={rope_theta}")
        print(f"output_layer_mode={output_layer_mode}, do_qk_permute={do_qk_permute}")
        print(f"export_attn_bias={export_attn_bias}")
        print(f"quant_variant={quant_variant}")

    weight_map = load_weight_map(model_dir)
    reader = TensorReader(model_dir, weight_map)

    writer = gguf.GGUFWriter(str(output_file), arch="fairy2i")
    writer.add_name(config.get("_name_or_path", "Fairy2i-Qwen2"))
    writer.add_context_length(int(config["max_position_embeddings"]))
    writer.add_embedding_length(hidden_complex)
    writer.add_block_count(n_layer)
    writer.add_feed_forward_length(ff_complex_padded)
    writer.add_head_count(n_head)
    writer.add_head_count_kv(n_head_kv)
    writer.add_layer_norm_rms_eps(float(config["rms_norm_eps"]))
    writer.add_rope_freq_base(rope_theta)
    if quant_variant == "legacy":
        writer.add_file_type(gguf.LlamaFileType.MOSTLY_IFAIRY)
    else:
        writer.add_file_type(gguf.LlamaFileType.ALL_F32)
    writer.add_vocab_size(int(config["vocab_size"]))
    writer.add_uint32("fairy2i.quant.residual_steps", args.residual_steps)
    writer.add_string("fairy2i.quant.codebook", "{+/-1,+/-i}")
    writer.add_string("fairy2i.quant.variant", quant_variant)
    if quant_variant == "tile64_v2":
        writer.add_uint32("fairy2i.quant.tile_size", TILE64)
        writer.add_string("fairy2i.quant.scale_stat", "dominant_mean_abs")

    if verbose:
        print("adding token embedding")
    tok_embd = reader.get("model.embed_tokens.weight")
    tok_embd_packed = pack_token_embedding(tok_embd, hidden_complex)
    writer.add_tensor("token_embd", tok_embd_packed, raw_dtype=gguf.GGMLQuantizationType.F32)
    del tok_embd, tok_embd_packed
    gc.collect()

    if verbose:
        print("adding output layers")
    output_norm = reader.get("model.norm.weight").to(torch.float32).cpu().numpy().astype(np.float32, copy=False)
    writer.add_tensor("output_norm", output_norm, raw_dtype=gguf.GGMLQuantizationType.F32)
    del output_norm
    gc.collect()

    output_w = reader.get("lm_head.weight")
    if output_layer_mode in ("ifairy", "both"):
        if verbose:
            print("adding output projection (wide-linear ifairy)")
        output_out_c = output_w.shape[0] // 2
        output_in_c = output_w.shape[1] // 2
        if quant_variant == "tile64_v2":
            output_packed = quantize_linear_to_ifairy64_stages(output_w, output_out_c, output_in_c)
            for stage_name, stage_data in output_packed.items():
                writer.add_tensor(
                    f"output.{stage_name}",
                    stage_data,
                    raw_dtype=gguf.GGMLQuantizationType.IFAIRY64,
                )
        else:
            output_packed = quantize_linear_to_ifairy_stages_legacy(output_w, output_out_c, output_in_c)
            for stage_name, stage_data in output_packed.items():
                writer.add_tensor(
                    f"output.{stage_name}",
                    stage_data,
                    raw_dtype=gguf.GGMLQuantizationType.F16_I2,
                )
        del output_packed

    if output_layer_mode in ("dense", "both"):
        if verbose:
            print("adding output projection (dense f16)")
        output_dense = output_w.to(torch.float16).cpu().numpy()
        writer.add_tensor("output", output_dense, raw_dtype=gguf.GGMLQuantizationType.F16)
        del output_dense

    del output_w
    gc.collect()

    linear_specs = [
        ("self_attn.q_proj.weight", "attn_q", "q"),
        ("self_attn.k_proj.weight", "attn_k", "k"),
        ("self_attn.v_proj.weight", "attn_v", None),
        ("self_attn.o_proj.weight", "attn_output", None),
        ("mlp.gate_proj.weight", "ffn_gate", None),
        ("mlp.up_proj.weight", "ffn_up", None),
        ("mlp.down_proj.weight", "ffn_down", None),
    ]

    bias_specs = [
        ("self_attn.q_proj.bias", "attn_q.bias"),
        ("self_attn.k_proj.bias", "attn_k.bias"),
        ("self_attn.v_proj.bias", "attn_v.bias"),
        ("self_attn.o_proj.bias", "attn_output.bias"),
    ]

    for il in range(n_layer):
        if verbose:
            print(f"layer {il + 1}/{n_layer}")

        attn_norm = reader.get(f"model.layers.{il}.input_layernorm.weight").to(torch.float32).cpu().numpy().astype(np.float32, copy=False)
        ffn_norm = reader.get(f"model.layers.{il}.post_attention_layernorm.weight").to(torch.float32).cpu().numpy().astype(np.float32, copy=False)
        writer.add_tensor(f"blk.{il}.attn_norm", attn_norm, raw_dtype=gguf.GGMLQuantizationType.F32)
        writer.add_tensor(f"blk.{il}.ffn_norm", ffn_norm, raw_dtype=gguf.GGMLQuantizationType.F32)
        del attn_norm, ffn_norm

        for hf_suffix, gguf_base, permute_kind in linear_specs:
            hf_key = f"model.layers.{il}.{hf_suffix}"
            w = reader.get(hf_key)

            if permute_kind == "q" and do_qk_permute:
                w = undo_llama_permute(w, n_head)
            elif permute_kind == "k" and do_qk_permute:
                w = undo_llama_permute(w, n_head_kv)

            out_c = w.shape[0] // 2
            in_c = w.shape[1] // 2
            out_target = ff_complex_padded if out_c == ff_complex else out_c
            in_target = ff_complex_padded if in_c == ff_complex else in_c

            if quant_variant == "tile64_v2":
                packed = quantize_linear_to_ifairy64_stages(w, out_target, in_target)
                for stage_name, stage_data in packed.items():
                    writer.add_tensor(
                        f"blk.{il}.{gguf_base}.{stage_name}",
                        stage_data,
                        raw_dtype=gguf.GGMLQuantizationType.IFAIRY64,
                    )
            else:
                packed = quantize_linear_to_ifairy_stages_legacy(w, out_target, in_target)
                for stage_name, stage_data in packed.items():
                    writer.add_tensor(
                        f"blk.{il}.{gguf_base}.{stage_name}",
                        stage_data,
                        raw_dtype=gguf.GGMLQuantizationType.F16_I2,
                    )

            del w, packed
            gc.collect()

        if export_attn_bias:
            for hf_suffix, gguf_name in bias_specs:
                add_optional_vector_tensor(
                    writer,
                    reader,
                    f"model.layers.{il}.{hf_suffix}",
                    f"blk.{il}.{gguf_name}",
                )

        gc.collect()

    set_vocab_qwen2(model_dir, config, writer)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"GGUF saved to: {output_file}")


if __name__ == "__main__":
    main()
