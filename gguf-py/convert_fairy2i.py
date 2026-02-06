#!/usr/bin/env python3

import argparse
import gc
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from safetensors import safe_open

import gguf
from gguf.constants import QK_IFAIRY


def round_up(x: int, base: int) -> int:
    return ((x + base - 1) // base) * base


def set_vocab_llama_hf(model_dir: Path, writer: gguf.GGUFWriter) -> None:
    vocab = gguf.LlamaHfVocab(model_dir)
    tokens = []
    scores = []
    toktypes = []

    for text, score, toktype in vocab.all_tokens():
        tokens.append(text)
        scores.append(score)
        toktypes.append(toktype)

    writer.add_tokenizer_model("llama")
    writer.add_tokenizer_pre("default")
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(toktypes)

    special_vocab = gguf.SpecialVocab(model_dir, n_vocab=len(tokens))
    special_vocab.add_to_gguf(writer)


def set_vocab(model_dir: Path, writer: gguf.GGUFWriter) -> None:
    tokenizer_json = model_dir / "tokenizer.json"
    if not tokenizer_json.is_file():
        raise FileNotFoundError(f"tokenizer.json not found in {model_dir}")

    set_vocab_llama_hf(model_dir, writer)

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
    choose_imag = ~choose_real

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
        choose_real = np.abs(stage_real) >= np.abs(stage_imag)
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


def quantize_linear_to_ifairy_stages(weight: torch.Tensor, out_target: int, in_target: int) -> dict[str, np.ndarray]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Fairy2i-W2 Hugging Face weights to GGUF")
    parser.add_argument("model_dir", type=Path, help="Path to Fairy2i-W2 model directory")
    parser.add_argument("output_file", type=Path, help="Output GGUF file path")
    parser.add_argument("--residual-steps", type=int, default=2, help="Residual quantization steps (only 2 is supported)")
    parser.add_argument("--verbose", action="store_true", help="Print conversion progress")
    args = parser.parse_args()

    if args.residual_steps != 2:
        raise ValueError("only --residual-steps 2 is currently supported")

    model_dir: Path = args.model_dir
    output_file: Path = args.output_file
    verbose: bool = args.verbose

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))

    hidden_real = int(config["hidden_size"])
    hidden_complex = hidden_real // 2
    n_layer = int(config["num_hidden_layers"])
    n_head = int(config["num_attention_heads"])
    n_head_kv = int(config.get("num_key_value_heads", n_head))

    ff_real = int(config["intermediate_size"])
    ff_complex = ff_real // 2
    ff_complex_padded = round_up(ff_complex, QK_IFAIRY)

    if verbose:
        print(f"hidden_real={hidden_real}, hidden_complex={hidden_complex}")
        print(f"ff_complex={ff_complex}, ff_complex_padded={ff_complex_padded}")

    weight_map = load_weight_map(model_dir)
    reader = TensorReader(model_dir, weight_map)

    writer = gguf.GGUFWriter(str(output_file), arch="fairy2i")
    writer.add_name(config.get("_name_or_path", "Fairy2i-W2"))
    writer.add_context_length(int(config["max_position_embeddings"]))
    writer.add_embedding_length(hidden_complex)
    writer.add_block_count(n_layer)
    writer.add_feed_forward_length(ff_complex_padded)
    writer.add_head_count(n_head)
    writer.add_head_count_kv(n_head_kv)
    writer.add_layer_norm_rms_eps(float(config["rms_norm_eps"]))
    writer.add_rope_freq_base(float(config.get("rope_theta", 10000.0)))
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_IFAIRY)
    writer.add_vocab_size(int(config["vocab_size"]))
    writer.add_uint32("fairy2i.quant.residual_steps", args.residual_steps)
    writer.add_string("fairy2i.quant.codebook", "{+/-1,+/-i}")

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

    if verbose:
        print("adding output projection (wide-linear ifairy)")
    output_w = reader.get("lm_head.weight")
    output_out_c = output_w.shape[0] // 2
    output_in_c = output_w.shape[1] // 2
    output_packed = quantize_linear_to_ifairy_stages(output_w, output_out_c, output_in_c)
    for stage_name, stage_data in output_packed.items():
        writer.add_tensor(
            f"output.{stage_name}",
            stage_data,
            raw_dtype=gguf.GGMLQuantizationType.F16_I2,
        )
    del output_w, output_packed
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

            if permute_kind == "q":
                w = undo_llama_permute(w, n_head)
            elif permute_kind == "k":
                w = undo_llama_permute(w, n_head_kv)

            out_c = w.shape[0] // 2
            in_c = w.shape[1] // 2
            out_target = ff_complex_padded if out_c == ff_complex else out_c
            in_target = ff_complex_padded if in_c == ff_complex else in_c

            packed = quantize_linear_to_ifairy_stages(w, out_target, in_target)
            for stage_name, stage_data in packed.items():
                writer.add_tensor(
                    f"blk.{il}.{gguf_base}.{stage_name}",
                    stage_data,
                    raw_dtype=gguf.GGMLQuantizationType.F16_I2,
                )

            del w, packed
            gc.collect()

    set_vocab(model_dir, writer)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"GGUF saved to: {output_file}")


if __name__ == "__main__":
    main()
