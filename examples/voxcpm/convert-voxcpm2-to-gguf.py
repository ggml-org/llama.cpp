#!/usr/bin/env python3
"""Convert openbmb/VoxCPM2 to a single GGUF file.

VoxCPM2 is a tokenizer-free multilingual TTS model with four stages:
  LocEnc  (12-layer transformer encoder)
  TSLM    (MiniCPM-4 LM backbone, 28 layers)
  RALM    (residual LM, 8 layers, same backbone arch)
  LocDiT  (12-layer diffusion transformer)
plus AudioVAE V2 (asymmetric audio autoencoder, 16 kHz → 48 kHz).

Source files:
  model.safetensors  — main weights (LM + encoder + DiT + projections)
  audiovae.pth       — AudioVAE V2 weights (PyTorch format)
  config.json        — all component configs in one flat JSON

Output tensor prefixes in the GGUF file:
  lm.*        TSLM + RALM backbone (base_lm.* in HF)
  enc.*       LocEnc
  dit.*       LocDiT
  vae.*       AudioVAE V2
  proj.*      inter-component projection layers

Multiple --q-type values are accepted; model is loaded once and all
variants written in parallel (same approach as convert-sdxl-to-gguf.py).

Usage:
  python convert-voxcpm2-to-gguf.py -m ./VoxCPM2 --q-type f16 Q4_K_M
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from gguf import GGUFWriter, GGMLQuantizationType  # noqa: E402
from gguf.quants import quantize_q8_0, can_quantize_to_q8_0  # noqa: E402

try:
    from safetensors.torch import load_file as safetensors_load
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


# ---------------------------------------------------------------------------
# Quantization registry  (shared with convert-sdxl-to-gguf.py)
# ---------------------------------------------------------------------------

_FTYPE_VALUE: dict[str, int] = {
    "f32": 0, "f16": 1, "q8_0": 7,
    "Q2_K": 10, "Q3_K_S": 11, "Q3_K_M": 12, "Q3_K_L": 13,
    "Q4_K_S": 14, "Q4_K_M": 15, "Q5_K_S": 16, "Q5_K_M": 17, "Q6_K": 18,
}

_K_QUANTS: frozenset[str] = frozenset(
    {"Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_K_S", "Q4_K_M", "Q5_K_S", "Q5_K_M", "Q6_K"}
)

_ALL_Q_TYPES = ["f32", "f16", "q8_0"] + sorted(_K_QUANTS)


def run_llama_quantize(f16_path: str, out_path: str, q_type: str, binary: str) -> str:
    if not os.path.isfile(binary):
        raise FileNotFoundError(
            f"llama-quantize not found at {binary!r}. "
            "Build with: cmake --build build --target llama-quantize"
        )
    cmd = [binary, f16_path, out_path, q_type]
    print(f"  [kquant] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return out_path


# ---------------------------------------------------------------------------
# Tensor loading
# ---------------------------------------------------------------------------

def load_safetensors_or_pth(path: str) -> dict[str, torch.Tensor]:
    if path.endswith(".pth") or path.endswith(".pt") or path.endswith(".bin"):
        return torch.load(path, map_location="cpu")
    if not HAS_SAFETENSORS:
        raise RuntimeError("Install safetensors: pip install safetensors")
    return safetensors_load(path)


def load_main_weights(model_dir: str) -> dict[str, torch.Tensor]:
    """Load the primary model weights (model.safetensors or sharded)."""
    d = model_dir
    if HAS_SAFETENSORS:
        shards = sorted(glob.glob(os.path.join(d, "model-*-of-*.safetensors")))
        if shards:
            print(f"  Loading {len(shards)} shards from model dir")
            out: dict[str, torch.Tensor] = {}
            for s in shards:
                out.update(safetensors_load(s))
            return out
        st = os.path.join(d, "model.safetensors")
        if os.path.exists(st):
            print(f"  Loading {st}")
            return safetensors_load(st)
    for fname in ("model.bin", "pytorch_model.bin"):
        p = os.path.join(d, fname)
        if os.path.exists(p):
            print(f"  Loading {p}")
            return torch.load(p, map_location="cpu")
    raise FileNotFoundError(f"No main weights found in {d!r}")


def load_audiovae_weights(model_dir: str) -> dict[str, torch.Tensor]:
    for fname in ("audiovae.pth", "audio_vae.pth", "audiovae.safetensors"):
        p = os.path.join(model_dir, fname)
        if os.path.exists(p):
            print(f"  Loading AudioVAE: {p}")
            return load_safetensors_or_pth(p)
    raise FileNotFoundError(
        f"AudioVAE weights not found in {model_dir!r}. "
        "Expected audiovae.pth or audiovae.safetensors"
    )


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------

def _quantize_f32_array(
    arr: np.ndarray, name: str, q_type: str, verbose: bool = True
) -> tuple[np.ndarray, GGMLQuantizationType | None]:
    n_dims = arr.ndim

    if n_dims == 1:
        if verbose:
            print(f"  {name} | f32 | {arr.shape}")
        return arr, None   # already f32

    if n_dims >= 3:        # conv/rnn kernels
        out = arr.astype(np.float16)
        if verbose:
            print(f"  {name} | f16 (conv/rnn) | {out.shape}")
        return out, None

    # 2-D weight matrices
    if q_type == "f32":
        if verbose:
            print(f"  {name} | f32 | {arr.shape}")
        return arr, None

    if q_type == "f16":
        out = arr.astype(np.float16)
        if verbose:
            print(f"  {name} | f16 | {out.shape}")
        return out, None

    if q_type == "q8_0":
        if can_quantize_to_q8_0(arr):
            out = quantize_q8_0(arr)
            if verbose:
                print(f"  {name} | q8_0 | {arr.shape}")
            return out, GGMLQuantizationType.Q8_0
        out = arr.astype(np.float16)
        if verbose:
            print(f"  {name} | f16 (q8_0 fallback row={arr.shape[-1]}) | {arr.shape}")
        return out, None

    raise ValueError(f"Unsupported q_type: {q_type!r}")


def add_tensor_to_all(
    writers: dict[str, GGUFWriter], name: str, data: torch.Tensor
) -> None:
    arr_f32 = data.squeeze().float().numpy()  # single conversion
    first = True
    for q_type, fout in writers.items():
        arr, raw_dtype = _quantize_f32_array(arr_f32, name, q_type, verbose=first)
        fout.add_tensor(name, arr, raw_dtype=raw_dtype)
        first = False


def _iter_tensors(tensors: dict[str, torch.Tensor], label: str):
    total = len(tensors)
    for i, (name, data) in enumerate(tensors.items(), 1):
        print(f"  [{i}/{total}]", end=" ")
        yield name, data
    print(f"  {label}: {total} tensors done")


# ---------------------------------------------------------------------------
# Tensor name mapping
# ---------------------------------------------------------------------------

# HF prefix → GGUF prefix for simple renames
_PREFIX_MAP = [
    ("base_lm.model.",         "lm."),   # some HF variants wrap with .model.
    ("base_lm.",               "lm."),
    ("encoder.",               "enc."),
    ("dit.",                   "dit."),
    ("enc_to_lm_proj",         "proj.enc_to_lm"),
    ("lm_to_dit_proj",         "proj.lm_to_dit"),
    ("res_to_dit_proj",        "proj.res_to_dit"),
    ("fusion_concat_proj",     "proj.fusion_concat"),
    ("stop_proj",              "proj.stop"),
    ("stop_head",              "proj.stop_head"),
]

# Common sub-tensor renames (applied after prefix mapping)
_SUBST = [
    ("self_attn.qkv_proj",  "attn_qkv"),
    ("self_attn.q_proj",    "attn_q"),
    ("self_attn.k_proj",    "attn_k"),
    ("self_attn.v_proj",    "attn_v"),
    ("self_attn.o_proj",    "attn_out"),
    ("cross_attn.q_proj",   "cross_attn_q"),
    ("cross_attn.k_proj",   "cross_attn_k"),
    ("cross_attn.v_proj",   "cross_attn_v"),
    ("cross_attn.o_proj",   "cross_attn_out"),
    ("mlp.gate_up_proj",    "ffn_gate_up"),
    ("mlp.up_proj",         "ffn_up"),
    ("mlp.gate_proj",       "ffn_gate"),
    ("mlp.down_proj",       "ffn_down"),
    ("embed_tokens",        "token_embd"),
    ("input_layernorm",     "ln1"),
    ("post_attention_layernorm", "ln2"),
    ("norm",                "norm"),
    (".layers.",            ".blk."),
]

# AudioVAE: top-level renames before vae. prefix
_VAE_SUBST = [
    ("encoder.",    "encoder."),
    ("decoder.",    "decoder."),
    (".weight",     ".weight"),
    (".bias",       ".bias"),
]


def get_main_tensor_name(hf_name: str) -> str | None:
    """Map a tensor name from model.safetensors to a GGUF name."""
    # Skip LoRA and compiled-model artifacts
    if "lora_" in hf_name or "._orig_mod." in hf_name:
        return None

    name = hf_name

    # Apply prefix map (first match wins)
    for hf_pre, gguf_pre in _PREFIX_MAP:
        if name.startswith(hf_pre):
            name = gguf_pre + name[len(hf_pre):]
            break

    # Apply sub-tensor substitutions
    for old, new in _SUBST:
        name = name.replace(old, new)

    return name


def get_vae_tensor_name(hf_name: str) -> str:
    """Map an AudioVAE tensor name to vae.* GGUF name."""
    name = hf_name
    # AudioVAE uses Conv1d / ConvTranspose1d kernels and GRU/LSTM for audio
    # Keep names as-is but apply vae. prefix; they're readable enough.
    return f"vae.{name}"


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def write_metadata(fout: GGUFWriter, cfg: dict, model_name: str) -> None:
    fout.add_name(model_name)
    fout.add_description("VoxCPM2 TTS model (LocEnc + TSLM/RALM + LocDiT + AudioVAE)")

    lm = cfg.get("lm_config", {})
    if lm:
        fout.add_uint32("voxcpm2.lm.vocab_size",         lm["vocab_size"])
        fout.add_uint32("voxcpm2.lm.embedding_length",   lm["hidden_size"])
        fout.add_uint32("voxcpm2.lm.feed_forward_length",lm["intermediate_size"])
        fout.add_uint32("voxcpm2.lm.block_count",        lm["num_hidden_layers"])
        fout.add_uint32("voxcpm2.lm.attention.head_count",    lm["num_attention_heads"])
        fout.add_uint32("voxcpm2.lm.attention.head_count_kv", lm["num_key_value_heads"])
        fout.add_float32("voxcpm2.lm.attention.layer_norm_rms_epsilon", lm["rms_norm_eps"])
        fout.add_uint32("voxcpm2.lm.context_length",     lm["max_position_embeddings"])
        if lm.get("scale_emb"):
            fout.add_float32("voxcpm2.lm.scale_emb",    float(lm["scale_emb"]))
        if lm.get("scale_depth"):
            fout.add_float32("voxcpm2.lm.scale_depth",  float(lm["scale_depth"]))
        if lm.get("dim_model_base"):
            fout.add_uint32("voxcpm2.lm.dim_model_base", lm["dim_model_base"])

    enc = cfg.get("encoder_config", {})
    if enc:
        fout.add_uint32("voxcpm2.enc.embedding_length",   enc["hidden_dim"])
        fout.add_uint32("voxcpm2.enc.feed_forward_length",enc["ffn_dim"])
        fout.add_uint32("voxcpm2.enc.attention.head_count", enc["num_heads"])
        fout.add_uint32("voxcpm2.enc.block_count",        enc["num_layers"])

    dit = cfg.get("dit_config", {})
    if dit:
        fout.add_uint32("voxcpm2.dit.embedding_length",   dit["hidden_dim"])
        fout.add_uint32("voxcpm2.dit.feed_forward_length",dit["ffn_dim"])
        fout.add_uint32("voxcpm2.dit.attention.head_count", dit["num_heads"])
        fout.add_uint32("voxcpm2.dit.block_count",        dit["num_layers"])
        cfm = dit.get("cfm_config", {})
        if cfm.get("solver"):
            fout.add_string("voxcpm2.dit.cfm_solver", cfm["solver"])

    vae = cfg.get("audio_vae_config", {})
    if vae:
        fout.add_uint32("voxcpm2.vae.encoder_dim",    vae["encoder_dim"])
        fout.add_uint32("voxcpm2.vae.latent_dim",     vae["latent_dim"])
        fout.add_uint32("voxcpm2.vae.decoder_dim",    vae["decoder_dim"])
        fout.add_uint32("voxcpm2.vae.sample_rate",    vae["sample_rate"])
        fout.add_uint32("voxcpm2.vae.out_sample_rate",vae["out_sample_rate"])
        fout.add_array("voxcpm2.vae.encoder_rates",   list(vae["encoder_rates"]))
        fout.add_array("voxcpm2.vae.decoder_rates",   list(vae["decoder_rates"]))

    if cfg.get("patch_size"):
        fout.add_uint32("voxcpm2.patch_size", cfg["patch_size"])
    if cfg.get("feat_dim"):
        fout.add_uint32("voxcpm2.feat_dim", cfg["feat_dim"])
    if cfg.get("residual_lm_num_layers"):
        fout.add_uint32("voxcpm2.ralm.block_count", cfg["residual_lm_num_layers"])


# ---------------------------------------------------------------------------
# Component writers
# ---------------------------------------------------------------------------

def add_main_model(writers: dict[str, GGUFWriter], model_dir: str) -> None:
    print("\n=== Main model weights (LM + LocEnc + LocDiT + projections) ===")
    tensors = load_main_weights(model_dir)
    for hf_name, data in _iter_tensors(tensors, "main"):
        gguf_name = get_main_tensor_name(hf_name)
        if gguf_name is None:
            print(f"skipping {hf_name}")
            continue
        add_tensor_to_all(writers, gguf_name, data)


def add_audio_vae(writers: dict[str, GGUFWriter], model_dir: str) -> None:
    print("\n=== AudioVAE V2 ===")
    tensors = load_audiovae_weights(model_dir)
    for hf_name, data in _iter_tensors(tensors, "audiovae"):
        add_tensor_to_all(writers, get_vae_tensor_name(hf_name), data)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Convert VoxCPM2 HuggingFace model to one or more GGUF files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python convert-voxcpm2-to-gguf.py -m ./VoxCPM2
  python convert-voxcpm2-to-gguf.py -m ./VoxCPM2 --q-type f16 q8_0 Q4_K_M
""",
    )
    ap.add_argument("-m", "--model-dir", required=True,
                    help="Path to the VoxCPM2 model directory")
    ap.add_argument("-o", "--output-dir", default=None,
                    help="Output directory (default: model dir)")
    ap.add_argument(
        "--q-type",
        nargs="+",
        choices=_ALL_Q_TYPES,
        default=["f16"],
        metavar="TYPE",
        help="One or more weight types (default: f16)",
    )
    ap.add_argument(
        "--llama-quantize",
        default="./llama-quantize",
        metavar="PATH",
        help="Path to llama-quantize binary (needed for K-quant types)",
    )
    ap.add_argument("--no-audio-vae", action="store_true",
                    help="Skip AudioVAE V2 weights")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    seen: set[str] = set()
    q_types: list[str] = []
    for qt in args.q_type:
        if qt not in seen:
            seen.add(qt)
            q_types.append(qt)

    model_dir = args.model_dir
    output_dir = args.output_dir or model_dir
    os.makedirs(output_dir, exist_ok=True)

    python_types = [qt for qt in q_types if qt not in _K_QUANTS]
    k_quant_types = [qt for qt in q_types if qt in _K_QUANTS]

    f16_is_temp = bool(k_quant_types) and "f16" not in python_types
    if f16_is_temp:
        python_types.append("f16")

    def out_path(qt: str, temp: bool = False) -> str:
        if temp:
            return os.path.join(output_dir, "voxcpm2-f16-tmp.gguf")
        return os.path.join(output_dir, f"voxcpm2-{qt.lower()}.gguf")

    paths = {
        qt: out_path(qt, temp=(qt == "f16" and f16_is_temp))
        for qt in python_types
    }

    # Load config
    cfg_path = os.path.join(model_dir, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    model_name = cfg.get("_name_or_path", os.path.basename(model_dir))

    # Open writers
    writers: dict[str, GGUFWriter] = {}
    for qt in python_types:
        fout = GGUFWriter(path=paths[qt], arch="voxcpm2")
        write_metadata(fout, cfg, model_name)
        fout.add_file_type(_FTYPE_VALUE[qt])
        writers[qt] = fout

    types_label = ", ".join(python_types) + (
        f"  [then K-quants: {', '.join(k_quant_types)}]" if k_quant_types else ""
    )
    print(f"Writing types: {types_label}")

    # Single pass — all writers receive every tensor
    add_main_model(writers, model_dir)
    if not args.no_audio_vae:
        add_audio_vae(writers, model_dir)

    output_files = [paths[qt] for qt in python_types if not (qt == "f16" and f16_is_temp)]

    for qt, fout in writers.items():
        print(f"\nFinalizing {paths[qt]}")
        fout.write_header_to_file()
        fout.write_kv_data_to_file()
        fout.write_tensors_to_file()
        fout.close()

    # Parallel K-quant pass
    if k_quant_types:
        f16_path = paths["f16"]
        n_workers = min(len(k_quant_types), os.cpu_count() or 1)
        print(f"\nK-quant pass: {len(k_quant_types)} type(s) × {n_workers} worker(s)")
        try:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                future_to_kq = {
                    pool.submit(
                        run_llama_quantize,
                        f16_path, out_path(kq), kq, args.llama_quantize
                    ): kq
                    for kq in k_quant_types
                }
                for future in as_completed(future_to_kq):
                    output_files.append(future.result())
        finally:
            if f16_is_temp and os.path.exists(f16_path):
                os.unlink(f16_path)
                print(f"Removed intermediate: {f16_path}")

    print("\nDone. Output files:")
    for p in sorted(output_files):
        print(f"  {p}")


if __name__ == "__main__":
    main()
