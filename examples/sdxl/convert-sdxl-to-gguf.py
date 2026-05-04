#!/usr/bin/env python3
"""Convert a HuggingFace SDXL-Turbo model directory to one or more GGUF files.

The output packs all four components into each file with prefixed tensor names:
  te1.*   - CLIP ViT-L/14 text encoder (text_encoder/)
  te2.*   - OpenCLIP ViT-bigG/14 text encoder (text_encoder_2/)
  vae.*   - Variational Autoencoder (vae/)
  unet.*  - Denoising U-Net (unet/)

Multiple --q-type values are accepted; the model is loaded once and all
requested variants are written in a single pass:

  python convert-sdxl-to-gguf.py -m ./sdxl-turbo --q-type f16 q8_0 Q4_K_M

Weight type rules (applied per tensor):
  1D (biases, norms)   → always f32
  4D (conv kernels)    → always f16
  2D (weight matrices) → --q-type value; q8_0 falls back to f16 for
                         rows not divisible by 32

Python-native types (no binary needed):  f32  f16  q8_0
K-quant types (require llama-quantize):  Q2_K  Q3_K_S  Q3_K_M  Q3_K_L
                                         Q4_K_S  Q4_K_M  Q5_K_S  Q5_K_M  Q6_K

For K-quants an f16 GGUF is written first (reusing one already requested if
available), llama-quantize is called once per K-quant type, then the
temporary f16 file is removed automatically.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Allow running from anywhere inside the llama.cpp tree
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
# Quantization type registry
# ---------------------------------------------------------------------------

_FTYPE_VALUE: dict[str, int] = {
    "f32":    0,
    "f16":    1,
    "q8_0":   7,
    "Q2_K":   10,
    "Q3_K_S": 11,
    "Q3_K_M": 12,
    "Q3_K_L": 13,
    "Q4_K_S": 14,
    "Q4_K_M": 15,
    "Q5_K_S": 16,
    "Q5_K_M": 17,
    "Q6_K":   18,
}

_K_QUANTS: frozenset[str] = frozenset(
    {"Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_K_S", "Q4_K_M", "Q5_K_S", "Q5_K_M", "Q6_K"}
)

_ALL_Q_TYPES = ["f32", "f16", "q8_0"] + sorted(_K_QUANTS)


def run_llama_quantize(f16_path: str, out_path: str, q_type: str, binary: str) -> None:
    if not os.path.isfile(binary):
        raise FileNotFoundError(
            f"llama-quantize binary not found at {binary!r}. "
            "Build it with: cmake --build build --target llama-quantize  "
            "or pass --llama-quantize /path/to/llama-quantize"
        )
    cmd = [binary, f16_path, out_path, q_type]
    print(f"\nRunning: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Tensor loading
# ---------------------------------------------------------------------------

def load_tensors(component_dir: str) -> dict[str, torch.Tensor]:
    d = component_dir
    candidates = [
        os.path.join(d, "model.safetensors"),
        os.path.join(d, "diffusion_pytorch_model.safetensors"),
        os.path.join(d, "diffusion_pytorch_model.fp16.safetensors"),
    ]
    for path in candidates:
        if os.path.exists(path):
            if not HAS_SAFETENSORS:
                raise RuntimeError(
                    f"Found {path} but 'safetensors' package is not installed. "
                    "Run: pip install safetensors"
                )
            print(f"  Loading {path}")
            return safetensors_load(path)

    for fname in ("pytorch_model.bin", "diffusion_pytorch_model.bin"):
        path = os.path.join(d, fname)
        if os.path.exists(path):
            print(f"  Loading {path}")
            return torch.load(path, map_location="cpu")

    raise FileNotFoundError(
        f"No model weights found in {component_dir!r}. "
        "Expected model.safetensors, diffusion_pytorch_model.safetensors, or pytorch_model.bin"
    )


def load_config(component_dir: str) -> dict[str, Any]:
    path = os.path.join(component_dir, "config.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Dtype / quantization logic
# ---------------------------------------------------------------------------

def convert_tensor(
    data: torch.Tensor, name: str, q_type: str, verbose: bool = True
) -> tuple[np.ndarray, GGMLQuantizationType | None]:
    """Return (array, raw_dtype) applying smart per-tensor dtype rules.

    1D (bias, norm, scalar) → f32, None
    4D (conv kernel)        → f16, None  (GGML conv has no f32 kernel support)
    2D (weight matrix)      → q_type; q8_0 falls back to f16 on incompatible shape
    """
    arr = data.squeeze().float().numpy()
    n_dims = arr.ndim

    if n_dims == 1:
        out = arr.astype(np.float32)
        if verbose:
            print(f"  {name} | f32 | {out.shape}")
        return out, None

    if n_dims == 4:
        out = arr.astype(np.float16)
        if verbose:
            print(f"  {name} | f16 (conv) | {out.shape}")
        return out, None

    if q_type == "f32":
        out = arr.astype(np.float32)
        if verbose:
            print(f"  {name} | f32 | {out.shape}")
        return out, None

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
        else:
            out = arr.astype(np.float16)
            if verbose:
                print(f"  {name} | f16 (q8_0 fallback, row={arr.shape[-1]}) | {arr.shape}")
            return out, None

    raise ValueError(f"Unsupported q_type: {q_type!r}")


def add_tensor_to_all(
    writers: dict[str, GGUFWriter], name: str, data: torch.Tensor
) -> None:
    """Convert and write a tensor to every open GGUF writer in one pass."""
    first = True
    for q_type, fout in writers.items():
        arr, raw_dtype = convert_tensor(data, name, q_type, verbose=first)
        fout.add_tensor(name, arr, raw_dtype=raw_dtype)
        first = False


# ---------------------------------------------------------------------------
# Tensor name mapping: Text Encoders
# ---------------------------------------------------------------------------

_TE_SKIP = frozenset([
    "logit_scale",
    "text_model.embeddings.position_ids",
])


def get_te_tensor_name(hf_name: str, prefix: str) -> str | None:
    if hf_name in _TE_SKIP:
        return None

    name = hf_name
    name = name.replace("text_model.encoder.layers.", "blk.")
    name = name.replace("text_model.embeddings.", "")
    name = name.replace("text_model.final_layer_norm.", "post_ln.")
    name = name.replace("text_model.", "")
    name = name.replace("embeddings.", "")
    name = name.replace("self_attn.", "attn_")
    name = re.sub(r"_proj\b", "", name)
    name = name.replace("layer_norm1", "ln1")
    name = name.replace("layer_norm2", "ln2")
    name = name.replace("layernorm", "ln")
    name = name.replace("layer_norm", "ln")
    name = name.replace("mlp.fc1", "ffn_down")
    name = name.replace("mlp.fc2", "ffn_up")
    name = name.replace("token_embedding", "token_embd")
    name = name.replace("position_embedding", "pos_embd")
    name = name.replace("text_projection", "proj")
    return f"{prefix}.{name}"


# ---------------------------------------------------------------------------
# Tensor name mapping: VAE
# ---------------------------------------------------------------------------

def get_vae_tensor_name(hf_name: str) -> str:
    name = hf_name

    name = re.sub(r"encoder\.down_blocks\.(\d+)\.resnets\.(\d+)", r"encoder.down.\1.res.\2", name)
    name = re.sub(r"encoder\.down_blocks\.(\d+)\.downsamplers\.0", r"encoder.down.\1.downsample", name)
    name = re.sub(r"encoder\.mid_block\.resnets\.0\b", "encoder.mid.res0", name)
    name = re.sub(r"encoder\.mid_block\.resnets\.1\b", "encoder.mid.res1", name)
    name = re.sub(r"encoder\.mid_block\.attentions\.0\b", "encoder.mid.attn", name)
    name = name.replace("encoder.conv_norm_out", "encoder.norm_out")

    name = re.sub(r"decoder\.up_blocks\.(\d+)\.resnets\.(\d+)", r"decoder.up.\1.res.\2", name)
    name = re.sub(r"decoder\.up_blocks\.(\d+)\.upsamplers\.0", r"decoder.up.\1.upsample", name)
    name = re.sub(r"decoder\.mid_block\.resnets\.0\b", "decoder.mid.res0", name)
    name = re.sub(r"decoder\.mid_block\.resnets\.1\b", "decoder.mid.res1", name)
    name = re.sub(r"decoder\.mid_block\.attentions\.0\b", "decoder.mid.attn", name)
    name = name.replace("decoder.conv_norm_out", "decoder.norm_out")

    name = name.replace(".processor.", ".")
    name = re.sub(r"\.to_q\b", ".q", name)
    name = re.sub(r"\.to_k\b", ".k", name)
    name = re.sub(r"\.to_v\b", ".v", name)
    name = re.sub(r"\.to_out\.0\b", ".out", name)

    return f"vae.{name}"


# ---------------------------------------------------------------------------
# Tensor name mapping: UNet
# ---------------------------------------------------------------------------

def get_unet_tensor_name(hf_name: str) -> str:
    name = hf_name

    name = name.replace("time_embedding.", "time_embed.")
    name = name.replace("add_embedding.", "add_embed.")

    name = re.sub(r"down_blocks\.(\d+)\.resnets\.(\d+)", r"down.\1.res.\2", name)
    name = re.sub(
        r"down_blocks\.(\d+)\.attentions\.(\d+)\.transformer_blocks\.(\d+)",
        r"down.\1.attn.\2.blk.\3",
        name,
    )
    name = re.sub(r"down_blocks\.(\d+)\.downsamplers\.0\b", r"down.\1.downsample", name)

    name = re.sub(r"mid_block\.resnets\.0\b", "mid.res0", name)
    name = re.sub(r"mid_block\.resnets\.1\b", "mid.res1", name)
    name = re.sub(
        r"mid_block\.attentions\.0\.transformer_blocks\.(\d+)",
        r"mid.attn.blk.\1",
        name,
    )

    name = re.sub(r"up_blocks\.(\d+)\.resnets\.(\d+)", r"up.\1.res.\2", name)
    name = re.sub(
        r"up_blocks\.(\d+)\.attentions\.(\d+)\.transformer_blocks\.(\d+)",
        r"up.\1.attn.\2.blk.\3",
        name,
    )
    name = re.sub(r"up_blocks\.(\d+)\.upsamplers\.0\b", r"up.\1.upsample", name)

    name = name.replace("conv_norm_out", "norm_out")

    name = re.sub(r"\.to_q\b", ".q", name)
    name = re.sub(r"\.to_k\b", ".k", name)
    name = re.sub(r"\.to_v\b", ".v", name)
    name = re.sub(r"\.to_out\.0\b", ".out", name)
    name = re.sub(r"\.ff\.net\.0\.proj\b", ".ffn_gate", name)
    name = re.sub(r"\.ff\.net\.2\b", ".ffn_up", name)

    return f"unet.{name}"


# ---------------------------------------------------------------------------
# Metadata writers
# ---------------------------------------------------------------------------

def write_te_metadata(fout: GGUFWriter, cfg: dict, prefix: str) -> None:
    fout.add_uint32(f"sdxl.{prefix}.vocab_size", cfg["vocab_size"])
    fout.add_uint32(f"sdxl.{prefix}.context_length", cfg["max_position_embeddings"])
    fout.add_uint32(f"sdxl.{prefix}.embedding_length", cfg["hidden_size"])
    fout.add_uint32(f"sdxl.{prefix}.feed_forward_length", cfg["intermediate_size"])
    fout.add_uint32(f"sdxl.{prefix}.block_count", cfg["num_hidden_layers"])
    fout.add_uint32(f"sdxl.{prefix}.attention.head_count", cfg["num_attention_heads"])
    fout.add_float32(f"sdxl.{prefix}.attention.layer_norm_epsilon", cfg["layer_norm_eps"])
    if "projection_dim" in cfg:
        fout.add_uint32(f"sdxl.{prefix}.projection_dim", cfg["projection_dim"])


def write_vae_metadata(fout: GGUFWriter, cfg: dict) -> None:
    fout.add_uint32("sdxl.vae.in_channels", cfg["in_channels"])
    fout.add_uint32("sdxl.vae.out_channels", cfg["out_channels"])
    fout.add_uint32("sdxl.vae.latent_channels", cfg["latent_channels"])
    fout.add_uint32("sdxl.vae.layers_per_block", cfg["layers_per_block"])
    fout.add_uint32("sdxl.vae.norm_num_groups", cfg["norm_num_groups"])
    fout.add_float32("sdxl.vae.scaling_factor", float(cfg["scaling_factor"]))
    fout.add_array("sdxl.vae.block_out_channels", list(cfg["block_out_channels"]))


def write_unet_metadata(fout: GGUFWriter, cfg: dict) -> None:
    fout.add_uint32("sdxl.unet.in_channels", cfg["in_channels"])
    fout.add_uint32("sdxl.unet.out_channels", cfg["out_channels"])
    fout.add_uint32("sdxl.unet.cross_attention_dim", cfg["cross_attention_dim"])
    fout.add_uint32("sdxl.unet.layers_per_block", cfg["layers_per_block"])
    fout.add_array("sdxl.unet.block_out_channels", list(cfg["block_out_channels"]))

    attn_head_dim = cfg["attention_head_dim"]
    if isinstance(attn_head_dim, int):
        attn_head_dim = [attn_head_dim]
    fout.add_array("sdxl.unet.attention_head_dim", list(attn_head_dim))

    tl = cfg.get("transformer_layers_per_block", [1])
    if isinstance(tl, int):
        tl = [tl]
    fout.add_array("sdxl.unet.transformer_layers_per_block", list(tl))

    if cfg.get("addition_embed_type") is not None:
        fout.add_string("sdxl.unet.addition_embed_type", cfg["addition_embed_type"])
    if cfg.get("addition_time_embed_dim") is not None:
        fout.add_uint32("sdxl.unet.addition_time_embed_dim", cfg["addition_time_embed_dim"])
    if cfg.get("projection_class_embeddings_input_dim") is not None:
        fout.add_uint32(
            "sdxl.unet.projection_class_embeddings_input_dim",
            cfg["projection_class_embeddings_input_dim"],
        )


# ---------------------------------------------------------------------------
# Component converters — write to all open writers simultaneously
# ---------------------------------------------------------------------------

def add_text_encoder(writers: dict[str, GGUFWriter], model_dir: str, prefix: str) -> None:
    component = "text_encoder" if prefix == "te1" else "text_encoder_2"
    comp_dir = os.path.join(model_dir, component)
    cfg = load_config(comp_dir)
    tensors = load_tensors(comp_dir)

    for fout in writers.values():
        write_te_metadata(fout, cfg, prefix)

    for hf_name, data in tensors.items():
        gguf_name = get_te_tensor_name(hf_name, prefix)
        if gguf_name is None:
            print(f"  skipping {hf_name}")
            continue
        add_tensor_to_all(writers, gguf_name, data)


def add_vae(writers: dict[str, GGUFWriter], model_dir: str) -> None:
    comp_dir = os.path.join(model_dir, "vae")
    cfg = load_config(comp_dir)
    tensors = load_tensors(comp_dir)

    for fout in writers.values():
        write_vae_metadata(fout, cfg)

    for hf_name, data in tensors.items():
        add_tensor_to_all(writers, get_vae_tensor_name(hf_name), data)


def add_unet(writers: dict[str, GGUFWriter], model_dir: str) -> None:
    comp_dir = os.path.join(model_dir, "unet")
    cfg = load_config(comp_dir)
    tensors = load_tensors(comp_dir)

    for fout in writers.values():
        write_unet_metadata(fout, cfg)

    for hf_name, data in tensors.items():
        add_tensor_to_all(writers, get_unet_tensor_name(hf_name), data)


def write_all_components(
    writers: dict[str, GGUFWriter], model_dir: str, args: argparse.Namespace
) -> None:
    if not args.no_text_enc_1:
        print("\n=== Text Encoder 1 (CLIP ViT-L/14) ===")
        add_text_encoder(writers, model_dir, "te1")

    if not args.no_text_enc_2:
        print("\n=== Text Encoder 2 (OpenCLIP ViT-bigG/14) ===")
        add_text_encoder(writers, model_dir, "te2")

    if not args.no_vae:
        print("\n=== VAE ===")
        add_vae(writers, model_dir)

    if not args.no_unet:
        print("\n=== UNet ===")
        add_unet(writers, model_dir)


def finalize_writers(writers: dict[str, GGUFWriter], paths: dict[str, str]) -> None:
    for q_type, fout in writers.items():
        print(f"\nWriting {paths[q_type]}")
        fout.write_header_to_file()
        fout.write_kv_data_to_file()
        fout.write_tensors_to_file()
        fout.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Convert SDXL-Turbo HuggingFace model to one or more GGUF files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # single variant
  python convert-sdxl-to-gguf.py -m ./sdxl-turbo --q-type Q4_K_M

  # multiple variants in one pass (model loaded once)
  python convert-sdxl-to-gguf.py -m ./sdxl-turbo --q-type f16 q8_0 Q4_K_M Q5_K_M
""",
    )
    ap.add_argument("-m", "--model-dir", required=True,
                    help="Path to the SDXL-Turbo model directory (HuggingFace format)")
    ap.add_argument("-o", "--output-dir", default=None,
                    help="Directory for the output GGUF files (default: model dir)")
    ap.add_argument(
        "--q-type",
        nargs="+",
        choices=_ALL_Q_TYPES,
        default=["f16"],
        metavar="TYPE",
        help=(
            f"One or more weight types (default: f16). "
            f"Python-native: f32 f16 q8_0. "
            f"K-quants (need llama-quantize): {' '.join(sorted(_K_QUANTS))}"
        ),
    )
    ap.add_argument(
        "--llama-quantize",
        default="./llama-quantize",
        metavar="PATH",
        help="Path to the llama-quantize binary, required for K-quant types "
             "(default: ./llama-quantize)",
    )
    ap.add_argument("--no-text-enc-1", action="store_true", default=False,
                    help="Skip Text Encoder 1 (CLIP ViT-L/14)")
    ap.add_argument("--no-text-enc-2", action="store_true", default=False,
                    help="Skip Text Encoder 2 (OpenCLIP ViT-bigG/14)")
    ap.add_argument("--no-vae", action="store_true", default=False,
                    help="Skip VAE")
    ap.add_argument("--no-unet", action="store_true", default=False,
                    help="Skip UNet")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Deduplicate while preserving order
    seen: set[str] = set()
    q_types: list[str] = []
    for qt in args.q_type:
        if qt not in seen:
            seen.add(qt)
            q_types.append(qt)

    model_dir = args.model_dir
    output_dir = args.output_dir if args.output_dir is not None else model_dir
    os.makedirs(output_dir, exist_ok=True)

    # Split into what Python can write directly vs what needs llama-quantize
    python_types = [qt for qt in q_types if qt not in _K_QUANTS]
    k_quant_types = [qt for qt in q_types if qt in _K_QUANTS]

    # K-quants need an f16 GGUF as input; reuse the one we're already writing
    # if possible, otherwise write a temporary one and delete it after.
    f16_is_temp = bool(k_quant_types) and "f16" not in python_types
    if f16_is_temp:
        python_types.append("f16")

    # Build output paths
    def out_path(qt: str, temp: bool = False) -> str:
        if temp:
            return os.path.join(output_dir, "sdxl-turbo-f16-tmp.gguf")
        return os.path.join(output_dir, f"sdxl-turbo-{qt.lower()}.gguf")

    paths = {
        qt: out_path(qt, temp=(qt == "f16" and f16_is_temp))
        for qt in python_types
    }

    # Read model name once
    model_index_path = os.path.join(model_dir, "model_index.json")
    if os.path.exists(model_index_path):
        with open(model_index_path, "r", encoding="utf-8") as f:
            model_name = json.load(f).get("_name_or_path", os.path.basename(model_dir))
    else:
        model_name = os.path.basename(model_dir)

    # Open all writers and write shared header metadata
    writers: dict[str, GGUFWriter] = {}
    for qt in python_types:
        fout = GGUFWriter(path=paths[qt], arch="sdxl")
        fout.add_name(model_name)
        fout.add_description("SDXL-Turbo model (text encoders + VAE + UNet)")
        fout.add_file_type(_FTYPE_VALUE[qt])
        writers[qt] = fout

    types_label = ", ".join(python_types) + (
        f"  [then K-quants: {', '.join(k_quant_types)}]" if k_quant_types else ""
    )
    print(f"Writing types: {types_label}")

    # Single pass over the model — all writers receive every tensor
    write_all_components(writers, model_dir, args)
    finalize_writers(writers, paths)

    output_files = [paths[qt] for qt in python_types if not (qt == "f16" and f16_is_temp)]

    # K-quant pass: call llama-quantize once per type using the f16 GGUF
    if k_quant_types:
        f16_path = paths["f16"]
        try:
            for kq in k_quant_types:
                kq_path = out_path(kq)
                run_llama_quantize(f16_path, kq_path, kq, args.llama_quantize)
                output_files.append(kq_path)
        finally:
            if f16_is_temp and os.path.exists(f16_path):
                os.unlink(f16_path)
                print(f"Removed intermediate: {f16_path}")

    print("\nDone. Output files:")
    for p in output_files:
        print(f"  {p}")


if __name__ == "__main__":
    main()
