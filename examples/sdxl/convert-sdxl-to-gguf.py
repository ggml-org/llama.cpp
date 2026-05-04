#!/usr/bin/env python3
"""Convert a HuggingFace SDXL-Turbo model directory to a single GGUF file.

The output packs all four components into one file with prefixed tensor names:
  te1.*   - CLIP ViT-L/14 text encoder (text_encoder/)
  te2.*   - OpenCLIP ViT-bigG/14 text encoder (text_encoder_2/)
  vae.*   - Variational Autoencoder (vae/)
  unet.*  - Denoising U-Net (unet/)

Supported weight types via --q-type:
  f32    - float32 (2D weights; 4D conv kernels always stay f16)
  f16    - float16 [default]
  q8_0   - Q8_0 for 2D weights; 1D biases/norms stay f32, 4D conv stay f16

For K-quants (Q2_K, Q4_K_S, Q4_K_M, Q5_K_M, Q6_K …) convert to f16 first
then quantize with the llama-quantize C++ binary:
  ./llama-quantize sdxl-turbo-f16.gguf sdxl-turbo-q4_k_m.gguf Q4_K_M

Usage:
  python convert-sdxl-to-gguf.py -m ./sdxl-turbo -o ./output
  python convert-sdxl-to-gguf.py -m ./sdxl-turbo --q-type q8_0
"""

from __future__ import annotations

import argparse
import json
import os
import re
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

# Maps --q-type string → GGUFFileType int (for add_file_type metadata)
_FTYPE_VALUE: dict[str, int] = {
    "f32":  0,   # ALL_F32
    "f16":  1,   # MOSTLY_F16
    "q8_0": 7,   # MOSTLY_Q8_0
}

_K_QUANT_NAMES = ("Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K",
                  "Q2_K_S", "Q4_K_S", "Q4_K_M", "Q5_K_S", "Q5_K_M")


# ---------------------------------------------------------------------------
# Tensor loading
# ---------------------------------------------------------------------------

def load_tensors(component_dir: str) -> dict[str, torch.Tensor]:
    """Load tensors from a component sub-directory (safetensors or pytorch)."""
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
    data: torch.Tensor, name: str, q_type: str
) -> tuple[np.ndarray, GGMLQuantizationType | None]:
    """Return (array, raw_dtype) applying smart per-tensor dtype rules.

    Rules:
      1D  (bias, norm, scalar) → always f32, raw_dtype=None
      4D  (conv kernel)        → always f16, raw_dtype=None
      2D  (weight matrix)      → q_type; fallback to f16 if shape is incompatible
    """
    arr = data.squeeze().float().numpy()
    n_dims = arr.ndim

    # 1D: biases, LayerNorm/GroupNorm weights, scalars
    if n_dims == 1:
        out = arr.astype(np.float32)
        print(f"  {name} | f32 | {out.shape}")
        return out, None

    # 4D: convolution kernels — GGML conv has no f32 kernel support
    if n_dims == 4:
        out = arr.astype(np.float16)
        print(f"  {name} | f16 (conv) | {out.shape}")
        return out, None

    # 2D and other: apply requested type
    if q_type == "f32":
        out = arr.astype(np.float32)
        print(f"  {name} | f32 | {out.shape}")
        return out, None

    if q_type == "f16":
        out = arr.astype(np.float16)
        print(f"  {name} | f16 | {out.shape}")
        return out, None

    if q_type == "q8_0":
        if can_quantize_to_q8_0(arr):
            out = quantize_q8_0(arr)
            print(f"  {name} | q8_0 | {arr.shape}")
            return out, GGMLQuantizationType.Q8_0
        else:
            # Row width not a multiple of 32 — fall back gracefully
            out = arr.astype(np.float16)
            print(f"  {name} | f16 (q8_0 fallback, row={arr.shape[-1]}) | {arr.shape}")
            return out, None

    raise ValueError(f"Unsupported q_type: {q_type!r}")


def add_gguf_tensor(
    fout: GGUFWriter,
    name: str,
    data: torch.Tensor,
    q_type: str,
) -> None:
    arr, raw_dtype = convert_tensor(data, name, q_type)
    fout.add_tensor(name, arr, raw_dtype=raw_dtype)


# ---------------------------------------------------------------------------
# Tensor name mapping: Text Encoders
# ---------------------------------------------------------------------------

_TE_SKIP = frozenset([
    "logit_scale",
    "text_model.embeddings.position_ids",
])


def get_te_tensor_name(hf_name: str, prefix: str) -> str | None:
    """Map a HuggingFace CLIP text-encoder tensor name to a prefixed GGUF name."""
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
    """Map a HuggingFace AutoencoderKL tensor name to a GGUF vae.* name."""
    name = hf_name

    # Encoder path
    name = re.sub(r"encoder\.down_blocks\.(\d+)\.resnets\.(\d+)", r"encoder.down.\1.res.\2", name)
    name = re.sub(r"encoder\.down_blocks\.(\d+)\.downsamplers\.0", r"encoder.down.\1.downsample", name)
    name = re.sub(r"encoder\.mid_block\.resnets\.0\b", "encoder.mid.res0", name)
    name = re.sub(r"encoder\.mid_block\.resnets\.1\b", "encoder.mid.res1", name)
    name = re.sub(r"encoder\.mid_block\.attentions\.0\b", "encoder.mid.attn", name)
    name = name.replace("encoder.conv_norm_out", "encoder.norm_out")

    # Decoder path
    name = re.sub(r"decoder\.up_blocks\.(\d+)\.resnets\.(\d+)", r"decoder.up.\1.res.\2", name)
    name = re.sub(r"decoder\.up_blocks\.(\d+)\.upsamplers\.0", r"decoder.up.\1.upsample", name)
    name = re.sub(r"decoder\.mid_block\.resnets\.0\b", "decoder.mid.res0", name)
    name = re.sub(r"decoder\.mid_block\.resnets\.1\b", "decoder.mid.res1", name)
    name = re.sub(r"decoder\.mid_block\.attentions\.0\b", "decoder.mid.attn", name)
    name = name.replace("decoder.conv_norm_out", "decoder.norm_out")

    # Attention sub-layers inside VAE (single-head, no cross-attn)
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
    """Map a HuggingFace UNet2DConditionModel tensor name to a GGUF unet.* name."""
    name = hf_name

    # Time / conditioning embeddings
    name = name.replace("time_embedding.", "time_embed.")
    name = name.replace("add_embedding.", "add_embed.")

    # Down blocks
    name = re.sub(r"down_blocks\.(\d+)\.resnets\.(\d+)", r"down.\1.res.\2", name)
    name = re.sub(
        r"down_blocks\.(\d+)\.attentions\.(\d+)\.transformer_blocks\.(\d+)",
        r"down.\1.attn.\2.blk.\3",
        name,
    )
    name = re.sub(r"down_blocks\.(\d+)\.downsamplers\.0\b", r"down.\1.downsample", name)

    # Mid block
    name = re.sub(r"mid_block\.resnets\.0\b", "mid.res0", name)
    name = re.sub(r"mid_block\.resnets\.1\b", "mid.res1", name)
    name = re.sub(
        r"mid_block\.attentions\.0\.transformer_blocks\.(\d+)",
        r"mid.attn.blk.\1",
        name,
    )

    # Up blocks
    name = re.sub(r"up_blocks\.(\d+)\.resnets\.(\d+)", r"up.\1.res.\2", name)
    name = re.sub(
        r"up_blocks\.(\d+)\.attentions\.(\d+)\.transformer_blocks\.(\d+)",
        r"up.\1.attn.\2.blk.\3",
        name,
    )
    name = re.sub(r"up_blocks\.(\d+)\.upsamplers\.0\b", r"up.\1.upsample", name)

    # Output
    name = name.replace("conv_norm_out", "norm_out")

    # Attention sub-layers
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

    if "addition_embed_type" in cfg and cfg["addition_embed_type"] is not None:
        fout.add_string("sdxl.unet.addition_embed_type", cfg["addition_embed_type"])
    if "addition_time_embed_dim" in cfg and cfg["addition_time_embed_dim"] is not None:
        fout.add_uint32("sdxl.unet.addition_time_embed_dim", cfg["addition_time_embed_dim"])
    if "projection_class_embeddings_input_dim" in cfg and cfg["projection_class_embeddings_input_dim"] is not None:
        fout.add_uint32(
            "sdxl.unet.projection_class_embeddings_input_dim",
            cfg["projection_class_embeddings_input_dim"],
        )


# ---------------------------------------------------------------------------
# Component converters
# ---------------------------------------------------------------------------

def add_text_encoder(fout: GGUFWriter, model_dir: str, prefix: str, q_type: str) -> None:
    """Write one text encoder (TE1 or TE2) into the GGUF file."""
    component = "text_encoder" if prefix == "te1" else "text_encoder_2"
    comp_dir = os.path.join(model_dir, component)
    cfg = load_config(comp_dir)
    tensors = load_tensors(comp_dir)

    write_te_metadata(fout, cfg, prefix)

    for hf_name, data in tensors.items():
        gguf_name = get_te_tensor_name(hf_name, prefix)
        if gguf_name is None:
            print(f"  skipping {hf_name}")
            continue
        add_gguf_tensor(fout, gguf_name, data, q_type)


def add_vae(fout: GGUFWriter, model_dir: str, q_type: str) -> None:
    comp_dir = os.path.join(model_dir, "vae")
    cfg = load_config(comp_dir)
    tensors = load_tensors(comp_dir)

    write_vae_metadata(fout, cfg)

    for hf_name, data in tensors.items():
        gguf_name = get_vae_tensor_name(hf_name)
        add_gguf_tensor(fout, gguf_name, data, q_type)


def add_unet(fout: GGUFWriter, model_dir: str, q_type: str) -> None:
    comp_dir = os.path.join(model_dir, "unet")
    cfg = load_config(comp_dir)
    tensors = load_tensors(comp_dir)

    write_unet_metadata(fout, cfg)

    for hf_name, data in tensors.items():
        gguf_name = get_unet_tensor_name(hf_name)
        add_gguf_tensor(fout, gguf_name, data, q_type)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Convert SDXL-Turbo HuggingFace model to GGUF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Weight type rules:
  1D tensors (biases, norms)  → always f32
  4D tensors (conv kernels)   → always f16
  2D tensors (weight matrices)→ --q-type value (with f16 fallback for q8_0 if shape incompatible)

For K-quants (Q4_K_M, Q2_K, etc.) first convert to f16, then run:
  ./llama-quantize sdxl-turbo-f16.gguf sdxl-turbo-q4_k_m.gguf Q4_K_M
""",
    )
    ap.add_argument("-m", "--model-dir", required=True,
                    help="Path to the SDXL-Turbo model directory (HuggingFace format)")
    ap.add_argument("-o", "--output-dir", default=None,
                    help="Directory for the output GGUF file (default: model dir)")
    ap.add_argument(
        "--q-type",
        choices=["f32", "f16", "q8_0"],
        default="f16",
        help="Weight quantization type for 2D tensors (default: f16)",
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


def main() -> None:
    args = parse_args()

    model_dir = args.model_dir
    output_dir = args.output_dir if args.output_dir is not None else model_dir
    os.makedirs(output_dir, exist_ok=True)

    q_type = args.q_type

    fname_out = os.path.join(output_dir, f"sdxl-turbo-{q_type}.gguf")
    fout = GGUFWriter(path=fname_out, arch="sdxl")

    # Read the top-level model_index.json for the model name
    model_index_path = os.path.join(model_dir, "model_index.json")
    if os.path.exists(model_index_path):
        with open(model_index_path, "r", encoding="utf-8") as f:
            model_index = json.load(f)
        model_name = model_index.get("_name_or_path", os.path.basename(model_dir))
    else:
        model_name = os.path.basename(model_dir)

    fout.add_name(model_name)
    fout.add_description("SDXL-Turbo model (text encoders + VAE + UNet)")
    fout.add_file_type(_FTYPE_VALUE[q_type])

    # --- Text Encoder 1 ---
    if not args.no_text_enc_1:
        print("\n=== Text Encoder 1 (CLIP ViT-L/14) ===")
        add_text_encoder(fout, model_dir, "te1", q_type)

    # --- Text Encoder 2 ---
    if not args.no_text_enc_2:
        print("\n=== Text Encoder 2 (OpenCLIP ViT-bigG/14) ===")
        add_text_encoder(fout, model_dir, "te2", q_type)

    # --- VAE ---
    if not args.no_vae:
        print("\n=== VAE ===")
        add_vae(fout, model_dir, q_type)

    # --- UNet ---
    if not args.no_unet:
        print("\n=== UNet ===")
        add_unet(fout, model_dir, q_type)

    print(f"\nWriting GGUF file: {fname_out}")
    fout.write_header_to_file()
    fout.write_kv_data_to_file()
    fout.write_tensors_to_file()
    fout.close()

    print(f"Done. Output: {fname_out}")

    if q_type == "f16":
        print(
            "\nTip: to produce K-quant variants run:\n"
            f"  ./llama-quantize {fname_out} "
            f"{fname_out.replace('-f16.gguf', '-q4_k_m.gguf')} Q4_K_M\n"
            "Available types: Q2_K  Q3_K_M  Q4_K_S  Q4_K_M  Q5_K_S  Q5_K_M  Q6_K"
        )


if __name__ == "__main__":
    main()
