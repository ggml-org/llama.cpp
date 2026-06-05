"""
Export a HuggingFace vision encoder to Apple CoreML (.mlpackage).

Convert the ViT + merger pipeline from a HF checkpoint into a .mlpackage that
can be compiled with `xcrun coremlcompiler compile` and loaded by the mtmd
CoreML backend at runtime (tools/mtmd/coreml/).

Currently supported:
  MiniCPM-V 4.6   — SigLIP 980 px, 27 layers, hidden_size=1152, insert_layer=6

Usage:
  python tools/mtmd/coreml/export_coreml.py \
      -m /path/to/MiniCPM-V-4_6 \
      --patch-h 32 --patch-w 32 \
      --precision float16 \
      -o ./coreml_minicpmv46_vit_all_f16.mlpackage
"""

import argparse
import json
import os
import sys
import zipfile

import numpy as np
import torch
import torch.nn as nn
import coremltools as ct
from coremltools.converters.mil.mil import types
from safetensors import safe_open

# Allow running this script directly from anywhere
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from models.modeling_siglip import (
    DownsampleMLP,
    FullVisionPipeline,
    SiglipVisionConfig,
    SiglipVisionEmbeddings,
    SiglipEncoder,
    SiglipEncoderLayer,
    ViTInsertMerger,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(f"[export_coreml] {msg}", flush=True)


def _load_json(model_path: str, filename: str) -> dict:
    """Load a JSON file from a directory or .zip archive."""
    if model_path.endswith(".zip"):
        with zipfile.ZipFile(model_path, "r") as zf:
            # HF zip: huggingface/config.json, huggingface/tokenizer_config.json, ...
            names = [n for n in zf.namelist() if n.endswith(filename)]
            if not names:
                raise FileNotFoundError(f"{filename} not found in zip: {model_path}")
            return json.loads(zf.read(names[0]))
    else:
        with open(os.path.join(model_path, filename), "r") as f:
            return json.load(f)


def _load_weights(model_path: str) -> dict[str, torch.Tensor]:
    """Load all safetensors tensors into a flat dict."""
    tensors: dict[str, torch.Tensor] = {}

    def _read_sf(path: str):
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    if model_path.endswith(".zip"):
        import tempfile
        with zipfile.ZipFile(model_path, "r") as zf:
            sf_names = [n for n in zf.namelist() if n.endswith(".safetensors")]
            for name in sf_names:
                data = zf.read(name)
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors")
                tmp.write(data)
                tmp.close()
                try:
                    _read_sf(tmp.name)
                finally:
                    os.unlink(tmp.name)
    else:
        # Try model.safetensors first, then sharded model-00001-of-NNNNN.safetensors
        single = os.path.join(model_path, "model.safetensors")
        if os.path.isfile(single):
            _read_sf(single)
        else:
            import glob as _glob
            shards = sorted(_glob.glob(os.path.join(model_path, "model-*-of-*.safetensors")))
            if not shards:
                raise FileNotFoundError(f"No .safetensors found in {model_path}")
            for s in shards:
                _read_sf(s)

    return tensors


# ---------------------------------------------------------------------------
# config detection
# ---------------------------------------------------------------------------

def _detect_model(model_path: str) -> dict:
    """
    Read config.json and return a dict with the information needed to
    build and export the CoreML pipeline.

    Returns dict with keys:
      family          — e.g. "minicpmv46"
      vision_config   — SiglipVisionConfig instance
      llm_embed_dim   — LLM hidden size from top-level config
      insert_layer_id — where to insert the merger (6 for v4.6)
    """
    config = _load_json(model_path, "config.json")

    vision_cfg_raw = config.get("vision_config")
    if vision_cfg_raw is None:
        raise ValueError("config.json missing 'vision_config'; not a MiniCPM-V (or similar) model?")

    vision_config = SiglipVisionConfig(
        hidden_size=vision_cfg_raw.get("hidden_size", 1152),
        intermediate_size=vision_cfg_raw.get("intermediate_size", 4304),
        num_hidden_layers=vision_cfg_raw.get("num_hidden_layers", 27),
        num_attention_heads=vision_cfg_raw.get("num_attention_heads", 16),
        num_channels=vision_cfg_raw.get("num_channels", 3),
        image_size=vision_cfg_raw.get("image_size", 980),
        patch_size=vision_cfg_raw.get("patch_size", 14),
        layer_norm_eps=vision_cfg_raw.get("layer_norm_eps", 1e-6),
        hidden_act=vision_cfg_raw.get("hidden_act", "gelu_pytorch_tanh"),
    )

    # llm_embed_dim: try top-level hidden_size first, then text_config
    llm_embed_dim = config.get("hidden_size") or \
                    (config.get("text_config") or {}).get("hidden_size") or 1024

    # MiniCPM-V family uses insert_layer_id = 6 for v4.x
    insert_layer_id = 6

    _log(f"detected: hidden={vision_config.hidden_size}, layers={vision_config.num_hidden_layers}, "
         f"llm_embed={llm_embed_dim}, image={vision_config.image_size}, patch={vision_config.patch_size}")

    return {
        "family": "minicpmv46",
        "vision_config": vision_config,
        "llm_embed_dim": llm_embed_dim,
        "insert_layer_id": insert_layer_id,
    }


# ---------------------------------------------------------------------------
# model build + weight load
# ---------------------------------------------------------------------------

def _build_and_load(model_path: str, info: dict, patch_h: int, patch_w: int) -> FullVisionPipeline:
    vision_config = info["vision_config"]
    llm_embed_dim = info["llm_embed_dim"]
    insert_layer_id = info["insert_layer_id"]
    num_patches = patch_h * patch_w
    nps = vision_config.image_size // vision_config.patch_size  # num_patches_per_side

    hidden_size = vision_config.hidden_size
    intermediate_size = vision_config.intermediate_size

    # Build components
    embeddings = SiglipVisionEmbeddings(vision_config)
    encoder = SiglipEncoder(vision_config)
    post_ln = nn.LayerNorm(hidden_size, eps=vision_config.layer_norm_eps)
    vit_merger = ViTInsertMerger(vision_config, num_patches)
    mlp_merger = DownsampleMLP(hidden_size * 4, llm_embed_dim)

    pipeline = FullVisionPipeline(
        embeddings, encoder, post_ln, vit_merger, mlp_merger,
        insert_layer_id, num_patches, nps,
    ).to(torch.float32)
    pipeline.eval()

    # Load weights — auto-detect format:
    #   Standard HF: model.vision_tower.{embeddings,encoder,post_layernorm,vit_merger}.*
    #                model.merger.mlp.0.{linear_1,linear_2,pre_norm}.*
    #   Old:         vpm.*, vit_merger.*, resampler.*
    #                (possibly under a "model." prefix)
    tensors = _load_weights(model_path)

    # Strip leading "model." if all keys share it
    normalized: dict[str, torch.Tensor] = {}
    for key, t in tensors.items():
        if key.startswith("model."):
            normalized[key[6:]] = t
        else:
            normalized[key] = t

    vpm_state: dict[str, torch.Tensor] = {}         # embeddings + encoder + post_ln
    merger_state: dict[str, torch.Tensor] = {}       # vit_merger
    resampler_state: dict[str, torch.Tensor] = {}    # mlp merger (resampler)

    # Detect format
    has_vision_tower = any(k.startswith("vision_tower.") for k in normalized)

    if has_vision_tower:
        # Standard HF format
        for key, t in normalized.items():
            if key.startswith("vision_tower.vit_merger."):
                merger_state[key[24:]] = t  # strip "vision_tower.vit_merger." (24 chars)
            elif key.startswith("vision_tower."):
                vpm_state[key[13:]] = t  # strip "vision_tower." (13 chars)
            elif key.startswith("merger.mlp.0."):
                sub = key[13:]  # strip "merger.mlp.0."
                # Map to match old-format resampler keys (expects double "mlp.0." nesting)
                #   linear_1.*  → mlp.0.mlp.0.*   → strip → mlp.0.*
                #   linear_2.*  → mlp.0.mlp.2.*   → strip → mlp.2.*
                #   pre_norm.*  → mlp.0.pre_norm.*  → strip → pre_norm.*
                if sub.startswith("linear_1."):
                    new_k = "mlp.0.mlp.0." + sub[9:]
                elif sub.startswith("linear_2."):
                    new_k = "mlp.0.mlp.2." + sub[9:]
                elif sub.startswith("pre_norm."):
                    new_k = "mlp.0.pre_norm." + sub[9:]
                else:
                    continue
                resampler_state[new_k] = t
    else:
        # Old format: vpm.*, vit_merger.*, resampler.* (or merger.*)
        for key, t in normalized.items():
            if key.startswith("vpm."):
                vpm_state[key[4:]] = t
            elif key.startswith("vit_merger."):
                merger_state[key[11:]] = t
            elif key.startswith("resampler."):
                resampler_state[key[10:]] = t
            elif key.startswith("merger."):
                resampler_state[key[7:]] = t

    _log(f"weights: vpm={len(vpm_state)}, vit_merger={len(merger_state)}, resampler={len(resampler_state)}")

    # map vpm keys to our sub-modules
    emb_state, enc_state, pln_state = {}, {}, {}
    for k, v in vpm_state.items():
        if k.startswith("embeddings."):
            emb_state[k[11:]] = v
        elif k.startswith("encoder."):
            enc_state[k[8:]] = v
        elif k.startswith("post_layernorm."):
            pln_state[k[15:]] = v
        else:
            emb_state[k] = v

    _load_or_warn(embeddings, emb_state, "embeddings")
    _load_or_warn(encoder, enc_state, "encoder")
    _load_or_warn(post_ln, pln_state, "post_layernorm")

    _load_or_warn(vit_merger, merger_state, "vit_merger",
                  ignore_missing={"window_indices", "reverse_indices"})

    # resampler.mlp.0.pre_norm.* → pre_norm.*
    # resampler.mlp.0.mlp.0.*   → mlp.0.*
    # resampler.mlp.0.mlp.2.*   → mlp.2.*
    ds_state = {k.replace("mlp.0.", "", 1): v for k, v in resampler_state.items()}
    _load_or_warn(mlp_merger, ds_state, "mlp_merger (resampler)")

    return pipeline


def _load_or_warn(module: nn.Module, state: dict, name: str, ignore_missing: set = None):
    """Load state_dict with friendly warnings for unmatched keys."""
    if ignore_missing is None:
        ignore_missing = set()
    missing, unexpected = module.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if k not in ignore_missing]
    if real_missing:
        _log(f"  [{name}] missing keys: {real_missing}")
    if unexpected:
        _log(f"  [{name}] unexpected keys: {unexpected}")


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

def export(
    model_path: str,
    output: str,
    patch_h: int = 32,
    patch_w: int = 32,
    precision: str = "float16",
    verify: bool = True,
) -> None:
    assert patch_h % 4 == 0 and patch_w % 4 == 0, "patch_h/patch_w must be multiples of 4 (two 2×2 stages)"

    info = _detect_model(model_path)
    vision_config = info["vision_config"]
    num_patches = patch_h * patch_w
    patch_size = vision_config.patch_size

    # --- 1. build & load ---
    _log("building model ...")
    pipeline = _build_and_load(model_path, info, patch_h, patch_w)

    _log(f"pipeline: pixel_values [1,3,{patch_size},{patch_size * num_patches}] + patch_w [1] i32 "
         f"→ [{1},{num_patches // 16},{info['llm_embed_dim']}]")

    # --- 2. trace ---
    _log("tracing ...")
    xe = torch.ones(1, 3, patch_size, patch_size * num_patches, dtype=torch.float32)
    pw = torch.tensor([patch_w], dtype=torch.int32)

    with torch.no_grad():
        trace_inputs = (xe, pw)
        traced = torch.jit.trace(pipeline, trace_inputs, strict=False)

        y_ref = pipeline(*trace_inputs)
        y_trace = traced(*trace_inputs)
        _log(f"trace diff: max={(y_ref - y_trace).abs().max().item():.6e}, "
             f"mean={(y_ref - y_trace).abs().mean().item():.6e}")

    # --- 3. CoreML convert ---
    _log(f"converting to CoreML ({precision}) ...")
    if precision == "float16":
        ct_prec = ct.precision.FLOAT16
        ct_cu = ct.ComputeUnit.CPU_AND_NE
        ps = "f16"
    else:
        ct_prec = ct.precision.FLOAT32
        ct_cu = ct.ComputeUnit.ALL
        ps = "f32"

    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="pixel_values", shape=xe.shape),
            ct.TensorType(name="patch_w", shape=pw.shape, dtype=types.int32),
        ],
        outputs=[ct.TensorType(name="output")],
        compute_precision=ct_prec,
        compute_units=ct_cu,
        minimum_deployment_target=ct.target.iOS15,
    )

    # metadata for runtime detect()
    mlmodel.short_description = f"MiniCPM-V 4.6 Full Vision Pipeline ({ps})"
    mlmodel.author = "llama.cpp mtmd coreml"
    mlmodel.license = "Apache 2.0"
    mlmodel.version = "4.6.0"

    try:
        spec = mlmodel._spec
        spec.description.metadata.userDefined.update({
            "model_type": "vit_all",
            "base_model": "MiniCPM-V 4.6",
            "precision": precision,
            "patch_h": str(patch_h),
            "patch_w": str(patch_w),
            "insert_layer_id": str(info["insert_layer_id"]),
            "input_pixels": str(xe.shape),
            "output_shape": str(y_ref.shape),
        })
    except Exception as e:
        _log(f"warning: failed to set userDefined metadata: {e}")

    # Ensure output path ends with .mlpackage
    if not output.endswith(".mlpackage"):
        output += ".mlpackage"

    mlmodel.save(output)
    _log(f"saved: {output}")

    # --- 4. verify ---
    if verify:
        _log("verifying ...")
        y_ml = mlmodel.predict({"pixel_values": xe.numpy(), "patch_w": pw.numpy()})["output"]
        y_ml_t = torch.as_tensor(y_ml)

        d = (y_ref - y_ml_t).abs()
        _log(f"  pytorch vs coreml: max={d.max().item():.6e}, mean={d.mean().item():.6e}")
        d2 = (y_trace - y_ml_t).abs()
        _log(f"  traced  vs coreml: max={d2.max().item():.6e}, mean={d2.mean().item():.6e}")

    _log("done. Compile with: xcrun coremlcompiler compile " + output + " <output_dir>")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Export HF vision encoder → CoreML .mlpackage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # MiniCPM-V 4.6, float16 (CPU + NE)
  python export_coreml.py -m /path/to/MiniCPM-V-4_6 --precision float16

  # float32 (CPU + NE + GPU)
  python export_coreml.py -m /path/to/MiniCPM-V-4_6 --precision float32

  # Custom patch grid (e.g. 16x64 for ultra-wide inputs)
  python export_coreml.py -m /path/to/MiniCPM-V-4_6 --patch-h 16 --patch-w 64
""",
    )
    ap.add_argument("-m", "--model", required=True,
                    help="HuggingFace model directory (or .zip archive)")
    ap.add_argument("-o", "--output", default="coreml_model.mlpackage",
                    help="Output .mlpackage path (default: coreml_model.mlpackage)")
    ap.add_argument("--patch-h", type=int, default=32,
                    help="Number of patches vertically (default: 32)")
    ap.add_argument("--patch-w", type=int, default=32,
                    help="Number of patches horizontally (default: 32)")
    ap.add_argument("--precision", choices=["float16", "float32"], default="float16",
                    help="Compute precision (default: float16)")
    ap.add_argument("--no-verify", action="store_true",
                    help="Skip PyTorch vs CoreML accuracy verification")
    args = ap.parse_args()

    _log(f"model: {args.model}")
    _log(f"output: {args.output}")
    _log(f"patch grid: {args.patch_h}x{args.patch_w}")
    _log(f"precision: {args.precision}")

    export(
        model_path=args.model,
        output=args.output,
        patch_h=args.patch_h,
        patch_w=args.patch_w,
        precision=args.precision,
        verify=not args.no_verify,
    )


if __name__ == "__main__":
    main()
