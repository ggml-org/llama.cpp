"""
Export a HuggingFace vision encoder to Apple CoreML (.mlpackage).

Convert the ViT + projector pipeline from a HF checkpoint into a .mlpackage
that can be compiled with `xcrun coremlcompiler compile` and loaded by the
mtmd CoreML backend at runtime (tools/mtmd/coreml/).

Supported models:
  minicpmv46 — MiniCPM-V 4.6 (SigLIP 980px, 27 layers, insert merger)
  llava15    — Llava 1.5 (CLIP ViT-L/14-336, 24 layers, MLP projector)

Usage:
  python tools/mtmd/coreml/export_coreml.py \
      --model minicpmv46 -m /path/to/model --precision float16

  python tools/mtmd/coreml/export_coreml.py \
      --model llava15 -m /path/to/model --precision float16
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

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from models.modeling_siglip import (
    DownsampleMLP,
    FullVisionPipeline,
    SiglipVisionConfig,
    SiglipVisionEmbeddings,
    SiglipEncoder,
    ViTInsertMerger,
)
from models.modeling_clip import (
    CLIPVisionConfig,
    CLIPVisionEmbeddings,
    CLIPVisionTransformer,
    MLPProjector,
    LlavaVisionPipeline,
)


# ===========================================================================
# common utilities
# ===========================================================================

def _log(msg: str) -> None:
    print(f"[export_coreml] {msg}", flush=True)


def _load_json(model_path: str, filename: str) -> dict:
    if model_path.endswith(".zip"):
        with zipfile.ZipFile(model_path, "r") as zf:
            names = [n for n in zf.namelist() if n.endswith(filename)]
            if not names:
                raise FileNotFoundError(f"{filename} not found in zip: {model_path}")
            return json.loads(zf.read(names[0]))
    else:
        with open(os.path.join(model_path, filename), "r") as f:
            return json.load(f)


def _load_weights(model_path: str) -> dict[str, torch.Tensor]:
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


def _load_state(module: nn.Module, state: dict, name: str, ignore_missing: set = None):
    if ignore_missing is None:
        ignore_missing = set()
    missing, unexpected = module.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if k not in ignore_missing]
    if real_missing:
        _log(f"  [{name}] missing keys: {real_missing}")
    if unexpected:
        _log(f"  [{name}] unexpected keys: {unexpected}")


def _trace_and_convert(
    pipeline: nn.Module,
    trace_inputs: tuple,
    ct_inputs: list,
    output_name: str,
    precision: str,
    metadata: dict,
    output_path: str,
    verify: bool,
) -> None:
    """Trace → CoreML convert → save → optionally verify. Model-agnostic."""
    _log("tracing ...")
    with torch.no_grad():
        traced = torch.jit.trace(pipeline, trace_inputs, strict=False)
        y_ref = pipeline(*trace_inputs)
        y_trace = traced(*trace_inputs)
        _log(f"trace diff: max={(y_ref - y_trace).abs().max().item():.6e}, "
             f"mean={(y_ref - y_trace).abs().mean().item():.6e}")

    _log(f"converting to CoreML ({precision}) ...")
    if precision == "float16":
        ct_prec = ct.precision.FLOAT16
        ct_cu = ct.ComputeUnit.CPU_AND_NE
    else:
        ct_prec = ct.precision.FLOAT32
        ct_cu = ct.ComputeUnit.ALL

    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=ct_inputs,
        outputs=[ct.TensorType(name=output_name)],
        compute_precision=ct_prec,
        compute_units=ct_cu,
        minimum_deployment_target=ct.target.iOS15,
    )

    mlmodel.author = "tianchi"
    mlmodel.license = "Apache 2.0"
    for k, v in metadata.items():
        setattr(mlmodel, k, v)

    try:
        spec = mlmodel._spec
        spec.description.metadata.userDefined.update(metadata.get("userDefined", {}))
    except Exception as e:
        _log(f"warning: failed to set userDefined metadata: {e}")

    if not output_path.endswith(".mlpackage"):
        output_path += ".mlpackage"

    mlmodel.save(output_path)
    _log(f"saved: {output_path}")

    if verify:
        _log("verifying ...")
        predict_inputs = {}
        for ct_in in ct_inputs:
            name = ct_in.name
            tensor = trace_inputs[0] if len(ct_inputs) == 1 \
                else {i.name: v for i, v in zip(ct_inputs, trace_inputs)}[name]
            predict_inputs[name] = tensor if isinstance(tensor, np.ndarray) else tensor.cpu().numpy()

        y_ml = mlmodel.predict(predict_inputs)[output_name]
        y_ml_t = torch.as_tensor(y_ml)

        d = (y_ref - y_ml_t).abs()
        _log(f"  pytorch vs coreml: max={d.max().item():.6e}, mean={d.mean().item():.6e}")

    _log("done. Compile with: xcrun coremlcompiler compile " + output_path + " <output_dir>")


# ===========================================================================
# MiniCPM-V 4.6
# ===========================================================================

def _export_minicpmv46(model_path: str, output: str, patch_h: int, patch_w: int,
                       precision: str, verify: bool) -> None:
    assert patch_h % 4 == 0 and patch_w % 4 == 0, "patch_h/patch_w must be multiples of 4"

    config = _load_json(model_path, "config.json")
    vision_cfg_raw = config.get("vision_config")
    if vision_cfg_raw is None:
        raise ValueError("config.json missing 'vision_config'")

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

    llm_embed_dim = config.get("hidden_size") or \
                    (config.get("text_config") or {}).get("hidden_size") or 1024
    insert_layer_id = 6
    num_patches = patch_h * patch_w
    nps = vision_config.image_size // vision_config.patch_size

    _log(f"detected: hidden={vision_config.hidden_size}, layers={vision_config.num_hidden_layers}, "
         f"llm_embed={llm_embed_dim}, image={vision_config.image_size}")

    # Build
    hidden_size = vision_config.hidden_size
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

    # Load weights
    tensors = _load_weights(model_path)
    normalized: dict[str, torch.Tensor] = {}
    for key, t in tensors.items():
        normalized[key[6:] if key.startswith("model.") else key] = t

    has_vision_tower = any(k.startswith("vision_tower.") for k in normalized)
    vpm_state, merger_state, resampler_state = {}, {}, {}

    if has_vision_tower:
        for key, t in normalized.items():
            if key.startswith("vision_tower.vit_merger."):
                merger_state[key[24:]] = t
            elif key.startswith("vision_tower."):
                vpm_state[key[13:]] = t
            elif key.startswith("merger.mlp.0."):
                sub = key[13:]
                if sub.startswith("linear_1."):
                    resampler_state["mlp.0.mlp.0." + sub[9:]] = t
                elif sub.startswith("linear_2."):
                    resampler_state["mlp.0.mlp.2." + sub[9:]] = t
                elif sub.startswith("pre_norm."):
                    resampler_state["mlp.0.pre_norm." + sub[9:]] = t
    else:
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

    _load_state(embeddings, emb_state, "embeddings")
    _load_state(encoder, enc_state, "encoder")
    _load_state(post_ln, pln_state, "post_layernorm")
    _load_state(vit_merger, merger_state, "vit_merger",
                ignore_missing={"window_indices", "reverse_indices"})

    ds_state = {k.replace("mlp.0.", "", 1): v for k, v in resampler_state.items()}
    _load_state(mlp_merger, ds_state, "mlp_merger")

    patch_size = vision_config.patch_size
    _log(f"pipeline: pixel_values [1,3,{patch_size},{patch_size * num_patches}] + patch_w [1] i32 "
         f"→ [{1},{num_patches // 16},{llm_embed_dim}]")

    xe = torch.ones(1, 3, patch_size, patch_size * num_patches, dtype=torch.float32)
    pw = torch.tensor([patch_w], dtype=torch.int32)

    _trace_and_convert(
        pipeline, (xe, pw),
        ct_inputs=[
            ct.TensorType(name="pixel_values", shape=xe.shape),
            ct.TensorType(name="patch_w", shape=pw.shape, dtype=types.int32),
        ],
        output_name="output",
        precision=precision,
        metadata={
            "short_description": f"MiniCPM-V 4.6 Full Vision Pipeline",
            "version": "4.6.0",
            "userDefined": {
                "model_type": "vit_all",
                "base_model": "MiniCPM-V 4.6",
                "precision": precision,
                "patch_h": str(patch_h),
                "patch_w": str(patch_w),
                "insert_layer_id": str(insert_layer_id),
                "input_pixels": str(xe.shape),
                "output_shape": str(torch.zeros(1, num_patches // 16, llm_embed_dim).shape),
            },
        },
        output_path=output,
        verify=verify,
    )


# ===========================================================================
# Llava 1.5
# ===========================================================================

def _export_llava15(model_path: str, output: str, precision: str, verify: bool) -> None:
    config = _load_json(model_path, "config.json")
    vision_cfg_raw = config.get("vision_config")
    if vision_cfg_raw is None:
        raise ValueError("config.json missing 'vision_config'")

    vision_config = CLIPVisionConfig(
        hidden_size=vision_cfg_raw.get("hidden_size", 1024),
        intermediate_size=vision_cfg_raw.get("intermediate_size", 4096),
        num_hidden_layers=vision_cfg_raw.get("num_hidden_layers", 24),
        num_attention_heads=vision_cfg_raw.get("num_attention_heads", 16),
        num_channels=vision_cfg_raw.get("num_channels", 3),
        image_size=vision_cfg_raw.get("image_size", 336),
        patch_size=vision_cfg_raw.get("patch_size", 14),
        layer_norm_eps=vision_cfg_raw.get("layer_norm_eps", 1e-6),
    )

    # Llava stores LLM hidden_size in text_config
    text_cfg = config.get("text_config", {})
    llm_embed_dim = text_cfg.get("hidden_size", 4096)

    _log(f"detected: hidden={vision_config.hidden_size}, layers={vision_config.num_hidden_layers}, "
         f"llm_embed={llm_embed_dim}, image={vision_config.image_size}")

    # Build
    vit = CLIPVisionTransformer(vision_config)
    projector = MLPProjector(
        vision_hidden=vision_config.hidden_size,
        intermediate=vision_config.intermediate_size,
        llm_hidden=llm_embed_dim,
    )
    pipeline = LlavaVisionPipeline(vit, projector).to(torch.float32)
    pipeline.eval()

    # Load weights
    tensors = _load_weights(model_path)
    vit_state, proj_state = {}, {}

    for key, t in tensors.items():
        if key == "vision_tower.vision_model.embeddings.class_embedding":
            vit_state["embeddings.class_embedding"] = t.view(1, 1, -1)
        elif key == "vision_tower.vision_model.embeddings.position_embedding.weight":
            vit_state["embeddings.position_embedding.weight"] = t
        elif key.startswith("vision_tower.vision_model."):
            stripped = key[26:]  # strip "vision_tower.vision_model."
            # nn.ModuleList keys: "encoder.0.self_attn..." not "encoder.layers.0.self_attn..."
            if stripped.startswith("encoder.layers."):
                stripped = "encoder." + stripped[15:]
            vit_state[stripped] = t
        elif key == "multi_modal_projector.linear_1.weight":
            proj_state["layers.0.weight"] = t
        elif key == "multi_modal_projector.linear_1.bias":
            proj_state["layers.0.bias"] = t
        elif key == "multi_modal_projector.linear_2.weight":
            proj_state["layers.2.weight"] = t
        elif key == "multi_modal_projector.linear_2.bias":
            proj_state["layers.2.bias"] = t

    _log(f"weights: vit={len(vit_state)}, projector={len(proj_state)}")

    _load_state(vit, vit_state, "vit",
                ignore_missing={"embeddings.patch_embedding.bias"})
    _load_state(projector, proj_state, "projector")

    image_size = vision_config.image_size
    n_tokens = vision_config.num_patches  # 576
    _log(f"pipeline: pixel_values [1,3,{image_size},{image_size}] → [{1},{n_tokens},{llm_embed_dim}]")

    xe = torch.ones(1, 3, image_size, image_size, dtype=torch.float32)

    _trace_and_convert(
        pipeline, (xe,),
        ct_inputs=[
            ct.TensorType(name="pixel_values", shape=xe.shape),
        ],
        output_name="output",
        precision=precision,
        metadata={
            "short_description": f"Llava 1.5 Vision Pipeline",
            "version": "1.5.0",
            "userDefined": {
                "model_type": "llava",
                "base_model": "Llava-1.5",
                "precision": precision,
                "image_size": str(image_size),
                "input_pixels": str(xe.shape),
                "output_shape": str(torch.zeros(1, n_tokens, llm_embed_dim).shape),
            },
        },
        output_path=output,
        verify=verify,
    )


# ===========================================================================
# CLI
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Export HF vision encoder → CoreML .mlpackage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # MiniCPM-V 4.6, float16
  python export_coreml.py --model minicpmv46 -m /path/to/MiniCPM-V-4.6 --precision float16

  # Llava 1.5, float32
  python export_coreml.py --model llava15 -m /path/to/llava-1.5-7b-hf --precision float32
""",
    )
    ap.add_argument("--model", required=True, choices=["minicpmv46", "llava15"],
                    help="Model identifier (dispatches to the correct pipeline)")
    ap.add_argument("-m", "--model-path", required=True,
                    help="HuggingFace model directory (or .zip archive)")
    ap.add_argument("-o", "--output", default="coreml_model.mlpackage",
                    help="Output .mlpackage path (default: coreml_model.mlpackage)")
    ap.add_argument("--patch-h", type=int, default=32,
                    help="patch grid height (minicpmv46 only, default: 32)")
    ap.add_argument("--patch-w", type=int, default=32,
                    help="patch grid width (minicpmv46 only, default: 32)")
    ap.add_argument("--precision", choices=["float16", "float32"], default="float16",
                    help="Compute precision (default: float16)")
    ap.add_argument("--no-verify", action="store_true",
                    help="Skip PyTorch vs CoreML accuracy verification")
    args = ap.parse_args()

    _log(f"model: {args.model}")
    _log(f"model-path: {args.model_path}")
    _log(f"output: {args.output}")
    _log(f"precision: {args.precision}")

    if args.model == "minicpmv46":
        _export_minicpmv46(
            model_path=args.model_path,
            output=args.output,
            patch_h=args.patch_h,
            patch_w=args.patch_w,
            precision=args.precision,
            verify=not args.no_verify,
        )
    elif args.model == "llava15":
        _export_llava15(
            model_path=args.model_path,
            output=args.output,
            precision=args.precision,
            verify=not args.no_verify,
        )


if __name__ == "__main__":
    main()
