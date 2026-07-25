#!/usr/bin/env python3
"""Convert an HF VLA component to a standalone GGUF.

The converter registry keeps model-specific tensor mapping separate from the
common ``general.architecture=vla`` metadata.

Usage (from llama.cpp root):
    PYTHONPATH=gguf-py python3 tools/vla/convert_hf_to_vla_gguf.py \\
        --model /path/to/MiniCPM-RobotManip \\
        --output vla-f32.gguf --control-horizon 30
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

_gguf_py = Path(__file__).resolve().parents[2] / "gguf-py"
if str(_gguf_py) not in sys.path:
    sys.path.insert(0, str(_gguf_py))

from gguf import GGUFWriter, GGUFReader  # noqa: E402


@dataclass
class ActionHeadHParams:
    action_dim: int
    state_dim: int
    action_horizon: int
    num_inference_timesteps: int
    num_timestep_buckets: int
    dit_layers: int
    dit_heads: int
    dit_head_dim: int
    dit_hidden: int
    dit_ffn: int
    cross_attention_dim: int
    output_dim: int
    dec_hidden: int
    n_future_tokens: int
    max_seq_len: int
    max_num_embodiments: int
    ln_eps: float
    norm_out_eps: float
    prediction_type: str
    proprio_inject: str
    interleave_self_attention: bool
    multi_embodiment: bool


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(f"ERROR: {msg}")


def _load_action_head_sd(robotmanip_path: str) -> dict[str, torch.Tensor]:
    path = Path(robotmanip_path)
    weights = path / "model.safetensors" if path.is_dir() else path
    _require(weights.is_file(), f"weights not found: {weights}")

    from safetensors import safe_open

    ah_sd: dict[str, torch.Tensor] = {}
    with safe_open(str(weights), framework="pt", device="cpu") as reader:
        for key in reader.keys():
            if not key.startswith("action_head."):
                continue
            ah_sd[key[len("action_head."):]] = reader.get_tensor(key).contiguous().clone()
    _require(ah_sd, "no action_head.* tensors found")
    print(f"Extracted {len(ah_sd)} action_head keys from {weights}")
    return ah_sd


def _build_name_map(n_layers: int) -> dict[str, str]:
    m: dict[str, str] = {}
    m["model.timestep_encoder.timestep_embedder.linear_1.weight"] = "act.time_emb.l1.weight"
    m["model.timestep_encoder.timestep_embedder.linear_1.bias"] = "act.time_emb.l1.bias"
    m["model.timestep_encoder.timestep_embedder.linear_2.weight"] = "act.time_emb.l2.weight"
    m["model.timestep_encoder.timestep_embedder.linear_2.bias"] = "act.time_emb.l2.bias"

    for i in range(n_layers):
        b = f"act.blk.{i}"
        p = f"model.transformer_blocks.{i}"
        m[f"{p}.norm1.linear.weight"] = f"{b}.adaln.weight"
        m[f"{p}.norm1.linear.bias"] = f"{b}.adaln.bias"
        m[f"{p}.attn1.to_q.weight"] = f"{b}.attn_q.weight"
        m[f"{p}.attn1.to_q.bias"] = f"{b}.attn_q.bias"
        m[f"{p}.attn1.to_k.weight"] = f"{b}.attn_k.weight"
        m[f"{p}.attn1.to_k.bias"] = f"{b}.attn_k.bias"
        m[f"{p}.attn1.to_v.weight"] = f"{b}.attn_v.weight"
        m[f"{p}.attn1.to_v.bias"] = f"{b}.attn_v.bias"
        m[f"{p}.attn1.to_out.0.weight"] = f"{b}.attn_o.weight"
        m[f"{p}.attn1.to_out.0.bias"] = f"{b}.attn_o.bias"
        m[f"{p}.ff.net.0.proj.weight"] = f"{b}.ff0.weight"
        m[f"{p}.ff.net.0.proj.bias"] = f"{b}.ff0.bias"
        m[f"{p}.ff.net.2.weight"] = f"{b}.ff2.weight"
        m[f"{p}.ff.net.2.bias"] = f"{b}.ff2.bias"

    m["model.proj_out_1.weight"] = "act.proj_out1.weight"
    m["model.proj_out_1.bias"] = "act.proj_out1.bias"
    m["model.proj_out_2.weight"] = "act.proj_out2.weight"
    m["model.proj_out_2.bias"] = "act.proj_out2.bias"

    # CategorySpecificLinear: PyTorch W [n_emb, in, out], b [n_emb, out]
    m["action_encoder.W1.W"] = "act.enc.w1.weight"
    m["action_encoder.W1.b"] = "act.enc.w1.bias"
    m["action_encoder.W2.W"] = "act.enc.w2.weight"
    m["action_encoder.W2.b"] = "act.enc.w2.bias"
    m["action_encoder.W3.W"] = "act.enc.w3.weight"
    m["action_encoder.W3.b"] = "act.enc.w3.bias"
    m["action_decoder.layer1.W"] = "act.dec.l1.weight"
    m["action_decoder.layer1.b"] = "act.dec.l1.bias"
    m["action_decoder.layer2.W"] = "act.dec.l2.weight"
    m["action_decoder.layer2.b"] = "act.dec.l2.bias"

    m["future_tokens.weight"] = "act.future_tokens"
    m["position_embedding.weight"] = "act.pos_embd"
    return m


def _prepare_tensor(pt_key: str, tensor: torch.Tensor) -> np.ndarray:
    # CategorySpecific W is PyTorch [n_emb, in, out]. Transpose to [n_emb, out, in]
    # so GGUF dim-reverse yields ggml ne=[in, out, n_emb] for mul_mat(W, x).
    t = tensor.to(torch.float32).detach().cpu()
    if t.ndim == 3 and pt_key.endswith(".W"):
        t = t.transpose(1, 2).contiguous()
    return np.ascontiguousarray(t.numpy())


def _infer_hparams(
    ah_sd: dict,
    *,
    action_horizon: int,
    prediction_type: str,
    proprio_inject: str,
    num_inference_timesteps: int,
    interleave_self_attention: bool,
) -> ActionHeadHParams:
    layer_ids = set()
    for k in ah_sd:
        m = re.match(r"model\.transformer_blocks\.(\d+)\.", k)
        if m:
            layer_ids.add(int(m.group(1)))
    _require(layer_ids, "no transformer_blocks.* keys found")
    dit_layers = max(layer_ids) + 1
    _require(layer_ids == set(range(dit_layers)), f"non-contiguous layers: {sorted(layer_ids)}")

    q0 = ah_sd["model.transformer_blocks.0.attn1.to_q.weight"]
    k0 = ah_sd["model.transformer_blocks.0.attn1.to_k.weight"]
    k1 = ah_sd["model.transformer_blocks.1.attn1.to_k.weight"]
    ff0 = ah_sd["model.transformer_blocks.0.ff.net.0.proj.weight"]
    w1 = ah_sd["action_encoder.W1.W"]  # [n_emb, in, out]
    w3 = ah_sd["action_encoder.W3.W"]
    dec2 = ah_sd["action_decoder.layer2.W"]  # [n_emb, in, out]
    future = ah_sd["future_tokens.weight"]
    pos = ah_sd["position_embedding.weight"]
    po2 = ah_sd["model.proj_out_2.weight"]

    dit_hidden = int(q0.shape[0])
    cross_attention_dim = int(k0.shape[1])
    _require(int(k1.shape[1]) == dit_hidden, "odd-layer attn_k in-dim must equal dit_hidden")
    dit_head_dim = 64 if dit_hidden % 64 == 0 else 0
    _require(dit_head_dim > 0, f"cannot infer head_dim from dit_hidden={dit_hidden}")
    dit_heads = dit_hidden // dit_head_dim
    dit_ffn = int(ff0.shape[0])

    _require(w1.ndim == 3, f"W1.W must be 3D CategorySpecific, got {tuple(w1.shape)}")
    n_emb = int(w1.shape[0])
    enc_in = int(w1.shape[1])
    enc_out = int(w1.shape[2])
    _require(enc_out == dit_hidden, f"W1 out {enc_out} != dit_hidden {dit_hidden}")
    _require(int(w3.shape[0]) == n_emb and int(w3.shape[2]) == dit_hidden, "W3 shape mismatch")

    if proprio_inject == "concat":
        _require(enc_in % 2 == 0, f"concat expects even enc in-dim, got {enc_in}")
        action_dim = enc_in // 2
        state_dim = enc_in // 2
    else:
        raise SystemExit(f"ERROR: unsupported proprio_inject={proprio_inject!r}")

    max_seq_len = int(pos.shape[0])
    n_future_tokens = int(future.shape[0])
    _require(action_horizon > 0, "action_horizon must be > 0")
    _require(
        action_horizon + n_future_tokens <= max_seq_len,
        f"horizon+future ({action_horizon}+{n_future_tokens}) > max_seq ({max_seq_len})",
    )

    _require(dec2.ndim == 3, f"decoder layer2.W must be 3D, got {tuple(dec2.shape)}")
    _require(int(dec2.shape[0]) == n_emb, "decoder n_emb mismatch")
    _require(int(dec2.shape[2]) == action_dim, f"dec.l2 out {dec2.shape[2]} != action_dim {action_dim}")
    dec_hidden = int(dec2.shape[1])
    output_dim = int(po2.shape[0])

    if prediction_type != "clean_action":
        raise SystemExit(f"ERROR: unsupported prediction_type={prediction_type!r}")
    if not interleave_self_attention:
        raise SystemExit("ERROR: interleave_self_attention=False unsupported")
    if num_inference_timesteps != 4:
        raise SystemExit(f"ERROR: num_inference_timesteps={num_inference_timesteps} unsupported")

    return ActionHeadHParams(
        action_dim=action_dim,
        state_dim=state_dim,
        action_horizon=action_horizon,
        num_inference_timesteps=num_inference_timesteps,
        num_timestep_buckets=1000,
        dit_layers=dit_layers,
        dit_heads=dit_heads,
        dit_head_dim=dit_head_dim,
        dit_hidden=dit_hidden,
        dit_ffn=dit_ffn,
        cross_attention_dim=cross_attention_dim,
        output_dim=output_dim,
        dec_hidden=dec_hidden,
        n_future_tokens=n_future_tokens,
        max_seq_len=max_seq_len,
        max_num_embodiments=n_emb,
        ln_eps=1e-5,
        norm_out_eps=1e-6,
        prediction_type=prediction_type,
        proprio_inject=proprio_inject,
        interleave_self_attention=interleave_self_attention,
        multi_embodiment=True,
    )


def _add_metadata(writer: GGUFWriter, hp: ActionHeadHParams) -> None:
    writer.add_string("vla.model_type", "minicpm_robot")
    writer.add_uint32("vla.control_dim", hp.action_dim)
    writer.add_uint32("vla.state_dim", hp.state_dim)
    writer.add_uint32("vla.control_horizon", hp.action_horizon)
    writer.add_uint32("vla.conditioning_dim", hp.cross_attention_dim)
    writer.add_uint32("vla.n_embodiments", hp.max_num_embodiments)

    writer.add_uint32("mra.action_dim", hp.action_dim)
    writer.add_uint32("mra.state_dim", hp.state_dim)
    writer.add_uint32("mra.action_horizon", hp.action_horizon)
    writer.add_uint32("mra.num_inference_timesteps", hp.num_inference_timesteps)
    writer.add_uint32("mra.num_timestep_buckets", hp.num_timestep_buckets)
    writer.add_uint32("mra.dit_layers", hp.dit_layers)
    writer.add_uint32("mra.dit_heads", hp.dit_heads)
    writer.add_uint32("mra.dit_head_dim", hp.dit_head_dim)
    writer.add_uint32("mra.dit_hidden", hp.dit_hidden)
    writer.add_uint32("mra.dit_ffn", hp.dit_ffn)
    writer.add_uint32("mra.cross_attention_dim", hp.cross_attention_dim)
    writer.add_uint32("mra.output_dim", hp.output_dim)
    writer.add_uint32("mra.dec_hidden", hp.dec_hidden)
    writer.add_uint32("mra.n_future_tokens", hp.n_future_tokens)
    writer.add_uint32("mra.max_seq_len", hp.max_seq_len)
    writer.add_uint32("mra.max_num_embodiments", hp.max_num_embodiments)
    writer.add_float32("mra.ln_eps", hp.ln_eps)
    writer.add_float32("mra.norm_out_eps", hp.norm_out_eps)
    writer.add_string("mra.prediction_type", hp.prediction_type)
    writer.add_string("mra.proprio_inject", hp.proprio_inject)
    writer.add_bool("mra.interleave_self_attention", hp.interleave_self_attention)
    writer.add_bool("mra.multi_embodiment", hp.multi_embodiment)


def convert_minicpm_robot(
    robotmanip_path: str,
    output_path: str,
    *,
    action_horizon: int,
    prediction_type: str,
    proprio_inject: str,
    num_inference_timesteps: int,
    interleave_self_attention: bool,
) -> None:
    ah_sd = _load_action_head_sd(robotmanip_path)
    hp = _infer_hparams(
        ah_sd,
        action_horizon=action_horizon,
        prediction_type=prediction_type,
        proprio_inject=proprio_inject,
        num_inference_timesteps=num_inference_timesteps,
        interleave_self_attention=interleave_self_attention,
    )
    print(
        f"Inferred: layers={hp.dit_layers} hidden={hp.dit_hidden} "
        f"heads={hp.dit_heads}x{hp.dit_head_dim} cross={hp.cross_attention_dim} "
        f"control/state={hp.action_dim}/{hp.state_dim} horizon={hp.action_horizon} "
        f"n_emb={hp.max_num_embodiments} future={hp.n_future_tokens} "
        f"pred={hp.prediction_type} proprio={hp.proprio_inject}"
    )

    name_map = _build_name_map(hp.dit_layers)
    expected = set(name_map.keys())
    present = set(ah_sd.keys())
    missing = sorted(expected - present)
    unexpected_weights = sorted(
        k for k in present - expected if k.endswith((".weight", ".bias", ".W", ".b"))
    )
    _require(not missing, f"missing required tensors ({len(missing)}): {missing[:8]}")
    _require(
        not unexpected_weights,
        f"unexpected weight tensors ({len(unexpected_weights)}): {unexpected_weights[:8]}",
    )

    writer = GGUFWriter(output_path, "vla")
    _add_metadata(writer, hp)

    prepared: dict[str, np.ndarray] = {}
    for pt_key in sorted(expected):
        gguf_name = name_map[pt_key]
        arr = _prepare_tensor(pt_key, ah_sd[pt_key])
        prepared[gguf_name] = arr
        writer.add_tensor(gguf_name, arr)

    _require(len(prepared) == len(expected), f"mapped {len(prepared)} != {len(expected)}")
    print(f"Mapped {len(prepared)}/{len(expected)} tensors")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print("\n--- L0 verification ---")
    reader = GGUFReader(output_path)
    gguf_tensors = {t.name: t for t in reader.tensors}
    errors = 0
    for pt_key in expected:
        gguf_name = name_map[pt_key]
        expect = prepared[gguf_name]
        if gguf_name not in gguf_tensors:
            print(f"  MISSING: {gguf_name}")
            errors += 1
            continue
        got = np.array(gguf_tensors[gguf_name].data)
        # GGUF may store dims reversed for multi-D tensors
        if got.shape != expect.shape:
            if got.shape[::-1] == expect.shape:
                got = got.reshape(expect.shape)
            else:
                print(f"  SHAPE MISMATCH: {gguf_name} gguf={got.shape} expect={expect.shape}")
                errors += 1
                continue
        if not np.allclose(expect, got, atol=0):
            print(f"  VALUE MISMATCH: {gguf_name}")
            errors += 1

    print(f"L0 result: {len(expected) - errors}/{len(expected)} OK, {errors} errors")
    if errors:
        sys.exit(1)
    print(f"L0 PASSED - {len(expected)} tensors verified")


def _detect_model_type(path: str, override: str | None) -> str:
    if override:
        return override
    config_path = Path(path) / "config.json"
    if not config_path.is_file():
        raise SystemExit("ERROR: --model-type is required when config.json is unavailable")
    config = json.loads(config_path.read_text())
    architectures = config.get("architectures") or []
    _require(architectures, f"no architectures in {config_path}")
    return str(architectures[0])


CONVERTERS = {
    "MiniCPMV_VLA": convert_minicpm_robot,
    "MiniCPMRobotForConditionalGeneration": convert_minicpm_robot,
    "minicpm_robot": convert_minicpm_robot,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert an HF VLA component to GGUF")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", dest="model_path",
                        help="HF model directory or weights file")
    source.add_argument("--robotmanip-path", dest="model_path",
                        help=argparse.SUPPRESS)
    parser.add_argument("--model-type", default=None,
                        help="converter registry key; inferred from config.json by default")
    parser.add_argument("--output", required=True)
    parser.add_argument("--control-horizon", type=int, default=30)
    parser.add_argument("--prediction-type", default="clean_action")
    parser.add_argument("--proprio-inject", default="concat")
    parser.add_argument("--num-inference-timesteps", type=int, default=4)
    parser.add_argument("--interleave-self-attention", action=argparse.BooleanOptionalAction,
                        default=True)
    args = parser.parse_args()

    src = args.model_path

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    model_type = _detect_model_type(src, args.model_type)
    converter = CONVERTERS.get(model_type)
    if converter is None:
        parser.error(
            f"unsupported VLA model type {model_type!r}; "
            f"available: {', '.join(sorted(CONVERTERS))}"
        )

    converter(
        src,
        args.output,
        action_horizon=args.control_horizon,
        prediction_type=args.prediction_type,
        proprio_inject=args.proprio_inject,
        num_inference_timesteps=args.num_inference_timesteps,
        interleave_self_attention=args.interleave_self_attention,
    )


if __name__ == "__main__":
    main()
