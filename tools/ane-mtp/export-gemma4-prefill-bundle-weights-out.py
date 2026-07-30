#!/usr/bin/env python3
"""Export a multifunction Gemma 4 ANE prefill bundle with weights as inputs.

A bundle is one Core ML mlmodelc that contains every `prefill_sN` function the
target runtime needs.  Unlike `export-gemma4-prefill-bundle.py`, this script
does NOT bake the Gemma 4 layer 0 weights into the mlmodelc.  Instead every
weight the layer slab consumes is declared as a model input.  The C++
runtime looks up the matching `blk.0.*` tensor in the source GGUF and
shares it with the ANE program via an IOSurface-backed MLMultiArray, so
the bundle on disk is just the MIL program and its metadata.

The embed table is also passed as an input — the runtime gathers the row
for each token in the active prompt and forwards the embedded values to
the slab, so we never have to ship the 2 GB vocab table to ANE.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


EPS = 1.0e-6


def load_tensor(source: Path, name: str) -> torch.Tensor:
    with safe_open(source / "model.safetensors", framework="pt", device="cpu") as f:
        return f.get_tensor(name).to(torch.float16)


def rms_norm(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + EPS)).to(
        value.dtype
    ) * (weight + 1.0)


def rope(value: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    pairs = value.shape[-1] // 2
    inv_freq = theta ** (-torch.arange(pairs, dtype=torch.float32) / pairs)
    angle = positions.float()[..., None] * inv_freq
    cosine = torch.cos(angle).to(value.dtype).unsqueeze(-2)
    sine = torch.sin(angle).to(value.dtype).unsqueeze(-2)
    first, second = value[..., :pairs], value[..., pairs:]
    return torch.cat((first * cosine - second * sine, second * cosine + first * sine), -1)


class Gemma4InitialSlabWeightsOut(torch.nn.Module):
    """Layer 0 slab. All weights are forward() inputs, not buffers."""

    def __init__(self, hidden: int, heads: int, kv_heads: int, head_dim: int,
            v_norm_inverse: torch.Tensor) -> None:
        super().__init__()
        self.hidden = hidden
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.register_buffer("v_norm_inverse", v_norm_inverse)

    def forward(
            self,
            token_ids: torch.Tensor,
            positions: torch.Tensor,
            embedded: torch.Tensor,
            attn_norm: torch.Tensor,
            q_weight: torch.Tensor,
            k_weight: torch.Tensor,
            v_weight: torch.Tensor,
            q_norm: torch.Tensor,
            k_norm: torch.Tensor,
            o_weight: torch.Tensor,
            post_attn: torch.Tensor,
            ffn_norm: torch.Tensor,
            gate_weight: torch.Tensor,
            up_weight: torch.Tensor,
            down_weight: torch.Tensor,
            post_ffn: torch.Tensor,
            scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, sequence = token_ids.shape
        current = embedded * math.sqrt(self.hidden)
        normed = rms_norm(current, attn_norm)
        query = F.linear(normed, q_weight).reshape(batch, sequence, self.heads, self.head_dim)
        keys = F.linear(normed, k_weight).reshape(batch, sequence, self.kv_heads, self.head_dim)
        values = F.linear(normed, v_weight).reshape(batch, sequence, self.kv_heads, self.head_dim)
        query = rope(rms_norm(query, q_norm), positions, 10000.0)
        keys = rope(rms_norm(keys, k_norm), positions, 10000.0)
        values = rms_norm(values, k_norm) * self.v_norm_inverse
        expanded_k = keys.repeat_interleave(self.heads // self.kv_heads, dim=2)
        expanded_v = values.repeat_interleave(self.heads // self.kv_heads, dim=2)
        query_heads = query.permute(0, 2, 1, 3)
        key_heads = expanded_k.permute(0, 2, 3, 1)
        scores = torch.matmul(query_heads, key_heads)
        causal = positions[:, None, :] <= positions[:, :, None]
        scores = torch.where(causal[:, None], scores, torch.full_like(scores, -1.0e4))
        probs = torch.softmax(scores.float(), dim=-1).to(current.dtype)
        attended = torch.matmul(probs, expanded_v.permute(0, 2, 1, 3))
        attended = attended.permute(0, 2, 1, 3).reshape(batch, sequence, -1)
        attended = rms_norm(F.linear(attended, o_weight), post_attn) + current
        ffn_input = rms_norm(attended, ffn_norm)
        ffn = F.gelu(F.linear(ffn_input, gate_weight), approximate="tanh") * F.linear(ffn_input, up_weight)
        output = (rms_norm(F.linear(ffn, down_weight), post_ffn) + attended) * scale
        return output, keys.reshape(batch, sequence, -1), values.reshape(batch, sequence, -1)


def export_function(
        source: Path,
        sequence: int,
        batch: int,
        hidden: int, heads: int, kv_heads: int, head_dim: int) -> Path:
    """Convert a single sequence bucket into a temp .mlpackage."""
    k_norm = load_tensor(source, "model.language_model.layers.0.self_attn.k_norm.weight")
    module = Gemma4InitialSlabWeightsOut(hidden, heads, kv_heads, head_dim,
            1.0 / (k_norm + 1.0)).eval()
    # Use any values for the weight inputs — they're only used to fix the
    # conversion-time shapes.  Real values arrive per-call from the GGUF.
    weight_dummies = {
        "embedded": torch.zeros((batch, sequence, hidden), dtype=torch.float16),
        "attn_norm": torch.zeros((hidden,), dtype=torch.float16),
        "q_weight": torch.zeros((heads * head_dim, hidden), dtype=torch.float16),
        "k_weight": torch.zeros((kv_heads * head_dim, hidden), dtype=torch.float16),
        "v_weight": torch.zeros((kv_heads * head_dim, hidden), dtype=torch.float16),
        "q_norm": torch.zeros((head_dim,), dtype=torch.float16),
        "k_norm": torch.zeros((head_dim,), dtype=torch.float16),
        "o_weight": torch.zeros((hidden, heads * head_dim), dtype=torch.float16),
        "post_attn": torch.zeros((hidden,), dtype=torch.float16),
        "ffn_norm": torch.zeros((hidden,), dtype=torch.float16),
        "gate_weight": torch.zeros((4 * hidden, hidden), dtype=torch.float16),
        "up_weight": torch.zeros((4 * hidden, hidden), dtype=torch.float16),
        "down_weight": torch.zeros((hidden, 4 * hidden), dtype=torch.float16),
        "post_ffn": torch.zeros((hidden,), dtype=torch.float16),
        "scale": torch.zeros((1,), dtype=torch.float16),
    }
    token_ids = torch.zeros((batch, sequence), dtype=torch.int32)
    positions = torch.arange(sequence, dtype=torch.int32).repeat(batch, 1)
    args = (token_ids, positions, *weight_dummies.values())
    program = torch.export.export(module, args).run_decompositions({})
    converted = ct.convert(
        program,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="token_ids", shape=token_ids.shape, dtype=int),
            ct.TensorType(name="positions", shape=positions.shape, dtype=int),
        ] + [
            ct.TensorType(name=k, shape=tuple(t.shape), dtype=np.float16)
            for k, t in weight_dummies.items()
        ],
        outputs=[
            ct.TensorType(name="hidden_states"),
            ct.TensorType(name="key_states"),
            ct.TensorType(name="value_states"),
        ],
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
    )
    package = Path(tempfile.mkdtemp(prefix=f"gemma4-prefill-weightsout-s{sequence}-")) / "prefill.mlpackage"
    converted.save(str(package))
    return package


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--sequences", type=int, nargs="+", default=[128, 256, 512],
            choices=[128, 256, 512, 1024])
    args = parser.parse_args()

    config = json.loads((args.source / "config.json").read_text())["text_config"]
    if config["model_type"] != "gemma4_unified_text" or config["layer_types"][0] != "sliding_attention":
        raise SystemExit("only Gemma 4 unified models with an initial sliding-attention layer are supported")
    hidden = config["hidden_size"]
    heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    if heads * head_dim != 4096 or kv_heads * head_dim != 2048:
        raise SystemExit("unexpected Gemma 4 initial-layer attention geometry")

    args.output.mkdir(parents=True, exist_ok=True)
    package_paths: list[tuple[int, Path]] = []
    for sequence in args.sequences:
        package = export_function(args.source, sequence, args.batch, hidden, heads, kv_heads, head_dim)
        package_paths.append((sequence, package))
        print(f"exported prefill_s{sequence} (weights as inputs) to {package}")

    descriptor = ct.utils.MultiFunctionDescriptor()
    for sequence, package in package_paths:
        descriptor.add_function(str(package), "main", f"prefill_s{sequence}")
    descriptor.default_function_name = f"prefill_s{args.sequences[0]}"
    combined = args.output / "prefill-bundle.mlpackage"
    if combined.exists():
        shutil.rmtree(combined)
    ct.utils.save_multifunction(descriptor, str(combined))

    args.output.mkdir(parents=True, exist_ok=True)
    subprocess.run(["xcrun", "coremlcompiler", "compile", str(combined),
            str(args.output)], check=True)
    named = args.output / "prefill-bundle.mlmodelc"
    compiled_candidate = args.output / "combined.mlmodelc"
    if not named.exists() and compiled_candidate.exists():
        compiled_candidate.rename(named)
    if not named.is_dir():
        raise SystemExit(f"Core ML compiler did not create {named}")

    functions_manifest: list[dict[str, object]] = []
    for sequence in args.sequences:
        functions_manifest.append({
            "name": f"prefill_s{sequence}",
            "role": "prefill",
            "bucket": sequence,
            "batch": args.batch,
            "hidden_size": hidden,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
        })
    manifest = {
        "format": "tessera-ane-prefill-bundle-weights-out-v1",
        "architecture": "gemma4",
        "execution_stage": "layer_slab",
        "layer_first": 0,
        "layer_last": 0,
        "hidden_layout": "token_major.f32.v1",
        "kv_layout": "llama.gemma4.kv_rows.f16.v1",
        "cache_requirement": "empty_contiguous_prompt",
        "batch": args.batch,
        "hidden_size": hidden,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "sequence_buckets": list(args.sequences),
        "functions": functions_manifest,
        "weight_layout": "gemma4.blk0.v1",
        "weight_inputs": [
            "embedded", "attn_norm", "q_weight", "k_weight", "v_weight",
            "q_norm", "k_norm", "o_weight", "post_attn", "ffn_norm",
            "gate_weight", "up_weight", "down_weight", "post_ffn", "scale",
        ],
    }
    (args.output / "prefill-bundle.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {named}")
    print(f"wrote {args.output / 'prefill-bundle.json'}")


if __name__ == "__main__":
    main()
