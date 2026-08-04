#!/usr/bin/env python3
"""Export a single multifunction Gemma 4 ANE prefill bundle.

A bundle is one Core ML mlmodelc that contains every `prefill_sN` function the
target runtime needs.  The Core ML multifunction machinery lets the converter
share the underlying weight table across functions, so the artifact's on-disk
size grows with the number of *unique* weight tables, not the number of
sequence buckets.  Every published function targets the same Gemma 4 layer 0
so all of them collapse onto a single weight file.

The manifest records the declared sequence buckets and the per-function
ABIs in the order they are exported.  The C++ runtime materializes the
multifunction package from a single embedded bundle and warms each function
under its declared name.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import coremltools as ct
import torch
import torch.nn.functional as F
from safetensors import safe_open

# The ane_state_layout.v1 manifest is the contract between this
# converter and the runtime's stateless-with-IOSurface-state
# design (common/ane-mtp.mm + ggml/src/ggml-ane/ggml-ane.mm).
# The converter emits it directly next to the .mlmodelc; the
# runtime reads it via the shared reader in
# common/ane-state-layout.h. See tools/ane-mtp/state_layout.py
# for the schema and tools/ane-mtp/test_emit_manifest.py
# for the 9-case unit suite.
sys.path.insert(0, str(Path(__file__).parent))
from state_layout import (  # noqa: E402
    ANE_MIN_ALLOC_BYTES,
    ANE_PAGE_BYTES,
    ANE_SIMD_ALIGN,
    DTYPE_BYTES,
    FunctionSpec,
    ROLE_PREFILL,
    StateLayout,
    StateSlot,
    manifest_path_for,
)


EPS = 1.0e-6


def load_tensor(source: Path, name: str) -> torch.Tensor:
    with safe_open(source / "model.safetensors", framework="pt", device="cpu") as f:
        return f.get_tensor(name).to(torch.float16)


def rms_norm(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + EPS)).to(
        value.dtype
    ) * (weight + 1.0)


def rms_norm_unweighted(value: torch.Tensor) -> torch.Tensor:
    return (value * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + EPS)).to(
        value.dtype
    )


def rope(value: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    pairs = value.shape[-1] // 2
    inv_freq = theta ** (-torch.arange(pairs, dtype=torch.float32) / pairs)
    angle = positions.float()[..., None] * inv_freq
    cosine = torch.cos(angle).to(value.dtype).unsqueeze(-2)
    sine = torch.sin(angle).to(value.dtype).unsqueeze(-2)
    first, second = value[..., :pairs], value[..., pairs:]
    return torch.cat((first * cosine - second * sine, second * cosine + first * sine), -1)


class Gemma4InitialSlab(torch.nn.Module):
    """One initial sliding-attention Gemma 4 layer as a self-contained ANE slab."""

    def __init__(self, source: Path, hidden: int, heads: int, kv_heads: int, head_dim: int) -> None:
        super().__init__()
        root = "model.language_model"
        layer = f"{root}.layers.0"
        names = {
            "embedding": f"{root}.embed_tokens.weight",
            "attn_norm": f"{layer}.input_layernorm.weight",
            "q": f"{layer}.self_attn.q_proj.weight",
            "k": f"{layer}.self_attn.k_proj.weight",
            "v": f"{layer}.self_attn.v_proj.weight",
            "q_norm": f"{layer}.self_attn.q_norm.weight",
            "k_norm": f"{layer}.self_attn.k_norm.weight",
            "o": f"{layer}.self_attn.o_proj.weight",
            "post_attn": f"{layer}.post_attention_layernorm.weight",
            "ffn_norm": f"{layer}.pre_feedforward_layernorm.weight",
            "gate": f"{layer}.mlp.gate_proj.weight",
            "up": f"{layer}.mlp.up_proj.weight",
            "down": f"{layer}.mlp.down_proj.weight",
            "post_ffn": f"{layer}.post_feedforward_layernorm.weight",
            "scale": f"{layer}.layer_scalar",
        }
        for local, remote in names.items():
            self.register_buffer(local, load_tensor(source, remote))
        self.register_buffer("v_norm_inverse", 1.0 / (self.k_norm + 1.0))
        self.hidden = hidden
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim

    def forward(self, token_ids: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, sequence = token_ids.shape
        current = F.embedding(token_ids.to(torch.int64), self.embedding) * math.sqrt(self.hidden)
        normed = rms_norm(current, self.attn_norm)
        query = F.linear(normed, self.q).reshape(batch, sequence, self.heads, self.head_dim)
        keys = F.linear(normed, self.k).reshape(batch, sequence, self.kv_heads, self.head_dim)
        values = F.linear(normed, self.v).reshape(batch, sequence, self.kv_heads, self.head_dim)
        query = rope(rms_norm(query, self.q_norm), positions, 10000.0)
        keys = rope(rms_norm(keys, self.k_norm), positions, 10000.0)
        values = rms_norm(values, self.k_norm) * self.v_norm_inverse
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
        attended = rms_norm(F.linear(attended, self.o), self.post_attn) + current
        ffn_input = rms_norm(attended, self.ffn_norm)
        ffn = F.gelu(F.linear(ffn_input, self.gate), approximate="tanh") * F.linear(ffn_input, self.up)
        output = (rms_norm(F.linear(ffn, self.down), self.post_ffn) + attended) * self.scale
        return output, keys.reshape(batch, sequence, -1), values.reshape(batch, sequence, -1)


def export_function(module: Gemma4InitialSlab, sequence: int, batch: int) -> Path:
    """Convert a single sequence bucket into a temp .mlpackage."""
    token_ids = torch.zeros((batch, sequence), dtype=torch.int32)
    positions = torch.arange(sequence, dtype=torch.int32).repeat(batch, 1)
    program = torch.export.export(module, (token_ids, positions)).run_decompositions({})
    converted = ct.convert(
        program,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="token_ids", shape=token_ids.shape, dtype=int),
            ct.TensorType(name="positions", shape=positions.shape, dtype=int),
        ],
        outputs=[
            ct.TensorType(name="hidden_states"),
            ct.TensorType(name="key_states"),
            ct.TensorType(name="value_states"),
        ],
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
    )
    package = Path(tempfile.mkdtemp(prefix=f"gemma4-prefill-s{sequence}-")) / "prefill.mlpackage"
    converted.save(str(package))
    return package


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
            help="destination directory; bundle is written here as prefill-bundle.mlmodelc")
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

    module = Gemma4InitialSlab(args.source, hidden, heads, kv_heads, head_dim).eval()
    args.output.mkdir(parents=True, exist_ok=True)
    package_paths: list[tuple[int, Path]] = []
    for sequence in args.sequences:
        package = export_function(module, sequence, args.batch)
        package_paths.append((sequence, package))
        print(f"exported prefill_s{sequence} to {package}")

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
    if not named.exists():
        # coremlcompiler sometimes names the output after the package stem; rename if so.
        if compiled_candidate.exists():
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
        "format": "tessera-ane-prefill-bundle-v1",
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
    }
    (args.output / "prefill-bundle.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {named}")
    print(f"wrote {args.output / 'prefill-bundle.json'}")

    # Emit the ane_state_layout.v1 manifest sidecar next to the
    # .mlmodelc. The runtime reads this JSON via the shared
    # reader in common/ane-state-layout.h to allocate the state
    # IOSurface and pin the per-function input/output slots.
    # We introspect the compiled .mlmodelc's metadata.json (the
    # same source the bridge tool uses) rather than re-deriving
    # the schema from coremltools' in-memory model, because the
    # .mlmodelc is what the runtime actually loads; the schema
    # may differ slightly from the source .mlpackage due to
    # Core ML's MIL optimization passes.
    state_manifest = build_state_layout_v1_manifest(
        mlmodelc_dir=named,
        bundle_name="prefill-bundle",
    )
    state_manifest.write_json(manifest_path_for(named, "prefill-bundle"))
    print(f"wrote {manifest_path_for(named, 'prefill-bundle')}")


def build_state_layout_v1_manifest(mlmodelc_dir: Path,
                                    bundle_name: str) -> StateLayout:
    """Read the .mlmodelc's metadata.json and build the
    ane_state_layout.v1 manifest. This is the production
    counterpart of the bridge tool
    (tools/ane-mtp/emit_manifest_from_mlmodelc.py); both
    produce the same JSON format. Kept here so the converter
    and the bridge can't drift.
    """
    metadata_path = mlmodelc_dir / "metadata.json"
    if not metadata_path.is_file():
        raise SystemExit(f"no metadata.json at {metadata_path}")
    with metadata_path.open() as f:
        meta = json.load(f)
    if not isinstance(meta, list) or len(meta) != 1:
        raise SystemExit(f"unexpected metadata.json shape: {type(meta)}")
    meta = meta[0]
    model_type_str = meta.get("modelType", {}).get("name", "")
    model_type = ("ml_program" if "mlProgram" in model_type_str
                  else "neural_network")
    functions = meta.get("functions") or []
    if not functions:
        raise SystemExit("metadata.json has no functions")

    def parse_shape(shape) -> list[int]:
        if isinstance(shape, list):
            return [int(d) for d in shape]
        if isinstance(shape, str):
            s = shape.strip().lstrip("[").rstrip("]")
            if not s:
                return []
            return [int(d.strip()) for d in s.split(",")]
        raise TypeError(f"unexpected shape type: {type(shape)}")

    def slot_bytes(dtype: str, shape) -> int:
        esize = DTYPE_BYTES[dtype.lower()]
        count = 1
        for d in parse_shape(shape):
            count *= d
        raw = count * esize
        return ((raw + ANE_SIMD_ALIGN - 1) // ANE_SIMD_ALIGN) * ANE_SIMD_ALIGN

    def parse_role(name: str) -> str:
        head = name.split("_", 1)[0]
        return {
            "prefill": "prefill",
            "mtp": "mtp",
            "dflash": "dflash",
            "hybrid": "hybrid",
            "sync": "sync",
            "reset": "reset",
        }.get(head, head)

    def parse_bucket(name: str) -> int:
        parts = name.split("_")
        if len(parts) < 2:
            return 0
        if parts[0] not in ("prefill", "dflash", "hybrid"):
            return 0
        try:
            return int(parts[1].lstrip("bsBS"))
        except ValueError:
            return 0

    def cml_dtype_to_ane(dtype: str) -> str:
        return {"Int32": "i32", "Float32": "f32", "Float16": "f16"}.get(
            dtype, dtype.lower())

    slots: list[StateSlot] = []
    functions_out: list[FunctionSpec] = []
    offset = 0
    for func in functions:
        fname = func["name"]
        role_str = parse_role(fname)
        bucket = parse_bucket(fname)
        is_ane = role_str not in ("sync", "reset")
        in_ids: list[int] = []
        out_ids: list[int] = []
        for inp in func.get("inputSchema", []):
            iname = inp["name"]
            idtype = inp["dataType"]
            ishape = parse_shape(inp["shape"])
            s = StateSlot(
                name=f"{fname}.{iname}",
                kind="input",
                dtype=cml_dtype_to_ane(idtype),
                shape=ishape,
                offset=offset,
                size_bytes=slot_bytes(idtype, ishape),
            )
            offset += s.size_bytes
            slots.append(s)
            in_ids.append(len(slots) - 1)
        for outp in func.get("outputSchema", []):
            oname = outp["name"]
            odtype = outp["dataType"]
            oshape = parse_shape(outp["shape"])
            kind = "state" if oname in ("key_states", "value_states") else "output"
            s = StateSlot(
                name=f"{fname}.{oname}",
                kind=kind,
                dtype=cml_dtype_to_ane(odtype),
                shape=oshape,
                offset=offset,
                size_bytes=slot_bytes(odtype, oshape),
            )
            offset += s.size_bytes
            slots.append(s)
            out_ids.append(len(slots) - 1)
        functions_out.append(FunctionSpec(
            name=fname,
            role=role_str,
            bucket=bucket,
            stateful=True,
            input_slots=[slots[i].name for i in in_ids],
            output_slots=[slots[i].name for i in out_ids],
            core_ml_function_name=fname,
            use_ane=is_ane,
        ))
    state_size = ((offset + ANE_PAGE_BYTES - 1) // ANE_PAGE_BYTES) * ANE_PAGE_BYTES
    if state_size < ANE_MIN_ALLOC_BYTES:
        state_size = ANE_MIN_ALLOC_BYTES
    return StateLayout(
        version=1,
        bundle_name=bundle_name,
        state_size_bytes=state_size,
        model_type=model_type,
        slots=slots,
        functions=functions_out,
        dependencies=[],
    )


if __name__ == "__main__":
    main()
