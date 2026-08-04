#!/usr/bin/env python3
"""Emit an ane_state_layout.v1 manifest from an existing .mlmodelc.

This is a bridge tool: it takes a real multifunction .mlmodelc
(such as /Volumes/Julian T7/models/gemma4-ane-prefill-bundle/
prefill-bundle.mlmodelc) and writes a manifest sidecar that
the new common/ane-mtp.mm load path can consume. It reads the
.mlmodelc's metadata.json (the Core ML runtime artifact, not
the converter-side .mlpackage) and produces the per-function
input/output slot table, sizes, and offsets that the
IOSurface-mapped stateful design needs.

The script does not regenerate the .mlmodelc. It just emits
the manifest sidecar that the existing converter tooling
emits as part of the bundle production. Use it as a one-time
adapter for the existing gemma4 prefill bundle so the new
load path has something to validate against.

Usage:
    python3 emit-manifest-from-mlmodelc.py \\
        --mlmodelc /path/to/bundle.mlmodelc \\
        --output /path/to/bundle.ane_state.v1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Slot layout: 16-byte aligned, ANE-friendly. Per the
# conversion-design Section 4, the state IOSurface is one big
# buffer; each function's input/output slots live at fixed
# offsets. We allocate slot-by-slot (per-function, not shared
# across functions) to keep the dispatch path's bookkeeping
# simple; a follow-on optimization could share input slots
# across bucket variants (the gemma4 case has 3 prefill
# variants that all consume token_ids + positions of
# different sizes, so a single shared input slot is not
# possible here).
ANE_SIMD_ALIGN = 16
ANE_PAGE_BYTES = 16 * 1024
ANE_MIN_ALLOC_BYTES = 64 * 1024

DTYPE_BYTES = {"Int32": 4, "Float32": 4, "Float16": 2}

ROLE_BY_FUNCTION_NAME = {
    "prefill": "prefill",
    "mtp": "mtp",
    "dflash": "dflash",
    "hybrid": "hybrid",
    "sync": "sync",
    "reset": "reset",
}


def parse_shape(shape) -> list[int]:
    """Parse a Core ML metadata shape (string '[1, 128]' or list [1, 128])."""
    if isinstance(shape, list):
        return [int(d) for d in shape]
    if isinstance(shape, str):
        s = shape.strip().lstrip("[").rstrip("]")
        if not s:
            return []
        return [int(d.strip()) for d in s.split(",")]
    raise TypeError(f"unexpected shape type: {type(shape)}")


def slot_size(dtype: str, shape) -> int:
    """Return the byte size of one slot, padded to ANE_SIMD_ALIGN."""
    esize = DTYPE_BYTES[dtype]
    count = 1
    for d in parse_shape(shape):
        count *= d
    raw = count * esize
    return ((raw + ANE_SIMD_ALIGN - 1) // ANE_SIMD_ALIGN) * ANE_SIMD_ALIGN


def parse_function_role(name: str) -> str:
    """Map a Core ML function name to the ane_state_layout role.

    Convention from the export tool: prefill_sN, mtp, dflash_bN,
    hybrid_bN, sync, reset. We extract the role from the prefix
    before the first underscore (or the whole name if no
    underscore).
    """
    head = name.split("_", 1)[0]
    return ROLE_BY_FUNCTION_NAME.get(head, head)


def parse_function_bucket(name: str) -> int:
    """Return the bucket dimension from a function name, or 0.

    For prefill_sN this is N. For dflash_bN this is N. For
    others, 0.
    """
    parts = name.split("_")
    if len(parts) < 2:
        return 0
    head = parts[0]
    if head not in ("prefill", "dflash", "hybrid"):
        return 0
    try:
        return int(parts[1].lstrip("bsBS"))
    except ValueError:
        return 0


def build_manifest(mlmodelc_dir: Path, bundle_name: str) -> dict:
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

    # Build the slot table. One slot per (function, input/output name).
    # Each function gets its own input slots and output slots; outputs
    # are persistent STATE-kind slots (K/V) or scratch output slots
    # (hidden_states). For the gemma4 prefill case:
    #   - token_ids, positions: INPUT-kind per function
    #   - hidden_states: OUTPUT-kind per function (consumed downstream)
    #   - key_states, value_states: STATE-kind per function (K/V cache)
    slots = []
    functions_out = []
    dependencies = []

    offset = 0
    for func in functions:
        fname = func["name"]
        role_str = parse_function_role(fname)
        bucket = parse_function_bucket(fname)
        is_ane = True
        if role_str in ("sync", "reset"):
            is_ane = False

        func_input_slot_ids = []
        func_output_slot_ids = []

        for inp in func.get("inputSchema", []):
            iname = inp["name"]
            idtype = inp["dataType"]
            ishape = parse_shape(inp["shape"])
            slot = {
                "name": f"{fname}.{iname}",
                "kind": "input",
                "dtype": _dtype_name(idtype),
                "shape": ishape,
                "offset": offset,
                "size_bytes": slot_size(idtype, ishape),
            }
            offset += slot["size_bytes"]
            slots.append(slot)
            func_input_slot_ids.append(len(slots) - 1)

        for outp in func.get("outputSchema", []):
            oname = outp["name"]
            odtype = outp["dataType"]
            oshape = parse_shape(outp["shape"])
            # K/V caches (key_states, value_states) are persistent
            # state; hidden_states is a per-call output.
            kind = "state" if oname in ("key_states", "value_states") else "output"
            slot = {
                "name": f"{fname}.{oname}",
                "kind": kind,
                "dtype": _dtype_name(odtype),
                "shape": oshape,
                "offset": offset,
                "size_bytes": slot_size(odtype, oshape),
            }
            offset += slot["size_bytes"]
            slots.append(slot)
            func_output_slot_ids.append(len(slots) - 1)

        functions_out.append({
            "name": fname,
            "role": role_str,
            "bucket": bucket,
            "stateful": True,
            "input_slots": [slots[i]["name"] for i in func_input_slot_ids],
            "output_slots": [slots[i]["name"] for i in func_output_slot_ids],
            "core_ml_function_name": fname,
            "use_ane": is_ane,
        })

    # Round state_size_bytes up to 16KB page and clamp to the ANE
    # minimum (64KB).
    state_size = ((offset + ANE_PAGE_BYTES - 1) // ANE_PAGE_BYTES) * ANE_PAGE_BYTES
    if state_size < ANE_MIN_ALLOC_BYTES:
        state_size = ANE_MIN_ALLOC_BYTES

    return {
        "version": 1,
        "bundle_name": bundle_name,
        "state_size_bytes": state_size,
        "model_type": model_type,
        "slots": slots,
        "functions": functions_out,
        "dependencies": dependencies,
    }


def _dtype_name(coreml_dtype: str) -> str:
    return {"Int32": "i32", "Float32": "f32", "Float16": "f16"}.get(
        coreml_dtype, coreml_dtype.lower())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlmodelc", type=Path, required=True,
                        help="path to the .mlmodelc directory")
    parser.add_argument("--output", type=Path, required=True,
                        help="output path for the manifest JSON")
    parser.add_argument("--bundle-name", type=str, default=None,
                        help="override the bundle name (default: "
                             "the .mlmodelc directory's stem)")
    args = parser.parse_args()
    bundle_name = args.bundle_name or args.mlmodelc.stem
    manifest = build_manifest(args.mlmodelc, bundle_name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {args.output}", file=sys.stderr)
    print(f"  bundle_name: {manifest['bundle_name']}", file=sys.stderr)
    print(f"  state_size_bytes: {manifest['state_size_bytes']}", file=sys.stderr)
    print(f"  model_type: {manifest['model_type']}", file=sys.stderr)
    print(f"  slots: {len(manifest['slots'])}", file=sys.stderr)
    print(f"  functions: {len(manifest['functions'])}", file=sys.stderr)


if __name__ == "__main__":
    main()
