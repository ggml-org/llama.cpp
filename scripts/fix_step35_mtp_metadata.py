#!/usr/bin/env python3
"""
Fix Step-3.5 GGUF metadata for MTP support.

Old (pre-MTP) Step-3.5 GGUFs were written with `step35.block_count = num_hidden_layers`
and per-layer arrays sized to the same length, so the appended MTP blocks have
no metadata slot. This script:

  * sets `step35.block_count = num_hidden_layers + num_nextn_predict_layers`
  * appends `num_nextn_predict_layers` entries to every known per-layer array
    (head_count, head_count_kv, sliding_window_pattern, swiglu.clamp_exp,
     swiglu.clamp_shexp) so the C++ loader's length check passes
  * writes `step35.nextn_predict_layers`
  * copies all tensors over unchanged

Defaults assume the MTP blocks are `sliding_attention` (Step-3.5-Flash):
head_count=96, head_count_kv=8, swa=True, swiglu_clamp_exp=0, swiglu_clamp_shexp=0.
Override with the per-array flags below if your model differs.

Run conversion with the up-to-date `conversion/step3.py` to produce a correct
GGUF in the first place; this script exists only to retrofit older outputs.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

# Allow running from a llama.cpp checkout without installing gguf-py.
if "NO_LOCAL_GGUF" not in os.environ:
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "gguf-py"))

import gguf  # noqa: E402

logger = logging.getLogger("fix-step35-mtp-metadata")


# Per-layer metadata keys (step35-specific) we know how to extend.
PER_LAYER_KEYS: dict[str, gguf.GGUFValueType] = {
    "step35.attention.head_count":             gguf.GGUFValueType.UINT32,
    "step35.attention.head_count_kv":          gguf.GGUFValueType.UINT32,
    "step35.attention.sliding_window_pattern": gguf.GGUFValueType.BOOL,
    "step35.swiglu_clamp_exp":                 gguf.GGUFValueType.FLOAT32,
    "step35.swiglu_clamp_shexp":               gguf.GGUFValueType.FLOAT32,
}

BLOCK_COUNT_KEY  = "step35.block_count"
NEXTN_LAYERS_KEY = "step35.nextn_predict_layers"


def get_field_contents(reader: gguf.GGUFReader, key: str) -> Any:
    field = reader.get_field(key)
    return field.contents() if field else None


def field_main_type(reader: gguf.GGUFReader, key: str) -> gguf.GGUFValueType | None:
    field = reader.get_field(key)
    if field is None or not field.types:
        return None
    return field.types[0]


def make_extended_value(
    reader:    gguf.GGUFReader,
    key:       str,
    sub_type:  gguf.GGUFValueType,
    n_total:   int,
    mtp_value: Any,
    n_mtp:     int,
) -> list[Any] | None:
    """Return the new array for ``key`` (length ``n_total``) or None if the key
    is absent and no broadcast is needed."""
    field = reader.get_field(key)
    if field is None:
        return None

    main_type = field.types[0]
    if main_type == gguf.GGUFValueType.ARRAY:
        current: list[Any] = list(field.contents())
        if len(current) >= n_total:
            logger.info("  %s already length %d (>= %d), trimming", key, len(current), n_total)
            return current[:n_total]
        pad = n_total - len(current)
        logger.info("  %s: extending %d -> %d (appending %d × %r)", key, len(current), n_total, pad, mtp_value)
        return current + [mtp_value] * pad

    # Scalar — broadcast to length n_total, replacing the trailing n_mtp entries.
    scalar = field.contents()
    logger.info("  %s: scalar %r -> array of %d (last %d = %r)", key, scalar, n_total, n_mtp, mtp_value)
    return [scalar] * (n_total - n_mtp) + [mtp_value] * n_mtp


def _bool(s: str) -> bool:
    t = s.strip().lower()
    if t in ("1", "true", "yes", "y"):
        return True
    if t in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected boolean, got {s!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input",  type=Path, help="input GGUF (Step-3.5, trunk-only)")
    parser.add_argument("output", type=Path, help="output GGUF (overwritten if exists)")
    parser.add_argument("--n-mtp", type=int, default=3,
                        help="number of MTP blocks to append (default: 3, matching Step-3.5-Flash)")

    # Per-MTP-layer values. Defaults are correct for Step-3.5-Flash whose MTP
    # blocks are `sliding_attention` type.
    parser.add_argument("--head-count",        type=int,   default=96,
                        help="MTP layer head_count (default 96, sliding_attention)")
    parser.add_argument("--head-count-kv",     type=int,   default=8,
                        help="MTP layer head_count_kv (default 8)")
    parser.add_argument("--swa",               type=_bool, default=True,
                        help="MTP layer sliding-window flag (default true)")
    parser.add_argument("--swiglu-clamp-exp",  type=float, default=0.0,
                        help="MTP layer swiglu clamp exp (default 0.0 = no clamp)")
    parser.add_argument("--swiglu-clamp-shexp", type=float, default=0.0,
                        help="MTP layer swiglu clamp shexp (default 0.0)")

    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s: %(message)s")

    if args.n_mtp <= 0:
        logger.error("--n-mtp must be > 0; got %d", args.n_mtp)
        sys.exit(1)

    logger.info("Reading %s", args.input)
    reader = gguf.GGUFReader(args.input, "r")

    arch = get_field_contents(reader, gguf.Keys.General.ARCHITECTURE)
    if arch != "step35":
        logger.error("Expected arch 'step35', got %r — this script is Step-3.5-specific.", arch)
        sys.exit(1)

    block_count_field = reader.get_field(BLOCK_COUNT_KEY)
    if block_count_field is None:
        logger.error("Missing %s in input GGUF.", BLOCK_COUNT_KEY)
        sys.exit(1)
    block_count = int(block_count_field.contents())

    existing_nextn = get_field_contents(reader, NEXTN_LAYERS_KEY)
    if existing_nextn is not None:
        # block_count already includes the MTP blocks; back them out so the
        # math below is independent of whether the input is old-style
        # (block_count = trunk) or new-style (block_count = trunk + nextn).
        n_main = block_count - int(existing_nextn)
        logger.info("Input declares nextn_predict_layers=%d; trunk has %d main blocks.",
                    existing_nextn, n_main)
    else:
        n_main = block_count
        logger.info("Input has no nextn_predict_layers key; treating block_count=%d as the trunk count.",
                    block_count)

    n_total = n_main + args.n_mtp
    logger.info("Block count: trunk=%d, MTP=%d -> total=%d", n_main, args.n_mtp, n_total)

    per_key_value: dict[str, Any] = {
        "step35.attention.head_count":             args.head_count,
        "step35.attention.head_count_kv":          args.head_count_kv,
        "step35.attention.sliding_window_pattern": args.swa,
        "step35.swiglu_clamp_exp":                 args.swiglu_clamp_exp,
        "step35.swiglu_clamp_shexp":               args.swiglu_clamp_shexp,
    }

    # Pre-compute the new array values so we can write them via the writer's
    # standard add_key_value() path.
    new_arrays: dict[str, list[Any]] = {}
    for key, sub_type in PER_LAYER_KEYS.items():
        new_val = make_extended_value(reader, key, sub_type, n_total, per_key_value[key], args.n_mtp)
        if new_val is not None:
            new_arrays[key] = new_val

    logger.info("Writing %s", args.output)
    writer = gguf.GGUFWriter(
        path  = args.output,
        arch  = arch,
        endianess = reader.endianess,
    )

    # Pass 1: copy every existing KV except those we're rewriting.
    rewritten = set(new_arrays.keys()) | {BLOCK_COUNT_KEY, NEXTN_LAYERS_KEY}
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        if field.name in rewritten:
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)

    # Pass 2: rewritten metadata.
    writer.add_uint32(BLOCK_COUNT_KEY,  n_total)
    writer.add_uint32(NEXTN_LAYERS_KEY, args.n_mtp)

    for key, values in new_arrays.items():
        sub_type = PER_LAYER_KEYS[key]
        writer.add_key_value(key, values, gguf.GGUFValueType.ARRAY, sub_type=sub_type)

    # Tensors: copy unchanged.
    total_bytes = 0
    for tensor in reader.tensors:
        total_bytes += tensor.n_bytes
        writer.add_tensor_info(
            tensor.name,
            tensor.data.shape,
            tensor.data.dtype,
            tensor.data.nbytes,
            tensor.tensor_type,
        )

    bar = tqdm(desc="Writing", total=total_bytes, unit="byte", unit_scale=True)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    for tensor in reader.tensors:
        writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
        bar.update(tensor.n_bytes)

    writer.close()
    bar.close()
    logger.info("Done. Wrote %s with block_count=%d, nextn_predict_layers=%d.",
                args.output, n_total, args.n_mtp)


if __name__ == "__main__":
    main()
