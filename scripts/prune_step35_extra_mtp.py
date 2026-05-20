#!/usr/bin/env python3
"""
Prune a Step-3.5 GGUF down to just the first MTP block.

The runtime only uses the first MTP block (single-block-MTP, Qwen/vLLM
style); trailing MTP blocks are loaded with TENSOR_NOT_REQUIRED so a pruned
GGUF works without further surgery. This script does the surgery: it drops
all tensors for blocks `blk.{n_main+1}..blk.{n_total-1}`, rewrites the
per-layer metadata arrays + block_count + nextn_predict_layers so the loader
sees the slimmer model, and writes a new GGUF. Saves ~one MTP block of
weights per pruned block (single-digit GB on Step-3.5-Flash).

  n_main = block_count - nextn_predict_layers   (transformer trunk)
  keep   = blk.0 .. blk.n_main                  (trunk + first MTP block)
  drop   = blk.{n_main+1} .. blk.{block_count-1}

After pruning the output has block_count = n_main + 1 and
nextn_predict_layers = 1.

Example:
  ./scripts/prune_step35_extra_mtp.py step3p5-flash-full.gguf step3p5-flash-mtp1.gguf
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

# Allow running from a llama.cpp checkout without installing gguf-py.
if "NO_LOCAL_GGUF" not in os.environ:
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "gguf-py"))

import gguf  # noqa: E402

logger = logging.getLogger("prune-step35-extra-mtp")

# Per-layer metadata keys (step35-specific) that we need to trim along with
# block_count. Matches the schema fix_step35_mtp_metadata.py knows about.
PER_LAYER_KEYS: dict[str, gguf.GGUFValueType] = {
    "step35.attention.head_count":             gguf.GGUFValueType.UINT32,
    "step35.attention.head_count_kv":          gguf.GGUFValueType.UINT32,
    "step35.attention.sliding_window_pattern": gguf.GGUFValueType.BOOL,
    "step35.swiglu_clamp_exp":                 gguf.GGUFValueType.FLOAT32,
    "step35.swiglu_clamp_shexp":               gguf.GGUFValueType.FLOAT32,
}

BLOCK_COUNT_KEY  = "step35.block_count"
NEXTN_LAYERS_KEY = "step35.nextn_predict_layers"

# Tensors with a "blk.<N>." prefix belong to block N. Anything that doesn't
# match (token_embd, output_norm, output, ...) is kept unconditionally.
BLOCK_TENSOR_RE = re.compile(r"^blk\.(\d+)\.")


def get_field_contents(reader: gguf.GGUFReader, key: str) -> Any:
    field = reader.get_field(key)
    return field.contents() if field else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input",  type=Path, help="input GGUF (Step-3.5 with full MTP)")
    parser.add_argument("output", type=Path, help="output GGUF (overwritten if exists)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s: %(message)s")

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

    nextn_field = reader.get_field(NEXTN_LAYERS_KEY)
    if nextn_field is None:
        logger.error("Input has no %s — nothing to prune (run fix_step35_mtp_metadata.py first if needed).",
                     NEXTN_LAYERS_KEY)
        sys.exit(1)
    n_mtp = int(nextn_field.contents())

    if n_mtp <= 1:
        logger.info("nextn_predict_layers=%d already <= 1; nothing to prune. Copying input unchanged.", n_mtp)

    n_main      = block_count - n_mtp
    keep_first  = n_main                    # first MTP block (index)
    keep_last   = n_main                    # inclusive — only one MTP block kept
    drop_first  = n_main + 1
    drop_last   = block_count - 1           # inclusive

    n_total_new = n_main + 1                # new block_count
    n_mtp_new   = 1                         # new nextn_predict_layers

    logger.info(
        "Pruning plan: trunk %d blocks (0..%d), keep MTP block %d, drop MTP blocks %d..%d -> new block_count=%d",
        n_main, n_main - 1 if n_main > 0 else -1, keep_first, drop_first, drop_last, n_total_new,
    )

    # Per-layer arrays: trim to length n_total_new (drop the trailing n_mtp-1 entries).
    new_arrays: dict[str, list[Any]] = {}
    for key in PER_LAYER_KEYS:
        field = reader.get_field(key)
        if field is None:
            continue
        if field.types[0] != gguf.GGUFValueType.ARRAY:
            # Scalar — irrelevant for per-layer trimming, will be copied as-is.
            continue
        current: list[Any] = list(field.contents())
        if len(current) <= n_total_new:
            logger.info("  %s: already length %d (<= %d), leaving as-is", key, len(current), n_total_new)
            continue
        trimmed = current[:n_total_new]
        logger.info("  %s: trimming length %d -> %d", key, len(current), n_total_new)
        new_arrays[key] = trimmed

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
    writer.add_uint32(BLOCK_COUNT_KEY,  n_total_new)
    writer.add_uint32(NEXTN_LAYERS_KEY, n_mtp_new)

    for key, values in new_arrays.items():
        sub_type = PER_LAYER_KEYS[key]
        writer.add_key_value(key, values, gguf.GGUFValueType.ARRAY, sub_type=sub_type)

    # Tensors: copy those that belong to a kept block (or to no block at all).
    kept_tensors  = []
    dropped_count = 0
    dropped_bytes = 0
    for tensor in reader.tensors:
        m = BLOCK_TENSOR_RE.match(tensor.name)
        if m is not None:
            blk_idx = int(m.group(1))
            if blk_idx > keep_last:
                dropped_count += 1
                dropped_bytes += tensor.n_bytes
                logger.debug("  drop %s (blk.%d)", tensor.name, blk_idx)
                continue
        kept_tensors.append(tensor)

    total_bytes = sum(t.n_bytes for t in kept_tensors)
    logger.info("Tensors: keeping %d (%.2f GB), dropping %d (%.2f GB)",
                len(kept_tensors), total_bytes / (1 << 30),
                dropped_count, dropped_bytes / (1 << 30))

    for tensor in kept_tensors:
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

    for tensor in kept_tensors:
        writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
        bar.update(tensor.n_bytes)

    writer.close()
    bar.close()
    logger.info("Done. Wrote %s (block_count=%d, nextn_predict_layers=%d).",
                args.output, n_total_new, n_mtp_new)


if __name__ == "__main__":
    main()
