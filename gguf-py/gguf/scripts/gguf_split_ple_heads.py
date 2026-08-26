#!/usr/bin/env python3
"""Split a qwen4exp n-gram table into one tensor per head.

The joined per_layer_token_embd is ~20.9 GiB, past the 4 GiB buffer a Vulkan device
will accept, so it can only ever sit on the host. Each n-gram head is a contiguous row
range of that table and is ~1.3 GiB, which fits. Rows are 160 elements, a whole number
of blocks for every 32-block type, so the heads split on block boundaries and the
quantized bytes are copied through untouched: no dequantize, no requantize, no loss.

Head bounds come from the file's own ple.head_offsets and ple.head_vocab_sizes.
"""
from __future__ import annotations

import argparse
import logging
import re
import os
import sys
from pathlib import Path

if "NO_LOCAL_GGUF" not in os.environ:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import gguf  # noqa: E402

logger = logging.getLogger("gguf-split-ple-heads")

JOINED = "per_layer_token_embd.weight"


def split(readers: list[gguf.GGUFReader], writer: gguf.GGUFWriter) -> None:
    reader = readers[0]
    offs = reader.fields.get("qwen4exp.ple.head_offsets")
    vocs = reader.fields.get("qwen4exp.ple.head_vocab_sizes")
    if offs is None or vocs is None:
        raise ValueError("file has no qwen4exp.ple.head_offsets / head_vocab_sizes; is it qwen4exp?")
    offs = [int(x) for x in offs.contents()]
    vocs = [int(x) for x in vocs.contents()]
    if len(offs) != len(vocs):
        raise ValueError(f"{len(offs)} head offsets but {len(vocs)} vocab sizes")

    for field in reader.fields.values():
        # a split input becomes one file, so its shard bookkeeping must not carry over
        if field.name == "GGUF.tensor_count" or field.name.startswith("split."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)

    tensors = [t for r in readers for t in r.tensors]
    joined = next((t for t in tensors if t.name == JOINED), None)
    if joined is None:
        raise ValueError(f"{JOINED} not found; already split?")
    if joined.data.shape[0] < offs[-1] + vocs[-1]:
        raise ValueError(f"heads need {offs[-1] + vocs[-1]} rows, table has {joined.data.shape[0]}")

    # tensor infos first, in the order the data is written below
    plan = []
    for t in tensors:
        if t.name == JOINED:
            for h, (o, v) in enumerate(zip(offs, vocs)):
                name = gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.PLE_NGRAM_EMBD].format(bid=h) + ".weight"
                data = t.data[o:o + v]
                logger.info(f"{name}: rows {o}..{o + v} ({data.nbytes / 2**30:.2f} GiB)")
                plan.append((name, data, t.tensor_type))
        else:
            plan.append((t.name, t.data, t.tensor_type))

    for name, data, qtype in plan:
        writer.add_tensor_info(name, data.shape, data.dtype, data.nbytes, qtype)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    done = 0
    total = sum(d.nbytes for _, d, _ in plan)
    for name, data, _ in plan:
        writer.write_tensor_data(data, tensor_endianess=reader.endianess)
        done += data.nbytes
        logger.info(f"  {done / total * 100:5.1f}%  {name}")

    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path,
                        help="qwen4exp GGUF with a joined n-gram table; pass shard 1 of a split file "
                             "and the rest are picked up, and the output is written unsplit")
    parser.add_argument("output", type=Path, help="where to write the split file")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    if args.output.exists():
        raise SystemExit(f"{args.output} exists")

    paths = [args.input]
    m = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", args.input.name)
    if m:
        stem, _, total = m.group(1), int(m.group(2)), int(m.group(3))
        paths = [args.input.parent / f"{stem}-{i:05d}-of-{total:05d}.gguf" for i in range(1, total + 1)]
        missing = [p.name for p in paths if not p.exists()]
        if missing:
            raise SystemExit(f"missing shards: {', '.join(missing)}")
        logger.info(f"reading {total} shards")

    readers = [gguf.GGUFReader(p, "r") for p in paths]
    arch = readers[0].fields["general.architecture"].contents()
    writer = gguf.GGUFWriter(args.output, arch=arch, endianess=readers[0].endianess)
    split(readers, writer)
    logger.info(f"wrote {args.output}")


if __name__ == "__main__":
    main()
