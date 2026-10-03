#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Necessary to load the local gguf package
sys.path.insert(0, str(Path(__file__).parent / "gguf-py"))

from gguf import GGMLQuantizationType, GGUFReader, GGUFValueType, GGUFWriter  # noqa: E402
from gguf.constants import LlamaFileType  # noqa: E402
from gguf.quants import dequantize, quantize  # noqa: E402


def bytes_to_human(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f}TiB"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompress a GGUF model to BF16X: near-lossless bfloat16 "
        "recompression at 11.5 bpw (1.39x smaller than BF16). Sign and mantissa "
        "stay exact; only weights already 2^-7 below their block max exponent "
        "(~2%) decode clamped. Requires the last tensor dim to be a multiple of 32.")
    parser.add_argument("input", help="GGUF model to read")
    parser.add_argument("output", help="BF16X GGUF model to write")
    parser.add_argument("--keep-embeddings", action="store_true",
                        help="leave token_embd*/output* tensors in their original format")
    parser.add_argument("--dry-run", action="store_true", help="report sizes without writing")
    args = parser.parse_args()

    reader = GGUFReader(args.input)

    arch_field = reader.fields.get("general.architecture")
    arch = arch_field.contents() if arch_field is not None else "llama"

    n_total = n_converted = 0
    bytes_in = bytes_out = 0
    tensors_out: list[tuple[str, object, GGMLQuantizationType | None]] = []

    for tensor in reader.tensors:
        ttype = tensor.tensor_type
        name = tensor.name
        n_bytes = int(tensor.n_bytes)
        n_total += 1
        bytes_in += n_bytes

        convertible = (
            ttype != GGMLQuantizationType.BF16X
            and tensor.data.dtype != np.uint8  # already-quantized byte blobs go through dequantize below
        )
        try:
            f32 = dequantize(tensor.data, ttype)
        except NotImplementedError:
            f32 = None
            convertible = False

        keep = args.keep_embeddings and ("token_embd" in name or "output" in name)
        if convertible and f32 is not None and f32.shape[-1] % 32 == 0 and not keep:
            packed = quantize(f32, GGMLQuantizationType.BF16X)
            packed_bytes = packed.nbytes
            tensors_out.append((name, packed, GGMLQuantizationType.BF16X))
            n_converted += 1
        else:
            packed_bytes = n_bytes
            tensors_out.append((name, tensor.data, ttype))
        bytes_out += packed_bytes

    print(f"* {n_converted}/{n_total} tensors -> BF16X")
    print(f"* tensor data: {bytes_to_human(bytes_in)} -> {bytes_to_human(bytes_out)} "
          f"({bytes_in / max(bytes_out, 1):.2f}x)")

    if args.dry_run:
        return

    writer = GGUFWriter(args.output, arch=arch)  # type: ignore[arg-type]
    # (the constructor already writes general.architecture)

    skip_keys = {"general.architecture", "general.file_type"}
    for key, field in reader.fields.items():
        if key in skip_keys:
            continue
        main_type = field.types[0]
        val = field.contents()
        if main_type == GGUFValueType.ARRAY:
            writer.add_key_value(key, val, GGUFValueType.ARRAY, sub_type=field.types[-1])
        else:
            writer.add_key_value(key, val, main_type)

    writer.add_file_type(LlamaFileType.MOSTLY_BF16X)

    for name, data, dtype in tensors_out:
        writer.add_tensor(name, data, raw_dtype=dtype)  # type: ignore[arg-type]

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"* wrote {args.output} ({bytes_to_human(os.path.getsize(args.output))})")


if __name__ == "__main__":
    main()
