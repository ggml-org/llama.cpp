#!/usr/bin/env python3
"""
Strip ggml view/reshape annotations (e.g. " (reshaped)") from imatrix tensor
names so they match the real model weight names at quantization time.

Background: when a weight is reshaped inline before ggml_mul_mat (as DeepSeek4
does for blk.N.attn_output_a.weight), the imatrix collector keys the importance
stats by the derived tensor name "<weight> (reshaped)". Quantization looks up the
plain weight name and fails with "Missing importance matrix ...". The data is
present under the wrong key; this rewrites the keys in place (into a new file).

Usage: fix_reshaped_imatrix.py input.imatrix.gguf output.imatrix.gguf
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Load the local gguf package (repo checkout), like the other gguf scripts do.
if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent.parent / "gguf-py").exists():
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "gguf-py"))

import gguf  # noqa: E402

# ggml suffixes appended to derived tensors (see ggml_format_name in ggml.c)
VIEW_SUFFIXES = (" (reshaped)", " (view)", " (cont)", " (transposed)", " (permuted)")
# imatrix stores two tensors per entry
STAT_SUFFIXES = (".in_sum2", ".counts")


def clean_name(name: str) -> str:
    stat = next((s for s in STAT_SUFFIXES if name.endswith(s)), "")
    base = name[: -len(stat)] if stat else name
    changed = True
    while changed:
        changed = False
        for s in VIEW_SUFFIXES:
            if base.endswith(s):
                base = base[: -len(s)]
                changed = True
    return base + stat


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    inp, outp = Path(sys.argv[1]), Path(sys.argv[2])
    print(f"* Loading: {inp}")
    reader = gguf.GGUFReader(inp, "r")

    # imatrix files carry no general.architecture; keep the copy faithful.
    writer = gguf.GGUFWriter(outp, arch="", endianess=reader.endianess)
    writer.kv_data[0].pop(gguf.Keys.General.ARCHITECTURE, None)

    alignment = reader.get_field(gguf.Keys.General.ALIGNMENT)
    if alignment is not None:
        writer.data_alignment = alignment.contents()

    # copy every KV field verbatim (skip virtual GGUF.* and the auto-added arch)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)

    renamed = 0
    for tensor in reader.tensors:
        new_name = clean_name(tensor.name)
        if new_name != tensor.name:
            renamed += 1
        writer.add_tensor_info(new_name, tensor.data.shape, tensor.data.dtype, tensor.data.nbytes, tensor.tensor_type)

    print(f"* Renaming {renamed} of {len(reader.tensors)} tensors")
    print(f"* Writing: {outp}")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor in reader.tensors:
        writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
    writer.close()
    print("* Done")


if __name__ == "__main__":
    main()
