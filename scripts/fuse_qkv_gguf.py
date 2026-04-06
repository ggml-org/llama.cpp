#!/usr/bin/env python3
"""Fuse Q/K/V tensors in an existing GGUF file into a single QKV tensor.

This script operates at the binary level to preserve ALL metadata (including
tokenizer) byte-for-byte from the original file.

Usage:
    python scripts/fuse_qkv_gguf.py input.gguf output.gguf
"""
import sys, struct, os, re
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gguf-py'))
from gguf import GGUFReader


def align_offset(offset, alignment=32):
    return (offset + alignment - 1) // alignment * alignment


def write_tensor_info(f, name, n_dims, dims, tensor_type, data_offset):
    """Write one tensor info entry in GGUF format."""
    name_bytes = name.encode('utf-8')
    f.write(struct.pack('<Q', len(name_bytes)))
    f.write(name_bytes)
    f.write(struct.pack('<I', n_dims))
    for d in dims:
        f.write(struct.pack('<Q', d))
    f.write(struct.pack('<I', tensor_type))
    f.write(struct.pack('<Q', data_offset))


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} input.gguf output.gguf")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    print(f"Reading {input_path}...")
    reader = GGUFReader(input_path)

    with open(input_path, 'rb') as f:
        magic = f.read(4)
        version = struct.unpack('<I', f.read(4))[0]
        n_tensors_orig = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]

    print(f"  Version: {version}, Tensors: {n_tensors_orig}, KV fields: {n_kv}")

    ti_start = min(t.field.offset for t in reader.tensors)
    kv_data_start = 24
    kv_data_end = ti_start

    print(f"  KV data: {kv_data_start} - {kv_data_end} ({kv_data_end - kv_data_start} bytes)")

    tensor_map = {t.name: t for t in reader.tensors}

    qkv_pattern = re.compile(r'^blk\.(\d+)\.attn_([qkv])\.weight$')
    layer_qkv = {}
    fused_names = set()

    for name, t in tensor_map.items():
        m = qkv_pattern.match(name)
        if m:
            layer = int(m.group(1))
            qkv_type = m.group(2)
            if layer not in layer_qkv:
                layer_qkv[layer] = {}
            layer_qkv[layer][qkv_type] = t
            fused_names.add(name)

    fuse_layers = sorted([l for l, d in layer_qkv.items() if len(d) == 3])
    print(f"  Fusing {len(fuse_layers)} layers: {fuse_layers[0]}-{fuse_layers[-1]}")

    output_tensors = []
    seen_layers = set()

    for t in reader.tensors:
        if t.name in fused_names:
            m = qkv_pattern.match(t.name)
            layer = int(m.group(1))
            if layer in seen_layers:
                continue
            seen_layers.add(layer)

            q = layer_qkv[layer]['q']
            k = layer_qkv[layer]['k']
            v = layer_qkv[layer]['v']

            assert q.tensor_type == k.tensor_type == v.tensor_type

            q_dims = [int(x) for x in q.field.parts[3]]
            k_dims = [int(x) for x in k.field.parts[3]]
            v_dims = [int(x) for x in v.field.parts[3]]

            assert q_dims[0] == k_dims[0] == v_dims[0]

            fused_ne0 = q_dims[0]
            fused_ne1 = q_dims[1] + k_dims[1] + v_dims[1]
            fused_name = f"blk.{layer}.attn_qkv.weight"

            fused_data = np.concatenate([
                np.array(q.data, copy=True),
                np.array(k.data, copy=True),
                np.array(v.data, copy=True),
            ])

            print(f"  Layer {layer}: Q{q_dims}+K{k_dims}+V{v_dims} -> QKV[{fused_ne0},{fused_ne1}]  {fused_data.nbytes} bytes")

            output_tensors.append((fused_name, 2, [fused_ne0, fused_ne1],
                                   int(q.tensor_type), fused_data.tobytes()))
        else:
            dims = [int(x) for x in t.field.parts[3]]
            n_dims = int(t.field.parts[2][0])
            output_tensors.append((t.name, n_dims, dims,
                                   int(t.tensor_type), bytes(t.data)))

    n_tensors_new = len(output_tensors)
    print(f"\n  {n_tensors_orig} -> {n_tensors_new} tensors")

    with open(input_path, 'rb') as f:
        f.seek(kv_data_start)
        kv_data_bytes = f.read(kv_data_end - kv_data_start)

    print(f"\nWriting {output_path}...")
    alignment = 32

    with open(output_path, 'wb') as f:
        f.write(magic)
        f.write(struct.pack('<I', version))
        f.write(struct.pack('<Q', n_tensors_new))
        f.write(struct.pack('<Q', n_kv))
        f.write(kv_data_bytes)

        data_offsets = []
        current_data_offset = 0
        for name, n_dims, dims, ttype, data in output_tensors:
            aligned = align_offset(current_data_offset, alignment)
            data_offsets.append(aligned)
            current_data_offset = aligned + len(data)

        for i, (name, n_dims, dims, ttype, data) in enumerate(output_tensors):
            write_tensor_info(f, name, n_dims, dims, ttype, data_offsets[i])

        ti_section_end = f.tell()
        tensor_data_start = align_offset(ti_section_end, alignment)
        if tensor_data_start > ti_section_end:
            f.write(b'\x00' * (tensor_data_start - ti_section_end))

        for i, (name, n_dims, dims, ttype, data) in enumerate(output_tensors):
            current_pos = f.tell() - tensor_data_start
            target_pos = data_offsets[i]
            if target_pos > current_pos:
                f.write(b'\x00' * (target_pos - current_pos))
            f.write(data)

        final_size = f.tell()

    print(f"  Output size: {final_size / 1e9:.2f} GB")
    print("  Done!")


if __name__ == '__main__':
    main()
