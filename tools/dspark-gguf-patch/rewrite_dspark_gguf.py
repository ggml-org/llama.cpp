#!/usr/bin/env python3
"""GGUF rewriter: rename tensors AND add V (copying K data, not shared offset)."""
import os
import struct
import sys
import numpy as np

sys.path.insert(0, '/Users/user/Developer/GitHub/llama.cpp/gguf-py')
from gguf import GGUFReader
from gguf.constants import GGMLQuantizationType

SRC = '/Volumes/Julian T7/models/drafters/dspark_gemma4_12b_q4pure.gguf'
DST = '/Volumes/Julian T7/models/drafters/dspark_gemma4_12b_q4pure_v2.gguf'

RENAMES = {
    b'markov.w1.weight':       b'markov_w1.weight',
    b'markov.w2.weight':       b'markov_w2.weight',
    b'confidence.proj.weight': b'conf_proj.weight',
    b'confidence.proj.bias':   b'conf_proj.bias',
}

print(f'Reading {SRC}...')
reader = GGUFReader(SRC)
real_fields = [(k, f) for k, f in reader.fields.items() if not k.startswith('GGUF.')]

pos = 24
for k, fld in real_fields:
    size = sum(int(p.nbytes) for p in fld.parts)
    pos += size
kv_end = pos

# Parse tensor index
orig_tensors = []
pos = kv_end
with open(SRC, 'rb') as f:
    for i in range(len(reader.tensors)):
        f.seek(pos)
        name_len = struct.unpack('<Q', f.read(8))[0]
        name = f.read(name_len)
        n_dims = struct.unpack('<I', f.read(4))[0]
        dims = struct.unpack(f'<{n_dims}Q', f.read(8 * n_dims))
        ttype = struct.unpack('<I', f.read(4))[0]
        offset = struct.unpack('<Q', f.read(8))[0]
        pos += 8 + name_len + 4 + 8 * n_dims + 4 + 8
        orig_tensors.append({
            'name': name, 'n_dims': n_dims, 'dims': dims, 'type': ttype, 'offset': offset,
            'n_bytes': reader.tensors[i].n_bytes,
        })

print(f'Tensor index ends at {pos}')

data_start = pos
pad = (32 - (data_start % 32)) % 32
data_start += pad
print(f'Data section: {data_start}..{os.path.getsize(SRC)} (with {pad} bytes padding)')

# Read data section
with open(SRC, 'rb') as f:
    f.seek(data_start)
    data_section = bytearray(f.read())

print(f'Data section size: {len(data_section)}')

# Build new tensor list (renames + V copies)
new_tensors = []
v_data_to_append = []  # (tensor_info, raw_data)

for t in orig_tensors:
    new_t = dict(t)
    if t['name'] in RENAMES:
        new_t['name'] = RENAMES[t['name']]
    new_tensors.append(new_t)

# Add V tensors (MQA) — copy K data to a new offset
# We'll append V's data to the end of data section, in the same alignment
for t in orig_tensors:
    if t['name'].startswith(b'blk.') and t['name'].endswith(b'.attn_k.weight'):
        parts = t['name'].split(b'.')
        if parts[0] == b'blk' and parts[1].isdigit():
            v_name = b'blk.' + parts[1] + b'.attn_v.weight'
            if not any(nt['name'] == v_name for nt in new_tensors):
                # V is a copy of K, so same n_bytes
                v_t = dict(t)
                v_t['name'] = v_name
                # Offset will be assigned later (at end of data)
                new_tensors.append(v_t)
                # Get K's data
                k_data_offset = t['offset']
                k_data = bytes(data_section[k_data_offset:k_data_offset + t['n_bytes']])
                v_data_to_append.append((v_t, k_data))
                print(f'  Add V: {v_name.decode()} copy of K ({t["n_bytes"]} bytes)')

# Sort new_tensors by data offset to maintain pack order
# Original order is preserved; new V tensors go at the end
# Calculate new offsets for V tensors (at end of data section)
current_data_size = len(data_section)
# Align to 32
if current_data_size % 32 != 0:
    pad_bytes = 32 - (current_data_size % 32)
    data_section.extend(b'\x00' * pad_bytes)
    current_data_size += pad_bytes

for v_t, v_data in v_data_to_append:
    v_t['offset'] = current_data_size
    data_section.extend(v_data)
    # Pad to alignment
    if len(data_section) % 32 != 0:
        pad_bytes = 32 - (len(data_section) % 32)
        data_section.extend(b'\x00' * pad_bytes)
    current_data_size = len(data_section)

print(f'New data section size: {len(data_section)}')

# Write new file
print(f'Writing {DST}...')
with open(DST, 'wb') as f:
    f.write(b'GGUF')
    with open(SRC, 'rb') as srcf:
        srcf.seek(4)
        f.write(srcf.read(20))  # version, n_tensors (orig), n_kv

    # Fix n_tensors
    f.seek(8)
    f.write(struct.pack('<Q', len(new_tensors)))
    f.seek(24)

    # Copy KV section
    with open(SRC, 'rb') as srcf:
        srcf.seek(24)
        f.write(srcf.read(kv_end - 24))

    # New tensor index
    for nt in new_tensors:
        f.write(struct.pack('<Q', len(nt['name'])))
        f.write(nt['name'])
        f.write(struct.pack('<I', nt['n_dims']))
        f.write(struct.pack(f'<{nt["n_dims"]}Q', *nt['dims']))
        f.write(struct.pack('<I', nt['type']))
        f.write(struct.pack('<Q', nt['offset']))

    # Alignment padding
    pos_now = f.tell()
    pad = (32 - (pos_now % 32)) % 32
    f.write(b'\x00' * pad)
    print(f'New data section starts at {f.tell()} (with {pad} bytes padding)')

    # Data section
    f.write(data_section)

print(f'Done. Wrote {DST}, size {os.path.getsize(DST)} (source: {os.path.getsize(SRC)})')

# Verify
new_reader = GGUFReader(DST)
print(f'\nVerification: {len(new_reader.tensors)} tensors')
for t in new_reader.tensors:
    if t.name in ['markov_w1.weight', 'markov_w2.weight', 'conf_proj.weight', 'conf_proj.bias']:
        print(f'  [OK] {t.name} shape={t.shape} dtype={t.tensor_type.name}')
for block in range(5):
    vn = f'blk.{block}.attn_v.weight'
    matches = [t for t in new_reader.tensors if t.name == vn]
    if matches:
        t = matches[0]
        print(f'  [OK] {vn} shape={t.shape} dtype={t.tensor_type.name} data_offset={t.data_offset} n_bytes={t.n_bytes}')
