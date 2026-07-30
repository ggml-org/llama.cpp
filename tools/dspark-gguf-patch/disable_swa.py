#!/usr/bin/env python3
import struct
fn = '/Volumes/Julian T7/models/drafters/dspark_gemma4_12b_q4pure_v2.gguf'
key = b'dflash.attention.sliding_window'
with open(fn, 'rb') as f:
    data = f.read()
idx = data.find(key)
print(f'Key at offset {idx}')
key_len = struct.unpack('<Q', data[idx-8:idx])[0]
print(f'key_len: {key_len}')
vtype_offset = idx + key_len
vtype = struct.unpack('<I', data[vtype_offset:vtype_offset+4])[0]
print(f'vtype: {vtype}')
val_offset = vtype_offset + 4
val = struct.unpack('<I', data[val_offset:val_offset+4])[0]
print(f'val: {val}')

new_data = data[:val_offset] + struct.pack('<I', 0) + data[val_offset+4:]
with open(fn, 'wb') as f:
    f.write(new_data)
print(f'Patched sliding_window from {val} to 0')

with open(fn, 'rb') as f:
    d = f.read()
val = struct.unpack('<I', d[val_offset:val_offset+4])[0]
print(f'New value: {val}')
