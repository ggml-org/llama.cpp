#!/usr/bin/env python3
"""Quick check of LoRA tensor norms in a GGUF file."""
import sys, struct, numpy as np

def read_gguf(path):
    with open(path, 'rb') as f:
        assert f.read(4) == b'GGUF'
        version = struct.unpack('<I', f.read(4))[0]
        n_tensors, n_kv = struct.unpack('<QQ', f.read(16))
        # skip KV pairs (simplified - just seek past them)
        # Read tensor infos
        tensors = []
        for _ in range(n_kv):
            # key
            klen = struct.unpack('<Q', f.read(8))[0]
            key = f.read(klen).decode()
            vtype = struct.unpack('<I', f.read(4))[0]
            # skip value based on type (simplified)
            if vtype == 8:   # string
                slen = struct.unpack('<Q', f.read(8))[0]; f.read(slen)
            elif vtype == 6: # float32
                f.read(4)
            elif vtype in (0,1,2,3,4,5,10,11,12): # int types
                sizes = {0:1,1:1,2:2,3:4,4:8,5:1,10:2,11:4,12:8}
                f.read(sizes.get(vtype,4))
            elif vtype == 9: # bool
                f.read(1)
            else:
                print(f"unknown kv type {vtype} for key {key}, stopping"); break

        data_offset = None
        for i in range(n_tensors):
            nlen = struct.unpack('<Q', f.read(8))[0]
            name = f.read(nlen).decode()
            ndims = struct.unpack('<I', f.read(4))[0]
            dims = struct.unpack('<' + 'Q'*ndims, f.read(8*ndims))
            dtype = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            tensors.append((name, dims, dtype, offset))

        # data section starts after alignment
        pos = f.tell()
        align = 32
        data_start = (pos + align - 1) & ~(align - 1)

        print(f"\nFile: {path}")
        print(f"Tensors: {n_tensors}")

        for name, dims, dtype, offset in tensors[:10]:  # first 10
            if dtype != 0:  # only F32 (type 0)
                print(f"  {name}: dims={dims} type={dtype} (non-F32, skipping norm)")
                continue
            f.seek(data_start + offset)
            n = 1
            for d in dims: n *= d
            data = np.frombuffer(f.read(n*4), dtype=np.float32)
            print(f"  {name}: dims={dims} norm={np.linalg.norm(data):.4f} max={np.abs(data).max():.4f} mean={np.abs(data).mean():.6f}")

if __name__ == '__main__':
    for p in sys.argv[1:]:
        try:
            read_gguf(p)
        except Exception as e:
            print(f"Error reading {p}: {e}")
