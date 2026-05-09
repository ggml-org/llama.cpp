#!/usr/bin/env python3
import argparse, hashlib, mmap, os, struct, sys
from pathlib import Path

GGUF_MAGIC   = 0x46554747          # "GGUF"
GGUF_VERSION = 3

FFN_PATTERNS = [
    "ffn_norm", "ffn_gate", "ffn_up", "ffn_down",
    "ffn_gate_inp", "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps",
]

def is_ffn_tensor(name: str) -> bool: return any(p in name for p in FFN_PATTERNS)

def get_ggml_type_size(type_id: int) -> tuple[int, int]:
    sizes = { 0:(4,1), 1:(2,1), 2:(18,32), 3:(20,32), 6:(22,32), 7:(24,32), 8:(34,32), 9:(36,32),
              10:(2,2), 11:(3,2), 12:(144,256), 13:(176,256), 14:(144,256), 15:(176,256), 16:(210,256), 17:(256,256),
              18:(2,1), 19:(2,1), 20:(2,1), 21:(2,1), 22:(2,1), 23:(2,1), 24:(2,1), 25:(2,1), 26:(2,1), 27:(2,1), 28:(2,1), 29:(2,1), 30:(8,1), 31:(2,1) }
    return sizes.get(type_id, (4, 1))

def calc_tensor_bytes(dims, dtype):
    n_elements = 1
    for d in dims: n_elements *= d
    type_size, block_size = get_ggml_type_size(dtype)
    return (n_elements * type_size) // block_size

def read_string(f) -> str:
    length = struct.unpack('<Q', f.read(8))[0]
    return f.read(length).decode('utf-8')

def read_raw_kv(f) -> tuple[str, bytes]:
    start = f.tell()
    key = read_string(f)
    vtype = struct.unpack('<I', f.read(4))[0]
    def skip(vt):
        sizes = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
        if vt in sizes: f.read(sizes[vt])
        elif vt == 8: f.read(struct.unpack('<Q', f.read(8))[0])
        elif vt == 9:
            et = struct.unpack('<I', f.read(4))[0]
            for _ in range(struct.unpack('<Q', f.read(8))[0]): skip(et)
    skip(vtype)
    end = f.tell()
    f.seek(start)
    return key, f.read(end - start)

def write_string(f, s: str):
    b = s.encode('utf-8')
    f.write(struct.pack('<Q', len(b)))
    f.write(b)

def write_kv_string(f, key: str, value: str):
    write_string(f, key)
    f.write(struct.pack('<I', 8))
    write_string(f, value)

def write_kv_u32(f, key: str, value: int):
    write_string(f, key)
    f.write(struct.pack('<I', 4))
    f.write(struct.pack('<I', value))

def split_gguf(src: Path, attn_dst: Path, ffn_dst: Path, dry_run=False, verify=False, layers=None):
    src_sha256 = hashlib.sha256(src.read_bytes()).hexdigest()
    with open(src, 'rb') as f:
        magic, version = struct.unpack('<II', f.read(8))
        tensor_count, kv_count = struct.unpack('<QQ', f.read(16))
        kvs, n_embd, n_layers = [], 4096, 32
        for _ in range(kv_count):
            key, raw = read_raw_kv(f)
            kvs.append((key, raw))
            if key.endswith('.embedding_length'): n_embd = struct.unpack('<I', raw[-4:])[0]
            if key.endswith('.block_count'): n_layers = struct.unpack('<I', raw[-4:])[0]

        tensor_infos = []
        for _ in range(tensor_count):
            name = read_string(f)
            n_dims = struct.unpack('<I', f.read(4))[0]
            dims = struct.unpack(f'<{n_dims}Q', f.read(8 * n_dims))
            dtype, offset = struct.unpack('<IQ', f.read(12))
            tensor_infos.append({'name': name, 'dims': dims, 'dtype': dtype, 'offset': offset})

        data_start = f.tell()
        remainder = data_start % 32
        if remainder: data_start += 32 - remainder

        attn_tensors = [t for t in tensor_infos if not is_ffn_tensor(t['name'])]
        ffn_tensors  = [t for t in tensor_infos if     is_ffn_tensor(t['name'])]

        if dry_run: return

        def write_slice(dst: Path, tensors: list, split_type: str):
            with open(dst, 'wb') as out:
                out.write(struct.pack('<II', GGUF_MAGIC, GGUF_VERSION))
                out.write(struct.pack('<QQ', len(tensors), len(kvs) + 7))
                for _, raw in kvs: out.write(raw)
                write_kv_string(out, 'split.type', split_type)
                write_kv_string(out, 'split.source_sha256', src_sha256)
                write_kv_u32   (out, 'split.n_embd', n_embd)
                write_kv_u32   (out, 'split.layer_first', 0)
                write_kv_u32   (out, 'split.layer_last', n_layers - 1)
                write_kv_u32   (out, 'split.wire_version', 1)
                write_kv_string(out, 'split.ffn_norm_placement', 'ffn')

                running_offset = 0
                new_infos = []
                for t in tensors:
                    nbytes = calc_tensor_bytes(t['dims'], t['dtype'])
                    nbytes_aligned = (nbytes + 31) // 32 * 32
                    new_infos.append((t, running_offset, nbytes))
                    running_offset += nbytes_aligned

                for t, off, _ in new_infos:
                    write_string(out, t['name'])
                    out.write(struct.pack('<I', len(t['dims'])))
                    out.write(struct.pack(f"<{len(t['dims'])}Q", *t['dims']))
                    out.write(struct.pack('<IQ', t['dtype'], off))

                pad = (32 - out.tell() % 32) % 32
                out.write(b'\x00' * pad)

                with open(src, 'rb') as src_f:
                    mm = mmap.mmap(src_f.fileno(), 0, access=mmap.ACCESS_READ)
                    for t, _, nbytes in new_infos:
                        chunk = mm[data_start + t['offset']: data_start + t['offset'] + nbytes]
                        out.write(chunk)
                        if (32 - len(chunk) % 32) % 32: out.write(b'\x00' * ((32 - len(chunk) % 32) % 32))
                    mm.close()

        write_slice(attn_dst, attn_tensors, 'attention')
        write_slice(ffn_dst,  ffn_tensors,  'ffn')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input',            required=True, type=Path)
    ap.add_argument('--output-attention', required=True, type=Path)
    ap.add_argument('--output-ffn',       required=True, type=Path)
    ap.add_argument('--dry-run',   action='store_true')
    ap.add_argument('--verify',    action='store_true')
    ap.add_argument('--layers',    default=None)
    args = ap.parse_args()
    split_gguf(args.input, args.output_attention, args.output_ffn, args.dry_run, args.verify, args.layers)
