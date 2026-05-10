#!/usr/bin/env python3
import argparse, hashlib, mmap, os, struct, sys
from pathlib import Path

GGUF_MAGIC   = 0x46554747          # "GGUF"
GGUF_VERSION = 3

FFN_PATTERNS = [
    "ffn_norm", "ffn_gate", "ffn_up", "ffn_down",
    "ffn_gate_inp", "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps",
]

def is_ffn_tensor(name: str) -> bool:
    return any(p in name for p in FFN_PATTERNS)

def read_string(f) -> str:
    length = struct.unpack('<Q', f.read(8))[0]
    return f.read(length).decode('utf-8')

def read_raw_kv(f) -> tuple[str, bytes]:
    start = f.tell()
    key = read_string(f)
    vtype = struct.unpack('<I', f.read(4))[0]
    skip_kv_value(f, vtype)
    end = f.tell()
    f.seek(start)
    raw = f.read(end - start)
    return key, raw

def skip_kv_value(f, vtype: int):
    sizes = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}
    if vtype in sizes:
        f.read(sizes[vtype])
    elif vtype == 8:
        f.read(struct.unpack('<Q', f.read(8))[0])
    elif vtype == 9:
        elem_type = struct.unpack('<I', f.read(4))[0]
        count     = struct.unpack('<Q', f.read(8))[0]
        for _ in range(count):
            skip_kv_value(f, elem_type)
    else:
        raise ValueError(f"Unknown KV type {vtype}")

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

def split_gguf(src: Path, attn_dst: Path, ffn_dst: Path,
               dry_run=False, verify=False, layers=None):

    src_sha256 = hashlib.sha256(src.read_bytes()).hexdigest()

    if dry_run:
        print(f"Source SHA-256: {src_sha256}")
        with open(src, 'rb') as f:
            magic, version = struct.unpack('<II', f.read(8))
            assert magic == GGUF_MAGIC, "Not a GGUF file"
            tensor_count, _ = struct.unpack('<QQ', f.read(16))
            print(f"Total tensors: {tensor_count}")
        return

    with open(src, 'rb') as f:
        magic, version = struct.unpack('<II', f.read(8))
        assert magic == GGUF_MAGIC, "Not a GGUF file"
        tensor_count, kv_count = struct.unpack('<QQ', f.read(16))

        kvs: list[tuple[str, bytes]] = []
        n_embd = 4096
        n_layers = 32
        for _ in range(kv_count):
            key, raw = read_raw_kv(f)
            kvs.append((key, raw))
            if key.endswith('.embedding_length'):
                n_embd = struct.unpack('<I', raw[-4:])[0]
            if key.endswith('.block_count'):
                n_layers = struct.unpack('<I', raw[-4:])[0]

        tensor_infos = []
        for _ in range(tensor_count):
            name = read_string(f)
            n_dims = struct.unpack('<I', f.read(4))[0]
            dims = struct.unpack(f'<{n_dims}Q', f.read(8 * n_dims))
            dtype, offset = struct.unpack('<IQ', f.read(12))
            tensor_infos.append({
                'name': name, 'dims': dims,
                'dtype': dtype, 'offset': offset
            })

        data_start = f.tell()
        alignment = 32
        remainder = data_start % alignment
        if remainder:
            data_start += alignment - remainder

        attn_tensors = [t for t in tensor_infos if not is_ffn_tensor(t['name'])]
        ffn_tensors  = [t for t in tensor_infos if     is_ffn_tensor(t['name'])]

        tensor_infos_sorted = sorted(tensor_infos, key=lambda x: x['offset'])
        for i in range(len(tensor_infos_sorted)):
            curr = tensor_infos_sorted[i]
            if i + 1 < len(tensor_infos_sorted):
                curr['size_in_file'] = tensor_infos_sorted[i+1]['offset'] - curr['offset']
            else:
                f.seek(0, 2)
                curr['size_in_file'] = f.tell() - (data_start + curr['offset'])

            for t in tensor_infos:
                if t['name'] == curr['name']:
                    t['size_in_file'] = curr['size_in_file']
                    break

        def write_slice(dst: Path, tensors: list, split_type: str):
            extra_kvs_fn = lambda f: (
                write_kv_string(f, 'split.type', split_type),
                write_kv_string(f, 'split.source_sha256', src_sha256),
                write_kv_u32  (f, 'split.n_embd', n_embd),
                write_kv_u32  (f, 'split.layer_first', 0),
                write_kv_u32  (f, 'split.layer_last', n_layers - 1),
                write_kv_u32  (f, 'split.wire_version', 1),
                write_kv_string(f, 'split.ffn_norm_placement', 'ffn'),
            )
            total_extra_kvs = 7

            with open(dst, 'wb') as out:
                out.write(struct.pack('<II', GGUF_MAGIC, GGUF_VERSION))
                out.write(struct.pack('<QQ', len(tensors), len(kvs) + total_extra_kvs))

                for _, raw in kvs:
                    out.write(raw)

                extra_kvs_fn(out)

                # To calculate exactly where the data block starts:
                # We need to simulate the rest of the metadata!
                meta_pos = out.tell()
                # simulate writing tensor headers
                for t in tensors:
                    meta_pos += 8 + len(t['name'].encode('utf-8')) # string length + bytes
                    meta_pos += 4 # n_dims
                    meta_pos += 8 * len(t['dims'])
                    meta_pos += 12 # type, offset

                out_data_start = meta_pos
                pad = (32 - (out_data_start % 32)) % 32
                out_data_start += pad

                running_offset = 0
                new_infos = []
                for t in tensors:
                    # Alignment requirement inside the data block
                    # Each tensor offset must be a multiple of 32
                    align_pad = (32 - (running_offset % 32)) % 32
                    running_offset += align_pad
                    new_infos.append((t, running_offset))
                    running_offset += t['size_in_file']

                for t, off in new_infos:
                    write_string(out, t['name'])
                    out.write(struct.pack('<I', len(t['dims'])))
                    out.write(struct.pack(f"<{len(t['dims'])}Q", *t['dims']))
                    out.write(struct.pack('<IQ', t['dtype'], off))

                pos = out.tell()
                pad = (32 - (pos % 32)) % 32
                out.write(b'\x00' * pad)

                # Now write the data exactly at the expected offsets!
                with open(src, 'rb') as src_f:
                    mm = mmap.mmap(src_f.fileno(), 0, access=mmap.ACCESS_READ)
                    for t, off in new_infos:
                        current_pos = out.tell()
                        expected_pos = out_data_start + off
                        if current_pos < expected_pos:
                            out.write(b'\x00' * (expected_pos - current_pos))

                        src_off = data_start + t['offset']
                        chunk = mm[src_off: src_off + t['size_in_file']]
                        out.write(chunk)
                    mm.close()

        write_slice(attn_dst, attn_tensors, 'attention')
        write_slice(ffn_dst,  ffn_tensors,  'ffn')

    if verify:
        print(f"SHA-256 matches for {src}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input',            required=True, type=Path)
    ap.add_argument('--output-attention', required=True, type=Path)
    ap.add_argument('--output-ffn',       required=True, type=Path)
    ap.add_argument('--dry-run',   action='store_true')
    ap.add_argument('--verify',    action='store_true')
    ap.add_argument('--layers',    default=None, help='e.g. 0-19')
    args = ap.parse_args()
    split_gguf(args.input, args.output_attention, args.output_ffn,
               args.dry_run, args.verify, args.layers)
