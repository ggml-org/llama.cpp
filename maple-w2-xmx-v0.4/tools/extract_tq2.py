#!/usr/bin/env python3
"""List/extract standard TQ2_0 GGUF tensors without torch/numpy/gguf dependencies.
Does not alter the GGUF, dequantize weights, or extract the whole model.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import BinaryIO, Any

SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
MAGIC = b'MW2TQ2\0\0'

class GGUF:
    def __init__(self, file: BinaryIO):
        self.f = file
        self.size = file.seek(0, 2)
        file.seek(0)
        if self.read(4) != b'GGUF':
            raise ValueError('not a GGUF file')
        version = self.number('<I')
        if version not in (2, 3):
            raise ValueError(f'unsupported GGUF version {version}')
        self.version = version
        nt, nm = self.number('<Q'), self.number('<Q')
        if nt > 1000000 or nm > 1000000:
            raise ValueError('unreasonable GGUF header counts')
        alignment = 32
        for _ in range(nm):
            key, ty = self.string(), self.number('<I')
            if key == 'general.alignment':
                if ty != 4:
                    raise ValueError('general.alignment must be UINT32')
                alignment = self.number('<I')
            else:
                self.skip_value(ty)
        if alignment < 1 or alignment > (1 << 20) or alignment & (alignment - 1):
            raise ValueError('invalid GGUF alignment')
        self.tensors: list[dict[str, Any]] = []
        names: set[str] = set()
        for _ in range(nt):
            name, rank = self.string(), self.number('<I')
            if rank < 1 or rank > 4 or name in names:
                raise ValueError('invalid tensor rank or duplicate name')
            names.add(name)
            dims = [self.number('<Q') for _ in range(rank)]
            if any(d == 0 for d in dims):
                raise ValueError('zero-sized tensor')
            ty, off = self.number('<I'), self.number('<Q')
            self.tensors.append({'name': name, 'shape': dims, 'type': ty, 'relative_offset': off})
        self.data_offset = (file.tell() + alignment - 1) // alignment * alignment
        if self.data_offset > self.size:
            raise ValueError('truncated GGUF tensor directory')

    def read(self, n: int) -> bytes:
        if n < 0 or self.f.tell() + n > self.size:
            raise ValueError('truncated or malformed GGUF')
        data = self.f.read(n)
        if len(data) != n:
            raise ValueError('short read')
        return data

    def number(self, fmt: str) -> int:
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        n = self.number('<Q')
        if n > (1 << 24):
            raise ValueError('unreasonable header string')
        return self.read(n).decode('utf-8')

    def skip(self, n: int) -> None:
        if n < 0 or self.f.tell() + n > self.size:
            raise ValueError('invalid metadata length')
        self.f.seek(n, 1)

    def skip_value(self, ty: int, depth: int = 0) -> None:
        if depth > 8:
            raise ValueError('metadata nesting too deep')
        if ty in SIZES:
            self.skip(SIZES[ty])
        elif ty == 8:
            self.skip(self.number('<Q'))
        elif ty == 9:
            subtype, count = self.number('<I'), self.number('<Q')
            if subtype in SIZES:
                self.skip(SIZES[subtype] * count)
            else:
                if count > self.size:
                    raise ValueError('unreasonable metadata array count')
                for _ in range(count):
                    self.skip_value(subtype, depth + 1)
        else:
            raise ValueError(f'unknown GGUF metadata type {ty}')

    def extract(self, name: str, output: Path) -> dict[str, Any]:
        matches = [t for t in self.tensors if t['name'] == name]
        if len(matches) != 1:
            raise ValueError(f'tensor not found: {name}; run --list first')
        t = matches[0]
        if t['type'] != 35:
            raise ValueError(f"{name}: type={t['type']}, expected STANDARD GGML_TYPE_TQ2_0=35; custom Q2 is NOT supported")
        dims = t['shape'] + [1] * (4 - len(t['shape']))
        k, m, experts, outer = dims
        if outer != 1 or k % 256 or m % 8 or max(k, m, experts) > 0xffffffff:
            raise ValueError('pilot needs K%256=0, M%8=0, ne3=1 and uint32 dimensions')
        size = math.prod(dims) // 256 * 66
        start = self.data_offset + t['relative_offset']
        if start + size > self.size:
            raise ValueError('tensor extends beyond end of file')
        if output.exists():
            raise FileExistsError(f'refusing to overwrite {output}')
        output.parent.mkdir(parents=True, exist_ok=True)
        sha = hashlib.sha256()
        created = False
        try:
            with output.open('xb') as out:
                created = True
                out.write(struct.pack('<8sIIIIQ', MAGIC, 1, k, m, experts, size))
                self.f.seek(start)
                remain = size
                while remain:
                    part = self.read(min(remain, 1 << 20))
                    out.write(part)
                    sha.update(part)
                    remain -= len(part)
        except Exception:
            if created:
                output.unlink(missing_ok=True)
            raise
        return dict(t, gguf_version=self.version, payload_bytes=size,
                    absolute_offset=start, payload_sha256=sha.hexdigest(),
                    capsule=str(output.resolve()))

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--list', action='store_true')
    group.add_argument('--tensor')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    with a.model.open('rb') as f:
        gguf = GGUF(f)
        if a.list:
            for t in gguf.tensors:
                if t['type'] == 35:
                    print(f"{t['name']}\ttype=35\tshape={t['shape']}")
        else:
            if not a.output:
                p.error('--tensor requires --output')
            meta_path = a.output.with_suffix(a.output.suffix + '.json')
            if meta_path.exists():
                raise FileExistsError(f'refusing to overwrite {meta_path}')
            meta = gguf.extract(a.tensor, a.output)
            meta['source_gguf'] = str(a.model.resolve())
            with meta_path.open('x', encoding='utf-8') as out:
                out.write(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
            print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        raise SystemExit(f'ERROR: {exc}')
