#!/usr/bin/env python3
"""Embed compiled ANE MTP bucket bundles in a minimal GGUF test fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from gguf import GGUFWriter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundles", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    writer = GGUFWriter(args.output, "ane-mtp-test")
    retained: list[np.ndarray] = []
    buckets: list[int] = []
    for batch in (1, 2, 4, 8):
        bundle = args.bundles / f"batch-{batch}.mlmodelc"
        if not bundle.is_dir():
            continue
        digest = hashlib.sha256()
        files = sorted(path for path in bundle.rglob("*") if path.is_file())
        prefix = f"mtp.ane.bucket.{batch}"
        for index, path in enumerate(files):
            relative = path.relative_to(bundle).as_posix()
            data = np.memmap(path, mode="r", dtype=np.uint8)
            retained.append(data)
            digest.update(relative.encode())
            digest.update(b"\0")
            digest.update(memoryview(data))
            name = f"{prefix}.file.{index:04d}"
            writer.add_tensor(name, data.view(np.int8))
            writer.add_string(f"{name}.path", relative)
        writer.add_uint32(f"{prefix}.file_count", len(files))
        writer.add_string(f"{prefix}.bundle_sha256", digest.hexdigest())
        manifest = args.bundles / f"batch-{batch}.json"
        if manifest.is_file():
            data = json.loads(manifest.read_text())
            functions = data.get("functions", [])
            if functions:
                writer.add_array(f"{prefix}.functions", functions)
            if "context" in data:
                writer.add_uint32(
                    f"{prefix}.context_length", int(data["context"])
                )
            if "sync_chunk" in data:
                writer.add_uint32(
                    f"{prefix}.sync_chunk", int(data["sync_chunk"])
                )
        buckets.append(batch)

    if not buckets:
        raise SystemExit(f"no batch-N.mlmodelc bundles found under {args.bundles}")
    writer.add_string("mtp.ane.format", "mlmodelc-buckets-v2")
    writer.add_array("mtp.ane.batch_buckets", buckets)
    writer.add_bool("mtp.ane.keep_warm", True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


if __name__ == "__main__":
    main()
