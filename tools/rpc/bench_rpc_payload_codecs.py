#!/usr/bin/env python3
"""Benchmark codecs against RPC compression-probe sample dumps."""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


@dataclass(frozen=True)
class Codec:
    name: str
    compress: Callable[[bytes], bytes]
    decompress: Callable[[bytes], bytes]


def timed_call(fn: Callable[[bytes], bytes], data: bytes, repeat: int) -> tuple[bytes, float]:
    results: list[bytes] = []
    times_ms: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter_ns()
        result = fn(data)
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
        results.append(result)
        times_ms.append(elapsed_ms)
    return results[-1], statistics.median(times_ms)


def run_cmd(args: list[str], data: bytes) -> bytes:
    proc = subprocess.run(args, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return proc.stdout


def discover_codecs(names: Iterable[str]) -> list[Codec]:
    requested = set(names)
    codecs: list[Codec] = []

    def want(name: str) -> bool:
        return not requested or name in requested

    if want("zlib-1"):
        codecs.append(Codec(
            "zlib-1",
            lambda data: zlib.compress(data, level=1),
            zlib.decompress,
        ))
    if want("zlib-6"):
        codecs.append(Codec(
            "zlib-6",
            lambda data: zlib.compress(data, level=6),
            zlib.decompress,
        ))

    zstd = shutil.which("zstd")
    if zstd is not None:
        if want("zstd-1"):
            codecs.append(Codec(
                "zstd-1",
                lambda data, zstd=zstd: run_cmd([zstd, "-q", "-1", "-c"], data),
                lambda data, zstd=zstd: run_cmd([zstd, "-q", "-d", "-c"], data),
            ))
        if want("zstd-fast"):
            codecs.append(Codec(
                "zstd-fast",
                lambda data, zstd=zstd: run_cmd([zstd, "-q", "--fast=1", "-c"], data),
                lambda data, zstd=zstd: run_cmd([zstd, "-q", "-d", "-c"], data),
            ))

    lz4 = shutil.which("lz4")
    if lz4 is not None and want("lz4-1"):
        codecs.append(Codec(
            "lz4-1",
            lambda data, lz4=lz4: run_cmd([lz4, "-q", "-1", "-c"], data),
            lambda data, lz4=lz4: run_cmd([lz4, "-q", "-d", "-c"], data),
        ))

    missing = requested - {codec.name for codec in codecs}
    for name in sorted(missing):
        print(f"warning: codec '{name}' is unavailable", file=sys.stderr)
    return codecs


def load_metadata(dump_dir: Path) -> list[dict]:
    metadata_path = dump_dir / "samples.jsonl"
    if metadata_path.exists():
        rows = []
        with metadata_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    rows = []
    for sample in sorted(dump_dir.glob("*.bin")):
        rows.append({
            "file": sample.name,
            "operation": "",
            "endpoint": "",
            "tensor": sample.stem,
            "bytes": sample.stat().st_size,
            "sampled_bytes": sample.stat().st_size,
            "windows": "",
        })
    return rows


def bench_sample(row: dict, dump_dir: Path, codecs: list[Codec], repeat: int) -> list[dict]:
    sample_path = dump_dir / row["file"]
    data = sample_path.read_bytes()
    results = []
    for codec in codecs:
        try:
            compressed, compress_ms = timed_call(codec.compress, data, repeat)
            decompressed, decompress_ms = timed_call(codec.decompress, compressed, repeat)
        except (subprocess.CalledProcessError, OSError, zlib.error) as exc:
            results.append({
                "kind": "sample",
                "codec": codec.name,
                "file": row["file"],
                "error": str(exc),
            })
            continue

        ok = decompressed == data
        sampled = len(data)
        results.append({
            "kind": "sample",
            "codec": codec.name,
            "file": row["file"],
            "operation": row.get("operation", ""),
            "endpoint": row.get("endpoint", ""),
            "tensor": row.get("tensor", ""),
            "payload_bytes": row.get("bytes", sampled),
            "sampled_bytes": sampled,
            "compressed_bytes": len(compressed),
            "ratio": len(compressed) / sampled if sampled else 0.0,
            "wire_bytes_if_skip_expansion": min(len(compressed), sampled),
            "would_send_compressed": len(compressed) < sampled,
            "compress_ms": compress_ms,
            "decompress_ms": decompress_ms,
            "roundtrip_ok": ok,
        })
    return results


def summarize_group(codec: str, group: str, rows: list[dict]) -> dict | None:
    if not rows:
        return None

    sampled_bytes = sum(row["sampled_bytes"] for row in rows)
    compressed_bytes = sum(row["compressed_bytes"] for row in rows)
    wire_bytes_if_skip = sum(row["wire_bytes_if_skip_expansion"] for row in rows)
    return {
        "kind": "summary",
        "group": group,
        "codec": codec,
        "samples": len(rows),
        "payload_bytes": sum(row["payload_bytes"] for row in rows),
        "sampled_bytes": sampled_bytes,
        "compressed_bytes": compressed_bytes,
        "weighted_ratio": compressed_bytes / sampled_bytes,
        "weighted_ratio_if_skip_expansion": wire_bytes_if_skip / sampled_bytes,
        "saved_bytes_if_skip_expansion": sampled_bytes - wire_bytes_if_skip,
        "compressible_samples": sum(1 for row in rows if row["would_send_compressed"]),
        "median_ratio": statistics.median(row["ratio"] for row in rows),
        "median_compress_ms": statistics.median(row["compress_ms"] for row in rows),
        "median_decompress_ms": statistics.median(row["decompress_ms"] for row in rows),
        "roundtrip_ok": all(row["roundtrip_ok"] for row in rows),
    }


def summarize(rows: list[dict]) -> list[dict]:
    summaries = []
    codecs = sorted({row["codec"] for row in rows if row.get("kind") == "sample" and "ratio" in row})
    for codec in codecs:
        codec_rows = [row for row in rows if row.get("codec") == codec and "ratio" in row]
        groups = {
            "all": codec_rows,
            "payload_ge_1mib": [row for row in codec_rows if row["payload_bytes"] >= 1024 * 1024],
            "payload_lt_64kib": [row for row in codec_rows if row["payload_bytes"] < 64 * 1024],
        }
        for group, group_rows in groups.items():
            summary = summarize_group(codec, group, group_rows)
            if summary is not None:
                summaries.append(summary)
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump_dir", type=Path, help="directory created by GGML_RPC_COMPRESSION_PROBE_DUMP_DIR")
    parser.add_argument("--repeat", type=int, default=3, help="per-codec timing repetitions (default: 3)")
    parser.add_argument(
        "--codec",
        action="append",
        default=[],
        help="codec to run; may be repeated; defaults to all available",
    )
    args = parser.parse_args()

    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    if not args.dump_dir.is_dir():
        parser.error(f"{args.dump_dir} is not a directory")

    metadata = load_metadata(args.dump_dir)
    codecs = discover_codecs(args.codec)
    if not metadata:
        parser.error("no sample metadata or .bin files found")
    if not codecs:
        parser.error("no codecs available")

    rows: list[dict] = []
    for row in metadata:
        rows.extend(bench_sample(row, args.dump_dir, codecs, args.repeat))

    rows.extend(summarize(rows))
    for row in rows:
        print(json.dumps(row, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
