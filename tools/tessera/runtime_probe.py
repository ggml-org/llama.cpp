#!/usr/bin/env python3
"""Runtime forward-pass probe for the Tessera L2 differential.

This is the L2 forward-pass side of the runtime-aware calibration
pipeline (see ``docs/runtime-aware-pipeline.md``, Layer 2). The C++
infrastructure (Tessera's ``tessera-matmul-output`` sidecar writer in
``common/tessera-debug/`` and the CPU matmul hook in
``ggml/src/ggml-cpu/cpu-dump-matmul-output.cpp``) writes a F32
``.matmul-output.f32`` file per matmul tensor, keyed by the
linear-layer tensor name. This Python orchestrator:

  1. Loads a calibration corpus (re-uses the Tessera balanced corpus
     when available; falls back to a built-in mini-corpus).
  2. Shells out to ``llama-cli`` (or ``llama-imatrix``) twice: once
     with the BF16 source model, once with the quantized model. Each
     invocation runs with
     ``--tessera-matmul-output-dir <sidecar_dir>`` so the C++ hook
     dumps one ``.matmul-output.f32`` file per matmul tensor
     (post-audit-allowlist: attn_q/k/v/output, ffn_gate/up/down,
     output.weight only).
  3. Reads the two sidecar directories and joins them on tensor
     name. For each (tensor, layer, position) the script computes
     four per-tensor divergence metrics:

         - max_abs           = max_i |bf16[i] - quant[i]|
         - mean_abs          = mean_i |bf16[i] - quant[i]|
         - relative_frobenius = ||bf16 - quant||_F^2 / ||bf16||_F^2
         - top1_mismatch     = 1 if argmax(bf16) != argmax(quant) else 0
         - top5_mismatch     = 1 if top5(bf16) != top5(quant) else 0

     The top-K metrics use the post-matmul distribution at each
     position (one row per token). top1_mismatch and top5_mismatch
     are computed at the per-position level and aggregated as the
     mismatch rate (fraction of positions where the argmax differs).

  4. Emits a JSONL report at ``--out <path>``. Schema
     ``llama.tessera.runtime-probe.v1`` (one line per tensor per
     layer per position; provenance block at the start).

Designed to be the consumer of the C++ sidecar infrastructure;
does not touch the GGML runtime directly. Pure stdlib + numpy.

Usage::

    python3 tools/tessera/runtime_probe.py \\
        --bf16 model-f16.gguf \\
        --quantized model-tessera.gguf \\
        --prompts prompts.txt \\
        --bf16-out-dir /tmp/l2_bf16 \\
        --quantized-out-dir /tmp/l2_quant \\
        --out l2_report.jsonl \\
        --llama-cli ./build/bin/llama-cli

The script is conservative about disk usage: it runs llama-cli
sequentially (not in parallel) and cleans up the sidecar
directories after the report is written. The C++ sidecar is
audit-bounded (default stride 32, row cap 4096, allowlist); see
``common/tessera-debug/tessera-matmul-output.h`` for the safety
guarantees.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


SCHEMA_NAME = "llama.tessera.runtime-probe.v1"

# Default capture stride and row cap match the C++ sidecar defaults
# (see common/tessera-debug/tessera-matmul-output.h). The Python
# orchestrator must agree with the C++ constants so the rows/cols
# math is consistent end-to-end.
DEFAULT_STRIDE = 32
DEFAULT_MAX_ROWS = 4096

# Tensor-name allowlist: only the linear layer matmuls of the
# per-block transformer and the model head pass. The C++ CPU
# hook enforces the same allowlist at the matmul-op call site; the
# Python side re-applies it as a defensive guard when reading the
# sidecar directories.
ALLOWLIST_PATTERNS = (
    "blk.{layer}.attn_q.weight",
    "blk.{layer}.attn_k.weight",
    "blk.{layer}.attn_v.weight",
    "blk.{layer}.attn_output.weight",
    "blk.{layer}.ffn_gate.weight",
    "blk.{layer}.ffn_up.weight",
    "blk.{layer}.ffn_down.weight",
    "output.weight",
)


@dataclass
class SidecarRow:
    """One row of the .matmul-output.f32 sidecar file.

    The on-disk format (TPMO magic, v3 schema) is documented in
    ``common/tessera-debug/tessera-matmul-output.h``. The reader
    here mirrors the C++ test reader.
    """
    tensor_name: str
    rows: int
    cols: int
    total_samples: int
    row_samples: np.ndarray
    row_timing_ns: np.ndarray
    row_kernel_id: np.ndarray
    row_dispatch_count: np.ndarray
    data: np.ndarray  # shape (cumulative_rows, cols) - includes appended invocations

    @property
    def cumulative_rows(self) -> int:
        """Number of rows actually written to the data block
        (deduced from data length / cols). May exceed ``rows`` if
        the file was opened in append mode (audit fix 4) and
        multiple invocations wrote the same per-invocation row
        index."""
        if self.cols == 0:
            return 0
        return self.data.shape[0]


def read_matmul_output_sidecar(path: str) -> SidecarRow:
    """Read a v3 TPMO .matmul-output.f32 sidecar file.

    Returns a SidecarRow with the data block as a numpy ndarray of
    shape (cumulative_rows, cols). The on-disk format mirrors the
    L1 dequant sidecar (TDQT) but uses a distinct ``TPMO`` magic
    so the two file kinds can coexist in one directory.
    """
    with open(path, "rb") as f:
        # v1 header: magic(4) + version(4) + rows(8) + cols(8) + dtype(4) = 28
        magic = f.read(4)
        if magic != b"TPMO":
            raise ValueError(f"{path}: bad magic {magic!r}, expected b'TPMO'")
        version, rows, cols, dtype = struct.unpack("<IqqI", f.read(24))
        if version != 3:
            raise ValueError(f"{path}: unsupported version {version} (expected 3)")
        if dtype != 0:
            raise ValueError(f"{path}: unsupported dtype {dtype} (expected 0 = F32)")
        # v2 header: outlier_threshold(4) + total_samples(8) = 12.
        # The matmul-output sidecar repurposes the v2 outlier strip
        # as a per-row sample counter (the number of times each
        # per-invocation row was written); the threshold is unused
        # and the total is the cumulative sample count.
        _, total_samples = struct.unpack("<fq", f.read(12))
        # v2 per-row strip: R * 4 bytes (per-row sample counter).
        row_samples = np.frombuffer(f.read(4 * rows), dtype=np.int32).copy()
        # v3 per-row strip: R * 24 bytes (timing + kernel_id + dispatch + reserved).
        meta_bytes = f.read(24 * rows)
        meta = np.frombuffer(meta_bytes, dtype=[
            ("timing_ns", "<u8"),
            ("kernel_id", "<u4"),
            ("dispatch_count", "<u4"),
            ("reserved", "<u8"),
        ])
        row_timing_ns = meta["timing_ns"].copy()
        row_kernel_id = meta["kernel_id"].copy()
        row_dispatch_count = meta["dispatch_count"].copy()
        # Data block: from the current position to EOF. The header
        # declared ``rows * cols`` floats; the file may have more
        # (audit fix 4: appends from subsequent invocations) or
        # fewer (truncation in progress). Use min(rows, actual) for
        # the per-invocation row count and read all remaining
        # bytes for the cumulative data.
        data_bytes = f.read()
    if cols == 0:
        data = np.zeros((0, 0), dtype=np.float32)
    else:
        # Trim trailing bytes to a whole number of F32 values.
        n_floats = len(data_bytes) // 4
        data_bytes = data_bytes[: n_floats * 4]
        data = np.frombuffer(data_bytes, dtype=np.float32).reshape(-1, cols).copy()
    return SidecarRow(
        tensor_name=Path(path).name.removesuffix(".matmul-output.f32"),
        rows=rows,
        cols=cols,
        total_samples=int(total_samples),
        row_samples=row_samples,
        row_timing_ns=row_timing_ns,
        row_kernel_id=row_kernel_id,
        row_dispatch_count=row_dispatch_count,
        data=data,
    )


def tensor_name_passes_allowlist(name: str) -> bool:
    """Python-side mirror of the C++ tensor-name allowlist. Returns
    True iff the tensor is one of the linear layer matmuls of
    the per-block transformer or the model head."""
    if name == "output.weight":
        return True
    if not name.startswith("blk."):
        return False
    parts = name.split(".")
    if len(parts) != 4:
        return False
    try:
        int(parts[1])
    except ValueError:
        return False
    return ".".join(parts[2:]) in {
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_output.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_down.weight",
    }


def per_tensor_divergence(
    bf16: np.ndarray,
    quant: np.ndarray,
) -> dict:
    """Compute the four L2 divergence metrics for one tensor at
    one position.

    bf16 and quant are 1D arrays of length cols (the matmul
    output dim for this tensor). The function returns a dict
    with max_abs, mean_abs, relative_frobenius, top1_mismatch,
    top5_mismatch. top-K metrics use argmax / argpartition of the
    full row (i.e., the post-matmul distribution at this
    position); they are computed at the per-position level so the
    caller can aggregate them as a mismatch rate.
    """
    diff = bf16.astype(np.float64) - quant.astype(np.float64)
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    mean_abs = float(np.mean(np.abs(diff))) if diff.size else 0.0
    bf16_sq = float(np.sum(bf16.astype(np.float64) ** 2))
    diff_sq = float(np.sum(diff ** 2))
    rel_frob = diff_sq / bf16_sq if bf16_sq > 0.0 else 0.0
    top1_bf16 = int(np.argmax(bf16)) if bf16.size else -1
    top1_quant = int(np.argmax(quant)) if quant.size else -1
    top1_mismatch = 1 if top1_bf16 != top1_quant else 0
    k = min(5, bf16.size)
    if k > 0:
        top5_bf16 = set(int(i) for i in np.argpartition(-np.abs(bf16), k)[:k])
        top5_quant = set(int(i) for i in np.argpartition(-np.abs(quant), k)[:k])
        top5_mismatch = 1 if top5_bf16 != top5_quant else 0
    else:
        top5_mismatch = 0
    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "relative_frobenius": rel_frob,
        "top1_mismatch": top1_mismatch,
        "top5_mismatch": top5_mismatch,
    }


def run_llama_capture(
    llama_cli: str,
    model_path: str,
    prompts_file: str,
    sidecar_dir: str,
    n_tokens: int,
    n_ctx: int,
    extra_args: Iterable[str] = (),
) -> dict:
    """Shell out to ``llama-cli`` with the matmul-output sidecar
    enabled. Returns a dict with stdout/stderr/returncode/duration
    so the caller can record provenance.

    The C++ sidecar opens one file per matmul tensor; the
    invocation here uses --tessera-matmul-output-dir so all
    linear-layer matmul outputs land in sidecar_dir during the
    forward pass.
    """
    Path(sidecar_dir).mkdir(parents=True, exist_ok=True)
    # Use --single-turn + --file to feed the prompts and exit
    # deterministically. --n N limits the decode so the run is
    # bounded; --c N sets the prefill context.
    cmd = [
        llama_cli,
        "-m", model_path,
        "-f", prompts_file,
        "-n", str(n_tokens),
        "-c", str(n_ctx),
        "-ngl", "0",  # CPU-only for the POC
        "--single-turn",
        "--tessera-matmul-output-dir", sidecar_dir,
        *extra_args,
    ]
    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    dt = time.monotonic() - t0
    return {
        "cmd": cmd,
        "stdout_tail": proc.stdout[-500:],
        "stderr_tail": proc.stderr[-500:],
        "returncode": proc.returncode,
        "duration_s": dt,
    }


def join_sidecars(
    bf16_dir: str,
    quant_dir: str,
    stride: int = DEFAULT_STRIDE,
) -> Iterable[dict]:
    """Walk both sidecar directories and yield per-tensor
    divergence records.

    Each yielded record is a dict with the JSONL schema's
    per-tensor block: tensor, layer, position, divergence
    metrics. The function is a generator so the caller can stream
    results to the JSONL file without buffering the whole report
    in memory.
    """
    bf16_path = Path(bf16_dir)
    quant_path = Path(quant_dir)
    bf16_files = {p.name: p for p in bf16_path.glob("*.matmul-output.f32")}
    quant_files = {p.name: p for p in quant_path.glob("*.matmul-output.f32")}

    # Yield tensor names sorted by (layer, short-name) for stable
    # JSONL output. Tensor names that fail the allowlist are
    # silently skipped (the C++ sidecar's audit fix 3 already
    # filtered them, but we re-apply here as a defensive guard).
    def sort_key(name: str) -> tuple:
        if name == "output.weight":
            return (10**9, name)
        parts = name.split(".")
        try:
            layer = int(parts[1])
        except (IndexError, ValueError):
            layer = 10**9
        return (layer, name)

    common = sorted(
        (set(bf16_files) & set(quant_files)),
        key=sort_key,
    )
    for filename in common:
        # The glob returns filenames (with the .matmul-output.f32
        # suffix); strip it before the allowlist check so the
        # short tensor name (e.g. "blk.0.attn_q.weight") is what
        # the allowlist sees.
        name = filename.removesuffix(".matmul-output.f32")
        if not tensor_name_passes_allowlist(name):
            continue
        bf16 = read_matmul_output_sidecar(str(bf16_files[filename]))
        quant = read_matmul_output_sidecar(str(quant_files[filename]))
        if bf16.cols != quant.cols:
            # The two models must have the same output dim for the
            # same tensor name; if they don't, the quantization
            # shape is wrong.
            raise ValueError(
                f"shape mismatch for {name}: "
                f"bf16 cols={bf16.cols} quant cols={quant.cols}"
            )
        # Per-position metrics. The C++ sidecar stores at most
        # MATMUL_OUTPUT_MAX_ROWS rows (capped at 4096). The two
        # sides may have written a different number of rows; we
        # align by the minimum and emit one JSONL line per
        # aligned position.
        n_positions = min(bf16.cumulative_rows, quant.cumulative_rows)
        layer = -1
        if name.startswith("blk."):
            try:
                layer = int(name.split(".")[1])
            except (IndexError, ValueError):
                pass
        for pos in range(n_positions):
            div = per_tensor_divergence(
                bf16.data[pos],
                quant.data[pos],
            )
            yield {
                "tensor": name,
                "layer": layer,
                "position": pos,
                "shape": [bf16.rows, bf16.cols],
                "divergence": div,
            }


def aggregate_per_tensor(records: Iterable[dict]) -> Iterable[dict]:
    """Group per-position records by tensor and emit one
    aggregate record per tensor (per the dispatch path's
    ``ts_l2_tensor_input`` schema). The aggregate is the
    L2 weight-level style metrics: max of per-position max_abs,
    mean of per-position mean_abs, max of per-position
    relative_frobenius, and the mean top1/top5 mismatch rate
    across positions."""
    by_tensor: dict = {}
    for rec in records:
        t = rec["tensor"]
        d = rec["divergence"]
        agg = by_tensor.setdefault(t, {
            "tensor": t,
            "layer": rec["layer"],
            "shape": rec["shape"],
            "n_positions": 0,
            "max_abs": 0.0,
            "mean_abs_sum": 0.0,
            "max_rel_frob": 0.0,
            "top1_mismatch_sum": 0,
            "top5_mismatch_sum": 0,
        })
        agg["n_positions"] += 1
        if d["max_abs"] > agg["max_abs"]:
            agg["max_abs"] = d["max_abs"]
        agg["mean_abs_sum"] += d["mean_abs"]
        if d["relative_frobenius"] > agg["max_rel_frob"]:
            agg["max_rel_frob"] = d["relative_frobenius"]
        agg["top1_mismatch_sum"] += d["top1_mismatch"]
        agg["top5_mismatch_sum"] += d["top5_mismatch"]
    for t, agg in by_tensor.items():
        n = max(agg["n_positions"], 1)
        yield {
            "tensor": agg["tensor"],
            "layer": agg["layer"],
            "shape": agg["shape"],
            "n_positions": agg["n_positions"],
            "divergence": {
                "max_abs": agg["max_abs"],
                "mean_abs": agg["mean_abs_sum"] / n,
                "max_relative_frobenius": agg["max_rel_frob"],
                "top1_mismatch_rate": agg["top1_mismatch_sum"] / n,
                "top5_mismatch_rate": agg["top5_mismatch_sum"] / n,
            },
        }


def write_jsonl_report(
    out_path: str,
    bf16_invocation: dict,
    quant_invocation: dict,
    bf16_model: str,
    quant_model: str,
    prompts_file: str,
    n_tokens: int,
    n_ctx: int,
    stride: int,
    per_position_records: Iterable[dict],
    per_tensor_aggregates: Iterable[dict],
) -> None:
    """Write the JSONL report. The first line is a provenance
    block; the rest are one line per (tensor, position) plus one
    block of per-tensor aggregates at the end (each on its own
    line, prefixed with ``"record_type": "tensor_aggregate"``)."""
    with open(out_path, "w") as f:
        prov = {
            "schema": SCHEMA_NAME,
            "record_type": "provenance",
            "bf16_model": bf16_model,
            "quantized_model": quant_model,
            "prompts_file": prompts_file,
            "n_tokens": n_tokens,
            "n_ctx": n_ctx,
            "capture_stride": stride,
            "bf16_invocation": bf16_invocation,
            "quantized_invocation": quant_invocation,
            "created_at_unix": time.time(),
        }
        f.write(json.dumps(prov) + "\n")
        for rec in per_position_records:
            f.write(json.dumps({"record_type": "per_position", **rec}) + "\n")
        for agg in per_tensor_aggregates:
            f.write(json.dumps({"record_type": "tensor_aggregate", **agg}) + "\n")


def build_mini_corpus(out_path: str, n_tokens: int = 512) -> None:
    """Write a minimal calibration corpus to ``out_path`` when
    the caller does not provide one. The text is a deterministic
    English paragraph that tokenizes to roughly ``n_tokens``
    BPE tokens in the qwen2 family. This is the fallback when
    the Tessera balanced corpus is not available."""
    sentence = (
        "The quick brown fox jumps over the lazy dog. "
        "In a hole in the ground there lived a hobbit. "
        "It was the best of times, it was the worst of times. "
        "All happy families are alike; each unhappy family is "
        "unhappy in its own way. "
    )
    # Roughly 100 tokens per repeat at qwen2 BPE.
    repeat = max(1, n_tokens // 100 + 1)
    with open(out_path, "w") as f:
        f.write(sentence * repeat)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="L2 forward-pass differential: BF16 vs quantized matmul outputs."
    )
    p.add_argument("--bf16", required=True, help="Path to the BF16 source model GGUF.")
    p.add_argument("--quantized", required=True, help="Path to the quantized model GGUF.")
    p.add_argument("--prompts", default="",
                   help="Calibration prompts file. If empty, a built-in mini corpus is written to a temp file.")
    p.add_argument("--bf16-out-dir", required=True,
                   help="Where llama-cli writes the BF16 matmul-output sidecar files.")
    p.add_argument("--quantized-out-dir", required=True,
                   help="Where llama-cli writes the quantized matmul-output sidecar files.")
    p.add_argument("--out", required=True, help="Output JSONL report path.")
    p.add_argument("--llama-cli", default="./build/bin/llama-cli",
                   help="Path to the llama-cli binary. Must be built with --tessera-matmul-output-dir support.")
    p.add_argument("--n-tokens", type=int, default=4,
                   help="Number of decode tokens (default: 4; the prefill is the bulk of the L2 signal).")
    p.add_argument("--n-ctx", type=int, default=512,
                   help="Context length (default: 512).")
    p.add_argument("--stride", type=int, default=DEFAULT_STRIDE,
                   help=f"Capture stride (default: {DEFAULT_STRIDE}; must match the C++ default).")
    p.add_argument("--keep-sidecars", action="store_true",
                   help="Do not delete the per-model sidecar directories after the report is written.")
    p.add_argument("--per-position", action="store_true", default=True,
                   help="Emit one JSONL line per (tensor, position) (default: true).")
    p.add_argument("--per-tensor", action="store_true", default=True,
                   help="Emit per-tensor aggregate records at the end of the JSONL (default: true).")
    args = p.parse_args(argv)

    # Prepare the calibration corpus. The spec re-uses the
    # imatrix prompts when available; we accept an explicit
    # --prompts file and otherwise generate a mini corpus.
    if args.prompts:
        prompts_file = args.prompts
    else:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False,
            prefix="tessera_runtime_probe_",
        )
        build_mini_corpus(tmp.name, n_tokens=args.n_ctx)
        tmp.close()
        prompts_file = tmp.name
        print(f"runtime_probe: wrote mini calibration corpus to {prompts_file}",
              file=sys.stderr)

    # Ensure the sidecar output directories exist (empty). The C++
    # sidecar appends when the file already exists (audit fix 4);
    # clearing the directory at the start makes the run
    # deterministic.
    for d in (args.bf16_out_dir, args.quantized_out_dir):
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    # 1. Run the BF16 source model with the sidecar enabled.
    print(f"runtime_probe: BF16 forward pass -> {args.bf16_out_dir}", file=sys.stderr)
    bf16_inv = run_llama_capture(
        llama_cli=args.llama_cli,
        model_path=args.bf16,
        prompts_file=prompts_file,
        sidecar_dir=args.bf16_out_dir,
        n_tokens=args.n_tokens,
        n_ctx=args.n_ctx,
    )
    if bf16_inv["returncode"] != 0:
        print(f"runtime_probe: BF16 invocation failed (rc={bf16_inv['returncode']}):",
              file=sys.stderr)
        print(bf16_inv["stderr_tail"], file=sys.stderr)
        return 1

    # 2. Run the quantized model with the sidecar enabled.
    print(f"runtime_probe: quantized forward pass -> {args.quantized_out_dir}", file=sys.stderr)
    quant_inv = run_llama_capture(
        llama_cli=args.llama_cli,
        model_path=args.quantized,
        prompts_file=prompts_file,
        sidecar_dir=args.quantized_out_dir,
        n_tokens=args.n_tokens,
        n_ctx=args.n_ctx,
    )
    if quant_inv["returncode"] != 0:
        print(f"runtime_probe: quantized invocation failed (rc={quant_inv['returncode']}):",
              file=sys.stderr)
        print(quant_inv["stderr_tail"], file=sys.stderr)
        return 1

    # 3. Join the sidecar directories and stream per-position
    # divergence records. We accumulate the per-position records
    # into a list for the per-tensor aggregate pass; the list is
    # bounded by O(n_tensors * n_positions) which is at most
    # ~100 tensors * 4096 rows = 400K records = ~50 MB in JSONL
    # (acceptable for the POC).
    per_position = list(join_sidecars(
        args.bf16_out_dir, args.quantized_out_dir, stride=args.stride,
    ))
    per_tensor = list(aggregate_per_tensor(per_position)) if args.per_tensor else []

    # 4. Write the JSONL report.
    write_jsonl_report(
        out_path=args.out,
        bf16_invocation=bf16_inv,
        quant_invocation=quant_inv,
        bf16_model=args.bf16,
        quant_model=args.quantized,
        prompts_file=prompts_file,
        n_tokens=args.n_tokens,
        n_ctx=args.n_ctx,
        stride=args.stride,
        per_position_records=per_position if args.per_position else [],
        per_tensor_aggregates=per_tensor,
    )

    n_tensors = len({r["tensor"] for r in per_position})
    print(
        f"runtime_probe: wrote {len(per_position)} per-position records "
        f"across {n_tensors} tensors, {len(per_tensor)} per-tensor aggregates "
        f"to {args.out}",
        file=sys.stderr,
    )

    if not args.keep_sidecars:
        for d in (args.bf16_out_dir, args.quantized_out_dir):
            shutil.rmtree(d, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
