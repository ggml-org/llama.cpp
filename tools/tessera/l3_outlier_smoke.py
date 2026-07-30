#!/usr/bin/env python3
"""Smoke test for the Tessera L3 outlier report.

Generates a small synthetic ``.dequant.f32`` v2 sidecar with a known
weight distribution, runs ``l3_outlier_report.py`` on it, and checks:

* The reader accepts the synthetic v2 file and recovers the expected
  per-row outlier count.
* The aggregate fraction is within a tight tolerance of the analytical
  value (we know exactly how many |x| > t events the synthetic input
  contains).
* A second synthetic tensor with a pathological outlier rate trips the
  5% ceiling and the report exits non-zero -- this is the gating
  behaviour the L3 metric and the L5 orchestrator rely on.

Run::

    python3 tools/tessera/l3_outlier_smoke.py

Exits 0 on success. Non-zero on any failure.
"""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_TOOL = Path(__file__).resolve().parent / "l3_outlier_report.py"
DEFAULT_THRESHOLD = 6.0
CEILING = 0.05  # matches tools/tessera/l3_outlier_report.py default


# ---------------------------------------------------------------------------
# Synthetic sidecar writer (v2)
# ---------------------------------------------------------------------------


SIDECAR_MAGIC = b"TDQT"
SIDECAR_HEADER_V2 = 40


def write_v2_sidecar(path: Path, weights: np.ndarray, threshold: float) -> None:
    """Write a v2 ``.dequant.f32`` sidecar matching the C++ writer layout.

    Layout:
        offset 0-3:   magic "TDQT"
        offset 4-7:   version = 2 (uint32 LE)
        offset 8-15:  rows (int64 LE)
        offset 16-23: cols (int64 LE)
        offset 24-27: dtype = 0 (F32, uint32 LE)
        offset 28-31: outlier_threshold (float32 LE)
        offset 32-39: outlier_count_total (int64 LE) -- recomputed at close
        offset 40-..: per-row int32 outlier_count strip
        then F32 data, row-major
    """
    if weights.dtype != np.float32:
        weights = weights.astype(np.float32)
    if weights.ndim != 2:
        raise ValueError("weights must be 2D")
    rows, cols = weights.shape
    per_row = np.sum(np.abs(weights) > threshold, axis=1).astype(np.int32)
    total = int(per_row.sum())

    header = bytearray()
    header += SIDECAR_MAGIC
    header += struct.pack("<I", 2)
    header += struct.pack("<q", rows)
    header += struct.pack("<q", cols)
    header += struct.pack("<I", 0)
    header += struct.pack("<f", threshold)
    header += struct.pack("<q", total)
    assert len(header) == SIDECAR_HEADER_V2

    with open(path, "wb") as f:
        f.write(header)
        f.write(per_row.astype("<i4").tobytes())
        f.write(weights.astype("<f4").tobytes())


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def synthetic_healthy(seed: int = 0) -> tuple[str, np.ndarray]:
    """A 64x256 weight with N(0, 1) noise -> ~0 outliers at threshold 6.0.

    Six-sigma on a unit normal is ~2e-9; across 64*256 = 16384 elements
    the expected outlier count is < 1e-4. We assert the count is exactly
    zero (the sample won't hit > 6.0 in practice).
    """
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((64, 256)).astype(np.float32)
    return "ffn_down.weight", w


def synthetic_pathological(seed: int = 1) -> tuple[str, np.ndarray]:
    """A 32x128 weight where 10% of values are seeded above threshold.

    Used to verify the report's ceiling gate triggers correctly.
    """
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((32, 128)).astype(np.float32)
    n = int(0.10 * w.size)
    flat = w.reshape(-1)
    flat[:n] = rng.uniform(7.0, 10.0, size=n).astype(np.float32)
    return "attn_qkv.weight", w


# ---------------------------------------------------------------------------
# Smoke test driver
# ---------------------------------------------------------------------------


def run_report(sidecar_dir: Path, output_path: Path) -> tuple[int, dict]:
    """Invoke l3_outlier_report.py on `sidecar_dir` and return (rc, parsed)."""
    cmd = [
        sys.executable,
        str(REPORT_TOOL),
        "--sidecar-dir", f"smoke:{sidecar_dir}",
        "--output", str(output_path),
        "--quiet",
        "--ceiling", str(CEILING),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    parsed: dict = {}
    if output_path.is_file():
        parsed = json.loads(output_path.read_text(encoding="utf-8"))
    return proc.returncode, parsed


def assert_eq(label: str, got: object, want: object) -> None:
    if got != want:
        raise AssertionError(f"{label}: got {got!r}, want {want!r}")


def main() -> int:
    if not REPORT_TOOL.is_file():
        print(f"missing report tool at {REPORT_TOOL}", file=sys.stderr)
        return 2

    tmp = Path(tempfile.mkdtemp(prefix="tessera_l3_smoke_"))
    try:
        # Case 1: healthy tensor -> report exits 0, count == 0.
        sidecar_dir = tmp / "healthy"
        sidecar_dir.mkdir(parents=True, exist_ok=True)
        name, w = synthetic_healthy()
        write_v2_sidecar(sidecar_dir / f"{name}.dequant.f32", w, DEFAULT_THRESHOLD)
        out_json = tmp / "healthy.json"
        rc, parsed = run_report(sidecar_dir, out_json)
        if rc != 0:
            print(f"healthy: report exited {rc} (expected 0)", file=sys.stderr)
            return 1
        groups = parsed.get("groups", [])
        assert_eq("healthy group count", len(groups), 1)
        tensors = groups[0]["tensors"]
        assert_eq("healthy tensor count", len(tensors), 1)
        t = tensors[0]
        assert_eq("healthy name", t["name"], name)
        assert_eq("healthy rows", t["rows"], w.shape[0])
        assert_eq("healthy cols", t["cols"], w.shape[1])
        assert_eq("healthy outlier_count", t["outlier_count"], 0)
        assert_eq("healthy threshold", t["threshold"], DEFAULT_THRESHOLD)
        expected_total = int(np.sum(np.abs(w) > DEFAULT_THRESHOLD))
        assert_eq("healthy total_elements", t["total_elements"], int(w.size))
        if t["outlier_count"] != expected_total:
            print(
                f"healthy: outlier_count={t['outlier_count']} but "
                f"analytical total={expected_total}",
                file=sys.stderr,
            )
            return 1

        # Case 2: pathological tensor -> report exits 2 (ceiling breach).
        sidecar_dir_p = tmp / "pathological"
        sidecar_dir_p.mkdir(parents=True, exist_ok=True)
        name_p, w_p = synthetic_pathological()
        write_v2_sidecar(
            sidecar_dir_p / f"{name_p}.dequant.f32",
            w_p,
            DEFAULT_THRESHOLD,
        )
        out_json_p = tmp / "pathological.json"
        rc_p, parsed_p = run_report(sidecar_dir_p, out_json_p)
        if rc_p != 2:
            print(
                f"pathological: report exited {rc_p} (expected 2, ceiling breach)",
                file=sys.stderr,
            )
            return 1
        tensors_p = parsed_p["groups"][0]["tensors"]
        t_p = tensors_p[0]
        expected_count = int(np.sum(np.abs(w_p) > DEFAULT_THRESHOLD))
        assert_eq("pathological outlier_count", t_p["outlier_count"], expected_count)
        if t_p["outlier_fraction"] <= CEILING:
            print(
                f"pathological: fraction {t_p['outlier_fraction']:.4%} did not "
                f"exceed ceiling {CEILING:.2%}",
                file=sys.stderr,
            )
            return 1

        # Case 3: mixed directory -> both tensors reported, the
        # pathological one drives the exit code.
        mixed = tmp / "mixed"
        mixed.mkdir(parents=True, exist_ok=True)
        write_v2_sidecar(mixed / f"{name}.dequant.f32", w, DEFAULT_THRESHOLD)
        write_v2_sidecar(
            mixed / f"{name_p}.dequant.f32", w_p, DEFAULT_THRESHOLD,
        )
        out_json_m = tmp / "mixed.json"
        rc_m, parsed_m = run_report(mixed, out_json_m)
        if rc_m != 2:
            print(f"mixed: report exited {rc_m} (expected 2)", file=sys.stderr)
            return 1
        assert_eq("mixed tensor count", len(parsed_m["groups"][0]["tensors"]), 2)

        print("l3_outlier_smoke: ok "
              "(healthy=0 outliers, pathological=ceiling breach, mixed=2 tensors)")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
