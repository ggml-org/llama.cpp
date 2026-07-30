#!/usr/bin/env python3
"""Smoke + unit tests for tools/tessera/latency_lut.py.

Builds synthetic v3 sidecar files (matching the layout in
common/tessera-debug/tessera-debug.h) and exercises the LUT builder
both in-process (via the module's helpers) and as a CLI subprocess.

Test cases (all run from a fresh tmpdir per test):

  1. Single tensor, single kernel            -> 1 LUT entry
  2. Two tensors same shape, same kernel     -> 1 entry, count=2
  3. Two tensors same shape, different kernel-> 2 entries
  4. Two tensors different shape, same kernel-> 2 entries
  5. Empty sidecar dir                       -> empty entries, summary=0
  6. mean_ns is per-row mean, not per-tensor sum
       (4x8 with row timings [10,20,30,40] -> mean_ns=25, total=100)
  7. CLI subprocess runs to completion, exit 0, JSON schema valid
  8. --include-l15 reads L1.5 sidecars too
  9. v1 sidecar is skipped with a stderr warning (no v3 strip)

Run with:

    python3 -m unittest tests/test-tessera-latency-lut.py

Exits 0 on success, non-zero on any failure. A short PASS/FAIL summary
is printed.
"""

from __future__ import annotations

import importlib.util
import json
import math
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np  # noqa: F401  (the l3 reader uses numpy for F32 data)

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools" / "tessera"
LUT_PATH = TOOLS / "latency_lut.py"

SIDECAR_MAGIC = b"TDQT"
DTYPE_F32 = 0


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LUT = _load_module("latency_lut", LUT_PATH)


# ---------------------------------------------------------------------------
# Synthetic v3 sidecar writer (matches l3_sidecar_v3_smoke.py layout).
# ---------------------------------------------------------------------------


def write_v3_sidecar(path, rows, cols, per_row_timing_ns, kernel_id,
                     per_row_outlier_count=None, threshold=6.0,
                     sidecar_kind="dequant"):
    """Write a synthetic v3 sidecar at `path`.

    The F32 data block is filled with zeros (the LUT does not need the
    weight values, only the v3 strip). The provenance JSON is NOT
    written; the latency_lut module does not read it.
    """
    if per_row_outlier_count is None:
        per_row_outlier_count = [0] * rows
    total_outliers = sum(int(x) for x in per_row_outlier_count)
    with open(path, "wb") as f:
        # v1 header (28 bytes)
        f.write(SIDECAR_MAGIC)
        f.write(struct.pack("<I", 3))
        f.write(struct.pack("<q", rows))
        f.write(struct.pack("<q", cols))
        f.write(struct.pack("<I", DTYPE_F32))
        # v2 header (12 bytes)
        f.write(struct.pack("<f", threshold))
        f.write(struct.pack("<q", total_outliers))
        # v2 per-row strip
        f.write(struct.pack("<%di" % rows, *per_row_outlier_count))
        # v3 per-row strip (24 bytes per row)
        for r in range(rows):
            t_ns = int(per_row_timing_ns[r])
            f.write(struct.pack("<Q", t_ns))
            f.write(struct.pack("<I", int(kernel_id)))
            f.write(struct.pack("<I", 1))
            f.write(struct.pack("<Q", 0))
        # F32 data block (row-major, zeros)
        data = np.zeros((rows, cols), dtype="<f4").tobytes()
        f.write(data)


def write_v1_sidecar(path, rows, cols):
    """Write a v1-only sidecar (28-byte header + F32 data) for the
    backward-compat test. The v3 reader returns zeros for the v3 fields
    on such a file; latency_lut should skip it with a warning."""
    with open(path, "wb") as f:
        f.write(SIDECAR_MAGIC)
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<q", rows))
        f.write(struct.pack("<q", cols))
        f.write(struct.pack("<I", DTYPE_F32))
        data = np.zeros((rows, cols), dtype="<f4").tobytes()
        f.write(data)


# ---------------------------------------------------------------------------
# TestCase
# ---------------------------------------------------------------------------


class LatencyLUTTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="latency-lut-test-"))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run_cli(self, sidecar_dir, out_path, fmt="json", group_by=None,
                 include_l15=False, check=True):
        cmd = [
            sys.executable, str(LUT_PATH),
            "--sidecar-dir", str(sidecar_dir),
            "--out", str(out_path),
            "--format", fmt,
        ]
        if group_by is not None:
            cmd += ["--group-by", group_by]
        if include_l15:
            cmd += ["--include-l15"]
        return subprocess.run(cmd, capture_output=True, text=True, check=check)

    # --- in-process unit tests --------------------------------------------

    def test_single_tensor_single_kernel(self):
        d = self.tmpdir / "case1"
        d.mkdir()
        write_v3_sidecar(d / "a.dequant.f32", 4, 8,
                         per_row_timing_ns=[10, 20, 30, 40],
                         kernel_id=0x42)
        recs = LUT.collect_sidecars(str(d))
        self.assertEqual(len(recs), 1)
        entries, summary = LUT.aggregate(recs, LUT.GROUP_SHAPE_KERNEL)
        self.assertEqual(summary["n_tensors"], 1)
        self.assertEqual(summary["n_groups"], 1)
        self.assertEqual(summary["n_kernel_ids"], 1)
        self.assertEqual(len(entries), 1)
        e = entries[0]
        self.assertEqual(e["shape"], "4x8")
        self.assertEqual(e["kernel_id"], 0x42)
        self.assertEqual(e["count"], 1)
        self.assertAlmostEqual(e["mean_ns"], 25.0, places=6)
        self.assertAlmostEqual(e["mean_total_ns"], 100.0, places=6)
        self.assertAlmostEqual(e["std_ns"], math.sqrt(125.0), places=4)

    def test_two_tensors_same_shape_same_kernel(self):
        d = self.tmpdir / "case2"
        d.mkdir()
        write_v3_sidecar(d / "a.dequant.f32", 4, 8,
                         per_row_timing_ns=[10, 20, 30, 40], kernel_id=0x42)
        write_v3_sidecar(d / "b.dequant.f32", 4, 8,
                         per_row_timing_ns=[50, 60, 70, 80], kernel_id=0x42)
        recs = LUT.collect_sidecars(str(d))
        self.assertEqual(len(recs), 2)
        entries, summary = LUT.aggregate(recs, LUT.GROUP_SHAPE_KERNEL)
        self.assertEqual(summary["n_tensors"], 2)
        self.assertEqual(summary["n_groups"], 1)
        self.assertEqual(summary["n_kernel_ids"], 1)
        self.assertEqual(len(entries), 1)
        e = entries[0]
        self.assertEqual(e["shape"], "4x8")
        self.assertEqual(e["kernel_id"], 0x42)
        self.assertEqual(e["count"], 2)
        # per-row means: 25 and 65 -> mean of means = 45
        self.assertAlmostEqual(e["mean_ns"], 45.0, places=6)
        # per-tensor totals: 100 and 260 -> mean = 180
        self.assertAlmostEqual(e["mean_total_ns"], 180.0, places=6)
        # both per-row stds are sqrt(125); mean of (std^2) = 125 -> sqrt = sqrt(125)
        self.assertAlmostEqual(e["std_ns"], math.sqrt(125.0), places=4)

    def test_two_tensors_same_shape_different_kernel(self):
        d = self.tmpdir / "case3"
        d.mkdir()
        write_v3_sidecar(d / "a.dequant.f32", 4, 8,
                         per_row_timing_ns=[10, 20, 30, 40], kernel_id=0x42)
        write_v3_sidecar(d / "b.dequant.f32", 4, 8,
                         per_row_timing_ns=[50, 60, 70, 80], kernel_id=0x99)
        recs = LUT.collect_sidecars(str(d))
        entries, summary = LUT.aggregate(recs, LUT.GROUP_SHAPE_KERNEL)
        self.assertEqual(summary["n_tensors"], 2)
        self.assertEqual(summary["n_groups"], 2)
        self.assertEqual(summary["n_kernel_ids"], 2)
        self.assertEqual(len(entries), 2)
        kernel_ids = sorted(e["kernel_id"] for e in entries)
        self.assertEqual(kernel_ids, [0x42, 0x99])
        for e in entries:
            self.assertEqual(e["shape"], "4x8")
            self.assertEqual(e["count"], 1)

    def test_two_tensors_different_shape_same_kernel(self):
        d = self.tmpdir / "case4"
        d.mkdir()
        write_v3_sidecar(d / "a.dequant.f32", 4, 8,
                         per_row_timing_ns=[10, 20, 30, 40], kernel_id=0x42)
        write_v3_sidecar(d / "b.dequant.f32", 8, 4,
                         per_row_timing_ns=[5, 15, 25, 35, 45, 55, 65, 75],
                         kernel_id=0x42)
        recs = LUT.collect_sidecars(str(d))
        entries, summary = LUT.aggregate(recs, LUT.GROUP_SHAPE_KERNEL)
        self.assertEqual(summary["n_tensors"], 2)
        self.assertEqual(summary["n_groups"], 2)
        self.assertEqual(summary["n_kernel_ids"], 1)
        self.assertEqual(len(entries), 2)
        shapes = sorted(e["shape"] for e in entries)
        self.assertEqual(shapes, ["4x8", "8x4"])
        for e in entries:
            self.assertEqual(e["kernel_id"], 0x42)
            self.assertEqual(e["count"], 1)

    def test_empty_sidecar_dir(self):
        d = self.tmpdir / "case5"
        d.mkdir()
        recs = LUT.collect_sidecars(str(d))
        self.assertEqual(recs, [])
        entries, summary = LUT.aggregate(recs, LUT.GROUP_SHAPE_KERNEL)
        self.assertEqual(entries, [])
        self.assertEqual(summary["n_tensors"], 0)
        self.assertEqual(summary["n_groups"], 0)
        self.assertEqual(summary["n_kernel_ids"], 0)

    def test_mean_is_per_row_not_per_tensor(self):
        # Spec: 4x8 with row timings [10, 20, 30, 40] -> mean_ns=25, total=100.
        # The mean must be the per-row mean, NOT the per-tensor sum.
        d = self.tmpdir / "case6"
        d.mkdir()
        write_v3_sidecar(d / "a.dequant.f32", 4, 8,
                         per_row_timing_ns=[10, 20, 30, 40], kernel_id=0x42)
        recs = LUT.collect_sidecars(str(d))
        entries, _ = LUT.aggregate(recs, LUT.GROUP_SHAPE_KERNEL)
        e = entries[0]
        self.assertAlmostEqual(
            e["mean_ns"], 25.0, places=6,
            msg="mean_ns must be the per-row mean (25.0), not the per-tensor sum (100.0)")
        self.assertNotAlmostEqual(e["mean_ns"], 100.0, places=6)
        self.assertAlmostEqual(e["mean_total_ns"], 100.0, places=6)

    def test_v1_sidecar_is_skipped(self):
        d = self.tmpdir / "v1case"
        d.mkdir()
        write_v1_sidecar(d / "old.dequant.f32", 4, 8)
        recs = LUT.collect_sidecars(str(d))
        self.assertEqual(recs, [])

    def test_include_l15(self):
        d = self.tmpdir / "l15case"
        d.mkdir()
        write_v3_sidecar(d / "a.dequant.f32", 4, 8,
                         per_row_timing_ns=[10, 20, 30, 40], kernel_id=0x42)
        write_v3_sidecar(d / "a.act.dequant.f32", 4, 8,
                         per_row_timing_ns=[10, 20, 30, 40], kernel_id=0x42)
        # Without --include-l15, only L1.
        recs = LUT.collect_sidecars(str(d), include_l15=False)
        self.assertEqual(len(recs), 1)
        # With --include-l15, both files.
        recs = LUT.collect_sidecars(str(d), include_l15=True)
        self.assertEqual(len(recs), 2)

    # --- CLI subprocess tests ---------------------------------------------

    def test_cli_json_subprocess(self):
        d = self.tmpdir / "case7"
        d.mkdir()
        write_v3_sidecar(d / "a.dequant.f32", 4, 8,
                         per_row_timing_ns=[10, 20, 30, 40], kernel_id=0x42)
        write_v3_sidecar(d / "b.dequant.f32", 4, 8,
                         per_row_timing_ns=[50, 60, 70, 80], kernel_id=0x42)
        out = d / "lut.json"
        r = self._run_cli(d, out, fmt="json", group_by="shape-kernel")
        self.assertEqual(r.returncode, 0,
                         msg="stderr=%s stdout=%s" % (r.stderr, r.stdout))
        obj = json.loads(out.read_text())
        self.assertEqual(obj["schema"], "llama.tessera.latency-lut.v1")
        self.assertEqual(obj["group_by"], "shape-kernel")
        self.assertIn("entries", obj)
        self.assertIn("summary", obj)
        for e in obj["entries"]:
            for f in ("shape", "kernel_id", "mean_ns", "std_ns",
                      "count", "mean_total_ns"):
                self.assertIn(f, e)
        self.assertEqual(obj["summary"]["n_tensors"], 2)
        self.assertEqual(obj["summary"]["n_groups"], 1)

    def test_cli_table_subprocess(self):
        d = self.tmpdir / "case8"
        d.mkdir()
        write_v3_sidecar(d / "a.dequant.f32", 4, 8,
                         per_row_timing_ns=[10, 20, 30, 40], kernel_id=0x42)
        out = d / "lut.txt"
        r = self._run_cli(d, out, fmt="table")
        self.assertEqual(r.returncode, 0, msg="stderr=%s" % r.stderr)
        body = out.read_text()
        # Header line is present.
        self.assertIn("Tessera latency LUT", body)
        self.assertIn("4x8", body)
        # ASCII only.
        body.encode("ascii")  # raises if non-ASCII

    def test_cli_empty_dir(self):
        d = self.tmpdir / "case9"
        d.mkdir()
        out = d / "lut.json"
        r = self._run_cli(d, out, fmt="json")
        self.assertEqual(r.returncode, 0, msg="stderr=%s" % r.stderr)
        obj = json.loads(out.read_text())
        self.assertEqual(obj["entries"], [])
        self.assertEqual(obj["summary"]["n_tensors"], 0)
        self.assertEqual(obj["summary"]["n_groups"], 0)
        self.assertEqual(obj["summary"]["n_kernel_ids"], 0)

    def test_cli_group_by_shape(self):
        d = self.tmpdir / "case10"
        d.mkdir()
        # Two tensors, same shape (4x8), different kernels. With
        # --group-by shape they should collapse into 1 group.
        write_v3_sidecar(d / "a.dequant.f32", 4, 8,
                         per_row_timing_ns=[10, 20, 30, 40], kernel_id=0x42)
        write_v3_sidecar(d / "b.dequant.f32", 4, 8,
                         per_row_timing_ns=[50, 60, 70, 80], kernel_id=0x99)
        out = d / "lut.json"
        r = self._run_cli(d, out, fmt="json", group_by="shape")
        self.assertEqual(r.returncode, 0, msg="stderr=%s" % r.stderr)
        obj = json.loads(out.read_text())
        self.assertEqual(obj["summary"]["n_tensors"], 2)
        self.assertEqual(obj["summary"]["n_groups"], 1)
        # n_kernel_ids in the summary counts distinct kernel_ids that
        # actually appear in `entries`; with group_by=shape there is
        # 1 entry, so it reports 1.
        self.assertEqual(obj["summary"]["n_kernel_ids"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
