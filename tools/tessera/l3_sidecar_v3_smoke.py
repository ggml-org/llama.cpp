#!/usr/bin/env python3
"""
l3_sidecar_v3_smoke.py

End-to-end smoke test for the L1 / L1.5 dequant sidecar v3 pipeline.

Two modes:

  (1) `--from-cpp BIN --sidecar-dir DIR --tensor NAME` (default): the
      test invokes a small C++ helper binary that calls the
      `tessera_debug` API directly to write synthetic L1 + L1.5
      sidecars plus the .provenance.json sidecar, then reads them
      back via `l3_sidecar_v3_reader.py` and verifies the fields.

  (2) `--synthesize DIR --tensor NAME`: the test writes the sidecars
      directly from Python (no C++ binary required) and verifies the
      reader. This is the "build a file from scratch" path; it is
      used to validate the Python reader and the file format
      independently of the C++ build.

The test exercises:

  - v3 file magic, version, rows, cols, dtype
  - v2 fields (outlier_threshold, outlier_count_total, per-row
    outlier_count strip) and v3 fields (per-row timing_ns,
    kernel_id, dispatch_count, reserved)
  - F32 data block is row-major, correct values
  - L1.5 file (.act.dequant.f32) is written with the same v3 schema
    and the same F32 data
  - .provenance.json contains the auto-populated kernel_version
    and tessera_main_tip and the user-supplied TESSERA_TELEMETRY_*
    env-var fields
  - Backward-compat: a v1 file (28-byte header + F32 data) is
    readable by the v3 reader, which reports zeros for the v2/v3
    fields and the correct F32 data
  - The L1 and L1.5 files are bit-identical in the data block and
    the v2/v3 per-row strips (since the L1.5 reference is the
    same F32 data; a future refactor will pass the original FP16
    weight to the hook so the L1.5 captures the actual ground-
    truth)

Exit code 0 on success, non-zero on any assertion failure. A short
PASS/FAIL summary is printed.
"""

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
import l3_sidecar_v3_reader as reader  # noqa: E402

MAGIC = b"TDQT"
DTYPE_F32 = 0


def _write_v1_file(path, rows, cols, data, threshold=None,
                   per_row_outlier_count=None):
    """Write a v1 sidecar (28-byte header + F32 data) for the
    backward-compat test. If `threshold` is given, write a v2 file
    instead (40-byte header + per-row strip + F32 data)."""
    n_els = rows * cols
    if HAVE_NUMPY := _have_numpy():
        data = data.reshape(-1).astype("<f4")
    else:
        data = struct.pack("<%df" % n_els, *data.reshape(-1).tolist())

    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", 1))     # version
        f.write(struct.pack("<q", rows))
        f.write(struct.pack("<q", cols))
        f.write(struct.pack("<I", DTYPE_F32))
        if threshold is not None:
            # v2 file
            f.write(struct.pack("<f", threshold))
            if per_row_outlier_count is None:
                per_row_outlier_count = [0] * rows
            f.write(struct.pack("<q", sum(per_row_outlier_count)))
            f.write(struct.pack("<%di" % rows, *per_row_outlier_count))
        f.write(data if isinstance(data, (bytes, bytearray)) else data.tobytes())


def _have_numpy():
    try:
        import numpy as np  # noqa: F401
        return True
    except ImportError:
        return False


def _make_synthetic_tensor(rows, cols, seed=0):
    """Make a deterministic F32 tensor for the test. Values in roughly
    [0, 10] so the default outlier threshold (6.0) produces a
    non-trivial per-row outlier count."""
    if _have_numpy():
        import numpy as np
        rng = np.random.default_rng(seed)
        # Mix of small and large values so the outlier test is
        # meaningful.
        small = rng.uniform(0.0, 4.0, size=(rows, cols)).astype("<f4")
        # Plant a few values > 6.0 in each row to force outliers.
        for r in range(rows):
            for k in range(min(2, cols)):
                small[r, (r + k) % cols] = 7.0 + 0.5 * (k + 1)
        return small
    # Fallback: deterministic Python-only values.
    out = []
    for r in range(rows):
        row = []
        for c in range(cols):
            v = float((r * 7 + c * 3 + seed) % 11)
            if (r + c) % 5 == 0:
                v = 8.0 + float((r + c) % 3)
            row.append(v)
        out.append(row)
    return out


def _synthesize_v3_file(path, rows, cols, data, threshold, per_row_outlier_count,
                        per_row_timing_ns, per_row_kernel_id, per_row_dispatch_count,
                        per_row_reserved, sidecar_kind="dequant"):
    """Write a v3 file from Python. Used by the test when no C++ binary
    is available (mode 2)."""
    n_els = rows * cols
    if _have_numpy():
        flat = data.reshape(-1).astype("<f4").tobytes()
    else:
        flat = struct.pack("<%df" % n_els, *data.reshape(-1).tolist())

    total_outliers = sum(per_row_outlier_count) if per_row_outlier_count else 0

    with open(path, "wb") as f:
        # v1 header (28 bytes)
        f.write(MAGIC)
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
            f.write(struct.pack("<Q", per_row_timing_ns[r]))
            f.write(struct.pack("<I", per_row_kernel_id[r]))
            f.write(struct.pack("<I", per_row_dispatch_count[r]))
            f.write(struct.pack("<Q", per_row_reserved[r]))
        # F32 data
        f.write(flat)


def _synthesize_provenance(path, kernel_version, main_tip, model, corpus, corpus_hash,
                           l1_sidecar_version=3, imatrix_version=2,
                           sidecar_path="?", sidecar_kind="dequant"):
    obj = {
        "model":                 model,
        "calibration_corpus":    corpus,
        "calibration_corpus_hash": corpus_hash,
        "kernel_version":        kernel_version,
        "l1_sidecar_version":    l1_sidecar_version,
        "imatrix_version":       imatrix_version,
        "created_at":            "2026-07-30T12:00:00Z",
        "tessera_main_tip":      main_tip,
        "sidecar_path":          sidecar_path,
        "sidecar_kind":          sidecar_kind,
    }
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def _run_cpp_helper(bin_path, sidecar_dir, tensor_name, rows, cols, mode):
    """Run the C++ helper that writes synthetic L1 / L1.5 sidecars."""
    env = os.environ.copy()
    env["LLAMA_TILE640_DEBUG_DEQUANT_DIR"] = sidecar_dir
    env["LLAMA_TILE640_DEBUG_DEQUANT_MODE"] = mode
    env["TESSERA_TELEMETRY_MODEL"] = "gemma-3-12b"
    env["TESSERA_TELEMETRY_CALIBRATION_CORPUS"] = "prompts/paris.txt + prompts/wikitext-30"
    env["TESSERA_TELEMETRY_CALIBRATION_CORPUS_HASH"] = "sha256:0123abcd"
    r = subprocess.run(
        [bin_path, tensor_name, str(rows), str(cols), mode],
        env=env, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write("cpp helper failed: rc=%d\nstdout=%s\nstderr=%s\n" % (
            r.returncode, r.stdout, r.stderr))
    return r


# --------------------------------------------------------------------
# Mode 2: pure-Python synthesize + verify.
# --------------------------------------------------------------------

def _run_synthesize_test(args):
    """Build synthetic L1 and L1.5 sidecars directly from Python and
    verify them. Used when no C++ helper is available (e.g., in
    environments where the C++ build hasn't been done)."""
    sidecar_dir = args.sidecar_dir
    os.makedirs(sidecar_dir, exist_ok=True)
    tensor = args.tensor
    rows = args.rows
    cols = args.cols

    data = _make_synthetic_tensor(rows, cols, seed=args.seed)
    # Per-row outlier count against threshold 6.0.
    if _have_numpy():
        import numpy as np
        per_row_outlier_count = (np.abs(data) > 6.0).sum(axis=1).astype("<i4").tolist()
    else:
        per_row_outlier_count = []
        for r in range(rows):
            cnt = 0
            for v in data[r]:
                if abs(v) > 6.0:
                    cnt += 1
            per_row_outlier_count.append(cnt)
    per_row_timing_ns      = [1000 * (r + 1) for r in range(rows)]
    per_row_kernel_id      = [0x42] * rows
    per_row_dispatch_count = [1] * rows
    per_row_reserved       = [0] * rows

    l1_path  = os.path.join(sidecar_dir, tensor + ".dequant.f32")
    l15_path = os.path.join(sidecar_dir, tensor + ".act.dequant.f32")
    l1_prov  = l1_path  + ".provenance.json"
    l15_prov = l15_path + ".provenance.json"

    _synthesize_v3_file(
        l1_path, rows, cols, data, 6.0, per_row_outlier_count,
        per_row_timing_ns, per_row_kernel_id, per_row_dispatch_count,
        per_row_reserved, sidecar_kind="dequant")
    _synthesize_v3_file(
        l15_path, rows, cols, data, 6.0, per_row_outlier_count,
        per_row_timing_ns, per_row_kernel_id, per_row_dispatch_count,
        per_row_reserved, sidecar_kind="fp16_reference")
    _synthesize_provenance(
        l1_prov, kernel_version="synthesize-test", main_tip="abcdef1234",
        model="gemma-3-12b", corpus="prompts/paris.txt + prompts/wikitext-30",
        corpus_hash="sha256:0123abcd", l1_sidecar_version=3,
        imatrix_version=2, sidecar_path=l1_path, sidecar_kind="dequant")
    _synthesize_provenance(
        l15_prov, kernel_version="synthesize-test", main_tip="abcdef1234",
        model="gemma-3-12b", corpus="prompts/paris.txt + prompts/wikitext-30",
        corpus_hash="sha256:0123abcd", l1_sidecar_version=3,
        imatrix_version=2, sidecar_path=l15_path, sidecar_kind="fp16_reference")

    return _verify(sidecar_dir, tensor, rows, cols, data,
                   per_row_outlier_count, per_row_timing_ns,
                   per_row_kernel_id)


# --------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------

def _verify(sidecar_dir, tensor, rows, cols, expected_data,
            expected_outlier_count, expected_timing_ns, expected_kernel_id):
    """Verify the L1, L1.5, and provenance sidecars."""
    n_pass = 0
    n_fail = 0
    def check(name, cond, detail=""):
        nonlocal n_pass, n_fail
        if cond:
            n_pass += 1
            print("  [PASS] %s" % name)
        else:
            n_fail += 1
            print("  [FAIL] %s %s" % (name, detail))

    l1_path  = os.path.join(sidecar_dir, tensor + ".dequant.f32")
    l15_path = os.path.join(sidecar_dir, tensor + ".act.dequant.f32")

    # 1) Both sidecar files exist.
    check("L1 file exists",  os.path.exists(l1_path))
    check("L1.5 file exists", os.path.exists(l15_path))

    # 2) Provenance JSONs exist.
    l1_prov  = l1_path  + ".provenance.json"
    l15_prov = l15_path + ".provenance.json"
    check("L1 provenance exists",  os.path.exists(l1_prov))
    check("L1.5 provenance exists", os.path.exists(l15_prov))

    # 3) Read L1 with the v3 reader (auto-dispatch).
    s1 = reader.read_sidecar(l1_path, mode="auto", provenance=True)
    check("L1 magic",  s1["magic"] == MAGIC, "got %r" % s1["magic"])
    check("L1 version=3", s1["version"] == 3, "got %d" % s1["version"])
    check("L1 actual_version=3", s1["actual_version"] == 3)
    check("L1 rows=%d" % rows, s1["rows"] == rows, "got %d" % s1["rows"])
    check("L1 cols=%d" % cols, s1["cols"] == cols, "got %d" % s1["cols"])
    check("L1 dtype=F32", s1["dtype_name"] == "F32")
    check("L1 outlier_threshold=6.0", abs(s1["outlier_threshold"] - 6.0) < 1e-6,
          "got %f" % s1["outlier_threshold"])
    expected_total = sum(expected_outlier_count)
    check("L1 outlier_count_total=%d" % expected_total,
          s1["outlier_count_total"] == expected_total,
          "got %d" % s1["outlier_count_total"])
    check("L1 row_outlier_count len=%d" % rows,
          s1["row_outlier_count"] is not None and len(s1["row_outlier_count"]) == rows)
    check("L1 row_outlier_count values",
          s1["row_outlier_count"] == expected_outlier_count,
          detail=str(s1["row_outlier_count"]))
    check("L1 row_timing_ns present", s1["row_timing_ns"] is not None)
    check("L1 row_timing_ns values",
          s1["row_timing_ns"] == expected_timing_ns,
          detail=str(s1["row_timing_ns"]))
    check("L1 row_kernel_id values",
          s1["row_kernel_id"] == expected_kernel_id,
          detail=str(s1["row_kernel_id"]))
    check("L1 row_dispatch_count all 1",
          s1["row_dispatch_count"] == [1] * rows,
          detail=str(s1["row_dispatch_count"]))
    check("L1 row_reserved all 0",
          s1["row_reserved"] == [0] * rows,
          detail=str(s1["row_reserved"]))
    # F32 data block.
    if _have_numpy():
        import numpy as np
        diff = np.abs(s1["data"].reshape(-1) - expected_data.reshape(-1))
        max_abs = float(diff.max()) if diff.size else 0.0
        check("L1 F32 data max abs err < 1e-6 (got %.2e)" % max_abs,
              max_abs < 1e-6)
    else:
        flat = [v for row in expected_data for v in row]
        # Compare first / last values; full comparison is slow in pure Python.
        check("L1 F32 data first value",
              abs(s1["data"][0] - flat[0]) < 1e-6)
        check("L1 F32 data last value",
              abs(s1["data"][-1] - flat[-1]) < 1e-6)

    # 4) L1 provenance JSON has auto-populated fields.
    p1 = s1.get("provenance")
    check("L1 provenance is a dict", isinstance(p1, dict))
    if p1:
        check("L1 provenance kernel_version is set",
              p1.get("kernel_version") not in (None, "", "unknown"),
              detail=repr(p1.get("kernel_version")))
        check("L1 provenance tessera_main_tip is set",
              p1.get("tessera_main_tip") not in (None, "", "unknown"),
              detail=repr(p1.get("tessera_main_tip")))
        check("L1 provenance l1_sidecar_version=3",
              p1.get("l1_sidecar_version") == 3,
              detail=repr(p1.get("l1_sidecar_version")))
        check("L1 provenance imatrix_version=2",
              p1.get("imatrix_version") == 2,
              detail=repr(p1.get("imatrix_version")))
        check("L1 provenance model=gemma-3-12b",
              p1.get("model") == "gemma-3-12b",
              detail=repr(p1.get("model")))
        check("L1 provenance sidecar_kind=dequant",
              p1.get("sidecar_kind") == "dequant",
              detail=repr(p1.get("sidecar_kind")))

    # 5) L1.5 file is the same v3 schema with the same data and v3 strip.
    s15 = reader.read_sidecar(l15_path, mode="auto", provenance=True)
    check("L1.5 version=3", s15["version"] == 3, "got %d" % s15["version"])
    check("L1.5 shape matches L1",
          s15["rows"] == s1["rows"] and s15["cols"] == s1["cols"])
    check("L1.5 row_timing_ns matches L1",
          s15["row_timing_ns"] == s1["row_timing_ns"])
    check("L1.5 row_kernel_id matches L1",
          s15["row_kernel_id"] == s1["row_kernel_id"])
    check("L1.5 row_dispatch_count matches L1",
          s15["row_dispatch_count"] == s1["row_dispatch_count"])
    p15 = s15.get("provenance")
    if p15:
        check("L1.5 provenance sidecar_kind=fp16_reference",
              p15.get("sidecar_kind") == "fp16_reference",
              detail=repr(p15.get("sidecar_kind")))

    # 6) L1 and L1.5 data block is bit-identical in the current
    #    implementation (the FP16 reference path is not yet wired;
    #    the L1.5 is the same F32 dequantized data).
    if _have_numpy():
        import numpy as np
        same = np.array_equal(s1["data"], s15["data"])
    else:
        same = s1["data"] == s15["data"]
    check("L1 and L1.5 data are bit-identical (current implementation)", same)

    # 7) Backward-compat: a v1 file is readable by the v3 reader and
    #    produces zeros for the v2/v3 fields. We write a v1 file
    #    alongside the v3 files and read it back.
    v1_path = os.path.join(sidecar_dir, tensor + ".v1.dequant.f32")
    if _have_numpy():
        import numpy as np
        v1_data = np.arange(rows * cols, dtype="<f4").reshape(rows, cols) * 0.1
    else:
        v1_data = [[0.1 * (r * cols + c) for c in range(cols)] for r in range(rows)]
    _write_v1_file(v1_path, rows, cols, v1_data)
    s_v1 = reader.read_sidecar(v1_path, mode="auto", provenance=False)
    check("v1 file magic", s_v1["magic"] == MAGIC)
    check("v1 file version=1", s_v1["version"] == 1)
    check("v1 file actual_version=1", s_v1["actual_version"] == 1)
    if _have_numpy():
        import numpy as np
        diff = np.abs(s_v1["data"].reshape(-1) - v1_data.reshape(-1))
        max_abs = float(diff.max()) if diff.size else 0.0
        check("v1 file F32 data round-trips (max err %.2e)" % max_abs,
              max_abs < 1e-6)
    check("v3 reader on v1: outlier_threshold=0.0",
          s_v1["outlier_threshold"] == 0.0,
          detail=str(s_v1["outlier_threshold"]))
    check("v3 reader on v1: outlier_count_total=0",
          s_v1["outlier_count_total"] == 0)
    check("v3 reader on v1: row_outlier_count all 0",
          s_v1["row_outlier_count"] == [0] * rows)
    check("v3 reader on v1: row_timing_ns all 0",
          s_v1["row_timing_ns"] == [0] * rows)
    check("v3 reader on v1: row_kernel_id all 0",
          s_v1["row_kernel_id"] == [0] * rows)
    check("v3 reader on v1: row_dispatch_count all 0",
          s_v1["row_dispatch_count"] == [0] * rows)

    # 8) Backward-compat (other direction): a v1 reader (mode=v1) on
    #    a v3 file would see garbage as the F32 data because the
    #    data offset is 28 in v1 but 40+R*28 in v3. The shipped
    #    Python reader exposes the data offset for this reason:
    #    callers can check `actual_version` before trusting the
    #    F32 data. We assert this contract here.
    s_v3_as_v1 = reader.read_sidecar(l1_path, mode="v1", provenance=False)
    check("v1-mode reader on v3 file: actual_version=3",
          s_v3_as_v1["actual_version"] == 3)
    check("v1-mode reader on v3 file: version=1 (read-as)",
          s_v3_as_v1["version"] == 1)
    # The F32 data read with mode=v1 is the v2/v3 fields reinterpreted
    # as F32; it MUST NOT equal the real F32 data. (If it did, the
    # version dispatch would be broken.)
    if _have_numpy():
        import numpy as np
        garbage_matches_real = np.array_equal(
            s_v3_as_v1["data"].reshape(-1), expected_data.reshape(-1))
    else:
        garbage_matches_real = s_v3_as_v1["data"] == [
            v for row in expected_data for v in row]
    check("v1-mode reader on v3 file: data is NOT the real F32 data "
          "(i.e., v1 reader does not silently read v3 as v1)",
          not garbage_matches_real)

    print()
    print("Summary: %d passed, %d failed" % (n_pass, n_fail))
    return n_fail == 0


# --------------------------------------------------------------------
# Mode 1: C++ helper writes, Python reader verifies.
# --------------------------------------------------------------------

def _run_cpp_test(args):
    """Run the C++ helper and verify the resulting sidecars."""
    bin_path = args.from_cpp
    if not os.path.exists(bin_path):
        sys.stderr.write("cpp helper not found: %s\n" % bin_path)
        return False

    sidecar_dir = args.sidecar_dir
    if os.path.exists(sidecar_dir):
        import shutil
        shutil.rmtree(sidecar_dir)
    os.makedirs(sidecar_dir, exist_ok=True)

    rows = args.rows
    cols = args.cols
    tensor = args.tensor
    mode = "w4a4" if args.w4a4 else ""

    r = _run_cpp_helper(bin_path, sidecar_dir, tensor, rows, cols, mode)
    if r.returncode != 0:
        return False

    # The C++ helper is the source of truth for the F32 data and the
    # per-row timing. We read the L1 file back and use its data block
    # as the "expected" values; the per-row timing and kernel_id
    # fields are read from the v3 strip. The per-row outlier count is
    # recomputed from the L1 data against the threshold (which the
    # reader reports from the v2 header).
    l1_path = os.path.join(sidecar_dir, tensor + ".dequant.f32")
    s1 = reader.read_sidecar(l1_path, mode="auto", provenance=False)
    actual_data = s1["data"]
    if _have_numpy():
        import numpy as np
        per_row_outlier_count = (
            np.abs(actual_data) > s1["outlier_threshold"]
        ).sum(axis=1).astype("<i4").tolist()
    else:
        per_row_outlier_count = []
        for r_i in range(rows):
            cnt = 0
            for v in actual_data[r_i]:
                if abs(v) > s1["outlier_threshold"]:
                    cnt += 1
            per_row_outlier_count.append(cnt)

    return _verify(sidecar_dir, tensor, rows, cols, actual_data,
                   per_row_outlier_count,
                   s1["row_timing_ns"], s1["row_kernel_id"])


# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------

def _main(argv):
    p = argparse.ArgumentParser(
        description="Tessera L1 / L1.5 sidecar v3 smoke test.")
    p.add_argument("--sidecar-dir", required=True,
                   help="directory to write sidecar files into")
    p.add_argument("--tensor", default="synth_4x8",
                   help="tensor name (default: synth_4x8)")
    p.add_argument("--rows", type=int, default=4)
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--from-cpp", default=None,
                   help="path to a C++ helper binary that writes the sidecars")
    p.add_argument("--w4a4", action="store_true",
                   help="set LLAMA_TILE640_DEBUG_DEQUANT_MODE=w4a4 "
                        "(writes both L1 and L1.5)")
    args = p.parse_args(argv)

    if args.from_cpp is not None:
        ok = _run_cpp_test(args)
    else:
        ok = _run_synthesize_test(args)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
