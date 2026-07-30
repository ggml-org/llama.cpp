#!/usr/bin/env python3
"""
test-tessera-per-layer-error.py

Smoke test for ``tools/tessera/per_layer_error_table.py``.

Builds synthetic v3 L1 / L1.5 sidecar files using the same
synthesize pattern as ``l3_sidecar_v3_smoke.py`` (no C++ build
required), then runs the CLI as a subprocess and verifies the
JSON / table output.

Test cases:

  1. L1 == L1.5 (data block is bit-identical) -> epsilon == 0
     for every tensor.
  2. L1 differs from L1.5 by a known delta ->
     epsilon matches ``||delta||^2 / ||L15||^2`` to 4 decimal
     places.
  3. Multi-tensor dir -> per-layer aggregation sums correctly
     across tensors within a layer, and the layer grouping
     matches the canonical ``blk.<N>`` prefix.
  4. Missing L1.5 file -> the tool skips the pair with a warning
     and still emits a valid output document.
  5. Missing L1 file -> same as (4), the other direction.

The test runs the CLI as a subprocess on the synthetic dir,
verifies exit 0, and checks the JSON schema. A short PASS / FAIL
summary is printed.
"""

import importlib.util
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS_DIR)
TOOL_PATH = os.path.join(ROOT, "tools", "tessera",
                         "per_layer_error_table.py")
READER_PATH = os.path.join(ROOT, "tools", "tessera",
                           "l3_sidecar_v3_reader.py")
sys.path.insert(0, os.path.dirname(READER_PATH))
import l3_sidecar_v3_reader as reader  # noqa: E402

SCHEMA = "llama.tessera.per-layer-error-table.v1"

MAGIC = b"TDQT"
DTYPE_F32 = 0
L1_SUFFIX = ".dequant.f32"
L15_SUFFIX = ".act.dequant.f32"


# --------------------------------------------------------------------
# Synthetic v3 file writer (mirrors l3_sidecar_v3_smoke.py).
# --------------------------------------------------------------------

def _write_v3(path: str, data: np.ndarray,
              per_row_timing_ns: list,
              per_row_kernel_id: list,
              per_row_dispatch_count: list,
              per_row_outlier_count: list,
              threshold: float = 6.0) -> None:
    """Write a v3 sidecar (40-byte header + 28R per-row strip + F32 data)."""
    rows, cols = data.shape
    total_outliers = int(sum(per_row_outlier_count))
    flat = data.reshape(-1).astype("<f4").tobytes()
    with open(path, "wb") as f:
        # v1 header
        f.write(MAGIC)
        f.write(struct.pack("<I", 3))
        f.write(struct.pack("<q", rows))
        f.write(struct.pack("<q", cols))
        f.write(struct.pack("<I", DTYPE_F32))
        # v2 header
        f.write(struct.pack("<f", threshold))
        f.write(struct.pack("<q", total_outliers))
        # v2 per-row strip
        f.write(struct.pack("<%di" % rows, *per_row_outlier_count))
        # v3 per-row strip (24 bytes per row)
        for r in range(rows):
            f.write(struct.pack("<Q", per_row_timing_ns[r]))
            f.write(struct.pack("<I", per_row_kernel_id[r]))
            f.write(struct.pack("<I", per_row_dispatch_count[r]))
            f.write(struct.pack("<Q", 0))
        # F32 data
        f.write(flat)


def _meta(rows: int) -> tuple:
    return (
        [1000 * (r + 1) for r in range(rows)],   # timing_ns
        [0x42] * rows,                            # kernel_id
        [1] * rows,                               # dispatch_count
        [0] * rows,                               # outlier_count
    )


def _synthesize_pair(d: str, name: str, l15: np.ndarray,
                     delta: np.ndarray) -> None:
    """Write an L1 / L1.5 pair under ``d`` with L1 = L15 - delta."""
    rows, cols = l15.shape
    l1 = (l15 - delta).astype("<f4")
    timing, kid, dcount, oc = _meta(rows)
    _write_v3(os.path.join(d, name + L1_SUFFIX),
              l1, timing, kid, dcount, oc)
    _write_v3(os.path.join(d, name + L15_SUFFIX),
              l15.astype("<f4"), timing, kid, dcount, oc)


def _deterministic_l15(rows: int, cols: int, seed: int) -> np.ndarray:
    """Deterministic, mostly-positive F32 reference tensor with a
    few values > 6.0 so the per-row outlier count is non-trivial."""
    rng = np.random.default_rng(seed)
    small = rng.uniform(0.5, 4.0, size=(rows, cols)).astype("<f4")
    for r in range(rows):
        for k in range(min(2, cols)):
            small[r, (r + k) % cols] = 7.0 + 0.5 * (k + 1)
    return small


# --------------------------------------------------------------------
# CLI runner.
# --------------------------------------------------------------------

def _run_cli(sidecar_dir: str, out_path: str,
             fmt: str = "json") -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, TOOL_PATH,
         "--sidecar-dir", sidecar_dir,
         "--out",       out_path,
         "--format",    fmt],
        capture_output=True, text=True)


# --------------------------------------------------------------------
# Tests.
# --------------------------------------------------------------------

class PerLayerErrorTableTest(unittest.TestCase):
    """Smoke test for the per-layer error table tool."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="ple_test_")
        self._out_json = os.path.join(self._tmp, "out.json")
        self._out_table = os.path.join(self._tmp, "out.txt")

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    # ---- 1) L1 == L1.5 -> epsilon == 0 -----------------------------

    def test_l1_equals_l15_gives_zero_epsilon(self) -> None:
        """Bit-identical L1 and L1.5 means every tensor has
        epsilon == 0.0 exactly."""
        d = os.path.join(self._tmp, "dir")
        os.makedirs(d, exist_ok=True)
        l15 = _deterministic_l15(4, 8, seed=1)
        _synthesize_pair(d, "blk.0.attn_q.weight", l15, delta=np.zeros_like(l15))
        _synthesize_pair(d, "blk.0.attn_k.weight", l15, delta=np.zeros_like(l15))

        r = _run_cli(d, self._out_json, fmt="json")
        self.assertEqual(
            r.returncode, 0,
            "CLI failed: stdout=%s stderr=%s" % (r.stdout, r.stderr))
        with open(self._out_json) as f:
            doc = json.load(f)
        self.assertEqual(doc["schema"], SCHEMA)
        self.assertEqual(len(doc["per_tensor"]), 2)
        for rec in doc["per_tensor"]:
            self.assertEqual(rec["epsilon"], 0.0,
                             "expected 0 epsilon for %s, got %r"
                             % (rec["name"], rec["epsilon"]))
        # Per-layer sums must be 0.
        self.assertEqual(len(doc["per_layer"]), 1)
        self.assertEqual(doc["per_layer"][0]["total_epsilon"], 0.0)
        self.assertEqual(doc["per_layer"][0]["n_tensors"], 2)
        # Summary.
        self.assertEqual(doc["summary"]["n_tensors"], 2)
        self.assertEqual(doc["summary"]["n_layers"], 1)
        self.assertEqual(doc["summary"]["mean_epsilon"], 0.0)
        self.assertEqual(doc["summary"]["max_epsilon"], 0.0)
        # Missing lists must be empty.
        self.assertEqual(doc["missing"]["l1_only"], [])
        self.assertEqual(doc["missing"]["l15_only"], [])

    # ---- 2) Known delta -> epsilon matches the formula --------------

    def test_known_delta_matches_formula(self) -> None:
        """A known delta gives an epsilon that matches
        ``||delta||^2 / ||L15||^2`` to 4 decimal places."""
        d = os.path.join(self._tmp, "dir")
        os.makedirs(d, exist_ok=True)
        l15 = _deterministic_l15(4, 8, seed=2).astype("<f4")
        # Pick a small delta so the test is non-trivial but stable.
        delta = np.full_like(l15, 0.05, dtype="<f4")
        delta[0, 0] = 0.10
        _synthesize_pair(d, "blk.0.attn_q.weight", l15, delta=delta)

        expected_num = float(np.sum(delta.astype(np.float32) ** 2))
        expected_den = float(np.sum(l15.astype(np.float32) ** 2))
        expected_eps = expected_num / expected_den

        r = _run_cli(d, self._out_json, fmt="json")
        self.assertEqual(r.returncode, 0, "stderr=%s" % r.stderr)
        with open(self._out_json) as f:
            doc = json.load(f)
        self.assertEqual(len(doc["per_tensor"]), 1)
        got = doc["per_tensor"][0]["epsilon"]
        self.assertAlmostEqual(got, expected_eps, places=4,
                               msg="epsilon %r vs expected %r"
                               % (got, expected_eps))
        # Sanity: the reader reported F32 data of the right shape.
        s15 = reader.read_sidecar(
            os.path.join(d, "blk.0.attn_q.weight" + L15_SUFFIX))
        self.assertEqual(s15["data"].shape, (4, 8))
        self.assertEqual(s15["dtype_name"], "F32")

    # ---- 3) Multi-tensor dir -> per-layer aggregation ---------------

    def test_multi_tensor_aggregation(self) -> None:
        """A directory with several tensors across multiple layers
        must aggregate per-layer totals correctly."""
        d = os.path.join(self._tmp, "dir")
        os.makedirs(d, exist_ok=True)
        # Two tensors in blk.0, two tensors in blk.1, plus one
        # non-block tensor (token_embd). Each pair has a different
        # known delta so the per-layer total is the sum of the
        # individual epsilons.
        l15_a = _deterministic_l15(4, 8, seed=3)
        l15_b = _deterministic_l15(4, 8, seed=4)
        l15_c = _deterministic_l15(4, 8, seed=5)
        l15_d_ = _deterministic_l15(4, 8, seed=6)
        l15_e = _deterministic_l15(4, 8, seed=7)
        delta_a = np.full_like(l15_a, 0.02)
        delta_b = np.full_like(l15_b, 0.04)
        delta_c = np.full_like(l15_c, 0.06)
        delta_d = np.full_like(l15_d_, 0.08)
        delta_e = np.full_like(l15_e, 0.10)
        _synthesize_pair(d, "blk.0.attn_q.weight", l15_a, delta_a)
        _synthesize_pair(d, "blk.0.attn_k.weight", l15_b, delta_b)
        _synthesize_pair(d, "blk.1.attn_q.weight", l15_c, delta_c)
        _synthesize_pair(d, "blk.1.attn_k.weight", l15_d_, delta_d)
        _synthesize_pair(d, "token_embd.weight",   l15_e, delta_e)

        r = _run_cli(d, self._out_json, fmt="json")
        self.assertEqual(r.returncode, 0, "stderr=%s" % r.stderr)
        with open(self._out_json) as f:
            doc = json.load(f)
        self.assertEqual(len(doc["per_tensor"]), 5)
        # Per-tensor epsilons.
        def _eps(name: str) -> float:
            for rec in doc["per_tensor"]:
                if rec["name"] == name:
                    return rec["epsilon"]
            self.fail("missing tensor %s" % name)
            return 0.0
        eps_a = _eps("blk.0.attn_q.weight")
        eps_b = _eps("blk.0.attn_k.weight")
        eps_c = _eps("blk.1.attn_q.weight")
        eps_d = _eps("blk.1.attn_k.weight")
        eps_e = _eps("token_embd.weight")
        # Per-layer: blk.0, blk.1, token_embd.
        layer_totals = {rec["layer"]: rec["total_epsilon"]
                        for rec in doc["per_layer"]}
        layer_counts = {rec["layer"]: rec["n_tensors"]
                        for rec in doc["per_layer"]}
        self.assertAlmostEqual(layer_totals["blk.0"], eps_a + eps_b, places=5)
        self.assertAlmostEqual(layer_totals["blk.1"], eps_c + eps_d, places=5)
        self.assertAlmostEqual(layer_totals["token_embd"], eps_e, places=5)
        self.assertEqual(layer_counts["blk.0"], 2)
        self.assertEqual(layer_counts["blk.1"], 2)
        self.assertEqual(layer_counts["token_embd"], 1)
        # Summary.
        self.assertEqual(doc["summary"]["n_tensors"], 5)
        self.assertEqual(doc["summary"]["n_layers"], 3)
        all_eps = [eps_a, eps_b, eps_c, eps_d, eps_e]
        self.assertAlmostEqual(doc["summary"]["mean_epsilon"],
                               sum(all_eps) / len(all_eps), places=6)
        self.assertAlmostEqual(doc["summary"]["max_epsilon"],
                               max(all_eps), places=6)
        # Layers must be sorted: blk.0, blk.1, token_embd.
        layer_order = [rec["layer"] for rec in doc["per_layer"]]
        self.assertEqual(layer_order, ["blk.0", "blk.1", "token_embd"])

    # ---- 4) Missing L1.5 -> skip with warning, no crash -------------

    def test_missing_l15_is_skipped(self) -> None:
        """An L1 file without a matching L1.5 must be skipped
        with a warning on stderr; the tool must still exit 0 and
        emit a valid output document that lists the missing
        pair in ``missing.l1_only``."""
        d = os.path.join(self._tmp, "dir")
        os.makedirs(d, exist_ok=True)
        l15 = _deterministic_l15(4, 8, seed=8)
        # L1.5 only for one tensor; L1 only (no L1.5) for another.
        _synthesize_pair(d, "blk.0.attn_q.weight", l15, delta=np.zeros_like(l15))
        timing, kid, dc, oc = _meta(4)
        _write_v3(os.path.join(d, "blk.0.attn_k.weight" + L1_SUFFIX),
                  l15, timing, kid, dc, oc)

        r = _run_cli(d, self._out_json, fmt="json")
        self.assertEqual(r.returncode, 0,
                         "CLI failed: stdout=%s stderr=%s"
                         % (r.stdout, r.stderr))
        # The warning was printed to stderr.
        self.assertIn("missing L1.5", r.stderr)
        with open(self._out_json) as f:
            doc = json.load(f)
        # Only the matched pair made it into per_tensor.
        self.assertEqual(len(doc["per_tensor"]), 1)
        self.assertEqual(doc["per_tensor"][0]["name"], "blk.0.attn_q.weight")
        # The unmatched L1 path is recorded under missing.l1_only.
        self.assertEqual(len(doc["missing"]["l1_only"]), 1)
        self.assertTrue(doc["missing"]["l1_only"][0]
                        .endswith("blk.0.attn_k.weight" + L1_SUFFIX))
        self.assertEqual(doc["missing"]["l15_only"], [])

    # ---- 5) Missing L1 -> skip with warning, no crash ---------------

    def test_missing_l1_is_skipped(self) -> None:
        """An L1.5 file without a matching L1 must be skipped
        with a warning on stderr; the tool must still exit 0 and
        emit a valid output document that lists the missing
        pair in ``missing.l15_only``."""
        d = os.path.join(self._tmp, "dir")
        os.makedirs(d, exist_ok=True)
        l15 = _deterministic_l15(4, 8, seed=9)
        # L1 only for one tensor; L1.5 only (no L1) for another.
        _synthesize_pair(d, "blk.0.attn_q.weight", l15, delta=np.zeros_like(l15))
        timing, kid, dc, oc = _meta(4)
        _write_v3(os.path.join(d, "blk.0.attn_k.weight" + L15_SUFFIX),
                  l15, timing, kid, dc, oc)

        r = _run_cli(d, self._out_json, fmt="json")
        self.assertEqual(r.returncode, 0,
                         "CLI failed: stdout=%s stderr=%s"
                         % (r.stdout, r.stderr))
        self.assertIn("missing L1", r.stderr)
        with open(self._out_json) as f:
            doc = json.load(f)
        self.assertEqual(len(doc["per_tensor"]), 1)
        self.assertEqual(doc["per_tensor"][0]["name"], "blk.0.attn_q.weight")
        self.assertEqual(doc["missing"]["l15_only"],
                         [os.path.join(d,
                                       "blk.0.attn_k.weight" + L15_SUFFIX)])

    # ---- 6) Table output is greppable and consistent ----------------

    def test_table_output_is_greppable(self) -> None:
        """The ``--format table`` output must be human-readable,
        contain the per-tensor and per-layer sections, and be
        consistent with the JSON output."""
        d = os.path.join(self._tmp, "dir")
        os.makedirs(d, exist_ok=True)
        l15 = _deterministic_l15(4, 8, seed=10)
        _synthesize_pair(d, "blk.0.attn_q.weight", l15,
                         delta=np.full_like(l15, 0.03))
        r = _run_cli(d, self._out_table, fmt="table")
        self.assertEqual(r.returncode, 0, "stderr=%s" % r.stderr)
        with open(self._out_table) as f:
            text = f.read()
        for marker in ("## Per-tensor", "## Per-layer", "## Summary",
                       "blk.0", "blk.0.attn_q.weight"):
            self.assertIn(marker, text,
                          "table output missing %r\n--- table ---\n%s"
                          % (marker, text))

    # ---- 7) Empty dir -> empty-but-valid output --------------------

    def test_empty_dir_emits_empty_valid_output(self) -> None:
        """A sidecar dir with no files at all must still produce
        a valid output document (empty per_tensor / per_layer)."""
        d = os.path.join(self._tmp, "empty")
        os.makedirs(d, exist_ok=True)
        r = _run_cli(d, self._out_json, fmt="json")
        self.assertEqual(r.returncode, 0, "stderr=%s" % r.stderr)
        with open(self._out_json) as f:
            doc = json.load(f)
        self.assertEqual(doc["schema"], SCHEMA)
        self.assertEqual(doc["per_tensor"], [])
        self.assertEqual(doc["per_layer"], [])
        self.assertEqual(doc["summary"]["n_tensors"], 0)
        self.assertEqual(doc["summary"]["n_layers"], 0)


if __name__ == "__main__":
    # Run unittest and print a short PASS/FAIL summary, mirroring the
    # style of l3_sidecar_v3_smoke.py.
    import unittest as _u
    suite = _u.defaultTestLoader.loadTestsFromTestCase(PerLayerErrorTableTest)
    runner = _u.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    n_pass = result.testsRun - len(result.failures) - len(result.errors)
    n_fail = len(result.failures) + len(result.errors)
    print()
    print("Summary: %d passed, %d failed" % (n_pass, n_fail))
    sys.exit(0 if n_fail == 0 else 1)
