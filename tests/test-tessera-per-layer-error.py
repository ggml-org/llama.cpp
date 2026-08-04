#!/usr/bin/env python3
"""
test-tessera-per-layer-error.py

Smoke test for ``tools/tessera/per_layer_error_table.py``.

Builds synthetic v3 L1 / L1.5 sidecar files using the same
synthesize pattern as ``l3_sidecar_v3_smoke.py`` (no C++ build
required), then runs the CLI as a subprocess and verifies the
NDJSON output via ``tools/tessera/_analytical_io.py``.

Phase B (polars scout) refactor: the producer now emits per-tensor
NDJSON only; the per-layer / per-network rollup is the consumer's
job. The tests use polars group_by to verify the rollup matches
the expected formulas, so the test exercises the same patterns
the architect uses in practice.

Test cases:

  1. L1 == L1.5 (data block is bit-identical) -> epsilon == 0
     for every tensor. Per-layer sum is 0.
  2. L1 differs from L1.5 by a known delta ->
     epsilon matches ``||delta||^2 / ||L15||^2`` to 4 decimal
     places.
  3. Multi-tensor dir -> per-layer aggregation (computed by
     the test via polars) sums correctly across tensors within
     a layer, and the layer grouping matches the canonical
     ``blk.<N>`` prefix.
  4. Missing L1.5 file -> the tool skips the pair with a warning
     and still emits a valid NDJSON document with the matched
     records.
  5. Missing L1 file -> same as (4), the other direction.
  6. --print-table flag prints a human-readable table to stdout.
  7. Empty dir -> empty-but-valid NDJSON (zero rows, schema-pinned
     column types).

The test runs the CLI as a subprocess on the synthetic dir,
verifies exit 0, and checks the polars-parsed output. A short
PASS / FAIL summary is printed.
"""

import importlib.util
import os
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import polars as pl

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS_DIR)
TOOL_PATH = os.path.join(ROOT, "tools", "tessera",
                         "per_layer_error_table.py")
READER_PATH = os.path.join(ROOT, "tools", "tessera",
                           "l3_sidecar_v3_reader.py")
ANALYTICAL_IO_PATH = os.path.join(ROOT, "tools", "tessera",
                                  "_analytical_io.py")
sys.path.insert(0, os.path.dirname(READER_PATH))
sys.path.insert(0, os.path.dirname(ANALYTICAL_IO_PATH))
import l3_sidecar_v3_reader as reader  # noqa: E402
from _analytical_io import read_analytical, polars_schema  # noqa: E402

SCHEMA = "tessera.per-layer-error-record.v1"

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
# Polars helpers (the per-layer rollup the producer no longer emits).
# --------------------------------------------------------------------

def _per_layer_rollup(df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate per-tensor epsilons into per-layer totals. Mirrors
    the ``aggregate_per_layer`` function the producer used to ship
    internally; reproduced here as the canonical consumer pattern.

    Returns a DataFrame with columns: ``layer, total_epsilon,
    n_tensors`` sorted by canonical block index (blk.0, blk.1, ...,
    then non-block layers alphabetically).
    """
    valid = df.filter(~pl.col("epsilon_is_nan"))
    rolled = valid.group_by("layer").agg(
        pl.col("epsilon").sum().alias("total_epsilon"),
        pl.len().alias("n_tensors"),
    )

    def _sort_key(layer: str) -> tuple:
        import re
        m = re.match(r"^blk\.(\d+)$", layer)
        if m is not None:
            return (0, "%010d" % int(m.group(1)))
        return (1, layer)

    order = sorted(rolled["layer"].to_list(), key=_sort_key)
    return rolled.filter(pl.col("layer").is_in(order)).sort(
        pl.col("layer").cast(pl.Enum(order)))


# --------------------------------------------------------------------
# CLI runner.
# --------------------------------------------------------------------

def _run_cli(sidecar_dir: str, out_path: str,
             *extra_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, TOOL_PATH,
         "--sidecar-dir", sidecar_dir,
         "--out",       out_path,
         *extra_args],
        capture_output=True, text=True)


# --------------------------------------------------------------------
# Tests.
# --------------------------------------------------------------------

class PerLayerErrorTableTest(unittest.TestCase):
    """Smoke test for the per-layer error table tool."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="ple_test_")
        self._out_ndjson = os.path.join(self._tmp, "out.ndjson")

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

        r = _run_cli(d, self._out_ndjson)
        self.assertEqual(
            r.returncode, 0,
            "CLI failed: stdout=%s stderr=%s" % (r.stdout, r.stderr))
        df = read_analytical(self._out_ndjson, "per_layer_error")
        self.assertEqual(df.height, 2)
        for rec in df.iter_rows(named=True):
            self.assertEqual(rec["epsilon"], 0.0,
                             "expected 0 epsilon for %s, got %r"
                             % (rec["tensor"], rec["epsilon"]))
        # Per-layer rollup (consumer-side polars).
        rolled = _per_layer_rollup(df)
        self.assertEqual(rolled.height, 1)
        self.assertEqual(rolled["total_epsilon"].to_list(), [0.0])
        self.assertEqual(rolled["n_tensors"].to_list(), [2])

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

        r = _run_cli(d, self._out_ndjson)
        self.assertEqual(r.returncode, 0, "stderr=%s" % r.stderr)
        df = read_analytical(self._out_ndjson, "per_layer_error")
        self.assertEqual(df.height, 1)
        got = df["epsilon"].to_list()[0]
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
        must aggregate per-layer totals correctly. The rollup is
        computed by the test via polars; the producer emits only
        per-tensor records."""
        d = os.path.join(self._tmp, "dir")
        os.makedirs(d, exist_ok=True)
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

        r = _run_cli(d, self._out_ndjson)
        self.assertEqual(r.returncode, 0, "stderr=%s" % r.stderr)
        df = read_analytical(self._out_ndjson, "per_layer_error")
        self.assertEqual(df.height, 5)

        def _eps(name: str) -> float:
            mask = df["tensor"] == name
            return df.filter(mask)["epsilon"].to_list()[0]

        eps_a = _eps("blk.0.attn_q.weight")
        eps_b = _eps("blk.0.attn_k.weight")
        eps_c = _eps("blk.1.attn_q.weight")
        eps_d = _eps("blk.1.attn_k.weight")
        eps_e = _eps("token_embd.weight")

        # Per-layer rollup via polars (consumer pattern).
        rolled = _per_layer_rollup(df)
        self.assertEqual(rolled.height, 3)
        layer_totals = dict(zip(rolled["layer"].to_list(),
                                rolled["total_epsilon"].to_list()))
        layer_counts = dict(zip(rolled["layer"].to_list(),
                                rolled["n_tensors"].to_list()))
        self.assertAlmostEqual(layer_totals["blk.0"], eps_a + eps_b, places=5)
        self.assertAlmostEqual(layer_totals["blk.1"], eps_c + eps_d, places=5)
        self.assertAlmostEqual(layer_totals["token_embd"], eps_e, places=5)
        self.assertEqual(layer_counts["blk.0"], 2)
        self.assertEqual(layer_counts["blk.1"], 2)
        self.assertEqual(layer_counts["token_embd"], 1)
        # Layers must be sorted: blk.0, blk.1, token_embd.
        self.assertEqual(rolled["layer"].to_list(),
                         ["blk.0", "blk.1", "token_embd"])

    # ---- 4) Missing L1.5 -> skip with warning, no crash -------------

    def test_missing_l15_is_skipped(self) -> None:
        """An L1 file without a matching L1.5 must be skipped
        with a warning on stderr; the tool must still exit 0 and
        emit a valid NDJSON document that contains only the
        matched tensor."""
        d = os.path.join(self._tmp, "dir")
        os.makedirs(d, exist_ok=True)
        l15 = _deterministic_l15(4, 8, seed=8)
        _synthesize_pair(d, "blk.0.attn_q.weight", l15, delta=np.zeros_like(l15))
        timing, kid, dc, oc = _meta(4)
        _write_v3(os.path.join(d, "blk.0.attn_k.weight" + L1_SUFFIX),
                  l15, timing, kid, dc, oc)

        r = _run_cli(d, self._out_ndjson)
        self.assertEqual(r.returncode, 0,
                         "CLI failed: stdout=%s stderr=%s"
                         % (r.stdout, r.stderr))
        # The warning was printed to stderr.
        self.assertIn("missing L1.5", r.stderr)
        df = read_analytical(self._out_ndjson, "per_layer_error")
        # Only the matched pair made it into the per-tensor records.
        self.assertEqual(df.height, 1)
        self.assertEqual(df["tensor"].to_list(), ["blk.0.attn_q.weight"])

    # ---- 5) Missing L1 -> skip with warning, no crash ---------------

    def test_missing_l1_is_skipped(self) -> None:
        """An L1.5 file without a matching L1 must be skipped
        with a warning on stderr; the tool must still exit 0 and
        emit a valid NDJSON document that contains only the
        matched tensor."""
        d = os.path.join(self._tmp, "dir")
        os.makedirs(d, exist_ok=True)
        l15 = _deterministic_l15(4, 8, seed=9)
        _synthesize_pair(d, "blk.0.attn_q.weight", l15, delta=np.zeros_like(l15))
        timing, kid, dc, oc = _meta(4)
        _write_v3(os.path.join(d, "blk.0.attn_k.weight" + L15_SUFFIX),
                  l15, timing, kid, dc, oc)

        r = _run_cli(d, self._out_ndjson)
        self.assertEqual(r.returncode, 0,
                         "CLI failed: stdout=%s stderr=%s"
                         % (r.stdout, r.stderr))
        self.assertIn("missing L1", r.stderr)
        df = read_analytical(self._out_ndjson, "per_layer_error")
        self.assertEqual(df.height, 1)
        self.assertEqual(df["tensor"].to_list(), ["blk.0.attn_q.weight"])

    # ---- 6) --print-table prints a human-readable table to stdout --

    def test_print_table_emits_human_readable_table(self) -> None:
        """The ``--print-table`` flag must print a human-readable
        per-tensor table to stdout. (The previous --format=table
        file output is gone; the table is now a stdout-only
        diagnostic that mirrors the NDJSON file output.)"""
        d = os.path.join(self._tmp, "dir")
        os.makedirs(d, exist_ok=True)
        l15 = _deterministic_l15(4, 8, seed=10)
        _synthesize_pair(d, "blk.0.attn_q.weight", l15,
                         delta=np.full_like(l15, 0.03))
        r = _run_cli(d, self._out_ndjson, "--print-table")
        self.assertEqual(r.returncode, 0, "stderr=%s" % r.stderr)
        for marker in ("# Tessera per-tensor",
                       "blk.0.attn_q.weight",
                       "epsilon"):
            self.assertIn(marker, r.stdout,
                          "table stdout missing %r\n--- stdout ---\n%s"
                          % (marker, r.stdout))
        # The NDJSON file is also written.
        df = read_analytical(self._out_ndjson, "per_layer_error")
        self.assertEqual(df.height, 1)

    # ---- 7) Empty dir -> empty-but-valid NDJSON (zero rows) --------

    def test_empty_dir_emits_empty_valid_output(self) -> None:
        """A sidecar dir with no files at all must still produce
        a valid NDJSON document. The polars reader returns a
        zero-row DataFrame with the schema-pinned column types."""
        d = os.path.join(self._tmp, "empty")
        os.makedirs(d, exist_ok=True)
        r = _run_cli(d, self._out_ndjson)
        self.assertEqual(r.returncode, 0, "stderr=%s" % r.stderr)
        df = read_analytical(self._out_ndjson, "per_layer_error")
        self.assertEqual(df.height, 0)
        # Schema-pinned types are still present.
        expected_types = polars_schema("per_layer_error")
        for col, dtype in expected_types.items():
            self.assertIn(col, df.columns,
                          "missing column %r in empty NDJSON" % col)
            self.assertEqual(df.schema[col], dtype,
                             "wrong dtype for column %r" % col)


if __name__ == "__main__":
    # Run unittest and print a short PASS/FAIL summary, mirroring the
    # style of l3_sidecar_v3_smoke.py.
    import unittest as _u
    suite = _u.defaultTestLoader.loadTestsFromTestCase(PerLayerErrorTableTest)
    runner = _u.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
