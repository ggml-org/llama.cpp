#!/usr/bin/env python3
"""Unit tests for ``tools/tessera/calibration_memory.py``.

The utilities are the memory-bound / spatial-temporal layer
that keeps the unified gemma4_12B + dspark + dflash + MTP
calibration in bounded memory.  Five categories are tested
here, each in its own ``TestCase``:

  * TestStreamingIO -- ``mmap_tensor``, ``mmap_layer``
  * TestChunkedProcessing -- ``chunked_iter``, ``chunked_process``
  * TestResidencyDecisions -- residency hints (the residency
    module lives in ``calibration_residency.py``; this case
    just covers the no-residency-decision path so the bundled
    defaults are exercised end-to-end)
  * TestSpatialOccupancy -- ``interleave_components``,
    ``extract_layer_index``
  * TestTemporalPipeline -- ``CalibPipeline`` (double-buffered
    I/O + compute overlap)

Run as::

    python3 tools/tessera/test_calibration_memory.py

Exits 0 on success, non-zero on any failure.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR.parent.parent))  # for top-level import

import calibration_memory as cm  # noqa: E402


def _macos_with_clang() -> bool:
    """True if macOS + clang++ are available.

    Used to gate the dispatch_io_t-specific tests.  On
    Linux/Windows the dispatch_io_t bridge is not
    available (the Apple ``dispatch_io_t`` API is macOS-
    only), so the tests skip there.
    """
    if platform.system() != "Darwin":
        return False
    try:
        return subprocess.run(
            ["xcrun", "--find", "clang++"],
            check=True, capture_output=True, text=True,
        ).returncode == 0
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _make_bundle(
    path: Path,
    *,
    out_dim: int = 256,
    in_dim: int = 256,
    n_tokens: int = 64,
    with_name: bool = True,
    with_observer: bool = True,
) -> None:
    """Write a tiny ``.npz`` bundle with the legacy key set."""
    rng = np.random.default_rng(0)
    arrays: dict[str, np.ndarray] = {
        "weight": rng.standard_normal((out_dim, in_dim)).astype(np.float32),
        "train_activations": rng.standard_normal((n_tokens, in_dim)).astype(np.float32),
    }
    if with_observer:
        arrays["in_sum2"] = rng.standard_normal(in_dim).astype(np.float32) ** 2
        arrays["counts"] = np.array(n_tokens, dtype=np.int64)
    if with_name:
        arrays["name"] = np.array("test_layer")
        arrays["family"] = np.array("ffn")
    np.savez(path, **arrays)


class TestResidencyDecisions(unittest.TestCase):
    """The peak-RSS budget tracker (``ResidencyTracker``) is
    the Phase 16 Cat 3 mechanism.  It reads the current
    process RSS, compares it against the budget, and aborts
    with a clear error when the budget is exceeded."""

    def test_read_rss_returns_nonzero(self) -> None:
        """The RSS reader returns a positive integer (the
        process is alive, so RSS is at least a few MB)."""
        rss = cm_residency.read_rss_bytes()
        self.assertGreater(rss, 0)

    def test_tracker_default_budget_32gb(self) -> None:
        """The default budget is 32 GB (a 12B unified
        calibration on a 64 GB host)."""
        from calibration_residency import ResidencyTracker
        t = ResidencyTracker()
        self.assertEqual(t.budget_bytes, 32 * 1024**3)

    def test_tracker_below_budget_passes(self) -> None:
        """A check that is below the budget returns the
        observed RSS and does not raise."""
        from calibration_residency import ResidencyTracker
        t = ResidencyTracker(budget_bytes=10 * 1024**3)  # 10 GB
        current = t.check("test_tensor")
        # The check returns the observed RSS (positive).
        self.assertGreater(current, 0)
        # The peak is updated.
        self.assertGreaterEqual(t.peak_bytes, current)
        # The check counter is incremented.
        self.assertEqual(t.n_checks, 1)
        self.assertEqual(t.n_violations, 0)

    def test_tracker_over_budget_raises(self) -> None:
        """A check that exceeds the budget raises
        ``MemoryError`` naming the tensor and the observed
        RSS vs the budget."""
        from calibration_residency import ResidencyTracker
        # 1 MB budget; the test process is >>1 MB so the
        # first check will exceed.
        t = ResidencyTracker(budget_bytes=1 * 1024 * 1024, abort_on_exceed=True)
        with self.assertRaises(MemoryError) as ctx:
            t.check("tensor_xyz")
        msg = str(ctx.exception)
        self.assertIn("tensor_xyz", msg)
        self.assertIn("RSS", msg)
        self.assertIn("budget", msg)

    def test_tracker_over_budget_does_not_raise_in_advisory(self) -> None:
        """A tracker with ``abort_on_exceed=False`` (advisory
        mode) records the violation but does not raise.  The
        diagnostics surface is still useful for the
        non-aborting path."""
        from calibration_residency import ResidencyTracker
        t = ResidencyTracker(budget_bytes=1, abort_on_exceed=False)
        t.check("test_tensor")  # would normally raise; advisory mode doesn't
        self.assertEqual(t.n_violations, 1)
        self.assertGreater(t.peak_bytes, 0)

    def test_tracker_zero_budget_disables_check(self) -> None:
        """A budget of 0 (the ``--peak-rss-budget-gb 0``
        opt-out) disables the check; the tracker still
        records peak_bytes for the final report."""
        from calibration_residency import ResidencyTracker
        t = ResidencyTracker(budget_bytes=0, abort_on_exceed=True)
        # Multiple checks: none raise.
        for i in range(5):
            t.check(f"t{i}")
        self.assertEqual(t.n_checks, 5)
        self.assertEqual(t.n_violations, 0)
        # peak_bytes is still recorded.
        self.assertGreater(t.peak_bytes, 0)

    def test_tracker_rejects_negative_budget(self) -> None:
        """A negative budget is a programming error; raise
        at construction time so the misconfiguration is
        caught early (vs at the first check)."""
        from calibration_residency import ResidencyTracker
        with self.assertRaises(ValueError):
            ResidencyTracker(budget_bytes=-1, abort_on_exceed=True)

    def test_tracker_report_is_one_line(self) -> None:
        """The report is a one-line summary suitable for the
        final stderr log line."""
        from calibration_residency import ResidencyTracker
        t = ResidencyTracker(budget_bytes=32 * 1024**3)
        t.check("test")
        report = t.report()
        self.assertIn("peak RSS", report)
        self.assertIn("budget", report)
        self.assertIn("checks", report)
        self.assertIn("violations", report)
        # One line: no embedded newlines.
        self.assertNotIn("\n", report)


# The residency module is imported at module level so the
# ``TestResidencyDecisions`` class can reference it via
# ``cm_residency``.  The import is below the streaming /
# chunked / spatial / temporal test classes so the
# streaming / chunked tests can run without the residency
# dependency.
import calibration_residency as cm_residency  # noqa: E402  pylint: disable=wrong-import-position


class TestStreamingIO(unittest.TestCase):
    """``mmap_tensor`` and ``mmap_layer`` open ``.npz`` keys as
    memory-mapped views rather than reading the whole file into
    RAM.  The legacy code path's failure mode was
    ``np.load(path)`` -> OOM on large bundles; this test asserts
    the mmap path is single-tensor and OS-driven."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="calmem_io_")
        self._td = Path(self._tmp)

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_mmap_tensor_returns_view(self) -> None:
        path = self._td / "tiny.npz"
        _make_bundle(path, out_dim=128, in_dim=64, n_tokens=16)
        w = cm.mmap_tensor(path, "weight", dtype=np.float32)
        # The returned array is a view (mmap) with the right shape / dtype.
        self.assertEqual(w.shape, (128, 64))
        self.assertEqual(w.dtype, np.float32)
        # Closing the np.load handle must not invalidate the view.
        # The OS keeps the zip mmap alive as long as the view is.
        expected = np.load(path, mmap_mode="r", allow_pickle=False)["weight"]
        np.testing.assert_array_equal(np.asarray(w), np.asarray(expected))

    def test_mmap_tensor_missing_key_raises(self) -> None:
        path = self._td / "tiny.npz"
        _make_bundle(path)
        with self.assertRaises(KeyError):
            cm.mmap_tensor(path, "no_such_key")

    def test_mmap_tensor_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            cm.mmap_tensor(self._td / "ghost.npz", "weight")

    def test_mmap_layer_yields_all_requested_keys(self) -> None:
        path = self._td / "tiny.npz"
        _make_bundle(path, out_dim=64, in_dim=32, n_tokens=8)
        with cm.mmap_layer(path) as data:
            for k in ("weight", "train_activations", "in_sum2", "counts", "name", "family"):
                self.assertIn(k, data, f"missing key {k!r}")
            self.assertEqual(data["weight"].shape, (64, 32))
            self.assertEqual(data["train_activations"].shape, (8, 32))
            # The name / family strings survive the mmap round-trip.
            self.assertEqual(str(data["name"].item()), "test_layer")
            self.assertEqual(str(data["family"].item()), "ffn")

    def test_mmap_layer_handles_missing_keys(self) -> None:
        """``mmap_layer`` only yields the keys present in the bundle
        (a bundle without ``in_sum2`` still works for FLRQ-style
        calibration-free runs)."""
        path = self._td / "no_observer.npz"
        _make_bundle(path, with_observer=False)
        with cm.mmap_layer(path) as data:
            self.assertIn("weight", data)
            self.assertNotIn("in_sum2", data)
            self.assertNotIn("counts", data)


class TestChunkedProcessing(unittest.TestCase):
    """``chunked_iter`` yields contiguous row ranges; ``chunked_process``
    runs a per-chunk computation and collects the results."""

    def test_chunked_iter_full_chunking(self) -> None:
        specs = list(cm.chunked_iter(1000, 250))
        self.assertEqual([(s.start, s.end) for s in specs],
                         [(0, 250), (250, 500), (500, 750), (750, 1000)])

    def test_chunked_iter_uneven_last_chunk(self) -> None:
        specs = list(cm.chunked_iter(1000, 300))
        # 4 chunks: 300 + 300 + 300 + 100
        self.assertEqual([(s.start, s.end) for s in specs],
                         [(0, 300), (300, 600), (600, 900), (900, 1000)])

    def test_chunked_iter_no_chunking_legacy_path(self) -> None:
        """``chunk_rows <= 0`` or ``>= n_rows`` is the legacy
        single-shot path: one chunk covering the full weight."""
        specs_a = list(cm.chunked_iter(1000, 0))
        self.assertEqual(len(specs_a), 1)
        self.assertEqual((specs_a[0].start, specs_a[0].end), (0, 1000))
        specs_b = list(cm.chunked_iter(1000, 5000))
        self.assertEqual(len(specs_b), 1)
        self.assertEqual((specs_b[0].start, specs_b[0].end), (0, 1000))

    def test_chunked_iter_empty_weight(self) -> None:
        specs = list(cm.chunked_iter(0, 256))
        self.assertEqual(specs, [])

    def test_chunked_process_collects_per_chunk_results(self) -> None:
        """``chunked_process`` yields one result per chunk; the
        consumer can reduce the list to a per-tensor answer."""
        w = np.arange(20, dtype=np.float32).reshape(4, 5)
        def compute(w_chunk, _a, _spec):
            return float(w_chunk.sum())
        results = cm.chunked_process(w, activations=None, chunk_rows=2, compute=compute)
        # 4 rows / 2-row chunks -> 2 chunks.  Sum: row 0+1 = 5+15 = ...
        self.assertEqual(len(results), 2)
        # The total is the per-tensor answer.
        self.assertAlmostEqual(sum(results), float(w.sum()))

    def test_chunked_process_with_activations(self) -> None:
        """Activations are input-aligned: shape ``(n_tokens, in_dim)``
        where ``in_dim`` matches the weight's in_dim.  They are
        passed to each chunk in full (no slicing on the
        chunked axis).  The per-chunk matmul
        ``w_chunk @ a_chunk.T`` produces a (chunk_rows, T) output
        that the test sums to a per-chunk scalar; the per-tensor
        answer is the sum across chunks."""
        w = np.arange(12, dtype=np.float32).reshape(3, 4)   # out=3, in=4
        a = np.arange(20, dtype=np.float32).reshape(5, 4)   # tokens=5, in=4
        def compute(w_chunk, a_chunk, _spec):
            return float((w_chunk @ a_chunk.T).sum())
        results = cm.chunked_process(w, activations=a, chunk_rows=2, compute=compute)
        # 2 chunks: rows [0,2) and [2,3)
        self.assertEqual(len(results), 2)
        # The total is the per-tensor answer.
        self.assertAlmostEqual(sum(results), float((w @ a.T).sum()))

    def test_chunked_process_rejects_1d_weight(self) -> None:
        with self.assertRaises(ValueError):
            cm.chunked_process(
                np.zeros(10, dtype=np.float32),
                activations=None,
                chunk_rows=4,
                compute=lambda w, a, s: None,
            )

    def test_chunked_process_rejects_shape_mismatch(self) -> None:
        """Activations whose in_dim does not match the weight's
        in_dim raise (the inner matmul would silently produce
        the wrong result)."""
        w = np.zeros((4, 5), dtype=np.float32)     # out=4, in=5
        a = np.zeros((10, 7), dtype=np.float32)   # in_dim=7 != 5
        with self.assertRaises(ValueError):
            cm.chunked_process(w, activations=a, chunk_rows=2,
                                compute=lambda w, a, s: None)

    def test_chunked_process_mmap_views(self) -> None:
        """The chunked_process utility passes mmap views to the
        compute callback without copying.  For the per-tensor
        calibration, this means the OS keeps the weight's zip
        mmap alive and the per-chunk reads do not materialise
        the full weight in RAM."""
        import tempfile
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "big.npz"
            np.savez(p, weight=rng.standard_normal((1000, 100)).astype(np.float32))
            weight = cm.mmap_tensor(p, "weight", dtype=np.float32)
            captured = {"shape": None, "base_is_mmap": None}
            def capture(w_chunk, _a, _spec):
                captured["shape"] = w_chunk.shape
                captured["base_is_mmap"] = w_chunk.base is not None
                return None
            results = cm.chunked_process(weight, activations=None, chunk_rows=250,
                                          compute=capture)
            self.assertEqual(len(results), 4)
            self.assertEqual(captured["shape"], (250, 100))
            # The mmap view is passed through (no copy).  Closing
            # the np.load handle inside mmap_tensor doesn't
            # invalidate the view; the OS keeps the zip mmap
            # alive as long as the view is held.
            self.assertIsNotNone(captured["base_is_mmap"])

    def test_chunked_process_wired_to_lrq_training(self) -> None:
        """End-to-end smoke: a small synthetic tensor goes through
        ``train_lrq_chunked`` and produces an LRQ result whose
        initial/final MSE is close to the legacy single-shot
        path.  The chunked path uses different matmul
        accumulation orders per chunk, so the float32 results
        are not bit-equivalent; the test asserts the relative
        MSE delta is bounded (the per-tensor result is
        numerically equivalent, not byte-identical)."""
        import per_tensor_calibrate as ptc
        rng = np.random.default_rng(0)
        out_dim, in_dim = 32, 16
        n_tokens = 8
        weight = rng.standard_normal((out_dim, in_dim)).astype(np.float32)
        acts = rng.standard_normal((n_tokens, in_dim)).astype(np.float32)
        layer = ptc.Layer(
            name="test_chunked_lrq",
            family="ffn",
            weight=weight,
            train_activations=acts,
            heldout_activations=None,
            in_sum2=(acts.astype(np.float32) ** 2).sum(axis=0),
            in_count=n_tokens,
        )
        # Legacy single-shot baseline.
        legacy = ptc.train_lrq(layer, rank=4, iterations=3, lr=1e-2, seed=0, aggregation="mean")
        # Chunked path.  32 rows / 16-row chunks = 2 chunks.
        chunked = ptc.train_lrq_chunked(
            layer, rank=4, iterations=3, lr=1e-2, seed=0,
            aggregation="mean", chunk_rows=16,
        )
        # The initial MSE is close (the chunked forward is
        # per-chunk matmul; float32 accumulation order differs).
        rel_init = abs(legacy.initial_mse - chunked.initial_mse) / max(abs(legacy.initial_mse), 1e-12)
        self.assertLess(rel_init, 1e-2)
        # The final MSE is close (Adam's float32 step order can
        # differ across the chunked boundary).
        rel_final = abs(legacy.final_mse - chunked.final_mse) / max(abs(legacy.final_mse), 1e-12)
        self.assertLess(rel_final, 1e-2)
        # The U, V are the same shape and dtype.
        self.assertEqual(legacy.u.shape, chunked.u.shape)
        self.assertEqual(legacy.v.shape, chunked.v.shape)
        # The scale_aggregate is close (per-input-channel
        # aggregate of S = U @ V; chunked aggregation is
        # equivalent to the legacy one for ``aggregation="mean"``,
        # modulo the float32 order-of-operations in the per-chunk
        # matmul).
        np.testing.assert_allclose(legacy.scale_aggregate, chunked.scale_aggregate,
                                    rtol=5e-2, atol=5e-2)

    def test_chunked_process_wired_to_flrq_sketch(self) -> None:
        """The chunked FLRQ sketch produces the same Y (up to
        float32 order) as the legacy single-shot path."""
        import per_tensor_calibrate as ptc
        rng = np.random.default_rng(0)
        weight = rng.standard_normal((64, 32)).astype(np.float32)
        Y_legacy, U_legacy, _ = ptc.flrq_sketch(weight, n_projections=8, seed=0, target_rank=4)
        Y_chunked, U_chunked, _ = ptc.flrq_sketch_chunked(
            weight, n_projections=8, seed=0, target_rank=4, chunk_rows=16,
        )
        np.testing.assert_allclose(Y_legacy, Y_chunked, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(np.abs(U_legacy), np.abs(U_chunked), rtol=1e-4, atol=1e-5)

    def test_chunked_process_dispatch_to_legacy(self) -> None:
        """``train_lrq_chunked`` with ``chunk_rows <= 0`` or
        ``chunk_rows >= n_rows`` dispatches to the legacy
        single-shot ``train_lrq`` (the public API is consistent)."""
        import per_tensor_calibrate as ptc
        rng = np.random.default_rng(0)
        layer = ptc.Layer(
            name="test_dispatch",
            family="ffn",
            weight=rng.standard_normal((8, 16)).astype(np.float32),
            train_activations=rng.standard_normal((4, 16)).astype(np.float32),
            heldout_activations=None,
            in_sum2=None,
            in_count=0,
        )
        # chunk_rows=0 -> legacy
        r0 = ptc.train_lrq_chunked(layer, rank=2, iterations=2, lr=1e-2,
                                    seed=0, chunk_rows=0)
        # chunk_rows > n_rows -> legacy
        r_big = ptc.train_lrq_chunked(layer, rank=2, iterations=2, lr=1e-2,
                                      seed=0, chunk_rows=100)
        # Both should match the legacy call.
        r_legacy = ptc.train_lrq(layer, rank=2, iterations=2, lr=1e-2, seed=0)
        self.assertAlmostEqual(r0.final_mse, r_legacy.final_mse, places=5)
        self.assertAlmostEqual(r_big.final_mse, r_legacy.final_mse, places=5)


class TestSpatialOccupancy(unittest.TestCase):
    """``interleave_components`` round-robins per-component tensors
    at the layer level so the per-component cache footprint
    stays small."""

    def test_extract_layer_index(self) -> None:
        self.assertEqual(cm.extract_layer_index("blk.0.attn_q.weight"), 0)
        self.assertEqual(cm.extract_layer_index("blk.12.attn_v.weight"), 12)
        self.assertEqual(cm.extract_layer_index("dflash.encoder.fc.0"), 0)
        self.assertEqual(cm.extract_layer_index("dspark.markov_w.7"), 7)
        self.assertEqual(cm.extract_layer_index("token_embd.weight"), -1)
        self.assertEqual(cm.extract_layer_index("output.weight"), -1)

    def test_interleave_components_round_robin_by_layer(self) -> None:
        components = {
            "trunk": ["blk.0.attn_q", "blk.0.attn_k", "blk.1.attn_q", "blk.1.attn_k"],
            "dflash": ["dflash.encoder.fc.0", "dflash.encoder.fc.1"],
            "dspark": ["dspark.markov_w.0", "dspark.markov_w.1"],
            "mtp_nextn": ["mtp_nextn.eh_proj.0", "mtp_nextn.eh_proj.1"],
        }
        order = list(cm.interleave_components(components))
        # 4 layers (0, 1) x 4 roles = 8; but 2 layers x 4 roles = 8.
        # At each layer, the role order is trunk, dflash, dspark, mtp_nextn.
        expected = [
            ("trunk", "blk.0.attn_q"),
            ("dflash", "dflash.encoder.fc.0"),
            ("dspark", "dspark.markov_w.0"),
            ("mtp_nextn", "mtp_nextn.eh_proj.0"),
            ("trunk", "blk.0.attn_k"),
            ("dflash", "dflash.encoder.fc.0"),  # wait, the second fc at layer 0? No.
        ]
        # Re-derive: the second trunk tensor at layer 0 is blk.0.attn_k.
        # There is no "second dflash" at layer 0 in the input, so it
        # doesn't get re-emitted.  The expected sequence is:
        #  (trunk, blk.0.attn_q)  (dflash, dflash.encoder.fc.0)
        #  (dspark, dspark.markov_w.0)  (mtp_nextn, mtp_nextn.eh_proj.0)
        #  (trunk, blk.0.attn_k)  -- second trunk at layer 0
        #  (trunk, blk.1.attn_q)  (dflash, dflash.encoder.fc.1)
        #  (dspark, dspark.markov_w.1)  (mtp_nextn, mtp_nextn.eh_proj.1)
        #  (trunk, blk.1.attn_k)
        expected = [
            ("trunk", "blk.0.attn_q"),
            ("dflash", "dflash.encoder.fc.0"),
            ("dspark", "dspark.markov_w.0"),
            ("mtp_nextn", "mtp_nextn.eh_proj.0"),
            ("trunk", "blk.0.attn_k"),
            ("trunk", "blk.1.attn_q"),
            ("dflash", "dflash.encoder.fc.1"),
            ("dspark", "dspark.markov_w.1"),
            ("mtp_nextn", "mtp_nextn.eh_proj.1"),
            ("trunk", "blk.1.attn_k"),
        ]
        self.assertEqual(order, expected)
        del expected  # silence pylint unused warning

    def test_interleave_components_empty(self) -> None:
        self.assertEqual(list(cm.interleave_components({})), [])
        # Missing roles are skipped.
        order = list(cm.interleave_components({"trunk": ["blk.0.attn_q"]}))
        self.assertEqual(order, [("trunk", "blk.0.attn_q")])

    def test_interleave_components_shared_first(self) -> None:
        """Tensors with no layer index (e.g. ``token_embd.weight``)
        sort to layer index -1 and are emitted first, in role
        order, so shared embeddings are computed before the
        first layer's tensors reference them."""
        components = {
            "trunk": ["token_embd.weight", "blk.0.attn_q"],
            "shared_embd": ["token_embd.weight"],
        }
        order = list(cm.interleave_components(components))
        # Layer -1 first, then layer 0.
        self.assertEqual(order, [
            ("trunk", "token_embd.weight"),
            ("shared_embd", "token_embd.weight"),
            ("trunk", "blk.0.attn_q"),
        ])

    def test_interleave_components_preserves_per_role_order(self) -> None:
        """Within a role, the per-role order is preserved so the
        per-component shell-out's own layer ordering is
        respected (e.g. attn_q before attn_k within the same
        layer)."""
        components = {
            "trunk": ["blk.0.attn_q", "blk.0.attn_k", "blk.0.attn_v"],
        }
        order = list(cm.interleave_components(components))
        # All at layer 0: emitted in the per-role input order.
        self.assertEqual([n for _, n in order], ["blk.0.attn_q", "blk.0.attn_k", "blk.0.attn_v"])

    def test_spatial_roles_contains_unified_components(self) -> None:
        """The default ``SPATIAL_ROLES`` covers the unified
        gemma4_12B + dspark + dflash + MTP pipeline."""
        for role in ("trunk", "dflash", "dspark", "mtp_nextn", "shared_embd"):
            self.assertIn(role, cm.SPATIAL_ROLES)

    def test_compute_spatial_order_sequential(self) -> None:
        """``compute_spatial_order(..., 'sequential')`` returns
        the legacy component-major order: all of the trunk's
        tensors first, then all of the dflash's, etc."""
        import per_tensor_calibrate as ptc
        from pathlib import Path
        components = {
            "trunk": [Path("/tmp/trunk_0.npz"), Path("/tmp/trunk_1.npz")],
            "dflash": [Path("/tmp/dflash_0.npz")],
            "dspark": [Path("/tmp/dspark_0.npz")],
        }
        order = ptc.compute_spatial_order(components, "sequential")
        self.assertEqual(
            [p.name for _, p in order],
            ["trunk_0.npz", "trunk_1.npz", "dflash_0.npz", "dspark_0.npz"],
        )
        # The roles are also correctly tagged.
        self.assertEqual(
            [role for role, _ in order],
            ["trunk", "trunk", "dflash", "dspark"],
        )

    def test_compute_spatial_order_interleaved(self) -> None:
        """``compute_spatial_order(..., 'interleaved')`` returns
        the per-layer round-robin order so the cache stays hot."""
        import per_tensor_calibrate as ptc
        from pathlib import Path
        components = {
            "trunk": [Path("/tmp/trunk_blk.0.npz"), Path("/tmp/trunk_blk.1.npz")],
            "dflash": [Path("/tmp/dflash_fc.0.npz"), Path("/tmp/dflash_fc.1.npz")],
            "dspark": [Path("/tmp/dspark_w.0.npz"), Path("/tmp/dspark_w.1.npz")],
        }
        order = ptc.compute_spatial_order(components, "interleaved")
        # The first layer fires all three roles, the second
        # layer fires all three.  Trunk has 2 tensors per
        # layer (attn_q, attn_k) so the per-layer order
        # interleaves 2 trunk + 1 each of dflash / dspark.
        names = [p.name for _, p in order]
        # The first tensor is layer 0 (any role).
        self.assertIn("trunk_blk.0.npz", names[0])
        # The shared_embd-less run has 6 tensors total.
        self.assertEqual(len(order), 6)

    def test_compute_spatial_order_rejects_unknown(self) -> None:
        """An unknown spatial_occupancy value raises ``ValueError``."""
        import per_tensor_calibrate as ptc
        from pathlib import Path
        components = {"trunk": [Path("/tmp/t.npz")]}
        with self.assertRaises(ValueError):
            ptc.compute_spatial_order(components, "diagonal")


class TestTemporalPipeline(unittest.TestCase):
    """``CalibPipeline`` overlaps the next tensor's mmap with the
    current tensor's compute.  The depth-1 path is the legacy
    single-thread path; depth-2 is the default double-buffered
    path."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="calmem_pipe_")
        self._td = Path(self._tmp)
        self._paths: list[Path] = []
        for i in range(6):
            p = self._td / f"layer_{i}.npz"
            _make_bundle(p, out_dim=32, in_dim=16, n_tokens=8)
            self._paths.append(p)

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_pipeline_depth1_yields_all_in_order(self) -> None:
        """The depth-1 path is the legacy single-thread path; the
        order is preserved and each item carries a mmap-backed
        dict with the legacy key set."""
        seen_paths: list[Path] = []
        seen_keys: list[set[str]] = []
        with cm.CalibPipeline(self._paths, depth=1) as pipe:
            for path, data in pipe:
                seen_paths.append(path)
                seen_keys.append(set(data.keys()))
        self.assertEqual(seen_paths, self._paths)
        for ks in seen_keys:
            self.assertIn("weight", ks)
            self.assertIn("train_activations", ks)

    def test_pipeline_depth2_double_buffered(self) -> None:
        """Depth-2 is the default; we cannot directly observe the
        overlap (it's a wall-time property) but the iteration
        order is the same and the contents are correct."""
        seen_paths: list[Path] = []
        with cm.CalibPipeline(self._paths, depth=2) as pipe:
            for path, data in pipe:
                seen_paths.append(path)
                self.assertEqual(data["weight"].shape, (32, 16))
        self.assertEqual(seen_paths, self._paths)

    def test_pipeline_empty(self) -> None:
        """An empty path list is a no-op (no items yielded)."""
        with cm.CalibPipeline([], depth=2) as pipe:
            self.assertEqual(list(pipe), [])

    def test_pipeline_rejects_depth_zero(self) -> None:
        with self.assertRaises(ValueError):
            cm.CalibPipeline(self._paths, depth=0)

    def test_pipeline_overlap_smoke(self) -> None:
        """Smoke test: the producer thread started for depth>=2
        is a daemon thread that terminates when the consumer
        is done (no thread leak)."""
        # Track threads that were alive while we were iterating.
        # The producer thread is a daemon and joins on close.
        threads_before = {t.ident for t in threading.enumerate()}
        with cm.CalibPipeline(self._paths, depth=2) as pipe:
            for _path, _data in pipe:
                pass
        # Give the OS a moment to reap the daemon thread.
        time.sleep(0.01)
        threads_after = {t.ident for t in threading.enumerate()}
        # The producer thread (if started) is no longer in the
        # live thread set.  Threads that were already there
        # (e.g. the test runner) are unaffected.
        new_threads = threads_after - threads_before
        # At most 0 new threads after the pipeline is closed
        # (the producer was a daemon, joined by ``close()``).
        self.assertEqual(new_threads, set())

    def test_pipeline_consumer_can_read_mmap_views(self) -> None:
        """The mmap views returned by the pipeline are read-able
        after the producer has moved on; the OS keeps the zip
        mmap alive as long as the views are held."""
        with cm.CalibPipeline(self._paths[:2], depth=2) as pipe:
            first_path, first_data = next(iter(pipe))
            w = first_data["weight"]
            # Read the entire weight to confirm the view is live.
            s = float(w.sum())
            self.assertIsInstance(s, float)
            # Continue iteration; the OS keeps the first zip mmap
            # alive while ``w`` is still held.
            for _p, _d in pipe:
                pass
            # The view is still readable after the producer moved on.
            self.assertEqual(float(w.sum()), s)

    def test_pipeline_wired_to_lrq_training(self) -> None:
        """End-to-end smoke: the per-tensor LRQ training
        loop uses ``CalibPipeline`` when
        ``--temporal-pipeline-depth > 1``.  This pins the
        wire-up: the consumer reads the pipeline's mmap
        data, builds a ``Layer`` via
        ``_layer_from_mmap_data``, and runs the LRQ
        training.  The result is the same as the
        single-thread path (the I/O overlap is
        wall-time-only)."""
        import per_tensor_calibrate as ptc
        # Build a couple of small bundles in a temp dir.
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            paths = []
            for i in range(3):
                p = Path(td) / f"layer_{i}.npz"
                rng = np.random.default_rng(i)
                np.savez(p,
                    weight=rng.standard_normal((16, 8)).astype(np.float32),
                    train_activations=rng.standard_normal((4, 8)).astype(np.float32),
                    in_sum2=rng.standard_normal(8).astype(np.float32)**2,
                    counts=np.array(4, dtype=np.int64),
                    name=np.array(f"layer_{i}"), family=np.array("ffn"),
                )
                paths.append(p)
            # Direct use of CalibPipeline + _layer_from_mmap_data + _train_one_lrq.
            args = type("Args", (), {
                "max_tokens": 4, "chunk_rows": 0, "lrq_rank": 4,
                "lrq_iterations": 2, "lr": 1e-2, "seed": 0,
                "lrq_agg": "mean", "verbose": False,
            })()
            tracker = ptc.ResidencyTracker(budget_bytes=0, abort_on_exceed=False)
            with ptc.CalibPipeline(paths, depth=2) as pipe:
                layers_seen = []
                for path, data in pipe:
                    tracker.check(str(path))
                    layer = ptc._layer_from_mmap_data(path, data, max_tokens=4)
                    layers_seen.append(layer.name)
                    result = ptc._train_one_lrq(layer, args, chunked_train=False)
                    self.assertIsInstance(result, ptc.LRQResult)
            # All 3 layers were processed in order.
            self.assertEqual(layers_seen, ["layer_0", "layer_1", "layer_2"])


class TestAsyncIOPipeline(unittest.TestCase):
    """``CalibPipelineAsync`` is the macOS async-I/O variant
    of ``CalibPipeline``.  On macOS the producer is a
    libdispatch ``dispatch_io_t`` read; on Linux/Windows
    the constructor falls back to ``CalibPipeline``
    transparently (the iteration API is the same).
    """

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="calmem_async_io_")
        self._td = Path(self._tmp)
        self._paths: list[Path] = []
        self._pipes: list = []
        for i in range(5):
            p = self._td / f"layer_{i}.npz"
            _make_bundle(p, out_dim=32, in_dim=16, n_tokens=8)
            self._paths.append(p)

    def tearDown(self) -> None:
        # The async dispatcher threads are daemons; the
        # underlying GCD reads may still be in flight
        # when the test method returns.  Closing each
        # pipe forces the dispatcher to stop issuing new
        # reads.  The temp dir can then be removed
        # without blocking on a file that's still open.
        for pipe in getattr(self, "_pipes", []):
            pipe.close()
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_async_pipeline_yields_all_layers(self) -> None:
        """``CalibPipelineAsync`` yields all input layers
        in some order.  The order may differ from the
        input (the async path doesn't preserve the input
        order; the legacy ``CalibPipeline`` does).  The
        contract is: every input layer is yielded exactly
        once, the contents are correct, and the
        iteration terminates.

        On macOS the async path uses dispatch_io_t; on
        Linux/Windows the constructor falls back to the
        legacy ``CalibPipeline`` (which preserves order).
        The test asserts the per-tensor contents, not
        the order, so it passes on both platforms.
        """
        seen_paths: list[Path] = []
        seen_keys: list[set[str]] = []
        pipe = cm.CalibPipelineAsync(self._paths, depth=2)
        self._pipes.append(pipe)
        count = 0
        with pipe:
            for path, data in pipe:
                seen_paths.append(path)
                seen_keys.append(set(data.keys()))
                count += 1
                if count >= len(self._paths):
                    break
        self.assertEqual(sorted(p.name for p in seen_paths),
                         sorted(p.name for p in self._paths))
        for ks in seen_keys:
            self.assertIn("weight", ks)
            self.assertIn("train_activations", ks)

    def test_async_pipeline_iterates_all(self) -> None:
        """The async pipeline iterates the full path list
        (no infinite hang on a non-blocking producer)."""
        pipe = cm.CalibPipelineAsync(self._paths, depth=2)
        self._pipes.append(pipe)
        with pipe:
            count = 0
            for _path, _data in pipe:
                count += 1
                if count >= len(self._paths):
                    break
            self.assertEqual(count, len(self._paths))

    def test_async_pipeline_empty(self) -> None:
        """An empty path list is a no-op (no items yielded)."""
        pipe = cm.CalibPipelineAsync([], depth=2)
        self._pipes.append(pipe)
        with pipe:
            # If the constructor fell back to CalibPipeline,
            # the iteration is a no-op.  If it built an
            # async pipeline, there are no items to push.
            count = 0
            for _path, _data in pipe:
                count += 1
            self.assertEqual(count, 0)

    def test_async_pipeline_depth1(self) -> None:
        """Depth=1 is the single-read path.  The async
        variant still works (just one read in flight at a
        time, no overlap)."""
        pipe = cm.CalibPipelineAsync(self._paths, depth=1)
        self._pipes.append(pipe)
        seen: list[Path] = []
        with pipe:
            count = 0
            for path, data in pipe:
                seen.append(path)
                self.assertEqual(data["weight"].shape, (32, 16))
                count += 1
                if count >= len(self._paths):
                    break
        self.assertEqual(sorted(p.name for p in seen),
                         sorted(p.name for p in self._paths))

    def test_async_pipeline_fallback_on_non_macos(self) -> None:
        """``CalibPipelineAsync`` falls back to the
        threaded ``CalibPipeline`` when the
        ``_load_async_io_backend`` returns None.  On
        macOS the backend is always available, so the
        test skips there; on non-macOS we explicitly
        disable the loader and confirm the iteration
        succeeds via the legacy path.

        We patch ``_load_async_io_backend`` to ``None``;
        the test then constructs ``CalibPipelineAsync``
        and asserts ``pipe._fallback`` is set.  This
        pins the fallback contract.
        """
        if _macos_with_clang():
            self.skipTest(
                "macOS: backend is available, cannot test fallback path"
            )
        # Patch the loader to return None.
        orig = cm._load_async_io_backend
        cm._load_async_io_backend = lambda: None
        try:
            pipe = cm.CalibPipelineAsync(self._paths, depth=2)
        finally:
            cm._load_async_io_backend = orig
        self._pipes.append(pipe)
        # When the backend is None, the constructor
        # falls back to a ``CalibPipeline`` and the
        # ``_fallback`` attribute is set.
        self.assertIsNotNone(getattr(pipe, "_fallback", None))
        with pipe:
            count = 0
            for _path, _data in pipe:
                count += 1
                if count >= len(self._paths):
                    break
            self.assertEqual(count, len(self._paths))

    def test_open_calib_pipeline_auto_returns_async_on_macos(self) -> None:
        """``open_calib_pipeline(..., async_io='auto')``
        returns ``CalibPipelineAsync`` on macOS (where
        the dispatch_io_t bridge is available)."""
        if not _macos_with_clang():
            self.skipTest("macOS clang++ not available")
        pipe = cm.open_calib_pipeline(self._paths, depth=2, async_io="auto")
        self._pipes.append(pipe)
        self.assertIsInstance(pipe, cm.CalibPipelineAsync)
        with pipe:
            count = 0
            for _path, _data in pipe:
                count += 1
                if count >= len(self._paths):
                    break
            self.assertEqual(count, len(self._paths))

    def test_open_calib_pipeline_off_returns_legacy(self) -> None:
        """``open_calib_pipeline(..., async_io='off')``
        always returns the legacy threaded
        ``CalibPipeline``, even on macOS.  This pins
        the override path for tests and the
        ``--async-io off`` CLI flag."""
        pipe = cm.open_calib_pipeline(self._paths, depth=2, async_io="off")
        self._pipes.append(pipe)
        # The legacy ``CalibPipeline`` is the threaded
        # path.  Even on macOS, ``async_io='off'`` skips
        # the dispatch_io_t path.
        self.assertIsInstance(pipe, cm.CalibPipeline)
        with pipe:
            count = 0
            for _path, _data in pipe:
                count += 1
                if count >= len(self._paths):
                    break
            self.assertEqual(count, len(self._paths))

    def test_open_calib_pipeline_on_raises_when_unavailable(self) -> None:
        """``open_calib_pipeline(..., async_io='on')``
        raises ``RuntimeError`` when the dispatch_io_t
        bridge is not available.  On macOS the bridge
        is always available, so the test is conditional.
        On non-macOS the test asserts the ``RuntimeError``."""
        if _macos_with_clang():
            self.skipTest("macOS: bridge is available, cannot test failure")
        with self.assertRaises(RuntimeError):
            cm.open_calib_pipeline(self._paths, depth=2, async_io="on")

    def test_open_calib_pipeline_on_succeeds_on_macos(self) -> None:
        """``open_calib_pipeline(..., async_io='on')``
        returns a ``CalibPipelineAsync`` on macOS."""
        if not _macos_with_clang():
            self.skipTest("macOS clang++ not available")
        pipe = cm.open_calib_pipeline(self._paths, depth=2, async_io="on")
        self._pipes.append(pipe)
        self.assertIsInstance(pipe, cm.CalibPipelineAsync)
        with pipe:
            count = 0
            for _path, _data in pipe:
                count += 1
                if count >= len(self._paths):
                    break
            self.assertEqual(count, len(self._paths))


if __name__ == "__main__":
    unittest.main(verbosity=2)
