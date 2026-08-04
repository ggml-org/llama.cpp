#!/usr/bin/env python3
"""Unit tests for ``tools/tessera/calibration_metal.py``.

Phase 16.5 (memopt-metal-dispatch).  The per-chunk LRQ / FLRQ
matmul is dispatched to the fastest available backend:

* Metal (MPS) on Apple Silicon (M1/M2/M3/M4)
* Accelerate (cblas_sgemm) on any Mac (Intel or Apple Silicon
  without Metal)
* numpy on Linux/Windows or when the C bridge fails to build

The tests assert the dispatch is correct on every platform
they run on:

* On macOS Apple Silicon: all three backends are available;
  the tests assert each backend produces results bit-equivalent
  (within float32 epsilon) to the numpy reference.
* On macOS Intel: Metal and Accelerate are both available;
  the tests assert both.  (Metal is slower than Accelerate on
  Intel integrated GPUs, but the dispatch is correct.)
* On Linux/Windows: only the numpy backend is available; the
  tests detect the absence of Metal/Accelerate and skip the
  backend-specific assertions with a clear message.

Run as::

    python3 tools/tessera/test_calibration_metal.py

Exits 0 on success, non-zero on any failure.  No disk or
network; the tests are pure-numpy / pure-cmath.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR.parent.parent))  # for top-level import

import calibration_metal as cm  # noqa: E402


# Tolerances for the float32 matmul equivalence.  cblas_sgemm
# uses a different reduction order than numpy's matmul; the
# accumulator rounds to F32 each multiply-add, so the per-
# element difference is bounded by ``K * eps`` (K is the
# inner-dim).  For K=64 that's 64 * 1.19e-7 = 7.6e-6; we
# use a 1e-4 tolerance for safety.
F32_TOL = 1e-4


def _macos_with_clang() -> bool:
    """True if macOS + clang++ are available (we can build the
    C bridge).  Used to gate the backend-specific tests."""
    if platform.system() != "Darwin":
        return False
    try:
        return subprocess.run(
            ["xcrun", "--find", "clang++"],
            check=True, capture_output=True, text=True,
        ).returncode == 0
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


class TestMatmulDispatch(unittest.TestCase):
    """The dispatch selects the fastest available backend for
    the host.  The test asserts:

      * The dispatch singleton returns a valid backend.
      * The dispatch name is one of the documented constants.
      * On macOS, the dispatch is either Metal or Accelerate
        (not numpy).  On Linux/Windows, the dispatch is numpy.
      * The ``chunked_matmul`` free function agrees with the
        ``matmul`` free function.
    """

    def test_dispatch_singleton(self) -> None:
        """The first call to ``get_matmul_backend`` returns a
        backend; subsequent calls return the same instance."""
        b1 = cm.get_matmul_backend()
        b2 = cm.get_matmul_backend()
        self.assertIs(b1, b2)
        self.assertIn(b1.name, (cm.BACKEND_NUMPY, cm.BACKEND_ACCELERATE, cm.BACKEND_METAL))

    def test_dispatch_name_helper(self) -> None:
        """``get_matmul_backend_name`` matches the singleton's name."""
        b1 = cm.get_matmul_backend()
        self.assertEqual(cm.get_matmul_backend_name(), b1.name)

    def test_dispatch_picks_metal_or_accelerate_on_macos(self) -> None:
        """On macOS, the dispatch is Metal (preferred) or Accelerate."""
        if not _macos_with_clang():
            self.skipTest("macOS with clang++ not available")
        backend = cm.get_matmul_backend()
        self.assertIn(
            backend.name, (cm.BACKEND_METAL, cm.BACKEND_ACCELERATE),
            f"macOS dispatch should be Metal or Accelerate, got {backend.name!r}",
        )

    def test_dispatch_picks_numpy_on_non_macos(self) -> None:
        """On Linux/Windows, the dispatch is numpy."""
        if platform.system() == "Darwin":
            self.skipTest("Darwin: dispatch may use Metal/Accelerate")
        backend = cm.get_matmul_backend()
        self.assertEqual(backend.name, cm.BACKEND_NUMPY)

    def test_chunked_matmul_matches_matmul(self) -> None:
        """The ``chunked_matmul`` free function and the ``matmul``
        free function return identical results (they share the
        same backend dispatch)."""
        rng = np.random.default_rng(0)
        a = rng.standard_normal((32, 32)).astype(np.float32)
        b = rng.standard_normal((32, 32)).astype(np.float32)
        np.testing.assert_array_equal(
            cm.chunked_matmul(a, b), cm.matmul(a, b),
        )

    def test_force_backend_affects_chunked_matmul(self) -> None:
        """The ``force_backend`` test affordance overrides the
        dispatch singleton for ``chunked_matmul``.  The
        ``matmul`` free function is unaffected (it always uses
        the singleton).  This pins the test affordance so
        backend-specific tests can be written deterministically."""
        rng = np.random.default_rng(0)
        a = rng.standard_normal((32, 32)).astype(np.float32)
        b = rng.standard_normal((32, 32)).astype(np.float32)
        # Force numpy.
        cm.force_backend(cm.MatmulBackend(
            name=cm.BACKEND_NUMPY, library=None, _lib=None, _sgemm=None,
            _lock=__import__("threading").Lock(),
        ))
        try:
            ref = cm.matmul_numpy(a, b)
            np.testing.assert_array_equal(cm.chunked_matmul(a, b), ref)
        finally:
            cm.force_backend(None)


class TestMatmulNumpy(unittest.TestCase):
    """``matmul_numpy`` is the legacy pure-numpy path.  It
    must be available on every platform (the test is the
    reference for the other backends' equivalence assertions).
    """

    def test_numpy_correctness_small(self) -> None:
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        np.testing.assert_array_equal(
            cm.matmul_numpy(a, b),
            np.array([[19.0, 22.0], [43.0, 50.0]], dtype=np.float32),
        )

    def test_numpy_correctness_larger(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.standard_normal((64, 64)).astype(np.float32)
        b = rng.standard_normal((64, 64)).astype(np.float32)
        np.testing.assert_allclose(
            cm.matmul_numpy(a, b), a @ b, rtol=1e-6, atol=1e-6,
        )

    def test_numpy_non_square(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.standard_normal((16, 32)).astype(np.float32)
        b = rng.standard_normal((32, 8)).astype(np.float32)
        np.testing.assert_allclose(
            cm.matmul_numpy(a, b), a @ b, rtol=1e-6, atol=1e-6,
        )


class TestMatmulAccelerate(unittest.TestCase):
    """``matmul_accelerate`` is the Accelerate cblas_sgemm path.
    Skipped on Linux/Windows (no Accelerate framework)."""

    @classmethod
    def setUpClass(cls) -> None:
        if not _macos_with_clang():
            raise unittest.SkipTest("Accelerate path requires macOS with clang++")

    def test_accelerate_correctness(self) -> None:
        """The Accelerate matmul agrees with numpy within float32
        epsilon for a representative (64, 64) square."""
        rng = np.random.default_rng(0)
        a = rng.standard_normal((64, 64)).astype(np.float32)
        b = rng.standard_normal((64, 64)).astype(np.float32)
        ref = a @ b
        result = cm.matmul_accelerate(a, b)
        np.testing.assert_allclose(result, ref, rtol=F32_TOL, atol=F32_TOL)

    def test_accelerate_non_square(self) -> None:
        """The Accelerate matmul handles non-square shapes
        (e.g. (16, 32) @ (32, 8) = (16, 8)).  This is the
        shape the per-chunk LRQ uses for the ``u_chunk @ v``
        matmul when ``rank < in_dim``."""
        rng = np.random.default_rng(0)
        a = rng.standard_normal((16, 32)).astype(np.float32)
        b = rng.standard_normal((32, 8)).astype(np.float32)
        ref = a @ b
        result = cm.matmul_accelerate(a, b)
        np.testing.assert_allclose(result, ref, rtol=F32_TOL, atol=F32_TOL)

    def test_accelerate_identity(self) -> None:
        """``A @ I = A`` for an identity matrix.  This pins
        the no-transpose case (the bridge only supports
        non-transposed matmul for now)."""
        rng = np.random.default_rng(0)
        a = rng.standard_normal((32, 32)).astype(np.float32)
        eye = np.eye(32, dtype=np.float32)
        result = cm.matmul_accelerate(a, eye)
        np.testing.assert_allclose(result, a, rtol=1e-6, atol=1e-6)

    def test_accelerate_falls_back_to_numpy_on_transpose(self) -> None:
        """The Accelerate bridge doesn't support transposed
        operands (it's a single-purpose fast path).  The
        matmul call returns a numpy result for transposed
        inputs.  The dispatch should never request a
        transposed matmul from the bridge; the call site
        uses ``b.copy().T`` to materialise the view."""
        # This is a smoke test: the bridge rejects the
        # transposed flag.  We can't easily exercise that
        # through the Python entry point (which always
        # passes 0/0), so we just confirm the non-transpose
        # path works.
        rng = np.random.default_rng(0)
        a = rng.standard_normal((8, 16)).astype(np.float32)
        b = rng.standard_normal((8, 16)).astype(np.float32)
        # a @ b.T
        result = cm.matmul_accelerate(a, b.T.copy())
        ref = a @ b.T
        np.testing.assert_allclose(result, ref, rtol=F32_TOL, atol=F32_TOL)


class TestMatmulMetal(unittest.TestCase):
    """``matmul_metal`` is the Metal Performance Shaders path.
    Skipped on Linux/Windows (no Metal framework)."""

    @classmethod
    def setUpClass(cls) -> None:
        if not _macos_with_clang():
            raise unittest.SkipTest("Metal path requires macOS with clang++")

    def test_metal_correctness(self) -> None:
        """The Metal matmul agrees with numpy within float32
        epsilon for a representative (64, 64) square.  The
        MPS path rounds to F32 at each multiply-add; the
        per-element difference is bounded by ``K * eps``."""
        rng = np.random.default_rng(0)
        a = rng.standard_normal((64, 64)).astype(np.float32)
        b = rng.standard_normal((64, 64)).astype(np.float32)
        ref = a @ b
        result = cm.matmul_metal(a, b)
        np.testing.assert_allclose(result, ref, rtol=F32_TOL, atol=F32_TOL)

    def test_metal_non_square(self) -> None:
        """The Metal matmul handles non-square shapes (the
        common case for the per-chunk LRQ is (4096, 16) @
        (16, 4096) = (4096, 4096))."""
        rng = np.random.default_rng(0)
        a = rng.standard_normal((16, 32)).astype(np.float32)
        b = rng.standard_normal((32, 8)).astype(np.float32)
        ref = a @ b
        result = cm.matmul_metal(a, b)
        np.testing.assert_allclose(result, ref, rtol=F32_TOL, atol=F32_TOL)

    def test_metal_empty_dims(self) -> None:
        """An empty result (m=0, n=0, k=0) returns a zero-shaped
        array without invoking the bridge.  The C bridge
        rejects zero dims; the Python side short-circuits to
        a zero allocation."""
        a = np.zeros((0, 0), dtype=np.float32)
        b = np.zeros((0, 0), dtype=np.float32)
        result = cm.matmul_metal(a, b)
        self.assertEqual(result.shape, (0, 0))

    def test_metal_shape_mismatch_raises(self) -> None:
        """An inner-dim mismatch raises ``ValueError`` rather
        than silently producing the wrong answer."""
        a = np.zeros((8, 16), dtype=np.float32)
        b = np.zeros((32, 8), dtype=np.float32)
        with self.assertRaises(ValueError):
            cm.matmul_metal(a, b)

    def test_metal_does_not_modify_inputs(self) -> None:
        """The Metal matmul does not modify the caller's
        ``a`` or ``b`` (the bridge copies out into a fresh
        output buffer).  This is the API contract for the
        per-chunk LRQ backward pass, which reuses the
        gradient buffers across iterations."""
        rng = np.random.default_rng(0)
        a = rng.standard_normal((64, 64)).astype(np.float32)
        b = rng.standard_normal((64, 64)).astype(np.float32)
        a_orig = a.copy()
        b_orig = b.copy()
        cm.matmul_metal(a, b)
        np.testing.assert_array_equal(a, a_orig)
        np.testing.assert_array_equal(b, b_orig)


class TestMatmulBackendEquivalence(unittest.TestCase):
    """On macOS, the three backends (numpy, Accelerate, Metal)
    produce equivalent results.  Skipped on Linux/Windows.
    """

    @classmethod
    def setUpClass(cls) -> None:
        if not _macos_with_clang():
            raise unittest.SkipTest("Backend equivalence requires macOS with clang++")

    def test_equivalence_via_force_backend(self) -> None:
        """The same ``a @ b`` is computed three times, once per
        backend.  All three results agree within float32
        epsilon.  We use ``force_backend`` to pin each call
        to a specific backend so the singleton is not in
        the way."""
        rng = np.random.default_rng(0)
        a = rng.standard_normal((64, 64)).astype(np.float32)
        b = rng.standard_normal((64, 64)).astype(np.float32)
        # Force numpy.
        cm.force_backend(cm.MatmulBackend(
            name=cm.BACKEND_NUMPY, library=None, _lib=None, _sgemm=None,
            _lock=__import__("threading").Lock(),
        ))
        try:
            ref_np = cm.chunked_matmul(a, b)
        finally:
            cm.force_backend(None)
        # Force Accelerate.
        cm.force_backend(cm.MatmulBackend(
            name=cm.BACKEND_ACCELERATE, library=cm._DEFAULT_ACCEL_LIBRARY,
            _lib=cm._load_accelerate_backend(cm._DEFAULT_ACCEL_LIBRARY),
            _sgemm=None,
            _lock=__import__("threading").Lock(),
        ))
        try:
            ref_accel = cm.chunked_matmul(a, b)
        finally:
            cm.force_backend(None)
        # Force Metal.
        cm.force_backend(cm.MatmulBackend(
            name=cm.BACKEND_METAL, library=cm._DEFAULT_METAL_LIBRARY,
            _lib=cm._load_metal_backend(cm._DEFAULT_METAL_LIBRARY),
            _sgemm=None,
            _lock=__import__("threading").Lock(),
        ))
        try:
            ref_metal = cm.chunked_matmul(a, b)
        finally:
            cm.force_backend(None)
        # All three agree within F32_TOL.
        np.testing.assert_allclose(ref_np, ref_accel, rtol=F32_TOL, atol=F32_TOL)
        np.testing.assert_allclose(ref_np, ref_metal, rtol=F32_TOL, atol=F32_TOL)
        np.testing.assert_allclose(ref_accel, ref_metal, rtol=F32_TOL, atol=F32_TOL)


class TestMatmulReuse(unittest.TestCase):
    """The Metal bridge reuses one MTLCommandQueue across calls.

    The legacy bridge allocated a new queue per call, which
    caused Metal's internal command buffer pool to grow
    unbounded.  After ~135 calls the device started returning
    NaN in the result buffer because the GPU was out of
    memory.  The fix pins one queue via ``dispatch_once``;
    the test below pins the regression by running 200
    matmul calls in a tight loop and asserting the result
    is always the same (no NaN, no drift).

    Skipped on Linux/Windows (no Metal framework).
    """

    @classmethod
    def setUpClass(cls) -> None:
        if not _macos_with_clang():
            raise unittest.SkipTest("Metal reuse test requires macOS with clang++")

    def test_metal_stable_across_200_calls(self) -> None:
        """200 consecutive Metal matmul calls all return the
        correct answer.  This pins the queue-reuse fix.

        The shape (1024, 256) @ (256, 16) is the per-chunk
        FLRQ R1-Sketch size at the test's 1024x256 weight.
        """
        rng = np.random.default_rng(0)
        a = rng.standard_normal((1024, 256)).astype(np.float32)
        b = rng.standard_normal((256, 16)).astype(np.float32)
        ref = a @ b
        for i in range(200):
            result = cm.matmul_metal(a, b)
            self.assertFalse(
                np.isnan(result).any(),
                f"iteration {i}: Metal returned NaN (queue-reuse regression)",
            )
            np.testing.assert_allclose(
                result, ref, rtol=F32_TOL, atol=F32_TOL,
                err_msg=f"iteration {i}: Metal result diverged from numpy",
            )

    def test_accelerate_stable_across_200_calls(self) -> None:
        """200 consecutive Accelerate matmul calls all return
        the correct answer.  Pin the test for the Accelerate
        bridge even though the queue-reuse bug was Metal-
        specific; the test catches any future regression
        in either bridge."""
        rng = np.random.default_rng(0)
        a = rng.standard_normal((1024, 256)).astype(np.float32)
        b = rng.standard_normal((256, 16)).astype(np.float32)
        ref = a @ b
        for i in range(200):
            result = cm.matmul_accelerate(a, b)
            self.assertFalse(
                np.isnan(result).any(),
                f"iteration {i}: Accelerate returned NaN",
            )
            np.testing.assert_allclose(
                result, ref, rtol=F32_TOL, atol=F32_TOL,
                err_msg=f"iteration {i}: Accelerate result diverged from numpy",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
