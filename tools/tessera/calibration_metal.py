"""Apple Silicon / Accelerate / Metal matmul dispatch for the
per-chunk calibration.

The legacy per-chunk LRQ / FLRQ matmul in
``tools/tessera/calibration_memory.py`` is pure numpy.  On
Apple Silicon (M1/M2/M3/M4) and Intel Macs, the same matmul
routed through Apple's first-party libraries is 2-4x faster:

* **Apple Metal Performance Shaders (MPS)** via
  ``MPSMatrixMultiplication``: GPU matmul on Apple Silicon
  unified memory.  The per-chunk numpy arrays are GPU-visible
  without an explicit copy; the chunked GEMM is small enough
  to keep residency local, which matches MPS's sweet spot.
* **Apple Accelerate.framework** via ``cblas_sgemm``:
  CPU SIMD.  Apple Silicon dispatches to AMX/NEON SIMD; Intel
  Mac dispatches to AVX-512 SIMD.  The same ``cblas_sgemm``
  entry point covers both.

On Linux/Windows (no Accelerate.framework) the dispatch falls
back to numpy.  The dispatch is a thin wrapper: the hot path
is ``matmul(a, b)`` which returns ``a @ b`` via the
fastest-available backend.

Why the C bridge instead of direct python bindings:
``pip`` does not have a Metal or Accelerate Python binding.
PyObjC has ``objc`` (and historically ``Accelerate``) but
those bindings vary across macOS versions and require a
``pip install`` step that the calibration harness does not
have.  The cheapest path is a small Objective-C++ wrapper
(``apple_metal_matmul.mm``) and a C++ wrapper
(``apple_accelerate_matmul.cpp``), each compiled on demand to
a small ``.dylib`` via ``xcrun clang++``, and loaded via
ctypes.  The .dylib is built once and cached under
``tools/tessera/.build/``; subsequent calls hit the cache.

Usage::

    from tools.tessera.calibration_metal import (
        MatmulBackend, matmul, matmul_numpy, matmul_accelerate,
        matmul_metal, get_matmul_backend, get_matmul_backend_name,
    )

    backend = MatmulBackend.detect()
    c = backend.matmul(a, b)        # a @ b with the best backend
    c = matmul(a, b)                # the dispatch entry point
"""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Avoid importing Objective-C headers; we use ctypes.  The
# ctypes.CDLL load is lazy so the module is importable on
# Linux/Windows without errors (the dispatch falls back to
# numpy there).
_THREAD_LOCK = threading.Lock()

_ROOT = Path(__file__).resolve().parent
_BUILD_DIR = _ROOT / ".build"
_ACCEL_SOURCE = _ROOT / "apple_accelerate_matmul.cpp"
_METAL_SOURCE = _ROOT / "apple_metal_matmul.mm"
_DEFAULT_ACCEL_LIBRARY = _BUILD_DIR / "libtessera_accel_matmul.dylib"
_DEFAULT_METAL_LIBRARY = _BUILD_DIR / "libtessera_metal_matmul.dylib"


# Backend name constants.  Stored as strings so the user
# can read them from logs and the dispatch is a free
# function (no enum dependency at the call site).
BACKEND_NUMPY = "numpy"
BACKEND_ACCELERATE = "accelerate"
BACKEND_METAL = "metal"


def _is_macos() -> bool:
    """True if running on macOS (any architecture)."""
    return platform.system() == "Darwin"


def _has_clang() -> bool:
    """True if the macOS ``xcrun clang++`` toolchain is on PATH.

    The C bridge needs clang++ to compile.  On a stripped
    macOS without Xcode Command Line Tools the build is
    skipped and the dispatch falls back to numpy.  This
    is fine: the calibration still runs, just slower.
    """
    if not _is_macos():
        return False
    try:
        return (
            subprocess.run(
                ["xcrun", "--find", "clang++"],
                check=True, capture_output=True, text=True,
            ).returncode
            == 0
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _build_accelerate_library(output: Path) -> Optional[Path]:
    """Compile ``apple_accelerate_matmul.cpp`` to a .dylib.

    Returns the path to the .dylib on success, or ``None`` on
    any build failure (the dispatch falls back to numpy).
    """
    if not _ACCEL_SOURCE.is_file():
        return None
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(
            [
                "xcrun", "clang++",
                "-std=c++17", "-O3",
                "-dynamiclib",
                str(_ACCEL_SOURCE),
                "-framework", "Accelerate",
                "-o", str(output),
            ],
            check=True, capture_output=True, text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    if proc.returncode != 0:
        return None
    return output


def _build_metal_library(output: Path) -> Optional[Path]:
    """Compile ``apple_metal_matmul.mm`` to a .dylib.

    Returns the path to the .dylib on success, or ``None`` on
    any build failure.  The Metal bridge needs the Metal
    Performance Shaders framework; we link against it
    explicitly so a missing framework yields a clear build
    error.
    """
    if not _METAL_SOURCE.is_file():
        return None
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(
            [
                "xcrun", "clang++",
                "-std=c++17", "-O3",
                "-dynamiclib",
                str(_METAL_SOURCE),
                "-framework", "Metal",
                "-framework", "MetalPerformanceShaders",
                "-framework", "Foundation",
                "-o", str(output),
            ],
            check=True, capture_output=True, text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    if proc.returncode != 0:
        return None
    return output


def _load_accelerate_backend(library: Path) -> Optional[ctypes.CDLL]:
    """Load the Accelerate .dylib and bind the matmul entry point.

    Returns ``None`` on any load failure (the dispatch
    falls back to numpy).  The caller is responsible for
    releasing the library; ctypes keeps it alive via the
    returned handle.
    """
    try:
        lib = ctypes.CDLL(str(library))
    except OSError:
        return None
    pointer = ctypes.POINTER(ctypes.c_float)
    try:
        sgemm = lib.tessera_accelerate_sgemm_f32
        sgemm.restype = ctypes.c_int
        sgemm.argtypes = [
            pointer, pointer, pointer,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_int, ctypes.c_int,
        ]
    except AttributeError:
        return None
    return lib


def _load_metal_backend(library: Path) -> Optional[ctypes.CDLL]:
    """Load the Metal .dylib and bind the matmul entry point.

    Returns ``None`` on any load failure.  Same ownership
    contract as ``_load_accelerate_backend``.
    """
    try:
        lib = ctypes.CDLL(str(library))
    except OSError:
        return None
    pointer = ctypes.POINTER(ctypes.c_float)
    try:
        sgemm = lib.tessera_metal_sgemm_f32
        sgemm.restype = ctypes.c_int
        sgemm.argtypes = [
            pointer, pointer, pointer,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_int, ctypes.c_int,
        ]
    except AttributeError:
        return None
    return lib


@dataclass
class MatmulBackend:
    """The dispatched matmul backend for the per-chunk calibration.

    The detection is one-shot: ``MatmulBackend.detect()``
    returns a process-wide singleton so the build cost is
    paid only once.  The matmul call itself is a
    ``dataclass`` method so the dispatch is at the call site
    (not the import level): a user who wants a specific
    backend can construct the dataclass directly.
    """

    name: str
    library: Optional[Path]
    _lib: Optional[ctypes.CDLL]
    _sgemm: Optional[object]
    _lock: threading.Lock

    @classmethod
    def detect(cls) -> "MatmulBackend":
        """Detect the fastest available matmul backend for this host.

        Priority: Metal (MPS) > Accelerate (cblas_sgemm) > numpy.
        The detection is process-wide: the first call pays
        the build / load cost; subsequent calls hit the
        cache.  Lock-protected so the singleton is built
        exactly once even under contention.
        """
        with _THREAD_LOCK:
            if not _is_macos() or not _has_clang():
                return cls(
                    name=BACKEND_NUMPY,
                    library=None, _lib=None, _sgemm=None,
                    _lock=threading.Lock(),
                )
            # Try Metal first.
            metal_path = Path(
                os.environ.get("TESSERA_METAL_MATMUL_LIBRARY",
                               _DEFAULT_METAL_LIBRARY)
            )
            if metal_path.is_file() or _METAL_SOURCE.is_file():
                if not metal_path.is_file() or (
                    _METAL_SOURCE.is_file()
                    and metal_path.stat().st_mtime < _METAL_SOURCE.stat().st_mtime
                ):
                    _build_metal_library(metal_path)
                lib = _load_metal_backend(metal_path)
                if lib is not None:
                    sgemm = lib.tessera_metal_sgemm_f32
                    return cls(
                        name=BACKEND_METAL,
                        library=metal_path, _lib=lib, _sgemm=sgemm,
                        _lock=threading.Lock(),
                    )
            # Try Accelerate as a fallback.
            accel_path = Path(
                os.environ.get("TESSERA_ACCEL_MATMUL_LIBRARY",
                               _DEFAULT_ACCEL_LIBRARY)
            )
            if accel_path.is_file() or _ACCEL_SOURCE.is_file():
                if not accel_path.is_file() or (
                    _ACCEL_SOURCE.is_file()
                    and accel_path.stat().st_mtime < _ACCEL_SOURCE.stat().st_mtime
                ):
                    _build_accelerate_library(accel_path)
                lib = _load_accelerate_backend(accel_path)
                if lib is not None:
                    sgemm = lib.tessera_accelerate_sgemm_f32
                    return cls(
                        name=BACKEND_ACCELERATE,
                        library=accel_path, _lib=lib, _sgemm=sgemm,
                        _lock=threading.Lock(),
                    )
            return cls(
                name=BACKEND_NUMPY,
                library=None, _lib=None, _sgemm=None,
                _lock=threading.Lock(),
            )

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute ``a @ b`` using the dispatched backend.

        The result is a fresh numpy array (the caller's
        ``b`` is not modified).  The dispatch is
        shape-aware: a non-2-D input, a 0-D input, or a
        non-FP32 input falls back to numpy.  The transpose
        case is also numpy-fallback because the C bridges
        are non-transposed (the Python call site does
        ``.copy()`` of ``v.T`` / ``u.T`` if needed; the
        legacy code never needs a transposed matmul beyond
        a view).
        """
        if self.name == BACKEND_NUMPY or self._sgemm is None:
            return a @ b
        if a.ndim != 2 or b.ndim != 2:
            return a @ b
        if a.dtype != np.float32 or b.dtype != np.float32:
            # Cast and recurse.  The cast is the cost of the
            # dispatch; in practice the per-chunk matmuls
            # are already F32.
            return (a.astype(np.float32)) @ (b.astype(np.float32))
        m, k = a.shape
        k2, n = b.shape
        if k != k2:
            raise ValueError(
                f"matmul shape mismatch: a is {a.shape}, b is {b.shape}"
            )
        if m == 0 or n == 0 or k == 0:
            return np.zeros((m, n), dtype=np.float32)
        a_c = np.ascontiguousarray(a)
        b_c = np.ascontiguousarray(b)
        out = np.empty((m, n), dtype=np.float32)
        pointer = ctypes.POINTER(ctypes.c_float)
        with self._lock:
            rc = self._sgemm(
                a_c.ctypes.data_as(pointer),
                b_c.ctypes.data_as(pointer),
                out.ctypes.data_as(pointer),
                m, n, k, 0, 0,
            )
        if rc != 0:
            # Bridge rejected the call (e.g. transposed).
            # Fall back to numpy so the calibration
            # continues.
            return a @ b
        return out

    @property
    def is_metal(self) -> bool:
        return self.name == BACKEND_METAL

    @property
    def is_accelerate(self) -> bool:
        return self.name == BACKEND_ACCELERATE

    @property
    def is_numpy(self) -> bool:
        return self.name == BACKEND_NUMPY


# The dispatch singleton.  Constructed on first access.
_DISPATCH_SINGLETON: Optional[MatmulBackend] = None
_DISPATCH_LOCK = threading.Lock()


def get_matmul_backend() -> MatmulBackend:
    """Return the process-wide matmul backend singleton.

    The first call detects and (if needed) builds the
    C bridge; subsequent calls hit the cache.  Tests that
    want a specific backend should construct
    ``MatmulBackend(...)`` directly with the name they
    want and call ``.matmul(a, b)``; the singleton is the
    production path.
    """
    global _DISPATCH_SINGLETON
    with _DISPATCH_LOCK:
        if _DISPATCH_SINGLETON is None:
            _DISPATCH_SINGLETON = MatmulBackend.detect()
        return _DISPATCH_SINGLETON


def get_matmul_backend_name() -> str:
    """Return the name of the process-wide matmul backend.

    Convenience: ``"metal"``, ``"accelerate"``, or
    ``"numpy"``.  Cheap to call (just reads the cached
    singleton).  Used by ``per_tensor_calibrate.py`` to
    log which backend is in use.
    """
    return get_matmul_backend().name


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The dispatched matmul: ``a @ b`` via the fastest backend.

    This is the call site for the per-chunk LRQ / FLRQ
    matmul.  The shape handling and the numpy fallback are
    documented on ``MatmulBackend.matmul``.
    """
    return get_matmul_backend().matmul(a, b)


def matmul_numpy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pure-numpy matmul: ``a @ b``.

    Always available.  This is the legacy path; the
    dispatch wrapper selects it on Linux/Windows.
    """
    return a @ b


def matmul_accelerate(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Force the Accelerate-backend matmul (or numpy if unavailable).

    Used by tests that want to assert the Accelerate path
    specifically.  Production code uses ``matmul``.
    """
    return MatmulBackend(
        name=BACKEND_ACCELERATE,
        library=_DEFAULT_ACCEL_LIBRARY,
        _lib=_load_accelerate_backend(_DEFAULT_ACCEL_LIBRARY),
        _sgemm=None,
        _lock=threading.Lock(),
    ).matmul(a, b)


def matmul_metal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Force the Metal-backend matmul (or numpy if unavailable).

    Used by tests that want to assert the Metal path
    specifically.  Production code uses ``matmul``.
    """
    return MatmulBackend(
        name=BACKEND_METAL,
        library=_DEFAULT_METAL_LIBRARY,
        _lib=_load_metal_backend(_DEFAULT_METAL_LIBRARY),
        _sgemm=None,
        _lock=threading.Lock(),
    ).matmul(a, b)


# Override hook: tests can monkey-patch this to force a
# specific backend.  When set to a non-None value, the
# ``matmul`` free function uses this backend instead of
# the auto-detected one.  Used by the unit tests in
# ``test_calibration_metal.py`` to assert each backend's
# result against numpy.
_FORCED_BACKEND: Optional[MatmulBackend] = None


def force_backend(backend: Optional[MatmulBackend]) -> None:
    """Pin the dispatch to a specific backend (test affordance)."""
    global _FORCED_BACKEND
    _FORCED_BACKEND = backend


def _matmul_dispatched(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The matmul entry point with the test-override hook applied."""
    if _FORCED_BACKEND is not None:
        return _FORCED_BACKEND.matmul(a, b)
    return matmul(a, b)


# Public entry: the test-override-aware free function.  The
# per-chunk calibration uses this so the unit tests can pin
# the backend.
def chunked_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-chunk matmul entry point with test-override support.

    This is the function the calibration should import and
    call at the matmul call sites.  In production it
    dispatches to the fastest available backend (Metal >
    Accelerate > numpy); in tests it can be pinned via
    ``force_backend``.
    """
    return _matmul_dispatched(a, b)


__all__ = [
    "BACKEND_NUMPY",
    "BACKEND_ACCELERATE",
    "BACKEND_METAL",
    "MatmulBackend",
    "get_matmul_backend",
    "get_matmul_backend_name",
    "matmul",
    "matmul_numpy",
    "matmul_accelerate",
    "matmul_metal",
    "chunked_matmul",
    "force_backend",
]
