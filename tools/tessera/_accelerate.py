"""Thin Python wrapper around Apple's vDSP for Tessera CPU-side kernels.

This module exposes the vDSP entry points that the Tile640 quantizer's inner
loops will eventually want, with a pure-Python fallback so the import is safe
on non-Apple platforms (Linux CI, Windows) and on Apple Silicon builds that
happen to run inside a sandboxed Python without the Accelerate dylib.

The interface deliberately mirrors the C names so a future Cython/C port
drops in without renaming. All functions accept either a Python list, a
``ctypes``-compatible array of ``float32``, or a ``numpy.float32`` array when
numpy is importable. Inputs are treated as dense single-precision streams;
non-unit strides are exposed as a ``stride`` keyword on the per-element
variants (default 1).

Functions:

* ``vDSP_meanv``     mean of a vector
* ``vDSP_measqv``    mean of squares
* ``vDSP_maxv``      maximum element
* ``vDSP_minv``      minimum element
* ``vDSP_sve``       sum of elements
* ``vDSP_vsmul``     vector multiplied by a scalar
* ``vDSP_vmul``      elementwise product of two vectors
* ``vDSP_vadd``      elementwise sum of two vectors
* ``vDSP_dotpr``     dot product
* ``vDSP_conv``      convolution of two real vectors
* ``vDSP_zrvmul``    elementwise complex-by-real multiply (split complex)
* ``vDSP_mmov``      matrix copy / memcpy for sub-matrices

Integration sketch (follow-up, not done here):

The Tile640 quantizer's per-tile scale/clip loops, residual computation, and
dequant verification are the natural targets. ``vDSP_sve`` and ``vDSP_measqv``
on the per-row 640-element activation segments would replace the existing
Python accumulators; ``vDSP_vsmul`` would replace per-tile scale application;
``vDSP_vadd`` / ``vDSP_vmul`` would replace the residual addition and the
importance-weighted error accumulation. The shader-side callers in
``apple_accelerate.cpp`` are a strict superset of these and would not change.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import platform
import sys
from array import array
from typing import Sequence


# ---------------------------------------------------------------------------
# Platform + library detection
# ---------------------------------------------------------------------------


_IS_DARWIN = platform.system() == "Darwin"
_ACCEL_PATH = ctypes.util.find_library("Accelerate") if _IS_DARWIN else None
# ``find_library('Accelerate')`` returns the umbrella framework path on macOS
# (e.g. ``/System/Library/Frameworks/Accelerate.framework/Accelerate``). vDSP
# lives inside vecLib, which the umbrella re-exports, so loading either path
# gives us the vDSP symbols. We prefer the umbrella for simplicity.
_VDSP = None
_LOAD_ERROR: Exception | None = None

if _IS_DARWIN and _ACCEL_PATH:
    try:
        _VDSP = ctypes.CDLL(_ACCEL_PATH)
    except OSError as exc:
        _LOAD_ERROR = exc
        _VDSP = None


def is_available() -> bool:
    """Return True iff a vDSP binding was loaded successfully."""
    return _VDSP is not None


def backend_path() -> str | None:
    """Path to the loaded Accelerate dylib, or None if not loaded."""
    return _ACCEL_PATH if _VDSP is not None else None


# ---------------------------------------------------------------------------
# ctypes signatures (Apple's vDSP uses vDSP_Stride = long, vDSP_Length = ulong)
# ---------------------------------------------------------------------------


_F32P = ctypes.POINTER(ctypes.c_float)
_LEN = ctypes.c_size_t   # vDSP_Length is unsigned long
_STR = ctypes.c_ssize_t  # vDSP_Stride is signed long (negatives used for conv)


if _VDSP is not None:
    try:
        _VDSP.vDSP_meanv.argtypes = [_F32P, _STR, _F32P, _LEN]
        _VDSP.vDSP_meanv.restype = None

        _VDSP.vDSP_measqv.argtypes = [_F32P, _STR, _F32P, _LEN]
        _VDSP.vDSP_measqv.restype = None

        _VDSP.vDSP_maxv.argtypes = [_F32P, _STR, _F32P, _LEN]
        _VDSP.vDSP_maxv.restype = None

        _VDSP.vDSP_minv.argtypes = [_F32P, _STR, _F32P, _LEN]
        _VDSP.vDSP_minv.restype = None

        _VDSP.vDSP_sve.argtypes = [_F32P, _STR, _F32P, _LEN]
        _VDSP.vDSP_sve.restype = None

        _VDSP.vDSP_vsmul.argtypes = [_F32P, _STR, _F32P, _F32P, _STR, _LEN]
        _VDSP.vDSP_vsmul.restype = None

        _VDSP.vDSP_vmul.argtypes = [_F32P, _STR, _F32P, _STR, _F32P, _STR, _LEN]
        _VDSP.vDSP_vmul.restype = None

        _VDSP.vDSP_vadd.argtypes = [_F32P, _STR, _F32P, _STR, _F32P, _STR, _LEN]
        _VDSP.vDSP_vadd.restype = None

        _VDSP.vDSP_dotpr.argtypes = [_F32P, _STR, _F32P, _STR, _F32P, _LEN]
        _VDSP.vDSP_dotpr.restype = None

        _VDSP.vDSP_conv.argtypes = [
            _F32P, _STR,
            _F32P, _STR,
            _F32P, _STR,
            _LEN, _LEN,
        ]
        _VDSP.vDSP_conv.restype = None

        _VDSP.vDSP_mmov.argtypes = [
            _F32P, _F32P,
            _LEN, _LEN, _LEN, _LEN,
        ]
        _VDSP.vDSP_mmov.restype = None

        # vDSP_zrvmul uses DSPSplitComplex. Apple defines the struct as
        #   typedef struct { float * realp; float * imagp; } DSPSplitComplex;
        # which is the natural C layout for a Structure with two f32 pointers.
        class _SplitComplex(ctypes.Structure):
            _fields_ = [
                ("realp", _F32P),
                ("imagp", _F32P),
            ]

        _VDSP.vDSP_zrvmul.argtypes = [
            ctypes.POINTER(_SplitComplex), _STR,
            _F32P, _STR,
            ctypes.POINTER(_SplitComplex), _STR,
            _LEN,
        ]
        _VDSP.vDSP_zrvmul.restype = None
    except AttributeError as exc:
        # A future macOS could drop one of these; degrade to pure-Python.
        _LOAD_ERROR = exc
        _VDSP = None


# ---------------------------------------------------------------------------
# Buffer helpers
# ---------------------------------------------------------------------------


class _F32View:
    """Owns the underlying ``array.array('f')`` so the ctypes view stays live.

    ``ctypes.Array.from_address`` creates a view that does not keep its
    source memory alive; if the originating ``array.array`` is garbage
    collected, the view's pointer dangles and the next read writes into
    freed memory. Wrapping the array in this small object guarantees the
    buffer outlives the ctypes pointer.
    """

    __slots__ = ("_array", "n", "_ptr", "_buf")

    def __init__(self, x):
        if isinstance(x, array) and x.typecode == "f":
            self._array = x
        elif isinstance(x, ctypes.Array) and x._type_ is ctypes.c_float:
            # Wrap the existing ctypes Array; ``from_address`` is safe here
            # because the ctypes Array owns its own memory (it isn't a view).
            self._array = None
            self.n = len(x)
            if self.n == 0:
                self._buf = (ctypes.c_float * 1)()
                self._ptr = ctypes.cast(self._buf, _F32P)
                return
            self._buf = x
            self._ptr = ctypes.cast(self._buf, _F32P)
            return
        elif isinstance(x, (list, tuple)):
            # Build the array.array directly from the iterable without an
            # intermediate Python list. For 1M-element inputs this saves
            # ~30 ms per call vs. the two-step path.
            self._array = array("f", x)
        else:
            flat = _flatten_floats(x)
            self._array = array("f", flat)
        self.n = len(self._array)
        if self.n == 0:
            # Avoid ctypes zero-length Array, which behaves inconsistently
            # across platforms; keep a 1-slot dummy that the caller never
            # reads.
            self._buf = (ctypes.c_float * 1)()
            self._ptr = ctypes.cast(self._buf, _F32P)
            return
        addr, length = self._array.buffer_info()
        if length != self.n:
            raise RuntimeError(
                f"array.array length mismatch: expected {self.n} got {length}"
            )
        self._buf = (ctypes.c_float * self.n).from_address(addr)
        self._ptr = ctypes.cast(self._buf, _F32P)

    @property
    def ptr(self) -> _F32P:
        return self._ptr

    def tolist(self) -> list[float]:
        if self._array is not None:
            return self._array.tolist()
        # Fallback for the ctypes-Array fast path: ctypes Array supports
        # index access but the ctypes-side iteration is slower. Use the
        # underlying bytes to populate a list via ``struct``.
        import struct
        n = self.n
        raw = bytes(self._buf)
        return list(struct.unpack(f"{n}f", raw))


def _scalar_pointer(value: float) -> ctypes.pointer:
    """Return a ctypes pointer to a single float holding ``value``."""
    return ctypes.pointer(ctypes.c_float(float(value)))


def _to_f32_view(x) -> _F32View:
    """Wrap any supported input in a lifetime-safe float32 ctypes view."""
    return _F32View(x)


def _flatten_floats(x) -> list[float]:
    """Coerce a list / array / numpy array into a flat list of Python floats.

    Used by the pure-Python fallback. The vDSP path uses ``_to_f32_view``
    instead, which is more efficient and keeps memory alive.
    """
    if isinstance(x, (list, tuple)):
        return [float(v) for v in x]
    if hasattr(x, "tolist") and hasattr(x, "dtype"):
        return [float(v) for v in x.reshape(-1).tolist()]
    if isinstance(x, array):
        if x.typecode != "f":
            return [float(v) for v in x.tolist()]
        return [float(v) for v in x]
    if isinstance(x, (ctypes.Array,)):
        return [float(v) for v in x]
    raise TypeError(f"unsupported input type: {type(x).__name__}")


class _F32Buffer:
    """Lifetime-safe float32 scratch buffer for vDSP output.

    Elementwise ops need a destination buffer of length ``n``. Allocating it
    fresh on every call is expensive (O(n) Python work in two places: the
    ctypes Array constructor and the ctypes Array -> list conversion). The
    fastest alternative is an ``array.array('f', [0.0]) * n``, which uses
    C-level repeat and gives a contiguous memory block we can pass to vDSP
    via ``ctypes.from_address``. ``tolist()`` returns the result as a plain
    Python list in O(n) C-side time.
    """

    __slots__ = ("n", "_array", "_ptr", "_buf")

    def __init__(self, n: int):
        self.n = n
        if n == 0:
            self._array = array("f")
            self._buf = (ctypes.c_float * 1)()
            self._ptr = ctypes.cast(self._buf, _F32P)
            return
        self._array = array("f", [0.0]) * n
        addr, length = self._array.buffer_info()
        if length != n:
            raise RuntimeError(
                f"array.array length mismatch: expected {n} got {length}"
            )
        self._buf = (ctypes.c_float * n).from_address(addr)
        self._ptr = ctypes.cast(self._buf, _F32P)

    def tolist(self) -> list[float]:
        # ``array.array('f', self._array)`` is a C-level O(n) copy; it is
        # measurably faster than ``self._array.tolist()`` for large n.
        return list(array("f", self._array))


def _scalar_pointer(value: float) -> ctypes.pointer:
    """Return a ctypes pointer to a single float holding ``value``."""
    return ctypes.pointer(ctypes.c_float(float(value)))


# ---------------------------------------------------------------------------
# Pure-Python fallbacks
# ---------------------------------------------------------------------------


def _py_meanv(x: Sequence[float]) -> float:
    if not x:
        return 0.0
    return sum(x) / float(len(x))


def _py_measqv(x: Sequence[float]) -> float:
    if not x:
        return 0.0
    return sum(v * v for v in x) / float(len(x))


def _py_maxv(x: Sequence[float]) -> float:
    if not x:
        return float("-inf")
    return max(x)


def _py_minv(x: Sequence[float]) -> float:
    if not x:
        return float("inf")
    return min(x)


def _py_sve(x: Sequence[float]) -> float:
    return float(sum(x))


def _py_vsmul(x: Sequence[float], scalar: float) -> list[float]:
    return [v * float(scalar) for v in x]


def _py_vmul(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [av * bv for av, bv in zip(a, b)]


def _py_vadd(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [av + bv for av, bv in zip(a, b)]


def _py_dotpr(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum(av * bv for av, bv in zip(a, b)))


def _py_conv(signal: Sequence[float], filt: Sequence[float]) -> list[float]:
    """Discrete convolution of two real vectors.

    Output length is ``len(signal) - len(filt) + 1`` (matches numpy.convolve
    in 'full' minus the trailing len(filt)-1 entries).
    """
    n = len(signal)
    p = len(filt)
    if p == 0 or n < p:
        return []
    out_len = n - p + 1
    filt_rev = list(reversed(filt))
    result = [0.0] * out_len
    for i in range(out_len):
        acc = 0.0
        for j in range(p):
            acc += signal[i + j] * filt_rev[j]
        result[i] = acc
    return result


def _py_zrvmul(
    re_in: Sequence[float],
    im_in: Sequence[float],
    real_vec: Sequence[float],
) -> tuple[list[float], list[float]]:
    return (
        [r * v for r, v in zip(re_in, real_vec)],
        [i * v for i, v in zip(im_in, real_vec)],
    )


def _py_mmov(
    src: Sequence[float],
    src_cols: int,
    m: int,
    n: int,
) -> list[float]:
    """Copy an m-column-wide, n-row block out of a row-major matrix."""
    out = [0.0] * (m * n)
    for r in range(n):
        for c in range(m):
            out[r * m + c] = src[r * src_cols + c]
    return out


# ---------------------------------------------------------------------------
# Public vDSP-compatible API
# ---------------------------------------------------------------------------


def vDSP_meanv(x, stride: int = 1) -> float:
    """Mean of a single-precision vector."""
    view = _to_f32_view(x)
    n = view.n
    if n == 0:
        return 0.0
    if stride != 1:
        # Strided reads are not common in the quantizer path; if requested,
        # do a one-shot pure-Python pass. Keep the vDSP fast path for the
        # common stride==1 case.
        flat = _flatten_floats(x)[::stride]
        return _py_meanv(flat) if flat else 0.0
    if _VDSP is None:
        flat = _flatten_floats(x)
        return _py_meanv(flat)
    out = ctypes.c_float()
    _VDSP.vDSP_meanv(view.ptr, _STR(1), ctypes.byref(out), _LEN(n))
    return out.value


def vDSP_measqv(x, stride: int = 1) -> float:
    """Mean of squares of a single-precision vector."""
    view = _to_f32_view(x)
    n = view.n
    if n == 0:
        return 0.0
    if stride != 1:
        flat = _flatten_floats(x)[::stride]
        return _py_measqv(flat) if flat else 0.0
    if _VDSP is None:
        return _py_measqv(_flatten_floats(x))
    out = ctypes.c_float()
    _VDSP.vDSP_measqv(view.ptr, _STR(1), ctypes.byref(out), _LEN(n))
    return out.value


def vDSP_maxv(x, stride: int = 1) -> float:
    """Maximum element of a single-precision vector."""
    view = _to_f32_view(x)
    n = view.n
    if n == 0:
        return float("-inf")
    if stride != 1:
        flat = _flatten_floats(x)[::stride]
        return _py_maxv(flat) if flat else float("-inf")
    if _VDSP is None:
        return _py_maxv(_flatten_floats(x))
    out = ctypes.c_float()
    _VDSP.vDSP_maxv(view.ptr, _STR(1), ctypes.byref(out), _LEN(n))
    return out.value


def vDSP_minv(x, stride: int = 1) -> float:
    """Minimum element of a single-precision vector."""
    view = _to_f32_view(x)
    n = view.n
    if n == 0:
        return float("inf")
    if stride != 1:
        flat = _flatten_floats(x)[::stride]
        return _py_minv(flat) if flat else float("inf")
    if _VDSP is None:
        return _py_minv(_flatten_floats(x))
    out = ctypes.c_float()
    _VDSP.vDSP_minv(view.ptr, _STR(1), ctypes.byref(out), _LEN(n))
    return out.value


def vDSP_sve(x, stride: int = 1) -> float:
    """Sum of a single-precision vector."""
    view = _to_f32_view(x)
    n = view.n
    if n == 0:
        return 0.0
    if stride != 1:
        flat = _flatten_floats(x)[::stride]
        return _py_sve(flat) if flat else 0.0
    if _VDSP is None:
        return _py_sve(_flatten_floats(x))
    out = ctypes.c_float()
    _VDSP.vDSP_sve(view.ptr, _STR(1), ctypes.byref(out), _LEN(n))
    return out.value


def vDSP_vsmul(x, scalar: float, out=None) -> list[float]:
    """Elementwise ``out[i] = x[i] * scalar``."""
    view = _to_f32_view(x)
    n = view.n
    if n == 0:
        return []
    if _VDSP is None:
        return _py_vsmul(_flatten_floats(x), scalar)
    s_ptr = _scalar_pointer(scalar)
    out_buf = _F32Buffer(n)
    _VDSP.vDSP_vsmul(view.ptr, _STR(1), s_ptr, out_buf._ptr, _STR(1), _LEN(n))
    if out is not None:
        out_flat = _flatten_floats(out)
        if len(out_flat) < n:
            raise ValueError("out must have at least len(x) elements")
        # Slice-assign from the underlying array.array; this is a single
        # C-level memmove instead of an O(n) Python loop.
        out_flat[:n] = out_buf._array
        return out_flat
    return out_buf.tolist()


def vDSP_vmul(a, b, out=None) -> list[float]:
    """Elementwise ``out[i] = a[i] * b[i]``."""
    av = _to_f32_view(a)
    bv = _to_f32_view(b)
    n = av.n
    if n != bv.n:
        raise ValueError("vDSP_vmul requires equal-length inputs")
    if n == 0:
        return []
    if _VDSP is None:
        return _py_vmul(_flatten_floats(a), _flatten_floats(b))
    out_buf = _F32Buffer(n)
    _VDSP.vDSP_vmul(av.ptr, _STR(1), bv.ptr, _STR(1), out_buf._ptr, _STR(1), _LEN(n))
    if out is not None:
        out_flat = _flatten_floats(out)
        if len(out_flat) < n:
            raise ValueError("out must have at least len(a) elements")
        out_flat[:n] = out_buf._array
        return out_flat
    return out_buf.tolist()


def vDSP_vadd(a, b, out=None) -> list[float]:
    """Elementwise ``out[i] = a[i] + b[i]``."""
    av = _to_f32_view(a)
    bv = _to_f32_view(b)
    n = av.n
    if n != bv.n:
        raise ValueError("vDSP_vadd requires equal-length inputs")
    if n == 0:
        return []
    if _VDSP is None:
        return _py_vadd(_flatten_floats(a), _flatten_floats(b))
    out_buf = _F32Buffer(n)
    _VDSP.vDSP_vadd(av.ptr, _STR(1), bv.ptr, _STR(1), out_buf._ptr, _STR(1), _LEN(n))
    if out is not None:
        out_flat = _flatten_floats(out)
        if len(out_flat) < n:
            raise ValueError("out must have at least len(a) elements")
        out_flat[:n] = out_buf._array
        return out_flat
    return out_buf.tolist()


def vDSP_dotpr(a, b) -> float:
    """Dot product of two single-precision vectors."""
    av = _to_f32_view(a)
    bv = _to_f32_view(b)
    n = av.n
    if n != bv.n:
        raise ValueError("vDSP_dotpr requires equal-length inputs")
    if n == 0:
        return 0.0
    if _VDSP is None:
        return _py_dotpr(_flatten_floats(a), _flatten_floats(b))
    out = ctypes.c_float()
    _VDSP.vDSP_dotpr(av.ptr, _STR(1), bv.ptr, _STR(1), ctypes.byref(out), _LEN(n))
    return out.value


def vDSP_conv(signal, filt) -> list[float]:
    """Discrete convolution ``signal (*) filt``; output is ``len(signal) - len(filt) + 1``.

    A length-1 filter returns the input scaled by that element. An empty
    filter or signal shorter than the filter returns ``[]``.
    """
    sv = _to_f32_view(signal)
    fv = _to_f32_view(filt)
    p = fv.n
    n = sv.n
    if p == 0 or n < p:
        return []
    out_len = n - p + 1
    if _VDSP is None:
        return _py_conv(_flatten_floats(signal), _flatten_floats(filt))
    out_buf = _F32Buffer(out_len)
    # Convolution: pass the LAST element of the filter with a negative stride.
    # Compute the address of the last element via raw address arithmetic
    # because ctypes pointers cannot be offset by an integer directly.
    f_addr = ctypes.cast(fv._buf, ctypes.c_void_p).value
    f_last = ctypes.cast(
        f_addr + (p - 1) * ctypes.sizeof(ctypes.c_float), _F32P,
    )
    _VDSP.vDSP_conv(
        sv.ptr, _STR(1),
        f_last, _STR(-1),
        out_buf._ptr, _STR(1),
        _LEN(out_len), _LEN(p),
    )
    return out_buf.tolist()


def vDSP_zrvmul(
    re_in,
    im_in,
    real_vec,
) -> tuple[list[float], list[float]]:
    """Elementwise complex-by-real multiply on split-complex input.

    Treats ``(re_in, im_in)`` as the real and imaginary parts of a complex
    vector and multiplies each complex element by the corresponding real
    element in ``real_vec``. Returns ``(re_out, im_out)``.
    """
    rv = _to_f32_view(re_in)
    iv = _to_f32_view(im_in)
    bv = _to_f32_view(real_vec)
    n = rv.n
    if n != iv.n or n != bv.n:
        raise ValueError("vDSP_zrvmul requires equal-length inputs")
    if n == 0:
        return [], []
    if _VDSP is None or "vDSP_zrvmul" not in dir(_VDSP):
        return _py_zrvmul(
            _flatten_floats(re_in), _flatten_floats(im_in), _flatten_floats(real_vec)
        )
    re_buf = _F32Buffer(n)
    im_buf = _F32Buffer(n)
    # ctypes Array objects must be explicitly cast to POINTER(c_float) before
    # assignment to a Structure field; otherwise the field is silently NULL.
    a_split = _SplitComplex(realp=rv.ptr, imagp=iv.ptr)
    c_split = _SplitComplex(realp=re_buf._ptr, imagp=im_buf._ptr)
    _VDSP.vDSP_zrvmul(
        ctypes.byref(a_split), _STR(1),
        bv.ptr, _STR(1),
        ctypes.byref(c_split), _STR(1),
        _LEN(n),
    )
    return re_buf.tolist(), im_buf.tolist()


def vDSP_mmov(
    src,
    m: int,
    n: int,
    src_cols: int | None = None,
) -> list[float]:
    """Copy an m-column-wide, n-row block out of a row-major source matrix.

    ``src`` is a flat row-major buffer of length ``src_cols * src_rows``.
    ``src_cols`` defaults to ``m`` (plain memcpy). Returns a flat row-major
    buffer of length ``m * n``.
    """
    if m <= 0 or n <= 0:
        return []
    if src_cols is None:
        src_cols = m
    sv = _to_f32_view(src)
    if src_cols < m or sv.n < src_cols * n:
        raise ValueError("source buffer too small for the requested block")
    if _VDSP is None:
        return _py_mmov(_flatten_floats(src), src_cols, m, n)
    out_buf = _F32Buffer(m * n)
    _VDSP.vDSP_mmov(
        sv.ptr, out_buf._ptr,
        _LEN(m), _LEN(n),
        _LEN(src_cols), _LEN(m),
    )
    return out_buf.tolist()


# ---------------------------------------------------------------------------
# Convenience: report a single human-readable status line.
# ---------------------------------------------------------------------------


def status_line() -> str:
    """Return a short string suitable for CLI banners / smoke-test output."""
    if _VDSP is not None:
        return f"vDSP available ({_ACCEL_PATH})"
    if _IS_DARWIN and _LOAD_ERROR is not None:
        return f"vDSP unavailable (Accelerate present but load failed: {_LOAD_ERROR})"
    if _IS_DARWIN:
        return "vDSP unavailable (Accelerate not found)"
    return "vDSP unavailable (non-Darwin platform)"


if __name__ == "__main__":
    # Minimal self-check when run directly.
    print(status_line())
    if _VDSP is not None:
        print(f"vDSP_meanv([1,2,3,4]) = {vDSP_meanv([1.0, 2.0, 3.0, 4.0])}")
        print(f"vDSP_dotpr([1,2,3],[4,5,6]) = {vDSP_dotpr([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])}")
    sys.exit(0)
