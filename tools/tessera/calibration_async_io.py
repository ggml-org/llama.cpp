"""Apple macOS async I/O via ``dispatch_io_t`` (GCD).

The legacy ``CalibPipeline`` in
``tools/tessera/calibration_memory.py`` uses a Python producer
thread that calls ``np.load(mmap_mode="r")`` synchronously.
On macOS the replacement uses libdispatch's
``dispatch_io_t`` to issue the read on a background GCD
queue; the consumer's compute overlaps with the next layer's
read.  This is the same producer/consumer pattern as the
threaded path, but the I/O scheduler is GCD instead of the
Python thread pool.

Why a C bridge: ``dispatch_io_t`` is a C-only API; there is
no Python binding in any maintained library (the ``dispatch``
PyObjC binding is not on PyPI for Python 3.14+ on Apple
Silicon).  The cheapest path is a small Objective-C++ wrapper
(``apple_dispatch_io.mm``) compiled to a ``.dylib`` on demand
and loaded via ctypes.  The bridge exposes one entry:

  * ``tessera_dispatch_read_file(path, callback, user)``:
    queues a ``dispatch_io_create_with_path`` +
    ``dispatch_io_read`` on a process-wide GCD queue.  The
    callback fires on the GCD queue with the file's bytes.

The Python wrapper in this module hands the bytes to numpy
(via ``np.frombuffer``) and delivers the resulting array to
a ``queue.Queue`` for the consumer thread.  ``CalibPipelineAsync``
(the actual pipeline class) lives in ``calibration_memory.py``;
this module is just the I/O backend.

On Linux/Windows: ``create_async_io_backend()`` returns
``None``.  ``CalibPipelineAsync`` falls back to
``CalibPipeline`` (the legacy threaded path) on those
platforms.
"""

from __future__ import annotations

import ctypes
import os
import platform
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# C bridge build
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent
_BUILD_DIR = _ROOT / ".build"
_DISPATCH_SOURCE = _ROOT / "apple_dispatch_io.mm"
_DEFAULT_DISPATCH_LIBRARY = _BUILD_DIR / "libtessera_dispatch_io.dylib"

_CALLBACK_TYPE = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p,    # data pointer (heap buffer; bridge owns it until callback returns)
    ctypes.c_size_t,    # size in bytes
    ctypes.c_int,       # error code (0 = success including empty; non-zero = error)
    ctypes.c_void_p,    # user pointer (Python object via ctypes)
)


def _is_macos() -> bool:
    return platform.system() == "Darwin"


def _has_clang() -> bool:
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


def _build_dispatch_library(output: Path) -> Optional[Path]:
    if not _DISPATCH_SOURCE.is_file():
        return None
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(
            [
                "xcrun", "clang++",
                "-std=c++17", "-O3",
                "-dynamiclib",
                str(_DISPATCH_SOURCE),
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


def _load_dispatch_library(library: Path) -> Optional[ctypes.CDLL]:
    try:
        lib = ctypes.CDLL(str(library))
    except OSError:
        return None
    try:
        read_file = lib.tessera_dispatch_read_file
        read_file.restype = ctypes.c_int
        read_file.argtypes = [
            ctypes.c_char_p,
            ctypes.c_void_p,    # callback (c_void_p, cast to function ptr)
            ctypes.c_void_p,    # user pointer
        ]
        free_buffer = lib.tessera_dispatch_free_buffer
        free_buffer.restype = None
        free_buffer.argtypes = [ctypes.c_void_p]
    except AttributeError:
        return None
    return lib


# Process-wide singleton state.  Each test that creates
# a new ``AsyncIOBackend`` would leak the ctypes handle
# (the GCD queue is shared via ``dispatch_once`` but the
# Python wrapper has its own state).  The singleton is
# created on the first call to ``create_async_io_backend``
# and reused thereafter.
_BACKEND_SINGLETON: Optional["AsyncIOBackend"] = None
_BACKEND_LOCK = threading.Lock()


def create_async_io_backend() -> Optional["AsyncIOBackend"]:
    """Return the macOS dispatch_io_t backend, or None on other
    platforms / when the build fails.

    The detection is process-wide: the first call pays the
    build / load cost; subsequent calls hit the cache.  Used
    by ``CalibPipelineAsync`` to gate the async path.  The
    backend is a process-wide singleton (one per
    process); creating a new instance per call would
    leak the underlying ctypes library handles.
    """
    global _BACKEND_SINGLETON
    if not _is_macos() or not _has_clang():
        return None
    with _BACKEND_LOCK:
        if _BACKEND_SINGLETON is not None:
            return _BACKEND_SINGLETON
        library = Path(
            os.environ.get("TESSERA_DISPATCH_IO_LIBRARY",
                           _DEFAULT_DISPATCH_LIBRARY)
        )
        if not library.is_file() or (
            _DISPATCH_SOURCE.is_file()
            and library.stat().st_mtime < _DISPATCH_SOURCE.stat().st_mtime
        ):
            if not _build_dispatch_library(library):
                return None
        lib = _load_dispatch_library(library)
        if lib is None:
            return None
        _BACKEND_SINGLETON = AsyncIOBackend(library=library, _lib=lib)
        return _BACKEND_SINGLETON


# ---------------------------------------------------------------------------
# Async I/O backend
# ---------------------------------------------------------------------------


class AsyncIOBackend:
    """macOS dispatch_io_t backend for per-layer reads.

    A single instance is process-wide: the GCD queue is
    process-scoped, and creating multiple instances would
    just create multiple GCD queues (the libdispatch
    scheduler serialises them anyway).  The backend exposes
    one method, ``read(path)``, which returns a
    ``queue.Queue`` that the consumer thread can poll.  The
    queue yields ``(bytes, error_message_or_none)`` tuples;
    a single tuple per file.

    The bridge hands the GCD-allocated buffer to the
    callback.  We wrap the bytes in a numpy array (the
    consumer uses ``np.load``-style access), and free the
    buffer when the consumer is done.  This is the
    "give the bytes to numpy once" pattern; the legacy
    ``np.load(mmap_mode="r")`` mmap'd the zip from disk on
    demand, which is more memory-efficient for huge bundles
    but does not benefit from the async I/O overlap.
    """

    def __init__(self, library: Path, _lib: ctypes.CDLL) -> None:
        self.library = library
        self._lib = _lib
        self._lock = threading.Lock()
        # Keepalive list for in-flight CFuncPtrs.  Each
        # ``read()`` appends its callback here; the
        # callback is NOT removed.  This is intentional
        # (see ``read()`` docstring for the GC / use-after-
        # free rationale).  For long-running processes
        # call ``cleanup()`` periodically to bound the
        # leak.
        self._in_flight: list = []

    def read(self, path: str | os.PathLike) -> "queue.Queue":
        """Issue an async read of ``path`` and return a queue.

        The caller pops one ``(bytes, error)`` from the
        queue.  ``bytes`` is the file's contents (or None
        on error); ``error`` is the error message (or
        None on success).  The queue is filled exactly
        once per call.

        GC / use-after-free note: the CFuncPtr returned
        by ``_CALLBACK_TYPE`` MUST stay alive for as long
        as GCD holds the function pointer (until the
        dispatch_io_read block fires).  The previous
        design attached the callback to a per-read
        ``Holder`` object, which formed a cycle (callback
        -> closure -> holder -> callback).  Python's GC
        collected the cycle and freed the CFuncPtr, while
        GCD was still in flight; the next call into the
        libffi trampoline segfaulted inside
        ``closure_fcn`` (the trace ended at
        ``__tessera_dispatch_read_file_block_invoke_2``).
        The fix attaches the callback to ``self._in_flight``
        (a backend attribute, reachable from the module's
        singleton); the GC won't collect a cycle that
        includes a module-reachable object, so the
        CFuncPtr stays alive for the process lifetime.
        The documented cost is a small per-read leak;
        ``cleanup()`` clears the list.
        """
        result_queue: "queue.Queue" = queue.Queue(maxsize=1)
        path_bytes = os.fspath(path).encode("utf-8")
        # Capture the keepalive list and the lib as
        # locals so the callback closure does NOT capture
        # ``self`` (avoiding a tight cycle; the cycle via
        # ``self._in_flight`` -> list -> callback is
        # intentionally kept because ``self._in_flight``
        # is reachable from the module via the backend
        # singleton).
        in_flight = self._in_flight
        lib = self._lib

        @_CALLBACK_TYPE
        def callback(data_ptr, size, error, _user):
            try:
                if error != 0:
                    # Real I/O error (the C bridge now
                    # carries the dispatch_io error code
                    # through, so a NULL data pointer with
                    # error==0 is success-with-empty, not
                    # an error).
                    result_queue.put((
                        None,
                        f"dispatch_io_read failed: errno={error}",
                    ))
                    return
                if not data_ptr or size == 0:
                    # Success-with-empty (data is NULL, size
                    # is 0, error is 0).  This is now
                    # unambiguous because the C bridge
                    # passes error=0 for success and the
                    # check above already filtered the
                    # error!=0 case.
                    result_queue.put((b"", None))
                    return
                # ``data_ptr`` is a c_void_p in the CFUNCTYPE
                # argtypes, but on Python 3.14 + ctypes the
                # C trampoline hands the raw integer address
                # to the Python callable in some cases.  Use
                # the cast-then-value pattern (same as the
                # original implementation) so the code works
                # for either representation.
                addr = ctypes.cast(data_ptr, ctypes.c_void_p).value
                # Build a c_char array VIEW and copy into a
                # Python bytes object so we own the bytes
                # (the C buffer is freed in the finally
                # block).
                data_array = (ctypes.c_char * size).from_address(addr)
                bs = bytes(data_array)
                result_queue.put((bs, None))
            finally:
                # The buffer is no longer needed; free it
                # (whether or not we successfully converted
                # to a Python bytes object).  Guarded with
                # try/except so a double-free (e.g. from a
                # C-side bug) doesn't mask the original
                # error.  On error the C bridge did not
                # allocate a buffer, so data_ptr is NULL
                # and the free is a no-op.
                if data_ptr:
                    try:
                        lib.tessera_dispatch_free_buffer(data_ptr)
                    except Exception:
                        pass

        in_flight.append(callback)

        with self._lock:
            rc = lib.tessera_dispatch_read_file(
                path_bytes,
                ctypes.cast(callback, ctypes.c_void_p),
                None,
            )
        if rc != 0:
            # The bridge rejected the call (e.g. the path
            # is invalid).  Deliver a synchronously-queued
            # error so the consumer doesn't block forever.
            result_queue.put((None, f"tessera_dispatch_read_file rc={rc}"))
        return result_queue

    def cleanup(self) -> None:
        """Drop all keepalive refs to completed CFuncPtrs.

        Safe to call after the last ``read()`` whose queue
        you care about has been drained.  Clears
        ``self._in_flight``; the CFuncPtrs are then
        GC-eligible.  Do NOT call while a read is in flight:
        the next GCD invocation would call into a freed
        trampoline and segfault.
        """
        self._in_flight.clear()


__all__ = [
    "create_async_io_backend",
    "AsyncIOBackend",
]
