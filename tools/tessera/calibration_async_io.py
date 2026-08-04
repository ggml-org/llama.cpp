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

    def read(self, path: str | os.PathLike) -> "queue.Queue":
        """Issue an async read of ``path`` and return a queue.

        The caller pops one ``(bytes, error)`` from the
        queue.  ``bytes`` is the file's contents (or None
        on error); ``error`` is the error message (or
        None on success).  The queue is filled exactly
        once per call.
        """
        result_queue: "queue.Queue" = queue.Queue(maxsize=1)
        path_bytes = os.fspath(path).encode("utf-8")

        # The user pointer must remain alive until the
        # callback fires.  We attach a holder object so
        # the ctypes handle isn't GC'd while the GCD
        # callback is in flight.  The holder owns a
        # strong reference to ``self`` (the backend) and
        # to the result queue.
        class Holder:
            def __init__(self, backend, q):
                self.backend = backend
                self.q = q
                # The callback closure references ``self``,
                # so the Holder stays alive until the
                # callback fires.

        holder = Holder(self, result_queue)

        @_CALLBACK_TYPE
        def callback(data_ptr, size, _user):
            # ``holder`` is captured from the enclosing
            # scope; the ctypes CFUNCTYPE keeps the closure
            # alive until the callback is replaced.  The
            # holder references the result queue so the GC
            # doesn't collect it before the GCD callback
            # fires.
            try:
                if not data_ptr:
                    holder.q.put((None, "dispatch_io_read failed"))
                    return
                addr = ctypes.cast(data_ptr, ctypes.c_void_p).value
                if not addr or size == 0:
                    holder.q.put((b"", None))
                    self._lib.tessera_dispatch_free_buffer(data_ptr)
                    return
                data = (ctypes.c_char * size).from_address(addr)
                bs = bytes(data)
                holder.q.put((bs, None))
            finally:
                # The buffer is no longer needed; free it
                # (whether or not we successfully converted
                # to a Python bytes object).
                if data_ptr:
                    self._lib.tessera_dispatch_free_buffer(data_ptr)

        # The callback must remain alive for the duration
        # of the read.  We attach it to the holder so the
        # GC doesn't collect it before the GCD callback
        # fires.
        holder.callback = callback

        with self._lock:
            rc = self._lib.tessera_dispatch_read_file(
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


__all__ = [
    "create_async_io_backend",
    "AsyncIOBackend",
]
