"""Regression tests for the ``dispatch_io_t`` async-I/O bridge.

Catches the use-after-free that used to segfault the multi-file
pytest sweep.  The previous design attached the CFuncPtr to a
per-read ``Holder`` object; the Holder was in a cycle, the GC
collected it, the CFuncPtr (and its libffi trampoline) was freed
while GCD still held the function pointer, and the next GCD
invocation segfaulted inside ``closure_fcn`` (the ctypes C
function that calls the Python callable).  Trace ended at
``__tessera_dispatch_read_file_block_invoke_2``.

The fix moves the CFuncPtr to ``AsyncIOBackend._in_flight`` (a
backend attribute, reachable from the module's singleton).  The
GC does not collect a cycle that includes a module-reachable
object, so the CFuncPtr stays alive for the process lifetime.
The cost is a small per-read leak, bounded by ``cleanup()``.

These tests issue many reads in quick succession to maximise the
chance of the GC running while a read is in flight.  Without the
fix the test process segfaults.
"""

from __future__ import annotations

import gc
import platform
import unittest

import pytest


def _is_macos() -> bool:
    return platform.system() == "Darwin"


@pytest.mark.skipif(not _is_macos(), reason="dispatch_io bridge is macOS-only")
def test_dispatch_io_many_concurrent_reads_no_segfault(tmp_path) -> None:
    """Issue many reads; force GC between them; verify no crash.

    Pre-fix: the test process segfaults inside the ctypes
    ``closure_fcn`` because the GC frees the CFuncPtr while
    GCD is still in flight.

    Post-fix: the CFuncPtr is held by ``AsyncIOBackend._in_flight``,
    which is reachable from the module via the backend
    singleton.  The cycle is not collected.
    """
    from calibration_async_io import create_async_io_backend

    backend = create_async_io_backend()
    if backend is None:
        pytest.skip("dispatch_io bridge not available (build failed)")

    test_file = tmp_path / "test.bin"
    test_file.write_bytes(b"hello world" * 100)

    queues = []
    for _ in range(50):
        q = backend.read(test_file)
        queues.append(q)
        # Force a GC after every few reads to maximise the
        # chance of catching the pre-fix use-after-free.
        if len(queues) % 5 == 0:
            gc.collect()

    for q in queues:
        data, error = q.get(timeout=10)
        assert error is None, error
        assert data == b"hello world" * 100

    backend.cleanup()


@pytest.mark.skipif(not _is_macos(), reason="dispatch_io bridge is macOS-only")
def test_dispatch_io_zero_size_read(tmp_path) -> None:
    """An empty file should now return (b"", None) (success),
    not surface as a failure.  The C bridge carries the
    dispatch_io error code through to Python, so a NULL data
    pointer with error=0 is success-with-empty-data, not a
    real I/O error.
    """
    from calibration_async_io import create_async_io_backend

    backend = create_async_io_backend()
    if backend is None:
        pytest.skip("dispatch_io bridge not available")

    test_file = tmp_path / "empty.bin"
    test_file.write_bytes(b"")

    q = backend.read(test_file)
    data, error = q.get(timeout=10)
    assert error is None, error
    assert data == b""

    backend.cleanup()


@pytest.mark.skipif(not _is_macos(), reason="dispatch_io bridge is macOS-only")
def test_dispatch_io_error_carries_through(tmp_path) -> None:
    """A read failure (file does not exist) should deliver an
    error tuple with error != None.  Verifies the C bridge
    actually carries the dispatch_io error code to Python
    (the fix from the prior commit).
    """
    from calibration_async_io import create_async_io_backend

    backend = create_async_io_backend()
    if backend is None:
        pytest.skip("dispatch_io bridge not available")

    q = backend.read(tmp_path / "does_not_exist.bin")
    data, error = q.get(timeout=10)
    assert data is None
    assert error is not None
    # The error message should mention the errno (proves the
    # error code was carried through from the C bridge).
    assert "errno=" in error, error

    backend.cleanup()
