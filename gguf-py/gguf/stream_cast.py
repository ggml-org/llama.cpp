# gguf-py/gguf/stream_cast.py
from __future__ import annotations
from typing import Any
import os
import sys
import numpy as np


def _slog(msg: str) -> None:
    """Conditional debug logging when GGUF_STREAM_LOG is set."""
    if os.environ.get("GGUF_STREAM_LOG"):
        print(f"[gguf-stream] {msg}", file=sys.stdout, flush=True)


def _chunk_elems(src_dtype: np.dtype, dst_dtype: np.dtype, chunk_mb: int) -> int:
    """
    Compute how many elements to process per chunk so that each chunk is
    approximately `chunk_mb` MiB of the *larger* of the source/destination itemsize.
    """
    try:
        mb = int(chunk_mb)
    except Exception:
        mb = 64
    mb = max(1, mb)
    item = max(np.dtype(src_dtype).itemsize, np.dtype(dst_dtype).itemsize)
    return max(1, (mb * 1024 * 1024) // item)


def write_cast(fout, src: np.ndarray, dst_dtype: Any, chunk_mb: int) -> None:
    """
    Stream `src.astype(dst_dtype)` to `fout` in fixed-size chunks to cap peak RSS.

    This matches the import site in lazy.py:
        from .stream_cast import write_cast

    Parameters
    ----------
    fout : file-like object
        Open file handle to write bytes to (must support .write()).
    src : np.ndarray
        Source ndarray to be converted and streamed.
    dst_dtype : Any
        Target dtype (anything accepted by np.dtype).
    chunk_mb : int
        Desired chunk size in MiB (will be clamped to >= 1).
    """
    dst = np.dtype(dst_dtype)
    flat = src.reshape(-1)
    n = flat.size
    ce = _chunk_elems(flat.dtype, dst, chunk_mb)

    _slog(
        f"write_cast: src={flat.dtype} -> dst={dst}; n={n}; "
        f"chunk={max(1, int(chunk_mb))} MiB; elems/chunk={ce}"
    )

    start = 0
    # local binding for tiny speed bump
    mv = memoryview
    while start < n:
        end = min(start + ce, n)
        # copy=False avoids an extra tmp when possible
        chunk = flat[start:end].astype(dst, copy=False)
        fout.write(mv(chunk).tobytes())
        start = end


# Optional: writer-side API that accepts chunk size in bytes (used by gguf_writer)
def stream_write(fout, src_arr: np.ndarray, dst_dtype: Any, chunk_bytes: int) -> None:
    """
    Same as write_cast, but the chunk size is given in bytes.
    Kept for compatibility with earlier helper drafts.
    """
    if not isinstance(chunk_bytes, int) or chunk_bytes <= 0:
        chunk_mb = 64
    else:
        # round bytes to MiB for the element count helper
        chunk_mb = max(1, chunk_bytes // (1024 * 1024))

    write_cast(fout, src_arr, dst_dtype, chunk_mb)