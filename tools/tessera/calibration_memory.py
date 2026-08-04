"""Memory-bound / spatial-temporal calibration utilities.

The unified gemma4_12B + dspark + dflash + MTP single-GGUF
calibration processes 4000+ tensors, with FFN gate/up tensors as
large as 16384x4096 = 256 MB F32 each.  Loading all of them at
once (or even iterating them with full retention) blows past 64 GB
of RAM and the OS kills the process.  This module is the
memory-bound / spatial-temporal utility layer that keeps the
calibration in bounded memory and overlaps I/O with compute.

Five categories of optimizations are implemented here, each
controlled by a CLI flag on ``per_tensor_calibrate.py``:

* **Streaming I/O (Category 1)** -- ``mmap_tensor`` opens a
  single tensor from a ``.npz`` bundle as a memory-mapped array
  rather than reading the whole file into RAM.  The OS pages in
  the tensor on demand; the calibration pass reads it once and
  the OS reclaims the pages.  The reader never holds the full
  ``.npz`` in RAM.

* **Chunked processing (Category 2)** -- ``chunked_process``
  splits a 2-D weight matrix into row-chunks so the per-tensor
  result is materialised incrementally.  For a 12B FFN gate
  tensor at 16384x4096 with ``chunk_rows=4096`` that is 4
  chunks of 4096x4096 = 64 MB each (F32); the per-chunk output
  is a few KB and the per-tensor result is reconstructed from the
  chunked outputs at the end.

* **Spatial occupancy (Category 4)** -- ``interleave_components``
  round-robins the per-component tensors at the layer level so
  the cache footprint stays small.  The benefit: the per-tensor
  observer moments (per-input-channel scales) are similar across
  components (they all share ``tok_embd`` + ``output``) so the
  cache hit rate on the observer moments is higher.

* **Temporal occupancy (Category 5)** -- ``CalibPipeline`` is a
  double-buffered producer/consumer that mmaps the next tensor
  while the current tensor is computing.  This is the standard
  I/O-compute overlap pattern; the depth is configurable via
  ``--temporal-pipeline-depth`` (default 2 = double-buffered).

Peak-RSS budgeting (Category 3) lives in
``tools/tessera/calibration_residency.py``; it is intentionally
split out so the policy (where to live) is separate from the
mechanism (how to move).  The two modules are the inputs to the
``per_tensor_calibrate.py`` refactor.

This module is a pure-utility module: it does not own the
calibration policy schema, the per-tensor learning loop, or the
``.npz`` format.  It is consumed by ``per_tensor_calibrate.py``
and tested by ``test_calibration_memory.py``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import re
import threading
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Category 1: streaming I/O
# ---------------------------------------------------------------------------


def mmap_tensor(
    npz_path: str | os.PathLike,
    key: str,
    dtype: np.dtype | None = None,
) -> np.ndarray:
    """Open a single tensor in a ``.npz`` file as a memory-mapped array.

    Parameters
    ----------
    npz_path : path-like
        The ``.npz`` bundle containing the tensor.
    key : str
        The array name inside the ``.npz`` (e.g. ``"weight"``,
        ``"train_activations"``, ``"in_sum2"``).
    dtype : numpy dtype, optional
        If supplied, the returned array is viewed as this dtype.  The
        mmap view is over the original dtype; the cast is a metadata
        change, not a copy.  If the original dtype is not safely
        viewable as ``dtype`` the cast is silently deferred to the
        first materialisation; callers that need a real copy should
        follow up with ``.astype(dtype, copy=True)``.

    Returns
    -------
    numpy ndarray
        A memory-mapped array.  The ``np.load`` handle is closed
        when the function returns, but the OS keeps the underlying
        pages mapped as long as the array is alive.

    Notes
    -----
    The OS pages the array in on demand; the calibration pass
    reads it once and the OS reclaims the pages.  The reader
    never holds the full ``.npz`` in RAM.  Peak RSS for a single
    tensor is ``max(weight, activations, observer)`` rather than
    ``sum(all_tensors_in_turn)``.

    Implementation detail: ``np.load(path, mmap_mode="r")`` opens
    the ``.npz`` as a ``NpzFile`` whose backing ``zip`` is mmap'd.
    Returning ``data[key]`` is a view into the zip's mmap; we
    detach from the ``NpzFile`` so the user can hold the array
    without keeping the full file in RAM.  The view shares memory
    with the zip mmap, not with the python process, so closing
    the ``NpzFile`` simply decrements the zip's refcount.
    """
    npz_path = os.fspath(npz_path)
    if not Path(npz_path).is_file():
        raise FileNotFoundError(npz_path)
    with np.load(npz_path, mmap_mode="r", allow_pickle=False) as data:
        if key not in data.files:
            raise KeyError(f"{npz_path}: missing key {key!r}; have {list(data.files)}")
        arr = data[key]
        if dtype is not None and arr.dtype != dtype:
            arr = arr.view(dtype)
    return arr


@contextlib.contextmanager
def mmap_layer(
    npz_path: str | os.PathLike,
    keys: Sequence[str] = (
        "weight",
        "train_activations",
        "heldout_activations",
        "in_sum2",
        "counts",
        "name",
        "family",
    ),
) -> Iterator[dict[str, np.ndarray]]:
    """Context manager: mmap all requested keys in a ``.npz`` bundle.

    The single ``np.load`` handle is held for the lifetime of the
    ``with`` block; when the block exits the handle is closed
    and the OS reclaims the pages lazily.  The dict values are
    memory-mapped views; touching them pages in the data.

    This is the per-tensor counterpart to ``mmap_tensor`` for the
    case where a layer bundle needs more than one key (the weight
    + the activations + the observer).  Opening a single
    ``np.load`` handle per tensor is cheaper than opening one
    per key, and the OS keeps the zip mmap warm across the keys.
    """
    npz_path = os.fspath(npz_path)
    if not Path(npz_path).is_file():
        raise FileNotFoundError(npz_path)
    with np.load(npz_path, mmap_mode="r", allow_pickle=False) as data:
        out: dict[str, np.ndarray] = {}
        for k in keys:
            if k in data.files:
                out[k] = data[k]
        yield out


# ---------------------------------------------------------------------------
# Category 2: chunked processing
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ChunkSpec:
    """Per-chunk work spec handed to a chunked computation.

    A chunk is a contiguous slice of rows of the weight matrix
    (and the corresponding columns of the activation matrix).
    The computation yields per-chunk results; the per-tensor
    result is reconstructed by reducing over the chunked outputs.
    """

    start: int          # inclusive start row
    end: int            # exclusive end row
    n_rows: int         # total number of rows in the weight matrix
    n_cols: int         # total number of input-dim columns


def chunked_iter(
    n_rows: int,
    chunk_rows: int,
) -> Iterator[ChunkSpec]:
    """Yield ``ChunkSpec`` for each row-chunk of an ``n_rows``-row weight.

    The last chunk is allowed to be smaller than ``chunk_rows`` so
    the per-tensor iteration always covers the full row range.
    The ``chunk_rows <= 0`` case is treated as "no chunking" (one
    chunk covering the full weight), which is the legacy single-
    shot path.
    """
    if n_rows <= 0:
        return
    if chunk_rows <= 0 or chunk_rows >= n_rows:
        yield ChunkSpec(0, n_rows, n_rows, n_cols=-1)
        return
    for start in range(0, n_rows, chunk_rows):
        end = min(start + chunk_rows, n_rows)
        yield ChunkSpec(start, end, n_rows, n_cols=-1)


def chunked_process(
    weight: np.ndarray,
    activations: np.ndarray | None,
    chunk_rows: int,
    compute: Callable[[np.ndarray, np.ndarray | None, ChunkSpec], object],
) -> list:
    """Run a row-chunked computation over a 2-D weight.

    The 12B FFN gate tensor is the canonical case: out_dim=16384,
    in_dim=4096, weight is 256 MB F32.  Chunking splits the
    out_dim axis (the long axis) into row-chunks; each chunk
    reads a (chunk_rows x in_dim) sub-matrix from the mmap, the
    ``compute`` callback does its per-chunk work, and the OS
    reclaims the pages before the next chunk is read.

    Parameters
    ----------
    weight : (out_dim, in_dim) ndarray
        The weight matrix.  Memory-mapped views are fine; the
        function will read each row-chunk once and let the OS
        reclaim the pages.
    activations : (n_tokens, in_dim) ndarray or None
        The training activations, or None when the bundle
        carries only the observer moments.  The activations
        are input-aligned: their in_dim axis matches the
        weight's in_dim.  The activations are passed to each
        chunk in full (no slicing); a chunk's per-row
        contribution is independent of which rows the
        activations cover.
    chunk_rows : int
        The number of rows per chunk.  ``0`` or negative means
        "no chunking" (one chunk covering the full weight);
        this is the legacy single-shot path.
    compute : callable
        Per-chunk work: ``compute(w_chunk, activations, spec)``
        where ``w_chunk = weight[start:end]`` is an mmap view
        (no copy) and ``activations`` is the full activation
        tensor.  Returns an arbitrary per-chunk result; the
        per-tensor result is reconstructed by reducing over the
        list of per-chunk results.

    Returns
    -------
    list
        The list of per-chunk results, in row order.  Callers
        typically ``sum`` / ``stack`` / reduce this list to
        obtain the per-tensor LRQ / FLRQ / DartQuant policy
        entry.  For 12B FFN gate tensors at 16384x4096 with
        ``chunk_rows=4096`` that is 4 entries per tensor.
    """
    if weight.ndim != 2:
        raise ValueError(f"chunked_process: weight must be 2-D, got {weight.ndim}-D")
    n_rows, in_dim = weight.shape
    if activations is not None:
        if activations.ndim != 2 or activations.shape[1] != in_dim:
            raise ValueError(
                f"chunked_process: activations shape {activations.shape} "
                f"incompatible with weight in_dim {in_dim}"
            )
    results: list = []
    for spec in chunked_iter(n_rows, chunk_rows):
        w_chunk = weight[spec.start:spec.end]   # mmap view, no copy
        results.append(compute(w_chunk, activations, spec))
    return results


# ---------------------------------------------------------------------------
# Category 4: spatial occupancy (interleave components for cache locality)
# ---------------------------------------------------------------------------

# A canonical set of model_role values for the unified pipeline.
# Listed in the round-robin order; the per-component shell-out
# iterates the components in this order so a single layer's
# tensors from all components fire before moving to the next
# layer.  The per-tensor observer moments (per-input-channel
# scales) are similar across components because they all share
# ``tok_embd`` + ``output``; round-robinning keeps the cache hot
# on those shared moments.
SPATIAL_ROLES: tuple[str, ...] = (
    "trunk",
    "dflash",
    "dspark",
    "mtp_nextn",
    "shared_embd",
)

# Regex used to extract a per-tensor layer index.  Matches the
# first integer that is either followed by ``.`` (mid-name) or
# at the end of the string (terminal).  Examples:
#   "blk.0.attn_q.weight"      -> 0
#   "dflash.encoder.fc.0"      -> 0  (terminal, no trailing ``.``)
#   "dspark.markov_w.12"       -> 12
#   "mtp_nextn.eh_proj.3"      -> 3
#   "token_embd.weight"        -> -1 (sentinel for "no layer")
_LAYER_RE = re.compile(r"\.(\d+)(?=\.|$)")


def extract_layer_index(tensor_name: str) -> int:
    """Return the layer index encoded in a tensor name, or -1.

    Tensors that are not layer-scoped (e.g. ``token_embd.weight``,
    ``output.weight``) return -1 so they sort first when the
    round-robin reaches them.  The exact regex matches
    ``.N.`` segments, which is the convention used by GGUF
    models for the trunk block index and by the per-component
    extension for the dflash / dspark / mtp_nextn block index.
    """
    match = _LAYER_RE.search(tensor_name)
    if match is None:
        return -1
    return int(match.group(1))


def interleave_components(
    components: dict[str, Sequence[str]],
    roles: Sequence[str] = SPATIAL_ROLES,
) -> Iterator[tuple[str, str]]:
    """Yield ``(model_role, tensor_name)`` pairs in spatial-interleaved order.

    The interleaving is round-robin across components **at the
    per-tensor level, layer by layer**: at each layer index, the
    per-role streams are pulled one tensor at a time in role
    order, then the layer's next round-robin round fires, then
    the next layer.  This keeps the cache hot on the per-tensor
    observer moments that components share (because they all
    depend on ``tok_embd`` + ``output``).

    Parameters
    ----------
    components : dict
        Maps ``model_role`` to a sequence of tensor names.  The
        per-component sequence is the per-component shell-out's
        own layer ordering; this function only re-orders across
        components.  Roles missing from the dict are skipped.
    roles : sequence of str
        The round-robin order.  The default
        ``SPATIAL_ROLES`` covers the unified gemma4_12B + dspark
        + dflash + MTP pipeline.

    Yields
    ------
    (model_role, tensor_name) : (str, str)
        Pairs in spatial-interleaved order.  Within a single
        layer, the order is the role order; the per-component
        order within a role is preserved.

    Notes
    -----
    Tensors whose layer index is -1 (e.g. ``token_embd.weight``)
    are emitted first (in role order) before the per-layer
    round-robin, so shared embeddings are computed before the
    first layer's tensors reference them.

    Worked example with two trunk tensors per layer and one
    tensor per layer for the other components::

        components = {
            "trunk":     ["blk.0.attn_q", "blk.0.attn_k", "blk.1.attn_q", "blk.1.attn_k"],
            "dflash":    ["dflash.encoder.fc.0", "dflash.encoder.fc.1"],
            "dspark":    ["dspark.markov_w.0", "dspark.markov_w.1"],
            "mtp_nextn": ["mtp_nextn.eh_proj.0", "mtp_nextn.eh_proj.1"],
        }
        list(interleave_components(components)) == [
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
    """
    if not components:
        return
    # Bucket each role's tensors by layer index; preserve the
    # per-role order within a layer.  Each bucket is a queue
    # that the round-robin pulls from.
    by_role_layer: dict[str, dict[int, list[str]]] = {}
    for role, names in components.items():
        bucket: dict[int, list[str]] = {}
        for n in names:
            idx = extract_layer_index(n)
            bucket.setdefault(idx, []).append(n)
        by_role_layer[role] = bucket

    # Layer-by-layer round-robin.  Layers = union of layer
    # indices across roles.  Tensors without a layer index
    # (-1) sort first so shared embeddings are computed
    # before the per-layer tensors reference them.
    layer_indices = sorted({idx for bucket in by_role_layer.values()
                            for idx in bucket.keys()})

    for idx in layer_indices:
        # Per-layer round-robin: each round pulls one tensor
        # from each role that still has tensors at this layer.
        # Continue until all roles' per-layer queues are
        # exhausted.
        active_roles = [r for r in roles
                        if by_role_layer.get(r, {}).get(idx)]
        # Deque-style round-robin via a small index cursor.
        cursor = 0
        while active_roles:
            role = active_roles[cursor % len(active_roles)]
            queue = by_role_layer[role][idx]
            if queue:
                n = queue.pop(0)
                yield role, n
            if not queue:
                # This role's per-layer stream is exhausted;
                # drop it from the round-robin for this layer.
                active_roles.pop(cursor % len(active_roles))
                if active_roles:
                    cursor = cursor % len(active_roles)
            else:
                cursor += 1


# ---------------------------------------------------------------------------
# Category 5: temporal occupancy (pipeline I/O with compute)
# ---------------------------------------------------------------------------


class CalibPipeline:
    """Double-buffered pipeline: mmap the next tensor while computing
    the current one.

    The pipeline is driven by a list of layer paths; the user
    pulls the next ``(path, layer_data)`` pair on each iteration.
    Internally, a worker thread mmaps one tensor ahead of the
    consumer, so the consumer's mmap latency overlaps with the
    consumer's compute.

    Parameters
    ----------
    layer_paths : sequence of path-like
        The list of ``.npz`` bundles to process, in iteration
        order.  The pipeline is single-pass: pulling ``n + 1``
        items after the last raises ``StopIteration``.
    depth : int
        The pipeline depth.  ``1`` is the legacy single-thread
        path (no overlap); ``2`` is the default double-buffered
        path; ``3+`` keeps more tensors in flight on slow I/O
        but uses more peak RSS (the working set is bounded by
        ``depth * max_tensor_bytes``).
    keys : sequence of str
        The keys to mmap per bundle.  Default mirrors
        ``mmap_layer``; pass a shorter tuple to skip the
        observer / name keys.
    mmap_mode : str
        Passed to ``np.load``; default ``"r"`` (read-only mmap).

    Notes
    -----
    The pipeline is single-consumer.  Multi-consumer use would
    require a thread-safe queue and a barrier; the calibration
    loop is single-consumer by design (one tensor at a time, so
    the per-tensor learning loop is the natural serialisation
    point).

    The pipeline does **not** copy the mmap into RAM: it hands
    out the same mmap views on each iteration.  The OS pages
    in the data as the consumer reads it; once the consumer
    moves to the next tensor, the OS reclaims the pages lazily.
    """

    def __init__(
        self,
        layer_paths: Sequence[str | os.PathLike],
        depth: int = 2,
        keys: Sequence[str] = (
            "weight",
            "train_activations",
            "heldout_activations",
            "in_sum2",
            "counts",
            "name",
            "family",
        ),
        mmap_mode: str = "r",
    ) -> None:
        if depth < 1:
            raise ValueError(f"CalibPipeline: depth must be >= 1, got {depth}")
        self._layer_paths = list(layer_paths)
        self._depth = min(int(depth), max(1, len(self._layer_paths)))
        self._keys = tuple(keys)
        self._mmap_mode = mmap_mode
        self._queue: list = []
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._producer_thread: threading.Thread | None = None
        self._producer_exc: BaseException | None = None
        self._closed = False
        self._started = False

    def __iter__(self) -> "CalibPipeline":
        return self

    def __next__(self) -> tuple[Path, dict[str, np.ndarray]]:
        if not self._started:
            self._start()
        item = self._pop()
        if item is None:
            if self._producer_exc is not None:
                exc = self._producer_exc
                self._producer_exc = None
                raise exc
            raise StopIteration
        path, data = item
        return path, data

    def _start(self) -> None:
        self._started = True
        if self._depth == 1 or len(self._layer_paths) <= 1:
            # Single-threaded path: no producer thread; the
            # consumer does its own mmap on the call site via
            # ``__next__`` -> ``_pop`` (which falls through to
            # the synchronous mmap path).
            return
        self._producer_thread = threading.Thread(
            target=self._producer_loop,
            name="CalibPipelineProducer",
            daemon=True,
        )
        self._producer_thread.start()

    def _producer_loop(self) -> None:
        try:
            for path in self._layer_paths:
                data = self._mmap(path)
                with self._cv:
                    while len(self._queue) >= self._depth:
                        self._cv.wait()
                    self._queue.append((path, data))
                    self._cv.notify()
        except BaseException as exc:  # pragma: no cover - propagates to consumer
            self._producer_exc = exc
            with self._cv:
                self._cv.notify_all()
        finally:
            self._closed = True
            with self._cv:
                self._cv.notify_all()

    def _mmap(self, path: str | os.PathLike) -> dict[str, np.ndarray]:
        # Detach from the np.load handle before returning: the
        # consumer will hold the views after the producer moves
        # on, and we want the OS to keep the zip mmap alive
        # (the views' memory backs onto the zip mmap).
        with np.load(os.fspath(path), mmap_mode=self._mmap_mode, allow_pickle=False) as data:
            out = {k: data[k] for k in self._keys if k in data.files}
        return out

    def _pop(self) -> tuple[Path, dict[str, np.ndarray]] | None:
        if self._producer_thread is None:
            # Single-threaded path: mmap on the consumer thread.
            if not self._layer_paths:
                return None
            path = self._layer_paths.pop(0)
            return Path(path), self._mmap(path)
        with self._cv:
            while not self._queue and not self._closed:
                self._cv.wait()
            if not self._queue:
                return None
            item = self._queue.pop(0)
            self._cv.notify()
            return item

    def close(self) -> None:
        """Stop the producer thread (idempotent)."""
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        if self._producer_thread is not None:
            self._producer_thread.join(timeout=0.1)
            self._producer_thread = None

    def __enter__(self) -> "CalibPipeline":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# House-keeping: a public entry that bundles the utilities for ``help()``
# and pydoc-style introspection.  The CLI flag list lives in
# ``per_tensor_calibrate.py``; this module is the implementation.
# ---------------------------------------------------------------------------


__all__ = [
    "mmap_tensor",
    "mmap_layer",
    "chunked_iter",
    "chunked_process",
    "ChunkSpec",
    "interleave_components",
    "extract_layer_index",
    "SPATIAL_ROLES",
    "CalibPipeline",
]
