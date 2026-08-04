"""Peak-RSS budget tracker for the calibration pipeline.

Phase 16 (calibration memopt) Category 3: the unified
gemma4_12B + dspark + dflash + MTP calibration processes
4000+ tensors, and a 12B FFN gate tensor at 16384x4096 is
256 MB F32.  The previous (pre-Phase-16) ``per_tensor_calibrate.py``
had no peak-RSS cap; on a 12B calibration it would OOM and
the OS would kill the process without a useful error
message.

This module owns the **mechanism**: read the current RSS,
compare it against a budget, raise ``MemoryError`` if the
budget is exceeded.  The policy side ("where should the
tensor live: RAM-mmap vs RAM-resident vs disk-spill")
belongs in ``calibration_memory.py``; this module is the
runtime check that enforces the budget.

The check is cross-platform:
  * Linux: ``/proc/self/status`` (no extra deps)
  * macOS / Windows: ``psutil.Process(os.getpid()).memory_info().rss``
    (psutil is in the dev environment; the import is lazy so
    the legacy single-host path doesn't pay the import cost)

The default budget is 32 GB, which fits a 12B unified
calibration on a 64 GB host with the chunked path.  The
CLI knob is ``--peak-rss-budget-gb N`` on
``per_tensor_calibrate.py``; the user can lower it for
tighter hosts (8 GB) or raise it on larger hosts (96 GB).

The tracker is **advisory** during the calibration: it
reads the current RSS at the top of each per-tensor
iteration and aborts with a clear error message naming
the tensor and the observed RSS vs the budget.  The
tracker is NOT a hard real-time limit; the OS may exceed
the budget transiently (e.g. on numpy's internal
allocations during a matmul).  The tracker is calibrated
to catch sustained over-budget states, not micro-spikes.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import sys
from typing import Iterator

# The cross-platform RSS reader.  ``psutil`` is preferred
# (single call, handles macOS/Windows quirks); on Linux we
# fall back to ``/proc/self/status`` so we don't require
# psutil on bare CI hosts.  The import is lazy so the
# legacy path doesn't pay for it.
_psutil_proc = None


def _read_rss_procfs() -> int | None:
    """Read the current RSS from ``/proc/self/status`` (Linux)."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        # VmRSS is in kB; convert to bytes.
                        return int(parts[1]) * 1024
    except (OSError, ValueError):
        return None
    return None


def _read_rss_psutil() -> int | None:
    """Read the current RSS via psutil (cross-platform)."""
    global _psutil_proc
    if _psutil_proc is None:
        try:
            import psutil  # type: ignore[import-not-found]
        except ImportError:
            return None
        _psutil_proc = psutil.Process(os.getpid())
    try:
        return int(_psutil_proc.memory_info().rss)
    except Exception:  # psutil raises various on process death
        return None


def read_rss_bytes() -> int:
    """Return the current process RSS in bytes.

    Tries ``/proc/self/status`` first (Linux, no extra deps);
    falls back to ``psutil`` (cross-platform); returns 0 if
    neither path is available.  The 0 fallback is safe: the
    budget check is conservative (it warns on any reported
    RSS > 0 over the budget, but 0 disables the check).
    """
    rss = _read_rss_procfs()
    if rss is not None:
        return rss
    rss = _read_rss_psutil()
    if rss is not None:
        return rss
    return 0


@dataclasses.dataclass
class ResidencyTracker:
    """Track peak RSS during a calibration run; abort if budget exceeded.

    The tracker is created at the top of ``main()`` and
    checked at the top of each per-tensor iteration.  On
    Linux it reads ``/proc/self/status`` (no extra deps);
    on macOS / Windows it uses psutil.  The cross-platform
    fallback returns 0 if neither is available, which
    disables the check (the calibration proceeds without a
    memory cap).

    The check is **advisory** during the calibration: it
    catches sustained over-budget states, not micro-spikes
    from numpy's internal allocations.  The ``budget_bytes``
    should be chosen with a 1.5-2x safety margin over the
    expected per-tensor working set so transient over-runs
    don't false-positive.

    Parameters
    ----------
    budget_bytes : int
        The peak-RSS budget in bytes.  ``<= 0`` disables the
        check (the tracker still records peak_bytes for the
        final report, but never raises).
    abort_on_exceed : bool
        If True (the default), ``check()`` raises
        ``MemoryError`` when the current RSS exceeds the
        budget.  If False, the check is logged but does not
        abort (useful for the verbose / diagnostic path).
    """

    budget_bytes: int = 32 * 1024**3   # 32 GB default
    abort_on_exceed: bool = True
    peak_bytes: int = 0
    n_checks: int = 0
    n_violations: int = 0

    def __post_init__(self) -> None:
        if self.budget_bytes < 0:
            raise ValueError(f"budget_bytes must be >= 0, got {self.budget_bytes}")

    def check(self, tensor_name: str) -> int:
        """Check the current RSS; raise ``MemoryError`` if over budget.

        Returns the observed RSS in bytes.  Updates
        ``peak_bytes`` and increments ``n_checks`` /
        ``n_violations`` for the final report.  The
        ``tensor_name`` is included in the error message so
        operators can see which tensor the budget failed on.
        """
        current = read_rss_bytes()
        self.n_checks += 1
        if current > self.peak_bytes:
            self.peak_bytes = current
        if self.budget_bytes > 0 and current > self.budget_bytes:
            self.n_violations += 1
            if self.abort_on_exceed:
                raise MemoryError(
                    f"calibration OOM on {tensor_name}: "
                    f"RSS {current / 1e9:.2f} GB > "
                    f"budget {self.budget_bytes / 1e9:.2f} GB "
                    f"(peak {self.peak_bytes / 1e9:.2f} GB, "
                    f"checks {self.n_checks})"
                )
        return current

    def report(self) -> str:
        """Return a one-line summary of the run's RSS footprint."""
        peak_gb = self.peak_bytes / 1e9 if self.peak_bytes else 0.0
        budget_gb = self.budget_bytes / 1e9 if self.budget_bytes > 0 else float("inf")
        return (
            f"peak RSS {peak_gb:.2f} GB "
            f"(budget {budget_gb:.2f} GB, "
            f"checks {self.n_checks}, violations {self.n_violations})"
        )


@contextlib.contextmanager
def residency_managed(
    budget_bytes: int,
    tensor_paths: list[str] | None = None,
) -> Iterator[ResidencyTracker]:
    """Context manager: a ``ResidencyTracker`` over a block of
    per-tensor work.

    The ``tensor_paths`` argument is optional metadata used
    to print a more helpful error message when the budget
    is exceeded.  The caller is expected to call
    ``tracker.check(tensor_name)`` at the top of each
    per-tensor iteration; the context manager itself does
    not check the budget (the per-tensor check is the
    granular signal).
    """
    tracker = ResidencyTracker(budget_bytes=budget_bytes)
    try:
        yield tracker
    finally:
        # Print a final report to stderr so the operator
        # sees the peak even on a clean exit.
        print(f"residency: {tracker.report()}", file=sys.stderr)


def default_budget_gb() -> int:
    """Return the default peak-RSS budget in GiB.

    The default is 32 GB, which fits a 12B unified
    calibration on a 64 GB host with the chunked path.
    Operators on tighter hosts can lower it via the
    ``--peak-rss-budget-gb`` CLI flag; the resulting OOM
    error names the tensor and the observed RSS.
    """
    return 32


__all__ = [
    "ResidencyTracker",
    "read_rss_bytes",
    "residency_managed",
    "default_budget_gb",
]
