#!/usr/bin/env python3
"""Crash-safe wrapper around ``llama-imatrix`` for the gemma4 12B smoke
test. The native imatrix binary loads the full model into the unified
memory (23 GB BF16 on a 16 GB M1) and runs inference for 30-60 min
- jetsam SIGKILL is a real risk. This wrapper enforces three guards:

  1. Memory precheck: refuse to start if ``model_size >
     memory_safety_fraction * physmem`` (default 0.6). Override with
     ``--force`` if you know what you're doing.
  2. ``--save-frequency 32``: the native imatrix binary default is 0
     (no intermediate saves) - a 30-min run that crashes at minute 25
     loses everything. 32 saves every ~30s so a crash loses at most
     one checkpoint.
  3. PID file + clean shutdown: the wrapper writes its own PID to
     ``<output>.pid``; ``kill -TERM`` the PID triggers a graceful
     SIGTERM to the child (imatrix will exit at the next chunk
     boundary; the atomic .tmp + rename in save_imatrix guarantees
     the previous checkpoint is intact).

Typical usage (gemma4 12B smoke test on M1, external SSD for outputs):

  tools/tessera/smoke_imatrix.py \\
    --model /Volumes/Julian\\ T7/models/gemma-4-12B-it-bf16.gguf \\
    --corpus /Volumes/Julian\\ T7/calibration_datav5.txt \\
    --output /Volumes/Julian\\ T7/runs/gemma4-12b-smoke/imatrix.gguf \\
    --save-frequency 32 --max-minutes 30

This wrapper is intentionally thin: it is NOT a replacement for
``llama-imatrix``. It is a script-level guard that catches the most
common crash causes (memory, save-loss) before the binary starts.
Crash-safe atomic writes inside imatrix (the .tmp + rename in
save_imatrix) are the load-bearing safety net; this wrapper is the
operator-visible layer that makes the safety choices explicit.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import resource
import shutil
import signal
import subprocess
import sys
import time
from typing import Optional


def _humanize_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _physmem_bytes() -> int:
    """Return the machine's physical memory in bytes (no third-party
    deps; works on Linux, macOS, BSD via /proc and sysctl). On macOS
    ``hw.memsize`` is the canonical source; on Linux ``MemTotal`` in
    /proc/meminfo; on Windows we fall back to a 0 sentinel so the
    caller can detect and skip the precheck."""
    if sys.platform == "darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True
            ).strip()
            return int(out)
        except (subprocess.CalledProcessError, ValueError):
            return 0
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        # "MemTotal:       16384000 kB"
                        kb = int(line.split()[1])
                        return kb * 1024
        except (OSError, ValueError):
            return 0
    return 0


def _model_size_bytes(path: pathlib.Path) -> int:
    """Return the on-disk size of the model GGUF. 0 if missing."""
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _preflight_memory(
    model_path: pathlib.Path,
    memory_safety_fraction: float,
    force: bool,
) -> tuple[int, int, float]:
    """Return (model_bytes, physmem_bytes, ratio). Aborts if ratio
    exceeds ``memory_safety_fraction`` and ``force`` is not set."""
    model = _model_size_bytes(model_path)
    phys = _physmem_bytes()
    if phys == 0 or model == 0:
        return model, phys, 0.0
    ratio = model / phys
    if ratio > memory_safety_fraction and not force:
        sys.stderr.write(
            f"smoke_imatrix: refusing to start: model is "
            f"{_humanize_bytes(model)} on "
            f"{_humanize_bytes(phys)} physmem "
            f"(ratio={ratio:.2f} > "
            f"memory_safety_fraction={memory_safety_fraction:.2f}).\n"
            f"smoke_imatrix: pass --force to override (NOT recommended;\n"
            f"  jetsam SIGKILL is likely).\n"
        )
        sys.exit(2)
    return model, phys, ratio


def _find_imatrix_binary(repo_root: pathlib.Path) -> pathlib.Path:
    """Resolve the imatrix binary. Default to ``build/bin/llama-imatrix``
    relative to the repo root; allow override via
    ``--imatrix-binary``. We do NOT search PATH by default because
    ``/Volumes/Julian T7/llama-cpp-build/bin/llama-imatrix`` is an
    older build that is not what we want."""
    candidates = [
        repo_root / "build" / "bin" / "llama-imatrix",
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        f"smoke_imatrix: cannot find llama-imatrix; tried {candidates}. "
        f"Build it with `cmake --build build --target llama-imatrix` or "
        f"pass --imatrix-binary PATH."
    )


def _write_pid_file(pid_path: pathlib.Path, pid: int) -> None:
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(f"{pid}\n")


def _remove_pid_file(pid_path: pathlib.Path) -> None:
    try:
        pid_path.unlink()
    except FileNotFoundError:
        pass


class _ChildSupervisor:
    """Forward SIGTERM/SIGINT to the child; on second SIGINT, SIGKILL
    (gives the child a chance to flush its checkpoint)."""

    def __init__(self, child: subprocess.Popen, pid_path: pathlib.Path):
        self._child = child
        self._pid_path = pid_path
        self._sigint_count = 0
        signal.signal(signal.SIGTERM, self._on_term)
        signal.signal(signal.SIGINT, self._on_term)

    def _on_term(self, signum, frame):
        self._sigint_count += 1
        if self._sigint_count == 1:
            sys.stderr.write(
                "smoke_imatrix: received signal; forwarding SIGTERM to "
                f"child pid {self._child.pid} (imatrix will exit at the "
                f"next chunk boundary).\n"
            )
            try:
                self._child.terminate()
            except ProcessLookupError:
                pass
        else:
            sys.stderr.write(
                "smoke_imatrix: second signal; sending SIGKILL to child.\n"
            )
            try:
                self._child.kill()
            except ProcessLookupError:
                pass

    def wait(self) -> int:
        return self._child.wait()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model", required=True, type=pathlib.Path,
        help="Path to the input GGUF (e.g. gemma-4-12B-it-bf16.gguf)."
    )
    parser.add_argument(
        "--corpus", required=True, type=pathlib.Path,
        help="Path to the calibration corpus text file."
    )
    parser.add_argument(
        "--output", required=True, type=pathlib.Path,
        help="Output imatrix.gguf path."
    )
    parser.add_argument(
        "--save-frequency", type=int, default=32,
        help="Save a checkpoint every N chunks (default 32 ~= 30s on gemma4 12B). "
             "Pass 0 to disable intermediate saves (NOT recommended; you will "
             "lose all progress on a crash)."
    )
    parser.add_argument(
        "--max-minutes", type=int, default=60,
        help="Soft cap wall time in minutes; SIGTERM the child when reached "
             "(default 60)."
    )
    parser.add_argument(
        "--memory-safety-fraction", type=float, default=0.6,
        help="Refuse to start if model_size / physmem > fraction (default 0.6). "
             "Pass --force to override."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Skip the memory precheck."
    )
    parser.add_argument(
        "--imatrix-binary", type=pathlib.Path, default=None,
        help="Path to llama-imatrix (default: <repo>/build/bin/llama-imatrix)."
    )
    parser.add_argument(
        "--ctx-size", type=int, default=512,
        help="Context size passed to llama-imatrix (default 512)."
    )
    parser.add_argument(
        "--chunks", type=int, default=0,
        help="Number of chunks to process (default 0 = unlimited; rely on "
             "--max-minutes for the wall-time cap)."
    )
    parser.add_argument(
        "--extra-arg", action="append", default=[],
        help="Extra arg passed verbatim to llama-imatrix (repeatable)."
    )
    args = parser.parse_args(argv)

    repo_root = pathlib.Path(__file__).resolve().parent.parent.parent
    binary = args.imatrix_binary or _find_imatrix_binary(repo_root)

    # Memory precheck
    model, phys, ratio = _preflight_memory(
        args.model, args.memory_safety_fraction, args.force
    )
    if phys:
        sys.stderr.write(
            f"smoke_imatrix: model={_humanize_bytes(model)}, "
            f"physmem={_humanize_bytes(phys)}, ratio={ratio:.2f}\n"
        )

    if not args.corpus.is_file():
        sys.stderr.write(
            f"smoke_imatrix: --corpus {args.corpus} not found.\n"
        )
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pid_path = args.output.with_suffix(args.output.suffix + ".pid")

    cmd = [
        str(binary),
        "-m", str(args.model),
        "-f", str(args.corpus),
        "-o", str(args.output),
        "-c", str(args.ctx_size),
        "--save-frequency", str(args.save_frequency),
    ]
    if args.chunks > 0:
        cmd += ["-n", str(args.chunks)]
    cmd += args.extra_arg

    sys.stderr.write(
        "smoke_imatrix: launching: " + " ".join(cmd) + "\n"
    )

    t0 = time.monotonic()
    child = subprocess.Popen(
        cmd, stdout=sys.stdout, stderr=sys.stderr,
        # NEW_SESSION so SIGTERM to the wrapper does not auto-cascade
        # to the child; the supervisor forwards explicitly.
        start_new_session=True,
    )
    _write_pid_file(pid_path, child.pid)
    supervisor = _ChildSupervisor(child, pid_path)

    # Wall-time cap
    rc: Optional[int] = None
    try:
        if args.max_minutes > 0:
            deadline = t0 + args.max_minutes * 60
            while True:
                try:
                    rc = supervisor.wait(timeout=5)
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() > deadline:
                        sys.stderr.write(
                            f"smoke_imatrix: --max-minutes "
                            f"({args.max_minutes}) reached; SIGTERM child.\n"
                        )
                        try:
                            child.terminate()
                        except ProcessLookupError:
                            pass
        else:
            rc = supervisor.wait()
    finally:
        _remove_pid_file(pid_path)
        elapsed = time.monotonic() - t0
        sys.stderr.write(
            f"smoke_imatrix: child exited rc={rc} after {elapsed:.1f}s.\n"
        )
    return rc if rc is not None else 1


if __name__ == "__main__":
    sys.exit(main())
