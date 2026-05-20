"""Filesystem scanner that finds local GGUF models and picks the best one.

The scanner is tiered: known paths first (fast), then the user's home tree,
then opt/media/mnt roots. A deadline caps total walk time so the GUI never
hangs. Hidden directories are skipped except for known cache locations
(``.cache``, ``.local``, ``.lmstudio``) where models commonly live.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

log = logging.getLogger(__name__)


# Paths checked first — covers HuggingFace, LM Studio, llama.cpp examples, Ollama, etc.
PRIORITY_ROOTS: tuple[str, ...] = (
    "~/Downloads",
    "~/Documents",
    "~/Models",
    "~/models",
    "~/.cache/huggingface",
    "~/.cache/lm-studio",
    "~/.cache/llama.cpp",
    "~/.lmstudio/models",
    "~/.ollama/models",
    "~/.local/share/models",
    "~/Library/Application Support/LM Studio/models",
    "~/Library/Caches/llama.cpp",
    "/opt/models",
    "/usr/local/share/models",
)

# Broader roots used when priority paths find nothing.
FALLBACK_ROOTS: tuple[str, ...] = (
    "~",
    "/opt",
    "/data",
    "/mnt",
    "/media",
    "/srv",
)

# Always skipped — system, build artefacts, scratch.
_SKIP_DIR_NAMES: frozenset[str] = frozenset({
    ".git", ".svn", ".hg",
    "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "venv", ".venv", "env", ".tox", ".nox",
    "build", "dist", "target", "out", ".gradle", ".idea", ".vscode",
})

# Absolute prefixes never worth walking (kernel + container scratch).
# Note: /tmp and /var are NOT skipped — temp downloads of GGUF files may live there.
_SKIP_ABS_PREFIXES: tuple[str, ...] = (
    "/proc", "/sys", "/dev", "/run", "/boot", "/snap", "/var/lib/docker",
)

# Hidden dirs that we DO descend into (caches commonly hold downloaded models).
_HIDDEN_ALLOWLIST: frozenset[str] = frozenset({
    ".cache", ".local", ".lmstudio", ".ollama",
})


@dataclass(frozen=True)
class FoundModel:
    """A GGUF file found on disk."""
    path: Path
    size_bytes: int
    mtime: float

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024 ** 3)


def system_ram_bytes() -> int:
    """Total system RAM. Linux uses /proc/meminfo; other platforms fall back to 8 GB."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except (FileNotFoundError, OSError, ValueError):
        pass
    if sys.platform == "darwin":
        try:
            import subprocess  # noqa: PLC0415
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return int(out.strip())
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            pass
    return 8 * 1024 ** 3


def _should_skip_abs(path: str) -> bool:
    return any(path == p or path.startswith(p + os.sep) for p in _SKIP_ABS_PREFIXES)


def _iter_gguf(root: Path, deadline: float,
               on_dir: Optional[Callable[[Path], None]] = None) -> Iterator[Path]:
    """Yield ``.gguf`` files under ``root`` until ``deadline`` (monotonic seconds).

    Skip-prefixes apply to subdirectories discovered during descent — an
    explicit ``root`` argument is always scanned, even if it sits below a
    normally-skipped prefix (e.g. an ``/var/cache`` path passed by the user).
    """
    stack: list[Path] = [root]
    while stack:
        if time.monotonic() > deadline:
            return
        current = stack.pop()
        if on_dir is not None:
            on_dir(current)
        try:
            with os.scandir(current) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            name = entry.name
                            if name in _SKIP_DIR_NAMES:
                                continue
                            if name.startswith(".") and name not in _HIDDEN_ALLOWLIST:
                                continue
                            if _should_skip_abs(entry.path):
                                continue
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            if entry.name.lower().endswith(".gguf"):
                                yield Path(entry.path)
                    except OSError:
                        continue
        except (PermissionError, OSError):
            continue


def scan(roots: Optional[list[str]] = None, *,
         max_seconds: float = 90.0,
         on_dir: Optional[Callable[[Path], None]] = None,
         on_found: Optional[Callable[[FoundModel], None]] = None) -> list[FoundModel]:
    """Find ``.gguf`` files under ``roots`` (defaults to priority + fallback paths)."""
    if roots is None:
        roots = list(PRIORITY_ROOTS) + list(FALLBACK_ROOTS)

    deadline = time.monotonic() + max_seconds
    visited_roots: set[Path] = set()
    seen_files: set[Path] = set()
    out: list[FoundModel] = []

    for root_str in roots:
        if time.monotonic() > deadline:
            break
        root = Path(os.path.expanduser(root_str))
        try:
            if not root.exists() or not root.is_dir():
                continue
            root = root.resolve()
        except OSError:
            continue
        if root in visited_roots:
            continue
        if any(str(root).startswith(str(v) + os.sep) for v in visited_roots):
            # Already covered by a higher root.
            continue
        visited_roots.add(root)

        log.info("scanning %s", root)
        for path in _iter_gguf(root, deadline, on_dir=on_dir):
            try:
                resolved = path.resolve()
            except OSError:
                resolved = path
            if resolved in seen_files:
                continue
            seen_files.add(resolved)
            try:
                stat = resolved.stat()
            except OSError:
                continue
            model = FoundModel(path=resolved, size_bytes=stat.st_size, mtime=stat.st_mtime)
            out.append(model)
            if on_found is not None:
                on_found(model)

    return out


def score(model: FoundModel, *, ram_bytes: Optional[int] = None) -> float:
    """Higher is better. Math-tuned and instruction-tuned models win;
    largest model that fits within ~70% of RAM wins among same-tier."""
    if ram_bytes is None:
        ram_bytes = system_ram_bytes()
    name = model.name.lower()

    s = 0.0
    if "math" in name:
        s += 100
    if any(tag in name for tag in ("instruct", "chat", "it-")):
        s += 40
    if any(tag in name for tag in ("qwen2.5-math", "deepseek-math", "mathstral", "wizard-math")):
        s += 60
    # Prefer Q4 — fits 8 GB and is the GCSE sweet spot.
    if "q4_k_m" in name:
        s += 35
    elif "q4_0" in name or "q4_k_s" in name or "iq4" in name:
        s += 25
    elif "q5" in name:
        s += 15
    elif "q8" in name or "fp16" in name or "f16" in name:
        s += 5

    # Fit in RAM with ~30% headroom.
    budget = int(ram_bytes * 0.7)
    if model.size_bytes <= budget:
        # Larger-up-to-budget is better.
        s += (model.size_bytes / max(budget, 1)) * 30
    else:
        s -= 80

    # Slight nudge toward recent files.
    age_days = max((time.time() - model.mtime) / 86400, 0)
    s += max(10 - age_days * 0.02, 0)

    return s


def auto_pick(models: list[FoundModel], *,
              ram_bytes: Optional[int] = None) -> Optional[FoundModel]:
    """Return the best model from a list, or None."""
    if not models:
        return None
    if ram_bytes is None:
        ram_bytes = system_ram_bytes()
    return max(models, key=lambda m: score(m, ram_bytes=ram_bytes))
