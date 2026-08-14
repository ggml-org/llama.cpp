#!/usr/bin/env python3
from __future__ import annotations

import os
import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

subprocess.run(
    [sys.executable, str(ROOT / "scripts/apply-tiered-upstream-compat.py")],
    cwd=ROOT,
    check=True,
)
runpy.run_path(
    str(ROOT / "scripts/apply-tiered-dram-pinned-fallback-impl.py"),
    run_name="__main__",
)
