#!/usr/bin/env python3
"""Convenience launcher: `python run.py --model path/to/model.gguf`.

Adds ``src/`` to the import path so the launcher works without an install.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from desktop_solver.gui import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
