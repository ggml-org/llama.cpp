#!/usr/bin/env python3
"""Save/load plain numpy tensors to/from a GGUF file, for the ggml-test skill's PyTorch <-> ggml comparison workflow. Thin wrapper around gguf-py.

Import from a script placed anywhere under the project (e.g. tmp/):

    import sys
    sys.path.insert(0, "skills/ggml-test/scripts")
    from gguf_io import save_tensors, load_tensors
"""
import sys
from pathlib import Path

_GGUF_PY = Path(__file__).resolve().parents[3] / "gguf-py"
if _GGUF_PY.exists() and str(_GGUF_PY) not in sys.path:
    sys.path.insert(0, str(_GGUF_PY))

import numpy as np
from gguf import GGUFWriter
from gguf.gguf_reader import GGUFReader


def save_tensors(path, tensors, kv=None):
    """tensors: dict[str, np.ndarray]. Array shape/dtype is preserved as-is -- gguf-py reverses the axis order internally to match ggml's ne[] convention (numpy's last axis becomes ggml's ne[0]), and load_tensors() below reverses it back, so round-tripping keeps the original numpy shape."""
    writer = GGUFWriter(str(path), "ggml-test")
    for k, v in (kv or {}).items():
        if isinstance(v, bool):
            writer.add_bool(k, v)
        elif isinstance(v, int):
            writer.add_int64(k, v)
        elif isinstance(v, float):
            writer.add_float32(k, v)
        else:
            writer.add_string(k, str(v))
    for name, arr in tensors.items():
        writer.add_tensor(name, np.ascontiguousarray(arr))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def load_tensors(path):
    """Returns dict[str, np.ndarray], shape as originally saved."""
    reader = GGUFReader(str(path))
    return {t.name: t.data.copy() for t in reader.tensors}
