#!/usr/bin/env python3
"""Generate the Python reference packing for the C++ bit-equivalence test.

Writes the Python ternary -> Tile640 packing and fitted scales for a
deterministic 4x640 (seed=42 Gaussian) weight page, so test_bit_equiv.cpp
can compare the C++ ts_ternarize_with_acts + ts_pack_tile640 output
byte-for-byte against quantize_v3.py.

Reference choices:
  - TESSERA_ACCELERATE is forced off before import so compute_scales uses
    the pure-NumPy least-squares path (the mathematical spec) rather than
    the Accelerate fast path. This keeps backend variance out of the
    reference. Must be set before quantize_v3 is imported, since the module
    probes this env var at load time.
  - ternarize_with_acts(weights, None, outlier_frac=0.0): no activation
    scaling and no outliers, matching the C++ call
    ts_ternarize_with_acts(weights, nullptr, alpha=0, clip=0).
"""
import os
# Force the pure-NumPy scale path for a deterministic reference.
os.environ["TESSERA_ACCELERATE"] = "0"

import sys
import numpy as np

sys.path.insert(0, "tools/tile640")
from quantize_v3 import ternarize_with_acts, pack_tile640, compute_scales

np.random.seed(42)
weights = np.random.randn(4, 640).astype(np.float32)

# Ternarize: no AWQ (act_scales=None), no outliers.
ternary, outlier_idx, outlier_vals = ternarize_with_acts(weights, None, 0.0)
assert outlier_idx.size == 0, "expected no outliers with outlier_frac=0"

packed, pages_per_row = pack_tile640(ternary, 4, 640)
page_scales, lane_scales = compute_scales(weights, ternary, 4, 640)

packed.astype(np.uint32).tofile("/tmp/bit_equiv_py_packed.bin")
page_scales.astype(np.float16).tofile("/tmp/bit_equiv_py_page_scales.bin")
lane_scales.astype(np.int8).tofile("/tmp/bit_equiv_py_lane_scales.bin")
weights.tofile("/tmp/bit_equiv_weights.bin")

print(f"Python: packed shape={packed.shape} dtype={packed.dtype} pages_per_row={pages_per_row}")
print(f"Python: page_scales shape={page_scales.shape} dtype={page_scales.dtype}")
print(f"Python: lane_scales shape={lane_scales.shape} dtype={lane_scales.dtype}")
