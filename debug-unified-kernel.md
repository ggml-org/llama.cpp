# Debug Log: Unified SYCL Kernel Regression (llama.cpp-373f)

## Problem Summary
- **Main branch**: `1, 2, 3, 4, 5,` → `1, 2, 3, 4, 5, 6, 7, 8, 9, 10` ✅
- **Feature branch**: `1, 2, 3, 4, 5,` → `1, 2, 3, 4, 5,\n\n` ❌

## ROOT CAUSE IDENTIFIED ✅

**Bug Location**: `ggml/src/ggml-sycl/mmvq.cpp`

**Root Cause**: The coalesced MMVQ kernels (`mul_mat_vec_q4_0_coalesced`, `mul_mat_vec_q8_0_coalesced`)
incorrectly calculate the scale offset using `nrows` (slice size) instead of `total_nrows` (full tensor).

**Bug in code** (line 206):
```cpp
const ggml_half * x_d = (const ggml_half *) ((const char *) vx + nrows * x_row_stride);
```

**Problem**: Coalesced layout stores scales after ALL quants for the ENTIRE tensor:
- Layout: `[all_quants_for_all_rows][all_scales_for_all_rows]`
- Scale offset MUST use `total_nrows`, not `row_diff`

**Call site (line 4204)** only passes `row_diff`:
```cpp
coalesced_mul_mat_vec_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
```

**Compare to SOA path (correct)**:
```cpp
reorder_mul_mat_vec_q4_0_q8_1_sycl(soa_base, ..., ne00, row_diff, ne01, row_low, stream);
```

## Fix Required

1. Add `total_nrows` and `row_low` parameters to coalesced kernels
2. Update scale offset: `vx + total_nrows * x_row_stride`
3. Update row indexing: `row_low + row` for quant access

## Investigation Log

### Session: 2026-01-26

**Step 1**: Confirmed bug - both unified and legacy paths produce broken output
**Step 2**: Identified dispatch path uses COALESCED layout (layout=2)
**Step 3**: Found MMVQ refactoring added `total_nrows`, `row_low` to SOA path but NOT coalesced
**Step 4**: Identified scale offset calculation bug in coalesced kernel

