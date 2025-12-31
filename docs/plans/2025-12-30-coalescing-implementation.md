# Coalesced Memory Layout Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add GPU-coalesced memory access patterns to SYCL backend for >80% memory bandwidth utilization.

**Architecture:** Extend existing SoA infrastructure with new COALESCED reorder mode. Adjacent threads access adjacent memory addresses within a warp (32 threads). Profile-driven approach: baseline first, then implement kernel-by-kernel.

**Tech Stack:** SYCL/DPC++, Intel oneAPI, VTune GPU Hotspots

---

## Phase 1: Profiling Baseline

### Task 1: Capture VTune Baseline for Q4_0 DMMV

**Files:**
- Create: `benchmark_results/vtune_soa_baseline/` (directory)
- Reference: `docs/DEBUG.md` (VTune commands)

**Step 1: Run VTune GPU Hotspots on Q4_0 model**

```bash
source /opt/intel/oneapi/setvars.sh --force
ONEAPI_DEVICE_SELECTOR=level_zero:1 vtune -collect gpu-hotspots \
  -knob gpu-sampling-interval=1 \
  -result-dir benchmark_results/vtune_soa_baseline \
  -- ./build/bin/llama-bench -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p 512 -n 128 -ngl 99 -fa 1

vtune -report summary -r benchmark_results/vtune_soa_baseline
```

**Step 2: Document baseline metrics**

Create `benchmark_results/vtune_soa_baseline/README.md`:
```markdown
# SoA Baseline (Q4_0)

## Memory Bandwidth
- Achieved: ??? GB/s
- Theoretical: 448 GB/s (Arc B50 Pro)
- Efficiency: ???%

## Top Kernels
1. kernel_name: ??? GB/s (??%)
2. ...
```

**Step 3: Commit baseline**

```bash
git add benchmark_results/vtune_soa_baseline/
git commit -m "perf: Add VTune SoA baseline for Q4_0"
```

---

## Phase 2: Infrastructure

### Task 2: Add COALESCED to reorder_mode enum

**Files:**
- Modify: `ggml/src/ggml-sycl/common.hpp`
- Test: Compile check only (no runtime change yet)

**Step 1: Locate reorder_mode enum**

```bash
grep -n "enum.*reorder_mode\|reorder_mode.*{" ggml/src/ggml-sycl/common.hpp
```

**Step 2: Add COALESCED value**

In `ggml/src/ggml-sycl/common.hpp`, find the enum and add:
```cpp
enum class reorder_mode {
    NONE = 0,
    SOA = 1,
    COALESCED = 2  // NEW: GPU-coalesced access pattern
};
```

**Step 3: Build to verify no compile errors**

```bash
./scripts/quick-rebuild.sh common.hpp
```
Expected: Build succeeds

**Step 4: Commit**

```bash
git add ggml/src/ggml-sycl/common.hpp
git commit -m "feat(sycl): Add COALESCED reorder_mode enum value"
```

---

### Task 3: Add is_coalesced() helper function

**Files:**
- Modify: `ggml/src/ggml-sycl/common.hpp` (same file as Task 2)

**Step 1: Find is_soa() function location**

```bash
grep -n "is_soa" ggml/src/ggml-sycl/common.hpp
```

**Step 2: Add is_coalesced() alongside is_soa()**

```cpp
// In optimized_feature struct or wherever is_soa() is defined:
bool is_coalesced() const {
    return mode == reorder_mode::COALESCED;
}
```

**Step 3: Build to verify**

```bash
./scripts/quick-rebuild.sh common.hpp
```
Expected: Build succeeds

**Step 4: Commit**

```bash
git add ggml/src/ggml-sycl/common.hpp
git commit -m "feat(sycl): Add is_coalesced() helper function"
```

---

### Task 4: Add GGML_SYCL_REORDER_MODE environment variable

**Files:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp`

**Step 1: Find existing env var parsing**

```bash
grep -n "GGML_SYCL_DISABLE_OPT\|getenv" ggml/src/ggml-sycl/ggml-sycl.cpp | head -20
```

**Step 2: Add reorder mode selection**

Near existing env var parsing:
```cpp
static reorder_mode get_reorder_mode() {
    const char* mode = getenv("GGML_SYCL_REORDER_MODE");
    if (mode == nullptr) {
        // Default: use SoA (existing behavior)
        return getenv("GGML_SYCL_DISABLE_OPT") ? reorder_mode::NONE : reorder_mode::SOA;
    }
    if (strcmp(mode, "none") == 0) return reorder_mode::NONE;
    if (strcmp(mode, "soa") == 0) return reorder_mode::SOA;
    if (strcmp(mode, "coalesced") == 0) return reorder_mode::COALESCED;
    fprintf(stderr, "WARN: Unknown GGML_SYCL_REORDER_MODE '%s', using soa\n", mode);
    return reorder_mode::SOA;
}
```

**Step 3: Build and test env var**

```bash
./scripts/quick-rebuild.sh ggml-sycl.cpp
GGML_SYCL_REORDER_MODE=coalesced ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-completion -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -ngl 99 --flash-attn on -p "Hi" -n 1 --seed 42 --temp 0
```
Expected: Runs (falls back to SoA since coalesced kernels don't exist yet)

**Step 4: Commit**

```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "feat(sycl): Add GGML_SYCL_REORDER_MODE environment variable"
```

---

## Phase 3: Q4_0 DMMV Coalesced Kernel

### Task 5: Write failing test for Q4_0 coalesced DMMV

**Files:**
- Create: `tests/test-dmmv-q4-0-coalesced.cpp`
- Modify: `tests/CMakeLists.txt`

**Step 1: Create test file**

```cpp
// tests/test-dmmv-q4-0-coalesced.cpp
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

// Test that coalesced layout produces same output as SoA
int main() {
    // This test will:
    // 1. Create Q4_0 data in AoS format
    // 2. Reorder to SoA, run DMMV, get output_soa
    // 3. Reorder to COALESCED, run DMMV, get output_coalesced
    // 4. Compare outputs (must match within tolerance)

    printf("TEST: Q4_0 DMMV Coalesced vs SoA comparison\n");

    // TODO: Implement after coalesced kernel exists
    // For now, fail to drive TDD
    fprintf(stderr, "FAIL: Coalesced DMMV kernel not implemented\n");
    return 1;
}
```

**Step 2: Add to CMakeLists.txt**

In `tests/CMakeLists.txt`, add after other SYCL tests:
```cmake
if (GGML_SYCL)
    add_executable(test-dmmv-q4-0-coalesced test-dmmv-q4-0-coalesced.cpp)
    target_link_libraries(test-dmmv-q4-0-coalesced PRIVATE ggml)
    target_compile_options(test-dmmv-q4-0-coalesced PRIVATE "-fsycl")
    target_link_options(test-dmmv-q4-0-coalesced PRIVATE "-fsycl")
endif()
```

**Step 3: Build and run test**

```bash
cmake --build build --target test-dmmv-q4-0-coalesced
./build/bin/test-dmmv-q4-0-coalesced
```
Expected: FAIL with "Coalesced DMMV kernel not implemented"

**Step 4: Commit failing test**

```bash
git add tests/test-dmmv-q4-0-coalesced.cpp tests/CMakeLists.txt
git commit -m "test(sycl): Add failing test for Q4_0 coalesced DMMV"
```

---

### Task 6: Implement Q4_0 coalesced reorder kernel

**Files:**
- Modify: `ggml/src/ggml-sycl/convert.cpp`
- Reference: Existing `reorder_qw_q4_0_soa` function

**Step 1: Find existing reorder function**

```bash
grep -n "reorder_qw_q4_0" ggml/src/ggml-sycl/convert.cpp
```

**Step 2: Add coalesced reorder kernel**

```cpp
// After reorder_qw_q4_0_soa, add:
static void reorder_qw_q4_0_coalesced(
    const block_q4_0* src,
    int8_t* dst,
    int64_t ne00,
    int64_t ne01,
    dpct::queue_ptr stream)
{
    // Coalesced layout: interleave across WARP_SIZE (32) threads
    // Thread i accesses bytes i, i+32, i+64, ...
    const int blocks_per_row = ne00 / QK4_0;
    const int warp_groups = (blocks_per_row + WARP_SIZE - 1) / WARP_SIZE;

    stream->parallel_for(
        sycl::nd_range<2>({(size_t)ne01, (size_t)(warp_groups * WARP_SIZE)},
                          {1, WARP_SIZE}),
        [=](sycl::nd_item<2> it) {
            int row = it.get_global_id(0);
            int tid = it.get_local_id(1);
            int warp_group = it.get_group(1);

            int block_idx = warp_group * WARP_SIZE + tid;
            if (block_idx >= blocks_per_row) return;

            // Source: standard block_q4_0 layout
            const block_q4_0* src_block = &src[row * blocks_per_row + block_idx];

            // Destination: coalesced layout
            // qs: each thread writes to position [warp_group * WARP_SIZE * QK4_0/2 + tid * QK4_0/2]
            // d: grouped after all qs for the warp group

            int dst_row_offset = row * (blocks_per_row * (QK4_0/2 + sizeof(sycl::half)));
            int qs_base = dst_row_offset + warp_group * WARP_SIZE * (QK4_0/2);

            // Write qs (interleaved by tid)
            for (int i = 0; i < QK4_0/2; i++) {
                dst[qs_base + i * WARP_SIZE + tid] = src_block->qs[i];
            }

            // Write d (grouped at end of warp group)
            int d_offset = dst_row_offset + blocks_per_row * (QK4_0/2) + block_idx * sizeof(sycl::half);
            *(sycl::half*)(dst + d_offset) = src_block->d;
        });
}
```

**Step 3: Build**

```bash
./scripts/quick-rebuild.sh convert.cpp
```
Expected: Build succeeds

**Step 4: Commit**

```bash
git add ggml/src/ggml-sycl/convert.cpp
git commit -m "feat(sycl): Add Q4_0 coalesced reorder kernel"
```

---

### Task 7: Implement Q4_0 coalesced DMMV kernel

**Files:**
- Modify: `ggml/src/ggml-sycl/dmmv.cpp`
- Reference: Existing `dequantize_mul_mat_vec_q4_0_sycl` function

**Step 1: Find existing DMMV function**

```bash
grep -n "dequantize_mul_mat_vec_q4_0" ggml/src/ggml-sycl/dmmv.cpp | head -5
```

**Step 2: Add coalesced DMMV kernel**

```cpp
// Add coalesced version that reads from coalesced layout
template<int BLOCK_SIZE>
static void dmmv_q4_0_coalesced_kernel(
    const int8_t* __restrict__ x_coalesced,
    const float* __restrict__ y,
    float* __restrict__ dst,
    int ncols,
    int nrows,
    sycl::nd_item<1> it)
{
    int tid = it.get_local_id(0);  // 0-31 within warp
    int row = it.get_group(0);
    if (row >= nrows) return;

    const int blocks_per_row = ncols / QK4_0;
    const int warp_groups = (blocks_per_row + WARP_SIZE - 1) / WARP_SIZE;

    float sum = 0.0f;

    for (int wg = 0; wg < warp_groups; wg++) {
        int block_idx = wg * WARP_SIZE + tid;
        if (block_idx >= blocks_per_row) continue;

        // Coalesced read: all 32 threads read adjacent bytes
        int row_offset = row * (blocks_per_row * (QK4_0/2 + sizeof(sycl::half)));
        int qs_base = row_offset + wg * WARP_SIZE * (QK4_0/2);

        // Read d (coalesced within warp)
        int d_offset = row_offset + blocks_per_row * (QK4_0/2) + block_idx * sizeof(sycl::half);
        float d = *(const sycl::half*)(x_coalesced + d_offset);

        // Read and dequantize qs
        for (int i = 0; i < QK4_0/2; i++) {
            int8_t qs = x_coalesced[qs_base + i * WARP_SIZE + tid];

            // Dequantize two values from nibbles
            int v0 = (qs & 0x0F) - 8;
            int v1 = ((qs >> 4) & 0x0F) - 8;

            int yi = block_idx * QK4_0 + i * 2;
            sum += d * v0 * y[yi];
            sum += d * v1 * y[yi + 1];
        }
    }

    // Warp reduction
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        sum += sycl::shift_group_left(it.get_sub_group(), sum, offset);
    }

    if (tid == 0) {
        dst[row] = sum;
    }
}
```

**Step 3: Add dispatch for coalesced mode**

```cpp
// In the main dispatch function, add:
if (src0_extra->optimized_feature.is_coalesced()) {
    stream->parallel_for(
        sycl::nd_range<1>({(size_t)(nrows * WARP_SIZE)}, {WARP_SIZE}),
        [=](sycl::nd_item<1> it) {
            dmmv_q4_0_coalesced_kernel<WARP_SIZE>(
                (const int8_t*)src0_d, src1_d, dst_d, ncols, nrows, it);
        });
    return;
}
```

**Step 4: Build**

```bash
./scripts/quick-rebuild.sh dmmv.cpp
```
Expected: Build succeeds

**Step 5: Commit**

```bash
git add ggml/src/ggml-sycl/dmmv.cpp
git commit -m "feat(sycl): Add Q4_0 coalesced DMMV kernel"
```

---

### Task 8: Update test and verify correctness

**Files:**
- Modify: `tests/test-dmmv-q4-0-coalesced.cpp`

**Step 1: Implement full test**

Update the test to actually compare SoA vs Coalesced outputs using the GGML API.

**Step 2: Run test**

```bash
cmake --build build --target test-dmmv-q4-0-coalesced
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-dmmv-q4-0-coalesced
```
Expected: PASS

**Step 3: Run end-to-end verification**

```bash
# SoA output
GGML_SYCL_REORDER_MODE=soa ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-completion -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -ngl 99 --flash-attn on -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0 > /tmp/soa.txt

# Coalesced output
GGML_SYCL_REORDER_MODE=coalesced ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-completion -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -ngl 99 --flash-attn on -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0 > /tmp/coalesced.txt

diff /tmp/soa.txt /tmp/coalesced.txt
```
Expected: No difference

**Step 4: Commit passing test**

```bash
git add tests/test-dmmv-q4-0-coalesced.cpp
git commit -m "test(sycl): Q4_0 coalesced DMMV test passing"
```

---

### Task 9: Benchmark Q4_0 coalesced vs SoA

**Files:**
- Create: `benchmark_results/vtune_coalesced_q4_0/`

**Step 1: Run coalesced benchmark**

```bash
GGML_SYCL_REORDER_MODE=coalesced ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-bench -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p 512 -n 128 -ngl 99 -fa 1
```

**Step 2: Compare with SoA baseline**

```bash
GGML_SYCL_REORDER_MODE=soa ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-bench -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p 512 -n 128 -ngl 99 -fa 1
```

**Step 3: Run VTune on coalesced**

```bash
GGML_SYCL_REORDER_MODE=coalesced ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  vtune -collect gpu-hotspots -knob gpu-sampling-interval=1 \
  -result-dir benchmark_results/vtune_coalesced_q4_0 \
  -- ./build/bin/llama-bench -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p 512 -n 128 -ngl 99 -fa 1
```

**Step 4: Document results**

Update `benchmark_results/vtune_coalesced_q4_0/README.md` with comparison.

**Step 5: Commit**

```bash
git add benchmark_results/
git commit -m "perf: Q4_0 coalesced benchmark results"
```

---

## Phase 4: Remaining Formats (repeat pattern)

### Task 10-12: Q8_0 DMMV Coalesced
- Same pattern as Tasks 5-9 for Q8_0 format

### Task 13-15: Q6_K DMMV Coalesced
- Same pattern for Q6_K format

### Task 16-18: MMQ Kernels
- Same pattern for matrix-multiply quantized kernels

### Task 19-21: MMVQ Kernels
- Same pattern for matrix-matrix vector quantized kernels

---

## Phase 5: Finalization

### Task 22: Update documentation

**Files:**
- Modify: `docs/ENV.md`
- Modify: `OPTIMIZATION_REPORT.md`

**Step 1: Add GGML_SYCL_REORDER_MODE to ENV.md**

**Step 2: Add coalesced benchmark results to OPTIMIZATION_REPORT.md**

**Step 3: Commit**

```bash
git add docs/ENV.md OPTIMIZATION_REPORT.md
git commit -m "docs: Add coalesced reorder mode documentation"
```

---

### Task 23: Merge to main branch

**Step 1: Verify all tests pass**

```bash
ctest --test-dir build --output-on-failure -R coalesced
```

**Step 2: Merge feature branch**

```bash
git checkout sycl-xmx-flash-attention
git merge feature/sycl-coalescing --no-ff -m "feat(sycl): Add coalesced memory layout optimization"
```

**Step 3: Push**

```bash
git push fork sycl-xmx-flash-attention
```

---

## Success Criteria

- [ ] VTune shows >80% memory bandwidth utilization for DMMV kernels
- [ ] Token output identical between SoA and COALESCED modes
- [ ] Performance improvement on tg benchmarks
- [ ] All unit tests passing
- [ ] Documentation updated
