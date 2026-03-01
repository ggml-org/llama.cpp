# Persistent TG Scratch Pool — Eliminate ggml Buffer Aliasing

> **For Claude:** REQUIRED SUB-SKILL: Use team-driven-development to implement this plan with agent teams.

**Goal:** Fix the persistent TG kernel producing wrong output by eliminating ggml buffer aliasing via a dedicated scratch pool allocated through the unified cache.

**Architecture:** Each persistent kernel operation gets a dedicated scratch output buffer (256-byte aligned, sub-allocated from one contiguous `sycl::malloc_device` pool). A forward-pass pointer remap rewires the plan's data-flow chain so no two operations share an output address. The UPDATE path (fast token path) uses `scratch_output(op_idx)` to return the stable scratch pointer instead of the ggml-aliased pointer.

**Tech Stack:** SYCL, unified cache (`unified_cache_add/sub_runtime_bytes`, `runtime_category::GRAPH`), ggml tensor library

**Beads Bug:** `llama.cpp-l714` (P0, in_progress)

---

## Team Topology

**Recommended implementers:** 1 (sequential dependency chain — all tasks touch related code)
**Reviewers:** 1 quality-reviewer

### Task Ordering (Sequential)

| Order | Task (Beads ID) | Description | Est. Lines |
|-------|-----------------|-------------|------------|
| 1 | FIX-1 (`llama.cpp-rbt4`) | Add output_bytes to all ops in full_build | ~15 |
| 2 | FIX-2 (`llama.cpp-d49e`) | Scratch pool infrastructure in UnifiedKernel | ~80 |
| 3 | FIX-3 (`llama.cpp-gwrk`) | Integrate build_scratch_pool() into full_build path | ~3 |
| 4 | FIX-4 (`llama.cpp-og3r`) | Replace output pointers with scratch in UPDATE path | ~30 |
| 5 | FIX-5 (`llama.cpp-2oyd`) | Fix SET_ROWS input resolution + full verification | ~5 |

### Dependency Graph

```
FIX-1 (output_bytes) ──→ FIX-2 (scratch pool infra)
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
             FIX-3 (full_build)   FIX-4 (UPDATE path)
                                        │
                                        ▼
                                 FIX-5 (SET_ROWS + verify)
```

### File Ownership Map

| File | Tasks | Conflict Risk |
|------|-------|---------------|
| `ggml/src/ggml-sycl/unified-kernel.hpp` | FIX-2 | None (single task) |
| `ggml/src/ggml-sycl/unified-kernel.cpp` | FIX-2 | None (single task) |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` | FIX-1, FIX-3, FIX-4, FIX-5 | Sequential (dependency chain) |

---

## Root Cause Analysis

ggml's `gallocr` reuses device addresses for compute tensors whose lifetimes don't overlap in sequential per-op dispatch. The persistent kernel fuses ALL ops into one GPU launch, so ALL intermediates are live simultaneously. When two ops share the same output address, the second overwrites the first's data before downstream consumers read it.

**Evidence**: Q_PROJ matmul output differs per call (matmul math works), but final SET_ROWS logits are identical across all calls. Even MAX_OPS=1 produces wrong output. Barriers, ROPE math, cos/sin caches all verified correct.

**Key insight**: `extra->data_device[dev]` is STABLE between tokens (same gallocr layout). The aliasing is within a single execution, not between calls.

---

## What Gets Scratch vs What Doesn't

| Operation | Gets scratch? | Why |
|-----------|--------------|-----|
| RMS_NORM | YES | Intermediate, aliased by gallocr |
| MUL_MAT | YES | Intermediate, aliased by gallocr |
| MUL | YES | Intermediate |
| ADD | YES | Intermediate |
| GLU/SILU_MUL | YES | Intermediate |
| ROPE | YES | Intermediate |
| SOFTMAX | YES | Intermediate |
| FLASH_ATTN | YES | Intermediate |
| GET_ROWS | NO | Uses dedicated `get_rows_stable_ptr` pool |
| SET_ROWS | NO | Output is ggml logits buffer — sampler reads directly |
| STRIDED_COPY | NO | Already uses dedicated alloc from `resolve_input_ptr` |
| CPY | NO | Runs outside persistent kernel (pre-executed) |

---

## Memory Budget

For Mistral 7B Q4_0 (batch=1, hidden=4096, intermediate=14336, n_heads=32):
- Per layer: ~14 ops x avg ~16KB each = ~224KB
- 32 layers + prefix/suffix: ~7.5MB total scratch
- Well within GRAPH budget (unified cache tracks via `runtime_category::GRAPH`)

---

## Task Details

### Task 1: FIX-1 — Add output_bytes to all persistent kernel ops (`llama.cpp-rbt4`)

**Depends on:** None
**File scope:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` lines 29885-32900 (full_build path)

**Description:**
`build_scratch_pool()` reads `OperationDescriptor.output_bytes` to know how much scratch per op. Currently only SET_ROWS (line 31334) and CONT (line 31411) populate this field. ALL ops need it.

**Implementation Guide:**

For each operation in the full_build path, add `output_bytes` computation before the `kernel.add_*()` call. Use `ggml_nbytes(node)` for element-wise ops. For MUL_MAT, use `M * N * sizeof(float)` (matmul output is always float).

**Exact locations** (search for each `kernel.add_` in full_build):
- `kernel.add_rms_norm(...)` — add `output_bytes = ggml_nbytes(node)` before
- `kernel.add_matmul(...)` — add `output_bytes = (int64_t)(M * N) * sizeof(float)` before
- `kernel.add_mul(...)` — add `output_bytes = ggml_nbytes(node)` before
- `kernel.add_add(...)` — add `output_bytes = ggml_nbytes(node)` before
- `kernel.add_glu(...)` — add `output_bytes = ggml_nbytes(node)` before
- `kernel.add_rope(...)` — add `output_bytes = ggml_nbytes(node)` before
- `kernel.add_softmax(...)` — add `output_bytes = ggml_nbytes(node)` before
- `kernel.add_flash_attn(...)` — add `output_bytes = ggml_nbytes(node)` before

Each `kernel.add_*()` method accepts or stores `output_bytes` in the OperationDescriptor. Check the signature to see if it's a parameter or if you need to set it on the descriptor after the call.

**Commit:**
```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "sycl: propagate output_bytes for all persistent kernel ops"
```

---

### Task 2: FIX-2 — Scratch pool infrastructure in UnifiedKernel (`llama.cpp-d49e`)

**Depends on:** FIX-1
**File scope:**
- Modify: `ggml/src/ggml-sycl/unified-kernel.hpp` (add fields + method declarations)
- Modify: `ggml/src/ggml-sycl/unified-kernel.cpp` (implement methods + wire cleanup)

**Description:**
Add scratch pool allocation, sub-allocation, forward-pass remap, and cleanup. Follow the existing `get_rows_stable_ptr()` grow-on-demand pattern.

**Implementation Guide:**

**Part A — Header** (`unified-kernel.hpp`):

After the `get_rows_pool_` fields (line 2707), add:
```cpp
// Scratch pool for persistent kernel — eliminates ggml buffer aliasing
void *                  scratch_pool_        = nullptr;
size_t                  scratch_pool_size_   = 0;
std::vector<void *>     scratch_outputs_;     // per-op scratch ptrs (nullptr = use ggml)
```

In public section, add:
```cpp
void   build_scratch_pool();
void * scratch_output(int op_idx) const;
void   free_scratch_pool();
```

**Part B — Implementation** (`unified-kernel.cpp`):

Implement `build_scratch_pool()`:
1. Walk `current_plan_->operations`, compute per-op output sizes from `output_bytes`
2. Skip SET_ROWS, GET_ROWS, STRIDED_COPY (they have own buffers)
3. Sum with 256-byte alignment per slot
4. Grow-on-demand: `sycl::malloc_device` + `unified_cache_add_runtime_bytes(device_id_, size, runtime_category::GRAPH)`
5. Sub-allocate: `scratch_outputs_[i] = pool_base + offset`
6. Forward-pass remap: walk ops, for each op remap `input`/`aux` if in remap table, register `output → scratch` in remap table, update `op.output = scratch`

Implement `scratch_output()`:
```cpp
void * UnifiedKernel::scratch_output(int op_idx) const {
    return (op_idx >= 0 && op_idx < (int)scratch_outputs_.size()) ? scratch_outputs_[op_idx] : nullptr;
}
```

Implement `free_scratch_pool()`:
```cpp
void UnifiedKernel::free_scratch_pool() {
    if (scratch_pool_) {
        if (scratch_pool_size_ > 0 && device_id_ >= 0)
            ggml_sycl::unified_cache_sub_runtime_bytes(device_id_, scratch_pool_size_, ggml_sycl::runtime_category::GRAPH);
        sycl::free(scratch_pool_, queue_);
        scratch_pool_ = nullptr;
        scratch_pool_size_ = 0;
    }
    scratch_outputs_.clear();
}
```

**Part C — Cleanup wiring**:
- `invalidate_plan_cache()` (line 6142): Add `free_scratch_pool();` at start
- `free_persistent_buffers()` (line 5449): Add `free_scratch_pool();` before `invalidate_plan_cache()`

**Pattern reference:** `get_rows_stable_ptr()` at lines 6162-6180 shows the exact grow-on-demand + budget tracking pattern.

**Commit:**
```bash
git add ggml/src/ggml-sycl/unified-kernel.hpp ggml/src/ggml-sycl/unified-kernel.cpp
git commit -m "sycl: add scratch pool infrastructure to eliminate ggml buffer aliasing"
```

---

### Task 3: FIX-3 — Integrate build_scratch_pool() into full_build (`llama.cpp-gwrk`)

**Depends on:** FIX-2
**File scope:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` line ~32901

**Description:**
Call `kernel.build_scratch_pool()` in the full_build path after all ops are added, before `execute_persistent()`.

**Implementation Guide:**

Find the execute_persistent() call at line ~32901:
```cpp
        try {
            kernel.execute_persistent();
```

Change to:
```cpp
        // Allocate scratch pool and remap aliased output pointers
        kernel.build_scratch_pool();

        try {
            kernel.execute_persistent();
```

**Commit:**
```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "sycl: integrate scratch pool into persistent TG full_build path"
```

---

### Task 4: FIX-4 — Replace output pointers in UPDATE path (`llama.cpp-og3r`)

**Depends on:** FIX-2
**File scope:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` lines 28874-29868 (UPDATE path)

**Description:**
Add `resolve_output_ptr` lambda and replace all output pointer assignments in the UPDATE path.

**Implementation Guide:**

**Step 1:** Add lambda after `resolve_input_ptr_no_materialize` (line 28874):
```cpp
auto resolve_output_ptr = [&](int op_idx, const ggml_tensor * node) -> void * {
    void * scratch = kernel.scratch_output(op_idx);
    if (scratch) return scratch;
    return get_tensor_ptr_view_fast(node);
};
```

**Step 2:** Replace each output pointer assignment:

| Line | Op | Old | New |
|------|----|-----|-----|
| 29076 | RMS_NORM | `get_tensor_ptr_view_fast(node)` | `resolve_output_ptr(op_idx, node)` |
| 29137 | MUL_MAT | `get_tensor_ptr_view_fast(node)` | `resolve_output_ptr(op_idx, node)` |
| 29186 | MUL | `get_tensor_ptr_view_fast(node)` | `resolve_output_ptr(op_idx, node)` |
| 29218 | ADD | `get_tensor_ptr_view_fast(node)` | `resolve_output_ptr(op_idx, node)` |
| 29250 | GLU | `get_tensor_ptr_view_fast(node)` | `resolve_output_ptr(op_idx, node)` |
| 29362 | ROPE | `get_tensor_ptr_view_fast(node)` | `resolve_output_ptr(op_idx, node)` |
| 29410 | SOFT_MAX | `get_tensor_ptr_view_fast(node)` | `resolve_output_ptr(op_idx, node)` |
| 29517 | FLASH_ATTN | `get_tensor_ptr_view_fast(node)` | `resolve_output_ptr(op_idx, node)` |

**Do NOT change:**
- Line 29058: GET_ROWS — uses `get_rows_stable_ptr` pattern (already correct)
- Lines 29702-29703: SET_ROWS inputs — handled by FIX-5
- Line 29704: SET_ROWS dst_ptr — sampler output, not aliased
- Line 29799-29800: CONT — uses dedicated alloc

The `materialized_ptrs[node]` assignments at lines 29102, 29173, 29205, 29237, 29266, 29384, 29479, 29674 already store the output variable, which will now contain the scratch pointer.

**Commit:**
```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "sycl: use scratch pool for output pointers in persistent TG UPDATE path"
```

---

### Task 5: FIX-5 — Fix SET_ROWS input resolution + verification (`llama.cpp-2oyd`)

**Depends on:** FIX-4
**File scope:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` line 29702

**Description:**
SET_ROWS reads from the logit matmul output via `get_tensor_ptr_view_fast(src0)`. After scratch remapping, the actual matmul output is in scratch. Change to `resolve_input_ptr_no_materialize(src0)`.

**Implementation Guide:**

**UPDATE path** (line 29702):
```cpp
// OLD:
const void * src_ptr = get_tensor_ptr_view_fast(src0);

// NEW:
const void * src_ptr = resolve_input_ptr_no_materialize(src0);
if (!src_ptr) src_ptr = get_tensor_ptr_view_fast(src0);  // fallback
```

`resolve_input_ptr_no_materialize()` checks `materialized_ptrs` first (which now contains scratch pointers from FIX-4). Falls back to ggml pointer only if the tensor wasn't materialized.

**Full build path** (line 31227): Verify that `build_scratch_pool()`'s forward-pass remap already handles this. The remap table maps old_output → scratch for all ops, and SET_ROWS's input field gets remapped if it matches a previous op's old output pointer. If this is the case, no change needed here.

**Verification:** Run the full test suite from the plan:
```bash
source /opt/intel/oneapi/setvars.sh --force
ninja -C build -j $(nproc)

# 1. Correctness
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
# MUST output: 6, 7, 8, 9, 10, 11, 12, 13, 14, 15

sleep 30

# 2. Performance
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128
# MUST: PP512 >= 1200, TG128 >= 68

sleep 30

# 3. Longer generation stability
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p 'The quick brown fox' -n 50 --seed 42 --temp 0

# 4. Legacy path
GGML_SYCL_PERSISTENT_TG=0 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```

**Commit:**
```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "sycl: fix SET_ROWS input resolution for scratch pool

Resolves llama.cpp-l714: persistent TG kernel produces wrong output
due to ggml buffer aliasing within single kernel execution."
```

---

## Risk Assessment

| Risk | Level | Mitigation |
|------|-------|-----------|
| Scratch pool OOM on large models | LOW | 7.5MB for Mistral 7B; budget-tracked via GRAPH category; fall back to non-persistent on alloc failure |
| Performance regression | LOW | Scratch is device-local; kernel writes to scratch at same speed; no extra copies |
| View tensor offset bugs in remap | MEDIUM | Forward-pass remap uses exact pointer matching; views with byte offsets won't match. UPDATE path resolves views through materialized_ptrs which handles offsets correctly |
| Stale scratch on plan invalidation | LOW | `invalidate_plan_cache()` calls `free_scratch_pool()`; next full_build allocates fresh |

---

## Key Architecture References

- **OperationDescriptor** struct: `unified-kernel.hpp:304-340` (output, input, aux, output_bytes fields)
- **DeviceOperation** struct: `unified-kernel.cpp:3964-4018` (device-side mirror of OperationDescriptor)
- **get_rows_stable_ptr()**: `unified-kernel.cpp:6162-6180` (grow-on-demand pattern to follow)
- **invalidate_plan_cache()**: `unified-kernel.cpp:6142-6160` (cleanup chain)
- **free_persistent_buffers()**: `unified-kernel.cpp:5449-5481` (budget untracking pattern)
- **launch_persistent_kernel()**: `unified-kernel.cpp:6237-6328` (OperationDescriptor → DeviceOperation conversion)
- **resolve_input_ptr_no_materialize**: `ggml-sycl.cpp:28847-28874` (materialized_ptrs lookup lambda)
- **UPDATE path**: `ggml-sycl.cpp:28985-29868` (fast token path)
- **full_build path**: `ggml-sycl.cpp:29869-32901` (first token path)
- **execute_persistent() call**: `ggml-sycl.cpp:32901` (launch site)
- **runtime_category::GRAPH**: `unified-cache.hpp:1095` (budget category for persistent TG)
