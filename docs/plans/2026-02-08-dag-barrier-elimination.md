# DAG Barrier Elimination Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use team-driven-development to implement this plan with agent teams.

**Goal:** Replace device-scope barriers in the persistent TG kernel with per-operation atomic dependency counters (DAG scheduling), eliminating ~580ms/token barrier overhead to reach ≥76 tok/s.

**Architecture:** The persistent kernel executes ~707 DeviceOperations per token for Mistral 7B. Currently each operation is separated by a device-scope barrier (~0.83ms each on Intel Arc B580's 18MB L2). DAG scheduling tracks per-operation dependencies via atomic counters: producer ops decrement successor ready-counters on completion; consumer ops begin as soon as their ready-counter reaches zero. No device-wide synchronization required.

**Tech Stack:** SYCL 2020, Intel oneAPI DPC++, Level Zero, Intel Arc B580 (Xe2, BMG-G21)

**Research basis:** Mirage MPK (arXiv 2512.22219), Hazy Research "No Bubbles" megakernel, Luminal megakernels — all use counter-based DAG scheduling. Full analysis in `/home/kainlan/.claude/plans/fluffy-prancing-treehouse.md`.

---

## Current State

Phase 1 bug fixes (Steps 1.1-1.3 from the research plan) are **implemented but don't compile**. The DAG infrastructure is ~90% complete:

| Component | Status |
|-----------|--------|
| `DeviceDAGState` struct | Done |
| `run_dag()` kernel method | Done (has 1 compile error) |
| `build_dag()` host method | Done (has 1 compile error) |
| `reset_dag_counters()` | Done |
| DAG edge construction (ggml-sycl.cpp) | Done |
| Fast-path DAG reset | Done |
| `GGML_SYCL_PERSISTENT_TG_DAG` env var | Done |
| Work-group count policy (DAG mode) | Done |
| Kernel dispatch branching | Done |

### Build Errors to Fix

**Error 1: Invalid `atomic_ref` default memory order** (`unified-kernel.cpp:4119`)
```cpp
// BROKEN: acquire not valid as DefaultOrder
sycl::atomic_ref<int, sycl::memory_order::acquire, ...> cc(*dag.completed_count);
```
Fix: Change to `sycl::memory_order::acq_rel` (valid default), use `.load(sycl::memory_order::acquire)` explicitly.

**Error 2: `n_tiles` not a member of `OperationDescriptor`** (`unified-kernel.cpp:5413`)
```cpp
// BROKEN: OperationDescriptor doesn't have n_tiles
tile_counts[i] = current_plan_->operations[i].n_tiles;
```
Fix: Remove tile_counts logic from `build_dag()`. Instead, upload `n_tiles` from `launch_persistent_kernel()` after `DeviceOperation` array is built (where `n_tiles` IS computed). This is a sequencing issue — tile counts depend on operation type + matrix dimensions, computed during DeviceOperation creation, which happens AFTER build_dag().

---

## Team Topology

**Recommended implementers:** 2 (based on 2 parallel tracks after build fix)
**Reviewers:** 1 quality-reviewer

### Parallel Tracks

| Track | Tasks | Description |
|-------|-------|-------------|
| A | 1, 2 | Build fixes + compilation (sequential, same files) |
| B | 3 | Correctness verification (after build) |
| C | 4 | Performance benchmarking (after build, parallel with B) |
| — | 5 | Enable by default (convergence, depends on B + C) |

### Dependency Graph

```
Task 1 (Fix build errors)
  └─► Task 2 (Build verification)
        ├─► Task 3 (Correctness testing)     [Track B]
        └─► Task 4 (Performance benchmarking) [Track C]
              │
              └─► Task 5 (Enable by default) ◄── Task 3
```

### File Ownership Map

| File | Tasks | Conflict Risk |
|------|-------|---------------|
| `ggml/src/ggml-sycl/unified-kernel.cpp` | 1 | None (single task) |
| `ggml/src/ggml-sycl/unified-kernel.hpp` | 1 | None (single task) |
| `ggml/src/ggml-sycl/dispatch.hpp` | 5 | None (runs last) |
| Runtime only (no file edits) | 2, 3, 4 | None |

---

## Task Details

### Task 1: Fix Build Errors in DAG Implementation

**Track:** A
**Depends on:** None
**Priority:** P0 (blocking everything else)
**File scope:**
- Modify: `ggml/src/ggml-sycl/unified-kernel.cpp:4119-4123` (atomic_ref fix)
- Modify: `ggml/src/ggml-sycl/unified-kernel.cpp:5410-5415` (remove n_tiles from build_dag)
- Modify: `ggml/src/ggml-sycl/unified-kernel.cpp:6175-6180` (add n_tiles upload in launch_persistent_kernel)

**Description:**

Two compilation errors prevent the DAG code from building. Both are in `unified-kernel.cpp`.

**Fix 1: atomic_ref default memory order (line 4119-4123)**

The `run_dag()` method's termination check uses `memory_order::acquire` as the `atomic_ref` default template parameter. SYCL 2020 only allows `relaxed`, `acq_rel`, or `seq_cst` as defaults.

Current (broken):
```cpp
sycl::atomic_ref<int, sycl::memory_order::acquire,
                 sycl::memory_scope::device,
                 sycl::access::address_space::global_space>
    cc(*dag.completed_count);
if (cc.load() >= n_ops) op_idx = -2;  // TERMINATE
```

Fixed:
```cpp
sycl::atomic_ref<int, sycl::memory_order::acq_rel,
                 sycl::memory_scope::device,
                 sycl::access::address_space::global_space>
    cc(*dag.completed_count);
if (cc.load(sycl::memory_order::acquire) >= n_ops) op_idx = -2;  // TERMINATE
```

**Fix 2: n_tiles sequencing (line 5410-5415 and ~6175)**

`build_dag()` tries to read `current_plan_->operations[i].n_tiles` but `OperationDescriptor` (the plan's operation type) doesn't have that field. Tile counts are computed later in `launch_persistent_kernel()` when building the `DeviceOperation` array (which DOES have `n_tiles` at line 3994).

Step A — Remove from `build_dag()` (lines 5410-5415):
```cpp
// DELETE these lines:
    // Read tile counts from the current plan
    std::vector<int> tile_counts(n_ops);
    for (int i = 0; i < n_ops; i++) {
        tile_counts[i] = current_plan_->operations[i].n_tiles;
    }
    host_n_tiles_ = tile_counts;
```

And delete the n_tiles upload line (5422):
```cpp
// DELETE:
    queue_.memcpy(dag_state_.n_tiles, tile_counts.data(), n_ops * sizeof(int));
```

Step B — Add n_tiles upload to `launch_persistent_kernel()`, after the `host_ops` DeviceOperation vector is built (~line 6178, after the loop that populates host_ops and before the device memcpy):
```cpp
    // Upload tile counts to DAG state (tile counts are computed during DeviceOperation creation)
    if (use_dag_mode && dag_allocated_) {
        std::vector<int> tile_counts(host_ops.size());
        for (size_t i = 0; i < host_ops.size(); i++) {
            tile_counts[i] = host_ops[i].n_tiles;
        }
        host_n_tiles_ = tile_counts;
        queue_.memcpy(dag_state_.n_tiles, tile_counts.data(), host_ops.size() * sizeof(int));
    }
```

**Acceptance Criteria:**
- [ ] `ninja -C build` completes with zero errors in unified-kernel.cpp
- [ ] No new warnings introduced
- [ ] `run_dag()` uses valid SYCL atomic_ref template parameters
- [ ] `build_dag()` no longer references `n_tiles` on `OperationDescriptor`
- [ ] n_tiles array is uploaded to device in `launch_persistent_kernel()` from DeviceOperation data

**Notes for implementer:**
- Source oneAPI first: `source /opt/intel/oneapi/setvars.sh --force`
- Build command: `ninja -C build -j $(nproc)`
- The `DeviceOperation` struct is at line 3965, `OperationDescriptor` at unified-kernel.hpp:302
- `host_ops` vector is built starting around line 6090 in `launch_persistent_kernel()`
- Look for the `queue_.memcpy(d_ops, host_ops.data(), ...)` line — insert the n_tiles upload just before it
- If additional SYCL_EXTERNAL errors appear, they likely mean `dispatch_operation()` calls functions not visible to device compiler — may need forward declarations

---

### Task 2: Build Verification

**Track:** A
**Depends on:** Task 1
**Priority:** P0

**Description:**

After Task 1's code fixes, verify the entire project builds cleanly. This is a gate task — nothing else can proceed until the build passes.

**Steps:**
```bash
source /opt/intel/oneapi/setvars.sh --force
ninja -C build -j $(nproc)
```

**Acceptance Criteria:**
- [ ] Build completes with exit code 0
- [ ] No errors in unified-kernel.cpp, unified-kernel.hpp, or ggml-sycl.cpp
- [ ] Binary `build/bin/llama-bench` exists and is executable
- [ ] Binary `build/bin/llama-completion` exists and is executable

**Notes for implementer:**
- Build takes ~10min with ccache, ~25min without
- If new errors appear, fix them — there may be cascading issues from the SYCL device compiler
- Common issue: `SYCL kernel cannot call an undefined function without SYCL_EXTERNAL attribute` — means a function called from kernel code isn't visible. Check if `dispatch_operation()` references any functions defined after the kernel lambda.

---

### Task 3: Correctness Testing

**Track:** B
**Depends on:** Task 2
**Priority:** P0

**Description:**

Verify DAG mode produces identical output to legacy barrier mode. Both modes execute the same operations — only the synchronization mechanism differs. Any output difference indicates a memory ordering or scheduling bug.

**Test 1: DAG mode deterministic output**
```bash
source /opt/intel/oneapi/setvars.sh --force
ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_PERSISTENT_TG=1 \
  ./build/bin/llama-completion -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```
Expected output should contain: `6, 7, 8, 9, 10, 11, 12, 13, 14, 15`

**Test 2: Legacy barriers (DAG disabled)**
```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_PERSISTENT_TG=1 GGML_SYCL_PERSISTENT_TG_DAG=0 \
  ./build/bin/llama-completion -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```

**Test 3: Standard dispatch (persistent kernel off)**
```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-completion -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```

**Acceptance Criteria:**
- [ ] All three tests produce identical token output
- [ ] No crashes, hangs, or GPU errors
- [ ] DAG mode completes within 30 seconds (not stuck in spin-wait)
- [ ] Run DAG mode 3 times consecutively — all produce identical output

**Notes for implementer:**
- CRITICAL: Use `ONEAPI_DEVICE_SELECTOR=level_zero:0` — without it the system hangs on multi-GPU
- If DAG mode hangs: likely a dependency cycle or missing reset. Check `reset_dag_counters()` runs before each token.
- If output differs: likely a memory ordering issue. The `acq_rel` on `tiles_done.fetch_add()` must happen-before successor's `ready_counter.load()`.
- Model path: `/Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf`

---

### Task 4: Performance Benchmarking

**Track:** C (parallel with Task 3)
**Depends on:** Task 2
**Priority:** P1

**Description:**

Benchmark DAG mode performance and verify no PP regression. The key metrics are:
- TG128 with persistent kernel DAG mode: target ≥76 tok/s (parity with master)
- PP512 standard mode: target ~1300 tok/s (no regression from DAG code)

**Benchmark 1: TG with DAG mode**
```bash
source /opt/intel/oneapi/setvars.sh --force
ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_PERSISTENT_TG=1 \
  ./build/bin/llama-bench -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 0 -n 128
```
Target: ≥76 tok/s. Baseline with barriers: 1.71 tok/s.

**Benchmark 2: TG with legacy barriers (comparison)**
```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_PERSISTENT_TG=1 GGML_SYCL_PERSISTENT_TG_DAG=0 \
  ./build/bin/llama-bench -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 0 -n 128
```
Expected: ~1.71 tok/s (unchanged from before).

**Benchmark 3: PP regression check**
```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-bench -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 0
```
Target: ~1300 tok/s.

**Benchmark 4: Work-group count sweep** (only if DAG TG ≥40 tok/s)
```bash
for wgs in 4 8 16 32 40 64; do
  ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_PERSISTENT_TG=1 \
  GGML_SYCL_PERSISTENT_TG_N_WGS=$wgs \
  ./build/bin/llama-bench -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 0 -n 128 2>&1 | tail -1
done
```

**Acceptance Criteria:**
- [ ] TG128 DAG mode: record actual tok/s (target ≥76)
- [ ] PP512: ≥1200 tok/s (no regression)
- [ ] Run each benchmark 3 times, report mean and std
- [ ] Document optimal work-group count if sweep is run

**Notes for implementer:**
- Each llama-bench run takes ~30-60 seconds
- PP benchmark does NOT use persistent kernel (it's a batch operation)
- Hardware: Intel Arc B580, 160 CUs, 12GB GDDR6
- The DAG work-group count defaults to max_cu/2 = 80. Override with GGML_SYCL_PERSISTENT_TG_N_WGS.

---

### Task 5: Enable Persistent TG by Default (Conditional)

**Track:** D (convergence)
**Depends on:** Task 3, Task 4
**Priority:** P2

**Description:**

If Task 3 passes correctness AND Task 4 shows TG ≥76 tok/s with no PP regression, change the default to enable persistent TG kernel.

**Only proceed if:**
- Task 3: All correctness tests pass
- Task 4: TG128 ≥76 tok/s AND PP512 ≥1200 tok/s

**File:** `ggml/src/ggml-sycl/dispatch.hpp:166`

Current code (persistent TG disabled by default):
```cpp
static bool env_persistent_tg_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = std::getenv("GGML_SYCL_PERSISTENT_TG");
        enabled = (env != nullptr && std::atoi(env) != 0) ? 1 : 0;
    }
    return enabled != 0;
}
```

Changed (enabled by default, opt-out with `GGML_SYCL_PERSISTENT_TG=0`):
```cpp
static bool env_persistent_tg_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = std::getenv("GGML_SYCL_PERSISTENT_TG");
        enabled = (env != nullptr && std::atoi(env) == 0) ? 0 : 1;
    }
    return enabled != 0;
}
```

**Post-change validation:**
```bash
# Default (persistent TG now ON):
ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  ./build/bin/llama-bench -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128

# Opt-out still works:
ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_PERSISTENT_TG=0 \
  ./build/bin/llama-bench -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128
```

**Acceptance Criteria:**
- [ ] Default mode uses persistent TG with DAG scheduling
- [ ] `GGML_SYCL_PERSISTENT_TG=0` disables persistent kernel
- [ ] Both PP and TG benchmarks pass with default settings
- [ ] Correctness test passes with default settings

**Notes for implementer:**
- DO NOT proceed with this task if performance targets are not met
- If TG is faster but < 76 tok/s, create a follow-up issue instead of enabling by default
- The opt-out logic flips: was "opt-in with =1", becomes "opt-out with =0"
