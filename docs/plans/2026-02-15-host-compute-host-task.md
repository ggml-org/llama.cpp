# HOST_COMPUTE All-Host-Task: Zero-Staging CPU Offload

> **For Claude:** REQUIRED SUB-SKILL: Use team-driven-development to implement this plan with agent teams.

**Goal:** Eliminate per-op staging overhead in CPU offload by running all CPU ops as `host_task` callbacks on the GPU in-order queue, with compute buffers allocated as host-pinned USM.

**Architecture:** When `GGML_SYCL_HOST_COMPUTE=1`, compute buffers are allocated via `sycl::malloc_host` (host-pinned USM). CPU ops execute as `host_task` on `gpu_q` instead of `parallel_for` on `cpu_q`. The in-order GPU queue naturally serializes GPU kernels and host_tasks — no cross-queue sync, no event chains, no staging copies needed. `get_host_ptr()` returns `t->data` directly for host-accessible buffers. `flush_output()` is a no-op.

**Tech Stack:** C++17, Intel oneAPI SYCL (icpx compiler), SYCL host_task API

**Beads Epic:** `llama.cpp-61i8`

---

## Team Topology

**Recommended implementers:** 2 (based on 2 parallel tracks after Task 1)
**Reviewers:** 1 (the team lead reviews between tasks)

### Parallel Tracks

| Track | Tasks | Description | Files |
|-------|-------|-------------|-------|
| A | 1, 2, 3 | cpu-dispatch.cpp host_task + mirror removal + boundary sync | cpu-dispatch.cpp, cpu-dispatch.hpp, ggml-sycl.cpp (lines 25160-25750) |
| B | 4 | unified-cache budget tracking | ggml-sycl.cpp (lines 6553-6579), unified-cache.hpp, unified-cache.cpp |
| — | 5 | Build + verify + benchmark (convergence) | No file changes |

### Dependency Graph

```
Task 1 (commit existing work)
  ├──→ Task 2 (remove mirror from cpu-dispatch)
  │      └──→ Task 3 (remove mirror from ggml-sycl + simplify boundary sync)
  │             └──→ Task 5 (build + verify)
  └──→ Task 4 (budget tracking — independent files)
                └──→ Task 5 (build + verify)
```

Tasks 2+3 (Track A) and Task 4 (Track B) can run in parallel.

### File Ownership Map

| File | Tasks | Conflict Risk |
|------|-------|---------------|
| `ggml/src/ggml-sycl/cpu-dispatch.cpp` | 1, 2 | Sequential (same track) |
| `ggml/src/ggml-sycl/cpu-dispatch.hpp` | 1, 2 | Sequential (same track) |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` lines 25160-25750 | 3 | None (Track A only) |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` lines 6553-6579 | 4 | None (Track B only, different section) |
| `ggml/src/ggml-sycl/unified-cache.hpp` | 4 | None (Track B only) |
| `ggml/src/ggml-sycl/unified-cache.cpp` | 4 | None (Track B only) |

### Model Recommendations

| Task | Recommended Model | Rationale |
|------|------------------|-----------|
| 1 | **haiku** | Mechanical: review existing code, build, commit |
| 2 | **haiku** | Mechanical: delete clearly-identified code blocks |
| 3 | **haiku** | Mechanical: delete mirror code + add simple skip guards |
| 4 | **sonnet** | Moderate: modify alloc intent + add host budget tracking |
| 5 | **haiku** | Mechanical: run commands, compare output |

---

## Pre-existing State

**IMPORTANT:** Tasks 1's code changes (beads `llama.cpp-6sn7` + `llama.cpp-9g8h`) are already implemented in the working tree but **not built or committed**. The modified files are:
- `ggml/src/ggml-sycl/cpu-dispatch.cpp` — host_task flag + all 10 CPU ops converted
- `ggml/src/ggml-sycl/cpu-dispatch.hpp` — `ggml_sycl_host_task_mode_set()` declaration added
- `ggml/src/ggml-sycl/ggml-sycl.cpp` — `ggml_sycl_host_task_mode_set()` activation added

---

## Task 1: Build and commit existing host_task work

**Track:** A
**Depends on:** None
**Beads:** `llama.cpp-6sn7`, `llama.cpp-9g8h`
**Model:** haiku
**File scope:**
- Already modified: `ggml/src/ggml-sycl/cpu-dispatch.cpp`
- Already modified: `ggml/src/ggml-sycl/cpu-dispatch.hpp`
- Already modified: `ggml/src/ggml-sycl/ggml-sycl.cpp`

**Description:**

The host_task infrastructure and all 10 CPU op conversions are already in the working tree. This task builds, verifies GPU-only correctness (no regression), and commits.

**What's already implemented:**

1. **Host task flag** in `cpu-dispatch.cpp` (after line 617):
   - `static bool g_host_task_mode = false;`
   - `host_task_mode_active()` — returns `g_host_task_mode`
   - `ggml_sycl_host_task_mode_set(bool)` — called from `graph_compute_impl`

2. **Declaration** in `cpu-dispatch.hpp` (line 65):
   - `void ggml_sycl_host_task_mode_set(bool active);`

3. **Activation** in `ggml-sycl.cpp` (line 25184):
   - `ggml_sycl_host_task_mode_set(cpu_offload_active && host_compute_enabled);`

4. **All 10 CPU ops** converted to dual-path (`host_task` on `gpu_q` OR `parallel_for` on `cpu_q`):
   - `cpu_rms_norm` (~line 1371): `sycl::sqrt` → `std::sqrt`
   - `cpu_binary_op` (~line 1480): row-wise with broadcasting
   - `cpu_silu` (~line 1626): `sycl::exp` → `std::exp`
   - `cpu_glu` (~line 1692): shared `glu_activate` lambda, `sycl::exp/erf` → `std::exp/erf`
   - `cpu_soft_max` (~line 1860): shared `softmax_row` lambda, `sycl::exp` → `std::exp`
   - `cpu_norm` (~line 1969): `sycl::sqrt` → `std::sqrt`
   - `cpu_scale` (~line 2078): trivial multiply
   - `cpu_cpy` (~line 2144): `memcpy` via `host_task`
   - `cpu_rope` (~line 2204): shared `rope_row` lambda, `sycl::cos/sin/fmax/fmin/log` → `std::` equivalents
   - `cpu_unary` (~line 2391): dispatch wrapper, no change needed

**Acceptance Criteria:**

- [ ] Build succeeds with `ninja -C build -j $(nproc)`
- [ ] GPU-only test produces correct output ("6, 7, 8, 9, 10...")
- [ ] Changes committed with descriptive message

**Implementation Guide:**

1. **Build:**

```bash
source /opt/intel/oneapi/setvars.sh --force
ninja -C build -j $(nproc)
```

Expected: Build succeeds (0 errors). Warnings about unused variables are OK.

2. **GPU-only correctness test:**

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```

Expected output contains: `6, 7, 8, 9, 10`

3. **Commit:**

```bash
git add ggml/src/ggml-sycl/cpu-dispatch.cpp ggml/src/ggml-sycl/cpu-dispatch.hpp ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "sycl: add host_task mode for HOST_COMPUTE CPU offload

When GGML_SYCL_HOST_COMPUTE=1 and CPU offload is active, all CPU ops
run as host_task callbacks on the GPU in-order queue instead of
parallel_for on the OpenCL CPU queue.  This eliminates cross-queue
sync overhead (~10x faster per op for TG-sized tensors).

Converted ops: rms_norm, binary_op (add/mul), silu, glu, soft_max,
norm, scale, cpy, rope, unary.  Each function has dual paths:
host_task on gpu_q (HOST_COMPUTE) or parallel_for on cpu_q (legacy).

The host_task path uses std:: math functions instead of sycl::
equivalents since host_task runs on the host CPU.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

**Notes for implementer:**
- Do NOT modify any code. Just build, test, and commit the existing working tree changes.
- The `host_compute_enabled` static variable at `ggml-sycl.cpp:25164` is already correct — it reads `GGML_SYCL_HOST_COMPUTE` env var.
- The mirror init block at `ggml-sycl.cpp:25168-25179` is still present — Task 3 removes it.

---

## Task 2: Remove compute buffer mirror from cpu-dispatch

**Track:** A
**Depends on:** Task 1
**Beads:** (part of `llama.cpp-l0di`)
**Model:** haiku
**File scope:**
- Modify: `ggml/src/ggml-sycl/cpu-dispatch.cpp` (lines 555-663, 1012-1016, 1054-1056)
- Modify: `ggml/src/ggml-sycl/cpu-dispatch.hpp` (lines 53-61)

**Description:**

Remove the compute buffer mirror infrastructure from `cpu-dispatch.cpp` and `cpu-dispatch.hpp`. The mirror was a rejected approach (HOST_COMPUTE makes compute buffers host-pinned from the start — no separate mirror needed). This is pure deletion of dead code.

**Acceptance Criteria:**

- [ ] All mirror-related code removed from cpu-dispatch.cpp and cpu-dispatch.hpp
- [ ] No references to `g_compute_mirror`, `is_in_compute_mirror`, `compute_mirror_host_ptr`, `ggml_sycl_compute_mirror_*` remain
- [ ] Committed

**Implementation Guide:**

1. **Delete mirror struct + helpers from `cpu-dispatch.cpp`**

Remove lines 555-663 entirely (from the comment `// Compute buffer mirror:` through the `sync_to_device` function and the blank line after it). This removes:
- `g_compute_mirror` struct
- `is_in_compute_mirror()` helper
- `compute_mirror_host_ptr()` offset mapping
- `ggml_sycl_compute_mirror_init()`
- `ggml_sycl_compute_mirror_teardown()`
- `ggml_sycl_compute_mirror_active()`
- `ggml_sycl_compute_mirror_sync_to_host()`
- `ggml_sycl_compute_mirror_sync_to_device()`

The host_task infrastructure (lines 619-639 before deletion, will shift up) must be KEPT.

2. **Delete mirror check from `flush_output()` in `cpu-dispatch.cpp`**

In the `flush_output` function (~line 999), find and remove these lines:

```cpp
    // Compute buffer mirror: skip per-op H2D flush.  CPU wrote to the host
    // mirror, and boundary sync in graph_compute_impl copies mirror→VRAM.
    if (g_compute_mirror.active && is_in_compute_mirror(t->data)) {
        return;
    }
```

3. **Delete mirror check from `get_host_output_ptr()` in `cpu-dispatch.cpp`**

In the `get_host_output_ptr` function (~line 1049), find and remove these lines:

```cpp
    // Compute buffer mirror: write directly to mirror.  Boundary sync handles H2D.
    if (g_compute_mirror.active && is_in_compute_mirror(t->data)) {
        return compute_mirror_host_ptr(t->data);
    }
```

4. **Delete mirror declarations from `cpu-dispatch.hpp`**

Remove lines 53-61:

```cpp
// Compute buffer mirror: host-pinned copy of the VRAM compute buffer.
// Enables zero-staging CPU offload — CPU ops read/write the mirror directly,
// with boundary memcpys at GPU↔CPU transitions instead of per-op staging.
void   ggml_sycl_compute_mirror_init(void * vram_base, size_t size, sycl::queue * q);
void   ggml_sycl_compute_mirror_teardown(sycl::queue * q);
bool   ggml_sycl_compute_mirror_active();
void   ggml_sycl_compute_mirror_sync_to_host(const ggml_tensor * t, sycl::queue * q);
void   ggml_sycl_compute_mirror_sync_to_device(const ggml_tensor * t, sycl::queue * q);
```

5. **Verify no remaining references:**

```bash
grep -rn 'compute_mirror\|g_compute_mirror\|is_in_compute_mirror\|compute_mirror_host_ptr' \
  ggml/src/ggml-sycl/cpu-dispatch.cpp ggml/src/ggml-sycl/cpu-dispatch.hpp
```

Expected: No matches.

6. **Commit:**

```bash
git add ggml/src/ggml-sycl/cpu-dispatch.cpp ggml/src/ggml-sycl/cpu-dispatch.hpp
git commit -m "sycl: remove compute buffer mirror from cpu-dispatch

The mirror was a rejected approach that created a separate host-pinned
copy of the VRAM compute buffer.  With HOST_COMPUTE, compute buffers
ARE host-pinned from the start — no mirror needed.  Removes dead code
to reduce confusion and eliminate unnecessary branches in hot paths
(flush_output, get_host_output_ptr).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

**Notes for implementer:**
- This is pure deletion — do NOT add any new code.
- The `ggml-sycl.cpp` mirror code (init block + boundary sync calls) is removed in Task 3, not here.
- After deletion, verify `cpu-dispatch.cpp` still has the host_task infrastructure (the `g_host_task_mode` section that was just below the mirror code).

---

## Task 3: Remove mirror from ggml-sycl.cpp + simplify boundary sync

**Track:** A
**Depends on:** Task 2
**Beads:** `llama.cpp-l0di`
**Model:** haiku
**File scope:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` (lines 25161-25179, 25601-25612, 25681-25694, 25732-25744)

**Description:**

Remove the compute buffer mirror initialization and boundary sync calls from `ggml-sycl.cpp`. Add HOST_COMPUTE skip guards at boundary sync blocks so that when host_task mode is active, the in-order queue handles all serialization (no boundary sync needed).

**Acceptance Criteria:**

- [ ] Mirror init block removed (lines 25161-25180)
- [ ] All `ggml_sycl_compute_mirror_*` calls removed from boundary sync
- [ ] HOST_COMPUTE skip guard added to boundary sync blocks
- [ ] Committed

**Implementation Guide:**

1. **Remove mirror init block from `ggml-sycl.cpp`**

Find and remove lines 25161-25180 (the block starting with `// Compute buffer mirror: opt-in via`):

```cpp
    // Compute buffer mirror: opt-in via GGML_SYCL_HOST_COMPUTE=1.
    // The per-op staging path is the default (proven correct).  The mirror has
    // a boundary-sync bug with recycled compute buffer offsets.
    static const bool host_compute_enabled = [] {
        const char * env = getenv("GGML_SYCL_HOST_COMPUTE");
        return env && (std::string(env) == "1" || std::string(env) == "true");
    }();
    if (cpu_offload_active && host_compute_enabled && !ggml_sycl_compute_mirror_active()) {
        for (int i = 0; i < cgraph->n_nodes; i++) {
            ggml_tensor * node = cgraph->nodes[i];
            if (node && node->buffer && node->buffer->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device)) {
                void * base = ggml_backend_buffer_get_base(node->buffer);
                size_t bsz  = ggml_backend_buffer_get_size(node->buffer);
                if (base && bsz > 0) {
                    ggml_sycl_compute_mirror_init(base, bsz, sycl_ctx->stream());
                    break;
                }
            }
        }
    }
```

**IMPORTANT:** Do NOT remove the `host_compute_enabled` static variable — move it above the mirror block, it's still needed by line 25184 (`ggml_sycl_host_task_mode_set`). The result should look like:

```cpp
    const bool cpu_offload_active = ggml_sycl_cpu_offload_enabled() && ggml_sycl_info().has_cpu_device;
    static const bool host_compute_enabled = [] {
        const char * env = getenv("GGML_SYCL_HOST_COMPUTE");
        return env && (std::string(env) == "1" || std::string(env) == "true");
    }();
    // Enable host_task mode when HOST_COMPUTE is active.
    // Compute buffers are host-pinned USM — CPU ops access t->data directly.
    // host_task on gpu_q eliminates cross-queue sync overhead (~10x faster per op).
    ggml_sycl_host_task_mode_set(cpu_offload_active && host_compute_enabled);
```

2. **Remove mirror sync from GPU island block (~line 25601-25612)**

Find and remove:

```cpp
                // Compute buffer mirror: island inputs may have been written to
                // the host mirror by preceding CPU ops.  Sync mirror→VRAM so
                // GPU island kernels read correct data.
                if (ggml_sycl_compute_mirror_active()) {
                    for (int s = 0; s < GGML_MAX_SRC && node->src[s]; s++) {
                        if (ggml_sycl_tensor_is_weight(node->src[s])) {
                            continue;
                        }
                        ggml_sycl_compute_mirror_sync_to_device(node->src[s], sycl_ctx->stream());
                    }
                    sycl_ctx->stream()->wait();
                }
```

3. **Remove mirror sync from GPU→CPU transition (~line 25681-25694)**

Find and remove:

```cpp
                    // Compute buffer mirror: sync VRAM→mirror for the CPU node's
                    // non-weight input tensors.  After this, get_host_ptr() returns
                    // the mirror pointer without any per-op D2H copy.
                    if (ggml_sycl_compute_mirror_active()) {
                        for (int s = 0; s < GGML_MAX_SRC && node->src[s]; s++) {
                            if (ggml_sycl_tensor_is_weight(node->src[s])) {
                                continue;
                            }
                            ggml_sycl_compute_mirror_sync_to_host(node->src[s], sycl_ctx->stream());
                        }
                        sycl_ctx->stream()->wait();
                        GGML_SYCL_DEBUG("[COMPUTE-MIRROR] GPU→CPU sync at node %d (%s)\n", i,
                                        node->name ? node->name : "(null)");
                    }
```

4. **Remove mirror sync from CPU→GPU transition (~line 25732-25744)**

Find and remove:

```cpp
                    // Compute buffer mirror: sync mirror→VRAM for the GPU node's
                    // non-weight input tensors so GPU kernels read up-to-date data.
                    if (ggml_sycl_compute_mirror_active()) {
                        for (int s = 0; s < GGML_MAX_SRC && node->src[s]; s++) {
                            if (ggml_sycl_tensor_is_weight(node->src[s])) {
                                continue;
                            }
                            ggml_sycl_compute_mirror_sync_to_device(node->src[s], sycl_ctx->stream());
                        }
                        sycl_ctx->stream()->wait();
                        GGML_SYCL_DEBUG("[COMPUTE-MIRROR] CPU→GPU sync at node %d (%s)\n", i,
                                        node->name ? node->name : "(null)");
                    }
```

5. **Add HOST_COMPUTE skip guard at GPU→CPU boundary sync**

The boundary sync block (starting at `if (node_on_cpu != prev_on_cpu)`) currently does per-tensor event waits or full queue waits at GPU→CPU transitions. When HOST_COMPUTE is active, the in-order gpu_q handles all serialization — no boundary sync is needed because host_tasks and GPU kernels are on the same queue.

After the `if (node_on_cpu)` opening brace (the GPU→CPU branch), add this guard:

```cpp
                if (node_on_cpu) {
                    // HOST_COMPUTE: in-order gpu_q serializes GPU kernels and
                    // host_tasks. No boundary sync needed — skip event waits
                    // and staging drain.
                    if (host_compute_enabled && cpu_offload_active) {
                        GGML_SYCL_DEBUG("[HOST_COMPUTE] GPU→CPU skip boundary sync at node %d (%s)\n",
                                        i, node->name ? node->name : "(null)");
                    } else {
                    // --- existing GPU→CPU boundary sync code (indented one more level) ---
```

And close the else brace after the existing GPU→CPU code ends (before the `} else {` that starts the CPU→GPU branch).

Similarly, for the CPU→GPU transition (the `else` branch), add:

```cpp
                } else {
                    if (host_compute_enabled && cpu_offload_active) {
                        GGML_SYCL_DEBUG("[HOST_COMPUTE] CPU→GPU skip boundary sync at node %d (%s)\n",
                                        i, node->name ? node->name : "(null)");
                    } else {
                    // --- existing CPU→GPU boundary sync code (indented one more level) ---
```

And for the GPU island block, add:

```cpp
            if (node_is_island) {
                if (host_compute_enabled && cpu_offload_active) {
                    GGML_SYCL_DEBUG("[HOST_COMPUTE] GPU island skip boundary sync at node %d (%s)\n",
                                    i, node->name ? node->name : "(null)");
                } else {
                    // existing island staging drain code
                    ggml_sycl_cpu_staging_drain();
                }
```

6. **Verify no remaining mirror references in ggml-sycl.cpp:**

```bash
grep -n 'compute_mirror' ggml/src/ggml-sycl/ggml-sycl.cpp
```

Expected: No matches.

7. **Commit:**

```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "sycl: remove mirror from ggml-sycl.cpp + add HOST_COMPUTE boundary skip

Remove compute buffer mirror init block and all mirror sync calls
from graph_compute_impl.  Add HOST_COMPUTE skip guards at GPU↔CPU
boundary sync blocks — when host_task mode is active, the in-order
gpu_q serializes everything and no boundary sync is needed.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

**Notes for implementer:**
- The `host_compute_enabled` static variable MUST be preserved (moved above the deleted mirror block).
- The skip guards use `host_compute_enabled && cpu_offload_active` — both variables already exist in scope.
- Do NOT modify any code outside the boundary sync section (lines ~25160-25750).
- The retained activation code inside the boundary sync is unrelated — leave it unchanged.

---

## Task 4: Add dual-budget tracking for HOST_COMPUTE compute buffers

**Track:** B (can run in parallel with Tasks 2 and 3)
**Depends on:** Task 1 (only for the `host_compute_enabled` variable, but Task 4 touches different files)
**Beads:** `llama.cpp-0h90`
**Model:** sonnet
**File scope:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` (lines 6553-6579 only — `alloc_buffer`)
- Modify: `ggml/src/ggml-sycl/unified-cache.hpp` (budget API)
- Modify: `ggml/src/ggml-sycl/unified-cache.cpp` (budget implementation)

**Description:**

When `GGML_SYCL_HOST_COMPUTE=1`, compute buffers are allocated as host-pinned USM. These allocations should be tracked with `runtime_category::HOST_COMPUTE` (not `COMPUTE`) and counted against a host memory budget. Currently, `alloc_buffer` tags `CpuOffloadCompute` buffers as `runtime_category::COMPUTE` — this task corrects that and adds host budget tracking.

**Acceptance Criteria:**

- [ ] `CpuOffloadCompute` buffers tagged as `HOST_COMPUTE` runtime category
- [ ] Host memory budget tracked in unified cache
- [ ] Budget logged at allocation time for debugging
- [ ] No changes to VRAM budget tracking (HOST_COMPUTE doesn't use VRAM)
- [ ] Committed

**Implementation Guide:**

1. **Change runtime_category for CpuOffloadCompute in `ggml-sycl.cpp` `alloc_buffer` (~line 6558)**

Find:

```cpp
    const ggml_sycl::runtime_category alloc_cat =
        is_compute_buft ? ggml_sycl::runtime_category::COMPUTE : ggml_sycl::runtime_category::KV_CACHE;
```

Replace with:

```cpp
    const bool is_cpu_offload_compute = buft_ctx->name.find("CpuOffloadCompute") != std::string::npos;
    const ggml_sycl::runtime_category alloc_cat =
        is_cpu_offload_compute ? ggml_sycl::runtime_category::HOST_COMPUTE
        : is_compute_buft     ? ggml_sycl::runtime_category::COMPUTE
                               : ggml_sycl::runtime_category::KV_CACHE;
```

2. **Add host budget tracking to unified-cache.hpp**

Find the existing budget tracking functions (around line 1269):

```cpp
void     unified_cache_add_runtime_bytes(int device, size_t bytes, runtime_category cat = runtime_category::OTHER);
```

After the existing runtime bytes functions, add:

```cpp
// Host memory budget tracking for HOST_COMPUTE allocations.
// Host-pinned USM doesn't consume VRAM but should be tracked to avoid
// exhausting system RAM.
void   unified_cache_add_runtime_host_bytes(int device, size_t bytes, runtime_category cat);
void   unified_cache_sub_runtime_host_bytes(int device, size_t bytes, runtime_category cat);
size_t unified_cache_get_runtime_host_bytes(int device);
```

3. **Implement host budget tracking in `unified-cache.cpp`**

Add a parallel tracking structure for host bytes. Find the existing `g_runtime_bytes` tracking (search for `g_runtime_reserved_baseline` or `runtime_bytes` in the file). Add alongside it:

```cpp
static std::atomic<size_t> g_runtime_host_bytes[GGML_SYCL_MAX_DEVICES] = {};

void unified_cache_add_runtime_host_bytes(int device, size_t bytes, runtime_category cat) {
    if (device < 0 || device >= GGML_SYCL_MAX_DEVICES) return;
    g_runtime_host_bytes[device].fetch_add(bytes, std::memory_order_relaxed);
    if (g_ggml_sycl_debug) {
        GGML_LOG_INFO("[UNIFIED-CACHE] device %d: +%zu host bytes (cat=%d, total_host=%zu)\n",
                      device, bytes, (int)cat, g_runtime_host_bytes[device].load());
    }
}

void unified_cache_sub_runtime_host_bytes(int device, size_t bytes, runtime_category cat) {
    if (device < 0 || device >= GGML_SYCL_MAX_DEVICES) return;
    size_t prev = g_runtime_host_bytes[device].fetch_sub(bytes, std::memory_order_relaxed);
    if (g_ggml_sycl_debug) {
        GGML_LOG_INFO("[UNIFIED-CACHE] device %d: -%zu host bytes (cat=%d, total_host=%zu)\n",
                      device, bytes, (int)cat, prev - bytes);
    }
}

size_t unified_cache_get_runtime_host_bytes(int device) {
    if (device < 0 || device >= GGML_SYCL_MAX_DEVICES) return 0;
    return g_runtime_host_bytes[device].load(std::memory_order_relaxed);
}
```

4. **Register HOST_COMPUTE allocations in `alloc_buffer`**

In `ggml-sycl.cpp` `alloc_buffer`, after the existing `unified_cache_add_runtime_bytes` call (search for it in the function), add a HOST_COMPUTE-specific host budget call. Find:

```cpp
        unified_cache_add_runtime_bytes(buft_ctx->device, size, alloc_cat);
```

After it, add:

```cpp
        if (alloc_cat == ggml_sycl::runtime_category::HOST_COMPUTE) {
            unified_cache_add_runtime_host_bytes(buft_ctx->device, size, alloc_cat);
        }
```

5. **Verify `cat_names[]` includes HOST_COMPUTE**

In `unified-cache.cpp`, find the `cat_names` array (search for `cat_names`). Verify it has an entry for HOST_COMPUTE at index 4. If the array looks like:

```cpp
static const char * cat_names[] = { "KV_CACHE", "COMPUTE", "STAGING", "GRAPH", "HOST_COMPUTE", "OTHER" };
```

Then it's correct. If HOST_COMPUTE is missing, add it at index 4 (must match the enum value).

6. **Commit:**

```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp ggml/src/ggml-sycl/unified-cache.hpp ggml/src/ggml-sycl/unified-cache.cpp
git commit -m "sycl: track HOST_COMPUTE allocations in host memory budget

CpuOffloadCompute buffers now tagged as runtime_category::HOST_COMPUTE
instead of COMPUTE.  Added host memory budget tracking to unified cache
so host-pinned USM allocations are monitored and reported.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

**Notes for implementer:**
- The `runtime_category::HOST_COMPUTE = 4` already exists in `unified-cache.hpp:1060`.
- The `cat_names[]` array in `unified-cache.cpp` MUST match the enum exactly or SEGFAULT. Verify index 4 = "HOST_COMPUTE".
- `unified_cache_add_runtime_bytes` at `alloc_buffer` may appear inside a conditional block — find it by searching the function body.
- This task only touches `ggml-sycl.cpp` lines 6553-6579 (alloc_buffer). Do NOT touch lines 25000+ (that's Task 3's territory).

---

## Task 5: Build, verify correctness, and benchmark

**Track:** Convergence
**Depends on:** Tasks 3 and 4
**Beads:** `llama.cpp-4tag`
**Model:** haiku
**File scope:** No file changes — run commands only

**Description:**

Build the complete implementation, verify correctness across all configurations, and benchmark performance.

**Acceptance Criteria:**

- [ ] Build succeeds
- [ ] GPU-only produces correct output (no regression)
- [ ] HOST_COMPUTE + CPU_OFFLOAD at 43% VRAM produces correct output
- [ ] HOST_COMPUTE + CPU_OFFLOAD at 30% VRAM produces correct output
- [ ] CPU_OFFLOAD without HOST_COMPUTE still works (staging path backward compat)
- [ ] GPU-only benchmark: PP512 >= 1200, TG128 >= 68
- [ ] HOST_COMPUTE benchmark shows improvement over staging baseline

**Implementation Guide:**

Always source oneAPI first:

```bash
source /opt/intel/oneapi/setvars.sh --force
```

1. **Build:**

```bash
ninja -C build -j $(nproc)
```

Expected: 0 errors. If build fails, check for missing `#include "cpu-dispatch.hpp"` in ggml-sycl.cpp.

2. **GPU-only correctness (no regression — HOST_COMPUTE should be inactive):**

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```

Expected output contains: `6, 7, 8, 9, 10`

3. **HOST_COMPUTE + CPU_OFFLOAD at 43% VRAM (mixed GPU/CPU layers):**

```bash
GGML_SYCL_HOST_COMPUTE=1 GGML_SYCL_CPU_OFFLOAD=1 GGML_SYCL_VRAM_BUDGET_PCT=43 \
  ONEAPI_DEVICE_SELECTOR="level_zero:0;opencl:cpu" ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```

Expected: Same output as GPU-only.

4. **HOST_COMPUTE + CPU_OFFLOAD at 30% VRAM (all CPU layers):**

```bash
GGML_SYCL_HOST_COMPUTE=1 GGML_SYCL_CPU_OFFLOAD=1 GGML_SYCL_VRAM_BUDGET_PCT=30 \
  ONEAPI_DEVICE_SELECTOR="level_zero:0;opencl:cpu" ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```

Expected: Same output.

5. **CPU_OFFLOAD without HOST_COMPUTE (backward compat — staging path):**

```bash
GGML_SYCL_CPU_OFFLOAD=1 GGML_SYCL_VRAM_BUDGET_PCT=30 \
  ONEAPI_DEVICE_SELECTOR="level_zero:0;opencl:cpu" ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```

Expected: Same output (staging path unchanged).

6. **GPU-only performance (wait 30-60 seconds before starting):**

```bash
NEW_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v pti | tr '\n' ':' | sed 's/:$//')
LD_LIBRARY_PATH="build/bin:$NEW_PATH" \
  ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128
```

Expected: PP512 >= 1200 tok/s, TG128 >= 68 tok/s (no regression).

**IMPORTANT:** Wait 30-60 seconds between benchmark runs. Arc B580 thermal throttles severely with back-to-back runs (70 → 2.4 tok/s).

7. **HOST_COMPUTE + CPU_OFFLOAD TG benchmark (wait 60s after step 6):**

```bash
LD_LIBRARY_PATH="build/bin:$NEW_PATH" \
  GGML_SYCL_HOST_COMPUTE=1 GGML_SYCL_CPU_OFFLOAD=1 GGML_SYCL_VRAM_BUDGET_PCT=30 \
  ONEAPI_DEVICE_SELECTOR="level_zero:0;opencl:cpu" ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 16 -n 128
```

Baseline (staging path without HOST_COMPUTE): ~1.5 tok/s at 30% VRAM.
Target: > 2 tok/s (any improvement from eliminating staging overhead).

8. **Report results and commit verification:**

If all correctness tests pass and performance is acceptable, close beads tasks:

```bash
bd close llama.cpp-6sn7 --reason="host_task infrastructure committed"
bd close llama.cpp-9g8h --reason="All 10 CPU ops converted to host_task"
bd close llama.cpp-l0di --reason="Mirror removed, boundary sync simplified"
bd close llama.cpp-0h90 --reason="HOST_COMPUTE budget tracking added"
bd close llama.cpp-4tag --reason="Verified: GPU-only PP=X TG=Y, HOST_COMPUTE 30%=Z tok/s"
bd sync
```

**Notes for implementer:**
- PTI library from `setvars.sh` can crash llama-bench. The `LD_LIBRARY_PATH` filter removes it.
- `ONEAPI_DEVICE_SELECTOR="level_zero:0;opencl:cpu"` is required for CPU offload — without `opencl:cpu`, there's no CPU SYCL device.
- If HOST_COMPUTE correctness fails (garbage output), the bug is likely in boundary sync (Task 3) or buffer type detection. Check `GGML_SYCL_DEBUG=1` output.
- If GPU-only regresses, check that the mirror removal didn't accidentally delete the host_task infrastructure.
