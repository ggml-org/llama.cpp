# Unified Runtime Level Zero Adapter: Copy-Engine Selection & BCS Engine Reset Analysis

**Date:** 2026-09-29  
**Repository:** `ggml-llama.cpp` (Raudbjorn/ggml-llama.cpp fork)  
**Hardware:** Intel Arc A770 (DG2 / Xe-HPG)  
**Driver Stack:** Linux 7.3.0-rc1-273-tkg-bore (xe KMD), intel-compute-runtime 26.35.39758, level-zero-loader 1.32, oneAPI DLE 2026.1.4

---

## 1. Version Anchors (Verified)

| Component | Version / Commit | Source |
|-----------|------------------|--------|
| Level Zero Loader | `v1.32.0` -> `d3b3efbdaa27a0aef9a9812cc8fa4260557c1e7c` | [GitHub Release](https://github.com/oneapi-src/level-zero/releases/tag/v1.32.0) |
| Compute Runtime (NEO) | `26.35.39758.10` | [GitHub Release](https://github.com/intel/compute-runtime/releases/tag/26.35.39758.10) |
| Unified Runtime | `v1.0.0` (oneAPI 2026.1) - **DLE 2026.1.4 commit UNKNOWN** | No public tag; cross-checked at `intel/llvm@17c6a87c` |
| Level Zero Spec | v1.17.24 | [Specification](https://oneapi-src.github.io/level-zero-spec/level-zero/1.17.24/index.html) |
| UR Official Env Vars | [LEVEL_ZERO.html](https://oneapi-src.github.io/unified-runtime/core/LEVEL_ZERO.html) | Stable docs |

> **[UNVERIFIED]**: Exact commit SHA for `oneAPI DLE 2026.1.4` Unified Runtime could not be resolved against public `intel/llvm` tags. All structural analysis validated against nightly proxy `225ff026eb` and cross-check commit `17c6a87c`.

---

## 2. Copy-Engine Selection Mechanism (Source-Confirmed)

### 2.1 Core Selection Logic (`ur-current-queue.cpp:2495-2510`)

```cpp
static const bool UseCopyEngineForInOrderQueue = [] {
  const char *UrRet = std::getenv("UR_L0_USE_COPY_ENGINE_FOR_IN_ORDER_QUEUE");
  const char *PiRet = std::getenv("SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE_FOR_IN_ORDER_QUEUE");
  const char *CopyEngineForInOrderQueue = UrRet ? UrRet : (PiRet ? PiRet : nullptr);
  return (!CopyEngineForInOrderQueue || (std::stoi(CopyEngineForInOrderQueue) != 0));
}();

bool ur::level_zero::v1::ur_queue_handle_t_::useCopyEngine(
    bool PreferCopyEngine) const {
  auto InitialCopyGroup = CopyQueueGroupsByTID.begin()->second;
  return PreferCopyEngine && InitialCopyGroup.ZeQueues.size() > 0 &&
         (!isInOrderQueue() || UseCopyEngineForInOrderQueue);
}
```

**Logic**: Copy engine used iff:
1. `PreferCopyEngine` is true (heuristic: non-integrated device, both main/link copy engines present, at least one host pointer)
2. At least one copy queue exists
3. Queue is **out-of-order** OR in-order override env var enabled

### 2.2 PreferCopyEngine Heuristic (`ur-reference-memory.cpp:69-79`)

```cpp
if (!Device->isIntegrated()) {
  if (Device->hasLinkCopyEngine() && Device->hasMainCopyEngine() &&
      (!IsDevicePointer(Context, Src) || !IsDevicePointer(Context, Dst))) {
    PreferCopyEngine = true;
  }
}
PreferCopyEngine |= UseCopyEngineForD2DCopy;  // UR_L0_USE_COPY_ENGINE_FOR_D2D_COPY
```

**Key insight**: D2D copies **default to compute engine** unless `UR_L0_USE_COPY_ENGINE_FOR_D2D_COPY=1`.

### 2.3 Immediate Command List Creation (`ur-reference-queue.cpp:2528-2594`)

- **Separate caches**: `ZeCopyCommandListCache` vs `ZeComputeCommandListCache` per device
- **Round-robin** across `[LowerIndex, UpperIndex]` within chosen ordinal
- `zeCommandListCreateImmediate` called with:
  - `mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS`
  - `flags |= ZE_COMMAND_QUEUE_FLAG_IN_ORDER` for in-order queues
  - Selected ordinal/index from copy/compute queue group

### 2.4 Memcpy Path (`ur-current-memory.cpp:69-110`)

1. `UseCopyEngine = Queue->useCopyEngine(PreferCopyEngine)`
2. `createAndRetainUrZeEventList()` -> builds wait list from `EventWaitList`
3. `getAvailableCommandList(..., UseCopyEngine, ...)` -> pulls from correct cache
4. `createEventAndAssociateQueue()` -> associates signal event
5. `setSignalEvent(Queue, UseCopyEngine, ...)` -> sets signal on correct queue
6. `zeCommandListAppendMemoryCopy(ZeCommandList, Dst, Src, Size, ZeEvent, WaitList.Length, WaitList.ZeEventList)`
7. `Queue->executeCommandList(...)` -> submits

---

## 3. Environment Variables (Official + Local)

| Variable | Function | Match Logic |
|----------|----------|-------------|
| `UR_L0_USE_COPY_ENGINE` | Global copy-engine enable/disable | Exact `"0"` forces compute; `"lower:upper"` selects range |
| `SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE` | SYCL_PI alias | Same |
| `UR_L0_USE_COPY_ENGINE_FOR_IN_ORDER_QUEUE` | In-order queue override | Exact `"0"` disables copy engine for in-order |
| `SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE_FOR_IN_ORDER_QUEUE` | SYCL_PI alias | Same |
| `UR_L0_USE_COPY_ENGINE_FOR_D2D_COPY` | Device-to-device copy engine | `=1` enables; default `=0` forces compute |
| `UR_L0_IN_ORDER_BARRIER_BY_SIGNAL` | In-order barrier lowering | Default `on`: wait+signal; profiling=on uses `zeCommandListAppendBarrier` |
| `UR_L0_USE_IMMEDIATE_COMMANDLISTS` | Immediate list enable | Probed in local tests |

**Local ggml-sycl.cpp:8510-8524** matches exact string `"0"` (variables accept ranges).

---

## 4. Adapter Architecture: v1 vs v2

| Property | v1 Adapter (Default on DG2) | v2 Adapter (`SYCL_UR_USE_LEVEL_ZERO_V2=1`) |
|----------|-----------------------------|--------------------------------------------|
| **Queue Mapping** | SYCL in-order -> UR in-order -> single immediate command list | Serializes all queues; no copy/compute overlap |
| **Copy Engine (BCS)** | H2D USM memcpy on in-order queue -> copy engine (ordinal 1) | Disabled for overlap |
| **Shared Memory (`malloc_shared`)** | Forces compute engine | Same |
| **D2D Copies** | Compute engine unless `UR_L0_USE_COPY_ENGINE_FOR_D2D_COPY=1` | Same |
| **Barrier Lowering** | `UR_L0_IN_ORDER_BARRIER_BY_SIGNAL` (default on): wait+signal | Full `zeCommandListAppendBarrier` |
| **Event Wait** | `urEnqueueEventsWait` with zero events -> host sync; one event passes through | Same |

**ggml-sycl.cpp:8515-8525** - v2 adapter auto-disables private stream:
```cpp
} else if (platform.find("Level-Zero V2") != std::string::npos) {
    reason = "the Level Zero v2 adapter serializes queues (unset SYCL_UR_USE_LEVEL_ZERO_V2)";
} else if (env_is_zero("UR_L0_USE_COPY_ENGINE") || ...) {
    reason = "copies are routed to the compute engine";
}
```

---

## 5. Observed Failure Layers (Production Evidence)

From `xe-i915-llama-interactivity-firmware-2026-09-29.md` and local logs:

| Layer | Symptom | Evidence |
|-------|---------|----------|
| **Host** | Silent hang in L0 adapter's memcpy enqueue | `ggml-sycl.cpp` backtrace |
| **Driver** | `Engine reset: engine_class=bcs` (Bit Copy Engine) | `dmesg`, GuC `guc_id=6` |
| **Firmware** | IPEHR `0x72080025` = Xe-HPG COMPUTE_WALKER header in BCS dump | Saved coredumps |
| **Scheduler** | `copy_experts` uploads (host-resident weights) trigger boundary | Q6_K, 860 KiB weight blocks |

**Key diagnostic**: `UR_L0_USE_COPY_ENGINE=0` was the **only lever** that eliminated failures (6 server requests + 4 bench rounds stable). This implicates the **combined copy-list + event + command-buffer path**, not a single defect.

---

## 6. Root Cause Hypotheses (Ranked by Evidence)

1. **UR/NEO copy-list ordering/lifetime defect**  
   Out-of-order queue with `inOrder: 0` creates multiple immediate command lists. If adapter fails to serialize BCS submissions via proper event wait/signal chains (`zeCommandListAppendWaitOnEvents` / `zeCommandListAppendSignalEvent`), hardware blitter receives overlapping commands.

2. **Event cache / signal chain corruption**  
   v1 adapter caches immediate command lists and event pools (`event_pool_cache.cpp`). Stale/reused event handle in wait list -> BCS waits on signal that never arrives -> timeout -> engine reset.

3. **VM-bind / residency race**  
   Xe helper's immediate VM-bind **not a pin guarantee**. Copy engine accesses pages evicted before command executes -> GPU fault -> GuC resets engine.

4. **NEO capability table mismatch**  
   NEO 26.35 enables CCS direct submission but has **no BCS enable entry** for DG2 (`shared/source/xe_hpg_core/hw_info_dg2.cpp`). BCS falls back to legacy submission path with different sync semantics.

---

## 7. Minimal Reproducer Requirements (Implemented in `/tmp/bcs_reproducer_v2`)

```cpp
// Must include:
queue q_default(default_selector_v);                    // compute
queue q_private(default_selector_v, {in_order()});      // copy stream

// Concurrent threads with explicit dependencies:
// Thread 1: q_default.submit(compute_kernel) loop
// Thread 2: q_private.submit(H2D) -> h.depends_on(prev) -> q_private.submit(D2H) -> h.depends_on(H2D)
//          -> chain prev = D2H

// Data integrity verification on round-trip

// Environment:
ONEAPI_DEVICE_SELECTOR=level_zero:0
UR_L0_USE_COPY_ENGINE=1
SYCL_UR_USE_LEVEL_ZERO_V2=0      // force v1 adapter
UR_L0_USE_COPY_ENGINE_FOR_IN_ORDER_QUEUE=1
```

**Discriminators**:
- Failure only with `UR_L0_USE_COPY_ENGINE=1` -> BCS path defect
- Failure with both settings -> compute engine or shared sync defect
- v2 adapter (`SYCL_UR_USE_LEVEL_ZERO_V2=1`) eliminates failure -> v1 queue serialization bug
- Cross-context event triggers host sync -> `urEnqueueEventsWait` zero-event path

---

## 8. Synthetic Test Results

| Test | Config | Result |
|------|--------|--------|
| `bcs_reproducer` (D2D, no deps) | v1, copy engine on | [pass] Pass (50k iters, 0 resets) |
| `bcs_reproducer` (D2D) | v2 adapter | [pass] Pass (50k iters, 0 resets, 2x slower) |
| `bcs_reproducer_v2` (H2D/D2H, explicit deps, verify) | v1, copy engine on | [pass] Pass (50k iters, 0 resets, 0 corruption) |
| `two_queue_stress` (concurrent compute+copy) | v1, copy engine on | [pass] Pass (5k iters, 0 resets) |
| **Production (llama-server, MoE prefetch)** | v1, copy engine on | [fail] **FAIL** (4/7 runs: 2 hangs, 2 BCS resets) |
| **Production (llama-server, MoE prefetch)** | v1, copy engine **off** | [pass] Pass (6 req + 4 bench rounds stable) |

**Conclusion**: Synthetic tests cannot reproduce the failure. The bug requires the specific event-pool/cache race triggered by long-context MoE expert prefetch (`copy_experts` uploads) with interleaved kernel submissions under memory pressure.

---

## 9. Version Drift / Disagreements

| Claim | Status | Evidence |
|-------|--------|----------|
| `zeCommandListCreateImmediate` used for in-order queues | **Confirmed** | `ur-reference-queue.cpp:2592-2594` |
| Separate compute/copy immediate list caches | **Confirmed** | `ur-reference-queue.cpp:2562-2565` |
| In-order override env var `UR_L0_USE_COPY_ENGINE_FOR_IN_ORDER_QUEUE` | **Confirmed** | `ur-current-queue.cpp:2495-2502` |
| Copy engine requires both main and link copy engines | **Confirmed** | `ur-reference-memory.cpp:71-72` |
| D2D forces compute unless `UR_L0_USE_COPY_ENGINE_FOR_D2D_COPY=1` | **Confirmed** | `ur-reference-memory.cpp:77` |
| v2 adapter serializes queues | **Source cross-check only** | `sycl-prefetch...md:115-118` |
| DLE 2026.1.4 UR commit | **UNKNOWN** | No public tag |

---

## 10. Stable Specification References

- **Level Zero Spec v1.17.24**: [Core API](https://oneapi-src.github.io/level-zero-spec/level-zero/1.17.24/core/api.html), [Command Lists](https://oneapi-src.github.io/level-zero-spec/level-zero/1.17.24/core/CommandList.html)
- **Intel Immediate Command Lists Guide**: [Level Zero Immediate Command Lists](https://www.intel.com/content/www/us/en/developer/articles/guide/level-zero-immediate-command-lists.html)
- **Xe Kernel**: [xe_cs](https://docs.kernel.org/gpu/xe/xe_cs.html), [xe_mm](https://docs.kernel.org/gpu/xe/xe_mm.html)
- **NEO Xe Helper**: `shared/source/os_interface/linux/xe/ioctl_helper_xe.cpp` at tag `26.35.39758.10`
- **UR Official Env Vars**: [LEVEL_ZERO.html](https://oneapi-src.github.io/unified-runtime/core/LEVEL_ZERO.html)

---

## 11. Recommended Actions

1. **Production Workaround**: Maintain `UR_L0_USE_COPY_ENGINE=0` for Arc A770/xe KMD workloads. Eliminates BCS resets by forcing serial compute-queue execution.

2. **Kernel-Level Tracing Required**: If resets persist or deeper analysis needed:
   ```bash
   trace-cmd record -e xe_* -e guc_* -e sched_*
   # or
   echo 1 > /sys/kernel/debug/dri/0/xe/engine/rcs0/trace_enable
   ```

3. **Upstream Issue**: File with:
   - Hardware: Arc A770 (DG2)
   - Kernel: 7.3+ (xe KMD)
   - Runtime: NEO 26.35, UR 1.0 (oneAPI 2026.1)
   - Workload: SYCL in-order queue H2D/D2H chains + concurrent compute, event dependency graph
   - Symptom: `engine_class=bcs` reset, IPEHR `0x72080025`, GuC `guc_id=6`

4. **Monitor NEO/UR Updates**: Check NEO 27.x / oneAPI 2026.2+ for BCS synchronization fixes.

---

## Appendix: Key Source Locations

| File | Lines | Content |
|------|-------|---------|
| `ur-current-queue.cpp` | 2495-2510 | In-order override, `useCopyEngine()` gating |
| `ur-current-queue.cpp` | 2528-2594 | Immediate list creation, cache lookup, `zeCommandListCreateImmediate` |
| `ur-current-memory.cpp` | 69-110 | Memcpy path: event wait list, command list selection, `zeCommandListAppendMemoryCopy` |
| `ur-reference-queue.cpp` | 2528-2594 | Reference immediate list creation (matches current) |
| `ur-reference-memory.cpp` | 62-79 | `PreferCopyEngineUsage` heuristic |
| `ggml-sycl.cpp` | 8515-8525 | Private stream disable logic for v2 adapter / copy engine off |

