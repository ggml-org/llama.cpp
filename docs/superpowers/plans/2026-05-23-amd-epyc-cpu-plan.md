# AMD EPYC CPU Optimization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add NUMA interleave allocation, CCD-aware thread affinity, and AVX-512 VNNI vec_dot for Q8_0/Q4_K/BF16 to the ggml-cpu backend, targeting 20-30 t/s on a 600GB Kimi K2.6 UD-Q4_K_XL model (baseline: 8-11 t/s).

**Architecture:** Three independent changes: (1) NUMA interleave via `set_mempolicy(MPOL_INTERLEAVE)`, (2) CCD topology probe from sysfs + thread pinning, (3) AVX-512 VNNI vec_dot implementations using `_mm512_dpbusd_epi32`. All guarded by `#ifdef __linux__` and compile-time `#ifdef __AVX512VNNI__`. Follows existing pattern in `quants.c` where `mul_sum_i8_pairs_float` already uses `__AVX512VNNI__` inline guards.

**Tech Stack:** Linux sysfs, `set_mempolicy`/`sched_setaffinity` syscalls, AVX-512 VNNI/BF16/BW/VL intrinsics, AOCC pragmas, CMake

---

## File Structure

| File | Responsibility |
|------|---------------|
| `ggml/include/ggml-cpu.h` | Public enum — add `GGML_NUMA_STRATEGY_INTERLEAVE`, `GGML_NUMA_STRATEGY_CCD` |
| `ggml/src/ggml-cpu/ggml-cpu.c` | NUMA interleave init/reset, CCD topology probe, CCD affinity in `set_numa_thread_affinity()` |
| `ggml/src/ggml-cpu/arch/x86/quants.c` | AVX-512 VNNI paths for Q8_0 and Q4_K (inline `#if defined(__AVX512VNNI__)` guards) |
| `ggml/src/ggml-cpu/vec.cpp` | AVX-512 BF16 path (inline `#if defined(__AVX512BF16__)` guard) |

**Dispatch via inline `#ifdef`:** Follows the existing pattern in `quants.c` where `mul_sum_i8_pairs_float` (line 106) and `mul_sum_us8_pairs_float` (line 122) use `#if defined(__AVX512VNNI__)` to select the optimal path. Each CMake variant (AVX2, AVX512-VNNI, etc.) compiles `quants.c` with its own flags, so the VNNI variant automatically gets the VNNI path. No new file, no weak symbols, no CMake changes needed.

---

### Task 1: NUMA Interleave Allocation

**Files:**
- Modify: `ggml/include/ggml-cpu.h:28-35` — add `GGML_NUMA_STRATEGY_INTERLEAVE = 5` to enum
- Modify: `ggml/src/ggml-cpu/ggml-cpu.c:679-765` — add interleave case to `ggml_numa_init()`

**Context:** The current `ggml_numa_init()` (line 679) probes `/sys/devices/system/node/node*` to enumerate NUMA nodes. We add a `GGML_NUMA_STRATEGY_INTERLEAVE` case that calls `set_mempolicy(MPOL_INTERLEAVE, nodemask)` to interleave all future malloc across all nodes. The existing `ggml_numa_init` already collects the node count, so we build a nodemask from that.

The NUMA strategy enum at `ggml/include/ggml-cpu.h:28-35` currently ends with `GGML_NUMA_STRATEGY_MIRROR = 4` followed by `GGML_NUMA_STRATEGY_COUNT`. Adding a new strategy before `COUNT` automatically updates the count.

- [ ] **Step 1: Add enum values to ggml-cpu.h**

In `ggml/include/ggml-cpu.h`, between line 33 (`GGML_NUMA_STRATEGY_MIRROR = 4,`) and line 34 (`GGML_NUMA_STRATEGY_COUNT`), insert:

```c
        GGML_NUMA_STRATEGY_INTERLEAVE = 5,
        GGML_NUMA_STRATEGY_CCD         = 6,
```

Resulting enum:
```c
    enum ggml_numa_strategy {
        GGML_NUMA_STRATEGY_DISABLED   = 0,
        GGML_NUMA_STRATEGY_DISTRIBUTE = 1,
        GGML_NUMA_STRATEGY_ISOLATE    = 2,
        GGML_NUMA_STRATEGY_NUMACTL    = 3,
        GGML_NUMA_STRATEGY_MIRROR     = 4,
        GGML_NUMA_STRATEGY_INTERLEAVE = 5,
        GGML_NUMA_STRATEGY_CCD        = 6,
        GGML_NUMA_STRATEGY_COUNT
    };
```

- [ ] **Step 2: Add NUMA interleave to ggml_numa_init()**

In `ggml/src/ggml-cpu/ggml-cpu.c`, after the node enumeration loop (line 704) and before line 707 (CPU enumeration), add a static helper function and the interleave logic.

First, add the helper function before `ggml_numa_init()` (around line 675):

```c
#if defined(__gnu_linux__)
#include <sys/syscall.h>
#ifndef SYS_set_mempolicy
#ifdef SYS_set_mempolicy
#define GGML_SYS_SET_MEMPOLICY SYS_set_mempolicy
#else
#define GGML_SYS_SET_MEMPOLICY 279  // x86_64 syscall number for set_mempolicy
#endif
#else
#define GGML_SYS_SET_MEMPOLICY SYS_set_mempolicy
#endif

static void ggml_cpu_set_numa_interleave(uint32_t n_nodes) {
    // Build nodemask: set bits 0..n_nodes-1
    unsigned long nodemask = (1UL << n_nodes) - 1;
    void *nmask = &nodemask;
    // Use syscall directly (no libnuma dependency)
    long ret = syscall(GGML_SYS_SET_MEMPOLICY, MPOL_INTERLEAVE, nmask, n_nodes * 8, 0, 0);
    if (ret == 0) {
        GGML_LOG_INFO("NUMA: set mempolicy INTERLEAVE across %d nodes\n", n_nodes);
    } else {
        GGML_LOG_WARN("NUMA: set_mempolicy(INTERLEAVE) failed: %s\n", strerror(-ret));
    }
}

static void ggml_cpu_reset_numa_interleave(void) {
    // Reset to default policy
    syscall(GGML_SYS_SET_MEMPOLICY, MPOL_DEFAULT, NULL, 0, 0, 0);
}
#else
static void ggml_cpu_set_numa_interleave(uint32_t n_nodes) { UNUSED(n_nodes); }
static void ggml_cpu_reset_numa_interleave(void) {}
#endif
```

Then in `ggml_numa_init()`, after line 692 (`g_state.numa.numa_strategy = numa_flag;`), add the interleave case:

```c
    // Apply NUMA interleave early, before any allocations
    if (numa_flag == GGML_NUMA_STRATEGY_INTERLEAVE) {
        // Need to enumerate nodes first to build nodemask
        // (node enumeration happens below, so we set a flag and apply after)
    }
```

Actually, cleaner: after the node enumeration completes (after line 704) and before the CPU enumeration, add:

```c
    // Apply interleave policy if requested
    if (numa_flag == GGML_NUMA_STRATEGY_INTERLEAVE && g_state.numa.n_nodes > 0) {
        ggml_cpu_set_numa_interleave(g_state.numa.n_nodes);
    }
```

- [ ] **Step 3: Add reset on backend close**

In `ggml/src/ggml-cpu/ggml-cpu.cpp`, find the backend destructor or free function. When the CPU backend is freed, call `ggml_cpu_reset_numa_interleave()` if `g_state.numa.numa_strategy == GGML_NUMA_STRATEGY_INTERLEAVE`.

If there's no explicit backend close hook, add a log message noting that the interleave policy persists until process exit (which is acceptable for most workloads).

- [ ] **Step 4: Verify build**

```bash
cmake -B build_epyc -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF 2>&1 | tail -5 && cmake --build build_epyc --config Release -j$(nproc) 2>&1 | grep -iE "(error|warn)" | head -5
```

Expected: No errors or warnings related to NUMA.

- [ ] **Step 5: Verify NUMA interleave at runtime**

```bash
# Quick test: run a binary that initializes ggml CPU backend with INTERLEAVE
# Check numactl --show before and after
numactl --show | grep policy
```

Expected: Policy changes to "interleaved" across detected nodes.

- [ ] **Step 6: Commit**

```bash
git add ggml/include/ggml-cpu.h ggml/src/ggml-cpu/ggml-cpu.c
git commit -m "$(cat <<'EOF'
cpu: add GGML_NUMA_STRATEGY_INTERLEAVE for multi-node memory interleave

Uses set_mempolicy(MPOL_INTERLEAVE) syscall to distribute allocations
across all NUMA nodes. Targets EPYC 9V74 with 12-channel DDR5 to saturate
all memory channels. No libnuma dependency — raw syscall only.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: CCD-Aware Thread Affinity

**Files:**
- Modify: `ggml/include/ggml-cpu.h:28-35` — `GGML_NUMA_STRATEGY_CCD` already added in Task 1
- Modify: `ggml/src/ggml-cpu/ggml-cpu-impl.h` — add `struct ggml_ccd_topology`
- Modify: `ggml/src/ggml-cpu/ggml-cpu.c:679-765` — CCD topology probe in `ggml_numa_init()`
- Modify: `ggml/src/ggml-cpu/ggml-cpu.c:2171-2214` — CCD case in `set_numa_thread_affinity()`

**Context:** `set_numa_thread_affinity()` (line 2171) switches on `g_state.numa.numa_strategy`. The `default` case (line 2196) returns early. We add `GGML_NUMA_STRATEGY_CCD` as a new case that pins thread N to `ccd_threads[N]` from the probed topology.

CCD topology is probed by reading `/sys/devices/system/cpu/cpu*/topology/core_defaults` (L3 cache domain ID). SMT siblings are identified from `/sys/devices/system/cpu/cpu*/topology/thread_siblings_list`. Primary threads (lower CPU ID per core) are scheduled first, then siblings.

The `struct ggml_numa_nodes` (line 568) lives inside `g_state`. We extend `ggml_state` (line 585) with a CCD topology struct.

- [ ] **Step 1: Add CCD topology struct to ggml-cpu.c**

In `ggml/src/ggml-cpu/ggml-cpu.c`, after `struct ggml_numa_nodes` (line 579) and before `struct ggml_state` (line 585), add:

```c
#if defined(__gnu_linux__)
struct ggml_ccd_topology {
    uint32_t n_ccds;
    uint32_t ccd_threads[GGML_NUMA_MAX_CPUS];       // thread IDs, CCD-ordered, primaries first
    uint32_t ccd_thread_count[GGML_NUMA_MAX_CPUS];  // thread count per CCD
    uint32_t ccd_for_cpu[GGML_NUMA_MAX_CPUS];       // CPU -> CCD ID mapping
    bool     is_sibling[GGML_NUMA_MAX_CPUS];        // true if this CPU is an SMT sibling
};
#endif

struct ggml_state {
    struct ggml_numa_nodes numa;
#if defined(__gnu_linux__)
    struct ggml_ccd_topology ccd;
#endif
};
```

- [ ] **Step 2: Add CCD topology probe function**

Before `ggml_numa_init()` (around line 670), add the probe function:

```c
#if defined(__gnu_linux__)
static void ggml_probe_ccd_topology(void) {
    struct stat st;
    char path[256];
    char buf[256];
    int rv;
    uint32_t n_ccds = 0;

    g_state.ccd.n_ccds = 0;
    memset(g_state.ccd.ccd_for_cpu, 0xFF, sizeof(g_state.ccd.ccd_for_cpu));
    memset(g_state.ccd.is_sibling, 0, sizeof(g_state.ccd.is_sibling));

    // Phase 1: read core_defaults for each CPU (L3 cache domain = CCD ID)
    for (uint32_t c = 0; c < g_state.numa.total_cpus; c++) {
        rv = snprintf(path, sizeof(path),
            "/sys/devices/system/cpu/cpu%u/topology/core_defaults", c);
        GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
        FILE *f = fopen(path, "r");
        if (!f) continue;
        if (fgets(buf, sizeof(buf), f)) {
            int ccd_id = atoi(buf);
            if (ccd_id >= 0) {
                g_state.ccd.ccd_for_cpu[c] = (uint32_t)ccd_id;
                if ((uint32_t)ccd_id + 1 > n_ccds) n_ccds = (uint32_t)ccd_id + 1;
            }
        }
        fclose(f);
    }

    if (n_ccds == 0) {
        GGML_LOG_WARN("CCD: core_defaults not available, CCD affinity disabled\n");
        return;
    }

    g_state.ccd.n_ccds = n_ccds;

    // Phase 2: identify SMT siblings from thread_siblings_list
    for (uint32_t c = 0; c < g_state.numa.total_cpus; c++) {
        rv = snprintf(path, sizeof(path),
            "/sys/devices/system/cpu/cpu%u/topology/thread_siblings_list", c);
        GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
        f = fopen(path, "r");
        if (!f) continue;
        if (fgets(buf, sizeof(buf), f)) {
            // Parse "0,1" or "0-1" format. Find siblings.
            // The sibling with lower CPU ID is primary, higher is sibling.
            char *saveptr = NULL;
            char *token = strtok_r(buf, ",-", &saveptr);
            while (token) {
                int sibling_id = atoi(token);
                if (sibling_id > (int)c) {
                    g_state.ccd.is_sibling[(uint32_t)sibling_id] = true;
                }
                token = strtok_r(NULL, ",-", &saveptr);
            }
        }
        fclose(f);
        if (!g_state.ccd.is_sibling[c] && c > 0) {
            // Verify: check if any lower CPU claims this one as sibling
            // If thread_siblings_list for a lower CPU includes c, then c is a sibling.
        }
    }

    // Phase 3: build ccd_threads array — primaries first, then siblings, grouped by CCD
    // Two passes per CCD: first count primaries and siblings separately, then fill the array.
    {
        uint32_t primary_count[GGML_NUMA_MAX_CPUS] = {0};
        uint32_t sibling_count[GGML_NUMA_MAX_CPUS] = {0};

        // Count primaries and siblings per CCD
        for (uint32_t c = 0; c < g_state.numa.total_cpus; c++) {
            uint32_t ccd_id = g_state.ccd.ccd_for_cpu[c];
            if (ccd_id >= n_ccds) continue;
            if (g_state.ccd.is_sibling[c]) {
                sibling_count[ccd_id]++;
            } else {
                primary_count[ccd_id]++;
            }
        }

        // Fill: primaries first (CCD-ordered), then siblings
        uint32_t write_idx = 0;
        // Write all primaries, CCD by CCD
        for (uint32_t ccd = 0; ccd < n_ccds; ccd++) {
            for (uint32_t c = 0; c < g_state.numa.total_cpus; c++) {
                if (g_state.ccd.ccd_for_cpu[c] == ccd && !g_state.ccd.is_sibling[c]) {
                    g_state.ccd.ccd_threads[write_idx++] = c;
                }
            }
        }
        // Write all siblings, CCD by CCD
        for (uint32_t ccd = 0; ccd < n_ccds; ccd++) {
            for (uint32_t c = 0; c < g_state.numa.total_cpus; c++) {
                if (g_state.ccd.ccd_for_cpu[c] == ccd && g_state.ccd.is_sibling[c]) {
                    g_state.ccd.ccd_threads[write_idx++] = c;
                }
            }
        }

        // Compute per-CCD thread counts for logging
        for (uint32_t ccd = 0; ccd < n_ccds; ccd++) {
            g_state.ccd.ccd_thread_count[ccd] = primary_count[ccd] + sibling_count[ccd];
        }
    }

    GGML_LOG_INFO("CCD: detected %u CCDs, %u total threads\n", n_ccds, g_state.numa.total_cpus);
    for (uint32_t ccd = 0; ccd < n_ccds; ccd++) {
        GGML_LOG_INFO("  CCD %u: %u threads\n", ccd, g_state.ccd.ccd_thread_count[ccd]);
    }
}
#else
static void ggml_probe_ccd_topology(void) {}
#endif
```

- [ ] **Step 3: Call probe from ggml_numa_init()**

In `ggml_numa_init()`, after the CPU-to-NUMA-node mapping loop (line 749) and before the numa_balancing check (line 751), add:

```c
    // Probe CCD topology for CCD-aware affinity
    if (numa_flag == GGML_NUMA_STRATEGY_CCD) {
        ggml_probe_ccd_topology();
    }
```

- [ ] **Step 4: Add CCD case to set_numa_thread_affinity()**

In `set_numa_thread_affinity()` (line 2180), add a case for `GGML_NUMA_STRATEGY_CCD`:

```c
        case GGML_NUMA_STRATEGY_CCD:
            {
                if (g_state.ccd.n_ccds == 0 || thread_n >= (int)g_state.numa.total_cpus) {
                    return;  // fallback: no CCD data, don't pin
                }
                uint32_t target_cpu = g_state.ccd.ccd_threads[thread_n];
                cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
                size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);
                CPU_ZERO_S(setsize, cpus);
                CPU_SET_S(target_cpu, setsize, cpus);
                rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
                if (rv) {
                    fprintf(stderr, "warning: pthread_setaffinity_np() (CCD) failed: %s\n", strerror(rv));
                }
                CPU_FREE(cpus);
            }
            return;
```

Insert this case before the `default:` case (line 2196).

- [ ] **Step 5: Verify build**

```bash
cmake --build build_epyc --config Release -j$(nproc) 2>&1 | grep -iE "(error|warn)" | head -5
```

- [ ] **Step 6: Verify CCD detection at runtime**

```bash
# Check that CCD topology is detected
grep -c core_defaults /sys/devices/system/cpu/cpu0/topology/core_defaults 2>/dev/null && echo "sysfs available" || echo "sysfs not available"
# Run a binary with GGML_NUMA_STRATEGY_CCD and check log output for "CCD: detected N CCDs"
```

- [ ] **Step 7: Commit**

```bash
git add ggml/src/ggml-cpu/ggml-cpu.c ggml/src/ggml-cpu/ggml-cpu-impl.h
git commit -m "$(cat <<'EOF'
cpu: add CCD-aware thread affinity (GGML_NUMA_STRATEGY_CCD)

Probes /sys/devices/system/cpu/*/topology/core_defaults to discover
L3 cache domains (CCDs on EPYC). Pins threads to CCD-local cores,
filling physical cores before SMT siblings. Falls back to no pinning
if sysfs core_defaults is unavailable.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: AVX-512 VNNI vec_dot for Q8_0

**Files:**
- Modify: `ggml/src/ggml-cpu/arch/x86/quants.c:1170-1236` — add `#if defined(__AVX512VNNI__)` path to `ggml_vec_dot_q8_0_q8_0`

**Context:** `ggml_vec_dot_q8_0_q8_0` (line 1170 in quants.c) currently has `#if defined(__AVX2__)` and `#elif defined(__AVX__)` paths. Add a new `#if defined(__AVX512VNNI__) && defined(__AVX512BW__) && defined(__AVX512VL__)` path before the AVX2 path. This is the exact same pattern used by `mul_sum_i8_pairs_float` (line 106).

The existing AVX2 path loads 32 bytes via `_mm256_loadu_si256`. The VNNI path loads 64 bytes via `_mm512_loadu_si512` and uses `_mm512_dpbusd_epi32`. Each block is QK8_0=32 bytes, so the VNNI path processes 2 blocks per DPBusD call.

- [ ] **Step 1: Add AVX-512 VNNI path to ggml_vec_dot_q8_0_q8_0**

In `ggml/src/ggml-cpu/arch/x86/quants.c`, at line 1170, restructure the `#if defined(__AVX2__)` to be `#if defined(__AVX512VNNI__) && defined(__AVX512BW__) && defined(__AVX512VL__)` first, then `#elif defined(__AVX2__)` second.

The VNNI path processes pairs of blocks (ib + 1 < nb) to fill a 64-byte load:

```c
#if defined(__AVX512VNNI__) && defined(__AVX512BW__) && defined(__AVX512VL__)
    // AVX-512 VNNI: 64 INT8 pairs per DPBusD, 2 blocks of QK8_0 per iteration
    __m512 acc = _mm512_setzero_ps();
    int ib = 0;

    // Horizontal sum helper for __m512 (8 FP32 elements)
    static inline float hsum_ps_8(__m512 a) {
        __m256 lo = _mm512_castps512_ps256(a);
        __m256 hi = _mm512_extractf32x8_ps(a, 1);
        a = _mm256_add_ps(lo, hi);
        a = _mm256_add_ps(a, _mm256_permute2f128_ps(a, a, 1));
        a = _mm256_hadd_ps(a, a);
        return _mm256_cvtss_f32(a);
    }

#if defined(__clang__) && defined(__amd64__)
    #pragma clang loop interleave(count=4) unroll(count=2)
#endif
    for (; ib + 1 < nb; ib += 2) {
        // Prefetch ahead — DDR5 latency ~100-150ns
        __builtin_prefetch(&x[ib + 4], 0, 3);
        __builtin_prefetch(&y[ib + 4], 0, 3);

        float d0 = GGML_CPU_FP16_TO_FP32(x[ib].d) * GGML_CPU_FP16_TO_FP32(y[ib].d);
        float d1 = GGML_CPU_FP16_TO_FP32(x[ib+1].d) * GGML_CPU_FP16_TO_FP32(y[ib+1].d);

        // Each block_q8_0 is 32 INT8 + fp16 scale, aligned to 32 bytes.
        // Two contiguous blocks = 64 bytes of qs data.
        const __m512i zx0 = _mm512_loadu_si512((const __m512i *)x[ib].qs);
        const __m512i zy0 = _mm512_loadu_si512((const __m512i *)y[ib].qs);
        const __m512i z_sum0 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), zx0, zy0);

        const __m512i zx1 = _mm512_loadu_si512((const __m512i *)x[ib+1].qs);
        const __m512i zy1 = _mm512_loadu_si512((const __m512i *)y[ib+1].qs);
        const __m512i z_sum1 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), zx1, zy1);

        __m512 s0 = _mm512_set1_ps(d0);
        __m512 s1 = _mm512_set1_ps(d1);
        acc = _mm512_fmadd_ps(s0, _mm512_cvtepi32_ps(z_sum0), acc);
        acc = _mm512_fmadd_ps(s1, _mm512_cvtepi32_ps(z_sum1), acc);
    }

    float sumf = hsum_ps_8(acc);
#elif defined(__AVX2__)
    // Existing AVX2 path (unchanged)
```

Then the existing `#elif defined(__AVX__)` and the scalar fallback remain unchanged.

**Critical alignment note:** `block_q8_0` has `GGML_ALIGNED(32)`. The `qs` array is 32 bytes. Two blocks' `qs` are NOT guaranteed to be contiguous in memory (each block struct has padding for the `d` field after `qs`). So `_mm512_loadu_si512((const __m512i *)x[ib].qs)` may read into `x[ib].d` or padding. The safe approach is to use two 256-bit loads per block and permute, or to load 64 bytes from `x[ib].qs` only if the blocks are guaranteed contiguous.

**Safer implementation for Q8_0:** Since `sizeof(block_q8_0) = 34` (32 qs + 2 fp16 + padding to 32 alignment = 34 or 40 bytes), two blocks are NOT contiguous for a 64-byte load. The VNNI path should process one block per iteration with two 256-bit DPBusD calls:

```c
    for (; ib < nb; ++ib) {
        float d = GGML_CPU_FP16_TO_FP32(x[ib].d) * GGML_CPU_FP16_TO_FP32(y[ib].d);
        // 32 bytes of qs — load as two 128-bit chunks, sign-extend to 256-bit, DPBusD
        const __m128i xl = _mm_loadu_si128((const __m128i *)x[ib].qs);
        const __m128i xh = _mm_loadu_si128((const __m128i *)x[ib].qs + 1);
        const __m128i yl = _mm_loadu_si128((const __m128i *)y[ib].qs);
        const __m128i yh = _mm_loadu_si128((const __m128i *)y[ib].qs + 1);
        // Use _mm256_dpbusd_epi32 on each 16-byte lane (16 INT8 pairs each)
        __m256i zl = _mm256_dpbusd_epi32(_mm256_setzero_si256(), _mm256_cvtepu8_epi16(xl), _mm256_cvtepi8_epi16(yl));
        __m256i zh = _mm256_dpbusd_epi32(_mm256_setzero_si256(), _mm256_cvtepu8_epi16(xh), _mm256_cvtepi8_epi16(yh));
        __m256i sum = _mm256_add_epi32(zl, zh);
        acc = _mm512_fmadd_ps(_mm512_set1_ps(d), _mm512_cvtepi32_ps(_mm256_broadcast_i32x4(sum)), acc);
    }
```

**Actually the simplest path:** The existing `mul_sum_i8_pairs_float` already uses `_mm256_dpbusd_epi32` on 256-bit registers. The Q8_0 function already calls it. So the current AVX2 path with VNNI support is already fast for 256-bit. The gain from full 512-bit loads would require repacking the data. For now, keep the existing structure and just ensure the AVX-512 variant gets the `__AVX512VNNI__` path in `mul_sum_i8_pairs_float`.

**Decision:** Since the existing `ggml_vec_dot_q8_0_q8_0` already calls `mul_sum_i8_pairs_float` which already has `__AVX512VNNI__` support (line 106), the Q8_0 path is already VNNI-accelerated for 256-bit loads. The only improvement is to add a 512-bit path when blocks can be loaded contiguously. For this task, add the 512-bit path with a contiguous check:

```c
#if defined(__AVX512VNNI__) && defined(__AVX512BW__) && defined(__AVX512VL__)
    // Fast path: only use 512-bit loads if we can verify contiguous qs data
    // Since block_q8_0 is 34 bytes (32 qs + 2 d), blocks are NOT contiguous for qs.
    // Fall through to AVX2 path which uses mul_sum_i8_pairs_float (already VNNI-accelerated).
    // This path can be expanded later with repacked data.
```

Given this analysis, Q8_0 is already well-served by the VNNI support in `mul_sum_i8_pairs_float`. Skip the dedicated 512-bit Q8_0 path — the AVX2 path with VNNI `mul_sum_i8_pairs_float` IS the optimal path for the current block layout.

**Task 3 becomes:** Verify that `mul_sum_i8_pairs_float` with `__AVX512VNNI__` is correctly compiled and tested. No code change needed — just verify.

- [ ] **Step 1: Verify VNNI path is active in mul_sum_i8_pairs_float**

The function at quants.c:122 already has:
```c
#if __AVXVNNIINT8__
    // Uses _mm256_dpbssd_epi32 (AVX-VNNI signed)
#else
    // Uses abs+sign + _mm256_dpbusd_epi32 (AVX512VNNI unsigned, line 108)
```

Confirm this compiles correctly for the AVX512-VNNI variant.

- [ ] **Step 2: Verify build**

```bash
cmake -B build_epyc -DGGML_CUDA=OFF -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF 2>&1 | tail -5
cmake --build build_epyc --config Release -j$(nproc) 2>&1 | grep -iE "(error|warn)" | head -5
```

- [ ] **Step 3: Verify correctness**

```bash
cd build_epyc && timeout 60 bin/test-backend-ops -b CPU0 -R "MUL_MAT(type=q8_0" 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 4: Commit (if changes needed)**

If no code changes were needed (VNNI already active via mul_sum_i8_pairs_float), skip the commit for this task. If any verification or minor optimization was made, commit with:

```bash
git add ggml/src/ggml-cpu/arch/x86/quants.c
git commit -m "$(cat <<'EOF'
cpu: verify Q8_0 vec_dot already uses AVX-512 VNNI via mul_sum_i8_pairs_float

No code change needed — the existing AVX2 path dispatches to
mul_sum_i8_pairs_float which has __AVX512VNNI__ guards (line 106).
The AVX512-VNNI variant of quants.c automatically gets the
_mm256_dpbusd_epi32 path.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: AVX-512 VNNI vec_dot for Q4_K

**Files:**
- Modify: `ggml/src/ggml-cpu/arch/x86/quants.c:1900-2076` — add `#if defined(__AVX512VNNI__)` path to `ggml_vec_dot_q4_K_q8_K`

**Context:** `ggml_vec_dot_q4_K_q8_K` (line 1900) currently uses AVX2 `_mm256_maddubs_epi16` for the nibble multiply. The function already has AVX2 and AVX paths. Add a VNNI path that processes 64 elements per sub-block using `_mm512_dpbusd_epi32`.

The key insight: Q4_K has 32 half-bytes (16 bytes) per sub-block of 64 elements (32 low + 32 high nibbles). Each nibble is paired with one Q8_K INT8 value. The Q8_K bsums correction (dmin) uses 256-bit operations regardless of the VNNI path.

- [ ] **Step 1: Add AVX-512 VNNI path to ggml_vec_dot_q4_K_q8_K**

At line 1919 in quants.c (currently `#if defined __AVX2__`), add a new VNNI guard before AVX2:

```c
#if defined(__AVX512VNNI__) && defined(__AVX512BW__) && defined(__AVX512VL__)

    // Helper: hsum for __m512i (8 int32 elements)
    static inline float hsum_epi32_8(__m512i a) {
        __m256i lo = _mm512_castsi512_si256(a);
        __m256i hi = _mm512_extracti64x4_epi64(a, 1);
        __m256i sum = _mm256_add_epi32(lo, hi);
        sum = _mm256_add_epi32(sum, _mm256_shuffle_i32x4(sum, sum, 1));
        sum = _mm256_add_epi32(sum, _mm256_shuffle_epi32(sum, 8));
        sum = _mm256_add_epi32(sum, _mm256_shuffle_epi32(sum, 1));
        return (float)_mm256_extract_epi32(sum, 0);
    }

    // Copy scale shuffle table from quants.c (static, not externally visible)
    // get_scale_shuffle_k4() is static in quants.c — replicate its 8-entry table here
    // or use _mm256_shuffle_epi8 directly with inline shuffle masks.
    // For now, use the existing AVX2 scale extraction, then VNNI for the multiply.

    const __m256i m4 = _mm256_set1_epi8(0xF);
    __m512 acc = _mm512_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();

   for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        const float dmin = -y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

        // Scale extraction — identical to AVX2 path
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));

        // dmin correction — AVX2 path (unchanged)
        const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y[i].bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

        const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = MM256_SET_M128I(sc128, sc128);

        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;
        __m512i sumi = _mm512_setzero_si512();

#if defined(__clang__) && defined(__amd64__)
        #pragma clang loop unroll(count=4)
#endif
        for (int j = 0; j < QK_K/64; ++j) {
            // Prefetch 3 cache lines ahead (DDR5 latency ~100-150 ns)
            __builtin_prefetch(q4 + 96, 0, 3);
            __builtin_prefetch(q8 + 96, 0, 3);

            const __m256i scale_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+0));
            const __m256i scale_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+1));

            // Load 32 half-bytes of Q4 data (64 nibbles)
            const __m256i q4bits = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
            const __m256i q4l = _mm256_and_si256(q4bits, m4);
            const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

            // Load 64 bytes of Q8 data (one per nibble)
            const __m512i q8_all = _mm512_loadu_si512((const __m512i*)q8); q8 += 64;
            // Split into low/high 32-byte halves for low/high nibble multiplication
            const __m256i q8l_256 = _mm512_castsi512_si256(q8_all);
            const __m256i q8h_256 = _mm512_extracti64x4_epi64(q8_all);

            // DPBusD on 256-bit: each processes 16 INT8 pairs → 4 INT32
            // q4 nibbles are unsigned [0,15], q8 is signed — use _mm256_dpbusd_epi32
            const __m256i zero = _mm256_setzero_si256();
            // Each 256-bit DPBusD handles 16 elements. Need 2 per 32 nibbles.
            // Lower 16 nibbles of q4l:
            const __m128i q4l_lo = _mm256_castsi256_si128(q4l);
            const __m128i q4l_hi = _mm256_extracti128_si256(q4l, 1);
            const __m128i q8l_lo = _mm256_castsi256_si128(q8l_256);
            const __m128i q8l_hi = _mm256_extracti128_si256(q8l_256, 1);

            __m256i s_l_lo = _mm256_dpbusd_epi32(zero, q4l_lo, q8l_lo);
            __m256i s_l_hi = _mm256_dpbusd_epi32(zero, q4l_hi, q8l_hi);
            __m256i sum_l = _mm256_add_epi32(s_l_lo, s_l_hi);

            const __m128i q4h_lo = _mm256_castsi256_si128(q4h);
            const __m128i q4h_hi = _mm256_extracti128_si256(q4h, 1);
            const __m128i q8h_lo = _mm256_castsi256_si128(q8h_256);
            const __m128i q8h_hi = _mm256_extracti128_si256(q8h_256, 1);

            __m256i s_h_lo = _mm256_dpbusd_epi32(zero, q4h_lo, q8h_lo);
            __m256i s_h_hi = _mm256_dpbusd_epi32(zero, q4h_hi, q8h_hi);
            __m256i sum_h = _mm256_add_epi32(s_h_lo, s_h_hi);

            // Scale and accumulate
            sum_l = _mm256_madd_epi16(scale_l, sum_l);
            sum_h = _mm256_madd_epi16(scale_h, sum_h);

            // Add to 512-bit accumulator
            sumi = _mm512_add_epi32(sumi, _mm512_cvtepi32_epi64(_mm256_add_epi32(sum_l, sum_h)));
        }

        __m512 vd = _mm512_set1_ps(d);
        acc = _mm512_fmadd_ps(vd, _mm512_cvtepi32_ps(sumi), acc);
    }

    float sumf = hsum_ps_8(acc);
    acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));
    *s = sumf + _mm_cvtss_f32(acc_m);

#elif defined __AVX2__
    // Existing AVX2 path (unchanged)
```

**Note on `get_scale_shuffle_k4()`:** This function returns `const __m128i*` shuffle masks. It's `static` in quants.c. Since the VNNI path is now inside quants.c, it can directly call `get_scale_shuffle_k4()` — no visibility issue.

**Note on `_mm256_dpbusd_epi32` vs `_mm512_dpbusd_epi32`:** The VNNI path here uses 256-bit DPBusD (same as `mul_sum_i8_pairs_float`) since the nibble unpacking is naturally 256-bit. The gain over AVX2's `_mm256_maddubs_epi16` is that DPBusD goes directly to INT32 (no second add needed), saving 2 instructions per sub-block.

- [ ] **Step 2: Verify build**

```bash
cmake --build build_epyc --config Release -j$(nproc) 2>&1 | grep -iE "(error|warn)" | head -10
```

- [ ] **Step 3: Verify correctness**

```bash
cd build_epyc && timeout 60 bin/test-backend-ops -b CPU0 -R "MUL_MAT(type=q4_0\|type=q4_k" 2>&1 | tail -10
```

Expected: PASS for all Q4_0 and Q4_K matmul tests.

- [ ] **Step 4: Commit**

```bash
git add ggml/src/ggml-cpu/arch/x86/quants.c
git commit -m "$(cat <<'EOF'
cpu: add AVX-512 VNNI path for Q4_K vec_dot

Uses _mm256_dpbusd_epi32 for direct INT8→INT32 multiply-add,
eliminating the _mm256_maddubs_epi16 + add step. Nibble unpack
stays 256-bit (natural width for 32 half-bytes). Scale extraction
and dmin correction unchanged from AVX2 path.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: AVX-512 BF16 vec_dot

**Files:**
- Modify: `ggml/src/ggml-cpu/vec.cpp` — add `#if defined(__AVX512BF16__)` path to `ggml_vec_dot_bf16`

**Context:** `ggml_vec_dot_bf16` in vec.cpp processes BF16 dot products. Add an AVX-512 path that uses `_mm512_cvtne2pack_ps` to convert 32 BF16→FP32 and FMA, or `_mm512_dpbf16_ps` if `__AVX512BF16__`.

- [ ] **Step 1: Add AVX-512 path to ggml_vec_dot_bf16**

In `ggml/src/ggml-cpu/vec.cpp`, find `ggml_vec_dot_bf16` and add an `#if defined(__AVX512BF16__)` path before any existing SIMD path:

```c
#if defined(__AVX512BF16__)
    __m512 acc = _mm512_setzero_ps();
    int i = 0;
    for (; i + 63 < n; i += 64) {
        __m256i xb = _mm256_loadu_si256((const __m256i*)(x + i));
        __m256i yb = _mm256_loadu_si256((const __m256i*)(y + i));
        // Split 256-bit into two 128-bit, extend to FP32, interleave
        __m512 xf = _mm512_cvtne2pack_ps(
            _mm512_cvtepi32_epi32(_mm512_cvtepu16_epi32(_mm256_castsi256_si128(xb))),
            _mm512_cvtepi32_epi32(_mm512_cvtepu16_epi32(_mm256_extracti128_si256(xb, 1))));
        __m512 yf = _mm512_cvtne2pack_ps(
            _mm512_cvtepi32_epi32(_mm512_cvtepu16_epi32(_mm256_castsi256_si128(yb))),
            _mm512_cvtepi32_epi32(_mm512_cvtepu16_epi32(_mm256_extracti128_si256(yb, 1))));
        acc = _mm512_fmadd_ps(xf, yf, acc);
    }
    // hsum_ps_8 helper (same as Task 3)
    float sumf = hsum_ps_8(acc);
    for (; i < n; i++) sumf += (float)x[i] * (float)y[i];
    *s = sumf;
#elif defined(__AVX2__)
    // Existing AVX2 path (unchanged)
```

- [ ] **Step 2: Verify build**

```bash
cmake --build build_epyc --config Release -j$(nproc) 2>&1 | grep -iE "(error|warn)" | head -10
```

- [ ] **Step 3: Verify correctness**

```bash
cd build_epyc && timeout 60 bin/test-backend-ops -b CPU0 -R "bf16" 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add ggml/src/ggml-cpu/vec.cpp
git commit -m "$(cat <<'EOF'
cpu: add AVX-512 BF16 vec_dot

Uses _mm512_cvtne2pack_ps to convert 32 BF16→FP32 per iteration
and _mm512_fmadd_ps for accumulation. Processes 64 BF16 pairs
per iteration vs 32 with AVX2.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Integration Testing

**Files:**
- Test: `build_epyc/bin/test-backend-ops` — verify all quant types on CPU backend

- [ ] **Step 1: Full clean build with all variants**

```bash
rm -rf build_epyc
cmake -B build_epyc -DGGML_CUDA=OFF -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF 2>&1 | grep -iE "(avx|vnni|bf16)" | head -10
cmake --build build_epyc --config Release -j$(nproc) 2>&1 | grep -iE "(error)" | head -5
```

Expected: Clean build, no errors. AVX2, AVX512, AVX512_VNNI, AVX512_BF16 variants all compile.

- [ ] **Step 2: Run full CPU test suite**

```bash
cd build_epyc && timeout 180 bin/test-backend-ops -b CPU0 2>&1 | tail -30
```

Expected: All tests pass. No regressions on any quant type.

- [ ] **Step 3: Verify Q8_0 + Q4_K specifically**

```bash
cd build_epyc && timeout 60 bin/test-backend-ops -b CPU0 -R "MUL_MAT(type=q8_0\|MUL_MAT(type=q4_k" 2>&1 | tail -10
```

Expected: PASS for all Q8_0 and Q4_K matmul tests.

- [ ] **Step 4: Verify BF16**

```bash
cd build_epyc && timeout 60 bin/test-backend-ops -b CPU0 -R "bf16" 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 5: Regression test — build without AVX-512**

```bash
cmake -B build_epyc_noavx512 -DGGML_CUDA=OFF -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build_epyc_noavx512 --config Release -j$(nproc) 2>&1 | grep -iE "(error)" | head -5
cd build_epyc_noavx512 && timeout 60 bin/test-backend-ops -b CPU0 -R "MUL_MAT(type=q8_0\|MUL_MAT(type=q4_k" 2>&1 | tail -10
```

Expected: Builds cleanly with AVX2 only. All tests pass using AVX2 paths.

- [ ] **Step 6: Final cleanup**

```bash
rm -rf build_epyc_noavx512
git log --oneline -5  # Should show: NUMA, CCD, Q4_K VNNI, BF16, integration
```

---

## Follow-up Work (Out of Scope)

- AVX-512 repacking (x8 block) for tiled GEMM — pre-repack Q4_K weights to 512-bit lanes
- Full AVX-512 sgemm kernel for FP16/FP32 matmul
- Hybrid TMA bridge: pinned RAM + async transfer benchmark (spec 2)
- per-CCD work stealing for load-balanced matmul across CCDs
- UD-Q4_K_XL specific vec_dot (if different from Q4_K)
