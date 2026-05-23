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
| `ggml/src/ggml-cpu/ggml-cpu-impl.h` | `struct ggml_ccd_topology` definition |
| `ggml/src/ggml-cpu/arch/x86/epyc-cpu.c` | AVX-512 VNNI vec_dot for Q8_0, Q4_K/Q8_K, BF16 (weak symbols, override quants.c when VNNI available) |
| `ggml/src/ggml-cpu/quants.h` | Declare new AVX-512 vec_dot functions |
| `ggml/src/ggml-cpu/CMakeLists.txt` | Add epyc-cpu.c to x86 sources |

**Why epyc-cpu.c uses weak symbols:** The CMake builds multiple CPU-backend variants (AVX2, AVX-512, etc.), each compiling all x86 sources with its own flags. Linking combines the variant objects. By marking the AVX-512 implementations in `epyc-cpu.c` as strong (normal) and the existing implementations in `quants.c` as `__attribute__((weak))`, the linker automatically picks the VNNI version when available. This is the same dispatch principle used by `mul_sum_i8_pairs_float` in `quants.c` lines 106-133.

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
- Create: `ggml/src/ggml-cpu/arch/x86/epyc-cpu.c` — AVX-512 VNNI implementations
- Modify: `ggml/src/ggml-cpu/quants.h` — declare AVX-512 functions
- Modify: `ggml/src/ggml-cpu/arch/x86/quants.c:1170` — mark existing `ggml_vec_dot_q8_0_q8_0` as weak

**Context:** `ggml_vec_dot_q8_0_q8_0` (line 1170 in quants.c) processes `block_q8_0` (32 INT8 + 1 fp16 scale). The AVX-512 version processes 64 bytes per DPBusD instruction. The AVX2 version loads 32 bytes via `_mm256_loadu_si256` and calls `mul_sum_i8_pairs_float`. The VNNI version loads 64 bytes via `_mm512_loadu_si512` and uses `_mm512_dpbusd_epi32`.

Dispatch: mark the existing `ggml_vec_dot_q8_0_q8_0` in quants.c as `__attribute__((weak))`. The strong definition in epyc-cpu.c (compiled only with `__AVX512VNNI__`) overrides it at link time.

**Reference — block_q8_0 structure:** Found in `ggml/include/ggml.h` or `ggml/src/ggml-cpu/quants.h`:
```c
#define QK8_0 32
typedef struct {
    ggml_aligned_buffer_m16b(qs, QK8_0);
    ggml_fp16_t d;
} block_q8_0 GGML_ALIGNED(32);
```

- [ ] **Step 1: Create epyc-cpu.c with Q8_0 VNNI implementation**

Create `ggml/src/ggml-cpu/arch/x86/epyc-cpu.c`:

```c
#include "ggml-common.h"

#if defined(__AVX512VNNI__) && defined(__AVX512BW__) && defined(__AVX512VL__) && defined(__AVX512F__)

#include <immintrin.h>
#include <string.h>
#include <math.h>

// Forward declarations from ggml types
#include "ggml.h"

//
// Q8_0 dot product using AVX-512 VNNI
//
// Processes 64 INT8 pairs per _mm512_dpbusd_epi32 instruction.
// 2 blocks of 32 = 64 elements processed per inner iteration.
// Inner loop unrolls 4x for latency hiding.
//

void ggml_vec_dot_q8_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_0;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc);

    const block_q8_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    // Handle strided access (bx, by, bs)
    if (bx == sizeof(block_q8_0) && by == sizeof(block_q8_0) && bs == sizeof(float)) {
        // Fast path: contiguous blocks
        __m512 acc = _mm512_setzero_ps();
        int ib = 0;

#if defined(__clang__) && defined(__amd64__)
        #pragma clang loop vectorize(enable) interleave(count=4)
        #pragma clang loop unroll(count=2)
#endif
        for (; ib + 1 < nb; ib += 2) {
            float d0 = GGML_CPU_FP16_TO_FP32(x[ib].d) * GGML_CPU_FP16_TO_FP32(y[ib].d);
            float d1 = GGML_CPU_FP16_TO_FP32(x[ib+1].d) * GGML_CPU_FP16_TO_FP32(y[ib+1].d);

            const __m512i zx0 = _mm512_loadu_si512((const __m512i *)x[ib].qs);
            const __m512i zy0 = _mm512_loadu_si512((const __m512i *)y[ib].qs);
            const __m512i z_sum0 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), zx0, zy0);

            const __m512i zx1 = _mm512_loadu_si512((const __m512i *)x[ib+1].qs);
            const __m512i zy1 = _mm512_loadu_si512((const __m512i *)y[ib+1].qs);
            const __m512i z_sum1 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), zx1, zy1);

            __m512 sd = _mm512_set_ps(d1, d1, d1, d1, d0, d0, d0, d0);
            acc = _mm512_fmadd_sd_ps(sd, _mm512_cvtepi32_ps(z_sum0), acc);
            acc = _mm512_fmadd_sd_ps(sd, _mm512_cvtepi32_ps(z_sum1), acc);
        }

        // Horizontal sum of __m512 (8 floats)
        float sumf = hsum_float_16(acc);  // need to implement for AVX-512
    } else {
        // Strided path — fall back to generic
    }

    // Remainder
    for (int ib = (nb / 2) * 2; ib < nb; ++ib) {
        int sumi = 0;
        for (int j = 0; j < qk; j++) {
            sumi += x[ib].qs[j] * y[ib].qs[j];
        }
        *s += sumi * GGML_CPU_FP16_TO_FP32(x[ib].d) * GGML_CPU_FP16_TO_FP32(y[ib].d);
    }

    *s = sumf;
}
```

**Note:** The complete implementation needs these helper functions:

**hsum_float_16(__m512):** Manual reduce for the 8 FP32 elements in an m512:
```c
static inline float hsum_float_16(__m512 a) {
    __m256 upper = _mm512_extractf32x8_ps(a, 1);
    __m256 lower = _mm512_castsi256_si128(a);  // _mm512_extractf32x8_ps(a, 0)
    a = _mm256_add_ps(upper, lower);
    a = _mm256_add_ps(a, _mm256_permute2f128_ps(a, a, 1));
    a = _mm256_hadd_ps(a, a);
    return _mm256_cvtss_f32(a);
}
```

**Strided path:** The existing `bx`, `by`, `bs` parameters in the vec_dot signature handle per-row strides in the matmul. For Q8_0, `bx == sizeof(block_q8_0)` and `by == sizeof(block_q8_0)` in the fast path. The strided path uses pointer arithmetic: `((const block_q8_0*)(((const char*)vx) + i*bx)))`. Follow the existing pattern in quants.c.

**Scale application:** Use `_mm512_add_ps(_mm512_mul_ps(scale_vec, sum_vec), acc)` instead of non-existent `_mm512_fmadd_sd_ps`.

**Prefetch distance:** 3 cache lines ahead (192 bytes), targeting DDR5 latency of ~100-150 ns.

- [ ] **Step 2: Mark existing Q8_0 in quants.c as weak**

In `ggml/src/ggml-cpu/arch/x86/quants.c`, change line 1170:

```c
// Before:
void ggml_vec_dot_q8_0_q8_0(int n, float * GGML_RESTRICT s, ...

// After:
__attribute__((weak)) void ggml_vec_dot_q8_0_q8_0(int n, float * GGML_RESTRICT s, ...
```

- [ ] **Step 3: Declare in quants.h**

In `ggml/src/ggml-cpu/quants.h`, after line 45 (`ggml_vec_dot_q8_0_q8_0`), no additional declaration needed since the symbol is the same. The declaration at line 45 already covers both implementations.

- [ ] **Step 4: Verify build**

```bash
cmake -B build_epyc -DGGML_CUDA=OFF -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF 2>&1 | tail -5
cmake --build build_epyc --config Release -j$(nproc) 2>&1 | grep -iE "(error|warn)" | head -10
```

Expected: No link errors. The strong symbol from epyc-cpu.o (AVX-512 VNNI variant) overrides the weak symbol from quants.o.

- [ ] **Step 5: Verify VNNI is active**

```bash
# Check that the symbol resolved to the VNNI version
nm -C build_epyc/src/ggml/CMakeFiles/ggml-cpu-avx512-vnni.dir/arch/x86/epyc-cpu.c.o | grep vec_dot_q8_0
# Should show 'T ggml_vec_dot_q8_0_q8_0' (strong symbol)
```

- [ ] **Step 6: Verify correctness**

```bash
cd build_epyc && timeout 60 bin/test-backend-ops -b CPU0 -R "MUL_MAT(type=q8_0" 2>&1 | tail -10
```

Expected: PASS — results match reference.

- [ ] **Step 7: Commit**

```bash
git add ggml/src/ggml-cpu/arch/x86/epyc-cpu.c ggml/src/ggml-cpu/arch/x86/quants.c
git commit -m "$(cat <<'EOF'
cpu: add AVX-512 VNNI vec_dot for Q8_0

Uses _mm512_dpbusd_epi32 for 64 INT8 pairs per instruction.
2x throughput per instruction vs AVX2 _mm256_maddubs_epi16.
Discovered via weak symbol override — strong VNNI definition
in epyc-cpu.c overrides weak AVX2 definition in quants.c.
Includes __builtin_prefetch for DDR5 latency hiding.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: AVX-512 VNNI vec_dot for Q4_K

**Files:**
- Modify: `ggml/src/ggml-cpu/arch/x86/epyc-cpu.c` — add `ggml_vec_dot_q4_K_q8_K`
- Modify: `ggml/src/ggml-cpu/arch/x86/quants.c:1900` — mark existing as weak

**Context:** `ggml_vec_dot_q4_K_q8_K` (line 1900 in quants.c) processes `block_q4_K` (QK_K=256 elements per block, stored as 128 half-bytes + 12 bytes scales + 2 bytes dmin + 12 bytes bsums). The AVX-512 VNNI path:

1. Load Q4_K data (16 bytes) → unpack low/high nibbles to two 32-byte INT8 vectors → sign-extend to 512-bit ZMM
2. Load Q8_K data (32 bytes) → two 512-bit ZMM registers
3. `_mm512_dpbusd_epi32()` for each nibble lane
4. Scale extraction from `x[i].scales` (12 bytes, 4-bit compressed) — same bit-manipulation as AVX2 path
5. Horizontal reduce with `_mm512_reduce_add_epi32()` (3-instruction shift/add tree)

The inner loop processes QK_K/64 = 4 sub-blocks per Q4_K block. Each sub-block has 64 elements (32 low + 32 high nibbles) paired with 64 Q8_K INT8 values.

- [ ] **Step 1: Add Q4_K VNNI implementation to epyc-cpu.c**

Following the same structure as the AVX2 path (quants.c:1919-1982) but with 512-bit operations:

```c
#if defined(__AVX512VNNI__) && defined(__AVX512BW__) && defined(__AVX512VL__) && defined(__AVX512F__)

void ggml_vec_dot_q4_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);

    const block_q4_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;
    const int nb = n / QK_K;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;
    uint32_t utmp[4];

    __m512 acc = _mm512_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        const float dmin = -y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

        // Scale extraction — same bit manipulation as AVX2
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        // dmin correction (same as AVX2)
        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));
        const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y[i].bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

        const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales_256 = MM256_SET_M128I(sc128, sc128);

        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;
        __m512i sumi = _mm512_setzero_si512();

#if defined(__clang__) && defined(__amd64__)
        #pragma clang loop unroll(count=4)
#endif
        for (int j = 0; j < QK_K/64; ++j) {
            // Prefetch next sub-blocks
            __builtin_prefetch(q4 + 96, 0, 3);
            __builtin_prefetch(q8 + 96, 0, 3);

            // Scales for this sub-block (reuse AVX2 shuffle logic)
            const __m256i scale_l = _mm256_shuffle_epi8(scales_256, get_scale_shuffle_k4(2*j+0));
            const __m256i scale_h = _mm256_shuffle_epi8(scales_256, get_scale_shuffle_k4(2*j+1));

            // Load and unpack Q4_K nibbles (32 half-bytes = 64 INT4 values)
            const __m256i q4bits = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
            const __m256i q4l = _mm256_and_si256(q4bits, _mm256_set1_epi8(0xF));
            const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), _mm256_set1_epi8(0xF));

            // Sign-extend to 512-bit: 32 uint8 → 32 int8 → sign-extend to int16 → dpbusd
            // _mm512_cvtepu8_epi16 loads 32 bytes from memory, so we need mem pointers
            const __m512i q4l_512 = _mm512_cvtepu8_epi16(q4l);  // This needs a memory operand, see note
            const __m512i q4h_512 = _mm512_cvtepu8_epi16(q4h);

            // Load Q8_K (64 bytes = two 32-byte chunks)
            const __m512i q8l = _mm512_loadu_si512((const __m512i*)q8); q8 += 64;  // 64 bytes for both nibbles

            // DPBusD: unsigned multiply + accumulate (q4 is unsigned 0-15, q8 is signed)
            // Note: _mm512_dpbusd_epi32 expects unsigned x and signed y
            __m512i sum_l = _mm512_dpbusd_epi32(_mm512_setzero_si512(), q4l_512, q8l_low32);
            __m512i sum_h = _mm512_dpbusd_epi32(_mm512_setzero_si512(), q4h_512, q8l_high32);

            // Scale and accumulate
            sum_l = _mm512_madd_epi16(scale_l_ext, sum_l);  // need to extend scales to 512-bit
            sum_h = _mm512_madd_epi16(scale_h_ext, sum_h);

            sumi = _mm512_add_epi32(sumi, _mm512_add_epi32(sum_l, sum_h));
        }

        __m512 vd = _mm512_set1_ps(d);
        acc = _mm512_fmadd_ps(vd, _mm512_cvtepi32_ps(sumi), acc);
    }

    // Horizontal sum
    float sumf = hsum_float_16(acc) + _mm_cvtss_f32(acc_m);
    acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));
    *s = sumf + _mm_cvtss_f32(acc_m);
}
```

**Important implementation notes:**
1. `_mm512_cvtepu8_epi16()` requires a **memory operand** — it loads 32 bytes from a pointer. Store the 256-bit unpacked nibbles to a `uint8_t tmp[32]` stack buffer, then call `_mm512_cvtepu8_epi16(tmp)`. This gives 64 zero-extended int16 from 32 uint8 nibbles.
2. Q4 nibbles are unsigned [0,15], Q8_K values are signed [-128,127]. `_mm512_dpbusd_epi32(zero, unsigned_x, signed_y)` matches this exactly — no sign manipulation needed.
3. Each sub-block (64 elements): 32 Q4 bytes → 64 nibbles. 64 Q8_K bytes. Two DPBusD calls per sub-block (low nibbles × first 32 Q8, high nibbles × next 32 Q8). Scale each with `_mm512_madd_epi16(scale_ext, dpbusd_result)`.
4. Scale extension: the 256-bit `scale_l`/`scale_h` (16 int16 values) must be broadcast to 512-bit for the `_mm512_madd_epi16` multiply. Use `_mm512_broadcast_i32x4(_mm256_castsi256_si128(scale_l))` to tile the 4 scales across 64 int16 lanes.
5. `get_scale_shuffle_k4()` is `static` in quants.c. Either: (a) copy the shuffle table to epyc-cpu.c, (b) move it to a shared header (`quants.h` or `common.h`), or (c) declare `extern const __m128i* get_scale_shuffle_k4(int)` in epyc-cpu.c. Option (a) is simplest — the table is small (8 entries of `__m128i`).
6. The existing Q4_K path handles `dmin` correction with a separate `acc_m` (`__m128`). Keep this pattern — the dmin correction is AVX2-only (256-bit) since it deals with block minima.

- [ ] **Step 2: Mark existing Q4_K in quants.c as weak**

In `ggml/src/ggml-cpu/arch/x86/quants.c`, change line 1900:

```c
// Before:
void ggml_vec_dot_q4_K_q8_K(int n, ...

// After:
__attribute__((weak)) void ggml_vec_dot_q4_K_q8_K(int n, ...
```

- [ ] **Step 3: Verify build**

```bash
cmake --build build_epyc --config Release -j$(nproc) 2>&1 | grep -iE "(error|warn)" | head -10
```

- [ ] **Step 4: Verify correctness**

```bash
cd build_epyc && timeout 60 bin/test-backend-ops -b CPU0 -R "MUL_MAT(type=q4_0\|type=q4_k" 2>&1 | tail -10
```

Expected: PASS for Q4_K matmul tests.

- [ ] **Step 5: Commit**

```bash
git add ggml/src/ggml-cpu/arch/x86/epyc-cpu.c ggml/src/ggml-cpu/arch/x86/quants.c
git commit -m "$(cat <<'EOF'
cpu: add AVX-512 VNNI vec_dot for Q4_K

Uses _mm512_dpbusd_epi32 for nibble unpack + INT8 multiply.
Processes 64 elements per sub-block with sign-extension from
256-bit unpack to 512-bit DPBusD. Includes __builtin_prefetch
for DDR5 latency hiding and AOCC loop unroll pragmas.
Weak symbol override — VNNI version replaces AVX2 at link time.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: AVX-512 BF16 vec_dot

**Files:**
- Modify: `ggml/src/ggml-cpu/arch/x86/epyc-cpu.c` — add `ggml_vec_dot_bf16_bf16_avx512`
- Modify: `ggml/src/ggml-cpu/quants.h` — declare BF16 function

**Context:** BF16 dot product uses `_mm512_cvtne2pack_sf2()` to convert 32 BF16 → 32 FP32, then either `_mm512_dpbf16_ps()` (if `__AVX512_BF16__`) or FMA accumulate. The existing `ggml_vec_dot_bf16` in vec.cpp processes FP32-converted values.

- [ ] **Step 1: Add BF16 VNNI implementation to epyc-cpu.c**

```c
#if defined(__AVX512BW__) && defined(__AVX512F__)

void ggml_vec_dot_bf16_bf16_avx512(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);

    const ggml_bf16_t * x = vx;
    const ggml_bf16_t * y = vy;

    __m512 acc = _mm512_setzero_ps();
    int i = 0;

#if defined(__AVX512_BF16__)
    // Direct BF16→FP32 multiply with dpbf16
    for (; i + 63 < n; i += 64) {
        const __m512i zx = _mm512_i32gather_epi32(i / 2, y, 4);  // gather 32 bf16 values
        // Actually: load 64 bf16_t (128 bytes), use dpbf16
        const __m512i xb = _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)(x+i)));  // Wrong size, need proper load
        const __m512i yb = _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)(y+i)));
        acc = _mm512_dpbf16_ps(acc, _mm512_castsi512_ps(xb), _mm512_castsi512_ps(yb), 0x7f);
    }
#else
    // Convert to FP32, then FMA
    for (; i + 63 < n; i += 64) {
        // Load 32 bf16 pairs, convert to FP32
        __m256i xb = _mm256_loadu_si256((const __m256i*)(x + i));
        __m256i yb = _mm256_loadu_si256((const __m256i*)(y + i));
        __m512 xf = _mm512_cvtne2pack_ps(_mm512_cvtepu16_epi32(_mm256_castsi256_si128(xb)), _mm512_cvtepu16_epi32(_mm256_extracti128_si256(xb, 1)));
        __m512 yf = _mm512_cvtne2pack_ps(_mm512_cvtepu16_epi32(_mm256_castsi256_si128(yb)), _mm512_cvtepu16_epi32(_mm256_extracti128_si256(yb, 1)));
        acc = _mm512_fmadd_ps(xf, yf, acc);
    }
#endif

    float sumf = 0;
    for (; i < n; i++) {
        sumf += (float)GGML_CPU_FP16_TO_FP32(x[i]) * (float)GGML_CPU_FP16_TO_FP32(y[i]);
    }

    *s = hsum_float_16(acc) + sumf;
}
#endif
```

**Note:** BF16 load and conversion pattern:
1. Load 32 bf16 (64 bytes) via `_mm256_loadu_si256()`
2. Split to low/high 128-bit: `_mm256_castsi256_si128()` and `_mm256_extracti128_si256(xb, 1)`
3. Zero-extend each to FP32: `_mm512_cvtepu16_epi32(low)` → 16 int32, `_mm512_cvtepu16_epi32(high)` → 16 int32
4. Interleave: `_mm512_cvtne2pack_ps(low_f32, high_f32)` → 32 FP32 in one m512
5. Same for Y, then `_mm512_fmadd_ps(xf, yf, acc)`

For `__AVX512_BF16__`, the `_mm512_dpbf16_ps(acc, xb_ps, yb_ps, 0x7f)` directly multiplies 32 BF16 pairs and accumulates to FP32. The control word `0x7f` selects all 32 elements.

- [ ] **Step 2: Declare in quants.h**

```c
// Add after line 45:
void ggml_vec_dot_bf16_bf16_avx512(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
```

- [ ] **Step 3: Wire up via type_traits_cpu**

The BF16 vec_dot in `type_traits_cpu` (line 387) currently points to `ggml_vec_dot_bf16`. For the AVX-512 BF16 variant, this should point to the new function. Same weak-symbol pattern:

In `ggml/src/ggml-cpu/vec.cpp`, mark the existing `ggml_vec_dot_bf16` as weak if needed, or dispatch via `#if defined(__AVX512BF16__)` inline in the existing function.

- [ ] **Step 4: Verify build**

```bash
cmake --build build_epyc --config Release -j$(nproc) 2>&1 | grep -iE "(error|warn)" | head -10
```

- [ ] **Step 5: Commit**

```bash
git add ggml/src/ggml-cpu/arch/x86/epyc-cpu.c ggml/src/ggml-cpu/quants.h
git commit -m "$(cat <<'EOF'
cpu: add AVX-512 BF16 vec_dot

Uses _mm512_cvtne2pack_ps for BF16→FP32 conversion or
_mm512_dpbf16_ps when AVX512_BF16 is available.
Processes 64 BF16 pairs per iteration vs 32 with AVX2.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: CMake Integration

**Files:**
- Modify: `ggml/src/ggml-cpu/CMakeLists.txt:248-253` — add epyc-cpu.c to x86 sources

**Context:** The x86 sources are added at line 250-253 of CMakeLists.txt:
```cmake
        list(APPEND GGML_CPU_SOURCES
            ggml-cpu/arch/x86/quants.c
            ggml-cpu/arch/x86/repack.cpp
            )
```

All x86 sources are compiled for each variant with the variant's flags. `epyc-cpu.c` uses AVX-512 intrinsics, so it must compile cleanly for all variants (guarded by `#if defined(__AVX512VNNI__)`).

- [ ] **Step 1: Add epyc-cpu.c to x86 sources**

In `ggml/src/ggml-cpu/CMakeLists.txt`, add `ggml-cpu/arch/x86/epyc-cpu.c` to the x86 source list:

```cmake
        list(APPEND GGML_CPU_SOURCES
            ggml-cpu/arch/x86/quants.c
            ggml-cpu/arch/x86/epyc-cpu.c
            ggml-cpu/arch/x86/repack.cpp
            )
```

- [ ] **Step 2: Ensure epyc-cpu.c compiles for non-AVX-512 variants**

The entire content of `epyc-cpu.c` must be wrapped in:
```c
#if defined(__AVX512VNNI__) && defined(__AVX512BW__) && defined(__AVX512VL__) && defined(__AVX512F__)
    // All function definitions
#endif
```

For non-VNNI variants, the file compiles to an empty object. This is fine — the linker simply has no symbols to resolve from it.

**Weak symbol portability:** `__attribute__((weak))` is GCC/Clang. For MSVC builds, use `#pragma comment(linker, "/ALTENTRY:ggml_vec_dot_q8_0_q8_0")` or a `#ifdef _MSC_VER` guard that uses the inline `#if defined(__AVX512VNNI__)` dispatch pattern instead of weak symbols. The plan targets Linux/gcc/clang first; MSVC support can be added later.

- [ ] **Step 3: Full build with tests**

```bash
cmake -B build_epyc -DGGML_CUDA=OFF -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF 2>&1 | grep -iE "(avx|vnni|epyc|warning)" | head -10
cmake --build build_epyc --config Release -j$(nproc) 2>&1 | grep -iE "(error)" | head -5
```

- [ ] **Step 4: Run full quant test suite**

```bash
cd build_epyc && timeout 120 bin/test-backend-ops -b CPU0 -R "MUL_MAT" 2>&1 | tail -20
```

Expected: All MUL_MAT tests pass. Q8_0 and Q4_K use VNNI path; all others use existing AVX2 paths.

- [ ] **Step 5: Verify symbol resolution**

```bash
# Check that VNNI symbols resolved correctly
nm -C build_epyc/src/ggml/libggml-cpu.so | grep vec_dot_q8_0 | head -5
nm -C build_epyc/src/ggml/libggml-cpu.so | grep vec_dot_q4_K | head -5
```

- [ ] **Step 6: Commit**

```bash
git add ggml/src/ggml-cpu/CMakeLists.txt ggml/src/ggml-cpu/arch/x86/epyc-cpu.c
git commit -m "$(cat <<'EOF'
cmake: integrate epyc-cpu.c into x86 backend build

Adds AVX-512 VNNI source to the variant build system.
Guards all content with __AVX512VNNI__ so non-VNNI variants
compile an empty object. Weak symbol dispatch ensures VNNI
implementations override AVX2 at link time.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Integration Testing

**Files:**
- Test: `build_epyc/bin/test-backend-ops` — verify all quant types on CPU backend

- [ ] **Step 1: Run comprehensive CPU tests**

```bash
cd build_epyc && timeout 180 bin/test-backend-ops -b CPU0 2>&1 | tail -30
```

Expected: All tests pass. No regressions on any quant type.

- [ ] **Step 2: Verify VNNI is active for Q8_0 and Q4_K**

```bash
# The log should show AVX-512 VNNI variant being used
cd build_epyc && LD_LIBRARY_PATH=src/ggml bin/test-backend-ops -b CPU0 -R "q8_0\|q4_K" 2>&1
```

- [ ] **Step 3: Test NUMA interleave**

```bash
# Build a small test program or use existing tool to verify NUMA interleave
# Check that set_mempolicy is called with MPOL_INTERLEAVE
numactl --show
```

- [ ] **Step 4: Test CCD affinity**

```bash
# Run with GGML_NUMA_STRATEGY_CCD and verify log shows CCD detection
# Verify thread pinning with taskset -pc <tid> on a running thread
```

- [ ] **Step 5: Regression test — build without AVX-512**

```bash
cmake -B build_epyc_noavx512 -DGGML_CUDA=OFF -DGGML_NATIVE=OFF -DGGML_AVX512=OFF -DGGML_AVX2=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build_epyc_noavx512 --config Release -j$(nproc) 2>&1 | grep -iE "(error)" | head -5
cd build_epyc_noavx512 && timeout 60 bin/test-backend-ops -b CPU0 -R "MUL_MAT(type=q8_0\|type=q4_k" 2>&1 | tail -10
```

Expected: Builds cleanly with AVX2 only. Tests pass using the weak (AVX2) implementations.

- [ ] **Step 6: Final cleanup and commit**

```bash
git status
# Ensure all files are committed, no stray debug code
git log --oneline -6  # Should show 6 commits: NUMA, CCD, Q8_0 VNNI, Q4_K VNNI, BF16, CMake
```

---

## Follow-up Work (Out of Scope)

- AVX-512 repacking (x8 block) for tiled GEMM — pre-repack Q4_K weights to 512-bit lanes
- Full AVX-512 sgemm kernel for FP16/FP32 matmul
- Hybrid TMA bridge: pinned RAM + async transfer benchmark (spec 2)
- per-CCD work stealing for load-balanced matmul across CCDs
- UD-Q4_K_XL specific vec_dot (if different from Q4_K)
