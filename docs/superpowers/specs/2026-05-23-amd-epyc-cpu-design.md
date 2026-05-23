# AMD EPYC CPU Optimization for ggml-cpu

> **Part 1 of 2**: CPU-only optimizations. Part 2 covers the hybrid TMA-to-RAM bridge.

**Goal:** Saturate the 460 GB/s bandwidth of 12-channel DDR5 on AMD EPYC 9V74 and utilize AVX-512 VNNI for Q4_K/Q8_0 quantized matmul, targeting 20-30 t/s on a 600GB Kimi K2.6 UD-Q4_K_XL model (baseline: 8-11 t/s).

**Architecture:** "EPYC Direct Path" — new `arch/x86/epyc-cpu.c` with AVX-512 VNNI vec_dot functions, sysfs-based NUMA/CCD topology probing, and runtime dispatch. Follows the Blackwell WGMMA pattern: compile-time `#ifdef` guards + runtime CPU feature checks. All existing AVX2 paths remain as fallback.

**Tech Stack:** Linux sysfs, `set_mempolicy`/`mbind` syscalls, AVX-512 (VNNI, BF16, BW, VL), GCC/Clang intrinsics, CMake

**System Context:**
- AMD EPYC 9V74: 8 CCDs × 10 cores × 2 SMT = 160 threads
- 768 GB DDR5, 12-channel interleaved (~460 GB/s peak bandwidth)
- Each CCD: 32 MB L3 cache, cross-CCD latency >100 ns via UPI
- RTX 5090 (32 GB VRAM): partial offload of ~10-20 layers
- Instruction set: AVX-512F, AVX-512BW, AVX-512VL, AVX-512_VNNI, AVX-512_BF16, AVX-VNNI, FMA, BMI2

---

## 1. NUMA Interleave Allocation

### Problem

`malloc()` is NUMA-blind. On a dual-socket EPYC with multiple NUMA domains, a 600GB model loaded via standard allocation will be placed on a single node's memory channels, leaving the remaining channels idle. The 12-channel DDR5 interleave is only achieved when allocations span all nodes.

### Design

Add `GGML_NUMA_STRATEGY_INTERLEAVE = 5` to `enum ggml_numa_strategy` in `ggml-cpu.h`.

`ggml_cpu_init_numa()` (called from `ggml_backend_cpu_init()`) performs:
1. Probes `/sys/devices/system/node/node*` to enumerate NUMA nodes
2. Builds a `nodemask_t` from the discovered nodes
3. Calls `set_mempolicy(MPOL_INTERLEAVE, &nodemask, 0)` — all subsequent `malloc()`/`aligned_alloc()` calls in the process are interleaved across all nodes
4. Logs: `GGML_LOG_INFO("NUMA: interleaving across %d nodes\n", n_nodes)`

`ggml_cpu_reset_numa()` (called on backend close) restores `MPOL_DEFAULT`.

**Cross-platform:**
- Linux: `set_mempolicy()` syscall (no libnuma dependency)
- Windows: no interleave policy equivalent; falls back to best-effort allocation across NUMA nodes
- Always graceful: if sysfs is unreadable or `set_mempolicy` fails, log a warning and continue with default malloc

### Files
- `ggml-cpu.h` — add `GGML_NUMA_STRATEGY_INTERLEAVE` to enum
- `ggml-cpu.c` — implement `ggml_cpu_init_numa()` and `ggml_cpu_reset_numa()`

---

## 2. CCD-Aware Thread Affinity

### Problem

The current NUMA DISTRIBUTE strategy (`thread_n % n_nodes`) round-robins threads across NUMA nodes but ignores L3 cache topology. On EPYC, NUMA nodes are large (spanning multiple CCDs). Threads on different CCDs within the same NUMA node still suffer >100 ns cross-CCD latency for L3 misses. Matmul chunks should stay within a single CCD's L3 domain.

Additionally, SMT hyperthreads share execution units but not memory bandwidth. For memory-bound workloads (600GB model), single-thread-per-core should be preferred before enabling SMT.

### Design

Add `GGML_NUMA_STRATEGY_CCD = 6` to `enum ggml_numa_strategy`.

**Topology probe:** At init, read `/sys/devices/system/cpu/cpu*/topology/core_defaults` for each online CPU. The `core_defaults` value is the L3 cache domain ID. Group CPUs by this ID to build CCD topology.

Store in `struct ggml_ccd_topology` (new in `ggml-cpu-impl.h`):
```c
struct ggml_ccd_topology {
    uint32_t n_ccds;
    // Per-CCD: list of hardware thread IDs, sorted to fill physical cores first, then SMT siblings.
    uint32_t ccd_threads[GGML_NUMA_MAX_CPUS];  // flat array of all thread IDs, in CCD-order
    uint32_t ccd_thread_count[GGML_NUMA_MAX_NODES]; // threads per CCD
    uint32_t ccd_for_cpu[GGML_NUMA_MAX_CPUS];   // CPU -> CCD mapping
};
```

**Thread assignment order:**
1. Fill each CCD with one thread per physical core (no SMT). For EPYC 9V74: 80 threads across 8 CCDs (10 per CCD).
2. Only after all physical cores are occupied, add SMT siblings. Threads 81-160 are the second hyperthread per core.

This is achieved by reading `/sys/devices/system/cpu/cpu*/topology/thread_siblings_list` to identify SMT sibling pairs per CCD. For each physical core, the thread with the lower CPU ID is marked primary. All primaries are scheduled first, then siblings.

**Affinity application:** `set_numa_thread_affinity()` gains a new case for `GGML_NUMA_STRATEGY_CCD`. Thread N gets assigned to `ccd_threads[N]` and pinned via `sched_setaffinity()` with a single-CPU mask.

**Cross-platform:**
- Linux: sysfs probe + `sched_setaffinity()`
- Windows: `GetNumaProcessorNodeEx()` + `SetThreadGroupAffinity()` — groups threads by NUMA node (Windows doesn't expose CCD-level topology reliably)
- Fallback: if sysfs `core_defaults` is missing (older kernels), fall back to NUMA DISTRIBUTE

### Files
- `ggml-cpu.h` — add `GGML_NUMA_STRATEGY_CCD` to enum
- `ggml-cpu-impl.h` — add `struct ggml_ccd_topology`
- `ggml-cpu.c` — CCD topology probe in `ggml_numa_init()`, new case in `set_numa_thread_affinity()`

---

## 3. AVX-512 VNNI Direct Path for Quantized Matmul

### Problem

`arch/x86/quants.c` has no AVX-512 path for Q4_K or Q8_0. The Q8_0 dot product uses AVX2 `_mm256_maddubs_epi16` (32 INT8 pairs → 16 INT16 sums → requires a second add to reach INT32). AVX-512 VNNI's `_mm512_dpbusd_epi32` fuses 32 signed-byte multiply-adds directly to INT32 in a single instruction, doubling the throughput per instruction and halving the register pressure.

Q4_K additionally suffers from nibble unpack overhead. AVX-512 VBMI (`_mm512_shuffle_i32x4`, `_mm512_permutexvar_ps`) can parallelize the high/low nibble extract across two 512-bit lanes.

### Design

**New file:** `ggml/src/ggml-cpu/arch/x86/epyc-cpu.c`

Compiled only when `__AVX512VNNI__` is defined. Flags: `-mavx512f -mavx512bw -mavx512vl -mavx512vnni -mavx512bf16`.

#### 3.1 `ggml_vec_dot_q8_0_q8_0_avx512_vnni()`

```
For each 64-byte chunk of X and Y:
  ZX = _mm512_loadu_si512(X)     // 64 INT8 values
  ZY = _mm512_loadu_si512(Y)     // 64 INT8 values
  Z_sum[i] += _mm512_dpbusd_epi32(Z_sum[i], ZX, Y)  // fuse mul+add to INT32
Horizontal reduce:
  _mm512_reduce_add_epi32(Z_sum)  // efficient shift/add tree, 3 vpaddd instructions
Convert to float, multiply by scales, accumulate.
```

Processes 64 INT8 pairs per DPBusD. Inner loop unrolls 4x (4 ZMM accumulators), then horizontal-sums. Scale is extracted per Q8_0 block (32 values + 1 float scale) and applied at block boundaries.

#### 3.2 `ggml_vec_dot_q4_K_q8_K_avx512_vnni()`

Q4_K stores 32 INT4 values per 32-byte block (packed as 16 half-byte data bytes), plus 12 scales (4-bit compressed) and two INT8 minima (m1, m2). The AVX-512 path:

1. Load 512 bits of Q4_K data (16 half-bytes across 2 blocks)
2. Unpack low nibbles: `data & 0x0F`, broadcast to 32 INT8
3. Unpack high nibbles: `(data >> 4) & 0x0F`, broadcast to 32 INT8
4. Load corresponding Q8_K as 512-bit INT8
5. `_mm512_dpbusd_epi32()` for each nibble lane
6. Scale extraction: read `q4.scales` (16 compressed 4-bit values), decompress via lookup table or bit-shift, apply per-4-element groups

Two parallel lanes (low/high nibble) each accumulate in separate ZMM registers. Horizontal reduce at block boundary, then scale and add to result.

#### 3.3 `ggml_vec_dot_bf16_bf16_avx512()`

For BF16 tensors:
1. `_mm512_cvtne2pack_sf2()` — convert 32 BF16 values to 32 FP32 (odd/even interleaved)
2. If `__AVX512_BF16__`: `_mm512_dpbf16_ps()` for direct BF16→FP32 matmul
3. Else: `_mm512_mul_ps()` + FP32 accumulate (FMA)

#### 3.4 AOCC Loop Vectorization and Prefetching

When compiled with AOCC (AMD Optimizing C/C++ Compiler, Clang-based), enable explicit vectorization pragmas to ensure the inner VNNI loops are fully unrolled and vectorized:

```c
#if defined(__clang__) && defined(__amd64__)
    #pragma clang loop vectorize(enable) interleave(count=4)
    #pragma clang loop unroll(count=4)
#endif
    for (int i = 0; i < count; i += 64) {
        // VNNI inner loop
    }
```

DDR5 prefetching: with 12 memory channels and ~460 GB/s bandwidth, the L3 cache fill rate is high, but the Q4_K loops process data sequentially. Use `__builtin_prefetch()` to pull the next 2-3 cache lines ahead of the VNNI consumer:

```c
const int PREFETCH_LINES = 3;
for (int i = 0; i < count; i += 64) {
    for (int p = 0; p < PREFETCH_LINES; ++p) {
        __builtin_prefetch(x8 + (i + p * 192), 0, 3);  // read, high temporal, L3
        __builtin_prefetch(y8 + (i + p * 192), 0, 3);
    }
    // ... VNNI compute ...
}
```

Prefetch distance (192 bytes = 3 × 64-byte cache lines) targets the DDR5 read latency (~100-150 ns). The `3` parameter means L3 cache level.

#### 3.5 Horizontal Summation
```c
z = _mm512_add_epi32(z, _mm512_shuffle_i32x4(z, z, 0x50)); // shift by 16
z = _mm512_add_epi32(z, _mm512_shuffle_i32x4(z, z, 0x08)); // shift by 8
z = _mm512_add_epi32(z, _mm512_shuffle_i32x4(z, z, 0x01)); // shift by 4
```

### Dispatch

Runtime dispatch via `ggml_type_traits_cpu`. At CPU backend init:
1. Check `ggml_cpu_has_avx512_vnni()` (calls CPUID via `arch/x86/cpu-feats.cpp`)
2. If true, patch `type_traits[GGML_TYPE_Q8_0].vec_dot` → `ggml_vec_dot_q8_0_q8_0_avx512_vnni`
3. Patch `type_traits[GGML_TYPE_Q4_K].vec_dot_type` companion similarly
4. Log: `GGML_LOG_INFO("AVX-512 VNNI: enabled for Q4_K/Q8_0\n")`

Follows the existing pattern used for llamafile sgemm dispatch.

### Files
- `ggml/src/ggml-cpu/arch/x86/epyc-cpu.c` — CREATE: VNNI vec_dot implementations
- `ggml/src/ggml-cpu/quants.h` — declare new functions
- `ggml/src/ggml-cpu/ggml-cpu.c` — runtime dispatch in backend init
- `ggml/src/ggml-cpu/arch/x86/CMakeLists.txt` — compile flags for epyc-cpu.c

---

## 4. Integration and Testing

### File Change Summary

| File | Change |
|------|--------|
| `ggml/src/ggml-cpu/arch/x86/epyc-cpu.c` | CREATE: AVX-512 VNNI vec_dot for Q4_K, Q8_0, BF16 (~400 lines) |
| `ggml/src/ggml-cpu/ggml-cpu.c` | MODIFY: NUMA interleave init, CCD probe, runtime vec_dot dispatch |
| `ggml/src/ggml-cpu/ggml-cpu.h` | MODIFY: `GGML_NUMA_STRATEGY_INTERLEAVE`, `GGML_NUMA_STRATEGY_CCD` |
| `ggml/src/ggml-cpu/ggml-cpu-impl.h` | MODIFY: `struct ggml_ccd_topology` |
| `ggml/src/ggml-cpu/arch/x86/CMakeLists.txt` | MODIFY: add epyc-cpu.c with AVX-512 flags |
| `ggml/src/ggml-cpu/quants.h` | MODIFY: declare new vec_dot functions |

### Testing

1. **Numerical correctness:** `test-backend-ops -b CPU0 -R "MUL_MAT(type=q[48]_0"` — compare AVX-512 VNNI output against AVX2 reference
2. **BF16 correctness:** `test-backend-ops -b CPU0 -R "vec_dot_bf16"`
3. **NUMA policy:** `numactl --show` before/after init to verify interleave policy
4. **CCD affinity:** `taskset -pc <tid>` on running threads to verify CCD-local pinning
5. **Regression on non-EPYC:** Build without AVX-512, run full CPU test suite — should hit only AVX2 paths
6. **Benchmark:** Kimi 600GB UD-Q4_K_XL on EPYC 9V74, target 20-30 t/s

### Zero Regression Guarantees

- All EPYC paths gated by `#ifdef __AVX512VNNI__` at compile time + CPUID check at runtime
- `#ifdef __linux__` guards all sysfs/NUMA code
- Intel/ARM/Legacy AMD: no code path change, AVX2 quants unchanged
- NUMA interleave and CCD affinity only activate when explicitly set via `ggml_numa_init()`
- Default behavior (`GGML_NUMA_STRATEGY_DISABLED`) is untouched

---

## 5. Follow-up Work (Out of Scope)

- AVX-512 repacking (x8 block) for tiled GEMM — pre-repack Q4_K weights to align with 512-bit lanes
- Full AVX-512 sgemm kernel (llamafile-style) for FP16/FP32 matmul
- Hybrid TMA bridge (spec 2): pinned RAM allocator + async transfer benchmark
- per-CCD work stealing for load-balanced matmul across CCDs
