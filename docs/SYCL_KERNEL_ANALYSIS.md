# SYCL Kernel Performance Analysis

Analysis of benchmark results from the XMX-Optimized Unified Kernel Architecture epic (llama.cpp-a3t).

**Analysis Date:** 2026-01-23
**Hardware:** Intel Arc B580 (level_zero:0/1)
**Benchmark Tool:** `tools/sycl-kernel-bench`

## Executive Summary

| Regime | Batch | Best Kernel Type | Key Finding |
|--------|-------|------------------|-------------|
| Memory-bound | 1-4 | MMVQ Coalesced/SoA | Layout matters more than algorithm |
| Transitional | 8-32 | XMX Small Tile | ESIMD variants competitive |
| Compute-bound | 64+ | XMX Large Tile / MMQ | Quant-dependent (Q6_K prefers MMQ) |

### Key Recommendations

1. **Q4_0/Q8_0**: Use `mmvq_coalesced` for batch≤4, XMX tiles for larger batches
2. **Q6_K**: Use `mmvq_soa_baseline` for batch≤4, switch to MMQ at batch≥16
3. **K-quants (Q2_K-Q5_K)**: SoA layout preferred; XMX double-buffer for batch≥8
4. **Layout conversion**: Always beneficial for memory-bound regime; use at load time

## Detailed Results by Regime

### Memory-Bound Regime (batch=1-4)

This regime is dominated by memory bandwidth. The key optimization is memory access pattern, not compute.

| Quant | Winner (batch=1) | Throughput (TPS) | Bandwidth (GB/s) |
|-------|------------------|------------------|------------------|
| Q4_0 | mmvq_coalesced | 53,124 | 502 |
| Q8_0 | mmvq_coalesced | 19,867 | 355 |
| Q6_K | mmvq_soa_baseline | 9,732 | 134 |
| Q4_K | mmvq_soa_baseline | 21,853 | 207 |
| Q5_K | mmvq_aos_baseline | 12,439 | 144 |

**Observations:**
- Coalesced layout provides 5-8x improvement over AoS for block-32 quants (Q4_0, Q8_0)
- SoA layout provides 2-3x improvement for K-quants
- Bandwidth utilization: 12-13% of peak (567 GB/s) - hardware latency limited
- XVE stalled/idle: 75-90% across all configs (memory latency bound)

### Transitional Regime (batch=8-64)

Mixed compute/memory bound. XMX kernels become competitive but not dominant.

| Quant | Winner (batch=8) | Winner (batch=16) | Winner (batch=32) |
|-------|------------------|-------------------|-------------------|
| Q4_0 | mmvq_esimd_v2 | mmvq_esimd_v2 | mmvq_esimd_v2 |
| Q8_0 | mmvq_xmx_tile_soa | mmvq_xmx_tile_soa | mmvq_xmx_cf2_soa |
| Q6_K | mmvq_coalesced | mmq_soa | mmq_soa |
| Q4_K | mmvq_xmx_double_buffer | mmvq_xmx_double_buffer | - |

**Observations:**
- ESIMD variants (esimd_v2) are surprisingly competitive for Q4_0
- Q6_K crosses over to MMQ very early (batch≥16) - unique pattern
- XMX tile variants with SoA layout perform best for Q8_0
- Double-buffer pattern wins for K-quants (Q4_K, Q5_K)

### Compute-Bound Regime (batch=64+)

XMX utilization becomes critical. Large tile sizes beneficial.

| Quant | Winner (batch=64) | Winner (batch=128) |
|-------|-------------------|---------------------|
| Q4_0 | mmvq_xmx_cf2_soa | mmq_aos |
| Q8_0 | mmvq_xmx_cf2_soa | mmvq_xmx_register_accum |
| Q6_K | mmq_coalesced | mmq_coalesced |
| Q4_K | mmvq_xmx_tile_64x64 | mmq_aos |

**Observations:**
- MMQ kernels dominate at batch=128 (production prompt processing)
- XMX cf2/cf4 variants with SoA layout competitive at batch=64
- Q6_K exclusively uses MMQ for batch≥16
- Register accumulation pattern wins for Q8_0 at very large batches

## Crossover Analysis

### MMVQ → XMX Transition

No clear single crossover point detected. MMVQ kernels remain competitive through batch=4 for all quants.

| Quant | MMVQ Max | Notes |
|-------|----------|-------|
| Q4_0 | 4 | Coalesced layout critical |
| Q8_0 | 4 | Coalesced layout critical |
| Q6_K | 4 | SoA layout; MMQ at batch≥16 |
| Q4_K | 4 | SoA layout preferred |

### XMX → MMQ Transition

| Quant | MMQ Win Point | Margin |
|-------|---------------|--------|
| Q6_K | batch ≥ 16 | +72.8% |
| Q4_0 | batch ≥ 128 | varies |
| Q4_K | batch ≥ 128 | varies |

## Surprising Findings

1. **Q6_K MMQ Dominance**: Q6_K switches to MMQ kernels very early (batch≥16), much earlier than other quants. This suggests the K-quant 256-block structure benefits from MMQ's batched approach.

2. **ESIMD Competitiveness**: ESIMD variants (esimd_v2) are competitive with XMX for Q4_0 in the transitional regime. This suggests explicit SIMD control can match XMX for some workloads.

3. **XMX Utilization Counters Unreliable**: xmx_util_pct always reports 0.00 even when DPAS instructions are confirmed via IGC VISA dumps (SimdSize=32). Use throughput_tops as the primary metric instead.

4. **B580 Memory Latency Bound**: DPAS bandwidth utilization is limited to 12-13% of peak (567 GB/s) due to XVE stalled/idle 75-90%. This is a hardware characteristic, not amenable to kernel-level optimization.

5. **Layout Matters More Than Algorithm**: In the memory-bound regime, the difference between AoS and Coalesced/SoA layouts (5-8x) exceeds the difference between kernel algorithms.

## Dispatch Thresholds

The analysis generates `ggml/src/ggml-sycl/dispatch_thresholds.hpp` with data-driven values:

```cpp
// Q4_0: MMVQ_MAX_BATCH=4, XMX_SMALL_MAX_BATCH=32, PREFER_COALESCED=true
// Q8_0: MMVQ_MAX_BATCH=4, XMX_SMALL_MAX_BATCH=32, PREFER_COALESCED=true
// Q6_K: MMVQ_MAX_BATCH=4, XMX_SMALL_MAX_BATCH=16, PREFER_COALESCED=false
```

## Recommendations

### For Production Integration

1. **Batch=1 (Token Generation)**:
   - Use DMMV kernel (existing fast path)
   - Layout conversion at load time (not runtime)

2. **Batch=2-4 (Small Batch)**:
   - Q4_0/Q8_0: `mmvq_coalesced` with COALESCED layout
   - K-quants: `mmvq_soa_baseline` with SOA layout
   - Enable layout conversion by default

3. **Batch=8-32 (Transitional)**:
   - Q4_0: `mmvq_esimd_v2` or `mmvq_xmx_tile_soa`
   - Q8_0: `mmvq_xmx_tile_soa`, `mmvq_xmx_cf2_soa`
   - Q6_K: Switch to MMQ at batch≥16
   - K-quants: `mmvq_xmx_double_buffer`

4. **Batch=64+ (Prompt Processing)**:
   - Use MMQ kernels for all quants
   - Q6_K especially benefits from MMQ

### Future Optimization Opportunities

1. **Reduce XVE Stall/Idle**: Current 75-90% stall rate indicates room for improvement in memory latency hiding, though this may require hardware changes.

2. **Hybrid Kernels**: Consider fusing dequant + DPAS to increase arithmetic intensity.

3. **Persistent Threads**: For very large batches, persistent thread patterns may improve XMX utilization.

4. **Device-Specific Tuning**: B580 vs B50 may have different optimal configurations.

## Data Files

| File | Description |
|------|-------------|
| `tools/sycl-kernel-bench/analysis/aggregated.csv` | Consolidated benchmark data (1245 results) |
| `tools/sycl-kernel-bench/analysis/crossovers.json` | Crossover points and winners per (quant, batch) |
| `tools/sycl-kernel-bench/analysis/thresholds.json` | Dispatch threshold values |
| `ggml/src/ggml-sycl/dispatch_thresholds.hpp` | C++ header with thresholds |

## Benchmark Source Files

- Tier 0: `benchmark_results/tier0-2026-01-21/`
- Tier 1: `benchmark_results/tier1-2026-01-22/`
- Tier 2: `benchmark_results/tier2-2026-01-21/`
- Tier 3: `benchmark_results/tier3-2026-01-22/`
- Tier 4: `benchmark_results/tier4-2026-01-22/`
- DPAS Exploration: `benchmark_results/li6-2026-01-22/`, `benchmark_results/a3t5-2026-01-*/`
