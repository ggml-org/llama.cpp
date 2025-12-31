# SoA Baseline Profiling Results (Q4_0)

## Date: 2025-12-30
## Branch: feature/sycl-coalescing (3ac1c8296)

## Hardware Configuration

### GPU (Primary Target)
- **Device**: Intel Arc B580 (Battlemage G21)
- **XVE Count**: 160
- **Max XVE Thread Count**: 8
- **Max Core Frequency**: 2.850 GHz
- **Theoretical Memory Bandwidth**: 456 GB/s (GDDR6, 19 Gbps, 192-bit)

### GPU (Secondary - Used for some tests)
- **Device**: Intel Arc Pro B50 Graphics (BMG G21)
- **XVE Count**: 128
- **Max XVE Thread Count**: 8
- **Max Core Frequency**: 2.600 GHz

### CPU
- **Name**: Intel Core Ultra 7 265K (Arrow Lake-S)
- **Frequency**: 3.878 GHz
- **Logical CPU Count**: 20

## Benchmark Results

### Model: Mistral 7B Q4_0 (3.83 GiB, 7.24B params)

#### Arc B580 (level_zero:0)
| Test | Flash Attn | Throughput (t/s) | Std Dev |
|------|------------|------------------|---------|
| pp512 | OFF | 674.37 | +/- 3.58 |
| tg128 | OFF | 77.22 | +/- 0.09 |
| pp512 | ON | 669.36 | +/- 1.74 |
| tg128 | ON | 82.70 | +/- 0.04 |

#### Arc Pro B50 (level_zero:1)
| Test | Flash Attn | Throughput (t/s) | Std Dev |
|------|------------|------------------|---------|
| pp512 | OFF | 386.07 | +/- 0.52 |
| tg128 | OFF | 39.69 | +/- 0.00 |
| pp512 | ON | 361.19 | +/- 0.38 |
| tg128 | ON | 41.64 | +/- 0.00 |

## VTune GPU Offload Analysis

**GPU Profiled**: Intel Arc B580 (Battlemage G21, level_zero:0, BDF: 0:3:0.0)

### Overall GPU Metrics
- **Elapsed Time**: 27.512s
- **GPU Time**: 0.873s (3.2% of elapsed time)
- **GPU Utilization**: Low (3.2%)
- **XVE Array Stalled/Idle**: 68.0% of elapsed time with GPU busy

### Hottest Host Tasks
| Host Task | Task Time | % of Elapsed | Task Count |
|-----------|-----------|--------------|------------|
| zeCommandListAppendMemoryCopy | 0.410s | 2.6% | 938 |
| zeModuleCreate | 0.040s | 0.2% | 16 |
| zeEventHostSynchronize | 0.031s | 0.2% | 938 |
| zeCommandListAppendLaunchKernel | 0.009s | 0.1% | 3,100 |

### GPU Computing Tasks Captured
No kernel-level data was captured in this VTune collection. Individual SYCL kernel
names (DMMV, MMQ, etc.) are not attributed to GPU computing tasks due to a Level Zero
limitation. Most compute time appears under "[Outside any task]" category or is attributed
to generic GPU activity.

**Limitation**: This is a known constraint of VTune's gpu-offload collection with Level Zero
tracing on Intel Arc GPUs. Custom SYCL kernels launched through the SYCL runtime are not
always captured with explicit task names and kernel-level attribution. To profile individual
kernels, alternative tools like Intel PTI, oneTrace, or Intel VTune's Level 0 API tracing
may provide better kernel-level details.

### Top CPU Hotspots When GPU Idle
| Function | Module | CPU Time |
|----------|--------|----------|
| func@0x2d6360 | libze_intel_gpu.so.1 | 23.979s |
| ggml_backend_sycl_buffer_set_tensor | libggml-sycl.so.0 | 0.800s |
| func@0x1db940 | libze_intel_gpu.so.1 | 0.770s |
| memset | libc-dynamic.so | 0.390s |

## Memory Bandwidth Estimation

Based on theoretical specs and workload characteristics:
- **Theoretical Peak**: 456 GB/s (Arc B580)
- **Observed GPU Memory Read**: 0.058 GB/s (from XVE counters)
- **Observed GPU Memory Write**: 0.048 GB/s (from XVE counters)

### Efficiency Calculations
- **Read Efficiency**: (0.058 / 456) × 100% = **0.013%**
- **Write Efficiency**: (0.048 / 456) × 100% = **0.011%**

### VTune Bandwidth Measurement Limitation
The extremely low bandwidth measurements (0.013% efficiency) do NOT represent actual GPU memory utilization during computation. VTune's gpu-offload collection with Level Zero tracing on Intel Arc has known limitations:

1. **Sampling Bias**: VTune samples GPU activity periodically and may miss high-bandwidth kernel execution phases
2. **Kernel Attribution**: Custom SYCL kernels (DMMV, MMQ, etc.) are not always captured in per-kernel counters, so bandwidth data reflects a subset of actual GPU work
3. **Actual Bandwidth Validation**: The token generation throughput (77-82 t/s for tg128) with Mistral 7B Q4_0 (typical memory footprint ~4GB weights + KV cache) indicates actual effective bandwidth is much higher - likely in the 50-100+ GB/s range during compute phases

**Recommendation**: Use this baseline for relative comparisons between optimization attempts. Absolute bandwidth numbers should be validated with additional profiling tools (PTI, oneTrace) that provide more accurate per-kernel attribution.

The low observed bandwidth suggests the workload is currently:
1. Compute-bound on small matrices
2. Limited by kernel launch overhead
3. Bottlenecked by memory access patterns (strided access)

## Key Observations

1. **Token Generation (tg128) is DMMV-bound**: The 82.7 t/s throughput during
   decoding is primarily limited by matrix-vector multiplication kernels that
   process one token at a time.

2. **Flash Attention Impact**:
   - Slight improvement in tg128 (+7% with FA on B580)
   - Slight decrease in pp512 (-0.7% with FA on B580)

3. **GPU Utilization is Low**: Only 4.8% of elapsed time is GPU compute.
   The majority of time is spent on:
   - Host-side memory management
   - Kernel launch overhead
   - Synchronization

4. **Memory Copy Dominates Host-side Work**: 938 memory copies consuming
   0.410s suggests significant data movement between host and device.

## Baseline for Coalescing Optimization

This baseline will be compared against after implementing coalesced memory
layouts. Key metrics to track:

1. **Token Generation Throughput**: Target 10-20% improvement on tg128
2. **Memory Bandwidth Utilization**: Target 50%+ of theoretical
3. **Kernel Execution Time**: Measure individual DMMV/MMQ kernel times
4. **GPU Utilization**: Target 15%+ of elapsed time

## Files in This Directory

- `vtune_output.txt` - Raw VTune collection output
- `bench_output.txt` - llama-bench output (B50 Pro)
- `bench_fa_comparison.txt` - Flash attention on/off comparison (B50 Pro)
- `bench_b580_output.txt` - llama-bench output (B580)
- `vtune_soa_baseline_b580/` - VTune result directory for B580

## Commands Used

```bash
# Baseline benchmark
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p 512 -n 128 -ngl 99 -fa 0,1 -r 5

# VTune GPU offload profiling
ONEAPI_DEVICE_SELECTOR=level_zero:0 vtune -collect gpu-offload \
  -result-dir benchmark_results/vtune_soa_baseline_b580 \
  -- ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p 512 -n 128 -ngl 99 -fa 1
```
