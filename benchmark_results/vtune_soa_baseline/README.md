# SoA Baseline (Q4_0)

**Date:** 2025-12-31
**Model:** Mistral 7B Q4_0 (3.83 GiB)
**GPU:** Intel Arc B580 (GPU 1 via ONEAPI_DEVICE_SELECTOR=level_zero:1)

## Benchmark Results

| Test   | Speed (t/s) |
|--------|-------------|
| pp512  | 363.44      |
| tg128  | 41.75       |

## VTune Summary

**Note:** VTune collected metrics on GPU 0 (iGPU) not GPU 1 (Arc B580).
This is a known limitation with multi-GPU systems and VTune's GPU collection.

### GPU 0 Metrics (for reference only)
- GPU Time: 2.869s
- XVE Array Stalled/Idle: 67.6% of elapsed time with GPU busy
- GPU L3 Bandwidth Bound: 4.5% of peak value
- Occupancy: 67.0% of peak value

### GPU 1 (Arc B580) - Target Device
- XVE Count: 160
- Max XVE Thread Count: 8
- Max Core Frequency: 2.850 GHz
- Theoretical Memory Bandwidth: 456 GB/s (GDDR6 19Gbps x 192-bit)

## Key Observations

1. **Low GPU Utilization (10.5%)**: Significant host-side bottlenecks
2. **XVE Stall Rate (67.6%)**: Memory access patterns causing stalls
3. **L3 Bandwidth Utilization (4.5%)**: Very low, indicating memory access inefficiency

## Target Improvements with Coalesced Layout

- Reduce XVE stall rate through coalesced memory access
- Increase L3 bandwidth utilization toward 50%+
- Improve tg128 throughput (current: 41.75 t/s)
