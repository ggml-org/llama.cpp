# SYCL Kernel Bench Test Matrix

This matrix defines the canonical benchmark configurations for MMVQ/MMQ kernels, plus DPAS device-opt sanity checks. It is designed to cover standard model shapes, MoE expert shapes, batch-size regimes, and edge cases while keeping the total runnable configs between 800–1300.

## How to validate

```
python3 tools/sycl-kernel-bench/validate_matrix.py --summary
```

## Dimension sets

| Set | Dimensions | Notes |
| --- | --- | --- |
| standard | 4096, 5120, 6144, 8192, 14336 | Common hidden/FFN dims |
| transitional | 4096, 8192 | Transitional batch regime focus |
| moe | 2048, 4096, 8192, 16384 | Expert dims (MoE) |
| coalesced_q6k | 8192, 16384 | Q6_K coalesced needs k/256 % 32 == 0 |
| memory_compare | 4096, 8192 | Representative sizes for memory-mode comparisons |

## Batch sets

| Set | Batches | Regime coverage |
| --- | --- | --- |
| core | 1, 4, 8, 16, 32, 64, 128, 256, 512 | memory → transitional → compute |
| transitional | 8, 16, 32, 64 | transitional regime focus |
| transitional_esimd | 8, 16, 32 | ESIMD transitional coverage (drop 64 to stay under cap) |
| hybrid_thresholds | 1, 4, 32, 64 | Tier-4 adaptive routing thresholds |
| moe | 2, 6, 96 | distinct from core to avoid duplicates |
| memory_compare | 1, 32, 128 | memory/transitional/compute reference points |
| compute | 64, 128 | compute-bound focus |

## Config sets (runnable)

| Name | Kernels | Quant types | Dims | Batches | Memory | Count |
| --- | --- | --- | --- | --- | --- | --- |
| core_aos | mmvq_aos | Q4_0, Q8_0, Q6_K, MXFP4, Q4_K, Q2_K, Q3_K, Q5_K | standard | core | USM_DEVICE | 360 |
| core_soa | mmvq_soa | Q4_0, Q8_0, Q6_K, MXFP4, Q4_K | standard | core | USM_DEVICE | 225 |
| core_coalesced | mmvq_coalesced | Q4_0, Q8_0, MXFP4 | standard | core | USM_DEVICE | 135 |
| tier2_xmx | mmvq_xmx_tile_8x8, mmvq_xmx_tile_16x16, mmvq_xmx_aos_direct, mmvq_xmx_soa_direct, mmvq_xmx_double_buffer | Q4_0, Q8_0, Q6_K, MXFP4, Q4_K | 4096 | 8, 16 | USM_DEVICE | 50 |
| tier2_esimd | mmvq_esimd_dpas_1x16x32, mmvq_esimd_dpas_8x16x32, mmvq_esimd_dpas_chained | all | 4096 | 8, 16 | USM_DEVICE | 48 |
| tier3_xmx | mmvq_xmx_tile_64x64, mmvq_xmx_register_accum, mmvq_xmx_multi_wg, mmvq_xmx_persistent | all | 4096 | 64, 128 | USM_DEVICE | 64 |
| tier3_esimd | mmvq_esimd_large_tile, mmvq_esimd_persistent, mmvq_esimd_lsc_prefetch | all | 4096 | 64, 128 | USM_DEVICE | 48 |
| mmq_aos_compute | mmq_aos | Q4_0, Q8_0, Q6_K, Q4_K, Q2_K, Q3_K, Q5_K | standard | 64, 128 | USM_DEVICE | 70 |
| mmq_soa_compute | mmq_soa | Q4_0, Q8_0, Q6_K, Q4_K | standard | 64, 128 | USM_DEVICE | 40 |
| mmq_coalesced_compute | mmq_coalesced | Q4_0, Q8_0, Q6_K | standard | 64, 128 | USM_DEVICE | 30 |
| tier4_hybrid_adaptive | mmvq_hybrid_adaptive | Q4_0, Q8_0 | 4096 | 1, 4, 32, 64 | USM_DEVICE | 8 |
| tier4_xmx_fused | mmvq_xmx_fused | Q4_0, Q8_0 | 4096 | 8, 16 | USM_DEVICE | 4 |
| tier4_coalesced_xmx_aligned | mmvq_coalesced_xmx_aligned | Q4_0, Q8_0 | 4096 | 8, 16 | USM_DEVICE | 4 |
| tier4_esimd_hybrid | mmvq_esimd_hybrid | Q4_0, Q8_0 | 4096 | 1, 4, 64 | USM_DEVICE | 6 |
| tier4_esimd_cooperative | mmvq_esimd_cooperative | Q4_0 | 4096 | 64, 128 | USM_DEVICE | 2 |
| tier4_q4_0_specialized | mmvq_q4_0_specialized | Q4_0 | 4096 | 64, 128 | USM_DEVICE | 2 |
| tier4_q6_k_specialized | mmvq_q6_k_specialized | Q6_K | 4096 | 64, 128 | USM_DEVICE | 2 |
| tier4_mxfp4_native | mmvq_mxfp4_native | MXFP4 | 4096 | 64, 128 | USM_DEVICE | 2 |
| coalesced_q6k | mmvq_coalesced | Q6_K | coalesced_q6k | 1, 32, 128, 512 | USM_DEVICE | 8 |
| moe_aos | mmvq_aos | Q4_0, Q8_0, Q6_K, MXFP4 | moe | moe | USM_DEVICE | 48 |
| moe_coalesced | mmvq_coalesced | Q4_0, Q8_0, MXFP4 | moe | moe | USM_DEVICE | 36 |
| memory_mode_compare | mmvq_aos, mmvq_coalesced | Q4_0, Q8_0 | memory_compare | memory_compare | USM_SHARED, BUFFER | 48 |

**Runnable total:** 1281 configs (includes 4 DPAS device-opt checks)

## DPAS device-opt checks

Each case uses `dpas_memory_patterns` with `dim_m=2048`, `dim_n=256`, `dim_k=4096`, `batch=1`, `Q8_0`.

| Name | Repeat | Notes |
| --- | --- | --- |
| dpas_device_opt_repeat1 | 1 | Heuristic sanity for repeat=1 |
| dpas_device_opt_repeat2 | 2 | Heuristic sanity for repeat=2 |
| dpas_device_opt_repeat4 | 4 | Heuristic sanity for repeat=4 |
| dpas_device_opt_repeat8 | 8 | Heuristic sanity for repeat=8 |

## Edge cases

| Name | Notes | Expected |
| --- | --- | --- |
| non_aligned_dim | K=4097 for Q4_0 | Expected failure |
| coalesced_q6k_non_tile | Q6_K coalesced with k=4096 | Expected failure |
| odd_k_dimension | K=127 for Q4_0 | Expected failure |
| non_aligned_batch | batch=63 | Runnable |
| minimum_dim | dim=64 | Runnable |
| batch_boundary_5 | batch=5 | Runnable |
| batch_boundary_31 | batch=31 | Runnable |
| batch_boundary_33 | batch=33 | Runnable |

## Stress tests (skip by default)

| Name | Notes |
| --- | --- |
| near_limit | ~11 GB allocation (dim=32768, batch=65536, Q4_0) |
| fragmented | Alternating alloc/free pattern (dim=16384, batch=8192, Q4_0) |

## Memory estimate formula

- **Weights:** `row_size_bytes(quant, K) * M`
- **Activations:** `batch * ceil(K/32) * 36 bytes` (Q8_1 blocks)
- **Output:** `batch * M * 4 bytes`
- **Budget:** 16 GB device minus 2 GB headroom (14 GB usable)

## Runtime estimate (for planning)

Regime selection by batch:
- `batch <= 4` → memory bound (assumed 1200 tps)
- `batch <= 64` → transitional (assumed 3000 tps)
- `batch > 64` → compute bound (assumed 8000 tps)

Estimated runtime per config:
```
iterations * (batch / assumed_tps)
```

## Notes

- Coalesced layout requires `(K / block_size) % 32 == 0`.
- Q6_K coalesced is restricted to K multiples of 8192.
- Expected-failure edge cases verify validation paths without polluting runnable counts.
