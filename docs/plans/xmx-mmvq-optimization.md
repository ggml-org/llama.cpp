# XMX-Optimized Unified Kernel Architecture

**Epic:** llama.cpp-a3t
**Created:** 2026-01-20
**Status:** Planning Complete

## Executive Summary

This document outlines a comprehensive plan to optimize MMVQ (Matrix-Matrix Vector Quantized) kernels for Intel Arc GPUs using XMX (Xe Matrix eXtensions). The goal is to create a unified kernel architecture that performs optimally across all batch sizes, from single-token generation (batch=1) through large prompt processing (batch=8192).

### Key Decisions

1. **Multiple specialized kernels** rather than single unified kernel
2. **On-the-fly SLM conversion** preferred over preprocessing
3. **Standalone benchmark binary** for isolated kernel testing
4. **Parallel development** with subagents for kernel variants
5. **Data-driven selection** based on profiling results

### Target Quantization Types

Q4_0, Q8_0, Q6_K, MXFP4, Q4_K, Q2_K, Q3_K, Q5_K

---

## Section 1: Architecture Overview

### 1.1 Current State

The SYCL backend currently uses three kernel paths:

| Kernel | Batch Range | Bottleneck | Current Performance |
|--------|-------------|------------|---------------------|
| DMMV | 1 | Memory bandwidth | Baseline |
| MMVQ | 1-4 | Memory bandwidth | Best for small batch |
| MMQ/XMX | 8+ | Compute | 5-11x slower than MMVQ at crossover |

**Problem:** Sharp performance cliff at MMVQ→XMX transition point.

### 1.2 Target Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Unified Dispatch Layer                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Batch=1-4   │  │ Batch=8-64  │  │ Batch=64+           │  │
│  │ Memory-Bound│  │ Transitional│  │ Compute-Bound       │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
└─────────┼────────────────┼────────────────────┼─────────────┘
          │                │                    │
          ▼                ▼                    ▼
   ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐
   │ Tier 1       │ │ Tier 2       │ │ Tier 3               │
   │ MMVQ Variants│ │ Small XMX    │ │ Large XMX Tiles      │
   │ - SLM Cache  │ │ - 8x8 Tiles  │ │ - 64x64 Tiles        │
   │ - Prefetch   │ │ - 16x16 Tiles│ │ - Persistent Thread  │
   │ - Wide Load  │ │ - ESIMD dpas │ │ - Multi-WG Coop      │
   └──────────────┘ └──────────────┘ └──────────────────────┘
```

### 1.3 Memory Layout Strategy

**Base Layout:** Coalesced Word-Major (current best for memory bandwidth)

**XMX Feeding:** On-the-fly conversion in SLM during kernel execution
- Avoids VRAM overhead of multiple layouts
- Conversion cost amortized across compute

---

## Section 2: Benchmark Infrastructure

### 2.1 Standalone Benchmark CLI

```bash
# sycl-kernel-bench - independent from llama-bench
./sycl-kernel-bench \
  --kernel=mmvq_slm_cached \
  --quant=Q4_0 \
  --batch=1,4,8,16,32,64 \
  --dim=4096 \
  --iterations=100 \
  --warmup=10 \
  --output=csv
```

### 2.2 Test Matrix

#### Standard Dimensions (Real Model Shapes)

| Dimension | Models |
|-----------|--------|
| 4096 | Mistral-7B, LLaMA-7B |
| 5120 | LLaMA-13B |
| 6144 | LLaMA-30B |
| 8192 | LLaMA-65B |
| 14336 | Mistral-7B FFN |

#### MoE Dimensions

| Dimension | Source |
|-----------|--------|
| 2048 | DeepSeek-V3 |
| 4096 | Mixtral FFN |
| 8192 | Large MoE |
| 16384 | GPT-OSS FFN up_proj |

#### Batch Sizes

```
1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192
```

#### Memory Pressure Tests

- Near-device-memory-limit allocations
- Fragmented memory scenarios
- USM vs buffer mode comparison

### 2.3 Output Format

```csv
kernel,quant,batch,dim_m,dim_n,dim_k,layout,throughput_tps,latency_us,bandwidth_gbps,xmx_util_pct,variance_pct
mmvq_coalesced,Q4_0,1,4096,4096,4096,COALESCED,1250.5,800.2,312.5,0.0,2.1
xmx_tile_16x16,Q4_0,64,4096,4096,4096,XMX_TILED,18500.3,54.1,285.2,78.5,1.8
```

---

## Section 3: Kernel Variants

### 3.1 Tier 0: Reference Kernels

| ID | Variant | Purpose |
|----|---------|---------|
| T0.1 | oneDNN FP16 GEMM | XMX throughput ceiling |
| T0.2 | oneDNN INT8 GEMM | Quantized ceiling (if viable) |
| T0.3 | Memory Copy | Bandwidth ceiling |
| T0.4 | Synthetic Compute | FLOPS ceiling |

### 3.2 Tier 1: Memory-Bound Kernels (batch=1-4)

#### Standard SYCL (6 variants)

| ID | Variant | Hypothesis |
|----|---------|------------|
| T1.1 | Baseline AoS | Current MMVQ performance |
| T1.2 | Baseline SoA | Existing coalesced layout |
| T1.3 | Coalesced Word-Major | Current best (verify) |
| T1.4 | SLM-Cached | Reduce global memory trips |
| T1.5 | Software Prefetch | Hide memory latency |
| T1.6 | Wide Vector Loads | 64/128-bit loads for bandwidth |

#### ESIMD (2 variants)

| ID | Variant | Hypothesis |
|----|---------|------------|
| T1.7 | ESIMD block_load | Explicit wide loads |
| T1.8 | ESIMD SLM | Manual SLM management |

### 3.3 Tier 2: Transitional Kernels (batch=8-64)

#### Standard SYCL (5 variants)

| ID | Variant | Hypothesis |
|----|---------|------------|
| T2.1 | 8x8 Tiles | Smallest XMX-viable tile |
| T2.2 | 16x16 Tiles | Standard tile size |
| T2.3 | AoS + XMX | Direct AoS feeding XMX |
| T2.4 | SoA + XMX | Coalesced data to XMX |
| T2.5 | Double Buffered | Hide conversion latency |

#### ESIMD (3 variants)

| ID | Variant | Hypothesis |
|----|---------|------------|
| T2.6 | ESIMD dpas 1x16x32 | Baseline XMX tile |
| T2.7 | ESIMD dpas 8x16x32 | Maximum repeat count |
| T2.8 | ESIMD Chained dpas | Multiple dpas in sequence |

### 3.4 Tier 3: Compute-Bound Kernels (batch=64+)

#### Standard SYCL (4 variants)

| ID | Variant | Hypothesis |
|----|---------|------------|
| T3.1 | Large 64x64 Tiles | Maximum register utilization |
| T3.2 | Register Accumulate | Reduce SLM pressure |
| T3.3 | Multi-WG Cooperative | Split work across work-groups |
| T3.4 | Persistent Threading | Reduce launch overhead |

#### ESIMD (3 variants)

| ID | Variant | Hypothesis |
|----|---------|------------|
| T3.5 | ESIMD Large Tiles | 64x64 with explicit control |
| T3.6 | ESIMD Persistent | Named barriers for cooperation |
| T3.7 | ESIMD LSC Prefetch | Explicit prefetch hints |

### 3.5 Tier 4: Experimental/Hybrid Kernels

#### Standard SYCL Hybrids (3 variants)

| ID | Variant | Hypothesis |
|----|---------|------------|
| T4.1 | Hybrid-Adaptive | Runtime batch-size selection |
| T4.2 | MMVQ-XMX-Fused | Use XMX even at batch=1 |
| T4.3 | Coalesced-XMX-Aligned | Best layout for both regimes |

#### ESIMD Hybrids (2 variants)

| ID | Variant | Hypothesis |
|----|---------|------------|
| T4.4 | ESIMD-Hybrid | Adaptive tiling based on batch |
| T4.5 | ESIMD-Cooperative | Named barriers for large tiles |

#### Per-Quant Specializations (3 focus areas)

| ID | Quant | Focus |
|----|-------|-------|
| T4.6 | Q4_0 | Most common, optimize heavily |
| T4.7 | Q6_K | K-quant specific patterns |
| T4.8 | MXFP4 | Native hardware support path |

### 3.6 ESIMD dpas Configuration Matrix

#### Tile Dimensions

| Config | Repeat | M | N | K |
|--------|--------|---|---|---|
| 1x16x32 | 1 | 1 | 16 | 32 |
| 2x16x32 | 2 | 2 | 16 | 32 |
| 4x16x32 | 4 | 4 | 16 | 32 |
| 8x16x32 | 8 | 8 | 16 | 32 |

#### Memory Access Patterns

1. Direct global load
2. SLM buffering
3. Register prefetch
4. Double buffering
5. LSC streaming loads
6. LSC prefetch hints

#### Priority Experiments

| Priority | Experiment | Question |
|----------|------------|----------|
| P0 | Baseline dpas throughput | What is theoretical XMX TOPS? |
| P0 | repeat=8 sustained | Can we sustain max repeat? |
| P1 | K=128 chaining for Q4_0 | Does block-aligned K help? |
| P1 | batch=1 XMX viability | Is XMX worth it for batch=1? |
| P2 | Dequantized vs native | INT8 vs dequant+FP16 overhead |
| P2 | GRF tradeoff | Occupancy vs register pressure |

---

## Section 4: Development Phases

### Phase 1: Infrastructure (Week 1)

**Tasks:**
- `llama.cpp-wzx`: Build standalone kernel microbenchmark infrastructure
- `llama.cpp-h66`: Define comprehensive test matrix

**Deliverables:**
- Working `sycl-kernel-bench` CLI
- Memory allocation helpers
- Timing infrastructure
- Test data generators

### Phase 2: Reference Kernels (Week 2)

**Tasks:**
- `llama.cpp-65i`: Implement reference/baseline kernels (Tier 0)

**Deliverables:**
- oneDNN FP16/INT8 ceiling measurements
- Memory bandwidth baseline
- Roofline chart data

### Phase 3: Memory-Bound Kernels (Week 3)

**Tasks:**
- `llama.cpp-241`: Implement memory-bound kernel variants (Tier 1)

**Deliverables:**
- 8 kernel variants (6 SYCL + 2 ESIMD)
- Per-quant specializations (64 configurations)
- Batch=1-4 benchmark data

### Phase 4: XMX Kernels (Week 4)

**Tasks (parallel execution):**
- `llama.cpp-yx7`: Transitional kernels (Tier 2)
- `llama.cpp-q4g`: Compute-bound kernels (Tier 3)
- `llama.cpp-li6`: ESIMD dpas configuration exploration
- `llama.cpp-6t3`: Experimental/hybrid variants (Tier 4)

**Deliverables:**
- 20+ kernel variants
- Crossover point identification
- XMX utilization measurements

### Phase 5: Analysis & Hybrid Dispatch (Week 5)

**Tasks:**
- `llama.cpp-7pg`: Analyze benchmark data and build adaptive dispatch

**Deliverables:**
- Consolidated benchmark analysis
- Dispatch threshold recommendations
- Hybrid kernel selection

### Phase 6: Production Integration (Week 6)

**Tasks:**
- `llama.cpp-o5z`: Production integration of optimized kernels

**Deliverables:**
- Updated `mmvq.cpp` with winning variants
- Updated dispatch logic in `ggml-sycl.cpp`
- Validation with real models

---

## Section 5: Success Criteria

### 5.1 Performance Metrics

| Criterion | Measurement | Target |
|-----------|-------------|--------|
| Coverage | Kernels benchmarked | All 36 variants × 8 quant types |
| Data Quality | Reproducible results | <5% variance across runs |
| Crossover Analysis | Transition points | Per-quant thresholds documented |
| XMX Utilization | % of theoretical peak | Measured and documented |
| Roofline Position | Distance from ceiling | Bottleneck quantified |

### 5.2 Kernel Selection Algorithm

```
Winner Selection:
1. Throughput: Highest tokens/sec in regime
2. Stability: <5% variance across 10 runs
3. Edge Cases: No failures at boundary conditions
4. Memory: No OOM at maximum test dimensions
5. Accuracy: Bit-exact match with reference
```

### 5.3 Validation Requirements

Before production integration:

- [ ] Passes accuracy validation against FP32 reference
- [ ] No memory leaks (verified with compute-sanitizer)
- [ ] Handles edge cases (batch=1, non-aligned dimensions)
- [ ] Works with graph capture enabled
- [ ] Compatible with multi-GPU dispatch

### 5.4 Regression Testing

```bash
# Required test configurations
./sycl-kernel-bench --validate --quant=all --batch=1,8,64,512
./build/bin/llama-bench -m mistral-7b-Q4_0.gguf -p 512 -n 128 -ngl 99
./build/bin/llama-completion -m mistral-7b-Q4_0.gguf -p "1,2,3,4,5," -n 15 --seed 42
```

---

## Section 6: Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| XMX slower than expected | Medium | High | Early Phase 2 baseline establishes ceiling |
| ESIMD compilation issues | Medium | Medium | Fallback to joint_matrix API |
| Quant-specific edge cases | High | Low | Per-quant specialization |
| Memory pressure at large batch | Medium | Medium | Test matrix includes stress configs |
| Dispatch overhead | Low | High | Measure separately, optimize if >1% |

---

## Appendix A: Code Standards

### Kernel Naming Convention

```
{regime}_{approach}_{tile}
Examples: mmvq_slm_cached, xmx_esimd_8x16x32, hybrid_adaptive
```

### Template Structure

```cpp
template <ggml_type QUANT, int TILE_M, int TILE_N>
void kernel_variant_name(
    const void* src, void* dst,
    int64_t M, int64_t N, int64_t K,
    sycl::queue& q);
```

---

## Appendix B: Beads Task Reference

| Phase | Task ID | Description |
|-------|---------|-------------|
| 1 | llama.cpp-wzx | Build standalone kernel microbenchmark infrastructure |
| 1 | llama.cpp-h66 | Define comprehensive test matrix |
| 2 | llama.cpp-65i | Implement reference/baseline kernels (Tier 0) |
| 3 | llama.cpp-241 | Implement memory-bound kernel variants (Tier 1) |
| 4 | llama.cpp-yx7 | Implement transitional kernel variants (Tier 2) |
| 4 | llama.cpp-q4g | Implement compute-bound kernel variants (Tier 3) |
| 4 | llama.cpp-li6 | Benchmark ESIMD dpas configurations |
| 4 | llama.cpp-6t3 | Implement experimental/hybrid variants (Tier 4) |
| 5 | llama.cpp-7pg | Analyze benchmark data and build adaptive dispatch |
| 6 | llama.cpp-o5z | Production integration of optimized kernels |

**Epic:** llama.cpp-a3t - XMX-Optimized Unified Kernel Architecture
