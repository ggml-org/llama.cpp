# TurboQuant SYCL Backend — Implementation & Engineering Playbook

This document is the Single Source of Truth for the development, architecture, and optimization of the TurboQuant SYCL backend for `llama.cpp`. It is designed to accelerate implementation velocity and ensure architectural consistency for both human engineers and AI agents.

---

## 1. Project Overview

**TurboQuant (TQ)** is a high-performance quantization suite (2, 3, 4-bit) based on PolarQuant/QJL (ICLR 2026). It utilizes the Fast Walsh-Hadamard Transform (WHT) to rotate tensors into a domain where Lloyd-Max quantization is highly effective, enabling aggressive KV cache and weight compression with minimal perplexity loss.

**SYCL Backend Maturity**: **Partial Prototype (Skeleton Functional)**.
*   **Target Hardware**: Intel Arc Graphics (A-series, B-series), Intel Data Center GPU Max (PVC), and integrated Xe Graphics.
*   **Current State**: `TURBO3_0` KV cache is functional via the Flash Attention (Vec) path. Weight quantization (`TQ3_1S`, `TQ4_1S`) kernels and hardware acceleration (XMX/DPAS) are fundamentally missing.
*   **Goal**: 100% feature parity with CUDA/Metal, fully optimized for Xe-core systolic arrays.

---

## 2. Repository Structure Deep Map

### `ggml/src/ggml-sycl/` (Primary Implementation Area)
*   **`ggml-sycl.cpp`**: The backend entry point. Contains op dispatch, USM management, and **Capability Checks** (`ggml_backend_sycl_device_supports_op`).
    *   *Risk*: Capability checks must strictly match kernel availability to prevent `GGML_ABORT` during inference.
*   **`turbo-quants.hpp`**: **Critical Path**. Defines centroids, nearest-neighbor lookup tables, and device functions for quantization/dequantization.
    *   *Status*: Missing implementation for 2-bit and 4-bit packing logic.
*   **`turbo-wht.hpp`**: Implements Fast Walsh-Hadamard Transform using subgroup shuffles and SLM.
    *   *Constraint*: Hardcoded SG size 32 must be audited for SG16/SG8 hardware (e.g., LNL/MTL iGPUs).
*   **`set_rows.cpp`**: Implements the quantization path for KV cache updates (`GGML_OP_SET_ROWS`).
    *   *Status*: Only `TURBO3_0` is implemented. `TURBO2/4` need integration here.
*   **`fattn-vec.hpp` / `fattn-tile.hpp`**: Flash Attention implementation.
    *   *Status*: `vec` is functional for `TURBO3_0`. `tile` is a work-in-progress prototype.
*   **`mmvq.cpp`**: Matrix-Vector Multiplication (GEMV). **Primary Blocker**. Missing all TQ weight types.
*   **`mmq.cpp`**: Matrix-Matrix Multiplication (GEMM). Missing all TQ weight types.
*   **`vecdotq.hpp`**: Low-level dot product primitives. Needs `TQ4_1S` and `TQ3_1S` software and hardware paths.
*   **`dequantize.hpp`**: Handles final dequantization to float for output layers.

### `tests/`
*   **`test-sycl-turbo.cpp`**: (Custom) Validates SYCL kernels against CPU references. **Always run this before committing quantization changes.**

---

## 3. Backend Architecture

### Runtime & Memory
*   **SYCL 2020**: Utilizing DPC++ compiler (`icpx`).
*   **USM (Unified Shared Memory)**: All allocations are `device` or `host` USM. No Buffer/Accessor abstraction is used to minimize overhead.
*   **Queue Management**: Optimistic dependency tracking. Most ops are submitted to the same queue, relying on in-order properties or manual `.wait()` calls.
    *   *Technical Debt*: Lacks native SYCL event-based graph dependencies.

### Inference Flow
1.  **Model Loading**: `ggml_sycl_supports_op` determines if the backend can handle the quants.
2.  **Quantization (KV Cache)**: During `SET_ROWS`, the input is rotated via WHT and Lloyd-Max quantized.
3.  **Attention**: Flash Attention kernels load quantized blocks, dequantize into registers, and compute scores.
4.  **GEMV (Weights)**: `mmvq` kernels compute matrix-vector products for FFN/Projection layers.

---

## 4. Kernel Architecture

*   **Subgroup Strategy**: Primary optimization mechanism. Uses shuffles (`permute_sub_group_by_xor`) for WHT and packing.
    *   **Rule**: Handle `sub_group_size` dynamically or use `[[sycl::reqd_sub_group_size(32)]]` with caution.
*   **Vectorization**: Prefer `sycl::vec<T, N>` for memory loads to saturate bandwidth.
*   **XMX / DPAS**: Intel's systolic arrays. Currently **Unused**.
    *   **Architecture Goal**: Utilize `sycl::ext::oneapi::experimental::matrix::joint_matrix` for weight multiplications.

---

## 5. Quantization System (Weights: TQ3_1S / TQ4_1S)

*   **Format**: Block size 32, PolarQuant.
*   **Missing Logic**:
    *   Subgroup-aware dequantization loops.
    *   `vec_dot` kernels combining TQ weights with Q8_1 activations.
*   **Implementation Rule**: Finalize bit-packing in `turbo-quants.hpp` before implementing `mmvq` dispatch.

---

## 6. Flash Attention

*   **`fattn-vec`**: Optimized for small sequence lengths/decoding.
    *   *Optimization*: High SLM usage. Minimize bank conflicts in subgroup reductions.
*   **`fattn-tile`**: Designed for high sequence lengths/prefill.
    *   *Status*: Requires redesign to support multi-bit TQ loaders efficiently.

---

## 7. Intel Arc / Xe Optimization Guide

*   **XE-Core**: 16 compute units per SM equivalent. 512 total on A770.
*   **L1 Cache / SLM**: Shared. Using too much SLM reduces occupancy.
*   **Occupancy**: Aim for at least 4-8 workgroups per Xe-core.
*   **Subgroup Behavior**: Intel GPUs execute in SIMD8, SIMD16, or SIMD32. The compiler makes the choice unless forced. **Always verify `reqd_sub_group_size` efficiency**.

---

## 8. Current Completion Status (Realistic)

| Subsystem | Implementation | Stability | Optimization |
| :--- | :---: | :---: | :---: |
| Core Backend | 90% | 40% | 40% |
| Quantization (TQ Weights) | 100% | 60% | 20% |
| KV Cache (TURBO3) | 100% | 40% | 30% |
| KV Cache (TURBO2/4) | 100% | 40% | 30% |
| Flash Attention Vec | 100% | 30% | 40% |
| XMX / DPAS Acceleration | 0% | 0% | 0% |


---

## 9. Active Critical Path & Priorities

1.  **Immediate**: Synchronize `ggml_sycl_supports_op` (Lying capability checks cause crashes).
2.  **High**: Implement `TURBO2_0` and `TURBO4_0` in `set_rows.cpp` to unlock KV cache parity.
3.  **Critical**: Implement `mmvq` for `TQ4_1S`. This is the single biggest missing feature.
4.  **Optimization**: Prototype `joint_matrix` (XMX) integration for GEMM.

---

## 10. AI-Agent Continuation Rules

*   **Research First**: Before implementing a kernel, check `turbo-quants.hpp` for the required device functions.
*   **No Lying**: Do not add types to `ggml_sycl_supports_op` until the `mmvq` path is confirmed functional.
*   **Validation**: Every change to a device function in `turbo-quants.hpp` must be verified using `tests/test-sycl-turbo.cpp`.
*   **Idiomatic SYCL**: Replace `dpct::permute_sub_group_by_xor` with native `sycl::select_from_group` where possible for better compiler optimization.
*   **Xe-First**: Do not assume CUDA tiling or warp-sync patterns work efficiently on Xe. Design for subgroups.

---

## 11. Engineering Standards

*   **TDD**: Write the test case in `test-sycl-turbo.cpp` before implementing the kernel.
*   **KISS**: Implement functional software paths before attempting hardware-specific (XMX) paths.
*   **Rollback Safety**: Keep non-functional stubs inside `#if 0` or gated by capability checks.

---

## 12. Phase 2: Stability Hardening & Deep Optimization (Updated)

The project has transitioned from the **Core Implementation Phase** to the **Stability Hardening + Deep Optimization Phase**. Engineering priorities now shift from "adding features" to "ensuring production-grade reliability and hardware-saturated performance."

### REAL Completion Assessment
| State | Maturity % | Description |
| :--- | :---: | :--- |
| **Functional** | 95% | 2/3/4-bit KV cache and 3/4-bit weights are functional and bit-exact with CPU. |
| **Stable** | 40% | Basic inference passes; lacks stress/fuzz/long-context validation. Hidden race conditions likely. |
| **Optimized** | 15% | Basic subgroup usage exists. **XMX/DPAS usage is 0%**. Performance is bandwidth-bound. |
| **Production** | 10% | Lacks automated regression, CI/CD benchmarks, and Xe-specific tuning. |

**Current Primary Blocker**: **XMX / DPAS Integration**. Without hardware matrix acceleration, the backend will remain 5-10x slower than CUDA.

---

## 13. Stability Hardening & Validation Infrastructure

Production readiness requires aggressive validation.
1.  **Deterministic Execution**: Every inference run MUST produce bit-exact matches across multiple iterations.
2.  **Stress Testing**: Validate long-context (>32k) and high-batch scenarios to expose USM pool fragmentation or driver-level timeouts.
3.  **Numerical Stability Audit**: Audit WHT rotation and normalization factors for precision loss in `half` accumulators.
4.  **Regression Suite**: `tests/test-sycl-turbo.cpp` is the gatekeeper. **Every PR must pass this test with MSE = 0.0.**

---

## 14. Intel Arc / Xe Optimization Guide (Deep-Dive)

### XMX / DPAS Strategy (The New Frontier)
*   **Architecture**: Xe-cores contain Systolic Arrays (XMX) that compute matrix products using `dpas` instructions.
*   **Implementation Path**: Use `sycl::ext::oneapi::experimental::matrix::joint_matrix` for high-level abstraction or `syclex::dpas` for low-level control.
*   **Tiling**: CUDA-style tiling does NOT directly translate. Use **Xe-native tiling** (e.g., 8x16 or 16x16 blocks) to saturate register file bandwidth.
*   **Layouts**: XMX requires specific memory layouts (A: Row-major, B: Column-major VNNI-packed).

### Memory & Occupancy
*   **Occupancy**: Arc GPUs have large register files. Aim for workgroup sizes that balance registers per thread vs. total active warps.
*   **Cache Locality**: Xe L1 cache is shared with SLM. High SLM usage **directly reduces** L1 cache availability.
*   **Subgroup Choice**: Prefer SG32 for compute-bound kernels; consider SG16 for register-pressured kernels.

---

## 15. Active Critical Path (Execution Roadmap)

1.  **Phase 1: Stabilization (Immediate)**: Deterministic replay checks and synchronization audit.
2.  **Phase 2: Validation Infra**: Add automated CI regression tests for all TQ bit-widths.
3.  **Phase 3: XMX Prototype**: Implement a baseline `joint_matrix` multiply for `TQ4_1S`.
4.  **Phase 4: Flash Attention Optimization**: Subgroup reduction tuning and bank-conflict removal in `fattn-vec`.
