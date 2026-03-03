//
// cpu-dispatch.hpp — CPU compute path for data-local inference
// When unified cache evicts weights to host pinned memory, this dispatches
// layer computation to a SYCL CPU device instead of streaming to GPU.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_CPU_DISPATCH_HPP
#define GGML_SYCL_CPU_DISPATCH_HPP

#include "common.hpp"

// Dispatch a single ggml operation to the CPU SYCL device.
// Returns true if handled, false if the op is unsupported on CPU.
bool ggml_sycl_compute_forward_cpu(ggml_backend_sycl_context & ctx, struct ggml_tensor * dst);

// Invalidate the activation quantization cache.  Call at the start of each
// graph compute to prevent stale cache hits across tokens.
void ggml_sycl_cpu_quant_cache_new_graph();

// Drain any pending async staging events (call at boundary sync points).
void ggml_sycl_cpu_staging_drain();

// Clear persistent staging cache for leaf tensors.
// Call on graph shape change (new token count changes masks).
void ggml_sycl_cpu_staging_cache_clear();

// Register the original host (mmap) pointer for a weight tensor.
// Called from set_tensor when weight data is uploaded to the device.
// The CPU dispatch path uses this to access quantized weight data directly
// from the mmap'd GGUF file, avoiding dequantization when using vec_dot.
void ggml_sycl_cpu_dispatch_register_host_ptr(const char * name, const void * host_ptr, size_t size);

// Fused CPU operation handlers — eliminate intermediate staging round-trips.
// Returns true if fusion was applied, false to fall back to individual dispatch.
bool ggml_sycl_compute_fused_rms_norm_mul(ggml_backend_sycl_context & ctx,
                                           ggml_tensor * rms_dst, ggml_tensor * mul_dst);
bool ggml_sycl_compute_fused_add_rms_norm(ggml_backend_sycl_context & ctx,
                                            ggml_tensor * add_dst, ggml_tensor * rms_dst);

// Retained activation API — eliminates per-op staging overhead
// by keeping intermediate results in host scratch memory between
// consecutive CPU-dispatched ops within a layer block.
void   ggml_sycl_cpu_retained_init(int device, sycl::queue * gpu_q);
void   ggml_sycl_cpu_retained_cleanup();
bool   ggml_sycl_cpu_retained_active();
void * ggml_sycl_cpu_retained_alloc_output(const ggml_tensor * dst);
void   ggml_sycl_cpu_retained_flush_all(int device, sycl::queue * gpu_q);
void   ggml_sycl_cpu_retained_flush_selective(int device, sycl::queue * gpu_q,
                                               const ggml_tensor * const * gpu_nodes,
                                               int n_gpu_nodes);
void   ggml_sycl_cpu_retained_deactivate();

// Check if a pointer is host-accessible USM (host or shared allocation).
// Used by can_batch_cpu() to verify HOST_COMPUTE buffers are CPU-accessible.
bool ggml_sycl_is_host_accessible_usm(void * ptr, int device);

// Tensor-split: compute vec_dot for a contiguous range of weight rows on CPU.
// src0_host must point to the FIRST row to process (pre-offset by caller).
// Output buffer receives n_rows float results.
void ggml_sycl_cpu_vec_dot_rows(ggml_type type, int ne00,
                                 const void * src0_host, const float * src1_host,
                                 float * output, int n_rows);

// Batch item for ggml_sycl_cpu_vec_dot_batched().
// Each item represents one tensor's CPU row range from tensor split.
struct cpu_vec_dot_batch_item {
    const void *  weight_data;   // host pointer to first CPU row (pre-offset by caller)
    const float * src1_host;     // host float32 activation [ne00]
    float *       output;        // output buffer for this item [n_rows]
    ggml_type     type;          // weight quant type (e.g. GGML_TYPE_Q4_0)
    int           ne00;          // K columns per row
    int           n_rows;        // number of CPU rows for this item
};

// Process multiple tensor split work items in a single TBB parallel_for.
// Deduplicates src1 quantization: items sharing the same src1_host pointer
// share one quantized copy. Distributes work across all TBB threads.
void ggml_sycl_cpu_vec_dot_batched(const cpu_vec_dot_batch_item * items, int n_items);

// Lookup the registered host (mmap) pointer for a weight tensor by name.
// Returns nullptr if not registered.
const void * ggml_sycl_cpu_dispatch_get_host_ptr(const char * name);

// HOST_COMPUTE host_task mode: when active, CPU ops run as host_task
// callbacks on gpu_q instead of parallel_for on cpu_q.  Activated when
// GGML_SYCL_HOST_COMPUTE=1 + CPU offload is active.
void   ggml_sycl_host_task_mode_set(bool active);

// BATCHED host_task mode: when active, CPU ops run as direct function calls
// inside a single batched host_task, not individual submissions.
void   ggml_sycl_batched_mode_set(bool active);
bool   ggml_sycl_batched_mode_active();

// ---------------------------------------------------------------------------
// MoE Expert CPU MUL_MAT — compute expert matmuls directly from host RAM
// ---------------------------------------------------------------------------

// Describes a single expert's matmul work for CPU dispatch.
// weight_host is a raw pointer into host RAM (e.g. mmap'd GGUF file).
// act_host is the activation vector (float, length K).
// output_host receives the result (float, length N).
struct cpu_expert_task {
    const void *  weight_host;   // Expert weight data in host RAM (quantized)
    const float * act_host;      // Activation vector (float32, length K)
    float *       output_host;   // Output buffer (float32, length N)
    ggml_type     type;          // Weight quant type (Q4_0, Q8_0, Q6_K, etc.)
    int           K;             // Input dimension (columns per weight row)
    int           N;             // Output rows (expert output dimension)
};

// Compute one expert's matmul on CPU, reading weights directly from host RAM.
// output[i] = dot(weight_row[i], quantize(act)) for i in [0, N).
// Uses the existing TBB thread pool for parallel row computation.
// Supports any quantized type with vec_dot (Q4_0, Q8_0, Q6_K, etc.).
void ggml_sycl_cpu_expert_mul_mat(const cpu_expert_task & task);

// Compute multiple experts concurrently on the CPU thread pool.
// Deduplicates activation quantization: tasks sharing the same act_host pointer
// and K dimension share one quantized copy.  All experts' rows are flattened
// into a single TBB parallel_for for maximum load balancing.
// n_threads is currently unused; thread count is controlled by the global
// TBB arena (see ggml_sycl_cpu_threads_hint). Parameter retained for future
// per-call control.
void ggml_sycl_cpu_expert_mul_mat_batched(
    const cpu_expert_task * tasks, int n_tasks,
    int n_threads = 0);

// ---------------------------------------------------------------------------
// CPU PP GEMM — batched GEMM on host-resident quantized weights for PP
// ---------------------------------------------------------------------------

// Compute batched GEMM on host-resident quantized weights for PP (large M).
// Primary path: quantize activations (F32 -> Q8_0), then parallel vec_dot
// per output element.  Falls back to dequant-to-F32 + dnnl_sgemm if vec_dot
// is unavailable for the weight type.
// weight_host: raw pointer to quantized weight rows [N rows, each ggml_row_size bytes]
// src1_host:   F32 activation matrix [M, K] row-major
// dst_host:    F32 output matrix [M, N] row-major
// Returns true on success, false if unsupported (no vec_dot or dequant function)
bool ggml_sycl_cpu_pp_gemm(ggml_type weight_type,
                            const void * weight_host, int64_t N, int64_t K,
                            const float * src1_host, int64_t M,
                            float * dst_host, int64_t ldc);

// ---------------------------------------------------------------------------
// Mixed-Precision Cache Miss Loading (HOBBIT-style)
// ---------------------------------------------------------------------------

// Precision mode for cache-miss expert computation.
enum class expert_miss_precision {
    FULL  = 0,   // Always use full precision (Q4_0/Q6_K as-is)
    MIXED = 1,   // Use INT4 for burst misses (>threshold per layer)
};

// Return the configured miss precision mode.
// Reads GGML_SYCL_EXPERT_MISS_PRECISION env var (full|mixed, default: mixed).
expert_miss_precision ggml_sycl_expert_miss_precision_mode();

// Return the burst miss threshold (default: 3).
// When miss count per layer exceeds this, mixed precision activates.
// Reads GGML_SYCL_EXPERT_MISS_BURST_THRESHOLD env var.
int ggml_sycl_expert_miss_burst_threshold();

// Compute multiple experts with adaptive precision based on miss count.
// When n_tasks > burst_threshold AND mode == MIXED, experts beyond the
// threshold are computed at reduced precision (truncated INT4 from Q4_0).
// First burst_threshold experts always use full precision.
void ggml_sycl_cpu_expert_mul_mat_adaptive(
    const cpu_expert_task * tasks, int n_tasks,
    int n_miss_total);

#endif // GGML_SYCL_CPU_DISPATCH_HPP
