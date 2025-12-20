// Quantized AllReduce for Tensor Parallelism (Flash Communication)
// =================================================================
// Based on Intel LLM-Scaler "Flash Communication" technique:
// 1. Quantize FP32 activations to INT16 on GPU before transfer (2x bandwidth reduction)
// 2. Transfer smaller quantized tensor through host
// 3. Dequantize + add on CPU, then copy result back
//
// Bandwidth analysis for 2-GPU host-staged AllReduce:
//   Standard:   dev0→host(4N) + dev1→host(4N) + host→dev0(4N) = 12N bytes
//   INT16:      dev0→host(2N) + dev1→host(2N) + host→dev0(4N) = 8N bytes (33% reduction!)
//
// INT16 precision: 65536 levels → 0.0015% max quantization error
// INT8 had only 256 levels → 0.4% max error → compounded to 5-24% error per element!
//
// Note: GPU-side quantization is required to achieve this reduction.
// If we copy FP32 first then quantize on CPU, there's no bandwidth benefit.
//
// Enable with: GGML_SYCL_QUANT_ALLREDUCE=1
// Buffers are pre-allocated during ggml_sycl_tp_init() for safe device allocation.

#ifndef GGML_SYCL_QUANTIZED_COMM_HPP
#define GGML_SYCL_QUANTIZED_COMM_HPP

#include "common.hpp"
#include <sycl/sycl.hpp>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <cfloat>

// Quantization parameters computed per-tensor
struct quant_params_t {
    float scale;     // scale = (max_val - min_val) / 65535 (INT16 range)
    float zero;      // zero = min_val
};

// =============================================================================
// GPU Kernels for quantization
// =============================================================================

// Fused min/max kernel for INT16 quantization
// Each work-group processes a chunk and atomically updates global min/max
template<int BLOCK_SIZE = 256>
inline void quantize_fp32_to_int16_kernel(
    const float* __restrict__ src,
    [[maybe_unused]] int16_t* __restrict__ dst,  // Actual quantization done in apply_quantize_kernel
    float* __restrict__ minmax,  // [min, max] - initialized to [FLT_MAX, -FLT_MAX]
    size_t n,
    sycl::nd_item<1> item,
    float* local_data  // SLM for reduction
) {
    const int lid = item.get_local_id(0);
    const int gid = item.get_global_id(0);

    // Phase 1: Find local min/max
    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;

    for (size_t i = gid; i < n; i += item.get_global_range(0)) {
        float val = src[i];
        local_min = sycl::fmin(local_min, val);
        local_max = sycl::fmax(local_max, val);
    }

    // Reduce within work-group using SLM
    local_data[lid] = local_min;
    local_data[lid + BLOCK_SIZE] = local_max;
    item.barrier(sycl::access::fence_space::local_space);

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            local_data[lid] = sycl::fmin(local_data[lid], local_data[lid + stride]);
            local_data[lid + BLOCK_SIZE] = sycl::fmax(local_data[lid + BLOCK_SIZE],
                                                       local_data[lid + BLOCK_SIZE + stride]);
        }
        item.barrier(sycl::access::fence_space::local_space);
    }

    // Thread 0 atomically updates global min/max
    if (lid == 0) {
        sycl::atomic_ref<float, sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
            atomic_min(minmax[0]);
        sycl::atomic_ref<float, sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
            atomic_max(minmax[1]);

        float wg_min = local_data[0];
        float wg_max = local_data[BLOCK_SIZE];

        // Atomic min
        float old_min = atomic_min.load();
        while (wg_min < old_min && !atomic_min.compare_exchange_weak(old_min, wg_min)) {}

        // Atomic max
        float old_max = atomic_max.load();
        while (wg_max > old_max && !atomic_max.compare_exchange_weak(old_max, wg_max)) {}
    }
}

// Second kernel: quantize using computed min/max (INT16)
inline void apply_quantize_kernel(
    const float* __restrict__ src,
    int16_t* __restrict__ dst,
    const float* __restrict__ minmax,  // [min, max]
    size_t n,
    sycl::nd_item<1> item
) {
    const size_t gid = item.get_global_id(0);
    if (gid >= n) return;

    float min_val = minmax[0];
    float max_val = minmax[1];
    float range = max_val - min_val;
    if (range < 1e-8f) range = 1e-8f;
    // INT16: 65536 levels (0 to 65535), centered at 32768
    float inv_scale = 65535.0f / range;

    float val = (src[gid] - min_val) * inv_scale;
    val = sycl::fmax(0.0f, sycl::fmin(65535.0f, val));
    // Store as signed INT16 centered at 0 (subtract 32768)
    dst[gid] = static_cast<int16_t>(val - 32768.0f);
}

// =============================================================================
// Quantized AllReduce Implementation
// =============================================================================

// Debug flag for quantization
static bool g_quant_debug = false;
static int g_quant_debug_count = 0;

// Perform GPU-side INT16 quantization
// Returns event that completes when INT16 data is ready on host
inline sycl::event quantize_and_copy_to_host(
    queue_ptr stream,
    const float* src_fp32,      // FP32 source on device
    int16_t* dev_q,             // INT16 device buffer
    float* dev_minmax,          // [min, max] device buffer
    int16_t* host_q,            // INT16 host destination
    quant_params_t* params_out, // Output quant params (on host)
    size_t n,
    bool debug = false
) {
    constexpr int BLOCK_SIZE = 256;
    const int num_blocks = std::min((int)((n + BLOCK_SIZE - 1) / BLOCK_SIZE), 256);

    // Debug: Read original FP32 values
    if (debug && g_quant_debug_count < 3) {
        float src_sample[4];
        stream->memcpy(src_sample, src_fp32, 4 * sizeof(float)).wait();
        fprintf(stderr, "QUANT_DEBUG: src_fp32[0..3]=[%.6f, %.6f, %.6f, %.6f] n=%zu\n",
                src_sample[0], src_sample[1], src_sample[2], src_sample[3], n);
    }

    // Initialize minmax
    float init_minmax[2] = {FLT_MAX, -FLT_MAX};
    stream->memcpy(dev_minmax, init_minmax, 2 * sizeof(float)).wait();

    // Kernel 1: Find min/max
    stream->submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> local_data(BLOCK_SIZE * 2, h);

        h.parallel_for(
            sycl::nd_range<1>(num_blocks * BLOCK_SIZE, BLOCK_SIZE),
            [=](sycl::nd_item<1> item) {
                quantize_fp32_to_int16_kernel<BLOCK_SIZE>(
                    src_fp32, dev_q, dev_minmax, n, item,
                    local_data.get_multi_ptr<sycl::access::decorated::no>().get()
                );
            }
        );
    }).wait();

    // Read min/max to compute params
    float minmax[2];
    stream->memcpy(minmax, dev_minmax, 2 * sizeof(float)).wait();

    if (debug && g_quant_debug_count < 3) {
        fprintf(stderr, "QUANT_DEBUG: minmax=[%.6f, %.6f]\n", minmax[0], minmax[1]);
        g_quant_debug_count++;
    }

    float range = minmax[1] - minmax[0];
    if (range < 1e-8f) range = 1e-8f;
    // INT16: 65536 levels
    params_out->scale = range / 65535.0f;
    params_out->zero = minmax[0];

    // Kernel 2: Quantize to INT16
    stream->submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(((n + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE, BLOCK_SIZE),
            [=](sycl::nd_item<1> item) {
                apply_quantize_kernel(src_fp32, dev_q, dev_minmax, n, item);
            }
        );
    }).wait();

    // Copy INT16 to host (2N bytes vs 4N standard, 33% bandwidth reduction)
    return stream->memcpy(host_q, dev_q, n * sizeof(int16_t));
}

// Dequantize INT16 + add on CPU
inline void dequant_add_cpu(
    const int16_t* q0, const quant_params_t& p0,
    const int16_t* q1, const quant_params_t& p1,
    float* dst,
    size_t n
) {
    for (size_t i = 0; i < n; i++) {
        // INT16 was stored centered at 0 (subtracted 32768 during quantization)
        float v0 = static_cast<float>(q0[i] + 32768) * p0.scale + p0.zero;
        float v1 = static_cast<float>(q1[i] + 32768) * p1.scale + p1.zero;
        dst[i] = v0 + v1;
    }
}

// =============================================================================
// Main quantized AllReduce function
// =============================================================================
//
// Performs AllReduce with INT16 quantization for 33% bandwidth reduction:
//   1. GPU0: quantize FP32→INT16, copy to host (2N bytes)
//   2. GPU1: quantize FP32→INT16, copy to host (2N bytes)
//   3. CPU: dequant(q0) + dequant(q1) → FP32 result
//   4. Copy FP32 result to GPU0 (4N bytes)
//
// Total: 8N bytes vs standard 12N bytes = 33% reduction
//
// INT16 precision: 65536 levels → 0.0015% max quantization error (vs INT8's 0.4%)
// This is sufficient for accumulating through 32+ transformer layers.
//
// IMPORTANT: Buffers must be pre-allocated via ggml_sycl_tp_init_quant_comm_buffers()
// before calling this function. The buffers are allocated during TP initialization
// to avoid device allocation issues during forward pass.

inline void quantized_allreduce(
    int main_device,
    int other_device,
    queue_ptr main_stream,
    queue_ptr other_stream,
    float* dst_dev0,       // FP32 partial result on device 0 (also output)
    float* src_dev1,       // FP32 partial result on device 1
    size_t n_elements,
    bool debug = false
) {
    // Get pre-allocated buffers
    ggml_sycl_tp_ensure_quant_comm_buffers(n_elements);
    auto* bufs = ggml_sycl_tp_get_quant_comm_buffers();

    if (!bufs) {
        GGML_LOG_ERROR("SYCL TP: Quant comm buffers not initialized. "
                       "Enable with GGML_SYCL_QUANT_ALLREDUCE=1\n");
        return;
    }

    if (bufs->capacity < n_elements) {
        GGML_LOG_ERROR("SYCL TP: Quant comm buffer too small: %zu < %zu\n",
                       bufs->capacity, n_elements);
        return;
    }

    quant_params_t params0, params1;

    // Quantize and copy both partials to host
    // Note: We process other_device first so main_device is set at the end
    if (debug) g_quant_debug_count = 0;  // Reset debug counter

    ggml_sycl_set_device(other_device);
    if (debug) fprintf(stderr, "QUANT_ALLREDUCE: Quantizing dev1 (other_device=%d)...\n", other_device);
    auto ev1 = quantize_and_copy_to_host(
        other_stream, src_dev1,
        bufs->dev_q[1], bufs->dev_minmax[1],
        bufs->host_q1, &params1, n_elements, debug
    );

    ggml_sycl_set_device(main_device);
    if (debug) fprintf(stderr, "QUANT_ALLREDUCE: Quantizing dev0 (main_device=%d)...\n", main_device);
    auto ev0 = quantize_and_copy_to_host(
        main_stream, dst_dev0,
        bufs->dev_q[0], bufs->dev_minmax[0],
        bufs->host_q0, &params0, n_elements, debug
    );

    // Wait for both copies
    ev0.wait();
    ev1.wait();

    if (debug) {
        fprintf(stderr, "QUANT_ALLREDUCE (INT16): params0(scale=%.9f, zero=%.6f) params1(scale=%.9f, zero=%.6f)\n",
                params0.scale, params0.zero, params1.scale, params1.zero);
        fprintf(stderr, "QUANT_ALLREDUCE (INT16): host_q0[0..3]=[%d, %d, %d, %d], host_q1[0..3]=[%d, %d, %d, %d]\n",
                (int)bufs->host_q0[0], (int)bufs->host_q0[1], (int)bufs->host_q0[2], (int)bufs->host_q0[3],
                (int)bufs->host_q1[0], (int)bufs->host_q1[1], (int)bufs->host_q1[2], (int)bufs->host_q1[3]);
        // Also show what dequantized values would be (INT16 centered at 32768)
        float dq0_0 = (bufs->host_q0[0] + 32768) * params0.scale + params0.zero;
        float dq0_1 = (bufs->host_q0[1] + 32768) * params0.scale + params0.zero;
        float dq1_0 = (bufs->host_q1[0] + 32768) * params1.scale + params1.zero;
        float dq1_1 = (bufs->host_q1[1] + 32768) * params1.scale + params1.zero;
        fprintf(stderr, "QUANT_ALLREDUCE (INT16): dq0[0..1]=[%.6f, %.6f], dq1[0..1]=[%.6f, %.6f]\n",
                dq0_0, dq0_1, dq1_0, dq1_1);
    }

    // Dequant + add on CPU
    dequant_add_cpu(
        bufs->host_q0, params0,
        bufs->host_q1, params1,
        bufs->host_result, n_elements
    );

    if (debug) {
        fprintf(stderr, "QUANT_ALLREDUCE: result[0..3]=[%.6f, %.6f, %.6f, %.6f]\n",
                bufs->host_result[0], bufs->host_result[1],
                bufs->host_result[2], bufs->host_result[3]);

        // Compare with simple FP32 sum (read src values for comparison)
        float orig0[4], orig1[4];
        main_stream->memcpy(orig0, dst_dev0, 4 * sizeof(float)).wait();
        other_stream->memcpy(orig1, src_dev1, 4 * sizeof(float)).wait();
        fprintf(stderr, "QUANT_COMPARE: orig0[0..3]=[%.6f, %.6f, %.6f, %.6f]\n",
                orig0[0], orig0[1], orig0[2], orig0[3]);
        fprintf(stderr, "QUANT_COMPARE: orig1[0..3]=[%.6f, %.6f, %.6f, %.6f]\n",
                orig1[0], orig1[1], orig1[2], orig1[3]);
        fprintf(stderr, "QUANT_COMPARE: expected_sum[0..3]=[%.6f, %.6f, %.6f, %.6f]\n",
                orig0[0]+orig1[0], orig0[1]+orig1[1], orig0[2]+orig1[2], orig0[3]+orig1[3]);
        fprintf(stderr, "QUANT_COMPARE: error[0..3]=[%.6f, %.6f, %.6f, %.6f]\n",
                bufs->host_result[0]-(orig0[0]+orig1[0]),
                bufs->host_result[1]-(orig0[1]+orig1[1]),
                bufs->host_result[2]-(orig0[2]+orig1[2]),
                bufs->host_result[3]-(orig0[3]+orig1[3]));
    }

    // Copy result back to device 0
    ggml_sycl_set_device(main_device);
    main_stream->memcpy(dst_dev0, bufs->host_result, n_elements * sizeof(float)).wait();
}

#endif // GGML_SYCL_QUANTIZED_COMM_HPP
