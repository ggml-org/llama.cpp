//
// cpu-dispatch.cpp — CPU compute path for data-local inference
//
// Executes operations on a CPU SYCL queue when weight data resides in host
// pinned memory (unified cache PINNED_HOST or MMAP tier).  Avoids unnecessary
// host-to-device transfers for layers evicted from VRAM.
//
// MUL_MAT uses quantized vec_dot (e.g., Q4_0×Q8_0) for small M (TG), or
// dnnl_sgemm for larger M (PP) after dequantizing to F32.  Element-wise ops
// (RMS_NORM, ADD, MUL) use portable SYCL parallel_for on the CPU queue.
//
// Supports F32, F16, and quantized types (via dequantize-to-F32 + sgemm).
//
// Activation staging: When compute buffers are device-resident (no HOST_COMPUTE),
// CPU kernels can't access device memory directly.  Double-buffered staging
// copies activations host↔device with SYCL event-based overlap.  Weights are
// already host-pinned so they need no staging.  With GGML_SYCL_HOST_COMPUTE=1,
// compute buffers are host-pinned and staging is bypassed entirely.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "cpu-dispatch.hpp"
#include "unified-cache.hpp"
#include "tensor-types.hpp"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>

#if GGML_SYCL_DNNL
#include "gemm.hpp"          // Provides dnnl.hpp → dnnl_sgemm()
#endif

// ---------------------------------------------------------------------------
// Host pointer registry: stores original mmap pointers for weight tensors.
// Populated during set_tensor (when the host data from the GGUF mmap is still
// available) and read during CPU dispatch to access quantized weight data
// directly without dequantization.
// ---------------------------------------------------------------------------

static std::mutex                                      g_host_ptr_mutex;
static std::unordered_map<std::string, const void *>   g_host_ptr_map;
static bool                                            g_host_ptr_owns_memory = false;

void ggml_sycl_cpu_dispatch_register_host_ptr(const char * name, const void * host_ptr, size_t size) {
    if (!name || !host_ptr || size == 0) return;
    std::lock_guard<std::mutex> lock(g_host_ptr_mutex);

    if (ggml_sycl_cpu_offload_enabled()) {
        // CPU offload mode: copy weight data to persistent host memory.
        // The original mmap pointer may be released by the model loader after
        // set_tensor completes, so we need our own copy for inference.
        // aligned_alloc(64) ensures AVX-512 alignment for vec_dot.
        size_t aligned_size = (size + 63) & ~size_t(63);
        void * copy = aligned_alloc(64, aligned_size);
        if (copy) {
            memcpy(copy, host_ptr, size);
            // Free any previous copy for this tensor
            if (g_host_ptr_owns_memory) {
                auto it = g_host_ptr_map.find(name);
                if (it != g_host_ptr_map.end()) {
                    free(const_cast<void *>(it->second));
                }
            }
            g_host_ptr_map[name] = copy;
            g_host_ptr_owns_memory = true;
        }
    } else {
        g_host_ptr_map[name] = host_ptr;
    }
}

static const void * cpu_dispatch_lookup_host_ptr(const char * name) {
    if (!name) return nullptr;
    std::lock_guard<std::mutex> lock(g_host_ptr_mutex);
    auto it = g_host_ptr_map.find(name);
    return (it != g_host_ptr_map.end()) ? it->second : nullptr;
}

// ---------------------------------------------------------------------------
// Retained activation state: eliminates per-op staging overhead
// ---------------------------------------------------------------------------
//
// When active, CPU op outputs stay in host scratch memory instead of being
// flushed to device. The next CPU op can read them directly without D2H copy.
// Activated at GPU→CPU transitions, flushed at CPU→GPU transitions.

static void *                                          g_retained_scratch     = nullptr;
static size_t                                          g_retained_scratch_cap = 0;
static size_t                                          g_retained_scratch_off = 0;  // bump allocator offset

struct retained_entry {
    void * host_ptr;   // pointer into g_retained_scratch
    size_t size;       // byte size of retained data
};
static std::unordered_map<const ggml_tensor *, retained_entry> g_retained_map;
static bool                                            g_retained_active      = false;
static sycl::queue *                                   g_retained_gpu_q       = nullptr;

// Allocate from scratch buffer (64-byte aligned for AVX-512)
static void * scratch_alloc(size_t size) {
    size_t aligned_off = (g_retained_scratch_off + 63) & ~size_t(63);
    if (aligned_off + size > g_retained_scratch_cap) {
        return nullptr;  // scratch full, fall back to staging
    }
    void * ptr = static_cast<char *>(g_retained_scratch) + aligned_off;
    g_retained_scratch_off = aligned_off + size;
    return ptr;
}

static void scratch_reset() {
    g_retained_scratch_off = 0;
}

static int g_retained_device = -1;

void ggml_sycl_cpu_retained_init(sycl::queue * gpu_q) {
    GGML_ASSERT(!g_retained_scratch || gpu_q == g_retained_gpu_q);
    if (!g_retained_scratch) {
        constexpr size_t DEFAULT_SCRATCH_SIZE = 4 * 1024 * 1024;  // 4MB
        g_retained_scratch = sycl::malloc_host(DEFAULT_SCRATCH_SIZE, *gpu_q);
        if (g_retained_scratch) {
            g_retained_scratch_cap = DEFAULT_SCRATCH_SIZE;
        }
    }
    g_retained_scratch_off = 0;
    g_retained_map.clear();
    g_retained_active = true;
    g_retained_gpu_q  = gpu_q;
}

void ggml_sycl_cpu_retained_init(int device, sycl::queue * gpu_q) {
    ggml_sycl_cpu_retained_init(gpu_q);
    g_retained_device = device;
}

void ggml_sycl_cpu_retained_cleanup() {
    if (g_retained_scratch && g_retained_gpu_q) {
        sycl::free(g_retained_scratch, *g_retained_gpu_q);
        g_retained_scratch     = nullptr;
        g_retained_scratch_cap = 0;
    }
    g_retained_map.clear();
    g_retained_active = false;
    g_retained_gpu_q  = nullptr;
    g_retained_device = -1;
}

bool ggml_sycl_cpu_retained_active() {
    return g_retained_active && g_retained_scratch;
}

void * ggml_sycl_cpu_retained_alloc_output(const ggml_tensor * dst) {
    if (!g_retained_active) return nullptr;

    size_t size = ggml_nbytes(dst);
    void * host_ptr = scratch_alloc(size);
    if (!host_ptr) return nullptr;

    g_retained_map[dst] = { host_ptr, size };
    return host_ptr;
}

void ggml_sycl_cpu_retained_flush_all(int device, sycl::queue * gpu_q) {
    if (g_retained_map.empty()) return;

    std::vector<sycl::event> events;
    events.reserve(g_retained_map.size());

    for (auto & [tensor, entry] : g_retained_map) {
        // Skip host-accessible tensors (no device copy needed)
        if (!tensor->buffer || ggml_backend_buffer_is_host(tensor->buffer)) {
            continue;
        }
        // Use proper device pointer lookup (matches flush_output pattern)
        void * device_ptr = ggml_sycl_get_data_ptr(tensor, device);
        if (!device_ptr) continue;

        events.push_back(
            gpu_q->memcpy(device_ptr, entry.host_ptr, entry.size)
        );
    }

    // Wait for all H2D copies to complete
    if (!events.empty()) {
        sycl::event::wait(events);
    }

    g_retained_map.clear();
    scratch_reset();
}

void ggml_sycl_cpu_retained_deactivate() {
    g_retained_map.clear();
    scratch_reset();
    g_retained_active = false;
}

// ---------------------------------------------------------------------------
// Activation staging: double-buffered host-pinned buffers for device↔host transfer
// ---------------------------------------------------------------------------
//
// GPU compute buffers use sycl::malloc_device() which CPU can't access.
// Weight tensors are already host-pinned (that's why we're here).
// Activation/output tensors need staging: copy device→host before CPU compute,
// then host→device after CPU writes output.
//
// Double-buffered staging with SYCL events (adapted from layer-streaming.hpp):
//   Two banks of 3 slots each (slot 0=src0, 1=src1, 2=dst).
//   Ops alternate between banks.  The previous op's flush can overlap with
//   the current op's stage-in because they use different buffers.
//   Events replace synchronous .wait() — we wait only when data is needed.

static constexpr int STAGING_SLOTS_PER_BANK = 3;
static constexpr int STAGING_BANKS          = 2;

static struct {
    void * ptr = nullptr;
    size_t cap = 0;
} g_cpu_staging[STAGING_BANKS][STAGING_SLOTS_PER_BANK];

static sycl::queue * g_cpu_staging_gpu_q = nullptr;

// Current bank index (alternates per op) and event tracking
static int         g_staging_bank      = 0;
static sycl::event g_staging_flush_evt;     // Event from previous op's flush_output
static bool        g_staging_flush_pending = false;

static void * staging_ensure(int bank, int slot, size_t nbytes, sycl::queue * gpu_q) {
    if (bank < 0 || bank >= STAGING_BANKS || slot < 0 || slot >= STAGING_SLOTS_PER_BANK) {
        return nullptr;
    }
    auto & entry = g_cpu_staging[bank][slot];
    if (nbytes <= entry.cap && entry.ptr) {
        return entry.ptr;
    }
    // Free old buffer using the same queue context it was allocated with.
    if (entry.ptr && g_cpu_staging_gpu_q) {
        GGML_ASSERT(gpu_q == g_cpu_staging_gpu_q && "staging queue changed between calls");
        sycl::free(entry.ptr, *g_cpu_staging_gpu_q);
    }
    g_cpu_staging_gpu_q = gpu_q;
    entry.ptr = sycl::malloc_host(nbytes, *gpu_q);
    entry.cap = nbytes;
    return entry.ptr;
}

// Begin a new staging operation.  Alternates to the next bank and waits for
// the previous op's flush to complete.  Since we alternate banks, the pending
// flush used the OTHER bank.  By waiting here we ensure the global flush
// event is drained before submitting new memcpys to the GPU queue.  The
// staging buffers for the bank we're about to use were last touched 2 ops ago
// and are already safe (waited on by the intervening op).
static void staging_begin_op() {
    g_staging_bank = 1 - g_staging_bank;

    // Drain the previous op's flush (submitted on the other bank).
    if (g_staging_flush_pending) {
        g_staging_flush_evt.wait();
        g_staging_flush_pending = false;
    }
}

// Get host-accessible pointer for a tensor.
// If tensor is in host-accessible memory, returns original pointer.
// For weight tensors: tries host_cache first (AOS data, no device copy needed).
// For activations/compute tensors: copies device→host via staging (event-based).
//
// out_event: if non-null, set to the memcpy event that must complete before
//            reading from the returned pointer.  If no staging was needed,
//            the event is left unchanged.
static void * get_host_ptr(const ggml_tensor * t, int device, int slot,
                           sycl::queue * gpu_q, sycl::event * out_event = nullptr) {
    // Check retained activation map first — if this tensor's data was
    // produced by a prior CPU op in the same layer block, return the
    // host pointer directly without any D2H copy.
    if (g_retained_active) {
        auto it = g_retained_map.find(t);
        if (it != g_retained_map.end()) {
            if (out_event) *out_event = sycl::event{};  // no-op event (already on host)
            return it->second.host_ptr;
        }
    }

    // Host-accessible buffers (weight mmap, host-pinned) → use tensor->data directly
    if (!t->buffer || ggml_backend_buffer_is_host(t->buffer)) {
        return t->data;
    }

    // For weight tensors: look up host-accessible data from cache or mmap.
    if (ggml_sycl_tensor_is_weight(t)) {
        if (ggml_sycl::unified_cache_enabled()) {
            ggml_sycl_cache_id key = ggml_backend_sycl_get_weight_cache_key(t, device);
            if (key.valid) {
                // Try unified cache — PINNED_HOST and MMAP entries are host-accessible.
                auto * cache = ggml_sycl::get_unified_cache_for_device(device);
                if (cache) {
                    ggml_sycl::cache_ptr_view view = cache->get_view(key, GGML_LAYOUT_AOS);
                    if (view.ptr && view.location != ggml_sycl::cache_location::DEVICE) {
                        return view.ptr;
                    }
                }

                // Try host_cache (pinned AOS copies of device-resident weights).
                auto * hcache = ggml_sycl::get_host_cache_for_device(device);
                if (hcache) {
                    int layer_id  = ggml_sycl::extract_layer_id(t->name);
                    int expert_id = ggml_sycl::extract_expert_id(t->name);
                    void * hp = hcache->get(key, ggml_sycl::cache_entry_type::DENSE_WEIGHT,
                                            layer_id, expert_id, GGML_LAYOUT_AOS);
                    if (hp) {
                        return hp;
                    }
                }
            }
        }

        // Fallback: retrieve original mmap host pointer from our static registry.
        // During set_tensor, we store the host data pointer (from the mmap'd GGUF
        // file) before the SYCL backend copies it to device memory.
        if (t->name) {
            const void * mmap_ptr = cpu_dispatch_lookup_host_ptr(t->name);
            if (mmap_ptr) {
                return const_cast<void *>(mmap_ptr);
            }
        }
        return nullptr;
    }

    // Non-weight tensors (activations, compute buffers) → stage device→host.
    void * ptr = ggml_sycl_get_data_ptr(t, device);
    if (!ptr) {
        return nullptr;
    }
    size_t nbytes = ggml_nbytes(t);
    void * host = staging_ensure(g_staging_bank, slot, nbytes, gpu_q);
    if (!host) {
        return nullptr;
    }
    sycl::event evt = gpu_q->memcpy(host, ptr, nbytes);
    if (out_event) {
        *out_event = evt;
    } else {
        // Fallback: if caller doesn't handle events, wait synchronously
        evt.wait();
    }
    return host;
}

// Copy output from host staging back to device memory (event-based).
// No-op if tensor is already in host-accessible memory.
// The flush event is tracked internally and awaited at the start of the next op.
static void flush_output(ggml_tensor * t, int device, sycl::queue * gpu_q) {
    if (!t->buffer || ggml_backend_buffer_is_host(t->buffer)) {
        return;
    }
    void * dev_ptr = ggml_sycl_get_data_ptr(t, device);
    auto & entry   = g_cpu_staging[g_staging_bank][2];
    if (!dev_ptr || !entry.ptr) {
        return;
    }
    size_t nbytes = ggml_nbytes(t);
    g_staging_flush_evt     = gpu_q->memcpy(dev_ptr, entry.ptr, nbytes);
    g_staging_flush_pending = true;
}

// Get host pointer for output tensor.
// Uses staging slot 2 of the current bank.
static void * get_host_output_ptr(ggml_tensor * t, int device, sycl::queue * gpu_q) {
    // Host-accessible buffer → use tensor->data directly
    if (!t->buffer || ggml_backend_buffer_is_host(t->buffer)) {
        return t->data;
    }
    // Device-resident: allocate staging but don't copy (will be written by kernel)
    size_t nbytes = ggml_nbytes(t);
    return staging_ensure(g_staging_bank, 2, nbytes, gpu_q);
}

// Helper: get output pointer from retained scratch or staging fallback.
// Sets *retained to true if output goes to scratch, false for staging.
static void * get_retained_or_staging_output(ggml_tensor * dst, int device,
                                              sycl::queue * gpu_q, bool * retained) {
    *retained = false;
    if (g_retained_active) {
        void * ptr = ggml_sycl_cpu_retained_alloc_output(dst);
        if (ptr) {
            *retained = true;
            return ptr;
        }
    }
    return get_host_output_ptr(dst, device, gpu_q);
}

// Wait for all pending staging events (call at boundary sync points).
void ggml_sycl_cpu_staging_drain() {
    if (g_staging_flush_pending) {
        g_staging_flush_evt.wait();
        g_staging_flush_pending = false;
    }
}

// ---------------------------------------------------------------------------
// MUL_MAT  (oneDNN on CPU queue)
// ---------------------------------------------------------------------------

static bool cpu_mul_mat(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
#if !GGML_SYCL_DNNL
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    return false;
#else
    const ggml_tensor * src0 = dst->src[0];  // weights
    const ggml_tensor * src1 = dst->src[1];  // activations

    if (!src0 || !src1) {
        return false;
    }

    // Activations must be F32, output must be F32
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    // Supported weight types: F32, F16, or any quantized type with to_float
    const bool src0_f32       = (src0->type == GGML_TYPE_F32);
    const bool src0_quantized = !src0_f32 && (src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type));

    if (!src0_f32 && !src0_quantized) {
        return false;
    }

    const auto * type_traits = src0_quantized ? ggml_get_type_traits(src0->type) : nullptr;
    if (src0_quantized && (!type_traits || !type_traits->to_float)) {
        return false;
    }

    // ggml MUL_MAT convention:  C^T = A * B^T  =>  C = B * A^T
    //   src0 = A  (weights)     [ne00 x ne01]  (ne00 = K, ne01 = N)
    //   src1 = B  (activations) [ne10 x ne11]  (ne10 = K, ne11 = M)
    //   dst  = C  (output)      [ne0  x ne1 ]  (ne0  = N, ne1  = M)
    //
    // We need: C[M,N] = src1[M,K] * src0^T[K,N]

    GGML_ASSERT(src0->ne[0] == src1->ne[0]);

    const dnnl_dim_t M = static_cast<dnnl_dim_t>(src1->ne[1]);
    const dnnl_dim_t N = static_cast<dnnl_dim_t>(src0->ne[1]);
    const dnnl_dim_t K = static_cast<dnnl_dim_t>(src0->ne[0]);

    const int device = ctx.device;
    sycl::queue * gpu_q = ctx.stream();

    staging_begin_op();

    // Stage tensors to host-accessible memory (event-based).
    sycl::event e0, e1;
    const void * src0_data = get_host_ptr(src0, device, 0, gpu_q, &e0);
    const void * src1_data = get_host_ptr(src1, device, 1, gpu_q, &e1);

    bool   retained_output;
    void * dst_data = get_retained_or_staging_output(dst, device, gpu_q, &retained_output);

    if (!src0_data || !src1_data || !dst_data) {
        return false;
    }

    // Wait for staging to complete before CPU compute
    e0.wait();
    e1.wait();

    // Batch dimensions (broadcast src0 if ne02/ne03 < ne12/ne13)
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];

    // AOS stride for host data (tensor nb may be SOA, but mmap/cache data is AOS)
    const int64_t nb01 = static_cast<int64_t>(ggml_row_size(src0->type, K));
    const int64_t nb02 = nb01 * src0->ne[1];
    const int64_t nb03 = nb02 * src0->ne[2];
    const int64_t nb12 = src1->nb[2];
    const int64_t nb13 = src1->nb[3];
    const int64_t nb2  = dst->nb[2];
    const int64_t nb3  = dst->nb[3];

    const dnnl_dim_t ldc = static_cast<dnnl_dim_t>(dst->nb[1] / sizeof(float));

    // For small M (TG batch=1..4), use quantized dot product when available.
    // This avoids dequantizing the entire N×K weight matrix to F32 and replaces
    // dnnl_sgemm GEMV with direct quantized vec_dot (e.g., Q4_0 × Q8_0).
    // Benefits: ~5x less memory bandwidth (quantized reads), no BLAS overhead,
    // L1-friendly access pattern (one 2KB weight row + 4KB activation per dot).
    const auto * cpu_traits = src0_quantized ? ggml_get_type_traits_cpu(src0->type) : nullptr;
    const bool   use_vec_dot = (M <= 4 && cpu_traits && cpu_traits->vec_dot);

    ggml_from_float_t from_float_fn = nullptr;
    size_t            q_row_size    = 0;
    std::vector<uint8_t> src1_q_buf;

    if (use_vec_dot) {
        const ggml_type vec_dot_type = cpu_traits->vec_dot_type;
        const auto * vdt_cpu_traits  = ggml_get_type_traits_cpu(vec_dot_type);
        from_float_fn = vdt_cpu_traits ? vdt_cpu_traits->from_float : nullptr;
        if (from_float_fn) {
            q_row_size = ggml_row_size(vec_dot_type, K);
            src1_q_buf.resize(static_cast<size_t>(M) * q_row_size);
        }
    }

    // Dequant/conversion buffer for non-F32 weights (only for GEMM fallback path)
    std::vector<float> src0_f32_buf;
    if (src0_quantized && !use_vec_dot) {
        src0_f32_buf.resize(static_cast<size_t>(N) * K);
    }

    for (int64_t i13 = 0; i13 < ne13; i13++) {
        for (int64_t i12 = 0; i12 < ne12; i12++) {
            const int64_t i02 = i12 % ne02;
            const int64_t i03 = i13 % ne03;

            const char * src0_batch = static_cast<const char *>(src0_data)
                                      + i02 * nb02 + i03 * nb03;
            const float * src1_batch = reinterpret_cast<const float *>(
                static_cast<const char *>(src1_data) + i12 * nb12 + i13 * nb13);
            float * dst_batch = reinterpret_cast<float *>(
                static_cast<char *>(dst_data) + i12 * nb2 + i13 * nb3);

            if (use_vec_dot && from_float_fn) {
                // Quantized dot product path: quantize activations, then vec_dot
                // per output element.  No weight dequantization needed.
                for (dnnl_dim_t m = 0; m < M; m++) {
                    from_float_fn(src1_batch + m * K,
                                  src1_q_buf.data() + m * q_row_size,
                                  K);
                }

                for (dnnl_dim_t n = 0; n < N; n++) {
                    const void * weight_row = src0_batch + n * nb01;
                    for (dnnl_dim_t m = 0; m < M; m++) {
                        float dot_result = 0.0f;
                        cpu_traits->vec_dot(
                            static_cast<int>(K), &dot_result, sizeof(float),
                            weight_row, 0,
                            src1_q_buf.data() + m * q_row_size, 0,
                            1);
                        dst_batch[m * ldc + n] = dot_result;
                    }
                }
            } else {
                // GEMM fallback: dequantize weights to F32, then dnnl_sgemm
                const float * weight_f32;
                dnnl_dim_t    weight_ld = K;

                if (src0_f32) {
                    weight_f32 = reinterpret_cast<const float *>(src0_batch);
                    weight_ld  = static_cast<dnnl_dim_t>(nb01 / sizeof(float));
                } else {
                    for (dnnl_dim_t row = 0; row < N; row++) {
                        const void * row_data = src0_batch + row * nb01;
                        type_traits->to_float(row_data, src0_f32_buf.data() + row * K, K);
                    }
                    weight_f32 = src0_f32_buf.data();
                }

                dnnl_sgemm('T', 'N', N, M, K,
                           1.0f, weight_f32, weight_ld, src1_batch, K,
                           0.0f, dst_batch, ldc);
            }
        }
    }

    // Copy output from host staging back to device compute buffer (async)
    if (!retained_output) {
        flush_output(dst, device, gpu_q);
    }
    return true;
#endif
}

// ---------------------------------------------------------------------------
// RMS_NORM  (SYCL parallel_for on CPU queue)
// ---------------------------------------------------------------------------

static bool cpu_rms_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    sycl::queue * cpu_q = ggml_sycl_get_cpu_queue();
    if (!cpu_q) {
        return false;
    }

    const ggml_tensor * src0 = dst->src[0];
    if (!src0) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(src0)) {
        return false;
    }

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    const int device = ctx.device;
    sycl::queue * gpu_q = ctx.stream();

    staging_begin_op();

    sycl::event e0;
    const float * src_data = static_cast<const float *>(get_host_ptr(src0, device, 0, gpu_q, &e0));

    bool    retained_output;
    float * dst_data = static_cast<float *>(get_retained_or_staging_output(dst, device, gpu_q, &retained_output));

    if (!src_data || !dst_data) {
        return false;
    }

    e0.wait();

    const int64_t ne00  = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    // One work-item per row — each computes RMS and normalizes.
    cpu_q->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(nrows)), [=](sycl::id<1> row_id) {
            const int64_t row     = static_cast<int64_t>(row_id[0]);
            const float * src_row = src_data + row * ne00;
            float *       dst_row = dst_data + row * ne00;

            float sum_sq = 0.0f;
            for (int64_t j = 0; j < ne00; j++) {
                sum_sq += src_row[j] * src_row[j];
            }

            const float scale = 1.0f / sycl::sqrt(sum_sq / static_cast<float>(ne00) + eps);

            for (int64_t j = 0; j < ne00; j++) {
                dst_row[j] = src_row[j] * scale;
            }
        });
    });

    cpu_q->wait();
    if (!retained_output) {
        flush_output(dst, device, gpu_q);
    }
    return true;
}

// ---------------------------------------------------------------------------
// ADD / MUL  (SYCL parallel_for on CPU queue, general ND broadcast)
// ---------------------------------------------------------------------------
//
// General broadcast: src1 dimensions can be 1 where src0 dimensions are > 1.
// The broadcast pattern uses modulo indexing across all 4 dimensions,
// following the same stride-based approach as ggml-cpu/binary-ops.cpp.
// src0 and dst always have the same shape.

enum class binary_op_type { OP_ADD, OP_MUL };

static bool cpu_binary_op(ggml_backend_sycl_context & ctx, ggml_tensor * dst, binary_op_type op) {
    sycl::queue * cpu_q = ggml_sycl_get_cpu_queue();
    if (!cpu_q) {
        return false;
    }

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    if (!src0 || !src1) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    // src0 and dst must be contiguous; src1 must be contiguous along dim 0
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        return false;
    }
    if (src1->nb[0] != sizeof(float)) {
        return false;
    }

    const int device = ctx.device;
    sycl::queue * gpu_q = ctx.stream();

    staging_begin_op();

    sycl::event e0, e1;
    const float * src0_data = static_cast<const float *>(get_host_ptr(src0, device, 0, gpu_q, &e0));
    const float * src1_data = static_cast<const float *>(get_host_ptr(src1, device, 1, gpu_q, &e1));

    bool    retained_output;
    float * dst_data = static_cast<float *>(get_retained_or_staging_output(dst, device, gpu_q, &retained_output));

    if (!src0_data || !src1_data || !dst_data) {
        return false;
    }

    e0.wait();
    e1.wait();

    // dst/src0 dimensions
    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    // src1 dimensions (may be smaller for broadcasting)
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];

    // src1 strides in floats
    const int64_t s11 = src1->nb[1] / sizeof(float);
    const int64_t s12 = src1->nb[2] / sizeof(float);
    const int64_t s13 = src1->nb[3] / sizeof(float);

    // Number of column repetitions within a row
    const int64_t nr0 = ne00 / ne10;

    // Total rows = ne01 * ne02 * ne03
    const int64_t total_rows = ne01 * ne02 * ne03;

    cpu_q->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_rows)), [=](sycl::id<1> row_id) {
            const int64_t ir  = static_cast<int64_t>(row_id[0]);
            const int64_t i03 = ir / (ne02 * ne01);
            const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            const int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

            // Broadcast: map src0 indices to src1 via modulo
            const int64_t i13 = i03 % ne13;
            const int64_t i12 = i02 % ne12;
            const int64_t i11 = i01 % ne11;

            // src0 and dst are contiguous, so row offset is straightforward
            const int64_t src0_row_off = ir * ne00;
            const int64_t dst_row_off  = ir * ne00;

            // src1 uses stride-based indexing (may not be contiguous in higher dims)
            const float * src1_row = src1_data + i13 * s13 + i12 * s12 + i11 * s11;

            const float * sp0 = src0_data + src0_row_off;
            float *       dp  = dst_data  + dst_row_off;

            if (op == binary_op_type::OP_ADD) {
                for (int64_t r = 0; r < nr0; r++) {
                    for (int64_t j = 0; j < ne10; j++) {
                        dp[r * ne10 + j] = sp0[r * ne10 + j] + src1_row[j];
                    }
                }
            } else {
                for (int64_t r = 0; r < nr0; r++) {
                    for (int64_t j = 0; j < ne10; j++) {
                        dp[r * ne10 + j] = sp0[r * ne10 + j] * src1_row[j];
                    }
                }
            }
        });
    });

    cpu_q->wait();
    if (!retained_output) {
        flush_output(dst, device, gpu_q);
    }
    return true;
}

static bool cpu_add(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    return cpu_binary_op(ctx, dst, binary_op_type::OP_ADD);
}

static bool cpu_mul(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    return cpu_binary_op(ctx, dst, binary_op_type::OP_MUL);
}

// ---------------------------------------------------------------------------
// SILU  (x * sigmoid(x), element-wise)
// ---------------------------------------------------------------------------

static bool cpu_silu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    sycl::queue * cpu_q = ggml_sycl_get_cpu_queue();
    if (!cpu_q) {
        return false;
    }

    const ggml_tensor * src0 = dst->src[0];
    if (!src0) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        return false;
    }

    const int device = ctx.device;
    sycl::queue * gpu_q = ctx.stream();

    staging_begin_op();

    sycl::event e0;
    const float * src_data = static_cast<const float *>(get_host_ptr(src0, device, 0, gpu_q, &e0));

    bool    retained_output;
    float * dst_data = static_cast<float *>(get_retained_or_staging_output(dst, device, gpu_q, &retained_output));

    if (!src_data || !dst_data) {
        return false;
    }

    e0.wait();

    const int64_t n = ggml_nelements(dst);

    cpu_q->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(n)), [=](sycl::id<1> i) {
            const float x = src_data[i];
            dst_data[i] = x / (1.0f + sycl::exp(-x));
        });
    });

    cpu_q->wait();
    if (!retained_output) {
        flush_output(dst, device, gpu_q);
    }
    return true;
}

// ---------------------------------------------------------------------------
// GLU  (SWIGLU, REGLU, GEGLU variants — fused gate*up)
// ---------------------------------------------------------------------------

static bool cpu_glu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    sycl::queue * cpu_q = ggml_sycl_get_cpu_queue();
    if (!cpu_q) {
        return false;
    }

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    if (!src0) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        return false;
    }
    if (src1 && !ggml_is_contiguous(src1)) {
        return false;
    }

    const enum ggml_glu_op glu_op = ggml_get_glu_op(dst);
    const int32_t swapped = ((const int32_t *)(dst->op_params))[1];

    const int device = ctx.device;
    sycl::queue * gpu_q = ctx.stream();

    staging_begin_op();

    sycl::event e0, e1;
    const float * src0_data = static_cast<const float *>(get_host_ptr(src0, device, 0, gpu_q, &e0));
    const float * src1_data = src1 ? static_cast<const float *>(get_host_ptr(src1, device, 1, gpu_q, &e1))
                                   : src0_data;

    if (!src0_data || !src1_data) {
        return false;
    }

    bool    retained_output;
    float * dst_data = static_cast<float *>(get_retained_or_staging_output(dst, device, gpu_q, &retained_output));
    if (!dst_data) {
        return false;
    }

    e0.wait();
    if (src1) {
        e1.wait();
    }

    const int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    const int64_t nrows = ggml_nrows(src0);

    // For single-source GLU, gate and up are split halves of src0
    // swapped controls which half is gate vs up
    const int64_t src0_row_stride = src0->nb[1] / sizeof(float);
    const int64_t src1_row_stride = src1 ? (src1->nb[1] / sizeof(float)) : src0_row_stride;
    const int64_t dst_row_stride  = dst->nb[1] / sizeof(float);

    const int64_t gate_offset = (!src1 && swapped) ? nc : 0;
    const int64_t up_offset   = (!src1 && swapped) ? 0 : nc;
    const bool has_src1 = (src1 != nullptr);

    cpu_q->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(nrows * nc)), [=](sycl::id<1> idx) {
            const int64_t row = static_cast<int64_t>(idx[0]) / nc;
            const int64_t col = static_cast<int64_t>(idx[0]) % nc;

            float gate_val, up_val;
            if (has_src1) {
                gate_val = src0_data[row * src0_row_stride + col];
                up_val   = src1_data[row * src1_row_stride + col];
            } else {
                gate_val = src0_data[row * src0_row_stride + gate_offset + col];
                up_val   = src0_data[row * src0_row_stride + up_offset + col];
            }

            float activated;
            switch (glu_op) {
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                    activated = gate_val / (1.0f + sycl::exp(-gate_val));  // silu
                    break;
                case GGML_GLU_OP_REGLU:
                    activated = gate_val > 0.0f ? gate_val : 0.0f;  // relu
                    break;
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_GEGLU_ERF:
                    activated = 0.5f * gate_val * (1.0f + sycl::erf(gate_val * 0.7071067811865475f));  // gelu
                    break;
                case GGML_GLU_OP_GEGLU_QUICK:
                    activated = gate_val / (1.0f + sycl::exp(-1.702f * gate_val));  // quick_gelu
                    break;
                default:
                    activated = gate_val;
                    break;
            }

            dst_data[row * dst_row_stride + col] = activated * up_val;
        });
    });

    cpu_q->wait();
    if (!retained_output) {
        flush_output(dst, device, gpu_q);
    }
    return true;
}

// ---------------------------------------------------------------------------
// SOFT_MAX  (row-wise softmax with scale and optional mask)
// ---------------------------------------------------------------------------

static bool cpu_soft_max(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    sycl::queue * cpu_q = ggml_sycl_get_cpu_queue();
    if (!cpu_q) {
        return false;
    }

    const ggml_tensor * src0 = dst->src[0];
    if (!src0) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        return false;
    }

    // Optional mask (src1) — F32 only for CPU path
    const ggml_tensor * src1 = dst->src[1];
    if (src1 && src1->type != GGML_TYPE_F32) {
        return false;
    }

    float scale    = 1.0f;
    float max_bias = 0.0f;
    memcpy(&scale,    (float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (float *) dst->op_params + 1, sizeof(float));

    // ALiBi not supported on CPU path for simplicity
    if (max_bias != 0.0f) {
        return false;
    }

    const int device = ctx.device;
    sycl::queue * gpu_q = ctx.stream();

    staging_begin_op();

    sycl::event e0, e1;
    const float * src_data  = static_cast<const float *>(get_host_ptr(src0, device, 0, gpu_q, &e0));
    const float * mask_data = src1 ? static_cast<const float *>(get_host_ptr(src1, device, 1, gpu_q, &e1))
                                   : nullptr;

    bool    retained_output;
    float * dst_data = static_cast<float *>(get_retained_or_staging_output(dst, device, gpu_q, &retained_output));

    if (!src_data || !dst_data) {
        return false;
    }

    e0.wait();
    if (src1) {
        e1.wait();
    }

    const int64_t ne00  = src0->ne[0];  // row width
    const int64_t nrows = ggml_nrows(src0);

    const int64_t mask_ne1 = src1 ? src1->nb[1] / sizeof(float) : 0;

    cpu_q->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(nrows)), [=](sycl::id<1> row_id) {
            const int64_t row = static_cast<int64_t>(row_id[0]);
            const float * sp  = src_data + row * ne00;
            float *       dp  = dst_data + row * ne00;

            // Mask row — broadcast across rows within each head
            const float * mp = nullptr;
            if (mask_data) {
                const int64_t mask_row = row % (mask_ne1 > 0 ? (nrows) : 1);
                mp = mask_data + mask_row * ne00;
            }

            // 1. Scale + mask + find max
            float max_val = -INFINITY;
            for (int64_t j = 0; j < ne00; j++) {
                float v = sp[j] * scale;
                if (mp) {
                    v += mp[j];
                }
                dp[j] = v;
                if (v > max_val) {
                    max_val = v;
                }
            }

            // 2. exp(x - max) and sum
            float sum = 0.0f;
            for (int64_t j = 0; j < ne00; j++) {
                dp[j] = sycl::exp(dp[j] - max_val);
                sum += dp[j];
            }

            // 3. Normalize
            const float inv_sum = 1.0f / sum;
            for (int64_t j = 0; j < ne00; j++) {
                dp[j] *= inv_sum;
            }
        });
    });

    cpu_q->wait();
    if (!retained_output) {
        flush_output(dst, device, gpu_q);
    }
    return true;
}

// ---------------------------------------------------------------------------
// NORM  (layer normalization: mean-subtract, variance-normalize)
// ---------------------------------------------------------------------------

static bool cpu_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    sycl::queue * cpu_q = ggml_sycl_get_cpu_queue();
    if (!cpu_q) {
        return false;
    }

    const ggml_tensor * src0 = dst->src[0];
    if (!src0) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(src0)) {
        return false;
    }

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    const int device = ctx.device;
    sycl::queue * gpu_q = ctx.stream();

    staging_begin_op();

    sycl::event e0;
    const float * src_data = static_cast<const float *>(get_host_ptr(src0, device, 0, gpu_q, &e0));

    bool    retained_output;
    float * dst_data = static_cast<float *>(get_retained_or_staging_output(dst, device, gpu_q, &retained_output));

    if (!src_data || !dst_data) {
        return false;
    }

    e0.wait();

    const int64_t ne00  = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    cpu_q->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(nrows)), [=](sycl::id<1> row_id) {
            const int64_t row     = static_cast<int64_t>(row_id[0]);
            const float * src_row = src_data + row * ne00;
            float *       dst_row = dst_data + row * ne00;

            // Compute mean
            float sum = 0.0f;
            for (int64_t j = 0; j < ne00; j++) {
                sum += src_row[j];
            }
            const float mean = sum / static_cast<float>(ne00);

            // Compute variance and mean-subtract
            float var = 0.0f;
            for (int64_t j = 0; j < ne00; j++) {
                float d = src_row[j] - mean;
                dst_row[j] = d;
                var += d * d;
            }
            var /= static_cast<float>(ne00);

            // Normalize
            const float scale = 1.0f / sycl::sqrt(var + eps);
            for (int64_t j = 0; j < ne00; j++) {
                dst_row[j] *= scale;
            }
        });
    });

    cpu_q->wait();
    if (!retained_output) {
        flush_output(dst, device, gpu_q);
    }
    return true;
}

// ---------------------------------------------------------------------------
// SCALE  (multiply all elements by a scalar)
// ---------------------------------------------------------------------------

static bool cpu_scale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    sycl::queue * cpu_q = ggml_sycl_get_cpu_queue();
    if (!cpu_q) {
        return false;
    }

    const ggml_tensor * src0 = dst->src[0];
    if (!src0) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        return false;
    }

    float scale;
    memcpy(&scale, dst->op_params, sizeof(float));

    const int device = ctx.device;
    sycl::queue * gpu_q = ctx.stream();

    staging_begin_op();

    sycl::event e0;
    const float * src_data = static_cast<const float *>(get_host_ptr(src0, device, 0, gpu_q, &e0));

    bool    retained_output;
    float * dst_data = static_cast<float *>(get_retained_or_staging_output(dst, device, gpu_q, &retained_output));

    if (!src_data || !dst_data) {
        return false;
    }

    e0.wait();

    const int64_t n = ggml_nelements(dst);

    cpu_q->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(n)), [=](sycl::id<1> i) {
            dst_data[i] = src_data[i] * scale;
        });
    });

    cpu_q->wait();
    if (!retained_output) {
        flush_output(dst, device, gpu_q);
    }
    return true;
}

// ---------------------------------------------------------------------------
// CPY / CONT  (copy or contiguify tensor data)
// ---------------------------------------------------------------------------

static bool cpu_cpy(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    sycl::queue * cpu_q = ggml_sycl_get_cpu_queue();
    if (!cpu_q) {
        return false;
    }

    const ggml_tensor * src0 = dst->src[0];
    if (!src0) {
        return false;
    }

    // Simple case: both contiguous, same type → memcpy
    if (src0->type != dst->type) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        return false;
    }

    const int device = ctx.device;
    sycl::queue * gpu_q = ctx.stream();

    staging_begin_op();

    sycl::event e0;
    const void * src_data = get_host_ptr(src0, device, 0, gpu_q, &e0);

    bool   retained_output;
    void * dst_data = get_retained_or_staging_output(dst, device, gpu_q, &retained_output);

    if (!src_data || !dst_data) {
        return false;
    }

    e0.wait();

    const size_t nbytes = ggml_nbytes(dst);
    memcpy(dst_data, src_data, nbytes);

    if (!retained_output) {
        flush_output(dst, device, gpu_q);
    }
    return true;
}

// ---------------------------------------------------------------------------
// ROPE  (rotary positional embeddings — NEOX and NORMAL modes, F32 only)
// ---------------------------------------------------------------------------

static bool cpu_rope(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    sycl::queue * cpu_q = ggml_sycl_get_cpu_queue();
    if (!cpu_q) {
        return false;
    }

    const ggml_tensor * src0 = dst->src[0];  // input tensor
    const ggml_tensor * src1 = dst->src[1];  // positions (int32)
    const ggml_tensor * src2 = dst->src[2];  // freq_factors (optional)

    if (!src0 || !src1) {
        return false;
    }
    // F32 only for CPU path
    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    // Extract op_params
    const int n_dims     = ((int32_t *) dst->op_params)[1];
    const int mode       = ((int32_t *) dst->op_params)[2];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];

    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    memcpy(&freq_base,   (int32_t *) dst->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));

    // Only support NORMAL and NEOX modes on CPU
    const bool is_neox   = (mode & GGML_ROPE_TYPE_NEOX) != 0;
    const bool is_normal = (mode == GGML_ROPE_TYPE_NORMAL);
    if (!is_neox && !is_normal) {
        return false;  // MROPE, VISION, IMROPE not supported
    }

    const int device = ctx.device;
    sycl::queue * gpu_q = ctx.stream();

    staging_begin_op();

    sycl::event e0, e1;
    const float *   src_data = static_cast<const float *>(get_host_ptr(src0, device, 0, gpu_q, &e0));
    const int32_t * pos_data = static_cast<const int32_t *>(get_host_ptr(src1, device, 1, gpu_q, &e1));

    bool    retained_output;
    float * dst_data = static_cast<float *>(get_retained_or_staging_output(dst, device, gpu_q, &retained_output));

    if (!src_data || !pos_data || !dst_data) {
        return false;
    }

    e0.wait();
    e1.wait();

    // Freq factors (optional src2) — must be host-accessible (no staging slot available)
    const float * freq_factors_data = nullptr;
    if (src2) {
        void * src2_ptr = ggml_sycl_get_data_ptr(src2, device);
        if (!src2_ptr) {
            return false;
        }
        // Only handle host-accessible freq_factors (no staging available for 4th tensor)
        if (src2->buffer && !ggml_backend_buffer_is_host(src2->buffer)) {
            return false;
        }
        freq_factors_data = static_cast<const float *>(src2_ptr);
    }

    const int64_t ne0 = src0->ne[0];  // head dim
    const int64_t ne1 = src0->ne[1];  // num heads
    const int64_t ne2 = src0->ne[2];  // seq len
    const int64_t ne3 = src0->ne[3];  // batch

    // Strides in float units
    const int64_t s01 = src0->nb[1] / sizeof(float);
    const int64_t s02 = src0->nb[2] / sizeof(float);
    const int64_t s03 = src0->nb[3] / sizeof(float);
    const int64_t d01 = dst->nb[1] / sizeof(float);
    const int64_t d02 = dst->nb[2] / sizeof(float);
    const int64_t d03 = dst->nb[3] / sizeof(float);

    // YaRN correction dims
    float corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    // Total work items = ne3 * ne2 * ne1 (one per row)
    const int64_t total_rows = ne3 * ne2 * ne1;

    // Capture locals for kernel
    const float cd0 = corr_dims[0];
    const float cd1 = corr_dims[1];

    cpu_q->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(total_rows)), [=](sycl::id<1> work_id) {
            const int64_t idx = static_cast<int64_t>(work_id[0]);
            const int64_t i3  = idx / (ne2 * ne1);
            const int64_t i2  = (idx / ne1) % ne2;
            const int64_t i1  = idx % ne1;

            const float * src_row = src_data + i3 * s03 + i2 * s02 + i1 * s01;
            float *       dst_row = dst_data + i3 * d03 + i2 * d02 + i1 * d01;

            const int32_t p = pos_data[i2];
            const float theta_base = static_cast<float>(p);

            float theta = theta_base;
            for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
                const float ff = freq_factors_data ? freq_factors_data[i0 / 2] : 1.0f;

                // YaRN rope_yarn inline
                const float theta_extrap = theta / ff;
                float theta_interp = freq_scale * theta_extrap;
                float theta_val = theta_interp;
                float mscale = attn_factor;

                if (ext_factor != 0.0f) {
                    const float y = (i0 / 2.0f - cd0) / sycl::fmax(0.001f, cd1 - cd0);
                    const float ramp_mix = (1.0f - sycl::fmin(1.0f, sycl::fmax(0.0f, y))) * ext_factor;
                    theta_val = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
                    mscale *= 1.0f + 0.1f * sycl::log(1.0f / freq_scale);
                }

                const float cos_theta = sycl::cos(theta_val) * mscale;
                const float sin_theta = sycl::sin(theta_val) * mscale;

                if (is_normal) {
                    // NORMAL: pairs are adjacent (i0, i0+1)
                    const float x0 = src_row[i0];
                    const float x1 = src_row[i0 + 1];
                    dst_row[i0]     = x0 * cos_theta - x1 * sin_theta;
                    dst_row[i0 + 1] = x0 * sin_theta + x1 * cos_theta;
                } else {
                    // NEOX: pairs are (i0/2, i0/2 + n_dims/2)
                    const int64_t ic = i0 / 2;
                    const float x0 = src_row[ic];
                    const float x1 = src_row[ic + n_dims / 2];
                    dst_row[ic]              = x0 * cos_theta - x1 * sin_theta;
                    dst_row[ic + n_dims / 2] = x0 * sin_theta + x1 * cos_theta;
                }

                theta *= theta_scale;
            }

            // Copy remaining dimensions beyond n_dims
            if (!is_normal) {
                for (int64_t i0 = n_dims; i0 < ne0; i0++) {
                    dst_row[i0] = src_row[i0];
                }
            } else {
                for (int64_t i0 = n_dims; i0 < ne0; i0++) {
                    dst_row[i0] = src_row[i0];
                }
            }
        });
    });

    cpu_q->wait();
    if (!retained_output) {
        flush_output(dst, device, gpu_q);
    }
    return true;
}

// ---------------------------------------------------------------------------
// UNARY dispatch (SILU and others)
// ---------------------------------------------------------------------------

static bool cpu_unary(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const enum ggml_unary_op op = ggml_get_unary_op(dst);
    switch (op) {
        case GGML_UNARY_OP_SILU:
            return cpu_silu(ctx, dst);
        default:
            GGML_SYCL_DEBUG("[SYCL-CPU] Unsupported unary op %d on CPU\n", static_cast<int>(op));
            return false;
    }
}


// ---------------------------------------------------------------------------
// Fused RMS_NORM + MUL  (single staging pass, saves 1 flush + 1 stage-in)
// ---------------------------------------------------------------------------
//
// Pattern: rms_dst = RMS_NORM(x), mul_dst = MUL(rms_dst, w)
// Fused:   mul_dst[j] = x[j] * rms_scale * w[j]  (no intermediate flush)
//
// Saves 2 staging transfers per fusion (2x per transformer layer):
//   - RMS_NORM output flush to device
//   - MUL input re-stage of that same data

bool ggml_sycl_compute_fused_rms_norm_mul(ggml_backend_sycl_context & ctx,
                                           ggml_tensor * rms_dst, ggml_tensor * mul_dst) {
    const ggml_tensor * rms_src0 = rms_dst->src[0];  // input to normalize
    const ggml_tensor * mul_src1 = mul_dst->src[1];   // element-wise weight

    if (!rms_src0 || !mul_src1) {
        return false;
    }
    if (rms_src0->type != GGML_TYPE_F32 || rms_dst->type != GGML_TYPE_F32 ||
        mul_src1->type != GGML_TYPE_F32 || mul_dst->type  != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(rms_src0) || !ggml_is_contiguous(mul_dst)) {
        return false;
    }

    float eps;
    memcpy(&eps, rms_dst->op_params, sizeof(float));

    const int device = ctx.device;
    sycl::queue * gpu_q = ctx.stream();

    staging_begin_op();

    // Stage: rms input (slot 0) + mul weight (slot 1) + mul output (slot 2)
    sycl::event e0, e1;
    const float * rms_in_data = static_cast<const float *>(get_host_ptr(rms_src0, device, 0, gpu_q, &e0));
    const float * mul_wt_data = static_cast<const float *>(get_host_ptr(mul_src1, device, 1, gpu_q, &e1));

    bool    retained_output;
    float * out_data = static_cast<float *>(get_retained_or_staging_output(mul_dst, device, gpu_q, &retained_output));

    if (!rms_in_data || !mul_wt_data || !out_data) {
        return false;
    }

    e0.wait();
    e1.wait();

    const int64_t ne00  = rms_src0->ne[0];
    const int64_t nrows = ggml_nrows(rms_src0);

    // mul_src1 dimensions for broadcasting
    const int64_t ne10 = mul_src1->ne[0];
    const int64_t ne11 = mul_src1->ne[1];
    const int64_t s11  = mul_src1->nb[1] / sizeof(float);

    for (int64_t row = 0; row < nrows; row++) {
        const float * src_row = rms_in_data + row * ne00;
        float *       dst_row = out_data    + row * ne00;

        // RMS normalization
        float sum_sq = 0.0f;
        for (int64_t j = 0; j < ne00; j++) {
            sum_sq += src_row[j] * src_row[j];
        }
        const float scale = 1.0f / sqrtf(sum_sq / static_cast<float>(ne00) + eps);

        // Fused multiply: rms_norm(x) * weight
        const int64_t wt_row_idx = row % ne11;
        const float * wt_row = mul_wt_data + wt_row_idx * s11;
        const int64_t nr0 = ne00 / ne10;
        for (int64_t r = 0; r < nr0; r++) {
            for (int64_t j = 0; j < ne10; j++) {
                dst_row[r * ne10 + j] = src_row[r * ne10 + j] * scale * wt_row[j];
            }
        }
    }

    if (!retained_output) {
        flush_output(mul_dst, device, gpu_q);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Fused ADD + RMS_NORM  (single staging pass, saves 1 stage-in)
// ---------------------------------------------------------------------------
//
// Pattern: add_dst = ADD(a, b), rms_dst = RMS_NORM(add_dst)
// Both outputs needed: add_dst is the residual (consumed downstream),
// rms_dst is the normalized input to attention/FFN.
//
// Saves 1 staging transfer per fusion (1x per transformer layer):
//   - RMS_NORM re-staging of add_dst from device

bool ggml_sycl_compute_fused_add_rms_norm(ggml_backend_sycl_context & ctx,
                                            ggml_tensor * add_dst, ggml_tensor * rms_dst) {
    const ggml_tensor * add_src0 = add_dst->src[0];
    const ggml_tensor * add_src1 = add_dst->src[1];
    const ggml_tensor * rms_src0 = rms_dst->src[0];  // should == add_dst

    if (!add_src0 || !add_src1 || rms_src0 != add_dst) {
        return false;
    }
    if (add_src0->type != GGML_TYPE_F32 || add_src1->type != GGML_TYPE_F32 ||
        add_dst->type  != GGML_TYPE_F32 || rms_dst->type  != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(add_src0) || !ggml_is_contiguous(add_dst) ||
        !ggml_is_contiguous(rms_dst)) {
        return false;
    }
    if (add_src1->nb[0] != sizeof(float)) {
        return false;
    }

    float eps;
    memcpy(&eps, rms_dst->op_params, sizeof(float));

    const int device = ctx.device;
    sycl::queue * gpu_q = ctx.stream();

    staging_begin_op();

    // Stage add inputs (slots 0, 1) + add output (slot 2)
    sycl::event e0, e1;
    const float * a_data = static_cast<const float *>(get_host_ptr(add_src0, device, 0, gpu_q, &e0));
    const float * b_data = static_cast<const float *>(get_host_ptr(add_src1, device, 1, gpu_q, &e1));

    // Output: use retained scratch for both add_dst and rms_dst if active
    bool    retained_add;
    float * add_out = static_cast<float *>(get_retained_or_staging_output(add_dst, device, gpu_q, &retained_add));

    if (!a_data || !b_data || !add_out) {
        return false;
    }

    e0.wait();
    e1.wait();

    const int64_t ne00 = add_src0->ne[0];
    const int64_t ne01 = add_src0->ne[1];
    const int64_t ne02 = add_src0->ne[2];
    const int64_t ne03 = add_src0->ne[3];
    const int64_t ne10 = add_src1->ne[0];
    const int64_t ne11 = add_src1->ne[1];
    const int64_t ne12 = add_src1->ne[2];
    const int64_t ne13 = add_src1->ne[3];
    const int64_t s11  = add_src1->nb[1] / sizeof(float);
    const int64_t s12  = add_src1->nb[2] / sizeof(float);
    const int64_t s13  = add_src1->nb[3] / sizeof(float);
    const int64_t nr0  = ne00 / ne10;

    // rms output: use retained scratch if active, else staging slot 0.
    // Cannot use the helper here because slot 2 may already be used for add_dst.
    const size_t rms_nbytes = ggml_nbytes(rms_dst);
    float * rms_out;
    bool    retained_rms = false;
    if (g_retained_active) {
        rms_out = static_cast<float *>(ggml_sycl_cpu_retained_alloc_output(rms_dst));
        retained_rms = (rms_out != nullptr);
    }
    if (!retained_rms) {
        rms_out = static_cast<float *>(staging_ensure(g_staging_bank, 0, rms_nbytes, gpu_q));
    }
    if (!rms_out) {
        return false;
    }

    const int64_t total_rows = ne01 * ne02 * ne03;

    for (int64_t ir = 0; ir < total_rows; ir++) {
        const int64_t i03 = ir / (ne02 * ne01);
        const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
        const int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

        const int64_t i13 = i03 % ne13;
        const int64_t i12 = i02 % ne12;
        const int64_t i11 = i01 % ne11;

        const float * a_row = a_data + ir * ne00;
        const float * b_row = b_data + i13 * s13 + i12 * s12 + i11 * s11;
        float * add_row     = add_out + ir * ne00;
        float * rms_row     = rms_out + ir * ne00;

        // Fused ADD + RMS_NORM
        float sum_sq = 0.0f;
        for (int64_t r = 0; r < nr0; r++) {
            for (int64_t j = 0; j < ne10; j++) {
                const float v = a_row[r * ne10 + j] + b_row[j];
                add_row[r * ne10 + j] = v;
                sum_sq += v * v;
            }
        }

        const float scale = 1.0f / sqrtf(sum_sq / static_cast<float>(ne00) + eps);
        for (int64_t j = 0; j < ne00; j++) {
            rms_row[j] = add_row[j] * scale;
        }
    }

    // Flush add_dst from staging slot 2
    if (!retained_add) {
        flush_output(add_dst, device, gpu_q);
    }

    // Flush rms_dst from staging slot 0 (or retained scratch)
    if (!retained_rms) {
        if (!rms_dst->buffer || ggml_backend_buffer_is_host(rms_dst->buffer)) {
            // Host-accessible: copy directly
            memcpy(rms_dst->data, rms_out, rms_nbytes);
        } else {
            void * rms_dev_ptr = ggml_sycl_get_data_ptr(rms_dst, device);
            if (rms_dev_ptr) {
                // In-order queue: this is sequenced after the add flush
                g_staging_flush_evt     = gpu_q->memcpy(rms_dev_ptr, rms_out, rms_nbytes);
                g_staging_flush_pending = true;
            }
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// Main dispatch entry point
// ---------------------------------------------------------------------------

bool ggml_sycl_compute_forward_cpu(ggml_backend_sycl_context & ctx, struct ggml_tensor * dst) {
    GGML_SYCL_DEBUG("[CPU-FWD] op=%s name=%s\n", ggml_op_name(dst->op), dst->name ? dst->name : "(null)");
    switch (dst->op) {
        case GGML_OP_MUL_MAT:
            return cpu_mul_mat(ctx, dst);
        case GGML_OP_RMS_NORM:
            return cpu_rms_norm(ctx, dst);
        case GGML_OP_ADD:
            return cpu_add(ctx, dst);
        case GGML_OP_MUL:
            return cpu_mul(ctx, dst);
        case GGML_OP_UNARY:
            return cpu_unary(ctx, dst);
        case GGML_OP_GLU:
            return cpu_glu(ctx, dst);
        case GGML_OP_SOFT_MAX:
            return cpu_soft_max(ctx, dst);
        case GGML_OP_NORM:
            return cpu_norm(ctx, dst);
        case GGML_OP_SCALE:
            return cpu_scale(ctx, dst);
        case GGML_OP_CPY:
        case GGML_OP_CONT:
            return cpu_cpy(ctx, dst);
        case GGML_OP_ROPE:
            return cpu_rope(ctx, dst);
        default:
            GGML_SYCL_DEBUG("[SYCL-CPU] Unsupported op %s on CPU, falling back to GPU\n",
                            ggml_op_name(dst->op));
            return false;
    }
}
