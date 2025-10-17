#include "ggml-rknpu2.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-quants.h"

#include "dma-alloc.h"

#include <rknn_api.h>
#include <rknn_matmul_api.h>

#include <arm_neon.h>
#include <omp.h>

#include <chrono>
#include <cassert>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>
#include <map>
#include <tuple>
#include <algorithm>
#include <memory>
#include <unordered_map>

#define UNUSED(x) (void)(x)

// Performance logging flag
#define RKNPU_PERF_LOG 0

// Backend logging flag
#define RKNPU_DEBUG 0

// RAII-struct for performance logging
#if RKNPU_PERF_LOG
struct RknpuPerfLogger {
    std::string name;
    std::chrono::high_resolution_clock::time_point start;

    RknpuPerfLogger(std::string name) : name(std::move(name)), start(std::chrono::high_resolution_clock::now()) {}

    ~RknpuPerfLogger() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        fprintf(stderr, "[PERF] %-40s: %8lld us\n", name.c_str(), (long long)duration);
    }
};
#endif

#if RKNPU_DEBUG
#define RKNPU_LOG_INFO(...) fprintf(stderr, __VA_ARGS__)
#else
#define RKNPU_LOG_INFO(...)
#endif

#define RKNPU_LOG_ERROR(...) fprintf(stderr, __VA_ARGS__)

// Macro for RKNN API calls
#define RKNN_CHECK(stmt, msg)                                           \
    do {                                                                \
        int ret = (stmt);                                               \
        if (ret < 0) {                                                  \
            RKNPU_LOG_ERROR("RKNN error %d at %s:%d: %s\n", ret,        \
                __FILE__, __LINE__, msg);                               \
            assert(false);                                              \
        }                                                               \
    } while (0)


// --- Helper functions ---

// Packing KxN FP16 (row-major: idx [k,n] -> k*N + n) into native RKNN: (N/16, K/32, 16, 32)
static void pack_B_segment_fp16_native_from_KN(
    uint16_t * dst, const uint16_t * src,
    int K, int N_total, int n_offset, int n_segment) {

    GGML_ASSERT(K % 32 == 0 && N_total > 0 && K > 0);
    GGML_ASSERT(n_offset % 16 == 0 && n_segment % 16 == 0 && n_offset + n_segment <= N_total);

    const size_t s0 = (size_t)(K / 32) * 16 * 32;
    const size_t s1 = 16 * 32;
    const size_t s2 = 32;

    for (int i = 0; i < n_segment / 16; ++i) {
        for (int j = 0; j < K / 32; ++j) {
            const size_t dst_block = (size_t) i * s0 + (size_t) j * s1;
            for (int ii = 0; ii < 16; ++ii) {
                const size_t n_global = (size_t)n_offset + (size_t)i * 16 + (size_t)ii;
                
                const uint16_t * src_ptr = src + n_global * K + j * 32;
                uint16_t * dst_ptr = dst + dst_block + ii * s2;

                uint16x8_t d0 = vld1q_u16(src_ptr + 0);
                uint16x8_t d1 = vld1q_u16(src_ptr + 8);
                uint16x8_t d2 = vld1q_u16(src_ptr + 16);
                uint16x8_t d3 = vld1q_u16(src_ptr + 24);

                vst1q_u16(dst_ptr + 0, d0);
                vst1q_u16(dst_ptr + 8, d1);
                vst1q_u16(dst_ptr + 16, d2);
                vst1q_u16(dst_ptr + 24, d3);
            }
        }
    }
}

// --- Hashers ---

// Function for hash combinations
template <class T>
inline void hash_combine(std::size_t& seed, const T& v) {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// Hasher for std::pair
struct PairHasher {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        std::size_t seed = 0;
        hash_combine(seed, p.first);
        hash_combine(seed, p.second);
        return seed;
    }
};

// Hasher for std::tuple
struct TupleHasher {
    template <typename... Ts>
    std::size_t operator()(const std::tuple<Ts...>& t) const {
        std::size_t seed = 0;
        std::apply([&](const auto&... args) {
            (hash_combine(seed, args), ...);
        }, t);
        return seed;
    }
};

// --- Segmenters ---

// Matrix segment information
struct MatrixSegment {
    int offset_n;  // Segment offset
    int size_n;    // Segment size
    int core_id;   // Segment core ID
};

// Split B-matrix into segments
static std::vector<MatrixSegment> compute_matrix_segments(int N, int num_cores = 3) {
    std::vector<MatrixSegment> segments;
    
    // N divisible by 16 for F16
    const int alignment = 16;
    
    // Basic segment size
    int base_segment_size = (N / num_cores / alignment) * alignment;
    int remaining = N - (base_segment_size * num_cores);
    
    int offset = 0;
    for (int i = 0; i < num_cores; i++) {
        MatrixSegment seg;
        seg.offset_n = offset;
        seg.size_n = base_segment_size;
        seg.core_id = i;
        
        if (i < remaining / alignment) {
            seg.size_n += alignment;
        }
        
        offset += seg.size_n;
        segments.push_back(seg);
    }
    
    return segments;
}

// --- Structs ---

// RKNN buffer context
struct ggml_backend_rknpu_buffer_context {
    rknn_tensor_mem * rknn_mem;
    DmaBuffer dma_buf;
    std::string name;
};

// RKNN matmul operation context
struct rknpu_matmul_context {
    rknn_matmul_info info;
    rknn_matmul_io_attr io_attr;
    rknn_matmul_ctx ctx = 0;

    rknpu_matmul_context(int M, int K, int N, rknn_matmul_type type) {
        memset(&info, 0, sizeof(info));
        info.M = M;
        info.K = K;
        info.N = N;
        info.type = type;
        info.B_layout = RKNN_MM_LAYOUT_NATIVE;
        info.AC_layout = RKNN_MM_LAYOUT_NORM;

        int ret = rknn_matmul_create(&ctx, &info, &io_attr);
        if (ret < 0) {
            RKNPU_LOG_ERROR("rknn_matmul_create failed for %dx%dx%d\n", M, K, N);
            ctx = 0;
        }
    }

    ~rknpu_matmul_context() {
        if (ctx != 0) {
            rknn_matmul_destroy(ctx);
        }
    }
};


// Backend main context
struct ggml_backend_rknpu_context {
    std::string name;
    std::mutex mutex;
    
    // RKNN matmul contexts cache
    std::unordered_map<std::tuple<int, int, int, int>, std::shared_ptr<rknpu_matmul_context>, TupleHasher> matmul_ctx_cache;
    
    // B-matrices cache
    std::unordered_map<std::tuple<const ggml_tensor*, int>, std::shared_ptr<rknn_tensor_mem>, TupleHasher> b_npu_buffer_cache;

    // A- and C-matrices cache
    std::unordered_map<std::tuple<int, int, int>, std::shared_ptr<rknn_tensor_mem>, TupleHasher> a_buffer_cache;
    std::unordered_map<std::tuple<int, int, int>, std::shared_ptr<rknn_tensor_mem>, TupleHasher> c_buffer_cache;

    std::shared_ptr<rknpu_matmul_context> get_matmul_ctx(int M, int K, int N, int core_id) {
        std::lock_guard<std::mutex> lock(mutex);
        auto key = std::make_tuple(M, K, N, core_id);
        auto it = matmul_ctx_cache.find(key);
        if (it != matmul_ctx_cache.end()) {
            return it->second;
        }
        auto ctx = std::make_shared<rknpu_matmul_context>(M, K, N, RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32);
        if (ctx->ctx == 0) {
            return nullptr;
        }

        rknn_core_mask core_mask;
        switch(core_id) {
            case 0: core_mask = RKNN_NPU_CORE_0; break;
            case 1: core_mask = RKNN_NPU_CORE_1; break;
            case 2: core_mask = RKNN_NPU_CORE_2; break;
            default: core_mask = RKNN_NPU_CORE_AUTO; break;
        }
        
        int ret = rknn_matmul_set_core_mask(ctx->ctx, core_mask);
        if (ret != RKNN_SUCC) {
            RKNPU_LOG_ERROR("Failed to set core mask %d for core %d\n", core_mask, core_id);
        }
        
        matmul_ctx_cache[key] = ctx;
        return ctx;
    }
};

// RKNN memory global context
struct rknpu_memory_context {
    rknn_matmul_ctx mem_ctx = 0;
    std::mutex mutex;

    rknpu_memory_context() {
        rknn_matmul_info dummy_info;
        memset(&dummy_info, 0, sizeof(dummy_info));
        dummy_info.M = 32;
        dummy_info.K = 32;
        dummy_info.N = 32;
        dummy_info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;

        rknn_matmul_io_attr dummy_io_attr;
        int ret = rknn_matmul_create(&mem_ctx, &dummy_info, &dummy_io_attr);
        if (ret < 0) {
            RKNPU_LOG_ERROR("Failed to create dummy matmul context for memory operations: %d\n", ret);
            mem_ctx = 0;
        }
    }

    ~rknpu_memory_context() {
        if (mem_ctx != 0) {
            rknn_matmul_destroy(mem_ctx);
        }
    }

    rknn_matmul_ctx get_ctx() {
        std::lock_guard<std::mutex> lock(mutex);
        return mem_ctx;
    }
};

static rknpu_memory_context & get_rknpu_memory_context() {
    static rknpu_memory_context g_mem_ctx;
    return g_mem_ctx;
}


//
// Backend
//

static const char * ggml_backend_rknpu_name(ggml_backend_t backend) {
    UNUSED(backend);
    return "RKNPU";
}

static void ggml_backend_rknpu_free(ggml_backend_t backend) {
    ggml_backend_rknpu_context * ctx = (ggml_backend_rknpu_context *)backend->context;
    delete ctx;
    delete backend;
}

// Function for getting buffer from cache or creating new one
static std::shared_ptr<rknn_tensor_mem> get_or_create_npu_buffer(
    ggml_backend_rknpu_context* backend_ctx,
    rknn_matmul_ctx matmul_ctx,
    size_t size,
    const std::tuple<int, int, int>& key,
    std::unordered_map<std::tuple<int, int, int>, std::shared_ptr<rknn_tensor_mem>, TupleHasher>& cache
) {
    std::lock_guard<std::mutex> lock(backend_ctx->mutex);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }

    rknn_tensor_mem* mem = rknn_create_mem(matmul_ctx, size);
    if (!mem) { return nullptr; }

    auto mem_ctx_for_deleter = get_rknpu_memory_context().get_ctx();
    auto deleter = [mem_ctx_for_deleter](rknn_tensor_mem* m) {
        if (m && mem_ctx_for_deleter != 0) {
            rknn_destroy_mem(mem_ctx_for_deleter, m);
        }
    };
    
    std::shared_ptr<rknn_tensor_mem> mem_shared(mem, deleter);
    cache[key] = mem_shared;
    return mem_shared;
}

static enum ggml_status ggml_backend_rknpu_graph_compute(ggml_backend_t backend, struct ggml_cgraph* cgraph) {
    auto* backend_ctx = (ggml_backend_rknpu_context*)backend->context;
    const int NUM_CORES = 3;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor* node = cgraph->nodes[i];
        if (node->op != GGML_OP_MUL_MAT) continue;

        #if RKNPU_PERF_LOG
        std::string node_name = std::string(node->name) + " (" + std::to_string(i) + ")";
        RknpuPerfLogger perf_total(node_name + " - TOTAL");
        #endif

        const struct ggml_tensor* src0 = node->src[0]; // Weights      :  (K x N)
        const struct ggml_tensor* src1 = node->src[1]; // Activations  :  (M x K)
        struct ggml_tensor* dst = node;

        const int M = (int)src1->ne[1];
        const int K = (int)src0->ne[0];
        const int N = (int)src0->ne[1];

        auto segments = compute_matrix_segments(N, NUM_CORES);
        
        std::vector<std::shared_ptr<rknpu_matmul_context>> matmul_ctxs(NUM_CORES);
        std::vector<std::shared_ptr<rknn_tensor_mem>> mem_B_segments(NUM_CORES);
        std::shared_ptr<rknn_tensor_mem> mem_A_shared;
        std::vector<std::shared_ptr<rknn_tensor_mem>> mem_C_segments(NUM_CORES);

        // ===== 1. Preraring contexts =====
        {
            #if RKNPU_PERF_LOG
            RknpuPerfLogger perf_ctx(node_name + " - Context preparation");
            #endif
            
            for (int core_id = 0; core_id < NUM_CORES; core_id++) {
                int N_segment = segments[core_id].size_n;
                if (N_segment == 0) continue;
                matmul_ctxs[core_id] = backend_ctx->get_matmul_ctx(M, K, N_segment, core_id);
                
                if (!matmul_ctxs[core_id] || matmul_ctxs[core_id]->ctx == 0) {
                    RKNPU_LOG_ERROR("Failed to create matmul context for core %d\n", core_id);
                    return GGML_STATUS_FAILED;
                }
            }
        }

        // ===== 2. Preparing B-matrix =====
        {
            #if RKNPU_PERF_LOG
            RknpuPerfLogger perf_b(node_name + " - B matrix preparation");
            #endif
            
            std::vector<bool> need_upload(NUM_CORES, false);
            std::vector<std::vector<uint16_t>> packed_segments(NUM_CORES);
            
            // Creating or getting memory from cache (sequentially)
            for (int core_id = 0; core_id < NUM_CORES; core_id++) {
                if (segments[core_id].size_n == 0) continue;
                std::lock_guard<std::mutex> lock(backend_ctx->mutex);
                auto cache_key = std::make_tuple(src0, core_id);
                auto it = backend_ctx->b_npu_buffer_cache.find(cache_key);

                if (it != backend_ctx->b_npu_buffer_cache.end()) {
                    mem_B_segments[core_id] = it->second;
                } else {
                    need_upload[core_id] = true;
                    auto& matmul_ctx = matmul_ctxs[core_id];
                    rknn_tensor_mem* mem = rknn_create_mem(matmul_ctx->ctx, matmul_ctx->io_attr.B.size);
                    if (!mem) return GGML_STATUS_FAILED;

                    auto mem_ctx_for_deleter = get_rknpu_memory_context().get_ctx();
                    auto deleter = [mem_ctx_for_deleter](rknn_tensor_mem* m) { if (m) rknn_destroy_mem(mem_ctx_for_deleter, m); };
                    mem_B_segments[core_id] = std::shared_ptr<rknn_tensor_mem>(mem, deleter);
                    backend_ctx->b_npu_buffer_cache[cache_key] = mem_B_segments[core_id];
                }
            }

            // Packing segments into native format (parallel)
            #pragma omp parallel for num_threads(NUM_CORES)
            for (int core_id = 0; core_id < NUM_CORES; core_id++) {
                if (need_upload[core_id]) {
                    auto& matmul_ctx = matmul_ctxs[core_id];
                    packed_segments[core_id].resize(matmul_ctx->io_attr.B.size / sizeof(uint16_t));
                    
                    pack_B_segment_fp16_native_from_KN(
                        packed_segments[core_id].data(),
                        (const uint16_t*)src0->data,
                        K, N, segments[core_id].offset_n, segments[core_id].size_n);
                }
            }

            // Setting and syncronizing memory in NPU (sequentially)
            for (int core_id = 0; core_id < NUM_CORES; core_id++) {
                 if (segments[core_id].size_n == 0) continue;
                if (need_upload[core_id]) {
                    memcpy(mem_B_segments[core_id]->virt_addr, packed_segments[core_id].data(), packed_segments[core_id].size() * sizeof(uint16_t));
                    RKNN_CHECK(rknn_mem_sync(matmul_ctxs[core_id]->ctx, mem_B_segments[core_id].get(), RKNN_MEMORY_SYNC_TO_DEVICE), "sync B segment");
                }
                RKNN_CHECK(rknn_matmul_set_io_mem(matmul_ctxs[core_id]->ctx, mem_B_segments[core_id].get(), &matmul_ctxs[core_id]->io_attr.B), "set_io_mem B segment");
            }
        }

        // ===== 3. Preparing A-matrix =====
        {
            #if RKNPU_PERF_LOG
            RknpuPerfLogger perf_a(node_name + " - A matrix preparation");
            #endif

            auto cache_key = std::make_tuple(M, K, -1);
            auto& matmul_ctx_0 = matmul_ctxs[0];

            mem_A_shared = get_or_create_npu_buffer(backend_ctx, matmul_ctx_0->ctx, matmul_ctx_0->io_attr.A.size, cache_key, backend_ctx->a_buffer_cache);
            if (!mem_A_shared) return GGML_STATUS_FAILED;

            const float* x = (const float*)src1->data;
            const int row_stride = (int)(src1->nb[1] / sizeof(float));
            uint16_t* dst_base = (uint16_t*)mem_A_shared->virt_addr;

            #pragma omp parallel for
            for (int m = 0; m < M; ++m) {
                const float* src_row = x + (size_t)m * row_stride;
                uint16_t* dst_row = dst_base + (size_t)m * K;
                int k = 0;
                for (; k <= K - 8; k += 8) {
                    float32x4_t f32_vec_0 = vld1q_f32(src_row + k);
                    float32x4_t f32_vec_1 = vld1q_f32(src_row + k + 4);
                    float16x8_t f16_vec = vcombine_f16(vcvt_f16_f32(f32_vec_0), vcvt_f16_f32(f32_vec_1));
                    vst1q_u16(dst_row + k, (uint16x8_t)f16_vec);
                }
                for (; k < K; ++k) {
                    dst_row[k] = GGML_FP32_TO_FP16(src_row[k]);
                }
            }

            RKNN_CHECK(rknn_mem_sync(matmul_ctx_0->ctx, mem_A_shared.get(), RKNN_MEMORY_SYNC_TO_DEVICE), "sync A TO_DEVICE");

            for (int core_id = 0; core_id < NUM_CORES; core_id++) {
                if (segments[core_id].size_n == 0) continue;
                RKNN_CHECK(rknn_matmul_set_io_mem(matmul_ctxs[core_id]->ctx, mem_A_shared.get(), &matmul_ctxs[core_id]->io_attr.A), "set_io_mem A for core");
            }
        }

        // ===== 4. Preparing C-matrix =====
        {
            #if RKNPU_PERF_LOG
            RknpuPerfLogger perf_c(node_name + " - C matrix preparation");
            #endif
            
            for (int core_id = 0; core_id < NUM_CORES; core_id++) {
                if (segments[core_id].size_n == 0) continue;
                auto& matmul_ctx = matmul_ctxs[core_id];
                auto cache_key = std::make_tuple(M, segments[core_id].size_n, core_id);
                mem_C_segments[core_id] = get_or_create_npu_buffer(backend_ctx, matmul_ctx->ctx, matmul_ctx->io_attr.C.size, cache_key, backend_ctx->c_buffer_cache);
                if (!mem_C_segments[core_id]) return GGML_STATUS_FAILED;
                RKNN_CHECK(rknn_matmul_set_io_mem(matmul_ctx->ctx, mem_C_segments[core_id].get(), &matmul_ctx->io_attr.C), "set_io_mem C");
            }
        }

        // ===== 5. Running operation =====
        {
            #if RKNPU_PERF_LOG
            RknpuPerfLogger perf_run(node_name + " - NPU parallel execution");
            #endif
            
            #pragma omp parallel for num_threads(NUM_CORES)
            for (int core_id = 0; core_id < NUM_CORES; core_id++) {
                if (segments[core_id].size_n > 0) {
                    int ret = rknn_matmul_run(matmul_ctxs[core_id]->ctx);
                    if (ret != RKNN_SUCC) {
                        RKNPU_LOG_ERROR("rknn_matmul_run failed for core %d with error %d\n", core_id, ret);
                    }
                }
            }
        }

        // ===== 6. Collecting results =====
        {
            #if RKNPU_PERF_LOG
            RknpuPerfLogger perf_gather(node_name + " - Result gathering");
            #endif

            float* dst_data = (float*)dst->data;
            
            for (int core_id = 0; core_id < NUM_CORES; core_id++) {
                if (segments[core_id].size_n == 0) continue;
                RKNN_CHECK(rknn_mem_sync(matmul_ctxs[core_id]->ctx, mem_C_segments[core_id].get(), RKNN_MEMORY_SYNC_FROM_DEVICE), "sync C FROM_DEVICE");
            }

            #pragma omp parallel for
            for (int m = 0; m < M; m++) {
                for (int core_id = 0; core_id < NUM_CORES; core_id++) {
                    if (segments[core_id].size_n == 0) continue;
                    
                    int N_offset = segments[core_id].offset_n;
                    int N_segment = segments[core_id].size_n;
                    float* src_segment_base = (float*)mem_C_segments[core_id]->virt_addr;

                    memcpy(dst_data + (size_t)m * N + N_offset,
                           src_segment_base + (size_t)m * N_segment,
                           N_segment * sizeof(float));
                }
            }
        }
    }

    return GGML_STATUS_SUCCESS;
}


//
// Buffer
//

static void ggml_backend_rknpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_rknpu_buffer_context * ctx = (ggml_backend_rknpu_buffer_context *)buffer->context;
    rknn_matmul_ctx mem_ctx = get_rknpu_memory_context().get_ctx();
    if (mem_ctx != 0 && ctx->rknn_mem != nullptr) {
        RKNN_CHECK(rknn_destroy_mem(mem_ctx, ctx->rknn_mem), "rknn_destroy_mem");
    }
    dma_free(ctx->dma_buf);
    delete ctx;
}

static void * ggml_backend_rknpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_rknpu_buffer_context * ctx = (ggml_backend_rknpu_buffer_context *)buffer->context;
    return ctx->rknn_mem->virt_addr;
}

static enum ggml_status ggml_backend_rknpu_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    UNUSED(buffer);
    UNUSED(tensor);
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_rknpu_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    auto * ctx = (ggml_backend_rknpu_buffer_context *) buffer->context;

    uint8_t * base = (uint8_t *) ctx->rknn_mem->virt_addr;
    uint8_t * dst  = (uint8_t *) tensor->data + offset;
    GGML_ASSERT(dst >= base && dst + size <= base + ctx->rknn_mem->size);

    memcpy(dst, data, size);

    auto mem_ctx = get_rknpu_memory_context().get_ctx();
    RKNN_CHECK(rknn_mem_sync(mem_ctx, ctx->rknn_mem, RKNN_MEMORY_SYNC_TO_DEVICE), "mem_sync write");
}

static void ggml_backend_rknpu_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_rknpu_buffer_context * ctx = (ggml_backend_rknpu_buffer_context *)buffer->context;
    rknn_matmul_ctx mem_ctx = get_rknpu_memory_context().get_ctx();

    RKNN_CHECK(rknn_mem_sync(mem_ctx, ctx->rknn_mem, RKNN_MEMORY_SYNC_FROM_DEVICE), "rknn_mem_sync from device");

    memcpy(data, (const char *)tensor->data + offset, size);
}

static void ggml_backend_rknpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_rknpu_buffer_context * ctx = (ggml_backend_rknpu_buffer_context *)buffer->context;
    memset(ctx->rknn_mem->virt_addr, value, ctx->rknn_mem->size);
    
    rknn_matmul_ctx mem_ctx = get_rknpu_memory_context().get_ctx();
    RKNN_CHECK(rknn_mem_sync(mem_ctx, ctx->rknn_mem, RKNN_MEMORY_SYNC_TO_DEVICE), "rknn_mem_sync to device after clear");
}


//
// Buffer Type
//

static const char * ggml_backend_rknpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    UNUSED(buft);
    return "RKNPU";
}

static ggml_backend_buffer_t ggml_backend_rknpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    UNUSED(buft);
    rknn_matmul_ctx mem_ctx = get_rknpu_memory_context().get_ctx();
    if (mem_ctx == 0) {
        RKNPU_LOG_ERROR("RKNPU memory context not initialized, cannot allocate buffer.\n");
        return NULL;
    }

    DmaBuffer dma_buf = dma_alloc(size);
    if (dma_buf.fd < 0) {
        RKNPU_LOG_ERROR("dma_alloc failed to allocate %zu bytes\n", size);
        return NULL;
    }
    
    rknn_tensor_mem * rknn_mem = rknn_create_mem_from_fd(mem_ctx, dma_buf.fd, dma_buf.virt_addr, size, 0);
    if (rknn_mem == NULL) {
        RKNPU_LOG_ERROR("rknn_create_mem_from_fd failed for size %zu\n", size);
        dma_free(dma_buf);
        return NULL;
    }

    ggml_backend_rknpu_buffer_context * ctx = new ggml_backend_rknpu_buffer_context{rknn_mem, dma_buf, "rknpu_dma_buffer"};
    
    static const ggml_backend_buffer_i rknpu_buffer_interface = {
        /* .free_buffer   = */ ggml_backend_rknpu_buffer_free_buffer,
        /* .get_base      = */ ggml_backend_rknpu_buffer_get_base,
        /* .init_tensor   = */ ggml_backend_rknpu_buffer_init_tensor,
        /* .memset_tensor = */ NULL,
        /* .set_tensor    = */ ggml_backend_rknpu_buffer_set_tensor,
        /* .get_tensor    = */ ggml_backend_rknpu_buffer_get_tensor,
        /* .cpy_tensor    = */ NULL,
        /* .clear         = */ ggml_backend_rknpu_buffer_clear,
        /* .reset         = */ NULL,
    };

    return ggml_backend_buffer_init(buft, rknpu_buffer_interface, ctx, size);
}

static size_t ggml_backend_rknpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    UNUSED(buft);
    return 64;
}

static size_t ggml_backend_rknpu_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    UNUSED(buft);
    return ggml_nbytes(tensor);
}


//
// Device
//

static const char * ggml_backend_rknpu_device_get_name(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return "RKNPU";
}

static const char * ggml_backend_rknpu_device_get_description(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return "Rockchip NPU";
}

static void ggml_backend_rknpu_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    UNUSED(dev);
    *free = 0;
    *total = 0;
}

static enum ggml_backend_dev_type ggml_backend_rknpu_device_get_type(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void ggml_backend_rknpu_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name = ggml_backend_rknpu_device_get_name(dev);
    props->description = ggml_backend_rknpu_device_get_description(dev);
    props->type = ggml_backend_rknpu_device_get_type(dev);
    ggml_backend_rknpu_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->device_id = NULL;

    props->caps.async = false;
    props->caps.host_buffer = false;
    props->caps.buffer_from_host_ptr = false;
    props->caps.events = false;
}

static bool ggml_backend_rknpu_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    UNUSED(dev);
    RKNPU_LOG_INFO("[RKNPU_SUPPORT] Checking support for op: %s, name: %s\n", ggml_op_name(op->op), op->name);

    switch (op->op) {
        case GGML_OP_NONE:
            RKNPU_LOG_INFO("  - ACCEPT: Op NONE is supported for RKNPU (leaf/weights).\n");
            return true;

        case GGML_OP_MUL_MAT: {
            const struct ggml_tensor * src0 = op->src[0]; // Weights
            const struct ggml_tensor * src1 = op->src[1]; // Activations

            RKNPU_LOG_INFO("  - src0: type=%s, dims=[%lld, %lld, %lld, %lld]\n", ggml_type_name(src0->type), src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3]);
            RKNPU_LOG_INFO("  - src1: type=%s, dims=[%lld, %lld, %lld, %lld]\n", ggml_type_name(src1->type), src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]);

            // Checking for weights in F16 and activations in F32
            if (src0->type != GGML_TYPE_F16 || src1->type != GGML_TYPE_F32) {
                RKNPU_LOG_INFO("  - REJECT: Unsupported type combination. Want src0=F16, src1=F32.\n");
                return false;
            }

            // Checking for K divisible 32, N divisible 16
            if (src0->ne[0] % 32 != 0 || src0->ne[1] % 16 != 0) {
                RKNPU_LOG_INFO("  - REJECT: K (%lld) must be a multiple of 32 and N (%lld) a multiple of 16.\n", src0->ne[0], src0->ne[1]);
                return false;
            }

            // Checking for exact dimentions
            if (src1->ne[0] != src0->ne[0]) {
                 RKNPU_LOG_INFO("  - REJECT: K dimensions do not match (%lld vs %lld).\n", src1->ne[0], src0->ne[0]);
                 return false;
            }

            // Checking contiguous memory
            if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
                RKNPU_LOG_INFO("  - REJECT: Tensors are not contiguous.\n");
                return false;
            }

            RKNPU_LOG_INFO("  - ACCEPT: Op %s can be offloaded to RKNPU.\n", op->name);
            return true;
        }
        default:
            RKNPU_LOG_INFO("  - REJECT: Op %s is not supported by RKNPU backend.\n", ggml_op_name(op->op));
            return false;
    }
}

static ggml_backend_t ggml_backend_rknpu_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    UNUSED(dev);
    UNUSED(params);

    ggml_backend_rknpu_context * ctx = new ggml_backend_rknpu_context{"RKNPU Backend"};
    
    static const struct ggml_backend_i rknpu_backend_interface = {
        /* .get_name           = */ ggml_backend_rknpu_name,
        /* .free               = */ ggml_backend_rknpu_free,
        /* .set_tensor_async   = */ NULL,
        /* .get_tensor_async   = */ NULL,
        /* .cpy_tensor_async   = */ NULL,
        /* .synchronize        = */ NULL,
        /* .graph_plan_create  = */ NULL,
        /* .graph_plan_free    = */ NULL,
        /* .graph_plan_update  = */ NULL,
        /* .graph_plan_compute = */ NULL,
        /* .graph_compute      = */ ggml_backend_rknpu_graph_compute,
        /* .event_record       = */ NULL,
        /* .event_wait         = */ NULL,
        /* .graph_optimize     = */ NULL,
    };

    return new ggml_backend{
        /* .guid    = */ {0},
        /* .iface   = */ rknpu_backend_interface,
        /* .device  = */ dev,
        /* .context = */ ctx,
    };
}


//
// Registry
//

static const char * ggml_backend_rknpu_reg_get_name(ggml_backend_reg_t reg) {
    UNUSED(reg);
    return "RKNPU";
}

static size_t ggml_backend_rknpu_reg_get_device_count(ggml_backend_reg_t reg) {
    UNUSED(reg);
    if (get_rknpu_memory_context().get_ctx() != 0) {
        return 1;
    }
    return 0;
}

static ggml_backend_dev_t ggml_backend_rknpu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    if (index != 0) {
        return NULL;
    }
    
    static const struct ggml_backend_buffer_type_i rknpu_buffer_type_interface = {
        /* .get_name       = */ ggml_backend_rknpu_buffer_type_get_name,
        /* .alloc_buffer   = */ ggml_backend_rknpu_buffer_type_alloc_buffer,
        /* .get_alignment  = */ ggml_backend_rknpu_buffer_type_get_alignment,
        /* .get_max_size   = */ NULL,
        /* .get_alloc_size = */ ggml_backend_rknpu_buffer_type_get_alloc_size,
        /* .is_host        = */ NULL,
    };

    static struct ggml_backend_buffer_type rknpu_buffer_type = {
        /* .iface   = */ rknpu_buffer_type_interface,
        /* .device  = */ NULL,
        /* .context = */ NULL,
    };

    static const struct ggml_backend_device_i rknpu_device_interface = {
        /* .get_name             = */ ggml_backend_rknpu_device_get_name,
        /* .get_description      = */ ggml_backend_rknpu_device_get_description,
        /* .get_memory           = */ ggml_backend_rknpu_device_get_memory,
        /* .get_type             = */ ggml_backend_rknpu_device_get_type,
        /* .get_props            = */ ggml_backend_rknpu_device_get_props,
        /* .init_backend         = */ ggml_backend_rknpu_device_init_backend,
        /* .get_buffer_type      = */ [](ggml_backend_dev_t dev) { UNUSED(dev); return &rknpu_buffer_type; },
        /* .get_host_buffer_type = */ NULL,
        /* .buffer_from_host_ptr = */ NULL,
        /* .supports_op          = */ ggml_backend_rknpu_device_supports_op,
        /* .supports_buft        = */ [](ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) { UNUSED(dev); return buft == &rknpu_buffer_type; },
        /* .offload_op           = */ NULL,
        /* .event_new            = */ NULL,
        /* .event_free           = */ NULL,
        /* .event_synchronize    = */ NULL,
    };

    static struct ggml_backend_device rknpu_device = {
        /* .iface   = */ rknpu_device_interface,
        /* .reg     = */ reg,
        /* .context = */ NULL,
    };
    
    if (rknpu_buffer_type.device == NULL) {
        rknpu_buffer_type.device = &rknpu_device;
    }

    return &rknpu_device;
}


//
// Public API
//

GGML_API ggml_backend_reg_t ggml_backend_rknpu2_reg(void) {
    static const struct ggml_backend_reg_i rknpu_reg_interface = {
        /* .get_name         = */ ggml_backend_rknpu_reg_get_name,
        /* .get_device_count = */ ggml_backend_rknpu_reg_get_device_count,
        /* .get_device       = */ ggml_backend_rknpu_reg_get_device,
        /* .get_proc_address = */ NULL,
    };

    static struct ggml_backend_reg rknpu_backend_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ rknpu_reg_interface,
        /* .context     = */ NULL,
    };

    return &rknpu_backend_reg;
}

#ifdef GGML_BACKEND_DL
GGML_BACKEND_DL_IMPL(ggml_backend_rknpu2_reg)
#endif