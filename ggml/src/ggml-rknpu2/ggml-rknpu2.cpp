#include "ggml-rknpu2.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-quants.h"

#include "rknpu2-allocation.h"
#include "rknpu2-quantization.h"
#include "rknpu2-calibration.h"
#include "rknpu2-configuration.h"

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
#include <random>
#include <limits>

#define UNUSED(x) (void)(x)

// Macro for RKNN API calls
#define RKNN_CHECK(stmt, msg)                                           \
    do {                                                                \
        int ret = (stmt);                                               \
        if (ret < 0) {                                                  \
            fprintf(stderr,"RKNN error %d at %s:%d: %s\n", ret,                \
                __FILE__, __LINE__, msg);                               \
            assert(false);                                              \
        }                                                               \
    } while (0)

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
static std::vector<MatrixSegment> compute_matrix_segments(int N, int num_cores, int alignment) {
    std::vector<MatrixSegment> segments;

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
    rknpu2_allocation::DmaBuffer dma_buf;
    std::string name;

    // Per-tensor scale for weights
    std::unordered_map<const struct ggml_tensor *, float> quantized_tensor_scales;

    // Per-tensor random sign vector for Hadamard Transform
    std::unordered_map<const struct ggml_tensor *, std::vector<float>> hadamard_s_vectors;

    std::mutex mutex;
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
        if (ret < 0) ctx = 0;
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
    std::unordered_map<std::tuple<int, int, int, int, int>, std::shared_ptr<rknpu_matmul_context>, TupleHasher> matmul_ctx_cache;

    // B-matrices handle cache (from fd)
    std::unordered_map<std::pair<ggml_backend_buffer_t, size_t>, std::shared_ptr<rknn_tensor_mem>, PairHasher> b_mem_handle_cache;

    // A- and C-matrices cache (from create_mem)
    std::unordered_map<std::tuple<int, int, int>, std::shared_ptr<rknn_tensor_mem>, TupleHasher> a_buffer_cache;
    std::unordered_map<std::tuple<int, int, int>, std::shared_ptr<rknn_tensor_mem>, TupleHasher> c_buffer_cache;

    std::shared_ptr<rknpu_matmul_context> get_matmul_ctx(int M, int K, int N, int core_id, rknn_matmul_type type) {
        std::lock_guard<std::mutex> lock(mutex);
        auto key = std::make_tuple(M, K, N, core_id, (int)type);
        auto it = matmul_ctx_cache.find(key);
        if (it != matmul_ctx_cache.end()) {
            return it->second;
        }
        auto ctx = std::make_shared<rknpu_matmul_context>(M, K, N, type);
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
            // Handle error
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
        if (ret < 0) mem_ctx = 0;
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

    // Getting the current device configuration once
    const auto& config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor* node = cgraph->nodes[i];
        if (node->op != GGML_OP_MUL_MAT) continue;

        const struct ggml_tensor* src0 = node->src[0]; // Weights      :  (K x N)
        const struct ggml_tensor* src1 = node->src[1]; // Activations  :  (M x K)
        struct ggml_tensor* dst = node;

        const ggml_type w_type = src0->type;

        const int M = (int)src1->ne[1];
        const int K = (int)src0->ne[0];
        const int N = (int)src0->ne[1];

        const bool is_q4_hadamard = (src0->type == GGML_TYPE_Q4_0);
        const int K_op = is_q4_hadamard ? rknpu2_calibration::next_power_of_two(K) : K;

        const auto* op_support = config.find_op_support(src0->type);
        const rknn_matmul_type matmul_type = op_support->mm_type;
        const int alignment = op_support->n_align;

        auto all_segments = compute_matrix_segments(N, config.core_count, alignment);

        std::vector<MatrixSegment> active_segments;
        for (const auto& seg : all_segments) {
            if (seg.size_n > 0) {
                active_segments.push_back(seg);
            }
        }

        if (active_segments.empty()) continue;

        const size_t num_active_segments = active_segments.size();
        std::vector<std::shared_ptr<rknpu_matmul_context>> matmul_ctxs(num_active_segments);
        std::vector<std::shared_ptr<rknn_tensor_mem>> mem_B_segments(num_active_segments);
        std::shared_ptr<rknn_tensor_mem> mem_A_shared;
        std::vector<std::shared_ptr<rknn_tensor_mem>> mem_C_segments(num_active_segments);

        // ===========================================
        // ========== 1. Preparing contexts ==========
        // ===========================================
        {
            for (size_t i = 0; i < num_active_segments; ++i) {
                const auto& seg = active_segments[i];
                matmul_ctxs[i] = backend_ctx->get_matmul_ctx(M, K_op, seg.size_n, seg.core_id, matmul_type);
                if (!matmul_ctxs[i] || matmul_ctxs[i]->ctx == 0) return GGML_STATUS_FAILED;
            }
        }

        // ===========================================
        // ========== 2. Preparing B-matrix ==========
        // ===========================================
        {
            ggml_backend_buffer_t src0_buffer = src0->buffer;
            auto* src0_buf_ctx = (ggml_backend_rknpu_buffer_context*)src0_buffer->context;
            size_t src0_base_offset_in_dma = (uintptr_t)src0->data - (uintptr_t)ggml_backend_buffer_get_base(src0_buffer);

            size_t type_size_packed;
            if (w_type == GGML_TYPE_F16) type_size_packed = 2;
            else if (w_type == GGML_TYPE_Q8_0) type_size_packed = 1;
            else type_size_packed = 0;

            size_t current_offset_in_tensor = 0;
            for (const auto& seg : all_segments) {
                for (size_t i = 0; i < num_active_segments; ++i) {
                    if (active_segments[i].offset_n == seg.offset_n) {
                        auto& matmul_ctx = matmul_ctxs[i];
                        size_t segment_size_bytes = matmul_ctx->io_attr.B.size;
                        size_t total_offset = src0_base_offset_in_dma + current_offset_in_tensor;
                        
                        auto cache_key = std::make_pair(src0_buffer, total_offset);
                        std::lock_guard<std::mutex> lock(backend_ctx->mutex);
                        auto it = backend_ctx->b_mem_handle_cache.find(cache_key);

                        if (it != backend_ctx->b_mem_handle_cache.end()) {
                            mem_B_segments[i] = it->second;
                        } else {
                            rknn_tensor_mem* mem = rknn_create_mem_from_fd(matmul_ctx->ctx, src0_buf_ctx->dma_buf.fd, src0_buf_ctx->dma_buf.virt_addr, segment_size_bytes, total_offset);
                            if (!mem) return GGML_STATUS_FAILED;
                            auto deleter = [ctx = matmul_ctx->ctx](rknn_tensor_mem* m) { if (m) rknn_destroy_mem(ctx, m); };
                            mem_B_segments[i] = std::shared_ptr<rknn_tensor_mem>(mem, deleter);
                            backend_ctx->b_mem_handle_cache[cache_key] = mem_B_segments[i];
                        }
                        RKNN_CHECK(rknn_matmul_set_io_mem(matmul_ctx->ctx, mem_B_segments[i].get(), &matmul_ctx->io_attr.B), "set_io_mem B segment");
                        break;
                    }
                }
                current_offset_in_tensor += type_size_packed > 0 ? (size_t)seg.size_n * K_op * type_size_packed : (size_t)seg.size_n * K_op / 2;
            }
        }

        // ===========================================
        // ========== 3. Preparing A-matrix ==========
        // ===========================================
        std::vector<float> scales_A(M);
        float scale_B = 1.0f;
        {
            auto cache_key = std::make_tuple(M, K, (int)w_type);
            auto& matmul_ctx_0 = matmul_ctxs[0];

            mem_A_shared = get_or_create_npu_buffer(backend_ctx, matmul_ctx_0->ctx, matmul_ctx_0->io_attr.A.size, cache_key, backend_ctx->a_buffer_cache);
            if (!mem_A_shared) return GGML_STATUS_FAILED;

            const float* x = (const float*)src1->data;
            const int row_stride = (int)(src1->nb[1] / sizeof(float));
            void* dst_base = mem_A_shared->virt_addr;

            switch (op_support->npu_type_a) {
                case rknpu2_configuration::NPU_TYPE_FP16: {
                    uint16_t* dst_ptr = (uint16_t*)dst_base;
                    #pragma omp parallel for
                    for (int m = 0; m < M; ++m) {
                        const float* src_row = x + (size_t)m * row_stride;
                        uint16_t* dst_row = dst_ptr + (size_t)m * K;
                        rknpu2_quantization::convert_fp32_to_fp16(src_row, dst_row, K);
                    }
                    break;
                }

                case rknpu2_configuration::NPU_TYPE_INT8: {
                    int8_t* dst_ptr = (int8_t*)dst_base;
                    #pragma omp parallel for
                    for (int m = 0; m < M; ++m) {
                        const float* src_row = x + (size_t)m * row_stride;
                        float amax_m = 0.0f;
                        for (int k = 0; k < K; ++k) {
                            amax_m = std::max(amax_m, std::abs(src_row[k]));
                        }
                        scales_A[m] = amax_m / 127.0f;
                        int8_t* dst_row = dst_ptr + (size_t)m * K;
                        rknpu2_quantization::quantize_fp32_to_int8(src_row, dst_row, K, scales_A[m]);
                    }
                    break;
                }

                case rknpu2_configuration::NPU_TYPE_INT4: {
                    auto* src0_buf_ctx = (ggml_backend_rknpu_buffer_context*)src0->buffer->context;
                    std::vector<float> s_vec;
                    {
                        std::lock_guard<std::mutex> lock(src0_buf_ctx->mutex);
                        auto it = src0_buf_ctx->hadamard_s_vectors.find(src0);
                        GGML_ASSERT(it != src0_buf_ctx->hadamard_s_vectors.end() && "Hadamard 's' vector not found");
                        s_vec = it->second;
                    }

                    uint8_t* dst_ptr = (uint8_t*)dst_base;
                    #pragma omp parallel for
                    for (int m = 0; m < M; ++m) {
                        const float* src_row = x + (size_t)m * row_stride;
                        std::vector<float> signed_row(K);
                        for(int k=0; k<K; ++k) signed_row[k] = src_row[k] * s_vec[k];

                        std::vector<float> rotated_row(K_op);
                        rknpu2_calibration::hadamard_transform(rotated_row.data(), signed_row.data(), K, K_op);

                        float amax_m = 0.0f;
                        for (int k = 0; k < K_op; ++k) amax_m = std::max(amax_m, std::abs(rotated_row[k]));
                        scales_A[m] = amax_m / 7.0f;
                        uint8_t* dst_row = dst_ptr + (size_t)m * (K_op / 2);
                        rknpu2_quantization::quantize_fp32_to_int4_packed(rotated_row.data(), dst_row, K_op, scales_A[m]);
                    }
                    break;
                }

                default:
                    // This should not be reached if config is correct
                    return GGML_STATUS_FAILED;
            }

            if (op_support->npu_type_a == rknpu2_configuration::NPU_TYPE_INT8 || op_support->npu_type_a == rknpu2_configuration::NPU_TYPE_INT4) {
                ggml_backend_buffer_t src0_buffer = src0->buffer;
                auto* src0_buf_ctx = (ggml_backend_rknpu_buffer_context*)src0_buffer->context;
                {
                    std::lock_guard<std::mutex> lock(src0_buf_ctx->mutex);
                    auto it = src0_buf_ctx->quantized_tensor_scales.find(src0);
                    GGML_ASSERT(it != src0_buf_ctx->quantized_tensor_scales.end() && "Quantized scale not found");
                    scale_B = it->second;
                }
            }

            RKNN_CHECK(rknn_mem_sync(matmul_ctx_0->ctx, mem_A_shared.get(), RKNN_MEMORY_SYNC_TO_DEVICE), "sync A TO_DEVICE");

            for (size_t i = 0; i < num_active_segments; i++) {
                RKNN_CHECK(rknn_matmul_set_io_mem(matmul_ctxs[i]->ctx, mem_A_shared.get(), &matmul_ctxs[i]->io_attr.A), "set_io_mem A for core");
            }
        }

        // ===========================================
        // ========== 4. Preparing C-matrix ==========
        // ===========================================
        {            
            for (size_t i = 0; i < num_active_segments; i++) {
                auto& matmul_ctx = matmul_ctxs[i];
                auto cache_key = std::make_tuple(M, active_segments[i].size_n, active_segments[i].core_id);
                mem_C_segments[i] = get_or_create_npu_buffer(backend_ctx, matmul_ctx->ctx, matmul_ctx->io_attr.C.size, cache_key, backend_ctx->c_buffer_cache);
                if (!mem_C_segments[i]) return GGML_STATUS_FAILED;
                RKNN_CHECK(rknn_matmul_set_io_mem(matmul_ctx->ctx, mem_C_segments[i].get(), &matmul_ctx->io_attr.C), "set_io_mem C");
            }
        }

        // ==========================================
        // ========== 5. Running operation ==========
        // ==========================================
        {            
            #pragma omp parallel for num_threads(num_active_segments)
            for (size_t i = 0; i < num_active_segments; i++) {
                int ret = rknn_matmul_run(matmul_ctxs[i]->ctx);
                if (ret != RKNN_SUCC) {
                    // Handle error
                }
            }
        }

        // ===========================================
        // ========== 6. Collecting results ==========
        // ===========================================
        {
            float* dst_data = (float*)dst->data;
            
            for (size_t i = 0; i < num_active_segments; i++) {
                RKNN_CHECK(rknn_mem_sync(matmul_ctxs[i]->ctx, mem_C_segments[i].get(), RKNN_MEMORY_SYNC_FROM_DEVICE), "sync C FROM_DEVICE");
            }

            #pragma omp parallel for
            for (int m = 0; m < M; m++) {
                switch (op_support->npu_type_c) {
                    case rknpu2_configuration::NPU_TYPE_FP32: {
                        for (size_t i = 0; i < num_active_segments; i++) {
                            int N_offset = active_segments[i].offset_n;
                            int N_segment = active_segments[i].size_n;
                            float* src_segment_base = (float*)mem_C_segments[i]->virt_addr;
                            memcpy(dst_data + (size_t)m * N + N_offset,
                                src_segment_base + (size_t)m * N_segment,
                                N_segment * sizeof(float));
                        }
                        break;
                    }

                    case rknpu2_configuration::NPU_TYPE_INT32: {
                        float dequant_scale = scales_A[m] * scale_B;
                        for (size_t i = 0; i < num_active_segments; i++) {
                            int N_offset = active_segments[i].offset_n;
                            int N_segment = active_segments[i].size_n;
                            float* dst_ptr = dst_data + (size_t)m * N + N_offset;
                            int32_t* src_segment_base = (int32_t*)mem_C_segments[i]->virt_addr;
                            int32_t* src_ptr = src_segment_base + (size_t)m * N_segment;
                            rknpu2_quantization::dequantize_int32_to_fp32(src_ptr, dst_ptr, N_segment, dequant_scale);
                        }
                        break;
                    }

                    case rknpu2_configuration::NPU_TYPE_INT16: {
                        float dequant_scale = scales_A[m] * scale_B;
                        if (op_support->npu_type_a == rknpu2_configuration::NPU_TYPE_INT4) {
                            dequant_scale /= (float)K_op;
                        }

                        for (size_t i = 0; i < num_active_segments; i++) {
                            int N_offset = active_segments[i].offset_n;
                            int N_segment = active_segments[i].size_n;
                            float* dst_ptr = dst_data + (size_t)m * N + N_offset;
                            int16_t* src_segment_base = (int16_t*)mem_C_segments[i]->virt_addr;
                            int16_t* src_ptr = src_segment_base + (size_t)m * N_segment;
                            rknpu2_quantization::dequantize_int16_to_fp32(src_ptr, dst_ptr, N_segment, dequant_scale);
                        }
                        break;
                    }
                    
                    default:
                        // This should not be reached if config is correct
                        break;
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
    rknpu2_allocation::free(ctx->dma_buf);
    delete ctx;
}

static void * ggml_backend_rknpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_rknpu_buffer_context * ctx = (ggml_backend_rknpu_buffer_context *)buffer->context;
    return ctx->dma_buf.virt_addr;
}

static enum ggml_status ggml_backend_rknpu_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    UNUSED(buffer);
    UNUSED(tensor);
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_rknpu_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    auto * ctx = (ggml_backend_rknpu_buffer_context *) buffer->context;
    uint8_t* dma_base = (uint8_t*)ctx->dma_buf.virt_addr;
    uint8_t* tensor_dma_ptr = dma_base + ((uintptr_t)tensor->data - (uintptr_t)ggml_backend_buffer_get_base(buffer));

    // Getting the current device configuration to drive the packing logic
    const auto& config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();
    const auto* op_support = config.find_op_support(tensor->type);

    // If there is a specific packing function defined for this tensor type, it's a weight matrix
    if (op_support && op_support->pack_func) {
        const int K = (int)tensor->ne[0];
        const int N = (int)tensor->ne[1];

        if (tensor->type == GGML_TYPE_F16) {
            auto segments = compute_matrix_segments(N, config.core_count, op_support->n_align);

            uint8_t* current_write_ptr = tensor_dma_ptr;
            std::vector<uint8_t> packed_data_temp;

            for (const auto& seg : segments) {
                if (seg.size_n == 0) continue;
                size_t segment_packed_size_bytes = (size_t)seg.size_n * K * sizeof(uint16_t);
                packed_data_temp.resize(segment_packed_size_bytes);

                op_support->pack_func(
                    packed_data_temp.data(),
                    (const uint8_t*)data,
                    K, N, seg.offset_n, seg.size_n
                );
                memcpy(current_write_ptr, packed_data_temp.data(), segment_packed_size_bytes);
                current_write_ptr += segment_packed_size_bytes;
            }
        // GGML_TYPE_Q8_0
        } else if (tensor->type == GGML_TYPE_Q8_0) {
            const block_q8_0* src_blocks = (const block_q8_0*)data;
            const size_t n_elements = (size_t)K * N;

            // Finding the global scale without storing the full dequantized matrix.
            float amax = 0.0f;
            #pragma omp parallel
            {
                std::vector<float> tmp_row(K);
                #pragma omp for reduction(max:amax)
                for (int n = 0; n < N; ++n) {
                    dequantize_row_q8_0(src_blocks + (size_t)n * (K / QK8_0), tmp_row.data(), K);
                    for (int k = 0; k < K; ++k) amax = std::max(amax, std::abs(tmp_row[k]));
                }
            }

            const float global_scale_b = amax / 127.0f;

            // Storing it in the buffer context cache
            {
                std::lock_guard<std::mutex> lock(ctx->mutex);
                ctx->quantized_tensor_scales[tensor] = global_scale_b;
            }

            // Dequantizing and re-quantizing directly row-by-row
            std::vector<int8_t> requantized_data(n_elements);
            #pragma omp parallel
            {
                std::vector<float> tmp_row(K);
                #pragma omp for
                for (int n = 0; n < N; ++n) {
                    dequantize_row_q8_0(src_blocks + (size_t)n * (K / QK8_0), tmp_row.data(), K);
                    rknpu2_quantization::quantize_fp32_to_int8(tmp_row.data(), requantized_data.data() + (size_t)n * K, K, global_scale_b);
                }
            }

            // Packing into native format by segments
            auto segments = compute_matrix_segments(N, config.core_count, op_support->n_align);
            uint8_t* current_write_ptr = tensor_dma_ptr;
            std::vector<uint8_t> packed_data_temp;

            for (const auto& seg : segments) {
                if (seg.size_n == 0) continue;
                size_t segment_packed_size = (size_t)seg.size_n * K;
                packed_data_temp.resize(segment_packed_size);

                op_support->pack_func(
                    packed_data_temp.data(),
                    (const uint8_t*)requantized_data.data(),
                    K, N, seg.offset_n, seg.size_n
                );
                memcpy(current_write_ptr, packed_data_temp.data(), segment_packed_size);
                current_write_ptr += segment_packed_size;
            }
        // GGML_TYPE_Q4_0
        } else if (tensor->type == GGML_TYPE_Q4_0) {
            if (op_support->npu_type_a == rknpu2_configuration::NPU_TYPE_INT4) {
                const block_q4_0* src_blocks = (const block_q4_0*)data;

                // Dequantizing original weights to FP32
                const size_t n_elements_original = (size_t)K * N;
                std::vector<float> dequantized_data(n_elements_original);
                #pragma omp parallel for
                for (int n = 0; n < N; ++n) {
                    dequantize_row_q4_0(src_blocks + (size_t)n * (K / QK4_0), dequantized_data.data() + (size_t)n * K, K);
                }

                // Padding K and generating the random sign vector 's'
                const int padded_K = rknpu2_calibration::next_power_of_two(K);
                std::vector<float> s_vec(padded_K);
                std::mt19937 gen(reinterpret_cast<uintptr_t>(tensor));
                std::uniform_int_distribution<int> distrib(0, 1);
                for(int k = 0; k < padded_K; ++k) {
                    s_vec[k] = (distrib(gen) == 0) ? -1.0f : 1.0f;
                }

                // Storing the vector for use during activation processing
                {
                    std::lock_guard<std::mutex> lock(ctx->mutex);
                    ctx->hadamard_s_vectors[tensor] = s_vec;
                }

                // Rotating the FP32 weight matrix using padding
                const size_t n_elements_padded = (size_t)N * padded_K;
                std::vector<float> rotated_b_fp32(n_elements_padded);
                #pragma omp parallel for
                for (int n = 0; n < N; ++n) {
                    const float* src_row = dequantized_data.data() + (size_t)n * K;
                    float* dst_row_rot = rotated_b_fp32.data() + (size_t)n * padded_K;
                    
                    std::vector<float> signed_row(K);
                    for(int k=0; k<K; ++k) signed_row[k] = src_row[k] * s_vec[k];

                    rknpu2_calibration::hadamard_transform(dst_row_rot, signed_row.data(), K, padded_K);
                }

                // Finding scale on the rotated data
                const float amax = rknpu2_calibration::calculate_entropy_amax(rotated_b_fp32.data(), n_elements_padded);
                const float global_scale_b = amax / 7.0f;

                {
                    std::lock_guard<std::mutex> lock(ctx->mutex);
                    ctx->quantized_tensor_scales[tensor] = global_scale_b;
                }

                // Quantizing data to INT4
                std::vector<uint8_t> packed_int4_data(n_elements_padded / 2);
                #pragma omp parallel for
                for (int n = 0; n < N; ++n) {
                    rknpu2_quantization::quantize_fp32_to_int4_packed(
                        rotated_b_fp32.data() + (size_t)n * padded_K,
                        packed_int4_data.data() + (size_t)n * (padded_K/2),
                        padded_K, global_scale_b
                    );
                }

                // Packing into native RKNN format with padded_K
                uint8_t* current_write_ptr = tensor_dma_ptr;
                std::vector<uint8_t> packed_data_temp;

                for (const auto& seg : compute_matrix_segments(N, config.core_count, op_support->n_align)) {
                    if (seg.size_n == 0) continue;
                    size_t segment_packed_size = (size_t)seg.size_n * padded_K / 2;
                    packed_data_temp.resize(segment_packed_size);

                    op_support->pack_func(
                        packed_data_temp.data(),
                        packed_int4_data.data(),
                        padded_K, N, seg.offset_n, seg.size_n
                    );
                    memcpy(current_write_ptr, packed_data_temp.data(), segment_packed_size);
                    current_write_ptr += segment_packed_size;
                }
            } else {
                const block_q4_0* src_blocks = (const block_q4_0*)data;
                const size_t n_elements = (size_t)K * N;

                // Finding the global scale without storing the full dequantized matrix.
                float amax = 0.0f;
                #pragma omp parallel
                {
                    std::vector<float> tmp_row(K);
                    #pragma omp for reduction(max:amax)
                    for (int n = 0; n < N; ++n) {
                        dequantize_row_q4_0(src_blocks + (size_t)n * (K / QK4_0), tmp_row.data(), K);
                        for (int k = 0; k < K; ++k) amax = std::max(amax, std::abs(tmp_row[k]));
                    }
                }

                const float global_scale_b = amax / 7.0f;

                // Storing it in the buffer context cache
                {
                    std::lock_guard<std::mutex> lock(ctx->mutex);
                    ctx->quantized_tensor_scales[tensor] = global_scale_b;
                }

                // Dequantizing and re-quantizing directly row-by-row
                std::vector<uint8_t> requantized_data(n_elements / 2);
                #pragma omp parallel
                {
                    std::vector<float> tmp_row(K);
                    #pragma omp for
                    for (int n = 0; n < N; ++n) {
                        dequantize_row_q4_0(src_blocks + (size_t)n * (K / QK4_0), tmp_row.data(), K);
                        rknpu2_quantization::quantize_fp32_to_int4_packed(tmp_row.data(), requantized_data.data() + (size_t)n * (K/2), K, global_scale_b);
                    }
                }

                // Packing into native format by segments
                auto segments = compute_matrix_segments(N, config.core_count, op_support->n_align);
                uint8_t* current_write_ptr = tensor_dma_ptr;
                std::vector<uint8_t> packed_data_temp;

                for (const auto& seg : segments) {
                    if (seg.size_n == 0) continue;
                    size_t segment_packed_size = (size_t)seg.size_n * K / 2;
                    packed_data_temp.resize(segment_packed_size);

                    op_support->pack_func(
                        packed_data_temp.data(), requantized_data.data(),
                        K, N, seg.offset_n, seg.size_n
                    );
                    memcpy(current_write_ptr, packed_data_temp.data(), segment_packed_size);
                    current_write_ptr += segment_packed_size;
                }
            }
        }
    // Other tensor types
    } else {
        memcpy(tensor_dma_ptr + offset, data, size);
    }
}

static void ggml_backend_rknpu_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_rknpu_buffer_context * ctx = (ggml_backend_rknpu_buffer_context *)buffer->context;
    uint8_t* dma_base = (uint8_t*)ctx->dma_buf.virt_addr;
    uint8_t* tensor_dma_ptr = dma_base + ((uintptr_t)tensor->data - (uintptr_t)ggml_backend_buffer_get_base(buffer));
    memcpy(data, tensor_dma_ptr + offset, size);
}

static void ggml_backend_rknpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_rknpu_buffer_context * ctx = (ggml_backend_rknpu_buffer_context *)buffer->context;
    memset(ctx->dma_buf.virt_addr, value, ctx->dma_buf.size);
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

    rknpu2_allocation::DmaBuffer dma_buf = rknpu2_allocation::alloc(size);
    if (dma_buf.fd < 0) {
        return NULL;
    }

    ggml_backend_rknpu_buffer_context * ctx = new ggml_backend_rknpu_buffer_context{
        dma_buf, "rknpu_dma_buffer", {}, {}, {}
    };

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

    // Getting the current device configuration
    const auto& config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();
    const auto* op_support = config.find_op_support(tensor->type);

    // Padding Q4_0 weights to the next power of two for Hadamard Transform.
    if (tensor->type == GGML_TYPE_Q4_0) {
        if (op_support && op_support->npu_type_a == rknpu2_configuration::NPU_TYPE_INT4) {
            const int K = (int)tensor->ne[0];
            const int N = (int)tensor->ne[1];

            const int padded_K = rknpu2_calibration::next_power_of_two(K);

            auto segments = compute_matrix_segments(N, config.core_count, op_support->n_align);
            size_t total_size = 0;
            for (const auto& seg : segments) {
                if (seg.size_n > 0) {
                    total_size += (size_t)seg.size_n * padded_K / 2;
                }
            }
            return total_size;
        }
    }

    // Fallback to default size calculation for other types.
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

    // Getting the current device configuration
    const auto& config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();

    switch (op->op) {
        case GGML_OP_NONE:
            return true;

        case GGML_OP_MUL_MAT: {
            const struct ggml_tensor * src0 = op->src[0]; // Weights
            const struct ggml_tensor * src1 = op->src[1]; // Activations

            // Finding if there is a supported operation for the given weight type
            const auto* op_support = config.find_op_support(src0->type);
            if (!op_support) {
                return false;
            }

            // Checking if activation type matches the supported operation
            if (src1->type != op_support->type_a) {
                return false;
            }

            // Checking for K alignment
            if (src0->ne[0] % op_support->k_align != 0) {
                return false;
            }

            // Checking for N alignment
            if (src0->ne[1] % op_support->n_align != 0) {
                return false;
            }

            // Checking for exact dimensions
            if (src1->ne[0] != src0->ne[0]) {
                 return false;
            }

            // Checking contiguous memory
            if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
                return false;
            }

            return true;
        }
        default:
            return false;
    }
}

static ggml_backend_t ggml_backend_rknpu_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    UNUSED(dev);
    UNUSED(params);

    // TODO: Make device selection dynamic (e.g., from params or env var)
    if (!rknpu2_configuration::Rknpu2ConfigManager::get_instance().select_device("RK3588")) return NULL;

    ggml_backend_rknpu_context * ctx = new ggml_backend_rknpu_context();
    
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