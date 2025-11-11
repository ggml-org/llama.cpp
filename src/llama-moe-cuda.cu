#include "llama-moe.h"

#if defined(LLAMA_MOE_ENABLE) && defined(GGML_USE_CUDA)

#include "llama-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/convert.cuh"

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <inttypes.h>
#include <limits>
#include <memory>
#include <vector>

namespace {

static inline void llama_cuda_try(cudaError_t result, const char * expr) {
    if (result != cudaSuccess) {
        LLAMA_LOG_ERROR("%s: CUDA call failed: %s (%d)\n", __func__, cudaGetErrorString(result), result);
        GGML_ABORT("%s", expr);
    }
}

constexpr float kInvSqrt2 = 0.70710678118654752440f;
constexpr float kSwigluOaiAlpha = 1.702f;
constexpr float kSwigluOaiLimit = 7.0f;

struct device_buffer {
    float * ptr = nullptr;

    void allocate(size_t count) {
        if (count == 0) {
            return;
        }
        if (ptr != nullptr) {
            return;
        }
        llama_cuda_try(cudaMalloc(&ptr, count * sizeof(float)), "cudaMalloc device_buffer");
    }

    ~device_buffer() {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }

    device_buffer() = default;
    device_buffer(const device_buffer &) = delete;
    device_buffer & operator=(const device_buffer &) = delete;
};

struct pinned_buffer {
    void * ptr = nullptr;
    size_t bytes = 0;

    void allocate(size_t nbytes) {
        if (nbytes == 0) {
            return;
        }
        if (ptr != nullptr && bytes >= nbytes) {
            return;
        }
        release();
        llama_cuda_try(cudaMallocHost(&ptr, nbytes), "cudaMallocHost pinned_buffer");
        bytes = nbytes;
    }

    template <typename T>
    T * data() {
        return reinterpret_cast<T *>(ptr);
    }

    ~pinned_buffer() {
        release();
    }

    void release() {
        if (ptr != nullptr) {
            cudaFreeHost(ptr);
            ptr = nullptr;
            bytes = 0;
        }
    }

    pinned_buffer() = default;
    pinned_buffer(const pinned_buffer &) = delete;
    pinned_buffer & operator=(const pinned_buffer &) = delete;
};

static inline void llama_cublas_try(cublasStatus_t result, const char * expr) {
    if (result == CUBLAS_STATUS_SUCCESS) {
        return;
    }
    LLAMA_LOG_ERROR("%s: cuBLAS call failed: status=%d\n", __func__, (int) result);
    GGML_ABORT("%s", expr);
}

__device__ inline float act_silu(float x) {
    return x / (1.0f + __expf(-x));
}

__device__ inline float act_gelu(float x) {
    const float cdf = 0.5f * (1.0f + erff(x * kInvSqrt2));
    return x * cdf;
}

__device__ inline float act_relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

__global__ void add_bias_kernel(float * data, const float * bias, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += bias[idx];
    }
}

__global__ void apply_activation_kernel(
        llm_ffn_op_type type,
        const float * gate,
        const float * up,
        float * hidden,
        int64_t n,
        bool has_gate) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    const float up_val = up[idx];
    const float gate_val = has_gate ? gate[idx] : up_val;

    switch (type) {
        case LLM_FFN_SILU:
        case LLM_FFN_SWIGLU:
            hidden[idx] = has_gate ? act_silu(gate_val) * up_val : act_silu(up_val);
            break;
        case LLM_FFN_SWIGLU_OAI_MOE: {
            if (has_gate) {
                const float x = fminf(gate_val, kSwigluOaiLimit);
                const float y = fminf(fmaxf(up_val, -kSwigluOaiLimit), kSwigluOaiLimit);
                const float out_glu = x / (1.0f + __expf(kSwigluOaiAlpha * (-x)));
                hidden[idx] = out_glu * (y + 1.0f);
            } else {
                hidden[idx] = act_silu(up_val);
            }
            break;
        }
        case LLM_FFN_GELU:
        case LLM_FFN_GEGLU:
            hidden[idx] = has_gate ? act_gelu(gate_val) * up_val : act_gelu(up_val);
            break;
        case LLM_FFN_RELU:
        case LLM_FFN_REGLU:
            hidden[idx] = has_gate ? act_relu(gate_val) * up_val : act_relu(up_val);
            break;
        case LLM_FFN_RELU_SQR: {
            const float r = act_relu(up_val);
            hidden[idx] = r * r;
            break;
        }
        default:
            hidden[idx] = up_val;
            break;
    }
}

__global__ void scale_and_accumulate_kernel(float * dst, const float * src, float scale, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] += scale * src[idx];
    }
}

inline int compute_grid(int64_t n, int block_size = 256) {
    return static_cast<int>((n + block_size - 1) / block_size);
}

static void add_bias(cudaStream_t stream, float * data, const float * bias, int64_t n) {
    if (bias == nullptr || n == 0) {
        return;
    }
    const int block = 256;
    const int grid = compute_grid(n, block);
    add_bias_kernel<<<grid, block, 0, stream>>>(data, bias, n);
}

static void apply_activation(
        cudaStream_t stream,
        llm_ffn_op_type type,
        const float * gate,
        const float * up,
        float * hidden,
        int64_t n,
        bool has_gate) {
    if (n == 0) {
        return;
    }
    const int block = 256;
    const int grid = compute_grid(n, block);
    apply_activation_kernel<<<grid, block, 0, stream>>>(type, gate, up, hidden, n, has_gate);
}

static void accumulate(cudaStream_t stream, float * dst, const float * src, float scale, int64_t n) {
    if (n == 0) {
        return;
    }
    const int block = 256;
    const int grid = compute_grid(n, block);
    scale_and_accumulate_kernel<<<grid, block, 0, stream>>>(dst, src, scale, n);
}

static void copy_and_scale_input(
        cublasHandle_t handle,
        cudaStream_t stream,
        const float * input,
        float * tmp,
        int64_t n,
        float scale) {
    if (tmp == nullptr || input == nullptr || n == 0) {
        return;
    }
    llama_cuda_try(cudaMemcpyAsync(tmp, input, n * sizeof(float), cudaMemcpyDeviceToDevice, stream),
            "cudaMemcpyAsync expert input scale");
    llama_cublas_try(cublasSetStream(handle, stream), "cublasSetStream");
    llama_cublas_try(cublasSscal(handle, static_cast<int>(n), &scale, tmp, 1), "cublasSscal expert input");
}

static void run_matvec(
        cublasHandle_t handle,
        const float * weight,
        const float * input,
        float * output,
        int64_t rows,
        int64_t cols) {
    if (weight == nullptr || input == nullptr || output == nullptr || rows == 0 || cols == 0) {
        LLAMA_LOG_ERROR("%s: invalid matvec args rows=%" PRId64 " cols=%" PRId64 "\n", __func__, rows, cols);
        GGML_ABORT("invalid matvec args");
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // Treat row-major weight as column-major transpose.
    llama_cublas_try(
        cublasSgemv(handle,
                    CUBLAS_OP_T,
                    static_cast<int>(cols),
                    static_cast<int>(rows),
                    &alpha,
                    weight,
                    static_cast<int>(cols),
                    input,
                    1,
                    &beta,
                    output,
                    1),
        "cublasSgemv expert matvec");
}

static void validate_handle(const llama_moe_expert_handle * handle, int64_t expected_cols, int64_t expected_rows = -1) {
    if (handle == nullptr) {
        LLAMA_LOG_ERROR("%s: expert handle missing\n", __func__);
        GGML_ABORT("missing expert handle");
    }
    if (handle->type != GGML_TYPE_F32 && !ggml_is_quantized(handle->type)) {
        LLAMA_LOG_ERROR("%s: expert handle type %d unsupported\n", __func__, (int) handle->type);
        GGML_ABORT("unsupported expert handle type");
    }
    if (handle->cols != expected_cols) {
        LLAMA_LOG_ERROR("%s: expert columns mismatch (expected=%" PRId64 " got=%" PRId64 ")\n",
                __func__, expected_cols, handle->cols);
        GGML_ABORT("expert columns mismatch");
    }
    if (expected_rows != -1 && handle->rows != expected_rows) {
        LLAMA_LOG_ERROR("%s: expert rows mismatch (expected=%" PRId64 " got=%" PRId64 ")\n",
                __func__, expected_rows, handle->rows);
        GGML_ABORT("expert rows mismatch");
    }
}

} // namespace

void llama_moe_dispatch_cuda(
        const llama_moe_dispatch_desc & desc,
        ggml_tensor * dst,
        ggml_tensor * input,
        ggml_tensor * selected,
        ggml_tensor * weights) {
    GGML_ASSERT(desc.cache != nullptr);
    GGML_ASSERT(desc.backend != nullptr);

    auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(desc.backend->context);
    GGML_ASSERT(cuda_ctx != nullptr);

    cudaStream_t stream = cuda_ctx->stream();
    cublasHandle_t handle = cuda_ctx->cublas_handle();
    llama_cublas_try(cublasSetStream(handle, stream), "cublasSetStream");

    ExpertCache * cache = desc.cache;
    cache->attach_stream(stream, cuda_ctx->device);
    const int device_id = cuda_ctx->device;

    const int64_t n_embd = desc.n_embd;
    const int64_t n_tokens = desc.n_tokens;
    const int64_t n_ff = desc.n_ff;
    const int64_t top_k = selected->ne[0];

    const float * input_d = static_cast<const float *>(input->data);
    const float * weights_d = static_cast<const float *>(weights->data);
    const int32_t * selected_d = static_cast<const int32_t *>(selected->data);
    float * output_d = static_cast<float *>(dst->data);

    const size_t input_stride = input->nb[1] / sizeof(float);
    const size_t output_stride = dst->nb[1] / sizeof(float);
    // zero output
    llama_cuda_try(cudaMemsetAsync(output_d, 0, n_embd * n_tokens * sizeof(float), stream),
            "cudaMemsetAsync moe dst");

    pinned_buffer selected_h;
    pinned_buffer weights_h;
    selected_h.allocate(top_k * n_tokens * sizeof(int32_t));
    weights_h.allocate(top_k * n_tokens * sizeof(float));

    llama_cuda_try(cudaMemcpyAsync(
            selected_h.data<int32_t>(),
            selected_d,
            top_k * n_tokens * sizeof(int32_t),
            cudaMemcpyDeviceToHost,
            stream),
            "cudaMemcpyAsync selected experts");

    llama_cuda_try(cudaMemcpyAsync(
            weights_h.data<float>(),
            weights_d,
            top_k * n_tokens * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream),
            "cudaMemcpyAsync expert weights");

    llama_cuda_try(cudaStreamSynchronize(stream), "cudaStreamSynchronize moe selection copy");

    device_buffer input_scaled;
    device_buffer up_buf;
    device_buffer gate_buf;
    device_buffer hidden_buf;
    device_buffer down_buf;
    device_buffer up_weight_deq;
    device_buffer gate_weight_deq;
    device_buffer down_weight_deq;
    device_buffer up_bias_deq;
    device_buffer gate_bias_deq;
    device_buffer down_bias_deq;

    if (desc.weight_before_ffn) {
        input_scaled.allocate(n_embd);
    }
    up_buf.allocate(n_ff);
    hidden_buf.allocate(n_ff);
    if (desc.has_gate) {
        gate_buf.allocate(n_ff);
    }
    down_buf.allocate(n_embd);

    for (int64_t t = 0; t < n_tokens; ++t) {
        const float * token_input = input_d + t * input_stride;
        float * token_output = output_d + t * output_stride;

        for (int64_t k = 0; k < top_k; ++k) {
            const int32_t expert_index = selected_h.data<int32_t>()[t * top_k + k];
            if (expert_index < 0 || expert_index >= desc.n_expert) {
                continue;
            }

            const float router_weight = weights_h.data<float>()[t * top_k + k];
            if (!desc.weight_before_ffn && std::abs(router_weight) < std::numeric_limits<float>::min()) {
                continue;
            }

            const auto fetch_handle = [&](llama_moe_weight_kind kind) -> const llama_moe_expert_handle * {
                const int32_t composed_id = llama_moe_compose_id(desc.layer, expert_index, kind);
                return cache->find(composed_id);
            };

            const llama_moe_expert_handle * up_h = fetch_handle(llama_moe_weight_kind::UP_WEIGHT);
            const llama_moe_expert_handle * gate_h = desc.has_gate ? fetch_handle(llama_moe_weight_kind::GATE_WEIGHT) : nullptr;
            const llama_moe_expert_handle * down_h = fetch_handle(llama_moe_weight_kind::DOWN_WEIGHT);
            const llama_moe_expert_handle * up_b_h = desc.has_up_bias ? fetch_handle(llama_moe_weight_kind::UP_BIAS) : nullptr;
            const llama_moe_expert_handle * gate_b_h = desc.has_gate_bias ? fetch_handle(llama_moe_weight_kind::GATE_BIAS) : nullptr;
            const llama_moe_expert_handle * down_b_h = desc.has_down_bias ? fetch_handle(llama_moe_weight_kind::DOWN_BIAS) : nullptr;

            validate_handle(up_h, n_embd, n_ff);
            validate_handle(down_h, n_ff, n_embd);
            if (desc.has_gate) {
                validate_handle(gate_h, n_embd, n_ff);
            }

            auto load_tensor = [&](const llama_moe_expert_handle * handle,
                                   llama_moe_weight_kind kind,
                                   device_buffer & scratch) -> const float * {
                if (handle == nullptr) {
                    return nullptr;
                }
                const int32_t composed_id = llama_moe_compose_id(desc.layer, expert_index, kind);
                void * raw = cache->ensure_loaded(composed_id, device_id);
                if (raw == nullptr) {
                    return nullptr;
                }
                if (!ggml_is_quantized(handle->type)) {
                    return static_cast<const float *>(raw);
                }

                const to_fp32_cuda_t to_fp32 = handle->is_contiguous && handle->nb[1] == 0
                    ? ggml_get_to_fp32_cuda(handle->type)
                    : nullptr;
                const to_fp32_nc_cuda_t to_fp32_nc = (!handle->is_contiguous || handle->nb[1] != 0)
                    ? ggml_get_to_fp32_nc_cuda(handle->type)
                    : nullptr;

                if (to_fp32 == nullptr && to_fp32_nc == nullptr) {
                    LLAMA_LOG_ERROR("%s: no CUDA dequantizer for tensor type %d\n", __func__, (int) handle->type);
                    GGML_ABORT("missing CUDA dequantizer");
                }

                size_t elems = 1;
                for (int i = 0; i < handle->n_dims; ++i) {
                    const int64_t dim = handle->ne[i];
                    if (dim <= 0) {
                        break;
                    }
                    elems *= static_cast<size_t>(dim);
                }
                scratch.allocate(elems);
                if (to_fp32) {
                    to_fp32(raw, scratch.ptr, static_cast<int64_t>(elems), stream);
                } else {
                    const int64_t ne0 = handle->ne[0];
                    const int64_t ne1 = handle->ne[1];
                    const int64_t ne2 = handle->ne[2];
                    const int64_t ne3 = handle->ne[3];
                    const int64_t nb1 = handle->nb[1];
                    const int64_t nb2 = handle->nb[2];
                    const int64_t nb3 = handle->nb[3];
                    to_fp32_nc(raw, scratch.ptr, ne0, ne1, ne2, ne3, nb1, nb2, nb3, stream);
                }
                return scratch.ptr;
            };

            const float * up_w = load_tensor(up_h, llama_moe_weight_kind::UP_WEIGHT, up_weight_deq);
            const float * gate_w = desc.has_gate ? load_tensor(gate_h, llama_moe_weight_kind::GATE_WEIGHT, gate_weight_deq) : nullptr;
            const float * down_w = load_tensor(down_h, llama_moe_weight_kind::DOWN_WEIGHT, down_weight_deq);
            const float * up_b = desc.has_up_bias ? load_tensor(up_b_h, llama_moe_weight_kind::UP_BIAS, up_bias_deq) : nullptr;
            const float * gate_b = desc.has_gate_bias ? load_tensor(gate_b_h, llama_moe_weight_kind::GATE_BIAS, gate_bias_deq) : nullptr;
            const float * down_b = desc.has_down_bias ? load_tensor(down_b_h, llama_moe_weight_kind::DOWN_BIAS, down_bias_deq) : nullptr;

            if (up_w == nullptr || down_w == nullptr || (desc.has_gate && gate_w == nullptr)) {
                LLAMA_LOG_ERROR("%s: missing expert weights for layer %d expert %d\n", __func__, desc.layer, expert_index);
                GGML_ABORT("missing expert weights");
            }

            float * expert_input = const_cast<float *>(token_input);
            if (desc.weight_before_ffn) {
                expert_input = input_scaled.ptr;
                copy_and_scale_input(handle, stream, token_input, input_scaled.ptr, n_embd, router_weight);
            }

            run_matvec(handle, up_w, expert_input, up_buf.ptr, n_ff, n_embd);
            add_bias(stream, up_buf.ptr, up_b, n_ff);

            if (desc.has_gate) {
                run_matvec(handle, gate_w, expert_input, gate_buf.ptr, n_ff, n_embd);
                add_bias(stream, gate_buf.ptr, gate_b, n_ff);
            }

            apply_activation(stream, desc.activation, gate_buf.ptr, up_buf.ptr, hidden_buf.ptr, n_ff, desc.has_gate);

            run_matvec(handle, down_w, hidden_buf.ptr, down_buf.ptr, n_embd, n_ff);
            add_bias(stream, down_buf.ptr, down_b, n_embd);

            if (desc.weight_before_ffn) {
                accumulate(stream, token_output, down_buf.ptr, 1.0f, n_embd);
            } else {
                accumulate(stream, token_output, down_buf.ptr, router_weight, n_embd);
            }
        }
    }
}

#endif // defined(LLAMA_MOE_ENABLE) && defined(GGML_USE_CUDA)
