#include <cmath>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <mudnncxx/mudnn.h>

#include "ggml-common.h"
#include "ggml-cuda.h"
#include "ggml-cuda/dequantize.cuh"
#include "mudnn.cuh"

float * get_unit_scale(int device);

namespace mudnn = musa::dnn;

// muDNN's low-precision MatMul (BatchMatMul::RunLt with MatMulLtParam scales) takes INT8 operands
// and an F32 accumulator only, so a quantized ggml weight has to be re-encoded as INT8 with one
// scale per output row before it can be used. The re-encoding, the activation quantization and the
// workspace management all stay inside the MUSA backend.
#define MUDNN_MM_Q_MIN_BATCH 32
#define MUDNN_MM_Q_KQUANT    256
#define MUDNN_MM_Q_THREADS   256

namespace {

// Scratch buffers are kept per device and only grow: the CUDA pool hands out fresh VMM mappings,
// which cost far more than the re-encoding itself once the operands are tens of megabytes.
struct mq_scratch {
    void * wq      = nullptr;
    void * aq      = nullptr;
    void * scale_w = nullptr;
    void * scale_a = nullptr;
    size_t wq_bytes      = 0;
    size_t aq_bytes      = 0;
    size_t scale_w_count = 0;
    size_t scale_a_count = 0;
};

std::unordered_map<int, mq_scratch> scratch_cache;
std::mutex scratch_mutex;

void mq_reserve(void *& ptr, size_t & have, size_t want) {
    if (have >= want) {
        return;
    }
    if (ptr != nullptr) {
        CUDA_CHECK(musaFree(ptr));
    }
    CUDA_CHECK(musaMalloc(&ptr, want));
    have = want;
}

mq_scratch & get_scratch(int device_id, size_t wq_bytes, size_t aq_bytes, size_t scale_w_count, size_t scale_a_count) {
    std::lock_guard<std::mutex> lock(scratch_mutex);
    mq_scratch & entry = scratch_cache[device_id];
    mq_reserve(entry.wq,      entry.wq_bytes,      wq_bytes);
    mq_reserve(entry.aq,      entry.aq_bytes,      aq_bytes);
    mq_reserve(entry.scale_w, entry.scale_w_count, scale_w_count * sizeof(float));
    mq_reserve(entry.scale_a, entry.scale_a_count, scale_a_count * sizeof(float));
    return entry;
}

constexpr float mq_int8_max = 127.0f;

__device__ __forceinline__ int8_t mq_quant_one(float v, float inv_scale) {
    float q = nearbyintf(v * inv_scale);
    q = fminf(fmaxf(q, -mq_int8_max), mq_int8_max);
    return (int8_t) q;
}

template <typename block_t>
__device__ __forceinline__ float mq_block_absmax(const void * row_base, int64_t iblk, const int type) {
    float2 v;
    float amax = 0.0f;
    for (int i = 0; i < 16; ++i) {
        if (type == 0) {
            dequantize_q4_0(row_base, iblk, i, v);
        } else if (type == 1) {
            dequantize_q5_0(row_base, iblk, i, v);
        } else {
            dequantize_q8_0(row_base, iblk, i, v);
        }
        amax = fmaxf(amax, fmaxf(fabsf(v.x), fabsf(v.y)));
    }
    return amax;
}

// Pass one over the weights: per-row absmax, which becomes the INT8 scale.
__global__ void mq_wmax_qk32_kernel(
        const void * __restrict__ vx, float * __restrict__ rowmax,
        const int64_t nrows, const int64_t nblk_row, const int64_t row_bytes, const int type) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nrows * nblk_row) {
        return;
    }
    const int64_t row  = idx / nblk_row;
    const int64_t iblk = idx % nblk_row;
    const void *  base = (const char *) vx + row * row_bytes;
    const float   amax = mq_block_absmax<block_q4_0>(base, iblk, type);
    atomicMax((int *) &rowmax[row], __float_as_int(amax));
}

__global__ void mq_wmax_super_kernel(
        const void * __restrict__ vx, float * __restrict__ rowmax,
        const int64_t nrows, const int64_t nblk_row, const int64_t row_bytes, const int type) {
    const int64_t idx = (int64_t) blockIdx.x;
    if (idx >= nrows * nblk_row) {
        return;
    }
    const int64_t row  = idx / nblk_row;
    const int64_t iblk = idx % nblk_row;
    const void *  base = (const char *) vx + row * row_bytes;
    const int     tid  = threadIdx.x;

    __shared__ float sval[MUDNN_MM_Q_KQUANT];
    if (type == 0) {
        if (tid < 32) {
            dequantize_q4_K(base, iblk, sval, tid);
        }
    } else {
        dequantize_q6_K(base, iblk, sval, tid);
    }
    __syncthreads();

    float amax = 0.0f;
    for (int i = tid; i < MUDNN_MM_Q_KQUANT; i += blockDim.x) {
        amax = fmaxf(amax, fabsf(sval[i]));
    }
    __shared__ float sreduce[64];
    sreduce[tid] = amax;
    __syncthreads();
    for (int step = blockDim.x / 2; step > 0; step >>= 1) {
        if (tid < step) {
            sreduce[tid] = fmaxf(sreduce[tid], sreduce[tid + step]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicMax((int *) &rowmax[row], __float_as_int(sreduce[0]));
    }
}

__global__ void mq_max_to_scale_kernel(float * __restrict__ value, const int64_t count) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        value[idx] = value[idx] > 0.0f ? value[idx] / mq_int8_max : 0.0f;
    }
}

// Pass two over the weights: write the INT8 payload with the per-row scale from pass one.
__global__ void mq_wquant_qk32_kernel(
        const void * __restrict__ vx, int8_t * __restrict__ vy, const float * __restrict__ rowscale,
        const int64_t nrows, const int64_t nblk_row, const int64_t row_bytes, const int type) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nrows * nblk_row) {
        return;
    }
    const int64_t row  = idx / nblk_row;
    const int64_t iblk = idx % nblk_row;
    const void *  base = (const char *) vx + row * row_bytes;
    const float   scl  = rowscale[row];
    const float   inv  = scl > 0.0f ? 1.0f / scl : 0.0f;

    int8_t * out = vy + (row * nblk_row + iblk) * 32;
    float2   v;
    for (int i = 0; i < 16; ++i) {
        if (type == 0) {
            dequantize_q4_0(base, iblk, i, v);
        } else if (type == 1) {
            dequantize_q5_0(base, iblk, i, v);
        } else {
            dequantize_q8_0(base, iblk, i, v);
        }
        out[2 * i + 0] = mq_quant_one(v.x, inv);
        out[2 * i + 1] = mq_quant_one(v.y, inv);
    }
}

__global__ void mq_wquant_super_kernel(
        const void * __restrict__ vx, int8_t * __restrict__ vy, const float * __restrict__ rowscale,
        const int64_t nrows, const int64_t nblk_row, const int64_t row_bytes, const int type) {
    const int64_t idx = (int64_t) blockIdx.x;
    if (idx >= nrows * nblk_row) {
        return;
    }
    const int64_t row  = idx / nblk_row;
    const int64_t iblk = idx % nblk_row;
    const void *  base = (const char *) vx + row * row_bytes;
    const int     tid  = threadIdx.x;

    __shared__ float sval[MUDNN_MM_Q_KQUANT];
    if (type == 0) {
        if (tid < 32) {
            dequantize_q4_K(base, iblk, sval, tid);
        }
    } else {
        dequantize_q6_K(base, iblk, sval, tid);
    }
    __syncthreads();

    const float scl = rowscale[row];
    const float inv = scl > 0.0f ? 1.0f / scl : 0.0f;
    int8_t *    out = vy + (row * nblk_row + iblk) * MUDNN_MM_Q_KQUANT;
    for (int i = tid; i < MUDNN_MM_Q_KQUANT; i += blockDim.x) {
        out[i] = mq_quant_one(sval[i], inv);
    }
}

// Activations: one scale per row (i.e. per token).
__global__ void mq_act_max_kernel(
        const void * __restrict__ vx, float * __restrict__ rowmax,
        const int64_t m, const int64_t k, const int src_f16) {
    const int64_t row = blockIdx.x;
    if (row >= m) {
        return;
    }
    float amax = 0.0f;
    if (!src_f16 && (k % 4) == 0) {
        const float4 * x4 = (const float4 *) ((const float *) vx + row * k);
        const int64_t n4 = k / 4;
        for (int64_t i = threadIdx.x; i < n4; i += blockDim.x) {
            const float4 v = x4[i];
            amax = fmaxf(amax, fabsf(v.x));
            amax = fmaxf(amax, fabsf(v.y));
            amax = fmaxf(amax, fabsf(v.z));
            amax = fmaxf(amax, fabsf(v.w));
        }
    } else {
        for (int64_t i = threadIdx.x; i < k; i += blockDim.x) {
            const float v = src_f16 ? __half2float(((const __half *) vx)[row * k + i])
                                    : ((const float *) vx)[row * k + i];
            amax = fmaxf(amax, fabsf(v));
        }
    }
    __shared__ float sreduce[MUDNN_MM_Q_THREADS];
    sreduce[threadIdx.x] = amax;
    __syncthreads();
    for (int step = blockDim.x / 2; step > 0; step >>= 1) {
        if ((int) threadIdx.x < step) {
            sreduce[threadIdx.x] = fmaxf(sreduce[threadIdx.x], sreduce[threadIdx.x + step]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        rowmax[row] = sreduce[0];
    }
}

__global__ void mq_act_quant_kernel(
        const void * __restrict__ vx, int8_t * __restrict__ vy, const float * __restrict__ rowscale,
        const int64_t m, const int64_t k, const int src_f16) {
    const int64_t row = blockIdx.x;
    if (row >= m) {
        return;
    }
    const float   scl  = rowscale[row];
    const float   inv  = scl > 0.0f ? 1.0f / scl : 0.0f;
    const int64_t base = row * k;
    if (!src_f16 && (k % 4) == 0) {
        const float4 * x4 = (const float4 *) ((const float *) vx + base);
        int32_t *      y4 = (int32_t *) (vy + base);
        const int64_t  n4 = k / 4;
        for (int64_t i = threadIdx.x; i < n4; i += blockDim.x) {
            const float4  v  = x4[i];
            const int32_t b0 = (int32_t) mq_quant_one(v.x, inv);
            const int32_t b1 = (int32_t) mq_quant_one(v.y, inv);
            const int32_t b2 = (int32_t) mq_quant_one(v.z, inv);
            const int32_t b3 = (int32_t) mq_quant_one(v.w, inv);
            y4[i] = (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24);
        }
    } else {
        for (int64_t i = threadIdx.x; i < k; i += blockDim.x) {
            const float v = src_f16 ? __half2float(((const __half *) vx)[base + i])
                                    : ((const float *) vx)[base + i];
            vy[base + i] = mq_quant_one(v, inv);
        }
    }
}

}  // namespace

// Q8_0 is left on the existing path on purpose: it has no measured headroom on this backend, so
// only the 4-bit family is taken over.
bool mudnnQuantMulMatSupported(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q6_K:
            return true;
        default:
            return false;
    }
}

bool mudnnMulMatQuant(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {
    if (!mudnnQuantMulMatSupported(src0->type)) {
        return false;
    }
    // This path encodes and caches a plain weight tensor. ggml_cuda_mul_mat_id() reaches
    // ggml_cuda_mul_mat() once per expert with stack-local GGML_OP_VIEW slices of a stacked weight,
    // and the route that normally handles those (the quantized matmul with ids) never runs for
    // them, so without this check the path claims MoE expert slices it was never written for.
    if (src0->op != GGML_OP_NONE || src0->view_src != nullptr) {
        return false;
    }
    if (src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_F32) {
        return false;
    }
    if (dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }

    const int64_t ne01 = src0->ne[1];
    const int64_t ne10 = src0->ne[0];
    const int64_t ne11 = src1->ne[1];
    if (ne11 < MUDNN_MM_Q_MIN_BATCH || ne01 < MUDNN_MM_Q_MIN_BATCH || ne10 < MUDNN_MM_Q_MIN_BATCH) {
        return false;
    }
    if (ne10 % MUDNN_MM_Q_KQUANT != 0) {
        return false;
    }

    const bool qk32  = src0->type == GGML_TYPE_Q4_0 || src0->type == GGML_TYPE_Q5_0;
    const int  wtype = src0->type == GGML_TYPE_Q4_0 ? 0
                     : src0->type == GGML_TYPE_Q5_0 ? 1
                     : src0->type == GGML_TYPE_Q8_0 ? 2
                     : src0->type == GGML_TYPE_Q4_K  ? 0 : 1;
    const int64_t nblk_row = qk32 ? ne10 / 32 : ne10 / MUDNN_MM_Q_KQUANT;
    const int64_t row_bytes = (int64_t) ggml_row_size(src0->type, ne10);

    musaStream_t stream = ctx.stream();

    int8_t * wq      = nullptr;
    float  * scale_w = nullptr;
    const bool cached = mudnnQuantMulMatCacheHit(src0, &wq, &scale_w);

    mq_scratch & scratch = get_scratch(
        ctx.device, cached ? 1 : (size_t) ne01 * ne10, (size_t) ne11 * ne10, cached ? 1 : (size_t) ne01, (size_t) ne11);
    int8_t * aq      = (int8_t *) scratch.aq;
    float *  scale_a = (float *) scratch.scale_a;
    if (!cached) {
        wq      = (int8_t *) scratch.wq;
        scale_w = (float *) scratch.scale_w;
    }

    if (!cached) {
        CUDA_CHECK(musaMemsetAsync(scale_w, 0, sizeof(float) * ne01, stream));

        const unsigned w_blocks = (unsigned) (ne01 * nblk_row);
        const unsigned w_threads = qk32 ? (w_blocks + MUDNN_MM_Q_THREADS - 1) / MUDNN_MM_Q_THREADS : w_blocks;
        if (qk32) {
            mq_wmax_qk32_kernel<<<w_threads, MUDNN_MM_Q_THREADS, 0, stream>>>(
                src0->data, scale_w, ne01, nblk_row, row_bytes, wtype);
        } else {
            mq_wmax_super_kernel<<<w_blocks, 64, 0, stream>>>(
                src0->data, scale_w, ne01, nblk_row, row_bytes, wtype);
        }
        CUDA_CHECK(musaGetLastError());
        mq_max_to_scale_kernel<<<(unsigned) ((ne01 + MUDNN_MM_Q_THREADS - 1) / MUDNN_MM_Q_THREADS), MUDNN_MM_Q_THREADS, 0, stream>>>(
            scale_w, ne01);
        CUDA_CHECK(musaGetLastError());

        if (qk32) {
            mq_wquant_qk32_kernel<<<w_threads, MUDNN_MM_Q_THREADS, 0, stream>>>(
                src0->data, wq, scale_w, ne01, nblk_row, row_bytes, wtype);
        } else {
            mq_wquant_super_kernel<<<w_blocks, 64, 0, stream>>>(
                src0->data, wq, scale_w, ne01, nblk_row, row_bytes, wtype);
        }
        CUDA_CHECK(musaGetLastError());
    }

    const int src_f16 = src1->type == GGML_TYPE_F16 ? 1 : 0;
    CUDA_CHECK(musaMemsetAsync(scale_a, 0, sizeof(float) * ne11, stream));
    mq_act_max_kernel<<<(unsigned) ne11, MUDNN_MM_Q_THREADS, 0, stream>>>(
        src1->data, scale_a, ne11, ne10, src_f16);
    CUDA_CHECK(musaGetLastError());
    mq_max_to_scale_kernel<<<(unsigned) ((ne11 + MUDNN_MM_Q_THREADS - 1) / MUDNN_MM_Q_THREADS), MUDNN_MM_Q_THREADS, 0, stream>>>(
        scale_a, ne11);
    CUDA_CHECK(musaGetLastError());

    mq_act_quant_kernel<<<(unsigned) ne11, MUDNN_MM_Q_THREADS, 0, stream>>>(
        src1->data, aq, scale_a, ne11, ne10, src_f16);
    CUDA_CHECK(musaGetLastError());

    const int64_t dims_a[2]  = { ne11, ne10 };
    const int64_t str_a[2]   = { ne10, 1 };
    const int64_t dims_b[2]  = { ne01, ne10 };
    const int64_t str_b[2]   = { ne10, 1 };
    const int64_t dims_c[2]  = { ne11, ne01 };
    const int64_t str_c[2]   = { ne01, 1 };
    const int64_t dims_sa[2] = { ne11, 1 };
    const int64_t str_sa[2]  = { 1, 1 };
    const int64_t dims_sb[2] = { ne01, 1 };
    const int64_t str_sb[2]  = { 1, 1 };
    const int64_t dims_one[2] = { 1, 1 };
    const int64_t str_one[2]  = { 1, 1 };

    float * scale_one = get_unit_scale(ctx.device);
    if (scale_one == nullptr) {
        return false;
    }

    mudnn::Tensor tensor_a, tensor_b, tensor_c, tensor_sa, tensor_sb, tensor_sd, tensor_none;

    MUDNN_CHECK(tensor_a.SetType(mudnn::Tensor::Type::INT8));
    MUDNN_CHECK(tensor_b.SetType(mudnn::Tensor::Type::INT8));
    MUDNN_CHECK(tensor_c.SetType(mudnn::Tensor::Type::FLOAT));
    MUDNN_CHECK(tensor_sa.SetType(mudnn::Tensor::Type::FLOAT));
    MUDNN_CHECK(tensor_sb.SetType(mudnn::Tensor::Type::FLOAT));
    MUDNN_CHECK(tensor_sd.SetType(mudnn::Tensor::Type::FLOAT));

    MUDNN_CHECK(tensor_a.SetNdInfo(2, dims_a, str_a));
    MUDNN_CHECK(tensor_b.SetNdInfo(2, dims_b, str_b));
    MUDNN_CHECK(tensor_c.SetNdInfo(2, dims_c, str_c));
    MUDNN_CHECK(tensor_sa.SetNdInfo(2, dims_sa, str_sa));
    MUDNN_CHECK(tensor_sb.SetNdInfo(2, dims_sb, str_sb));
    MUDNN_CHECK(tensor_sd.SetNdInfo(2, dims_one, str_one));

    MUDNN_CHECK(tensor_a.SetAddr(aq));
    MUDNN_CHECK(tensor_b.SetAddr(wq));
    MUDNN_CHECK(tensor_c.SetAddr(dst->data));
    MUDNN_CHECK(tensor_sa.SetAddr(scale_a));
    MUDNN_CHECK(tensor_sb.SetAddr(scale_w));
    MUDNN_CHECK(tensor_sd.SetAddr(scale_one));

    mudnn::BatchMatMul op;
    MUDNN_CHECK(op.SetTranspose(false, true));
    MUDNN_CHECK(op.SetAlpha(1.0));
    MUDNN_CHECK(op.SetBeta(0.0));
    MUDNN_CHECK(op.SetGamma(0.0));

    mudnn::MatMulLtParam param;
    MUDNN_CHECK(param.SetEpilogue(mudnn::MatMulLtParam::MatMulLtEpilogueMode::MATMULLT_EPILOGUE_DEFAULT));
    MUDNN_CHECK(param.SetScale(tensor_sa, tensor_sb, tensor_sd, tensor_sd));

    mudnn::Handle * handle = get_cached_handle(ctx.device);
    MUDNN_CHECK(handle->SetStream(stream));

    size_t workspace_size = 0;
    if (op.GetWorkspaceSize(*handle, workspace_size, tensor_c, tensor_a, tensor_b, tensor_none, param) != mudnn::Status::SUCCESS) {
        return false;
    }
    ggml_cuda_pool_alloc<char> workspace(ctx.pool(), workspace_size == 0 ? 1 : workspace_size);
    char * workspace_ptr = workspace.get();
    if (workspace_ptr == nullptr) {
        return false;
    }
    const mudnn::MemoryMaintainer maintainer = [workspace_ptr](size_t) {
        return mudnn::MemoryHandler(workspace_ptr, [](void *) {});
    };

    const mudnn::Status status = op.RunLt(
        *handle, tensor_c, tensor_a, tensor_b, tensor_none, tensor_none, param, maintainer);

    return status == mudnn::Status::SUCCESS;
}

// The INT8 cache lives in extra space at the end of the weight tensor's own allocation, and that
// space exists only because ggml_backend_cuda_buffer_type_get_alloc_size() adds
// mudnnQuantMulMatCacheSize() to the size it hands to the allocator. This function is therefore the
// single definition of the space: the allocation, the repack that fills it and the hit that reads
// it all have to derive from it, or the repack ends up writing bytes nobody reserved.
#define MUDNN_MM_Q_CACHE_ALIGN 256

static size_t mq_cache_payload(const ggml_tensor * src0) {
    return (size_t) src0->ne[1] * src0->ne[0] + (size_t) src0->ne[1] * sizeof(float);
}

size_t mudnnQuantMulMatCacheSize(const ggml_tensor * src0) {
    // A view shares its parent's allocation: the bytes behind its data belong to another tensor,
    // never to a cache, so a view neither owns one nor gets one reserved.
    if (src0->view_src != nullptr) {
        return 0;
    }
    if (src0->op != GGML_OP_NONE || !ggml_is_contiguous(src0)) {
        return 0;
    }
    switch (src0->type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q6_K:
            break;
        default:
            return 0;
    }
    if (src0->ne[0] % MUDNN_MM_Q_KQUANT != 0) {
        return 0;
    }
    // The INT8 payload, one float scale per output row, and the slack the cache base may need to
    // start on an aligned address.
    return mq_cache_payload(src0) + MUDNN_MM_Q_CACHE_ALIGN;
}

// data + ggml_nbytes() is not vector aligned for every supported shape -- Q6_K at ne[0] = 512 puts
// it at an offset of 4 mod 16 -- while the scale kernels take 4-byte atomic and store accesses on
// it, so the cache starts at the next MUDNN_MM_Q_CACHE_ALIGN boundary inside the reserved space.
// The payload is a multiple of MUDNN_MM_Q_KQUANT bytes, so the per-row scales stay aligned too.
static char * mq_cache_base(const ggml_tensor * src0) {
    return (char *) GGML_PAD((uintptr_t) ((char *) src0->data + ggml_nbytes(src0)), MUDNN_MM_Q_CACHE_ALIGN);
}

// True only when the buffer that holds src0 really carries the reservation. The buffer's own
// allocation size is the function that sized this tensor's slot, so a buffer that does not reserve
// the cache -- a host buffer, one on another device, a split buffer -- answers with something
// smaller than weight plus cache and is rejected here. The second test covers a tensor whose slot
// is shorter than its nominal size, e.g. one placed by hand inside a larger buffer.
static bool mq_cache_reserved(const ggml_tensor * src0, size_t want) {
    if (src0->buffer == nullptr) {
        return false;
    }
    if (ggml_backend_buffer_get_alloc_size(src0->buffer, src0) < ggml_nbytes(src0) + want) {
        return false;
    }
    return (const char *) mq_cache_base(src0) + mq_cache_payload(src0) <=
           (const char *) ggml_backend_buffer_get_base(src0->buffer) + ggml_backend_buffer_get_size(src0->buffer);
}

// A device buffer holding 1.0f, allocated once per device. A per-call pool allocation plus a
// 4-byte host-to-device copy costs about 96 us per call on this stack, which is pure overhead in
// a path that runs once per matmul.
float * get_unit_scale(int device) {
    static std::unordered_map<int, float *> unit_scales;
    static std::mutex unit_mutex;
    std::lock_guard<std::mutex> lock(unit_mutex);
    const auto it = unit_scales.find(device);
    if (it != unit_scales.end()) {
        return it->second;
    }
    ggml_cuda_set_device(device);
    float * ptr = nullptr;
    if (musaMalloc((void **) &ptr, sizeof(float)) != musaSuccess) {
        return nullptr;
    }
    const float one = 1.0f;
    if (musaMemcpy(ptr, &one, sizeof(float), musaMemcpyHostToDevice) != musaSuccess) {
        musaFree(ptr);
        return nullptr;
    }
    unit_scales[device] = ptr;
    return ptr;
}

// The extra space only holds valid INT8 weights after a repack has run for this exact tensor.
// The allocation size alone cannot tell, because a tensor whose data was uploaded by another
// path would otherwise be read as uninitialised memory.
std::unordered_map<const ggml_tensor *, const void *> repacked_tensors;
std::mutex repacked_mutex;

bool mudnnQuantMulMatCacheRepacked(const ggml_tensor * src0) {
    std::lock_guard<std::mutex> lock(repacked_mutex);
    const auto it = repacked_tensors.find(src0);
    return it != repacked_tensors.end() && it->second == src0->data;
}

bool mudnnQuantMulMatCacheHit(const ggml_tensor * src0, int8_t ** wq, float ** scale_w) {
    const size_t want = mudnnQuantMulMatCacheSize(src0);
    if (want == 0) {
        return false;
    }
    // The registry is the gate: only a tensor whose extra space was actually filled with INT8
    // weights may be read as one.
    if (!mudnnQuantMulMatCacheRepacked(src0)) {
        return false;
    }
    // Same reservation the repack needed before it wrote anything.
    if (!mq_cache_reserved(src0, want)) {
        return false;
    }
    *wq = (int8_t *) mq_cache_base(src0);
    *scale_w = (float *) (*wq + src0->ne[1] * src0->ne[0]);
    return true;
}

bool mudnnQuantMulMatCacheRepack(int device, musaStream_t stream, ggml_tensor * src0) {
    ggml_cuda_set_device(device);
    // The repack writes the cache, so it has to prove the same reservation the hit path requires,
    // from the same predicate that sized the allocation. It must not consult the registry: that
    // entry is written at the end of this function, so gating on it here would make the gate
    // circular and the repack would never run, leaving every matmul to re-encode the weight inline.
    const size_t want = mudnnQuantMulMatCacheSize(src0);
    if (want == 0) {
        return false;
    }
    // Only the plain MUSA buffer for this device is sized by the allocation size the reservation
    // above accounts for.
    if (src0->buffer == nullptr || ggml_backend_buffer_get_type(src0->buffer) != ggml_backend_cuda_buffer_type(device)) {
        return false;
    }
    if (!mq_cache_reserved(src0, want)) {
        return false;
    }
    int8_t * wq      = (int8_t *) mq_cache_base(src0);
    float  * scale_w = (float *) (wq + (size_t) src0->ne[1] * src0->ne[0]);
    const int64_t ne01      = src0->ne[1];
    const int64_t ne10      = src0->ne[0];
    const int64_t row_bytes = (int64_t) ggml_row_size(src0->type, ne10);
    const bool    qk32      = src0->type == GGML_TYPE_Q4_0 || src0->type == GGML_TYPE_Q5_0;
    // The encode kernels take the compact weight code, not the ggml type: passing src0->type here
    // dequantized every cache entry with the wrong block layout (Q4_K and Q4_0 as Q6_K/Q8_0), so a
    // cache filled here never matched the same tensor encoded inline.
    const int     wtype     = qk32 ? (src0->type == GGML_TYPE_Q4_0 ? 0 : 1)
                                   : (src0->type == GGML_TYPE_Q4_K  ? 0 : 1);
    const int64_t nblk_row  = qk32 ? ne10 / 32 : ne10 / MUDNN_MM_Q_KQUANT;
    const int64_t w_blocks  = ne01 * nblk_row;

    CUDA_CHECK(musaMemsetAsync(scale_w, 0, sizeof(float) * ne01, stream));
    if (qk32) {
        mq_wmax_qk32_kernel<<<(unsigned) ((w_blocks + MUDNN_MM_Q_THREADS - 1) / MUDNN_MM_Q_THREADS), MUDNN_MM_Q_THREADS, 0, stream>>>(
            src0->data, scale_w, ne01, nblk_row, row_bytes, wtype);
    } else {
        mq_wmax_super_kernel<<<(unsigned) w_blocks, 64, 0, stream>>>(
            src0->data, scale_w, ne01, nblk_row, row_bytes, wtype);
    }
    CUDA_CHECK(musaGetLastError());
    mq_max_to_scale_kernel<<<(unsigned) ((ne01 + MUDNN_MM_Q_THREADS - 1) / MUDNN_MM_Q_THREADS), MUDNN_MM_Q_THREADS, 0, stream>>>(
        scale_w, ne01);
    CUDA_CHECK(musaGetLastError());
    if (qk32) {
        mq_wquant_qk32_kernel<<<(unsigned) ((w_blocks + MUDNN_MM_Q_THREADS - 1) / MUDNN_MM_Q_THREADS), MUDNN_MM_Q_THREADS, 0, stream>>>(
            src0->data, wq, scale_w, ne01, nblk_row, row_bytes, wtype);
    } else {
        mq_wquant_super_kernel<<<(unsigned) w_blocks, 64, 0, stream>>>(
            src0->data, wq, scale_w, ne01, nblk_row, row_bytes, wtype);
    }
    CUDA_CHECK(musaGetLastError());
    CUDA_CHECK(musaDeviceSynchronize());
    {
        std::lock_guard<std::mutex> lock(repacked_mutex);
        repacked_tensors[src0] = src0->data;
    }
    return true;
}
