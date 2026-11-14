#include "token-mask.cuh"
#include "ggml-token-mask.h"

static constexpr int TOKEN_MASK_TILE = 4096;

static __device__ int32_t token_mask_lower_bound(const int32_t * ids, int32_t count, int64_t value) {
    int32_t lo = 0, hi = count;
    while (lo < hi) {
        const int32_t mid = lo + (hi - lo)/2;
        if (ids[mid] < value) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

template<bool identity>
static __global__ void token_mask_f32(
        const char * logits, const char * ids, const int32_t * payload, const char * desc,
        float * dst, int64_t candidates, int64_t rows, int64_t payload_size,
        size_t logits_stride, size_t ids_stride, size_t desc_stride) {
    __shared__ int32_t first, last;
    for (int64_t row = blockIdx.y; row < rows; row += gridDim.y) {
        const int32_t * d = (const int32_t *) (desc + row*desc_stride);
        if (!ggml_token_mask_valid_desc(d, payload_size, candidates, identity)) {
            if (threadIdx.x == 0) {
                printf("TOKEN_MASK: invalid descriptor at row %lld\n", (long long) row);
                __trap();
            }
            return;
        }
        const bool sparse = d[0] == GGML_TOKEN_MASK_ALLOW_IDS || d[0] == GGML_TOKEN_MASK_DENY_IDS;
        const int32_t * list = payload + d[2];
        if (sparse) {
            for (int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
                    i < d[1]; i += (int64_t) gridDim.x*blockDim.x) {
                const int32_t id = list[i];
                if (id < 0 || id >= d[3] || (i && id <= list[i - 1])) {
                    printf("TOKEN_MASK: invalid payload at row %lld\n", (long long) row);
                    __trap();
                }
            }
        }
        const float * src = (const float *) (logits + row*logits_stride);
        for (int64_t begin = (int64_t) blockIdx.x*TOKEN_MASK_TILE; begin < candidates;
                begin += (int64_t) gridDim.x*TOKEN_MASK_TILE) {
            const int64_t end = min(begin + TOKEN_MASK_TILE, candidates);
            if (identity && sparse) {
                if (threadIdx.x == 0) {
                    first = token_mask_lower_bound(list, d[1], begin);
                    last = token_mask_lower_bound(list, d[1], end);
                }
                for (int64_t col = begin + threadIdx.x; col < end; col += blockDim.x) {
                    dst[row*candidates + col] = d[0] == GGML_TOKEN_MASK_ALLOW_IDS ? -INFINITY : src[col];
                }
                __syncthreads();
                for (int64_t i = (int64_t) first + threadIdx.x; i < last; i += blockDim.x) {
                    const int32_t id = list[i];
                    if (id >= begin && id < end) {
                        dst[row*candidates + id] = d[0] == GGML_TOKEN_MASK_ALLOW_IDS ? src[id] : -INFINITY;
                    }
                }
                __syncthreads();
            } else {
                for (int64_t col = begin + threadIdx.x; col < end; col += blockDim.x) {
                    const int32_t id = identity ? (int32_t) col : ((const int32_t *) (ids + row*ids_stride))[col];
                    dst[row*candidates + col] = ggml_token_mask_allowed(d, payload, id) ? src[col] : -INFINITY;
                }
            }
        }
    }
}

void ggml_cuda_op_token_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * logits = dst->src[0];
    const ggml_tensor * ids     = dst->src[1];
    const ggml_tensor * payload = dst->src[2];
    const ggml_tensor * desc    = dst->src[3];
    GGML_ASSERT(ggml_token_mask_valid_layout(logits, ids, payload, desc));
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && ggml_are_same_shape(logits, dst) && ggml_is_contiguous(dst));

    const int block_size = 256;
    const dim3 blocks(
        (unsigned) MIN((logits->ne[0] + TOKEN_MASK_TILE - 1)/TOKEN_MASK_TILE, INT64_C(65535)),
        (unsigned) MIN(logits->ne[1], INT64_C(65535)));
    if (ids) {
        token_mask_f32<false><<<blocks, block_size, 0, ctx.stream()>>>(
            (const char *) logits->data, (const char *) ids->data,
            (const int32_t *) payload->data, (const char *) desc->data, (float *) dst->data,
            logits->ne[0], logits->ne[1], payload->ne[0], logits->nb[1], ids->nb[1], desc->nb[1]);
    } else {
        token_mask_f32<true><<<blocks, block_size, 0, ctx.stream()>>>(
            (const char *) logits->data, nullptr,
            (const int32_t *) payload->data, (const char *) desc->data, (float *) dst->data,
            logits->ne[0], logits->ne[1], payload->ne[0], logits->nb[1], 0, desc->nb[1]);
    }
}
