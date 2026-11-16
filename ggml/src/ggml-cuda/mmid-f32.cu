#include "mmid-f32.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#    include "dequantize.cuh"

static bool mmid_f32_layout_supported(const ggml_tensor * dst,
                                      ggml_type           weight_type,
                                      std::size_t         block_bytes,
                                      int                 block_elements) {
    if (dst->op != GGML_OP_MUL_MAT_ID || ggml_prec_get_src(dst, 1) != GGML_PREC_F32) {
        return false;
    }
    const auto * a   = dst->src[0];
    const auto * b   = dst->src[1];
    const auto * ids = dst->src[2];
    if (a->type != weight_type || b->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 ||
        dst->type != GGML_TYPE_F32 || a->ne[0] <= 0 || a->ne[0] > 4096 || a->ne[0] % block_elements || a->ne[1] <= 0 ||
        a->ne[1] > 4096 || a->ne[2] <= 0 || a->ne[2] > 128 || a->ne[3] != 1 || b->ne[0] != a->ne[0] ||
        b->ne[2] != ids->ne[1] || b->ne[3] != 1 || ids->ne[0] <= 0 || ids->ne[0] > 64 || ids->ne[0] > a->ne[2] ||
        ids->ne[1] != 1 || ids->ne[2] != 1 || ids->ne[3] != 1 || (b->ne[1] != 1 && b->ne[1] != ids->ne[0]) ||
        dst->ne[0] != a->ne[1] || dst->ne[1] != ids->ne[0] || dst->ne[2] != ids->ne[1] || dst->ne[3] != 1) {
        return false;
    }
    const std::size_t row = std::size_t(a->ne[0] / block_elements) * block_bytes;
    if (!(a->nb[0] == block_bytes && a->nb[1] == row && a->nb[2] >= std::size_t(a->ne[1]) * row &&
          a->nb[2] <= 8192 * row && a->nb[2] % block_bytes == 0 && b->nb[0] == sizeof(float) &&
          b->nb[1] >= std::size_t(b->ne[0]) * sizeof(float) && b->nb[1] <= 8192 * sizeof(float) &&
          b->nb[1] % sizeof(float) == 0 && ids->nb[0] == sizeof(int32_t) && dst->nb[0] == sizeof(float) &&
          dst->nb[1] >= std::size_t(dst->ne[0]) * sizeof(float) && dst->nb[1] <= 8192 * sizeof(float) &&
          dst->nb[1] % sizeof(float) == 0)) {
        return false;
    }
    return true;
}

bool ggml_cuda_mul_mat_id_f32_supported(const ggml_tensor * dst, int cc) {
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || dst->op != GGML_OP_MUL_MAT_ID || ggml_prec_get_src(dst, 1) != GGML_PREC_F32) {
        return false;
    }
    switch (dst->src[0]->type) {
        case GGML_TYPE_Q8_0:
            return mmid_f32_layout_supported(dst, GGML_TYPE_Q8_0, sizeof(block_q8_0), QK8_0);
        case GGML_TYPE_Q5_0:
            return mmid_f32_layout_supported(dst, GGML_TYPE_Q5_0, sizeof(block_q5_0), QK5_0);
        case GGML_TYPE_Q4_K:
            return mmid_f32_layout_supported(dst, GGML_TYPE_Q4_K, sizeof(block_q4_K), QK_K);
        default:
            return false;
    }
}

// One Q5_0 block is 32 values. dequantize_q5_0 returns the pair (iqs, iqs+16).
static __device__ float mmid_dot_q5_0(const block_q5_0 * row, int ib, const float * y) {
    const float * yb  = y + ib * QK5_0;
    float         acc = 0.0f;
#    pragma unroll
    for (int iqs = 0; iqs < QK5_0 / 2; ++iqs) {
        float2 v;
        dequantize_q5_0(row, ib, iqs, v);
        acc = fmaf(v.x, yb[iqs], acc);
        acc = fmaf(v.y, yb[iqs + QK5_0 / 2], acc);
    }
    return acc;
}

// One Q4_K superblock is 256 values. dequantize_q4_K needs 32 threads and writes
// a buffer. Here one thread owns the row, so the same nibble formula is summed in place.
static __device__ float mmid_dot_q4_K(const block_q4_K * row, int ib, const float * y) {
    const block_q4_K & b    = row[ib];
    const float        dall = __low2half(b.dm);
    const float        dmin = __high2half(b.dm);
    const float *      yb   = y + ib * QK_K;
    float              acc  = 0.0f;
#    pragma unroll
    for (int il = 0; il < 4; ++il) {
        uint8_t sc, mn;
        get_scale_min_k4(2 * il, b.scales, sc, mn);
        const float d1 = dall * sc;
        const float m1 = dmin * mn;
        get_scale_min_k4(2 * il + 1, b.scales, sc, mn);
        const float     d2 = dall * sc;
        const float     m2 = dmin * mn;
        const uint8_t * q  = b.qs + 32 * il;
#    pragma unroll
        for (int j = 0; j < 32; ++j) {
            acc = fmaf(d1 * float(q[j] & 0xF) - m1, yb[64 * il + j], acc);
            acc = fmaf(d2 * float(q[j] >> 4) - m2, yb[64 * il + 32 + j], acc);
        }
    }
    return acc;
}

static __device__ float mmid_dot_q8_0(const block_q8_0 * row, int ib, const float * y) {
    const block_q8_0 & q   = row[ib];
    const float        d   = __half2float(q.d);
    const float *      yb  = y + ib * QK8_0;
    float              acc = 0.0f;
#    pragma unroll
    for (int i = 0; i < QK8_0; i += 2) {
        acc = fmaf(__fmul_rn(d, float(q.qs[i])), yb[i], acc);
        acc = fmaf(__fmul_rn(d, float(q.qs[i + 1])), yb[i + 1], acc);
    }
    return acc;
}

// One block owns 128 consecutive rows of one slot. The activation stays F32
// and is loaded once; every row in the tile reuses that copy.
template <ggml_type type>
static __global__ void mul_mat_id_vec_f32(const char * __restrict__ weights,
                                          const char * __restrict__ x,
                                          const int32_t * __restrict__ ids,
                                          char * __restrict__ dst,
                                          int    k,
                                          int    m,
                                          int    experts,
                                          size_t expert_stride,
                                          size_t row_stride,
                                          size_t x_stride,
                                          size_t dst_stride,
                                          int    x_rows) {
    constexpr int           qk             = type == GGML_TYPE_Q5_0 ? QK5_0 : type == GGML_TYPE_Q4_K ? QK_K : QK8_0;
    constexpr int           rows_per_block = 128;
    const int               slot           = blockIdx.y;
    const int               row            = blockIdx.x * rows_per_block + threadIdx.x;
    const int               id             = ids[slot];
    extern __shared__ float yshared[];

    float * out = reinterpret_cast<float *>(dst + size_t(slot) * dst_stride);
    // A negative id skips the slot and zeroes its output row, per ggml_mul_mat_id.
    // Do not read expert 0, and do not sync: every thread sees the same id.
    if (id < 0) {
        if (row < m) {
            out[row] = 0.0f;
        }
        return;
    }
    if (id >= experts) {
        if (row < m) {
            out[row] = __int_as_float(0x7fc00000);
        }
        return;
    }

    const float * y = reinterpret_cast<const float *>(x + size_t(slot % x_rows) * x_stride);
    for (int i = threadIdx.x; i < k; i += rows_per_block) {
        yshared[i] = y[i];
    }
    __syncthreads();

    if (row >= m) {
        return;
    }

    const char * rowp = weights + size_t(id) * expert_stride + size_t(row) * row_stride;
    const int    nqb  = k / qk;
    float        acc  = 0.0f;
    for (int ib = 0; ib < nqb; ++ib) {
        if constexpr (type == GGML_TYPE_Q5_0) {
            acc += mmid_dot_q5_0(reinterpret_cast<const block_q5_0 *>(rowp), ib, yshared);
        } else if constexpr (type == GGML_TYPE_Q4_K) {
            acc += mmid_dot_q4_K(reinterpret_cast<const block_q4_K *>(rowp), ib, yshared);
        } else {
            acc += mmid_dot_q8_0(reinterpret_cast<const block_q8_0 *>(rowp), ib, yshared);
        }
    }
    out[row] = acc;
}

// The activation stays F32. Only one-token inputs reach this launcher.
template <ggml_type type> static void launch_mul_mat_id_vec_f32(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * a   = dst->src[0];
    const ggml_tensor * b   = dst->src[1];
    const ggml_tensor * ids = dst->src[2];
    GGML_ASSERT(dst->ne[2] == 1);
    const int     k              = int(a->ne[0]);
    const int     m              = int(a->ne[1]);
    const int     slots          = int(ids->ne[0]);
    constexpr int rows_per_block = 128;
    const int     tiles          = (m + rows_per_block - 1) / rows_per_block;
    mul_mat_id_vec_f32<type><<<dim3(tiles, slots), rows_per_block, size_t(k) * sizeof(float), ctx.stream()>>>(
        static_cast<const char *>(a->data), static_cast<const char *>(b->data), static_cast<const int32_t *>(ids->data),
        static_cast<char *>(dst->data), k, m, int(a->ne[2]), a->nb[2], a->nb[1], b->nb[1], dst->nb[1], int(b->ne[1]));
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_mul_mat_id_f32(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    switch (dst->src[0]->type) {
        case GGML_TYPE_Q8_0:
            launch_mul_mat_id_vec_f32<GGML_TYPE_Q8_0>(ctx, dst);
            break;
        case GGML_TYPE_Q5_0:
            launch_mul_mat_id_vec_f32<GGML_TYPE_Q5_0>(ctx, dst);
            break;
        case GGML_TYPE_Q4_K:
            launch_mul_mat_id_vec_f32<GGML_TYPE_Q4_K>(ctx, dst);
            break;
        default:
            GGML_ABORT("unsupported F32 expert weight type");
    }
}
#endif
