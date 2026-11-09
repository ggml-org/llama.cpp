#include "mmid-f32.cuh"
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include "dequantize.cuh"

static bool mmid_f32_layout_supported(const ggml_tensor * dst, ggml_type weight_type,
                                std::size_t block_bytes, int block_elements, int max_tokens) {
    if (dst->op != GGML_OP_MUL_MAT_ID || ggml_get_op_params_i32(dst, 3) != GGML_PREC_F32) return false;
    const auto * a=dst->src[0]; const auto * b=dst->src[1]; const auto * ids=dst->src[2];
    if (a->type != weight_type || b->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 || dst->type != GGML_TYPE_F32 ||
        a->ne[0] <= 0 || a->ne[0] > 4096 || a->ne[0] % block_elements || a->ne[1] <= 0 || a->ne[1] > 4096 ||
        a->ne[2] <= 0 || a->ne[2] > 128 || a->ne[3] != 1 ||
        b->ne[0] != a->ne[0] || b->ne[2] != ids->ne[1] || b->ne[3] != 1 ||
        ids->ne[0] <= 0 || ids->ne[0] > 64 || ids->ne[0] > a->ne[2] ||
        ids->ne[1] <= 0 || ids->ne[1] > max_tokens || ids->ne[2] != 1 || ids->ne[3] != 1 ||
        (b->ne[1] != 1 && b->ne[1] != ids->ne[0]) ||
        dst->ne[0] != a->ne[1] || dst->ne[1] != ids->ne[0] || dst->ne[2] != ids->ne[1] || dst->ne[3] != 1) return false;
    const std::size_t row=std::size_t(a->ne[0] / block_elements)*block_bytes;
    if (!(a->nb[0] == block_bytes && a->nb[1] == row &&
        a->nb[2] >= std::size_t(a->ne[1])*row && a->nb[2] <= 8192*row && a->nb[2] % block_bytes == 0 &&
        b->nb[0] == sizeof(float) && b->nb[1] >= std::size_t(b->ne[0])*sizeof(float) &&
        b->nb[1] <= 8192*sizeof(float) && b->nb[1] % sizeof(float) == 0 && ids->nb[0] == sizeof(int32_t) &&
        dst->nb[0] == sizeof(float) && dst->nb[1] >= std::size_t(dst->ne[0])*sizeof(float) &&
        dst->nb[1] <= 8192*sizeof(float) && dst->nb[1] % sizeof(float) == 0)) return false;
    if (ids->ne[1] == 1) return true;
    return b->nb[2] >= std::size_t(b->ne[1])*b->nb[1] && b->nb[2] <= 8192*b->nb[1] && b->nb[2] % sizeof(float) == 0 &&
        ids->nb[1] >= std::size_t(ids->ne[0])*sizeof(int32_t) && ids->nb[1] <= 8192*sizeof(int32_t) &&
        ids->nb[1] % sizeof(int32_t) == 0 &&
        dst->nb[2] >= std::size_t(dst->ne[1])*dst->nb[1] && dst->nb[2] <= 8192*dst->nb[1] &&
        dst->nb[2] % sizeof(float) == 0;
}

bool ggml_cuda_mul_mat_id_f32_supported(const ggml_tensor * dst, int cc) {
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || dst->op != GGML_OP_MUL_MAT_ID ||
            ggml_get_op_params_i32(dst, 3) != GGML_PREC_F32) {
        return false;
    }
    switch (dst->src[0]->type) {
        case GGML_TYPE_Q8_0:
            return mmid_f32_layout_supported(dst, GGML_TYPE_Q8_0, sizeof(block_q8_0), QK8_0, 1);
        case GGML_TYPE_Q5_0:
            return mmid_f32_layout_supported(dst, GGML_TYPE_Q5_0, sizeof(block_q5_0), QK5_0, 1);
        case GGML_TYPE_Q4_K:
            return mmid_f32_layout_supported(dst, GGML_TYPE_Q4_K, sizeof(block_q4_K), QK_K, 64);
        default:
            return false;
    }
}

// One Q5_0 block is 32 values. dequantize_q5_0 returns the pair (iqs, iqs+16).
static __device__ float mmid_dot_q5_0(const block_q5_0 * row, int ib, const float * y) {
    const float * yb = y + ib * QK5_0;
    float acc = 0.0f;
#pragma unroll
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
    const block_q4_K & b = row[ib];
    const float dall = __low2half(b.dm);
    const float dmin = __high2half(b.dm);
    const float * yb = y + ib * QK_K;
    float acc = 0.0f;
#pragma unroll
    for (int il = 0; il < 4; ++il) {
        uint8_t sc, mn;
        get_scale_min_k4(2 * il, b.scales, sc, mn);
        const float d1 = dall * sc;
        const float m1 = dmin * mn;
        get_scale_min_k4(2 * il + 1, b.scales, sc, mn);
        const float d2 = dall * sc;
        const float m2 = dmin * mn;
        const uint8_t * q = b.qs + 32 * il;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            acc = fmaf(d1 * float(q[j] & 0xF) - m1, yb[64 * il + j], acc);
            acc = fmaf(d2 * float(q[j] >> 4) - m2, yb[64 * il + 32 + j], acc);
        }
    }
    return acc;
}

static __device__ float mmid_dot_q8_0(const block_q8_0 * row, int ib, const float * y) {
    const block_q8_0 & q = row[ib];
    const float d = __half2float(q.d);
    const float * yb = y + ib * QK8_0;
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < QK8_0; i += 2) {
        acc = fmaf(__fmul_rn(d, float(q.qs[i])),     yb[i],     acc);
        acc = fmaf(__fmul_rn(d, float(q.qs[i + 1])), yb[i + 1], acc);
    }
    return acc;
}

// One block owns 128 consecutive rows of one slot. The activation stays F32
// and is loaded once; every row in the tile reuses that copy.
template <ggml_type type>
static __global__ void mul_mat_id_vec_f32(const char * __restrict__ weights, const char * __restrict__ x,
        const int32_t * __restrict__ ids, char * __restrict__ dst, int k, int m, int experts,
        size_t expert_stride, size_t row_stride, size_t x_stride, size_t dst_stride, int x_rows) {
    constexpr int qk = type == GGML_TYPE_Q5_0 ? QK5_0 : type == GGML_TYPE_Q4_K ? QK_K : QK8_0;
    constexpr int rows_per_block = 128;
    const int slot = blockIdx.y;
    const int row  = blockIdx.x * rows_per_block + threadIdx.x;
    const int id   = ids[slot];
    extern __shared__ float yshared[];

    float * out = reinterpret_cast<float *>(dst + size_t(slot) * dst_stride);
    // An invalid id is not an expert. Do not read expert 0, and do not sync:
    // every thread in the block sees the same id.
    if (id < 0 || id >= experts) {
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
    const int nqb = k / qk;
    float acc = 0.0f;
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

// This launch is one token. Q4_K with more tokens uses mul_mat_id_q4k_dp4a_f32.
template <ggml_type type>
static void launch_mul_mat_id_vec_f32(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * a   = dst->src[0];
    const ggml_tensor * b   = dst->src[1];
    const ggml_tensor * ids = dst->src[2];
    GGML_ASSERT(dst->ne[2] == 1);
    const int k     = int(a->ne[0]);
    const int m     = int(a->ne[1]);
    const int slots = int(ids->ne[0]);
    constexpr int rows_per_block = 128;
    const int tiles = (m + rows_per_block - 1) / rows_per_block;
    mul_mat_id_vec_f32<type><<<dim3(tiles, slots), rows_per_block, size_t(k) * sizeof(float), ctx.stream()>>>(
        static_cast<const char *>(a->data), static_cast<const char *>(b->data),
        static_cast<const int32_t *>(ids->data), static_cast<char *>(dst->data),
        k, m, int(a->ne[2]), a->nb[2], a->nb[1], b->nb[1], dst->nb[1], int(b->ne[1]));
    CUDA_CHECK(cudaGetLastError());
}

static __device__ int mmid_id_at(const char * ids, int token, int slot, size_t ids_nb1) {
    return *reinterpret_cast<const int32_t *>(ids + size_t(token) * ids_nb1 + size_t(slot) * sizeof(int32_t));
}

// 32 floats -> int8. Scale stays fp32 so amax=1e5 does not overflow.
static __device__ void mmid_quant_f32_i8(const float * y, int8_t * qs, float & scale) {
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) {
        amax = fmaxf(amax, fabsf(y[i]));
    }
    if (amax == 0.0f) {
        scale = 0.0f;
        for (int i = 0; i < 32; ++i) {
            qs[i] = 0;
        }
        return;
    }
    scale = amax / 127.0f;
    const float inv = 1.0f / scale;
    for (int i = 0; i < 32; ++i) {
        const float v = fminf(127.0f, fmaxf(-127.0f, rintf(y[i] * inv)));
        qs[i] = (int8_t) v;
    }
}

// One Q4_K superblock against NT int8 rows. Nibbles match mmid_dot_q4_K:
// low nibble of q[j] is y[64*il+j], high nibble is y[64*il+32+j].
template <int NT>
static __device__ void mmid_row_q4k_dp4a(const block_q4_K * rowp, int nqb, int k,
        const int8_t * qs, const float * scale, float acc[NT]) {
    const int nblk = k / 32;
    for (int ib = 0; ib < nqb; ++ib) {
        const block_q4_K & b = rowp[ib];
        const float dall = __low2half(b.dm);
        const float dmin = __high2half(b.dm);
#pragma unroll
        for (int il = 0; il < 4; ++il) {
            uint8_t sc, mn;
            get_scale_min_k4(2 * il, b.scales, sc, mn);
            const float d1 = dall * sc;
            const float m1 = dmin * mn;
            get_scale_min_k4(2 * il + 1, b.scales, sc, mn);
            const float d2 = dall * sc;
            const float m2 = dmin * mn;
            const uint8_t * q = b.qs + 32 * il;
            const int blk_lo = ib * 8 + 2 * il;
            int sum_lo[NT], sum_hi[NT], ysum_lo[NT], ysum_hi[NT];
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                sum_lo[t] = 0;
                sum_hi[t] = 0;
                ysum_lo[t] = 0;
                ysum_hi[t] = 0;
            }
            for (int j = 0; j < 32; j += 4) {
                const uint32_t raw = *reinterpret_cast<const uint32_t *>(q + j);
                const int lo = int(raw & 0x0F0F0F0F);
                const int hi = int((raw >> 4) & 0x0F0F0F0F);
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    const int8_t * ylo = qs + t * k + blk_lo * 32 + j;
                    const int8_t * yhi = ylo + 32;
                    const int ulo = *reinterpret_cast<const int *>(ylo);
                    const int uhi = *reinterpret_cast<const int *>(yhi);
                    sum_lo[t] = ggml_cuda_dp4a(lo, ulo, sum_lo[t]);
                    sum_hi[t] = ggml_cuda_dp4a(hi, uhi, sum_hi[t]);
                    ysum_lo[t] = ggml_cuda_dp4a(0x01010101, ulo, ysum_lo[t]);
                    ysum_hi[t] = ggml_cuda_dp4a(0x01010101, uhi, ysum_hi[t]);
                }
            }
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                const float slo = scale[t * nblk + blk_lo];
                const float shi = scale[t * nblk + blk_lo + 1];
                acc[t] += slo * (d1 * float(sum_lo[t]) - m1 * float(ysum_lo[t]))
                        + shi * (d2 * float(sum_hi[t]) - m2 * float(ysum_hi[t]));
            }
        }
    }
}

// One thread owns one row. Up to four y rows are quantized once per expert tile.
static __global__ void mul_mat_id_q4k_dp4a_f32(const char * __restrict__ weights, const char * __restrict__ x,
        const char * __restrict__ ids, char * __restrict__ dst, int k, int m, int experts, int slots, int ntokens,
        size_t expert_stride, size_t row_stride, size_t x_nb1, size_t x_nb2, size_t ids_nb1,
        size_t dst_nb1, size_t dst_nb2, int x_rows) {
    const int expert = int(blockIdx.y);
    const int row    = int(blockIdx.x * blockDim.x + threadIdx.x);

    extern __shared__ float yshared[];
    int8_t * qs = reinterpret_cast<int8_t *>(yshared);
    float * scale = reinterpret_cast<float *>(qs + 4 * k);

    __shared__ int nmatch;
    if (threadIdx.x == 0) {
        int n = 0;
        for (int token = 0; token < ntokens; ++token) {
            for (int slot = 0; slot < slots; ++slot) {
                if (mmid_id_at(ids, token, slot, ids_nb1) == expert) {
                    ++n;
                }
            }
        }
        nmatch = n;
    }
    __syncthreads();

    if (expert == 0 && row < m) {
        for (int token = 0; token < ntokens; ++token) {
            for (int slot = 0; slot < slots; ++slot) {
                const int id = mmid_id_at(ids, token, slot, ids_nb1);
                if (id < 0 || id >= experts) {
                    float * out = reinterpret_cast<float *>(dst + size_t(token) * dst_nb2 + size_t(slot) * dst_nb1);
                    out[row] = __int_as_float(0x7fc00000);
                }
            }
        }
    }
    if (nmatch == 0) {
        return;
    }

    const int nqb = k / QK_K;
    const int nblk = k / 32;
    const block_q4_K * rowp = row < m ? reinterpret_cast<const block_q4_K *>(
            weights + size_t(expert) * expert_stride + size_t(row) * row_stride) : nullptr;
    int done = 0;
    while (done < nmatch) {
        int tok[4] = { 0, 0, 0, 0 };
        int sl[4]  = { 0, 0, 0, 0 };
        int nt = 0;
        int seen = 0;
        for (int token = 0; token < ntokens; ++token) {
            for (int slot = 0; slot < slots; ++slot) {
                if (mmid_id_at(ids, token, slot, ids_nb1) != expert) {
                    continue;
                }
                if (seen++ < done) {
                    continue;
                }
                if (nt < 4) {
                    tok[nt] = token;
                    sl[nt]  = slot;
                    ++nt;
                }
            }
        }
        if (nt == 0) {
            break;
        }
        done += nt;
        const float * yp[4];
        for (int t = 0; t < nt; ++t) {
            yp[t] = reinterpret_cast<const float *>(x + size_t(tok[t]) * x_nb2 + size_t(sl[t] % x_rows) * x_nb1);
        }
        for (int g = threadIdx.x; g < nt * nblk; g += int(blockDim.x)) {
            const int t = g / nblk;
            const int blk = g - t * nblk;
            mmid_quant_f32_i8(yp[t] + blk * 32, qs + t * k + blk * 32, scale[t * nblk + blk]);
        }
        __syncthreads();
        if (rowp != nullptr) {
            float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            if (nt == 4) {
                mmid_row_q4k_dp4a<4>(rowp, nqb, k, qs, scale, acc);
            } else if (nt == 3) {
                mmid_row_q4k_dp4a<3>(rowp, nqb, k, qs, scale, acc);
            } else if (nt == 2) {
                mmid_row_q4k_dp4a<2>(rowp, nqb, k, qs, scale, acc);
            } else {
                mmid_row_q4k_dp4a<1>(rowp, nqb, k, qs, scale, acc);
            }
            for (int t = 0; t < nt; ++t) {
                float * out = reinterpret_cast<float *>(dst + size_t(tok[t]) * dst_nb2 + size_t(sl[t]) * dst_nb1);
                out[row] = acc[t];
            }
        }
        __syncthreads();
    }
}

static void launch_mul_mat_id_q4k_dp4a(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * a   = dst->src[0];
    const ggml_tensor * b   = dst->src[1];
    const ggml_tensor * ids = dst->src[2];
    GGML_ASSERT(dst->ne[2] > 1);
    const int k       = int(a->ne[0]);
    const int m       = int(a->ne[1]);
    const int experts = int(a->ne[2]);
    const int slots   = int(ids->ne[0]);
    const int ntokens = int(ids->ne[1]);
    const int threads = 128;
    const int nblk    = k / 32;
    const size_t shmem = size_t(4) * k + size_t(4) * nblk * sizeof(float);
    const int tiles = (m + threads - 1) / threads;
    mul_mat_id_q4k_dp4a_f32<<<dim3(tiles, experts), threads, shmem, ctx.stream()>>>(
        static_cast<const char *>(a->data), static_cast<const char *>(b->data),
        static_cast<const char *>(ids->data), static_cast<char *>(dst->data),
        k, m, experts, slots, ntokens, a->nb[2], a->nb[1], b->nb[1], b->nb[2], ids->nb[1],
        dst->nb[1], dst->nb[2], int(b->ne[1]));
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
            if (dst->ne[2] == 1) {
                launch_mul_mat_id_vec_f32<GGML_TYPE_Q4_K>(ctx, dst);
            } else {
                launch_mul_mat_id_q4k_dp4a(ctx, dst);
            }
            break;
        default:
            GGML_ABORT("unsupported F32 expert weight type");
    }
}
#endif
