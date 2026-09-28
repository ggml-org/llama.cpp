#include "common.cuh"
#include "convert.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-sparse.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#define FATTN_SPARSE_LIST_NTHREADS 128

// one list of kv rows per (kv head, query): the visible rows of the selected blocks, padded with -1
// the counts of all lists follow the lists, this is the layout that the sparse MMA FA kernel reads
static __global__ void flash_attn_sparse_build_lists(
        const int * __restrict__ blk_idx, const int * __restrict__ q_pos, const int * __restrict__ pos_cell,
        int * __restrict__ lists, int * __restrict__ counts,
        const int n_sel, const int blk, const int n_kv, const int n_pos, const int n_kv_max,
        const int64_t s_i1, const int64_t s_i2) {
    constexpr int nwarps = FATTN_SPARSE_LIST_NTHREADS/WARP_SIZE;

    const int t    = blockIdx.x;
    const int h    = blockIdx.y;
    const int tid  = threadIdx.x;
    const int warp = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;

    const int * bids = blk_idx + h*s_i1 + t*s_i2;
    const int   p_q  = q_pos[t];
    int       * list = lists + (int64_t(h)*gridDim.x + t)*n_kv_max;

    __shared__ int warp_counts[nwarps];

    int n = 0;
    for (int is = 0; is < n_sel; ++is) {
        const int b = bids[is];
        if (b < 0) {
            continue;
        }
        const int p1 = min(min((b + 1)*blk, p_q + 1), n_pos);
        for (int p0 = b*blk; p0 < p1; p0 += FATTN_SPARSE_LIST_NTHREADS) {
            const int p = p0 + tid;
            const int c = p < p1 ? pos_cell[p] : -1;
            const bool valid = c >= 0 && c < n_kv;

            const uint32_t ballot = __ballot_sync(0xFFFFFFFF, valid);
            if (lane == 0) {
                warp_counts[warp] = __popc(ballot);
            }
            __syncthreads();

            int offset = n;
            int total  = 0;
#pragma unroll
            for (int w = 0; w < nwarps; ++w) {
                offset += w < warp ? warp_counts[w] : 0;
                total  += warp_counts[w];
            }
            if (valid) {
                list[offset + __popc(ballot & ((1u << lane) - 1))] = c;
            }
            n += total;
            __syncthreads();
        }
    }

    for (int i = n + tid; i < n_kv_max; i += FATTN_SPARSE_LIST_NTHREADS) {
        list[i] = -1;
    }
    if (tid == 0) {
        counts[int64_t(h)*gridDim.x + t] = n;
    }
}

// [D, gqa, n_batch, n_head_kv] -> [D, gqa*n_head_kv, n_batch]
static __global__ void flash_attn_sparse_permute_dst(
        const float * __restrict__ src, float * __restrict__ dst, const int D, const int gqa, const int n_batch, const int n_head_kv) {
    const int g = blockIdx.x;
    const int t = blockIdx.y;
    const int h = blockIdx.z;

    const float * s = src + ((int64_t(h)*n_batch + t)*gqa + g)*D;
    float       * d = dst + ((int64_t(t)*n_head_kv + h)*gqa + g)*D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        d[i] = s[i];
    }
}

// runs the sparse gather path of the MMA FA kernel: q is viewed as [D, n_batch, gqa, n_head_kv] and k/v as [D, n_kv, 1, n_head_kv]
// so that each (kv head, query) pair gets its own index list and needs no mask
template <int D, int ncols2>
static void launch_flash_attn_sparse_mma(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q        = dst->src[0];
    const ggml_tensor * k        = dst->src[1];
    const ggml_tensor * v        = dst->src[2];
    const ggml_tensor * blk_idx  = dst->src[3];
    const ggml_tensor * q_pos    = dst->src[4];
    const ggml_tensor * pos_cell = dst->src[5];

    const float scale = ggml_get_op_params_f32(dst, 0);
    const int   blk   = ggml_get_op_params_i32(dst, 1);

    const int64_t n_batch   = q->ne[1];
    const int64_t n_head    = q->ne[2];
    const int64_t n_head_kv = k->ne[2];
    const int64_t gqa       = n_head/n_head_kv;
    const int64_t n_kv      = k->ne[1];
    const int64_t n_pos     = pos_cell->ne[0];
    const int64_t n_sel     = blk_idx->ne[0];
    const int64_t n_kv_max  = n_sel*blk;
    const int64_t n_lists   = n_batch*n_head_kv;

    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<int>   lists(ctx.pool(), n_kv_max*n_lists + n_lists);
    ggml_cuda_pool_alloc<float> tmp  (ctx.pool(), D*n_head*n_batch);

    for (int64_t s = 0; s < q->ne[3]; ++s) {
        {
            const dim3 block_nums(n_batch, n_head_kv, 1);
            const dim3 block_dims(FATTN_SPARSE_LIST_NTHREADS, 1, 1);
            flash_attn_sparse_build_lists<<<block_nums, block_dims, 0, stream>>>(
                (const int *) ((const char *) blk_idx->data + s*blk_idx->nb[3]),
                (const int *) ((const char *) q_pos->data   + s*q_pos->nb[1]),
                (const int *) ((const char *) pos_cell->data + s*pos_cell->nb[1]),
                lists.ptr, lists.ptr + n_kv_max*n_lists,
                n_sel, blk, n_kv, n_pos, n_kv_max,
                blk_idx->nb[1]/sizeof(int), blk_idx->nb[2]/sizeof(int));
            CUDA_CHECK(cudaGetLastError());
        }

        ggml_tensor q4 = *q;
        q4.data  = (char *) q->data + s*q->nb[3];
        q4.ne[2] = gqa;
        q4.ne[3] = n_head_kv;
        q4.nb[3] = q->nb[2]*gqa;

        ggml_tensor k4 = *k;
        k4.data  = (char *) k->data + s*k->nb[3];
        k4.ne[2] = 1;
        k4.ne[3] = n_head_kv;
        k4.nb[3] = k->nb[2];
        k4.nb[2] = k->nb[2]*n_head_kv;

        ggml_tensor v4 = *v;
        v4.data  = (char *) v->data + s*v->nb[3];
        v4.ne[2] = 1;
        v4.ne[3] = n_head_kv;
        v4.nb[3] = v->nb[2];
        v4.nb[2] = v->nb[2]*n_head_kv;

        ggml_tensor lt = {};
        lt.type  = GGML_TYPE_I32;
        lt.ne[0] = n_kv_max;
        lt.data  = lists.ptr;

        ggml_tensor fa = {};
        fa.op    = GGML_OP_FLASH_ATTN_EXT;
        fa.type  = GGML_TYPE_F32;
        fa.ne[0] = D;
        fa.ne[1] = gqa;
        fa.ne[2] = n_batch;
        fa.ne[3] = n_head_kv;
        fa.nb[0] = sizeof(float);
        for (int i = 1; i < GGML_MAX_DIMS; ++i) {
            fa.nb[i] = fa.nb[i - 1]*fa.ne[i - 1];
        }
        fa.data   = tmp.ptr;
        fa.src[0] = &q4;
        fa.src[1] = &k4;
        fa.src[2] = &v4;
        fa.src[5] = &lt;
        ggml_set_op_params_f32(&fa, 0, scale);
        ggml_set_op_params_i32(&fa, 3, GGML_PREC_F32);
        ggml_set_op_params_i32(&fa, 4, n_kv_max);

        ggml_cuda_flash_attn_ext_mma_f16_case<D, D, 1, ncols2>(ctx, &fa);

        {
            const dim3 block_nums(gqa, n_batch, n_head_kv);
            flash_attn_sparse_permute_dst<<<block_nums, D, 0, stream>>>(
                tmp.ptr, (float *) ((char *) dst->data + s*dst->nb[3]), D, gqa, n_batch, n_head_kv);
            CUDA_CHECK(cudaGetLastError());
        }
    }
}

static bool flash_attn_sparse_use_mma(const ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    return GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) &&
        k->ne[0] == 128 && k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16 &&
        (q->ne[2]/k->ne[2]) % 8 == 0;
}
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// one warp per query head, each lane holds D/WARP_SIZE elements of Q and of the VKQ accumulator
template <int D, typename T>
static __global__ void flash_attn_sparse_vec(
        const char * __restrict__ Q, const char * __restrict__ K, const char * __restrict__ V,
        const char * __restrict__ blk_idx, const char * __restrict__ q_pos, const char * __restrict__ pos_cell,
        float * __restrict__ dst,
        const int n_head, const int n_batch, const int gqa, const int n_sel, const int blk, const int n_kv, const int n_pos,
        const float scale,
        const int64_t nbq1, const int64_t nbq2, const int64_t nbq3,
        const int64_t nbk1, const int64_t nbk2, const int64_t nbk3,
        const int64_t nbv1, const int64_t nbv2, const int64_t nbv3,
        const int64_t nbi1, const int64_t nbi2, const int64_t nbi3,
        const int64_t nbp1, const int64_t nbc1) {
    constexpr int per_lane = D/WARP_SIZE;

    const int lane = threadIdx.x;
    const int iq1  = blockIdx.x;
    const int iq2  = blockIdx.y*blockDim.y + threadIdx.y;
    const int iq3  = blockIdx.z;

    if (iq2 >= n_head) {
        return;
    }

    const int ik2 = iq2/gqa;

    const float * q = (const float *) (Q + iq1*nbq1 + iq2*nbq2 + iq3*nbq3);
    float q_reg[per_lane];
#pragma unroll
    for (int i = 0; i < per_lane; ++i) {
        q_reg[i] = q[i*WARP_SIZE + lane]*scale;
    }

    const int       p_q  = *(const int *) (q_pos + iq1*sizeof(int) + iq3*nbp1);
    const int     * bids =  (const int *) (blk_idx + ik2*nbi1 + iq1*nbi2 + iq3*nbi3);
    const int     * pc   =  (const int *) (pos_cell + iq3*nbc1);
    const char    * K_h  = K + ik2*nbk2 + iq3*nbk3;
    const char    * V_h  = V + ik2*nbv2 + iq3*nbv3;

    float M = -INFINITY;
    float S = 0.0f;
    float acc[per_lane] = { 0.0f };

    for (int is = 0; is < n_sel; ++is) {
        const int b = bids[is];
        if (b < 0) {
            continue;
        }
        const int p1 = min(min((b + 1)*blk, p_q + 1), n_pos);
        for (int p = b*blk; p < p1; ++p) {
            const int ic = pc[p];
            if (ic < 0 || ic >= n_kv) {
                continue;
            }

            const T * k = (const T *) (K_h + ic*nbk1);
            float s = 0.0f;
#pragma unroll
            for (int i = 0; i < per_lane; ++i) {
                s += q_reg[i]*ggml_cuda_cast<float>(k[i*WARP_SIZE + lane]);
            }
            s = warp_reduce_sum(s);

            const float M_new = fmaxf(M, s);
            const float ms    = expf(M - M_new);
            const float vs    = expf(s - M_new);
            M = M_new;
            S = S*ms + vs;

            const T * v = (const T *) (V_h + ic*nbv1);
#pragma unroll
            for (int i = 0; i < per_lane; ++i) {
                acc[i] = acc[i]*ms + vs*ggml_cuda_cast<float>(v[i*WARP_SIZE + lane]);
            }
        }
    }

    const float S_inv = S == 0.0f ? 0.0f : 1.0f/S;

    // permute(0, 2, 1, 3)
    float * d = dst + ((int64_t(iq3)*n_batch + iq1)*n_head + iq2)*D;
#pragma unroll
    for (int i = 0; i < per_lane; ++i) {
        d[i*WARP_SIZE + lane] = acc[i]*S_inv;
    }
}

template <int D, typename T>
static void launch_flash_attn_sparse_vec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q        = dst->src[0];
    const ggml_tensor * k        = dst->src[1];
    const ggml_tensor * v        = dst->src[2];
    const ggml_tensor * blk_idx  = dst->src[3];
    const ggml_tensor * q_pos    = dst->src[4];
    const ggml_tensor * pos_cell = dst->src[5];

    const float scale = ggml_get_op_params_f32(dst, 0);
    const int   blk   = ggml_get_op_params_i32(dst, 1);

    const int n_head  = q->ne[2];
    const int n_batch = q->ne[1];
    const int gqa     = q->ne[2]/k->ne[2];
    const int n_kv    = k->ne[1];
    const int n_pos   = pos_cell->ne[0];

    const int nwarps = 4;
    const dim3 block_dims(WARP_SIZE, nwarps, 1);
    const dim3 block_nums(n_batch, (n_head + nwarps - 1)/nwarps, q->ne[3]);

    flash_attn_sparse_vec<D, T><<<block_nums, block_dims, 0, ctx.stream()>>>(
        (const char *) q->data, (const char *) k->data, (const char *) v->data,
        (const char *) blk_idx->data, (const char *) q_pos->data, (const char *) pos_cell->data,
        (float *) dst->data,
        n_head, n_batch, gqa, blk_idx->ne[0], blk, n_kv, n_pos, scale,
        q->nb[1], q->nb[2], q->nb[3],
        k->nb[1], k->nb[2], k->nb[3],
        v->nb[1], v->nb[2], v->nb[3],
        blk_idx->nb[1], blk_idx->nb[2], blk_idx->nb[3],
        q_pos->nb[1], pos_cell->nb[1]);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_flash_attn_sparse(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (flash_attn_sparse_use_mma(dst)) {
        // all q heads of a kv head in one tile, so that each index list is gathered once
        if ((dst->src[0]->ne[2]/dst->src[1]->ne[2]) % 16 == 0) {
            launch_flash_attn_sparse_mma<128, 16>(ctx, dst);
        } else {
            launch_flash_attn_sparse_mma<128, 8>(ctx, dst);
        }
        return;
    }
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

    switch (dst->src[1]->type) {
        case GGML_TYPE_F32:  launch_flash_attn_sparse_vec<128, float>      (ctx, dst); break;
        case GGML_TYPE_F16:  launch_flash_attn_sparse_vec<128, half>       (ctx, dst); break;
        case GGML_TYPE_BF16: launch_flash_attn_sparse_vec<128, nv_bfloat16>(ctx, dst); break;
        default: GGML_ABORT("fatal error");
    }
}

bool ggml_cuda_flash_attn_sparse_supported(int device, const ggml_tensor * dst) {
    GGML_UNUSED(device);

    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];

    if (k->ne[0] != 128 || v->ne[0] != 128 || v->type != k->type) {
        return false;
    }
    if (k->type != GGML_TYPE_F32 && k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_BF16) {
        return false;
    }
    return q->nb[0] == sizeof(float) && k->nb[0] == ggml_type_size(k->type) && v->nb[0] == ggml_type_size(v->type);
}
