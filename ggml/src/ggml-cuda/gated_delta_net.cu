#include "gated_delta_net.cuh"

constexpr int d_v_per_warp = 4;
constexpr int num_warps    = 4;
constexpr int block_dv     = num_warps * d_v_per_warp;  // 16

template <int S_v, bool KDA>
__global__ void __launch_bounds__(ggml_cuda_get_physical_warp_size() * num_warps, 2) gated_delta_net_cuda(
        const float * __restrict__ q,
        const float * __restrict__ k,
        const float * __restrict__ v,
        const float * __restrict__ g,
        const float * __restrict__ beta,
        const float * __restrict__ curr_state,
        float * __restrict__ dst,
        int64_t H,
        int64_t n_tokens,
        int64_t n_seqs,
        int64_t sq1,
        int64_t sq2,
        int64_t sq3,
        int64_t sv1,
        int64_t sv2,
        int64_t sv3,
        int64_t sb1,
        int64_t sb2,
        int64_t sb3,
        uint3   neqk1_magic,
        uint3   rq3_magic,
        float   scale) {
    constexpr int warp_size     = ggml_cuda_get_physical_warp_size();
    constexpr int active_lanes  = (S_v < warp_size) ? S_v : warp_size;
    constexpr int d_qk_per_lane = S_v / active_lanes;
    static_assert(d_qk_per_lane >= 1, "S_v must >= 1");
    static_assert(S_v % active_lanes == 0, "S_v must be a multiple of active_lanes");
    static_assert(S_v % block_dv == 0, "S_v must be a multiple of block_dv");

    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      lane     = threadIdx.x;
    const int      warp_id  = threadIdx.y;

    // each warp owns d_v_per_warp rows of the state matrix; the warp's lanes shard the QK axis
    const uint32_t dv_base  = blockIdx.z * block_dv + warp_id * d_v_per_warp;
    const uint32_t dqk_base = lane * d_qk_per_lane;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    // dst layout: [attn_data | final_state]
    const int64_t attn_score_elems = (int64_t) S_v * H * n_tokens * n_seqs;
    float *       attn_data        = dst;
    float *       state_out        = dst + attn_score_elems;

    const int64_t state_off = ((int64_t) sequence * H + h_idx) * S_v * S_v;
    state_out += state_off;
    curr_state += state_off;
    attn_data += ((int64_t) sequence * n_tokens * H + h_idx) * S_v;

    auto load_qk_lane = [&] __device__(float(&reg)[d_qk_per_lane], const float * base) {
        if constexpr (S_v < warp_size) {
            reg[0] = (lane < active_lanes) ? base[lane] : 0.0f;
        } else {
            ggml_cuda_memcpy_1<d_qk_per_lane * sizeof(float)>(reg, base + dqk_base);
        }
    };
    auto store_qk_lane = [&] __device__(const float(&reg)[d_qk_per_lane], float * base) {
        if constexpr (S_v < warp_size) {
            if (lane < active_lanes) {
                base[lane] = reg[0];
            }
        } else {
            ggml_cuda_memcpy_1<d_qk_per_lane * sizeof(float)>(base + dqk_base, reg);
        }
    };

    // state is stored transposed: M[r][c] = S[c][r]
    __align__(16) float s_tile[d_v_per_warp][d_qk_per_lane];
#pragma unroll
    for (int r = 0; r < d_v_per_warp; ++r) {
        load_qk_lane(s_tile[r], curr_state + (int64_t) (dv_base + r) * S_v);
    }

    __align__(16) float k_reg[d_qk_per_lane];
    load_qk_lane(k_reg, k + (int64_t) iq3 * sq3 + (int64_t) iq1 * sq1);

    for (int t = 0; t < n_tokens; ++t) {
        const float * v_t = v + (int64_t) sequence * sv3 + (int64_t) t * sv2 + (int64_t) h_idx * sv1;

        const int64_t gb_off   = (int64_t) sequence * sb3 + (int64_t) t * sb2 + (int64_t) h_idx * sb1;
        const float   beta_val = beta[gb_off];

        __align__(16) float alpha_lane[d_qk_per_lane];
        float               alpha_scalar = 0.0f;
        if constexpr (KDA) {
            __align__(16) float g_reg[d_qk_per_lane];
            load_qk_lane(g_reg, g + gb_off * S_v);
#pragma unroll
            for (int c = 0; c < d_qk_per_lane; ++c) {
                alpha_lane[c] = expf(g_reg[c]);
            }
        } else {
            alpha_scalar = expf(g[gb_off]);
        }

        // only first d_v_per_warp lanes hold a real v[dv_base + lane]; broadcast via __shfl_sync
        float v_local = 0.0f;
        if (lane < d_v_per_warp) {
            v_local = v_t[dv_base + lane];
        }

        // stage A: state update
#pragma unroll
        for (int r = 0; r < d_v_per_warp; ++r) {
            float partial = 0.0f;
#pragma unroll
            for (int c = 0; c < d_qk_per_lane; ++c) {
                if constexpr (KDA) {
                    partial += alpha_lane[c] * s_tile[r][c] * k_reg[c];
                } else {
                    partial += s_tile[r][c] * k_reg[c];
                }
            }
            partial = warp_reduce_sum<warp_size>(partial);

            const float v_r   = __shfl_sync(0xffffffff, v_local, r, warp_size);
            const float delta = beta_val * (v_r - (KDA ? 1.0f : alpha_scalar) * partial);

#pragma unroll
            for (int c = 0; c < d_qk_per_lane; ++c) {
                if constexpr (KDA) {
                    s_tile[r][c] = alpha_lane[c] * s_tile[r][c] + delta * k_reg[c];
                } else {
                    s_tile[r][c] = alpha_scalar * s_tile[r][c] + delta * k_reg[c];
                }
            }
        }

        // prefetch k for next token while issuing the q load for this token
        if (t + 1 < n_tokens) {
            load_qk_lane(k_reg, k + (int64_t) iq3 * sq3 + (int64_t) (t + 1) * sq2 + (int64_t) iq1 * sq1);
        }

        __align__(16) float q_reg[d_qk_per_lane];
        load_qk_lane(q_reg, q + (int64_t) iq3 * sq3 + (int64_t) t * sq2 + (int64_t) iq1 * sq1);

        // stage B: attention output
        float attn_val = 0.0f;
#pragma unroll
        for (int r = 0; r < d_v_per_warp; ++r) {
            float partial = 0.0f;
#pragma unroll
            for (int c = 0; c < d_qk_per_lane; ++c) {
                partial += s_tile[r][c] * q_reg[c];
            }
            partial = warp_reduce_sum<warp_size>(partial);
            if (lane == r) {
                attn_val = partial;
            }
        }

        if (lane < d_v_per_warp) {
            float * attn_t         = attn_data + (int64_t) t * S_v * H;
            attn_t[dv_base + lane] = attn_val * scale;
        }
    }

    // store final state
#pragma unroll
    for (int r = 0; r < d_v_per_warp; ++r) {
        store_qk_lane(s_tile[r], state_out + (int64_t) (dv_base + r) * S_v);
    }
}

template <bool KDA>
static void launch_gated_delta_net(const float * q_d,
                                   const float * k_d,
                                   const float * v_d,
                                   const float * g_d,
                                   const float * b_d,
                                   const float * s_d,
                                   float *       dst_d,
                                   int64_t       S_v,
                                   int64_t       H,
                                   int64_t       n_tokens,
                                   int64_t       n_seqs,
                                   int64_t       sq1,
                                   int64_t       sq2,
                                   int64_t       sq3,
                                   int64_t       sv1,
                                   int64_t       sv2,
                                   int64_t       sv3,
                                   int64_t       sb1,
                                   int64_t       sb2,
                                   int64_t       sb3,
                                   int64_t       neqk1,
                                   int64_t       rq3,
                                   float         scale,
                                   cudaStream_t  stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int   warp_size   = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int   n_block_dv  = (int) ((S_v + block_dv - 1) / block_dv);
    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    dim3 grid_dims((unsigned) H, (unsigned) n_seqs, (unsigned) n_block_dv);
    dim3 block_dims(warp_size, num_warps, 1);

    switch (S_v) {
        case 16:
            gated_delta_net_cuda<16, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
                neqk1_magic, rq3_magic, scale);
            break;
        case 32:
            gated_delta_net_cuda<32, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
                neqk1_magic, rq3_magic, scale);
            break;
        case 64:
            gated_delta_net_cuda<64, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
                neqk1_magic, rq3_magic, scale);
            break;
        case 128:
            gated_delta_net_cuda<128, KDA><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
                neqk1_magic, rq3_magic, scale);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t, nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t, nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t, nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    GGML_ASSERT(nbq1 % 16 == 0);
    GGML_ASSERT(nbq2 % 16 == 0);
    GGML_ASSERT(nbq3 % 16 == 0);

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    if (kda) {
        launch_gated_delta_net<true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1,
                                     sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, stream);
    } else {
        launch_gated_delta_net<false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1,
                                      sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, stream);
    }
}
