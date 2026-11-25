#include <sycl/sycl.hpp>
#include "dpct/helper.hpp"
#include "common.hpp"
#include "ggml.h"
#include "gated_delta_net.hpp"
#include <cmath>


// Warps per work-group. 2 is 13% faster than 4 at 4 columns per warp on Arc Pro B70.
static constexpr int gdn_warps = 2;
// State columns per warp when there are enough heads, see launch_gated_delta_net.
static constexpr int gdn_cols_wide = 4;

// C is the number of state columns per warp: a token's q/k shard is loaded once and shared by these columns.
template <int S_v, int C, bool KDA, bool keep_rs_t>
void gated_delta_net_sycl(const float *     q,
                          const float *     k,
                          const float *     v,
                          const float *     g,
                          const float *     beta,
                          const float *     curr_state,
                          float *           dst,
                          float *           state,
                          int64_t           H,
                          int64_t           n_tokens,
                          int64_t           sq1,
                          int64_t           sq2,
                          int64_t           sq3,
                          int64_t           sv1,
                          int64_t           sv2,
                          int64_t           sv3,
                          int64_t           sb1,
                          int64_t           sb2,
                          int64_t           sb3,
                          const sycl::uint3 neqk1_magic,
                          const sycl::uint3 rq3_magic,
                          float             scale,
                          int64_t           state_slot_stride,
                          int               K) {
    auto           item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const uint32_t h_idx    = item_ct1.get_group(2);
    const uint32_t sequence = item_ct1.get_group(1);
    // each warp owns C columns, using warp-level primitives to reduce across rows
    const int      lane     = item_ct1.get_local_id(2);
    const int      warp     = item_ct1.get_group(0) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);
    const int      col0     = warp * C;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_sycl_get_physical_warp_size() < S_v ? ggml_sycl_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    static_assert(S_v % (C * gdn_warps) == 0, "S_v must be a multiple of the columns per work-group");
    static_assert(C <= warp_size, "the attn store puts column c in lane c");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[C][rows_per_lane];
#pragma unroll
    for (int c = 0; c < C; c++) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i   = r * warp_size + lane;
            s_shard[c][r] = curr_state[(col0 + c) * S_v + i];
        }
    }

    // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
    // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.

    const float * qp = q + iq3 * sq3 + iq1 * sq1 + lane;
    const float * kp = k + iq3 * sq3 + iq1 * sq1 + lane;
    const float * vp = v + sequence * sv3 + h_idx * sv1 + col0;
    const float * bp = beta + sequence * sb3 + h_idx * sb1;
    const float * gp = g + (sequence * sb3 + h_idx * sb1) * (KDA ? S_v : 1);

    // q/k of the next token are loaded one iteration ahead to hide the load latency.
    float q_next[rows_per_lane];
    float k_next[rows_per_lane];
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        q_next[r] = qp[r * warp_size];
        k_next[r] = kp[r * warp_size];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float   beta_val = *bp;
        const float * g_t      = gp;
        bp += sb2;
        gp += sb2 * (KDA ? S_v : 1);

        float q_reg[rows_per_lane];
        float k_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            q_reg[r] = q_next[r];
            k_reg[r] = k_next[r];
        }
        if (t + 1 < n_tokens) {
            qp += sq2;
            kp += sq2;
        }
        // Load outside the branch: the last token reads its own row again.
        // A guarded load is 20% slower at 32 heads on Arc Pro B70.
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            q_next[r] = qp[r * warp_size];
            k_next[r] = kp[r * warp_size];
        }

        float v_reg[C];
#pragma unroll
        for (int c = 0; c < C; c++) {
            v_reg[c] = vp[c];
        }
        vp += sv2;

        // attn[col] = g * (S^T q)[col] + delta[col] * (k . q), so no reduction waits for delta
        float kq_shard = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            kq_shard += k_reg[r] * q_reg[r];
        }
        const float kq = warp_reduce_sum<warp_size>(kq_shard);

        float out_lane = 0.0f;
        if constexpr (!KDA) {
            const float g_val = sycl::native::exp(*g_t);

#pragma unroll
            for (int c = 0; c < C; c++) {
                // x: kv[col] = (S^T k)[col], y: (S^T q)[col]
                sycl::float2 acc(0.0f, 0.0f);
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    acc.x() += s_shard[c][r] * k_reg[r];
                    acc.y() += s_shard[c][r] * q_reg[r];
                }
                acc = warp_reduce_sum<warp_size>(acc);

                // delta[col] = (v[col] - g * kv[col]) * beta
                const float delta_col = (v_reg[c] - g_val * acc.x()) * beta_val;

                // S[i][col] = g * S[i][col] + k[i] * delta[col]
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    s_shard[c][r] = g_val * s_shard[c][r] + k_reg[r] * delta_col;
                }

                // the reductions broadcast, so lane c already holds column c's result
                out_lane = (lane == c) ? (g_val * acc.y() + delta_col * kq) * scale : out_lane;
            }
        } else {
            float g_reg[rows_per_lane];
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                g_reg[r] = sycl::native::exp(g_t[r * warp_size + lane]);
            }

#pragma unroll
            for (int c = 0; c < C; c++) {
                // x: kv[col] = sum_i g[i] * S[i][col] * k[i], y: sum_i g[i] * S[i][col] * q[i]
                sycl::float2 acc(0.0f, 0.0f);
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const float gs = g_reg[r] * s_shard[c][r];
                    acc.x() += gs * k_reg[r];
                    acc.y() += gs * q_reg[r];
                }
                acc = warp_reduce_sum<warp_size>(acc);

                // delta[col] = (v[col] - kv[col]) * beta
                const float delta_col = (v_reg[c] - acc.x()) * beta_val;

                // S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    s_shard[c][r] = g_reg[r] * s_shard[c][r] + k_reg[r] * delta_col;
                }

                out_lane = (lane == c) ? (acc.y() + delta_col * kq) * scale : out_lane;
            }
        }

        if (lane < C) {
            attn_data[col0 + lane] = out_lane;
        }

        attn_data += S_v * H;


    // Write state back to global memory
        if constexpr (keep_rs_t) {
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * slot_col = state + target_slot * state_slot_stride + col0 * S_v + lane;
#pragma unroll
                for (int c = 0; c < C; c++) {
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        slot_col[c * S_v + r * warp_size] = s_shard[c][r];
                    }
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
        float * out_col = state + col0 * S_v + lane;
#pragma unroll
        for (int c = 0; c < C; c++) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                out_col[c * S_v + r * warp_size] = s_shard[c][r];
            }
        }
    }
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(const float *   q_d,
                                   const float *   k_d,
                                   const float *   v_d,
                                   const float *   g_d,
                                   const float *   b_d,
                                   const float *   s_d,
                                   float *         dst_d,
                                   float *         state_d,
                                   int64_t         S_v,
                                   int64_t         H,
                                   int64_t         n_tokens,
                                   int64_t         n_seqs,
                                   int64_t         sq1,
                                   int64_t         sq2,
                                   int64_t         sq3,
                                   int64_t         sv1,
                                   int64_t         sv2,
                                   int64_t         sv3,
                                   int64_t         sb1,
                                   int64_t         sb2,
                                   int64_t         sb3,
                                   int64_t         neqk1,
                                   int64_t         rq3,
                                   float           scale,
                                   int64_t         state_slot_stride,
                                   int             K,
                                   dpct::queue_ptr stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_sycl_info().devices[ggml_sycl_get_device()].warp_size;

    const sycl::uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const sycl::uint3 rq3_magic   = init_fastdiv_values(rq3);

    auto launch = [&](auto sv_c, auto c_c) {
        constexpr int sv = decltype(sv_c)::value;
        constexpr int C  = decltype(c_c)::value;
        dpct::dim3 grid_dims(H, n_seqs, sv / (C * gdn_warps));
        dpct::dim3 block_dims(warp_size <= sv ? warp_size : sv, gdn_warps, 1);
        stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                             [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 gated_delta_net_sycl<sv, C, KDA, keep_rs_t>(
                                     q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2, sq3, sv1, sv2,
                                     sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
                             });
    };
    // gdn_cols_wide columns per warp cut the q/k loads but launch fewer warps, so with few heads 1 column is faster.
    // The 1024-warp switch point is measured on Arc Pro B70.
    auto launch_sv = [&](auto sv_c) {
        if (H * n_seqs * S_v / gdn_cols_wide >= 1024) {
            launch(sv_c, std::integral_constant<int, gdn_cols_wide>{});
        } else {
            launch(sv_c, std::integral_constant<int, 1>{});
        }
    };

    switch (S_v) {
        case 16:  launch_sv(std::integral_constant<int, 16>{});  break;
        case 32:  launch_sv(std::integral_constant<int, 32>{});  break;
        case 64:  launch_sv(std::integral_constant<int, 64>{});  break;
        case 128: launch_sv(std::integral_constant<int, 128>{}); break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void ggml_sycl_op_gated_delta_net_impl(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              const ggml_sycl_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

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

    dpct::queue_ptr stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> dst tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    }
}

void ggml_sycl_op_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_sycl_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    ggml_sycl_op_gated_delta_net(ctx, dst);
}

void ggml_sycl_op_gated_delta_net_fused_cache(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              ggml_sycl_gated_delta_net_fused_cache cache) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    ggml_sycl_op_gated_delta_net_impl(ctx, dst, &cache);
}
