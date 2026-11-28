#include <sycl/sycl.hpp>
#include "dpct/helper.hpp"
#include "common.hpp"
#include "ggml.h"
#include "gated_delta_net.hpp"
#include <cmath>

#ifdef GGML_SYCL_DMMV_HAS_ESIMD
#    include <sycl/ext/intel/esimd.hpp>

static void launch_gated_delta_net_esimd(const float *   q,
                                         const float *   k,
                                         const float *   v,
                                         const float *   g,
                                         const float *   beta,
                                         const float *   curr_state,
                                         float *         dst,
                                         float *         state,
                                         int64_t         H,
                                         int64_t         n_tokens,
                                         int64_t         n_seqs,
                                         int64_t         sq1,
                                         int64_t         sq2,
                                         int64_t         sq3,
                                         int64_t         sk1,
                                         int64_t         sk2,
                                         int64_t         sk3,
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
    constexpr int S_v = 128;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>((size_t) n_seqs, (size_t) H, S_v), sycl::range<3>(1, 1, 16)),
        [=](sycl::nd_item<3> it) [[intel::sycl_explicit_simd]] {
            using namespace sycl::ext::intel::esimd;
            const int col      = it.get_global_id(2);
            const int h_idx    = it.get_group(1);
            const int sequence = it.get_group(0);
            const int iq1      = h_idx % (int) neqk1;
            const int iq3      = sequence / (int) rq3;

            const int64_t    head      = sequence * H + h_idx;
            simd<float, S_v> s         = block_load<float, S_v>(curr_state + head * S_v * S_v + col * S_v);
            float *          state_col = state + head * S_v * S_v + col * S_v;
            float *          attn      = dst + (sequence * n_tokens * H + h_idx) * S_v + col;

            for (int t = 0; t < n_tokens; ++t) {
                const simd<float, S_v> q_t      = block_load<float, S_v>(q + iq3 * sq3 + t * sq2 + iq1 * sq1);
                const simd<float, S_v> k_t      = block_load<float, S_v>(k + iq3 * sk3 + t * sk2 + iq1 * sk1);
                const float            v_val    = v[sequence * sv3 + t * sv2 + h_idx * sv1 + col];
                const int64_t          gb       = sequence * sb3 + t * sb2 + h_idx * sb1;
                const float            g_val    = exp(g[gb]);
                const float            beta_val = beta[gb];

                // kv = S^T k, delta = (v - g kv) beta, S = g S + k delta, attn = S^T q
                const float kv              = reduce<float>(s * k_t, std::plus<>());
                const float delta           = (v_val - g_val * kv) * beta_val;
                s                           = g_val * s + k_t * delta;
                attn[(int64_t) t * S_v * H] = reduce<float>(s * q_t, std::plus<>()) * scale;

                const int slot = (int) n_tokens - 1 - t;
                if (slot < K) {
                    block_store<float, S_v>(state_col + slot * state_slot_stride, s);
                }
            }
        });
}
#endif

// Warps per work-group. 2 is 13% faster than 4 at 4 columns per warp on Arc Pro B70.
static constexpr int gdn_warps = 2;
// State columns per warp when there are enough heads, see launch_gated_delta_net.
static constexpr int gdn_cols_wide = 4;

// replay: pending tokens to run before the new ones (see ggml_gated_delta_net_ext)
struct gdn_replay_args {
    const float *   pk;    // [S_v, H_k, n_pend, n_seqs]
    const float *   pv;    // [S_v, H, n_pend, n_seqs]
    const float *   pgb;   // [g0, H, 2*n_pend, n_seqs], gates then betas
    const int32_t * n_rep; // [n_seqs]
    int64_t         sk1, sk2, sk3; // head, token and seq strides in floats
    int64_t         sv1, sv2, sv3;
    int64_t         sg1, sg2, sg3;
    int             n_pend;
    bool            keep;  // write the state after the replay, not the final one

    // optional (not KDA): store the k, v, g, beta inputs of the new tokens as pending (fused cpys)
    float *         st[4];
    int64_t         st2[4]; // token stride in floats
    int64_t         st3[4]; // seq stride in floats
};

// C is the number of state columns per warp: a token's q/k shard is loaded once and shared by these columns.
template <int S_v, int C, bool KDA, bool keep_rs_t, bool replay>
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
                          int64_t           sk1,
                          int64_t           sk2,
                          int64_t           sk3,
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
                          int               K,
                          const gdn_replay_args rep) {
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

    if constexpr (replay) {
        // the same state update as the main loop below, without the attention output
        const int     n_r = rep.n_rep[sequence];
        const float * pk  = rep.pk  + sequence * rep.sk3 + iq1   * rep.sk1 + lane;
        const float * pv  = rep.pv  + sequence * rep.sv3 + h_idx * rep.sv1 + col0;
        const float * pgb = rep.pgb + sequence * rep.sg3 + h_idx * rep.sg1;

        for (int t = 0; t < n_r; t++) {
            const float * g_t      = pgb + t * rep.sg2;
            const float   beta_val = pgb[(rep.n_pend + t) * rep.sg2];

            float k_reg[rows_per_lane];
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                k_reg[r] = pk[t * rep.sk2 + r * warp_size];
            }

            if constexpr (!KDA) {
                const float g_val = sycl::native::exp(*g_t);
#pragma unroll
                for (int c = 0; c < C; c++) {
                    float kv_shard = 0.0f;
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        kv_shard += s_shard[c][r] * k_reg[r];
                    }
                    const float kv_col    = warp_reduce_sum<warp_size>(kv_shard);
                    const float delta_col = (pv[t * rep.sv2 + c] - g_val * kv_col) * beta_val;
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        s_shard[c][r] = g_val * s_shard[c][r] + k_reg[r] * delta_col;
                    }
                }
            } else {
                float g_reg[rows_per_lane];
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    g_reg[r] = sycl::native::exp(g_t[r * warp_size + lane]);
                }
#pragma unroll
                for (int c = 0; c < C; c++) {
                    float kv_shard = 0.0f;
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        kv_shard += g_reg[r] * s_shard[c][r] * k_reg[r];
                    }
                    const float kv_col    = warp_reduce_sum<warp_size>(kv_shard);
                    const float delta_col = (pv[t * rep.sv2 + c] - kv_col) * beta_val;
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        s_shard[c][r] = g_reg[r] * s_shard[c][r] + k_reg[r] * delta_col;
                    }
                }
            }
        }

        if (rep.keep) {
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

    // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
    // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.

    const float * qp = q + iq3 * sq3 + iq1 * sq1 + lane;
    const float * kp = k + iq3 * sk3 + iq1 * sk1 + lane;
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
            kp += sk2;
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

        if constexpr (replay && !KDA) {
            if (rep.st[0] != nullptr) {
                // keep the inputs of this token pending: one work-group per k head writes k, each warp its v columns
                const bool first_wg = item_ct1.get_group(0) == 0 && item_ct1.get_local_id(1) == 0;
                if (first_wg && h_idx < neqk1_magic.z()) {
                    float * dk = rep.st[0] + sequence * rep.st3[0] + t * rep.st2[0] + h_idx * S_v + lane;
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        dk[r * warp_size] = k_reg[r];
                    }
                }
                if (lane == 0) {
                    float * dv = rep.st[1] + sequence * rep.st3[1] + t * rep.st2[1] + h_idx * S_v + col0;
#pragma unroll
                    for (int c = 0; c < C; c++) {
                        dv[c] = v_reg[c];
                    }
                    if (first_wg) {
                        rep.st[2][sequence * rep.st3[2] + t * rep.st2[2] + h_idx] = *g_t;
                        rep.st[3][sequence * rep.st3[3] + t * rep.st2[3] + h_idx] = beta_val;
                    }
                }
            }
        }

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
        if (replay && rep.keep) {
            return;
        }
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

template <bool KDA, bool keep_rs_t, bool replay>
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
                                   int64_t         sk1,
                                   int64_t         sk2,
                                   int64_t         sk3,
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
                                   const gdn_replay_args & rep,
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
                                 gated_delta_net_sycl<sv, C, KDA, keep_rs_t, replay>(
                                     q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2, sq3, sk1, sk2, sk3, sv1, sv2,
                                     sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, rep);
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
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    // k can have its own layout (a permuted or padded view), only its rows must be contiguous
    const int64_t sk1 = nbk1 / sizeof(float);
    const int64_t sk2 = nbk2 / sizeof(float);
    const int64_t sk3 = nbk3 / sizeof(float);
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

    const ggml_tensor * src_pk = dst->src[6];
    gdn_replay_args     rep    = {};
    if (src_pk != nullptr) {
        const ggml_tensor * src_pv  = dst->src[7];
        const ggml_tensor * src_pgb = dst->src[8];
        const ggml_tensor * src_nr  = dst->src[9];

        GGML_ASSERT(K == 1);
        GGML_ASSERT(src_pv != nullptr && src_pgb != nullptr && src_nr != nullptr);
        GGML_ASSERT(src_pk->type == GGML_TYPE_F32 && src_pv->type == GGML_TYPE_F32 && src_pgb->type == GGML_TYPE_F32);
        GGML_ASSERT(src_nr->type == GGML_TYPE_I32 && ggml_is_contiguous(src_nr) && ggml_nelements(src_nr) == n_seqs);
        GGML_ASSERT(src_pk->nb[0] == sizeof(float) && src_pv->nb[0] == sizeof(float) && src_pgb->nb[0] == sizeof(float));

        rep.pk     = (const float *) src_pk->data;
        rep.pv     = (const float *) src_pv->data;
        rep.pgb    = (const float *) src_pgb->data;
        rep.n_rep  = (const int32_t *) src_nr->data;
        rep.sk1    = src_pk->nb[1] / sizeof(float);
        rep.sk2    = src_pk->nb[2] / sizeof(float);
        rep.sk3    = src_pk->nb[3] / sizeof(float);
        rep.sv1    = src_pv->nb[1] / sizeof(float);
        rep.sv2    = src_pv->nb[2] / sizeof(float);
        rep.sv3    = src_pv->nb[3] / sizeof(float);
        rep.sg1    = src_pgb->nb[1] / sizeof(float);
        rep.sg2    = src_pgb->nb[2] / sizeof(float);
        rep.sg3    = src_pgb->nb[3] / sizeof(float);
        rep.n_pend = ggml_get_op_params_i32(dst, 1);
        rep.keep   = ggml_get_op_params_i32(dst, 2) != 0;
        for (int i = 0; i < 4; ++i) {
            rep.st[i]  = cache != nullptr ? cache->in_dst[i] : nullptr;
            rep.st2[i] = cache != nullptr ? cache->in_nb2[i] : 0;
            rep.st3[i] = cache != nullptr ? cache->in_nb3[i] : 0;
        }
    }

#ifdef GGML_SYCL_DMMV_HAS_ESIMD
    if (src_pk == nullptr && !kda && S_v == 128 && g_ggml_sycl_enable_esimd && state_slot_stride % 4 == 0 && (uintptr_t) state_d % 16 == 0) {
        launch_gated_delta_net_esimd(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, n_seqs, sq1, sq2, sq3,
                                     sk1, sk2, sk3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        return;
    }
#endif

#define GDN_LAUNCH(KDA, KEEP, REPLAY)                                                                  \
    launch_gated_delta_net<KDA, KEEP, REPLAY>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, S_v, H, n_tokens, \
                                              n_seqs, sq1, sq2, sq3, sk1, sk2, sk3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3,  \
                                              scale, state_slot_stride, K, rep, stream)

    if (kda) {
        if (keep_rs) {
            GDN_LAUNCH(true, true, false);
        } else if (src_pk != nullptr) {
            GDN_LAUNCH(true, false, true);
        } else {
            GDN_LAUNCH(true, false, false);
        }
    } else {
        if (keep_rs) {
            GDN_LAUNCH(false, true, false);
        } else if (src_pk != nullptr) {
            GDN_LAUNCH(false, false, true);
        } else {
            GDN_LAUNCH(false, false, false);
        }
    }

#undef GDN_LAUNCH
}

void ggml_sycl_op_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_sycl_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/dst->src[6] != nullptr ? 10 : 6);
    ggml_sycl_op_gated_delta_net(ctx, dst);
}

void ggml_sycl_op_gated_delta_net_fused_cache(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              ggml_sycl_gated_delta_net_fused_cache cache) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/dst->src[6] != nullptr ? 10 : 6);
    ggml_sycl_op_gated_delta_net_impl(ctx, dst, &cache);
}
