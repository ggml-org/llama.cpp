// Gated Delta Net (GDN) fused kernel for SYCL backend
// Ported from Vulkan compute shader: gated_delta_net.comp
//
// Supports both GDA (scalar gate) and KDA (key-dependent, vector gate) variants.
// Parallelization: one workgroup per (head, seq), S_V threads per workgroup.
// Each thread manages one column of the S_V x S_V state matrix.

#include <sycl/sycl.hpp>

#include "common.hpp"

template <uint32_t S_V, bool KDA>
static void gated_delta_net_f32_kernel(
        const dpct::queue_ptr stream,
        uint32_t H, uint32_t n_tokens, uint32_t n_seqs, uint32_t s_off,
        uint32_t sq1, uint32_t sq2, uint32_t sq3,
        uint32_t sv1, uint32_t sv2, uint32_t sv3,
        uint32_t sb1, uint32_t sb2, uint32_t sb3,
        uint32_t neq1, uint32_t rq3,
        float scale,
        const float * q, const float * k, const float * v,
        const float * g, const float * beta, const float * state,
        float * dst) {

    sycl::range<2> block_dims(1, S_V);
    sycl::range<2> grid_dims(n_seqs, H);

    stream->submit([&](sycl::handler & cgh) {
        auto s_k = sycl::local_accessor<float, 1>(sycl::range<1>(S_V), cgh);
        auto s_q = sycl::local_accessor<float, 1>(sycl::range<1>(S_V), cgh);
        auto s_g = sycl::local_accessor<float, 1>(sycl::range<1>(KDA ? S_V : 1), cgh);

        cgh.parallel_for(sycl::nd_range<2>(grid_dims * block_dims, block_dims),
            [=](sycl::nd_item<2> item) {
            const uint32_t head_id = item.get_group(1);
            const uint32_t seq_id  = item.get_group(0);
            const uint32_t col     = item.get_local_id(1);

            const uint32_t iq1 = head_id % neq1;
            const uint32_t iq3 = seq_id / rq3;

            const uint32_t state_size = S_V * S_V;
            const uint32_t state_base = (seq_id * H + head_id) * state_size;

            // Load state column into registers
            float st[S_V];
#pragma unroll
            for (uint32_t i = 0; i < S_V; i++) {
                st[i] = state[state_base + col * S_V + i];
            }

            uint32_t attn_off = (seq_id * n_tokens * H + head_id) * S_V;

            for (uint32_t t = 0; t < n_tokens; t++) {
                const uint32_t q_off  = iq3 * sq3 + t * sq2 + iq1 * sq1;
                const uint32_t k_off  = q_off;
                const uint32_t v_off  = seq_id * sv3 + t * sv2 + head_id * sv1;
                const uint32_t gb_off = seq_id * sb3 + t * sb2 + head_id * sb1;

                // Load q, k into shared memory
                s_q[col] = q[q_off + col];
                s_k[col] = k[k_off + col];

                if constexpr (KDA) {
                    const uint32_t g_base = gb_off * S_V;
                    s_g[col] = sycl::exp(g[g_base + col]);
                }

                item.barrier(sycl::access::fence_space::local_space);

                const float v_val    = v[v_off + col];
                const float beta_val = beta[gb_off];

                if constexpr (!KDA) {
                    // GDA: scalar gate
                    const float g_val = sycl::exp(g[gb_off]);

                    // kv_col = dot(state_column, k)
                    float kv_col = 0.0f;
#pragma unroll
                    for (uint32_t i = 0; i < S_V; i += 4) {
                        const sycl::float4 & sv = (const sycl::float4 &)(st[i]);
                        const sycl::float4 & kv = (const sycl::float4 &)(s_k[i]);
                        kv_col += sv.x()*kv.x() + sv.y()*kv.y() + sv.z()*kv.z() + sv.w()*kv.w();
                    }

                    // delta = (v - g * kv_col) * beta
                    const float delta_col = (v_val - g_val * kv_col) * beta_val;

                    // Update state and compute attention output
                    float attn_col = 0.0f;
#pragma unroll
                    for (uint32_t i = 0; i < S_V; i += 4) {
                        sycl::float4 & sv = (sycl::float4 &)(st[i]);
                        const sycl::float4 & kv = (const sycl::float4 &)(s_k[i]);
                        const sycl::float4 & qv = (const sycl::float4 &)(s_q[i]);

                        // state = g * state + k * delta
                        sv = g_val * sv + kv * delta_col;

                        attn_col += sv.x()*qv.x() + sv.y()*qv.y() + sv.z()*qv.z() + sv.w()*qv.w();
                    }

                    dst[attn_off + col] = attn_col * scale;
                } else {
                    // KDA: vector gate (key-dependent attention)

                    // kv_col = dot(exp(g) * state_column, k)
                    float kv_col = 0.0f;
#pragma unroll
                    for (uint32_t i = 0; i < S_V; i += 4) {
                        const sycl::float4 & gv = (const sycl::float4 &)(s_g[i]);
                        const sycl::float4 & sv = (const sycl::float4 &)(st[i]);
                        const sycl::float4 & kv = (const sycl::float4 &)(s_k[i]);
                        kv_col += gv.x()*sv.x()*kv.x() + gv.y()*sv.y()*kv.y()
                               +  gv.z()*sv.z()*kv.z() + gv.w()*sv.w()*kv.w();
                    }

                    // delta = (v - kv_col) * beta
                    const float delta_col = (v_val - kv_col) * beta_val;

                    // Update state and compute attention output
                    float attn_col = 0.0f;
#pragma unroll
                    for (uint32_t i = 0; i < S_V; i += 4) {
                        const sycl::float4 & gv = (const sycl::float4 &)(s_g[i]);
                        sycl::float4 & sv       = (sycl::float4 &)(st[i]);
                        const sycl::float4 & kv = (const sycl::float4 &)(s_k[i]);
                        const sycl::float4 & qv = (const sycl::float4 &)(s_q[i]);

                        // state = exp(g) * state + k * delta
                        sv = gv * sv + kv * delta_col;

                        attn_col += sv.x()*qv.x() + sv.y()*qv.y() + sv.z()*qv.z() + sv.w()*qv.w();
                    }

                    dst[attn_off + col] = attn_col * scale;
                }

                attn_off += S_V * H;
                item.barrier(sycl::access::fence_space::local_space);
            }

            // Write updated state back
#pragma unroll
            for (uint32_t i = 0; i < S_V; i++) {
                dst[s_off + state_base + col * S_V + i] = st[i];
            }
        });
    });
}

template <uint32_t S_V>
static void dispatch_gdn_kda(
        const dpct::queue_ptr stream, bool kda,
        uint32_t H, uint32_t n_tokens, uint32_t n_seqs, uint32_t s_off,
        uint32_t sq1, uint32_t sq2, uint32_t sq3,
        uint32_t sv1, uint32_t sv2, uint32_t sv3,
        uint32_t sb1, uint32_t sb2, uint32_t sb3,
        uint32_t neq1, uint32_t rq3,
        float scale,
        const float * q, const float * k, const float * v,
        const float * g, const float * beta, const float * state,
        float * dst) {
    if (kda) {
        gated_delta_net_f32_kernel<S_V, true>(stream, H, n_tokens, n_seqs, s_off,
            sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neq1, rq3, scale,
            q, k, v, g, beta, state, dst);
    } else {
        gated_delta_net_f32_kernel<S_V, false>(stream, H, n_tokens, n_seqs, s_off,
            sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neq1, rq3, scale,
            q, k, v, g, beta, state, dst);
    }
}

void ggml_sycl_op_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);

    const ggml_tensor * src_q     = dst->src[0];
    const ggml_tensor * src_k     = dst->src[1];
    const ggml_tensor * src_v     = dst->src[2];
    const ggml_tensor * src_g     = dst->src[3];
    const ggml_tensor * src_beta  = dst->src[4];
    const ggml_tensor * src_state = dst->src[5];

    const float * q_d     = static_cast<const float *>(src_q->data);
    const float * k_d     = static_cast<const float *>(src_k->data);
    const float * v_d     = static_cast<const float *>(src_v->data);
    const float * g_d     = static_cast<const float *>(src_g->data);
    const float * beta_d  = static_cast<const float *>(src_beta->data);
    const float * state_d = static_cast<const float *>(src_state->data);
    float * dst_d         = static_cast<float *>(dst->data);

    GGML_ASSERT(src_q->type     == GGML_TYPE_F32);
    GGML_ASSERT(src_k->type     == GGML_TYPE_F32);
    GGML_ASSERT(src_v->type     == GGML_TYPE_F32);
    GGML_ASSERT(src_g->type     == GGML_TYPE_F32);
    GGML_ASSERT(src_beta->type  == GGML_TYPE_F32);
    GGML_ASSERT(src_state->type == GGML_TYPE_F32);

    const uint32_t S_v      = (uint32_t)src_v->ne[0];
    const uint32_t H        = (uint32_t)src_v->ne[1];
    const uint32_t n_tokens = (uint32_t)src_v->ne[2];
    const uint32_t n_seqs   = (uint32_t)src_v->ne[3];

    // Strides for q (in float elements)
    const uint32_t sq1 = (uint32_t)(src_q->nb[1] / sizeof(float));
    const uint32_t sq2 = (uint32_t)(src_q->nb[2] / sizeof(float));
    const uint32_t sq3 = (uint32_t)(src_q->nb[3] / sizeof(float));

    // Strides for v
    const uint32_t sv1 = (uint32_t)(src_v->nb[1] / sizeof(float));
    const uint32_t sv2 = (uint32_t)(src_v->nb[2] / sizeof(float));
    const uint32_t sv3 = (uint32_t)(src_v->nb[3] / sizeof(float));

    // Strides for beta (and gate dimension mapping)
    const uint32_t sb1 = (uint32_t)(src_beta->nb[1] / sizeof(float));
    const uint32_t sb2 = (uint32_t)(src_beta->nb[2] / sizeof(float));
    const uint32_t sb3 = (uint32_t)(src_beta->nb[3] / sizeof(float));

    // GQA/MQA support
    const uint32_t neq1 = (uint32_t)src_q->ne[1];
    const uint32_t rq3  = (uint32_t)(src_v->ne[3] / src_q->ne[3]);

    // Output offset where state begins
    const uint32_t s_off = (uint32_t)(src_v->ne[0] * src_v->ne[1] * src_v->ne[2] * src_v->ne[3]);

    const float scale = 1.0f / sqrtf((float)S_v);

    // KDA = key-dependent attention (vector gate)
    const bool kda = (src_g->ne[0] == (int64_t)S_v);

    dpct::queue_ptr stream = ctx.stream();

    switch (S_v) {
        case 32:
            dispatch_gdn_kda<32>(stream, kda, H, n_tokens, n_seqs, s_off,
                sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neq1, rq3, scale,
                q_d, k_d, v_d, g_d, beta_d, state_d, dst_d);
            break;
        case 64:
            dispatch_gdn_kda<64>(stream, kda, H, n_tokens, n_seqs, s_off,
                sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neq1, rq3, scale,
                q_d, k_d, v_d, g_d, beta_d, state_d, dst_d);
            break;
        case 128:
            dispatch_gdn_kda<128>(stream, kda, H, n_tokens, n_seqs, s_off,
                sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neq1, rq3, scale,
                q_d, k_d, v_d, g_d, beta_d, state_d, dst_d);
            break;
        default:
            GGML_ABORT("Unsupported head size %u for GDN", S_v);
    }
}
