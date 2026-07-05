//
// XMX (DPAS) flash-attention — first ggml-sycl integration cut. SCAFFOLD.
//
// Validated standalone (docs/research/xmx_fa_*.cpp): fused online-softmax FA with
// QK^T and PV on DPAS at sub-group 8, f16 in / f32 accumulate. This wires that into
// the ggml-sycl fattn dispatch for the f16-KV, D=128, causal (no explicit mask),
// GQA-packed decode case. OFF unless GGML_SYCL_FA_XMX is set. Correctness on real
// ggml tensors is pending the CPU-vs-SYCL harness; q8_0/turbo dequant-into-tile,
// explicit masks, ALiBi, D in {256,512}, and prefill are follow-ups.
//

#include "fattn-xmx.hpp"
#include "fattn-common.hpp"
#include <sycl/sycl.hpp>

namespace sycl_matrix = sycl::ext::oneapi::experimental::matrix;

// DPAS tile + the mandatory DG2 sub-group (8, NOT the backend WARP_SIZE 16).
static constexpr int XMX_TM = 8, XMX_TN = 8, XMX_TK = 16, XMX_SG = 8;

// Per-operand KV kind (template param). K and V must share a kind (see entry).
enum xmx_kv_kind : int { XMX_F16 = 0, XMX_Q8_0 = 1, XMX_TURBO2 = 2, XMX_TURBO3 = 3, XMX_TURBO4 = 4 };

// Dequant one KV element (block base already offset to (kv0+j)'s row start; d in 0..D-1).
template <int KIND>
static inline sycl::half xmx_load_kv(const char * row, int d) {
    if constexpr (KIND == XMX_Q8_0) {
        const block_q8_0 * b = (const block_q8_0 *) row + d / QK8_0;
        return sycl::half((float) b->d * (float) b->qs[d % QK8_0]);
    } else if constexpr (KIND == XMX_TURBO2) {
        const block_turbo2_0 * b = (const block_turbo2_0 *) row + d / QK_TURBO2;
        return sycl::half(dequantize_turbo2_0(b, d % QK_TURBO2, (float) b->norm));
    } else if constexpr (KIND == XMX_TURBO3) {
        const block_turbo3_0 * b = (const block_turbo3_0 *) row + d / QK_TURBO3;
        return sycl::half(dequantize_turbo3_0(b, d % QK_TURBO3, (float) b->norm));
    } else if constexpr (KIND == XMX_TURBO4) {
        const block_turbo4_0 * b = (const block_turbo4_0 *) row + d / QK_TURBO4;
        return sycl::half(dequantize_turbo4_0(b, d % QK_TURBO4, (float) b->norm));
    } else {  // XMX_F16
        return ((const sycl::half *) row)[d];
    }
}

template <int D, int Br, int Bc, int KK, int VK>
static void flash_attn_ext_xmx_kernel(
        const char * __restrict__ Q, const char * __restrict__ K, const char * __restrict__ V,
        float * __restrict__ dst,
        const float scale,
        const int   n_kv,
        const int   gqa_ratio,
        const int64_t nb01, const int64_t nb02,          // Q row / head strides (bytes)
        const int64_t nb11, const int64_t nb12,          // K row / head strides (bytes)
        const int64_t nb21, const int64_t nb22,          // V row / head strides (bytes)
        const int     ne1,                               // n_q (query count for this head)
        const char * __restrict__ maskb, const int64_t mask_nb1,  // additive mask (f16) or null
        const sycl::nd_item<3> & it,
        sycl::half * Qh, sycl::half * Kt_sh, sycl::half * V_sh,
        float * S_sh, sycl::half * P_sh, float * PV_sh, float * O_sh, float * m_sh, float * l_sh) {

    auto sg = it.get_sub_group();
    const int lane   = it.get_local_id(2);                 // 0..XMX_SG-1; lane i owns query row i
    const int head   = it.get_group(1);                    // query head
    const int q0     = it.get_group(2) * Br;               // first query row of this block
    const int kv_head = head / gqa_ratio;
    const int qr = q0 + lane;                              // this lane's query row
    const sycl::half * maskh = maskb ? (const sycl::half *)(maskb + (size_t) qr * mask_nb1) : nullptr;

    const char * Qh_base = Q + head * nb02;                // this head's Q
    const char * K_base  = K + kv_head * nb12;
    const char * V_base  = V + kv_head * nb22;

    // Convert this block's Q (f32) -> f16 into SLM [Br,D]. Rows past ne1 -> 0.
    for (int idx = lane; idx < Br * D; idx += XMX_SG) {
        const int r = idx / D, d = idx % D;
        const int qr = q0 + r;
        float v = 0.0f;
        if (qr < ne1) v = ((const float *) (Qh_base + qr * nb01))[d];
        Qh[r * D + d] = sycl::half(v);
    }
    for (int d = 0; d < D; ++d) O_sh[lane * D + d] = 0.0f;
    m_sh[lane] = -1e30f; l_sh[lane] = 0.0f;
    sycl::group_barrier(sg);

    for (int kv0 = 0; kv0 < n_kv; kv0 += Bc) {
        // stage K-block transposed [D,Bc] and V-block [Bc,D] into SLM as f16,
        // dequantizing q8_0/turbo blocks on the fly (rotated domain for turbo).
        for (int idx = lane; idx < Bc * D; idx += XMX_SG) {
            const int j = idx / D, d = idx % D;
            Kt_sh[d * Bc + j] = xmx_load_kv<KK>(K_base + (kv0 + j) * nb11, d);
            V_sh [j * D + d]  = xmx_load_kv<VK>(V_base + (kv0 + j) * nb21, d);
        }
        sycl::group_barrier(sg);

        // S[Br,Bc] = Qh . Kt_sh
        for (int nt = 0; nt < Bc / XMX_TN; ++nt) {
            sycl_matrix::joint_matrix<sycl::sub_group, sycl::half, sycl_matrix::use::a, XMX_TM, XMX_TK, sycl_matrix::layout::row_major> a;
            sycl_matrix::joint_matrix<sycl::sub_group, sycl::half, sycl_matrix::use::b, XMX_TK, XMX_TN, sycl_matrix::layout::row_major> b;
            sycl_matrix::joint_matrix<sycl::sub_group, float, sycl_matrix::use::accumulator, XMX_TM, XMX_TN> c;
            sycl_matrix::joint_matrix_fill(sg, c, 0.0f);
            for (int k = 0; k < D; k += XMX_TK) {
                sycl_matrix::joint_matrix_load(sg, a, sycl::multi_ptr<const sycl::half, sycl::access::address_space::local_space>(Qh + k), D);
                sycl_matrix::joint_matrix_load(sg, b, sycl::multi_ptr<const sycl::half, sycl::access::address_space::local_space>(Kt_sh + k * Bc + nt * XMX_TN), Bc);
                sycl_matrix::joint_matrix_mad(sg, c, a, b, c);
            }
            sycl_matrix::joint_matrix_store(sg, c, sycl::multi_ptr<float, sycl::access::address_space::local_space>(S_sh + nt * XMX_TN), Bc, sycl_matrix::layout::row_major);
        }
        sycl::group_barrier(sg);

        // online softmax (lane owns row). Fold in scale + additive mask, store the
        // masked/scaled score back so the exp pass reuses it. (slope=1: ALiBi TODO.)
        float rmax = -1e30f;
        for (int j = 0; j < Bc; ++j) {
            float sj = S_sh[lane * Bc + j] * scale;
            if (maskh) sj += (float) maskh[kv0 + j];
            S_sh[lane * Bc + j] = sj;
            rmax = sycl::fmax(rmax, sj);
        }
        const float m_new = sycl::fmax(m_sh[lane], rmax);
        const float corr  = sycl::native::exp(m_sh[lane] - m_new);
        float psum = 0.0f;
        for (int j = 0; j < Bc; ++j) { const float p = sycl::native::exp(S_sh[lane * Bc + j] - m_new); P_sh[lane * Bc + j] = sycl::half(p); psum += p; }
        l_sh[lane] = l_sh[lane] * corr + psum;
        for (int d = 0; d < D; ++d) O_sh[lane * D + d] *= corr;
        m_sh[lane] = m_new;
        sycl::group_barrier(sg);

        // PV[Br,D] = P . V_sh
        for (int nt = 0; nt < D / XMX_TN; ++nt) {
            sycl_matrix::joint_matrix<sycl::sub_group, sycl::half, sycl_matrix::use::a, XMX_TM, XMX_TK, sycl_matrix::layout::row_major> a;
            sycl_matrix::joint_matrix<sycl::sub_group, sycl::half, sycl_matrix::use::b, XMX_TK, XMX_TN, sycl_matrix::layout::row_major> b;
            sycl_matrix::joint_matrix<sycl::sub_group, float, sycl_matrix::use::accumulator, XMX_TM, XMX_TN> c;
            sycl_matrix::joint_matrix_fill(sg, c, 0.0f);
            for (int k = 0; k < Bc; k += XMX_TK) {
                sycl_matrix::joint_matrix_load(sg, a, sycl::multi_ptr<const sycl::half, sycl::access::address_space::local_space>(P_sh + k), Bc);
                sycl_matrix::joint_matrix_load(sg, b, sycl::multi_ptr<const sycl::half, sycl::access::address_space::local_space>(V_sh + k * D + nt * XMX_TN), D);
                sycl_matrix::joint_matrix_mad(sg, c, a, b, c);
            }
            sycl_matrix::joint_matrix_store(sg, c, sycl::multi_ptr<float, sycl::access::address_space::local_space>(PV_sh + nt * XMX_TN), D, sycl_matrix::layout::row_major);
        }
        sycl::group_barrier(sg);
        for (int d = 0; d < D; ++d) O_sh[lane * D + d] += PV_sh[lane * D + d];
        sycl::group_barrier(sg);
    }

    // write output: ggml FA dst is [D, n_head, n_q(*seq)] contiguous.
    if (qr < ne1) {
        float * o = dst + (int64_t)(qr) * (D * it.get_group_range(1)) + head * D;
        for (int d = 0; d < D; ++d) o[d] = O_sh[lane * D + d] / l_sh[lane];
    }
}

template <int D, int Br, int Bc, int KK, int VK>
static void submit_xmx(queue_ptr stream,
        const char * Qd, const char * Kd, const char * Vd, float * Od,
        float scale, int n_kv, int gqa_ratio,
        int64_t nb01, int64_t nb02, int64_t nb11, int64_t nb12, int64_t nb21, int64_t nb22,
        int ne1, int n_head, int q_blocks, const char * maskb, int64_t mask_nb1) {
    sycl::range<3> global(1, n_head, q_blocks * XMX_SG);
    sycl::range<3> local (1, 1, XMX_SG);
    stream->submit([&](sycl::handler & h) {
        sycl::local_accessor<sycl::half, 1> Qh   (Br * D, h);
        sycl::local_accessor<sycl::half, 1> Kt_sh(D * Bc, h);
        sycl::local_accessor<sycl::half, 1> V_sh (Bc * D, h);
        sycl::local_accessor<float,    1>  S_sh (Br * Bc, h);
        sycl::local_accessor<sycl::half, 1> P_sh (Br * Bc, h);
        sycl::local_accessor<float,    1>  PV_sh(Br * D,  h);
        sycl::local_accessor<float,    1>  O_sh (Br * D,  h);
        sycl::local_accessor<float,    1>  m_sh (Br, h);
        sycl::local_accessor<float,    1>  l_sh (Br, h);
        h.parallel_for(sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(XMX_SG)]] {
            flash_attn_ext_xmx_kernel<D, Br, Bc, KK, VK>(
                Qd, Kd, Vd, Od, scale, n_kv, gqa_ratio,
                nb01, nb02, nb11, nb12, nb21, nb22, ne1, maskb, mask_nb1, it,
                Qh.get_multi_ptr<sycl::access::decorated::no>().get(),
                Kt_sh.get_multi_ptr<sycl::access::decorated::no>().get(),
                V_sh.get_multi_ptr<sycl::access::decorated::no>().get(),
                S_sh.get_multi_ptr<sycl::access::decorated::no>().get(),
                P_sh.get_multi_ptr<sycl::access::decorated::no>().get(),
                PV_sh.get_multi_ptr<sycl::access::decorated::no>().get(),
                O_sh.get_multi_ptr<sycl::access::decorated::no>().get(),
                m_sh.get_multi_ptr<sycl::access::decorated::no>().get(),
                l_sh.get_multi_ptr<sycl::access::decorated::no>().get());
        });
    });
}

void ggml_sycl_flash_attn_ext_xmx(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    auto kv_kind = [](ggml_type t) -> int {
        switch (t) {
            case GGML_TYPE_F16:      return XMX_F16;
            case GGML_TYPE_Q8_0:     return XMX_Q8_0;
            case GGML_TYPE_TURBO2_0: return XMX_TURBO2;
            case GGML_TYPE_TURBO3_0: return XMX_TURBO3;
            case GGML_TYPE_TURBO4_0: return XMX_TURBO4;
            default:                 return -1;
        }
    };
    const int kk = kv_kind(K->type);
    const int vk = kv_kind(V->type);
    GGML_ASSERT(kk >= 0 && vk >= 0 && "XMX FA: unsupported KV type");
    GGML_ASSERT(kk == vk && "XMX FA: mixed K/V types not supported");
    GGML_ASSERT(Q->ne[0] == 128);
    if (mask) {
        GGML_ASSERT(mask->type == GGML_TYPE_F16 && "XMX FA: mask must be f16");
        GGML_ASSERT(mask->ne[2] == 1 && "XMX FA: per-head mask not yet supported");
    }

    constexpr int D = 128, Br = 8, Bc = 16;
    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const int n_kv      = K->ne[1];
    const int n_head    = Q->ne[2];
    const int n_head_kv = K->ne[2];
    const int gqa_ratio = n_head / n_head_kv;
    const int ne1       = Q->ne[1];                 // queries per head
    const int q_blocks  = (ne1 + Br - 1) / Br;

    const char * Qd = (const char *) Q->data;
    const char * Kd = (const char *) K->data;
    const char * Vd = (const char *) V->data;
    float      * Od = (float *)      dst->data;

    const int64_t nb01 = Q->nb[1], nb02 = Q->nb[2];
    const int64_t nb11 = K->nb[1], nb12 = K->nb[2];
    const int64_t nb21 = V->nb[1], nb22 = V->nb[2];
    const char * maskb     = mask ? (const char *) mask->data : nullptr;
    const int64_t mask_nb1 = mask ? mask->nb[1] : 0;

    if (getenv("GGML_SYCL_FA_XMX_DEBUG")) {
        fprintf(stderr, "[XMX-FA] D=%d kk=%d vk=%d n_q=%d n_head=%d n_kv=%d mask=%d\n",
                D, kk, vk, ne1, n_head, n_kv, maskb ? 1 : 0);
    }

    queue_ptr stream = ctx.stream();
#define XMX_LAUNCH_ARGS stream, Qd, Kd, Vd, Od, scale, n_kv, gqa_ratio, nb01, nb02, nb11, nb12, nb21, nb22, ne1, n_head, q_blocks, maskb, mask_nb1
    switch (kk) {  // kk == vk (asserted above)
        case XMX_F16:    submit_xmx<D, Br, Bc, XMX_F16,    XMX_F16   >(XMX_LAUNCH_ARGS); break;
        case XMX_Q8_0:   submit_xmx<D, Br, Bc, XMX_Q8_0,   XMX_Q8_0  >(XMX_LAUNCH_ARGS); break;
        case XMX_TURBO2: submit_xmx<D, Br, Bc, XMX_TURBO2, XMX_TURBO2>(XMX_LAUNCH_ARGS); break;
        case XMX_TURBO3: submit_xmx<D, Br, Bc, XMX_TURBO3, XMX_TURBO3>(XMX_LAUNCH_ARGS); break;
        case XMX_TURBO4: submit_xmx<D, Br, Bc, XMX_TURBO4, XMX_TURBO4>(XMX_LAUNCH_ARGS); break;
        default: GGML_ABORT("XMX FA: unreachable KV kind");
    }
#undef XMX_LAUNCH_ARGS
}
