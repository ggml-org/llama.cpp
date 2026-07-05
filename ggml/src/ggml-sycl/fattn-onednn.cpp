#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "fattn-onednn.hpp"
#include "fattn-tile.hpp"

// set minimum query length to treat as prefill (32)
#define GGML_SYCL_FA_ONEDNN_MIN_Q 32

bool ggml_sycl_flash_attn_ext_onednn_supported(const ggml_tensor * dst) {
#if !GGML_SYCL_DNNL
    GGML_UNUSED(dst);
    return false;
#else
    if (!g_ggml_sycl_fa_onednn) {
        return false;
    }
    // Battlemage (Xe2) only: the sole architecture validated for this path (single B70 here; dual
    // Arc B60 for the multi-device case). Other archs fall back to the existing FA kernel.
    const gpu_arch arch = ggml_sycl_info().devices[ggml_sycl_get_device()].hw_info.arch;
    if (arch != gpu_arch::intel_gpu_bmg_g21 && arch != gpu_arch::intel_gpu_bmg_g31) {
        return false;
    }
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    // gate for f16 KV only for now
    // need to implement quantized KV
    if (K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16) {
        return false;
    }
    // gate for the following cases
    // 1. if the oneDNN graph Add node has no input --> skip
    // 2. types other than f16 need different logical_tensor declaration
    // 3. the mask must be shape [1, 1, q, seq]
    // 4. sinks: excludes attention sink (Xiao et al., 2024) that can't be modeled by oneDNN graph
    if (!mask || mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || mask->ne[3] != 1 || sinks) {
        return false;
    }
    float max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }
    // K and V must share head_dim: the SDPA graph uses a single `d` for both.
    const int64_t d = K->ne[0];
    if (V->ne[0] != d || Q->ne[3] != 1) {
        return false;
    }
    // GQA must divide evenly.
    if (K->ne[2] == 0 || Q->ne[2] % K->ne[2] != 0) {
        return false;
    }
    // Prefill only.
    if (Q->ne[1] < GGML_SYCL_FA_ONEDNN_MIN_Q) {
        return false;
    }
    return true;
#endif
}

#if GGML_SYCL_DNNL

#include "dnnl.hpp"
#include "dnnl_sycl.hpp"
#include "oneapi/dnnl/dnnl_graph.hpp"   // graph API lives only under oneapi/dnnl/, not at the include root

using namespace dnnl;
using namespace dnnl::graph;

// strided src (f16 or f32) -> contiguous f16 [ne0,ne1,ne2,ne3] (ne0 innermost). nb* are BYTE strides.
template <typename src_t>
static void cont_to_f16_sycl(const char * src, sycl::half * dst,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
        size_t nb1, size_t nb2, size_t nb3, dpct::queue_ptr stream) {
    const int64_t n = ne0 * ne1 * ne2 * ne3;
    stream->parallel_for(sycl::range<1>(n), [=](sycl::id<1> ix) {
        const int64_t gid = ix[0];
        int64_t       i   = gid;
        const int64_t i0 = i % ne0; i /= ne0;
        const int64_t i1 = i % ne1; i /= ne1;
        const int64_t i2 = i % ne2; const int64_t i3 = i / ne2;
        const src_t * p = (const src_t *) (src + i1 * nb1 + i2 * nb2 + i3 * nb3) + i0;
        dst[gid] = (sycl::half) (*p);
    });
}

// oneDNN SDPA out (f16 contiguous [mb,H,q,d]) -> ggml dst (f32 [head_dim,H,n_tok,mb], contiguous).
static void permute_sdpa_out_sycl(const sycl::half * out, float * dst,
        int64_t mb, int64_t H, int64_t q, int64_t d, dpct::queue_ptr stream) {
    const int64_t n = mb * H * q * d;
    stream->parallel_for(sycl::range<1>(n), [=](sycl::id<1> ix) {
        const int64_t gid = ix[0];
        int64_t       i   = gid;
        const int64_t e = i % d; i /= d;
        const int64_t t = i % q; i /= q;
        const int64_t h = i % H; const int64_t b = i / H;
        dst[e + h * d + t * d * H + b * d * H * q] = (float) out[gid];
    });
}

struct sdpa_partition {
    compiled_partition          cp;
    std::vector<logical_tensor> ins;
    logical_tensor              out;
    size_t id_q = 0, id_k = 0, id_v = 0, id_scale = 0, id_mask = 0;
    bool   ok = false;
};

// Build + compile the contiguous-input GQA SDPA graph (MatMul->Divide->Add->SoftMax->MatMul), f32 out.
// Mirrors the hardware-verified scratch/onednn_sdpa_probe.cpp build_gqa (partitions=1, sdp_primitive_kernel_t).
static sdpa_partition build_sdpa(const engine & eng, int H, int Hkv, int q, int seq, int d) {
    using ltype = logical_tensor::layout_type;
    using dt    = logical_tensor::data_type;
    using ldims = logical_tensor::dims;
    const dt    fi = dt::f32, t = dt::f16;
    const int   rep = H / Hkv;
    const ldims q_sz = {1, Hkv, rep, q, d}, kv_sz = {1, Hkv, 1, seq, d}, s_sz = {1, Hkv, rep, q, seq},
                sc = {1, 1, 1, 1, 1}, msk = {1, 1, 1, q, seq}, o_sz = {1, Hkv, rep, q, d};
    int64_t        id = 0;
    sdpa_partition E;

    auto query  = logical_tensor(id++, t,  q_sz, ltype::strided);
    auto key    = logical_tensor(id++, t,  kv_sz, ltype::strided);
    auto score  = logical_tensor(id++, fi, s_sz, ltype::strided);
    auto bmm1   = op(id++, op::kind::MatMul, "bmm1");
    bmm1.set_attr<bool>(op::attr::transpose_b, true);          // key is [.., seq, d]
    bmm1.add_inputs({query, key}); bmm1.add_outputs({score});

    auto scale  = logical_tensor(id++, t,  sc,   ltype::strided);
    auto scaled = logical_tensor(id++, fi, s_sz, ltype::strided);
    auto sdiv   = op(id++, op::kind::Divide, "scale_div");     // score / (1/kq_scale) == score * kq_scale
    sdiv.add_inputs({score, scale}); sdiv.add_outputs({scaled});

    auto mask   = logical_tensor(id++, t,  msk,  ltype::strided);
    auto masked = logical_tensor(id++, fi, s_sz, ltype::strided);
    auto madd   = op(id++, op::kind::Add, "mask_add");
    madd.add_inputs({scaled, mask}); madd.add_outputs({masked});

    auto probs  = logical_tensor(id++, t,  s_sz, ltype::strided);
    auto smax   = op(id++, op::kind::SoftMax, "softmax");
    smax.set_attr<int64_t>(op::attr::axis, -1);
    smax.set_attr<std::string>(op::attr::mode, "inf_as_zero");
    smax.add_inputs({masked}); smax.add_outputs({probs});

    auto value  = logical_tensor(id++, t,  kv_sz, ltype::strided);
    // f16 output is REQUIRED to hit sdp_primitive_kernel_t (the systolic micro-kernel); an f32 output
    // falls to larger_partition_kernel_t which materializes N^2 (confirmed: scratch/onednn_sdpa_kernel_probe.cpp).
    // converted to the f32 ggml dst in the permute below.
    auto output = logical_tensor(id++, t,  o_sz, ltype::strided);   // f16 contiguous [mb,Hkv,rep,q,d]
    auto bmm2   = op(id++, op::kind::MatMul, "bmm2");
    bmm2.add_inputs({probs, value}); bmm2.add_outputs({output});

    dnnl::graph::graph g(eng.get_kind());
    g.add_op(bmm1); g.add_op(sdiv); g.add_op(madd); g.add_op(smax); g.add_op(bmm2);
    g.finalize();

    auto parts = g.get_partitions();
    if (parts.size() != 1 || !parts[0].is_supported()) {
        return E;   // ok stays false -> caller falls back to TILE
    }
    E.ins      = parts[0].get_input_ports();
    E.out      = parts[0].get_output_ports()[0];
    E.cp       = parts[0].compile(E.ins, {E.out}, eng);
    E.out      = E.cp.query_logical_tensor(E.out.get_id());
    E.id_q     = query.get_id(); E.id_k = key.get_id(); E.id_v = value.get_id();
    E.id_scale = scale.get_id(); E.id_mask = mask.get_id();
    E.ok       = true;
    return E;
}

void ggml_sycl_flash_attn_ext_onednn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) try {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const int64_t d   = K->ne[0];   // head_dim
    const int64_t seq = K->ne[1];   // n_kv
    const int64_t Hkv = K->ne[2];   // n_head_kv
    const int64_t H   = Q->ne[2];   // n_head
    const int64_t q   = Q->ne[1];   // n_tok
    const int64_t mb  = Q->ne[3];   // batch (== 1, gated)

    float kq_scale = 1.0f;
    memcpy(&kq_scale, (const float *) dst->op_params + 0, sizeof(float));

    dpct::queue_ptr stream = ctx.stream();
    dnnl::engine    eng    = ctx.engine_dnnl(stream);
    dnnl::stream    strm   = ctx.stream_dnnl(stream);

    // cont/cast inputs to contiguous f16 (head-major) -- the layout the fast systolic path wants.
    ggml_sycl_pool_alloc<sycl::half> Qf(ctx.pool(), (size_t) H   * q   * d);
    ggml_sycl_pool_alloc<sycl::half> Kf(ctx.pool(), (size_t) Hkv * seq * d);
    ggml_sycl_pool_alloc<sycl::half> Vf(ctx.pool(), (size_t) Hkv * seq * d);
    cont_to_f16_sycl<float>     ((const char *) Q->data, Qf.get(), d, q,   H,   mb, Q->nb[1], Q->nb[2], Q->nb[3], stream);
    cont_to_f16_sycl<sycl::half>((const char *) K->data, Kf.get(), d, seq, Hkv, mb, K->nb[1], K->nb[2], K->nb[3], stream);
    cont_to_f16_sycl<sycl::half>((const char *) V->data, Vf.get(), d, seq, Hkv, mb, V->nb[1], V->nb[2], V->nb[3], stream);

    // divide-by-(1/scale) reproduces ggml's score *= kq_scale on the proven probe graph.
    const sycl::half scale_h = (sycl::half) (1.0f / kq_scale);
    ggml_sycl_pool_alloc<sycl::half> scbuf(ctx.pool(), 1);
    stream->memcpy(scbuf.get(), &scale_h, sizeof(sycl::half));

    ggml_sycl_pool_alloc<sycl::half> outf(ctx.pool(), (size_t) H * q * d);   // f16 contiguous SDPA out [mb,H,q,d]

    // compile once per (device, shape), reuse across layers/calls.
    static std::unordered_map<std::string, sdpa_partition> cache;
    char keyb[96];
    snprintf(keyb, sizeof(keyb), "%d:%lld:%lld:%lld:%lld:%lld", ggml_sycl_get_device(),
             (long long) H, (long long) Hkv, (long long) q, (long long) seq, (long long) d);
    auto it = cache.find(keyb);
    if (it == cache.end()) {
        it = cache.emplace(keyb, build_sdpa(eng, (int) H, (int) Hkv, (int) q, (int) seq, (int) d)).first;
    }
    sdpa_partition & E = it->second;
    // _supported() is authoritative: if it accepted this op the partition must build.
    // A failure here is a gap in _supported() -- surface it, don't mask it with a fallback.
    GGML_ASSERT(E.ok && "oneDNN SDPA partition failed to build for a _supported() shape");

    auto id2ptr = [&](size_t r) -> void * {
        if (r == E.id_q)     return Qf.get();
        if (r == E.id_k)     return Kf.get();
        if (r == E.id_v)     return Vf.get();
        if (r == E.id_scale) return scbuf.get();
        if (r == E.id_mask)  return (void *) mask->data;
        return nullptr;
    };
    std::vector<tensor> ti;
    ti.reserve(E.ins.size());
    for (auto & lt : E.ins) {
        ti.emplace_back(lt, eng, id2ptr(lt.get_id()));
    }
    tensor to(E.out, eng, outf.get());
    E.cp.execute(strm, ti, {to});

    permute_sdpa_out_sycl(outf.get(), (float *) dst->data, mb, H, q, d, stream);
    // Single device: no sync is required, and actually PP perf is ~6% > wait_and_throw() (tested on llama-3.1-8b & qwen3.6-27b, both Q8_0, with Arc B70).
    // Any future multi-GPU refactor MUST re-measure this single-device path and keep the best
    // single-device PP speed. Otherwise (multiple devices/streams can race the reuse):
    if (ggml_sycl_info().device_count > 1) {
        // cont_to_f16 -> oneDNN execute -> permute is async on this stream, but the
        // pool_alloc*s above free their device buffers at host return. Without this wait the next
        // scheduler op re-acquires those bytes while the GPU is still computing the SDPA, turning
        // it into garbage and collapsing multi-turn output to a single repeated token ("GGGGG...").
        stream->wait_and_throw();
    }
}
catch (const std::exception & e) {
    // any oneDNN/SYCL failure is non-fatal: fall back to the existing kernel (strictly additive).
    GGML_LOG_WARN("%s: oneDNN SDPA failed (%s); falling back to TILE kernel\n", __func__, e.what());
    ggml_sycl_flash_attn_ext_tile(ctx, dst);
}

#endif // GGML_SYCL_DNNL
