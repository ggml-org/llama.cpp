#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include "dpct/helper.hpp"
#include "common.hpp"
#include "fattn-common.hpp"
#include "fattn-dpas.hpp"

// GGML_SYCL_SUPPORT_XMX=OFF builds skip the DPAS below, as do spir64_gen builds for sub-group 8
// targets (Xe-HPG) or mixed target lists, which cannot compile it
#if !defined(GGML_SYCL_SUPPORT_XMX) || (defined(GGML_SYCL_XMX_AOT_SG) && GGML_SYCL_XMX_AOT_SG != 16)
#define GGML_SYCL_FA_DPAS_BUILD 0
#else
#define GGML_SYCL_FA_DPAS_BUILD 1
#endif

// the sg8 probe at the end of this file targets sub-group 8 instead: it builds when XMX is on and
// the AOT sub-group, if one is fixed, is 8. The operational build is JIT, where both bodies build.
#if !defined(GGML_SYCL_SUPPORT_XMX) || (defined(GGML_SYCL_XMX_AOT_SG) && GGML_SYCL_XMX_AOT_SG != 8)
#define GGML_SYCL_FA_SG8_PROBE_BUILD 0
#else
#define GGML_SYCL_FA_SG8_PROBE_BUILD 1
#endif

// the sg8 FA body (XMX remedy 7, stage B) follows the same split: it builds when XMX is on and
// the AOT sub-group, if one is fixed, is 8. In a JIT build both bodies are present.
#if !defined(GGML_SYCL_SUPPORT_XMX) || (defined(GGML_SYCL_XMX_AOT_SG) && GGML_SYCL_XMX_AOT_SG != 8)
#define GGML_SYCL_FA_DPAS_SG8_BUILD 0
#else
#define GGML_SYCL_FA_DPAS_SG8_BUILD 1
#endif

#if GGML_SYCL_FA_DPAS_BUILD || GGML_SYCL_FA_DPAS_SG8_BUILD
static constexpr int FA_DPAS_WG    = 256; // work-group size, both bodies
static constexpr int FA_DPAS_MAX_Q = 31;  // larger batches go to the prefill kernels
static constexpr int FA_DPAS_MAX_G = 32;  // query heads per KV head

// head sizes with a compiled kernel (multiples of the 32-dim q4_0 / q8_0 block)
static bool fattn_dpas_head_ok(int64_t d) {
    return d == 64 || d == 96 || d == 128 || d == 256 || d == 512;
}
#endif // GGML_SYCL_FA_DPAS_BUILD || GGML_SYCL_FA_DPAS_SG8_BUILD

#if GGML_SYCL_FA_DPAS_BUILD
static constexpr int   FA_DPAS_NSG  = 16;       // sub-groups of 16 lanes
static constexpr float FA_DPAS_MINF = -1.0e30f; // finite start value of the running max

static bool fattn_dpas_support_arch(gpu_arch arch) {
    return arch == gpu_arch::intel_gpu_bmg_g21 || arch == gpu_arch::intel_gpu_bmg_g31 ||
           arch == gpu_arch::intel_gpu_lnl_m   || arch == gpu_arch::intel_gpu_ptl_h   ||
           arch == gpu_arch::intel_gpu_ptl_u   || arch == gpu_arch::intel_gpu_wcl;
}
#endif // GGML_SYCL_FA_DPAS_BUILD

#if GGML_SYCL_FA_DPAS_SG8_BUILD
static constexpr int   FA_DPAS_SG8_NSG  = 32;       // sub-groups of 8 lanes
static constexpr float FA_DPAS_SG8_MINF = -1.0e30f; // finite start value of the running max

// ACM (Xe1) devices: the same device list as the stage 1 sg8 predicate in common.hpp
static bool fattn_dpas_sg8_support_arch(gpu_arch arch) {
    return arch == gpu_arch::intel_gpu_acm_g10 || arch == gpu_arch::intel_gpu_acm_g11 ||
           arch == gpu_arch::intel_gpu_acm_g12;
}

// the Xe2 shape gates, narrowed to the sg8 v1 target cell: single token, K q8_0, D 256, and the
// staged decode case of the Xe2 dispatch (16-byte aligned K/V, G <= 16). Everything else on ACM
// keeps falling through to the VEC / TILE kernels.
static bool fattn_dpas_sg8_supported(const ggml_tensor * dst) {
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    // the K side is wired as q8_0 only in v1 (the s4 form stays probe-only); V follows its own type
    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_Q8_0 ||
        (V->type != GGML_TYPE_Q4_0 && V->type != GGML_TYPE_Q8_0) || (sinks && sinks->type != GGML_TYPE_F32)) {
        return false;
    }
    if (!fattn_dpas_head_ok(K->ne[0]) || K->ne[0] != 256 || V->ne[0] != K->ne[0] || Q->ne[0] != K->ne[0]) {
        return false;
    }
    if (Q->ne[1] != 1 || K->ne[1] < 1) {
        return false;
    }
    if (K->ne[2] < 1 || V->ne[2] != K->ne[2] || Q->ne[2] % K->ne[2] != 0 || Q->ne[2] / K->ne[2] > FA_DPAS_MAX_G) {
        return false;
    }
    if (Q->ne[3] != K->ne[3] || V->ne[3] != K->ne[3]) {
        return false;
    }

    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }

    if (mask && (mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || mask->nb[0] != sizeof(sycl::half) ||
                 mask->ne[0] < K->ne[1] || mask->ne[1] < Q->ne[1])) {
        return false;
    }
    if (sinks && (sinks->nb[0] != sizeof(float) || sinks->ne[0] < Q->ne[2])) {
        return false;
    }
    if (Q->nb[0] != sizeof(float)) {
        return false;
    }
    // q4_0 / q8_0 rows are read as dwords from the dword at or before the row start: rows must start on 2-byte bounds
    // (head sizes with an odd block count, e.g. 96, put every other head 2 bytes into a dword)
    // so a row's first/last dword may hold 2 bytes outside it; aligned dwords never cross the 128-byte buffer alignment
    for (const ggml_tensor * t : { K, V }) {
        if (t->nb[0] != ggml_type_size(t->type) || t->nb[1] % 2 != 0 || t->nb[2] % 2 != 0 || t->nb[3] % 2 != 0 ||
            (uintptr_t) t->data % 2 != 0) {
            return false;
        }
    }
    // the staged decode case: K/V strides and data 16-byte aligned, at most 16 query heads per KV head
    if (Q->ne[2] / K->ne[2] > 16) {
        return false;
    }
    for (const ggml_tensor * t : { K, V }) {
        if (t->nb[1] % 16 != 0 || t->nb[2] % 16 != 0 || t->nb[3] % 16 != 0 || (uintptr_t) t->data % 16 != 0) {
            return false;
        }
    }
    return true;
}
#endif // GGML_SYCL_FA_DPAS_SG8_BUILD

bool ggml_sycl_flash_attn_ext_dpas_supported(int device, const ggml_tensor * dst) {
#if GGML_SYCL_FA_DPAS_SG8_BUILD
    // ACM devices take the sg8 judgment when its toggle is on; the Xe2 judgment below is unchanged
    if (g_ggml_sycl_fa_dpas && g_ggml_sycl_fa_dpas_sg8 &&
        fattn_dpas_sg8_support_arch(ggml_sycl_info().devices[device].hw_info.arch)) {
        return fattn_dpas_sg8_supported(dst);
    }
#endif
#if !GGML_SYCL_FA_DPAS_BUILD
    GGML_UNUSED(device);
    GGML_UNUSED(dst);
    return false;
#else
    if (!g_ggml_sycl_fa_dpas || !fattn_dpas_support_arch(ggml_sycl_info().devices[device].hw_info.arch)) {
        return false;
    }

    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    if (Q->type != GGML_TYPE_F32 || (K->type != GGML_TYPE_Q4_0 && K->type != GGML_TYPE_Q8_0) ||
        (V->type != GGML_TYPE_Q4_0 && V->type != GGML_TYPE_Q8_0) ||
        (K->type == GGML_TYPE_Q4_0 && V->type == GGML_TYPE_Q8_0) || (sinks && sinks->type != GGML_TYPE_F32)) {
        return false;
    }
    if (!fattn_dpas_head_ok(K->ne[0]) || V->ne[0] != K->ne[0] || Q->ne[0] != K->ne[0]) {
        return false;
    }
    if (Q->ne[1] < 1 || Q->ne[1] > FA_DPAS_MAX_Q || K->ne[1] < 1) {
        return false;
    }
    if (K->ne[2] < 1 || V->ne[2] != K->ne[2] || Q->ne[2] % K->ne[2] != 0 || Q->ne[2] / K->ne[2] > FA_DPAS_MAX_G) {
        return false;
    }
    if (Q->ne[3] != K->ne[3] || V->ne[3] != K->ne[3]) {
        return false;
    }

    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }

    if (mask && (mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || mask->nb[0] != sizeof(sycl::half) ||
                 mask->ne[0] < K->ne[1] || mask->ne[1] < Q->ne[1])) {
        return false;
    }
    if (sinks && (sinks->nb[0] != sizeof(float) || sinks->ne[0] < Q->ne[2])) {
        return false;
    }
    if (Q->nb[0] != sizeof(float)) {
        return false;
    }
    // q4_0 / q8_0 rows are read as dwords from the dword at or before the row start: rows must start on 2-byte bounds
    // (head sizes with an odd block count, e.g. 96, put every other head 2 bytes into a dword)
    // so a row's first/last dword may hold 2 bytes outside it; aligned dwords never cross the 128-byte buffer alignment
    for (const ggml_tensor * t : { K, V }) {
        if (t->nb[0] != ggml_type_size(t->type) || t->nb[1] % 2 != 0 || t->nb[2] % 2 != 0 || t->nb[3] % 2 != 0 ||
            (uintptr_t) t->data % 2 != 0) {
            return false;
        }
    }
    return true;
#endif
}

#if GGML_SYCL_FA_DPAS_BUILD || GGML_SYCL_FA_DPAS_SG8_BUILD

typedef short fa_dpas_short8 __attribute__((ext_vector_type(8)));
typedef int   fa_dpas_int4   __attribute__((ext_vector_type(4)));
typedef int   fa_dpas_int8   __attribute__((ext_vector_type(8)));
typedef float fa_dpas_float8 __attribute__((ext_vector_type(8)));

#endif // GGML_SYCL_FA_DPAS_BUILD || GGML_SYCL_FA_DPAS_SG8_BUILD

#if GGML_SYCL_FA_DPAS_BUILD

// Xe2 DPAS (SIMD16, systolic depth 8, repeat count 8) as inline vISA: joint_matrix has no s4 type.
// d = a x b. a: 8 rows x 32 s8, 2 per lane per row. b: 32 x 16 s4, 16 bytes per lane, one column per lane.
static inline void fa_dpas_s8_s4(const fa_dpas_short8 & a, const fa_dpas_int4 & b, fa_dpas_int8 & d) {
#ifdef __SYCL_DEVICE_ONLY__
    __asm__("{\n"
            ".decl DB v_type=G type=ud num_elts=64 align=GRF alias=<%1,0>\n"
            ".decl DA v_type=G type=ud num_elts=64 align=GRF alias=<%2,0>\n"
            "dpas.s4.s8.8.8 (M1, 16) %0.0 %%null.0 DB.0 DA(0,0)\n"
            "}\n"
            : "=rw"(d)
            : "rw"(b), "rw"(a));
#else
    d = 0;
    GGML_UNUSED(a);
    GGML_UNUSED(b);
#endif
}

// d = a x b. a: 8 rows x 32 s8, 2 per lane per row. b: 32 x 16 s8, 32 bytes per lane, one column per lane.
static inline void fa_dpas_s8_s8(const fa_dpas_short8 & a, const fa_dpas_int8 & b, fa_dpas_int8 & d) {
#ifdef __SYCL_DEVICE_ONLY__
    __asm__("{\n"
            ".decl DB v_type=G type=ud num_elts=128 align=GRF alias=<%1,0>\n"
            ".decl DA v_type=G type=ud num_elts=64 align=GRF alias=<%2,0>\n"
            "dpas.s8.s8.8.8 (M1, 16) %0.0 %%null.0 DB.0 DA(0,0)\n"
            "}\n"
            : "=rw"(d)
            : "rw"(b), "rw"(a));
#else
    d = 0;
    GGML_UNUSED(a);
    GGML_UNUSED(b);
#endif
}

// acc += a x b. a: 8 rows x 16 f16, one per lane per row. b: 16 x 16 f16 as pairs, 8 dwords per lane, one column per lane.
static inline void fa_dpas_f16(const fa_dpas_short8 & a, const fa_dpas_int8 & b, fa_dpas_float8 & acc) {
#ifdef __SYCL_DEVICE_ONLY__
    __asm__("{\n"
            ".decl DB v_type=G type=ud num_elts=128 align=GRF alias=<%1,0>\n"
            ".decl DA v_type=G type=ud num_elts=64 align=GRF alias=<%2,0>\n"
            "dpas.hf.hf.8.8 (M1, 16) %0.0 %0.0 DB.0 DA(0,0)\n"
            "}\n"
            : "+rw"(acc)
            : "rw"(b), "rw"(a));
#else
    GGML_UNUSED(a);
    GGML_UNUSED(b);
    GGML_UNUSED(acc);
#endif
}

#endif // GGML_SYCL_FA_DPAS_BUILD

#if GGML_SYCL_FA_DPAS_BUILD || GGML_SYCL_FA_DPAS_SG8_BUILD

// Block b of a row = f16 scale + NQ dwords of quants (q4_0: 18 bytes, NQ 4; q8_0: 34 bytes, NQ 8), at byte (2 + 4 NQ) b.
// w = the NQ + 1 dwords from that byte / 4. An odd block starts 2 bytes into w[0].
template <int NQ> static inline void fa_dpas_split_block(const uint32_t * w, bool odd, uint32_t & sc, uint32_t * q) {
    if (!odd) {
        sc = w[0] & 0xFFFFu;
#pragma unroll
        for (int i = 0; i < NQ; ++i) {
            q[i] = (w[i] >> 16) | (w[i + 1] << 16);
        }
    } else {
        sc = w[0] >> 16;
#pragma unroll
        for (int i = 0; i < NQ; ++i) {
            q[i] = w[i + 1];
        }
    }
}

// Stage TK rows of RW dwords into SLM at row stride SW. Keys past k_end repeat key k_begin.
// WG is a parameter so the sg8 body can share this helper; both bodies launch 256-wide groups.
template <int TK, int SW, int ST, int RW, int WG = FA_DPAS_WG>
static void fa_dpas_stage(uint32_t * sKV, const int tid, const char * base, const int64_t nb, const int t0,
                          const int k_begin, const int k_end) {
    if constexpr (ST == 2) {
        for (int e = tid; e < TK * RW; e += WG) {
            const int key = e / RW;
            const int q   = e - key * RW;
            const int j   = t0 + key < k_end ? t0 + key : k_begin;
            sKV[key * SW + q] = ((const uint32_t *) (base + (int64_t) j * nb))[q];
        }
        return;
    }
    constexpr int RV  = ST == 1 ? RW / 4 : 1;
    constexpr int NV  = TK * RV;
    constexpr int PER = (NV + WG - 1) / WG;
    sycl::uint4   v[PER];
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int e = tid + i * WG;
        if (e < NV) {
            const int key = e / RV;
            const int q   = e - key * RV;
            const int j   = t0 + key < k_end ? t0 + key : k_begin;
            v[i]          = ((const sycl::uint4 *) (base + (int64_t) j * nb))[q];
        }
    }
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int e = tid + i * WG;
        if (e < NV) {
            const int key = e / RV;
            const int q   = e - key * RV;
            const int o   = key * SW + 4 * q;
            sKV[o + 0]    = v[i][0];
            sKV[o + 1]    = v[i][1];
            sKV[o + 2]    = v[i][2];
            sKV[o + 3]    = v[i][3];
        }
    }
}

// The NQ + 1 dwords of block b of a row: of tile key kk in SLM when staged, else of the row itself.
template <int BB, int NQ, int SW, int ST>
static void fa_dpas_load_block(const uint32_t * sKV, const uint32_t * row, const int ph, const int kk, const int b,
                               uint32_t * wv) {
    const int wd = (2 * ph + BB * b) / 4;
    if constexpr (ST) {
        GGML_UNUSED(row);
#pragma unroll
        for (int i = 0; i < NQ + 1; ++i) {
            wv[i] = sKV[kk * SW + wd + i];
        }
    } else {
        GGML_UNUSED(kk);
#pragma unroll
        for (int i = 0; i < NQ + 1; ++i) {
            wv[i] = row[wd + i];
        }
    }
}

struct fa_dpas_params {
    const char *   Q;
    const char *   K;
    const char *   V;
    const char *   mask;
    float *        dst;
    float *        parts;  // per-split partial results, nsplit > 1 only
    sycl::float2 * meta;   // per-split (max, sum), nsplit > 1 only
    float          scale;
    const float *  sinks;  // per query head softmax sink logit, or null
    int            G;      // query heads per KV head
    int            nq;     // query tokens per work-group
    int            nqc;    // token chunks
    int            ne01;
    int            ne02;
    int            ne11;
    int            ne33;
    int            nsplit;
    int            chunk;  // keys per split
    int64_t        nb01, nb02, nb03;
    int64_t        nb11, nb12, nb13;
    int64_t        nb21, nb22, nb23;
    int64_t        nb31, nb33;
};

#endif // GGML_SYCL_FA_DPAS_BUILD || GGML_SYCL_FA_DPAS_SG8_BUILD

#if GGML_SYCL_FA_DPAS_BUILD

// One work-group = one KV head x nq query tokens x one slice of the keys: S = Q K^T by s8 x s4 / s8 x s8 DPAS on the
// raw K blocks, online softmax, O^T += V^T P^T by f16 DPAS. ST: stage K/V tiles in SLM (1: 16-byte, 2: dword loads).
// LG: 256-register file (no spills at 32 rows, same occupancy). Q8K / Q8V: K / V are q8_0, else q4_0; mixed pairs
// (K q8_0, V q4_0) are allowed, each side follows its own type.
template <int D, int NR16, int KG, int TC, int ST, bool LG, bool Q8K, bool Q8V>
static void fattn_dpas_kernel(const fa_dpas_params p, const int nkvh, const int ne03, dpct::queue_ptr stream) {
    constexpr int RMAX  = 16 * NR16;
    constexpr int RTMAX = RMAX / 8;                         // 8-row tiles in Q.K
    constexpr int TK    = 16 * KG;                          // keys per tile
    static_assert(QK4_0 == QK8_0, "q4_0 and q8_0 blocks hold the same number of dims");
    constexpr int BBK   = Q8K ? (int) sizeof(block_q8_0) : (int) sizeof(block_q4_0); // bytes per K block
    constexpr int BBV   = Q8V ? (int) sizeof(block_q8_0) : (int) sizeof(block_q4_0); // bytes per V block
    constexpr int NQK   = (BBK - 2) / 4;                    // quant dwords per K block
    constexpr int NQV   = (BBV - 2) / 4;                    // quant dwords per V block
    constexpr int NB    = D / QK4_0;                        // blocks per row
    constexpr int DPS   = D <= 128 ? 8 : D <= 256 ? 16 : 32; // head dims per sub-group in P.V, inside one 32-dim block
    constexpr int NPV   = D / DPS;                          // sub-groups doing P.V, the rest only reach its barrier
    constexpr bool ALLPV = NPV == FA_DPAS_NSG;              // every sub-group does P.V: no runtime check in the loops
    constexpr int XT    = DPS / 8;                          // 8-dim tiles per sub-group in P.V
    constexpr int PS    = TK + 16;                          // f16 row stride of P
    constexpr int RWK   = NB * BBK / 4;                     // dwords per K row
    constexpr int RWV   = NB * BBV / 4;                     // dwords per V row
    constexpr bool ODD  = (NB * BBK) % 4 != 0;              // odd block count: rows may start 2 bytes into a dword (same for V: both block sizes are 2 mod 4)
    constexpr int SW    = (RWK > RWV ? RWK : RWV) + 1;      // odd SLM row stride: no bank conflicts for lane = key
    static_assert(D % QK4_0 == 0 && D % DPS == 0 && NPV <= FA_DPAS_NSG && QK4_0 % DPS == 0, "unsupported head size");
    static_assert(ST != 1 || (RWK % 4 == 0 && RWV % 4 == 0), "16-byte staging needs rows of 16-byte multiples");

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<fa_dpas_short8, 1> sQ8(sycl::range<1>(RTMAX * NB * 16), cgh); // Q s8 [tile][block][lane]
        sycl::local_accessor<fa_dpas_float8, 1> sDQ(sycl::range<1>(RTMAX * NB), cgh);      // Q scales [tile][block]
        sycl::local_accessor<float, 1>          sS(sycl::range<1>(RMAX * TK), cgh);        // S [row][key]
        sycl::local_accessor<fa_dpas_int8, 1>   sP(sycl::range<1>(RMAX * PS / 16), cgh);   // P f16 [row][key]
        sycl::local_accessor<float, 1>          sM(sycl::range<1>(RMAX), cgh);
        sycl::local_accessor<float, 1>          sL(sycl::range<1>(RMAX), cgh);
        sycl::local_accessor<float, 1>          sA(sycl::range<1>(RMAX), cgh);
        sycl::local_accessor<uint32_t, 1>       sKV(sycl::range<1>(ST ? TK * SW : 1), cgh);
        const sycl::nd_range<3> ndr(sycl::range<3>((size_t) ne03 * p.nqc, nkvh, (size_t) p.nsplit * FA_DPAS_WG),
                                    sycl::range<3>(1, 1, FA_DPAS_WG));
        auto kern = [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(16)]] {
                const int tid   = it.get_local_id(2);
                const int split = it.get_group(2);
                const int kvh   = it.get_group(1);
                const int seq   = it.get_group(0) / p.nqc;
                const int c0    = (it.get_group(0) % p.nqc) * p.nq;
                const int G     = p.G;
                const int R     = p.nq * G;
                const int RT    = (R + 7) / 8;
                auto      sg    = it.get_sub_group();
                const int w     = tid / 16;
                const int lane  = tid % 16;
                short *   sQs   = (short *) &sQ8[0];
                short *   sPs   = (short *) &sP[0];
                uint32_t * sKVp = &sKV[0];

                // Q -> s8, one scale per row and block. Lane l holds the 2 dims of K column elements 2 l, 2 l + 1:
                // q4_0 packs dims l and l + 16 into byte l, q8_0 stores dims in order
                for (int e = tid; e < RT * 8 * NB; e += FA_DPAS_WG) {
                    const int  r    = e / NB;
                    const int  b    = e % NB;
                    const int  c    = c0 + r / G;
                    const bool real = r < R && c < p.ne01;
                    const float * qr = (const float *) (p.Q + seq * p.nb03 + (int64_t) (real ? c : 0) * p.nb01 +
                                                        (int64_t) (kvh * G + (real ? r % G : 0)) * p.nb02) + b * QK4_0;
                    float x[QK4_0];
                    float amax = 0.0f;
#pragma unroll
                    for (int i = 0; i < QK4_0; ++i) {
                        x[i] = real ? qr[i] * p.scale : 0.0f;
                        amax = sycl::fmax(amax, sycl::fabs(x[i]));
                    }
                    const float dq = amax / 127.0f;
                    const float iq = amax > 0.0f ? 127.0f / amax : 0.0f;
                    const int   t  = r / 8;
                    const int   m  = r % 8;
                    ((float *) &sDQ[0])[(t * NB + b) * 8 + m] = dq;
#pragma unroll
                    for (int l = 0; l < 16; ++l) {
                        const int lo = (int) sycl::rint(x[Q8K ? 2 * l : l] * iq);
                        const int hi = (int) sycl::rint(x[Q8K ? 2 * l + 1 : l + 16] * iq);
                        sQs[((t * NB + b) * 16 + l) * 8 + m] = (short) ((lo & 0xFF) | ((hi & 0xFF) << 8));
                    }
                }
                // rows without a query keep P = 0
                for (int e = tid; e < RMAX * PS; e += FA_DPAS_WG) {
                    sPs[e] = 0;
                }
                if (tid < RMAX) {
                    // the sink is one more key with logit sinks[head] and no value; the first key slice owns it
                    const bool sink = p.sinks && split == 0 && tid < R;
                    sM[tid] = sink ? p.sinks[kvh * G + tid % G] : FA_DPAS_MINF;
                    sL[tid] = sink ? 1.0f : 0.0f;
                    sA[tid] = 1.0f;
                }
                it.barrier(sycl::access::fence_space::local_space);

                const int    k_begin = split * p.chunk;
                const int    k_end   = sycl::min(k_begin + p.chunk, p.ne11);
                const char * Kh      = p.K + seq * p.nb13 + kvh * p.nb12;
                const char * Vh      = p.V + seq * p.nb23 + kvh * p.nb22;
                const sycl::half * mrow = p.mask ? (const sycl::half *) (p.mask + (seq % p.ne33) * p.nb33) : nullptr;

                // O^T: lane = query row, element m = head dim
                fa_dpas_float8 o[NR16][XT];
#pragma unroll
                for (int t = 0; t < NR16; ++t) {
#pragma unroll
                    for (int x = 0; x < XT; ++x) {
                        o[t][x] = 0.0f;
                    }
                }
                // the dims of this sub-group lie in one V block
                const int vb  = (DPS * w) / QK4_0;
                const int vd0 = (DPS * w) % QK4_0;

                // start loading the rows of keys t0 .. t0 + TK - 1 into L1 while the current phase computes
                auto prefetch_rows = [&](const char * base, int64_t nb, int t0, int rw) {
                    const int NL = rw * 4 / 64 + 1; // cache lines per row
                    for (int e = tid; e < TK * NL; e += FA_DPAS_WG) {
                        const int j = t0 + e / NL;
                        if (j < k_end) {
                            char * pa = const_cast<char *>(base) + (int64_t) j * nb + (e % NL) * 64;
                            sycl::ext::oneapi::experimental::prefetch(pa, sycl::ext::oneapi::experimental::properties{ sycl::ext::oneapi::experimental::prefetch_hint_L1 });
                        }
                    }
                };

                for (int t0 = k_begin; t0 < k_end; t0 += TK) {
                    if constexpr (ST) {
                        fa_dpas_stage<TK, SW, ST, RWK>(sKVp, tid, Kh, p.nb11, t0, k_begin, k_end);
                        it.barrier(sycl::access::fence_space::local_space);
                    }
                    prefetch_rows(Vh, p.nb21, t0, RWV);

                    // S = Q K^T, job = (key group, TC row tiles)
                    const int nch = (RT + TC - 1) / TC;
                    for (int job = w; job < KG * nch; job += FA_DPAS_NSG) {
                        const int  kg = job % KG;
                        const int  ch = job / KG;
                        const int  j  = t0 + kg * 16 + lane;
                        const bool jv = j < k_end;
                        const char *     kb = Kh + (int64_t) (jv ? j : k_begin) * p.nb11;
                        const int        kp = (ST || !ODD) ? 0 : (int) (((uintptr_t) kb >> 1) & 1); // row starts 2 bytes into a dword
                        const uint32_t * kr = (const uint32_t *) (kb - 2 * kp);
                        fa_dpas_float8 s[TC];
#pragma unroll
                        for (int tt = 0; tt < TC; ++tt) {
                            s[tt] = 0.0f;
                        }
#pragma unroll
                        for (int b = 0; b < NB; ++b) {
                            uint32_t wv[NQK + 1];
                            uint32_t sc;
                            uint32_t q[NQK];
                            fa_dpas_load_block<BBK, NQK, SW, ST>(sKVp, kr, kp, kg * 16 + lane, b, wv);
                            fa_dpas_split_block<NQK>(wv, (b + kp) % 2 != 0, sc, q);
                            const float dk = static_cast<float>(sycl::bit_cast<sycl::half>((uint16_t) sc));
#pragma unroll
                            for (int tt = 0; tt < TC; ++tt) {
                                const int t = ch * TC + tt;
                                if (t < RT) {
                                    fa_dpas_int8 ia;
                                    if constexpr (Q8K) {
                                        const fa_dpas_int8 bv = { (int) q[0], (int) q[1], (int) q[2], (int) q[3],
                                                                  (int) q[4], (int) q[5], (int) q[6], (int) q[7] };
                                        fa_dpas_s8_s8(sQ8[(t * NB + b) * 16 + lane], bv, ia);
                                    } else {
                                        // u4 (n + 8) -> s4 n
                                        const fa_dpas_int4 bv = { (int) (q[0] ^ 0x88888888u), (int) (q[1] ^ 0x88888888u),
                                                                  (int) (q[2] ^ 0x88888888u), (int) (q[3] ^ 0x88888888u) };
                                        fa_dpas_s8_s4(sQ8[(t * NB + b) * 16 + lane], bv, ia);
                                    }
                                    const fa_dpas_float8 ds = sDQ[t * NB + b] * dk;
                                    s[tt] += __builtin_convertvector(ia, fa_dpas_float8) * ds;
                                }
                            }
                        }
#pragma unroll
                        for (int tt = 0; tt < TC; ++tt) {
                            const int t = ch * TC + tt;
                            if (t < RT) {
#pragma unroll
                                for (int m = 0; m < 8; ++m) {
                                    const int r = t * 8 + m;
                                    const int c = c0 + r / G;
                                    if (r < R) {
                                        const bool  ok = jv && c < p.ne01;
                                        const float mv = (ok && mrow) ? static_cast<float>(mrow[((int64_t) c * p.nb31) / (int64_t) sizeof(sycl::half) + j]) : 0.0f;
                                        sS[r * TK + kg * 16 + lane] = ok ? s[tt][m] + mv : -INFINITY;
                                    }
                                }
                            }
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                    if constexpr (ST) {
                        fa_dpas_stage<TK, SW, ST, RWV>(sKVp, tid, Vh, p.nb21, t0, k_begin, k_end); // the softmax does not use sKV
                    }

                    // online softmax per row -> P
                    for (int r = w; r < R; r += FA_DPAS_NSG) {
                        float sv[KG];
                        float mt = -INFINITY;
#pragma unroll
                        for (int i = 0; i < KG; ++i) {
                            sv[i] = sS[r * TK + i * 16 + lane];
                            mt    = sycl::fmax(mt, sv[i]);
                        }
                        mt = sycl::reduce_over_group(sg, mt, sycl::maximum<float>());
                        const float m_old = sM[r];
                        const float m_new = sycl::fmax(m_old, mt);
                        float       lt    = 0.0f;
#pragma unroll
                        for (int i = 0; i < KG; ++i) {
                            const float pv = sycl::native::exp(sv[i] - m_new);
                            sPs[r * PS + i * 16 + lane] = (short) sycl::bit_cast<uint16_t>(sycl::half(pv));
                            lt += pv;
                        }
                        lt = sycl::reduce_over_group(sg, lt, sycl::plus<float>());
                        if (lane == 0) {
                            const float a = sycl::native::exp(m_old - m_new);
                            sA[r] = a;
                            sL[r] = sL[r] * a + lt;
                            sM[r] = m_new;
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    prefetch_rows(Kh, p.nb11, t0 + TK, RWK);

                    // O^T = diag(alpha) O^T + V^T P^T. A = V^T (8 dims x 16 keys, lane = key), B = P^T (16 keys x 16 rows)
#pragma unroll
                    for (int t = 0; t < NR16; ++t) {
                        const float a = sA[t * 16 + lane];
#pragma unroll
                        for (int x = 0; x < XT; ++x) {
                            o[t][x] *= a;
                        }
                    }
#pragma unroll 1
                    for (int kg = 0; kg < ((ALLPV || w < NPV) ? KG : 0); ++kg) {
                        const int  j  = t0 + kg * 16 + lane;
                        const bool jv = j < k_end;
                        const char *     vbp = Vh + (int64_t) (jv ? j : k_begin) * p.nb21;
                        const int        vp  = (ST || !ODD) ? 0 : (int) (((uintptr_t) vbp >> 1) & 1);
                        const uint32_t * vr  = (const uint32_t *) (vbp - 2 * vp);
                        uint32_t wv[NQV + 1];
                        uint32_t sc;
                        uint32_t q[NQV];
                        fa_dpas_load_block<BBV, NQV, SW, ST>(sKVp, vr, vp, kg * 16 + lane, vb, wv);
                        fa_dpas_split_block<NQV>(wv, (vb + vp) % 2 != 0, sc, q);
                        const float dv = jv ? static_cast<float>(sycl::bit_cast<sycl::half>((uint16_t) sc)) : 0.0f;
                        fa_dpas_short8 va[XT];
#pragma unroll
                        for (int x = 0; x < XT; ++x) {
#pragma unroll
                            for (int m = 0; m < 8; ++m) {
                                const int d = vd0 + x * 8 + m;
                                int       v;
                                if constexpr (Q8V) {
                                    v = (int8_t) ((q[d / 4] >> (8 * (d % 4))) & 0xFF);
                                } else {
                                    const int byte = d % 16;
                                    v = (int) ((q[byte / 4] >> (8 * (byte % 4) + 4 * (d / 16))) & 0xF) - 8;
                                }
                                va[x][m] = (short) sycl::bit_cast<uint16_t>(sycl::half((float) v * dv));
                            }
                        }
#pragma unroll
                        for (int t = 0; t < NR16; ++t) {
                            const fa_dpas_int8 pb = sP[((t * 16 + lane) * PS + kg * 16) / 16];
#pragma unroll
                            for (int x = 0; x < XT; ++x) {
                                fa_dpas_f16(va[x], pb, o[t][x]);
                            }
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }

#pragma unroll
                for (int t = 0; t < NR16; ++t) {
                    const int r = t * 16 + lane;
                    const int c = c0 + r / G;
                    if ((ALLPV || w < NPV) && r < R && c < p.ne01) {
                        const int64_t jdu = ((int64_t) seq * p.ne01 + c) * p.ne02 + kvh * G + r % G;
                        const float   l   = sL[r];
#pragma unroll
                        for (int x = 0; x < XT; ++x) {
#pragma unroll
                            for (int m = 0; m < 8; ++m) {
                                const int d = w * DPS + x * 8 + m;
                                if (p.nsplit == 1) {
                                    p.dst[jdu * D + d] = l > 0.0f ? o[t][x][m] / l : 0.0f;
                                } else {
                                    p.parts[(jdu * p.nsplit + split) * D + d] = o[t][x][m];
                                }
                            }
                        }
                    }
                }
                if (p.nsplit > 1 && tid < R && c0 + tid / G < p.ne01) {
                    const int64_t jdu = ((int64_t) seq * p.ne01 + c0 + tid / G) * p.ne02 + kvh * G + tid % G;
                    p.meta[jdu * p.nsplit + split] = sycl::float2(sM[tid], sL[tid]);
                }
            };
        if constexpr (LG) {
            cgh.parallel_for(ndr, sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> }, kern);
        } else {
            cgh.parallel_for(ndr, kern);
        }
    });
}

template <int D, int NR16, int KG, int TC, int ST, bool Q8K, bool Q8V>
static void fattn_dpas_case(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    constexpr int TK = 16 * KG;

    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const int ne01 = Q->ne[1];
    const int ne11 = K->ne[1];
    const int nkvh = K->ne[2];
    const int ne03 = Q->ne[3];

    fa_dpas_params p;
    p.Q    = (const char *) Q->data;
    p.K    = (const char *) K->data;
    p.V    = (const char *) V->data;
    p.mask = mask ? (const char *) mask->data : nullptr;
    p.sinks = dst->src[4] ? (const float *) dst->src[4]->data : nullptr;
    p.dst  = (float *) dst->data;
    memcpy(&p.scale, (const float *) dst->op_params + 0, sizeof(float));
    p.G    = Q->ne[2] / K->ne[2];
    p.nq   = std::max(1, std::min(ne01, 16 * NR16 / p.G));
    p.nqc  = (ne01 + p.nq - 1) / p.nq;
    p.ne01 = ne01;
    p.ne02 = Q->ne[2];
    p.ne11 = ne11;
    p.ne33 = mask ? mask->ne[3] : 1;
    p.nb01 = Q->nb[1];
    p.nb02 = Q->nb[2];
    p.nb03 = Q->nb[3];
    p.nb11 = K->nb[1];
    p.nb12 = K->nb[2];
    p.nb13 = K->nb[3];
    p.nb21 = V->nb[1];
    p.nb22 = V->nb[2];
    p.nb23 = V->nb[3];
    p.nb31 = mask ? mask->nb[1] : 0;
    p.nb33 = mask ? mask->nb[3] : 0;

    // split the keys so that all work-groups run in one wave
    // SLM fits about 4 work-groups per Xe2 core with 16 rows, 2 with 32. Xe2 core = 8 EUs, nsm = EUs / 16.
    const int xe_cores = std::max(1, ggml_sycl_info().devices[ctx.device].nsm * 2);
    const int wave     = xe_cores * (NR16 == 1 ? 4 : 2);
    const int ntiles   = (ne11 + TK - 1) / TK;
    int nsplit = std::max(1, std::min(wave / (nkvh * ne03 * p.nqc), ntiles));
    p.chunk    = ((ntiles + nsplit - 1) / nsplit) * TK;
    nsplit     = (ne11 + p.chunk - 1) / p.chunk;
    p.nsplit   = nsplit;

    ggml_sycl_pool_alloc<float>        parts(ctx.pool());
    ggml_sycl_pool_alloc<sycl::float2> meta(ctx.pool());
    if (nsplit > 1) {
        parts.alloc((size_t) nsplit * ggml_nelements(dst));
        meta.alloc((size_t) nsplit * ggml_nrows(dst));
    }
    p.parts = parts.get();
    p.meta  = meta.get();

    dpct::queue_ptr stream = ctx.stream();
    // the large register file pays off for 32-row launches from about 4K keys
    if (NR16 == 2 && ne11 >= 4096) {
        fattn_dpas_kernel<D, NR16, KG, TC, ST, true, Q8K, Q8V>(p, nkvh, ne03, stream);
    } else {
        fattn_dpas_kernel<D, NR16, KG, TC, ST, false, Q8K, Q8V>(p, nkvh, ne03, stream);
    }

    if (nsplit > 1) {
        float *        parts_p = parts.get();
        sycl::float2 * meta_p  = meta.get();
        float *        dst_p   = (float *) dst->data;
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<uint8_t, 1> lm(sycl::range<1>(nsplit * sizeof(sycl::float2)), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(Q->ne[3], Q->ne[2], (size_t) ne01 * D), sycl::range<3>(1, 1, D)),
                [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(16)]] {
                    flash_attn_combine_results<D>(parts_p, meta_p, dst_p, nsplit,
                                                  lm.get_multi_ptr<sycl::access::decorated::no>().get());
                });
        });
    }
}

template <bool Q8K, bool Q8V> static void fattn_dpas_dispatch(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const int D    = K->ne[0];
    const int G    = Q->ne[2] / K->ne[2];
    const int ne01 = Q->ne[1];

    bool aligned16 = true;
    for (const ggml_tensor * t : { K, V }) {
        aligned16 = aligned16 && t->nb[1] % 16 == 0 && t->nb[2] % 16 == 0 && t->nb[3] % 16 == 0 &&
                    (uintptr_t) t->data % 16 == 0;
    }

    // single token decode: staged K/V tiles are faster at long context
    if (D == 256 && ne01 == 1 && G <= 16 && aligned16) {
        fattn_dpas_case<256, 1, 8, 1, 1, Q8K, Q8V>(ctx, dst);
        return;
    }
    const bool rows16 = ne01 * G <= 16;
    // 256-key tiles for one row tile; 128 at head size 512 so S and the Q copy stay in SLM for the planned waves
#define FATTN_DPAS_D(D_, KG16_)                                                                        \
    case D_:                                                                                           \
        if (rows16) {                                                                                  \
            fattn_dpas_case<D_, 1, KG16_, 1, 0, Q8K, Q8V>(ctx, dst);                               \
        } else {                                                                                       \
            fattn_dpas_case<D_, 2, 8, 2, 0, Q8K, Q8V>(ctx, dst);                                   \
        }                                                                                              \
        break;
    // q8_0 K at head size 128: its 136-byte rows gather poorly per lane, stage them in SLM by coalesced dword loads
    bool aligned4 = true;
    for (const ggml_tensor * t : { K, V }) {
        aligned4 = aligned4 && t->nb[1] % 4 == 0 && t->nb[2] % 4 == 0 && t->nb[3] % 4 == 0 && (uintptr_t) t->data % 4 == 0;
    }
    if (Q8K && D == 128 && aligned4) {
        if (rows16) {
            fattn_dpas_case<128, 1, 16, 1, 2, Q8K, Q8V>(ctx, dst);
        } else {
            fattn_dpas_case<128, 2, 8, 2, 2, Q8K, Q8V>(ctx, dst);
        }
        return;
    }
    switch (D) {
        FATTN_DPAS_D( 64, 16)
        FATTN_DPAS_D( 96, 16)
        FATTN_DPAS_D(128, 16)
        FATTN_DPAS_D(256, 16)
        FATTN_DPAS_D(512, 8)
        default:
            GGML_ABORT("fattn-dpas: unsupported head size %d", D);
    }
#undef FATTN_DPAS_D
}

#endif // GGML_SYCL_FA_DPAS_BUILD

#if GGML_SYCL_FA_DPAS_SG8_BUILD

// Xe1 DPAS (SIMD8, systolic depth 8, repeat count 8) as inline vISA: the forms the stage A probe
// verified on ACM, with the probe's operand layouts unchanged.
// d = a x b. a: 8 rows x 32 s8, 4 per lane per row. b: 32 x 8 s8, 32 bytes per lane, one column per lane.
static inline void fa_dpas8_s8_s8(const fa_dpas_int8 & a, const fa_dpas_int8 & b, fa_dpas_int8 & d) {
#ifdef __SYCL_DEVICE_ONLY__
    __asm__("{\n"
            ".decl DB v_type=G type=ud num_elts=64 align=GRF alias=<%1,0>\n"
            ".decl DA v_type=G type=ud num_elts=64 align=GRF alias=<%2,0>\n"
            "dpas.s8.s8.8.8 (M1, 8) %0.0 %%null.0 DB.0 DA(0,0)\n"
            "}\n"
            : "=rw"(d)
            : "rw"(b), "rw"(a));
#else
    d = 0;
    GGML_UNUSED(a);
    GGML_UNUSED(b);
#endif
}

// acc += a x b. a: 8 rows x 16 f16, 2 per lane per row (keys 2 l and 2 l + 1, one dword per row).
// b: 16 x 8 f16, one column per lane (the column's 16 key values, 8 dwords). The contraction is
// 16 f16 per instruction at any exec size (systolic depth 8 x 2), as the stage A fix established.
static inline void fa_dpas8_f16(const fa_dpas_int8 & a, const fa_dpas_int8 & b, fa_dpas_float8 & acc) {
#ifdef __SYCL_DEVICE_ONLY__
    __asm__("{\n"
            ".decl DB v_type=G type=ud num_elts=64 align=GRF alias=<%1,0>\n"
            ".decl DA v_type=G type=ud num_elts=64 align=GRF alias=<%2,0>\n"
            "dpas.hf.hf.8.8 (M1, 8) %0.0 %0.0 DB.0 DA(0,0)\n"
            "}\n"
            : "+rw"(acc)
            : "rw"(b), "rw"(a));
#else
    GGML_UNUSED(a);
    GGML_UNUSED(b);
    GGML_UNUSED(acc);
#endif
}

// The sg8 twin of fattn_dpas_kernel for the staged decode case, duplicated on purpose (pass 6
// section 3): the Xe2 body above and its object code stay untouched. One work-group = one KV head
// x nq query tokens x one slice of the keys: S = Q K^T by s8 x s8 DPAS on the raw K blocks, online
// softmax, O^T += V^T P^T by f16 DPAS, all at sub-group 8. ST: stage K/V tiles in SLM (1: 16-byte
// loads), the only mode v1 instantiates. Q8V: V is q8_0, else q4_0; K is always q8_0.
template <int D, int NR16, int KG, int TC, int ST, bool Q8V>
static void fattn_dpas_kernel_sg8(const fa_dpas_params p, const int nkvh, const int ne03, dpct::queue_ptr stream) {
    constexpr int RMAX   = 16 * NR16;
    constexpr int RTMAX  = RMAX / 8;                    // 8-row tiles in Q.K and in P.V
    constexpr int TK     = 16 * KG;                     // keys per tile
    constexpr int KG8    = TK / 8;                      // key groups of 8 in Q.K and the softmax
    static_assert(QK4_0 == QK8_0, "q4_0 and q8_0 blocks hold the same number of dims");
    constexpr int BBK    = (int) sizeof(block_q8_0);    // bytes per K block (K is q8_0 in v1)
    constexpr int BBV    = Q8V ? (int) sizeof(block_q8_0) : (int) sizeof(block_q4_0); // bytes per V block
    constexpr int NQK    = (BBK - 2) / 4;               // quant dwords per K block
    constexpr int NQV    = (BBV - 2) / 4;               // quant dwords per V block
    constexpr int NB     = D / QK4_0;                   // blocks per row
    constexpr int DPS    = 8;                           // head dims per sub-group in P.V
    constexpr int NPV    = D / DPS;                     // sub-groups doing P.V, the rest only reach its barrier
    constexpr bool ALLPV = NPV == FA_DPAS_SG8_NSG;      // every sub-group does P.V: no runtime check in the loops
    constexpr int XT     = DPS / 8;                     // 8-dim tiles per sub-group in P.V
    constexpr int PS     = TK + 16;                     // f16 row stride of P
    constexpr int RWK    = NB * BBK / 4;                // dwords per K row
    constexpr int RWV    = NB * BBV / 4;                // dwords per V row
    constexpr bool ODD   = (NB * BBK) % 4 != 0;         // odd block count: rows may start 2 bytes into a dword (same for V: both block sizes are 2 mod 4)
    constexpr int SW     = (RWK > RWV ? RWK : RWV) + 1; // odd SLM row stride: no bank conflicts for lane = key
    static_assert(D % QK4_0 == 0 && D % DPS == 0 && NPV <= FA_DPAS_SG8_NSG && QK4_0 % DPS == 0, "unsupported head size");
    static_assert(ST != 1 || (RWK % 4 == 0 && RWV % 4 == 0), "16-byte staging needs rows of 16-byte multiples");

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<fa_dpas_int8, 1>   sQ8(sycl::range<1>(RTMAX * NB * 8), cgh); // Q s8 [tile][block][lane]
        sycl::local_accessor<fa_dpas_float8, 1> sDQ(sycl::range<1>(RTMAX * NB), cgh);    // Q scales [tile][block]
        sycl::local_accessor<float, 1>          sS(sycl::range<1>(RMAX * TK), cgh);      // S [row][key]
        sycl::local_accessor<fa_dpas_int8, 1>   sP(sycl::range<1>(RMAX * PS / 16), cgh);  // P f16 [row][key]
        sycl::local_accessor<float, 1>          sM(sycl::range<1>(RMAX), cgh);
        sycl::local_accessor<float, 1>          sL(sycl::range<1>(RMAX), cgh);
        sycl::local_accessor<float, 1>          sA(sycl::range<1>(RMAX), cgh);
        sycl::local_accessor<uint32_t, 1>       sKV(sycl::range<1>(ST ? TK * SW : 1), cgh);
        const sycl::nd_range<3> ndr(sycl::range<3>((size_t) ne03 * p.nqc, nkvh, (size_t) p.nsplit * FA_DPAS_WG),
                                    sycl::range<3>(1, 1, FA_DPAS_WG));
        auto kern = [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(8)]] {
                const int tid   = it.get_local_id(2);
                const int split = it.get_group(2);
                const int kvh   = it.get_group(1);
                const int seq   = it.get_group(0) / p.nqc;
                const int c0    = (it.get_group(0) % p.nqc) * p.nq;
                const int G     = p.G;
                const int R     = p.nq * G;
                const int RT    = (R + 7) / 8;
                auto      sg    = it.get_sub_group();
                const int w     = tid / 8;
                const int lane  = tid % 8;
                int *     sQi   = (int *) &sQ8[0];
                short *   sPs   = (short *) &sP[0];
                uint32_t * sKVp = &sKV[0];

                // Q -> s8, one scale per row and block. Lane l holds the 4 dims 4 l .. 4 l + 3 of each
                // row, one dword per row (q8_0 stores dims in order).
                for (int e = tid; e < RT * 8 * NB; e += FA_DPAS_WG) {
                    const int  r    = e / NB;
                    const int  b    = e % NB;
                    const int  c    = c0 + r / G;
                    const bool real = r < R && c < p.ne01;
                    const float * qr = (const float *) (p.Q + seq * p.nb03 + (int64_t) (real ? c : 0) * p.nb01 +
                                                        (int64_t) (kvh * G + (real ? r % G : 0)) * p.nb02) + b * QK4_0;
                    float x[QK4_0];
                    float amax = 0.0f;
#pragma unroll
                    for (int i = 0; i < QK4_0; ++i) {
                        x[i] = real ? qr[i] * p.scale : 0.0f;
                        amax = sycl::fmax(amax, sycl::fabs(x[i]));
                    }
                    const float dq = amax / 127.0f;
                    const float iq = amax > 0.0f ? 127.0f / amax : 0.0f;
                    const int   t  = r / 8;
                    const int   m  = r % 8;
                    ((float *) &sDQ[0])[(t * NB + b) * 8 + m] = dq;
#pragma unroll
                    for (int l = 0; l < 8; ++l) {
                        uint32_t pk = 0;
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            pk |= (uint32_t) (((int) sycl::rint(x[4 * l + j] * iq)) & 0xFF) << (8 * j);
                        }
                        sQi[((t * NB + b) * 8 + l) * 8 + m] = (int) pk;
                    }
                }
                // rows without a query keep P = 0
                for (int e = tid; e < RMAX * PS; e += FA_DPAS_WG) {
                    sPs[e] = 0;
                }
                if (tid < RMAX) {
                    // the sink is one more key with logit sinks[head] and no value; the first key slice owns it
                    const bool sink = p.sinks && split == 0 && tid < R;
                    sM[tid] = sink ? p.sinks[kvh * G + tid % G] : FA_DPAS_SG8_MINF;
                    sL[tid] = sink ? 1.0f : 0.0f;
                    sA[tid] = 1.0f;
                }
                it.barrier(sycl::access::fence_space::local_space);

                const int    k_begin = split * p.chunk;
                const int    k_end   = sycl::min(k_begin + p.chunk, p.ne11);
                const char * Kh      = p.K + seq * p.nb13 + kvh * p.nb12;
                const char * Vh      = p.V + seq * p.nb23 + kvh * p.nb22;
                const sycl::half * mrow = p.mask ? (const sycl::half *) (p.mask + (seq % p.ne33) * p.nb33) : nullptr;

                // O^T: lane = query row, element m = head dim
                fa_dpas_float8 o[RTMAX][XT];
#pragma unroll
                for (int t = 0; t < RTMAX; ++t) {
#pragma unroll
                    for (int x = 0; x < XT; ++x) {
                        o[t][x] = 0.0f;
                    }
                }
                // the dims of this sub-group lie in one V block
                const int vb  = (DPS * w) / QK4_0;
                const int vd0 = (DPS * w) % QK4_0;

                // start loading the rows of keys t0 .. t0 + TK - 1 into L1 while the current phase computes
                auto prefetch_rows = [&](const char * base, int64_t nb, int t0, int rw) {
                    const int NL = rw * 4 / 64 + 1; // cache lines per row
                    for (int e = tid; e < TK * NL; e += FA_DPAS_WG) {
                        const int j = t0 + e / NL;
                        if (j < k_end) {
                            char * pa = const_cast<char *>(base) + (int64_t) j * nb + (e % NL) * 64;
                            sycl::ext::oneapi::experimental::prefetch(pa, sycl::ext::oneapi::experimental::properties{ sycl::ext::oneapi::experimental::prefetch_hint_L1 });
                        }
                    }
                };

                for (int t0 = k_begin; t0 < k_end; t0 += TK) {
                    if constexpr (ST) {
                        fa_dpas_stage<TK, SW, ST, RWK, FA_DPAS_WG>(sKVp, tid, Kh, p.nb11, t0, k_begin, k_end);
                        it.barrier(sycl::access::fence_space::local_space);
                    }
                    prefetch_rows(Vh, p.nb21, t0, RWV);

                    // S = Q K^T, job = (key group of 8, TC row tiles)
                    const int nch = (RT + TC - 1) / TC;
                    for (int job = w; job < KG8 * nch; job += FA_DPAS_SG8_NSG) {
                        const int  kg = job % KG8;
                        const int  ch = job / KG8;
                        const int  j  = t0 + kg * 8 + lane;
                        const bool jv = j < k_end;
                        const char *     kb = Kh + (int64_t) (jv ? j : k_begin) * p.nb11;
                        const int        kp = (ST || !ODD) ? 0 : (int) (((uintptr_t) kb >> 1) & 1); // row starts 2 bytes into a dword
                        const uint32_t * kr = (const uint32_t *) (kb - 2 * kp);
                        fa_dpas_float8 s[TC];
#pragma unroll
                        for (int tt = 0; tt < TC; ++tt) {
                            s[tt] = 0.0f;
                        }
#pragma unroll
                        for (int b = 0; b < NB; ++b) {
                            uint32_t wv[NQK + 1];
                            uint32_t sc;
                            uint32_t q[NQK];
                            fa_dpas_load_block<BBK, NQK, SW, ST>(sKVp, kr, kp, kg * 8 + lane, b, wv);
                            fa_dpas_split_block<NQK>(wv, (b + kp) % 2 != 0, sc, q);
                            const float dk = static_cast<float>(sycl::bit_cast<sycl::half>((uint16_t) sc));
#pragma unroll
                            for (int tt = 0; tt < TC; ++tt) {
                                const int t = ch * TC + tt;
                                if (t < RT) {
                                    const fa_dpas_int8 bv = { (int) q[0], (int) q[1], (int) q[2], (int) q[3],
                                                              (int) q[4], (int) q[5], (int) q[6], (int) q[7] };
                                    fa_dpas_int8 ia;
                                    fa_dpas8_s8_s8(sQ8[(t * NB + b) * 8 + lane], bv, ia);
                                    const fa_dpas_float8 ds = sDQ[t * NB + b] * dk;
                                    s[tt] += __builtin_convertvector(ia, fa_dpas_float8) * ds;
                                }
                            }
                        }
#pragma unroll
                        for (int tt = 0; tt < TC; ++tt) {
                            const int t = ch * TC + tt;
                            if (t < RT) {
#pragma unroll
                                for (int m = 0; m < 8; ++m) {
                                    const int r = t * 8 + m;
                                    const int c = c0 + r / G;
                                    if (r < R) {
                                        const bool  ok = jv && c < p.ne01;
                                        const float mv = (ok && mrow) ? static_cast<float>(mrow[((int64_t) c * p.nb31) / (int64_t) sizeof(sycl::half) + j]) : 0.0f;
                                        sS[r * TK + kg * 8 + lane] = ok ? s[tt][m] + mv : -INFINITY;
                                    }
                                }
                            }
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                    if constexpr (ST) {
                        fa_dpas_stage<TK, SW, ST, RWV, FA_DPAS_WG>(sKVp, tid, Vh, p.nb21, t0, k_begin, k_end); // the softmax does not use sKV
                    }

                    // online softmax per row -> P
                    for (int r = w; r < R; r += FA_DPAS_SG8_NSG) {
                        float sv[KG8];
                        float mt = -INFINITY;
#pragma unroll
                        for (int i = 0; i < KG8; ++i) {
                            sv[i] = sS[r * TK + i * 8 + lane];
                            mt    = sycl::fmax(mt, sv[i]);
                        }
                        mt = sycl::reduce_over_group(sg, mt, sycl::maximum<float>());
                        const float m_old = sM[r];
                        const float m_new = sycl::fmax(m_old, mt);
                        float       lt    = 0.0f;
#pragma unroll
                        for (int i = 0; i < KG8; ++i) {
                            const float pv = sycl::native::exp(sv[i] - m_new);
                            sPs[r * PS + i * 8 + lane] = (short) sycl::bit_cast<uint16_t>(sycl::half(pv));
                            lt += pv;
                        }
                        lt = sycl::reduce_over_group(sg, lt, sycl::plus<float>());
                        if (lane == 0) {
                            const float a = sycl::native::exp(m_old - m_new);
                            sA[r] = a;
                            sL[r] = sL[r] * a + lt;
                            sM[r] = m_new;
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    prefetch_rows(Kh, p.nb11, t0 + TK, RWK);

                    // O^T = diag(alpha) O^T + V^T P^T. A = V^T (8 dims x 16 keys, lane holds the key pair
                    // 2 l, 2 l + 1 of the group), B = P^T (16 keys x 8 rows)
#pragma unroll
                    for (int t = 0; t < RTMAX; ++t) {
                        const float a = sA[t * 8 + lane];
#pragma unroll
                        for (int x = 0; x < XT; ++x) {
                            o[t][x] *= a;
                        }
                    }
#pragma unroll 1
                    for (int kg = 0; kg < ((ALLPV || w < NPV) ? KG : 0); ++kg) {
                        // dequantize this lane's two keys of the 16-key group, one V block each
                        uint32_t qw[2][NQV];
                        float    dvv[2];
#pragma unroll
                        for (int h = 0; h < 2; ++h) {
                            const int  kk = kg * 16 + 2 * lane + h;
                            const int  j  = t0 + kk;
                            const bool jv = j < k_end;
                            const char *     vbp = Vh + (int64_t) (jv ? j : k_begin) * p.nb21;
                            const int        vp  = (ST || !ODD) ? 0 : (int) (((uintptr_t) vbp >> 1) & 1);
                            const uint32_t * vr  = (const uint32_t *) (vbp - 2 * vp);
                            uint32_t wv[NQV + 1];
                            uint32_t sc;
                            fa_dpas_load_block<BBV, NQV, SW, ST>(sKVp, vr, vp, kk, vb, wv);
                            fa_dpas_split_block<NQV>(wv, (vb + vp) % 2 != 0, sc, qw[h]);
                            dvv[h] = jv ? static_cast<float>(sycl::bit_cast<sycl::half>((uint16_t) sc)) : 0.0f;
                        }
                        fa_dpas_int8 va[XT];
#pragma unroll
                        for (int x = 0; x < XT; ++x) {
#pragma unroll
                            for (int m = 0; m < 8; ++m) {
                                const int d  = vd0 + x * 8 + m;
                                uint32_t  pk = 0;
#pragma unroll
                                for (int h = 0; h < 2; ++h) {
                                    int v;
                                    if constexpr (Q8V) {
                                        v = (int8_t) ((qw[h][d / 4] >> (8 * (d % 4))) & 0xFF);
                                    } else {
                                        const int byte = d % 16;
                                        v = (int) ((qw[h][byte / 4] >> (8 * (byte % 4) + 4 * (d / 16))) & 0xF) - 8;
                                    }
                                    pk |= (uint32_t) sycl::bit_cast<uint16_t>(sycl::half((float) v * dvv[h])) << (16 * h);
                                }
                                va[x][m] = (int) pk;
                            }
                        }
#pragma unroll
                        for (int t = 0; t < RTMAX; ++t) {
                            const fa_dpas_int8 pb = sP[((t * 8 + lane) * PS + kg * 16) / 16];
#pragma unroll
                            for (int x = 0; x < XT; ++x) {
                                fa_dpas8_f16(va[x], pb, o[t][x]);
                            }
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }

#pragma unroll
                for (int t = 0; t < RTMAX; ++t) {
                    const int r = t * 8 + lane;
                    const int c = c0 + r / G;
                    if ((ALLPV || w < NPV) && r < R && c < p.ne01) {
                        const int64_t jdu = ((int64_t) seq * p.ne01 + c) * p.ne02 + kvh * G + r % G;
                        const float   l   = sL[r];
#pragma unroll
                        for (int x = 0; x < XT; ++x) {
#pragma unroll
                            for (int m = 0; m < 8; ++m) {
                                const int d = w * DPS + x * 8 + m;
                                if (p.nsplit == 1) {
                                    p.dst[jdu * D + d] = l > 0.0f ? o[t][x][m] / l : 0.0f;
                                } else {
                                    p.parts[(jdu * p.nsplit + split) * D + d] = o[t][x][m];
                                }
                            }
                        }
                    }
                }
                if (p.nsplit > 1 && tid < R && c0 + tid / G < p.ne01) {
                    const int64_t jdu = ((int64_t) seq * p.ne01 + c0 + tid / G) * p.ne02 + kvh * G + tid % G;
                    p.meta[jdu * p.nsplit + split] = sycl::float2(sM[tid], sL[tid]);
                }
            };
        cgh.parallel_for(ndr, kern);
    });
}

template <bool Q8V> static void fattn_dpas_case_sg8(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    constexpr int D    = 256;
    constexpr int NR16 = 1;
    constexpr int KG   = 8;
    constexpr int TK   = 16 * KG;

    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const int ne01 = Q->ne[1];
    const int ne11 = K->ne[1];
    const int nkvh = K->ne[2];
    const int ne03 = Q->ne[3];

    fa_dpas_params p;
    p.Q    = (const char *) Q->data;
    p.K    = (const char *) K->data;
    p.V    = (const char *) V->data;
    p.mask = mask ? (const char *) mask->data : nullptr;
    p.sinks = dst->src[4] ? (const float *) dst->src[4]->data : nullptr;
    p.dst  = (float *) dst->data;
    memcpy(&p.scale, (const float *) dst->op_params + 0, sizeof(float));
    p.G    = Q->ne[2] / K->ne[2];
    p.nq   = std::max(1, std::min(ne01, 16 * NR16 / p.G));
    p.nqc  = (ne01 + p.nq - 1) / p.nq;
    p.ne01 = ne01;
    p.ne02 = Q->ne[2];
    p.ne11 = ne11;
    p.ne33 = mask ? mask->ne[3] : 1;
    p.nb01 = Q->nb[1];
    p.nb02 = Q->nb[2];
    p.nb03 = Q->nb[3];
    p.nb11 = K->nb[1];
    p.nb12 = K->nb[2];
    p.nb13 = K->nb[3];
    p.nb21 = V->nb[1];
    p.nb22 = V->nb[2];
    p.nb23 = V->nb[3];
    p.nb31 = mask ? mask->nb[1] : 0;
    p.nb33 = mask ? mask->nb[3] : 0;

    // split the keys so that all work-groups run in one wave (the Xe2 heuristic, unchanged in v1)
    const int xe_cores = std::max(1, ggml_sycl_info().devices[ctx.device].nsm * 2);
    const int wave     = xe_cores * (NR16 == 1 ? 4 : 2);
    const int ntiles   = (ne11 + TK - 1) / TK;
    int nsplit = std::max(1, std::min(wave / (nkvh * ne03 * p.nqc), ntiles));
    p.chunk    = ((ntiles + nsplit - 1) / nsplit) * TK;
    nsplit     = (ne11 + p.chunk - 1) / p.chunk;
    p.nsplit   = nsplit;

    ggml_sycl_pool_alloc<float>        parts(ctx.pool());
    ggml_sycl_pool_alloc<sycl::float2> meta(ctx.pool());
    if (nsplit > 1) {
        parts.alloc((size_t) nsplit * ggml_nelements(dst));
        meta.alloc((size_t) nsplit * ggml_nrows(dst));
    }
    p.parts = parts.get();
    p.meta  = meta.get();

    dpct::queue_ptr stream = ctx.stream();
    fattn_dpas_kernel_sg8<D, NR16, KG, 1, 1, Q8V>(p, nkvh, ne03, stream);

    if (nsplit > 1) {
        float *        parts_p = parts.get();
        sycl::float2 * meta_p  = meta.get();
        float *        dst_p   = (float *) dst->data;
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<uint8_t, 1> lm(sycl::range<1>(nsplit * sizeof(sycl::float2)), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(Q->ne[3], Q->ne[2], (size_t) ne01 * D), sycl::range<3>(1, 1, D)),
                [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(8)]] {
                    flash_attn_combine_results<D>(parts_p, meta_p, dst_p, nsplit,
                                                  lm.get_multi_ptr<sycl::access::decorated::no>().get());
                });
        });
    }
}

static void fattn_dpas_sg8_dispatch(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    bool aligned16 = true;
    for (const ggml_tensor * t : { K, V }) {
        aligned16 = aligned16 && t->nb[1] % 16 == 0 && t->nb[2] % 16 == 0 && t->nb[3] % 16 == 0 &&
                    (uintptr_t) t->data % 16 == 0;
    }
    // supported() has narrowed ACM to this case; anything else reaching here is a wiring bug
    if (K->ne[0] != 256 || Q->ne[1] != 1 || Q->ne[2] / K->ne[2] > 16 || !aligned16) {
        GGML_ABORT("fattn-dpas sg8: shape outside the v1 staged decode case");
    }
    if (V->type == GGML_TYPE_Q8_0) {
        fattn_dpas_case_sg8<true>(ctx, dst);
    } else {
        fattn_dpas_case_sg8<false>(ctx, dst);
    }
}

#endif // GGML_SYCL_FA_DPAS_SG8_BUILD

void ggml_sycl_flash_attn_ext_dpas(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
#if GGML_SYCL_FA_DPAS_SG8_BUILD
    if (fattn_dpas_sg8_support_arch(ggml_sycl_info().devices[ctx.device].hw_info.arch)) {
        fattn_dpas_sg8_dispatch(ctx, dst);
        return;
    }
#endif
#if GGML_SYCL_FA_DPAS_BUILD
    const bool q8k = dst->src[1]->type == GGML_TYPE_Q8_0;
    const bool q8v = dst->src[2]->type == GGML_TYPE_Q8_0;
    if (q8k && q8v) {
        fattn_dpas_dispatch<true, true>(ctx, dst);
    } else if (q8k) {
        fattn_dpas_dispatch<true, false>(ctx, dst);
    } else if (!q8v) {
        fattn_dpas_dispatch<false, false>(ctx, dst);
    } else {
        GGML_ABORT("fattn-dpas: unsupported K/V type combination (K q4_0, V q8_0)");
    }
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("fattn-dpas: not built for this target");
#endif
}

// Stage A probe (XMX remedy 7): one launch that runs the (M1, 8) DPAS forms of the planned sg8
// kernel on fixed inputs and checks the results against a scalar recompute in the kernel. It only
// observes: FA dispatch is unchanged and supported() stays false on ACM while stage A runs.
// Variant full stages the operands in SLM and barriers like the real kernel; bare reads global
// memory and skips the barriers, so a wedge in full alone points at the staging rhythm.
#if GGML_SYCL_FA_SG8_PROBE_BUILD

typedef short fa_probe_short8 __attribute__((ext_vector_type(8)));
typedef int   fa_probe_int4   __attribute__((ext_vector_type(4)));
typedef int   fa_probe_int8   __attribute__((ext_vector_type(8)));
typedef float fa_probe_float8 __attribute__((ext_vector_type(8)));

// d = a x b at (M1, 8). a: 8 rows x 32 s8, 4 per lane per row (dims 4 l .. 4 l + 3 in lane l).
// b: 32 x 8 s8, 32 bytes per lane, one column per lane.
static inline void fa_probe_s8_s8(const fa_probe_int8 & a, const fa_probe_int8 & b, fa_probe_int8 & d) {
#ifdef __SYCL_DEVICE_ONLY__
    __asm__("{\n"
            ".decl DB v_type=G type=ud num_elts=64 align=GRF alias=<%1,0>\n"
            ".decl DA v_type=G type=ud num_elts=64 align=GRF alias=<%2,0>\n"
            "dpas.s8.s8.8.8 (M1, 8) %0.0 %%null.0 DB.0 DA(0,0)\n"
            "}\n"
            : "=rw"(d)
            : "rw"(b), "rw"(a));
#else
    d = 0;
    GGML_UNUSED(a);
    GGML_UNUSED(b);
#endif
}

// d = a x b at (M1, 8). a as in fa_probe_s8_s8. b: 32 x 8 s4, 16 bytes per lane, one column per lane.
static inline void fa_probe_s8_s4(const fa_probe_int8 & a, const fa_probe_int4 & b, fa_probe_int8 & d) {
#ifdef __SYCL_DEVICE_ONLY__
    __asm__("{\n"
            ".decl DB v_type=G type=ud num_elts=32 align=GRF alias=<%1,0>\n"
            ".decl DA v_type=G type=ud num_elts=64 align=GRF alias=<%2,0>\n"
            "dpas.s4.s8.8.8 (M1, 8) %0.0 %%null.0 DB.0 DA(0,0)\n"
            "}\n"
            : "=rw"(d)
            : "rw"(b), "rw"(a));
#else
    d = 0;
    GGML_UNUSED(a);
    GGML_UNUSED(b);
#endif
}

// acc += a x b at (M1, 8). The contraction is 16 f16 per instruction at any exec size
// (systolic depth 8 x 2), as in production. a: 8 rows x 16 f16, 2 per lane per row
// (keys 2 l and 2 l + 1 in lane l, one dword per row). b: 16 x 8 f16, one column per
// lane (the column's 16 key values, 8 dwords). Both operands are 8 dwords per lane.
static inline void fa_probe_f16(const fa_probe_int8 & a, const fa_probe_int8 & b, fa_probe_float8 & acc) {
#ifdef __SYCL_DEVICE_ONLY__
    __asm__("{\n"
            ".decl DB v_type=G type=ud num_elts=64 align=GRF alias=<%1,0>\n"
            ".decl DA v_type=G type=ud num_elts=64 align=GRF alias=<%2,0>\n"
            "dpas.hf.hf.8.8 (M1, 8) %0.0 %0.0 DB.0 DA(0,0)\n"
            "}\n"
            : "+rw"(acc)
            : "rw"(b), "rw"(a));
#else
    GGML_UNUSED(a);
    GGML_UNUSED(b);
    GGML_UNUSED(acc);
#endif
}

// operand regions in one int32 buffer: [s8 A | s8_s8 B x IT8 | s8_s4 B x IT4 | f16 A | f16 B x ITHF]
static constexpr int FA_PROBE_OFF_A8  = 0;   // 64 dwords: [lane][row]
static constexpr int FA_PROBE_OFF_B8  = 64;  // IT8 x 64: [iter][column][dword]
static constexpr int FA_PROBE_OFF_B4  = 320; // IT4 x 32: [iter][column][dword]
static constexpr int FA_PROBE_OFF_AHF = 384; // 64 dwords of halves: [key][row], 16 keys
static constexpr int FA_PROBE_OFF_BHF = 448; // ITHF x 64: [iter][column][key] halves, 16 keys
static constexpr int FA_PROBE_NIN     = 576;
static constexpr int FA_PROBE_IT8     = 4;
static constexpr int FA_PROBE_IT4     = 2;
static constexpr int FA_PROBE_ITHF    = 2;
static constexpr int FA_PROBE_WG      = 256; // 32 sub-groups of 8 lanes, as the sg8 kernel
static constexpr int FA_PROBE_NRES    = 3 * 64 + 3; // per-form 8 x 8 results, then per-form checksums

// fixed input patterns; small values, so every product and sum is exact in int32 and in f32
static int   fa_probe_a8_val(int m, int d)            { return (m * 7 + d * 3) % 5 - 2; }
static int   fa_probe_b8_val(int i, int d, int c)     { return (d * 5 + c * 11 + i * 13) % 7 - 3; }
static int   fa_probe_b4_val(int i, int d, int c)     { return (d * 3 + c * 7 + i * 5) % 9 - 4; }
static float fa_probe_ahf_val(int m, int k)           { return ((m * 3 + k * 7) % 5 - 2) * 0.5f; }
static float fa_probe_bhf_val(int i, int k, int c)    { return ((k * 7 + c * 3 + i * 2) % 5 - 2) * 0.5f; }

static void fa_probe_fill(int32_t * in) {
    for (int i = 0; i < FA_PROBE_NIN; ++i) {
        in[i] = 0;
    }
    for (int l = 0; l < 8; ++l) {
        for (int m = 0; m < 8; ++m) {
            uint32_t w = 0;
            for (int j = 0; j < 4; ++j) {
                w |= (uint32_t) (fa_probe_a8_val(m, 4 * l + j) & 0xFF) << (8 * j);
            }
            in[FA_PROBE_OFF_A8 + l * 8 + m] = (int32_t) w;
        }
    }
    for (int i = 0; i < FA_PROBE_IT8; ++i) {
        for (int c = 0; c < 8; ++c) {
            for (int j = 0; j < 8; ++j) {
                uint32_t w = 0;
                for (int q = 0; q < 4; ++q) {
                    w |= (uint32_t) (fa_probe_b8_val(i, 4 * j + q, c) & 0xFF) << (8 * q);
                }
                in[FA_PROBE_OFF_B8 + i * 64 + c * 8 + j] = (int32_t) w;
            }
        }
    }
    // s4 columns: byte b holds dims 2 b (low nibble) and 2 b + 1 (high nibble), as in q4_0 order
    for (int i = 0; i < FA_PROBE_IT4; ++i) {
        for (int c = 0; c < 8; ++c) {
            for (int b = 0; b < 16; ++b) {
                const uint32_t byte = (uint32_t) ((fa_probe_b4_val(i, 2 * b, c) & 0xF) |
                                                  ((fa_probe_b4_val(i, 2 * b + 1, c) & 0xF) << 4));
                in[FA_PROBE_OFF_B4 + i * 32 + c * 4 + b / 4] |= (int32_t) (byte << (8 * (b % 4)));
            }
        }
    }
    for (int k = 0; k < 16; ++k) {
        for (int m = 0; m < 8; ++m) {
            const int idx = k * 8 + m;
            in[FA_PROBE_OFF_AHF + idx / 2] |=
                (int32_t) ((uint32_t) sycl::bit_cast<uint16_t>(sycl::half(fa_probe_ahf_val(m, k))) << (16 * (idx % 2)));
        }
    }
    for (int i = 0; i < FA_PROBE_ITHF; ++i) {
        for (int c = 0; c < 8; ++c) {
            for (int k = 0; k < 16; ++k) {
                const int idx = c * 16 + k;
                in[FA_PROBE_OFF_BHF + i * 64 + idx / 2] |=
                    (int32_t) ((uint32_t) sycl::bit_cast<uint16_t>(sycl::half(fa_probe_bhf_val(i, k, c))) << (16 * (idx % 2)));
            }
        }
    }
}

static void fa_probe_expect(float * exp, float * cksum) {
    for (int f = 0; f < 3; ++f) {
        cksum[f] = 0.0f;
    }
    for (int c = 0; c < 8; ++c) {
        for (int m = 0; m < 8; ++m) {
            int   e8 = 0;
            int   e4 = 0;
            float eh = 0.0f;
            for (int i = 0; i < FA_PROBE_IT8; ++i) {
                for (int d = 0; d < 32; ++d) {
                    e8 += fa_probe_a8_val(m, d) * fa_probe_b8_val(i, d, c);
                }
            }
            for (int i = 0; i < FA_PROBE_IT4; ++i) {
                for (int d = 0; d < 32; ++d) {
                    e4 += fa_probe_a8_val(m, d) * fa_probe_b4_val(i, d, c);
                }
            }
            for (int i = 0; i < FA_PROBE_ITHF; ++i) {
                for (int k = 0; k < 16; ++k) {
                    eh += fa_probe_ahf_val(m, k) * fa_probe_bhf_val(i, k, c);
                }
            }
            exp[0 * 64 + c * 8 + m] = (float) e8;
            exp[1 * 64 + c * 8 + m] = (float) e4;
            exp[2 * 64 + c * 8 + m] = eh;
            cksum[0] += (float) e8;
            cksum[1] += (float) e4;
            cksum[2] += eh;
        }
    }
}

static void fa_probe_submit(const int32_t * in, float * res, int32_t * flag, bool full, dpct::queue_ptr stream) {
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<int32_t, 1> sIn(sycl::range<1>(FA_PROBE_NIN), cgh);
        const sycl::nd_range<1> ndr{ sycl::range<1>(FA_PROBE_WG), sycl::range<1>(FA_PROBE_WG) };
        auto kern = [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(8)]] {
                const int tid  = (int) it.get_local_id(0);
                const int lane = tid % 8;
                const int w    = tid / 8; // sub-group index; every sub-group runs the same forms
                auto      sg   = it.get_sub_group();
                if (full) {
                    for (int e = tid; e < FA_PROBE_NIN; e += FA_PROBE_WG) {
                        sIn[e] = in[e];
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }
                auto rd = [&](int i) -> int32_t { return full ? sIn[i] : in[i]; };
                auto rd_half = [&](int base, int idx) -> uint16_t {
                    const uint32_t wd = (uint32_t) rd(base + idx / 2);
                    return (uint16_t) ((idx % 2) ? (wd >> 16) : (wd & 0xFFFFu));
                };
                auto s8_at = [&](int base, int m, int d) -> int {
                    return (int) (int8_t) (((uint32_t) rd(base + (d / 4) * 8 + m) >> (8 * (d % 4))) & 0xFFu);
                };

                fa_probe_int8 a8;
#pragma unroll
                for (int m = 0; m < 8; ++m) {
                    a8[m] = rd(FA_PROBE_OFF_A8 + lane * 8 + m);
                }

                // s8_s8: the KQ form of the sg8 kernel
                fa_probe_float8 acc8;
                acc8 = 0.0f;
#pragma unroll
                for (int i = 0; i < FA_PROBE_IT8; ++i) {
                    fa_probe_int8 b8;
                    fa_probe_int8 d8;
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        b8[j] = rd(FA_PROBE_OFF_B8 + i * 64 + lane * 8 + j);
                    }
                    fa_probe_s8_s8(a8, b8, d8);
                    acc8 += __builtin_convertvector(d8, fa_probe_float8);
                }
                bool  ok8 = true;
                float ck8 = 0.0f;
#pragma unroll
                for (int m = 0; m < 8; ++m) {
                    int e = 0;
#pragma unroll
                    for (int i = 0; i < FA_PROBE_IT8; ++i) {
#pragma unroll
                        for (int d = 0; d < 32; ++d) {
                            e += s8_at(FA_PROBE_OFF_A8, m, d) *
                                 (int) (int8_t) (((uint32_t) rd(FA_PROBE_OFF_B8 + i * 64 + lane * 8 + d / 4) >> (8 * (d % 4))) & 0xFFu);
                        }
                    }
                    ok8 = ok8 && ((float) e == acc8[m]);
                    ck8 += acc8[m];
                }
                const int   all_ok8 = sycl::reduce_over_group(sg, ok8 ? 1 : 0, sycl::minimum<int>());
                const float tot8    = sycl::reduce_over_group(sg, ck8, sycl::plus<float>());
                if (full) {
                    it.barrier(sycl::access::fence_space::local_space);
                }
                if (w == 0) {
#pragma unroll
                    for (int m = 0; m < 8; ++m) {
                        res[0 * 64 + lane * 8 + m] = acc8[m];
                    }
                    if (lane == 0) {
                        flag[0]         = all_ok8;
                        res[3 * 64 + 0] = tot8;
                    }
                }

                // s8_s4: reference only, stage B v1 does not wire the s4 K side
                fa_probe_float8 acc4;
                acc4 = 0.0f;
#pragma unroll
                for (int i = 0; i < FA_PROBE_IT4; ++i) {
                    fa_probe_int4 b4;
                    fa_probe_int8 d4;
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        b4[j] = rd(FA_PROBE_OFF_B4 + i * 32 + lane * 4 + j);
                    }
                    fa_probe_s8_s4(a8, b4, d4);
                    acc4 += __builtin_convertvector(d4, fa_probe_float8);
                }
                bool  ok4 = true;
                float ck4 = 0.0f;
#pragma unroll
                for (int m = 0; m < 8; ++m) {
                    int e = 0;
#pragma unroll
                    for (int i = 0; i < FA_PROBE_IT4; ++i) {
#pragma unroll
                        for (int d = 0; d < 32; ++d) {
                            const uint32_t byte = ((uint32_t) rd(FA_PROBE_OFF_B4 + i * 32 + lane * 4 + (d / 2) / 4) >>
                                                   (8 * ((d / 2) % 4))) & 0xFFu;
                            const int nib = (int) ((d % 2) ? (byte >> 4) : (byte & 0xFu));
                            e += s8_at(FA_PROBE_OFF_A8, m, d) * (nib >= 8 ? nib - 16 : nib);
                        }
                    }
                    ok4 = ok4 && ((float) e == acc4[m]);
                    ck4 += acc4[m];
                }
                const int   all_ok4 = sycl::reduce_over_group(sg, ok4 ? 1 : 0, sycl::minimum<int>());
                const float tot4    = sycl::reduce_over_group(sg, ck4, sycl::plus<float>());
                if (full) {
                    it.barrier(sycl::access::fence_space::local_space);
                }
                if (w == 0) {
#pragma unroll
                    for (int m = 0; m < 8; ++m) {
                        res[1 * 64 + lane * 8 + m] = acc4[m];
                    }
                    if (lane == 0) {
                        flag[1]         = all_ok4;
                        res[3 * 64 + 1] = tot4;
                    }
                }

                // hf_hf: the PV form of the sg8 kernel
                fa_probe_float8 acch;
                acch = 0.0f;
#pragma unroll
                for (int i = 0; i < FA_PROBE_ITHF; ++i) {
                    fa_probe_int8 ah;
                    fa_probe_int8 bh;
#pragma unroll
                    for (int m = 0; m < 8; ++m) {
                        ah[m] = (int32_t) ((uint32_t) rd_half(FA_PROBE_OFF_AHF, (2 * lane) * 8 + m) |
                                           ((uint32_t) rd_half(FA_PROBE_OFF_AHF, (2 * lane + 1) * 8 + m) << 16));
                    }
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        bh[j] = rd(FA_PROBE_OFF_BHF + i * 64 + lane * 8 + j);
                    }
                    fa_probe_f16(ah, bh, acch);
                }
                bool  okh = true;
                float ckh = 0.0f;
#pragma unroll
                for (int m = 0; m < 8; ++m) {
                    float e = 0.0f;
#pragma unroll
                    for (int i = 0; i < FA_PROBE_ITHF; ++i) {
#pragma unroll
                        for (int k = 0; k < 16; ++k) {
                            const float av = static_cast<float>(sycl::bit_cast<sycl::half>(rd_half(FA_PROBE_OFF_AHF, k * 8 + m)));
                            const float bv = static_cast<float>(
                                sycl::bit_cast<sycl::half>(rd_half(FA_PROBE_OFF_BHF + i * 64, lane * 16 + k)));
                            e += av * bv;
                        }
                    }
                    okh = okh && (sycl::fabs(acch[m] - e) <= 1e-4f * sycl::fmax(1.0f, sycl::fabs(e)));
                    ckh += acch[m];
                }
                const int   all_okh = sycl::reduce_over_group(sg, okh ? 1 : 0, sycl::minimum<int>());
                const float toth    = sycl::reduce_over_group(sg, ckh, sycl::plus<float>());
                if (full) {
                    it.barrier(sycl::access::fence_space::local_space);
                }
                if (w == 0) {
#pragma unroll
                    for (int m = 0; m < 8; ++m) {
                        res[2 * 64 + lane * 8 + m] = acch[m];
                    }
                    if (lane == 0) {
                        flag[2]         = all_okh;
                        res[3 * 64 + 2] = toth;
                    }
                }
            };
        cgh.parallel_for(ndr, kern);
    });
}

void ggml_sycl_fa_sg8_probe(ggml_backend_sycl_context * ctx) {
    const int mode = g_ggml_sycl_fa_sg8_probe; // 0 off, 1 full, 2 bare
    if (mode == 0 || ctx == nullptr) {
        return;
    }
    const int dev = ctx->device;
    if (!ggml_sycl_xmx_sg8_supports_arch(ggml_sycl_info().devices[dev].hw_info.arch)) {
        return;
    }
    static bool done[GGML_SYCL_MAX_DEVICES] = {};
    if (done[dev]) {
        return;
    }
    done[dev] = true;

    const char * variant = mode == 1 ? "full" : "bare";
    GGML_LOG_INFO("[SG8-PROBE] variant=%s start dev=%d\n", variant, dev);

    int32_t in_h[FA_PROBE_NIN];
    float   exp_h[3 * 64];
    float   exp_ck[3];
    fa_probe_fill(in_h);
    fa_probe_expect(exp_h, exp_ck);

    float res_h[FA_PROBE_NRES];
    for (int i = 0; i < FA_PROBE_NRES; ++i) {
        res_h[i] = 0.0f;
    }
    int32_t flag_h[3] = { 0, 0, 0 };

    sycl::queue & q      = *ctx->stream();
    int32_t *     in_d   = sycl::malloc_device<int32_t>(FA_PROBE_NIN, q);
    float *       res_d  = sycl::malloc_device<float>(FA_PROBE_NRES, q);
    int32_t *     flag_d = sycl::malloc_device<int32_t>(3, q);
    bool          run    = in_d != nullptr && res_d != nullptr && flag_d != nullptr;
    if (run) {
        try {
            q.memcpy(in_d, in_h, sizeof(in_h)).wait();
            q.memcpy(flag_d, flag_h, sizeof(flag_h)).wait();
            fa_probe_submit(in_d, res_d, flag_d, mode == 1, ctx->stream());
            q.wait_and_throw();
            q.memcpy(res_h, res_d, sizeof(res_h)).wait();
            q.memcpy(flag_h, flag_d, sizeof(flag_h)).wait();
        } catch (const sycl::exception & e) {
            GGML_LOG_ERROR("[SG8-PROBE] variant=%s sycl exception: %s\n", variant, e.what());
            run = false;
        }
    }
    if (in_d != nullptr) {
        sycl::free(in_d, q);
    }
    if (res_d != nullptr) {
        sycl::free(res_d, q);
    }
    if (flag_d != nullptr) {
        sycl::free(flag_d, q);
    }

    static const char * form_name[3] = { "s8_s8", "s8_s4", "hf_hf" };
    for (int f = 0; f < 3; ++f) {
        bool  ok = run && flag_h[f] == 1;
        float ck = 0.0f;
        for (int e = 0; e < 64; ++e) {
            const float got = res_h[f * 64 + e];
            const float exp = exp_h[f * 64 + e];
            ck += got;
            ok = ok && (f == 2 ? std::fabs(got - exp) <= 1e-4f * std::max(1.0f, std::fabs(exp)) : got == exp);
        }
        GGML_LOG_INFO("[SG8-PROBE] variant=%s form=%s %s checksum=%.1f expected=%.1f dev_checksum=%.1f\n",
                      variant, form_name[f], ok ? "ok" : "ng", (double) ck, (double) exp_ck[f],
                      (double) res_h[3 * 64 + f]);
    }
}

#else // GGML_SYCL_FA_SG8_PROBE_BUILD

void ggml_sycl_fa_sg8_probe(ggml_backend_sycl_context * ctx) {
    GGML_UNUSED(ctx);
}

#endif // GGML_SYCL_FA_SG8_PROBE_BUILD
