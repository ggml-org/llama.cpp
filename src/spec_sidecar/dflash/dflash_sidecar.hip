// DFlash2 HIP drafter sidecar DLL — loaded by llama.cpp's impl_draft_dflash.
// SPDX-License-Identifier: MIT
// Release ABI 5 adds lazy accepted-prefix commits, device-input paths,
// sequence-scoped state operations, and compact stochastic top-k proposals.
// API: spec_dflash_init(dir, n_seq) / spec_dflash_chunk(seq, poss, feats, n) /
//      spec_dflash_draft(seq, id_last, n_past, out_ids[7]).
#include "dflash_kernels.hip"
#include "../artifact_manifest.h"
#include "spec_sidecar/sidecar_abi.h"

#include <chrono>
#include <cinttypes>

#define DFC(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "[dflash-sidecar] %s @%d\n", hipGetErrorString(e_), __LINE__); return -1; } } while (0)

using Tens = spec_sidecar_artifact::TensorDesc;

static void deq_q4k_host(const uint8_t * src, float * dst, size_t n) {
    const block_q4_K * bs = (const block_q4_K *) src;
    for (size_t b = 0; b < n / 256; ++b) {
        const block_q4_K * bk = &bs[b];
        const float d = df_f16(bk->d), dmin = df_f16(bk->dmin);
        float * out = dst + b * 256;
        for (int g64 = 0; g64 < 4; ++g64) {
            float sc0, mn0, sc1, mn1;
            q4k_scale_min(g64 * 2 + 0, bk->scales, &sc0, &mn0);
            q4k_scale_min(g64 * 2 + 1, bk->scales, &sc1, &mn1);
            const uint8_t * qs = bk->qs + g64 * 32;
            for (int l = 0; l < 32; ++l) {
                out[g64 * 64 + l]      = d * sc0 * (qs[l] & 0xF) - dmin * mn0;
                out[g64 * 64 + 32 + l] = d * sc1 * (qs[l] >> 4)  - dmin * mn1;
            }
        }
    }
}

#define DISPATCH_Q4K(N, W, X, Y, NIN, NOUT, ST) do { \
    switch (N) { \
    case 1: gemv8t_q4_K<1><<<((NOUT)+7)/8, 512, 0, ST>>>(W, X, Y, NIN, NOUT); break; \
    case 2: gemv8t_q4_K<2><<<((NOUT)+7)/8, 512, 0, ST>>>(W, X, Y, NIN, NOUT); break; \
    case 3: gemv8t_q4_K<3><<<((NOUT)+7)/8, 512, 0, ST>>>(W, X, Y, NIN, NOUT); break; \
    case 4: gemv8t_q4_K<4><<<((NOUT)+7)/8, 512, 0, ST>>>(W, X, Y, NIN, NOUT); break; \
    case 5: gemv8t_q4_K<5><<<((NOUT)+7)/8, 512, 0, ST>>>(W, X, Y, NIN, NOUT); break; \
    case 6: gemv8t_q4_K<6><<<((NOUT)+7)/8, 512, 0, ST>>>(W, X, Y, NIN, NOUT); break; \
    case 7: gemv8t_q4_K<7><<<((NOUT)+7)/8, 512, 0, ST>>>(W, X, Y, NIN, NOUT); break; \
    default: gemv8t_q4_K<8><<<((NOUT)+7)/8, 512, 0, ST>>>(W, X, Y, NIN, NOUT); break; } } while (0)
#define DISPATCH_Q6K(N, W, X, Y, NIN, NOUT, ST) do { \
    switch (N) { \
    case 1: gemv8t_q6_K<1><<<((NOUT)+7)/ 8, 512, 0, ST>>>(W, X, Y, NIN, NOUT); break; \
    case 2: gemv8t_q6_K<2><<<((NOUT)+7)/ 8, 512, 0, ST>>>(W, X, Y, NIN, NOUT); break; \
    case 3: gemv8t_q6_K<3><<<((NOUT)+7)/ 8, 512, 0, ST>>>(W, X, Y, NIN, NOUT); break; \
    case 4: gemv8t_q6_K<4><<<((NOUT)+7)/ 8, 512, 0, ST>>>(W, X, Y, NIN, NOUT); break; \
    case 5: gemv8t_q6_K<5><<<((NOUT)+7)/ 8, 512, 0, ST>>>(W, X, Y, NIN, NOUT); break; \
    case 6: gemv8t_q6_K<6><<<((NOUT)+7)/ 8, 512, 0, ST>>>(W, X, Y, NIN, NOUT); break; \
    case 7: gemv8t_q6_K<7><<<((NOUT)+7)/ 8, 512, 0, ST>>>(W, X, Y, NIN, NOUT); break; \
    default: gemv8t_q6_K<8><<<((NOUT)+7)/ 8, 512, 0, ST>>>(W, X, Y, NIN, NOUT); break; } } while (0)
#define DISPATCH_Q4K_Q8(N, W, X, Q, Y, NIN, NOUT, ST) do { \
    switch (N) { \
    case 1: gemv_q4_K_q8_1<1><<<((NOUT)+7)/8, 256, 0, ST>>>(W, Q, Y, NIN, NOUT); break; \
    case 2: gemv_q4_K_q8_1<2><<<((NOUT)+7)/8, 256, 0, ST>>>(W, Q, Y, NIN, NOUT); break; \
    case 3: gemv_q4_K_q8_1<3><<<((NOUT)+7)/8, 256, 0, ST>>>(W, Q, Y, NIN, NOUT); break; \
    case 4: gemv_q4_K_q8_1<4><<<((NOUT)+7)/8, 256, 0, ST>>>(W, Q, Y, NIN, NOUT); break; \
    case 5: gemv_q4_K_q8_1<5><<<((NOUT)+7)/8, 256, 0, ST>>>(W, Q, Y, NIN, NOUT); break; \
    case 6: gemv_q4_K_q8_1<6><<<((NOUT)+7)/8, 256, 0, ST>>>(W, Q, Y, NIN, NOUT); break; \
    case 7: gemv_q4_K_q8_1<7><<<((NOUT)+7)/8, 256, 0, ST>>>(W, Q, Y, NIN, NOUT); break; \
    default: gemv_q4_K_q8_1<8><<<((NOUT)+7)/8, 256, 0, ST>>>(W, Q, Y, NIN, NOUT); break; } } while (0)
#define DISPATCH_Q6K_Q8(N, W, X, Q, Y, NIN, NOUT, ST) do { \
    switch (N) { \
    case 1: gemv_q6_K_q8_1<1><<<((NOUT)+7)/8, 256, 0, ST>>>(W, Q, Y, NIN, NOUT); break; \
    case 2: gemv_q6_K_q8_1<2><<<((NOUT)+7)/8, 256, 0, ST>>>(W, Q, Y, NIN, NOUT); break; \
    case 3: gemv_q6_K_q8_1<3><<<((NOUT)+7)/8, 256, 0, ST>>>(W, Q, Y, NIN, NOUT); break; \
    case 4: gemv_q6_K_q8_1<4><<<((NOUT)+7)/8, 256, 0, ST>>>(W, Q, Y, NIN, NOUT); break; \
    case 5: gemv_q6_K_q8_1<5><<<((NOUT)+7)/8, 256, 0, ST>>>(W, Q, Y, NIN, NOUT); break; \
    case 6: gemv_q6_K_q8_1<6><<<((NOUT)+7)/8, 256, 0, ST>>>(W, Q, Y, NIN, NOUT); break; \
    case 7: gemv_q6_K_q8_1<7><<<((NOUT)+7)/8, 256, 0, ST>>>(W, Q, Y, NIN, NOUT); break; \
    default: gemv_q6_K_q8_1<8><<<((NOUT)+7)/8, 256, 0, ST>>>(W, Q, Y, NIN, NOUT); break; } } while (0)
#define DISPATCH_Q4K_Q8_NATIVE(N, W, X, Q, Y, NIN, NOUT, ST) do { \
    switch (N) { \
    case 1: gemv_mmvq2_q4_K_q8_1<1><<<((NOUT)+1)/2, 32, 0, ST>>>(W, Q, Y, NIN, NOUT); break; \
    case 2: gemv_mmvq2_q4_K_q8_1<2><<<((NOUT)+1)/2, 32, 0, ST>>>(W, Q, Y, NIN, NOUT); break; \
    case 3: gemv_mmvq2_q4_K_q8_1<3><<<((NOUT)+1)/2, 32, 0, ST>>>(W, Q, Y, NIN, NOUT); break; \
    case 4: gemv_mmvq2_q4_K_q8_1<4><<<((NOUT)+1)/2, 32, 0, ST>>>(W, Q, Y, NIN, NOUT); break; \
    case 5: gemv_mmvq2_q4_K_q8_1<5><<<((NOUT)+1)/2, 32, 0, ST>>>(W, Q, Y, NIN, NOUT); break; \
    case 6: gemv_mmvq2_q4_K_q8_1<6><<<((NOUT)+1)/2, 32, 0, ST>>>(W, Q, Y, NIN, NOUT); break; \
    case 7: gemv_mmvq2_q4_K_q8_1<7><<<((NOUT)+1)/2, 32, 0, ST>>>(W, Q, Y, NIN, NOUT); break; \
    default: gemv_mmvq2_q4_K_q8_1<8><<<((NOUT)+1)/2, 32, 0, ST>>>(W, Q, Y, NIN, NOUT); break; } } while (0)
#define DISPATCH_Q6K_Q8_NATIVE(N, W, X, Q, Y, NIN, NOUT, ST) do { \
    switch (N) { \
    case 1: gemv_mmvq2_q6_K_q8_1<1><<<((NOUT)+1)/2, 32, 0, ST>>>(W, Q, Y, NIN, NOUT); break; \
    case 2: gemv_mmvq2_q6_K_q8_1<2><<<((NOUT)+1)/2, 32, 0, ST>>>(W, Q, Y, NIN, NOUT); break; \
    case 3: gemv_mmvq2_q6_K_q8_1<3><<<((NOUT)+1)/2, 32, 0, ST>>>(W, Q, Y, NIN, NOUT); break; \
    case 4: gemv_mmvq2_q6_K_q8_1<4><<<((NOUT)+1)/2, 32, 0, ST>>>(W, Q, Y, NIN, NOUT); break; \
    case 5: gemv_mmvq2_q6_K_q8_1<5><<<((NOUT)+1)/2, 32, 0, ST>>>(W, Q, Y, NIN, NOUT); break; \
    case 6: gemv_mmvq2_q6_K_q8_1<6><<<((NOUT)+1)/2, 32, 0, ST>>>(W, Q, Y, NIN, NOUT); break; \
    case 7: gemv_mmvq2_q6_K_q8_1<7><<<((NOUT)+1)/2, 32, 0, ST>>>(W, Q, Y, NIN, NOUT); break; \
    default: gemv_mmvq2_q6_K_q8_1<8><<<((NOUT)+1)/2, 32, 0, ST>>>(W, Q, Y, NIN, NOUT); break; } } while (0)
#define DISPATCH_Q4K_ANY(N, W, X, Q, Y, NIN, NOUT, USE_Q8, ST) do { \
    if (USE_Q8) DISPATCH_Q4K_Q8(N, W, X, Q, Y, NIN, NOUT, ST); \
    else        DISPATCH_Q4K(N, W, X, Y, NIN, NOUT, ST); } while (0)
#define DISPATCH_Q6K_ANY(N, W, X, Q, Y, NIN, NOUT, USE_Q8, ST) do { \
    if (USE_Q8) DISPATCH_Q6K_Q8(N, W, X, Q, Y, NIN, NOUT, ST); \
    else        DISPATCH_Q6K(N, W, X, Y, NIN, NOUT, ST); } while (0)
#define DISPATCH_Q4K_ANY_NATIVE(N, W, X, Q, Y, NIN, NOUT, USE_Q8, USE_NATIVE, ST) do { \
    if (USE_Q8) { \
        if (USE_NATIVE) DISPATCH_Q4K_Q8_NATIVE(N, W, X, Q, Y, NIN, NOUT, ST); \
        else             DISPATCH_Q4K_Q8(N, W, X, Q, Y, NIN, NOUT, ST); \
    } else { \
        DISPATCH_Q4K(N, W, X, Y, NIN, NOUT, ST); \
    } } while (0)
#define DISPATCH_Q6K_ANY_NATIVE(N, W, X, Q, Y, NIN, NOUT, USE_Q8, USE_NATIVE, ST) do { \
    if (USE_Q8) { \
        if (USE_NATIVE && !S.q8_native_q4_only) DISPATCH_Q6K_Q8_NATIVE(N, W, X, Q, Y, NIN, NOUT, ST); \
        else                                  DISPATCH_Q6K_Q8(N, W, X, Q, Y, NIN, NOUT, ST); \
    } else { \
        DISPATCH_Q6K(N, W, X, Y, NIN, NOUT, ST); \
    } } while (0)

namespace {
struct DLayer {
    const block_q4_K *conv_a_proj, *conv_f_proj, *wq, *wk, *wo, *up, *gate;
    const void * wv; bool wv_q6;
    const void * down; bool down_q6;
    const float *attn_norm, *qn, *kn, *ffn_norm, *conv_a_base, *conv_f_base;
};
struct DSeqState {
    __half * Kc[DF_LAYERS] = {};
    __half * Vc[DF_LAYERS] = {};
    __half * Kp[DF_LAYERS] = {};
    __half * Vp[DF_LAYERS] = {};
    int pending_capacity = 0;
    int32_t pending_pos_min = 0;
    int32_t pending_pos_max = 0;
    int32_t state_pos_min = 0;
    int32_t state_pos_max = 0; // exclusive; only committed target rows
    uint64_t state_epoch = 1;
    int kv_capacity = 0; // allocated Kc/Vc rows; grows geometrically on demand
};
struct DState {
    bool ok = false;
    std::vector<Tens> G;
    int32_t n_seq = 1;
    int device = -1;
    hipStream_t st = nullptr;
    hipStream_t owned_st = nullptr;
    bool owns_st = true;
    DLayer L[DF_LAYERS];
    const block_q4_K *W_fc, *W_selhid;
    const block_q4_0d *W_embd;
    const block_q6_K *W_head;
    const float *w_enc_norm, *w_out_norm;
    std::vector<float> sel_prev, sel_next;
    std::vector<int32_t> remap;
    int n_head = 40960; // rows in the loaded target head (sliced or full)
    float *feat_d = nullptr, *henc = nullptr, *x = nullptr, *xn = nullptr, *dyn_a = nullptr, *dyn_f = nullptr, *cg = nullptr, *q = nullptr, *k = nullptr, *v = nullptr, *att = nullptr, *wo_out = nullptr,
          *fu = nullptr, *fg = nullptr, *ftmp = nullptr, *resid = nullptr, *tE = nullptr, *logits = nullptr, *selhid = nullptr, *tk_pmax = nullptr, *tk_vals = nullptr;
    df_block_q8_1 * q8_feat = nullptr, *q8_xn = nullptr, *q8_cg = nullptr,
                   *q8_att = nullptr, *q8_fg = nullptr, *tE_q8 = nullptr;
    bool q8_head = false;
    bool q8_all = false;
    bool q8_process = false;
    bool q8_native_mmvq = false;
    bool q8_native_q4_only = false;
    int q8_layers = 5;
    float * q8_probe = nullptr;
    int q8_probe_slots = 0;
    int *pos_d[8] = {}, *tok_d[8] = {}, *tk_pidx = nullptr, *tk_ids = nullptr;
    int max_pos = 131072; // hard position ceiling; KV storage grows on demand
    DSeqState seq[8];
    int tkparts = 0;
    // Host scratch copied with the existing compact top-k readback. It lets
    // stochastic drafting choose a predecessor without a second graph or a
    // full-vocabulary transfer.
    int last_tk_ids[8 * DF_TOPK] = {};
    float last_tk_vals[8 * DF_TOPK] = {};
    float last_selhid[8 * DF_RANK] = {};
};
DState S;

// Sidecar calls run on an explicit stream, but HIP's current device is still
// thread-local state shared with llama.cpp. Keep device-placement experiments
// from leaking that state into the target backend.
struct DFlashDeviceGuard {
    int previous = -1;
    bool valid = false;
    bool changed = false;

    explicit DFlashDeviceGuard(int target = -1) {
        if (hipGetDevice(&previous) != hipSuccess) {
            return;
        }
        valid = true;
        if (target >= 0 && !set_target(target)) {
            return;
        }
    }

    bool set_target(int target) {
        if (!valid || target < 0 || previous == target) {
            return valid;
        }
        if (hipSetDevice(target) != hipSuccess) {
            valid = false;
            return false;
        }
        changed = true;
        return true;
    }

    ~DFlashDeviceGuard() {
        if (changed && previous >= 0) {
            (void) hipSetDevice(previous);
        }
    }
};

static bool valid_seq(int32_t seq_id) {
    return seq_id >= 0 && seq_id < S.n_seq && seq_id < 8;
}

static DSeqState * get_seq(int32_t seq_id) {
    return valid_seq(seq_id) ? &S.seq[seq_id] : nullptr;
}
Tens * DT(const std::string & n) {
    return spec_sidecar_artifact::find_tensor(S.G, n.c_str());
}
} // namespace

// See the MTP sidecar for the rationale. The single-block directional copy is
// only used by the cold context-shift path, never by draft().
__global__ void shift_kv_rows(__half * kv, int src_min, int src_max, int delta, int width) {
    // Move downward low-to-high and upward high-to-low so overlapping source
    // and destination ranges behave like memmove.
    const int step = delta > 0 ? -1 : 1;
    int pos = delta > 0 ? src_max - 1 : src_min;
    for (int r = 0; r < src_max - src_min; ++r, pos += step) {
        const size_t src = (size_t) pos * (size_t) width;
        const size_t dst = (size_t) (pos + delta) * (size_t) width;
        for (int i = (int) threadIdx.x; i < width; i += (int) blockDim.x) {
            kv[dst + (size_t) i] = kv[src + (size_t) i];
        }
        __syncthreads();
    }
}

#ifdef _WIN32
#define DF_API extern "C" __declspec(dllexport)
#else
#define DF_API extern "C"
#endif

static constexpr int32_t DF_RELEASE_ABI = 5;

static_assert(DF_TOPK == SPEC_SIDECAR_DFLASH_DRAFT_TOP_K,
              "DFlash stochastic top-k ABI mismatch");

static hipGraphExec_t g_draft_gexec[8] = {};

// Grow the committed KV cache to hold at least `needed` rows. Storage starts
// at 16K positions (~335 MB/seq) and doubles on demand up to S.max_pos, so
// long generations scale automatically instead of hitting a fixed cap. The
// draft hipGraph bakes in the Kc/Vc pointers, so growth invalidates the
// cached graph for this sequence; the next draft recaptures it (one-time).
static int ensure_seq_storage(DSeqState & q, int needed) {
    if (needed <= q.kv_capacity) return 0;
    if (needed > S.max_pos) return -1;
    int capacity = q.kv_capacity > 0 ? q.kv_capacity : 16384;
    while (capacity < needed) capacity *= 2;
    if (capacity > S.max_pos) capacity = S.max_pos;
    const size_t kv_bytes = (size_t) capacity * DF_KV * DF_HDIM * sizeof(__half);
    const size_t row_bytes = (size_t) DF_KV * DF_HDIM * sizeof(__half);
    DFC(hipStreamSynchronize(S.st));
    for (int il = 0; il < DF_LAYERS; ++il) {
        __half * nk = nullptr, * nv = nullptr;
        DFC(hipMalloc(&nk, kv_bytes));
        DFC(hipMalloc(&nv, kv_bytes));
        if (q.state_pos_max > 0) {
            DFC(hipMemcpyAsync(nk, q.Kc[il], (size_t) q.state_pos_max * row_bytes, hipMemcpyDeviceToDevice, S.st));
            DFC(hipMemcpyAsync(nv, q.Vc[il], (size_t) q.state_pos_max * row_bytes, hipMemcpyDeviceToDevice, S.st));
        }
        DFC(hipStreamSynchronize(S.st));
        if (q.Kc[il] != nullptr) DFC(hipFree(q.Kc[il]));
        if (q.Vc[il] != nullptr) DFC(hipFree(q.Vc[il]));
        q.Kc[il] = nk;
        q.Vc[il] = nv;
    }
    const int seq_idx = (int) (&q - S.seq);
    if (seq_idx >= 0 && seq_idx < 8 && g_draft_gexec[seq_idx] != nullptr) {
        DFC(hipGraphExecDestroy(g_draft_gexec[seq_idx]));
        g_draft_gexec[seq_idx] = nullptr;
    }
    if (q.kv_capacity > 0) {
        fprintf(stderr, "[dflash-sidecar] seq %d KV cache grown %d -> %d rows\n", seq_idx, q.kv_capacity, capacity);
    }
    q.kv_capacity = capacity;
    return 0;
}

static int ensure_pending_storage(DSeqState & q, int count) {
    if (count <= q.pending_capacity) return 0;
    if (q.pending_pos_max != q.state_pos_max) {
        fprintf(stderr, "[dflash-sidecar] cannot resize pending KV while a commit is in flight\n");
        return -1;
    }
    DFC(hipStreamSynchronize(S.st));
    for (int il = 0; il < DF_LAYERS; ++il) {
        if (q.Kp[il] != nullptr) DFC(hipFree(q.Kp[il]));
        if (q.Vp[il] != nullptr) DFC(hipFree(q.Vp[il]));
    }
    const int capacity = std::max(count, std::max(8, q.pending_capacity * 2));
    const size_t kv_bytes = (size_t) capacity * DF_KV * DF_HDIM * sizeof(__half);
    for (int il = 0; il < DF_LAYERS; ++il) {
        DFC(hipMalloc(&q.Kp[il], kv_bytes));
        DFC(hipMalloc(&q.Vp[il], kv_bytes));
    }
    q.pending_capacity = capacity;
    return 0;
}

static void discard_pending(DSeqState & q) {
    q.pending_pos_min = q.state_pos_max;
    q.pending_pos_max = q.state_pos_max;
}

DF_API int spec_dflash_release_abi() {
    return DF_RELEASE_ABI;
}

DF_API int spec_dflash_check(int32_t n_embd_enc, int32_t block_size, int32_t n_seq) {
    return (n_embd_enc == 25600 && block_size == DF_BLOCK && n_seq >= 1 && n_seq <= 8) ? 0 : -1;
}

DF_API int spec_dflash_state_size() {
    return (int) sizeof(spec_sidecar_state);
}

DF_API int spec_dflash_get_state(int32_t seq_id, void * data, int size) {
    DSeqState * q = get_seq(seq_id);
    if (!S.ok || q == nullptr || data == nullptr || size != (int) sizeof(spec_sidecar_state)) {
        return -1;
    }
    spec_sidecar_state state = {};
    state.magic   = SPEC_SIDECAR_STATE_MAGIC;
    state.version = SPEC_SIDECAR_STATE_VERSION;
    state.kind    = SPEC_SIDECAR_STATE_KIND_DFLASH;
    state.pos_min = q->state_pos_min;
    state.pos_max = q->state_pos_max;
    state.epoch   = q->state_epoch;
    std::memcpy(data, &state, sizeof(state));
    return 0;
}

static bool spec_dflash_state_matches(int32_t seq_id, const void * data, int size,
        spec_sidecar_state & state) {
    DSeqState * q = get_seq(seq_id);
    if (!S.ok || q == nullptr || data == nullptr || size != (int) sizeof(state)) return false;
    std::memcpy(&state, data, sizeof(state));
    return state.magic == SPEC_SIDECAR_STATE_MAGIC &&
           state.version == SPEC_SIDECAR_STATE_VERSION &&
           state.kind == SPEC_SIDECAR_STATE_KIND_DFLASH &&
           state.epoch == q->state_epoch &&
           state.pos_min == q->state_pos_min &&
           state.pos_min >= 0 && state.pos_min <= state.pos_max &&
           state.pos_max <= q->state_pos_max && state.pos_max <= S.max_pos;
}

DF_API int spec_dflash_set_state(int32_t seq_id, const void * data, int size) {
    DSeqState * q = get_seq(seq_id);
    spec_sidecar_state state = {};
    if (q == nullptr || !spec_dflash_state_matches(seq_id, data, size, state)) return -1;
    q->state_pos_max = state.pos_max;
    discard_pending(*q);
    return 0;
}

DF_API int spec_dflash_reset_state(int32_t seq_id) {
    DSeqState * q = get_seq(seq_id);
    if (!S.ok || q == nullptr) return -1;
    if (++q->state_epoch == 0) q->state_epoch = 1;
    q->state_pos_min = 0;
    q->state_pos_max = 0;
    discard_pending(*q);
    // Give grown KV storage back on reset: a long generation borrows memory
    // and returns it when the request ends. The next >16K request pays one
    // re-alloc plus a graph recapture (~15 ms, once).
    if (q->kv_capacity > 16384) {
        DFlashDeviceGuard device_guard(S.device);
        if (device_guard.valid) {
            (void) hipStreamSynchronize(S.st);
            for (int il = 0; il < DF_LAYERS; ++il) {
                if (q->Kc[il] != nullptr) { (void) hipFree(q->Kc[il]); q->Kc[il] = nullptr; }
                if (q->Vc[il] != nullptr) { (void) hipFree(q->Vc[il]); q->Vc[il] = nullptr; }
            }
            const int seq_idx = (int) (q - S.seq);
            if (seq_idx >= 0 && seq_idx < 8 && g_draft_gexec[seq_idx] != nullptr) {
                (void) hipGraphExecDestroy(g_draft_gexec[seq_idx]);
                g_draft_gexec[seq_idx] = nullptr;
            }
            fprintf(stderr, "[dflash-sidecar] seq %d KV cache shrunk %d -> 0 rows on reset\n", seq_idx, q->kv_capacity);
            q->kv_capacity = 0;
        }
    }
    return 0;
}

DF_API int spec_dflash_truncate_state(int32_t seq_id, int32_t pos_max) {
    DSeqState * q = get_seq(seq_id);
    if (!S.ok || q == nullptr || pos_max < q->state_pos_min || pos_max > q->state_pos_max) return -1;
    q->state_pos_max = pos_max;
    discard_pending(*q);
    return 0;
}

DF_API int spec_dflash_commit_state(int32_t seq_id, int32_t pos_max) {
    DSeqState * q = get_seq(seq_id);
    if (!S.ok || q == nullptr || pos_max < q->state_pos_max ||
            pos_max > q->pending_pos_max || pos_max > S.max_pos) return -1;
    DFlashDeviceGuard device_guard(S.device);
    if (!device_guard.valid) return -1;
    const bool sync_timing = getenv("LLAMA_SPEC_HIP_SYNC_TIMING") != nullptr;
    const auto t0 = sync_timing ? std::chrono::high_resolution_clock::now()
                                : std::chrono::high_resolution_clock::time_point{};
    int n_rows_committed = 0;
    if (pos_max > q->state_pos_max) {
        if (ensure_seq_storage(*q, pos_max) != 0 || q->pending_pos_min != q->state_pos_max) return -1;
        const size_t src_row = (size_t) (q->state_pos_max - q->pending_pos_min);
        const size_t n_rows = (size_t) (pos_max - q->state_pos_max);
        n_rows_committed = (int) n_rows;
        const size_t row_bytes = (size_t) DF_KV * DF_HDIM * sizeof(__half);
        for (int il = 0; il < DF_LAYERS; ++il) {
            DFC(hipMemcpyAsync(q->Kc[il] + (size_t) q->state_pos_max * DF_KV * DF_HDIM,
                               q->Kp[il] + src_row * DF_KV * DF_HDIM,
                               n_rows * row_bytes, hipMemcpyDeviceToDevice, S.st));
            DFC(hipMemcpyAsync(q->Vc[il] + (size_t) q->state_pos_max * DF_KV * DF_HDIM,
                               q->Vp[il] + src_row * DF_KV * DF_HDIM,
                               n_rows * row_bytes, hipMemcpyDeviceToDevice, S.st));
        }
        q->state_pos_max = pos_max;
    }
    discard_pending(*q);
    if (sync_timing) {
        DFC(hipStreamSynchronize(S.st));
        static uint64_t n_calls = 0;
        static uint64_t n_rows = 0;
        static double total_ms = 0.0;
        total_ms += std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count();
        n_rows += (uint64_t) n_rows_committed;
        if (++n_calls % 25 == 0) {
            fprintf(stderr, "[dflash-sidecar] commit complete avg %.3f ms (%.1f rows) over %" PRIu64 " calls\n",
                    total_ms / (double) n_calls, (double) n_rows / (double) n_calls, n_calls);
        }
    }
    return 0;
}

DF_API int spec_dflash_rebase_state(int32_t seq_id, int32_t pos_min, int32_t pos_max, int32_t delta) {
    DSeqState * q = get_seq(seq_id);
    if (!S.ok || q == nullptr || pos_min < 0 || pos_min > pos_max || pos_max > S.max_pos ||
            q->pending_pos_max != q->state_pos_max) return -1;
    DFlashDeviceGuard device_guard(S.device);
    if (!device_guard.valid) return -1;
    if (delta == 0 || q->state_pos_max <= pos_min) {
        if (delta != 0 && ++q->state_epoch == 0) q->state_epoch = 1;
        return 0;
    }
    if (pos_max < q->state_pos_max || q->kv_capacity <= 0) return -1;

    const int64_t new_min = (int64_t) pos_min + delta;
    const int64_t new_max = (int64_t) q->state_pos_max + delta;
    if (new_min < 0 || new_max < new_min || new_max > S.max_pos ||
            ensure_seq_storage(*q, (int) new_max) != 0) return -1;

    DFC(hipStreamSynchronize(S.st));
    for (int il = 0; il < DF_LAYERS; ++il) {
        shift_kv_rows<<<1, 256, 0, S.st>>>(q->Kc[il], pos_min, q->state_pos_max, delta, DF_KV * DF_HDIM);
        DFC(hipGetLastError());
        shift_kv_rows<<<1, 256, 0, S.st>>>(q->Vc[il], pos_min, q->state_pos_max, delta, DF_KV * DF_HDIM);
        DFC(hipGetLastError());
    }
    DFC(hipStreamSynchronize(S.st));
    if (q->state_pos_min >= pos_min) q->state_pos_min += delta;
    q->state_pos_max = (int32_t) new_max;
    discard_pending(*q);
    if (++q->state_epoch == 0) q->state_epoch = 1;
    return 0;
}

DF_API int spec_dflash_attach_target_stream(void * stream, int32_t device) {
    if (!S.ok || device != S.device) return -1;
    DFlashDeviceGuard device_guard(S.device);
    if (!device_guard.valid) return -1;
    if (stream == nullptr) {
        if (S.st != S.owned_st) {
            DFC(hipStreamSynchronize(S.st));
            S.st = S.owned_st;
            S.owns_st = true;
        }
        return 0;
    }
    const hipStream_t target = reinterpret_cast<hipStream_t>(stream);
    if (target != S.st) {
        DFC(hipStreamSynchronize(S.st));
        S.st = target;
        S.owns_st = false;
    }
    return 0;
}

static bool expect_tensor(const std::vector<Tens> & tensors, const std::string & name,
                          const char * dtype, std::initializer_list<int64_t> shape,
                          uint64_t nbytes, std::string & error) {
    const Tens * tensor = spec_sidecar_artifact::find_tensor(tensors, name.c_str());
    if (tensor == nullptr) {
        error = "missing tensor: " + name;
        return false;
    }
    if (!spec_sidecar_artifact::tensor_matches(*tensor, dtype, shape, nbytes)) {
        error = "tensor schema mismatch: " + name;
        return false;
    }
    return true;
}

static bool validate_dflash_schema(const std::vector<Tens> & tensors, std::string & error) {
    if (tensors.size() != 81) {
        error = "DFlash manifest must contain exactly 81 tensors";
        return false;
    }
    if (!expect_tensor(tensors, "enc.output_norm.weight",       "0",  {5120},          20480, error) ||
        !expect_tensor(tensors, "fc.weight",                    "12", {25600, 5120}, 73728000, error) ||
        !expect_tensor(tensors, "output_norm.weight",           "0",  {5120},          20480, error) ||
        !expect_tensor(tensors, "selector_hidden.weight",       "12", {5120, 256},     737280, error) ||
        !expect_tensor(tensors, "selector_predecessor.weight",  "12", {256, 248320}, 35758080, error) ||
        !expect_tensor(tensors, "selector_successor.weight",    "12", {256, 248320}, 35758080, error)) {
        return false;
    }
    for (int layer = 0; layer < DF_LAYERS; ++layer) {
        char prefix[32];
        std::snprintf(prefix, sizeof(prefix), "blk.%d.", layer);
        const auto name = [&](const char * suffix) { return std::string(prefix) + suffix; };
        const bool q6 = layer == 2 || layer == 4;
        if (!expect_tensor(tensors, name("attn_conv_base"),        "0",  {5120, 2, 2}, 81920, error) ||
            !expect_tensor(tensors, name("attn_conv_proj.weight"), "12", {5120, 1280}, 3686400, error) ||
            !expect_tensor(tensors, name("attn_k.weight"),         "12", {5120, 1024}, 2949120, error) ||
            !expect_tensor(tensors, name("attn_k_norm.weight"),    "0",  {128}, 512, error) ||
            !expect_tensor(tensors, name("attn_norm.weight"),      "0",  {5120}, 20480, error) ||
            !expect_tensor(tensors, name("attn_output.weight"),    "12", {4096, 5120}, 11796480, error) ||
            !expect_tensor(tensors, name("attn_q.weight"),         "12", {5120, 4096}, 11796480, error) ||
            !expect_tensor(tensors, name("attn_q_norm.weight"),    "0",  {128}, 512, error) ||
            !expect_tensor(tensors, name("attn_v.weight"), q6 ? "14" : "12", {5120, 1024}, q6 ? 4300800 : 2949120, error) ||
            !expect_tensor(tensors, name("ffn_conv_base"),         "0",  {5120, 2, 2}, 81920, error) ||
            !expect_tensor(tensors, name("ffn_conv_proj.weight"),  "12", {5120, 1280}, 3686400, error) ||
            !expect_tensor(tensors, name("ffn_down.weight"), q6 ? "14" : "12", {17408, 5120}, q6 ? 73113600 : 50135040, error) ||
            !expect_tensor(tensors, name("ffn_gate.weight"),       "12", {5120, 17408}, 50135040, error) ||
            !expect_tensor(tensors, name("ffn_norm.weight"),       "0",  {5120}, 20480, error) ||
            !expect_tensor(tensors, name("ffn_up.weight"),         "12", {5120, 17408}, 50135040, error)) {
            return false;
        }
    }
    return true;
}

static bool validate_target_schema(const std::vector<Tens> & tensors, std::string & error) {
    struct Exp { const char * name, * dtype; int64_t ne0, ne1; uint64_t nbytes; };
    static const Exp EXPECTED[] = {
        {"token_embd.weight",                         "2",  5120, 248320, 715161600},
        {"blk.64.attn_k.weight",                     "2",  5120,   1024,   2949120},
        {"blk.64.attn_k_norm.weight",                "0",   256,      0,      1024},
        {"blk.64.attn_norm.weight",                  "0",  5120,      0,     20480},
        {"blk.64.attn_output.weight",                "2",  6144,   5120,  17694720},
        {"blk.64.attn_q.weight",                     "2",  5120,  12288,  35389440},
        {"blk.64.attn_q_norm.weight",                "0",   256,      0,      1024},
        {"blk.64.attn_v.weight",                     "2",  5120,   1024,   2949120},
        {"blk.64.ffn_down.weight",                   "2", 17408,   5120,  50135040},
        {"blk.64.ffn_gate.weight",                   "2",  5120,  17408,  50135040},
        {"blk.64.ffn_up.weight",                     "2",  5120,  17408,  50135040},
        {"blk.64.nextn.eh_proj.weight",              "2", 10240,   5120,  29491200},
        {"blk.64.nextn.enorm.weight",                "0",  5120,      0,     20480},
        {"blk.64.nextn.hnorm.weight",                "0",  5120,      0,     20480},
        {"blk.64.nextn.shared_head_norm.weight",     "0",  5120,      0,     20480},
        {"blk.64.post_attention_norm.weight",        "0",  5120,      0,     20480},
        {"blk.64.nextn.shared_head_head.weight",     "2",  5120,  40960, 117964800},
    };
    if (tensors.size() != 1 && tensors.size() != sizeof(EXPECTED) / sizeof(EXPECTED[0])) {
        error = "target manifest must contain exactly 1 or 17 tensors";
        return false;
    }
    const size_t count = tensors.size() == 1 ? 1 : sizeof(EXPECTED) / sizeof(EXPECTED[0]);
    for (size_t i = 0; i < count; ++i) {
        const Exp & item = EXPECTED[i];
        const Tens * tensor = spec_sidecar_artifact::find_tensor(tensors, item.name);
        const bool matches = tensor != nullptr && (item.ne1 == 0
            ? spec_sidecar_artifact::tensor_matches(*tensor, item.dtype, {item.ne0}, item.nbytes)
            : spec_sidecar_artifact::tensor_matches(*tensor, item.dtype, {item.ne0, item.ne1}, item.nbytes));
        if (!matches) {
            error = std::string("target tensor schema mismatch: ") + item.name;
            return false;
        }
    }
    return true;
}

DF_API int spec_dflash_init(const char * dir, int32_t n_seq) {
    if (S.ok) return n_seq == S.n_seq ? 0 : -1;
    DFlashDeviceGuard caller_device;
    if (!caller_device.valid) return -1;
    if (n_seq < 1 || n_seq > 8) return -1;
    S.n_seq = n_seq;
    // Q8_1-activation GEMV is the default on the qualified gfx1030 sidecar;
    // LLAMA_SPEC_HIP_DFLASH_Q8_DISABLE restores the pure-f32 kernels for A/B.
    S.q8_all = getenv("LLAMA_SPEC_HIP_DFLASH_Q8_DISABLE") == nullptr ||
               getenv("LLAMA_SPEC_HIP_DFLASH_Q8_ALL") != nullptr;
    S.q8_process = S.q8_all && getenv("LLAMA_SPEC_HIP_DFLASH_Q8_PROCESS") != nullptr;
    // Q4_K is automatically selected only for the exact gfx1030 sidecar
    // build, and only reaches a kernel when an existing Q8 dispatch gate is
    // active. The full selector remains an audit-only override because the
    // native-shaped Q6 path regresses. The disable switch is an opt-out for
    // A/B and rollback; there is no Q4 performance opt-in in normal use.
#if defined(SPEC_SIDECAR_GFX1030)
    constexpr bool gfx1030_sidecar = true;
#else
    constexpr bool gfx1030_sidecar = false;
#endif
    const bool native_all = gfx1030_sidecar && getenv("LLAMA_SPEC_HIP_DFLASH_Q8_NATIVE_MMVQ") != nullptr;
    const bool native_q4 = gfx1030_sidecar &&
        (native_all || getenv("LLAMA_SPEC_HIP_DFLASH_DISABLE_NATIVE_Q4_MMVQ") == nullptr);
    S.q8_native_q4_only = native_q4 && !native_all;
    S.q8_native_mmvq = native_q4 || native_all;
    S.q8_layers = 5;
    if (const char * layers = getenv("LLAMA_SPEC_HIP_DFLASH_Q8_LAYERS")) {
        S.q8_layers = std::max(0, std::min(DF_LAYERS, std::atoi(layers)));
    }
    const bool q8_no_head = getenv("LLAMA_SPEC_HIP_DFLASH_Q8_NO_HEAD") != nullptr;
    S.q8_head = !q8_no_head && (S.q8_all || getenv("LLAMA_SPEC_HIP_DFLASH_Q8_HEAD") != nullptr);
    if (const char * requested = getenv("LLAMA_SPEC_HIP_DFLASH_DEVICE")) {
        const int device = atoi(requested);
        int device_count = 0;
        DFC(hipGetDeviceCount(&device_count));
        if (device < 0 || device >= device_count || !caller_device.set_target(device)) {
            fprintf(stderr, "[dflash-sidecar] invalid requested device %d (count=%d)\n",
                    device, device_count);
            return -1;
        }
    }
    DFC(hipGetDevice(&S.device));
    char pb[1024];
    std::string validation_error;
    if (!spec_sidecar_artifact::make_path(pb, sizeof(pb), dir, "dflash_manifest.json") ||
        !spec_sidecar_artifact::load_manifest(pb, S.G, validation_error)) {
        fprintf(stderr, "[dflash-sidecar] invalid DFlash manifest: %s\n", validation_error.c_str());
        return -1;
    }
    if (!validate_dflash_schema(S.G, validation_error)) {
        fprintf(stderr, "[dflash-sidecar] %s\n", validation_error.c_str());
        return -1;
    }
    if (!spec_sidecar_artifact::make_path(pb, sizeof(pb), dir, "dflash_weights.bin")) return -1;
    FILE * wf = fopen(pb, "rb");
    uint64_t wf_size = 0;
    if (!wf || !spec_sidecar_artifact::file_size(wf, wf_size) ||
        !spec_sidecar_artifact::validate_blob_layout(S.G, wf_size, validation_error)) {
        fprintf(stderr, "[dflash-sidecar] invalid DFlash weights layout: %s\n", validation_error.c_str());
        if (wf) fclose(wf);
        return -1;
    }

    // Validate the complete target embedding blob, not only the descriptor used
    // by this sidecar. prepare_assets emits either the embedding alone or the
    // complete 17-tensor MTP blob.
    std::vector<Tens> target_manifest;
    if (!spec_sidecar_artifact::make_path(pb, sizeof(pb), dir, "drafter_manifest.json") ||
        !spec_sidecar_artifact::load_manifest(pb, target_manifest, validation_error) ||
        !validate_target_schema(target_manifest, validation_error)) {
        fprintf(stderr, "[dflash-sidecar] invalid target manifest: %s\n", validation_error.c_str());
        fclose(wf);
        return -1;
    }
    const Tens * token_embedding = spec_sidecar_artifact::find_tensor(target_manifest, "token_embd.weight");
    if (!spec_sidecar_artifact::make_path(pb, sizeof(pb), dir, "drafter_weights.bin")) { fclose(wf); return -1; }
    FILE * target_wf = fopen(pb, "rb");
    uint64_t target_wf_size = 0;
    if (!target_wf || !spec_sidecar_artifact::file_size(target_wf, target_wf_size) ||
        !spec_sidecar_artifact::validate_blob_layout(target_manifest, target_wf_size, validation_error)) {
        fprintf(stderr, "[dflash-sidecar] invalid target weights layout: %s\n", validation_error.c_str());
        if (target_wf) fclose(target_wf);
        fclose(wf);
        return -1;
    }

    // Validate the selected target head and its host-side remap before any HIP
    // allocation/upload. This keeps malformed artifacts out of GPU read paths.
    const bool full_head = getenv("LLAMA_SPEC_HIP_FULL_HEAD") != nullptr;
    const int expected_rows = full_head ? DF_VOCAB : 40960;
    const uint64_t expected_head_size = static_cast<uint64_t>(expected_rows) * 4200;
    if (!spec_sidecar_artifact::make_path(pb, sizeof(pb), dir, full_head ? "target_head.bin" : "target_head_sliced.bin")) {
        fclose(target_wf); fclose(wf); return -1;
    }
    FILE * head_file = fopen(pb, "rb");
    uint64_t head_size = 0;
    if (!head_file || !spec_sidecar_artifact::file_size(head_file, head_size) || head_size != expected_head_size) {
        fprintf(stderr, "[dflash-sidecar] target-head size mismatch: expected %llu bytes\n",
                static_cast<unsigned long long>(expected_head_size));
        if (head_file) fclose(head_file);
        fclose(target_wf); fclose(wf);
        return -1;
    }
    S.n_head = expected_rows;
    std::vector<int32_t> remap_h(static_cast<size_t>(S.n_head));
    if (full_head) {
        for (int i = 0; i < S.n_head; ++i) remap_h[i] = i;
    } else {
        if (!spec_sidecar_artifact::make_path(pb, sizeof(pb), dir, "draft_head_ids.bin")) {
            fclose(head_file); fclose(target_wf); fclose(wf); return -1;
        }
        FILE * remap_file = fopen(pb, "rb");
        uint64_t remap_size = 0;
        const uint64_t expected_remap_size = static_cast<uint64_t>(S.n_head) * sizeof(int32_t);
        if (!remap_file || !spec_sidecar_artifact::file_size(remap_file, remap_size) ||
            remap_size != expected_remap_size ||
            !spec_sidecar_artifact::read_exact_at(remap_file, 0, remap_h.data(), expected_remap_size)) {
            fprintf(stderr, "[dflash-sidecar] remap size/read mismatch\n");
            if (remap_file) fclose(remap_file);
            fclose(head_file); fclose(target_wf); fclose(wf);
            return -1;
        }
        fclose(remap_file);
        if (!spec_sidecar_artifact::validate_remap(remap_h, DF_VOCAB, validation_error)) {
            fprintf(stderr, "[dflash-sidecar] invalid remap: %s\n", validation_error.c_str());
            fclose(head_file); fclose(target_wf); fclose(wf);
            return -1;
        }
    }

    // All external metadata/ranges are now validated. The normal upload path is
    // unchanged from here onward and runtime kernels do no manifest work.
    std::vector<char> hb;
    for (auto & t : S.G) {
        hb.resize(static_cast<size_t>(t.nbytes));
        if (!spec_sidecar_artifact::read_exact_at(wf, t.offset, hb.data(), t.nbytes)) { fclose(wf); return -1; }
        DFC(hipMalloc(&t.dev, static_cast<size_t>(t.nbytes)));
        DFC(hipMemcpy(t.dev, hb.data(), static_cast<size_t>(t.nbytes), hipMemcpyHostToDevice));
    }
    // selector tables -> host f32
    S.sel_prev.resize((size_t) DF_VOCAB * DF_RANK);
    S.sel_next.resize((size_t) DF_VOCAB * DF_RANK);
    for (auto & pr : { std::make_pair("selector_predecessor.weight", S.sel_prev.data()),
                       std::make_pair("selector_successor.weight",   S.sel_next.data()) }) {
        Tens * t = DT(pr.first);
        if (!t) { fclose(wf); return -1; }
        hb.resize(static_cast<size_t>(t->nbytes));
        if (!spec_sidecar_artifact::read_exact_at(wf, t->offset, hb.data(), t->nbytes)) { fclose(wf); return -1; }
        deq_q4k_host((const uint8_t *) hb.data(), pr.second, (size_t) DF_VOCAB * DF_RANK);
    }
    fclose(wf);

    // target token_embd pinned host (from drafter_weights.bin)
    {
        void * hptr = nullptr;
        DFC(hipHostMalloc(&hptr, static_cast<size_t>(token_embedding->nbytes), hipHostMallocMapped));
        if (!spec_sidecar_artifact::read_exact_at(target_wf, token_embedding->offset, hptr, token_embedding->nbytes)) {
            fclose(target_wf); return -1;
        }
        fclose(target_wf);
        void * dp = nullptr;
        DFC(hipHostGetDevicePointer(&dp, hptr, 0));
        S.W_embd = (const block_q4_0d *) dp;
    }
    // target head: sliced 40960 rows + id remap by default; full head uses identity
    {
        std::vector<char> buf(static_cast<size_t>(head_size));
        if (!spec_sidecar_artifact::read_exact_at(head_file, 0, buf.data(), head_size)) { fclose(head_file); return -1; }
        fclose(head_file);
        void * d = nullptr;
        DFC(hipMalloc(&d, static_cast<size_t>(head_size)));
        DFC(hipMemcpy(d, buf.data(), static_cast<size_t>(head_size), hipMemcpyHostToDevice));
        S.W_head = (const block_q6_K *) d;
        fprintf(stderr, "[dflash-sidecar] head: %s (%d rows)\n", full_head ? "FULL" : "sliced", S.n_head);
    }
    S.remap = std::move(remap_h);
    if (!full_head) {
        fprintf(stderr, "[dflash-sidecar] release-abi: %d bundled-remap\n", DF_RELEASE_ABI);
    }
    for (int il = 0; il < DF_LAYERS; ++il) {
        auto nm = [&](const char * s2) { char b2[128]; snprintf(b2, sizeof(b2), "blk.%d.%s", il, s2); return std::string(b2); };
        auto req = [&](const std::string & n) -> Tens * { Tens * t = DT(n); if (!t) { fprintf(stderr, "[dflash-sidecar] missing %s\n", n.c_str()); } return t; };
        Tens *t;
        if (!(t = req(nm("attn_conv_proj.weight")))) return -1; S.L[il].conv_a_proj = (const block_q4_K *) t->dev;
        if (!(t = req(nm("ffn_conv_proj.weight"))))  return -1; S.L[il].conv_f_proj = (const block_q4_K *) t->dev;
        if (!(t = req(nm("attn_q.weight")))) return -1; S.L[il].wq = (const block_q4_K *) t->dev;
        if (!(t = req(nm("attn_k.weight")))) return -1; S.L[il].wk = (const block_q4_K *) t->dev;
        if (!(t = req(nm("attn_v.weight")))) return -1; S.L[il].wv = t->dev; S.L[il].wv_q6 = t->dtype == "14";
        if (!(t = req(nm("attn_output.weight")))) return -1; S.L[il].wo = (const block_q4_K *) t->dev;
        if (!(t = req(nm("ffn_up.weight"))))   return -1; S.L[il].up   = (const block_q4_K *) t->dev;
        if (!(t = req(nm("ffn_gate.weight")))) return -1; S.L[il].gate = (const block_q4_K *) t->dev;
        if (!(t = req(nm("ffn_down.weight")))) return -1; S.L[il].down = t->dev; S.L[il].down_q6 = t->dtype == "14";
        if (!(t = req(nm("attn_norm.weight")))) return -1; S.L[il].attn_norm = (const float *) t->dev;
        if (!(t = req(nm("attn_q_norm.weight")))) return -1; S.L[il].qn = (const float *) t->dev;
        if (!(t = req(nm("attn_k_norm.weight")))) return -1; S.L[il].kn = (const float *) t->dev;
        if (!(t = req(nm("ffn_norm.weight")))) return -1; S.L[il].ffn_norm = (const float *) t->dev;
        if (!(t = req(nm("attn_conv_base")))) return -1; S.L[il].conv_a_base = (const float *) t->dev;
        if (!(t = req(nm("ffn_conv_base")))) return -1; S.L[il].conv_f_base = (const float *) t->dev;
    }
    Tens * t;
    if (!(t = DT("fc.weight"))) return -1; S.W_fc = (const block_q4_K *) t->dev;
    if (!(t = DT("enc.output_norm.weight"))) return -1; S.w_enc_norm = (const float *) t->dev;
    if (!(t = DT("output_norm.weight"))) return -1; S.w_out_norm = (const float *) t->dev;
    if (!(t = DT("selector_hidden.weight"))) return -1; S.W_selhid = (const block_q4_K *) t->dev;

    // Optional override of the hard position ceiling (storage still grows on
    // demand from 16K rows; this only bounds the growth).
    if (const char * mp = getenv("LLAMA_SPEC_HIP_MAX_POS")) {
        long long v = atoll(mp);
        if (v < 4096) v = 4096;
        if (v > 131072) v = 131072;
        S.max_pos = (int) v;
    }
    DFC(hipStreamCreate(&S.owned_st));
    S.st = S.owned_st;
    S.owns_st = true;
    DFC(hipMalloc(&S.feat_d, (size_t) 8 * 25600 * 4));
    DFC(hipMalloc(&S.henc,   (size_t) 8 * DF_EMBD * 4));
    DFC(hipMalloc(&S.x,      (size_t) DF_BLOCK * DF_EMBD * 4));
    DFC(hipMalloc(&S.xn,     (size_t) DF_BLOCK * DF_EMBD * 4));
    DFC(hipMalloc(&S.dyn_a,  (size_t) DF_BLOCK * DF_CONVDYN * 4));
    DFC(hipMalloc(&S.dyn_f,  (size_t) DF_BLOCK * DF_CONVDYN * 4));
    DFC(hipMalloc(&S.cg,     (size_t) DF_BLOCK * DF_EMBD * 4));
    DFC(hipMalloc(&S.q,      (size_t) DF_BLOCK * DF_HEADS * DF_HDIM * 4));
    DFC(hipMalloc(&S.k,      (size_t) 8 * DF_KV * DF_HDIM * 4));
    DFC(hipMalloc(&S.v,      (size_t) 8 * DF_KV * DF_HDIM * 4));
    DFC(hipMalloc(&S.att,    (size_t) DF_BLOCK * DF_HEADS * DF_HDIM * 4));
    DFC(hipMalloc(&S.wo_out, (size_t) DF_BLOCK * DF_EMBD * 4));
    DFC(hipMalloc(&S.fu,     (size_t) DF_BLOCK * DF_FF * 4));
    DFC(hipMalloc(&S.fg,     (size_t) DF_BLOCK * DF_FF * 4));
    DFC(hipMalloc(&S.ftmp,   (size_t) DF_BLOCK * DF_EMBD * 4));
    DFC(hipMalloc(&S.resid,  (size_t) DF_BLOCK * DF_EMBD * 4));
    DFC(hipMalloc(&S.tE,     (size_t) DF_BLOCK * DF_EMBD * 4));
    if (S.q8_all) {
        DFC(hipMalloc(&S.q8_feat, (size_t) DF_BLOCK * (25600 / 32) * sizeof(df_block_q8_1)));
        DFC(hipMalloc(&S.q8_xn,   (size_t) DF_BLOCK * (DF_EMBD / 32) * sizeof(df_block_q8_1)));
        DFC(hipMalloc(&S.q8_cg,   (size_t) DF_BLOCK * (DF_EMBD / 32) * sizeof(df_block_q8_1)));
        DFC(hipMalloc(&S.q8_att,  (size_t) DF_BLOCK * ((DF_HEADS * DF_HDIM) / 32) * sizeof(df_block_q8_1)));
        DFC(hipMalloc(&S.q8_fg,   (size_t) DF_BLOCK * (DF_FF / 32) * sizeof(df_block_q8_1)));
    }
    if (S.q8_head) {
        DFC(hipMalloc(&S.tE_q8, (size_t) DF_BLOCK * (DF_EMBD / 32) * sizeof(df_block_q8_1)));
    }
    if (S.q8_all) {
        DFC(hipMalloc(&S.q8_probe, (size_t) 60 * 4 * sizeof(float)));
    }
    DFC(hipMalloc(&S.logits, (size_t) DF_BLOCK * S.n_head * 4));
    DFC(hipMalloc(&S.selhid, (size_t) DF_BLOCK * DF_RANK * 4));
    for (int seq_id = 0; seq_id < 8; ++seq_id) {
        DFC(hipMalloc(&S.pos_d[seq_id], 8 * sizeof(int32_t)));
        DFC(hipMalloc(&S.tok_d[seq_id], 8 * sizeof(int32_t)));
    }
    S.tkparts = (S.n_head + 2047) / 2048;
    DFC(hipMalloc(&S.tk_pmax, (size_t) 8 * S.tkparts * 4));
    DFC(hipMalloc(&S.tk_pidx, (size_t) 8 * S.tkparts * 4));
    DFC(hipMalloc(&S.tk_ids,  8 * DF_TOPK * 4));
    DFC(hipMalloc(&S.tk_vals, 8 * DF_TOPK * 4));
    S.ok = true;
    fprintf(stderr, "[dflash-sidecar] initialized (max_pos=%d, device=%d, q8_head=%d, q8_all=%d, q8_process=%d, q8_layers=%d, q8_native_mmvq=%d, q8_native_q4_only=%d)\n",
            S.max_pos, S.device, S.q8_head ? 1 : 0, S.q8_all ? 1 : 0, S.q8_process ? 1 : 0, S.q8_layers,
            S.q8_native_mmvq ? 1 : 0, S.q8_native_q4_only ? 1 : 0);
    return 0;
}

// Encoder + KV inject for up to n rows of target features. Target rows land in
// a pending cache; only commit_state() copies the accepted prefix to Kc/Vc.
static int spec_dflash_chunk_impl(int32_t seq_id, const int32_t * poss,
        const float * feats, const void * const * layer_features_device,
        int n_layers, int layer_width, bool device_features, int n) {
    DSeqState * q = get_seq(seq_id);
    if (!S.ok || q == nullptr || n < 0 ||
            (n > 0 && (!poss || (device_features ? layer_features_device == nullptr : feats == nullptr)))) return -1;
    if (n == 0) return 0;
    DFlashDeviceGuard device_guard(S.device);
    if (!device_guard.valid) return -1;
    if (n > S.max_pos - q->state_pos_max || q->pending_pos_max != q->state_pos_max ||
            ensure_seq_storage(*q, q->state_pos_max + n) != 0 || ensure_pending_storage(*q, n) != 0) return -1;
    if (device_features && (n_layers != DF_LAYERS || layer_width != DF_EMBD / DF_LAYERS)) return -1;
    if (device_features) {
        for (int il = 0; il < DF_LAYERS; ++il) {
            if (layer_features_device[il] == nullptr) return -1;
        }
    }
    for (int r = 0; r < n; ++r) {
        if (poss[r] < 0 || poss[r] >= S.max_pos || poss[r] != q->state_pos_max + r) {
            fprintf(stderr, "[dflash-sidecar] seq %d chunk: non-contiguous position %d at row %d (expected %d)\n",
                    seq_id, poss[r], r, q->state_pos_max + r);
            return -1;
        }
    }
    static int n_calls = 0, n_rows = 0;
    static double tot_ms = 0.0;
    const bool dbg = getenv("LLAMA_SPEC_HIP_DEBUG") != nullptr;
    auto t0 = std::chrono::high_resolution_clock::now();
    const int NROT = 128;
    q->pending_pos_min = q->state_pos_max;
    q->pending_pos_max = q->state_pos_max + n;
    for (int off = 0; off < n; off += 8) {
        const int nb = std::min(8, n - off);
        if (device_features) {
            const float * layer0 = static_cast<const float *>(layer_features_device[0]) + (size_t) off * layer_width;
            const float * layer1 = static_cast<const float *>(layer_features_device[1]) + (size_t) off * layer_width;
            const float * layer2 = static_cast<const float *>(layer_features_device[2]) + (size_t) off * layer_width;
            const float * layer3 = static_cast<const float *>(layer_features_device[3]) + (size_t) off * layer_width;
            const float * layer4 = static_cast<const float *>(layer_features_device[4]) + (size_t) off * layer_width;
            df_pack_layers5<<<dim3((layer_width + 255) / 256, nb), 256, 0, S.st>>>(
                    layer0, layer1, layer2, layer3, layer4, S.feat_d, layer_width, nb);
            df_sanitize<<<(nb * DF_EMBD + 255) / 256, 256, 0, S.st>>>(S.feat_d, nb * DF_EMBD);
        } else {
            DFC(hipMemcpyAsync(S.feat_d, feats + (size_t) off * 25600,
                               (size_t) nb * 25600 * sizeof(float), hipMemcpyHostToDevice, S.st));
        }
        const int pos_base = poss[off];
        if (S.q8_process) {
            df_quantize_q8_1<<<dim3(25600 / 256, nb), 256, 0, S.st>>>(
                    S.feat_d, S.q8_feat, 25600);
        }
        DISPATCH_Q4K_ANY_NATIVE(nb, S.W_fc, S.feat_d, S.q8_feat, S.henc, 25600, DF_EMBD, S.q8_process, S.q8_native_mmvq, S.st);
        if (dbg && getenv("LLAMA_SPEC_HIP_DFLASH_TRACE_PROCESS") != nullptr) {
            static bool compared_process = false;
            if (!compared_process) {
                compared_process = true;
                DISPATCH_Q4K(nb, S.W_fc, S.feat_d, S.resid, 25600, DF_EMBD, S.st);
                float h_q8[8] = {}, h_f32[8] = {};
                DFC(hipStreamSynchronize(S.st));
                DFC(hipMemcpy(h_q8, S.henc, sizeof(h_q8), hipMemcpyDeviceToHost));
                DFC(hipMemcpy(h_f32, S.resid, sizeof(h_f32), hipMemcpyDeviceToHost));
                std::vector<float> h_input((size_t) nb * 25600);
                DFC(hipMemcpy(h_input.data(), S.feat_d, h_input.size() * sizeof(float), hipMemcpyDeviceToHost));
                FILE * input_file = std::fopen("/tmp/dflash-sidecar-fc-input.bin", "wb");
                if (input_file != nullptr) {
                    std::fwrite(h_input.data(), sizeof(float), h_input.size(), input_file);
                    std::fclose(input_file);
                }
                if (S.q8_process) {
                    std::vector<df_block_q8_1> h_q8_input((size_t) nb * (25600 / 32));
                    DFC(hipMemcpy(h_q8_input.data(), S.q8_feat,
                                  h_q8_input.size() * sizeof(df_block_q8_1), hipMemcpyDeviceToHost));
                    FILE * q8_file = std::fopen("/tmp/dflash-sidecar-fc-q8.bin", "wb");
                    if (q8_file != nullptr) {
                        std::fwrite(h_q8_input.data(), sizeof(df_block_q8_1), h_q8_input.size(), q8_file);
                        std::fclose(q8_file);
                    }
                }
                fprintf(stderr, "[dflash-sidecar] process fc compare q8=%d q8=%.9g,%.9g,%.9g,%.9g f32=%.9g,%.9g,%.9g,%.9g\n",
                        S.q8_process ? 1 : 0, h_q8[0], h_q8[1], h_q8[2], h_q8[3],
                        h_f32[0], h_f32[1], h_f32[2], h_f32[3]);
            }
        }
        df_rmsnorm<<<nb, 256, 0, S.st>>>(S.henc, S.w_enc_norm, S.xn, DF_EMBD);
        if (S.q8_process) {
            df_quantize_q8_1<<<dim3(DF_EMBD / 256, nb), 256, 0, S.st>>>(
                    S.xn, S.q8_xn, DF_EMBD);
        }
        for (int il = 0; il < DF_LAYERS; ++il) {
            DISPATCH_Q4K_ANY_NATIVE(nb, S.L[il].wk, S.xn, S.q8_xn, S.k, DF_EMBD, DF_KV * DF_HDIM, S.q8_process, S.q8_native_mmvq, S.st);
            if (S.L[il].wv_q6) DISPATCH_Q6K_ANY_NATIVE(nb, (const block_q6_K *) S.L[il].wv, S.xn, S.q8_xn, S.v, DF_EMBD, DF_KV * DF_HDIM, S.q8_process, S.q8_native_mmvq, S.st);
            else               DISPATCH_Q4K_ANY_NATIVE(nb, (const block_q4_K *) S.L[il].wv, S.xn, S.q8_xn, S.v, DF_EMBD, DF_KV * DF_HDIM, S.q8_process, S.q8_native_mmvq, S.st);
            df_rmsnorm_head<<<dim3(DF_KV, nb), 128, 0, S.st>>>(S.k, S.L[il].kn, DF_KV);
            df_rope_base<<<dim3(DF_KV, nb), NROT / 2, 0, S.st>>>(S.k, DF_KV, pos_base, NROT);
            df_kv_store_base<<<dim3((DF_KV * DF_HDIM + 255) / 256, nb), 256, 0, S.st>>>(
                    S.k, S.v, q->Kp[il], q->Vp[il], pos_base, q->pending_pos_min, nb);
        }
        if (!device_features) DFC(hipStreamSynchronize(S.st));
    }
    if (device_features) DFC(hipGetLastError());
    if (dbg) {
        DFC(hipStreamSynchronize(S.st));
        auto t1 = std::chrono::high_resolution_clock::now();
        tot_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        n_rows += n;
        if (++n_calls % 25 == 0) {
            fprintf(stderr, "[dflash-sidecar] chunk avg %.2f ms (%.1f rows) over %d calls\n",
                    tot_ms / n_calls, (double) n_rows / n_calls, n_calls);
        }
    }
    return 0;
}

DF_API int spec_dflash_chunk(int32_t seq_id, const int32_t * poss, const float * feats, int n) {
    return spec_dflash_chunk_impl(seq_id, poss, feats, nullptr, 0, 0, false, n);
}

DF_API int spec_dflash_chunk_device(int32_t seq_id, const int32_t * poss,
        const void * const * layer_features_device, int n_layers, int layer_width, int n) {
    return spec_dflash_chunk_impl(seq_id, poss, nullptr, layer_features_device,
                                  n_layers, layer_width, true, n);
}

DF_API int spec_dflash_draft(int32_t seq_id, int32_t id_last, int32_t n_past, int32_t * out_ids) {
    DSeqState * q = get_seq(seq_id);
    if (!S.ok || q == nullptr || !out_ids || id_last < 0 || id_last >= DF_VOCAB) return -1;
    if (n_past < 1 || n_past != q->state_pos_max || q->kv_capacity < n_past) return -1;
    // The hard ceiling is not an error: report an empty draft so the host
    // falls through to the stacked n-gram drafters instead of disabling the
    // sidecar for the rest of the process.
    if (static_cast<uint64_t>(n_past) + DF_BLOCK > static_cast<uint64_t>(S.max_pos)) return 0;
    DFlashDeviceGuard device_guard(S.device);
    if (!device_guard.valid) return -1;
    const int NROT = 128;
    static int n_calls = 0;
    static double tot_ms = 0.0;
    const bool dbg = getenv("LLAMA_SPEC_HIP_DEBUG") != nullptr;
    auto t0 = std::chrono::high_resolution_clock::now();

    int toks[8], poss[8];
    toks[0] = id_last;
    for (int i = 1; i < 8; ++i) toks[i] = 248070;
    for (int i = 0; i < 8; ++i) poss[i] = n_past + i;
    DFC(hipMemcpyAsync(S.tok_d[seq_id], toks, 32, hipMemcpyHostToDevice, S.st));
    DFC(hipMemcpyAsync(S.pos_d[seq_id], poss, 32, hipMemcpyHostToDevice, S.st));

    // one hipGraph for the whole ~205-kernel draft: all kernel args are static
    // device pointers (contents vary per call), n_past is read from pos_d[0]
    int q8_probe_slot = 0;
    if (g_draft_gexec[seq_id] != nullptr) {
        DFC(hipGraphLaunch(g_draft_gexec[seq_id], S.st));
        goto readback;
    }
    DFC(hipStreamBeginCapture(S.st, hipStreamCaptureModeThreadLocal));
    df_embed_rows<<<dim3((DF_EMBD + 255) / 256, DF_BLOCK), 256, 0, S.st>>>(S.W_embd, S.tok_d[seq_id], S.x, DF_EMBD);
    for (int il = 0; il < DF_LAYERS; ++il) {
        auto & L = S.L[il];
        const bool q8_layer = S.q8_all && il < S.q8_layers;
        df_rmsnorm<<<DF_BLOCK, 256, 0, S.st>>>(S.x, L.attn_norm, S.xn, DF_EMBD);
        if (q8_layer) {
            df_quantize_q8_1<<<dim3(DF_EMBD / 256, DF_BLOCK), 256, 0, S.st>>>(
                    S.xn, S.q8_xn, DF_EMBD);
            if (S.q8_process) df_probe_f32<<<1, 1, 0, S.st>>>(S.xn, S.q8_probe + 4 * q8_probe_slot++, DF_BLOCK * DF_EMBD);
        }
        DISPATCH_Q4K_ANY_NATIVE(DF_BLOCK, L.conv_a_proj, S.xn, S.q8_xn, S.dyn_a, DF_EMBD, DF_CONVDYN, q8_layer, S.q8_native_mmvq, S.st);
        df_conv_gate<<<dim3((DF_EMBD + 255) / 256, DF_BLOCK), 256, 0, S.st>>>(S.xn, S.dyn_a, L.conv_a_base, S.cg, 0, DF_BLOCK);
        if (q8_layer) {
            df_quantize_q8_1<<<dim3(DF_EMBD / 256, DF_BLOCK), 256, 0, S.st>>>(
                    S.cg, S.q8_cg, DF_EMBD);
            if (S.q8_process) df_probe_f32<<<1, 1, 0, S.st>>>(S.cg, S.q8_probe + 4 * q8_probe_slot++, DF_BLOCK * DF_EMBD);
        }
        DISPATCH_Q4K_ANY_NATIVE(DF_BLOCK, L.wq, S.cg, S.q8_cg, S.q, DF_EMBD, DF_HEADS * DF_HDIM, q8_layer, S.q8_native_mmvq, S.st);
        DISPATCH_Q4K_ANY_NATIVE(DF_BLOCK, L.wk, S.cg, S.q8_cg, S.k, DF_EMBD, DF_KV * DF_HDIM, q8_layer, S.q8_native_mmvq, S.st);
        if (L.wv_q6) DISPATCH_Q6K_ANY_NATIVE(DF_BLOCK, (const block_q6_K *) L.wv, S.cg, S.q8_cg, S.v, DF_EMBD, DF_KV * DF_HDIM, q8_layer, S.q8_native_mmvq, S.st);
        else         DISPATCH_Q4K_ANY_NATIVE(DF_BLOCK, (const block_q4_K *) L.wv, S.cg, S.q8_cg, S.v, DF_EMBD, DF_KV * DF_HDIM, q8_layer, S.q8_native_mmvq, S.st);
        df_rmsnorm_head<<<dim3(DF_HEADS, DF_BLOCK), 128, 0, S.st>>>(S.q, L.qn, DF_HEADS);
        df_rmsnorm_head<<<dim3(DF_KV, DF_BLOCK), 128, 0, S.st>>>(S.k, L.kn, DF_KV);
        df_rope<<<dim3(DF_HEADS, DF_BLOCK), NROT / 2, 0, S.st>>>(S.q, DF_HEADS, S.pos_d[seq_id], NROT);
        df_rope<<<dim3(DF_KV, DF_BLOCK), NROT / 2, 0, S.st>>>(S.k, DF_KV, S.pos_d[seq_id], NROT);
        df_attn<<<dim3(DF_HEADS, DF_BLOCK), 256, 0, S.st>>>(S.q, q->Kc[il], q->Vc[il], S.k, S.v, S.att, n_past, S.pos_d[seq_id]);
        if (q8_layer) {
            df_quantize_q8_1<<<dim3((DF_HEADS * DF_HDIM) / 256, DF_BLOCK), 256, 0, S.st>>>(
                    S.att, S.q8_att, DF_HEADS * DF_HDIM);
            if (S.q8_process) df_probe_f32<<<1, 1, 0, S.st>>>(S.att, S.q8_probe + 4 * q8_probe_slot++, DF_BLOCK * DF_HEADS * DF_HDIM);
        }
        DISPATCH_Q4K_ANY_NATIVE(DF_BLOCK, L.wo, S.att, S.q8_att, S.wo_out, DF_HEADS * DF_HDIM, DF_EMBD, q8_layer, S.q8_native_mmvq, S.st);
        df_conv_gate<<<dim3((DF_EMBD + 255) / 256, DF_BLOCK), 256, 0, S.st>>>(S.wo_out, S.dyn_a, L.conv_a_base, S.cg, 1, DF_BLOCK);
        df_add<<<(DF_BLOCK * DF_EMBD + 255) / 256, 256, 0, S.st>>>(S.x, S.cg, DF_BLOCK * DF_EMBD);
        df_rmsnorm<<<DF_BLOCK, 256, 0, S.st>>>(S.x, L.ffn_norm, S.xn, DF_EMBD);
        if (q8_layer) {
            df_quantize_q8_1<<<dim3(DF_EMBD / 256, DF_BLOCK), 256, 0, S.st>>>(
                    S.xn, S.q8_xn, DF_EMBD);
            if (S.q8_process) df_probe_f32<<<1, 1, 0, S.st>>>(S.xn, S.q8_probe + 4 * q8_probe_slot++, DF_BLOCK * DF_EMBD);
        }
        DISPATCH_Q4K_ANY_NATIVE(DF_BLOCK, L.conv_f_proj, S.xn, S.q8_xn, S.dyn_f, DF_EMBD, DF_CONVDYN, q8_layer, S.q8_native_mmvq, S.st);
        df_conv_gate<<<dim3((DF_EMBD + 255) / 256, DF_BLOCK), 256, 0, S.st>>>(S.xn, S.dyn_f, L.conv_f_base, S.cg, 0, DF_BLOCK);
        if (q8_layer) {
            df_quantize_q8_1<<<dim3(DF_EMBD / 256, DF_BLOCK), 256, 0, S.st>>>(
                    S.cg, S.q8_cg, DF_EMBD);
            if (S.q8_process) df_probe_f32<<<1, 1, 0, S.st>>>(S.cg, S.q8_probe + 4 * q8_probe_slot++, DF_BLOCK * DF_EMBD);
        }
        DISPATCH_Q4K_ANY_NATIVE(DF_BLOCK, L.up, S.cg, S.q8_cg, S.fu, DF_EMBD, DF_FF, q8_layer, S.q8_native_mmvq, S.st);
        DISPATCH_Q4K_ANY_NATIVE(DF_BLOCK, L.gate, S.cg, S.q8_cg, S.fg, DF_EMBD, DF_FF, q8_layer, S.q8_native_mmvq, S.st);
        if (q8_layer) {
            if (S.q8_process) df_probe_f32<<<1, 1, 0, S.st>>>(S.fu, S.q8_probe + 4 * q8_probe_slot++, DF_BLOCK * DF_FF);
            if (S.q8_process) df_probe_f32<<<1, 1, 0, S.st>>>(S.fg, S.q8_probe + 4 * q8_probe_slot++, DF_BLOCK * DF_FF);
        }
        df_silu_mul<<<(DF_BLOCK * DF_FF + 255) / 256, 256, 0, S.st>>>(S.fg, S.fu, DF_BLOCK * DF_FF);
        if (q8_layer) {
            df_quantize_q8_1<<<dim3(DF_FF / 256, DF_BLOCK), 256, 0, S.st>>>(
                    S.fg, S.q8_fg, DF_FF);
            if (S.q8_process) df_probe_f32<<<1, 1, 0, S.st>>>(S.fg, S.q8_probe + 4 * q8_probe_slot++, DF_BLOCK * DF_FF);
        }
        if (L.down_q6) DISPATCH_Q6K_ANY_NATIVE(DF_BLOCK, (const block_q6_K *) L.down, S.fg, S.q8_fg, S.ftmp, DF_FF, DF_EMBD, q8_layer, S.q8_native_mmvq, S.st);
        else           DISPATCH_Q4K_ANY_NATIVE(DF_BLOCK, (const block_q4_K *) L.down, S.fg, S.q8_fg, S.ftmp, DF_FF, DF_EMBD, q8_layer, S.q8_native_mmvq, S.st);
        df_conv_gate<<<dim3((DF_EMBD + 255) / 256, DF_BLOCK), 256, 0, S.st>>>(S.ftmp, S.dyn_f, L.conv_f_base, S.cg, 1, DF_BLOCK);
        df_add<<<(DF_BLOCK * DF_EMBD + 255) / 256, 256, 0, S.st>>>(S.x, S.cg, DF_BLOCK * DF_EMBD);
    }
    df_rmsnorm<<<DF_BLOCK, 256, 0, S.st>>>(S.x, S.w_out_norm, S.tE, DF_EMBD);
    if (S.q8_head) {
        df_quantize_q8_1<<<dim3(DF_EMBD / 256, DF_BLOCK), 256, 0, S.st>>>(
                S.tE, S.tE_q8, DF_EMBD);
        if (S.q8_process) {
            df_probe_f32<<<1, 1, 0, S.st>>>(S.tE, S.q8_probe + 4 * q8_probe_slot++, DF_BLOCK * DF_EMBD);
        }
        DISPATCH_Q6K_ANY_NATIVE(DF_BLOCK, S.W_head, S.tE, S.tE_q8, S.logits, DF_EMBD, S.n_head, true, S.q8_native_mmvq, S.st);
    } else {
        gemv8t_q6_K<DF_BLOCK><<<(S.n_head + 7) / 8, 512, 0, S.st>>>(
                S.W_head, S.tE, S.logits, DF_EMBD, S.n_head);
    }
    if (S.q8_head) {
        DISPATCH_Q4K_ANY_NATIVE(DF_BLOCK, S.W_selhid, S.tE, S.tE_q8, S.selhid, DF_EMBD, DF_RANK, true, S.q8_native_mmvq, S.st);
    } else {
        DISPATCH_Q4K(DF_BLOCK, S.W_selhid, S.tE, S.selhid, DF_EMBD, DF_RANK, S.st);
    }
    for (int s2 = 0; s2 < DF_TOPK; ++s2) {
        dfa_argmax_partial<<<dim3(S.tkparts, DF_BLOCK), 256, 0, S.st>>>(S.logits, S.n_head, S.tk_pmax, S.tk_pidx, S.tkparts);
        dfa_argmax_final_mask<<<DF_BLOCK, 1, 0, S.st>>>(S.logits, S.n_head, S.tk_pmax, S.tk_pidx, S.tkparts, S.tk_ids, S.tk_vals, s2, DF_TOPK);
    }
    {
        hipGraph_t graph = nullptr;
        DFC(hipStreamEndCapture(S.st, &graph));
        DFC(hipGraphInstantiate(&g_draft_gexec[seq_id], graph, nullptr, nullptr, 0));
        S.q8_probe_slots = q8_probe_slot;
        DFC(hipGraphDestroy(graph));
        fprintf(stderr, "[dflash-sidecar] draft graph captured for seq %d\n", seq_id);
        DFC(hipGraphLaunch(g_draft_gexec[seq_id], S.st));
    }
readback:
    int   h_ids[8 * DF_TOPK];
    float h_vals[8 * DF_TOPK];
    float sh[8 * DF_RANK];
    DFC(hipMemcpyAsync(h_ids, S.tk_ids, sizeof(h_ids), hipMemcpyDeviceToHost, S.st));
    DFC(hipMemcpyAsync(h_vals, S.tk_vals, sizeof(h_vals), hipMemcpyDeviceToHost, S.st));
    DFC(hipMemcpyAsync(sh, S.selhid, sizeof(sh), hipMemcpyDeviceToHost, S.st));
    DFC(hipStreamSynchronize(S.st));
    if (dbg && S.q8_all && S.q8_head) {
        static bool trace_q8_values = false;
        if (!trace_q8_values) {
            trace_q8_values = true;
            df_block_q8_1 h_xn[8] = {}, h_cg[8] = {}, h_fg[8] = {}, h_te[8] = {};
            float h_te_f[8] = {}, h_logits[8] = {}, h_probe[60 * 4] = {};
            DFC(hipMemcpy(h_xn, S.q8_xn, sizeof(h_xn), hipMemcpyDeviceToHost));
            DFC(hipMemcpy(h_cg, S.q8_cg, sizeof(h_cg), hipMemcpyDeviceToHost));
            DFC(hipMemcpy(h_fg, S.q8_fg, sizeof(h_fg), hipMemcpyDeviceToHost));
            DFC(hipMemcpy(h_te, S.tE_q8, sizeof(h_te), hipMemcpyDeviceToHost));
            DFC(hipMemcpy(h_te_f, S.tE, sizeof(h_te_f), hipMemcpyDeviceToHost));
            DFC(hipMemcpy(h_logits, S.logits, sizeof(h_logits), hipMemcpyDeviceToHost));
            DFC(hipMemcpy(h_probe, S.q8_probe, sizeof(h_probe), hipMemcpyDeviceToHost));
            fprintf(stderr, "[dflash-sidecar] q8 trace xn d/s=%.6g/%.6g cg=%.6g/%.6g fg=%.6g/%.6g tE=%.6g/%.6g tE_f=%.6g logits=%.6g,%.6g,%.6g,%.6g\n",
                    df_f16(h_xn[0].d), df_f16(h_xn[0].s),
                    df_f16(h_cg[0].d), df_f16(h_cg[0].s),
                    df_f16(h_fg[0].d), df_f16(h_fg[0].s),
                    df_f16(h_te[0].d), df_f16(h_te[0].s),
                    h_te_f[0], h_logits[0], h_logits[1], h_logits[2], h_logits[3]);
            for (int slot = 0; slot < S.q8_probe_slots; ++slot) {
                fprintf(stderr, "[dflash-sidecar] q8 probe slot=%d lo=%.6g hi=%.6g bad=%.0f ratio=%.6g\n",
                        slot, h_probe[slot * 4 + 0], h_probe[slot * 4 + 1],
                        h_probe[slot * 4 + 2], h_probe[slot * 4 + 3]);
            }
        }
    }
    std::memcpy(S.last_tk_ids, h_ids, sizeof(h_ids));
    std::memcpy(S.last_tk_vals, h_vals, sizeof(h_vals));
    std::memcpy(S.last_selhid, sh, sizeof(sh));
    if (dbg) {
        static int trace_blocks = 0;
        if (trace_blocks++ < 1) {
            for (int t = 1; t < 4; ++t) {
                fprintf(stderr, "[dflash-sidecar] raw topk t=%d", t);
                for (int kx = 0; kx < 8; ++kx) {
                    fprintf(stderr, " %d:%.6g", h_ids[t * DF_TOPK + kx], h_vals[t * DF_TOPK + kx]);
                }
                fprintf(stderr, "\n");
            }
        }
    }

    int pred_idx = 0;
    int n_out = 0;
    for (int t = 1; t < 8; ++t) {
        const float * hid = sh + (size_t) t * DF_RANK;
        const int ps = (t == 1) ? -1 : h_ids[(t - 1) * DF_TOPK + pred_idx];
        const int pred_id = (t == 1) ? id_last : ((ps >= 0 && ps < S.n_head) ? S.remap[ps] : -1);
        if (pred_id < 0 || pred_id >= DF_VOCAB) break;
        const float * pv = S.sel_prev.data() + (size_t) pred_id * DF_RANK;
        float best = -1e30f; int besti = -1;
        for (int kx = 0; kx < DF_TOPK; ++kx) {
            const int cs = h_ids[t * DF_TOPK + kx];
            if (cs < 0 || cs >= S.n_head) continue;
            const int cid = S.remap[cs];
            if (cid < 0 || cid >= DF_VOCAB) continue;
            const float * nx = S.sel_next.data() + (size_t) cid * DF_RANK;
            float s2 = 0.0f;
            for (int rr = 0; rr < DF_RANK; ++rr) s2 += nx[rr] * (pv[rr] * hid[rr]);
            s2 += h_vals[t * DF_TOPK + kx];
            if (s2 == s2 && (besti < 0 || s2 > best)) { best = s2; besti = kx; }
        }
        if (besti < 0) break;
        pred_idx = besti;
        const int ts2 = h_ids[t * DF_TOPK + besti];
        if (ts2 < 0 || ts2 >= S.n_head) break;
        const int tokid = S.remap[ts2];
        if (tokid < 0 || tokid >= DF_VOCAB) break;
        out_ids[n_out++] = tokid;
    }
    if (dbg) {
        auto t1 = std::chrono::high_resolution_clock::now();
        tot_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (++n_calls % 25 == 0) {
            fprintf(stderr, "[dflash-sidecar] draft avg %.2f ms over %d calls\n", tot_ms / n_calls, n_calls);
        }
    }
    return n_out;
}

// Stochastic DFlash reuses the same single graph and compact top-k readback as
// greedy DFlash. The selector scores are reconstructed for the sampled
// predecessor on the host (7 * 16 * 256 multiply-adds), avoiding both a full
// vocabulary transfer and a second decoder execution.
DF_API int spec_dflash_stochastic_top_k() {
    return DF_TOPK;
}

DF_API int spec_dflash_draft_stochastic(int32_t seq_id, int32_t id_last, int32_t n_past,
        float temperature, float p_min, uint64_t rng_key, int max_draft,
        int32_t * out_ids, int32_t * out_dist_ids, float * out_dist_probs) {
    if (!S.ok || max_draft < 1 || max_draft > 7 || out_ids == nullptr ||
            out_dist_ids == nullptr || out_dist_probs == nullptr ||
            !std::isfinite(temperature) || temperature <= 0.0f ||
            !std::isfinite(p_min) || p_min < 0.0f) {
        return -1;
    }

    int32_t greedy_ids[7] = {};
    const int greedy_rc = spec_dflash_draft(seq_id, id_last, n_past, greedy_ids);
    if (greedy_rc < 0) {
        return -1;
    }
    if (greedy_rc == 0) {
        // Hard position ceiling: no draft this cycle; S.last_* is stale.
        return 0;
    }

    int pred_idx = 0;
    int n_out = 0;
    for (int t = 1; t < 8 && n_out < max_draft; ++t) {
        const int ps = t == 1 ? -1 : S.last_tk_ids[(t - 1) * DF_TOPK + pred_idx];
        const int pred_id = t == 1 ? id_last : ((ps >= 0 && ps < S.n_head) ? S.remap[ps] : -1);
        if (pred_id < 0 || pred_id >= DF_VOCAB) {
            break;
        }

        const float * hid = S.last_selhid + (size_t) t * DF_RANK;
        const float * pv = S.sel_prev.data() + (size_t) pred_id * DF_RANK;
        float scores[DF_TOPK] = {};
        float probs[DF_TOPK] = {};
        float max_score = -1e30f;
        int valid = 0;
        for (int kx = 0; kx < DF_TOPK; ++kx) {
            const int cs = S.last_tk_ids[t * DF_TOPK + kx];
            if (cs < 0 || cs >= S.n_head) {
                continue;
            }
            const int cid = S.remap[cs];
            if (cid < 0 || cid >= DF_VOCAB) {
                continue;
            }
            const float * nx = S.sel_next.data() + (size_t) cid * DF_RANK;
            float score = S.last_tk_vals[t * DF_TOPK + kx];
            for (int rr = 0; rr < DF_RANK; ++rr) {
                score += nx[rr] * (pv[rr] * hid[rr]);
            }
            if (!std::isfinite(score)) {
                continue;
            }
            scores[kx] = score;
            max_score = std::max(max_score, score);
            ++valid;
        }
        if (valid == 0) {
            break;
        }
        float sum = 0.0f;
        for (int kx = 0; kx < DF_TOPK; ++kx) {
            const int cs = S.last_tk_ids[t * DF_TOPK + kx];
            const int cid = cs >= 0 && cs < S.n_head ? S.remap[cs] : -1;
            if (cid < 0 || cid >= DF_VOCAB || !std::isfinite(scores[kx])) {
                probs[kx] = 0.0f;
                continue;
            }
            probs[kx] = std::exp((scores[kx] - max_score) / temperature);
            sum += probs[kx];
        }
        if (!(sum > 0.0f) || !std::isfinite(sum)) {
            break;
        }
        int selected = DF_TOPK - 1;
        double cumulative = 0.0;
        const double u = spec_sidecar_stochastic_uniform(rng_key, (uint32_t) n_out);
        for (int kx = 0; kx < DF_TOPK; ++kx) {
            probs[kx] /= sum;
            cumulative += probs[kx];
            if (u < cumulative) {
                selected = kx;
                break;
            }
        }
        const int cs = S.last_tk_ids[t * DF_TOPK + selected];
        const int cid = cs >= 0 && cs < S.n_head ? S.remap[cs] : -1;
        if (cid < 0 || cid >= DF_VOCAB || probs[selected] < p_min) {
            break;
        }
        out_ids[n_out] = cid;
        for (int kx = 0; kx < DF_TOPK; ++kx) {
            const int candidate = S.last_tk_ids[t * DF_TOPK + kx];
            out_dist_ids[(size_t) n_out * DF_TOPK + kx] =
                    candidate >= 0 && candidate < S.n_head ? S.remap[candidate] : -1;
            out_dist_probs[(size_t) n_out * DF_TOPK + kx] = probs[kx];
        }
        ++n_out;
        pred_idx = selected;
    }
    return n_out;
}
