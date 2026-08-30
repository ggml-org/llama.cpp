// SPDX-License-Identifier: MIT
// HIP draft sidecar: C API around the MTP-block kernels, built as a DLL that
// llama.cpp's speculative driver loads at runtime (LLAMA_SPEC_HIP_SIDECAR).
// Release ABI 4 adds lazy accepted-prefix commits, device-input paths,
// sequence-scoped state operations, and compact stochastic top-k proposals.
// The harness build (-DSIDECAR_HARNESS) replays golden.bin through the same
// sequence-scoped lazy-catchup/commit API.
#include "kernels.hip"
#include "kernels_b.hip"
#include "catchup_alignment.h"
#include "../artifact_manifest.h"
#include "spec_sidecar/sidecar_abi.h"

#if defined(__has_include)
#    if __has_include(<rocprim/device/device_topk.hpp>)
#        include <rocprim/device/device_topk.hpp>
#        include <rocprim/iterator/counting_iterator.hpp>
#        define SPEC_SIDECAR_ROCPRIM_TOPK
#    endif
#endif

// ---------------- host: weights ----------------
using Tensor = spec_sidecar_artifact::TensorDesc;
static constexpr int SC_N_VOCAB = 248320;

// ---------------- sidecar state ----------------
namespace {
#define N_SCR 4
#define SC_MAX_SEQ 8
struct Scratch {
    float *x_tok, *x_h, *e_n, *h_n, *cat, *proj, *attin, *qfull, *q, *gate, *k, *v, *attout,
          *ffn_in, *ffn_g, *ffn_u, *ffn_tmp, *resid, *hnorm2;
#if defined(__gfx1030__) || defined(SPEC_SIDECAR_GFX1030)
    block_q8_1 *q8;
    block_q8_1_sum_hi *q8_sum_hi;
#endif
    int * id_dev;
};
#define N_BATCH 8
struct BatchScratch { // packed [N_BATCH][...] buffers for batched catch-up
    float *x_tok, *e_n, *h_n, *cat, *proj, *attin, *qfull, *q, *gate, *k, *v, *attout,
          *ffn_in, *ffn_g, *ffn_u, *ffn_tmp, *resid;
};
struct SeqState {
    __half * Kc = nullptr;
    __half * Vc = nullptr;
    __half * Kp = nullptr; // pending target rows; never read by draft()
    __half * Vp = nullptr;
    float  * Hp = nullptr; // host-path target h rows retained until commit
    float  * h_committed = nullptr;
    int pending_capacity = 0;
    int pending_h_capacity = 0;
    int32_t pending_pos_min = 0;
    int32_t pending_pos_max = 0;
    int32_t state_pos_min = 0;
    int32_t state_pos_max = 0; // exclusive; only committed target rows
    uint64_t state_epoch = 1;
    bool h_valid = false;
    bool allocated = false;
};
struct State {
    bool ok = false;
    std::vector<Tensor> tensors;
    int32_t n_seq = 1;
    int device = -1;
    hipStream_t st = nullptr;       // active target/draft stream
    hipStream_t owned_st = nullptr; // fallback stream owned by the sidecar
    bool owns_st = true;
    Scratch scr[N_SCR];             // shared serialized scratch; scr[0] is draft scratch
    BatchScratch bs;
    float *part_m = nullptr, *part_l = nullptr, *part_acc = nullptr; // split-K attention partials
    float *logits = nullptr, *pmax = nullptr;
    float *draft_logits = nullptr, *draft_probs = nullptr;
    int *pidx = nullptr, *tok_out = nullptr;
    int *draft_ids = nullptr, *draft_count = nullptr, *draft_active = nullptr;
#if defined(SPEC_SIDECAR_ROCPRIM_TOPK)
    void *topk_temp = nullptr;
    size_t topk_temp_bytes = 0;
    bool topk_rocprim = false;
#endif
    int nparts = 0;
    size_t max_pos = 131072;
    SeqState seq[SC_MAX_SEQ];
    hipEvent_t ev_a = nullptr, ev_b = nullptr; // GPU-side timing (debug)
    // weights
    const block_q4_0 *W_embd, *W_ehp, *W_q, *W_k, *W_v, *W_o, *W_fg, *W_fu, *W_fd, *W_head;
    const float *w_enorm, *w_hnorm, *w_attn, *w_qn, *w_kn, *w_post, *w_shn;
    int * remap_d = nullptr;
};
State S;

static SeqState * get_seq(int32_t seq_id) {
    return seq_id >= 0 && seq_id < S.n_seq && seq_id < SC_MAX_SEQ ? &S.seq[seq_id] : nullptr;
}

Tensor * find_tensor(const char * name) {
    return spec_sidecar_artifact::find_tensor(S.tensors, name);
}

#define SC_CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "[hip-sidecar] %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); return -1; } } while (0)

// Move a contiguous position range in-place. A single block and an explicit
// direction make overlapping moves deterministic without allocating a second
// context-sized KV buffer. Rebase is rare (context shift); the draft hot path
// never calls this kernel.
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

#if defined(__gfx1030__) || defined(SPEC_SIDECAR_GFX1030)
static bool sidecar_gfx1030_gemv_enabled() {
    static const bool enabled = [] {
        const char * value = getenv("LLAMA_SPEC_HIP_GFX1030_GEMV");
        return value == nullptr || (std::atoi(value) != 0);
    }();
    return enabled;
}
#endif

// phase 1: inputs -> projection -> Q/K/V -> rope -> KV store (rows independent)
void block_phase1(Scratch & c, hipStream_t st, const float * xh_in, int pos,
        __half * Kc, __half * Vc) {
    auto & s = S;
    rmsnorm<<<1, 256, 0, st>>>(c.x_tok, s.w_enorm, c.e_n, N_EMBD);
    rmsnorm<<<1, 256, 0, st>>>(xh_in,   s.w_hnorm, c.h_n, N_EMBD);
    concat2<<<(2 * N_EMBD + 255) / 256, 256, 0, st>>>(c.e_n, c.h_n, c.cat, N_EMBD);
    gemv_q4_0<<<N_EMBD, 32, 0, st>>>(s.W_ehp, c.cat, c.proj, 2 * N_EMBD, N_EMBD);
    rmsnorm<<<1, 256, 0, st>>>(c.proj, s.w_attn, c.attin, N_EMBD);
    gemv_q4_0<<<2 * N_HEAD * HEAD_DIM, 32, 0, st>>>(s.W_q, c.attin, c.qfull, N_EMBD, 2 * N_HEAD * HEAD_DIM);
    gemv_q4_0<<<N_HEAD_KV * HEAD_DIM, 32, 0, st>>>(s.W_k, c.attin, c.k, N_EMBD, N_HEAD_KV * HEAD_DIM);
    gemv_q4_0<<<N_HEAD_KV * HEAD_DIM, 32, 0, st>>>(s.W_v, c.attin, c.v, N_EMBD, N_HEAD_KV * HEAD_DIM);
    split_q_gate<<<N_HEAD, HEAD_DIM, 0, st>>>(c.qfull, c.q, c.gate);
    rmsnorm_heads<<<N_HEAD, 256, 0, st>>>(c.q, s.w_qn, c.q, N_HEAD);
    rmsnorm_heads<<<N_HEAD_KV, 256, 0, st>>>(c.k, s.w_kn, c.k, N_HEAD_KV);
    rope_neox<<<N_HEAD, N_ROT / 2, 0, st>>>(c.q, N_HEAD, pos);
    rope_neox<<<N_HEAD_KV, N_ROT / 2, 0, st>>>(c.k, N_HEAD_KV, pos);
    kv_store<<<(N_HEAD_KV * HEAD_DIM + 255) / 256, 256, 0, st>>>(c.k, c.v, Kc, Vc, pos);
}

// phase 2: attention (reads global KV, all rows' stores must be complete) -> ffn
void block_phase2(Scratch & c, hipStream_t st, int pos, int * tok_dst,
        __half * Kc, __half * Vc, bool emit_logits = false) {
    auto & s = S;
    // split-K attention (winner of v3/v4/v5 shootout): workgroup per (q-head,
    // 4096-position chunk). The GQA shared-K variants lost to LDS pressure.
    const int att_parts = (pos + 1 + ATTN_SPLIT - 1) / ATTN_SPLIT;
    attn_1q_split<<<dim3(N_HEAD, att_parts), 256, 0, st>>>(c.q, Kc, Vc, pos + 1, s.part_m, s.part_l, s.part_acc);
    attn_merge<<<N_HEAD, HEAD_DIM, 0, st>>>(s.part_m, s.part_l, s.part_acc, c.attout, att_parts);
    sigmoid_gate<<<(N_HEAD * HEAD_DIM + 255) / 256, 256, 0, st>>>(c.attout, c.gate, N_HEAD * HEAD_DIM);
    gemv_q4_0<<<N_EMBD, 32, 0, st>>>(s.W_o, c.attout, c.resid, N_HEAD * HEAD_DIM, N_EMBD);
    add_inplace<<<(N_EMBD + 255) / 256, 256, 0, st>>>(c.resid, c.proj, N_EMBD);
#if defined(__gfx1030__) || defined(SPEC_SIDECAR_GFX1030)
    if (sidecar_gfx1030_gemv_enabled()) {
        rmsnorm_q8_1<<<1, 256, 0, st>>>(c.resid, s.w_post, c.ffn_in,
                                        c.q8, c.q8_sum_hi, N_EMBD);
        // W_fg and W_fu consume the same normalized activation. Quantize once
        // and keep the Q8_1 payload resident for both sibling projections.
        gemv_q4_0_q8_1_dot8<<<N_FF, 32, 0, st>>>(s.W_fg, c.q8, c.q8_sum_hi,
                                                 c.ffn_g, N_EMBD, N_FF);
        gemv_q4_0_q8_1_dot8<<<N_FF, 32, 0, st>>>(s.W_fu, c.q8, c.q8_sum_hi,
                                                 c.ffn_u, N_EMBD, N_FF);
    } else
#endif
    {
        rmsnorm<<<1, 256, 0, st>>>(c.resid, s.w_post, c.ffn_in, N_EMBD);
        gemv_q4_0<<<N_FF, 32, 0, st>>>(s.W_fg, c.ffn_in, c.ffn_g, N_EMBD, N_FF);
        gemv_q4_0<<<N_FF, 32, 0, st>>>(s.W_fu, c.ffn_in, c.ffn_u, N_EMBD, N_FF);
    }
    silu_mul<<<(N_FF + 255) / 256, 256, 0, st>>>(c.ffn_g, c.ffn_u, N_FF);
#if defined(__gfx1030__) || defined(SPEC_SIDECAR_GFX1030)
    if (sidecar_gfx1030_gemv_enabled()) {
        quantize_q8_1<<<1, 256, 0, st>>>(c.ffn_g, c.q8, c.q8_sum_hi, N_FF);
        gemv_q4_0_q8_1_dot8<<<N_EMBD, 32, 0, st>>>(s.W_fd, c.q8, c.q8_sum_hi,
                                                    c.ffn_tmp, N_FF, N_EMBD);
    } else
#endif
    {
        gemv_q4_0<<<N_EMBD, 32, 0, st>>>(s.W_fd, c.ffn_g, c.ffn_tmp, N_FF, N_EMBD);
    }
    add_inplace<<<(N_EMBD + 255) / 256, 256, 0, st>>>(c.resid, c.ffn_tmp, N_EMBD);
    if (tok_dst != nullptr || emit_logits) {
#if defined(__gfx1030__) || defined(SPEC_SIDECAR_GFX1030)
        if (sidecar_gfx1030_gemv_enabled()) {
            // Keep hnorm2 in float for the next autoregressive step while
            // producing the Q8_1 activation needed by the shared head.
            rmsnorm_q8_1<<<1, 256, 0, st>>>(c.resid, s.w_shn, c.hnorm2,
                                            c.q8, c.q8_sum_hi, N_EMBD);
            gemv_q4_0_q8_1_dot8<<<N_HEAD_ROWS, 32, 0, st>>>(s.W_head, c.q8,
                                                            c.q8_sum_hi, s.logits,
                                                            N_EMBD, N_HEAD_ROWS);
        } else
#endif
        {
            rmsnorm<<<1, 256, 0, st>>>(c.resid, s.w_shn, c.hnorm2, N_EMBD);
            gemv_q4_0<<<N_HEAD_ROWS, 32, 0, st>>>(s.W_head, c.hnorm2, s.logits, N_EMBD, N_HEAD_ROWS);
        }
        if (tok_dst != nullptr) {
            argmax_partial<<<s.nparts, 256, 0, st>>>(s.logits, N_HEAD_ROWS, s.pmax, s.pidx);
            argmax_final<<<1, 1, 0, st>>>(s.pmax, s.pidx, s.nparts, N_HEAD_ROWS,
                                          s.remap_d, SC_N_VOCAB, tok_dst);
        }
    }
}
} // namespace

#ifdef _WIN32
#define SC_API extern "C" __declspec(dllexport)
#else
#define SC_API extern "C"
#endif

static constexpr int32_t SC_RELEASE_ABI = 4;

static_assert(MTP_DRAFT_TOP_K == SPEC_SIDECAR_MTP_DRAFT_TOP_K,
              "MTP stochastic top-k ABI mismatch");

static bool valid_seq(int32_t seq_id) {
    return seq_id >= 0 && seq_id < S.n_seq && seq_id < SC_MAX_SEQ;
}

static SeqState * get_seq_checked(int32_t seq_id) {
    return valid_seq(seq_id) ? &S.seq[seq_id] : nullptr;
}

static int ensure_seq_storage(SeqState & q) {
    if (q.allocated) {
        return 0;
    }
    const size_t kv_bytes = S.max_pos * N_HEAD_KV * HEAD_DIM * sizeof(__half);
    SC_CHECK(hipMalloc(&q.Kc, kv_bytes));
    SC_CHECK(hipMalloc(&q.Vc, kv_bytes));
    SC_CHECK(hipMalloc(&q.h_committed, N_EMBD * sizeof(float)));
    q.allocated = true;
    return 0;
}

static int ensure_pending_storage(SeqState & q, int count) {
    if (count <= q.pending_capacity) {
        return 0;
    }
    if (q.pending_pos_max != q.state_pos_max) {
        fprintf(stderr, "[hip-sidecar] cannot resize pending KV while a commit is in flight\n");
        return -1;
    }
    SC_CHECK(hipStreamSynchronize(S.st));
    if (q.Kp != nullptr) SC_CHECK(hipFree(q.Kp));
    if (q.Vp != nullptr) SC_CHECK(hipFree(q.Vp));
    const int capacity = std::max(count, std::max(8, q.pending_capacity * 2));
    const size_t kv_bytes = (size_t) capacity * N_HEAD_KV * HEAD_DIM * sizeof(__half);
    SC_CHECK(hipMalloc(&q.Kp, kv_bytes));
    SC_CHECK(hipMalloc(&q.Vp, kv_bytes));
    q.pending_capacity = capacity;
    return 0;
}

static int ensure_pending_h_storage(SeqState & q, int count) {
    if (count <= q.pending_h_capacity) {
        return 0;
    }
    SC_CHECK(hipStreamSynchronize(S.st));
    if (q.Hp != nullptr) SC_CHECK(hipFree(q.Hp));
    const int capacity = std::max(count, std::max(8, q.pending_h_capacity * 2));
    SC_CHECK(hipMalloc(&q.Hp, (size_t) capacity * N_EMBD * sizeof(float)));
    q.pending_h_capacity = capacity;
    return 0;
}

static void discard_pending(SeqState & q) {
    q.pending_pos_min = q.state_pos_max;
    q.pending_pos_max = q.state_pos_max;
}

SC_API int spec_hip_release_abi() {
    return SC_RELEASE_ABI;
}

// ABI/model handshake: the host passes what it believes the model is; refuse
// to run against a mismatched drafter (wrong arch/width/head slice/sequence count).
SC_API int spec_hip_check(int32_t n_embd, int32_t head_rows, int32_t n_seq) {
    return (n_embd == N_EMBD && head_rows == N_HEAD_ROWS && n_seq >= 1 && n_seq <= SC_MAX_SEQ) ? 0 : -1;
}

SC_API int spec_hip_state_size() {
    return (int) sizeof(spec_sidecar_state);
}

SC_API int spec_hip_get_state(int32_t seq_id, void * data, int size) {
    SeqState * q = get_seq_checked(seq_id);
    if (!S.ok || q == nullptr || data == nullptr || size != (int) sizeof(spec_sidecar_state)) {
        return -1;
    }
    spec_sidecar_state state = {};
    state.magic   = SPEC_SIDECAR_STATE_MAGIC;
    state.version = SPEC_SIDECAR_STATE_VERSION;
    state.kind    = SPEC_SIDECAR_STATE_KIND_MTP;
    state.pos_min = q->state_pos_min;
    state.pos_max = q->state_pos_max;
    state.epoch   = q->state_epoch;
    std::memcpy(data, &state, sizeof(state));
    return 0;
}

static bool spec_hip_state_matches(int32_t seq_id, const void * data, int size,
        spec_sidecar_state & state) {
    SeqState * q = get_seq_checked(seq_id);
    if (!S.ok || q == nullptr || data == nullptr || size != (int) sizeof(state)) {
        return false;
    }
    std::memcpy(&state, data, sizeof(state));
    return state.magic == SPEC_SIDECAR_STATE_MAGIC &&
           state.version == SPEC_SIDECAR_STATE_VERSION &&
           state.kind == SPEC_SIDECAR_STATE_KIND_MTP &&
           state.epoch == q->state_epoch &&
           state.pos_min == q->state_pos_min &&
           state.pos_min >= 0 && state.pos_min <= state.pos_max &&
           state.pos_max <= q->state_pos_max &&
           state.pos_max <= (int32_t) S.max_pos;
}

SC_API int spec_hip_set_state(int32_t seq_id, const void * data, int size) {
    SeqState * q = get_seq_checked(seq_id);
    spec_sidecar_state state = {};
    if (q == nullptr || !spec_hip_state_matches(seq_id, data, size, state)) {
        return -1;
    }
    q->h_valid = q->h_valid && spec_sidecar_mtp::committed_hidden_matches_tip(
            q->state_pos_max, state.pos_max);
    q->state_pos_max = state.pos_max;
    discard_pending(*q);
    return 0;
}

SC_API int spec_hip_reset_state(int32_t seq_id) {
    SeqState * q = get_seq_checked(seq_id);
    if (!S.ok || q == nullptr) {
        return -1;
    }
    if (++q->state_epoch == 0) {
        q->state_epoch = 1;
    }
    q->state_pos_min = 0;
    q->state_pos_max = 0;
    q->h_valid = false;
    discard_pending(*q);
    return 0;
}

// Discard a logically speculative suffix without copying it. Increasing the
// cursor is deliberately reserved for commit_state().
SC_API int spec_hip_truncate_state(int32_t seq_id, int32_t pos_max) {
    SeqState * q = get_seq_checked(seq_id);
    if (!S.ok || q == nullptr || pos_max < q->state_pos_min || pos_max > q->state_pos_max) {
        return -1;
    }
    q->h_valid = q->h_valid && spec_sidecar_mtp::committed_hidden_matches_tip(
            q->state_pos_max, pos_max);
    q->state_pos_max = pos_max;
    discard_pending(*q);
    return 0;
}

SC_API int spec_hip_commit_state(int32_t seq_id, int32_t pos_max, const float * hidden_device) {
    SeqState * q = get_seq_checked(seq_id);
    if (!S.ok || q == nullptr || pos_max < q->state_pos_max ||
            pos_max > q->pending_pos_max || pos_max > (int32_t) S.max_pos) {
        return -1;
    }
    if (pos_max > q->state_pos_max) {
        if (!q->allocated || q->Kp == nullptr || q->Vp == nullptr ||
                q->pending_pos_min != q->state_pos_max) {
            return -1;
        }
        const size_t src_row = (size_t) (q->state_pos_max - q->pending_pos_min);
        const size_t n_rows = (size_t) (pos_max - q->state_pos_max);
        const size_t row_bytes = (size_t) N_HEAD_KV * HEAD_DIM * sizeof(__half);
        SC_CHECK(hipMemcpyAsync(q->Kc + (size_t) q->state_pos_max * N_HEAD_KV * HEAD_DIM,
                                q->Kp + src_row * N_HEAD_KV * HEAD_DIM,
                                n_rows * row_bytes, hipMemcpyDeviceToDevice, S.st));
        SC_CHECK(hipMemcpyAsync(q->Vc + (size_t) q->state_pos_max * N_HEAD_KV * HEAD_DIM,
                                q->Vp + src_row * N_HEAD_KV * HEAD_DIM,
                                n_rows * row_bytes, hipMemcpyDeviceToDevice, S.st));
        if (q->h_committed == nullptr) return -1;
        // Preserve the target hidden row at the accepted boundary. The direct
        // path uses the target-owned pointer; only the host fallback retains a
        // temporary device copy of all pending hidden rows.
        if (hidden_device != nullptr) {
            SC_CHECK(hipMemcpyAsync(q->h_committed, hidden_device,
                                    N_EMBD * sizeof(float), hipMemcpyDeviceToDevice, S.st));
        } else {
            const size_t hidden_row = (size_t) (pos_max - 1 - q->pending_pos_min);
            if (q->Hp == nullptr || hidden_row >= (size_t) q->pending_h_capacity) return -1;
            SC_CHECK(hipMemcpyAsync(q->h_committed, q->Hp + hidden_row * N_EMBD,
                                    N_EMBD * sizeof(float), hipMemcpyDeviceToDevice, S.st));
        }
        q->h_valid = true;
        q->state_pos_max = pos_max;
    }
    discard_pending(*q);
    return 0;
}

SC_API int spec_hip_rebase_state(int32_t seq_id, int32_t pos_min, int32_t pos_max, int32_t delta) {
    SeqState * q = get_seq_checked(seq_id);
    if (!S.ok || q == nullptr || pos_min < 0 || pos_min > pos_max || pos_max > (int32_t) S.max_pos ||
            q->pending_pos_max != q->state_pos_max) {
        return -1;
    }
    if (delta == 0 || q->state_pos_max <= pos_min) {
        if (delta != 0 && ++q->state_epoch == 0) {
            q->state_epoch = 1;
        }
        return 0;
    }
    if (pos_max < q->state_pos_max || !q->allocated) {
        return -1;
    }

    const int64_t new_min = (int64_t) pos_min + delta;
    const int64_t new_max = (int64_t) q->state_pos_max + delta;
    if (new_min < 0 || new_max < new_min || new_max > (int64_t) S.max_pos) {
        return -1;
    }

    SC_CHECK(hipStreamSynchronize(S.st));
    shift_kv_rows<<<1, 256, 0, S.st>>>(q->Kc, pos_min, q->state_pos_max, delta, N_HEAD_KV * HEAD_DIM);
    SC_CHECK(hipGetLastError());
    shift_kv_rows<<<1, 256, 0, S.st>>>(q->Vc, pos_min, q->state_pos_max, delta, N_HEAD_KV * HEAD_DIM);
    SC_CHECK(hipGetLastError());
    SC_CHECK(hipStreamSynchronize(S.st));

    if (q->state_pos_min >= pos_min) {
        q->state_pos_min += delta;
    }
    q->state_pos_max = (int32_t) new_max;
    discard_pending(*q);
    if (++q->state_epoch == 0) {
        q->state_epoch = 1;
    }
    return 0;
}

SC_API int spec_hip_attach_target_stream(void * stream, int32_t device) {
    if (!S.ok || device != S.device) {
        if (getenv("LLAMA_SPEC_HIP_DEBUG") != nullptr) {
            fprintf(stderr, "[hip-sidecar] target stream device mismatch: sidecar=%d target=%d\n",
                    S.device, device);
        }
        return -1;
    }
    if (stream == nullptr) {
        if (S.st != S.owned_st) {
            SC_CHECK(hipStreamSynchronize(S.st));
            S.st = S.owned_st;
            S.owns_st = true;
        }
        return 0;
    }
    const hipStream_t target = reinterpret_cast<hipStream_t>(stream);
    if (target != S.st) {
        SC_CHECK(hipStreamSynchronize(S.st));
        S.st = target;
        S.owns_st = false;
    }
    return 0;
}

// init: weights_dir holds drafter_manifest.json + drafter_weights.bin;
// ids_path = little-endian i32 slice->full-vocab remap table (40960 entries).
// requested_device >= 0 binds all sidecar allocations and streams to that
// target device. -1 preserves the legacy "current device" behavior for the
// standalone harness and older callers.
static int spec_hip_init_impl(const char * weights_dir, const char * ids_path,
        int32_t n_seq, int32_t requested_device) {
    if (requested_device < -1) return -1;
    if (S.ok) {
        return n_seq == S.n_seq &&
                (requested_device < 0 || requested_device == S.device) ? 0 : -1;
    }
    if (n_seq < 1 || n_seq > SC_MAX_SEQ) return -1;
    S.n_seq = n_seq;
    if (requested_device >= 0) {
        SC_CHECK(hipSetDevice(requested_device));
    }
    SC_CHECK(hipGetDevice(&S.device));
    if (requested_device >= 0 && S.device != requested_device) return -1;
    char pbuf[1024];
    std::string validation_error;
    if (!spec_sidecar_artifact::make_path(pbuf, sizeof(pbuf), weights_dir, "drafter_manifest.json")) {
        fprintf(stderr, "[hip-sidecar] invalid or overlong weights directory\n");
        return -1;
    }
    if (!spec_sidecar_artifact::load_manifest(pbuf, S.tensors, validation_error)) {
        fprintf(stderr, "[hip-sidecar] invalid manifest %s: %s\n", pbuf, validation_error.c_str());
        return -1;
    }

    // model-identity validation: exact expected tensor set for the Qwen3.8-27B
    // nextn drafter. Any name/dtype/shape/size mismatch refuses init.
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
    if (S.tensors.size() != sizeof(EXPECTED)/sizeof(EXPECTED[0])) {
        fprintf(stderr, "[hip-sidecar] manifest has %zu tensors, expected %zu\n", S.tensors.size(), sizeof(EXPECTED)/sizeof(EXPECTED[0]));
        return -1;
    }
    for (const auto & e : EXPECTED) {
        Tensor * t = find_tensor(e.name);
        if (!t) { fprintf(stderr, "[hip-sidecar] missing tensor %s\n", e.name); return -1; }
        const bool matches = e.ne1 == 0
            ? spec_sidecar_artifact::tensor_matches(*t, e.dtype, {e.ne0}, e.nbytes)
            : spec_sidecar_artifact::tensor_matches(*t, e.dtype, {e.ne0, e.ne1}, e.nbytes);
        if (!matches) {
            fprintf(stderr, "[hip-sidecar] tensor schema mismatch: %s\n", e.name);
            return -1;
        }
    }

    if (!spec_sidecar_artifact::make_path(pbuf, sizeof(pbuf), weights_dir, "drafter_weights.bin")) return -1;
    FILE * wf = fopen(pbuf, "rb");
    if (!wf) { fprintf(stderr, "[hip-sidecar] cannot open %s\n", pbuf); return -1; }
    uint64_t weights_size = 0;
    if (!spec_sidecar_artifact::file_size(wf, weights_size) ||
        !spec_sidecar_artifact::validate_blob_layout(S.tensors, weights_size, validation_error)) {
        fprintf(stderr, "[hip-sidecar] invalid weights layout: %s\n", validation_error.c_str());
        fclose(wf);
        return -1;
    }

    // Validate the complete little-endian remap before any HIP allocation or
    // artifact upload. A valid sliced head is an injective subset of the target
    // vocabulary, exactly N_HEAD_ROWS entries long.
    std::vector<int32_t> remap_h(N_HEAD_ROWS);
    FILE * rf = ids_path == nullptr ? nullptr : fopen(ids_path, "rb");
    uint64_t remap_size = 0;
    const uint64_t expected_remap_size = static_cast<uint64_t>(N_HEAD_ROWS) * sizeof(int32_t);
    if (!rf || !spec_sidecar_artifact::file_size(rf, remap_size) || remap_size != expected_remap_size ||
        !spec_sidecar_artifact::read_exact_at(rf, 0, remap_h.data(), expected_remap_size)) {
        fprintf(stderr, "[hip-sidecar] invalid remap table (%s): expected %llu bytes\n",
                ids_path ? ids_path : "null", static_cast<unsigned long long>(expected_remap_size));
        if (rf) fclose(rf);
        fclose(wf);
        return -1;
    }
    fclose(rf);
    if (!spec_sidecar_artifact::validate_remap(remap_h, SC_N_VOCAB, validation_error)) {
        fprintf(stderr, "[hip-sidecar] invalid remap table: %s\n", validation_error.c_str());
        fclose(wf);
        return -1;
    }

    std::vector<char> hostbuf;
    for (auto & t : S.tensors) {
        // token_embd is only ever read one row (2.9KB) per draft step: keep it in
        // pinned host memory (zero VRAM) instead of adding 760MB of device pressure
        const bool host_resident = t.name == "token_embd.weight";
        if (host_resident) {
            void * hptr = nullptr;
            SC_CHECK(hipHostMalloc(&hptr, static_cast<size_t>(t.nbytes), hipHostMallocMapped));
            if (!spec_sidecar_artifact::read_exact_at(wf, t.offset, hptr, t.nbytes)) { fclose(wf); fprintf(stderr, "[hip-sidecar] short read %s\n", t.name.c_str()); return -1; }
            SC_CHECK(hipHostGetDevicePointer(&t.dev, hptr, 0));
        } else {
            hostbuf.resize(static_cast<size_t>(t.nbytes));
            if (!spec_sidecar_artifact::read_exact_at(wf, t.offset, hostbuf.data(), t.nbytes)) { fclose(wf); fprintf(stderr, "[hip-sidecar] short read %s\n", t.name.c_str()); return -1; }
            SC_CHECK(hipMalloc(&t.dev, static_cast<size_t>(t.nbytes)));
            SC_CHECK(hipMemcpy(t.dev, hostbuf.data(), static_cast<size_t>(t.nbytes), hipMemcpyHostToDevice));
        }
    }
    fclose(wf);
    SC_CHECK(hipMalloc(&S.remap_d, expected_remap_size));
    SC_CHECK(hipMemcpy(S.remap_d, remap_h.data(), expected_remap_size, hipMemcpyHostToDevice));

    auto & s = S;
    for (int i = 0; i < N_SCR; ++i) {
        auto & c = s.scr[i];
        SC_CHECK(hipMalloc(&c.x_tok, N_EMBD * 4));  SC_CHECK(hipMalloc(&c.x_h, N_EMBD * 4));
        SC_CHECK(hipMalloc(&c.e_n, N_EMBD * 4));    SC_CHECK(hipMalloc(&c.h_n, N_EMBD * 4));
        SC_CHECK(hipMalloc(&c.cat, 2 * N_EMBD * 4)); SC_CHECK(hipMalloc(&c.proj, N_EMBD * 4));
        SC_CHECK(hipMalloc(&c.attin, N_EMBD * 4));
        SC_CHECK(hipMalloc(&c.qfull, 2 * N_HEAD * HEAD_DIM * 4));
        SC_CHECK(hipMalloc(&c.q, N_HEAD * HEAD_DIM * 4)); SC_CHECK(hipMalloc(&c.gate, N_HEAD * HEAD_DIM * 4));
        SC_CHECK(hipMalloc(&c.k, N_HEAD_KV * HEAD_DIM * 4)); SC_CHECK(hipMalloc(&c.v, N_HEAD_KV * HEAD_DIM * 4));
        SC_CHECK(hipMalloc(&c.attout, N_HEAD * HEAD_DIM * 4));
        SC_CHECK(hipMalloc(&c.ffn_in, N_EMBD * 4)); SC_CHECK(hipMalloc(&c.ffn_g, N_FF * 4)); SC_CHECK(hipMalloc(&c.ffn_u, N_FF * 4));
        SC_CHECK(hipMalloc(&c.ffn_tmp, N_EMBD * 4));
        SC_CHECK(hipMalloc(&c.resid, N_EMBD * 4));  SC_CHECK(hipMalloc(&c.hnorm2, N_EMBD * 4));
#if defined(__gfx1030__) || defined(SPEC_SIDECAR_GFX1030)
        SC_CHECK(hipMalloc(&c.q8, (size_t) (N_FF / 32) * sizeof(block_q8_1)));
        SC_CHECK(hipMalloc(&c.q8_sum_hi, (size_t) (N_FF / 32) * sizeof(block_q8_1_sum_hi)));
#endif
        SC_CHECK(hipMalloc(&c.id_dev, 4));
    }
    {
        auto & b = s.bs;
        SC_CHECK(hipMalloc(&b.x_tok, N_BATCH * N_EMBD * 4));
        SC_CHECK(hipMalloc(&b.e_n,   N_BATCH * N_EMBD * 4));
        SC_CHECK(hipMalloc(&b.h_n,   N_BATCH * N_EMBD * 4));
        SC_CHECK(hipMalloc(&b.cat,   N_BATCH * 2 * N_EMBD * 4));
        SC_CHECK(hipMalloc(&b.proj,  N_BATCH * N_EMBD * 4));
        SC_CHECK(hipMalloc(&b.attin, N_BATCH * N_EMBD * 4));
        SC_CHECK(hipMalloc(&b.qfull, N_BATCH * 2 * N_HEAD * HEAD_DIM * 4));
        SC_CHECK(hipMalloc(&b.q,     N_BATCH * N_HEAD * HEAD_DIM * 4));
        SC_CHECK(hipMalloc(&b.gate,  N_BATCH * N_HEAD * HEAD_DIM * 4));
        SC_CHECK(hipMalloc(&b.k,     N_BATCH * N_HEAD_KV * HEAD_DIM * 4));
        SC_CHECK(hipMalloc(&b.v,     N_BATCH * N_HEAD_KV * HEAD_DIM * 4));
        SC_CHECK(hipMalloc(&b.attout, N_BATCH * N_HEAD * HEAD_DIM * 4));
        SC_CHECK(hipMalloc(&b.ffn_in, N_BATCH * N_EMBD * 4));
        SC_CHECK(hipMalloc(&b.ffn_g, N_BATCH * N_FF * 4));
        SC_CHECK(hipMalloc(&b.ffn_u, N_BATCH * N_FF * 4));
        SC_CHECK(hipMalloc(&b.ffn_tmp, N_BATCH * N_EMBD * 4));
        SC_CHECK(hipMalloc(&b.resid, N_BATCH * N_EMBD * 4));
    }
    SC_CHECK(hipMalloc(&s.part_m,   N_HEAD * ATTN_MAX_PARTS * 4));
    SC_CHECK(hipMalloc(&s.part_l,   N_HEAD * ATTN_MAX_PARTS * 4));
    SC_CHECK(hipMalloc(&s.part_acc, (size_t) N_HEAD * ATTN_MAX_PARTS * HEAD_DIM * 4));
    SC_CHECK(hipMalloc(&s.logits, N_HEAD_ROWS * 4));
    s.nparts = (N_HEAD_ROWS + 256 * 8 - 1) / (256 * 8);
    SC_CHECK(hipMalloc(&s.pmax, (size_t) s.nparts * MTP_DRAFT_TOP_K * 4));
    SC_CHECK(hipMalloc(&s.pidx, (size_t) s.nparts * MTP_DRAFT_TOP_K * 4));
    SC_CHECK(hipMalloc(&s.tok_out, 8 * 4));
    SC_CHECK(hipMalloc(&s.draft_ids, (size_t) 8 * MTP_DRAFT_TOP_K * 4));
    SC_CHECK(hipMalloc(&s.draft_logits, (size_t) 8 * MTP_DRAFT_TOP_K * 4));
    SC_CHECK(hipMalloc(&s.draft_probs, (size_t) 8 * MTP_DRAFT_TOP_K * 4));
    SC_CHECK(hipMalloc(&s.draft_count, 4));
    SC_CHECK(hipMalloc(&s.draft_active, 4));
    if (const char * mp = getenv("LLAMA_SPEC_HIP_MAX_POS")) {
        const long long v = atoll(mp);
        // hard capacity: the split-attention partial buffers cover at most
        // ATTN_MAX_PARTS * ATTN_SPLIT positions; a larger max_pos would send the
        // kernel grid out of bounds. Clamp to [4096, that capacity] and cap at
        // the model's trained context.
        const long long cap = (long long) ATTN_MAX_PARTS * ATTN_SPLIT;
        long long c2 = v;
        if (c2 < 4096) c2 = 4096;
        if (c2 > cap) c2 = cap;
        if (c2 > 131072) c2 = 131072;
        if (c2 != v) fprintf(stderr, "[hip-sidecar] LLAMA_SPEC_HIP_MAX_POS=%lld clamped to %lld\n", v, c2);
        s.max_pos = (size_t) c2;
    }
    SC_CHECK(hipStreamCreate(&s.owned_st));
    s.st = s.owned_st;
    s.owns_st = true;
    SC_CHECK(hipEventCreate(&s.ev_a));
    SC_CHECK(hipEventCreate(&s.ev_b));

#if defined(SPEC_SIDECAR_ROCPRIM_TOPK)
    {
        const rocprim::counting_iterator<int> indices(0);
        size_t bytes = 0;
        const hipError_t status = rocprim::topk_pairs<rocprim::default_config, true>(
                nullptr, bytes, s.logits, s.pmax, indices, s.pidx,
                (uint32_t) N_HEAD_ROWS, (uint32_t) MTP_DRAFT_TOP_K,
                rocprim::identity_decomposer{}, s.st);
        if (status == hipSuccess) {
            s.topk_temp_bytes = bytes;
            if (bytes == 0 || hipMalloc(&s.topk_temp, bytes) == hipSuccess) {
                s.topk_rocprim = true;
            } else {
                fprintf(stderr, "[hip-sidecar] rocPRIM top-k scratch allocation failed; using fallback\n");
            }
        } else {
            fprintf(stderr, "[hip-sidecar] rocPRIM top-k query failed (%s); using fallback\n",
                    hipGetErrorString(status));
        }
        if (getenv("LLAMA_SPEC_HIP_DEBUG") != nullptr) {
            fprintf(stderr, "[hip-sidecar] stochastic top-k: %s\n",
                    s.topk_rocprim ? "rocPRIM" : "fallback");
        }
    }
#endif

    auto W  = [&](const char * n) { return (const block_q4_0 *) find_tensor(n)->dev; };
    auto Wf = [&](const char * n) { return (const float *) find_tensor(n)->dev; };
    s.W_embd = W("token_embd.weight");
    s.W_ehp  = W("blk.64.nextn.eh_proj.weight");
    s.w_enorm = Wf("blk.64.nextn.enorm.weight");
    s.w_hnorm = Wf("blk.64.nextn.hnorm.weight");
    s.w_attn  = Wf("blk.64.attn_norm.weight");
    s.W_q = W("blk.64.attn_q.weight");
    s.W_k = W("blk.64.attn_k.weight");
    s.W_v = W("blk.64.attn_v.weight");
    s.W_o = W("blk.64.attn_output.weight");
    s.w_qn = Wf("blk.64.attn_q_norm.weight");
    s.w_kn = Wf("blk.64.attn_k_norm.weight");
    s.w_post = Wf("blk.64.post_attention_norm.weight");
    s.W_fg = W("blk.64.ffn_gate.weight");
    s.W_fu = W("blk.64.ffn_up.weight");
    s.W_fd = W("blk.64.ffn_down.weight");
    s.w_shn = Wf("blk.64.nextn.shared_head_norm.weight");
    s.W_head = W("blk.64.nextn.shared_head_head.weight");

    S.ok = true;
    fprintf(stderr, "[hip-sidecar] initialized: %zu tensors, KV %zu positions f16, device=%d\n",
            S.tensors.size(), S.max_pos, S.device);
#if defined(__gfx1030__) || defined(SPEC_SIDECAR_GFX1030)
    if (getenv("LLAMA_SPEC_HIP_DEBUG") != nullptr) {
        fprintf(stderr, "[hip-sidecar] gfx1030 GEMV path: %s\n",
                sidecar_gfx1030_gemv_enabled() ? "enabled" : "disabled");
    }
#endif
    return 0;
}

SC_API int spec_hip_init_device(const char * weights_dir, const char * ids_path,
        int32_t n_seq, int32_t device) {
    return spec_hip_init_impl(weights_dir, ids_path, n_seq, device);
}

SC_API int spec_hip_init(const char * weights_dir, const char * ids_path, int32_t n_seq) {
    return spec_hip_init_impl(weights_dir, ids_path, n_seq, -1);
}

// Catch-up rows are staged in pending KV. The host path synchronizes only to
// keep pageable input storage alive; the device path stays entirely async.
static int spec_hip_catchup_impl(int32_t seq_id, const int32_t * toks, const int32_t * poss,
        const float * h_rows, bool h_rows_device, int n) {
    SeqState * q = get_seq_checked(seq_id);
    if (!S.ok || q == nullptr || n < 0 ||
            (n > 0 && (toks == nullptr || poss == nullptr || h_rows == nullptr))) return -1;
    if (n == 0) return 0;
    if (!spec_sidecar_mtp::can_begin_catchup(q->state_pos_max, q->h_valid)) {
        fprintf(stderr, "[hip-sidecar] catchup: prior hidden state unavailable at position %d\n",
                q->state_pos_max);
        return -1;
    }
    if (n > (int) S.max_pos - q->state_pos_max ||
            q->pending_pos_max != q->state_pos_max || ensure_seq_storage(*q) != 0 ||
            ensure_pending_storage(*q, n) != 0) return -1;
    for (int r = 0; r < n; ++r) {
        if (toks[r] < 0 || toks[r] >= SC_N_VOCAB) {
            fprintf(stderr, "[hip-sidecar] catchup: invalid token %d at row %d\n", toks[r], r);
            return -1;
        }
        if (poss[r] < 0 || (size_t) poss[r] >= S.max_pos ||
                poss[r] != q->state_pos_max + r) {
            fprintf(stderr, "[hip-sidecar] catchup: seq %d non-contiguous position %d at row %d (expected %d)\n",
                    seq_id, poss[r], r, q->state_pos_max + r);
            return -1;
        }
    }
    static int n_calls = 0, n_rows = 0;
    static double total_ms = 0.0, total_gpu = 0.0;
    const bool dbg = getenv("LLAMA_SPEC_HIP_DEBUG") != nullptr;
    static int last_input_mode = -1;
    const int input_mode = h_rows_device ? 1 : 0;
    if (dbg && input_mode != last_input_mode) {
        fprintf(stderr, "[hip-sidecar] catch-up input: %s (sidecar device=%d)\n",
                h_rows_device ? "DIRECT D2D" : "HOST FALLBACK", S.device);
        last_input_mode = input_mode;
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    if (dbg) SC_CHECK(hipEventRecord(S.ev_a, S.st));
    q->pending_pos_min = q->state_pos_max;
    q->pending_pos_max = q->state_pos_max + n;
    auto & b = S.bs;
    hipStream_t st = S.st;
    const float * pending_h_rows = h_rows;
    if (!h_rows_device) {
        if (ensure_pending_h_storage(*q, n) != 0) return -1;
        SC_CHECK(hipMemcpyAsync(q->Hp, h_rows, (size_t) n * N_EMBD * sizeof(float),
                                hipMemcpyHostToDevice, st));
        pending_h_rows = q->Hp;
    }
    static const bool full_catchup = getenv("LLAMA_SPEC_HIP_FULL_CATCHUP") != nullptr;
    for (int r0 = 0; r0 < n; r0 += N_BATCH) {
        const int nb = std::min(N_BATCH, n - r0);
        // Native MTP pairs token x_p with target hidden h_(p-1). The first row
        // consumes the previously committed target hidden (or zeros at BOS),
        // and all remaining rows consume the preceding row from this batch.
        const int32_t hidden_row0 = spec_sidecar_mtp::hidden_source_row(r0);
        if (hidden_row0 < 0) {
            if (q->h_valid) {
                SC_CHECK(hipMemcpyAsync(b.ffn_tmp, q->h_committed,
                                        N_EMBD * sizeof(float), hipMemcpyDeviceToDevice, st));
            } else {
                SC_CHECK(hipMemsetAsync(b.ffn_tmp, 0, N_EMBD * sizeof(float), st));
            }
            if (nb > 1) {
                SC_CHECK(hipMemcpyAsync(b.ffn_tmp + N_EMBD, pending_h_rows,
                                        (size_t) (nb - 1) * N_EMBD * sizeof(float),
                                        hipMemcpyDeviceToDevice, st));
            }
        } else {
            SC_CHECK(hipMemcpyAsync(b.ffn_tmp, pending_h_rows + (size_t) hidden_row0 * N_EMBD,
                                    (size_t) nb * N_EMBD * sizeof(float),
                                    hipMemcpyDeviceToDevice, st));
        }
        const int pos_base = poss[r0];
        const int32_t t0 = toks[r0];
        const int32_t t1 = nb > 1 ? toks[r0 + 1] : 0;
        const int32_t t2 = nb > 2 ? toks[r0 + 2] : 0;
        const int32_t t3 = nb > 3 ? toks[r0 + 3] : 0;
        const int32_t t4 = nb > 4 ? toks[r0 + 4] : 0;
        const int32_t t5 = nb > 5 ? toks[r0 + 5] : 0;
        const int32_t t6 = nb > 6 ? toks[r0 + 6] : 0;
        const int32_t t7 = nb > 7 ? toks[r0 + 7] : 0;
        const dim3 gcol((N_EMBD + 255) / 256, nb);
        deq_rows_q4_0_b8<<<gcol, 256, 0, st>>>(S.W_embd, t0, t1, t2, t3, t4, t5, t6, t7,
                b.x_tok, N_EMBD, nb);
        rmsnorm_b<<<nb, 256, 0, st>>>(b.x_tok, S.w_enorm, b.e_n, N_EMBD);
        rmsnorm_b<<<nb, 256, 0, st>>>(b.ffn_tmp, S.w_hnorm, b.h_n, N_EMBD);
        concat2_b<<<dim3((2 * N_EMBD + 255) / 256, nb), 256, 0, st>>>(b.e_n, b.h_n, b.cat, N_EMBD);
        gemv_q4_0_b<<<dim3(N_EMBD, nb), 32, 0, st>>>(S.W_ehp, b.cat, b.proj, 2 * N_EMBD, N_EMBD);
        rmsnorm_b<<<nb, 256, 0, st>>>(b.proj, S.w_attn, b.attin, N_EMBD);
        gemv_q4_0_b<<<dim3(N_HEAD_KV * HEAD_DIM, nb), 32, 0, st>>>(S.W_k, b.attin, b.k, N_EMBD, N_HEAD_KV * HEAD_DIM);
        gemv_q4_0_b<<<dim3(N_HEAD_KV * HEAD_DIM, nb), 32, 0, st>>>(S.W_v, b.attin, b.v, N_EMBD, N_HEAD_KV * HEAD_DIM);
        rmsnorm_heads_b<<<dim3(N_HEAD_KV, nb), 256, 0, st>>>(b.k, S.w_kn, b.k, N_HEAD_KV);
        rope_neox_b_base<<<dim3(N_HEAD_KV, nb), N_ROT / 2, 0, st>>>(b.k, N_HEAD_KV, pos_base);
        kv_store_b_base<<<dim3((N_HEAD_KV * HEAD_DIM + 255) / 256, nb), 256, 0, st>>>(
                b.k, b.v, q->Kp, q->Vp, pos_base, q->pending_pos_min);
        if (full_catchup) {
            // The debug full path uses a dual-cache attention view. It is not
            // used by the production KV-only path, but remains safe under lazy staging.
            gemv_q4_0_b<<<dim3(2 * N_HEAD * HEAD_DIM, nb), 32, 0, st>>>(S.W_q, b.attin, b.qfull, N_EMBD, 2 * N_HEAD * HEAD_DIM);
            split_q_gate_b<<<dim3(N_HEAD, nb), HEAD_DIM, 0, st>>>(b.qfull, b.q, b.gate);
            rmsnorm_heads_b<<<dim3(N_HEAD, nb), 256, 0, st>>>(b.q, S.w_qn, b.q, N_HEAD);
            rope_neox_b_base<<<dim3(N_HEAD, nb), N_ROT / 2, 0, st>>>(b.q, N_HEAD, pos_base);
            attn_1q_b_base<<<dim3(N_HEAD, nb), 256, 0, st>>>(b.q, q->Kc, q->Vc, q->Kp, q->Vp,
                    q->pending_pos_min, b.attout, pos_base);
            sigmoid_gate<<<(nb * N_HEAD * HEAD_DIM + 255) / 256, 256, 0, st>>>(b.attout, b.gate, nb * N_HEAD * HEAD_DIM);
            gemv_q4_0_b<<<dim3(N_EMBD, nb), 32, 0, st>>>(S.W_o, b.attout, b.resid, N_HEAD * HEAD_DIM, N_EMBD);
            add_inplace<<<(nb * N_EMBD + 255) / 256, 256, 0, st>>>(b.resid, b.proj, nb * N_EMBD);
            rmsnorm_b<<<nb, 256, 0, st>>>(b.resid, S.w_post, b.ffn_in, N_EMBD);
            gemv_q4_0_b<<<dim3(N_FF, nb), 32, 0, st>>>(S.W_fg, b.ffn_in, b.ffn_g, N_EMBD, N_FF);
            gemv_q4_0_b<<<dim3(N_FF, nb), 32, 0, st>>>(S.W_fu, b.ffn_in, b.ffn_u, N_EMBD, N_FF);
            silu_mul<<<(nb * N_FF + 255) / 256, 256, 0, st>>>(b.ffn_g, b.ffn_u, nb * N_FF);
            gemv_q4_0_b<<<dim3(N_EMBD, nb), 32, 0, st>>>(S.W_fd, b.ffn_g, b.ffn_tmp, N_FF, N_EMBD);
        }
        if (!h_rows_device) SC_CHECK(hipStreamSynchronize(st));
    }
    if (h_rows_device) SC_CHECK(hipGetLastError());
    if (dbg) {
        SC_CHECK(hipEventRecord(S.ev_b, S.st));
        SC_CHECK(hipEventSynchronize(S.ev_b));
        float gpu = 0.0f;
        SC_CHECK(hipEventElapsedTime(&gpu, S.ev_a, S.ev_b));
        total_gpu += gpu;
        auto t1 = std::chrono::high_resolution_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        n_rows += n;
        if (++n_calls % 25 == 0) {
            fprintf(stderr, "[hip-sidecar] catchup avg %.2f ms wall / %.2f ms gpu (%.1f rows/call) over %d calls\n",
                    total_ms / n_calls, total_gpu / n_calls, (double) n_rows / n_calls, n_calls);
        }
    }
    return 0;
}

SC_API int spec_hip_catchup(int32_t seq_id, const int32_t * toks, const int32_t * poss,
        const float * h_rows, int n) {
    return spec_hip_catchup_impl(seq_id, toks, poss, h_rows, false, n);
}

SC_API int spec_hip_catchup_device(int32_t seq_id, const int32_t * toks, const int32_t * poss,
        const float * h_rows_device, int n) {
    return spec_hip_catchup_impl(seq_id, toks, poss, h_rows_device, true, n);
}

// draft n tokens greedily from (id_last, n_past, h); out_ids receives full-vocab ids
static int spec_hip_draft_impl(int32_t seq_id, int32_t id_last, int32_t n_past,
        const float * h, bool h_device, int n, int32_t * out_ids) {
    SeqState * q = get_seq_checked(seq_id);
    if (!S.ok || q == nullptr || n < 1 || n > 8 || out_ids == nullptr ||
            (!h_device && h == nullptr)) return -1;
    if (id_last < 0 || id_last >= SC_N_VOCAB) {
        fprintf(stderr, "[hip-sidecar] draft: invalid id_last %d\n", id_last);
        return -1;
    }
    if (n_past < 0 || n_past != q->state_pos_max || !q->allocated ||
            (h_device && !q->h_valid) ||
            static_cast<size_t>(n_past) + static_cast<size_t>(n) > S.max_pos) return -1;
    // fault-injection hook for hardening tests: returns garbage ids once
    if (getenv("LLAMA_SPEC_HIP_FAULT_BADID") != nullptr) {
        static bool fired = false;
        if (!fired) {
            fired = true;
            for (int j = 0; j < n; ++j) out_ids[j] = 999999999;
            return 0;
        }
    }
    static int n_calls = 0;
    static double total_ms = 0.0, total_gpu = 0.0;
    const bool dbg = getenv("LLAMA_SPEC_HIP_DEBUG") != nullptr;
    auto t0 = std::chrono::high_resolution_clock::now();
    if (dbg) SC_CHECK(hipEventRecord(S.ev_a, S.st));
    auto & c = S.scr[0];
    if (h_device) {
        // commit_state has already copied the accepted hidden row into the
        // sequence-local device buffer; no host round-trip is needed here.
        h = q->h_committed;
    } else {
        SC_CHECK(hipMemcpyAsync(c.x_h, h, N_EMBD * sizeof(float), hipMemcpyHostToDevice, S.st));
    }
    SC_CHECK(hipMemcpyAsync(c.id_dev, &id_last, sizeof(id_last), hipMemcpyHostToDevice, S.st));
    for (int j = 0; j < n; ++j) {
        const int * tok_src = j == 0 ? c.id_dev : S.tok_out + (j - 1);
        const float * xh_in = j == 0 ? h_device ? q->h_committed : c.x_h : c.hnorm2;
        deq_row_q4_0<<<(N_EMBD + 255) / 256, 256, 0, S.st>>>(S.W_embd, tok_src, c.x_tok, N_EMBD, SC_N_VOCAB);
        block_phase1(c, S.st, xh_in, n_past + j, q->Kc, q->Vc);
        block_phase2(c, S.st, n_past + j, S.tok_out + j, q->Kc, q->Vc);
    }
    if (dbg) SC_CHECK(hipEventRecord(S.ev_b, S.st));
    SC_CHECK(hipMemcpyAsync(out_ids, S.tok_out, sizeof(int32_t) * n, hipMemcpyDeviceToHost, S.st));
    SC_CHECK(hipStreamSynchronize(S.st));
    for (int j = 0; j < n; ++j) {
        if (out_ids[j] < 0 || out_ids[j] >= SC_N_VOCAB) {
            fprintf(stderr, "[hip-sidecar] draft produced invalid token at step %d\n", j);
            return -1;
        }
    }
    if (dbg) {
        float gpu = 0.0f;
        SC_CHECK(hipEventElapsedTime(&gpu, S.ev_a, S.ev_b));
        static double win_gpu = 0.0, win_wall = 0.0;
        static int win_n = 0;
        auto t1 = std::chrono::high_resolution_clock::now();
        win_gpu  += gpu;
        win_wall += std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_gpu += gpu;
        total_ms  += std::chrono::duration<double, std::milli>(t1 - t0).count();
        n_calls++;
        if (++win_n == 25) {
            fprintf(stderr, "[hip-sidecar] draft last25 %.2f ms wall / %.2f ms gpu @pos=%d (lifetime %.2f/%.2f over %d)\n",
                    win_wall / win_n, win_gpu / win_n, n_past, total_ms / n_calls, total_gpu / n_calls, n_calls);
            win_gpu = win_wall = 0.0;
            win_n = 0;
        }
    }
    return n;
}

SC_API int spec_hip_draft(int32_t seq_id, int32_t id_last, int32_t n_past,
        const float * h, int n, int32_t * out_ids) {
    return spec_hip_draft_impl(seq_id, id_last, n_past, h, false, n, out_ids);
}

SC_API int spec_hip_draft_device(int32_t seq_id, int32_t id_last, int32_t n_past,
        int n, int32_t * out_ids) {
    return spec_hip_draft_impl(seq_id, id_last, n_past, nullptr, true, n, out_ids);
}

// Stochastic drafting keeps the autoregressive chain on the device. Only the
// compact top-k q rows and selected IDs cross back to the host; the caller's
// key determines the proposal uniforms without requiring mutable sidecar RNG
// state in checkpoints.
static int spec_hip_draft_stochastic_impl(int32_t seq_id, int32_t id_last, int32_t n_past,
        const float * h, bool h_device, float temperature, float p_min, uint64_t rng_key,
        int n, int32_t * out_ids, int32_t * out_dist_ids, float * out_dist_probs) {
    SeqState * q = get_seq_checked(seq_id);
    if (!S.ok || q == nullptr || n < 1 || n > 8 || out_ids == nullptr ||
            out_dist_ids == nullptr || out_dist_probs == nullptr ||
            (!h_device && h == nullptr) || !std::isfinite(temperature) || temperature <= 0.0f ||
            !std::isfinite(p_min) || p_min < 0.0f) {
        return -1;
    }
    if (id_last < 0 || id_last >= SC_N_VOCAB || n_past < 0 ||
            n_past != q->state_pos_max || !q->allocated ||
            (h_device && !q->h_valid) ||
            static_cast<size_t>(n_past) + static_cast<size_t>(n) > S.max_pos) {
        return -1;
    }

    auto & c = S.scr[0];
    if (h_device) {
        h = q->h_committed;
    } else {
        SC_CHECK(hipMemcpyAsync(c.x_h, h, N_EMBD * sizeof(float), hipMemcpyHostToDevice, S.st));
    }
    SC_CHECK(hipMemcpyAsync(c.id_dev, &id_last, sizeof(id_last), hipMemcpyHostToDevice, S.st));
    const int zero = 0;
    const int one = 1;
    SC_CHECK(hipMemcpyAsync(S.draft_count, &zero, sizeof(zero), hipMemcpyHostToDevice, S.st));
    SC_CHECK(hipMemcpyAsync(S.draft_active, &one, sizeof(one), hipMemcpyHostToDevice, S.st));

    for (int j = 0; j < n; ++j) {
        const int * tok_src = j == 0 ? c.id_dev : S.tok_out + (j - 1);
        const float * xh_in = j == 0 ? (h_device ? q->h_committed : c.x_h) : c.hnorm2;
        deq_row_q4_0<<<(N_EMBD + 255) / 256, 256, 0, S.st>>>(S.W_embd, tok_src, c.x_tok, N_EMBD, SC_N_VOCAB);
        block_phase1(c, S.st, xh_in, n_past + j, q->Kc, q->Vc);
        block_phase2(c, S.st, n_past + j, nullptr, q->Kc, q->Vc, true);
#if defined(SPEC_SIDECAR_ROCPRIM_TOPK)
        if (S.topk_rocprim) {
            const rocprim::counting_iterator<int> indices(0);
            size_t topk_bytes = S.topk_temp_bytes;
            const hipError_t status = rocprim::topk_pairs<rocprim::default_config, true>(
                    S.topk_temp, topk_bytes, S.logits, S.pmax, indices, S.pidx,
                    (uint32_t) N_HEAD_ROWS, (uint32_t) MTP_DRAFT_TOP_K,
                    rocprim::identity_decomposer{}, S.st);
            SC_CHECK(status);
            topk_final<<<1, 256, 0, S.st>>>(S.pmax, S.pidx, 1, S.remap_d, SC_N_VOCAB,
                    S.draft_ids + (size_t) j * MTP_DRAFT_TOP_K,
                    S.draft_logits + (size_t) j * MTP_DRAFT_TOP_K);
        } else
#endif
        {
            topk_partial<<<S.nparts, 256, 0, S.st>>>(S.logits, N_HEAD_ROWS, S.pmax, S.pidx);
            topk_final<<<1, 256, 0, S.st>>>(S.pmax, S.pidx, S.nparts, S.remap_d, SC_N_VOCAB,
                    S.draft_ids + (size_t) j * MTP_DRAFT_TOP_K,
                    S.draft_logits + (size_t) j * MTP_DRAFT_TOP_K);
        }
        sample_topk<<<1, 1, 0, S.st>>>(
                S.draft_ids + (size_t) j * MTP_DRAFT_TOP_K,
                S.draft_logits + (size_t) j * MTP_DRAFT_TOP_K,
                MTP_DRAFT_TOP_K, temperature, p_min,
                spec_sidecar_stochastic_uniform(rng_key, (uint32_t) j), j,
                S.tok_out, S.draft_probs, S.draft_count, S.draft_active);
    }

    int h_count = 0;
    int h_ids[8] = {};
    int h_dist_ids[8 * MTP_DRAFT_TOP_K] = {};
    float h_dist_probs[8 * MTP_DRAFT_TOP_K] = {};
    SC_CHECK(hipMemcpyAsync(&h_count, S.draft_count, sizeof(h_count), hipMemcpyDeviceToHost, S.st));
    SC_CHECK(hipMemcpyAsync(h_ids, S.tok_out, sizeof(h_ids), hipMemcpyDeviceToHost, S.st));
    SC_CHECK(hipMemcpyAsync(h_dist_ids, S.draft_ids, sizeof(h_dist_ids), hipMemcpyDeviceToHost, S.st));
    SC_CHECK(hipMemcpyAsync(h_dist_probs, S.draft_probs, sizeof(h_dist_probs), hipMemcpyDeviceToHost, S.st));
    SC_CHECK(hipStreamSynchronize(S.st));
    if (h_count < 0 || h_count > n) {
        return -1;
    }
    for (int j = 0; j < h_count; ++j) {
        if (h_ids[j] < 0 || h_ids[j] >= SC_N_VOCAB) {
            return -1;
        }
        std::memcpy(out_dist_ids + (size_t) j * MTP_DRAFT_TOP_K,
                h_dist_ids + (size_t) j * MTP_DRAFT_TOP_K,
                MTP_DRAFT_TOP_K * sizeof(int32_t));
        std::memcpy(out_dist_probs + (size_t) j * MTP_DRAFT_TOP_K,
                h_dist_probs + (size_t) j * MTP_DRAFT_TOP_K,
                MTP_DRAFT_TOP_K * sizeof(float));
        out_ids[j] = h_ids[j];
    }
    return h_count;
}

SC_API int spec_hip_stochastic_top_k() {
    return MTP_DRAFT_TOP_K;
}

SC_API int spec_hip_draft_stochastic(int32_t seq_id, int32_t id_last, int32_t n_past,
        const float * h, float temperature, float p_min, uint64_t rng_key, int n,
        int32_t * out_ids, int32_t * out_dist_ids, float * out_dist_probs) {
    return spec_hip_draft_stochastic_impl(seq_id, id_last, n_past, h, false,
            temperature, p_min, rng_key, n, out_ids, out_dist_ids, out_dist_probs);
}

SC_API int spec_hip_draft_stochastic_device(int32_t seq_id, int32_t id_last, int32_t n_past,
        float temperature, float p_min, uint64_t rng_key, int n, int32_t * out_ids,
        int32_t * out_dist_ids, float * out_dist_probs) {
    return spec_hip_draft_stochastic_impl(seq_id, id_last, n_past, nullptr, true,
            temperature, p_min, rng_key, n, out_ids, out_dist_ids, out_dist_probs);
}

// ---------------- harness: replay golden.bin through the API ----------------
#ifdef SIDECAR_HARNESS
int main(int argc, char ** argv) {
    const char * wdir = argc > 1 ? argv[1] : ".";
    const char * ids_path = argc > 2 ? argv[2] : "draft_head_ids.bin";
    const char * golden_path = argc > 3 ? argv[3] : nullptr;
    if (spec_hip_init(wdir, ids_path, 1) != 0) return 1;

    char pbuf[1024];
    if (golden_path) {
        snprintf(pbuf, sizeof(pbuf), "%s", golden_path);
    } else {
        snprintf(pbuf, sizeof(pbuf), "%s/golden.bin", wdir);
    }
    FILE * g = fopen(pbuf, "rb");
    if (!g) { printf("cannot open golden.bin\n"); return 1; }
    uint32_t magic, ne;
    fread(&magic, 4, 1, g); fread(&ne, 4, 1, g);
    if (magic != 0x50444853 || ne != (uint32_t) N_EMBD) { printf("bad golden header\n"); return 1; }

    std::vector<float> hrow(N_EMBD);
    int cycles = 0, tok_match = 0, tok_total = 0, cycles_exact = 0;
    int pend_id_last = 0, pend_n_past = 0;
    bool have_draft_in = false;
    std::vector<int32_t> my_ids;
    double gpu_ms = 0.0;

    while (true) {
        uint32_t typ;
        if (fread(&typ, 4, 1, g) != 1) break;
        if (typ == 0) {
            uint32_t n; fread(&n, 4, 1, g);
            std::vector<int32_t> toks(n), poss(n);
            std::vector<float> hs((size_t) n * N_EMBD);
            for (uint32_t r = 0; r < n; ++r) {
                fread(&toks[r], 4, 1, g); fread(&poss[r], 4, 1, g);
                fread(hs.data() + (size_t) r * N_EMBD, 4, N_EMBD, g);
            }
            if (spec_hip_catchup(0, toks.data(), poss.data(), hs.data(), (int) n) != 0 ||
                    spec_hip_commit_state(0, poss.back() + 1, nullptr) != 0) return 1;
        } else if (typ == 1) {
            fread(&pend_id_last, 4, 1, g); fread(&pend_n_past, 4, 1, g);
            fread(hrow.data(), 4, N_EMBD, g);
            have_draft_in = true;
            my_ids.assign(3, -1);
            auto t0 = std::chrono::high_resolution_clock::now();
            if (spec_hip_draft(0, pend_id_last, pend_n_past, hrow.data(), 3, my_ids.data()) != 0) return 1;
            auto t1 = std::chrono::high_resolution_clock::now();
            gpu_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        } else if (typ == 2) {
            uint32_t n; fread(&n, 4, 1, g);
            std::vector<int32_t> exp_ids(n);
            if (n) fread(exp_ids.data(), 4, n, g);
            if (!have_draft_in) continue;
            have_draft_in = false;
            cycles++;
            bool exact = (n == my_ids.size());
            for (uint32_t i = 0; i < n; ++i) {
                tok_total++;
                if (i < my_ids.size() && my_ids[i] == exp_ids[i]) tok_match++;
                else exact = false;
            }
            if (exact) cycles_exact++;
        }
    }
    fclose(g);
    printf("PARITY(api): %d/%d cycles exact (%.1f%%), %d/%d tokens (%.1f%%)\n",
           cycles_exact, cycles, 100.0 * cycles_exact / (cycles ? cycles : 1),
           tok_match, tok_total, 100.0 * tok_match / (tok_total ? tok_total : 1));
    printf("draft wall: %.2f ms per draft\n", gpu_ms / (cycles ? cycles : 1));
    return 0;
}
#endif
