/*
 * Test: GGML_OP_GATED_DELTA_NET emit_mode==1 (replay ingredients) vs emit_mode==0 (full snapshots).
 *
 * Proves the core DRC-phase-1 claim: replaying the small per-token (k, v, g, beta)
 * ingredients captured by emit_mode==1 through a fresh K==1, emit_mode==0 call
 * reconstructs the exact same recurrent state as a from-scratch full run would
 * have produced at that position -- without needing q or a full S_v x S_v state
 * snapshot for every retained step.
 *
 * Scenario: N=6 total tokens, a checkpoint taken after c=2 tokens, and a "draft"
 * of r-c=3 more tokens (positions 2,3,4) that need to be replayed on top of the
 * checkpoint to reach the state after r=5 tokens. This mirrors the real usage:
 * checkpoint = last accepted position, replay = tokens between the checkpoint
 * and the new rollback point.
 *
 * kda=false and n_seqs=1, H_k==H_v (no GQA broadcast) for this first proof --
 * those are straightforward generalizations of the same ingredient encoding,
 * not additional risk to the core claim.
 *
 * This is the only test that exercises emit_mode==1 end-to-end through a real
 * replay reconstruction; test-backend-ops.cpp's GATED_DELTA_NET cases only
 * sanity-check that emit_mode==1 builds/runs/produces finite output.
 */

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <memory>
#include <algorithm>

struct ggml_backend_ptr {
    ggml_backend_t ptr = nullptr;
    ~ggml_backend_ptr() { if (ptr) ggml_backend_free(ptr); }
    ggml_backend_t get() { return ptr; }
};

struct ggml_backend_buffer_ptr {
    ggml_backend_buffer_t ptr = nullptr;
    ~ggml_backend_buffer_ptr() { if (ptr) ggml_backend_buffer_free(ptr); }
    ggml_backend_buffer_t get() { return ptr; }
};

static float max_abs(const float * a, const float * b, size_t n) {
    float m = 0;
    for (size_t i = 0; i < n; i++) m = std::max(m, fabsf(a[i] - b[i]));
    return m;
}

static uint64_t g_rng = 0x1234567890abcdefULL;
static float rnd(float lo, float hi) {
    g_rng = g_rng * 6364136223846793005ULL + 1442695040888963407ULL;
    float u = (float) ((g_rng >> 33) & 0xffffff) / (float) 0x1000000;
    return lo + u * (hi - lo);
}

// Runs one ggml_gated_delta_net call on `backend` with the given host-side inputs
// (each sized for n_tokens tokens, H heads, S_v head width, n_seqs==1, no GQA) and
// returns the full output buffer (attn scores followed by K snapshot/ingredient rows).
static std::vector<float> run_gdn(
        ggml_backend_t backend,
        const std::vector<float> & q, const std::vector<float> & k, const std::vector<float> & v,
        const std::vector<float> & g, const std::vector<float> & beta, const std::vector<float> & state,
        int64_t S_v, int64_t H, int64_t n_tokens, int64_t K, int32_t emit_mode) {
    ggml_init_params iparams = { /*.mem_size=*/ 1024 * 1024, /*.mem_buffer=*/ nullptr, /*.no_alloc=*/ true };
    ggml_context * ctx = ggml_init(iparams);

    ggml_tensor * tq    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, n_tokens, 1);
    ggml_tensor * tk    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, n_tokens, 1);
    ggml_tensor * tv    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, n_tokens, 1);
    ggml_tensor * tg    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1,   H, n_tokens, 1); // kda=false: scalar gate
    ggml_tensor * tbeta = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1,   H, n_tokens, 1);
    ggml_tensor * tstate = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, S_v, H, 1);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_tensor * out = ggml_gated_delta_net_ext(ctx, tq, tk, tv, tg, tbeta, tstate, K, emit_mode);
    ggml_build_forward_expand(gf, out);

    // allocate AFTER the graph (including its output tensor) is fully built, so `out` gets
    // real backing memory too -- allocating before this point leaves out->data == NULL.
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    ggml_backend_buffer_ptr buf;
    buf.ptr = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);

    ggml_backend_tensor_set(tq,     q.data(),     0, q.size()     * sizeof(float));
    ggml_backend_tensor_set(tk,     k.data(),     0, k.size()     * sizeof(float));
    ggml_backend_tensor_set(tv,     v.data(),     0, v.size()     * sizeof(float));
    ggml_backend_tensor_set(tg,     g.data(),     0, g.size()     * sizeof(float));
    ggml_backend_tensor_set(tbeta,  beta.data(),  0, beta.size()  * sizeof(float));
    ggml_backend_tensor_set(tstate, state.data(), 0, state.size() * sizeof(float));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "FAIL: gated_delta_net graph compute failed\n");
        ggml_free(ctx);
        return {};
    }

    std::vector<float> result(ggml_nelements(out));
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));

    ggml_free(ctx);
    return result;
}

int main() {
    const int64_t S_v = 16;
    const int64_t H   = 4;
    const int64_t N   = 6; // total tokens
    const int64_t c   = 2; // checkpoint position (tokens accepted so far)
    const int64_t r   = 5; // rollback target (ground truth: state after r tokens)

    ggml_backend_ptr backend;
    backend.ptr = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend.get()) {
        fprintf(stderr, "FAIL: could not init CPU backend\n");
        return 1;
    }

    // full-sequence host buffers, shared across all calls below (slices taken as needed)
    std::vector<float> q_full(S_v * H * N), k_full(S_v * H * N), v_full(S_v * H * N);
    std::vector<float> g_full(H * N), beta_full(H * N);
    for (auto & x : q_full) x = rnd(-1.0f, 1.0f);
    for (auto & x : k_full) x = rnd(-1.0f, 1.0f);
    for (auto & x : v_full) x = rnd(-0.3f, 5.0f);
    for (auto & x : g_full) x = rnd(-5.0f, -1e-4f);   // gate is exp(g), g < 0 like real usage
    for (auto & x : beta_full) x = rnd(0.0f, 1.0f);
    std::vector<float> s0(S_v * S_v * H);
    for (auto & x : s0) x = rnd(-0.1f, 0.1f);

    auto slice = [&](const std::vector<float> & full, int64_t row_width, int64_t t0, int64_t t1) {
        return std::vector<float>(full.begin() + t0 * row_width, full.begin() + t1 * row_width);
    };

    // --- 1. ground truth: full run over the first r tokens, K=1, emit_mode=0 ---
    std::vector<float> out_truth = run_gdn(backend.get(),
        slice(q_full, S_v * H, 0, r), slice(k_full, S_v * H, 0, r), slice(v_full, S_v * H, 0, r),
        slice(g_full, H, 0, r), slice(beta_full, H, 0, r), s0,
        S_v, H, /*n_tokens=*/r, /*K=*/1, /*emit_mode=*/0);
    const int64_t attn_r = S_v * H * r;
    std::vector<float> state_truth(out_truth.begin() + attn_r, out_truth.begin() + attn_r + S_v * S_v * H);

    // --- 2. checkpoint: full run over the first c tokens, K=1, emit_mode=0 ---
    std::vector<float> out_ckpt = run_gdn(backend.get(),
        slice(q_full, S_v * H, 0, c), slice(k_full, S_v * H, 0, c), slice(v_full, S_v * H, 0, c),
        slice(g_full, H, 0, c), slice(beta_full, H, 0, c), s0,
        S_v, H, /*n_tokens=*/c, /*K=*/1, /*emit_mode=*/0);
    const int64_t attn_c = S_v * H * c;
    std::vector<float> state_ckpt(out_ckpt.begin() + attn_c, out_ckpt.begin() + attn_c + S_v * S_v * H);

    // --- 3. ingredients: full run over all N tokens, K=N, emit_mode=1 (captures every position) ---
    std::vector<float> out_ingr = run_gdn(backend.get(),
        q_full, k_full, v_full, g_full, beta_full, s0,
        S_v, H, /*n_tokens=*/N, /*K=*/N, /*emit_mode=*/1);
    const int64_t attn_N = S_v * H * N;
    const int64_t snap_size = 4 * S_v * H; // ingredient rows per slot (k,v,g,beta x S_v x H)

    // emit_mode=1's trailing final-state block should match a from-scratch full-N-token run,
    // independent of the replay mechanism below (this checks the Phase 2 op-contract extension).
    std::vector<float> out_truth_N = run_gdn(backend.get(),
        q_full, k_full, v_full, g_full, beta_full, s0,
        S_v, H, /*n_tokens=*/N, /*K=*/1, /*emit_mode=*/0);
    std::vector<float> state_truth_N(out_truth_N.begin() + attn_N, out_truth_N.begin() + attn_N + S_v * S_v * H);
    const int64_t final_state_offset = attn_N + N * snap_size;
    std::vector<float> final_state_ingr(out_ingr.begin() + final_state_offset, out_ingr.begin() + final_state_offset + S_v * S_v * H);
    const float max_diff_final = max_abs(final_state_ingr.data(), state_truth_N.data(), final_state_ingr.size());
    printf("[gdn-ingredient-replay] emit_mode=1 trailing final-state vs from-scratch N-token run: max_abs=%.3e\n", max_diff_final);
    if (max_diff_final > 1e-4f) {
        fprintf(stderr, "FAIL: emit_mode=1's trailing final-state block does not match a from-scratch run (max_abs=%.3e)\n", max_diff_final);
        return 1;
    }

    // when n_tokens > K, a further trailing block ("ckpt_state") should hold the state
    // immediately before the K-token retained window starts -- i.e. after the first
    // (n_tokens - K) tokens -- captured inline instead of via a second op call over the prefix.
    // This is the fix for the always-triggered redundant prefix-recompute found in the Phase 3
    // benchmark (see the DRC plan). Use K=4 < N=6 here specifically to exercise it.
    const int64_t K_small = 4;
    std::vector<float> out_ingr_ckpt = run_gdn(backend.get(),
        q_full, k_full, v_full, g_full, beta_full, s0,
        S_v, H, /*n_tokens=*/N, /*K=*/K_small, /*emit_mode=*/1);
    const int64_t prefix_len = N - K_small;
    std::vector<float> out_truth_prefix = run_gdn(backend.get(),
        slice(q_full, S_v * H, 0, prefix_len), slice(k_full, S_v * H, 0, prefix_len), slice(v_full, S_v * H, 0, prefix_len),
        slice(g_full, H, 0, prefix_len), slice(beta_full, H, 0, prefix_len), s0,
        S_v, H, /*n_tokens=*/prefix_len, /*K=*/1, /*emit_mode=*/0);
    const int64_t attn_prefix = S_v * H * prefix_len;
    std::vector<float> state_truth_prefix(out_truth_prefix.begin() + attn_prefix, out_truth_prefix.begin() + attn_prefix + S_v * S_v * H);
    const int64_t ckpt_offset = attn_N + K_small * snap_size + S_v * S_v * H; // ingredients, then final-state, then ckpt
    std::vector<float> ckpt_state_ingr(out_ingr_ckpt.begin() + ckpt_offset, out_ingr_ckpt.begin() + ckpt_offset + S_v * S_v * H);
    const float max_diff_ckpt = max_abs(ckpt_state_ingr.data(), state_truth_prefix.data(), ckpt_state_ingr.size());
    printf("[gdn-ingredient-replay] emit_mode=1 ckpt_state (n_tokens=%lld > K=%lld) vs from-scratch %lld-token prefix run: max_abs=%.3e\n",
           (long long) N, (long long) K_small, (long long) prefix_len, max_diff_ckpt);
    if (max_diff_ckpt > 1e-4f) {
        fprintf(stderr, "FAIL: emit_mode=1's ckpt_state block does not match a from-scratch prefix run (max_abs=%.3e)\n", max_diff_ckpt);
        return 1;
    }

    // extract ingredients for tokens [c, r) in forward order, building a synthetic replay batch
    const int64_t n_replay = r - c;
    std::vector<float> q_replay(S_v * H * n_replay, 0.0f); // dummy: q doesn't affect state
    std::vector<float> k_replay(S_v * H * n_replay);
    std::vector<float> v_replay(S_v * H * n_replay);
    std::vector<float> g_replay(H * n_replay);
    std::vector<float> beta_replay(H * n_replay);

    for (int64_t rt = 0; rt < n_replay; rt++) {
        const int64_t t    = c + rt;   // original token index
        const int64_t slot = t;        // emit_mode=1 chronological order; K=N here so slot == t directly
        for (int64_t h = 0; h < H; h++) {
            const float * ingr = &out_ingr[attn_N + slot * snap_size + h * 4 * S_v];
            std::copy(ingr,               ingr + S_v, &k_replay[rt * S_v * H + h * S_v]);
            std::copy(ingr + S_v,         ingr + 2*S_v, &v_replay[rt * S_v * H + h * S_v]);
            g_replay[rt * H + h]    = ingr[2 * S_v]; // broadcast scalar: any of the S_v copies works
            beta_replay[rt * H + h] = ingr[3 * S_v];
        }
    }

    // --- 4. replay: checkpoint state + extracted ingredients, K=1, emit_mode=0 ---
    std::vector<float> out_replay = run_gdn(backend.get(),
        q_replay, k_replay, v_replay, g_replay, beta_replay, state_ckpt,
        S_v, H, /*n_tokens=*/n_replay, /*K=*/1, /*emit_mode=*/0);
    const int64_t attn_replay = S_v * H * n_replay;
    std::vector<float> state_replay(out_replay.begin() + attn_replay, out_replay.begin() + attn_replay + S_v * S_v * H);

    const float max_diff = max_abs(state_replay.data(), state_truth.data(), state_replay.size());
    printf("[gdn-ingredient-replay] N=%lld c=%lld r=%lld S_v=%lld H=%lld: state_replay vs state_truth max_abs=%.3e\n",
           (long long) N, (long long) c, (long long) r, (long long) S_v, (long long) H, max_diff);

    if (max_diff > 1e-4f) {
        fprintf(stderr, "FAIL: ingredient replay did not reconstruct the ground-truth state (max_abs=%.3e)\n", max_diff);
        return 1;
    }

    printf("OK: ingredient replay reconstructs the ground-truth full-snapshot state\n");
    return 0;
}
