// SPDX-License-Identifier: MIT
// Qwen3.6/Qwen3.5-MoE 35B-A3B MTP sidecar.
//
// This provider is intentionally separate from src/spec_sidecar/mtp: the
// Qwen35MoE MTP block is a full-attention block followed by an 8-of-256 MoE,
// with a 2048-wide hidden state. Its artifact is generated from the trained
// block in the target GGUF and is fail-closed by an exact manifest contract.
#include "kernels.hip"
#include "batch_kernels.hip"
#include "../artifact_manifest.h"
#include "../mtp/catchup_alignment.h"
#include "spec_sidecar/sidecar_abi.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(__has_include)
#    if __has_include(<rocprim/device/device_topk.hpp>)
#        include <rocprim/device/device_topk.hpp>
#        include <rocprim/iterator/counting_iterator.hpp>
#        define QWEN35MOE_SIDECAR_ROCPRIM_TOPK
#    endif
#endif

namespace {

using Tensor = spec_sidecar_artifact::TensorDesc;
using namespace qwen35moe_sidecar;

static constexpr int SC_RELEASE_ABI = SPEC_SIDECAR_MTP_RELEASE_ABI;
static constexpr int SC_MAX_SEQ = 8;
static constexpr int SC_MAX_POS_DEFAULT = 262144;
static constexpr size_t N_PARTS = (N_HEAD_ROWS + 256 * 8 - 1) / (256 * 8);

static_assert(MTP_DRAFT_TOP_K == SPEC_SIDECAR_MTP_DRAFT_TOP_K,
        "Qwen35MoE stochastic top-k ABI mismatch");

struct SeqState {
    __half * key_committed = nullptr;
    __half * value_committed = nullptr;
    __half * key_pending = nullptr;
    __half * value_pending = nullptr;
    float * hidden_committed = nullptr;
    float * hidden_pending = nullptr;
    int pending_capacity = 0;
    int pending_hidden_capacity = 0;
    int32_t pending_pos_min = 0;
    int32_t pending_pos_max = 0;
    int32_t state_pos_min = 0;
    int32_t state_pos_max = 0;
    uint64_t state_epoch = 1;
    bool hidden_valid = false;
    bool allocated = false;
};

struct Scratch {
    float * token = nullptr;
    float * hidden = nullptr;
    float * token_norm = nullptr;
    float * hidden_norm = nullptr;
    float * concat = nullptr;
    float * projected = nullptr;
    float * attention_input = nullptr;
    float * query_full = nullptr;
    float * query = nullptr;
    float * attention_gate = nullptr;
    float * key = nullptr;
    float * value = nullptr;
    float * attention = nullptr;
    float * residual = nullptr;
    float * post_norm = nullptr;
    float * router = nullptr;
    int32_t * expert_ids = nullptr;
    float * expert_weights = nullptr;
    float * expert_gate = nullptr;
    float * expert_up = nullptr;
    float * expert_outputs = nullptr;
    float * moe = nullptr;
    float * shared_gate = nullptr;
    float * shared_up = nullptr;
    float * shared_output = nullptr;
    float * shared_scalar = nullptr;
    float * h_next = nullptr;
    float * logits = nullptr;
    float * partial_max = nullptr;
    int * partial_idx = nullptr;
#if defined(__gfx1030__) || defined(SPEC_SIDECAR_GFX1030)
    block_q8_1 * q8_embd = nullptr;
    block_q8_1_sum_hi * q8_embd_hi = nullptr;
    block_q8_1 * q8_ffn = nullptr;
    block_q8_1_sum_hi * q8_ffn_hi = nullptr;
#endif
    int * id_dev = nullptr;
    int * tok_out = nullptr;
    int * draft_count = nullptr;
    int * draft_active = nullptr;
    int * draft_ids = nullptr;
    float * draft_logits = nullptr;
    float * draft_probs = nullptr;
    float * zero = nullptr;
};

static constexpr int CATCHUP_BATCH = 64;

struct CatchupScratch {
    float * token = nullptr;
    float * hidden = nullptr;
    float * token_norm = nullptr;
    float * hidden_norm = nullptr;
    float * concat = nullptr;
    float * projected = nullptr;
    float * attention_input = nullptr;
    float * key = nullptr;
    float * value = nullptr;
};

struct Weights {
    const block_q4_0 * token_embedding = nullptr;
    const block_q4_0 * eh_proj = nullptr;
    const float * enorm = nullptr;
    const float * hnorm = nullptr;
    const float * attn_norm = nullptr;
    const block_q4_0 * attn_q = nullptr;
    const float * attn_q_norm = nullptr;
    const block_q4_0 * attn_k = nullptr;
    const float * attn_k_norm = nullptr;
    const block_q4_0 * attn_v = nullptr;
    const block_q4_0 * attn_output = nullptr;
    const float * post_attention_norm = nullptr;
    const block_q4_0 * ffn_gate_exps = nullptr;
    const block_q4_0 * ffn_up_exps = nullptr;
    const block_q4_0 * ffn_down_exps = nullptr;
    const float * ffn_gate_inp = nullptr;
    const block_q4_0 * ffn_gate_shexp = nullptr;
    const block_q4_0 * ffn_up_shexp = nullptr;
    const block_q4_0 * ffn_down_shexp = nullptr;
    const float * ffn_gate_inp_shexp = nullptr;
    const float * shared_head_norm = nullptr;
    const block_q4_0 * output = nullptr;
};

struct State {
    bool ok = false;
    int32_t n_seq = 1;
    int device = -1;
    int max_pos = SC_MAX_POS_DEFAULT;
    size_t n_parts = N_PARTS;
    hipStream_t stream = nullptr;
    hipStream_t owned_stream = nullptr;
    bool owns_stream = true;
    bool parallel_router = false;
    bool batched_catchup = false;
    std::vector<Tensor> tensors;
    Weights weights;
    Scratch scratch;
    CatchupScratch catchup;
    SeqState seq[SC_MAX_SEQ];
    int32_t * remap = nullptr;
    hipEvent_t event_a = nullptr;
    hipEvent_t event_b = nullptr;
#if defined(QWEN35MOE_SIDECAR_ROCPRIM_TOPK)
    void * topk_temp = nullptr;
    size_t topk_temp_bytes = 0;
    bool topk_rocprim = false;
#endif
};

State S;

#define SC_CHECK(call) do { \
    const hipError_t error = (call); \
    if (error != hipSuccess) { \
        std::fprintf(stderr, "[qwen35moe-sidecar] %s at %s:%d\n", \
                hipGetErrorString(error), __FILE__, __LINE__); \
        return -1; \
    } \
} while (0)

static Tensor * find_tensor(const char * name) {
    return spec_sidecar_artifact::find_tensor(S.tensors, name);
}

static bool expect_tensor(const char * name, const char * dtype,
        std::initializer_list<int64_t> shape, uint64_t nbytes) {
    const Tensor * tensor = spec_sidecar_artifact::find_tensor(S.tensors, name);
    return tensor != nullptr && spec_sidecar_artifact::tensor_matches(*tensor, dtype, shape, nbytes);
}

static SeqState * get_seq(int32_t seq_id) {
    return seq_id >= 0 && seq_id < S.n_seq && seq_id < SC_MAX_SEQ ? &S.seq[seq_id] : nullptr;
}

static int ensure_seq_storage(SeqState & seq) {
    if (seq.allocated) return 0;
    const size_t bytes = static_cast<size_t>(S.max_pos) * N_HEAD_KV * HEAD_DIM * sizeof(__half);
    SC_CHECK(hipMalloc(&seq.key_committed, bytes));
    SC_CHECK(hipMalloc(&seq.value_committed, bytes));
    SC_CHECK(hipMalloc(&seq.hidden_committed, N_EMBD * sizeof(float)));
    seq.allocated = true;
    return 0;
}

static int ensure_pending_storage(SeqState & seq, int count) {
    if (count <= seq.pending_capacity) return 0;
    if (seq.pending_pos_max != seq.state_pos_max) return -1;
    SC_CHECK(hipStreamSynchronize(S.stream));
    if (seq.key_pending != nullptr) SC_CHECK(hipFree(seq.key_pending));
    if (seq.value_pending != nullptr) SC_CHECK(hipFree(seq.value_pending));
    const int capacity = std::max(count, std::max(8, seq.pending_capacity * 2));
    const size_t bytes = static_cast<size_t>(capacity) * N_HEAD_KV * HEAD_DIM * sizeof(__half);
    SC_CHECK(hipMalloc(&seq.key_pending, bytes));
    SC_CHECK(hipMalloc(&seq.value_pending, bytes));
    seq.pending_capacity = capacity;
    return 0;
}

static int ensure_pending_hidden_storage(SeqState & seq, int count) {
    if (count <= seq.pending_hidden_capacity) return 0;
    SC_CHECK(hipStreamSynchronize(S.stream));
    if (seq.hidden_pending != nullptr) SC_CHECK(hipFree(seq.hidden_pending));
    const int capacity = std::max(count, std::max(8, seq.pending_hidden_capacity * 2));
    SC_CHECK(hipMalloc(&seq.hidden_pending, static_cast<size_t>(capacity) * N_EMBD * sizeof(float)));
    seq.pending_hidden_capacity = capacity;
    return 0;
}

static void discard_pending(SeqState & seq) {
    seq.pending_pos_min = seq.state_pos_max;
    seq.pending_pos_max = seq.state_pos_max;
}

// Build the MTP input and K/V projection. Catch-up only needs this half of the
// block; drafting continues through query attention, MoE, and the output head.
static void project_input(Scratch & c, const int * token, const float * hidden,
        int position, int cache_position, __half * key_cache, __half * value_cache,
        bool need_query) {
    deq_row_q4_0<<<(N_EMBD + 255) / 256, 256, 0, S.stream>>>(
            S.weights.token_embedding, token, c.token, N_EMBD, SC_N_VOCAB);
    rmsnorm<<<1, 256, 0, S.stream>>>(c.token, S.weights.enorm, c.token_norm, N_EMBD);
    rmsnorm<<<1, 256, 0, S.stream>>>(hidden, S.weights.hnorm, c.hidden_norm, N_EMBD);
    concat2<<<(2 * N_EMBD + 255) / 256, 256, 0, S.stream>>>(
            c.token_norm, c.hidden_norm, c.concat, N_EMBD);
    gemv_q4_0<<<N_EMBD, 32, 0, S.stream>>>(
            S.weights.eh_proj, c.concat, c.projected, 2 * N_EMBD, N_EMBD);
    rmsnorm<<<1, 256, 0, S.stream>>>(
            c.projected, S.weights.attn_norm, c.attention_input, N_EMBD);

    gemv_q4_0<<<N_HEAD_KV * HEAD_DIM, 32, 0, S.stream>>>(
            S.weights.attn_k, c.attention_input, c.key, N_EMBD, N_HEAD_KV * HEAD_DIM);
    gemv_q4_0<<<N_HEAD_KV * HEAD_DIM, 32, 0, S.stream>>>(
            S.weights.attn_v, c.attention_input, c.value, N_EMBD, N_HEAD_KV * HEAD_DIM);
    rmsnorm_heads<<<N_HEAD_KV, 256, 0, S.stream>>>(
            c.key, S.weights.attn_k_norm, c.key, N_HEAD_KV);
    rope_neox<<<N_HEAD_KV, N_ROT / 2, 0, S.stream>>>(c.key, N_HEAD_KV, position);
    kv_store<<<(N_HEAD_KV * HEAD_DIM + 255) / 256, 256, 0, S.stream>>>(
            c.key, c.value, key_cache, value_cache, cache_position);

    if (!need_query) return;
    gemv_q4_0<<<2 * N_HEAD * HEAD_DIM, 32, 0, S.stream>>>(
            S.weights.attn_q, c.attention_input, c.query_full, N_EMBD, 2 * N_HEAD * HEAD_DIM);
    split_q_gate<<<N_HEAD, HEAD_DIM, 0, S.stream>>>(c.query_full, c.query, c.attention_gate);
    rmsnorm_heads<<<N_HEAD, 256, 0, S.stream>>>(
            c.query, S.weights.attn_q_norm, c.query, N_HEAD);
    rope_neox<<<N_HEAD, N_ROT / 2, 0, S.stream>>>(c.query, N_HEAD, position);
}

static void run_full_block(Scratch & c, const int * token, const float * hidden,
        int position, __half * key_cache, __half * value_cache, bool emit_logits) {
    project_input(c, token, hidden, position, position, key_cache, value_cache, true);
    attn_1q<<<N_HEAD, 256, 0, S.stream>>>(
            c.query, key_cache, value_cache, c.attention, position + 1);
    multiply_sigmoid<<<(N_HEAD * HEAD_DIM + 255) / 256, 256, 0, S.stream>>>(
            c.attention, c.attention_gate, N_HEAD * HEAD_DIM);
    gemv_q4_0<<<N_EMBD, 32, 0, S.stream>>>(
            S.weights.attn_output, c.attention, c.residual, N_HEAD * HEAD_DIM, N_EMBD);
    add_inplace<<<(N_EMBD + 255) / 256, 256, 0, S.stream>>>(c.residual, c.projected, N_EMBD);
#if defined(__gfx1030__) || defined(SPEC_SIDECAR_GFX1030)
    rmsnorm_q8_1<<<1, 256, 0, S.stream>>>(
            c.residual, S.weights.post_attention_norm, c.post_norm,
            c.q8_embd, c.q8_embd_hi, N_EMBD);
#else
    rmsnorm<<<1, 256, 0, S.stream>>>(
            c.residual, S.weights.post_attention_norm, c.post_norm, N_EMBD);
#endif

    gemv_f32<<<N_EXPERT, 32, 0, S.stream>>>(
            S.weights.ffn_gate_inp, c.post_norm, c.router, N_EMBD, N_EXPERT);
    if (S.parallel_router) {
        router_top8_parallel<<<1, N_EXPERT, 0, S.stream>>>(
                c.router, c.expert_ids, c.expert_weights);
    } else {
        router_top8<<<1, 1, 0, S.stream>>>(
                c.router, c.expert_ids, c.expert_weights);
    }
    dim3 expert_grid(N_FF, N_EXPERT_USED);
#if defined(__gfx1030__) || defined(SPEC_SIDECAR_GFX1030)
    gemv_q4_q8_experts_dot8<<<expert_grid, 32, 0, S.stream>>>(
            S.weights.ffn_gate_exps, c.q8_embd, c.q8_embd_hi, c.expert_ids,
            c.expert_gate, N_EMBD, N_FF, N_EXPERT_USED, 0);
    gemv_q4_q8_experts_dot8<<<expert_grid, 32, 0, S.stream>>>(
            S.weights.ffn_up_exps, c.q8_embd, c.q8_embd_hi, c.expert_ids,
            c.expert_up, N_EMBD, N_FF, N_EXPERT_USED, 0);
#else
    gemv_q4_experts<<<expert_grid, 32, 0, S.stream>>>(
            S.weights.ffn_gate_exps, c.post_norm, c.expert_ids,
            c.expert_gate, N_EMBD, N_FF, N_EXPERT_USED);
    gemv_q4_experts<<<expert_grid, 32, 0, S.stream>>>(
            S.weights.ffn_up_exps, c.post_norm, c.expert_ids,
            c.expert_up, N_EMBD, N_FF, N_EXPERT_USED);
#endif
    silu_mul<<<(N_EXPERT_USED * N_FF + 255) / 256, 256, 0, S.stream>>>(
            c.expert_gate, c.expert_up, N_EXPERT_USED * N_FF);
    dim3 down_grid(N_EMBD, N_EXPERT_USED);
#if defined(__gfx1030__) || defined(SPEC_SIDECAR_GFX1030)
    quantize_q8_1_batched<<<N_EXPERT_USED, 256, 0, S.stream>>>(
            c.expert_gate, c.q8_ffn, c.q8_ffn_hi, N_FF, N_FF);
    gemv_q4_q8_experts_dot8<<<down_grid, 32, 0, S.stream>>>(
            S.weights.ffn_down_exps, c.q8_ffn, c.q8_ffn_hi, c.expert_ids,
            c.expert_outputs, N_FF, N_EMBD, N_EXPERT_USED, N_FF / 32);
#else
    gemv_q4_experts<<<down_grid, 32, 0, S.stream>>>(
            S.weights.ffn_down_exps, c.expert_gate, c.expert_ids,
            c.expert_outputs, N_FF, N_EMBD, N_EXPERT_USED);
#endif
    weighted_sum<<<(N_EMBD + 255) / 256, 256, 0, S.stream>>>(
            c.expert_outputs, c.expert_weights, c.moe, N_EMBD, N_EXPERT_USED);

#if defined(__gfx1030__) || defined(SPEC_SIDECAR_GFX1030)
    gemv_q4_q8_1_dot8<<<N_FF, 32, 0, S.stream>>>(
            S.weights.ffn_gate_shexp, c.q8_embd, c.q8_embd_hi, c.shared_gate, N_EMBD, N_FF);
    gemv_q4_q8_1_dot8<<<N_FF, 32, 0, S.stream>>>(
            S.weights.ffn_up_shexp, c.q8_embd, c.q8_embd_hi, c.shared_up, N_EMBD, N_FF);
#else
    gemv_q4_0<<<N_FF, 32, 0, S.stream>>>(
            S.weights.ffn_gate_shexp, c.post_norm, c.shared_gate, N_EMBD, N_FF);
    gemv_q4_0<<<N_FF, 32, 0, S.stream>>>(
            S.weights.ffn_up_shexp, c.post_norm, c.shared_up, N_EMBD, N_FF);
#endif
    silu_mul<<<(N_FF + 255) / 256, 256, 0, S.stream>>>(c.shared_gate, c.shared_up, N_FF);
#if defined(__gfx1030__) || defined(SPEC_SIDECAR_GFX1030)
    quantize_q8_1_batched<<<1, 256, 0, S.stream>>>(
            c.shared_gate, c.q8_ffn, c.q8_ffn_hi, N_FF, N_FF);
    gemv_q4_q8_1_dot8<<<N_EMBD, 32, 0, S.stream>>>(
            S.weights.ffn_down_shexp, c.q8_ffn, c.q8_ffn_hi,
            c.shared_output, N_FF, N_EMBD);
#else
    gemv_q4_0<<<N_EMBD, 32, 0, S.stream>>>(
            S.weights.ffn_down_shexp, c.shared_gate, c.shared_output, N_FF, N_EMBD);
#endif
    gemv_f32<<<1, 32, 0, S.stream>>>(
            S.weights.ffn_gate_inp_shexp, c.post_norm, c.shared_scalar, N_EMBD, 1);
    scale_sigmoid<<<(N_EMBD + 255) / 256, 256, 0, S.stream>>>(
            c.shared_output, c.shared_scalar, N_EMBD);

    add_inplace<<<(N_EMBD + 255) / 256, 256, 0, S.stream>>>(c.moe, c.shared_output, N_EMBD);
    add_inplace<<<(N_EMBD + 255) / 256, 256, 0, S.stream>>>(c.residual, c.moe, N_EMBD);
    rmsnorm<<<1, 256, 0, S.stream>>>(
            c.residual, S.weights.shared_head_norm, c.h_next, N_EMBD);
    if (emit_logits) {
#if defined(__gfx1030__) || defined(SPEC_SIDECAR_GFX1030)
        quantize_q8_1_batched<<<1, 256, 0, S.stream>>>(
                c.h_next, c.q8_embd, c.q8_embd_hi, N_EMBD, N_EMBD);
        gemv_q4_q8_1_dot8<<<N_HEAD_ROWS, 32, 0, S.stream>>>(
                S.weights.output, c.q8_embd, c.q8_embd_hi,
                c.logits, N_EMBD, N_HEAD_ROWS);
#else
        gemv_q4_0<<<N_HEAD_ROWS, 32, 0, S.stream>>>(
                S.weights.output, c.h_next, c.logits, N_EMBD, N_HEAD_ROWS);
#endif
    }
}

static int32_t resolve_max_context(int32_t requested) {
    if (requested < 1) return -1;
    int64_t effective = std::min<int64_t>(requested, SC_MAX_POS_DEFAULT);
    if (const char * value = std::getenv("LLAMA_SPEC_HIP_MAX_POS")) {
        const int64_t cap = std::atoll(value);
        if (cap < 1) return -1;
        effective = std::min(effective, cap);
    }
    return effective <= INT32_MAX ? static_cast<int32_t>(effective) : -1;
}

static int initialize_impl(const char * weights_dir, const char * ids_path,
        int32_t n_seq, int32_t requested_device, int32_t requested_max_context) {
    const int32_t max_context = resolve_max_context(requested_max_context);
    if (requested_device < -1 || n_seq < 1 || n_seq > SC_MAX_SEQ || max_context < 1) return -1;
    if (S.ok) {
        return n_seq == S.n_seq && max_context == S.max_pos &&
                (requested_device < 0 || requested_device == S.device) ? 0 : -1;
    }
    S.n_seq = n_seq;
    S.max_pos = max_context;
#if defined(__gfx1030__) || defined(SPEC_SIDECAR_GFX1030)
    S.parallel_router = true;
    S.batched_catchup = true;
#endif
    if (const char * value = std::getenv("LLAMA_SPEC_QWEN35MOE_PARALLEL_ROUTER")) {
        S.parallel_router = std::strcmp(value, "0") != 0 &&
                std::strcmp(value, "off") != 0 && std::strcmp(value, "false") != 0;
    }
    if (const char * value = std::getenv("LLAMA_SPEC_QWEN35MOE_BATCHED_CATCHUP")) {
        S.batched_catchup = std::strcmp(value, "0") != 0 &&
                std::strcmp(value, "off") != 0 && std::strcmp(value, "false") != 0;
    }
    if (requested_device >= 0) SC_CHECK(hipSetDevice(requested_device));
    SC_CHECK(hipGetDevice(&S.device));
    if (requested_device >= 0 && S.device != requested_device) return -1;

    char path[1024] = {};
    std::string error;
    if (!spec_sidecar_artifact::make_path(path, sizeof(path), weights_dir, "drafter_manifest.json") ||
            !spec_sidecar_artifact::load_manifest(path, S.tensors, error)) {
        std::fprintf(stderr, "[qwen35moe-sidecar] invalid manifest: %s\n", error.c_str());
        return -1;
    }

    if (S.tensors.size() != 22 ||
        !expect_tensor("token_embd.weight", "2", {2048, 248320}, 286064640) ||
        !expect_tensor("blk.40.attn_k.weight", "2", {2048, 512}, 589824) ||
        !expect_tensor("blk.40.attn_k_norm.weight", "0", {256}, 1024) ||
        !expect_tensor("blk.40.attn_norm.weight", "0", {2048}, 8192) ||
        !expect_tensor("blk.40.attn_output.weight", "2", {4096, 2048}, 4718592) ||
        !expect_tensor("blk.40.attn_q.weight", "2", {2048, 8192}, 9437184) ||
        !expect_tensor("blk.40.attn_q_norm.weight", "0", {256}, 1024) ||
        !expect_tensor("blk.40.attn_v.weight", "2", {2048, 512}, 589824) ||
        !expect_tensor("blk.40.ffn_down_exps.weight", "2", {512, 2048, 256}, 150994944) ||
        !expect_tensor("blk.40.ffn_down_shexp.weight", "2", {512, 2048}, 589824) ||
        !expect_tensor("blk.40.ffn_gate_exps.weight", "2", {2048, 512, 256}, 150994944) ||
        !expect_tensor("blk.40.ffn_gate_inp.weight", "0", {2048, 256}, 2097152) ||
        !expect_tensor("blk.40.ffn_gate_inp_shexp.weight", "0", {2048}, 8192) ||
        !expect_tensor("blk.40.ffn_gate_shexp.weight", "2", {2048, 512}, 589824) ||
        !expect_tensor("blk.40.ffn_up_exps.weight", "2", {2048, 512, 256}, 150994944) ||
        !expect_tensor("blk.40.ffn_up_shexp.weight", "2", {2048, 512}, 589824) ||
        !expect_tensor("blk.40.nextn.eh_proj.weight", "2", {4096, 2048}, 4718592) ||
        !expect_tensor("blk.40.nextn.enorm.weight", "0", {2048}, 8192) ||
        !expect_tensor("blk.40.nextn.hnorm.weight", "0", {2048}, 8192) ||
        !expect_tensor("blk.40.nextn.shared_head_norm.weight", "0", {2048}, 8192) ||
        !expect_tensor("blk.40.post_attention_norm.weight", "0", {2048}, 8192) ||
        !expect_tensor("output.weight", "2", {2048, 40960}, 47185920)) {
        std::fprintf(stderr, "[qwen35moe-sidecar] artifact schema mismatch\n");
        return -1;
    }

    if (!spec_sidecar_artifact::make_path(path, sizeof(path), weights_dir, "drafter_weights.bin")) return -1;
    FILE * weights_file = std::fopen(path, "rb");
    if (weights_file == nullptr) return -1;
    uint64_t weights_size = 0;
    if (!spec_sidecar_artifact::file_size(weights_file, weights_size) ||
            !spec_sidecar_artifact::validate_blob_layout(S.tensors, weights_size, error)) {
        std::fprintf(stderr, "[qwen35moe-sidecar] invalid weights layout: %s\n", error.c_str());
        std::fclose(weights_file);
        return -1;
    }

    std::vector<int32_t> remap_h(N_HEAD_ROWS);
    FILE * ids_file = ids_path == nullptr ? nullptr : std::fopen(ids_path, "rb");
    const uint64_t ids_size = static_cast<uint64_t>(N_HEAD_ROWS) * sizeof(int32_t);
    uint64_t actual_ids_size = 0;
    if (ids_file == nullptr || !spec_sidecar_artifact::file_size(ids_file, actual_ids_size) ||
            actual_ids_size != ids_size ||
            !spec_sidecar_artifact::read_exact_at(ids_file, 0, remap_h.data(), ids_size)) {
        std::fprintf(stderr, "[qwen35moe-sidecar] invalid full-vocabulary ID table\n");
        if (ids_file != nullptr) std::fclose(ids_file);
        std::fclose(weights_file);
        return -1;
    }
    std::fclose(ids_file);
    if (!spec_sidecar_artifact::validate_remap(remap_h, SC_N_VOCAB, error)) {
        std::fprintf(stderr, "[qwen35moe-sidecar] invalid ID table: %s\n", error.c_str());
        std::fclose(weights_file);
        return -1;
    }

    std::vector<uint8_t> host_buffer;
    for (Tensor & tensor : S.tensors) {
        const bool host_resident = tensor.name == "token_embd.weight";
        if (host_resident) {
            void * host = nullptr;
            SC_CHECK(hipHostMalloc(&host, static_cast<size_t>(tensor.nbytes), hipHostMallocMapped));
            if (!spec_sidecar_artifact::read_exact_at(weights_file, tensor.offset, host, tensor.nbytes)) {
                std::fclose(weights_file);
                return -1;
            }
            SC_CHECK(hipHostGetDevicePointer(&tensor.dev, host, 0));
        } else {
            host_buffer.resize(static_cast<size_t>(tensor.nbytes));
            if (!spec_sidecar_artifact::read_exact_at(
                        weights_file, tensor.offset, host_buffer.data(), tensor.nbytes)) {
                std::fclose(weights_file);
                return -1;
            }
            SC_CHECK(hipMalloc(&tensor.dev, static_cast<size_t>(tensor.nbytes)));
            SC_CHECK(hipMemcpy(tensor.dev, host_buffer.data(), static_cast<size_t>(tensor.nbytes), hipMemcpyHostToDevice));
        }
    }
    std::fclose(weights_file);
    SC_CHECK(hipMalloc(&S.remap, ids_size));
    SC_CHECK(hipMemcpy(S.remap, remap_h.data(), ids_size, hipMemcpyHostToDevice));

#define Q4(name) reinterpret_cast<const block_q4_0 *>(find_tensor(name)->dev)
#define F32(name) reinterpret_cast<const float *>(find_tensor(name)->dev)
    S.weights.token_embedding = Q4("token_embd.weight");
    S.weights.eh_proj = Q4("blk.40.nextn.eh_proj.weight");
    S.weights.enorm = F32("blk.40.nextn.enorm.weight");
    S.weights.hnorm = F32("blk.40.nextn.hnorm.weight");
    S.weights.attn_norm = F32("blk.40.attn_norm.weight");
    S.weights.attn_q = Q4("blk.40.attn_q.weight");
    S.weights.attn_q_norm = F32("blk.40.attn_q_norm.weight");
    S.weights.attn_k = Q4("blk.40.attn_k.weight");
    S.weights.attn_k_norm = F32("blk.40.attn_k_norm.weight");
    S.weights.attn_v = Q4("blk.40.attn_v.weight");
    S.weights.attn_output = Q4("blk.40.attn_output.weight");
    S.weights.post_attention_norm = F32("blk.40.post_attention_norm.weight");
    S.weights.ffn_gate_exps = Q4("blk.40.ffn_gate_exps.weight");
    S.weights.ffn_up_exps = Q4("blk.40.ffn_up_exps.weight");
    S.weights.ffn_down_exps = Q4("blk.40.ffn_down_exps.weight");
    S.weights.ffn_gate_inp = F32("blk.40.ffn_gate_inp.weight");
    S.weights.ffn_gate_shexp = Q4("blk.40.ffn_gate_shexp.weight");
    S.weights.ffn_up_shexp = Q4("blk.40.ffn_up_shexp.weight");
    S.weights.ffn_down_shexp = Q4("blk.40.ffn_down_shexp.weight");
    S.weights.ffn_gate_inp_shexp = F32("blk.40.ffn_gate_inp_shexp.weight");
    S.weights.shared_head_norm = F32("blk.40.nextn.shared_head_norm.weight");
    S.weights.output = Q4("output.weight");
#undef Q4
#undef F32

    Scratch & c = S.scratch;
#define MALLOC_F32(field, count) SC_CHECK(hipMalloc(&c.field, static_cast<size_t>(count) * sizeof(float)))
    MALLOC_F32(token, N_EMBD);
    MALLOC_F32(hidden, N_EMBD);
    MALLOC_F32(token_norm, N_EMBD);
    MALLOC_F32(hidden_norm, N_EMBD);
    MALLOC_F32(concat, 2 * N_EMBD);
    MALLOC_F32(projected, N_EMBD);
    MALLOC_F32(attention_input, N_EMBD);
    MALLOC_F32(query_full, 2 * N_HEAD * HEAD_DIM);
    MALLOC_F32(query, N_HEAD * HEAD_DIM);
    MALLOC_F32(attention_gate, N_HEAD * HEAD_DIM);
    MALLOC_F32(key, N_HEAD_KV * HEAD_DIM);
    MALLOC_F32(value, N_HEAD_KV * HEAD_DIM);
    MALLOC_F32(attention, N_HEAD * HEAD_DIM);
    MALLOC_F32(residual, N_EMBD);
    MALLOC_F32(post_norm, N_EMBD);
    MALLOC_F32(router, N_EXPERT);
    MALLOC_F32(expert_weights, N_EXPERT_USED);
    MALLOC_F32(expert_gate, N_EXPERT_USED * N_FF);
    MALLOC_F32(expert_up, N_EXPERT_USED * N_FF);
    MALLOC_F32(expert_outputs, N_EXPERT_USED * N_EMBD);
    MALLOC_F32(moe, N_EMBD);
    MALLOC_F32(shared_gate, N_FF);
    MALLOC_F32(shared_up, N_FF);
    MALLOC_F32(shared_output, N_EMBD);
    MALLOC_F32(shared_scalar, 1);
    MALLOC_F32(h_next, N_EMBD);
    MALLOC_F32(logits, N_HEAD_ROWS);
    // Argmax uses the first N_PARTS entries; top-k keeps TOP_K candidates
    // per partition in the same buffers.
    MALLOC_F32(partial_max, N_PARTS * MTP_DRAFT_TOP_K);
    SC_CHECK(hipMalloc(&c.partial_idx, N_PARTS * MTP_DRAFT_TOP_K * sizeof(int)));
#if defined(__gfx1030__) || defined(SPEC_SIDECAR_GFX1030)
    SC_CHECK(hipMalloc(&c.q8_embd, (N_EMBD / 32) * sizeof(block_q8_1)));
    SC_CHECK(hipMalloc(&c.q8_embd_hi, (N_EMBD / 32) * sizeof(block_q8_1_sum_hi)));
    SC_CHECK(hipMalloc(&c.q8_ffn, N_EXPERT_USED * (N_FF / 32) * sizeof(block_q8_1)));
    SC_CHECK(hipMalloc(&c.q8_ffn_hi, N_EXPERT_USED * (N_FF / 32) * sizeof(block_q8_1_sum_hi)));
#endif
    SC_CHECK(hipMalloc(&c.id_dev, sizeof(int)));
    SC_CHECK(hipMalloc(&c.tok_out, 8 * sizeof(int)));
    SC_CHECK(hipMalloc(&c.expert_ids, N_EXPERT_USED * sizeof(int32_t)));
    SC_CHECK(hipMalloc(&c.draft_count, sizeof(int)));
    SC_CHECK(hipMalloc(&c.draft_active, sizeof(int)));
    SC_CHECK(hipMalloc(&c.draft_ids, 8 * MTP_DRAFT_TOP_K * sizeof(int)));
    SC_CHECK(hipMalloc(&c.draft_logits, 8 * MTP_DRAFT_TOP_K * sizeof(float)));
    SC_CHECK(hipMalloc(&c.draft_probs, 8 * MTP_DRAFT_TOP_K * sizeof(float)));
    SC_CHECK(hipMalloc(&c.zero, N_EMBD * sizeof(float)));
    SC_CHECK(hipMemset(c.zero, 0, N_EMBD * sizeof(float)));
#undef MALLOC_F32

    if (S.batched_catchup) {
        CatchupScratch & b = S.catchup;
#define MALLOC_CATCHUP(field, count) SC_CHECK(hipMalloc(&b.field, static_cast<size_t>(count) * sizeof(float)))
        MALLOC_CATCHUP(token, CATCHUP_BATCH * N_EMBD);
        MALLOC_CATCHUP(hidden, CATCHUP_BATCH * N_EMBD);
        MALLOC_CATCHUP(token_norm, CATCHUP_BATCH * N_EMBD);
        MALLOC_CATCHUP(hidden_norm, CATCHUP_BATCH * N_EMBD);
        MALLOC_CATCHUP(concat, CATCHUP_BATCH * 2 * N_EMBD);
        MALLOC_CATCHUP(projected, CATCHUP_BATCH * N_EMBD);
        MALLOC_CATCHUP(attention_input, CATCHUP_BATCH * N_EMBD);
        MALLOC_CATCHUP(key, CATCHUP_BATCH * N_HEAD_KV * HEAD_DIM);
        MALLOC_CATCHUP(value, CATCHUP_BATCH * N_HEAD_KV * HEAD_DIM);
#undef MALLOC_CATCHUP
    }

    for (SeqState & seq : S.seq) seq = {};
    SC_CHECK(hipStreamCreate(&S.owned_stream));
    S.stream = S.owned_stream;
    SC_CHECK(hipEventCreate(&S.event_a));
    SC_CHECK(hipEventCreate(&S.event_b));
#if defined(QWEN35MOE_SIDECAR_ROCPRIM_TOPK)
    {
        const char * value = std::getenv("LLAMA_SPEC_QWEN35MOE_ROCPRIM_TOPK");
        const bool enabled = value == nullptr || (std::strcmp(value, "0") != 0 &&
                std::strcmp(value, "off") != 0 && std::strcmp(value, "false") != 0);
        if (enabled) {
            const rocprim::counting_iterator<int> indices(0);
            size_t bytes = 0;
            const hipError_t status = rocprim::topk_pairs<rocprim::default_config, true>(
                    nullptr, bytes, c.logits, c.partial_max, indices, c.partial_idx,
                    static_cast<uint32_t>(N_HEAD_ROWS), static_cast<uint32_t>(MTP_DRAFT_TOP_K),
                    rocprim::identity_decomposer{}, S.stream);
            if (status == hipSuccess) {
                S.topk_temp_bytes = bytes;
                if (bytes == 0 || hipMalloc(&S.topk_temp, bytes) == hipSuccess) {
                    S.topk_rocprim = true;
                } else {
                    std::fprintf(stderr, "[qwen35moe-sidecar] rocPRIM top-k scratch allocation failed; using fallback\n");
                }
            } else {
                std::fprintf(stderr, "[qwen35moe-sidecar] rocPRIM top-k query failed (%s); using fallback\n",
                        hipGetErrorString(status));
            }
        }
        if (std::getenv("LLAMA_SPEC_HIP_DEBUG") != nullptr) {
            std::fprintf(stderr, "[qwen35moe-sidecar] stochastic top-k: %s\n",
                    S.topk_rocprim ? "rocPRIM" : "fallback");
        }
    }
#endif
    S.ok = true;
    std::fprintf(stderr, "[qwen35moe-sidecar] initialized: %zu tensors, KV %d positions f16, device=%d, router=%s, catchup=%s\n",
            S.tensors.size(), S.max_pos, S.device, S.parallel_router ? "parallel" : "scalar",
            S.batched_catchup ? "batched" : "scalar");
    return 0;
}

static int catchup_impl(int32_t seq_id, const int32_t * tokens, const int32_t * positions,
        const float * hidden_rows, bool hidden_device, int count) {
    SeqState * seq = get_seq(seq_id);
    if (!S.ok || seq == nullptr || count < 0 ||
            (count > 0 && (tokens == nullptr || positions == nullptr || hidden_rows == nullptr))) return -1;
    if (count == 0) return 0;
    if (!spec_sidecar_mtp::can_begin_catchup(seq->state_pos_max, seq->hidden_valid)) return -1;
    if (count > S.max_pos - seq->state_pos_max || seq->pending_pos_max != seq->state_pos_max ||
            ensure_seq_storage(*seq) != 0 || ensure_pending_storage(*seq, count) != 0) return -1;
    for (int i = 0; i < count; ++i) {
        if (tokens[i] < 0 || tokens[i] >= SC_N_VOCAB || positions[i] != seq->state_pos_max + i ||
                positions[i] < 0 || positions[i] >= S.max_pos) return -1;
    }
    if (!hidden_device && ensure_pending_hidden_storage(*seq, count) != 0) return -1;
    if (!hidden_device) {
        SC_CHECK(hipMemcpyAsync(seq->hidden_pending, hidden_rows,
                static_cast<size_t>(count) * N_EMBD * sizeof(float), hipMemcpyHostToDevice, S.stream));
    }
    seq->pending_pos_min = seq->state_pos_max;
    seq->pending_pos_max = seq->state_pos_max + count;
    Scratch & c = S.scratch;
    if (S.batched_catchup) {
        CatchupScratch & b = S.catchup;
        const float * pending_hidden = hidden_device ? hidden_rows : seq->hidden_pending;
        for (int row0 = 0; row0 < count; row0 += CATCHUP_BATCH) {
            const int n_rows = std::min(CATCHUP_BATCH, count - row0);
            if (row0 == 0) {
                SC_CHECK(hipMemcpyAsync(b.hidden,
                        seq->hidden_valid ? seq->hidden_committed : c.zero,
                        N_EMBD * sizeof(float), hipMemcpyDeviceToDevice, S.stream));
                if (n_rows > 1) {
                    SC_CHECK(hipMemcpyAsync(b.hidden + N_EMBD, pending_hidden,
                            static_cast<size_t>(n_rows - 1) * N_EMBD * sizeof(float),
                            hipMemcpyDeviceToDevice, S.stream));
                }
            } else {
                SC_CHECK(hipMemcpyAsync(b.hidden,
                        pending_hidden + static_cast<size_t>(row0 - 1) * N_EMBD,
                        static_cast<size_t>(n_rows) * N_EMBD * sizeof(float),
                        hipMemcpyDeviceToDevice, S.stream));
            }

            for (int token_row = 0; token_row < n_rows; token_row += 8) {
                const int n_token_rows = std::min(8, n_rows - token_row);
                const int32_t t0 = tokens[row0 + token_row];
                const int32_t t1 = n_token_rows > 1 ? tokens[row0 + token_row + 1] : 0;
                const int32_t t2 = n_token_rows > 2 ? tokens[row0 + token_row + 2] : 0;
                const int32_t t3 = n_token_rows > 3 ? tokens[row0 + token_row + 3] : 0;
                const int32_t t4 = n_token_rows > 4 ? tokens[row0 + token_row + 4] : 0;
                const int32_t t5 = n_token_rows > 5 ? tokens[row0 + token_row + 5] : 0;
                const int32_t t6 = n_token_rows > 6 ? tokens[row0 + token_row + 6] : 0;
                const int32_t t7 = n_token_rows > 7 ? tokens[row0 + token_row + 7] : 0;
                deq_rows_q4_0_b8<<<dim3((N_EMBD + 255) / 256, n_token_rows), 256, 0, S.stream>>>(
                        S.weights.token_embedding, t0, t1, t2, t3, t4, t5, t6, t7,
                        b.token + static_cast<size_t>(token_row) * N_EMBD, n_token_rows);
            }
            rmsnorm_b<<<n_rows, 256, 0, S.stream>>>(
                    b.token, S.weights.enorm, b.token_norm, N_EMBD);
            rmsnorm_b<<<n_rows, 256, 0, S.stream>>>(
                    b.hidden, S.weights.hnorm, b.hidden_norm, N_EMBD);
            concat2_b<<<dim3((2 * N_EMBD + 255) / 256, n_rows), 256, 0, S.stream>>>(
                    b.token_norm, b.hidden_norm, b.concat);
            gemv_q4_0_b<<<dim3(N_EMBD, n_rows), 32, 0, S.stream>>>(
                    S.weights.eh_proj, b.concat, b.projected, 2 * N_EMBD, N_EMBD);
            rmsnorm_b<<<n_rows, 256, 0, S.stream>>>(
                    b.projected, S.weights.attn_norm, b.attention_input, N_EMBD);
            gemv_q4_0_b<<<dim3(N_HEAD_KV * HEAD_DIM, n_rows), 32, 0, S.stream>>>(
                    S.weights.attn_k, b.attention_input, b.key, N_EMBD, N_HEAD_KV * HEAD_DIM);
            gemv_q4_0_b<<<dim3(N_HEAD_KV * HEAD_DIM, n_rows), 32, 0, S.stream>>>(
                    S.weights.attn_v, b.attention_input, b.value, N_EMBD, N_HEAD_KV * HEAD_DIM);
            rmsnorm_heads_b<<<dim3(N_HEAD_KV, n_rows), 256, 0, S.stream>>>(
                    b.key, S.weights.attn_k_norm, b.key, N_HEAD_KV);
            rope_neox_b<<<dim3(N_HEAD_KV, n_rows), N_ROT / 2, 0, S.stream>>>(
                    b.key, N_HEAD_KV, positions[row0]);
            kv_store_b<<<dim3((N_HEAD_KV * HEAD_DIM + 255) / 256, n_rows), 256, 0, S.stream>>>(
                    b.key, b.value, seq->key_pending, seq->value_pending,
                    positions[row0], seq->pending_pos_min);
        }
    } else {
        for (int i = 0; i < count; ++i) {
            const float * hidden = nullptr;
            if (i == 0) {
                SC_CHECK(hipMemcpyAsync(c.hidden,
                        seq->hidden_valid ? seq->hidden_committed : c.zero,
                        N_EMBD * sizeof(float), hipMemcpyDeviceToDevice, S.stream));
                hidden = c.hidden;
            } else if (hidden_device) {
                hidden = hidden_rows + static_cast<size_t>(i - 1) * N_EMBD;
            } else {
                hidden = seq->hidden_pending + static_cast<size_t>(i - 1) * N_EMBD;
            }
            SC_CHECK(hipMemcpyAsync(c.id_dev, &tokens[i], sizeof(int), hipMemcpyHostToDevice, S.stream));
            project_input(c, c.id_dev, hidden, positions[i], positions[i] - seq->pending_pos_min,
                    seq->key_pending, seq->value_pending, false);
        }
    }
    if (!hidden_device) SC_CHECK(hipStreamSynchronize(S.stream));
    return 0;
}

static int draft_impl(int32_t seq_id, int32_t last_token, int32_t past_tokens,
        const float * hidden, bool hidden_device, int count, int32_t * output_ids) {
    SeqState * seq = get_seq(seq_id);
    if (!S.ok || seq == nullptr || count < 1 || count > 8 || output_ids == nullptr ||
            (!hidden_device && hidden == nullptr) || last_token < 0 || last_token >= SC_N_VOCAB ||
            past_tokens < 0 || past_tokens != seq->state_pos_max || !seq->allocated ||
            (hidden_device && !seq->hidden_valid) || past_tokens + count > S.max_pos) return -1;
    Scratch & c = S.scratch;
    if (hidden_device) {
        hidden = seq->hidden_committed;
    } else {
        SC_CHECK(hipMemcpyAsync(c.hidden, hidden, N_EMBD * sizeof(float), hipMemcpyHostToDevice, S.stream));
        hidden = c.hidden;
    }
    SC_CHECK(hipMemcpyAsync(c.id_dev, &last_token, sizeof(last_token), hipMemcpyHostToDevice, S.stream));
    for (int i = 0; i < count; ++i) {
        const int * token = i == 0 ? c.id_dev : c.tok_out + i - 1;
        const float * input_hidden = i == 0 ? hidden : c.h_next;
        run_full_block(c, token, input_hidden, past_tokens + i,
                seq->key_committed, seq->value_committed, true);
        argmax_partial<<<N_PARTS, 256, 0, S.stream>>>(
                c.logits, N_HEAD_ROWS, c.partial_max, c.partial_idx);
        argmax_final<<<1, 1, 0, S.stream>>>(
                c.partial_max, c.partial_idx, N_PARTS, c.tok_out + i);
    }
    SC_CHECK(hipMemcpyAsync(output_ids, c.tok_out,
            static_cast<size_t>(count) * sizeof(int32_t), hipMemcpyDeviceToHost, S.stream));
    SC_CHECK(hipStreamSynchronize(S.stream));
    for (int i = 0; i < count; ++i) if (output_ids[i] < 0 || output_ids[i] >= SC_N_VOCAB) return -1;
    return count;
}

static int stochastic_impl(int32_t seq_id, int32_t last_token, int32_t past_tokens,
        const float * hidden, bool hidden_device, float temperature, float p_min,
        uint64_t rng_key, int count, int32_t * output_ids, int32_t * dist_ids, float * dist_probs) {
    SeqState * seq = get_seq(seq_id);
    if (!S.ok || seq == nullptr || count < 1 || count > 8 || output_ids == nullptr ||
            dist_ids == nullptr || dist_probs == nullptr || (!hidden_device && hidden == nullptr) ||
            last_token < 0 || last_token >= SC_N_VOCAB || past_tokens < 0 ||
            past_tokens != seq->state_pos_max || !seq->allocated || (hidden_device && !seq->hidden_valid) ||
            past_tokens + count > S.max_pos || !std::isfinite(temperature) || temperature <= 0.0f ||
            !std::isfinite(p_min) || p_min < 0.0f) return -1;
    Scratch & c = S.scratch;
    if (hidden_device) {
        hidden = seq->hidden_committed;
    } else {
        SC_CHECK(hipMemcpyAsync(c.hidden, hidden, N_EMBD * sizeof(float), hipMemcpyHostToDevice, S.stream));
        hidden = c.hidden;
    }
    const int zero = 0;
    const int one = 1;
    SC_CHECK(hipMemcpyAsync(c.id_dev, &last_token, sizeof(last_token), hipMemcpyHostToDevice, S.stream));
    SC_CHECK(hipMemcpyAsync(c.draft_count, &zero, sizeof(zero), hipMemcpyHostToDevice, S.stream));
    SC_CHECK(hipMemcpyAsync(c.draft_active, &one, sizeof(one), hipMemcpyHostToDevice, S.stream));
    for (int i = 0; i < count; ++i) {
        const int * token = i == 0 ? c.id_dev : c.tok_out + i - 1;
        const float * input_hidden = i == 0 ? hidden : c.h_next;
        run_full_block(c, token, input_hidden, past_tokens + i,
                seq->key_committed, seq->value_committed, true);
#if defined(QWEN35MOE_SIDECAR_ROCPRIM_TOPK)
        if (S.topk_rocprim) {
            const rocprim::counting_iterator<int> indices(0);
            size_t bytes = S.topk_temp_bytes;
            const hipError_t status = rocprim::topk_pairs<rocprim::default_config, true>(
                    S.topk_temp, bytes, c.logits, c.partial_max, indices, c.partial_idx,
                    static_cast<uint32_t>(N_HEAD_ROWS), static_cast<uint32_t>(MTP_DRAFT_TOP_K),
                    rocprim::identity_decomposer{}, S.stream);
            SC_CHECK(status);
            sort_remap_topk<<<1, MTP_DRAFT_TOP_K, 0, S.stream>>>(
                    c.partial_max, c.partial_idx, S.remap,
                    c.draft_ids + static_cast<size_t>(i) * MTP_DRAFT_TOP_K,
                    c.draft_logits + static_cast<size_t>(i) * MTP_DRAFT_TOP_K);
        } else
#endif
        {
            topk_partial<<<N_PARTS, 256, 0, S.stream>>>(
                    c.logits, c.partial_max, c.partial_idx, N_PARTS);
            topk_final<<<1, 1, 0, S.stream>>>(
                    c.partial_max, c.partial_idx, N_PARTS, S.remap,
                    c.draft_ids + static_cast<size_t>(i) * MTP_DRAFT_TOP_K,
                    c.draft_logits + static_cast<size_t>(i) * MTP_DRAFT_TOP_K);
        }
        sample_topk<<<1, 1, 0, S.stream>>>(
                c.draft_ids + static_cast<size_t>(i) * MTP_DRAFT_TOP_K,
                c.draft_logits + static_cast<size_t>(i) * MTP_DRAFT_TOP_K,
                temperature, p_min, spec_sidecar_stochastic_uniform(rng_key, static_cast<uint32_t>(i)),
                i, c.tok_out, c.draft_probs, c.draft_count, c.draft_active);
    }
    int host_count = 0;
    int host_ids[8] = {};
    int host_dist_ids[8 * MTP_DRAFT_TOP_K] = {};
    float host_dist_probs[8 * MTP_DRAFT_TOP_K] = {};
    SC_CHECK(hipMemcpyAsync(&host_count, c.draft_count, sizeof(host_count), hipMemcpyDeviceToHost, S.stream));
    SC_CHECK(hipMemcpyAsync(host_ids, c.tok_out, sizeof(host_ids), hipMemcpyDeviceToHost, S.stream));
    SC_CHECK(hipMemcpyAsync(host_dist_ids, c.draft_ids, sizeof(host_dist_ids), hipMemcpyDeviceToHost, S.stream));
    SC_CHECK(hipMemcpyAsync(host_dist_probs, c.draft_probs, sizeof(host_dist_probs), hipMemcpyDeviceToHost, S.stream));
    SC_CHECK(hipStreamSynchronize(S.stream));
    if (host_count < 0 || host_count > count) return -1;
    for (int i = 0; i < host_count; ++i) {
        output_ids[i] = host_ids[i];
        std::memcpy(dist_ids + static_cast<size_t>(i) * MTP_DRAFT_TOP_K,
                host_dist_ids + static_cast<size_t>(i) * MTP_DRAFT_TOP_K,
                MTP_DRAFT_TOP_K * sizeof(int32_t));
        std::memcpy(dist_probs + static_cast<size_t>(i) * MTP_DRAFT_TOP_K,
                host_dist_probs + static_cast<size_t>(i) * MTP_DRAFT_TOP_K,
                MTP_DRAFT_TOP_K * sizeof(float));
    }
    return host_count;
}

} // namespace

#ifdef _WIN32
#define SC_API extern "C" __declspec(dllexport)
#else
#define SC_API extern "C"
#endif

SC_API int spec_hip_release_abi() { return SC_RELEASE_ABI; }
SC_API int spec_hip_check(int32_t n_embd, int32_t head_rows, int32_t n_seq) {
    return n_embd == N_EMBD && head_rows == N_HEAD_ROWS && n_seq >= 1 && n_seq <= SC_MAX_SEQ ? 0 : -1;
}
SC_API int spec_hip_state_size() { return static_cast<int>(sizeof(spec_sidecar_state)); }
SC_API int spec_hip_get_state(int32_t seq_id, void * data, int size) {
    SeqState * seq = get_seq(seq_id);
    if (!S.ok || seq == nullptr || data == nullptr || size != static_cast<int>(sizeof(spec_sidecar_state))) return -1;
    spec_sidecar_state state = {};
    state.magic = SPEC_SIDECAR_STATE_MAGIC;
    state.version = SPEC_SIDECAR_STATE_VERSION;
    state.kind = SPEC_SIDECAR_STATE_KIND_MTP;
    state.pos_min = seq->state_pos_min;
    state.pos_max = seq->state_pos_max;
    state.epoch = seq->state_epoch;
    std::memcpy(data, &state, sizeof(state));
    return 0;
}

static bool state_matches(SeqState * seq, const void * data, int size, spec_sidecar_state & state) {
    if (!S.ok || seq == nullptr || data == nullptr || size != static_cast<int>(sizeof(state))) return false;
    std::memcpy(&state, data, sizeof(state));
    return state.magic == SPEC_SIDECAR_STATE_MAGIC && state.version == SPEC_SIDECAR_STATE_VERSION &&
           state.kind == SPEC_SIDECAR_STATE_KIND_MTP && state.epoch == seq->state_epoch &&
           state.pos_min == seq->state_pos_min && state.pos_min >= 0 && state.pos_min <= state.pos_max &&
           state.pos_max <= seq->state_pos_max && state.pos_max <= S.max_pos;
}

SC_API int spec_hip_set_state(int32_t seq_id, const void * data, int size) {
    SeqState * seq = get_seq(seq_id);
    spec_sidecar_state state = {};
    if (!state_matches(seq, data, size, state)) return -1;
    seq->hidden_valid = seq->hidden_valid &&
            spec_sidecar_mtp::committed_hidden_matches_tip(seq->state_pos_max, state.pos_max);
    seq->state_pos_max = state.pos_max;
    discard_pending(*seq);
    return 0;
}
SC_API int spec_hip_reset_state(int32_t seq_id) {
    SeqState * seq = get_seq(seq_id);
    if (!S.ok || seq == nullptr) return -1;
    if (++seq->state_epoch == 0) ++seq->state_epoch;
    seq->state_pos_min = 0;
    seq->state_pos_max = 0;
    seq->hidden_valid = false;
    discard_pending(*seq);
    return 0;
}
SC_API int spec_hip_truncate_state(int32_t seq_id, int32_t pos_max) {
    SeqState * seq = get_seq(seq_id);
    if (!S.ok || seq == nullptr || pos_max < seq->state_pos_min || pos_max > seq->state_pos_max) return -1;
    seq->hidden_valid = seq->hidden_valid &&
            spec_sidecar_mtp::committed_hidden_matches_tip(seq->state_pos_max, pos_max);
    seq->state_pos_max = pos_max;
    discard_pending(*seq);
    return 0;
}
SC_API int spec_hip_commit_state(int32_t seq_id, int32_t pos_max, const float * hidden_device) {
    SeqState * seq = get_seq(seq_id);
    if (!S.ok || seq == nullptr || pos_max < seq->state_pos_max || pos_max > seq->pending_pos_max ||
            pos_max > S.max_pos || (pos_max > seq->state_pos_max &&
            (!seq->allocated || seq->key_pending == nullptr || seq->value_pending == nullptr))) return -1;
    if (pos_max > seq->state_pos_max) {
        const size_t source_row = static_cast<size_t>(seq->state_pos_max - seq->pending_pos_min);
        const size_t count = static_cast<size_t>(pos_max - seq->state_pos_max);
        const size_t row_bytes = static_cast<size_t>(N_HEAD_KV * HEAD_DIM) * sizeof(__half);
        SC_CHECK(hipMemcpyAsync(seq->key_committed + static_cast<size_t>(seq->state_pos_max) * N_HEAD_KV * HEAD_DIM,
                seq->key_pending + source_row * N_HEAD_KV * HEAD_DIM, count * row_bytes,
                hipMemcpyDeviceToDevice, S.stream));
        SC_CHECK(hipMemcpyAsync(seq->value_committed + static_cast<size_t>(seq->state_pos_max) * N_HEAD_KV * HEAD_DIM,
                seq->value_pending + source_row * N_HEAD_KV * HEAD_DIM, count * row_bytes,
                hipMemcpyDeviceToDevice, S.stream));
        if (hidden_device != nullptr) {
            SC_CHECK(hipMemcpyAsync(seq->hidden_committed, hidden_device,
                    N_EMBD * sizeof(float), hipMemcpyDeviceToDevice, S.stream));
        } else {
            const size_t hidden_row = static_cast<size_t>(pos_max - 1 - seq->pending_pos_min);
            if (seq->hidden_pending == nullptr || hidden_row >= static_cast<size_t>(seq->pending_hidden_capacity)) return -1;
            SC_CHECK(hipMemcpyAsync(seq->hidden_committed,
                    seq->hidden_pending + hidden_row * N_EMBD,
                    N_EMBD * sizeof(float), hipMemcpyDeviceToDevice, S.stream));
        }
        seq->hidden_valid = true;
        seq->state_pos_max = pos_max;
    }
    discard_pending(*seq);
    return 0;
}

__global__ static void shift_rows(__half * values, int first, int last, int delta, int width) {
    const int direction = delta > 0 ? -1 : 1;
    int position = delta > 0 ? last - 1 : first;
    for (int row = 0; row < last - first; ++row, position += direction) {
        const size_t source = static_cast<size_t>(position) * width;
        const size_t destination = static_cast<size_t>(position + delta) * width;
        for (int i = static_cast<int>(threadIdx.x); i < width; i += static_cast<int>(blockDim.x)) {
            values[destination + i] = values[source + i];
        }
        __syncthreads();
    }
}

SC_API int spec_hip_rebase_state(int32_t seq_id, int32_t pos_min, int32_t pos_max, int32_t delta) {
    SeqState * seq = get_seq(seq_id);
    if (!S.ok || seq == nullptr || pos_min < 0 || pos_min > pos_max || pos_max > S.max_pos ||
            seq->pending_pos_max != seq->state_pos_max) return -1;
    if (delta == 0 || seq->state_pos_max <= pos_min) {
        if (delta != 0 && ++seq->state_epoch == 0) ++seq->state_epoch;
        return 0;
    }
    const int64_t new_min = static_cast<int64_t>(pos_min) + delta;
    const int64_t new_max = static_cast<int64_t>(seq->state_pos_max) + delta;
    if (!seq->allocated || new_min < 0 || new_max < new_min || new_max > S.max_pos) return -1;
    SC_CHECK(hipStreamSynchronize(S.stream));
    shift_rows<<<1, 256, 0, S.stream>>>(seq->key_committed, pos_min, seq->state_pos_max, delta, N_HEAD_KV * HEAD_DIM);
    shift_rows<<<1, 256, 0, S.stream>>>(seq->value_committed, pos_min, seq->state_pos_max, delta, N_HEAD_KV * HEAD_DIM);
    SC_CHECK(hipStreamSynchronize(S.stream));
    if (seq->state_pos_min >= pos_min) seq->state_pos_min += delta;
    seq->state_pos_max = static_cast<int32_t>(new_max);
    discard_pending(*seq);
    if (++seq->state_epoch == 0) ++seq->state_epoch;
    return 0;
}

SC_API int spec_hip_attach_target_stream(void * stream, int32_t device) {
    if (!S.ok || device != S.device) return -1;
    SC_CHECK(hipSetDevice(S.device));
    if (stream == nullptr) {
        if (S.stream != S.owned_stream) {
            SC_CHECK(hipStreamSynchronize(S.stream));
            S.stream = S.owned_stream;
            S.owns_stream = true;
        }
        return 0;
    }
    const hipStream_t target = reinterpret_cast<hipStream_t>(stream);
    if (target != S.stream) {
        SC_CHECK(hipStreamSynchronize(S.stream));
        S.stream = target;
        S.owns_stream = false;
    }
    return 0;
}

SC_API int spec_hip_init_context(const char * weights_dir, const char * ids_path,
        int32_t n_seq, int32_t max_context) {
    return initialize_impl(weights_dir, ids_path, n_seq, -1, max_context);
}
SC_API int spec_hip_init_device_context(const char * weights_dir, const char * ids_path,
        int32_t n_seq, int32_t device, int32_t max_context) {
    return initialize_impl(weights_dir, ids_path, n_seq, device, max_context);
}
SC_API int spec_hip_init(const char * weights_dir, const char * ids_path, int32_t n_seq) {
    return initialize_impl(weights_dir, ids_path, n_seq, -1, SC_MAX_POS_DEFAULT);
}
SC_API int spec_hip_init_device(const char * weights_dir, const char * ids_path,
        int32_t n_seq, int32_t device) {
    return initialize_impl(weights_dir, ids_path, n_seq, device, SC_MAX_POS_DEFAULT);
}
SC_API int spec_hip_catchup(int32_t seq_id, const int32_t * tokens, const int32_t * positions,
        const float * hidden_rows, int count) {
    return catchup_impl(seq_id, tokens, positions, hidden_rows, false, count);
}
SC_API int spec_hip_catchup_device(int32_t seq_id, const int32_t * tokens, const int32_t * positions,
        const float * hidden_rows_device, int count) {
    return catchup_impl(seq_id, tokens, positions, hidden_rows_device, true, count);
}
SC_API int spec_hip_draft(int32_t seq_id, int32_t last_token, int32_t past_tokens,
        const float * hidden, int max_draft, int32_t * output_ids) {
    return draft_impl(seq_id, last_token, past_tokens, hidden, false, max_draft, output_ids);
}
SC_API int spec_hip_draft_device(int32_t seq_id, int32_t last_token, int32_t past_tokens,
        int max_draft, int32_t * output_ids) {
    return draft_impl(seq_id, last_token, past_tokens, nullptr, true, max_draft, output_ids);
}
SC_API int spec_hip_stochastic_top_k() { return MTP_DRAFT_TOP_K; }
SC_API int spec_hip_draft_stochastic(int32_t seq_id, int32_t last_token, int32_t past_tokens,
        const float * hidden, float temperature, float p_min, uint64_t rng_key, int max_draft,
        int32_t * output_ids, int32_t * dist_ids, float * dist_probs) {
    return stochastic_impl(seq_id, last_token, past_tokens, hidden, false, temperature, p_min,
            rng_key, max_draft, output_ids, dist_ids, dist_probs);
}
SC_API int spec_hip_draft_stochastic_device(int32_t seq_id, int32_t last_token, int32_t past_tokens,
        float temperature, float p_min, uint64_t rng_key, int max_draft, int32_t * output_ids,
        int32_t * dist_ids, float * dist_probs) {
    return stochastic_impl(seq_id, last_token, past_tokens, nullptr, true, temperature, p_min,
            rng_key, max_draft, output_ids, dist_ids, dist_probs);
}
