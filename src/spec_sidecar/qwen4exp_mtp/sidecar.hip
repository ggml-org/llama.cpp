// SPDX-License-Identifier: MIT
// Qwen3.8 Flash Next MTP sidecar.
//
// The production ABI is enabled only for the dedicated opt-in CMake target.
// Test builds retain the historical phase-boundary helpers used by the exact
// native-parity fixtures.
#include "kernels.hip"
#if defined(QWEN4EXP_SIDECAR_PRODUCTION) || defined(QWEN4EXP_SIDECAR_NATIVE_ATTN_TEST)
#define QWEN4EXP_SIDECAR_HAS_NATIVE_ATTN 1
#include "native_attention.h"
#endif
#ifdef QWEN4EXP_SIDECAR_NATIVE_MMQ_TEST
#include "native_mmq.h"
#endif
#include "moe_kernels.hip"
#include "output_kernels.hip"
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

using namespace qwen4exp_sidecar;
using Tensor = spec_sidecar_artifact::TensorDesc;

namespace {

static constexpr int32_t SC_RELEASE_ABI = SPEC_SIDECAR_MTP_RELEASE_ABI;
static constexpr int SC_MAX_SEQ = 8;
static constexpr int SC_MAX_POS = 131072;
static constexpr int KV_WIDTH = N_HEAD_KV * HEAD_DIM;
static constexpr int IDX_WIDTH = INDEX_DIM;
static constexpr int QSA_MAX_BLOCKS = SC_MAX_POS / QSA_RATIO;
static constexpr int QSA_STAGE_A_COUNT =
        ((QSA_MAX_BLOCKS + QSA_SORT_CAPACITY - 1) / QSA_SORT_CAPACITY) *
        QSA_MAX_RANKED_BLOCKS;
static constexpr int QSA_STAGE_B_COUNT =
        ((QSA_STAGE_A_COUNT + QSA_SORT_CAPACITY - 1) / QSA_SORT_CAPACITY) *
        QSA_MAX_RANKED_BLOCKS;
static constexpr int QSA_MAX_ATTENTION_PARTS = 512;
static constexpr int QUANTIZED_MATRIX_ROW_PADDING = 512;

// Match ggml_backend_cuda_buffer_type_get_alloc_size(): CUDA MMQ kernels may
// read the final quantized row through MATRIX_ROW_PADDING, and ggml zeros that
// allocation tail specifically to prevent masked out-of-range values becoming NaN.
static size_t quantized_tensor_tail_padding(const Tensor & tensor) {
    if (tensor.shape.empty()) return 0;
    const int64_t remainder = tensor.shape[0] % QUANTIZED_MATRIX_ROW_PADDING;
    if (remainder == 0) return 0;
    const int64_t values = QUANTIZED_MATRIX_ROW_PADDING - remainder;
    if (tensor.dtype == "2") { // GGML_TYPE_Q4_0
        return static_cast<size_t>(values / 32) * sizeof(block_q4_0);
    }
    if (tensor.dtype == "14") { // GGML_TYPE_Q6_K
        return static_cast<size_t>(values / 256) * sizeof(block_q6_K);
    }
    return 0;
}

struct SeqState {
    __half * key_committed = nullptr;
    __half * value_committed = nullptr;
    __half * index_committed = nullptr;
    __half * key_pending = nullptr;
    __half * value_pending = nullptr;
    __half * index_pending = nullptr;
    float * hidden_committed = nullptr;
    float * hidden_pending = nullptr;
    int pending_capacity = 0;
    int32_t pending_pos_min = 0;
    int32_t pending_pos_max = 0;
    int32_t state_pos_min = 0;
    int32_t state_pos_max = 0;
    uint64_t state_epoch = 1;
    bool hidden_valid = false;
    bool allocated = false;
};

struct CatchupScratch {
    float * token = nullptr;
    float * token_norm = nullptr;
    float * token_norm_hc = nullptr;
    float * hidden_norm = nullptr;
    float * packed_eh = nullptr;
    float * eh = nullptr;
    float * hc_norm = nullptr;
    float * hc_low = nullptr;
    float * hc_gate = nullptr;
    float * hc_inject = nullptr;
    float * mixed = nullptr;
    float * key = nullptr;
    float * value = nullptr;
    float * index_key = nullptr;
    block_q8_1 * packed_eh_q8 = nullptr;
    block_q8_1 * hc_norm_q8 = nullptr;
    block_q8_1 * hc_low_q8 = nullptr;
    block_q8_1 * mixed_q8 = nullptr;
    block_q8_1_sum_hi * hc_norm_hi = nullptr;
    block_q8_1_sum_hi * hc_low_hi = nullptr;
    block_q8_1_sum_hi * mixed_hi = nullptr;
    float * phase1_mixed = nullptr;
    float * query_full = nullptr;
    float * query = nullptr;
    float * query_normalized = nullptr;
    float * attention_gate = nullptr;
    float * attention = nullptr;
    float * attention_pregate = nullptr;
    block_q8_1 * attention_q8 = nullptr;
    block_q8_1_sum_hi * attention_hi = nullptr;
    float * attention_output = nullptr;
    float * attention_hc = nullptr;
    float * router_logits = nullptr;
    int32_t * expert_ids = nullptr;
    float * expert_weights = nullptr;
    float * expert_gate = nullptr;
    float * expert_up = nullptr;
    block_q8_1 * expert_q8 = nullptr;
    block_q8_1_sum_hi * expert_hi = nullptr;
    float * expert_output = nullptr;
    float * moe_output = nullptr;
    float * ffn_mixed_snapshot = nullptr;
    float * shared_gate = nullptr;
    float * shared_up = nullptr;
    block_q8_1 * shared_q8 = nullptr;
    block_q8_1_sum_hi * shared_hi = nullptr;
    float * shared_output = nullptr;
    float * shared_scalar = nullptr;
    block_q8_1_layout_128 * output_q8 = nullptr;
    float * logits = nullptr;
    float * topk_values_1 = nullptr;
    int32_t * topk_indices_1 = nullptr;
    float * topk_values_2 = nullptr;
    int32_t * topk_indices_2 = nullptr;
    float * topk_values = nullptr;
    int32_t * topk_indices = nullptr;
    float * topk_probabilities = nullptr;
    bool proposal_valid = false;
    float * chain_query_full = nullptr;
    float * chain_query_normalized = nullptr;
    float * chain_query = nullptr;
    float * chain_attention_pregate = nullptr;
    float * chain_attention_hc = nullptr;
    float * chain_hidden = nullptr;
    float * chain_logits = nullptr;
    int32_t * chain_tokens = nullptr;
    int32_t * chain_dist_ids = nullptr;
    float * chain_dist_probs = nullptr;
    int32_t * chain_count = nullptr;
    int32_t * chain_active = nullptr;
    bool allocated = false;
};

#ifdef QWEN4EXP_SIDECAR_NATIVE_MMQ_TEST
struct PromptBatchScratch {
    static constexpr int MAX_ROWS = 15;
    float * hc_norm = nullptr;
    float * hc_low = nullptr;
    float * hc_gate = nullptr;
    float * hc_inject = nullptr;
    float * mixed = nullptr;
    float * key = nullptr;
    float * value = nullptr;
    float * index_key = nullptr;
    uint8_t * q8 = nullptr;
    size_t q8_bytes = 0;
    __half * index_input_f16 = nullptr;
    __half * index_output_f16 = nullptr;
    int32_t * nonfinite = nullptr;
    int rows = 0;
    bool allocated = false;
};
#endif

struct QsaScratch {
    float * mixed = nullptr;
    float * pooled = nullptr;
    float * normalized = nullptr;
    float * query = nullptr;
    float * head_dots = nullptr;
    float * block_heads = nullptr;
    float * scores = nullptr;
    uint8_t * selected = nullptr;
    int32_t * selected_blocks = nullptr;
    uint64_t * candidate_keys_a = nullptr;
    uint64_t * candidate_keys_b = nullptr;
    int32_t * candidate_blocks_a = nullptr;
    int32_t * candidate_blocks_b = nullptr;
#ifdef QWEN4EXP_SIDECAR_HAS_NATIVE_ATTN
    float * attention_query = nullptr;
    __half * attention_mask = nullptr;
    float * attention_parts = nullptr;
    float2 * attention_meta = nullptr;
    float * attention = nullptr;
    int last_attention_seq = -1;
    int last_attention_physical = 0;
    int last_attention_parallel = 0;
#endif
    int query_position = -1;
    int last_qsa_seq = -1;
    int last_complete_blocks = 0;
    int last_position = -1;
    int last_ranked_blocks = 0;
    bool allocated = false;
};

struct Weights {
    const block_q4_0 * token_embedding = nullptr;
    const float * embedding_norm = nullptr;
    const float * hidden_norm = nullptr;
    const block_q4_0 * eh = nullptr;
    const float * hc_norm = nullptr;
    const block_q4_0 * hc_down = nullptr;
    const block_q4_0 * hc_up = nullptr;
    const block_q4_0 * hc_inject = nullptr;
    const block_q4_0 * query = nullptr;
    const block_q4_0 * key = nullptr;
    const block_q4_0 * value = nullptr;
    const float * query_norm = nullptr;
    const float * key_norm = nullptr;
    const __half * index_key = nullptr;
    const __half * index_query = nullptr;
    const float * index_key_norm = nullptr;
    const float * index_query_norm = nullptr;
    const float * ffn_hc_norm = nullptr;
    const block_q4_0 * ffn_hc_down = nullptr;
    const block_q4_0 * ffn_hc_up = nullptr;
    const block_q4_0 * ffn_hc_inject = nullptr;
    const block_q4_0 * attention_output = nullptr;
    const float * router = nullptr;
    const block_q4_0 * expert_gate = nullptr;
    const block_q4_0 * expert_up = nullptr;
    const block_q4_0 * expert_down = nullptr;
    const block_q4_0 * shared_gate = nullptr;
    const block_q4_0 * shared_up = nullptr;
    const block_q4_0 * shared_down = nullptr;
    const float * shared_router = nullptr;
    const float * output_hc_norm = nullptr;
    const block_q4_0 * output_hc_down = nullptr;
    const block_q4_0 * output_hc_up = nullptr;
    const block_q6_K * output = nullptr;
};

struct State {
    bool ok = false;
    int32_t n_seq = 1;
    int device = -1;
    int32_t max_pos = SC_MAX_POS;
    hipStream_t stream = nullptr;
    hipStream_t owned_stream = nullptr;
    std::vector<Tensor> tensors;
    std::vector<void *> device_allocations;
    std::vector<void *> host_allocations;
    uint64_t artifact_bytes = 0;
    Weights weights;
    CatchupScratch catchup;
#ifdef QWEN4EXP_SIDECAR_NATIVE_MMQ_TEST
    PromptBatchScratch prompt_batch;
#endif
    QsaScratch qsa;
    SeqState seq[SC_MAX_SEQ];
};

State S;

struct ExpectedTensor {
    const char * name;
    const char * dtype;
    int64_t dims[3];
    int ndims;
    uint64_t nbytes;
};

static const ExpectedTensor EXPECTED_TENSORS[] = {
    {"output.weight",                         "14", {2560, 248320, 0}, 2, 521472000},
    {"output_hc_down.weight",                  "2", {10240, 320, 0}, 2, 1843200},
    {"output_hc_norm.weight",                  "0", {10240, 0, 0}, 1, 40960},
    {"output_hc_up.weight",                    "2", {320, 10240, 0}, 2, 1843200},
    {"token_embd.weight",                      "2", {2560, 248320, 0}, 2, 357580800},
    {"blk.48.attn_k.weight",                   "2", {2560, 512, 0}, 2, 737280},
    {"blk.48.attn_k_norm.weight",              "0", {256, 0, 0}, 1, 1024},
    {"blk.48.attn_output.weight",              "2", {6144, 2560, 0}, 2, 8847360},
    {"blk.48.attn_q.weight",                   "2", {2560, 12288, 0}, 2, 17694720},
    {"blk.48.attn_q_norm.weight",              "0", {256, 0, 0}, 1, 1024},
    {"blk.48.attn_v.weight",                   "2", {2560, 512, 0}, 2, 737280},
    {"blk.48.ffn_down_exps.weight",            "2", {640, 2560, 512}, 3, 471859200},
    {"blk.48.ffn_down_shexp.weight",           "2", {640, 2560, 0}, 2, 921600},
    {"blk.48.ffn_gate_exps.weight",            "2", {2560, 640, 512}, 3, 471859200},
    {"blk.48.ffn_gate_inp.weight",             "0", {2560, 512, 0}, 2, 5242880},
    {"blk.48.ffn_gate_inp_shexp.weight",       "0", {2560, 0, 0}, 1, 10240},
    {"blk.48.ffn_gate_shexp.weight",           "2", {2560, 640, 0}, 2, 921600},
    {"blk.48.ffn_up_exps.weight",              "2", {2560, 640, 512}, 3, 471859200},
    {"blk.48.ffn_up_shexp.weight",             "2", {2560, 640, 0}, 2, 921600},
    {"blk.48.hc_attn_down.weight",             "2", {10240, 320, 0}, 2, 1843200},
    {"blk.48.hc_attn_inject.weight",           "2", {10240, 4, 0}, 2, 23040},
    {"blk.48.hc_attn_norm.weight",             "0", {10240, 0, 0}, 1, 40960},
    {"blk.48.hc_attn_up.weight",               "2", {320, 10240, 0}, 2, 1843200},
    {"blk.48.hc_ffn_down.weight",              "2", {10240, 320, 0}, 2, 1843200},
    {"blk.48.hc_ffn_inject.weight",            "2", {10240, 4, 0}, 2, 23040},
    {"blk.48.hc_ffn_norm.weight",              "0", {10240, 0, 0}, 1, 40960},
    {"blk.48.hc_ffn_up.weight",                "2", {320, 10240, 0}, 2, 1843200},
    {"blk.48.indexer.k_norm.weight",           "0", {128, 0, 0}, 1, 512},
    {"blk.48.indexer.k_proj.weight",           "1", {2560, 128, 0}, 2, 655360},
    {"blk.48.indexer.q_norm.weight",           "0", {128, 0, 0}, 1, 512},
    {"blk.48.indexer.q_proj.weight",           "1", {2560, 512, 0}, 2, 2621440},
    {"blk.48.nextn.eh_proj.weight",            "2", {5120, 2560, 0}, 2, 7372800},
    {"blk.48.nextn.enorm.weight",              "0", {2560, 0, 0}, 1, 10240},
    {"blk.48.nextn.hnorm.weight",              "0", {10240, 0, 0}, 1, 40960},
};

static bool tensor_matches(const Tensor & tensor, const ExpectedTensor & expected) {
    if (tensor.name != expected.name || tensor.dtype != expected.dtype ||
            tensor.nbytes != expected.nbytes ||
            tensor.shape.size() != static_cast<size_t>(expected.ndims)) return false;
    for (int i = 0; i < expected.ndims; ++i) {
        if (tensor.shape[static_cast<size_t>(i)] != expected.dims[i]) return false;
    }
    return true;
}

static int release_artifact_allocations() {
    int result = 0;
    for (void * pointer : S.device_allocations) {
        if (pointer != nullptr && hipFree(pointer) != hipSuccess) result = -1;
    }
    for (void * pointer : S.host_allocations) {
        if (pointer != nullptr && hipHostFree(pointer) != hipSuccess) result = -1;
    }
    S.device_allocations.clear();
    S.host_allocations.clear();
    S.tensors.clear();
    S.artifact_bytes = 0;
    return result;
}

static void bind_phase1_weights() {
    auto device = [](const char * name) -> void * {
        Tensor * tensor = spec_sidecar_artifact::find_tensor(S.tensors, name);
        return tensor != nullptr ? tensor->dev : nullptr;
    };
    S.weights.token_embedding = static_cast<const block_q4_0 *>(device("token_embd.weight"));
    S.weights.embedding_norm = static_cast<const float *>(device("blk.48.nextn.enorm.weight"));
    S.weights.hidden_norm = static_cast<const float *>(device("blk.48.nextn.hnorm.weight"));
    S.weights.eh = static_cast<const block_q4_0 *>(device("blk.48.nextn.eh_proj.weight"));
    S.weights.hc_norm = static_cast<const float *>(device("blk.48.hc_attn_norm.weight"));
    S.weights.hc_down = static_cast<const block_q4_0 *>(device("blk.48.hc_attn_down.weight"));
    S.weights.hc_up = static_cast<const block_q4_0 *>(device("blk.48.hc_attn_up.weight"));
    S.weights.hc_inject = static_cast<const block_q4_0 *>(device("blk.48.hc_attn_inject.weight"));
    S.weights.query = static_cast<const block_q4_0 *>(device("blk.48.attn_q.weight"));
    S.weights.key = static_cast<const block_q4_0 *>(device("blk.48.attn_k.weight"));
    S.weights.value = static_cast<const block_q4_0 *>(device("blk.48.attn_v.weight"));
    S.weights.query_norm = static_cast<const float *>(device("blk.48.attn_q_norm.weight"));
    S.weights.key_norm = static_cast<const float *>(device("blk.48.attn_k_norm.weight"));
    S.weights.index_key = static_cast<const __half *>(device("blk.48.indexer.k_proj.weight"));
    S.weights.index_query = static_cast<const __half *>(device("blk.48.indexer.q_proj.weight"));
    S.weights.index_key_norm = static_cast<const float *>(device("blk.48.indexer.k_norm.weight"));
    S.weights.index_query_norm = static_cast<const float *>(device("blk.48.indexer.q_norm.weight"));
    S.weights.ffn_hc_norm = static_cast<const float *>(device("blk.48.hc_ffn_norm.weight"));
    S.weights.ffn_hc_down = static_cast<const block_q4_0 *>(device("blk.48.hc_ffn_down.weight"));
    S.weights.ffn_hc_up = static_cast<const block_q4_0 *>(device("blk.48.hc_ffn_up.weight"));
    S.weights.ffn_hc_inject = static_cast<const block_q4_0 *>(device("blk.48.hc_ffn_inject.weight"));
    S.weights.attention_output = static_cast<const block_q4_0 *>(device("blk.48.attn_output.weight"));
    S.weights.router = static_cast<const float *>(device("blk.48.ffn_gate_inp.weight"));
    S.weights.expert_gate = static_cast<const block_q4_0 *>(device("blk.48.ffn_gate_exps.weight"));
    S.weights.expert_up = static_cast<const block_q4_0 *>(device("blk.48.ffn_up_exps.weight"));
    S.weights.expert_down = static_cast<const block_q4_0 *>(device("blk.48.ffn_down_exps.weight"));
    S.weights.shared_gate = static_cast<const block_q4_0 *>(device("blk.48.ffn_gate_shexp.weight"));
    S.weights.shared_up = static_cast<const block_q4_0 *>(device("blk.48.ffn_up_shexp.weight"));
    S.weights.shared_down = static_cast<const block_q4_0 *>(device("blk.48.ffn_down_shexp.weight"));
    S.weights.shared_router = static_cast<const float *>(device("blk.48.ffn_gate_inp_shexp.weight"));
    S.weights.output_hc_norm = static_cast<const float *>(device("output_hc_norm.weight"));
    S.weights.output_hc_down = static_cast<const block_q4_0 *>(device("output_hc_down.weight"));
    S.weights.output_hc_up = static_cast<const block_q4_0 *>(device("output_hc_up.weight"));
    S.weights.output = static_cast<const block_q6_K *>(device("output.weight"));
}

template <typename T>
static int allocate_tracked(T *& pointer, size_t count) {
    void * allocation = nullptr;
    const hipError_t status = hipMalloc(&allocation, count * sizeof(T));
    if (status != hipSuccess) {
        std::fprintf(stderr, "[qwen4exp-sidecar] scratch allocation failed: %s\n",
                hipGetErrorString(status));
        return -1;
    }
    pointer = static_cast<T *>(allocation);
    S.device_allocations.push_back(allocation);
    return 0;
}

static int ensure_catchup_scratch() {
    CatchupScratch & c = S.catchup;
    if (c.allocated) return 0;
    if (allocate_tracked(c.token, N_EMBD) != 0 ||
            allocate_tracked(c.token_norm, N_EMBD) != 0 ||
            allocate_tracked(c.token_norm_hc, HC_WIDTH) != 0 ||
            allocate_tracked(c.hidden_norm, HC_WIDTH) != 0 ||
            allocate_tracked(c.packed_eh, N_HC * 2 * N_EMBD) != 0 ||
            allocate_tracked(c.eh, HC_WIDTH) != 0 ||
            allocate_tracked(c.hc_norm, HC_WIDTH) != 0 ||
            allocate_tracked(c.hc_low, HC_LOW_RANK) != 0 ||
            allocate_tracked(c.hc_gate, HC_WIDTH) != 0 ||
            allocate_tracked(c.hc_inject, N_HC) != 0 ||
            allocate_tracked(c.mixed, N_EMBD) != 0 ||
            allocate_tracked(c.key, KV_WIDTH) != 0 ||
            allocate_tracked(c.value, KV_WIDTH) != 0 ||
            allocate_tracked(c.index_key, IDX_WIDTH) != 0 ||
            allocate_tracked(c.packed_eh_q8, N_HC * (2 * N_EMBD / 32)) != 0 ||
            allocate_tracked(c.hc_norm_q8, HC_WIDTH / 32) != 0 ||
            allocate_tracked(c.hc_low_q8, HC_LOW_RANK / 32) != 0 ||
            allocate_tracked(c.mixed_q8, N_EMBD / 32) != 0 ||
            allocate_tracked(c.hc_norm_hi, HC_WIDTH / 32) != 0 ||
            allocate_tracked(c.hc_low_hi, HC_LOW_RANK / 32) != 0 ||
            allocate_tracked(c.mixed_hi, N_EMBD / 32) != 0 ||
            allocate_tracked(c.phase1_mixed, N_EMBD) != 0 ||
            allocate_tracked(c.query_full, 2 * N_HEAD * HEAD_DIM) != 0 ||
            allocate_tracked(c.query, N_HEAD * HEAD_DIM) != 0 ||
            allocate_tracked(c.query_normalized, N_HEAD * HEAD_DIM) != 0 ||
            allocate_tracked(c.attention_gate, N_HEAD * HEAD_DIM) != 0 ||
            allocate_tracked(c.attention, N_HEAD * HEAD_DIM) != 0 ||
            allocate_tracked(c.attention_pregate, N_HEAD * HEAD_DIM) != 0 ||
            allocate_tracked(c.attention_q8, N_HEAD * HEAD_DIM / 32) != 0 ||
            allocate_tracked(c.attention_hi, N_HEAD * HEAD_DIM / 32) != 0 ||
            allocate_tracked(c.attention_output, N_EMBD) != 0 ||
            allocate_tracked(c.attention_hc, HC_WIDTH) != 0 ||
            allocate_tracked(c.router_logits, N_EXPERT) != 0 ||
            allocate_tracked(c.expert_ids, N_EXPERT_USED) != 0 ||
            allocate_tracked(c.expert_weights, N_EXPERT_USED) != 0 ||
            allocate_tracked(c.expert_gate, N_EXPERT_USED * N_FF) != 0 ||
            allocate_tracked(c.expert_up, N_EXPERT_USED * N_FF) != 0 ||
            allocate_tracked(c.expert_q8, N_EXPERT_USED * (N_FF / 32)) != 0 ||
            allocate_tracked(c.expert_hi, N_EXPERT_USED * (N_FF / 32)) != 0 ||
            allocate_tracked(c.expert_output, N_EXPERT_USED * N_EMBD) != 0 ||
            allocate_tracked(c.moe_output, N_EMBD) != 0 ||
            allocate_tracked(c.ffn_mixed_snapshot, N_EMBD) != 0 ||
            allocate_tracked(c.shared_gate, N_FF) != 0 ||
            allocate_tracked(c.shared_up, N_FF) != 0 ||
            allocate_tracked(c.shared_q8, N_FF / 32) != 0 ||
            allocate_tracked(c.shared_hi, N_FF / 32) != 0 ||
            allocate_tracked(c.shared_output, N_EMBD) != 0 ||
            allocate_tracked(c.shared_scalar, 1) != 0 ||
            allocate_tracked(c.output_q8, N_EMBD / 128) != 0 ||
            allocate_tracked(c.logits, N_VOCAB) != 0 ||
            allocate_tracked(c.topk_values_1, 122 * PROPOSAL_TOP_K) != 0 ||
            allocate_tracked(c.topk_indices_1, 122 * PROPOSAL_TOP_K) != 0 ||
            allocate_tracked(c.topk_values_2, 2 * PROPOSAL_TOP_K) != 0 ||
            allocate_tracked(c.topk_indices_2, 2 * PROPOSAL_TOP_K) != 0 ||
            allocate_tracked(c.topk_values, PROPOSAL_TOP_K) != 0 ||
            allocate_tracked(c.topk_indices, PROPOSAL_TOP_K) != 0 ||
            allocate_tracked(c.topk_probabilities, PROPOSAL_TOP_K) != 0 ||
            allocate_tracked(c.chain_query_full, 8 * 2 * N_HEAD * HEAD_DIM) != 0 ||
            allocate_tracked(c.chain_query_normalized, 8 * N_HEAD * HEAD_DIM) != 0 ||
            allocate_tracked(c.chain_query, 8 * N_HEAD * HEAD_DIM) != 0 ||
            allocate_tracked(c.chain_attention_pregate, 8 * N_HEAD * HEAD_DIM) != 0 ||
            allocate_tracked(c.chain_attention_hc, 8 * HC_WIDTH) != 0 ||
            allocate_tracked(c.chain_hidden, 8 * HC_WIDTH) != 0 ||
            allocate_tracked(c.chain_logits, 8 * N_VOCAB) != 0 ||
            allocate_tracked(c.chain_tokens, 8) != 0 ||
            allocate_tracked(c.chain_dist_ids, 8 * PROPOSAL_TOP_K) != 0 ||
            allocate_tracked(c.chain_dist_probs, 8 * PROPOSAL_TOP_K) != 0 ||
            allocate_tracked(c.chain_count, 1) != 0 ||
            allocate_tracked(c.chain_active, 1) != 0) return -1;
    c.allocated = true;
    return 0;
}

#ifdef QWEN4EXP_SIDECAR_NATIVE_MMQ_TEST
static int ensure_prompt_batch_scratch() {
    PromptBatchScratch & b = S.prompt_batch;
    if (b.allocated) return 0;
    constexpr int rows = PromptBatchScratch::MAX_ROWS;
    b.q8_bytes = native_q4_prompt_mmq_scratch_bytes(HC_WIDTH, rows);
    if (b.q8_bytes == 0 ||
            allocate_tracked(b.hc_norm, static_cast<size_t>(rows) * HC_WIDTH) != 0 ||
            allocate_tracked(b.hc_low, static_cast<size_t>(rows) * HC_LOW_RANK) != 0 ||
            allocate_tracked(b.hc_gate, static_cast<size_t>(rows) * HC_WIDTH) != 0 ||
            allocate_tracked(b.hc_inject, static_cast<size_t>(rows) * N_HC) != 0 ||
            allocate_tracked(b.mixed, static_cast<size_t>(rows) * N_EMBD) != 0 ||
            allocate_tracked(b.key, static_cast<size_t>(rows) * KV_WIDTH) != 0 ||
            allocate_tracked(b.value, static_cast<size_t>(rows) * KV_WIDTH) != 0 ||
            allocate_tracked(b.index_key, static_cast<size_t>(rows) * IDX_WIDTH) != 0 ||
            allocate_tracked(b.q8, b.q8_bytes) != 0 ||
            allocate_tracked(b.index_input_f16, static_cast<size_t>(rows) * N_EMBD) != 0 ||
            allocate_tracked(b.index_output_f16, static_cast<size_t>(rows) * IDX_WIDTH) != 0 ||
            allocate_tracked(b.nonfinite, 1) != 0) {
        return -1;
    }
    b.allocated = true;
    return 0;
}
#endif

static int ensure_qsa_scratch() {
    QsaScratch & q = S.qsa;
    if (q.allocated) return 0;
    if (allocate_tracked(q.mixed, N_EMBD) != 0 ||
            allocate_tracked(q.pooled, static_cast<size_t>(QSA_MAX_BLOCKS) * INDEX_DIM) != 0 ||
            allocate_tracked(q.normalized, static_cast<size_t>(QSA_MAX_BLOCKS) * INDEX_DIM) != 0 ||
            allocate_tracked(q.query, INDEX_HEADS * INDEX_DIM) != 0 ||
            allocate_tracked(q.head_dots, static_cast<size_t>(INDEX_HEADS) * QSA_MAX_BLOCKS) != 0 ||
            allocate_tracked(q.block_heads, static_cast<size_t>(INDEX_HEADS) * QSA_MAX_BLOCKS) != 0 ||
            allocate_tracked(q.scores, QSA_MAX_BLOCKS) != 0 ||
            allocate_tracked(q.selected, SC_MAX_POS) != 0 ||
            allocate_tracked(q.selected_blocks, QSA_MAX_RANKED_BLOCKS) != 0 ||
            allocate_tracked(q.candidate_keys_a, QSA_STAGE_A_COUNT) != 0 ||
            allocate_tracked(q.candidate_keys_b, QSA_STAGE_B_COUNT) != 0 ||
            allocate_tracked(q.candidate_blocks_a, QSA_STAGE_A_COUNT) != 0 ||
            allocate_tracked(q.candidate_blocks_b, QSA_STAGE_B_COUNT) != 0) return -1;
    q.allocated = true;
    return 0;
}

#ifdef QWEN4EXP_SIDECAR_HAS_NATIVE_ATTN
static int ensure_native_attention_scratch() {
    QsaScratch & q = S.qsa;
    if (ensure_qsa_scratch() != 0) return -1;
    if (q.attention_query != nullptr) return 0;
    if (allocate_tracked(q.attention_query, N_HEAD * HEAD_DIM) != 0 ||
            allocate_tracked(q.attention_mask, SC_MAX_POS) != 0 ||
            allocate_tracked(q.attention_parts,
                static_cast<size_t>(N_HEAD) * QSA_MAX_ATTENTION_PARTS * HEAD_DIM) != 0 ||
            allocate_tracked(q.attention_meta,
                static_cast<size_t>(N_HEAD) * QSA_MAX_ATTENTION_PARTS) != 0 ||
            allocate_tracked(q.attention, N_HEAD * HEAD_DIM) != 0) return -1;
    return 0;
}
#endif

static int32_t resolve_max_context(int32_t requested) {
    if (requested < 1) return -1;
    int64_t effective = std::min<int64_t>(requested, SC_MAX_POS);
    if (const char * value = std::getenv("LLAMA_SPEC_HIP_MAX_POS")) {
        const int64_t override_limit = std::atoll(value);
        if (override_limit < 1) return -1;
        effective = std::min(effective, override_limit);
    }
    if (effective != requested) {
        std::fprintf(stderr, "[qwen4exp-sidecar] target context %d limited to %lld positions\n",
                requested, (long long) effective);
    }
    return (int32_t) effective;
}

static int initialize_artifacts(const char * weights_dir, const char * ids_path,
        int32_t n_seq, int32_t requested_device, int32_t requested_max_context) {
    if (requested_device < -1) return -1;
    const int32_t max_context = resolve_max_context(requested_max_context);
    if (max_context < 1) return -1;
    if (S.ok) {
        return n_seq == S.n_seq && S.max_pos == max_context &&
                (requested_device < 0 || requested_device == S.device) ? 0 : -1;
    }
    if (n_seq < 1 || n_seq > SC_MAX_SEQ) return -1;
    if (requested_device >= 0 && hipSetDevice(requested_device) != hipSuccess) return -1;
    char path[1024];
    std::string error;
    std::vector<Tensor> tensors;
    if (!spec_sidecar_artifact::make_path(path, sizeof(path), weights_dir, "drafter_manifest.json") ||
            !spec_sidecar_artifact::load_manifest(path, tensors, error)) {
        std::fprintf(stderr, "[qwen4exp-sidecar] invalid manifest: %s\n", error.c_str());
        return -1;
    }
    if (tensors.size() != sizeof(EXPECTED_TENSORS) / sizeof(EXPECTED_TENSORS[0])) {
        std::fprintf(stderr, "[qwen4exp-sidecar] manifest tensor count %zu != 34\n", tensors.size());
        return -1;
    }
    for (const ExpectedTensor & expected : EXPECTED_TENSORS) {
        const Tensor * tensor = spec_sidecar_artifact::find_tensor(tensors, expected.name);
        if (tensor == nullptr || !tensor_matches(*tensor, expected)) {
            std::fprintf(stderr, "[qwen4exp-sidecar] tensor schema mismatch: %s\n", expected.name);
            return -1;
        }
    }
    if (!spec_sidecar_artifact::make_path(path, sizeof(path), weights_dir, "drafter_weights.bin")) return -1;
    FILE * weights = std::fopen(path, "rb");
    uint64_t weights_size = 0;
    if (weights == nullptr || !spec_sidecar_artifact::file_size(weights, weights_size) ||
            !spec_sidecar_artifact::validate_blob_layout(tensors, weights_size, error)) {
        std::fprintf(stderr, "[qwen4exp-sidecar] invalid weights blob: %s\n", error.c_str());
        if (weights != nullptr) std::fclose(weights);
        return -1;
    }
    FILE * ids = ids_path != nullptr ? std::fopen(ids_path, "rb") : nullptr;
    uint64_t ids_size = 0;
    std::vector<int32_t> identity(N_VOCAB);
    const uint64_t expected_ids_size = static_cast<uint64_t>(N_VOCAB) * sizeof(int32_t);
    if (ids == nullptr || !spec_sidecar_artifact::file_size(ids, ids_size) ||
            ids_size != expected_ids_size ||
            !spec_sidecar_artifact::read_exact_at(ids, 0, identity.data(), expected_ids_size)) {
        std::fprintf(stderr, "[qwen4exp-sidecar] invalid identity ID table\n");
        if (ids != nullptr) std::fclose(ids);
        std::fclose(weights);
        return -1;
    }
    std::fclose(ids);
    for (int32_t row = 0; row < N_VOCAB; ++row) {
        if (identity[static_cast<size_t>(row)] != row) {
            std::fprintf(stderr, "[qwen4exp-sidecar] non-identity ID at row %d\n", row);
            std::fclose(weights);
            return -1;
        }
    }

    S.n_seq = n_seq;
    S.max_pos = max_context;
    S.tensors = std::move(tensors);
    S.artifact_bytes = weights_size;
    hipError_t status = hipGetDevice(&S.device);
    if (status == hipSuccess && requested_device >= 0 && S.device != requested_device) {
        status = hipErrorInvalidDevice;
    }
    if (status == hipSuccess) status = hipStreamCreate(&S.owned_stream);
    if (status != hipSuccess) {
        std::fprintf(stderr, "[qwen4exp-sidecar] HIP initialization failed: %s\n",
                hipGetErrorString(status));
        std::fclose(weights);
        S = State{};
        return -1;
    }
    S.stream = S.owned_stream;
    std::vector<uint8_t> host_buffer;
    for (Tensor & tensor : S.tensors) {
        const bool host_resident = tensor.name == "token_embd.weight";
        const size_t tensor_bytes = static_cast<size_t>(tensor.nbytes);
        const size_t tail_padding = quantized_tensor_tail_padding(tensor);
        const size_t allocation_bytes = tensor_bytes + tail_padding;
        if (host_resident) {
            void * host = nullptr;
            status = hipHostMalloc(&host, allocation_bytes, hipHostMallocMapped);
            if (status == hipSuccess && !spec_sidecar_artifact::read_exact_at(
                    weights, tensor.offset, host, tensor.nbytes)) status = hipErrorInvalidValue;
            if (status == hipSuccess && tail_padding != 0) {
                std::memset(static_cast<uint8_t *>(host) + tensor_bytes, 0, tail_padding);
            }
            if (status == hipSuccess) status = hipHostGetDevicePointer(&tensor.dev, host, 0);
            if (host != nullptr) S.host_allocations.push_back(host);
        } else {
            host_buffer.resize(tensor_bytes);
            if (!spec_sidecar_artifact::read_exact_at(
                    weights, tensor.offset, host_buffer.data(), tensor.nbytes)) {
                status = hipErrorInvalidValue;
            } else {
                status = hipMalloc(&tensor.dev, allocation_bytes);
                if (status == hipSuccess) {
                    S.device_allocations.push_back(tensor.dev);
                    status = hipMemcpy(tensor.dev, host_buffer.data(),
                            tensor_bytes, hipMemcpyHostToDevice);
                }
                if (status == hipSuccess && tail_padding != 0) {
                    status = hipMemset(static_cast<uint8_t *>(tensor.dev) + tensor_bytes,
                            0, tail_padding);
                }
            }
        }
        if (status != hipSuccess) {
            std::fprintf(stderr, "[qwen4exp-sidecar] tensor upload failed for %s: %s\n",
                    tensor.name.c_str(), hipGetErrorString(status));
            std::fclose(weights);
            release_artifact_allocations();
            (void) hipStreamDestroy(S.owned_stream);
            S = State{};
            return -1;
        }
    }
    std::fclose(weights);
    bind_phase1_weights();
    S.ok = true;
    std::fprintf(stderr, "[qwen4exp-sidecar] validated and loaded %zu tensors (%llu bytes)\n",
            S.tensors.size(), static_cast<unsigned long long>(S.artifact_bytes));
    return 0;
}

static int phase1_finish_hc_norm() {
    CatchupScratch & c = S.catchup;
    const Weights & w = S.weights;
    rms_norm_rows<<<1, 1024, 0, S.stream>>>(
            c.token, w.embedding_norm, c.token_norm, N_EMBD);
    repeat_embedding<<<(HC_WIDTH + 255) / 256, 256, 0, S.stream>>>(
            c.token_norm, c.token_norm_hc);
    pack_eh<<<(N_HC * 2 * N_EMBD + 255) / 256, 256, 0, S.stream>>>(
            c.token_norm_hc, c.hidden_norm, c.packed_eh);
    quantize_q8_1<<<dim3(2 * N_EMBD / 32, N_HC), 32, 0, S.stream>>>(
            c.packed_eh, c.packed_eh_q8, 2 * N_EMBD);
    gemv_q4_q8_batched<<<dim3(N_EMBD, N_HC), 32, 0, S.stream>>>(
            w.eh, c.packed_eh_q8, c.eh, 2 * N_EMBD, N_EMBD);
    rms_norm_rows<<<N_HC, 1024, 0, S.stream>>>(
            c.eh, w.hc_norm, c.hc_norm, N_EMBD);
    return hipGetLastError() == hipSuccess ? 0 : -1;
}

static int phase1_prepare_hc_norm(int32_t token) {
    CatchupScratch & c = S.catchup;
    const block_q4_0 * embedding = S.weights.token_embedding +
            static_cast<size_t>(token) * (N_EMBD / 32);
    dequant_q4_row<<<(N_EMBD + 255) / 256, 256, 0, S.stream>>>(
            embedding, c.token, N_EMBD);
    return phase1_finish_hc_norm();
}

static int phase1_prepare_hc_norm_device(const int32_t * token) {
    CatchupScratch & c = S.catchup;
    if (token == nullptr) return -1;
    dequant_q4_selected_row<<<(N_EMBD + 255) / 256, 256, 0, S.stream>>>(
            S.weights.token_embedding, token, c.token, N_EMBD, N_VOCAB);
    return phase1_finish_hc_norm();
}

static int phase1_from_prepared_hc_norm(SeqState & seq,
        int32_t position, int pending_row) {
    CatchupScratch & c = S.catchup;
    const Weights & w = S.weights;
    quantize_q8_1_sum_hi<<<HC_WIDTH / 32, 32, 0, S.stream>>>(
            c.hc_norm, c.hc_norm_q8, c.hc_norm_hi, HC_WIDTH);
    gemv_q4_q8_dot8_batched<<<HC_LOW_RANK, 32, 0, S.stream>>>(
            w.hc_down, c.hc_norm_q8, c.hc_norm_hi,
            c.hc_low, HC_WIDTH, HC_LOW_RANK);
    silu_scaled<<<(HC_LOW_RANK + 255) / 256, 256, 0, S.stream>>>(
            c.hc_low, HC_LOW_RANK, 1.0f / N_HC);
    quantize_q8_1_sum_hi<<<HC_LOW_RANK / 32, 32, 0, S.stream>>>(
            c.hc_low, c.hc_low_q8, c.hc_low_hi, HC_LOW_RANK);
    gemv_q4_q8_dot8_batched<<<HC_WIDTH, 32, 0, S.stream>>>(
            w.hc_up, c.hc_low_q8, c.hc_low_hi,
            c.hc_gate, HC_LOW_RANK, HC_WIDTH);
    sigmoid_values<<<(HC_WIDTH + 255) / 256, 256, 0, S.stream>>>(
            c.hc_gate, HC_WIDTH);
    gemv_q4_q8_dot8_batched<<<N_HC, 32, 0, S.stream>>>(
            w.hc_inject, c.hc_norm_q8, c.hc_norm_hi,
            c.hc_inject, HC_WIDTH, N_HC);
    hc_mixed<<<(N_EMBD + 255) / 256, 256, 0, S.stream>>>(
            c.hc_norm, c.hc_gate, c.mixed);
    quantize_q8_1_sum_hi<<<N_EMBD / 32, 32, 0, S.stream>>>(
            c.mixed, c.mixed_q8, c.mixed_hi, N_EMBD);
    gemv_q4_q8_dot8_batched<<<KV_WIDTH, 32, 0, S.stream>>>(
            w.key, c.mixed_q8, c.mixed_hi, c.key, N_EMBD, KV_WIDTH);
    gemv_q4_q8_dot8_batched<<<KV_WIDTH, 32, 0, S.stream>>>(
            w.value, c.mixed_q8, c.mixed_hi, c.value, N_EMBD, KV_WIDTH);
    rms_norm_heads<<<N_HEAD_KV, HEAD_DIM, 0, S.stream>>>(
            c.key, w.key_norm, c.key, N_HEAD_KV);
    rope_neox<<<N_HEAD_KV, N_ROT / 2, 0, S.stream>>>(
            c.key, N_HEAD_KV, position);
    gemv_f16_f32_batched<256><<<IDX_WIDTH, 256, 0, S.stream>>>(
            w.index_key, c.mixed, c.index_key, N_EMBD, IDX_WIDTH);
    store_cache_row_f16<<<(KV_WIDTH + 255) / 256, 256, 0, S.stream>>>(
            c.key, c.value, c.index_key, seq.key_pending,
            seq.value_pending, seq.index_pending, pending_row);
    const hipError_t status = hipGetLastError();
    if (status != hipSuccess) {
        std::fprintf(stderr, "[qwen4exp-sidecar] catch-up phase 1 failed: %s\n",
                hipGetErrorString(status));
        return -1;
    }
    return 0;
}

static int phase1_from_normalized_hidden(SeqState & seq, int32_t token,
        int32_t position, int pending_row) {
    if (phase1_prepare_hc_norm(token) != 0) return -1;
    return phase1_from_prepared_hc_norm(seq, position, pending_row);
}

static int phase1_from_normalized_hidden_device_token(SeqState & seq,
        const int32_t * token, int32_t position, int pending_row) {
    if (phase1_prepare_hc_norm_device(token) != 0) return -1;
    return phase1_from_prepared_hc_norm(seq, position, pending_row);
}

static int prepare_draft_query(int32_t position) {
    CatchupScratch & c = S.catchup;
    const Weights & w = S.weights;
    c.proposal_valid = false;
    if (position < 0 || position >= S.max_pos ||
            hipMemcpyAsync(c.phase1_mixed, c.mixed, N_EMBD * sizeof(float),
                hipMemcpyDeviceToDevice, S.stream) != hipSuccess) return -1;
    gemv_q4_q8_dot8_batched<<<2 * N_HEAD * HEAD_DIM, 32, 0, S.stream>>>(
            w.query, c.mixed_q8, c.mixed_hi, c.query_full,
            N_EMBD, 2 * N_HEAD * HEAD_DIM);
    split_query_gate<<<N_HEAD, HEAD_DIM, 0, S.stream>>>(
            c.query_full, c.query, c.attention_gate);
    rms_norm_heads<<<N_HEAD, HEAD_DIM, 0, S.stream>>>(
            c.query, w.query_norm, c.query, N_HEAD);
    if (hipMemcpyAsync(c.query_normalized, c.query, N_HEAD * HEAD_DIM * sizeof(float),
                hipMemcpyDeviceToDevice, S.stream) != hipSuccess) return -1;
    rope_neox<<<N_HEAD, N_ROT / 2, 0, S.stream>>>(c.query, N_HEAD, position);
    const hipError_t status = hipGetLastError();
    if (status != hipSuccess) {
        std::fprintf(stderr, "[qwen4exp-sidecar] query preparation failed: %s\n",
                hipGetErrorString(status));
        return -1;
    }
    return 0;
}

static int finish_draft_from_attention(float temperature) {
    CatchupScratch & c = S.catchup;
    const Weights & w = S.weights;
    c.proposal_valid = false;
    if (!(temperature > 0.0f && temperature <= 100.0f)) return -1;
    if (hipMemcpyAsync(c.attention_pregate, c.attention, N_HEAD * HEAD_DIM * sizeof(float),
                hipMemcpyDeviceToDevice, S.stream) != hipSuccess) return -1;
    gate_attention<<<(N_HEAD * HEAD_DIM + 255) / 256, 256, 0, S.stream>>>(
            c.attention, c.attention_gate, N_HEAD * HEAD_DIM);
    quantize_q8_1_sum_hi<<<N_HEAD * HEAD_DIM / 32, 32, 0, S.stream>>>(
            c.attention, c.attention_q8, c.attention_hi, N_HEAD * HEAD_DIM);
    gemv_q4_q8_dot8_batched<<<N_EMBD, 32, 0, S.stream>>>(
            w.attention_output, c.attention_q8, c.attention_hi,
            c.attention_output, N_HEAD * HEAD_DIM, N_EMBD);
    hc_combine<<<(HC_WIDTH + 255) / 256, 256, 0, S.stream>>>(
            c.eh, c.attention_output, c.hc_inject);
    if (hipMemcpyAsync(c.attention_hc, c.eh, HC_WIDTH * sizeof(float),
                hipMemcpyDeviceToDevice, S.stream) != hipSuccess) return -1;

    rms_norm_rows<<<N_HC, 1024, 0, S.stream>>>(
            c.eh, w.ffn_hc_norm, c.hc_norm, N_EMBD);
    quantize_q8_1_sum_hi<<<HC_WIDTH / 32, 32, 0, S.stream>>>(
            c.hc_norm, c.hc_norm_q8, c.hc_norm_hi, HC_WIDTH);
    gemv_q4_q8_dot8_batched<<<HC_LOW_RANK, 32, 0, S.stream>>>(
            w.ffn_hc_down, c.hc_norm_q8, c.hc_norm_hi,
            c.hc_low, HC_WIDTH, HC_LOW_RANK);
    silu_scaled<<<(HC_LOW_RANK + 255) / 256, 256, 0, S.stream>>>(
            c.hc_low, HC_LOW_RANK, 1.0f / N_HC);
    quantize_q8_1_sum_hi<<<HC_LOW_RANK / 32, 32, 0, S.stream>>>(
            c.hc_low, c.hc_low_q8, c.hc_low_hi, HC_LOW_RANK);
    gemv_q4_q8_dot8_batched<<<HC_WIDTH, 32, 0, S.stream>>>(
            w.ffn_hc_up, c.hc_low_q8, c.hc_low_hi,
            c.hc_gate, HC_LOW_RANK, HC_WIDTH);
    sigmoid_values<<<(HC_WIDTH + 255) / 256, 256, 0, S.stream>>>(c.hc_gate, HC_WIDTH);
    gemv_q4_q8_dot8_batched<<<N_HC, 32, 0, S.stream>>>(
            w.ffn_hc_inject, c.hc_norm_q8, c.hc_norm_hi,
            c.hc_inject, HC_WIDTH, N_HC);
    hc_mixed<<<(N_EMBD + 255) / 256, 256, 0, S.stream>>>(
            c.hc_norm, c.hc_gate, c.mixed);
    if (hipMemcpyAsync(c.ffn_mixed_snapshot, c.mixed, N_EMBD * sizeof(float),
                hipMemcpyDeviceToDevice, S.stream) != hipSuccess) return -1;
    gemv_f32_f32_batched<256><<<N_EXPERT, 256, 0, S.stream>>>(
            w.router, c.mixed, c.router_logits, N_EMBD, N_EXPERT);
    router_top10_native<<<1, N_EXPERT, 0, S.stream>>>(
            c.router_logits, c.expert_ids, c.expert_weights);
    quantize_q8_1_sum_hi<<<N_EMBD / 32, 32, 0, S.stream>>>(
            c.mixed, c.mixed_q8, c.mixed_hi, N_EMBD);
    gemv_q4_q8_dot8_experts<<<dim3(N_FF, N_EXPERT_USED), 32, 0, S.stream>>>(
            w.expert_gate, c.mixed_q8, c.mixed_hi, c.expert_ids,
            c.expert_gate, N_EMBD, N_FF, N_EXPERT_USED, 0);
    gemv_q4_q8_dot8_experts<<<dim3(N_FF, N_EXPERT_USED), 32, 0, S.stream>>>(
            w.expert_up, c.mixed_q8, c.mixed_hi, c.expert_ids,
            c.expert_up, N_EMBD, N_FF, N_EXPERT_USED, 0);
    silu_multiply<<<(N_EXPERT_USED * N_FF + 255) / 256, 256, 0, S.stream>>>(
            c.expert_gate, c.expert_up, N_EXPERT_USED * N_FF);
    quantize_q8_1_sum_hi<<<dim3(N_FF / 32, N_EXPERT_USED), 32, 0, S.stream>>>(
            c.expert_gate, c.expert_q8, c.expert_hi, N_FF);
    gemv_q4_q8_dot8_experts<<<dim3(N_EMBD, N_EXPERT_USED), 32, 0, S.stream>>>(
            w.expert_down, c.expert_q8, c.expert_hi, c.expert_ids,
            c.expert_output, N_FF, N_EMBD, N_EXPERT_USED, N_FF / 32);
    weighted_expert_sum_native<<<(N_EMBD + 255) / 256, 256, 0, S.stream>>>(
            c.expert_output, c.expert_weights, c.moe_output, N_EMBD, N_EXPERT_USED);

    gemv_q4_q8_dot8_batched<<<N_FF, 32, 0, S.stream>>>(
            w.shared_gate, c.mixed_q8, c.mixed_hi, c.shared_gate, N_EMBD, N_FF);
    gemv_q4_q8_dot8_batched<<<N_FF, 32, 0, S.stream>>>(
            w.shared_up, c.mixed_q8, c.mixed_hi, c.shared_up, N_EMBD, N_FF);
    silu_multiply<<<(N_FF + 255) / 256, 256, 0, S.stream>>>(
            c.shared_gate, c.shared_up, N_FF);
    quantize_q8_1_sum_hi<<<N_FF / 32, 32, 0, S.stream>>>(
            c.shared_gate, c.shared_q8, c.shared_hi, N_FF);
    gemv_q4_q8_dot8_batched<<<N_EMBD, 32, 0, S.stream>>>(
            w.shared_down, c.shared_q8, c.shared_hi, c.shared_output, N_FF, N_EMBD);
    gemv_f32_f32_batched<256><<<1, 256, 0, S.stream>>>(
            w.shared_router, c.mixed, c.shared_scalar, N_EMBD, 1);
    sigmoid_scale<<<(N_EMBD + 255) / 256, 256, 0, S.stream>>>(
            c.shared_output, c.shared_scalar, N_EMBD);
    add_vectors<<<(N_EMBD + 255) / 256, 256, 0, S.stream>>>(
            c.moe_output, c.shared_output, N_EMBD);
    hc_combine<<<(HC_WIDTH + 255) / 256, 256, 0, S.stream>>>(
            c.eh, c.moe_output, c.hc_inject);

    rms_norm_rows<<<N_HC, 1024, 0, S.stream>>>(
            c.eh, w.output_hc_norm, c.hc_norm, N_EMBD);
    quantize_q8_1_sum_hi<<<HC_WIDTH / 32, 32, 0, S.stream>>>(
            c.hc_norm, c.hc_norm_q8, c.hc_norm_hi, HC_WIDTH);
    gemv_q4_q8_dot8_batched<<<HC_LOW_RANK, 32, 0, S.stream>>>(
            w.output_hc_down, c.hc_norm_q8, c.hc_norm_hi,
            c.hc_low, HC_WIDTH, HC_LOW_RANK);
    silu_scaled<<<(HC_LOW_RANK + 255) / 256, 256, 0, S.stream>>>(
            c.hc_low, HC_LOW_RANK, 1.0f / N_HC);
    quantize_q8_1_sum_hi<<<HC_LOW_RANK / 32, 32, 0, S.stream>>>(
            c.hc_low, c.hc_low_q8, c.hc_low_hi, HC_LOW_RANK);
    gemv_q4_q8_dot8_batched<<<HC_WIDTH, 32, 0, S.stream>>>(
            w.output_hc_up, c.hc_low_q8, c.hc_low_hi,
            c.hc_gate, HC_LOW_RANK, HC_WIDTH);
    sigmoid_values<<<(HC_WIDTH + 255) / 256, 256, 0, S.stream>>>(c.hc_gate, HC_WIDTH);
    hc_mixed<<<(N_EMBD + 255) / 256, 256, 0, S.stream>>>(
            c.hc_norm, c.hc_gate, c.mixed);
    quantize_q8_1_layout_128<<<N_EMBD / 128, 32, 0, S.stream>>>(
            c.mixed, c.output_q8, N_EMBD);
    gemv_q6_k_q8_1_layout_128<<<N_VOCAB, 32, 0, S.stream>>>(
            w.output, c.output_q8, c.logits, N_EMBD, N_VOCAB);
    proposal_topk_vocab_partial<<<122, 256, 0, S.stream>>>(
            c.logits, N_VOCAB, c.topk_values_1, c.topk_indices_1);
    proposal_topk_candidate_partial<<<2, 256, 0, S.stream>>>(
            c.topk_values_1, c.topk_indices_1, 122 * PROPOSAL_TOP_K,
            c.topk_values_2, c.topk_indices_2);
    proposal_topk_final<<<1, 256, 0, S.stream>>>(
            c.topk_values_2, c.topk_indices_2, 2 * PROPOSAL_TOP_K,
            c.topk_values, c.topk_indices);
    proposal_normalize_top32<<<1, 1, 0, S.stream>>>(
            c.topk_values, temperature, c.topk_probabilities);
    const hipError_t status = hipGetLastError();
    if (status != hipSuccess) {
        std::fprintf(stderr, "[qwen4exp-sidecar] draft graph finish failed: %s\n",
                hipGetErrorString(status));
        return -1;
    }
    c.proposal_valid = true;
    return 0;
}

static int draft_one_short(SeqState & seq, int32_t position) {
    CatchupScratch & c = S.catchup;
    if (position < 0 || position >= 32 || prepare_draft_query(position) != 0) return -1;
    attention_dense_f16_32<<<N_HEAD, 32, 0, S.stream>>>(
            c.query, seq.key_committed, seq.value_committed,
            c.attention, position + 1);
    return finish_draft_from_attention(1.0f);
}

#define SC_CHECK(call) do { \
    const hipError_t status_ = (call); \
    if (status_ != hipSuccess) { \
        std::fprintf(stderr, "[qwen4exp-sidecar] %s at %s:%d\n", \
                hipGetErrorString(status_), __FILE__, __LINE__); \
        return -1; \
    } \
} while (0)

static SeqState * get_seq(int32_t seq_id) {
    return seq_id >= 0 && seq_id < S.n_seq && seq_id < SC_MAX_SEQ ? &S.seq[seq_id] : nullptr;
}

static void discard_pending(SeqState & seq) {
    seq.pending_pos_min = seq.state_pos_max;
    seq.pending_pos_max = seq.state_pos_max;
}

static void invalidate_qsa_for_seq(int32_t seq_id) {
    QsaScratch & q = S.qsa;
    if (q.last_qsa_seq != seq_id) return;
    q.last_qsa_seq = -1;
    q.last_complete_blocks = 0;
    q.last_position = -1;
    q.last_ranked_blocks = 0;
#ifdef QWEN4EXP_SIDECAR_HAS_NATIVE_ATTN
    q.last_attention_seq = -1;
    q.last_attention_physical = 0;
    q.last_attention_parallel = 0;
#endif
}

static int ensure_seq_storage(SeqState & seq) {
    if (seq.allocated) return 0;
    const size_t kv_bytes = static_cast<size_t>(S.max_pos) * KV_WIDTH * sizeof(__half);
    const size_t index_bytes = static_cast<size_t>(S.max_pos) * IDX_WIDTH * sizeof(__half);
    SC_CHECK(hipMalloc(&seq.key_committed, kv_bytes));
    SC_CHECK(hipMalloc(&seq.value_committed, kv_bytes));
    SC_CHECK(hipMalloc(&seq.index_committed, index_bytes));
    SC_CHECK(hipMalloc(&seq.hidden_committed, static_cast<size_t>(HC_WIDTH) * sizeof(float)));
    // Long attention rounds its physical cache width to 256 cells. Clear the
    // untouched tail so masked reads can never ingest allocator residue/NaNs.
    SC_CHECK(hipMemsetAsync(seq.key_committed, 0, kv_bytes, S.stream));
    SC_CHECK(hipMemsetAsync(seq.value_committed, 0, kv_bytes, S.stream));
    SC_CHECK(hipMemsetAsync(seq.index_committed, 0, index_bytes, S.stream));
    SC_CHECK(hipMemsetAsync(seq.hidden_committed, 0,
            static_cast<size_t>(HC_WIDTH) * sizeof(float), S.stream));
    seq.allocated = true;
    return 0;
}

static int ensure_pending_storage(SeqState & seq, int count) {
    if (count <= seq.pending_capacity) return 0;
    if (seq.pending_pos_max != seq.state_pos_max) {
        std::fprintf(stderr, "[qwen4exp-sidecar] pending resize with uncommitted rows\n");
        return -1;
    }
    SC_CHECK(hipStreamSynchronize(S.stream));
    if (seq.key_pending != nullptr) SC_CHECK(hipFree(seq.key_pending));
    if (seq.value_pending != nullptr) SC_CHECK(hipFree(seq.value_pending));
    if (seq.index_pending != nullptr) SC_CHECK(hipFree(seq.index_pending));
    if (seq.hidden_pending != nullptr) SC_CHECK(hipFree(seq.hidden_pending));
    const int capacity = std::max(count, std::max(8, seq.pending_capacity * 2));
    const size_t kv_bytes = static_cast<size_t>(capacity) * KV_WIDTH * sizeof(__half);
    const size_t index_bytes = static_cast<size_t>(capacity) * IDX_WIDTH * sizeof(__half);
    const size_t hidden_bytes = static_cast<size_t>(capacity) * HC_WIDTH * sizeof(float);
    SC_CHECK(hipMalloc(&seq.key_pending, kv_bytes));
    SC_CHECK(hipMalloc(&seq.value_pending, kv_bytes));
    SC_CHECK(hipMalloc(&seq.index_pending, index_bytes));
    SC_CHECK(hipMalloc(&seq.hidden_pending, hidden_bytes));
    seq.pending_capacity = capacity;
    return 0;
}

static int load_normalized_hidden_row(const float * source,
        bool hidden_device, bool already_normalized) {
    CatchupScratch & c = S.catchup;
    if (already_normalized) {
        const hipMemcpyKind kind = hidden_device ?
                hipMemcpyDeviceToDevice : hipMemcpyHostToDevice;
        return hipMemcpyAsync(c.hidden_norm, source,
                static_cast<size_t>(HC_WIDTH) * sizeof(float), kind, S.stream) == hipSuccess ? 0 : -1;
    }
    if (hidden_device) {
        rms_norm_rows<<<1, 1024, 0, S.stream>>>(
                source, S.weights.hidden_norm, c.hidden_norm, HC_WIDTH);
        return hipGetLastError() == hipSuccess ? 0 : -1;
    }
    if (hipMemcpyAsync(c.hidden_norm, source,
                static_cast<size_t>(HC_WIDTH) * sizeof(float),
                hipMemcpyHostToDevice, S.stream) != hipSuccess) return -1;
    rms_norm_rows<<<1, 1024, 0, S.stream>>>(
            c.hidden_norm, S.weights.hidden_norm, c.hidden_norm, HC_WIDTH);
    return hipGetLastError() == hipSuccess ? 0 : -1;
}

#ifdef QWEN4EXP_SIDECAR_NATIVE_MMQ_TEST
static int catchup_phase1_native_batch(SeqState & seq,
        const int32_t * tokens, const int32_t * positions,
        const float * hidden_rows, int rows,
        bool hidden_device, bool already_normalized) {
    CatchupScratch & c = S.catchup;
    PromptBatchScratch & b = S.prompt_batch;
    const Weights & w = S.weights;
    b.rows = 0;
    if (rows < 9 || rows > PromptBatchScratch::MAX_ROWS ||
            ensure_prompt_batch_scratch() != 0) return -1;
    for (int row = 0; row < rows; ++row) {
        const float * source = hidden_rows + static_cast<size_t>(row) * HC_WIDTH;
        if (load_normalized_hidden_row(source, hidden_device, already_normalized) != 0 ||
                phase1_prepare_hc_norm(tokens[row]) != 0 ||
                hipMemcpyAsync(b.hc_norm + static_cast<size_t>(row) * HC_WIDTH,
                    c.hc_norm, static_cast<size_t>(HC_WIDTH) * sizeof(float),
                    hipMemcpyDeviceToDevice, S.stream) != hipSuccess) return -1;
    }
    if (launch_native_q4_prompt_mmq(S.stream, w.hc_down, b.hc_norm,
                HC_WIDTH, HC_LOW_RANK, rows, b.q8, b.q8_bytes, b.hc_low) != 0) return -1;
    silu_scaled<<<(rows * HC_LOW_RANK + 255) / 256, 256, 0, S.stream>>>(
            b.hc_low, rows * HC_LOW_RANK, 1.0f / N_HC);
    if (launch_native_q4_prompt_mmq(S.stream, w.hc_up, b.hc_low,
                HC_LOW_RANK, HC_WIDTH, rows, b.q8, b.q8_bytes, b.hc_gate) != 0) return -1;
    sigmoid_values<<<(rows * HC_WIDTH + 255) / 256, 256, 0, S.stream>>>(
            b.hc_gate, rows * HC_WIDTH);
    if (launch_native_q4_prompt_mmq(S.stream, w.hc_inject, b.hc_norm,
                HC_WIDTH, N_HC, rows, b.q8, b.q8_bytes, b.hc_inject) != 0) return -1;
    for (int row = 0; row < rows; ++row) {
        hc_mixed<<<(N_EMBD + 255) / 256, 256, 0, S.stream>>>(
                b.hc_norm + static_cast<size_t>(row) * HC_WIDTH,
                b.hc_gate + static_cast<size_t>(row) * HC_WIDTH,
                b.mixed + static_cast<size_t>(row) * N_EMBD);
    }
    if (launch_native_q4_prompt_mmq(S.stream, w.key, b.mixed,
                N_EMBD, KV_WIDTH, rows, b.q8, b.q8_bytes, b.key) != 0 ||
            launch_native_q4_prompt_mmq(S.stream, w.value, b.mixed,
                N_EMBD, KV_WIDTH, rows, b.q8, b.q8_bytes, b.value) != 0) return -1;
    for (int row = 0; row < rows; ++row) {
        float * key = b.key + static_cast<size_t>(row) * KV_WIDTH;
        rms_norm_heads<<<N_HEAD_KV, HEAD_DIM, 0, S.stream>>>(
                key, w.key_norm, key, N_HEAD_KV);
        rope_neox<<<N_HEAD_KV, N_ROT / 2, 0, S.stream>>>(
                key, N_HEAD_KV, positions[row]);
    }
    if (launch_native_f16_prompt_batch(S.stream, w.index_key, b.mixed,
                N_EMBD, IDX_WIDTH, rows,
                b.index_input_f16, static_cast<size_t>(rows) * N_EMBD,
                b.index_output_f16, static_cast<size_t>(rows) * IDX_WIDTH,
                b.index_key) != 0 ||
            hipMemsetAsync(b.nonfinite, 0, sizeof(int32_t), S.stream) != hipSuccess) return -1;
    const size_t key_value_count = static_cast<size_t>(rows) * KV_WIDTH;
    const size_t index_count = static_cast<size_t>(rows) * IDX_WIDTH;
    mark_nonfinite_f32<<<(key_value_count + 255) / 256, 256, 0, S.stream>>>(
            b.key, key_value_count, b.nonfinite);
    mark_nonfinite_f32<<<(key_value_count + 255) / 256, 256, 0, S.stream>>>(
            b.value, key_value_count, b.nonfinite);
    mark_nonfinite_f32<<<(index_count + 255) / 256, 256, 0, S.stream>>>(
            b.index_key, index_count, b.nonfinite);
    int32_t nonfinite = 0;
    if (hipGetLastError() != hipSuccess ||
            hipMemcpyAsync(&nonfinite, b.nonfinite, sizeof(nonfinite),
                hipMemcpyDeviceToHost, S.stream) != hipSuccess ||
            hipStreamSynchronize(S.stream) != hipSuccess || nonfinite != 0) return -1;
    float_to_half<<<(rows * KV_WIDTH + 255) / 256, 256, 0, S.stream>>>(
            b.key, seq.key_pending, rows * KV_WIDTH);
    float_to_half<<<(rows * KV_WIDTH + 255) / 256, 256, 0, S.stream>>>(
            b.value, seq.value_pending, rows * KV_WIDTH);
    float_to_half<<<(rows * IDX_WIDTH + 255) / 256, 256, 0, S.stream>>>(
            b.index_key, seq.index_pending, rows * IDX_WIDTH);

    const int last = rows - 1;
    if (hipMemcpyAsync(c.hc_gate, b.hc_gate + static_cast<size_t>(last) * HC_WIDTH,
                static_cast<size_t>(HC_WIDTH) * sizeof(float), hipMemcpyDeviceToDevice, S.stream) != hipSuccess ||
            hipMemcpyAsync(c.hc_inject, b.hc_inject + static_cast<size_t>(last) * N_HC,
                static_cast<size_t>(N_HC) * sizeof(float), hipMemcpyDeviceToDevice, S.stream) != hipSuccess ||
            hipMemcpyAsync(c.mixed, b.mixed + static_cast<size_t>(last) * N_EMBD,
                static_cast<size_t>(N_EMBD) * sizeof(float), hipMemcpyDeviceToDevice, S.stream) != hipSuccess ||
            hipMemcpyAsync(c.key, b.key + static_cast<size_t>(last) * KV_WIDTH,
                static_cast<size_t>(KV_WIDTH) * sizeof(float), hipMemcpyDeviceToDevice, S.stream) != hipSuccess ||
            hipMemcpyAsync(c.value, b.value + static_cast<size_t>(last) * KV_WIDTH,
                static_cast<size_t>(KV_WIDTH) * sizeof(float), hipMemcpyDeviceToDevice, S.stream) != hipSuccess ||
            hipMemcpyAsync(c.index_key, b.index_key + static_cast<size_t>(last) * IDX_WIDTH,
                static_cast<size_t>(IDX_WIDTH) * sizeof(float), hipMemcpyDeviceToDevice, S.stream) != hipSuccess) {
        return -1;
    }
    quantize_q8_1_sum_hi<<<N_EMBD / 32, 32, 0, S.stream>>>(
            c.mixed, c.mixed_q8, c.mixed_hi, N_EMBD);
    if (hipGetLastError() != hipSuccess) return -1;
    b.rows = rows;
    return 0;
}
#endif

static int catchup_phase1_impl(int32_t seq_id, const int32_t * tokens,
        const int32_t * positions, const float * hidden_rows, int count,
        bool hidden_device, bool already_normalized) {
    SeqState * seq = get_seq(seq_id);
    if (!S.ok || seq == nullptr || count < 0 ||
            (count > 0 && (tokens == nullptr || positions == nullptr || hidden_rows == nullptr))) return -1;
    if (count == 0) return 0;
#if !defined(QWEN4EXP_SIDECAR_BATCH_DIAGNOSTIC) && \
        !defined(QWEN4EXP_SIDECAR_NATIVE_MMQ_TEST)
    // Multirow prompt catch-up requires its separate native MMQ/F16 contract.
    if (count > 1) return -1;
#elif defined(QWEN4EXP_SIDECAR_NATIVE_MMQ_TEST) && \
        !defined(QWEN4EXP_SIDECAR_BATCH_DIAGNOSTIC)
    // Current-tree dense widths 2–8 dispatch MMVQ rather than this J=16 MMQ.
    // Reject them until that separate batch contract is reproduced exactly.
    if (count > 1 && (count < 9 || count > PromptBatchScratch::MAX_ROWS)) return -1;
#endif
    if (count > S.max_pos - seq->state_pos_max ||
            seq->pending_pos_max != seq->state_pos_max ||
            ensure_seq_storage(*seq) != 0 || ensure_pending_storage(*seq, count) != 0 ||
            ensure_catchup_scratch() != 0) return -1;
    for (int row = 0; row < count; ++row) {
        if (tokens[row] < 0 || tokens[row] >= N_VOCAB ||
                positions[row] != seq->state_pos_max + row) return -1;
    }
    seq->pending_pos_min = seq->state_pos_max;
    seq->pending_pos_max = seq->state_pos_max + count;
#ifdef QWEN4EXP_SIDECAR_NATIVE_MMQ_TEST
    if (count > 1) {
        if (catchup_phase1_native_batch(*seq, tokens, positions, hidden_rows, count,
                    hidden_device, already_normalized) != 0 ||
                (!hidden_device && hipStreamSynchronize(S.stream) != hipSuccess)) {
            discard_pending(*seq);
            return -1;
        }
        return 0;
    }
#endif
    for (int row = 0; row < count; ++row) {
        const float * source = hidden_rows + static_cast<size_t>(row) * HC_WIDTH;
        if (load_normalized_hidden_row(source, hidden_device, already_normalized) != 0 ||
                phase1_from_normalized_hidden(*seq, tokens[row], positions[row], row) != 0) {
            discard_pending(*seq);
            return -1;
        }
    }
    if (hidden_device) {
        if (hipGetLastError() != hipSuccess) {
            discard_pending(*seq);
            return -1;
        }
    } else if (hipStreamSynchronize(S.stream) != hipSuccess) {
        discard_pending(*seq);
        return -1;
    }
    return 0;
}

#ifdef QWEN4EXP_SIDECAR_PRODUCTION
// Runtime target batches expose h_p beside token x_p. Qwen4Exp MTP consumes
// x_p with h_(p-1), so preserve the raw target rows for commit and shift the
// hidden source by one while constructing the draft K/V/index cache.
static int catchup_runtime_impl(int32_t seq_id, const int32_t * tokens,
        const int32_t * positions, const float * hidden_rows, int count,
        bool hidden_device) {
    const bool debug = std::getenv("LLAMA_SPEC_HIP_DEBUG") != nullptr;
    SeqState * seq = get_seq(seq_id);
    int current_device = -2;
    if (debug) {
        (void) hipGetDevice(&current_device);
        std::fprintf(stderr,
                "[qwen4exp-sidecar] catch-up enter seq=%d count=%d path=%s provider_device=%d current_device=%d state=[%d,%d) pending=[%d,%d) hidden_valid=%d input_pos=[%d,%d]\n",
                seq_id, count, hidden_device ? "D2D" : "HOST", S.device, current_device,
                seq != nullptr ? seq->state_pos_min : -1,
                seq != nullptr ? seq->state_pos_max : -1,
                seq != nullptr ? seq->pending_pos_min : -1,
                seq != nullptr ? seq->pending_pos_max : -1,
                seq != nullptr ? static_cast<int>(seq->hidden_valid) : -1,
                count > 0 && positions != nullptr ? positions[0] : -1,
                count > 0 && positions != nullptr ? positions[count - 1] : -1);
    }
    if (!S.ok || seq == nullptr || count < 0 ||
            (count > 0 && (tokens == nullptr || positions == nullptr || hidden_rows == nullptr))) {
        std::fprintf(stderr, "[qwen4exp-sidecar] catch-up rejected: invalid provider/sequence/input\n");
        return -1;
    }
    if (count == 0) return 0;
    if (seq->state_pos_max > 0 && !seq->hidden_valid) {
        std::fprintf(stderr,
                "[qwen4exp-sidecar] catch-up rejected: committed hidden tip missing at state_pos_max=%d\n",
                seq->state_pos_max);
        return -1;
    }
    if (count > S.max_pos - seq->state_pos_max) {
        std::fprintf(stderr,
                "[qwen4exp-sidecar] catch-up rejected: capacity count=%d state_pos_max=%d max_pos=%d\n",
                count, seq->state_pos_max, S.max_pos);
        return -1;
    }
    if (seq->pending_pos_max != seq->state_pos_max) {
        std::fprintf(stderr,
                "[qwen4exp-sidecar] catch-up rejected: uncommitted pending state=[%d,%d) pending=[%d,%d)\n",
                seq->state_pos_min, seq->state_pos_max,
                seq->pending_pos_min, seq->pending_pos_max);
        return -1;
    }
    if (ensure_seq_storage(*seq) != 0) {
        std::fprintf(stderr, "[qwen4exp-sidecar] catch-up rejected: sequence allocation failed\n");
        return -1;
    }
    if (ensure_pending_storage(*seq, count) != 0) {
        std::fprintf(stderr, "[qwen4exp-sidecar] catch-up rejected: pending allocation failed for count=%d\n", count);
        return -1;
    }
    if (ensure_catchup_scratch() != 0) {
        std::fprintf(stderr, "[qwen4exp-sidecar] catch-up rejected: scratch allocation failed\n");
        return -1;
    }
    for (int row = 0; row < count; ++row) {
        if (tokens[row] < 0 || tokens[row] >= N_VOCAB) {
            std::fprintf(stderr,
                    "[qwen4exp-sidecar] catch-up rejected: invalid token row=%d token=%d\n",
                    row, tokens[row]);
            return -1;
        }
        const int32_t expected = seq->state_pos_max + row;
        if (positions[row] != expected) {
            std::fprintf(stderr,
                    "[qwen4exp-sidecar] catch-up rejected: position mismatch row=%d actual=%d expected=%d state_pos_max=%d count=%d\n",
                    row, positions[row], expected, seq->state_pos_max, count);
            return -1;
        }
    }

    const size_t hidden_bytes = static_cast<size_t>(count) * HC_WIDTH * sizeof(float);
    const hipMemcpyKind hidden_kind = hidden_device ? hipMemcpyDeviceToDevice : hipMemcpyHostToDevice;
    hipError_t status = hipMemcpyAsync(seq->hidden_pending, hidden_rows, hidden_bytes,
            hidden_kind, S.stream);
    if (status != hipSuccess) {
        std::fprintf(stderr,
                "[qwen4exp-sidecar] catch-up hidden copy failed: %s (bytes=%zu path=%s)\n",
                hipGetErrorString(status), hidden_bytes, hidden_device ? "D2D" : "HOST");
        return -1;
    }

    seq->pending_pos_min = seq->state_pos_max;
    seq->pending_pos_max = seq->state_pos_max + count;
    for (int row = 0; row < count; ++row) {
        if (row == 0 && seq->state_pos_max == 0) {
            status = hipMemsetAsync(S.catchup.hidden_norm, 0,
                    static_cast<size_t>(HC_WIDTH) * sizeof(float), S.stream);
            if (status != hipSuccess) {
                std::fprintf(stderr,
                        "[qwen4exp-sidecar] catch-up BOS hidden clear failed: %s\n",
                        hipGetErrorString(status));
                discard_pending(*seq);
                return -1;
            }
        } else {
            const float * previous_hidden = row == 0 ? seq->hidden_committed :
                    seq->hidden_pending + static_cast<size_t>(row - 1) * HC_WIDTH;
            if (load_normalized_hidden_row(previous_hidden, true, false) != 0) {
                std::fprintf(stderr,
                        "[qwen4exp-sidecar] catch-up hidden normalization failed at row=%d position=%d\n",
                        row, positions[row]);
                discard_pending(*seq);
                return -1;
            }
        }
        if (phase1_from_normalized_hidden(*seq, tokens[row], positions[row], row) != 0) {
            std::fprintf(stderr,
                    "[qwen4exp-sidecar] catch-up phase 1 rejected at row=%d position=%d token=%d\n",
                    row, positions[row], tokens[row]);
            discard_pending(*seq);
            return -1;
        }
    }
    status = hipGetLastError();
    if (status != hipSuccess) {
        std::fprintf(stderr,
                "[qwen4exp-sidecar] catch-up final launch check failed: %s\n",
                hipGetErrorString(status));
        discard_pending(*seq);
        return -1;
    }
    if (!hidden_device || debug) {
        status = hipStreamSynchronize(S.stream);
        if (status != hipSuccess) {
            std::fprintf(stderr,
                    "[qwen4exp-sidecar] catch-up final stream sync failed: %s (path=%s count=%d positions=[%d,%d])\n",
                    hipGetErrorString(status), hidden_device ? "D2D" : "HOST", count,
                    positions[0], positions[count - 1]);
            discard_pending(*seq);
            return -1;
        }
    }
    if (debug) {
        std::fprintf(stderr,
                "[qwen4exp-sidecar] catch-up staged seq=%d count=%d positions=[%d,%d]\n",
                seq_id, count, positions[0], positions[count - 1]);
    }
    return 0;
}

static int build_qsa_mask_runtime(SeqState & seq, int32_t seq_id, int32_t position) {
    QsaScratch & q = S.qsa;
    CatchupScratch & c = S.catchup;
    if (position < 32 || position >= S.max_pos || seq.state_pos_min != 0 ||
            ensure_native_attention_scratch() != 0) return -1;
    const int valid_tokens = position + 1;
    const int complete_blocks = valid_tokens / QSA_RATIO;
    if (complete_blocks < 1 || complete_blocks > QSA_MAX_BLOCKS) return -1;

    gemv_f16_f32_batched<256><<<INDEX_HEADS * INDEX_DIM, 256, 0, S.stream>>>(
            S.weights.index_query, c.mixed, q.query, N_EMBD, INDEX_HEADS * INDEX_DIM);
    rms_norm_shared_rows<<<INDEX_HEADS, 256, 0, S.stream>>>(
            q.query, S.weights.index_query_norm, q.query, INDEX_DIM);
    index_rope_neox<<<INDEX_HEADS, N_ROT / 2, 0, S.stream>>>(
            q.query, INDEX_HEADS, position, 0);
    if (hipMemsetAsync(q.selected, 0, SC_MAX_POS, S.stream) != hipSuccess) return -1;
    qsa_pool_index_f16<<<complete_blocks, INDEX_DIM, 0, S.stream>>>(
            seq.index_committed, q.pooled, complete_blocks);
    rms_norm_shared_rows<<<complete_blocks, 256, 0, S.stream>>>(
            q.pooled, S.weights.index_key_norm, q.normalized, INDEX_DIM);
    index_rope_neox<<<complete_blocks, N_ROT / 2, 0, S.stream>>>(
            q.normalized, complete_blocks, 0, QSA_RATIO);
    index_head_dots_native<<<complete_blocks, 64, 0, S.stream>>>(
            q.normalized, q.query, q.head_dots, complete_blocks);
    index_head_relu_transpose_native<<<
            (complete_blocks * INDEX_HEADS + 255) / 256, 256, 0, S.stream>>>(
            q.head_dots, q.block_heads, complete_blocks);
    index_sum_heads_native<<<complete_blocks, 32, 0, S.stream>>>(
            q.block_heads, q.scores, complete_blocks);

    int partitions = (complete_blocks + QSA_SORT_CAPACITY - 1) / QSA_SORT_CAPACITY;
    int candidate_count = partitions * QSA_MAX_RANKED_BLOCKS;
    qsa_select_candidates_scores<<<partitions, QSA_SORT_THREADS, 0, S.stream>>>(
            q.scores, complete_blocks, q.candidate_keys_a, q.candidate_blocks_a);
    const uint64_t * final_keys = q.candidate_keys_a;
    const int32_t * final_blocks = q.candidate_blocks_a;
    if (candidate_count > QSA_SORT_CAPACITY) {
        partitions = (candidate_count + QSA_SORT_CAPACITY - 1) / QSA_SORT_CAPACITY;
        qsa_select_candidates_pairs<<<partitions, QSA_SORT_THREADS, 0, S.stream>>>(
                q.candidate_keys_a, q.candidate_blocks_a, candidate_count,
                q.candidate_keys_b, q.candidate_blocks_b);
        candidate_count = partitions * QSA_MAX_RANKED_BLOCKS;
        final_keys = q.candidate_keys_b;
        final_blocks = q.candidate_blocks_b;
    }
    if (candidate_count > QSA_SORT_CAPACITY) {
        partitions = (candidate_count + QSA_SORT_CAPACITY - 1) / QSA_SORT_CAPACITY;
        qsa_select_candidates_pairs<<<partitions, QSA_SORT_THREADS, 0, S.stream>>>(
                q.candidate_keys_b, q.candidate_blocks_b, candidate_count,
                q.candidate_keys_a, q.candidate_blocks_a);
        candidate_count = partitions * QSA_MAX_RANKED_BLOCKS;
        final_keys = q.candidate_keys_a;
        final_blocks = q.candidate_blocks_a;
    }
    if (candidate_count > QSA_SORT_CAPACITY) return -1;
    qsa_select_mask_final<<<1, QSA_SORT_THREADS, 0, S.stream>>>(
            final_keys, final_blocks, candidate_count, complete_blocks,
            position, q.selected, q.selected_blocks);
    if (hipGetLastError() != hipSuccess) return -1;

    const int target_tokens = std::min(valid_tokens, INDEX_TOP_K + QSA_RATIO - 1);
    const int ranked_tokens = target_tokens - valid_tokens % QSA_RATIO;
    q.last_complete_blocks = complete_blocks;
    q.last_position = position;
    q.last_ranked_blocks = ranked_tokens / QSA_RATIO + (ranked_tokens % QSA_RATIO != 0);
    q.last_qsa_seq = seq_id;
    q.last_attention_seq = -1;
    return 0;
}

static int draft_step_runtime(SeqState & seq, int32_t seq_id,
        int32_t position, float temperature) {
    CatchupScratch & c = S.catchup;
    if (prepare_draft_query(position) != 0) return -1;
    if (position < 32) {
        attention_dense_f16_32<<<N_HEAD, 32, 0, S.stream>>>(
                c.query, seq.key_committed, seq.value_committed,
                c.attention, position + 1);
    } else {
        QsaScratch & q = S.qsa;
        const int physical = ((position + 1 + 255) / 256) * 256;
        if (physical > S.max_pos || build_qsa_mask_runtime(seq, seq_id, position) != 0) return -1;
        qsa_mask_to_f16<<<(physical + 255) / 256, 256, 0, S.stream>>>(
                q.selected, q.attention_mask, physical);
        int parallel = 0;
        if (hipGetLastError() != hipSuccess ||
                launch_native_sparse_attention(S.stream, c.query,
                    seq.key_committed, seq.value_committed, q.attention_mask,
                    physical, QSA_MAX_ATTENTION_PARTS, q.attention_parts,
                    q.attention_meta, c.attention, &parallel) != 0) return -1;
        q.last_attention_seq = seq_id;
        q.last_attention_physical = physical;
        q.last_attention_parallel = parallel;
    }
    return finish_draft_from_attention(temperature);
}

static int draft_runtime_impl(int32_t seq_id, int32_t token, int32_t position,
        const float * hidden, bool hidden_device, float temperature, float p_min,
        uint64_t rng_key, bool stochastic, int count, int32_t * output_tokens,
        int32_t * output_dist_ids, float * output_dist_probs) {
    SeqState * seq = get_seq(seq_id);
    CatchupScratch & c = S.catchup;
    if (!S.ok || seq == nullptr || token < 0 || token >= N_VOCAB ||
            count < 1 || count > 8 || output_tokens == nullptr ||
            (!hidden_device && hidden == nullptr) || position != seq->state_pos_max ||
            position < 0 || position + count > S.max_pos || !seq->allocated ||
            (hidden_device && !seq->hidden_valid) ||
            (stochastic && (output_dist_ids == nullptr || output_dist_probs == nullptr ||
                !std::isfinite(temperature) || temperature <= 0.0f || temperature > 100.0f ||
                !std::isfinite(p_min) || p_min < 0.0f)) ||
            seq->pending_pos_max != seq->state_pos_max ||
            ensure_pending_storage(*seq, count) != 0 || ensure_catchup_scratch() != 0) return -1;

    if (stochastic) {
        const int32_t zero = 0;
        const int32_t one = 1;
        if (hipMemcpyAsync(c.chain_count, &zero, sizeof(zero),
                    hipMemcpyHostToDevice, S.stream) != hipSuccess ||
                hipMemcpyAsync(c.chain_active, &one, sizeof(one),
                    hipMemcpyHostToDevice, S.stream) != hipSuccess) return -1;
    }

    const float * initial_hidden = hidden_device ? seq->hidden_committed : hidden;
    seq->pending_pos_min = position;
    seq->pending_pos_max = position + count;
    if (load_normalized_hidden_row(initial_hidden, hidden_device, false) != 0 ||
            phase1_from_normalized_hidden(*seq, token, position, 0) != 0) {
        discard_pending(*seq);
        return -1;
    }

    for (int step = 0; step < count; ++step) {
        const int32_t current_position = position + step;
        copy_cache_rows_f16<<<(KV_WIDTH + 255) / 256, 256, 0, S.stream>>>(
                seq->key_committed, seq->key_pending, current_position, step, 1, KV_WIDTH);
        copy_cache_rows_f16<<<(KV_WIDTH + 255) / 256, 256, 0, S.stream>>>(
                seq->value_committed, seq->value_pending, current_position, step, 1, KV_WIDTH);
        copy_cache_rows_f16<<<(IDX_WIDTH + 255) / 256, 256, 0, S.stream>>>(
                seq->index_committed, seq->index_pending, current_position, step, 1, IDX_WIDTH);
        if (draft_step_runtime(*seq, seq_id, current_position,
                    stochastic ? temperature : 1.0f) != 0) {
            discard_pending(*seq);
            return -1;
        }
        if (stochastic) {
            proposal_sample_top32<<<1, 1, 0, S.stream>>>(
                    c.topk_indices, c.topk_probabilities, p_min,
                    spec_sidecar_stochastic_uniform(rng_key, static_cast<uint32_t>(step)),
                    step, c.chain_tokens, c.chain_dist_ids, c.chain_dist_probs,
                    c.chain_count, c.chain_active);
        } else if (hipMemcpyAsync(c.chain_tokens + step, c.topk_indices,
                    sizeof(int32_t), hipMemcpyDeviceToDevice, S.stream) != hipSuccess) {
            discard_pending(*seq);
            return -1;
        }
        if (step + 1 < count) {
            rms_norm_rows<<<1, 1024, 0, S.stream>>>(
                    c.eh, S.weights.hidden_norm, c.hidden_norm, HC_WIDTH);
            if (phase1_from_normalized_hidden_device_token(*seq, c.chain_tokens + step,
                        current_position + 1, step + 1) != 0) {
                discard_pending(*seq);
                return -1;
            }
        }
    }

    int32_t emitted = count;
    if (stochastic && (hipMemcpyAsync(&emitted, c.chain_count, sizeof(emitted),
                    hipMemcpyDeviceToHost, S.stream) != hipSuccess ||
            hipMemcpyAsync(output_dist_ids, c.chain_dist_ids,
                    static_cast<size_t>(count) * PROPOSAL_TOP_K * sizeof(int32_t),
                    hipMemcpyDeviceToHost, S.stream) != hipSuccess ||
            hipMemcpyAsync(output_dist_probs, c.chain_dist_probs,
                    static_cast<size_t>(count) * PROPOSAL_TOP_K * sizeof(float),
                    hipMemcpyDeviceToHost, S.stream) != hipSuccess)) {
        discard_pending(*seq);
        return -1;
    }
    if (hipMemcpyAsync(output_tokens, c.chain_tokens,
                static_cast<size_t>(count) * sizeof(int32_t),
                hipMemcpyDeviceToHost, S.stream) != hipSuccess ||
            hipStreamSynchronize(S.stream) != hipSuccess || emitted < 0 || emitted > count) {
        discard_pending(*seq);
        return -1;
    }
    for (int step = 0; step < emitted; ++step) {
        if (output_tokens[step] < 0 || output_tokens[step] >= N_VOCAB) {
            discard_pending(*seq);
            return -1;
        }
    }
    discard_pending(*seq);
    return emitted;
}
#endif

static bool state_matches(int32_t seq_id, const void * data, int size,
        spec_sidecar_state & snapshot) {
    SeqState * seq = get_seq(seq_id);
    if (!S.ok || seq == nullptr || data == nullptr || size != static_cast<int>(sizeof(snapshot))) {
        return false;
    }
    std::memcpy(&snapshot, data, sizeof(snapshot));
    return snapshot.magic == SPEC_SIDECAR_STATE_MAGIC &&
            snapshot.version == SPEC_SIDECAR_STATE_VERSION &&
            snapshot.kind == SPEC_SIDECAR_STATE_KIND_MTP &&
            snapshot.epoch == seq->state_epoch &&
            snapshot.pos_min == seq->state_pos_min &&
            snapshot.pos_min >= 0 && snapshot.pos_min <= snapshot.pos_max &&
            snapshot.pos_max <= seq->state_pos_max && snapshot.pos_max <= S.max_pos;
}

#ifdef QWEN4EXP_SIDECAR_LIFECYCLE_TEST
__global__ void fill_test_rows(__half * cache, int rows, int width,
        int logical_pos_min, float kind_offset) {
    const int row = static_cast<int>(blockIdx.y);
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row < rows && i < width) {
        const float value = kind_offset + static_cast<float>(logical_pos_min + row) +
                0.125f * static_cast<float>(i & 3);
        cache[static_cast<size_t>(row) * width + i] = __float2half(value);
    }
}

static int stage_test_rows(SeqState & seq, int32_t position, int count) {
    if (count < 1 || position != seq.state_pos_max ||
            count > S.max_pos - seq.state_pos_max ||
            seq.pending_pos_max != seq.state_pos_max ||
            ensure_seq_storage(seq) != 0 || ensure_pending_storage(seq, count) != 0) {
        return -1;
    }
    seq.pending_pos_min = position;
    seq.pending_pos_max = position + count;
    fill_test_rows<<<dim3((KV_WIDTH + 255) / 256, count), 256, 0, S.stream>>>(
            seq.key_pending, count, KV_WIDTH, position, 0.0f);
    fill_test_rows<<<dim3((KV_WIDTH + 255) / 256, count), 256, 0, S.stream>>>(
            seq.value_pending, count, KV_WIDTH, position, 32.0f);
    fill_test_rows<<<dim3((IDX_WIDTH + 255) / 256, count), 256, 0, S.stream>>>(
            seq.index_pending, count, IDX_WIDTH, position, 64.0f);
    SC_CHECK(hipGetLastError());
    return 0;
}
#endif

static int free_seq(SeqState & seq) {
    if (seq.key_committed != nullptr) SC_CHECK(hipFree(seq.key_committed));
    if (seq.value_committed != nullptr) SC_CHECK(hipFree(seq.value_committed));
    if (seq.index_committed != nullptr) SC_CHECK(hipFree(seq.index_committed));
    if (seq.key_pending != nullptr) SC_CHECK(hipFree(seq.key_pending));
    if (seq.value_pending != nullptr) SC_CHECK(hipFree(seq.value_pending));
    if (seq.index_pending != nullptr) SC_CHECK(hipFree(seq.index_pending));
    if (seq.hidden_committed != nullptr) SC_CHECK(hipFree(seq.hidden_committed));
    if (seq.hidden_pending != nullptr) SC_CHECK(hipFree(seq.hidden_pending));
    seq = SeqState{};
    return 0;
}

} // namespace

#ifdef _WIN32
#define SC_API extern "C" __declspec(dllexport)
#else
#define SC_API extern "C"
#endif

SC_API int spec_hip_release_abi() {
    return SC_RELEASE_ABI;
}

SC_API int spec_hip_check(int32_t embedding_width, int32_t head_rows, int32_t n_seq) {
    return embedding_width == HC_WIDTH && head_rows == N_VOCAB &&
            n_seq >= 1 && n_seq <= SC_MAX_SEQ ? 0 : -1;
}

SC_API int spec_hip_state_size() {
    return static_cast<int>(sizeof(spec_sidecar_state));
}

SC_API int spec_hip_get_state(int32_t seq_id, void * data, int size) {
    SeqState * seq = get_seq(seq_id);
    if (!S.ok || seq == nullptr || data == nullptr || size != static_cast<int>(sizeof(spec_sidecar_state))) {
        return -1;
    }
    spec_sidecar_state snapshot = {};
    snapshot.magic = SPEC_SIDECAR_STATE_MAGIC;
    snapshot.version = SPEC_SIDECAR_STATE_VERSION;
    snapshot.kind = SPEC_SIDECAR_STATE_KIND_MTP;
    snapshot.pos_min = seq->state_pos_min;
    snapshot.pos_max = seq->state_pos_max;
    snapshot.epoch = seq->state_epoch;
    std::memcpy(data, &snapshot, sizeof(snapshot));
    return 0;
}

SC_API int spec_hip_set_state(int32_t seq_id, const void * data, int size) {
    SeqState * seq = get_seq(seq_id);
    spec_sidecar_state snapshot = {};
    if (seq == nullptr || !state_matches(seq_id, data, size, snapshot)) return -1;
    seq->hidden_valid = seq->hidden_valid && spec_sidecar_mtp::committed_hidden_matches_tip(
            seq->state_pos_max, snapshot.pos_max);
    seq->state_pos_max = snapshot.pos_max;
    discard_pending(*seq);
    invalidate_qsa_for_seq(seq_id);
    return 0;
}

SC_API int spec_hip_reset_state(int32_t seq_id) {
    SeqState * seq = get_seq(seq_id);
    if (!S.ok || seq == nullptr) return -1;
    if (++seq->state_epoch == 0) seq->state_epoch = 1;
    seq->state_pos_min = 0;
    seq->state_pos_max = 0;
    seq->hidden_valid = false;
    discard_pending(*seq);
    invalidate_qsa_for_seq(seq_id);
    return 0;
}

SC_API int spec_hip_truncate_state(int32_t seq_id, int32_t pos_max) {
    SeqState * seq = get_seq(seq_id);
    if (!S.ok || seq == nullptr || pos_max < seq->state_pos_min || pos_max > seq->state_pos_max) {
        return -1;
    }
    seq->hidden_valid = seq->hidden_valid && spec_sidecar_mtp::committed_hidden_matches_tip(
            seq->state_pos_max, pos_max);
    seq->state_pos_max = pos_max;
    discard_pending(*seq);
    invalidate_qsa_for_seq(seq_id);
    return 0;
}

SC_API int spec_hip_commit_state(int32_t seq_id, int32_t pos_max,
        const float * hidden_device) {
    SeqState * seq = get_seq(seq_id);
    if (!S.ok || seq == nullptr || pos_max < seq->state_pos_max ||
            pos_max > seq->pending_pos_max || pos_max > S.max_pos) {
        return -1;
    }
    if (pos_max > seq->state_pos_max) {
        if (!seq->allocated || seq->key_pending == nullptr || seq->value_pending == nullptr ||
                seq->index_pending == nullptr || seq->pending_pos_min != seq->state_pos_max) {
            return -1;
        }
        const size_t source_row = static_cast<size_t>(seq->state_pos_max - seq->pending_pos_min);
        const size_t rows = static_cast<size_t>(pos_max - seq->state_pos_max);
        SC_CHECK(hipMemcpyAsync(seq->key_committed + static_cast<size_t>(seq->state_pos_max) * KV_WIDTH,
                seq->key_pending + source_row * KV_WIDTH,
                rows * KV_WIDTH * sizeof(__half), hipMemcpyDeviceToDevice, S.stream));
        SC_CHECK(hipMemcpyAsync(seq->value_committed + static_cast<size_t>(seq->state_pos_max) * KV_WIDTH,
                seq->value_pending + source_row * KV_WIDTH,
                rows * KV_WIDTH * sizeof(__half), hipMemcpyDeviceToDevice, S.stream));
        SC_CHECK(hipMemcpyAsync(seq->index_committed + static_cast<size_t>(seq->state_pos_max) * IDX_WIDTH,
                seq->index_pending + source_row * IDX_WIDTH,
                rows * IDX_WIDTH * sizeof(__half), hipMemcpyDeviceToDevice, S.stream));
        const float * hidden_source = hidden_device;
#ifdef QWEN4EXP_SIDECAR_PRODUCTION
        if (hidden_source == nullptr && seq->hidden_pending != nullptr) {
            hidden_source = seq->hidden_pending + (source_row + rows - 1) * HC_WIDTH;
        }
#endif
        seq->hidden_valid = hidden_source != nullptr;
        if (hidden_source != nullptr) {
            SC_CHECK(hipMemcpyAsync(seq->hidden_committed, hidden_source,
                    static_cast<size_t>(HC_WIDTH) * sizeof(float),
                    hipMemcpyDeviceToDevice, S.stream));
        }
        seq->state_pos_max = pos_max;
        invalidate_qsa_for_seq(seq_id);
    }
    discard_pending(*seq);
    return 0;
}

SC_API int spec_hip_rebase_state(int32_t seq_id, int32_t pos_min,
        int32_t pos_max, int32_t delta) {
    SeqState * seq = get_seq(seq_id);
    if (!S.ok || seq == nullptr || pos_min < 0 || pos_min > pos_max ||
            pos_max > S.max_pos || seq->pending_pos_max != seq->state_pos_max) {
        return -1;
    }
    if (delta == 0 || seq->state_pos_max <= pos_min) {
        if (delta != 0) {
            if (++seq->state_epoch == 0) seq->state_epoch = 1;
            invalidate_qsa_for_seq(seq_id);
        }
        return 0;
    }
    if (pos_max < seq->state_pos_max || !seq->allocated) return -1;
    const int64_t new_min = static_cast<int64_t>(pos_min) + delta;
    const int64_t new_max = static_cast<int64_t>(seq->state_pos_max) + delta;
    if (new_min < 0 || new_max < new_min || new_max > S.max_pos) return -1;
    SC_CHECK(hipStreamSynchronize(S.stream));
    shift_cache_rows_f16<<<1, 256, 0, S.stream>>>(
            seq->key_committed, pos_min, seq->state_pos_max, delta, KV_WIDTH);
    shift_cache_rows_f16<<<1, 256, 0, S.stream>>>(
            seq->value_committed, pos_min, seq->state_pos_max, delta, KV_WIDTH);
    shift_cache_rows_f16<<<1, 256, 0, S.stream>>>(
            seq->index_committed, pos_min, seq->state_pos_max, delta, IDX_WIDTH);
    SC_CHECK(hipGetLastError());
    SC_CHECK(hipStreamSynchronize(S.stream));
    if (seq->state_pos_min >= pos_min) seq->state_pos_min += delta;
    seq->state_pos_max = static_cast<int32_t>(new_max);
    discard_pending(*seq);
    if (++seq->state_epoch == 0) seq->state_epoch = 1;
    invalidate_qsa_for_seq(seq_id);
    return 0;
}

SC_API int spec_hip_attach_target_stream(void * stream, int32_t device) {
    if (!S.ok || device != S.device || hipSetDevice(S.device) != hipSuccess) return -1;
    const hipStream_t next = stream == nullptr ? S.owned_stream : reinterpret_cast<hipStream_t>(stream);
    if (next != S.stream) {
        SC_CHECK(hipStreamSynchronize(S.stream));
        S.stream = next;
    }
    return 0;
}

#ifdef QWEN4EXP_SIDECAR_PRODUCTION
SC_API int spec_hip_init_device_context(const char * weights_dir, const char * ids_path,
        int32_t n_seq, int32_t device, int32_t max_context) {
    return initialize_artifacts(weights_dir, ids_path, n_seq, device, max_context);
}

SC_API int spec_hip_init_context(const char * weights_dir, const char * ids_path,
        int32_t n_seq, int32_t max_context) {
    return initialize_artifacts(weights_dir, ids_path, n_seq, -1, max_context);
}

SC_API int spec_hip_init_device(const char * weights_dir, const char * ids_path,
        int32_t n_seq, int32_t device) {
    return initialize_artifacts(weights_dir, ids_path, n_seq, device, SC_MAX_POS);
}
#endif

SC_API int spec_hip_init(const char * weights_dir, const char * ids_path, int32_t n_seq) {
#ifdef QWEN4EXP_SIDECAR_LIFECYCLE_TEST
    (void) weights_dir;
    (void) ids_path;
    if (S.ok) return n_seq == S.n_seq ? 0 : -1;
    if (n_seq < 1 || n_seq > SC_MAX_SEQ) return -1;
    S.n_seq = n_seq;
    S.max_pos = 32;
    SC_CHECK(hipGetDevice(&S.device));
    SC_CHECK(hipStreamCreate(&S.owned_stream));
    S.stream = S.owned_stream;
    S.ok = true;
    return 0;
#elif defined(QWEN4EXP_SIDECAR_ARTIFACT_TEST) || defined(QWEN4EXP_SIDECAR_PRODUCTION)
    return initialize_artifacts(weights_dir, ids_path, n_seq, -1, SC_MAX_POS);
#else
    (void) weights_dir;
    (void) ids_path;
    (void) n_seq;
    std::fprintf(stderr, "[qwen4exp-sidecar] provider graph is not complete\n");
    return -1;
#endif
}

SC_API int spec_hip_catchup(int32_t seq_id, const int32_t * tokens,
        const int32_t * positions, const float * hidden_rows, int count) {
#ifdef QWEN4EXP_SIDECAR_LIFECYCLE_TEST
    SeqState * seq = get_seq(seq_id);
    if (!S.ok || seq == nullptr || count < 0 ||
            (count > 0 && (tokens == nullptr || positions == nullptr || hidden_rows == nullptr))) return -1;
    if (count == 0) return 0;
    for (int row = 0; row < count; ++row) {
        if (tokens[row] < 0 || tokens[row] >= N_VOCAB ||
                positions[row] != seq->state_pos_max + row) return -1;
    }
    return stage_test_rows(*seq, positions[0], count);
#elif defined(QWEN4EXP_SIDECAR_PRODUCTION)
    return catchup_runtime_impl(seq_id, tokens, positions, hidden_rows, count, false);
#elif defined(QWEN4EXP_SIDECAR_ARTIFACT_TEST)
    return catchup_phase1_impl(seq_id, tokens, positions, hidden_rows, count, false, false);
#else
    (void) seq_id; (void) tokens; (void) positions; (void) hidden_rows; (void) count;
    return -1;
#endif
}

SC_API int spec_hip_catchup_device(int32_t seq_id, const int32_t * tokens,
        const int32_t * positions, const float * hidden_rows_device, int count) {
#ifdef QWEN4EXP_SIDECAR_LIFECYCLE_TEST
    return spec_hip_catchup(seq_id, tokens, positions, hidden_rows_device, count);
#elif defined(QWEN4EXP_SIDECAR_PRODUCTION)
    return catchup_runtime_impl(seq_id, tokens, positions, hidden_rows_device, count, true);
#elif defined(QWEN4EXP_SIDECAR_ARTIFACT_TEST)
    return catchup_phase1_impl(seq_id, tokens, positions, hidden_rows_device, count, true, false);
#else
    (void) seq_id; (void) tokens; (void) positions; (void) hidden_rows_device; (void) count;
    return -1;
#endif
}

SC_API int spec_hip_draft(int32_t seq_id, int32_t token, int32_t position,
        const float * hidden, int count, int32_t * output_tokens) {
#ifdef QWEN4EXP_SIDECAR_PRODUCTION
    return draft_runtime_impl(seq_id, token, position, hidden, false,
            1.0f, 0.0f, 0, false, count, output_tokens, nullptr, nullptr);
#else
    (void) seq_id; (void) token; (void) position; (void) hidden; (void) count; (void) output_tokens;
    return -1;
#endif
}
SC_API int spec_hip_draft_device(int32_t seq_id, int32_t token, int32_t position,
        int count, int32_t * output_tokens) {
#ifdef QWEN4EXP_SIDECAR_PRODUCTION
    return draft_runtime_impl(seq_id, token, position, nullptr, true,
            1.0f, 0.0f, 0, false, count, output_tokens, nullptr, nullptr);
#else
    (void) seq_id; (void) token; (void) position; (void) count; (void) output_tokens;
    return -1;
#endif
}
SC_API int spec_hip_stochastic_top_k() {
    return SPEC_SIDECAR_MTP_DRAFT_TOP_K;
}
SC_API int spec_hip_draft_stochastic(int32_t seq_id, int32_t token, int32_t position,
        const float * hidden, float temperature, float p_min, uint64_t rng_key,
        int count, int32_t * output_tokens, int32_t * output_dist_ids,
        float * output_dist_probs) {
#ifdef QWEN4EXP_SIDECAR_PRODUCTION
    return draft_runtime_impl(seq_id, token, position, hidden, false,
            temperature, p_min, rng_key, true, count, output_tokens,
            output_dist_ids, output_dist_probs);
#else
    (void) seq_id; (void) token; (void) position; (void) hidden;
    (void) temperature; (void) p_min; (void) rng_key; (void) count;
    (void) output_tokens; (void) output_dist_ids; (void) output_dist_probs;
    return -1;
#endif
}
SC_API int spec_hip_draft_stochastic_device(int32_t seq_id, int32_t token,
        int32_t position, float temperature, float p_min, uint64_t rng_key,
        int count, int32_t * output_tokens, int32_t * output_dist_ids,
        float * output_dist_probs) {
#ifdef QWEN4EXP_SIDECAR_PRODUCTION
    return draft_runtime_impl(seq_id, token, position, nullptr, true,
            temperature, p_min, rng_key, true, count, output_tokens,
            output_dist_ids, output_dist_probs);
#else
    (void) seq_id; (void) token; (void) position;
    (void) temperature; (void) p_min; (void) rng_key; (void) count;
    (void) output_tokens; (void) output_dist_ids; (void) output_dist_probs;
    return -1;
#endif
}

#if defined(QWEN4EXP_SIDECAR_LIFECYCLE_TEST) || defined(QWEN4EXP_SIDECAR_ARTIFACT_TEST)
SC_API int qwen4exp_test_tensor_count() {
    return S.ok ? static_cast<int>(S.tensors.size()) : -1;
}

SC_API uint64_t qwen4exp_test_artifact_bytes() {
    return S.ok ? S.artifact_bytes : 0;
}
#endif

#ifdef QWEN4EXP_SIDECAR_ARTIFACT_TEST
SC_API int qwen4exp_test_set_empty_cursor(int32_t seq_id, int32_t position) {
    SeqState * seq = get_seq(seq_id);
    if (!S.ok || seq == nullptr || position < 0 || position >= S.max_pos ||
            seq->allocated || seq->state_pos_min != 0 || seq->state_pos_max != 0) return -1;
    seq->state_pos_min = position;
    seq->state_pos_max = position;
    discard_pending(*seq);
    return 0;
}

SC_API int qwen4exp_test_catchup_normalized(int32_t seq_id, int32_t token,
        int32_t position, const float * hidden_norm) {
    return catchup_phase1_impl(seq_id, &token, &position, hidden_norm, 1, false, true);
}

SC_API int qwen4exp_test_seed_long_index_f16(int32_t seq_id,
        const void * index_rows, int positions) {
    SeqState * seq = get_seq(seq_id);
    if (!S.ok || seq == nullptr || index_rows == nullptr ||
            positions < 1 || positions > S.max_pos ||
            seq->state_pos_min != 0 || seq->state_pos_max != 0 ||
            seq->pending_pos_max != seq->state_pos_max || ensure_seq_storage(*seq) != 0) return -1;
    if (hipMemcpyAsync(seq->index_committed, index_rows,
                static_cast<size_t>(positions) * IDX_WIDTH * sizeof(__half),
                hipMemcpyHostToDevice, S.stream) != hipSuccess ||
            hipStreamSynchronize(S.stream) != hipSuccess) return -1;
    seq->state_pos_min = 0;
    seq->state_pos_max = positions;
    discard_pending(*seq);
    invalidate_qsa_for_seq(seq_id);
    return 0;
}

#ifdef QWEN4EXP_SIDECAR_NATIVE_ATTN_TEST
SC_API int qwen4exp_test_seed_long_kv_f16(int32_t seq_id,
        const void * key_rows, const void * value_rows, int positions) {
    SeqState * seq = get_seq(seq_id);
    const int expected_physical = seq != nullptr ?
            ((seq->state_pos_max + 255) / 256) * 256 : 0;
    if (!S.ok || seq == nullptr || !seq->allocated || key_rows == nullptr ||
            value_rows == nullptr || positions != expected_physical ||
            positions < 256 || positions > S.max_pos ||
            seq->state_pos_min != 0 || seq->state_pos_max < 1 ||
            seq->pending_pos_max != seq->state_pos_max) return -1;
    const size_t bytes = static_cast<size_t>(positions) * KV_WIDTH * sizeof(__half);
    if (hipMemcpyAsync(seq->key_committed, key_rows, bytes,
                hipMemcpyHostToDevice, S.stream) != hipSuccess ||
            hipMemcpyAsync(seq->value_committed, value_rows, bytes,
                hipMemcpyHostToDevice, S.stream) != hipSuccess ||
            hipStreamSynchronize(S.stream) != hipSuccess) return -1;
    if (S.qsa.last_attention_seq == seq_id) {
        S.qsa.last_attention_seq = -1;
        S.qsa.last_attention_physical = 0;
        S.qsa.last_attention_parallel = 0;
    }
    return 0;
}
#endif

SC_API int qwen4exp_test_index_query_from_mixed(const float * mixed,
        int32_t position) {
    QsaScratch & q = S.qsa;
    if (!S.ok || mixed == nullptr || position < 0 || position >= S.max_pos ||
            ensure_qsa_scratch() != 0 ||
            hipMemcpyAsync(q.mixed, mixed, N_EMBD * sizeof(float),
                hipMemcpyHostToDevice, S.stream) != hipSuccess) return -1;
    gemv_f16_f32_batched<256><<<INDEX_HEADS * INDEX_DIM, 256, 0, S.stream>>>(
            S.weights.index_query, q.mixed, q.query, N_EMBD, INDEX_HEADS * INDEX_DIM);
    rms_norm_shared_rows<<<INDEX_HEADS, 256, 0, S.stream>>>(
            q.query, S.weights.index_query_norm, q.query, INDEX_DIM);
    index_rope_neox<<<INDEX_HEADS, N_ROT / 2, 0, S.stream>>>(
            q.query, INDEX_HEADS, position, 0);
    if (hipGetLastError() != hipSuccess || hipStreamSynchronize(S.stream) != hipSuccess) return -1;
    q.query_position = position;
    q.last_qsa_seq = -1;
    q.last_complete_blocks = 0;
    q.last_position = -1;
    q.last_ranked_blocks = 0;
#ifdef QWEN4EXP_SIDECAR_NATIVE_ATTN_TEST
    q.last_attention_seq = -1;
    q.last_attention_physical = 0;
    q.last_attention_parallel = 0;
#endif
    return 0;
}

SC_API int qwen4exp_test_qsa_committed(int32_t seq_id, int32_t position,
        const float * normalized_query) {
    SeqState * seq = get_seq(seq_id);
    QsaScratch & q = S.qsa;
    if (!S.ok || seq == nullptr || !seq->allocated ||
            (normalized_query == nullptr && q.query_position != position) ||
            position < 0 || position >= seq->state_pos_max || position >= S.max_pos ||
            seq->state_pos_min != 0 || ensure_qsa_scratch() != 0) return -1;
    const int valid_tokens = position + 1;
    const int complete_blocks = valid_tokens / QSA_RATIO;
    if (complete_blocks < 1 || complete_blocks > QSA_MAX_BLOCKS) return -1;
    const int target_tokens = std::min(valid_tokens, INDEX_TOP_K + QSA_RATIO - 1);
    const int ranked_tokens = target_tokens - valid_tokens % QSA_RATIO;
    const int ranked_blocks = ranked_tokens / QSA_RATIO + (ranked_tokens % QSA_RATIO != 0);
    if ((normalized_query != nullptr && hipMemcpyAsync(q.query, normalized_query,
                INDEX_HEADS * INDEX_DIM * sizeof(float), hipMemcpyHostToDevice, S.stream) != hipSuccess) ||
            hipMemsetAsync(q.selected, 0, SC_MAX_POS, S.stream) != hipSuccess) return -1;
    qsa_pool_index_f16<<<complete_blocks, INDEX_DIM, 0, S.stream>>>(
            seq->index_committed, q.pooled, complete_blocks);
    rms_norm_shared_rows<<<complete_blocks, 256, 0, S.stream>>>(
            q.pooled, S.weights.index_key_norm, q.normalized, INDEX_DIM);
    index_rope_neox<<<complete_blocks, N_ROT / 2, 0, S.stream>>>(
            q.normalized, complete_blocks, 0, QSA_RATIO);
    index_head_dots_native<<<complete_blocks, 64, 0, S.stream>>>(
            q.normalized, q.query, q.head_dots, complete_blocks);
    index_head_relu_transpose_native<<<
            (complete_blocks * INDEX_HEADS + 255) / 256, 256, 0, S.stream>>>(
            q.head_dots, q.block_heads, complete_blocks);
    index_sum_heads_native<<<complete_blocks, 32, 0, S.stream>>>(
            q.block_heads, q.scores, complete_blocks);

    int partitions = (complete_blocks + QSA_SORT_CAPACITY - 1) / QSA_SORT_CAPACITY;
    int candidate_count = partitions * QSA_MAX_RANKED_BLOCKS;
    qsa_select_candidates_scores<<<partitions, QSA_SORT_THREADS, 0, S.stream>>>(
            q.scores, complete_blocks, q.candidate_keys_a, q.candidate_blocks_a);
    const uint64_t * final_keys = q.candidate_keys_a;
    const int32_t * final_blocks = q.candidate_blocks_a;
    if (candidate_count > QSA_SORT_CAPACITY) {
        partitions = (candidate_count + QSA_SORT_CAPACITY - 1) / QSA_SORT_CAPACITY;
        qsa_select_candidates_pairs<<<partitions, QSA_SORT_THREADS, 0, S.stream>>>(
                q.candidate_keys_a, q.candidate_blocks_a, candidate_count,
                q.candidate_keys_b, q.candidate_blocks_b);
        candidate_count = partitions * QSA_MAX_RANKED_BLOCKS;
        final_keys = q.candidate_keys_b;
        final_blocks = q.candidate_blocks_b;
    }
    if (candidate_count > QSA_SORT_CAPACITY) {
        partitions = (candidate_count + QSA_SORT_CAPACITY - 1) / QSA_SORT_CAPACITY;
        qsa_select_candidates_pairs<<<partitions, QSA_SORT_THREADS, 0, S.stream>>>(
                q.candidate_keys_b, q.candidate_blocks_b, candidate_count,
                q.candidate_keys_a, q.candidate_blocks_a);
        candidate_count = partitions * QSA_MAX_RANKED_BLOCKS;
        final_keys = q.candidate_keys_a;
        final_blocks = q.candidate_blocks_a;
    }
    if (candidate_count > QSA_SORT_CAPACITY) return -1;
    qsa_select_mask_final<<<1, QSA_SORT_THREADS, 0, S.stream>>>(
            final_keys, final_blocks, candidate_count, complete_blocks,
            position, q.selected, q.selected_blocks);
    if (hipGetLastError() != hipSuccess || hipStreamSynchronize(S.stream) != hipSuccess) return -1;
    q.last_complete_blocks = complete_blocks;
    q.last_position = position;
    q.last_ranked_blocks = ranked_blocks;
    q.last_qsa_seq = seq_id;
#ifdef QWEN4EXP_SIDECAR_NATIVE_ATTN_TEST
    q.last_attention_seq = -1;
#endif
    return 0;
}

#ifdef QWEN4EXP_SIDECAR_NATIVE_ATTN_TEST
SC_API int qwen4exp_test_qsa_sparse_attention(int32_t seq_id,
        const float * attention_query, int physical) {
    SeqState * seq = get_seq(seq_id);
    QsaScratch & q = S.qsa;
    const int expected_physical = seq != nullptr ?
            ((seq->state_pos_max + 255) / 256) * 256 : 0;
    if (!S.ok || seq == nullptr || !seq->allocated || attention_query == nullptr ||
            physical != expected_physical || physical < 256 || physical > S.max_pos ||
            seq->state_pos_min != 0 || seq->state_pos_max != q.last_position + 1 ||
            q.last_qsa_seq != seq_id || ensure_native_attention_scratch() != 0 ||
            hipMemcpyAsync(q.attention_query, attention_query,
                N_HEAD * HEAD_DIM * sizeof(float),
                hipMemcpyHostToDevice, S.stream) != hipSuccess) return -1;
    qsa_mask_to_f16<<<(physical + 255) / 256, 256, 0, S.stream>>>(
            q.selected, q.attention_mask, physical);
    int parallel = 0;
    if (hipGetLastError() != hipSuccess ||
            launch_native_sparse_attention(S.stream, q.attention_query,
                seq->key_committed, seq->value_committed, q.attention_mask,
                physical, QSA_MAX_ATTENTION_PARTS, q.attention_parts,
                q.attention_meta, q.attention, &parallel) != 0 ||
            hipStreamSynchronize(S.stream) != hipSuccess) return -1;
    q.last_attention_seq = seq_id;
    q.last_attention_physical = physical;
    q.last_attention_parallel = parallel;
    return 0;
}

SC_API int qwen4exp_test_attention_shape(int * physical, int * parallel) {
    QsaScratch & q = S.qsa;
    if (!S.ok || physical == nullptr || parallel == nullptr ||
            q.last_attention_seq < 0 || q.last_attention_physical <= 0 ||
            q.last_attention_parallel <= 0) return -1;
    *physical = q.last_attention_physical;
    *parallel = q.last_attention_parallel;
    return 0;
}

SC_API int qwen4exp_test_qsa_full_graph_current(int32_t seq_id,
        int32_t position, float temperature) {
    SeqState * seq = get_seq(seq_id);
    QsaScratch & q = S.qsa;
    CatchupScratch & c = S.catchup;
    const int physical = seq != nullptr ?
            ((seq->state_pos_max + 255) / 256) * 256 : 0;
    if (!S.ok || seq == nullptr || !seq->allocated || position != q.last_position ||
            seq->state_pos_min != 0 || seq->state_pos_max != position + 1 ||
            q.last_qsa_seq != seq_id || physical < 256 || physical > S.max_pos ||
            ensure_native_attention_scratch() != 0 ||
            prepare_draft_query(position) != 0) return -1;
    qsa_mask_to_f16<<<(physical + 255) / 256, 256, 0, S.stream>>>(
            q.selected, q.attention_mask, physical);
    int parallel = 0;
    if (hipGetLastError() != hipSuccess ||
            launch_native_sparse_attention(S.stream, c.query,
                seq->key_committed, seq->value_committed, q.attention_mask,
                physical, QSA_MAX_ATTENTION_PARTS, q.attention_parts,
                q.attention_meta, q.attention, &parallel) != 0 ||
            hipMemcpyAsync(c.attention, q.attention,
                N_HEAD * HEAD_DIM * sizeof(float),
                hipMemcpyDeviceToDevice, S.stream) != hipSuccess ||
            finish_draft_from_attention(temperature) != 0 ||
            hipStreamSynchronize(S.stream) != hipSuccess) return -1;
    q.last_attention_seq = seq_id;
    q.last_attention_physical = physical;
    q.last_attention_parallel = parallel;
    return 0;
}

SC_API int qwen4exp_test_qsa_full_graph(int32_t seq_id, int32_t token,
        const float * hidden_row, const float * attention_query,
        const float * attention_override, int physical, float temperature) {
    SeqState * seq = get_seq(seq_id);
    QsaScratch & q = S.qsa;
    const int expected_physical = seq != nullptr ?
            ((seq->state_pos_max + 255) / 256) * 256 : 0;
    if (!S.ok || seq == nullptr || !seq->allocated || hidden_row == nullptr ||
            (attention_override == nullptr && attention_query == nullptr) ||
            token < 0 || token >= N_VOCAB || physical != expected_physical ||
            physical < 256 || physical > S.max_pos ||
            seq->state_pos_min != 0 || seq->state_pos_max >= S.max_pos ||
            seq->pending_pos_max != seq->state_pos_max ||
            q.last_position + 1 != seq->state_pos_max ||
            q.last_qsa_seq != seq_id) return -1;
    const int32_t position = seq->state_pos_max;
    if (catchup_phase1_impl(seq_id, &token, &position,
                hidden_row, 1, false, false) != 0) return -1;
    int result = -1;
    do {
        if (prepare_draft_query(q.last_position) != 0) break;
        if (attention_override != nullptr) {
            if (hipMemcpyAsync(S.catchup.attention, attention_override,
                        N_HEAD * HEAD_DIM * sizeof(float),
                        hipMemcpyHostToDevice, S.stream) != hipSuccess) break;
        } else {
            if (qwen4exp_test_qsa_sparse_attention(seq_id,
                        attention_query, physical) != 0 ||
                    hipMemcpyAsync(S.catchup.attention, q.attention,
                        N_HEAD * HEAD_DIM * sizeof(float),
                        hipMemcpyDeviceToDevice, S.stream) != hipSuccess) break;
        }
        if (finish_draft_from_attention(temperature) != 0 ||
                hipStreamSynchronize(S.stream) != hipSuccess) break;
        result = 0;
    } while (false);
    discard_pending(*seq);
    return result;
}

SC_API int qwen4exp_test_normalize_proposal(float temperature) {
    CatchupScratch & c = S.catchup;
    if (!S.ok || !c.proposal_valid ||
            !(temperature > 0.0f && temperature <= 100.0f)) return -1;
    proposal_normalize_top32<<<1, 1, 0, S.stream>>>(
            c.topk_values, temperature, c.topk_probabilities);
    return hipGetLastError() == hipSuccess &&
            hipStreamSynchronize(S.stream) == hipSuccess ? 0 : -1;
}
#endif

SC_API int qwen4exp_test_read_qsa(int kind, void * data, int size) {
    QsaScratch & q = S.qsa;
    if (!S.ok || !q.allocated || (kind != 5 && q.last_position < 0) || data == nullptr) return -1;
    const void * source = nullptr;
    int expected = 0;
    switch (kind) {
        case 0: source = q.pooled; expected = q.last_complete_blocks * INDEX_DIM * sizeof(float); break;
        case 1: source = q.normalized; expected = q.last_complete_blocks * INDEX_DIM * sizeof(float); break;
        case 2: source = q.scores; expected = q.last_complete_blocks * sizeof(float); break;
        case 3: source = q.selected; expected = (q.last_position + 1) * sizeof(uint8_t); break;
        case 4: source = q.selected_blocks; expected = q.last_ranked_blocks * sizeof(int32_t); break;
        case 5: source = q.query; expected = INDEX_HEADS * INDEX_DIM * sizeof(float); break;
#ifdef QWEN4EXP_SIDECAR_NATIVE_ATTN_TEST
        case 6:
            if (q.last_attention_seq < 0) return -1;
            source = q.attention;
            expected = N_HEAD * HEAD_DIM * sizeof(float);
            break;
#endif
        default: return -1;
    }
    if (size != expected) return -1;
    return hipMemcpy(data, source, static_cast<size_t>(size), hipMemcpyDeviceToHost) == hipSuccess ? 0 : -1;
}

SC_API int qwen4exp_test_seed_short_prefix(int32_t seq_id, int32_t position,
        const float * key_rows, const float * value_rows, int count) {
    SeqState * seq = get_seq(seq_id);
    if (!S.ok || seq == nullptr || key_rows == nullptr || value_rows == nullptr ||
            count < 1 || position < 0 || position + count > 32 ||
            seq->allocated || seq->state_pos_min != 0 || seq->state_pos_max != 0 ||
            ensure_seq_storage(*seq) != 0 || ensure_catchup_scratch() != 0) return -1;
    for (int row = 0; row < count; ++row) {
        const size_t offset = static_cast<size_t>(row) * KV_WIDTH;
        if (hipMemcpyAsync(S.catchup.key, key_rows + offset, KV_WIDTH * sizeof(float),
                    hipMemcpyHostToDevice, S.stream) != hipSuccess ||
                hipMemcpyAsync(S.catchup.value, value_rows + offset, KV_WIDTH * sizeof(float),
                    hipMemcpyHostToDevice, S.stream) != hipSuccess) return -1;
        float_to_half<<<(KV_WIDTH + 255) / 256, 256, 0, S.stream>>>(
                S.catchup.key, seq->key_committed + static_cast<size_t>(position + row) * KV_WIDTH,
                KV_WIDTH);
        float_to_half<<<(KV_WIDTH + 255) / 256, 256, 0, S.stream>>>(
                S.catchup.value, seq->value_committed + static_cast<size_t>(position + row) * KV_WIDTH,
                KV_WIDTH);
    }
    if (hipStreamSynchronize(S.stream) != hipSuccess) return -1;
    seq->state_pos_min = position;
    seq->state_pos_max = position + count;
    discard_pending(*seq);
    return 0;
}

SC_API int qwen4exp_test_draft_one_short(int32_t seq_id, int32_t token,
        int32_t position, const float * hidden_row) {
    SeqState * seq = get_seq(seq_id);
    if (!S.ok || seq == nullptr || hidden_row == nullptr || position != seq->state_pos_max ||
            position >= 32 || catchup_phase1_impl(seq_id, &token, &position,
                hidden_row, 1, false, false) != 0) return -1;
    copy_cache_rows_f16<<<(KV_WIDTH + 255) / 256, 256, 0, S.stream>>>(
            seq->key_committed, seq->key_pending, position, 0, 1, KV_WIDTH);
    copy_cache_rows_f16<<<(KV_WIDTH + 255) / 256, 256, 0, S.stream>>>(
            seq->value_committed, seq->value_pending, position, 0, 1, KV_WIDTH);
    copy_cache_rows_f16<<<(IDX_WIDTH + 255) / 256, 256, 0, S.stream>>>(
            seq->index_committed, seq->index_pending, position, 0, 1, IDX_WIDTH);
    if (draft_one_short(*seq, position) != 0 || hipStreamSynchronize(S.stream) != hipSuccess) {
        discard_pending(*seq);
        return -1;
    }
    discard_pending(*seq);
    return 0;
}

SC_API int qwen4exp_test_draft_chain_short(int32_t seq_id, int32_t token,
        int32_t position, const float * hidden_row,
        int count, int32_t * output_tokens) {
    SeqState * seq = get_seq(seq_id);
    CatchupScratch & c = S.catchup;
    if (!S.ok || seq == nullptr || hidden_row == nullptr || output_tokens == nullptr ||
            count < 1 || count > 8 || position != seq->state_pos_max ||
            position + count > 32 || seq->pending_pos_max != seq->state_pos_max ||
            ensure_pending_storage(*seq, count) != 0 || ensure_catchup_scratch() != 0 ||
            catchup_phase1_impl(seq_id, &token, &position, hidden_row, 1, false, false) != 0) return -1;
    seq->pending_pos_min = position;
    seq->pending_pos_max = position + count;
    for (int step = 0; step < count; ++step) {
        const int32_t current_position = position + step;
        copy_cache_rows_f16<<<(KV_WIDTH + 255) / 256, 256, 0, S.stream>>>(
                seq->key_committed, seq->key_pending, current_position, step, 1, KV_WIDTH);
        copy_cache_rows_f16<<<(KV_WIDTH + 255) / 256, 256, 0, S.stream>>>(
                seq->value_committed, seq->value_pending, current_position, step, 1, KV_WIDTH);
        copy_cache_rows_f16<<<(IDX_WIDTH + 255) / 256, 256, 0, S.stream>>>(
                seq->index_committed, seq->index_pending, current_position, step, 1, IDX_WIDTH);
        if (draft_one_short(*seq, current_position) != 0 ||
                hipMemcpyAsync(c.chain_query_full + static_cast<size_t>(step) * 2 * N_HEAD * HEAD_DIM,
                    c.query_full, 2 * N_HEAD * HEAD_DIM * sizeof(float), hipMemcpyDeviceToDevice, S.stream) != hipSuccess ||
                hipMemcpyAsync(c.chain_query_normalized + static_cast<size_t>(step) * N_HEAD * HEAD_DIM,
                    c.query_normalized, N_HEAD * HEAD_DIM * sizeof(float), hipMemcpyDeviceToDevice, S.stream) != hipSuccess ||
                hipMemcpyAsync(c.chain_query + static_cast<size_t>(step) * N_HEAD * HEAD_DIM,
                    c.query, N_HEAD * HEAD_DIM * sizeof(float), hipMemcpyDeviceToDevice, S.stream) != hipSuccess ||
                hipMemcpyAsync(c.chain_attention_pregate + static_cast<size_t>(step) * N_HEAD * HEAD_DIM,
                    c.attention_pregate, N_HEAD * HEAD_DIM * sizeof(float), hipMemcpyDeviceToDevice, S.stream) != hipSuccess ||
                hipMemcpyAsync(c.chain_attention_hc + static_cast<size_t>(step) * HC_WIDTH,
                    c.attention_hc, HC_WIDTH * sizeof(float), hipMemcpyDeviceToDevice, S.stream) != hipSuccess ||
                hipMemcpyAsync(c.chain_hidden + static_cast<size_t>(step) * HC_WIDTH,
                    c.eh, HC_WIDTH * sizeof(float), hipMemcpyDeviceToDevice, S.stream) != hipSuccess ||
                hipMemcpyAsync(c.chain_logits + static_cast<size_t>(step) * N_VOCAB,
                    c.logits, N_VOCAB * sizeof(float), hipMemcpyDeviceToDevice, S.stream) != hipSuccess ||
                hipMemcpyAsync(c.chain_tokens + step, c.topk_indices, sizeof(int32_t),
                    hipMemcpyDeviceToDevice, S.stream) != hipSuccess) {
            discard_pending(*seq);
            return -1;
        }
        if (step + 1 < count) {
            rms_norm_rows<<<1, 1024, 0, S.stream>>>(
                    c.eh, S.weights.hidden_norm, c.hidden_norm, HC_WIDTH);
            if (phase1_from_normalized_hidden_device_token(*seq, c.chain_tokens + step,
                        current_position + 1, step + 1) != 0) {
                discard_pending(*seq);
                return -1;
            }
        }
    }
    if (hipMemcpyAsync(output_tokens, c.chain_tokens,
                static_cast<size_t>(count) * sizeof(int32_t),
                hipMemcpyDeviceToHost, S.stream) != hipSuccess ||
            hipStreamSynchronize(S.stream) != hipSuccess) {
        discard_pending(*seq);
        return -1;
    }
    for (int step = 0; step < count; ++step) {
        if (output_tokens[step] < 0 || output_tokens[step] >= N_VOCAB) {
            discard_pending(*seq);
            return -1;
        }
    }
    discard_pending(*seq);
    return 0;
}

SC_API int qwen4exp_test_read_chain_cache(int32_t seq_id, int kind,
        int32_t position, void * data, int size) {
    SeqState * seq = get_seq(seq_id);
    if (!S.ok || seq == nullptr || !seq->allocated || data == nullptr ||
            position < seq->state_pos_max || position >= 32) return -1;
    const __half * cache = kind == 0 ? seq->key_committed :
            kind == 1 ? seq->value_committed : kind == 2 ? seq->index_committed : nullptr;
    const int width = kind < 2 ? KV_WIDTH : IDX_WIDTH;
    if (cache == nullptr || size != width * static_cast<int>(sizeof(__half))) return -1;
    return hipMemcpy(data, cache + static_cast<size_t>(position) * width,
            static_cast<size_t>(size), hipMemcpyDeviceToHost) == hipSuccess ? 0 : -1;
}

SC_API int qwen4exp_test_read_chain_step(int step, int kind, void * data, int size) {
    if (!S.ok || step < 0 || step >= 8 || data == nullptr) return -1;
    const void * source = nullptr;
    int expected = 0;
    if (kind == 0) {
        source = S.catchup.chain_hidden + static_cast<size_t>(step) * HC_WIDTH;
        expected = HC_WIDTH * sizeof(float);
    } else if (kind == 1) {
        source = S.catchup.chain_logits + static_cast<size_t>(step) * N_VOCAB;
        expected = N_VOCAB * sizeof(float);
    } else if (kind == 2) {
        source = S.catchup.chain_attention_hc + static_cast<size_t>(step) * HC_WIDTH;
        expected = HC_WIDTH * sizeof(float);
    } else if (kind == 3) {
        source = S.catchup.chain_attention_pregate + static_cast<size_t>(step) * N_HEAD * HEAD_DIM;
        expected = N_HEAD * HEAD_DIM * sizeof(float);
    } else if (kind == 4) {
        source = S.catchup.chain_query + static_cast<size_t>(step) * N_HEAD * HEAD_DIM;
        expected = N_HEAD * HEAD_DIM * sizeof(float);
    } else if (kind == 5) {
        source = S.catchup.chain_query_full + static_cast<size_t>(step) * 2 * N_HEAD * HEAD_DIM;
        expected = 2 * N_HEAD * HEAD_DIM * sizeof(float);
    } else if (kind == 6) {
        source = S.catchup.chain_query_normalized + static_cast<size_t>(step) * N_HEAD * HEAD_DIM;
        expected = N_HEAD * HEAD_DIM * sizeof(float);
    } else {
        return -1;
    }
    if (size != expected) return -1;
    return hipMemcpy(data, source, static_cast<size_t>(size), hipMemcpyDeviceToHost) == hipSuccess ? 0 : -1;
}

SC_API int qwen4exp_test_read_one_step(int kind, void * data, int size) {
    const void * source = nullptr;
    int expected = 0;
    if (!S.ok || data == nullptr) return -1;
    switch (kind) {
        case 0: source = S.catchup.attention_hc; expected = HC_WIDTH * sizeof(float); break;
        case 1: source = S.catchup.eh; expected = HC_WIDTH * sizeof(float); break;
        case 2: source = S.catchup.mixed; expected = N_EMBD * sizeof(float); break;
        case 3: source = S.catchup.logits; expected = N_VOCAB * sizeof(float); break;
        case 4: source = S.catchup.key; expected = KV_WIDTH * sizeof(float); break;
        case 5: source = S.catchup.value; expected = KV_WIDTH * sizeof(float); break;
        case 6: source = S.catchup.phase1_mixed; expected = N_EMBD * sizeof(float); break;
        case 7: source = S.catchup.query; expected = N_HEAD * HEAD_DIM * sizeof(float); break;
        case 8: source = S.catchup.attention_gate; expected = N_HEAD * HEAD_DIM * sizeof(float); break;
        case 9: source = S.catchup.attention_pregate; expected = N_HEAD * HEAD_DIM * sizeof(float); break;
        case 10: source = S.catchup.topk_indices; expected = PROPOSAL_TOP_K * sizeof(int32_t); break;
        case 11: source = S.catchup.topk_values; expected = PROPOSAL_TOP_K * sizeof(float); break;
        case 12: source = S.catchup.topk_probabilities; expected = PROPOSAL_TOP_K * sizeof(float); break;
        case 13: source = S.catchup.ffn_mixed_snapshot; expected = N_EMBD * sizeof(float); break;
        default: return -1;
    }
    if (size != expected) return -1;
    return hipMemcpy(data, source, static_cast<size_t>(size), hipMemcpyDeviceToHost) == hipSuccess ? 0 : -1;
}
#endif

#ifdef QWEN4EXP_SIDECAR_NATIVE_MMQ_TEST
SC_API int qwen4exp_test_read_prompt_batch(int kind, void * data, int size) {
    PromptBatchScratch & b = S.prompt_batch;
    const void * source = nullptr;
    int expected = 0;
    if (!S.ok || !b.allocated || data == nullptr) return -1;
    const int rows = b.rows;
    if (rows < 9 || rows > PromptBatchScratch::MAX_ROWS) return -1;
    switch (kind) {
        case 0: source = b.hc_norm; expected = rows * HC_WIDTH * sizeof(float); break;
        case 1: source = b.hc_gate; expected = rows * HC_WIDTH * sizeof(float); break;
        case 2: source = b.hc_inject; expected = rows * N_HC * sizeof(float); break;
        case 3: source = b.mixed; expected = rows * N_EMBD * sizeof(float); break;
        case 4: source = b.key; expected = rows * KV_WIDTH * sizeof(float); break;
        case 5: source = b.value; expected = rows * KV_WIDTH * sizeof(float); break;
        case 6: source = b.index_key; expected = rows * IDX_WIDTH * sizeof(float); break;
        default: return -1;
    }
    if (size != expected) return -1;
    SC_CHECK(hipMemcpyAsync(data, source, static_cast<size_t>(size),
            hipMemcpyDeviceToHost, S.stream));
    SC_CHECK(hipStreamSynchronize(S.stream));
    return 0;
}
#endif

#if defined(QWEN4EXP_SIDECAR_LIFECYCLE_TEST) || defined(QWEN4EXP_SIDECAR_ARTIFACT_TEST)
SC_API int qwen4exp_test_cache_value(int32_t seq_id, int kind,
        int32_t position, int index, float * value) {
    SeqState * seq = get_seq(seq_id);
    if (!S.ok || seq == nullptr || !seq->allocated || value == nullptr ||
            position < seq->state_pos_min || position >= seq->state_pos_max) return -1;
    const __half * cache = kind == 0 ? seq->key_committed :
            kind == 1 ? seq->value_committed : kind == 2 ? seq->index_committed : nullptr;
    const int width = kind < 2 ? KV_WIDTH : IDX_WIDTH;
    if (cache == nullptr || index < 0 || index >= width) return -1;
    __half result;
    SC_CHECK(hipMemcpyAsync(&result, cache + static_cast<size_t>(position) * width + index,
            sizeof(result), hipMemcpyDeviceToHost, S.stream));
    SC_CHECK(hipStreamSynchronize(S.stream));
    *value = __half2float(result);
    return 0;
}

SC_API int qwen4exp_test_read_cache_row(int32_t seq_id, int kind,
        int32_t position, void * data, int size) {
    SeqState * seq = get_seq(seq_id);
    if (!S.ok || seq == nullptr || !seq->allocated || data == nullptr ||
            position < seq->state_pos_min || position >= seq->state_pos_max) return -1;
    const __half * cache = kind == 0 ? seq->key_committed :
            kind == 1 ? seq->value_committed : kind == 2 ? seq->index_committed : nullptr;
    const int width = kind < 2 ? KV_WIDTH : IDX_WIDTH;
    if (cache == nullptr || size != width * static_cast<int>(sizeof(__half))) return -1;
    SC_CHECK(hipMemcpyAsync(data, cache + static_cast<size_t>(position) * width,
            static_cast<size_t>(size), hipMemcpyDeviceToHost, S.stream));
    SC_CHECK(hipStreamSynchronize(S.stream));
    return 0;
}

SC_API int qwen4exp_test_read_cache_rows(int32_t seq_id, int kind,
        int32_t position, int count, void * data, int size) {
    SeqState * seq = get_seq(seq_id);
    if (!S.ok || seq == nullptr || !seq->allocated || data == nullptr || count < 1 ||
            position < seq->state_pos_min || position > seq->state_pos_max - count) return -1;
    const __half * cache = kind == 0 ? seq->key_committed :
            kind == 1 ? seq->value_committed : kind == 2 ? seq->index_committed : nullptr;
    const int width = kind < 2 ? KV_WIDTH : IDX_WIDTH;
    const size_t expected = static_cast<size_t>(count) * width * sizeof(__half);
    if (cache == nullptr || expected != static_cast<size_t>(size)) return -1;
    SC_CHECK(hipMemcpyAsync(data, cache + static_cast<size_t>(position) * width,
            expected, hipMemcpyDeviceToHost, S.stream));
    SC_CHECK(hipStreamSynchronize(S.stream));
    return 0;
}

SC_API int qwen4exp_test_hidden_valid(int32_t seq_id) {
    SeqState * seq = get_seq(seq_id);
    return S.ok && seq != nullptr && seq->hidden_valid ? 1 : 0;
}

#endif

#if defined(QWEN4EXP_SIDECAR_LIFECYCLE_TEST) || defined(QWEN4EXP_SIDECAR_ARTIFACT_TEST)
SC_API int qwen4exp_test_shutdown() {
    if (!S.ok) return 0;
    SC_CHECK(hipStreamSynchronize(S.stream));
    if (S.stream != S.owned_stream) S.stream = S.owned_stream;
    for (int i = 0; i < SC_MAX_SEQ; ++i) {
        if (free_seq(S.seq[i]) != 0) return -1;
    }
    if (release_artifact_allocations() != 0) return -1;
    SC_CHECK(hipStreamDestroy(S.owned_stream));
    S = State{};
    return 0;
}
#endif
