#pragma once

#include "llama-cpp.h"
#include "llama.h"

#include "common.h"

#include <cstdint>
#include <string>
#include <vector>

//
// shared helpers of the context state cases
//

extern const double NMSE_THRESHOLD;

//
// comparison
//

// normalized mean squared error = mse(a, b) / mse(a, 0), infinity if any logit is not finite

// accumulates logit comparisons and derives their NMSE, so that a case can compare an entire
// multi-seq batch row by row and still have a single verdict
struct logits_diff {
    double   mse_ab   = 0.0;
    double   mse_a0   = 0.0;
    float    diff_max = 0.0f;
    size_t   n_nan    = 0;
    size_t   n        = 0;
    int64_t  i_first  = -1; // flat index of the first differing element

    void add(const float * a, const float * b, size_t n_elem);

    double nmse() const;
};

logits_diff compare_logits(const float * a, const float * b, size_t n);

//
// generation
//

// logits produced by a run, compared against by the replay cases
struct generation_result {
    llama_tokens tokens;
    std::vector<std::vector<float>> logits;

    bool empty() const { return tokens.empty(); }
};

//
// decode
//

bool get_current_logits(llama_context * ctx, std::vector<float> & out);

// decode tokens [p_begin, p_end) of one sequence
bool decode_seq(llama_context * ctx, const llama_tokens & tokens, llama_pos p_begin, llama_pos p_end,
                llama_seq_id seq_id, bool last_output);

bool decode_tokens(llama_context * ctx, const llama_tokens & tokens, llama_seq_id seq_id = 0);

// tokens derived from (seq, pos): the recurrent cases do not use the tokenizer, so they do
// not depend on the prompt or on the prompt length
llama_token gen_token(int32_t n_vocab, llama_seq_id seq_id, llama_pos pos);

bool decode_gen(llama_context * ctx, int32_t n_vocab, llama_seq_id seq_id, llama_pos p_begin, llama_pos p_end);

bool decode_one(llama_context * ctx, llama_token tok, llama_pos pos, llama_seq_id seq_id = 0);

// greedy-free replay: sample for reporting, but feed the sampled tokens back
bool generate_tokens(llama_context * ctx, llama_sampler * smpl, int & n_past, int32_t n_predict, llama_seq_id seq_id,
                     generation_result & result);

// as generate_tokens, but feeds the expected tokens back so that the compared logits stay on
// the reference trajectory
bool generate_tokens_compare(llama_context * ctx, llama_sampler * smpl, int & n_past, int32_t n_predict, llama_seq_id seq_id,
                             const generation_result & expected, double nmse_eps);

//
// state
//

// llama_state_seq_get_size_ext + get_data_ext, with the length verified
bool get_seq_state(llama_context * ctx, llama_seq_id seq_id, uint32_t flags, std::vector<uint8_t> & state);

// llama_state_seq_set_data_ext, with the length verified
bool set_seq_state(llama_context * ctx, llama_seq_id seq_id, uint32_t flags, const std::vector<uint8_t> & state);

// drop a sequence entirely
bool seq_rm(llama_context * ctx, llama_seq_id seq_id);

// load a saved session and replay its last token, leaving n_past ready for the next token
bool load_state_replay_last(llama_context * ctx, const std::string & path, const llama_tokens & tokens, int & n_past);

// false when the sizes differ or any byte differs, the first difference is logged
bool blobs_equal(const std::vector<uint8_t> & a, const std::vector<uint8_t> & b);

// overwrite the tensor data with 0xff bytes (NaN when read as f16/f32), so that the restore fails
bool corrupt_state(std::vector<uint8_t> & data);

//
// context
//

// per-test context overrides, sentinel values mean "take the value from common_params"
struct ctx_opts {
    uint32_t n_ctx     = 0; // 0 = from params
    uint32_t n_batch   = 0; // 0 = from params
    uint32_t n_ubatch  = 0; // 0 = from params
    uint32_t n_seq_max = 0; // 0 = from params
    uint32_t n_rs_seq  = 0; // 0 = from params, n_batch / n_ubatch are raised to fit it
    int      kv_unified = -1; // -1 = from params, 0 = off, 1 = on

    llama_flash_attn_type fa = LLAMA_FLASH_ATTN_TYPE_AUTO; // AUTO = from params

    ggml_type type_k = GGML_TYPE_COUNT; // COUNT = from params
    ggml_type type_v = GGML_TYPE_COUNT; // COUNT = from params

    uint8_t fill = 0; // fill the cache buffers with this byte, 0 = disabled

    // a probe that expects the context to be refused should not report it as an error
    bool quiet = false;
};

llama_context_ptr init_ctx(llama_model * model, llama_context_params cparams, uint8_t fill);
llama_context_ptr make_ctx(llama_model * model, const common_params & params, const ctx_opts & opts);

llama_sampler_ptr make_dist_sampler(uint32_t seed);

// these archs reserve the final pp graph with n_seqs = 1, so any multi-seq graph has a
// different layout and re-reserves by design - they must not be run with a multi-seq batch
// when GGML_SCHED_DEBUG_REALLOC is enabled
// see [TAG_RESERVE_DIAG_DECAY] in llama-context.cpp
bool arch_reserves_single_seq(const llama_model * model);

//
// the state of a single model run, handed to every case
//

struct model_run {
    common_params     params;        // per-model copy, out_file is scoped to this model
    std::string       name;          // model file name, used for the table and for temp files
    llama_model *     model          = nullptr;
    llama_tokens      tokens;
    generation_result baseline;
    bool              have_baseline  = false;
    uint8_t           fill           = 0; // set by the runner before each case

    // a context with these options, the fill of this run applied; nullptr when it failed to
    // init, the reason is logged
    llama_context_ptr ctx(const ctx_opts & opts = {}) const;

    // a seeded distribution sampler
    llama_sampler_ptr sampler() const;

    bool is_recurrent() const;
    bool has_vocab() const;
};
