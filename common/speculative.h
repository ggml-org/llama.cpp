#pragma once

#include "llama.h"
#include "common.h"

struct common_speculative;

struct common_speculative_token_dist {
    llama_tokens ids;
    std::vector<float> probs;
};

// comma separated list the provided types
std::string common_speculative_type_name_str(const std::vector<enum common_speculative_type> & types);

// comma separated list of all types
const char * common_speculative_all_types_str();

// parse user provided types
std::vector<enum common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names);

// infer the spec types from the GGUF metadata of a draft model; empty if unknown
std::vector<enum common_speculative_type> common_speculative_types_from_gguf(const std::string & path);

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

// return the max number of draft tokens based on the speculative parameters
int32_t common_speculative_n_max(const common_params_speculative * spec);

common_params common_base_params_to_speculative(const common_params & params);

// Probe an opt-in sidecar before constructing any host draft model/context.
// Returns the selected sidecar type and marks params.draft.sidecar_only on
// success; returns NONE when native draft loading should be retained.
common_speculative_type common_speculative_sidecar_preflight(
        common_params_speculative & params, const llama_model * model_tgt,
        uint32_t n_seq, std::string & error);

// Cheap pre-target check used by --fit to avoid creating a host draft model
// solely for memory measurement when the explicit sidecar contract is present.
bool common_speculative_sidecar_candidate(const common_params_speculative & params,
        const std::string & target_model_path, uint32_t n_seq);

struct common_speculative_output_limits {
    int32_t total;
    int32_t per_seq;
};

// return the output limits needed for speculative decoding
common_speculative_output_limits common_speculative_get_output_limits(
        int32_t n_batch, int32_t n_parallel, int32_t n_draft);

common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq);

void common_speculative_free(common_speculative * spec);

struct common_speculative_draft_params {
    // this flag is used to chain the drafts through all the available implementations
    // after the first successful draft from an implementation, we set it
    //   to false to prevent further drafts for that sequence
    // at the end of the draft() call, all drafting flags will be reset to false
    bool drafting = false;

    // overrides individual configurations (-1 disabled)
    // can be used to constrain the max draft based on the remaining context size
    int32_t n_max = -1;

    // Set by the server only when the request supplied speculative.n_max.
    // This is internal metadata; it does not add a command-line/API setting.
    bool n_max_user_override = false;

    llama_pos   n_past;
    llama_token id_last;

    // TODO: remove in the future by keeping track of the prompt from the _begin() call and the consecutive accept calls
    const llama_tokens * prompt;

    // the generated draft from the last _draft() call
    llama_tokens * result;

    // optional sparse proposal distributions, one per draft token
    std::vector<common_speculative_token_dist> * dists = nullptr;

    float temperature = 0.0f;
    uint32_t seed = LLAMA_DEFAULT_SEED;
};

common_speculative_draft_params & common_speculative_get_draft_params(common_speculative * spec, llama_seq_id seq_id);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt);

// process the batch and update the internal state of the speculative context
bool common_speculative_process(common_speculative * spec, const llama_batch & batch);

// generate drafts for the sequences specified with `common_speculative_get_draft_params`
void common_speculative_draft(common_speculative * spec);

// informs the speculative context that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, llama_seq_id, uint16_t n_accepted);

// (optional) checkpoint and lifecycle state. State is keyed by implementation;
// sidecars store only a small cursor/epoch while keeping device KV resident.
bool common_speculative_get_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data);
bool common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data);
void common_speculative_reset_state(common_speculative * spec, llama_seq_id seq_id);
// Release per-request state while allowing implementations to retain state tied
// to a resident target prompt. Before reusing that prompt, prepare validates the
// implementation cursor; false means the caller must replay from position zero.
void common_speculative_release_state(common_speculative * spec, llama_seq_id seq_id);
bool common_speculative_prepare_prompt_state(
        common_speculative * spec, llama_seq_id seq_id, llama_pos pos_next, bool can_reuse_resident);
// Discard a state suffix without committing new target rows.
bool common_speculative_truncate_state(common_speculative * spec, llama_seq_id seq_id, llama_pos pos_max);
// Commit target rows that are known to be accepted (prompt/ordinary decode or
// the accepted prefix of speculative verification).
bool common_speculative_commit_state(common_speculative * spec, llama_seq_id seq_id, llama_pos pos_max);
bool common_speculative_rebase_state(common_speculative * spec, llama_seq_id seq_id,
        llama_pos pos_min, llama_pos pos_max, llama_pos delta);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec);

struct common_speculative_deleter {
    void operator()(common_speculative * s) { common_speculative_free(s); }
};

typedef std::unique_ptr<common_speculative, common_speculative_deleter> common_speculative_ptr;

struct common_speculative_init_result {
    common_speculative_init_result(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
    ~common_speculative_init_result();

    llama_model   * model();
    llama_context * context();
    bool sidecar_only() const;
    common_speculative_type sidecar_type() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using common_speculative_init_result_ptr = std::unique_ptr<common_speculative_init_result>;

common_speculative_init_result_ptr common_speculative_init_from_params(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
