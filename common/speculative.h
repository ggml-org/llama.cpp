#pragma once

#include "llama.h"
#include "common.h"

struct common_speculative;

// comma separated list of all types
std::string common_speculative_type_name_str();

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq);

void common_speculative_free(common_speculative * spec);

// optionally call once at the beginning of a new generation
// TODO: when common_speculative_process() is implemented, we can remove this _begin() function and
//       implement all the logic within common_speculative_process()
void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt);

// TODO: implement [TAG_COMMON_SPECULATIVE_PROCESS]
//bool common_speculative_process(common_speculative * spec, const llama_batch & batch);

struct common_speculative_draft_params {
    // this flag helps chain the drafts through all the implementations
    // after the first successful draft from an implementation, we set it
    //   to false to prevent further drafts for that sequence
    bool drafting = true;

    // overrides individual configurations (-1 disabled)
    // can be used to constraint the max draft based on the remaining context size
    int32_t n_max = -1;

    llama_pos   n_past;
    llama_token id_last;

    const llama_tokens * prompt;

    llama_tokens * result;
};

using common_speculative_draft_params_vec = std::vector<common_speculative_draft_params>;

// generate drafts for the sequences specified in dparams
// requires that `dparams.size() == n_seq` using during common_speculative_init()
void common_speculative_draft(
                     common_speculative * spec,
    common_speculative_draft_params_vec & dparams);

// informs the speculative decoder that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, llama_seq_id, uint16_t n_accepted);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec);

struct common_speculative_deleter {
    void operator()(common_speculative * s) { common_speculative_free(s); }
};

typedef std::unique_ptr<common_speculative, common_speculative_deleter> common_speculative_ptr;
