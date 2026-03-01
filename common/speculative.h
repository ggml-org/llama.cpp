#pragma once

#include "llama.h"
#include "common.h"

// common/speculative.h has two interfaces:
//
// 1) struct common_speculative with init, begin, draft, accept and print_stats
//    Simple interface, see examples/speculative/speculative.cpp
//
// 2) struct common_speculative_session with struct common_speculative_callback
//    Complex interface which supports checkpoints, see tools/server/server-context.cpp
//

struct common_speculative;

// comma separated list of all types
std::string common_speculative_type_name_str();

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

// check if the llama_context is compatible for speculative decoding
// note: clears the memory of the context
bool common_speculative_is_compat(llama_context * ctx_tgt);

common_speculative * common_speculative_init(
        common_params_speculative & params,
        llama_context             * ctx_tgt);

void common_speculative_free(common_speculative * spec);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, const llama_tokens & prompt);

// sample up to n_draft tokens and add them to the batch using the draft model
llama_tokens common_speculative_draft(
                     common_speculative * spec,
        const common_params_speculative & params,
                     const llama_tokens & prompt,
                            llama_token   id_last);

// informs the speculative decoder that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, uint16_t n_accepted);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec);



// Interactions with server
//

// callback implemented by the server
struct common_speculative_callback {
    virtual ~common_speculative_callback();

    // Add a token to the draft sequence.
    virtual void batch_add_token(const llama_token token, bool logits) = 0;

    // Sample and accept tokens from the main model.
    virtual llama_tokens sampler_sample_and_accept_n(const llama_tokens & drafted) = 0;

    // Deletes a part of the context.
    // Returns true if the memory was modified.
    virtual bool memory_seq_rm(llama_pos p0, llama_pos p1) = 0;

    // Creates a checkpoint of the current state of the context.
    // Returns the size of the checkpoint in bytes.
    virtual size_t create_checkpoint() = 0;

    // Restore a checkpoint previously created by create_checkpoint().
    // Returns the size of the restored checkpoint in bytes.
    virtual size_t restore_checkpoint(size_t ckpt_size_part_expected) = 0;

    // Delete a checkpoint previously created by create_checkpoint().
    virtual void delete_checkpoint() = 0;
};

struct common_speculative_accept_response {
    llama_tokens tokens;
    size_t       draft_size_initial;
    bool         skip_acceptance;

    common_speculative_accept_response(llama_tokens t, size_t draft_size_initial, bool skip)
        : tokens(std::move(t)), draft_size_initial(draft_size_initial), skip_acceptance(skip) {}
};

// speculative decoding which may use checkpoints to rewind in tokens history
struct common_speculative_session {

    common_speculative_session(
                  common_speculative_callback & callback,
            const common_params_speculative   & params,
                  llama_context               * ctx_tgt);

    ~common_speculative_session();

    // dont copy
    common_speculative_session(const common_speculative_session &) = delete;
    common_speculative_session & operator=(const common_speculative_session &) = delete;


    // call once at the beginning of a new generation
    // some spec implementations use the prompt history to initialize lookup maps
    void begin(const llama_tokens & prompt_history);

    bool has_batch_dft();

    // do speculative decoding to compute a draft of tokens
    llama_tokens compute_draft(const llama_tokens & prompt,
                                      llama_token   id_last,
                                      int           n_draft_max_slot);

    // check if and how far the current draft is accepted
    common_speculative_accept_response sample_and_accept();

    // rewind (because of a draft not fully accepted)
    void rewind(const llama_pos p0);

    // print statistics
    void print_stats() const;

    // reset and delete structures
    void reset();

    private:
        struct impl;
        impl * p_impl;

};

