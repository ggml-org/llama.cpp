#pragma once

#ifndef __cplusplus
#error "This header is for C++ only"
#endif

#include <memory>
#include <vector>

#include "llama.h"

struct llama_model_deleter {
    void operator()(llama_model * model) { llama_model_free(model); }
};

struct llama_context_deleter {
    void operator()(llama_context * context) { llama_free(context); }
};

struct llama_sampler_deleter {
    void operator()(llama_sampler * sampler) { llama_sampler_free(sampler); }
};

struct llama_adapter_lora_deleter {
    void operator()(llama_adapter_lora * adapter) { llama_adapter_lora_free(adapter); }
};

struct llama_batch_ext_deleter {
    void operator()(llama_batch_ext * batch) { llama_batch_ext_free(batch); }
};

typedef std::unique_ptr<llama_model, llama_model_deleter> llama_model_ptr;
typedef std::unique_ptr<llama_context, llama_context_deleter> llama_context_ptr;
typedef std::unique_ptr<llama_sampler, llama_sampler_deleter> llama_sampler_ptr;
typedef std::unique_ptr<llama_adapter_lora, llama_adapter_lora_deleter> llama_adapter_lora_ptr;

struct llama_batch_ext_ptr : std::unique_ptr<llama_batch_ext, llama_batch_ext_deleter> {
    llama_batch_ext_ptr() : std::unique_ptr<llama_batch_ext, llama_batch_ext_deleter>() {}
    llama_batch_ext_ptr(llama_batch_ext * batch) : std::unique_ptr<llama_batch_ext, llama_batch_ext_deleter>(batch) {}

    // Convenience C++ wrapper to create a batch from text tokens, without worrying about manually freeing it
    // First token will be at position pos0
    // The sequence ID will be fixed to seq_id
    // If output_last is true, the last token will have output set
    static llama_batch_ext_ptr init_from_text(llama_token * tokens,
                                                  int32_t   n_tokens,
                                                llama_pos   pos0,
                                             llama_seq_id   seq_id,
                                                     bool   output_last) {
        llama_batch_ext * batch = llama_batch_ext_init(n_tokens, 1);
        for (int32_t i = 0; i < n_tokens; i++) {
            llama_batch_ext_add_text(batch, tokens[i], pos0 + i, &seq_id, 1, false);
        }
        if (output_last) {
            llama_batch_ext_set_output_last(batch);
        }
        return llama_batch_ext_ptr(batch);
    }

    // Convenience C++ wrapper to create a batch from text embeddings, without worrying about manually freeing it
    static llama_batch_ext_ptr init_from_embd(float * embd,
                                             size_t   n_tokens,
                                             size_t   n_embd,
                                          llama_pos   pos0,
                                       llama_seq_id   seq_id) {
        return llama_batch_ext_ptr(llama_batch_ext_init_from_embd(embd, n_tokens, n_embd, pos0, seq_id));
    }

    // Wrapper to add a single token to the batch, support multiple sequence IDs
    int32_t add_text(llama_token token, llama_pos pos, const std::vector<llama_seq_id> & seq_id, bool output_last) {
        int32_t output_id = llama_batch_ext_add_text(this->get(), token, pos, seq_id.data(), seq_id.size(), false);
        if (output_last) {
            llama_batch_ext_set_output_last(this->get());
        }
        return output_id;
    }

    // Wrapper to add a single token to the batch (single sequence ID)
    int32_t add_text(llama_token token, llama_pos pos, llama_seq_id seq_id, bool output_last) {
        int32_t output_id = llama_batch_ext_add_text(this->get(), token, pos, &seq_id, 1, false);
        if (output_last) {
            llama_batch_ext_set_output_last(this->get());
        }
        return output_id;
    }

    void clear() {
        llama_batch_ext_clear(this->get());
    }

    int32_t n_tokens() const {
        return llama_batch_ext_get_n_tokens(this->get());
    }
};
