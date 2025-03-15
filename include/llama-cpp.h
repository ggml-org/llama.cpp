#pragma once

#ifndef __cplusplus
#error "This header is for C++ only"
#endif

#include <memory>

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

    // convenience function to create a batch from text tokens, without worrying about manually freeing it
    static llama_batch_ext_ptr init_from_text(llama_token * tokens,
                                             int32_t   n_tokens,
                                             int32_t   pos0,
                                             int32_t   seq_id,
                                                bool   output_last) {
        return llama_batch_ext_ptr(llama_batch_ext_init_from_text(tokens, n_tokens, pos0, seq_id, output_last));
    }

    // convenience function to create a batch from text embeddings, without worrying about manually freeing it
    static llama_batch_ext_ptr init_from_embd(float * embd,
                                        size_t   n_tokens,
                                        size_t   n_embd,
                                       int32_t   pos0,
                                       int32_t   seq_id) {
        return llama_batch_ext_ptr(llama_batch_ext_init_from_embd(embd, n_tokens, n_embd, pos0, seq_id));
    }
};
