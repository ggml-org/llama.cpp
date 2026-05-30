
#include <llama.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define JSLLAMA_API __attribute__ ((visibility ("default")))

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    void* model;
    void* ctx;
    void* vocab;
    void* sampler;
    int32_t n_ctx;
    int32_t n_batch;
    int32_t n_threads;
    int32_t n_gpu_layers;
} jsllama_handle;

JSLLAMA_API void jsllama_backend_init(void) {
    llama_backend_init();
}

JSLLAMA_API void jsllama_backend_free(void) {
    llama_backend_free();
}

JSLLAMA_API jsllama_handle* jsllama_load_model(const char* path, int32_t n_ctx, int32_t n_batch, int32_t n_threads, int32_t n_gpu_layers) {
    jsllama_handle* handle = (jsllama_handle*)calloc(1, sizeof(jsllama_handle));
    if (!handle) return NULL;

    struct llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;

    handle->model = llama_model_load_from_file(path, model_params);
    if (!handle->model) {
        free(handle);
        return NULL;
    }

    handle->vocab = (void*)llama_model_get_vocab((const struct llama_model*)handle->model);

    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = n_batch;
    ctx_params.n_ubatch = n_batch < 512 ? n_batch : 512;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;

    handle->ctx = llama_init_from_model((struct llama_model*)handle->model, ctx_params);
    if (!handle->ctx) {
        llama_model_free((struct llama_model*)handle->model);
        free(handle);
        return NULL;
    }

    handle->n_ctx = n_ctx;
    handle->n_batch = n_batch;
    handle->n_threads = n_threads;
    handle->n_gpu_layers = n_gpu_layers;

    struct llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    handle->sampler = (void*)llama_sampler_chain_init(sampler_params);
    
    llama_sampler_chain_add((struct llama_sampler*)handle->sampler, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add((struct llama_sampler*)handle->sampler, llama_sampler_init_penalties(64, 1.25f, 0.1f, 0.0f));
    llama_sampler_chain_add((struct llama_sampler*)handle->sampler, llama_sampler_init_greedy());

    return handle;
}

JSLLAMA_API void jsllama_free_model(jsllama_handle* handle) {
    if (!handle) return;
    
    if (handle->sampler) {
        llama_sampler_free((struct llama_sampler*)handle->sampler);
    }
    if (handle->ctx) {
        llama_free((struct llama_context*)handle->ctx);
    }
    if (handle->model) {
        llama_model_free((struct llama_model*)handle->model);
    }
    free(handle);
}

JSLLAMA_API int32_t jsllama_tokenize(jsllama_handle* handle, const char* text, int32_t text_len, int32_t* tokens, int32_t max_tokens, int add_special, int parse_special) {
    return llama_tokenize(
        (const struct llama_vocab*)handle->vocab,
        text, text_len,
        tokens, max_tokens,
        add_special, parse_special
    );
}

JSLLAMA_API int32_t jsllama_decode(jsllama_handle* handle, const int32_t* tokens, int32_t n_tokens, int32_t n_past) {
    if (n_tokens <= 0) {
        return -1;
    }
    struct llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    
    batch.n_tokens = n_tokens;
    
    for (int32_t i = 0; i < n_tokens; i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = n_past + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == n_tokens - 1) ? 1 : 0;
    }
    
    int32_t ret = llama_decode((struct llama_context*)handle->ctx, batch);
    llama_batch_free(batch);
    
    return ret;
}

JSLLAMA_API int32_t jsllama_decode_single(jsllama_handle* handle, int32_t token, int32_t n_past) {
    struct llama_batch batch = llama_batch_init(1, 0, 1);
    
    batch.n_tokens = 1;
    batch.token[0] = token;
    batch.pos[0] = n_past;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = 1;
    
    int32_t ret = llama_decode((struct llama_context*)handle->ctx, batch);
    llama_batch_free(batch);
    
    return ret;
}

JSLLAMA_API int32_t jsllama_sample(jsllama_handle* handle) {
    return llama_sampler_sample((struct llama_sampler*)handle->sampler, (struct llama_context*)handle->ctx, -1);
}

JSLLAMA_API int jsllama_is_eog(jsllama_handle* handle, int32_t token) {
    return llama_vocab_is_eog((const struct llama_vocab*)handle->vocab, token);
}

JSLLAMA_API int32_t jsllama_token_to_piece(jsllama_handle* handle, int32_t token, char* buf, int32_t buf_len) {
    return llama_token_to_piece(
        (const struct llama_vocab*)handle->vocab,
        token,
        buf, buf_len,
        0, false
    );
}

JSLLAMA_API void jsllama_kv_cache_clear(jsllama_handle* handle) {
    llama_memory_clear(llama_get_memory((struct llama_context*)handle->ctx), true);
}

JSLLAMA_API int32_t jsllama_get_vocab_size(jsllama_handle* handle) {
    return llama_vocab_n_tokens((const struct llama_vocab*)handle->vocab);
}

JSLLAMA_API float* jsllama_get_logits(jsllama_handle* handle, int32_t idx) {
    return llama_get_logits_ith((struct llama_context*)handle->ctx, idx);
}

#ifdef __cplusplus
}
#endif
