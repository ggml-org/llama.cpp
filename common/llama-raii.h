// RAII wrappers for llama.cpp C API resources.
//
// Shared by tools/hs-extract, tools/hs-extract-batch, tests/test-hidden-states,
// and examples/hidden-states. All wrappers support move semantics; tools that
// don't need moves simply don't use them.
//
// Usage:
//   LlamaBackend backend;          // calls llama_backend_init()
//   LlamaModel model(...);          // owns llama_model*
//   LlamaContext ctx(...);          // owns llama_context*
//   LlamaBatch batch;               // owns llama_batch (call .init() first)
//
// All resources are automatically freed on scope exit. Copy is disabled;
// move is allowed.

#pragma once

#include "llama.h"

struct LlamaBackend {
    LlamaBackend() { llama_backend_init(); }
    ~LlamaBackend() { llama_backend_free(); }
    LlamaBackend(const LlamaBackend &) = delete;
    LlamaBackend & operator=(const LlamaBackend &) = delete;
};

struct LlamaModel {
    llama_model * model;
    LlamaModel() : model(nullptr) {}
    LlamaModel(llama_model * m) : model(m) {}
    ~LlamaModel() { if (model) llama_model_free(model); }
    LlamaModel(const LlamaModel &) = delete;
    LlamaModel & operator=(const LlamaModel &) = delete;
    LlamaModel(LlamaModel && other) noexcept : model(other.model) { other.model = nullptr; }
    LlamaModel & operator=(LlamaModel && other) noexcept {
        if (this != &other) { if (model) llama_model_free(model); model = other.model; other.model = nullptr; }
        return *this;
    }
    operator llama_model *() const { return model; }
    explicit operator bool() const { return model != nullptr; }
};

struct LlamaContext {
    llama_context * ctx;
    LlamaContext() : ctx(nullptr) {}
    LlamaContext(llama_context * c) : ctx(c) {}
    ~LlamaContext() { if (ctx) llama_free(ctx); }
    LlamaContext(const LlamaContext &) = delete;
    LlamaContext & operator=(const LlamaContext &) = delete;
    LlamaContext(LlamaContext && other) noexcept : ctx(other.ctx) { other.ctx = nullptr; }
    LlamaContext & operator=(LlamaContext && other) noexcept {
        if (this != &other) { if (ctx) llama_free(ctx); ctx = other.ctx; other.ctx = nullptr; }
        return *this;
    }
    operator llama_context *() const { return ctx; }
    explicit operator bool() const { return ctx != nullptr; }
};

struct LlamaBatch {
    llama_batch batch;
    bool initialized;
    LlamaBatch() : batch{}, initialized(false) {}
    void init(int32_t n_tokens, int32_t embd, int32_t n_seq_max) {
        batch = llama_batch_init(n_tokens, embd, n_seq_max);
        initialized = true;
    }
    ~LlamaBatch() { if (initialized) llama_batch_free(batch); }
    LlamaBatch(const LlamaBatch &) = delete;
    LlamaBatch & operator=(const LlamaBatch &) = delete;
};
