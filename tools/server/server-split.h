#pragma once

#include "common.h"
#include "speculative.h"
#include <map>

struct server_split {
    virtual ~server_split() = default;
    virtual llama_model * model() const = 0;
    virtual llama_context * context() const = 0;
    virtual llama_context * draft_context() const = 0;
    virtual common_speculative * speculative() const = 0;
    virtual void prepare(bool prefill, size_t remaining) = 0;
    virtual void external_begin() = 0;
    virtual void external_end() = 0;
    virtual void observe(size_t tokens, double ms) = 0;
};

std::unique_ptr<server_split> server_split_init(common_params & params, const std::map<ggml_backend_dev_t, size_t> & reserved);

struct server_split_external {
    server_split * split;
    explicit server_split_external(server_split * split) : split(split) { if (split) split->external_begin(); }
    ~server_split_external() { if (split) split->external_end(); }
    server_split_external(const server_split_external &) = delete;
    server_split_external & operator=(const server_split_external &) = delete;
};
