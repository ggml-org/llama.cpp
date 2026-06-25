#pragma once

#include "server-common.h"
#include "server-task.h"

#include "sampling.h"
#include "speculative.h"

#include <climits>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace server_schema {

struct field_eval_context {
    task_params & params;
    const llama_vocab * vocab = nullptr;
    const std::vector<llama_logit_bias> * logit_bias_eog = nullptr;
    field_eval_context(task_params & params) : params(params) {}
};

using field_handler = std::function<void(field_eval_context &, const json &)>;

struct field {
    std::vector<const char *> name;
    const char * desc = "";
    field_handler custom_handler;
    field() = default;
    field(const char * n) : name({n}) {}
    virtual ~field() = default;
    field * set_desc(const char * s) {
        desc = s;
        return this;
    }
    // if 'name' is present, use it, otherwise look for aliases following the order they were added
    field * add_alias(const char * n) {
        name.push_back(n);
        return this;
    }
    field * set_handler(field_handler h) { this->custom_handler = h; return this; }
    virtual void eval(field_eval_context & ctx, const json & data) = 0;
};

template <typename T = int32_t>
struct field_num : public field {
    T & val;
    T min = std::numeric_limits<T>::lowest();
    T max = std::numeric_limits<T>::max();
    bool is_hard_limit = false; // if true, throw error if the value is invalid
    field_num(const char * n, T & val) : field(n), val(val) {}
    // limits are inclusive, min <= value <= max
    field_num * set_limits(T min, T max) {
        this->min = min;
        this->max = max;
        return this;
    }
    field_num * set_hard_limits(T min, T max) {
        set_limits(min, max);
        is_hard_limit = true;
        return this;
    }
    virtual void eval(field_eval_context & ctx, const json & data) override;
};

struct field_str : public field {
    field_str(const char * n) : field(n) {}
    virtual void eval(field_eval_context & ctx, const json & data) override;
};

struct field_bool : public field {
    bool & val;
    field_bool(const char * n, bool & val) : field(n), val(val) {}
    virtual void eval(field_eval_context & ctx, const json & data) override;
};

struct field_json : public field {
    field_json(const char * n) : field(n) {}
    virtual void eval(field_eval_context & ctx, const json & data) override;
};

struct field_nested : public field {
    std::vector<std::unique_ptr<field>> subfields;
    field_nested(const char * n) : field(n) {}
    field_nested * add_subfield(field * f) {
        subfields.emplace_back(std::unique_ptr<field>(f));
        return this;
    }
    virtual void eval(field_eval_context & ctx, const json & data) override;
};

std::vector<std::unique_ptr<field>> make_llama_cmpl_schema(
                    const common_params & params_base,
                    task_params & params);

task_params eval_llama_cmpl_schema(
                    const llama_vocab * vocab,
                    const common_params & params_base,
                    const int n_ctx_slot,
                    const std::vector<llama_logit_bias> & logit_bias_eog,
                    const json & data);

} // namespace server_schema
