#pragma once

#include "chat.h"
#include "chat-auto-parser.h"
#include "chat-auto-parser-helpers.h"
#include "chat-peg-parser.h"
#include "common.h"
#include "ggml.h"
#include "json-schema-to-grammar.h"
#include "json.h"

#include <functional>
#include <optional>
#include <set>
#include <string>
#include <vector>

using json = common_json;

// iterate over the function tools of an OpenAI-style tools array
void foreach_function(const json & tools, const std::function<void(const json &)> & fn);

// iterate over the parameters of a function tool, flagging the ones listed as required
void foreach_parameter(const json & function, const std::function<void(const std::string &, const json &, bool)> & fn);

// render a template; the override arguments let a parser feed in messages, tools or context it has rewritten
std::string common_chat_template_direct_apply_impl(
    const common_chat_template & tmpl,
    const autoparser::generation_params & inputs,
    const std::optional<json> & messages_override = std::nullopt,
    const std::optional<json> & tools_override = std::nullopt,
    const std::optional<json> & additional_context = std::nullopt);

// the suffix a template appends when add_generation_prompt is set
std::string common_chat_template_generation_prompt_impl(
    const common_chat_template & tmpl,
    const autoparser::generation_params & inputs,
    const std::optional<json> & messages_override = std::nullopt,
    const std::optional<json> & tools_override = std::nullopt,
    const std::optional<json> & additional_context = std::nullopt);

bool is_lfm2_template(const std::string & src);

namespace workaround {

void convert_tool_responses_gemma4(json & messages);

}

common_chat_params common_chat_params_init_cohere2moe(const common_chat_template & tmpl, const autoparser::generation_params & inputs);

common_chat_params common_chat_params_init_deepseek_v3_2(const common_chat_template & tmpl, const autoparser::generation_params & inputs);

common_chat_params common_chat_params_init_functionary_v3_2(const common_chat_template & tmpl, const autoparser::generation_params & inputs);

common_chat_params common_chat_params_init_gemma4(const common_chat_template & tmpl, const autoparser::generation_params & inputs);

common_chat_params common_chat_params_init_gigachat_v3(const common_chat_template & tmpl, const autoparser::generation_params & inputs);

common_chat_params common_chat_params_init_gpt_oss(const common_chat_template & tmpl, const autoparser::generation_params & inputs);

common_chat_params common_chat_params_init_kimi_k2(const common_chat_template & tmpl, const autoparser::generation_params & inputs);

common_chat_params common_chat_params_init_kimi_k3(const common_chat_template & tmpl, const autoparser::generation_params & inputs);

// tool_list_tokens preserves the LFM2 system tool-list markers; LFM2.5 renders without them
common_chat_params common_chat_params_init_lfm2(const common_chat_template & tmpl, const autoparser::generation_params & inputs, bool tool_list_tokens);

common_chat_params common_chat_params_init_minicpm5(const common_chat_template & tmpl, const autoparser::generation_params & inputs);

common_chat_params common_chat_params_init_minimax_m3(const common_chat_template & tmpl, const autoparser::generation_params & inputs);

common_chat_params common_chat_params_init_ministral_3(const common_chat_template & tmpl, const autoparser::generation_params & inputs);

common_chat_params common_chat_params_init_muse_glimmer(const common_chat_template & tmpl, const autoparser::generation_params & inputs);

common_chat_params common_chat_params_init_qwen3_coder(const common_chat_template & tmpl, const autoparser::generation_params & inputs);
