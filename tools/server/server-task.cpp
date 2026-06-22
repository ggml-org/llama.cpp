#include "server-task.h"

#include "build-info.h"
#include "server-chat.h"
#include "chat.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"
#include "server-common.h"

#include <algorithm>

using json = nlohmann::ordered_json;

//
// task_params
//

json task_params::format_logit_bias(const std::vector<llama_logit_bias> & logit_bias) const {
    json data = json::array();
    for (const auto & lb : logit_bias) {
        data.push_back(json{
            {"bias", lb.bias},
            {"token", lb.token},
        });
    }
    return data;
}

json task_params::to_json(bool only_metrics) const {
    std::vector<std::string> samplers;
    samplers.reserve(sampling.samplers.size());
    for (const auto & sampler : sampling.samplers) {
        samplers.emplace_back(common_sampler_type_to_str(sampler));
    }

    json lora = json::array();
    for (auto & it : this->lora) {
        lora.push_back({{"id", it.first}, {"scale", it.second}});
    }

    if (only_metrics) {
        return json {
            {"seed",                      sampling.seed},
            {"temperature",               sampling.temp},
            {"dynatemp_range",            sampling.dynatemp_range},
            {"dynatemp_exponent",         sampling.dynatemp_exponent},
            {"top_k",                     sampling.top_k},
            {"top_p",                     sampling.top_p},
            {"min_p",                     sampling.min_p},
            {"top_n_sigma",               sampling.top_n_sigma},
            {"xtc_probability",           sampling.xtc_probability},
            {"xtc_threshold",             sampling.xtc_threshold},
            {"typical_p",                 sampling.typ_p},
            {"repeat_last_n",             sampling.penalty_last_n},
            {"repeat_penalty",            sampling.penalty_repeat},
            {"presence_penalty",          sampling.penalty_present},
            {"frequency_penalty",         sampling.penalty_freq},
            {"dry_multiplier",            sampling.dry_multiplier},
            {"dry_base",                  sampling.dry_base},
            {"dry_allowed_length",        sampling.dry_allowed_length},
            {"dry_penalty_last_n",        sampling.dry_penalty_last_n},
            {"mirostat",                  sampling.mirostat},
            {"mirostat_tau",              sampling.mirostat_tau},
            {"mirostat_eta",              sampling.mirostat_eta},
            {"max_tokens",                n_predict},
            {"n_predict",                 n_predict}, // TODO: deduplicate?
            {"n_keep",                    n_keep},
            {"n_discard",                 n_discard},
            {"ignore_eos",                sampling.ignore_eos},
            {"stream",                    stream},
            {"n_probs",                   sampling.n_probs},
            {"min_keep",                  sampling.min_keep},
            {"chat_format",               common_chat_format_name(chat_parser_params.format)},
            {"reasoning_format",          common_reasoning_format_name(chat_parser_params.reasoning_format)},
            {"reasoning_in_content",      chat_parser_params.reasoning_in_content},
            {"generation_prompt",         chat_parser_params.generation_prompt},
            {"samplers",                  samplers},
            {"speculative.types",         common_speculative_type_name_str(speculative.types)},
            {"timings_per_token",         timings_per_token},
            {"post_sampling_probs",       post_sampling_probs},
            {"backend_sampling",          sampling.backend_sampling},
            {"lora",                      lora},
        };
    }

    auto grammar_triggers = json::array();
    for (const auto & trigger : sampling.grammar_triggers) {
        server_grammar_trigger ct(trigger);
        grammar_triggers.push_back(ct.to_json());
    }

    return json {
        {"seed",                      sampling.seed},
        {"temperature",               sampling.temp},
        {"dynatemp_range",            sampling.dynatemp_range},
        {"dynatemp_exponent",         sampling.dynatemp_exponent},
        {"top_k",                     sampling.top_k},
        {"top_p",                     sampling.top_p},
        {"min_p",                     sampling.min_p},
        {"top_n_sigma",               sampling.top_n_sigma},
        {"xtc_probability",           sampling.xtc_probability},
        {"xtc_threshold",             sampling.xtc_threshold},
        {"typical_p",                 sampling.typ_p},
        {"repeat_last_n",             sampling.penalty_last_n},
        {"repeat_penalty",            sampling.penalty_repeat},
        {"presence_penalty",          sampling.penalty_present},
        {"frequency_penalty",         sampling.penalty_freq},
        {"dry_multiplier",            sampling.dry_multiplier},
        {"dry_base",                  sampling.dry_base},
        {"dry_allowed_length",        sampling.dry_allowed_length},
        {"dry_penalty_last_n",        sampling.dry_penalty_last_n},
        {"dry_sequence_breakers",     sampling.dry_sequence_breakers},
        {"mirostat",                  sampling.mirostat},
        {"mirostat_tau",              sampling.mirostat_tau},
        {"mirostat_eta",              sampling.mirostat_eta},
        {"stop",                      antiprompt},
        {"max_tokens",                n_predict},
        {"n_predict",                 n_predict}, // TODO: deduplicate?
        {"n_keep",                    n_keep},
        {"n_discard",                 n_discard},
        {"ignore_eos",                sampling.ignore_eos},
        {"stream",                    stream},
        {"logit_bias",                format_logit_bias(sampling.logit_bias)},
        {"n_probs",                   sampling.n_probs},
        {"min_keep",                  sampling.min_keep},
        {"grammar",                   common_grammar_value(sampling.grammar)},
        {"grammar_lazy",              sampling.grammar_lazy},
        {"grammar_triggers",          grammar_triggers},
        {"preserved_tokens",          sampling.preserved_tokens},
        {"chat_format",               common_chat_format_name(chat_parser_params.format)},
        {"reasoning_format",          common_reasoning_format_name(chat_parser_params.reasoning_format)},
        {"reasoning_in_content",      chat_parser_params.reasoning_in_content},
        {"generation_prompt",         chat_parser_params.generation_prompt},
        {"samplers",                  samplers},
        {"speculative.types",         common_speculative_type_name_str(speculative.types)},
        {"timings_per_token",         timings_per_token},
        {"post_sampling_probs",       post_sampling_probs},
        {"backend_sampling",          sampling.backend_sampling},
        {"lora",                      lora},
    };
}

//
// task_result_state
//
task_result_state::task_result_state(const common_chat_parser_params & chat_parser_params)
    : chat_parser_params(chat_parser_params)
    , oai_resp_id("resp_" + random_string())
    , oai_resp_reasoning_id("rs_" + random_string())
    , oai_resp_message_id("msg_" + random_string()) {
    if (chat_parser_params.is_continuation && !chat_parser_params.echo) {
        // initialize chat_msg to avoid emitting a delta containing the assistant prefill
        chat_msg = common_chat_parse("", true, chat_parser_params);
    }
}

common_chat_msg task_result_state::update_chat_msg(
        const std::string & text_added,
        bool is_partial,
        std::vector<common_chat_msg_diff> & diffs,
        bool filter_tool_calls) {
    generated_text += text_added;
    auto msg_prv_copy = chat_msg;
    //SRV_DBG("Parsing chat message: %s\n", generated_text.c_str());
    auto new_msg = common_chat_parse(
        generated_text,
        is_partial,
        chat_parser_params);
    if (!new_msg.empty()) {
        new_msg.set_tool_call_ids(generated_tool_call_ids, gen_tool_call_id);
        chat_msg = new_msg;
        auto all_diffs = common_chat_msg_diff::compute_diffs(msg_prv_copy, chat_msg);

        if (!filter_tool_calls) {
            diffs = std::move(all_diffs);
        } else {
            for (auto & d : all_diffs) {
                // If this is a new type of delta, flush all currently pending tool call names
                for (size_t i = 0; i < chat_msg.tool_calls.size(); ++i) {
                    if (sent_tool_call_names.count(i) || chat_msg.tool_calls[i].name.empty()) {
                        continue;
                    }
                    if (d.tool_call_index != i || !d.tool_call_delta.arguments.empty()) {
                        common_chat_msg_diff header;
                        header.tool_call_index      = i;
                        header.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                        header.tool_call_delta.name = chat_msg.tool_calls[i].name;
                        diffs.push_back(std::move(header));
                        sent_tool_call_names.insert(i);
                    }
                }

                if (d.tool_call_index == std::string::npos) {
                    diffs.push_back(std::move(d));
                } else {
                    size_t i = d.tool_call_index;
                    if (sent_tool_call_names.count(i)) {
                        if (!d.tool_call_delta.arguments.empty()) {
                            d.tool_call_delta.name = "";
                            d.tool_call_delta.id   = "";
                            diffs.push_back(std::move(d));
                        }
                    } else {
                        // Not sent yet.
                        if (!d.tool_call_delta.arguments.empty() || !is_partial) {
                            d.tool_call_delta.name = chat_msg.tool_calls[i].name;
                            d.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                            diffs.push_back(std::move(d));
                            sent_tool_call_names.insert(i);
                        } else {
                            // Suppress
                        }
                    }
                }
            }
            // Final check at EOF
            if (!is_partial) {
                for (size_t i = 0; i < chat_msg.tool_calls.size(); ++i) {
                    if (!sent_tool_call_names.count(i) && !chat_msg.tool_calls[i].name.empty()) {
                        common_chat_msg_diff header;
                        header.tool_call_index      = i;
                        header.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                        header.tool_call_delta.name = chat_msg.tool_calls[i].name;
                        diffs.push_back(std::move(header));
                        sent_tool_call_names.insert(i);
                    }
                }
            }
        }
    }
    return chat_msg;
}

//

// result_timings
//

json result_timings::to_json() const {
    json base = {
        {"cache_n",                cache_n},

        {"prompt_n",               prompt_n},
        {"prompt_ms",              prompt_ms},
        {"prompt_per_token_ms",    prompt_per_token_ms},
        {"prompt_per_second",      prompt_per_second},

        {"predicted_n",            predicted_n},
        {"predicted_ms",           predicted_ms},
        {"predicted_per_token_ms", predicted_per_token_ms},
        {"predicted_per_second",   predicted_per_second},
    };

    if (draft_n > 0) {
        base["draft_n"] = draft_n;
        base["draft_n_accepted"] = draft_n_accepted;
    }

    return base;
}

//
// result_prompt_progress
//
json result_prompt_progress::to_json() const {
    return json {
        {"total",     total},
        {"cache",     cache},
        {"processed", processed},
        {"time_ms",   time_ms},
    };
}

static inline std::string stop_type_to_str(stop_type type) {
    switch (type) {
        case STOP_TYPE_EOS:   return "eos";
        case STOP_TYPE_WORD:  return "word";
        case STOP_TYPE_LIMIT: return "limit";
        default:              return "none";
    }
}

//
// completion_token_output
//

json completion_token_output::to_json(bool post_sampling_probs) const {
    json probs_for_token = json::array();
    for (const auto & p : probs) {
        std::string txt(p.txt);
        txt.resize(validate_utf8(txt));
        probs_for_token.push_back(json {
            {"id",      p.tok},
            {"token",   txt},
            {"bytes",   str_to_bytes(p.txt)},
            {
                post_sampling_probs ? "prob" : "logprob",
                post_sampling_probs ? p.prob : logarithm(p.prob)
            },
        });
    }
    return probs_for_token;
}

json completion_token_output::probs_vector_to_json(const std::vector<completion_token_output> & probs, bool post_sampling_probs) {
    json out = json::array();
    for (const auto & p : probs) {
        std::string txt(p.text_to_send);
        txt.resize(validate_utf8(txt));
        out.push_back(json {
            {"id",           p.tok},
            {"token",        txt},
            {"bytes",        str_to_bytes(p.text_to_send)},
            {
                post_sampling_probs ? "prob" : "logprob",
                post_sampling_probs ? p.prob : logarithm(p.prob)
            },
            {
                post_sampling_probs ? "top_probs" : "top_logprobs",
                p.to_json(post_sampling_probs)
            },
        });
    }
    return out;
}

float completion_token_output::logarithm(float x) {
    // nlohmann::json converts -inf to null, so we need to prevent that
    return x == 0.0f ? std::numeric_limits<float>::lowest() : std::log(x);
}

std::vector<unsigned char> completion_token_output::str_to_bytes(const std::string & str) {
    std::vector<unsigned char> bytes;
    for (unsigned char c : str) {
        bytes.push_back(c);
    }
    return bytes;
}

//
// server_task_result_cmpl_final
//
json server_task_result_cmpl_final::to_json() {
    GGML_ASSERT(is_updated && "update() must be called before to_json()");
    switch (res_type) {
        case TASK_RESPONSE_TYPE_NONE:
            return to_json_non_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CMPL:
            return to_json_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            return stream ? to_json_oaicompat_chat_stream() : to_json_oaicompat_chat();
        case TASK_RESPONSE_TYPE_OAI_RESP:
            return stream ? to_json_oaicompat_resp_stream() : to_json_oaicompat_resp();
        case TASK_RESPONSE_TYPE_OAI_ASR:
            return to_json_oaicompat_asr();
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            return stream ? to_json_anthropic_stream() : to_json_anthropic();
        default:
            GGML_ASSERT(false && "Invalid task_response_type");
    }
}

json server_task_result_cmpl_final::to_json_non_oaicompat() {
    json res = json {
        {"index",               index},
        {"content",             content},
        {"tokens",              tokens},
        {"id_slot",             id_slot},
        {"stop",                true},
        {"model",               oaicompat_model},
        {"tokens_predicted",    n_decoded},
        {"tokens_evaluated",    n_prompt_tokens},
        {"generation_settings", generation_params.to_json()},
        {"prompt",              prompt},
        {"has_new_line",        has_new_line},
        {"truncated",           truncated},
        {"stop_type",           stop_type_to_str(stop)},
        {"stopping_word",       stopping_word},
        {"tokens_cached",       n_tokens_cached},
        {"timings",             timings.to_json()},
    };
    if (!stream && !probs_output.empty()) {
        res["completion_probabilities"] = completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs);
    }
    return response_fields.empty() ? res : json_get_nested_values(response_fields, res);
}

json server_task_result_cmpl_final::usage_json_oaicompat() {
    return json {
        {"completion_tokens", n_decoded},
        {"prompt_tokens",     n_prompt_tokens},
        {"total_tokens",      n_decoded + n_prompt_tokens},
        {"prompt_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
    };
}

json server_task_result_cmpl_final::to_json_oaicompat() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (!stream && probs_output.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }
    json finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = "stop";
    }
    json res = json {
        {"choices",            json::array({
            json{
                {"text",          content},
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", finish_reason},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "text_completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (timings.prompt_n >= 0) {
        res.push_back({"timings", timings.to_json()});
    }

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat() {
    std::string finish_reason = "length";
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content;
    }
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = msg.tool_calls.empty() ? "stop" : "tool_calls";
    }

    json choice {
        {"finish_reason", finish_reason},
        {"index", index},
        {"message", msg.to_json_oaicompat()},
    };

    if (!stream && probs_output.size() > 0) {
        choice["logprobs"] = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }

    std::time_t t = std::time(0);

    json res = json {
        {"choices",            json::array({choice})},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "chat.completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (timings.prompt_n >= 0) {
        res.push_back({"timings", timings.to_json()});
    }

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat_stream() {
    std::time_t t = std::time(0);
    std::string finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = oaicompat_msg.tool_calls.empty() ? "stop" : "tool_calls";
    }

    json deltas = json::array();
    for (const auto & diff : oaicompat_msg_diffs) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", index},
                    {"delta", server_chat_msg_diff_to_json_oaicompat(diff)},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object", "chat.completion.chunk"},
        });
    }

    deltas.push_back({
        {"choices", json::array({
            json {
                {"finish_reason", finish_reason},
                {"index", index},
                {"delta", json::object()},
            },
        })},
        {"created",            t},
        {"id",                 oaicompat_cmpl_id},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "chat.completion.chunk"},
    });

    if (include_usage) {
        // OpenAI API spec for chat.completion.chunks specifies an empty `choices` array for the last chunk when including usage
        // https://platform.openai.com/docs/api-reference/chat_streaming/streaming#chat_streaming/streaming-choices
        deltas.push_back({
            {"choices", json::array()},
            {"created",            t},
            {"id",                 oaicompat_cmpl_id},
            {"model",              oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object",             "chat.completion.chunk"},
            {"usage",              usage_json_oaicompat()},
        });
    }

    if (timings.prompt_n >= 0) {
        deltas.back().push_back({"timings", timings.to_json()});
    }

    // extra fields for debugging purposes
    if (verbose && !deltas.empty()) {
        deltas.front()["__verbose"] = to_json_non_oaicompat();
    }

    return deltas;
}

static std::string build_output_text(const std::vector<json> & output) {
    std::string result;
    for (const auto & item : output) {
        if (json_value(item, "type", std::string()) == "message") {
            for (const auto & part : item.at("content")) {
                if (json_value(part, "type", std::string()) == "output_text") {
                    result += part.at("text").get<std::string>();
                }
            }
        }
    }
    return result;
}

static json build_oai_resp_metadata(const std::string & oai_resp_id,
                                    const std::string & oaicompat_model,
                                    const std::vector<json> & output,
                                    const std::string & output_text,
                                    int n_prompt_tokens,
                                    int n_decoded,
                                    int n_prompt_tokens_cache,
                                    const std::string & status = "completed") {
    std::time_t t = std::time(0);
    return json {
        {"completed_at",         status == "completed" ? json(t) : json(nullptr)},
        {"created_at",           t},
        {"id",                   oai_resp_id},
        {"model",                oaicompat_model},
        {"object",               "response"},
        {"output",               output},
        {"output_text",          output_text},
        {"status",               status},
        {"usage",                json {
            {"input_tokens",          n_prompt_tokens},
            {"output_tokens",         n_decoded},
            {"total_tokens",          n_decoded + n_prompt_tokens},
            {"input_tokens_details",  json{{"cached_tokens", n_prompt_tokens_cache}}},
            {"output_tokens_details", json{{"reasoning_tokens", 0}}},
        }},
        {"incomplete_details",   nullptr},
        {"previous_response_id", nullptr},
        {"instructions",         nullptr},
        {"error",                nullptr},
        {"tools",                json::array()},
        {"tool_choice",          "auto"},
        {"truncation",           "disabled"},
        {"parallel_tool_calls",  false},
        {"text",                 json{{"format", json{{"type", "text"}}}}},
        {"top_p",                1.0},
        {"presence_penalty",     0.0},
        {"frequency_penalty",    0.0},
        {"top_logprobs",         0},
        {"temperature",          1.0},
        {"reasoning",            nullptr},
        {"max_output_tokens",    nullptr},
        {"max_tool_calls",       nullptr},
        {"store",                false},
        {"background",           false},
        {"service_tier",         "default"},
        {"safety_identifier",    nullptr},
        {"prompt_cache_key",     nullptr},
        {"metadata",             json::object()},
    };
}

static json parse_tool_arguments(const std::string & arguments) {
    if (arguments.empty()) {
        return json::object();
    }
    try {
        return json::parse(arguments);
    } catch (const std::exception &) {
        return json::object();
    }
}

static std::string get_responses_tool_type(
        const std::unordered_map<std::string, json> & responses_tool_metadata,
        const std::string & tool_name) {
    const auto it = responses_tool_metadata.find(tool_name);
    if (it == responses_tool_metadata.end()) {
        return "function";
    }
    return json_value(it->second, "type", std::string("function"));
}

static bool get_custom_tool_input_value_if_present(const json & parsed_args, std::string & input) {
    if (parsed_args.contains("input")) {
        input = parsed_args.at("input").is_string()
            ? parsed_args.at("input").get<std::string>()
            : parsed_args.at("input").dump();
        return true;
    }
    if (parsed_args.contains("patch")) {
        input = parsed_args.at("patch").is_string()
            ? parsed_args.at("patch").get<std::string>()
            : parsed_args.at("patch").dump();
        return true;
    }
    return false;
}

static std::string get_custom_tool_input_value(
        const json & parsed_args,
        const std::string & raw_arguments) {
    std::string input;
    if (get_custom_tool_input_value_if_present(parsed_args, input)) {
        return input;
    }
    return raw_arguments;
}

static std::string normalize_apply_patch_hunk_header(const std::string & line) {
    if (line.rfind("@@ -", 0) != 0) {
        return line;
    }

    const size_t new_range = line.find(" +", 4);
    if (new_range == std::string::npos) {
        return line;
    }

    const size_t range_end = line.find(" @@", new_range + 2);
    if (range_end == std::string::npos) {
        return line;
    }

    std::string context = line.substr(range_end + 3);
    while (!context.empty() && context.front() == ' ') {
        context.erase(context.begin());
    }
    return context.empty() ? "@@" : "@@ " + context;
}

static std::string normalize_apply_patch_input(const std::string & input) {
    std::string output;
    output.reserve(input.size());

    size_t pos = 0;
    while (pos < input.size()) {
        const size_t line_end = input.find('\n', pos);
        const size_t end = line_end == std::string::npos ? input.size() : line_end;
        std::string line = input.substr(pos, end - pos);
        bool has_cr = false;
        if (!line.empty() && line.back() == '\r') {
            has_cr = true;
            line.pop_back();
        }

        output += normalize_apply_patch_hunk_header(line);
        if (has_cr) {
            output += '\r';
        }
        if (line_end != std::string::npos) {
            output += '\n';
        }
        pos = line_end == std::string::npos ? input.size() : line_end + 1;
    }

    return output;
}

static std::string get_custom_tool_input_value(
        const std::string & tool_name,
        const json & parsed_args,
        const std::string & raw_arguments) {
    std::string input = get_custom_tool_input_value(parsed_args, raw_arguments);
    if (tool_name == "apply_patch") {
        input = normalize_apply_patch_input(input);
    }
    return input;
}

static std::string best_effort_custom_tool_delta(
        const std::string & accumulated_arguments,
        const std::string & previous_input) {
    const json parsed_args = parse_tool_arguments(accumulated_arguments);
    std::string current_input;
    if (!get_custom_tool_input_value_if_present(parsed_args, current_input)) {
        return std::string();
    }
    if (current_input.size() < previous_input.size()) {
        return std::string();
    }
    return current_input.substr(previous_input.size());
}

static json build_local_shell_action(const json & args, const std::string & raw_arguments) {
    if (args.contains("action") && args.at("action").is_object()) {
        return args.at("action");
    }

    json action = {
        {"type", "exec"},
        {"command", json::array()},
    };

    if (args.contains("command")) {
        const json & command = args.at("command");
        if (command.is_array()) {
            action["command"] = command;
        } else if (command.is_string()) {
            action["command"] = json::array({"bash", "-lc", command.get<std::string>()});
        }
    } else if (args.contains("cmd") && args.at("cmd").is_string()) {
        action["command"] = json::array({"bash", "-lc", args.at("cmd").get<std::string>()});
    } else if (!raw_arguments.empty()) {
        action["command"] = json::array({"bash", "-lc", raw_arguments});
    }

    for (const char * key : {"timeout_ms", "working_directory", "env", "user"}) {
        if (args.contains(key)) {
            action[key] = args.at(key);
        }
    }

    return action;
}

static std::string first_web_search_query(const json & action) {
    for (const char * key : {"query", "input", "q", "search_query"}) {
        if (action.contains(key) && action.at(key).is_string()) {
            return action.at(key).get<std::string>();
        }
    }
    if (action.contains("queries") && action.at("queries").is_array()) {
        for (const auto & query : action.at("queries")) {
            if (query.is_string() && !query.get<std::string>().empty()) {
                return query.get<std::string>();
            }
            if (query.is_object() && query.contains("q") && query.at("q").is_string()) {
                return query.at("q").get<std::string>();
            }
        }
    }
    if (action.contains("search_query") && action.at("search_query").is_array()) {
        return first_web_search_query(json{{"queries", action.at("search_query")}});
    }
    if (action.contains("url") && action.at("url").is_string()) {
        std::string query = action.at("url").get<std::string>();
        if (action.contains("pattern") && action.at("pattern").is_string()) {
            query += " " + action.at("pattern").get<std::string>();
        }
        return query;
    }
    return "";
}

static json build_web_search_local_shell_call_item(
        const common_chat_tool_call & tool_call,
        const std::string & status,
        const json & action,
        const std::string & wrapper) {
    if (wrapper.empty() || wrapper.find_first_of(" \t\r\n") != std::string::npos) {
        SRV_WRN("%s", "Ignoring X-Llama-Responses-Web-Search-Wrapper: expected a command name or path without arguments\n");
        return nullptr;
    }

    if (status == "in_progress") {
        return json {
            {"type",    "local_shell_call"},
            {"status",  status},
            {"call_id", tool_call.id},
            {"action",  json{{"type", "exec"}, {"command", json::array()}}},
        };
    }

    const std::string action_type = json_value(action, "type", std::string("search"));
    json command = json::array({wrapper});
    if (action_type == "open_page" || action_type == "find_in_page") {
        const std::string url = json_value(action, "url", std::string());
        if (url.empty()) {
            SRV_WRN("%s", "Ignoring Responses web_search shell bridge call: missing url\n");
            return nullptr;
        }
        command.push_back("extract");
        command.push_back(url);
        if (action_type == "find_in_page" && action.contains("pattern") && action.at("pattern").is_string()) {
            command.push_back("--query");
            command.push_back(action.at("pattern").get<std::string>());
        }
    } else {
        const std::string query = first_web_search_query(action);
        if (query.empty()) {
            SRV_WRN("%s", "Ignoring Responses web_search shell bridge call: missing query\n");
            return nullptr;
        }
        command.push_back("search");
        command.push_back(query);
    }
    command.push_back("--json");

    const json shell_action = json {
        {"type", "exec"},
        {"command", command},
        {"timeout_ms", 60000},
    };
    return json {
        {"type",    "local_shell_call"},
        {"status",  status},
        {"call_id", tool_call.id},
        {"action",  shell_action},
    };
}

static std::string file_search_query_from_args(const json & args) {
    for (const char * key : {"query", "pattern", "filename", "name"}) {
        if (args.contains(key) && args.at(key).is_string()) {
            const std::string value = args.at(key).get<std::string>();
            if (!value.empty()) {
                return value;
            }
        }
    }
    return "";
}

static json build_file_search_local_shell_call_item(
        const common_chat_tool_call & tool_call,
        const std::string & status,
        const json & args,
        const std::string & wrapper) {
    if (wrapper.empty() || wrapper.find_first_of(" \t\r\n") != std::string::npos) {
        SRV_WRN("%s", "Ignoring X-Llama-Responses-File-Search-Wrapper: expected a command name or path without arguments\n");
        return nullptr;
    }

    if (status == "in_progress") {
        return json {
            {"type",    "local_shell_call"},
            {"status",  status},
            {"call_id", tool_call.id},
            {"action",  json{{"type", "exec"}, {"command", json::array()}}},
        };
    }

    const std::string query = file_search_query_from_args(args);
    if (query.empty()) {
        SRV_WRN("%s", "Ignoring Responses file_search shell bridge call: missing query\n");
        return nullptr;
    }

    std::string path = json_value(args, "path", std::string("."));
    if (path.empty() || path[0] == '/' || path.find("..") != std::string::npos) {
        path = ".";
    }

    const std::string mode = json_value(args, "mode",
        json_value(args, "search_type", std::string("content")));
    const bool files_mode = mode == "files" || mode == "file" || mode == "filename" || mode == "path";

    json command = json::array({wrapper});
    command.push_back("--hidden");
    command.push_back("--glob");
    command.push_back("!**/.git/**");
    command.push_back("--glob");
    command.push_back("!**/node_modules/**");
    command.push_back("--glob");
    command.push_back("!**/build/**");
    command.push_back("--glob");
    command.push_back("!**/dist/**");

    if (files_mode) {
        command.push_back("--files");
        command.push_back("--glob");
        command.push_back("*" + query + "*");
        command.push_back(path);
    } else {
        command.push_back("-n");
        command.push_back("--max-columns");
        command.push_back("240");
        command.push_back("--max-columns-preview");
        command.push_back("--max-filesize");
        command.push_back("1M");
        command.push_back("--");
        command.push_back(query);
        command.push_back(path);
    }

    const json shell_action = json {
        {"type", "exec"},
        {"command", command},
        {"timeout_ms", 30000},
    };
    return json {
        {"type",    "local_shell_call"},
        {"status",  status},
        {"call_id", tool_call.id},
        {"action",  shell_action},
    };
}

static json server_build_responses_tool_output_item(
        const common_chat_tool_call & tool_call,
        const std::unordered_map<std::string, json> & responses_tool_metadata,
        const std::string & status,
        const std::string & item_id,
        const std::string & responses_web_search_wrapper,
        const std::string & responses_file_search_wrapper) {
    const auto it = responses_tool_metadata.find(tool_call.name);
    const json parsed_args = parse_tool_arguments(tool_call.arguments);

    const json & meta = it != responses_tool_metadata.end() ? it->second : json::object();
    const std::string tool_type = get_responses_tool_type(responses_tool_metadata, tool_call.name);
    const std::string tool_name = json_value(meta, "name", tool_call.name);

    if (tool_type == "custom") {
        return json {
            {"type",    "custom_tool_call"},
            {"status",  status},
            {"call_id", tool_call.id},
            {"name",    tool_name},
            {"input",   get_custom_tool_input_value(tool_name, parsed_args, tool_call.arguments)},
        };
    }

    if (tool_type == "local_shell") {
        return json {
            {"type",    "local_shell_call"},
            {"status",  status},
            {"call_id", tool_call.id},
            {"action",  build_local_shell_action(parsed_args, tool_call.arguments)},
        };
    }

    if (tool_type == "tool_search") {
        json arguments = parsed_args.contains("arguments") ? parsed_args.at("arguments") : parsed_args;
        return json {
            {"type",      "tool_search_call"},
            {"status",    status},
            {"call_id",   tool_call.id},
            {"execution", json_value(meta, "execution", json_value(parsed_args, "execution", std::string("client")))},
            {"arguments", arguments},
        };
    }

    if (tool_type == "web_search") {
        json action = json::object();
        if (parsed_args.contains("action") && parsed_args.at("action").is_object()) {
            action = parsed_args.at("action");
        } else if (parsed_args.contains("query") || parsed_args.contains("queries") ||
                   parsed_args.contains("input") || parsed_args.contains("q") || parsed_args.contains("search_query")) {
            action = json{{"type", "search"}};
            if (parsed_args.contains("queries")) {
                action["queries"] = parsed_args.at("queries");
            }
            for (const char * key : {"query", "input", "q", "search_query"}) {
                if (parsed_args.contains(key)) {
                    action[key] = parsed_args.at(key);
                }
            }
        } else if (parsed_args.contains("url")) {
            const std::string action_type = parsed_args.contains("pattern") ? "find_in_page" : "open_page";
            action = json {
                {"type", action_type},
                {"url",  parsed_args.at("url")},
            };
            if (parsed_args.contains("pattern")) {
                action["pattern"] = parsed_args.at("pattern");
            }
        }
        if (!responses_web_search_wrapper.empty()) {
            const json shell_item = build_web_search_local_shell_call_item(
                    tool_call,
                    status,
                    action,
                    responses_web_search_wrapper);
            if (!shell_item.is_null()) {
                return shell_item;
            }
        }
        return json {
            {"type",   "web_search_call"},
            {"id",     item_id.empty() ? "ws_" + random_string() : item_id},
            {"status", status},
            {"action", action},
        };
    }

    if (tool_type == "file_search") {
        if (!responses_file_search_wrapper.empty()) {
            const json shell_item = build_file_search_local_shell_call_item(
                    tool_call,
                    status,
                    parsed_args,
                    responses_file_search_wrapper);
            if (!shell_item.is_null()) {
                return shell_item;
            }
        }

        const std::string query = file_search_query_from_args(parsed_args);
        return json {
            {"type",    "file_search_call"},
            {"id",      item_id.empty() ? "fs_" + random_string() : item_id},
            {"status",  status},
            {"queries", query.empty() ? json::array() : json::array({query})},
        };
    }

    if (tool_type == "image_generation") {
        json output_item = {
            {"type",   "image_generation_call"},
            {"id",     item_id.empty() ? "ig_" + random_string() : item_id},
            {"status", status},
            {"result", json_value(parsed_args, "result", std::string())},
        };
        const std::string revised_prompt = json_value(parsed_args, "revised_prompt",
            json_value(parsed_args, "prompt", std::string()));
        if (!revised_prompt.empty()) {
            output_item["revised_prompt"] = revised_prompt;
        }
        return output_item;
    }

    return json {
        {"type",      "function_call"},
        {"id",        item_id.empty() ? "fc_" + tool_call.id : item_id},
        {"call_id",   "call_" + tool_call.id},
        {"name",      tool_name},
        {"arguments", tool_name == "update_plan" ? ([&]() { json args = parsed_args;
            if (args.contains("plan") && args.at("plan").is_array()) {
                for (auto & item : args["plan"]) {
                    if (item.is_object() && item.contains("status") && item.at("status").is_string()) {
                        const std::string s = item.at("status").get<std::string>();
                        item["status"] = s == "Pending" ? "pending" : s
                                           == "InProgress" || s == "In Progress" ? "in_progress" : s
                                           == "Completed" ? "completed" : item["status"];
                    }
                } return args.dump();
            } return tool_call.arguments; })() : tool_call.arguments},
        {"status",    status},
    };
}

static json build_responses_reasoning_item(const std::string & id, const std::string & text, const std::string & status) {
    json item = {
        {"id",                id},
        {"summary",           json::array()},
        {"type",              "reasoning"},
        {"content",           json::array()},
        {"encrypted_content", ""},
        {"status",            status},
    };
    if (!text.empty()) {
        item["summary"].push_back({{"type", "summary_text"}, {"text", text}});
        item["content"].push_back({{"type", "reasoning_text"}, {"text", text}});
    }
    return item;
}

static json build_responses_content_part(const std::string & text) {
    return json {
        {"type", "output_text"}, {"annotations", json::array()},
        {"logprobs", json::array()}, {"text", text},
    };
}

static json build_responses_message_item(const std::string & id, const common_chat_msg & msg, const bool has_tool_calls) {
    return json {
        {"content", json::array({build_responses_content_part(msg.content)})},
        {"id",     id},
        {"phase",  has_tool_calls ? "commentary" : "final_answer"},
        {"role",   msg.role.empty() ? "assistant" : msg.role},
        {"status", "completed"},
        {"type",   "message"},
    };
}

static json build_responses_sse(const char * event, int & seq_num, const json & fields) {
    json data = {
        {"type",            event},
        {"sequence_number", seq_num++},
    };
    for (const auto & field : fields.items()) {
        data[field.key()] = field.value();
    }
    return json {{"event", event}, {"data", data}};
}

json server_task_result_cmpl_final::to_json_oaicompat_resp() {
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content;
    }

    std::vector<json> output;

    if (msg.reasoning_content != "") {
        output.push_back(build_responses_reasoning_item("rs_" + random_string(), msg.reasoning_content, "completed"));
    }

    if (msg.content != "") {
        const bool has_tool_calls = !oaicompat_msg.tool_calls.empty();
        output.push_back(build_responses_message_item("msg_" + random_string(), msg, has_tool_calls));
    }

    for (const common_chat_tool_call & tool_call : oaicompat_msg.tool_calls) {
        output.push_back(server_build_responses_tool_output_item(
            tool_call,
            generation_params.responses_tool_metadata,
            "completed",
            "",
            generation_params.responses_web_search_wrapper,
            generation_params.responses_file_search_wrapper));
    }

    std::string output_text = build_output_text(output);
    json res = build_oai_resp_metadata(oai_resp_id, oaicompat_model, output, output_text,
                                       n_prompt_tokens, n_decoded, n_prompt_tokens_cache);
    if (stop == STOP_TYPE_LIMIT && msg.content.empty() && msg.tool_calls.empty()) {
        res["status"]             = "incomplete";
        res["completed_at"]       = nullptr;
        res["incomplete_details"] = json{{"reason", "max_output_tokens"}};
    }
    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp_stream() {
    std::vector<json> server_sent_events;
    std::vector<json> output;
    int & seq_num = oai_resp_seq_num;
    int output_idx = 0;

    if (oaicompat_msg.reasoning_content != "") {
        const json output_item = build_responses_reasoning_item(
            oai_resp_reasoning_id,
            oaicompat_msg.reasoning_content,
            "completed");

        if (!oai_resp_reasoning_done) {
            server_sent_events.push_back(build_responses_sse("response.output_item.done", seq_num, {
                {"output_index", output_idx}, {"item", output_item},
            }));
        }
        output.push_back(output_item);
        output_idx++;
    }

    if (oaicompat_msg.content != "") {
        const bool has_tool_calls = !oaicompat_msg.tool_calls.empty();
        const json content_part = build_responses_content_part(oaicompat_msg.content);
        common_chat_msg msg = oaicompat_msg;
        msg.role = msg.role.empty() ? "assistant" : msg.role;
        const json output_item = build_responses_message_item(oai_resp_message_id, msg, has_tool_calls);
        if (!oai_resp_message_done) {
            server_sent_events.push_back(build_responses_sse("response.output_text.done", seq_num, {
                {"output_index", output_idx}, {"content_index", 0},
                {"item_id", oai_resp_message_id}, {"text", oaicompat_msg.content},
                {"logprobs", json::array()},
            }));
            server_sent_events.push_back(build_responses_sse("response.content_part.done", seq_num, {
                {"output_index", output_idx}, {"content_index", 0},
                {"item_id", oai_resp_message_id}, {"part", content_part},
            }));
            server_sent_events.push_back(build_responses_sse("response.output_item.done", seq_num, {
                {"output_index", output_idx}, {"item", output_item},
            }));
        }
        output.push_back(output_item);
        output_idx++;
    }

    for (size_t tc_idx = 0; tc_idx < oaicompat_msg.tool_calls.size(); tc_idx++) {
        const common_chat_tool_call & tool_call = oaicompat_msg.tool_calls[tc_idx];
        const std::string fc_id = tc_idx < oai_resp_fc_item_ids.size()
            ? oai_resp_fc_item_ids[tc_idx]
            : "fc_" + random_string(); // fallback for non-streaming path
        const json output_item = server_build_responses_tool_output_item(
            tool_call,
            generation_params.responses_tool_metadata,
            "completed",
            fc_id,
            generation_params.responses_web_search_wrapper,
            generation_params.responses_file_search_wrapper);
        const std::string tool_type = json_value(output_item, "type", std::string());
        if (tool_type == "function_call") {
            server_sent_events.push_back(build_responses_sse("response.function_call_arguments.done", seq_num, {
                {"output_index", output_idx}, {"item_id", fc_id},
                {"arguments", json_value(output_item, "arguments", tool_call.arguments)},
            }));
        } else if (tool_type == "custom_tool_call") {
            server_sent_events.push_back(build_responses_sse("response.custom_tool_call_input.done", seq_num, {
                {"output_index", output_idx}, {"item_id", fc_id},
                {"input", json_value(output_item, "input", std::string())},
            }));
        }
        server_sent_events.push_back(build_responses_sse("response.output_item.done", seq_num, {
            {"output_index", output_idx}, {"item", output_item},
        }));
        output.push_back(output_item);
        output_idx++;
    }

    std::string output_text = build_output_text(output);
    json resp = build_oai_resp_metadata(oai_resp_id, oaicompat_model, output, output_text,
                                        n_prompt_tokens, n_decoded, n_prompt_tokens_cache);

    const char * event = "response.completed";
    if (stop == STOP_TYPE_LIMIT && oaicompat_msg.content.empty() && oaicompat_msg.tool_calls.empty()) {
        resp["status"]             = "incomplete";
        resp["completed_at"]       = nullptr;
        resp["incomplete_details"] = json{{"reason", "max_output_tokens"}};
        event = "response.incomplete";
    }
    server_sent_events.push_back(build_responses_sse(event, seq_num, {
        {"response", resp},
    }));

    return server_sent_events;
}

json server_task_result_cmpl_final::to_json_oaicompat_asr() {
    json event = json {
        {"type",  "transcript.text.done"},
        {"text",  oaicompat_msg.content},
        {"usage", json {
            {"type",         "tokens"},
            {"input_tokens",  n_prompt_tokens},
            {"output_tokens", n_decoded},
            {"total_tokens",  n_decoded + n_prompt_tokens},
            {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
        }},
    };
    return event;
}

json server_task_result_cmpl_final::to_json_anthropic() {
    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    json content_blocks = json::array();

    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content;
    }

    // thinking block comes first (Anthropic extended thinking format)
    if (!msg.reasoning_content.empty()) {
        content_blocks.push_back({
            {"type", "thinking"},
            {"thinking", msg.reasoning_content},
            {"signature", ""}  // empty signature for local models (no cryptographic verification)
        });
    }

    if (!msg.content.empty()) {
        content_blocks.push_back({
            {"type", "text"},
            {"text", msg.content}
        });
    }

    for (const auto & tool_call : msg.tool_calls) {
        json tool_use_block = {
            {"type", "tool_use"},
            {"id", tool_call.id},
            {"name", tool_call.name}
        };

        try {
            tool_use_block["input"] = json::parse(tool_call.arguments);
        } catch (const std::exception &) {
            tool_use_block["input"] = json::object();
        }

        content_blocks.push_back(tool_use_block);
    }

    json res = {
        {"id", oaicompat_cmpl_id},
        {"type", "message"},
        {"role", "assistant"},
        {"content", content_blocks},
        {"model", oaicompat_model},
        {"stop_reason", stop_reason},
        {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)},
        {"usage", {
            {"cache_read_input_tokens", n_prompt_tokens_cache},
            {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
            {"output_tokens", n_decoded}
        }}
    };

    return res;
}

json server_task_result_cmpl_final::to_json_anthropic_stream() {
    json events = json::array();

    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    bool has_thinking = !oaicompat_msg.reasoning_content.empty();
    bool has_text     = !oaicompat_msg.content.empty();
    size_t num_tool_calls = oaicompat_msg.tool_calls.size();

    // content block indices: thinking (0) -> text (0 or 1) -> tool_use (n+)
    size_t thinking_block_index = 0;
    size_t text_block_index     = has_thinking ? 1 : 0;

    bool thinking_block_started = false;
    bool text_block_started     = false;
    std::unordered_set<size_t> tool_calls_started;

    for (const auto & diff : oaicompat_msg_diffs) {
        // handle thinking/reasoning content
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                });
                thinking_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
            });
        }

        // handle regular text content
        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                });
                text_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
            });
        }

        // handle tool calls
        if (diff.tool_call_index != std::string::npos) {
            size_t content_block_index = (has_thinking ? 1 : 0) + (has_text ? 1 : 0) + diff.tool_call_index;

            if (tool_calls_started.find(diff.tool_call_index) == tool_calls_started.end()) {
                const auto & full_tool_call = oaicompat_msg.tool_calls[diff.tool_call_index];

                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", full_tool_call.id},
                            {"name", full_tool_call.name}
                        }}
                    }}
                });
                tool_calls_started.insert(diff.tool_call_index);
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                });
            }
        }
    }

    // close content blocks in order
    if (has_thinking) {
        // Anthropic API requires a signature_delta before closing thinking blocks
        // We use an empty signature since we can't generate a cryptographic signature for local models
        events.push_back({
            {"event", "content_block_delta"},
            {"data", {
                {"type", "content_block_delta"},
                {"index", thinking_block_index},
                {"delta", {
                    {"type", "signature_delta"},
                    {"signature", ""}
                }}
            }}
        });
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", thinking_block_index}
            }}
        });
    }

    if (has_text) {
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", text_block_index}
            }}
        });
    }

    for (size_t i = 0; i < num_tool_calls; i++) {
        size_t content_block_index = (has_thinking ? 1 : 0) + (has_text ? 1 : 0) + i;
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", content_block_index}
            }}
        });
    }

    events.push_back({
        {"event", "message_delta"},
        {"data", {
            {"type", "message_delta"},
            {"delta", {
                {"stop_reason", stop_reason},
                {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)}
            }},
            {"usage", {
                {"output_tokens", n_decoded}
            }}
        }}
    });

    events.push_back({
        {"event", "message_stop"},
        {"data", {
            {"type", "message_stop"}
        }}
    });

    return events;
}

//
// server_task_result_cmpl_partial
//
void server_task_result_cmpl_partial::update(task_result_state & state) {
    is_updated = true;
    if (is_begin) {
        return; // begin marker only flushes headers, skip parsing
    }
    state.update_chat_msg(content, true, oaicompat_msg_diffs);

    // Copy current state for use in to_json_*() (reflects state BEFORE this chunk)
    thinking_block_started = state.thinking_block_started;
    text_block_started     = state.text_block_started;

    oai_resp_id            = state.oai_resp_id;
    oai_resp_reasoning_id  = state.oai_resp_reasoning_id;
    oai_resp_message_id    = state.oai_resp_message_id;
    oai_resp_fc_id         = state.oai_resp_fc_id;
    oai_resp_fc_item_id    = state.oai_resp_fc_item_id;
    oai_resp_fc_tool_type  = state.oai_resp_fc_tool_type;
    oai_resp_fc_arguments  = state.oai_resp_fc_arguments;
    oai_resp_fc_custom_input = state.oai_resp_fc_custom_input;
    oai_resp_seq_num       = state.oai_resp_seq_num;
    oai_resp_output_idx    = state.oai_resp_output_idx;
    oai_resp_reasoning_output_idx = state.oai_resp_reasoning_output_idx;
    oai_resp_reasoning_done = state.oai_resp_reasoning_done;
    oai_resp_message_done = state.oai_resp_message_done;
    oai_resp_reasoning_content = state.chat_msg.reasoning_content;
    oai_resp_message_content = state.chat_msg.content;

    // track if the accumulated message has any reasoning content
    anthropic_has_reasoning = !state.chat_msg.reasoning_content.empty();

    // Pre-compute state updates based on diffs (for next chunk)
    // Also advance seq_num/output_idx to match events that to_json_oaicompat_resp() will emit
    if (n_decoded == 1) {
        state.oai_resp_seq_num += 2; // response.created + response.in_progress
    }
    for (const common_chat_msg_diff & diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty()) {
            if (!state.thinking_block_started) {
                state.thinking_block_started = true;
                state.oai_resp_reasoning_output_idx = state.oai_resp_output_idx;
                state.oai_resp_seq_num += 2; // output_item.added + reasoning_summary_part.added
                state.oai_resp_output_idx++;
            }
            state.oai_resp_seq_num += 2; // reasoning_summary_text.delta + reasoning_text.delta
        }
        if (!diff.content_delta.empty()) {
            if (!state.text_block_started) {
                if (state.thinking_block_started && !state.oai_resp_reasoning_done) {
                    state.oai_resp_reasoning_done = true;
                    state.oai_resp_seq_num++; // reasoning output_item.done
                }
                state.text_block_started = true;
                state.oai_resp_seq_num += 2; // output_item.added + content_part.added
                state.oai_resp_output_idx++;
            }
            state.oai_resp_seq_num++; // output_text.delta
        }
        if (!diff.tool_call_delta.name.empty()) {
            if (state.thinking_block_started && !state.oai_resp_reasoning_done) {
                state.oai_resp_reasoning_done = true;
                state.oai_resp_seq_num++; // reasoning output_item.done
            }
            if (state.text_block_started && !state.oai_resp_message_done) {
                state.oai_resp_message_done = true;
                state.oai_resp_seq_num += 3; // output_text.done + content_part.done + output_item.done
            }
            state.oai_resp_fc_id = diff.tool_call_delta.id;
            state.oai_resp_fc_item_id = "fc_" + random_string();
            oai_resp_fc_item_id = state.oai_resp_fc_item_id;
            state.oai_resp_fc_tool_type = get_responses_tool_type(responses_tool_metadata, diff.tool_call_delta.name);
            state.oai_resp_fc_arguments.clear();
            state.oai_resp_fc_custom_input.clear();
            state.oai_resp_fc_item_ids.push_back(state.oai_resp_fc_item_id);
            state.oai_resp_seq_num++;    // output_item.added
            state.oai_resp_output_idx++;
        }
        if (!diff.tool_call_delta.arguments.empty()) {
            const std::string tool_type = state.oai_resp_fc_tool_type;
            if (tool_type == "function") {
                state.oai_resp_seq_num++; // function_call_arguments.delta
            }
            const std::string next_arguments = state.oai_resp_fc_arguments + diff.tool_call_delta.arguments;
            if (tool_type == "custom") {
                const std::string delta = best_effort_custom_tool_delta(next_arguments, state.oai_resp_fc_custom_input);
                const json parsed_args = parse_tool_arguments(next_arguments);
                std::string current_input;
                if (get_custom_tool_input_value_if_present(parsed_args, current_input)) {
                    state.oai_resp_fc_custom_input = current_input;
                }
                if (!delta.empty()) {
                    state.oai_resp_seq_num++; // custom_tool_call_input.delta
                }
            }
            state.oai_resp_fc_arguments = next_arguments;
        }
    }
}

json server_task_result_cmpl_partial::to_json() {
    GGML_ASSERT(is_updated && "update() must be called before to_json()");
    if (is_begin) {
        return nullptr; // simply signal to HTTP handler to send the headers and status code
    }
    switch (res_type) {
        case TASK_RESPONSE_TYPE_NONE:
            return to_json_non_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CMPL:
            return to_json_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            return to_json_oaicompat_chat();
        case TASK_RESPONSE_TYPE_OAI_RESP:
            return to_json_oaicompat_resp();
        case TASK_RESPONSE_TYPE_OAI_ASR:
            return to_json_oaicompat_asr();
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            return to_json_anthropic();
        default:
            GGML_ASSERT(false && "Invalid task_response_type");
    }
}

json server_task_result_cmpl_partial::to_json_non_oaicompat() {
    // non-OAI-compat JSON
    json res = json {
        {"index",            index},
        {"content",          content},
        {"tokens",           tokens},
        {"stop",             false},
        {"id_slot",          id_slot},
        {"tokens_predicted", n_decoded},
        {"tokens_evaluated", n_prompt_tokens},
    };
    // populate the timings object when needed (usually for the last response or with timings_per_token enabled)
    if (timings.prompt_n > 0) {
        res.push_back({"timings", timings.to_json()});
    }
    if (is_progress) {
        res.push_back({"prompt_progress", progress.to_json()});
    }
    if (!prob_output.probs.empty()) {
        res["completion_probabilities"] = completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs);
    }
    return res;
}

json server_task_result_cmpl_partial::to_json_oaicompat() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (prob_output.probs.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs)},
        };
    }
    json res = json {
        {"choices",            json::array({
            json{
                {"text",          content},
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", nullptr},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "text_completion"},
        {"id",                 oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (timings.prompt_n >= 0) {
        res.push_back({"timings", timings.to_json()});
    }
    if (is_progress) {
        res.push_back({"prompt_progress", progress.to_json()});
    }

    return res;
}

json server_task_result_cmpl_partial::to_json_oaicompat_chat() {
    bool first = n_decoded == 1;
    std::time_t t = std::time(0);
    json choices;

    std::vector<json> deltas;
    auto add_delta = [&](const json & delta) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", index},
                    {"delta", delta},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object", "chat.completion.chunk"},
        });
    };
    // We have to send an initial update to conform to openai behavior
    if (first || is_progress) {
        add_delta({
            {"role", "assistant"},
            {"content", nullptr},
        });
    }

    for (const auto & diff : oaicompat_msg_diffs) {
        add_delta(server_chat_msg_diff_to_json_oaicompat(diff));
    }

    if (!deltas.empty()) {
        auto & last_json = deltas[deltas.size() - 1];
        GGML_ASSERT(last_json.at("choices").size() >= 1);

        if (prob_output.probs.size() > 0) {
            last_json.at("choices").at(0)["logprobs"] = json {
                {"content", completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs)},
            };
        }

        if (timings.prompt_n >= 0) {
            last_json.push_back({"timings", timings.to_json()});
        }
        if (is_progress) {
            last_json.push_back({"prompt_progress", progress.to_json()});
        }
    }

    return deltas;
}

json server_task_result_cmpl_partial::to_json_oaicompat_resp() {
    std::vector<json> events;
    int & seq_num    = oai_resp_seq_num;
    int & output_idx = oai_resp_output_idx;
    auto maybe_close_reasoning = [&]() {
        if (!thinking_block_started || oai_resp_reasoning_done) {
            return;
        }
        events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type",            "response.output_item.done"},
                {"sequence_number", seq_num++},
                {"output_index",    oai_resp_reasoning_output_idx < 0 ? output_idx - 1 : oai_resp_reasoning_output_idx},
                {"item",            build_responses_reasoning_item(
                    oai_resp_reasoning_id,
                    oai_resp_reasoning_content,
                    "completed")},
            }},
        });
        oai_resp_reasoning_done = true;
    };
    auto maybe_close_text = [&]() {
        if (!text_block_started || oai_resp_message_done) {
            return;
        }
        const json content_part = build_responses_content_part(oai_resp_message_content);
        common_chat_msg msg;
        msg.role = "assistant";
        msg.content = oai_resp_message_content;
        events.push_back(build_responses_sse("response.output_text.done", seq_num, {
            {"output_index", output_idx - 1}, {"content_index", 0},
            {"item_id", oai_resp_message_id}, {"text", oai_resp_message_content},
            {"logprobs", json::array()},
        }));
        events.push_back(build_responses_sse("response.content_part.done", seq_num, {
            {"output_index", output_idx - 1}, {"content_index", 0},
            {"item_id", oai_resp_message_id}, {"part", content_part},
        }));
        events.push_back(build_responses_sse("response.output_item.done", seq_num, {
            {"output_index", output_idx - 1},
            {"item", build_responses_message_item(oai_resp_message_id, msg, true)},
        }));
        oai_resp_message_done = true;
    };

    if (n_decoded == 1) {
        // Build initial response object with all required fields but empty output and zeroed usage
        json initial_resp = build_oai_resp_metadata(
            oai_resp_id, oaicompat_model, {}, "",
            0, 0, 0, "in_progress");

        events.push_back(json {
            {"event", "response.created"},
            {"data", json {
                {"type",            "response.created"},
                {"sequence_number", seq_num++},
                {"response",        initial_resp},
            }},
        });
        events.push_back(json {
            {"event", "response.in_progress"},
            {"data", json {
                {"type",            "response.in_progress"},
                {"sequence_number", seq_num++},
                {"response",        initial_resp},
            }},
        });
    }

    for (const common_chat_msg_diff & diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_block_started) {
                events.push_back(json {
                    {"event", "response.output_item.added"},
                    {"data", json {
                        {"type",            "response.output_item.added"},
                        {"sequence_number", seq_num++},
                        {"output_index",    output_idx++},
                        {"item",            build_responses_reasoning_item(oai_resp_reasoning_id, "", "in_progress")},
                    }},
                });
                events.push_back(json {
                    {"event", "response.reasoning_summary_part.added"},
                    {"data", json {
                        {"type",            "response.reasoning_summary_part.added"},
                        {"sequence_number", seq_num++},
                        {"output_index",    output_idx - 1},
                        {"summary_index",   0},
                        {"item_id",         oai_resp_reasoning_id},
                    }},
                });
                thinking_block_started = true;
            }
            events.push_back(json {
                {"event", "response.reasoning_summary_text.delta"},
                {"data", json {
                    {"type",            "response.reasoning_summary_text.delta"},
                    {"sequence_number", seq_num++},
                    {"output_index",    output_idx - 1},
                    {"summary_index",   0},
                    {"delta",           diff.reasoning_content_delta},
                    {"item_id",         oai_resp_reasoning_id},
                }},
            });
            events.push_back(json {
                {"event", "response.reasoning_text.delta"},
                {"data", json {
                    {"type",            "response.reasoning_text.delta"},
                    {"sequence_number", seq_num++},
                    {"output_index",    output_idx - 1},
                    {"content_index",   0},
                    {"delta",           diff.reasoning_content_delta},
                    {"item_id",         oai_resp_reasoning_id},
                }},
            });
        }

        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                maybe_close_reasoning();
                events.push_back(json {
                    {"event", "response.output_item.added"},
                    {"data", json {
                        {"type",            "response.output_item.added"},
                        {"sequence_number", seq_num++},
                        {"output_index",    output_idx++},
                        {"item", json {
                            {"content", json::array()},
                            {"id",      oai_resp_message_id},
                            {"phase",   "commentary"},
                            {"role",    "assistant"},
                            {"status",  "in_progress"},
                            {"type",    "message"},
                        }},
                    }},
                });
                events.push_back(json {
                    {"event", "response.content_part.added"},
                    {"data", json {
                        {"type",            "response.content_part.added"},
                        {"sequence_number", seq_num++},
                        {"output_index",    output_idx - 1},
                        {"content_index",   0},
                        {"item_id",         oai_resp_message_id},
                        {"part", json {
                            {"type", "output_text"},
                            {"text", ""},
                        }},
                    }},
                });
                text_block_started = true;
            }
            events.push_back(json {
                {"event", "response.output_text.delta"},
                {"data", json {
                    {"type",            "response.output_text.delta"},
                    {"sequence_number", seq_num++},
                    {"output_index",    output_idx - 1},
                    {"content_index",   0},
                    {"item_id",         oai_resp_message_id},
                    {"delta",           diff.content_delta},
                }},
            });
        }

        if (!diff.tool_call_delta.name.empty()) {
            maybe_close_reasoning();
            maybe_close_text();
            oai_resp_fc_tool_type = get_responses_tool_type(responses_tool_metadata, diff.tool_call_delta.name);
            oai_resp_fc_arguments.clear();
            oai_resp_fc_custom_input.clear();
            const common_chat_tool_call tool_call {
                diff.tool_call_delta.name,
                "",
                diff.tool_call_delta.id,
            };
            const json output_item = server_build_responses_tool_output_item(
                tool_call,
                responses_tool_metadata,
                "in_progress",
                oai_resp_fc_item_id,
                responses_web_search_wrapper,
                responses_file_search_wrapper);
            if (json_value(output_item, "type", std::string()) == "function_call") {
                oai_resp_fc_tool_type = "function";
            }
            events.push_back(json {
                {"event", "response.output_item.added"},
                {"data", json {
                    {"type",            "response.output_item.added"},
                    {"sequence_number", seq_num++},
                    {"output_index",    output_idx++},
                    {"item",            output_item},
                }},
            });
        }

        if (!diff.tool_call_delta.arguments.empty()) {
            if (oai_resp_fc_tool_type == "function") {
                events.push_back(json {
                    {"event", "response.function_call_arguments.delta"},
                    {"data", json {
                        {"type",            "response.function_call_arguments.delta"},
                        {"sequence_number", seq_num++},
                        {"output_index",    output_idx - 1},
                        {"delta",           diff.tool_call_delta.arguments},
                        {"item_id",         oai_resp_fc_item_id},
                    }},
                });
            } else if (oai_resp_fc_tool_type == "custom") {
                const std::string next_arguments = oai_resp_fc_arguments + diff.tool_call_delta.arguments;
                const std::string delta = best_effort_custom_tool_delta(next_arguments, oai_resp_fc_custom_input);
                const json parsed_args = parse_tool_arguments(next_arguments);
                oai_resp_fc_arguments = next_arguments;
                std::string current_input;
                if (get_custom_tool_input_value_if_present(parsed_args, current_input)) {
                    oai_resp_fc_custom_input = current_input;
                }
                if (!delta.empty()) {
                    events.push_back(json {
                        {"event", "response.custom_tool_call_input.delta"},
                        {"data", json {
                            {"type",            "response.custom_tool_call_input.delta"},
                            {"sequence_number", seq_num++},
                            {"output_index",    output_idx - 1},
                            {"delta",           delta},
                            {"item_id",         oai_resp_fc_item_id},
                        }},
                    });
                }
            }
        }
    }
    return events;
}

json server_task_result_cmpl_partial::to_json_oaicompat_asr() {
    json event = json {
        {"type", "transcript.text.delta"},
        {"delta", content},
    };
    return event;
}

json server_task_result_cmpl_partial::to_json_anthropic() {
    json events = json::array();
    bool first = (n_decoded == 1);
    // use member variables to track block state across streaming calls
    // (anthropic_thinking_block_started, anthropic_text_block_started)

    if (first) {
        events.push_back({
            {"event", "message_start"},
            {"data", {
                {"type", "message_start"},
                {"message", {
                    {"id", oaicompat_cmpl_id},
                    {"type", "message"},
                    {"role", "assistant"},
                    {"content", json::array()},
                    {"model", oaicompat_model},
                    {"stop_reason", nullptr},
                    {"stop_sequence", nullptr},
                    {"usage", {
                        {"cache_read_input_tokens", n_prompt_tokens_cache},
                        {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
                        {"output_tokens", 0}
                    }}
                }}
            }}
        });
    }

    // content block indices: thinking (0) -> text (0 or 1) -> tool_use (n+)
    size_t thinking_block_index = 0;
    // use anthropic_has_reasoning (set in update()) to know if ANY reasoning was generated
    size_t text_block_index     = anthropic_has_reasoning ? 1 : 0;

    // use local copies of streaming state (copied from task_result_state in update())
    // these reflect the state BEFORE this chunk was processed
    bool thinking_started = thinking_block_started;
    bool text_started     = text_block_started;

    for (const auto & diff : oaicompat_msg_diffs) {
        // handle thinking/reasoning content
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                });
                thinking_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
            });
        }

        // handle regular text content
        if (!diff.content_delta.empty()) {
            if (!text_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                });
                text_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
            });
        }

        // handle tool calls
        if (diff.tool_call_index != std::string::npos) {
            // use anthropic_has_reasoning for thinking block count (persists across calls)
            size_t content_block_index = (anthropic_has_reasoning ? 1 : 0) + (text_started ? 1 : 0) + diff.tool_call_index;

            if (!diff.tool_call_delta.name.empty()) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", diff.tool_call_delta.id},
                            {"name", diff.tool_call_delta.name}
                        }}
                    }}
                });
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                });
            }
        }
    }

    return events;
}

//
// server_task_result_embd
//
json server_task_result_embd::to_json() {
    return res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? to_json_oaicompat()
        : to_json_non_oaicompat();
}

json server_task_result_embd::to_json_non_oaicompat() {
    return json {
        {"index",     index},
        {"embedding", embedding},
    };
}

json server_task_result_embd::to_json_oaicompat() {
    return json {
        {"index",            index},
        {"embedding",        embedding[0]},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_rerank
//
json server_task_result_rerank::to_json() {
    return json {
        {"index",            index},
        {"score",            score},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_error
//
json server_task_result_error::to_json() {
    json res = format_error_response(err_msg, err_type);
    if (err_type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
        res["n_prompt_tokens"] = n_prompt_tokens;
        res["n_ctx"]           = n_ctx;
    }
    return res;
}

//
// server_task_result_metrics
//
json server_task_result_metrics::to_json() {
    return json {
        { "idle",                            n_idle_slots },
        { "processing",                      n_processing_slots },
        { "deferred",                        n_tasks_deferred },
        { "t_start",                         t_start },

        { "n_prompt_tokens_processed_total", n_prompt_tokens_processed_total },
        { "t_tokens_generation_total",       t_tokens_generation_total },
        { "n_tokens_predicted_total",        n_tokens_predicted_total },
        { "t_prompt_processing_total",       t_prompt_processing_total },

        { "n_tokens_max",                    n_tokens_max },

        { "n_prompt_tokens_processed",       n_prompt_tokens_processed },
        { "t_prompt_processing",             t_prompt_processing },
        { "n_tokens_predicted",              n_tokens_predicted },
        { "t_tokens_generation",             t_tokens_generation },

        { "n_decode_total",                  n_decode_total },
        { "n_busy_slots_total",              n_busy_slots_total },

        { "slots",                           slots_data },
    };
}

//
// server_task_result_slot_save_load
//
json server_task_result_slot_save_load::to_json() {
    if (is_save) {
        return json {
            { "id_slot",   id_slot },
            { "filename",  filename },
            { "n_saved",   n_tokens },
            { "n_written", n_bytes },
            { "timings", {
                { "save_ms", t_ms }
            }},
        };
    }

    return json {
        { "id_slot",    id_slot },
        { "filename",   filename },
        { "n_restored", n_tokens },
        { "n_read",     n_bytes },
        { "timings", {
            { "restore_ms", t_ms }
        }},
    };
}

//
// server_task_result_slot_erase
//
json server_task_result_slot_erase::to_json() {
    return json {
        { "id_slot",  id_slot },
        { "n_erased", n_erased },
    };
}

//
// server_task_result_get_lora
//

json server_task_result_get_lora::to_json() {
    json result = json::array();
    for (size_t i = 0; i < loras.size(); ++i) {
        auto & lora = loras[i];
        json entry = {
            {"id",            i},
            {"path",          lora.info.path},
            {"scale",         lora.info.scale},
            {"task_name",     lora.info.task_name},
            {"prompt_prefix", lora.info.prompt_prefix},
        };
        if (!lora.alora_invocation_tokens.empty()) {
            entry["alora_invocation_string"] = lora.alora_invocation_string;
            entry["alora_invocation_tokens"] = lora.alora_invocation_tokens;
        }
        result.push_back(std::move(entry));
    }
    return result;
}

//
// server_task_result_apply_lora
//

json server_task_result_apply_lora::to_json() {
    return json {{ "success", true }};
}

//
// server_prompt_cache
//
size_t server_prompt_cache::size() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.size();
    }

    return res;
}

size_t server_prompt_cache::n_tokens() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.n_tokens();
    }

    return res;
}

server_prompt * server_prompt_cache::alloc(const server_prompt & prompt, size_t state_size_tgt, size_t state_size_dft) {
    // first check if the current state is contained fully in the cache
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int cur_lcp_len = it->tokens.get_common_prefix(prompt.tokens);

        if (cur_lcp_len == (int) prompt.tokens.size()) {
            SRV_INF("%s", " - prompt is already in the cache, skipping\n");
            return nullptr;
        }
    }

    // next, remove any cached prompts that are fully contained in the current prompt
    for (auto it = states.begin(); it != states.end();) {
        const int len = it->tokens.get_common_prefix(prompt.tokens);

        if (len == (int) it->tokens.size()) {
            SRV_WRN(" - removing obsolete cached prompt with length %d\n", len);

            it = states.erase(it);
        } else {
            ++it;
        }
    }

    std::vector<uint8_t> state_data_tgt;
    std::vector<uint8_t> state_data_dft;

    // check if we can allocate enough memory for the new state
    try {
        state_data_tgt.resize(state_size_tgt);
        state_data_dft.resize(state_size_dft);
    } catch (const std::bad_alloc & e) {
        SRV_ERR("failed to allocate memory for prompt cache state: %s\n", e.what());

        limit_size = std::max<size_t>(1, 0.4*size());

        SRV_WRN(" - cache size limit reduced to %.3f MiB\n", limit_size / (1024.0 * 1024.0));

        update();

        return nullptr;
    }

    states.push_back({
        /*.tokens      =*/ prompt.tokens.clone(),
        /*.data        =*/ {
            /*.main =*/ std::move(state_data_tgt),
            /*.drft =*/ std::move(state_data_dft),
        },
        /*.checkpoints =*/ prompt.checkpoints,
    });

    return &states.back();
}

bool server_prompt_cache::load(server_prompt & prompt, const server_tokens & tokens_new, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot) {
    const int lcp_best = prompt.tokens.get_common_prefix(tokens_new);

    float f_keep_best = prompt.tokens.size() > 0 ? float(lcp_best) / prompt.tokens.size() : -1.0f; // empty slot: any cache entry wins
    float sim_best    = float(lcp_best) / tokens_new.size();

    SRV_INF(" - looking for better prompt, base f_keep = %.3f, sim = %.3f\n", f_keep_best, sim_best);

    auto it_best = states.end();

    // find the most similar cached prompt, that would also preserve the most context
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int lcp_cur = it->tokens.get_common_prefix(tokens_new);

        const float f_keep_cur = float(lcp_cur) / it->tokens.size();
        const float sim_cur    = float(lcp_cur) / tokens_new.size();

        // don't trash large prompts
        if (f_keep_cur < 0.25f) {
            continue;
        }

        if (f_keep_best < f_keep_cur && sim_best < sim_cur) {
            f_keep_best = f_keep_cur;
            sim_best    = sim_cur;

            it_best = it;
        }
    }

    if (it_best != states.end()) {
        SRV_INF(" - found better prompt with f_keep = %.3f, sim = %.3f\n", f_keep_best, sim_best);

        {
            auto & data = it_best->data.main;

            const size_t size = data.size();
            const size_t n = llama_state_seq_set_data_ext(ctx_tgt, data.data(), size, id_slot, 0);
            if (n != size) {
                SRV_ERR("failed to restore state with size %zu\n", size);

                return false;
            }

            data.clear();
            data.shrink_to_fit();
        }

        {
            auto & data = it_best->data.drft;

            if (!data.empty()) {
                GGML_ASSERT(ctx_dft);

                const size_t size = data.size();
                const size_t n = llama_state_seq_set_data_ext(ctx_dft, data.data(), size, id_slot, 0);
                if (n != size) {
                    SRV_WRN("failed to restore state with size %zu\n", size);

                    return false;
                }

                data.clear();
                data.shrink_to_fit();
            }
        }

        prompt = std::move(*it_best);

        states.erase(it_best);
    }

    return true;
}

void server_prompt_cache::update() {
    if (limit_size > 0) {
        // always keep at least one state, regardless of the limits
        while (states.size() > 1 && size() > limit_size) {
            if (states.empty()) {
                break;
            }

            SRV_WRN(" - cache size limit reached, removing oldest entry (size = %.3f MiB)\n", states.front().size() / (1024.0 * 1024.0));

            states.pop_front();
        }
    }

    // average size per token
    const float size_per_token = std::max<float>(1.0f, float(size()) / (std::max<size_t>(1, n_tokens())));

    // dynamically increase the token limit if it can fit in the memory limit
    const size_t limit_tokens_cur = limit_size > 0 ? std::max<size_t>(limit_tokens, limit_size/size_per_token) : limit_tokens;

    if (limit_tokens > 0) {
        while (states.size() > 1 && n_tokens() > limit_tokens_cur) {
            if (states.empty()) {
                break;
            }

            SRV_WRN(" - cache token limit (%zu, est: %zu) reached, removing oldest entry (size = %.3f MiB)\n",
                    limit_tokens, limit_tokens_cur, states.front().size() / (1024.0 * 1024.0));

            states.pop_front();
        }
    }

    SRV_INF(" - cache state: %zu prompts, %.3f MiB (limits: %.3f MiB, %zu tokens, %zu est)\n",
            states.size(), size() / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0), limit_tokens, limit_tokens_cur);

    for (const auto & state : states) {
        SRV_INF("   - prompt %p: %7d tokens, checkpoints: %2zu, %9.3f MiB\n",
                (const void *)&state, state.n_tokens(), state.checkpoints.size(), state.size() / (1024.0 * 1024.0));
    }
}
