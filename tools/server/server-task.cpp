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

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <system_error>
#include <type_traits>
#include <utility>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

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
            {"adaptive_target",           sampling.adaptive_target},
            {"adaptive_decay",            sampling.adaptive_decay},
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
        {"adaptive_target",           sampling.adaptive_target},
        {"adaptive_decay",            sampling.adaptive_decay},
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
    // the JSON library converts -inf to null, so we need to prevent that
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
        {"timings",             stats.to_json()},
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
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
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
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
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

    if (stats.is_set()) {
        deltas.back()["timings"] = stats.to_json();
    }

    // extra fields for debugging purposes
    if (verbose && !deltas.empty()) {
        deltas.front()["__verbose"] = to_json_non_oaicompat();
    }

    return deltas;
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
        output.push_back(json {
            {"id",      "rs_" + random_string()},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({ json {
                {"text", msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
            {"status",            "completed"},
        });
    }

    if (msg.content != "") {
        output.push_back(json {
            {"content", json::array({ json {
                {"type",        "output_text"},
                {"annotations", json::array()},
                {"logprobs",    json::array()},
                {"text",        msg.content},
            }})},
            {"id",     "msg_" + random_string()},
            {"role",   msg.role},
            {"status", "completed"},
            {"type",   "message"},
        });
    }

    for (const common_chat_tool_call & tool_call : oaicompat_msg.tool_calls) {
        output.push_back(json {
            {"id",        "fc_" + tool_call.id},
            {"type",      "function_call"},
            {"status",    "completed"},
            {"arguments", tool_call.arguments},
            {"call_id",   "call_" + tool_call.id},
            {"name",      tool_call.name},
        });
    }

    std::time_t t = std::time(0);
    json res = {
        {"completed_at", t},
        {"created_at",   t},
        {"id",           oai_resp_id},
        {"model",        oaicompat_model},
        {"object",       "response"},
        {"output",       output},
        {"status",       "completed"},
        {"usage",        json {
            {"input_tokens",  n_prompt_tokens},
            {"output_tokens", n_decoded},
            {"total_tokens",  n_decoded + n_prompt_tokens},
            {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
        }},
    };

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp_stream() {
    std::vector<json> server_sent_events;
    std::vector<json> output;

    if (oaicompat_msg.reasoning_content != "") {
        const json output_item = json {
            {"id",      oai_resp_reasoning_id},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({ json {
                {"text", oaicompat_msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
        };

        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    if (oaicompat_msg.content != "") {
        server_sent_events.push_back(json {
            {"event", "response.output_text.done"},
            {"data", json {
                {"type",    "response.output_text.done"},
                {"item_id", oai_resp_message_id},
                {"text",    oaicompat_msg.content}
            }}
        });

        const json content_part = {
            {"type",        "output_text"},
            {"annotations", json::array()},
            {"logprobs",    json::array()},
            {"text",        oaicompat_msg.content}
        };

        server_sent_events.push_back(json {
            {"event", "response.content_part.done"},
            {"data", json {
                {"type",    "response.content_part.done"},
                {"item_id", oai_resp_message_id},
                {"part",    content_part}
            }}
        });
        const json output_item = {
            {"type",    "message"},
            {"status",  "completed"},
            {"id",      oai_resp_message_id},
            {"content", json::array({content_part})},
            {"role",    "assistant"}
        };

        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    for (const common_chat_tool_call & tool_call : oaicompat_msg.tool_calls) {
        const json output_item = {
            {"id",        "fc_" + tool_call.id},
            {"type",      "function_call"},
            {"status",    "completed"},
            {"arguments", tool_call.arguments},
            {"call_id",   "call_" + tool_call.id},
            {"name",      tool_call.name}
        };
        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    std::time_t t = std::time(0);
    server_sent_events.push_back(json {
        {"event", "response.completed"},
        {"data", json {
            {"type", "response.completed"},
            {"response", json {
                {"id",         oai_resp_id},
                {"object",     "response"},
                {"created_at", t},
                {"status",     "completed"},
                {"model",      oaicompat_model},
                {"output",     output},
                {"usage",      json {
                    {"input_tokens",  n_prompt_tokens},
                    {"output_tokens", n_decoded},
                    {"total_tokens",  n_decoded + n_prompt_tokens},
                    {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
                }}
            }},
        }}
    });

    if (stats.is_set()) {
        server_sent_events.back().at("data")["timings"] = stats.to_json();
    }

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

    oai_resp_created       = state.oai_resp_created;
    oai_resp_id            = state.oai_resp_id;
    oai_resp_reasoning_id  = state.oai_resp_reasoning_id;
    oai_resp_message_id    = state.oai_resp_message_id;
    oai_resp_fc_id         = state.oai_resp_fc_id;

    // track if the accumulated message has any reasoning content
    anthropic_has_reasoning = !state.chat_msg.reasoning_content.empty();

    if (res_type == TASK_RESPONSE_TYPE_OAI_RESP && !state.oai_resp_created && (is_progress || n_decoded == 1)) {
        state.oai_resp_created = true;
    }

    // Pre-compute state updates based on diffs (for next chunk)
    for (const common_chat_msg_diff & diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty() && !state.thinking_block_started) {
            state.thinking_block_started = true;
        }
        if (!diff.content_delta.empty() && !state.text_block_started) {
            state.text_block_started = true;
        }
        if (!diff.tool_call_delta.name.empty()) {
            state.oai_resp_fc_id = diff.tool_call_delta.id;
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
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }
    if (is_progress) {
        res["prompt_progress"] = progress.to_json();
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
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }
    if (is_progress) {
        res["prompt_progress"] = progress.to_json();
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

        if (stats.is_set()) {
            last_json["timings"] = stats.to_json();
        }
        if (is_progress) {
            last_json["prompt_progress"] = progress.to_json();
        }
    }

    return deltas;
}

json server_task_result_cmpl_partial::to_json_oaicompat_resp() {
    std::vector<json> events;

    if (!oai_resp_created) {
        events.push_back(json {
            {"event", "response.created"},
            {"data", json {
                {"type", "response.created"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
        events.push_back(json {
            {"event", "response.in_progress"},
            {"data", json {
                {"type", "response.in_progress"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
    } else if (is_progress) {
        events.push_back(json {
            {"event", "response.in_progress"},
            {"data", json {
                {"type", "response.in_progress"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
    }

    for (const common_chat_msg_diff & diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_block_started) {
                events.push_back(json {
                    {"event", "response.output_item.added"},
                    {"data", json {
                        {"type", "response.output_item.added"},
                        {"item", json {
                            {"id",                oai_resp_reasoning_id},
                            {"summary",           json::array()},
                            {"type",              "reasoning"},
                            {"content",           json::array()},
                            {"encrypted_content", ""},
                            {"status",            "in_progress"},
                        }},
                    }},
                });
                thinking_block_started = true;
            }
            events.push_back(json {
                {"event", "response.reasoning_text.delta"},
                {"data", json {
                    {"type",    "response.reasoning_text.delta"},
                    {"delta",   diff.reasoning_content_delta},
                    {"item_id", oai_resp_reasoning_id},
                }},
            });
        }

        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back(json {
                    {"event", "response.output_item.added"},
                    {"data", json {
                        {"type", "response.output_item.added"},
                        {"item", json {
                            {"content", json::array()},
                            {"id",      oai_resp_message_id},
                            {"role",    "assistant"},
                            {"status",  "in_progress"},
                            {"type",    "message"},
                        }},
                    }},
                });
                events.push_back(json {
                    {"event", "response.content_part.added"},
                    {"data", json {
                        {"type",    "response.content_part.added"},
                        {"item_id", oai_resp_message_id},
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
                    {"type",    "response.output_text.delta"},
                    {"item_id", oai_resp_message_id},
                    {"delta",   diff.content_delta},
                }},
            });
        }

        if (!diff.tool_call_delta.name.empty()) {
            events.push_back(json {
                {"event", "response.output_item.added"},
                {"data", json {
                    {"type",  "response.output_item.added"},
                    {"item", json {
                        {"id",        "fc_" + diff.tool_call_delta.id},
                        {"arguments", ""},
                        {"call_id",   "call_" + diff.tool_call_delta.id},
                        {"name",      diff.tool_call_delta.name},
                        {"type",      "function_call"},
                        {"status",    "in_progress"},
                    }},
                }},
            });
            oai_resp_fc_id = diff.tool_call_delta.id;
        }

        if (!diff.tool_call_delta.arguments.empty()) {
            events.push_back(json {
                {"event", "response.function_call_arguments.delta"},
                {"data", json {
                    {"type",    "response.function_call_arguments.delta"},
                    {"delta",   diff.tool_call_delta.arguments},
                    {"item_id", "fc_" + oai_resp_fc_id},
                }},
            });
        }
    }

    if (!events.empty()) {
        json & data = events.back().at("data");
        if (stats.is_set()) {
            data["timings"] = stats.to_json();
        }
        if (is_progress) {
            data["prompt_progress"] = progress.to_json();
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
json server_task_result_slots::to_json() {
    return slots_data;
}

json server_task_result_metrics::to_json() {
    // not used, /metrics renders prometheus text via to_metrics()
    return json{};
}

// metrics definition: https://prometheus.io/docs/practices/naming/#metric-names
std::string server_task_result_metrics::to_metrics() {
    const std::vector<metric_item> counters = {
        {
            "prompt_tokens_total",
            "Number of prompt tokens processed, excluding cached tokens",
            (double) metrics.prompt.count
        }, {
            "prompt_tokens_cached_total",
            "Number of prompt tokens reused from the cache",
            (double) metrics.n_prompt_cached
        }, {
            "prompt_seconds_total",
            "Total time spent processing prompts",
            metrics.prompt.time / 1.e6
        }, {
            "tokens_predicted_total",
            "Number of generation tokens processed",
            (double) metrics.predict.count
        }, {
            "tokens_predicted_seconds_total",
            "Total time spent generating tokens",
            metrics.predict.time / 1.e6
        }, {
            "n_decode_total",
            "Total number of llama_decode() calls, excluding speculative decoding and multimodal decoding",
            (double) metrics.n_decode
        }, {
            "n_tokens_max",
            "Largest observed sequence length (prompt + generation)",
            (double) metrics.n_tokens_max
        }, {
            "spec_decode_num_draft_tokens_total",
            "Speculative: Total draft tokens generated",
            (double) metrics.n_draft_tokens
        }, {
            "spec_decode_num_accepted_tokens_total",
            "Speculative: Total draft tokens accepted by the target model",
            (double) metrics.n_draft_accepted
        }, {
            "spec_decode_num_drafts_total",
            "Speculative: Total speculative decoding verification steps",
            (double) metrics.n_draft_verif_steps
        },
    };

    const std::vector<metric_item> gauges = {
        {
            "prompt_tokens_seconds",
            "Average prompt throughput in tokens/s",
            metrics.prompt_bucket.n_per_second()
        }, {
            "predicted_tokens_seconds",
            "Average generation throughput in tokens/s",
            metrics.predict_bucket.n_per_second()
        }, {
            "requests_processing",
            "Number of requests processing",
            (double) n_processing_slots
        }, {
            "requests_deferred",
            "Number of requests deferred",
            (double) n_tasks_deferred
        }, {
            "n_busy_slots_per_decode",
            "Average number of busy slots per llama_decode() call",
            (double) metrics.n_busy_slots / std::max((double) metrics.n_decode, 1.0)
        },
    };

    std::stringstream prometheus;

    auto add_items = [&prometheus](const char * type, const std::vector<metric_item> & items) {
        for (const auto & item : items) {
            prometheus << "# HELP llamacpp:" << item.name << " " << item.description << "\n"
                       << "# TYPE llamacpp:" << item.name << " " << type             << "\n"
                       << "llamacpp:"        << item.name << " " << item.value       << "\n";
        }
    };

    add_items("counter", counters);
    add_items("gauge",   gauges);

    // labeled counter: one time series per draft position
    if (!metrics.n_accepted_per_pos.empty()) {
        prometheus << "# HELP llamacpp:spec_decode_num_accepted_tokens_per_pos_total"
                      " Accepted tokens per draft position\n"
                   << "# TYPE llamacpp:spec_decode_num_accepted_tokens_per_pos_total counter\n";
        for (size_t i = 0; i < metrics.n_accepted_per_pos.size(); i++) {
            prometheus << "llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position=\""
                       << i << "\"} " << metrics.n_accepted_per_pos[i] << "\n";
        }
    }

    return prometheus.str();
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
server_prompt_data::~server_prompt_data() {
    clear();
}

server_prompt_data::server_prompt_data(server_prompt_data && other) noexcept {
    *this = std::move(other);
}

server_prompt_data & server_prompt_data::operator=(server_prompt_data && other) noexcept {
    if (this == &other) {
        return *this;
    }

    clear();

    main         = std::move(other.main);
    drft         = std::move(other.drft);
    cache_file   = std::move(other.cache_file);
    mapping      = other.mapping;
    mapping_size = other.mapping_size;
    main_size    = other.main_size;
    drft_size    = other.drft_size;
    remove_file_on_destroy = other.remove_file_on_destroy;

#ifdef _WIN32
    file_handle    = other.file_handle;
    mapping_handle = other.mapping_handle;
    other.file_handle    = nullptr;
    other.mapping_handle = nullptr;
#endif

    other.mapping      = nullptr;
    other.mapping_size = 0;
    other.main_size    = 0;
    other.drft_size    = 0;
    other.remove_file_on_destroy = true;
    other.cache_file.clear();

    return *this;
}

void server_prompt_data::clear() {
    if (mapping != nullptr) {
#ifdef _WIN32
        UnmapViewOfFile(mapping);
#else
        munmap(mapping, mapping_size);
#endif
        mapping = nullptr;
    }

#ifdef _WIN32
    if (mapping_handle != nullptr) {
        CloseHandle((HANDLE) mapping_handle);
        mapping_handle = nullptr;
    }
    if (file_handle != nullptr) {
        CloseHandle((HANDLE) file_handle);
        file_handle = nullptr;
    }
#endif

    if (!cache_file.empty()) {
        if (remove_file_on_destroy) {
            std::error_code ec;
            std::filesystem::remove(cache_file, ec);
            if (ec) {
                SRV_WRN("failed to remove disk cache file %s: %s\n", cache_file.c_str(), ec.message().c_str());
            }

            ec.clear();
            const std::string metadata_file = cache_file + ".meta";
            std::filesystem::remove(metadata_file, ec);
            if (ec) {
                SRV_WRN("failed to remove disk cache metadata %s: %s\n", metadata_file.c_str(), ec.message().c_str());
            }
        }
        cache_file.clear();
    }

    mapping_size = 0;
    main_size = 0;
    drft_size = 0;
    remove_file_on_destroy = true;
    main.clear();
    drft.clear();
}

void server_prompt_data::discard() {
    remove_file_on_destroy = true;
    clear();
}

static bool server_prompt_cache_alloc_disk(
        server_prompt_data & data,
        const std::string & cache_dir,
        uint64_t & next_file_id,
        size_t size) {
    if (size == 0) {
        SRV_ERR("%s", "cannot allocate an empty disk cache state\n");
        return false;
    }

    if (next_file_id == 0) {
        next_file_id = (uint64_t) ggml_time_us();
    }

    for (;;) {
        const std::filesystem::path path = std::filesystem::path(cache_dir) /
            ("llama-prompt-cache-" + std::to_string(next_file_id++) + ".bin");
        const std::string path_str = path.string();

#ifdef _WIN32
        HANDLE file = CreateFileA(
            path_str.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr,
            CREATE_NEW, FILE_ATTRIBUTE_TEMPORARY, nullptr);
        if (file == INVALID_HANDLE_VALUE) {
            if (GetLastError() == ERROR_FILE_EXISTS) {
                continue;
            }
            SRV_ERR("failed to create disk cache file %s (error %lu)\n", path_str.c_str(), GetLastError());
            return false;
        }

        LARGE_INTEGER file_size;
        file_size.QuadPart = size;
        if (!SetFilePointerEx(file, file_size, nullptr, FILE_BEGIN) || !SetEndOfFile(file)) {
            SRV_ERR("failed to resize disk cache file %s (error %lu)\n", path_str.c_str(), GetLastError());
            CloseHandle(file);
            std::error_code ec;
            std::filesystem::remove(path, ec);
            return false;
        }

        HANDLE mapping_handle = CreateFileMappingA(file, nullptr, PAGE_READWRITE, 0, 0, nullptr);
        if (mapping_handle == nullptr) {
            SRV_ERR("failed to map disk cache file %s (error %lu)\n", path_str.c_str(), GetLastError());
            CloseHandle(file);
            std::error_code ec;
            std::filesystem::remove(path, ec);
            return false;
        }

        void * mapping = MapViewOfFile(mapping_handle, FILE_MAP_ALL_ACCESS, 0, 0, size);
        if (mapping == nullptr) {
            SRV_ERR("failed to open disk cache mapping %s (error %lu)\n", path_str.c_str(), GetLastError());
            CloseHandle(mapping_handle);
            CloseHandle(file);
            std::error_code ec;
            std::filesystem::remove(path, ec);
            return false;
        }

        data.cache_file    = path_str;
        data.mapping       = (uint8_t *) mapping;
        data.mapping_size  = size;
        data.file_handle   = file;
        data.mapping_handle = mapping_handle;
#else
        if (size > (size_t) std::numeric_limits<off_t>::max()) {
            SRV_ERR("disk cache state is too large: %zu bytes\n", size);
            return false;
        }

        const int fd = open(path_str.c_str(), O_RDWR | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR);
        if (fd < 0) {
            if (errno == EEXIST) {
                continue;
            }
            SRV_ERR("failed to create disk cache file %s: %s\n", path_str.c_str(), strerror(errno));
            return false;
        }

        if (ftruncate(fd, (off_t) size) != 0) {
            SRV_ERR("failed to resize disk cache file %s: %s\n", path_str.c_str(), strerror(errno));
            close(fd);
            std::error_code ec;
            std::filesystem::remove(path, ec);
            return false;
        }

        void * mapping = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        const int mmap_errno = errno;
        close(fd);
        if (mapping == MAP_FAILED) {
            SRV_ERR("failed to map disk cache file %s: %s\n", path_str.c_str(), strerror(mmap_errno));
            std::error_code ec;
            std::filesystem::remove(path, ec);
            return false;
        }

        data.cache_file   = path_str;
        data.mapping      = (uint8_t *) mapping;
        data.mapping_size = size;
#endif

        return true;
    }
}

namespace {

constexpr char SERVER_PROMPT_CACHE_META_MAGIC[8] = {'L', 'L', 'P', 'C', 'A', 'C', 'H', 'E'};
constexpr uint32_t SERVER_PROMPT_CACHE_META_VERSION = 1;
constexpr uint32_t SERVER_PROMPT_CACHE_META_FLAG_MTMD = 1u << 0;
constexpr uint64_t SERVER_PROMPT_CACHE_META_HEADER_SIZE = 64;
constexpr uint64_t SERVER_PROMPT_CACHE_META_CHECKPOINT_SIZE = 80;

template <typename T>
bool server_prompt_cache_meta_write(std::ostream & output, const T & value) {
    static_assert(std::is_trivially_copyable<T>::value, "cache metadata values must be trivially copyable");
    output.write(reinterpret_cast<const char *>(&value), sizeof(value));
    return output.good();
}

template <typename T>
bool server_prompt_cache_meta_read(std::istream & input, T & value) {
    static_assert(std::is_trivially_copyable<T>::value, "cache metadata values must be trivially copyable");
    input.read(reinterpret_cast<char *>(&value), sizeof(value));
    return input.good();
}

bool server_prompt_cache_u64_add(uint64_t & value, uint64_t addend) {
    if (addend > std::numeric_limits<uint64_t>::max() - value) {
        return false;
    }
    value += addend;
    return true;
}

bool server_prompt_cache_u64_mul(uint64_t lhs, uint64_t rhs, uint64_t & result) {
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool server_prompt_cache_range_valid(uint64_t offset, uint64_t size, uint64_t total) {
    return offset <= total && size <= total - offset;
}

void server_prompt_cache_remove_files(const std::filesystem::path & data_path) {
    std::error_code ec;
    std::filesystem::remove(data_path, ec);
    if (ec) {
        SRV_WRN("failed to remove disk cache file %s: %s\n", data_path.string().c_str(), ec.message().c_str());
    }

    ec.clear();
    const std::filesystem::path metadata_path = data_path.string() + ".meta";
    std::filesystem::remove(metadata_path, ec);
    if (ec) {
        SRV_WRN("failed to remove disk cache metadata %s: %s\n", metadata_path.string().c_str(), ec.message().c_str());
    }

    ec.clear();
    const std::filesystem::path temporary_path = metadata_path.string() + ".tmp";
    std::filesystem::remove(temporary_path, ec);
    if (ec) {
        SRV_WRN("failed to remove temporary disk cache metadata %s: %s\n", temporary_path.string().c_str(), ec.message().c_str());
    }
}

bool server_prompt_cache_flush_disk(server_prompt_data & data) {
#ifdef _WIN32
    if (!FlushViewOfFile(data.mapping, data.mapping_size)) {
        SRV_ERR("failed to flush disk cache mapping %s (error %lu)\n", data.cache_file.c_str(), GetLastError());
        return false;
    }
    if (!FlushFileBuffers((HANDLE) data.file_handle)) {
        SRV_ERR("failed to flush disk cache file %s (error %lu)\n", data.cache_file.c_str(), GetLastError());
        return false;
    }
#else
    if (msync(data.mapping, data.mapping_size, MS_SYNC) != 0) {
        SRV_ERR("failed to flush disk cache file %s: %s\n", data.cache_file.c_str(), strerror(errno));
        return false;
    }
#endif
    return true;
}

bool server_prompt_cache_open_disk(
        server_prompt_data & data,
        const std::filesystem::path & path,
        size_t size) {
    const std::string path_str = path.string();

#ifdef _WIN32
    HANDLE file = CreateFileA(
        path_str.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr,
        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        SRV_WRN("failed to open disk cache file %s (error %lu)\n", path_str.c_str(), GetLastError());
        return false;
    }

    LARGE_INTEGER file_size;
    if (!GetFileSizeEx(file, &file_size) || file_size.QuadPart < 0 || (uint64_t) file_size.QuadPart != size) {
        SRV_WRN("disk cache file has an unexpected size: %s\n", path_str.c_str());
        CloseHandle(file);
        return false;
    }

    HANDLE mapping_handle = CreateFileMappingA(file, nullptr, PAGE_READWRITE, 0, 0, nullptr);
    if (mapping_handle == nullptr) {
        SRV_WRN("failed to map disk cache file %s (error %lu)\n", path_str.c_str(), GetLastError());
        CloseHandle(file);
        return false;
    }

    void * mapping = MapViewOfFile(mapping_handle, FILE_MAP_ALL_ACCESS, 0, 0, size);
    if (mapping == nullptr) {
        SRV_WRN("failed to open disk cache mapping %s (error %lu)\n", path_str.c_str(), GetLastError());
        CloseHandle(mapping_handle);
        CloseHandle(file);
        return false;
    }

    data.file_handle    = file;
    data.mapping_handle = mapping_handle;
#else
    const int fd = open(path_str.c_str(), O_RDWR);
    if (fd < 0) {
        SRV_WRN("failed to open disk cache file %s: %s\n", path_str.c_str(), strerror(errno));
        return false;
    }

    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 0 || (uint64_t) st.st_size != size) {
        SRV_WRN("disk cache file has an unexpected size: %s\n", path_str.c_str());
        close(fd);
        return false;
    }

    void * mapping = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    const int mmap_errno = errno;
    close(fd);
    if (mapping == MAP_FAILED) {
        SRV_WRN("failed to map disk cache file %s: %s\n", path_str.c_str(), strerror(mmap_errno));
        return false;
    }
#endif

    data.cache_file = path_str;
    data.mapping = (uint8_t *) mapping;
    data.mapping_size = size;
    data.remove_file_on_destroy = false;

    return true;
}

bool server_prompt_cache_write_metadata(
        const server_prompt_cache_state & state,
        const std::string & cache_key) {
    const std::vector<char> serialized_tokens = state.prompt.tokens.serialize();

    const std::filesystem::path metadata_path = state.data.cache_file + ".meta";
    const std::filesystem::path temporary_path = metadata_path.string() + ".tmp";

    std::ofstream output(temporary_path, std::ios::binary | std::ios::trunc);
    if (!output) {
        SRV_ERR("failed to create disk cache metadata %s\n", temporary_path.string().c_str());
        return false;
    }

    const uint32_t version = SERVER_PROMPT_CACHE_META_VERSION;
    const uint32_t flags = state.prompt.tokens.has_mtmd ? SERVER_PROMPT_CACHE_META_FLAG_MTMD : 0;
    const uint64_t key_size = cache_key.size();
    const uint64_t tokens_size = serialized_tokens.size();
    const uint64_t payload_size = state.data.mapping_size;
    const uint64_t main_size = state.data.main_size;
    const uint64_t drft_size = state.data.drft_size;
    const uint64_t checkpoint_count = state.checkpoints_disk.size();

    output.write(SERVER_PROMPT_CACHE_META_MAGIC, sizeof(SERVER_PROMPT_CACHE_META_MAGIC));
    bool ok = output.good() &&
        server_prompt_cache_meta_write(output, version) &&
        server_prompt_cache_meta_write(output, flags) &&
        server_prompt_cache_meta_write(output, key_size) &&
        server_prompt_cache_meta_write(output, tokens_size) &&
        server_prompt_cache_meta_write(output, payload_size) &&
        server_prompt_cache_meta_write(output, main_size) &&
        server_prompt_cache_meta_write(output, drft_size) &&
        server_prompt_cache_meta_write(output, checkpoint_count);

    if (ok && key_size > 0) {
        output.write(cache_key.data(), key_size);
        ok = output.good();
    }
    if (ok && tokens_size > 0) {
        output.write(serialized_tokens.data(), tokens_size);
        ok = output.good();
    }

    for (const auto & checkpoint : state.checkpoints_disk) {
        const int64_t n_tokens = checkpoint.n_tokens;
        const int32_t id_task = checkpoint.id_task;
        const uint32_t reserved = 0;
        const int64_t pos_min = checkpoint.pos_min;
        const int64_t pos_max = checkpoint.pos_max;
        const uint64_t offset_tgt = checkpoint.offset_tgt;
        const uint64_t size_tgt = checkpoint.size_tgt;
        const uint64_t offset_dft = checkpoint.offset_dft;
        const uint64_t size_dft = checkpoint.size_dft;
        const uint64_t offset_spec = checkpoint.offset_spec;
        const uint64_t size_spec = checkpoint.size_spec;

        ok = ok &&
            server_prompt_cache_meta_write(output, n_tokens) &&
            server_prompt_cache_meta_write(output, id_task) &&
            server_prompt_cache_meta_write(output, reserved) &&
            server_prompt_cache_meta_write(output, pos_min) &&
            server_prompt_cache_meta_write(output, pos_max) &&
            server_prompt_cache_meta_write(output, offset_tgt) &&
            server_prompt_cache_meta_write(output, size_tgt) &&
            server_prompt_cache_meta_write(output, offset_dft) &&
            server_prompt_cache_meta_write(output, size_dft) &&
            server_prompt_cache_meta_write(output, offset_spec) &&
            server_prompt_cache_meta_write(output, size_spec);
    }

    output.close();
    ok = ok && !output.fail();

    if (!ok) {
        SRV_ERR("failed to write disk cache metadata %s\n", temporary_path.string().c_str());
        std::error_code ec;
        std::filesystem::remove(temporary_path, ec);
        return false;
    }

    std::error_code ec;
    std::filesystem::rename(temporary_path, metadata_path, ec);
    if (ec) {
        SRV_ERR("failed to commit disk cache metadata %s: %s\n", metadata_path.string().c_str(), ec.message().c_str());
        ec.clear();
        std::filesystem::remove(temporary_path, ec);
        return false;
    }

    return true;
}

bool server_prompt_cache_read_metadata(
        const std::filesystem::path & metadata_path,
        const std::string & expected_cache_key,
        bool expected_has_mtmd,
        server_prompt_cache_state & state) {
    std::error_code ec;
    const uintmax_t metadata_file_size = std::filesystem::file_size(metadata_path, ec);
    if (ec || metadata_file_size < SERVER_PROMPT_CACHE_META_HEADER_SIZE) {
        return false;
    }

    std::ifstream input(metadata_path, std::ios::binary);
    if (!input) {
        return false;
    }

    char magic[sizeof(SERVER_PROMPT_CACHE_META_MAGIC)];
    input.read(magic, sizeof(magic));

    uint32_t version = 0;
    uint32_t flags = 0;
    uint64_t key_size = 0;
    uint64_t tokens_size = 0;
    uint64_t payload_size = 0;
    uint64_t main_size = 0;
    uint64_t drft_size = 0;
    uint64_t checkpoint_count = 0;

    bool ok = input.good() &&
        server_prompt_cache_meta_read(input, version) &&
        server_prompt_cache_meta_read(input, flags) &&
        server_prompt_cache_meta_read(input, key_size) &&
        server_prompt_cache_meta_read(input, tokens_size) &&
        server_prompt_cache_meta_read(input, payload_size) &&
        server_prompt_cache_meta_read(input, main_size) &&
        server_prompt_cache_meta_read(input, drft_size) &&
        server_prompt_cache_meta_read(input, checkpoint_count);

    uint64_t checkpoint_bytes = 0;
    uint64_t expected_metadata_size = SERVER_PROMPT_CACHE_META_HEADER_SIZE;
    ok = ok && memcmp(magic, SERVER_PROMPT_CACHE_META_MAGIC, sizeof(magic)) == 0 &&
        version == SERVER_PROMPT_CACHE_META_VERSION &&
        (flags & ~SERVER_PROMPT_CACHE_META_FLAG_MTMD) == 0 &&
        ((flags & SERVER_PROMPT_CACHE_META_FLAG_MTMD) != 0) == expected_has_mtmd &&
        key_size <= 64 * 1024 &&
        tokens_size > 0 &&
        tokens_size <= 1024ull * 1024ull * 1024ull &&
        tokens_size <= std::numeric_limits<size_t>::max() &&
        tokens_size % sizeof(llama_token) == 0 &&
        payload_size > 0 &&
        payload_size <= std::numeric_limits<size_t>::max() &&
        main_size <= std::numeric_limits<size_t>::max() &&
        drft_size <= std::numeric_limits<size_t>::max() &&
        checkpoint_count <= 1024 * 1024 &&
        checkpoint_count <= std::numeric_limits<size_t>::max() &&
        server_prompt_cache_u64_mul(checkpoint_count, SERVER_PROMPT_CACHE_META_CHECKPOINT_SIZE, checkpoint_bytes) &&
        server_prompt_cache_u64_add(expected_metadata_size, key_size) &&
        server_prompt_cache_u64_add(expected_metadata_size, tokens_size) &&
        server_prompt_cache_u64_add(expected_metadata_size, checkpoint_bytes) &&
        expected_metadata_size == metadata_file_size &&
        main_size <= payload_size &&
        drft_size <= payload_size - main_size;
    if (!ok) {
        return false;
    }

    std::string cache_key(key_size, '\0');
    if (key_size > 0) {
        input.read(cache_key.data(), key_size);
    }
    if (!input.good() || cache_key != expected_cache_key) {
        return false;
    }

    std::vector<char> serialized_tokens(tokens_size);
    input.read(serialized_tokens.data(), tokens_size);
    if (!input.good()) {
        return false;
    }

    llama_tokens packed_tokens(tokens_size / sizeof(llama_token));
    memcpy(packed_tokens.data(), serialized_tokens.data(), tokens_size);
    try {
        state.prompt.tokens = server_tokens::deserialize(packed_tokens, expected_has_mtmd);
    } catch (const std::exception & e) {
        SRV_WRN("failed to deserialize tokens from disk cache metadata %s: %s\n", metadata_path.string().c_str(), e.what());
        return false;
    }
    if (state.prompt.tokens.empty()) {
        return false;
    }

    state.data.main_size = main_size;
    state.data.drft_size = drft_size;
    state.checkpoints_disk.reserve(checkpoint_count);

    uint64_t next_offset = main_size + drft_size;
    for (uint64_t i = 0; i < checkpoint_count; ++i) {
        int64_t n_tokens = 0;
        int32_t id_task = -1;
        uint32_t reserved = 0;
        int64_t pos_min = 0;
        int64_t pos_max = 0;
        uint64_t offset_tgt = 0;
        uint64_t size_tgt = 0;
        uint64_t offset_dft = 0;
        uint64_t size_dft = 0;
        uint64_t offset_spec = 0;
        uint64_t size_spec = 0;

        ok =
            server_prompt_cache_meta_read(input, n_tokens) &&
            server_prompt_cache_meta_read(input, id_task) &&
            server_prompt_cache_meta_read(input, reserved) &&
            server_prompt_cache_meta_read(input, pos_min) &&
            server_prompt_cache_meta_read(input, pos_max) &&
            server_prompt_cache_meta_read(input, offset_tgt) &&
            server_prompt_cache_meta_read(input, size_tgt) &&
            server_prompt_cache_meta_read(input, offset_dft) &&
            server_prompt_cache_meta_read(input, size_dft) &&
            server_prompt_cache_meta_read(input, offset_spec) &&
            server_prompt_cache_meta_read(input, size_spec);

        ok = ok && reserved == 0 &&
            n_tokens >= 0 && (uint64_t) n_tokens <= state.prompt.tokens.size() &&
            pos_min >= std::numeric_limits<llama_pos>::min() && pos_min <= std::numeric_limits<llama_pos>::max() &&
            pos_max >= std::numeric_limits<llama_pos>::min() && pos_max <= std::numeric_limits<llama_pos>::max() &&
            offset_tgt == next_offset && server_prompt_cache_range_valid(offset_tgt, size_tgt, payload_size);
        if (!ok || !server_prompt_cache_u64_add(next_offset, size_tgt)) {
            return false;
        }
        ok = offset_dft == next_offset && server_prompt_cache_range_valid(offset_dft, size_dft, payload_size);
        if (!ok || !server_prompt_cache_u64_add(next_offset, size_dft)) {
            return false;
        }
        ok = offset_spec == next_offset && server_prompt_cache_range_valid(offset_spec, size_spec, payload_size);
        if (!ok || !server_prompt_cache_u64_add(next_offset, size_spec)) {
            return false;
        }

        state.checkpoints_disk.push_back({
            n_tokens,
            id_task,
            (llama_pos) pos_min,
            (llama_pos) pos_max,
            (size_t) offset_tgt,
            (size_t) size_tgt,
            (size_t) offset_dft,
            (size_t) size_dft,
            (size_t) offset_spec,
            (size_t) size_spec,
        });
    }

    if (!input.good() || next_offset != payload_size) {
        return false;
    }

    std::filesystem::path data_path = metadata_path;
    data_path.replace_extension();
    if (!server_prompt_cache_open_disk(state.data, data_path, payload_size)) {
        return false;
    }

    return true;
}

} // namespace

server_prompt_cache::server_prompt_cache(
    int32_t limit_size_mib,
    size_t limit_tokens,
    const std::string & cache_dir,
    const std::string & cache_key,
    bool has_mtmd) :
    limit_size(1024ull*1024ull*(limit_size_mib < 0 ? 0 : limit_size_mib)),
    limit_tokens(limit_tokens),
    cache_dir(cache_dir),
    cache_key(cache_key),
    has_mtmd(has_mtmd) {
    if (cache_dir.empty()) {
        return;
    }

    std::vector<std::filesystem::path> metadata_paths;
    std::error_code ec;
    for (const auto & entry : std::filesystem::directory_iterator(cache_dir, ec)) {
        if (ec) {
            break;
        }
        if (!entry.is_regular_file(ec) || ec) {
            ec.clear();
            continue;
        }

        const std::string filename = entry.path().filename().string();
        const bool is_cache_file = filename.rfind("llama-prompt-cache-", 0) == 0;
        if (!is_cache_file) {
            continue;
        }

        if (string_ends_with(filename, ".bin.meta")) {
            metadata_paths.push_back(entry.path());
        } else if (string_ends_with(filename, ".bin.meta.tmp")) {
            std::filesystem::remove(entry.path(), ec);
            ec.clear();
        }
    }
    if (ec) {
        SRV_WRN("failed to scan prompt cache directory %s: %s\n", cache_dir.c_str(), ec.message().c_str());
        return;
    }

    std::sort(metadata_paths.begin(), metadata_paths.end(), [](const auto & lhs, const auto & rhs) {
        std::error_code lhs_ec;
        std::error_code rhs_ec;
        const auto lhs_time = std::filesystem::last_write_time(lhs, lhs_ec);
        const auto rhs_time = std::filesystem::last_write_time(rhs, rhs_ec);
        if (lhs_ec || rhs_ec) {
            return lhs.string() < rhs.string();
        }
        return lhs_time < rhs_time;
    });

    for (const auto & metadata_path : metadata_paths) {
        server_prompt_cache_state state;
        bool loaded = false;
        try {
            loaded = server_prompt_cache_read_metadata(metadata_path, cache_key, has_mtmd, state);
        } catch (const std::exception & e) {
            SRV_WRN("failed to read disk prompt cache metadata %s: %s\n", metadata_path.string().c_str(), e.what());
        }
        if (!loaded) {
            std::filesystem::path data_path = metadata_path;
            data_path.replace_extension();
            SRV_WRN("removing incompatible or invalid disk prompt cache entry %s\n", data_path.string().c_str());
            server_prompt_cache_remove_files(data_path);
            continue;
        }
        states.push_back(std::move(state));
    }

    // Raw files without a committed metadata sidecar are interrupted writes.
    for (const auto & entry : std::filesystem::directory_iterator(cache_dir, ec)) {
        if (ec) {
            break;
        }
        if (!entry.is_regular_file(ec) || ec) {
            ec.clear();
            continue;
        }
        const std::string filename = entry.path().filename().string();
        if (filename.rfind("llama-prompt-cache-", 0) != 0 || !string_ends_with(filename, ".bin")) {
            continue;
        }
        const std::filesystem::path metadata_path = entry.path().string() + ".meta";
        if (!std::filesystem::exists(metadata_path, ec) || ec) {
            ec.clear();
            SRV_WRN("removing incomplete disk prompt cache entry %s\n", entry.path().string().c_str());
            server_prompt_cache_remove_files(entry.path());
        }
    }

    const size_t restored = states.size();
    update();
    SRV_INF("restored %zu persistent prompt cache entries (%.3f MiB) from %s\n",
            states.size(), size() / (1024.0 * 1024.0), cache_dir.c_str());
    if (states.size() < restored) {
        SRV_INF("evicted %zu restored prompt cache entries to satisfy configured limits\n", restored - states.size());
    }
}

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
        res += state.prompt.n_tokens();
    }

    return res;
}

server_prompt_cache_state * server_prompt_cache::alloc(const server_prompt & prompt, size_t state_size_tgt, size_t state_size_dft) {
    // first check if the current state is contained fully in the cache
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int cur_lcp_len = it->prompt.tokens.get_common_prefix(prompt.tokens);

        if (cur_lcp_len == (int) prompt.tokens.size()) {
            SRV_TRC("%s", " - prompt is already in the cache, skipping\n");
            return nullptr;
        }
    }

    // calculate checkpoints size to see if it will fit with the prompt
    size_t checkpoints_size = 0;
    for (const auto & ckpt : prompt.checkpoints) {
        checkpoints_size += ckpt.size();
    }

    const size_t state_size_new = state_size_tgt + state_size_dft + checkpoints_size;

    // skip over-limit entries to avoid disturbing the cache
    if (limit_size > 0 && state_size_new > limit_size) {
        SRV_WRN(" - prompt state size %.3f MiB exceeds cache size limit %.3f MiB, skipping\n",
                state_size_new / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0));
        return nullptr;
    }

    // remove any cached prompts that are fully contained in the current prompt
    for (auto it = states.begin(); it != states.end();) {
        const int len = it->prompt.tokens.get_common_prefix(prompt.tokens);

        if (len == (int) it->prompt.tokens.size()) {
            SRV_TRC(" - removing obsolete cached prompt with length %d\n", len);

            it->data.discard();
            it = states.erase(it);
        } else {
            ++it;
        }
    }

    if (limit_size > 0) {
        // make room before allocating the new vectors to avoid breaching the limit
        while (!states.empty() && size() + state_size_new > limit_size) {
            SRV_WRN(" - making room for prompt cache entry, removing oldest entry (size = %.3f MiB)\n",
                    states.front().size() / (1024.0 * 1024.0));

            states.front().data.discard();
            states.pop_front();
        }
    }

    server_prompt_cache_state state;
    state.prompt.tokens = prompt.tokens.clone();

    if (!cache_dir.empty()) {
        if (!server_prompt_cache_alloc_disk(state.data, cache_dir, next_file_id, state_size_new)) {
            return nullptr;
        }

        state.data.main_size = state_size_tgt;
        state.data.drft_size = state_size_dft;

        size_t offset = state_size_tgt + state_size_dft;
        state.checkpoints_disk.reserve(prompt.checkpoints.size());
        for (const auto & checkpoint : prompt.checkpoints) {
            server_prompt_cache_checkpoint checkpoint_disk;
            checkpoint_disk.n_tokens = checkpoint.n_tokens;
            checkpoint_disk.id_task  = checkpoint.id_task;
            checkpoint_disk.pos_min  = checkpoint.pos_min;
            checkpoint_disk.pos_max  = checkpoint.pos_max;

            checkpoint_disk.offset_tgt = offset;
            checkpoint_disk.size_tgt   = checkpoint.data_tgt.size();
            if (checkpoint_disk.size_tgt > 0) {
                memcpy(state.data.mapping + offset, checkpoint.data_tgt.data(), checkpoint_disk.size_tgt);
                offset += checkpoint_disk.size_tgt;
            }

            checkpoint_disk.offset_dft = offset;
            checkpoint_disk.size_dft   = checkpoint.data_dft.size();
            if (checkpoint_disk.size_dft > 0) {
                memcpy(state.data.mapping + offset, checkpoint.data_dft.data(), checkpoint_disk.size_dft);
                offset += checkpoint_disk.size_dft;
            }

            checkpoint_disk.offset_spec = offset;
            checkpoint_disk.size_spec   = checkpoint.data_spec.size();
            if (checkpoint_disk.size_spec > 0) {
                memcpy(state.data.mapping + offset, checkpoint.data_spec.data(), checkpoint_disk.size_spec);
                offset += checkpoint_disk.size_spec;
            }

            state.checkpoints_disk.push_back(checkpoint_disk);
        }
        GGML_ASSERT(offset == state_size_new);
    } else {
        state.prompt.checkpoints = prompt.checkpoints;

        try {
            state.data.main.resize(state_size_tgt);
            state.data.drft.resize(state_size_dft);
        } catch (const std::bad_alloc & e) {
            SRV_ERR("failed to allocate memory for prompt cache state: %s\n", e.what());

            limit_size = std::max<size_t>(1, 0.4*size());

            SRV_WRN(" - cache size limit reduced to %.3f MiB\n", limit_size / (1024.0 * 1024.0));

            update();

            return nullptr;
        }
    }

    states.push_back(std::move(state));

    return &states.back();
}

bool server_prompt_cache::commit(server_prompt_cache_state * state) {
    if (state == nullptr || !state->data.is_disk()) {
        return state != nullptr;
    }

    auto it = std::find_if(states.begin(), states.end(), [state](const auto & candidate) {
        return &candidate == state;
    });
    if (it == states.end()) {
        return false;
    }

    bool ok = server_prompt_cache_flush_disk(it->data);
    try {
        ok = ok && server_prompt_cache_write_metadata(*it, cache_key);
    } catch (const std::exception & e) {
        SRV_ERR("failed to serialize disk prompt cache metadata: %s\n", e.what());
        ok = false;
    }

    if (!ok) {
        it->data.discard();
        states.erase(it);
        return false;
    }

    it->data.remove_file_on_destroy = false;
    return true;
}

void server_prompt_cache::discard(server_prompt_cache_state * state) {
    if (state == nullptr) {
        return;
    }

    auto it = std::find_if(states.begin(), states.end(), [state](const auto & candidate) {
        return &candidate == state;
    });
    if (it == states.end()) {
        return;
    }

    it->data.discard();
    states.erase(it);
}

llama_pos server_prompt_pos_min_thold(llama_pos pos_next, int32_t n_swa, bool has_new_tokens) {
    return std::max(0, pos_next - n_swa - (has_new_tokens ? 0 : 1));
}

int server_prompt_checkpoint_reuse(
        const server_tokens & tokens,
        int64_t               n_tokens,
        llama_pos             checkpoint_pos_min,
        llama_pos             checkpoint_pos_max,
        llama_pos             pos_next,
        llama_pos             pos_min_thold) {
    if (checkpoint_pos_max > pos_next) {
        return -1;
    }

    if (checkpoint_pos_min >= pos_min_thold && checkpoint_pos_min != 0) {
        return -1;
    }

    const llama_pos checkpoint_pos_next = std::min(
        pos_next, std::max(checkpoint_pos_min + 1, checkpoint_pos_max));
    return (int) std::min(tokens.size_up_to_pos(checkpoint_pos_next), (size_t) n_tokens);
}

bool server_prompt_cache::load(
        server_prompt & prompt,
        const server_tokens & tokens_new,
        llama_context * ctx_tgt,
        llama_context * ctx_dft,
        int32_t id_slot,
        bool cache_prompt,
        bool needs_checkpoint,
        int32_t n_swa,
        float slot_prompt_similarity) {
    if (tokens_new.empty()) {
        return true;
    }

    const int lcp_best = prompt.tokens.get_common_prefix(tokens_new);

    float f_keep_best = prompt.tokens.size() > 0 ? float(lcp_best) / prompt.tokens.size() : -1.0f; // empty slot: any cache entry wins
    float f_sim_best  = float(lcp_best) / tokens_new.size();

    const bool has_disk_cache = !cache_dir.empty();
    int effective_prefix_reuse_base = 0;
    if (has_disk_cache && !prompt.tokens.empty() && lcp_best > 0) {
        effective_prefix_reuse_base = lcp_best;

        const llama_pos pos_next = prompt.tokens.pos_next(lcp_best);
        const bool has_new_tokens = lcp_best < (int) tokens_new.size();
        const llama_pos pos_min_thold = server_prompt_pos_min_thold(pos_next, n_swa, has_new_tokens);
        const llama_pos pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), id_slot);

        if (pos_min >= pos_min_thold) {
            effective_prefix_reuse_base = 0;
            for (auto it = prompt.checkpoints.rbegin(); it != prompt.checkpoints.rend(); ++it) {
                const int reuse = server_prompt_checkpoint_reuse(
                    prompt.tokens, it->n_tokens, it->pos_min, it->pos_max, pos_next, pos_min_thold);
                if (reuse >= 0) {
                    effective_prefix_reuse_base = reuse;
                    break;
                }
            }
        }

        if (effective_prefix_reuse_base == (int) tokens_new.size()) {
            effective_prefix_reuse_base--;
        }
    }

    if (has_disk_cache) {
        SRV_TRC(" - looking for better prompt, base f_keep = %.3f, f_sim = %.3f, effective_prefix_reuse = %d\n",
                f_keep_best, f_sim_best, effective_prefix_reuse_base);
    } else {
        SRV_TRC(" - looking for better prompt, base f_keep = %.3f, f_sim = %.3f\n", f_keep_best, f_sim_best);
    }

    auto it_best = states.end();
    int effective_prefix_reuse_best = -1;
    size_t mapping_size_best = 0;

    // find the most similar cached prompt, that would also preserve the most context
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int lcp_cur = it->prompt.tokens.get_common_prefix(tokens_new);

        const float f_keep_cur = float(lcp_cur) / it->prompt.tokens.size();
        const float f_sim_cur  = float(lcp_cur) / tokens_new.size();

        if (it->data.is_disk()) {
            if (!cache_prompt) {
                continue;
            }

            int effective_prefix_reuse = lcp_cur;
            if (needs_checkpoint) {
                effective_prefix_reuse = 0;

                const llama_pos pos_next = it->prompt.tokens.pos_next(lcp_cur);
                const bool has_new_tokens = lcp_cur < (int) tokens_new.size();
                const llama_pos pos_min_thold = server_prompt_pos_min_thold(pos_next, n_swa, has_new_tokens);
                for (auto checkpoint = it->checkpoints_disk.rbegin(); checkpoint != it->checkpoints_disk.rend(); ++checkpoint) {
                    const int reuse = server_prompt_checkpoint_reuse(
                        it->prompt.tokens,
                        checkpoint->n_tokens,
                        checkpoint->pos_min,
                        checkpoint->pos_max,
                        pos_next,
                        pos_min_thold);
                    if (reuse >= 0) {
                        effective_prefix_reuse = reuse;
                        break;
                    }
                }
            }

            if (effective_prefix_reuse == (int) tokens_new.size()) {
                effective_prefix_reuse--;
            }

            SRV_TRC("   - prompt with length %7zu, lcp = %7d, f_keep = %.3f, f_sim = %.3f, effective_prefix_reuse = %d\n",
                    it->prompt.tokens.size(), lcp_cur, f_keep_cur, f_sim_cur, effective_prefix_reuse);

            const float effective_similarity = float(effective_prefix_reuse) / tokens_new.size();
            const bool is_admissible =
                effective_prefix_reuse > effective_prefix_reuse_base &&
                effective_similarity > slot_prompt_similarity;
            const bool is_better = is_admissible &&
                (it_best == states.end() ||
                 effective_prefix_reuse > effective_prefix_reuse_best ||
                 (effective_prefix_reuse == effective_prefix_reuse_best && it->data.mapping_size <= mapping_size_best));
            if (is_better) {
                effective_prefix_reuse_best = effective_prefix_reuse;
                mapping_size_best = it->data.mapping_size;
                f_keep_best = f_keep_cur;
                f_sim_best  = f_sim_cur;
                it_best = it;
            }
            continue;
        }

        SRV_TRC("   - prompt with length %7zu, lcp = %7d, f_keep = %.3f, f_sim = %.3f\n",
                it->prompt.tokens.size(), lcp_cur, f_keep_cur, f_sim_cur);

        // don't trash large prompts
        if (f_keep_cur < 0.25f) {
            continue;
        }

        if (f_keep_best < f_keep_cur && f_sim_best < f_sim_cur) {
            f_keep_best = f_keep_cur;
            f_sim_best  = f_sim_cur;

            it_best = it;
        }
    }

    if (it_best != states.end()) {
        if (it_best->data.is_disk()) {
            SRV_TRC(" - found better prompt with f_keep = %.3f, f_sim = %.3f, effective_prefix_reuse = %d\n",
                    f_keep_best, f_sim_best, effective_prefix_reuse_best);
        } else {
            SRV_TRC(" - found better prompt with f_keep = %.3f, f_sim = %.3f\n", f_keep_best, f_sim_best);
        }

        if (it_best->data.is_disk()) {
            auto & data = it_best->data;

            const size_t n_tgt = llama_state_seq_set_data_ext(ctx_tgt, data.mapping, data.main_size, id_slot, 0);
            if (n_tgt != data.main_size) {
                SRV_ERR("failed to restore state with size %zu\n", data.main_size);
                data.discard();
                states.erase(it_best);
                return false;
            }

            if (data.drft_size > 0) {
                if (ctx_dft == nullptr) {
                    SRV_ERR("%s", "cannot restore draft state without a draft context\n");
                    data.discard();
                    states.erase(it_best);
                    return false;
                }

                const size_t n_dft = llama_state_seq_set_data_ext(ctx_dft, data.mapping + data.main_size, data.drft_size, id_slot, 0);
                if (n_dft != data.drft_size) {
                    SRV_WRN("failed to restore state with size %zu\n", data.drft_size);
                    data.discard();
                    states.erase(it_best);
                    return false;
                }
            }

            server_prompt restored;
            restored.tokens = it_best->prompt.tokens.clone();
            try {
                for (const auto & checkpoint_disk : it_best->checkpoints_disk) {
                    common_prompt_checkpoint checkpoint;
                    checkpoint.n_tokens = checkpoint_disk.n_tokens;
                    checkpoint.id_task  = checkpoint_disk.id_task;
                    checkpoint.pos_min  = checkpoint_disk.pos_min;
                    checkpoint.pos_max  = checkpoint_disk.pos_max;

                    checkpoint.data_tgt.assign(
                        data.mapping + checkpoint_disk.offset_tgt,
                        data.mapping + checkpoint_disk.offset_tgt + checkpoint_disk.size_tgt);
                    checkpoint.data_dft.assign(
                        data.mapping + checkpoint_disk.offset_dft,
                        data.mapping + checkpoint_disk.offset_dft + checkpoint_disk.size_dft);
                    checkpoint.data_spec.assign(
                        data.mapping + checkpoint_disk.offset_spec,
                        data.mapping + checkpoint_disk.offset_spec + checkpoint_disk.size_spec);

                    restored.checkpoints.push_back(std::move(checkpoint));
                }
            } catch (const std::bad_alloc & e) {
                SRV_ERR("failed to allocate memory for restored prompt checkpoints: %s\n", e.what());
                return false;
            }

            prompt = std::move(restored);

            std::error_code ec;
            std::filesystem::last_write_time(
                data.cache_file + ".meta", std::filesystem::file_time_type::clock::now(), ec);
            if (ec) {
                SRV_WRN("failed to update disk cache access time %s: %s\n", data.cache_file.c_str(), ec.message().c_str());
            }

            // Disk entries are reusable and persistent. Move the hit to the
            // back so size-based eviction remains least-recently-used.
            states.splice(states.end(), states, it_best);
            return true;
        } else {
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
        }

        prompt = std::move(it_best->prompt);

        states.erase(it_best);
    }

    return true;
}

void server_prompt_cache::update() {
    if (limit_size > 0) {
        while (!states.empty() && size() > limit_size) {
            SRV_WRN(" - cache size limit reached, removing oldest entry (size = %.3f MiB)\n", states.front().size() / (1024.0 * 1024.0));

            states.front().data.discard();
            states.pop_front();
        }
    }

    // average size per token
    const float size_per_token = std::max<float>(1.0f, float(size()) / (std::max<size_t>(1, n_tokens())));

    // dynamically increase the token limit if it can fit in the memory limit
    const size_t limit_tokens_cur = limit_size > 0 ? std::max<size_t>(limit_tokens, limit_size/size_per_token) : limit_tokens;

    if (limit_tokens > 0) {
        while (!states.empty() && n_tokens() > limit_tokens_cur) {
            SRV_WRN(" - cache token limit (%zu, est: %zu) reached, removing oldest entry (size = %.3f MiB)\n",
                    limit_tokens, limit_tokens_cur, states.front().size() / (1024.0 * 1024.0));

            states.front().data.discard();
            states.pop_front();
        }
    }

    SRV_TRC(" - cache state: %zu prompts, %.3f MiB (limits: %.3f MiB, %zu tokens, %zu est)\n",
            states.size(), size() / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0), limit_tokens, limit_tokens_cur);

    for (const auto & state : states) {
        SRV_TRC("   - prompt %p: %7d tokens, checkpoints: %2zu, %9.3f MiB\n",
                (const void *)&state, state.prompt.n_tokens(),
                state.data.is_disk() ? state.checkpoints_disk.size() : state.prompt.checkpoints.size(),
                state.size() / (1024.0 * 1024.0));
    }
}
