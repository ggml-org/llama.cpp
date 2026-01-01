#pragma once

// Internal header for chat template format implementations.
// This header is NOT part of the public API and should only be included by:
// - common/chat.cpp (main implementation)
// - common/chat-parsers/*.cpp (per-format implementations)

#include "chat.h"
#include "chat-parser.h"
#include "chat-peg-parser.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "peg-parser.h"
#include "regex-partial.h"

#include <minja/chat-template.hpp>

#include <nlohmann/json.hpp>

#include <chrono>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>

// JSON type alias
using json = nlohmann::ordered_json;

// Template type alias (from minja)
typedef minja::chat_template common_chat_template;

// Parameters for template-based format initialization functions
struct templates_params {
    json messages;
    json tools;
    common_chat_tool_choice tool_choice;
    json json_schema;
    bool parallel_tool_calls;
    common_reasoning_format reasoning_format;
    bool stream;
    std::string grammar;
    bool add_generation_prompt = true;
    bool enable_thinking = true;
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    json extra_context;
    bool add_bos;
    bool add_eos;
    bool is_inference = true;
    // When true, use experimental new PEG parsers from chat-parsers/*.cpp instead of legacy parsers
    bool experimental_new_parsers = false;
};

// Helper to iterate over function tools
inline void foreach_function(const json & tools, const std::function<void(const json &)> & fn) {
    for (const auto & tool : tools) {
        if (!tool.contains("type") || tool.at("type") != "function" || !tool.contains("function")) {
            continue;
        }
        fn(tool);
    }
}

// Helper to iterate over function tools
inline void foreach_function(
    const json & tools,
    const std::function<void(
        const json &,
        const std::string &,
        const json &,
        const common_schema_info &
    )> & fn_name_resolved_params)
{
    for (const auto & tool : tools) {
        if (!tool.contains("type") || tool.at("type") != "function" || !tool.contains("function")) {
            continue;
        }
        const auto & function = tool.at("function");
        const std::string & name = function.at("name");
        auto parameters = function.at("parameters");

        auto schema_info = common_schema_info();
        schema_info.resolve_refs(parameters);

        fn_name_resolved_params(function, name, parameters, schema_info);
    }
}

enum class ParameterType { Optional, Required, Additional };

// Helper to iterate over function parameters
inline void foreach_parameter(
    common_chat_peg_builder & p,
    const json & params,
    const std::function<void(const std::string &, const common_peg_parser &, const json &, ParameterType)> & fn)
{
    if (!params.contains("properties") || !params.at("properties").is_object()) {
        return;
    }
    const auto & props = params.at("properties");
    std::vector<std::string> prop_names;
    std::set<std::string> required;
    if (params.contains("required") && params.at("required").is_array()) {
        params.at("required").get_to(required);
    }
    for (const auto & [name, prop] : props.items()) {
        prop_names.push_back(name);
        bool is_required = (required.find(name) != required.end());
        fn(name, p.literal(name), prop, is_required ? ParameterType::Required : ParameterType::Optional);
    }

    // Default to false for stricter parsing - only allow explicitly defined parameters
    bool allow_additional = false;
    // bool additional_has_schema = false;
    json additional_schema;
    if (params.contains("additionalProperties")) {
        const json & additional = params.at("additionalProperties");
        if (additional.is_boolean()) {
            allow_additional = additional.get<bool>();
        } else if (additional.is_object()) {
            allow_additional = true;
            // additional_has_schema = true;
            additional_schema = additional;
        }
    }
    if (allow_additional) {
        // TODO: generate parser rule for string NOT in existing property names
        // use gbnf_excluding_pattern(prop_names + {">"})?
        auto additional_name = p.tag(Tag::TOOL_ARG_NAME, p.until(">"));
        fn("additional", additional_name, additional_schema, ParameterType::Additional);
    }
}

// Helper to iterate over function parameters
inline void foreach_parameter_legacy(const json & function, const std::function<void(const std::string &, const json &, bool)> & fn) {
    if (!function.contains("parameters") || !function.at("parameters").is_object()) {
        return;
    }
    const auto & params = function.at("parameters");
    if (!params.contains("properties") || !params.at("properties").is_object()) {
        return;
    }
    const auto & props = params.at("properties");
    std::set<std::string> required;
    if (params.contains("required") && params.at("required").is_array()) {
        params.at("required").get_to(required);
    }
    for (const auto & [name, prop] : props.items()) {
        bool is_required = (required.find(name) != required.end());
        fn(name, prop, is_required);
    }
    // Note: legacy parses handle additionalProperties themselves (if at all)
}

// Format time for template contexts
inline std::string format_time(const std::chrono::system_clock::time_point & now, const std::string & format) {
    auto time = std::chrono::system_clock::to_time_t(now);
    auto local_time = *std::localtime(&time);
    std::ostringstream ss;
    ss << std::put_time(&local_time, format.c_str());
    return ss.str();
}

// Apply chat template with inputs
inline std::string apply(
    const common_chat_template & tmpl,
    const struct templates_params & inputs,
    const std::optional<json> & messages_override = std::nullopt,
    const std::optional<json> & tools_override = std::nullopt,
    const std::optional<json> & additional_context = std::nullopt)
{
    minja::chat_template_inputs tmpl_inputs;
    tmpl_inputs.messages = messages_override ? *messages_override : inputs.messages;
    if (tools_override) {
        tmpl_inputs.tools = *tools_override;
    } else {
        tmpl_inputs.tools = inputs.tools.empty() ? json() : inputs.tools;
    }
    tmpl_inputs.add_generation_prompt = inputs.add_generation_prompt;
    tmpl_inputs.extra_context = inputs.extra_context;
    tmpl_inputs.extra_context["enable_thinking"] = inputs.enable_thinking;
    if (additional_context) {
        tmpl_inputs.extra_context.merge_patch(*additional_context);
    }

    minja::chat_template_options tmpl_opts;
    auto result = tmpl.apply(tmpl_inputs, tmpl_opts);
    if (inputs.add_bos && string_starts_with(result, tmpl.bos_token())) {
        result = result.substr(tmpl.bos_token().size());
    }
    if (inputs.add_eos && string_ends_with(result, tmpl.eos_token())) {
        result = result.substr(0, result.size() - tmpl.eos_token().size());
    }
    return result;
}

// Type for format initialization functions
typedef common_chat_params (*common_chat_format_init_fn)(
    const common_chat_template & tmpl,
    const struct templates_params & params
);

// Type for format initialization functions that need extra inputs
typedef common_chat_params (*common_chat_format_init_fn_with_inputs)(
    const common_chat_template & tmpl,
    const struct templates_params & params,
    const common_chat_templates_inputs & inputs
);

// Type for llama_3_x style init that takes extra bool
typedef common_chat_params (*common_chat_format_init_fn_llama3x)(
    const common_chat_template & tmpl,
    const struct templates_params & params,
    bool allow_python_tag_builtin_tools
);

// Forward declarations for experimental new PEG parser implementations in chat-parsers/
common_chat_params common_chat_params_init_mistral_nemo_peg(const common_chat_template & tmpl, const struct templates_params & inputs);
common_chat_params common_chat_params_init_magistral_peg(const common_chat_template & tmpl, const struct templates_params & inputs);
common_chat_params common_chat_params_init_command_r7b_peg(const common_chat_template & tmpl, const struct templates_params & inputs);
common_chat_params common_chat_params_init_deepseek_r1_peg(const common_chat_template & tmpl, const struct templates_params & inputs);
common_chat_params common_chat_params_init_deepseek_v3_1_peg(const common_chat_template & tmpl, const struct templates_params & inputs);
common_chat_params common_chat_params_init_firefunction_v2_peg(const common_chat_template & tmpl, const struct templates_params & inputs);
common_chat_params common_chat_params_init_hermes_2_pro_peg(const common_chat_template & tmpl, const struct templates_params & inputs);
common_chat_params common_chat_params_init_llama_3_x_peg(const common_chat_template & tmpl, const struct templates_params & inputs, bool allow_python_tag_builtin_tools);
common_chat_params common_chat_params_init_ministral_3_peg(const common_chat_template & tmpl, const struct templates_params & inputs);
common_chat_params common_chat_params_init_nemotron_v3_peg(const common_chat_template & tmpl, const struct templates_params & inputs);
common_chat_params common_chat_params_init_seed_oss_peg(const common_chat_template & tmpl, const struct templates_params & inputs);
common_chat_params common_chat_params_init_nemotron_v2_peg(const common_chat_template & tmpl, const struct templates_params & inputs);
common_chat_params common_chat_params_init_lfm2_peg(const common_chat_template & tmpl, const struct templates_params & inputs);
common_chat_params common_chat_params_init_apertus_peg(const common_chat_template & tmpl, const struct templates_params & inputs);
common_chat_params common_chat_params_init_minimax_m2_peg(const common_chat_template & tmpl, const struct templates_params & inputs);
common_chat_params common_chat_params_init_qwen3_coder_xml_peg(const common_chat_template & tmpl, const struct templates_params & inputs);
common_chat_params common_chat_params_init_kimi_k2_peg(const common_chat_template & tmpl, const struct templates_params & inputs);
common_chat_params common_chat_params_init_apriel_1_5_peg(const common_chat_template & tmpl, const struct templates_params & inputs);
common_chat_params common_chat_params_init_xiaomi_mimo_peg(const common_chat_template & tmpl, const struct templates_params & inputs);
common_chat_params common_chat_params_init_glm_4_5_peg(const common_chat_template & tmpl, const struct templates_params & inputs);
common_chat_params common_chat_params_init_granite_peg(const common_chat_template & tmpl, const struct templates_params & inputs);
common_chat_params common_chat_params_init_functionary_v3_1_llama_3_1_peg(const common_chat_template & tmpl, const struct templates_params & inputs);
common_chat_params common_chat_params_init_functionary_v3_2_peg(const common_chat_template & tmpl, const struct templates_params & inputs);
common_chat_params common_chat_params_init_gpt_oss_peg(const common_chat_template & tmpl, const struct templates_params & inputs);
common_chat_params common_chat_params_init_generic_peg(const common_chat_template & tmpl, const struct templates_params & inputs);

inline void common_chat_build_peg_grammar(const struct templates_params & inputs, const common_peg_arena & parser, common_chat_params & data){
    if (!inputs.grammar.empty()) {
        // Throw something upstream??
        data.grammar = inputs.grammar;
    } else if (!inputs.json_schema.is_null() && !inputs.experimental_new_parsers) {
        // Legacy path: use json_schema_to_grammar directly (bypasses PEG parser)
        // New parsers handle json_schema internally via p.schema()
        data.grammar = json_schema_to_grammar(inputs.json_schema);
    } else {
        data.parser = parser.save();
        if (data.parser.empty()) {
            throw std::runtime_error(std::string("Empty parser for ") + common_chat_format_name(data.format));
        }
        data.grammar_lazy = !data.grammar_triggers.empty() && inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO;
        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                auto schema = function.at("parameters");
                builder.resolve_refs(schema);
            });
            parser.build_grammar(builder, data.grammar_lazy);
        });
    }
}

// Format struct for XML-style tool calls with individual parameters
// Example: <tool_call><function=name><parameter=key>value</parameter></function></tool_call>
struct generic_tool_call_format {
    // Tool calls array wrapper (all default to eps if not set)
    std::optional<common_peg_parser> tool_calls_start;
    std::optional<common_peg_parser> tool_calls_sep;
    std::optional<common_peg_parser> tool_calls_end;

    // Individual tool call structure (required)
    std::optional<common_peg_parser> tool_call_start;           // e.g., <tool_call><function=
    std::optional<common_peg_parser> tool_call_name_params_sep; // e.g., >
    std::optional<common_peg_parser> tool_call_end;             // e.g., </function></tool_call>

    // Parameter structure (required)
    std::optional<common_peg_parser> param_start;           // e.g., <parameter=
    std::optional<common_peg_parser> param_name_value_sep;  // e.g., >
    std::vector<std::string> param_ends;  // e.g., </parameter> (string for schema_or_raw_string_until)

    bool allow_raw_string_param_value = true;
};

inline common_peg_parser build_generic_tool_calls_peg_parser(
    common_chat_peg_builder & p,
    const struct templates_params & inputs,
    const generic_tool_call_format & format
)
{
    using Tag = common_chat_peg_tag;

    // Validate required fields
    if (!format.tool_call_start || !format.tool_call_name_params_sep || !format.tool_call_end) {
        throw std::runtime_error("tool_call_start, tool_call_name_params_sep, and tool_call_end are required");
    }
    if (!format.param_start || !format.param_name_value_sep || format.param_ends.empty()) {
        throw std::runtime_error("param_start, param_name_value_sep, and param_end are required");
    }

    // Default to eps() if not set
    auto tool_calls_start = format.tool_calls_start ? *format.tool_calls_start : p.eps();
    auto tool_calls_sep = format.tool_calls_sep ? *format.tool_calls_sep : p.eps();
    auto tool_calls_end = format.tool_calls_end ? *format.tool_calls_end : p.eps();

    auto tool_call = p.choice();
    foreach_function(inputs.tools, [&](const auto &, const auto & name, const json & parameters, const auto & schema_info) {
        auto args = p.sequence();
        foreach_parameter(p, parameters, [&](const std::string & param_name, const common_peg_parser & param_p, const json & param_schema, ParameterType param_type) {
            auto close = p.choice();
            for (const auto & end : format.param_ends) {
                close |= p.literal(end);
            }
            auto arg = p.rule("tool-" + name + "-arg-" + param_name,
                p.tag(Tag::TOOL_ARG_OPEN, *format.param_start)
                + p.tag(Tag::TOOL_ARG_NAME, param_p)
                + *format.param_name_value_sep
                + (format.allow_raw_string_param_value
                    ? p.schema_or_raw_string_until("tool-" + name + "-arg-" + param_name + "-schema", param_schema, format.param_ends,
                        schema_info, Tag::TOOL_ARG_STRING_VALUE, Tag::TOOL_ARG_JSON_VALUE, true)
                    : p.schema(p.json(), "tool-" + name + "-arg-" + param_name, param_schema))
                + p.tag(Tag::TOOL_ARG_CLOSE, close));
            switch (param_type) {
                case ParameterType::Required:
                    args += arg;
                    break;
                case ParameterType::Optional:
                    args += p.optional(arg);
                    break;
                case ParameterType::Additional:
                    args += p.repeat(arg, 0, -1);
                    break;
                default:
                    throw std::runtime_error("Unhandled param type");
            }
        });

        tool_call |= p.rule("tool-" + name,
            p.tag(Tag::TOOL_OPEN, *format.tool_call_start)
            // Wrap name + delimiter in atomic so TOOL_NAME isn't emitted prematurely.
            // Without this, "special_function" would match as complete when input is
            // "special_function_" (prefix of "special_function_with_opt"), causing
            // streaming regressions (tool count decreases when more input arrives).
            + p.atomic(p.literal_tag(Tag::TOOL_NAME, name) + *format.tool_call_name_params_sep)
            + args
            + p.tag(Tag::TOOL_CLOSE, *format.tool_call_end));
    });

    return
        tool_calls_start
        + tool_call + p.repeat(tool_calls_sep + tool_call, 0, inputs.parallel_tool_calls ? -1 : 0)
        + tool_calls_end;
}