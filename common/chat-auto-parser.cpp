#include "chat-auto-parser.h"

#include "chat-peg-parser.h"
#include "chat.h"
#include "common.h"
#include "ggml.h"
#include "json-schema-to-grammar.h"
#include "log.h"

#include <algorithm>
#include <minja/chat-template.hpp>
#include <minja/minja.hpp>
#include <stdexcept>

using json = nlohmann::ordered_json;

static bool string_ends_with(const std::string & str, const std::string & suffix) {
    return str.size() >= suffix.size() && str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static void foreach_function(const json & tools, const std::function<void(const json &)> & callback) {
    if (tools.is_array()) {
        for (const auto & tool : tools) {
            if (tool.contains("type") && tool.at("type") == "function" && tool.contains("function")) {
                callback(tool);
            }
        }
    }
}

// Definition of templates_params to match the one in chat.cpp
struct templates_params {
    json                                  messages;
    json                                  tools;
    common_chat_tool_choice               tool_choice;
    json                                  json_schema;
    bool                                  parallel_tool_calls;
    common_reasoning_format               reasoning_format;
    bool                                  stream;
    std::string                           grammar;
    bool                                  add_generation_prompt;
    bool                                  enable_thinking;
    std::chrono::system_clock::time_point now;
    json                                  extra_context;
    bool                                  add_bos;
    bool                                  add_eos;
    bool                                  is_inference;
    bool                                  add_inference;
};

const char * TemplatePattern::format_to_str(TemplatePattern::ToolCallFormat format) {
    switch (format) {
        case JSON_NATIVE:
            return "JSON_NATIVE";
        case XML_CONSTRUCTED:
            return "XML_CONSTRUCTED";
        case CONTENT_ONLY:
            return "CONTENT_ONLY";
        case UNKNOWN:
            return "UNKNOWN";
        default:
            return "(unknown)";
    }
}

// Implementation of the apply function to get prompt from template
static std::string apply(const minja::chat_template &    tmpl,
                         const struct templates_params & inputs,
                         const std::optional<json> &     messages_override  = std::nullopt,
                         const std::optional<json> &     tools_override     = std::nullopt,
                         const std::optional<json> &     additional_context = std::nullopt) {
    minja::chat_template_inputs tmpl_inputs;
    tmpl_inputs.messages = messages_override ? *messages_override : inputs.messages;
    if (tools_override) {
        tmpl_inputs.tools = *tools_override;
    } else {
        tmpl_inputs.tools = inputs.tools.empty() ? json() : inputs.tools;
    }
    tmpl_inputs.add_generation_prompt            = inputs.add_generation_prompt;
    tmpl_inputs.extra_context                    = inputs.extra_context;
    tmpl_inputs.extra_context["enable_thinking"] = inputs.enable_thinking;
    if (additional_context) {
        tmpl_inputs.extra_context.merge_patch(*additional_context);
    }
    // TODO: add flag to control date/time, if only for testing purposes.
    // tmpl_inputs.now = std::chrono::system_clock::now();

    minja::chat_template_options tmpl_opts;
    // To avoid double BOS / EOS tokens, we're manually removing begining / trailing tokens
    // instead of using `chat_template_options.use_bos_token = false`, since these tokens
    // may be needed inside the template / between messages too.
    try {
        auto result = tmpl.apply(tmpl_inputs, tmpl_opts);
        return result;
    } catch (const std::exception & e) {
        // If template application fails, return an empty string to indicate failure
        LOG_DBG("Template application failed: %s", e.what());
        return "";
    }
}

// Find the difference between two strings
static std::string find_string_difference(const std::string & base, const std::string & extended) {
    size_t common_prefix = 0;
    while (common_prefix < base.length() && common_prefix < extended.length() &&
           base[common_prefix] == extended[common_prefix]) {
        common_prefix++;
    }
    return extended.substr(common_prefix);
}

// TemplatePattern implementation using pure differential analysis
TemplatePattern TemplateAnalyzer::analyze_template(const minja::chat_template & tmpl) {
    TemplatePattern pattern;

    // Perform differential analysis to discover patterns
    auto discovered = analyze_by_differential(tmpl);

    // Set format based on discovered patterns
    pattern.format = determine_format_from_patterns(discovered);

    // Set reasoning support based on discovered markers
    if (!discovered.reasoning_start_marker.empty()) {
        pattern.has_reasoning_support = true;
    }

    // Trim reasoning markers to avoid whitespace issues
    auto trim_marker = [](std::string & s) {
        if (s.empty()) {
            return;
        }
        size_t first = s.find_first_not_of(" \n\t\r");
        if (first == std::string::npos) {
            s.clear();
            return;
        }
        size_t last = s.find_last_not_of(" \n\t\r");
        s           = s.substr(first, (last - first + 1));
    };
    trim_marker(discovered.reasoning_start_marker);
    trim_marker(discovered.reasoning_end_marker);

    // Store discovered patterns
    pattern.special_markers = {
        { "tool_call_opener",       discovered.tool_call_opener       },
        { "tool_call_closer",       discovered.tool_call_closer       },
        { "function_opener",        discovered.function_opener        },
        { "function_closer",        discovered.function_closer        },
        { "function_name_suffix",   discovered.function_name_suffix   },
        { "parameter_opener",       discovered.parameter_opener       },
        { "parameter_closer",       discovered.parameter_closer       },
        { "argument_separator",     discovered.argument_separator     },
        { "tool_call_start_marker", discovered.tool_call_start_marker },
        { "tool_call_end_marker",   discovered.tool_call_end_marker   },
        { "parameter_key_prefix",   discovered.parameter_key_prefix   },
        { "parameter_key_suffix",   discovered.parameter_key_suffix   },
        { "reasoning_start_marker", discovered.reasoning_start_marker },
        { "reasoning_end_marker",   discovered.reasoning_end_marker   }
    };

    return pattern;
}

// Pure differential analysis - no hardcoded patterns
DiscoveredPattern TemplateAnalyzer::analyze_by_differential(const minja::chat_template & tmpl) {
    DiscoveredPattern patterns;

    try {
        LOG_DBG("=== STARTING TEMPLATE DIFFERENTIAL ANALYSIS ===");

        // Helper to refine patterns for JSON Native (e.g. models with custom tags like Nemotron)
        auto refine_json_native = [&](DiscoveredPattern & p, TemplatePattern::ToolCallFormat & fmt,
                                      const std::string & b_out, const std::string & t_out,
                                      const std::string & t_diff) {
            if (t_diff.empty()) {
                return;
            }
            size_t f1 = t_diff.find("test_function_name");
            if (f1 == std::string::npos) {
                return;
            }
            size_t br = t_diff.rfind('{', f1);
            if (br == std::string::npos) {
                return;
            }

            std::string mid = t_diff.substr(br + 1, f1 - (br + 1));
            if (mid.find("\"name\"") != std::string::npos || mid.find("'name'") != std::string::npos ||
                mid.find("name") != std::string::npos) {
                if (fmt != TemplatePattern::JSON_NATIVE) {
                    LOG_DBG("Heuristic: Overriding format to JSON_NATIVE due to JSON signature");
                    fmt = TemplatePattern::JSON_NATIVE;
                }

                // Refine markers from full output to include swallowed symbols like <
                // Refine markers from full output to include swallowed symbols like <
                // Use rfind to avoid matching example tools in system prompt
                size_t ff1 = t_out.rfind("test_function_name");
                size_t fbr = (ff1 != std::string::npos) ? t_out.rfind('{', ff1) : std::string::npos;

                // Try to find turn boundary
                size_t                   turn_start   = std::string::npos;
                std::vector<std::string> turn_headers = { "<SPECIAL_11>Assistant\n", "<|im_start|>assistant\n",
                                                          "Assistant\n", "assistant\n", "Assistant: " };
                for (const auto & header : turn_headers) {
                    size_t p_header = t_out.rfind(header, fbr);
                    if (p_header != std::string::npos) {
                        if (turn_start == std::string::npos || p_header > turn_start) {
                            turn_start = p_header + header.length();
                        }
                    }
                }

                if (turn_start != std::string::npos) {
                    p.tool_call_start_marker = t_out.substr(turn_start, fbr - turn_start);
                } else {
                    // Fallback to divergence point
                    size_t d_pos = 0;
                    while (d_pos < b_out.length() && d_pos < t_out.length() && b_out[d_pos] == t_out[d_pos]) {
                        d_pos++;
                    }
                    size_t s_pos = d_pos;
                    if (s_pos > 0 && t_out[s_pos - 1] == '<') {
                        s_pos--;
                    }
                    if (fbr >= s_pos) {
                        p.tool_call_start_marker = t_out.substr(s_pos, fbr - s_pos);
                    }
                }

                size_t last_brace = t_diff.rfind('}');
                if (last_brace != std::string::npos && last_brace > f1) {
                    std::string after   = t_diff.substr(last_brace + 1);
                    size_t      tag_end = after.find('>');
                    if (tag_end != std::string::npos) {
                        p.tool_call_end_marker = after.substr(0, tag_end + 1);
                    } else {
                        size_t sym_end = after.find_first_not_of("]>} \n\t\r");
                        if (sym_end != std::string::npos) {
                            p.tool_call_end_marker = after.substr(0, sym_end);
                        } else {
                            p.tool_call_end_marker = after;
                        }
                    }
                }

                auto trim = [](std::string & str) {
                    size_t f = str.find_first_not_of(" \n\t\r");
                    if (f == std::string::npos) {
                        str.clear();
                        return;
                    }
                    size_t l = str.find_last_not_of(" \n\t\r");
                    str      = str.substr(f, (l - f + 1));
                };
                trim(p.tool_call_start_marker);
                trim(p.tool_call_end_marker);

                LOG_DBG("Heuristic markers refined: start='%s', end='%s'", p.tool_call_start_marker.c_str(),
                        p.tool_call_end_marker.c_str());
            }
        };

        // Test messages for differential analysis
        json base_msg = {
            { "role",    "assistant" },
            { "content", "MARKER"    }
        };

        json tool_msg1 = {
            { "role",       "assistant"                                                                          },
            { "content",    ""                                                                                   },
            { "tool_calls",
             json::array(
                  { { { "type", "function" },
                      { "function", { { "name", "test_function_name" }, { "arguments", json::object() } } } } }) }
        };

        json tool_msg2 = {
            { "role",       "assistant"                                                                              },
            { "content",    ""                                                                                       },
            { "tool_calls",
             json::array(
                  { { { "type", "function" },
                      { "function",
                        { { "name", "test_function_name" },
                          { "arguments", json::object({ { "param1", "value1" }, { "param2", "value2" } }) } } } } }) }
        };

        json tool_msg3 = {
            { "role",       "assistant"                                                                             },
            { "content",    ""                                                                                      },
            { "tool_calls",
             json::array(
                  { { { "type", "function" },
                      { "function", { { "name", "test_function_name" }, { "arguments", json::object() } } } },
                    { { "type", "function" },
                      { "function", { { "name", "another_test_function" }, { "arguments", json::object() } } } } }) }
        };

        // Apply template to get outputs - always provide both tools and messages with tool_calls
        minja::chat_template_inputs inputs;
        inputs.tools = {
            { { "type", "function" },
             { "function",
                { { "name", "test_function_name" },
                  { "description", "A test function" },
                  { "parameters",
                    { { "type", "object" },
                      { "properties",
                        { { "param1", { { "type", "string" }, { "description", "First parameter" } } },
                          { "param2", { { "type", "string" }, { "description", "Second parameter" } } } } },
                      { "required", json::array({ "param1", "param2" }) } } } } } },
            { { "type", "function" },
             { "function",
                { { "name", "another_test_function" },
                  { "description", "Another test function" },
                  { "parameters",
                    { { "type", "object" },
                      { "properties",
                        { { "param1", { { "type", "string" }, { "description", "First parameter" } } } } },
                      { "required", json::array({ "param1" }) } } } } }           }
        };

        inputs.messages  = { base_msg };
        auto base_output = tmpl.apply(inputs);
        LOG_DBG("Base output: %s", base_output.c_str());

        inputs.messages   = { tool_msg1 };
        auto tool1_output = tmpl.apply(inputs);
        LOG_DBG("Tool1 output: %s", tool1_output.c_str());

        inputs.messages   = { tool_msg2 };
        auto tool2_output = tmpl.apply(inputs);
        LOG_DBG("Tool2 output: %s", tool2_output.c_str());

        inputs.messages   = { tool_msg3 };
        auto tool3_output = tmpl.apply(inputs);
        LOG_DBG("Tool3 output: %s", tool3_output.c_str());

        // Analyze differences to discover patterns
        std::string tool1_diff = find_string_difference(base_output, tool1_output);
        std::string tool2_diff = find_string_difference(base_output, tool2_output);
        std::string tool3_diff = find_string_difference(base_output, tool3_output);

        // Special handling for templates that require generation prompt (Nemotron)
        LOG_DBG("Tool1 diff length: %zu", tool1_diff.length());
        LOG_DBG("Tool2 diff length: %zu", tool2_diff.length());
        LOG_DBG("Tool3 diff length: %zu", tool3_diff.length());

        // If all diffs are empty, try a different approach for Nemotron-style templates
        if (tool1_diff.empty() && tool2_diff.empty() && tool3_diff.empty()) {
            LOG_DBG("All diffs are empty, trying alternative approach for tool call detection");
            // According to manual testing, we need to call the template with both add_generation_prompt: true and false
            json nemotron_base_msg = {
                { "role",    "assistant" },
                { "content", "MARKER"    }
            };
            json nemotron_tool_msg = {
                { "role",       "assistant"                                                                         },
                { "content",    ""                                                                                  },
                { "tool_calls",
                 json::array({ { { "type", "function" },
                                  { "function",
                                    { { "name", "test_function_name" },
                                      { "arguments",
                                        json::object({ { "param1", "value1" }, { "param2", "value2" } }) } } } } }) }
            };

            // Create outputs with add_generation_prompt: false (default)
            auto get_nemotron_diffs = [&](bool gen_prompt, std::string & b_out, std::string & t1_out,
                                          std::string & t1_diff, std::string & t2_diff, std::string & t3_diff) {
                minja::chat_template_inputs b_inputs;
                b_inputs.tools                 = inputs.tools;
                b_inputs.messages              = { nemotron_base_msg };
                b_inputs.add_generation_prompt = gen_prompt;
                b_out                          = tmpl.apply(b_inputs);

                auto get_diff = [&](const json & msg, std::string & out) {
                    minja::chat_template_inputs t_inputs;
                    t_inputs.tools                 = inputs.tools;
                    t_inputs.messages              = { msg };
                    t_inputs.add_generation_prompt = gen_prompt;
                    out                            = tmpl.apply(t_inputs);
                    return find_string_difference(b_out, out);
                };

                t1_diff = get_diff(tool_msg1, t1_out);
                t2_diff = get_diff(tool_msg2, t1_out);  // We only need one t_out for refinement
                t3_diff = get_diff(tool_msg3, t1_out);

                return !t1_diff.empty() || !t2_diff.empty() || !t3_diff.empty();
            };

            std::string n_base_false, n_t1_out_false, n_t1_diff_false, n_t2_diff_false, n_t3_diff_false;
            std::string n_base_true, n_t1_out_true, n_t1_diff_true, n_t2_diff_true, n_t3_diff_true;

            bool false_ok = get_nemotron_diffs(false, n_base_false, n_t1_out_false, n_t1_diff_false, n_t2_diff_false,
                                               n_t3_diff_false);
            bool true_ok =
                get_nemotron_diffs(true, n_base_true, n_t1_out_true, n_t1_diff_true, n_t2_diff_true, n_t3_diff_true);

            if (false_ok) {
                tool1_diff   = n_t1_diff_false;
                tool2_diff   = n_t2_diff_false;
                tool3_diff   = n_t3_diff_false;
                base_output  = n_base_false;
                tool1_output = n_t1_out_false;
            } else if (true_ok) {
                tool1_diff   = n_t1_diff_true;
                tool2_diff   = n_t2_diff_true;
                tool3_diff   = n_t3_diff_true;
                base_output  = n_base_true;
                tool1_output = n_t1_out_true;
            } else {
                // Fallback to cross diff if needed...
                std::string cross = find_string_difference(n_base_false, n_base_true);
                if (!cross.empty()) {
                    tool1_diff   = cross;
                    tool2_diff   = cross;
                    tool3_diff   = cross;
                    base_output  = n_base_false;
                    tool1_output = n_base_true;
                }
            }
        }

        // Find the common prefix and tool call structure
        patterns = extract_patterns_from_differences(tool1_diff, tool2_diff, tool3_diff);

        // Analyze reasoning patterns
        analyze_reasoning(tmpl, patterns);

        // Debug print the discovered patterns
        LOG_DBG("=== DISCOVERED PATTERNS ===");
        LOG_DBG("tool_call_opener: '%s'", patterns.tool_call_opener.c_str());
        LOG_DBG("tool_call_closer: '%s'", patterns.tool_call_closer.c_str());
        LOG_DBG("function_opener: '%s'", patterns.function_opener.c_str());
        LOG_DBG("function_closer: '%s'", patterns.function_closer.c_str());
        LOG_DBG("parameter_opener: '%s'", patterns.parameter_opener.c_str());
        LOG_DBG("parameter_closer: '%s'", patterns.parameter_closer.c_str());
        LOG_DBG("argument_separator: '%s'", patterns.argument_separator.c_str());
        LOG_DBG("tool_call_start_marker: '%s'", patterns.tool_call_start_marker.c_str());
        LOG_DBG("tool_call_end_marker: '%s'", patterns.tool_call_end_marker.c_str());

        // Detect the format
        auto detected_format = determine_format_from_patterns(patterns);

        refine_json_native(patterns, detected_format, base_output, tool1_output, tool1_diff);

        LOG_DBG("=== DETECTED FORMAT ===");
        LOG_DBG("Format: %s", TemplatePattern::format_to_str(detected_format));

        // Additional validation: if we detected a format but have no meaningful markers,
        // we should be conservative and fall back to generic parser
        if (detected_format != TemplatePattern::UNKNOWN) {
            // Check if we have meaningful tool call markers for the detected format
            bool has_meaningful_markers = false;

            if (detected_format == TemplatePattern::JSON_NATIVE) {
                // For JSON_NATIVE, we need clear JSON tool call markers in the output, not just in definitions
                if (!patterns.tool_call_start_marker.empty() &&
                    (patterns.tool_call_start_marker.find('{') != std::string::npos ||
                     patterns.tool_call_start_marker.find('[') != std::string::npos)) {
                    has_meaningful_markers = true;
                } else if (!patterns.function_opener.empty() &&
                           patterns.function_opener.find("{\"name\"") != std::string::npos) {
                    // Check if function opener contains actual JSON tool call structure
                    has_meaningful_markers = true;
                }
            } else if (detected_format == TemplatePattern::XML_CONSTRUCTED) {
                // For XML_CONSTRUCTED, we need clear XML-like markers
                if (!patterns.tool_call_start_marker.empty() &&
                    (patterns.tool_call_start_marker.find('<') != std::string::npos ||
                     patterns.function_opener.find('<') != std::string::npos)) {
                    has_meaningful_markers = true;
                }
            }

            if (!has_meaningful_markers) {
                LOG_DBG("Detected format %s but no meaningful tool call markers found - falling back to generic parser",
                        TemplatePattern::format_to_str(detected_format));
                detected_format = TemplatePattern::UNKNOWN;
            }
        }

        if (detected_format == TemplatePattern::UNKNOWN) {
            if (!tool1_diff.empty()) {
                LOG_DBG("Format is still UNKNOWN but we have diffs - assuming content only");
            } else {
                // No diffs found - this template doesn't support tool calls
                LOG_DBG("No tool call patterns detected - assuming content only");
            }
        }

    } catch (const std::exception & e) {
        LOG_DBG("Template differential analysis failed: %s", e.what());
    }

    LOG_DBG("=== ENDING TEMPLATE DIFFERENTIAL ANALYSIS ===");

    return patterns;
}

// Extract patterns from the differences
DiscoveredPattern TemplateAnalyzer::extract_patterns_from_differences(const std::string & tool1_diff,
                                                                      const std::string & tool2_diff,
                                                                      const std::string & tool3_diff) {
    LOG_DBG("=== EXTRACTING PATTERNS FROM DIFFERENCES ===");

    DiscoveredPattern patterns;

    // Find function name positions in the differences using a more general approach
    size_t func1_pos  = tool1_diff.find("test_function_name");
    size_t func2_pos  = tool2_diff.find("test_function_name");
    size_t func3_pos1 = tool3_diff.find("test_function_name");
    size_t func3_pos2 = tool3_diff.find("another_test_function");

    LOG_DBG("Function name positions - func1_pos: %zu, func2_pos: %zu, func3_pos1: %zu, func3_pos2: %zu", func1_pos,
            func2_pos, func3_pos1, func3_pos2);

    if (func1_pos != std::string::npos && func2_pos != std::string::npos) {
        LOG_DBG("Found function names, extracting patterns...");

        // Extract everything before function_name as tool_call_opener
        // We assume the diff starts after the MARKER (since it's a diff from base)
        patterns.tool_call_opener = tool1_diff.substr(0, func1_pos);

        // Extract parameter structure from tool2_diff with improved logic
        size_t param1_pos       = tool2_diff.find("\"param1\"");
        bool   param_has_quotes = (param1_pos != std::string::npos);

        size_t param2_pos = tool2_diff.find("\"param2\"");
        size_t value1_pos = tool2_diff.find("\"value1\"");
        size_t value2_pos = tool2_diff.find("\"value2\"");

        // If JSON-style quotes not found, try without quotes
        if (param1_pos == std::string::npos) {
            param1_pos = tool2_diff.find("param1");
        }

        if (param_has_quotes) {
            if (param1_pos != std::string::npos) {
                param1_pos++;
            }
            if (param2_pos != std::string::npos) {
                param2_pos++;
            }
            if (value1_pos != std::string::npos) {
                value1_pos++;
            }
            if (value2_pos != std::string::npos) {
                value2_pos++;
            }
        }
        if (param2_pos == std::string::npos) {
            param2_pos = tool2_diff.find("param2");
        }
        if (value1_pos == std::string::npos) {
            value1_pos = tool2_diff.find("value1");
        }
        if (value2_pos == std::string::npos) {
            value2_pos = tool2_diff.find("value2");
        }

        if (param1_pos != std::string::npos && value1_pos != std::string::npos) {
            // Extract parameter opener (everything from param name to value)
            // Original logic: patterns.parameter_opener = tool2_diff.substr(param1_pos, value1_pos - param1_pos);
            // This captures "param1" or "param1>" or "param1: " depending on format

            // Refined logic: extract prefix and suffix around "param1"

            // Find start of parameter block by looking at what's before param1
            // We assume parameter block starts after function name and some separator
            // But we don't know the separator length.

            // Let's look at the text before "param1".
            // If we have "<parameter=param1>", then text before is "<parameter=".
            // How far back?
            // We can compare tool1_diff (no args) and tool2_diff (args).
            // But extract_patterns doesn't have easy access to aligned diffs.

            // Heuristic: search backwards for common delimiters
            size_t      search_start = (param1_pos > 20) ? param1_pos - 20 : 0;
            std::string pre_param    = tool2_diff.substr(search_start, param1_pos - search_start);

            // Look for the last occurence of newline, space, or >
            size_t delim_pos = pre_param.find_last_of('\n');
            if (delim_pos == std::string::npos) {
                delim_pos = pre_param.find_last_of('>');
            }
            // if (delim_pos == std::string::npos) delim_pos = pre_param.find_last_of(" ");

            if (delim_pos != std::string::npos) {
                patterns.parameter_key_prefix = pre_param.substr(delim_pos + 1);
            } else {
                // If no delimiter found, assume the whole pre_param is prefix?
                // Or maybe just take the last few chars?
                // For Qwen3: "<parameter=" is 11 chars.
                // For Seed: "<parameter=" is 11 chars.
                // For Nemotron: "\"" is 1 char.

                // Let's assume it starts with <, {, [, ", or space
                size_t start_marker = pre_param.find_last_of("<{[ \"");
                if (start_marker != std::string::npos) {
                    patterns.parameter_key_prefix = pre_param.substr(start_marker);
                } else {
                    patterns.parameter_key_prefix = pre_param;
                }
            }

            // Trim prefix if it's not empty
            if (!patterns.parameter_key_prefix.empty()) {
                size_t first = patterns.parameter_key_prefix.find_first_not_of(" \n\t");
                if (first == std::string::npos) {
                    // All whitespace? keep it? or empty?
                    // patterns.parameter_key_prefix = "";
                } else {
                    patterns.parameter_key_prefix = patterns.parameter_key_prefix.substr(first);
                }
            }

            // Key suffix is between param1 and value1
            size_t key_end = param1_pos + std::string("param1").length();

            if (value1_pos > key_end) {
                patterns.parameter_key_suffix = tool2_diff.substr(key_end, value1_pos - key_end);

                // Trim suffix
                if (!patterns.parameter_key_suffix.empty()) {
                    size_t first = patterns.parameter_key_suffix.find_first_not_of(" \n\t");
                    size_t last  = patterns.parameter_key_suffix.find_last_not_of(" \n\t");
                    if (first != std::string::npos && last != std::string::npos) {
                        patterns.parameter_key_suffix = patterns.parameter_key_suffix.substr(first, (last - first + 1));
                    }
                }
            }

            // Determine end of value1
            size_t value1_len = 6;  // "value1"
            (void) value1_len;
            size_t actual_val1_pos = value1_pos;
            (void) actual_val1_pos;
            if (value1_pos > 0 && tool2_diff[value1_pos - 1] == '"' && value1_pos + 6 < tool2_diff.length() &&
                tool2_diff[value1_pos + 6] == '"') {
                // Quotes are part of the value if found by find("value1") inside quotes
                // But find("value1") returns index of v.
                // So if we have "value1", index points to v.
                // We want to capture the value structure.
                // If it's quoted, we might want to capture quotes as part of value parser?
                // No, build_constructed_parser expects value content.
                // But extract_patterns should define delimiters.

                // If quoted, suffix should start after quote.
                // And prefix should end before quote.

                // Let's assume value1_pos points to value content.
                // end_of_val1 should be after content.
                // If there are quotes, they are part of key_suffix/closer?
                // For JSON: "key": "value", "key2"...
                // key_suffix = ": \""
                // closer = "\", "

                // So end_of_val1 should be after "value1".
                // And we let gap logic handle the quote.
            }
            size_t end_of_val1 = value1_pos + 6;

            // Extract argument separator (between first and second values)
            if (param2_pos != std::string::npos && value2_pos != std::string::npos) {
                if (param2_pos > end_of_val1) {
                    std::string gap = tool2_diff.substr(end_of_val1, param2_pos - end_of_val1);

                    // Gap contains: [value_closer] [parameter_closer] [separator] [next_key_prefix]
                    // We know next_key_prefix (patterns.parameter_key_prefix).

                    if (!patterns.parameter_key_prefix.empty() &&
                        gap.length() >= patterns.parameter_key_prefix.length() &&
                        gap.substr(gap.length() - patterns.parameter_key_prefix.length()) ==
                            patterns.parameter_key_prefix) {
                        std::string closer_and_sep =
                            gap.substr(0, gap.length() - patterns.parameter_key_prefix.length());

                        // Try to split into closer and separator
                        // For XML: </parameter>\n
                        // For JSON: ",

                        // Heuristic: separator is usually whitespace or comma+whitespace at the end
                        size_t last_non_sep = closer_and_sep.find_last_not_of(" \n\t,");
                        if (last_non_sep != std::string::npos && last_non_sep < closer_and_sep.length() - 1) {
                            patterns.parameter_closer   = closer_and_sep.substr(0, last_non_sep + 1);
                            patterns.argument_separator = closer_and_sep.substr(last_non_sep + 1);
                        } else {
                            patterns.parameter_closer   = closer_and_sep;
                            patterns.argument_separator = "";
                        }
                    } else {
                        // Fallback
                        patterns.parameter_closer = gap;
                    }

                    // Trim parameter_closer
                    if (!patterns.parameter_closer.empty()) {
                        size_t first = patterns.parameter_closer.find_first_not_of(" \n\t");
                        size_t last  = patterns.parameter_closer.find_last_not_of(" \n\t");
                        if (first != std::string::npos && last != std::string::npos) {
                            patterns.parameter_closer = patterns.parameter_closer.substr(first, (last - first + 1));
                        }
                    }
                }
            } else {
                // If only one param, we need to find closer from end of function?
                // We can't easily. But usually closer is same as above.
                // If we didn't find param2, we rely on what we found above.
                // But we haven't found it yet if param2 is missing.
                // In our test case, tool2 has 2 params. So we are good.
            }
        }

        // Extract function structure - find patterns around function name
        const std::string & func_context = tool1_diff;
        // Look for opening tags before the function name
        // Use rfind from func1_pos to find the closest opener
        size_t              open_pos     = func_context.rfind('<', func1_pos);
        if (open_pos != std::string::npos && open_pos < func1_pos) {
            // Check if there is a closer '>' BEFORE the function name (e.g. <opener>func_name)
            size_t close_pos = func_context.find('>', open_pos);
            if (close_pos != std::string::npos && close_pos < func1_pos) {
                patterns.function_opener = func_context.substr(open_pos, close_pos - open_pos + 1);
            } else {
                // The function name is likely inside the tag (e.g. <function=func_name)
                // In this case, the opener is everything from < up to func_name
                patterns.function_opener = func_context.substr(open_pos, func1_pos - open_pos);
            }
        } else {
            // If no XML-style opener found, try to find other patterns
            // Look for patterns that might surround the function name
            size_t start_pos = 0;
            // Search backwards for a potential opener
            for (int i = (int) func1_pos - 1; i >= 0; i--) {
                if (func_context[i] == '{' || func_context[i] == '[' || func_context[i] == '(' ||
                    func_context[i] == '<') {
                    start_pos                = i;
                    patterns.function_opener = func_context.substr(start_pos, func1_pos - start_pos);
                    break;
                }
            }
        }

        // Extract function name suffix (e.g. ">" in <function=name>)
        // It's the text between function name end and:
        // 1. parameter_key_prefix (if params exist)
        // 2. function_closer (if no params)
        // 3. tool_call_end_marker (if no closer)
        // But we don't have parameter_key_prefix yet? Ah we do if we processed tool2.

        size_t func_name_end = func1_pos + std::string("test_function_name").length();

        // Check for suffix immediately following function name
        if (func_name_end < func_context.length()) {
            char next_char = func_context[func_name_end];
            if (next_char == '>' || next_char == ']' || next_char == '}') {
                patterns.function_name_suffix = std::string(1, next_char);
            } else if (next_char == '"') {
                // Check if quote is followed by > (e.g. name="foo">)
                if (func_name_end + 1 < func_context.length() && func_context[func_name_end + 1] == '>') {
                    patterns.function_name_suffix = "\">";
                } else {
                    patterns.function_name_suffix = "\"";
                }
            }
        }
        LOG_DBG("Extracted function_name_suffix: '%s'", patterns.function_name_suffix.c_str());

        // Look for closing tags after the function name AND suffix
        size_t search_start = func_name_end;
        if (!patterns.function_name_suffix.empty()) {
            search_start += patterns.function_name_suffix.length();
        }
        patterns.function_closer = find_closing_pattern(func_context, search_start);

        // If function closer matches parameter closer, look for the next closer
        if (!patterns.function_closer.empty() && !patterns.parameter_closer.empty() &&
            patterns.function_closer == patterns.parameter_closer) {
            size_t pos = func_context.find(patterns.function_closer, search_start);
            if (pos != std::string::npos) {
                std::string next_closer = find_closing_pattern(func_context, pos + patterns.function_closer.length());
                if (!next_closer.empty()) {
                    patterns.function_closer = next_closer;
                }
            }
        }

        // If function_closer is empty, try to infer it from the context
        if (patterns.function_closer.empty()) {
            // Look for common closing patterns after the function name
            std::string after_func = func_context.substr(search_start);
            for (char c : after_func) {
                if (c == '}' || c == ']' || c == ')' || c == '>') {
                    patterns.function_closer = std::string(1, c);
                    break;
                }
                if (c == ' ' || c == '\n' || c == '\t') {
                    continue;  // Skip whitespace
                }
                break;         // Stop at first meaningful character that's not a closer
            }
        }

        LOG_DBG("After processing function context:");
        LOG_DBG("  function_opener: '%s'", patterns.function_opener.c_str());
        LOG_DBG("  function_closer: '%s'", patterns.function_closer.c_str());
        LOG_DBG("  tool_call_opener: '%s'", patterns.tool_call_opener.c_str());

        // Extract tool call structure
        // If tool_call_opener is "longer" than function_opener, it likely contains the tool call start
        // e.g. "<tool_call><function=" -> start is "<tool_call>"
        if (patterns.function_opener.length() > 0 &&
            patterns.tool_call_opener.length() > patterns.function_opener.length()) {
            // Check if tool_call_opener ends with function_opener
            if (patterns.tool_call_opener.rfind(patterns.function_opener) ==
                patterns.tool_call_opener.length() - patterns.function_opener.length()) {
                patterns.tool_call_start_marker = patterns.tool_call_opener.substr(
                    0, patterns.tool_call_opener.length() - patterns.function_opener.length());
            } else {
                patterns.tool_call_start_marker = patterns.tool_call_opener;
            }
        } else {
            // Try to find what comes before function opener
            patterns.tool_call_start_marker = find_tool_call_start(tool1_diff);
        }

        patterns.tool_call_end_marker = find_tool_call_end(func_context, func1_pos);

        // Trim tool_call_end_marker
        if (!patterns.tool_call_end_marker.empty()) {
            size_t first = patterns.tool_call_end_marker.find_first_not_of(" \n\t");
            size_t last  = patterns.tool_call_end_marker.find_last_not_of(" \n\t");
            if (first != std::string::npos && last != std::string::npos) {
                patterns.tool_call_end_marker = patterns.tool_call_end_marker.substr(first, (last - first + 1));
            }
        }

        LOG_DBG("After finding tool call markers:");
        LOG_DBG("  tool_call_start_marker: '%s'", patterns.tool_call_start_marker.c_str());
        LOG_DBG("  tool_call_end_marker: '%s'", patterns.tool_call_end_marker.c_str());

        // If we couldn't find proper markers, try to infer them from all differences
        if (patterns.tool_call_opener.empty()) {
            patterns.tool_call_opener = infer_tool_call_opener(tool1_diff, tool2_diff, tool3_diff);
            // Truncate if it overlaps with function name
            if (func1_pos != std::string::npos && patterns.tool_call_opener.length() > func1_pos) {
                patterns.tool_call_opener = patterns.tool_call_opener.substr(0, func1_pos);
            }
        }
        if (patterns.tool_call_closer.empty()) {
            patterns.tool_call_closer = infer_tool_call_closer(tool1_diff, tool2_diff, tool3_diff);
        }

        // Ensure all patterns are properly set, even if they're inferred
        if (patterns.tool_call_start_marker.empty()) {
            // Try to infer from common patterns in all diffs
            patterns.tool_call_start_marker = find_common_start_pattern(tool1_diff, tool2_diff, tool3_diff);

            // Truncate if it overlaps with function name
            if (func1_pos != std::string::npos && patterns.tool_call_start_marker.length() > func1_pos) {
                patterns.tool_call_start_marker = patterns.tool_call_start_marker.substr(0, func1_pos);
            }
        }

        // Trim tool_call_start_marker (moved here to apply to inferred marker as well)
        if (!patterns.tool_call_start_marker.empty()) {
            size_t first = patterns.tool_call_start_marker.find_first_not_of(" \n\t\r");
            size_t last  = patterns.tool_call_start_marker.find_last_not_of(" \n\t\r");
            if (first != std::string::npos && last != std::string::npos) {
                patterns.tool_call_start_marker = patterns.tool_call_start_marker.substr(first, (last - first + 1));
            }
            // Explicitly remove trailing newlines just in case
            while (!patterns.tool_call_start_marker.empty() &&
                   (patterns.tool_call_start_marker.back() == '\n' || patterns.tool_call_start_marker.back() == '\r')) {
                patterns.tool_call_start_marker.pop_back();
            }
        }

        if (patterns.tool_call_end_marker.empty()) {
            // Try to infer from common patterns at the end of diffs
            patterns.tool_call_end_marker = find_common_end_pattern(tool1_diff, tool2_diff, tool3_diff);
        }
    }

    return patterns;
}

// Helper to find closing patterns
std::string TemplateAnalyzer::find_closing_pattern(const std::string & diff, size_t func_pos) {
    // Look for common closing patterns in order of preference
    std::vector<std::string> closers = { "</", "}", "]", ">", " " };

    std::string best_pattern;
    size_t      best_pos = std::string::npos;

    for (const auto & pattern : closers) {
        size_t pos = diff.find(pattern, func_pos);
        if (pos != std::string::npos) {
            // Check if this pattern is better (earlier)
            if (pos < best_pos) {
                // For XML-style, we need to find the full closing tag
                if (pattern == "</") {
                    size_t end_pos = diff.find('>', pos);
                    if (end_pos != std::string::npos) {
                        best_pattern = diff.substr(pos, end_pos - pos + 1);
                        best_pos     = pos;
                    }
                } else {
                    best_pattern = pattern;
                    best_pos     = pos;
                }
            }
        }
    }
    return best_pattern;
}

std::string TemplateAnalyzer::find_tool_call_start(const std::string & diff) {
    // Look for patterns that might indicate the start of a tool call
    // The diff typically starts with the tool call structure

    // Look for common patterns that might indicate tool call start
    std::vector<std::string> start_patterns = { "<", "[", "{", "call", "func", "tool", "TOOL" };
    for (const auto & pattern : start_patterns) {
        size_t pos = diff.find(pattern);
        // Only consider if it's at the very beginning or very close to it
        if (pos < 5) {
            // If it starts with <, find the closing >
            if (pattern == "<") {
                size_t end_pos = diff.find('>', pos);
                if (end_pos != std::string::npos) {
                    return diff.substr(pos, end_pos - pos + 1);
                }
            }
            // If it starts with [ or {, just return that char
            if (pattern == "[" || pattern == "{") {
                return diff.substr(pos, 1);
            }

            // For words like "tool", try to find full word/tag
            size_t end_pos = diff.find_first_of(">]} \n", pos);
            if (end_pos != std::string::npos) {
                if (diff[end_pos] == '>' || diff[end_pos] == ']' || diff[end_pos] == '}') {
                    return diff.substr(pos, end_pos - pos + 1);
                }
                // Don't include space/newline
                return diff.substr(pos, end_pos - pos);
            }
            return diff.substr(pos, pattern.length());
        }
    }
    return "";
}

std::string TemplateAnalyzer::find_tool_call_end(const std::string & diff, size_t func_pos) {
    // Check if we have a corresponding closer for the opener
    char        opener_char = 0;
    std::string start_tag_name;

    // Check start of diff for opener
    if (diff.length() > 0) {
        if (diff[0] == '[' || diff.find('[') < 5) {
            opener_char = '[';
        } else if (diff[0] == '{' || diff.find('{') < 5) {
            opener_char = '{';
        } else if (diff[0] == '<' || diff.find('<') < 5) {
            opener_char = '<';
        }

        // Try to extract tag name if it's XML-like
        if (opener_char == '<') {
            size_t tag_start = diff.find('<');
            size_t tag_end   = diff.find_first_of(" >\n", tag_start);
            if (tag_start != std::string::npos && tag_end != std::string::npos) {
                start_tag_name = diff.substr(tag_start + 1, tag_end - tag_start - 1);
            }
        }
    }

    // If we found a start tag name, prioritize finding its closer
    if (!start_tag_name.empty()) {
        std::string expected_closer = "</" + start_tag_name + ">";
        size_t      pos             = diff.find(expected_closer, func_pos);
        if (pos != std::string::npos) {
            // Check for ] before this closer if opener was [
            if (opener_char == '[') {
                size_t bracket_pos = diff.rfind(']', pos);
                if (bracket_pos != std::string::npos && bracket_pos > func_pos) {
                    return diff.substr(bracket_pos, (pos + expected_closer.length()) - bracket_pos);
                }
            }
            return expected_closer;
        }
    }

    // Find what comes after the tool call section by looking for common closing patterns after the function
    std::vector<std::string> end_patterns = { "</", "}", "]", ">", "\n", " " };

    std::string best_pattern;
    size_t      best_pos = std::string::npos;

    for (const auto & pattern : end_patterns) {
        size_t pos = diff.find(pattern, func_pos);
        if (pos != std::string::npos) {
            if (pos < best_pos) {
                // For XML-style, we need to find the full closing tag
                if (pattern == "</") {
                    size_t end_pos = diff.find('>', pos);
                    if (end_pos != std::string::npos) {
                        std::string closing_tag = diff.substr(pos, end_pos - pos + 1);
                        // If we opened with [, check if ] is before this tag
                        if (opener_char == '[') {
                            // Check for ] before </
                            size_t bracket_pos = diff.rfind(']', pos);
                            if (bracket_pos != std::string::npos && bracket_pos > func_pos) {
                                best_pattern = diff.substr(bracket_pos, (end_pos + 1) - bracket_pos);
                                best_pos     = pos;  // Use pos of </ as reference for best_pos? or bracket_pos?
                                // Let's keep pos as reference to avoid confusion, but store full string
                            } else {
                                best_pattern = closing_tag;
                                best_pos     = pos;
                            }
                        } else {
                            best_pattern = closing_tag;
                            best_pos     = pos;
                        }
                    }
                } else if ((pattern == "]" && opener_char == '[') || (pattern == "}" && opener_char == '{')) {
                    // Prioritize specific closers
                    best_pattern = pattern;
                    best_pos     = pos;
                    // If we found specific closer, it's likely the best one if it's the earliest
                } else {
                    best_pattern = pattern;
                    best_pos     = pos;
                }
            }
        }
    }

    return best_pattern;
}

// Helper to infer tool call opener from multiple differences
std::string TemplateAnalyzer::infer_tool_call_opener(const std::string & diff1,
                                                     const std::string & diff2,
                                                     const std::string & diff3) {
    // Find common patterns in all three differences
    std::vector<std::string> differences    = { diff1, diff2, diff3 };
    std::string              common_pattern = find_common_substring(differences);
    return common_pattern;
}

// Helper to infer tool call closer from multiple differences
std::string TemplateAnalyzer::infer_tool_call_closer(const std::string & diff1,
                                                     const std::string & diff2,
                                                     const std::string & diff3) {
    // Find common patterns at the end of all three differences
    std::vector<std::string> differences    = { diff1, diff2, diff3 };
    std::string              common_pattern = find_common_suffix(differences);
    return common_pattern;
}

// Find common substring in a vector of strings
std::string TemplateAnalyzer::find_common_substring(const std::vector<std::string> & strings) {
    if (strings.empty()) {
        return "";
    }
    if (strings.size() == 1) {
        return strings[0];
    }

    std::string common = strings[0];
    for (size_t i = 1; i < strings.size(); ++i) {
        const std::string & current = strings[i];
        std::string         temp_common;
        for (size_t j = 0; j < common.length() && j < current.length(); ++j) {
            if (common[j] == current[j]) {
                temp_common += common[j];
            } else {
                break;
            }
        }
        common = temp_common;
    }
    return common;
}

// Find common suffix in a vector of strings
std::string TemplateAnalyzer::find_common_suffix(const std::vector<std::string> & strings) {
    if (strings.empty()) {
        return "";
    }
    if (strings.size() == 1) {
        return strings[0];
    }

    std::string common = strings[0];
    for (size_t i = 1; i < strings.size(); ++i) {
        const std::string & current = strings[i];
        std::string         temp_common;
        size_t              min_len = std::min(common.length(), current.length());
        for (size_t j = 0; j < min_len; ++j) {
            if (common[common.length() - 1 - j] == current[current.length() - 1 - j]) {
                temp_common = common[common.length() - 1 - j] + temp_common;
            } else {
                break;
            }
        }
        common = temp_common;
    }
    return common;
}

// Helper to find common start pattern in differences
std::string TemplateAnalyzer::find_common_start_pattern(const std::string & diff1,
                                                        const std::string & diff2,
                                                        const std::string & diff3) {
    // Find the common prefix among the three differences
    std::string common    = diff1;
    // Compare with diff2
    size_t      min_len   = std::min(common.length(), diff2.length());
    size_t      match_len = 0;
    for (size_t i = 0; i < min_len; ++i) {
        if (common[i] == diff2[i]) {
            match_len++;
        } else {
            break;
        }
    }
    common = common.substr(0, match_len);

    // Compare with diff3
    min_len   = std::min(common.length(), diff3.length());
    match_len = 0;
    for (size_t i = 0; i < min_len; ++i) {
        if (common[i] == diff3[i]) {
            match_len++;
        } else {
            break;
        }
    }
    common = common.substr(0, match_len);

    // Extract a reasonable length pattern that might be a start marker
    if (common.length() > 20) {
        // Look for meaningful tokens within the common string
        size_t pos = common.find_last_of(" \n\t<[{");
        if (pos != std::string::npos && pos > 0) {
            return common.substr(pos + 1);  // Return after the delimiter
        }
        return common.substr(0, 20);        // Limit length
    }
    return common;
}

// Helper to find common end pattern in differences
std::string TemplateAnalyzer::find_common_end_pattern(const std::string & diff1,
                                                      const std::string & diff2,
                                                      const std::string & diff3) {
    // Find common suffix by comparing from the end
    size_t len1 = diff1.length();
    size_t len2 = diff2.length();
    size_t len3 = diff3.length();

    size_t min_len   = std::min({ len1, len2, len3 });
    size_t match_len = 0;

    for (size_t i = 0; i < min_len; ++i) {
        char c1 = diff1[len1 - 1 - i];
        char c2 = diff2[len2 - 1 - i];
        char c3 = diff3[len3 - 1 - i];

        if (c1 == c2 && c2 == c3) {
            match_len++;
        } else {
            break;
        }
    }

    if (match_len > 0) {
        std::string common_suffix = diff1.substr(len1 - match_len);
        // Reverse to get the actual suffix
        std::reverse(common_suffix.begin(), common_suffix.end());

        // Extract a reasonable length pattern that might be an end marker
        if (common_suffix.length() > 20) {
            // Look for meaningful tokens within the common suffix
            size_t pos = common_suffix.find_first_of(" \n\t>}]");
            if (pos != std::string::npos) {
                return common_suffix.substr(0, pos);  // Return up to the delimiter
            }
            return common_suffix.substr(0, 20);       // Limit length
        }
        return common_suffix;
    }
    return "";
}

// Determine format based on discovered patterns
TemplatePattern::ToolCallFormat TemplateAnalyzer::determine_format_from_patterns(const DiscoveredPattern & patterns) {
    LOG_DBG("=== DETERMINING FORMAT FROM PATTERNS ===");
    LOG_DBG("Checking patterns:");
    LOG_DBG("  function_opener: '%s'", patterns.function_opener.c_str());
    LOG_DBG("  tool_call_opener: '%s'", patterns.tool_call_opener.c_str());
    LOG_DBG("  tool_call_start_marker: '%s'", patterns.tool_call_start_marker.c_str());

    // If all patterns are empty, this template doesn't have tool call markers
    if (patterns.tool_call_opener.empty() && patterns.tool_call_closer.empty() && patterns.function_opener.empty() &&
        patterns.function_closer.empty() && patterns.parameter_opener.empty() && patterns.parameter_closer.empty() &&
        patterns.argument_separator.empty() && patterns.tool_call_start_marker.empty() &&
        patterns.tool_call_end_marker.empty()) {
        LOG_DBG("All patterns are empty - template doesn't support tool calls");
        return TemplatePattern::UNKNOWN;
    }

    // Early check for JSON patterns in tool_call_opener that should override XML patterns
    // This handles cases like Qwen3Next where function_opener is XML but tool_call_opener contains JSON
    if (!patterns.tool_call_opener.empty()) {
        // Check for explicit JSON tool call structure like {"name":
        if (patterns.tool_call_opener.find("{\"name\":") != std::string::npos ||
            patterns.tool_call_opener.find("{\"name\":") != std::string::npos ||
            patterns.tool_call_opener.find("{&quot;name&quot;:") != std::string::npos) {
            LOG_DBG("Detected JSON_NATIVE format from tool_call_opener JSON structure");
            return TemplatePattern::JSON_NATIVE;
        }
    }

    // Look for XML-like patterns in function opener
    if (!patterns.function_opener.empty() && patterns.function_opener.find('<') == 0) {
        LOG_DBG("Detected XML_CONSTRUCTED format from function_opener");
        return TemplatePattern::XML_CONSTRUCTED;
    }

    // Look for JSON-like patterns in function opener
    if (!patterns.function_opener.empty() && patterns.function_opener.find('{') == 0) {
        LOG_DBG("Detected JSON_NATIVE format from function_opener");
        return TemplatePattern::JSON_NATIVE;
    }

    // Check if it's a constructed format (XML-like) in tool call start marker
    if (!patterns.tool_call_start_marker.empty() &&
        (patterns.tool_call_start_marker.find('<') == 0 || patterns.tool_call_start_marker.find('[') == 0)) {
        LOG_DBG("Detected XML_CONSTRUCTED format from tool_call_start_marker");
        return TemplatePattern::XML_CONSTRUCTED;
    }

    // Check for XML-like patterns in tool call opener (for formats like Qwen3-Coder and Seed-OSS)
    if (!patterns.tool_call_opener.empty() && (patterns.tool_call_opener.find("<function=") != std::string::npos ||
                                               patterns.tool_call_opener.find('<') != std::string::npos)) {
        LOG_DBG("Detected XML_CONSTRUCTED format from tool_call_opener");
        return TemplatePattern::XML_CONSTRUCTED;
    }

    // Check for JSON-like patterns in tool call opener (for formats like Nemotron)
    if (!patterns.tool_call_opener.empty() &&
        (patterns.tool_call_opener.find('{') != std::string::npos ||
         patterns.tool_call_opener.find('[') != std::string::npos ||
         patterns.tool_call_opener.find("TOOLCALL") != std::string::npos)) {  // For Nemotron-style
        LOG_DBG("Detected JSON_NATIVE format from tool_call_opener");
        return TemplatePattern::JSON_NATIVE;
    }

    // Check for JSON-like patterns in the overall structure
    if (!patterns.tool_call_start_marker.empty() &&
        (patterns.tool_call_start_marker.find('{') != std::string::npos ||
         patterns.tool_call_start_marker.find('[') != std::string::npos ||
         patterns.tool_call_start_marker.find("TOOLCALL") != std::string::npos)) {
        LOG_DBG("Detected JSON_NATIVE format from tool_call_start_marker structure");
        return TemplatePattern::JSON_NATIVE;
    }

    // Check for TOOLCALL pattern in any of the discovered patterns (for Nemotron-style)
    if (patterns.tool_call_opener.find("TOOLCALL") != std::string::npos ||
        patterns.tool_call_closer.find("TOOLCALL") != std::string::npos ||
        patterns.tool_call_start_marker.find("TOOLCALL") != std::string::npos) {
        LOG_DBG("Detected JSON_NATIVE format from TOOLCALL pattern");
        return TemplatePattern::JSON_NATIVE;
    }

    // Check for JSON structure in tool arguments (for Nemotron-style JSON tool calls)
    if (patterns.tool_call_opener.find("\"name\"") != std::string::npos &&
        patterns.tool_call_opener.find("\"arguments\"") != std::string::npos) {
        LOG_DBG("Detected JSON_NATIVE format from JSON structure in arguments");
        return TemplatePattern::JSON_NATIVE;
    }

    // If we have some patterns but couldn't determine the format, we need to be conservative
    // Only if we have substantial patterns should we assume a specific format
    bool has_substantial_patterns = false;

    // Check if we have meaningful function or tool call markers
    if (!patterns.function_opener.empty() || !patterns.tool_call_start_marker.empty()) {
        has_substantial_patterns = true;
        LOG_DBG("Found substantial patterns but couldn't determine specific format");
    }

    if (!has_substantial_patterns) {
        LOG_DBG("No substantial patterns found - falling back to UNKNOWN");
        return TemplatePattern::UNKNOWN;
    }

    // Default to unknown if we can't determine the format
    LOG_DBG("Could not determine format from patterns - returning UNKNOWN");
    return TemplatePattern::UNKNOWN;
}

// Perform reasoning analysis and update patterns
void TemplateAnalyzer::analyze_reasoning(const minja::chat_template & tmpl, DiscoveredPattern & patterns) {
    LOG_DBG("=== ANALYZING REASONING ===");

    // Method 1: Check if template renders reasoning_content
    json reasoning_msg = {
        { "role",              "assistant" },
        { "content",           "MARKER"    },
        { "reasoning_content", "THOUGHT"   }
    };

    json base_msg = {
        { "role",    "assistant" },
        { "content", "MARKER"    }
    };

    minja::chat_template_inputs inputs;
    inputs.messages       = { reasoning_msg };
    auto reasoning_output = tmpl.apply(inputs);

    inputs.messages  = { base_msg };
    auto base_output = tmpl.apply(inputs);

    // Check if reasoning output is different
    if (reasoning_output != base_output) {
        // Find "THOUGHT" in reasoning output
        size_t thought_pos = reasoning_output.find("THOUGHT");
        if (thought_pos != std::string::npos) {
            // Find prefix and suffix relative to base
            // Since we don't know where reasoning is inserted, we use diff or direct search

            // Assume reasoning is inserted before content
            // base: PREFIX + MARKER + SUFFIX
            // reasoning: PREFIX + START + THOUGHT + END + MARKER + SUFFIX

            // Find MARKER pos in reasoning output
            size_t marker_pos = reasoning_output.find("MARKER");
            if (marker_pos != std::string::npos && marker_pos > thought_pos) {
                // End tag is between THOUGHT and MARKER
                size_t thought_end            = thought_pos + 7;
                patterns.reasoning_end_marker = reasoning_output.substr(thought_end, marker_pos - thought_end);

                // Start tag is before THOUGHT. How far back?
                std::string diff             = find_string_difference(base_output, reasoning_output);
                size_t      diff_thought_pos = diff.find("THOUGHT");
                if (diff_thought_pos != std::string::npos) {
                    patterns.reasoning_start_marker = diff.substr(0, diff_thought_pos);
                }
            }
        }
    }

    // Method 2: Check enable_thinking + add_generation_prompt
    if (patterns.reasoning_start_marker.empty()) {
        minja::chat_template_inputs inputs_prompt;
        inputs_prompt.messages                         = { base_msg };
        inputs_prompt.add_generation_prompt            = true;
        inputs_prompt.extra_context["enable_thinking"] = false;
        auto prompt_no_think                           = tmpl.apply(inputs_prompt);

        inputs_prompt.extra_context["enable_thinking"] = true;
        auto prompt_think                              = tmpl.apply(inputs_prompt);

        if (prompt_think != prompt_no_think) {
            // Find difference at the end
            std::string diff = find_string_difference(prompt_no_think, prompt_think);
            if (!diff.empty()) {
                patterns.reasoning_start_marker = diff;

                // Infer end marker
                if (patterns.reasoning_start_marker[0] == '<') {
                    std::string tag_name    = patterns.reasoning_start_marker.substr(1);
                    size_t      end_bracket = tag_name.find_first_of(" >");
                    if (end_bracket != std::string::npos) {
                        tag_name = tag_name.substr(0, end_bracket);
                    }
                    patterns.reasoning_end_marker = "</" + tag_name + ">";
                } else {
                    // Try [THINK] -> [/THINK]
                    if (patterns.reasoning_start_marker.front() == '[' &&
                        patterns.reasoning_start_marker.back() == ']') {
                        std::string name =
                            patterns.reasoning_start_marker.substr(1, patterns.reasoning_start_marker.length() - 2);
                        patterns.reasoning_end_marker = "[/" + name + "]";
                    }
                }
            }
        }

        // Method 3: If no markers found, check if the generation prompt ends with a tag (MiniMax case)
        if (patterns.reasoning_start_marker.empty()) {
            std::string prompt = prompt_think;  // Reuse calculated prompt
            // Trim whitespace from end for check
            while (!prompt.empty() && std::isspace(static_cast<unsigned char>(prompt.back()))) {
                prompt.pop_back();
            }

            // Look for tag at the end
            size_t last_open_angle    = prompt.rfind('<');
            size_t last_close_angle   = prompt.rfind('>');
            size_t last_open_bracket  = prompt.rfind('[');
            size_t last_close_bracket = prompt.rfind(']');

            size_t last_open  = std::string::npos;
            size_t last_close = std::string::npos;

            if (last_open_angle != std::string::npos && last_close_angle != std::string::npos &&
                last_close_angle > last_open_angle) {
                last_open  = last_open_angle;
                last_close = last_close_angle;
            } else if (last_open_bracket != std::string::npos && last_close_bracket != std::string::npos &&
                       last_close_bracket > last_open_bracket) {
                last_open  = last_open_bracket;
                last_close = last_close_bracket;
            }

            if (last_open != std::string::npos && last_close == prompt.length() - 1) {
                std::string tag                 = prompt.substr(last_open);
                // Assume any tag at end of generation prompt is reasoning start
                patterns.reasoning_start_marker = tag;
                // Infer end marker
                if (tag[0] == '<') {
                    std::string name              = tag.substr(1, tag.length() - 2);
                    patterns.reasoning_end_marker = "</" + name + ">";
                } else if (tag[0] == '[') {
                    std::string name              = tag.substr(1, tag.length() - 2);
                    patterns.reasoning_end_marker = "[/" + name + "]";
                }
            }
        }
    }

    LOG_DBG("Reasoning markers: start='%s', end='%s'", patterns.reasoning_start_marker.c_str(),
            patterns.reasoning_end_marker.c_str());
}

// Simplified methods that just delegate to the differential analysis
std::vector<std::string> TemplateAnalyzer::extract_preserved_tokens(const minja::chat_template & tmpl) {
    auto                     discovered = analyze_by_differential(tmpl);
    std::vector<std::string> tokens;

    // Collect non-empty patterns as preserved tokens
    if (!discovered.tool_call_opener.empty()) {
        tokens.push_back(discovered.tool_call_opener);
    }
    if (!discovered.tool_call_closer.empty()) {
        tokens.push_back(discovered.tool_call_closer);
    }
    if (!discovered.function_opener.empty()) {
        tokens.push_back(discovered.function_opener);
    }
    if (!discovered.function_closer.empty()) {
        tokens.push_back(discovered.function_closer);
    }
    if (!discovered.parameter_opener.empty()) {
        tokens.push_back(discovered.parameter_opener);
    }
    if (!discovered.parameter_closer.empty()) {
        tokens.push_back(discovered.parameter_closer);
    }
    if (!discovered.tool_call_start_marker.empty()) {
        tokens.push_back(discovered.tool_call_start_marker);
    }
    if (!discovered.tool_call_end_marker.empty()) {
        tokens.push_back(discovered.tool_call_end_marker);
    }
    if (!discovered.reasoning_start_marker.empty()) {
        tokens.push_back(discovered.reasoning_start_marker);
    }
    if (!discovered.reasoning_end_marker.empty()) {
        tokens.push_back(discovered.reasoning_end_marker);
    }

    return tokens;
}

common_chat_params UniversalPEGGenerator::generate_parser(const TemplatePattern &         pattern,
                                                          const minja::chat_template &    tmpl,
                                                          const struct templates_params & inputs) {
    common_chat_params data;
    TemplatePattern    local_pattern = pattern;

    try {
        LOG_DBG("=== GENERATING PEG PARSER ===\n");
        LOG_DBG("Pattern format: %d\n", local_pattern.format);
        LOG_DBG("Markers:\n");
        LOG_DBG("  tool_call_start: '%s'\n", local_pattern.special_markers.at("tool_call_start_marker").c_str());
        LOG_DBG("  tool_call_end:   '%s'\n", local_pattern.special_markers.at("tool_call_end_marker").c_str());
        LOG_DBG("  function_opener: '%s'\n", local_pattern.special_markers.at("function_opener").c_str());
        LOG_DBG("  reasoning_start: '%s'\n", local_pattern.special_markers.at("reasoning_start_marker").c_str());
        LOG_DBG("  reasoning_end:   '%s'\n", local_pattern.special_markers.at("reasoning_end_marker").c_str());

        // Calculate prompt first to detect forced thinking
        data.prompt = apply(tmpl, inputs);

        bool        thinking_forced_open = false;
        std::string start_marker         = local_pattern.special_markers.at("reasoning_start_marker");

        // Robust check for forced thinking (trim whitespace)
        std::string prompt_trimmed = data.prompt;
        while (!prompt_trimmed.empty() && std::isspace(static_cast<unsigned char>(prompt_trimmed.back()))) {
            prompt_trimmed.pop_back();
        }

        LOG_DBG("Prompt trimmed end: '%s'", prompt_trimmed.length() > 20 ?
                                                ("..." + prompt_trimmed.substr(prompt_trimmed.length() - 20)).c_str() :
                                                prompt_trimmed.c_str());

        if (!start_marker.empty()) {
            if (string_ends_with(prompt_trimmed, start_marker)) {
                if (!inputs.enable_thinking) {
                    data.prompt += local_pattern.special_markers.at("reasoning_end_marker");
                } else {
                    fprintf(stderr, "Thinking forced open via start marker match\n");
                    thinking_forced_open = true;
                }
            }
        } else if (prompt_trimmed.length() > 2 && prompt_trimmed.back() == '>' && inputs.enable_thinking) {
            fprintf(stderr, "Checking inference for prompt ending with >\n");
            // ... generic inference ...
            size_t open = prompt_trimmed.rfind('<');
            if (open != std::string::npos) {
                std::string tag = prompt_trimmed.substr(open);

                // Check if this looks like a reasoning tag (not just any tag)
                // Reasoning tags typically contain words like "think", "reason", "thought", etc.
                std::string tag_name  = tag.substr(1, tag.length() - 2);
                size_t      space_pos = tag_name.find(' ');
                if (space_pos != std::string::npos) {
                    tag_name = tag_name.substr(0, space_pos);
                }

                // Only infer reasoning if the tag name suggests reasoning
                std::string lower_tag_name = tag_name;
                std::transform(lower_tag_name.begin(), lower_tag_name.end(), lower_tag_name.begin(), ::tolower);

                if (lower_tag_name.find("think") != std::string::npos ||
                    lower_tag_name.find("reason") != std::string::npos ||
                    lower_tag_name.find("thought") != std::string::npos ||
                    lower_tag_name.find("reflect") != std::string::npos) {
                    LOG_DBG("Inferred reasoning tag from prompt: '%s'", tag.c_str());
                    local_pattern.special_markers["reasoning_start_marker"] = tag;
                    start_marker                                            = tag;

                    // Infer end marker
                    std::string end_marker;
                    std::string name  = tag.substr(1, tag.length() - 2);
                    size_t      space = name.find(' ');
                    if (space != std::string::npos) {
                        name = name.substr(0, space);
                    }
                    end_marker = "</" + name + ">";
                    if (tag[0] == '[') {  // Fix: check tag type for end marker inference
                        std::string name_sq = tag.substr(1, tag.length() - 2);
                        end_marker          = "[/" + name_sq + "]";
                    }

                    local_pattern.special_markers["reasoning_end_marker"] = end_marker;
                    fprintf(stderr, "Thinking forced open via inference\n");
                    thinking_forced_open = true;
                } else {
                    LOG_DBG("Tag '%s' does not appear to be a reasoning tag, skipping inference", tag.c_str());
                }
            }
        }

        data.thinking_forced_open = thinking_forced_open;
        fprintf(stderr, "Detailed thinking forced open: %d (marker=%s, prompt_end=%c, enable=%d)\n",
                thinking_forced_open, start_marker.c_str(), prompt_trimmed.empty() ? '?' : prompt_trimmed.back(),
                static_cast<int>(inputs.enable_thinking));

        // ... validation ...
        if (local_pattern.format == TemplatePattern::JSON_NATIVE) {
            if (local_pattern.special_markers.at("tool_call_start_marker").empty() &&
                local_pattern.special_markers.at("tool_call_opener").empty() &&
                local_pattern.special_markers.at("function_opener").empty()) {
                LOG_DBG("JSON_NATIVE format detected but no meaningful markers - falling back to generic parser");
                throw std::runtime_error("Template analysis failed: JSON_NATIVE format without meaningful markers");
            }
        } else if (local_pattern.format == TemplatePattern::XML_CONSTRUCTED) {
            if (local_pattern.special_markers.at("tool_call_start_marker").empty() &&
                local_pattern.special_markers.at("function_opener").empty()) {
                LOG_DBG("XML_CONSTRUCTED format detected but no meaningful markers - falling back to generic parser");
                throw std::runtime_error("Template analysis failed: XML_CONSTRUCTED format without meaningful markers");
            }
        }

        common_peg_arena arena;

        if (local_pattern.format == TemplatePattern::JSON_NATIVE) {
            arena       = build_native_parser(local_pattern, tmpl, inputs, thinking_forced_open);
            data.format = COMMON_CHAT_FORMAT_PEG_NATIVE;
            LOG_DBG("Generated JSON_NATIVE parser successfully");
        } else if (local_pattern.format == TemplatePattern::XML_CONSTRUCTED) {
            arena       = build_constructed_parser(local_pattern, tmpl, inputs, thinking_forced_open);
            data.format = COMMON_CHAT_FORMAT_PEG_CONSTRUCTED;
            LOG_DBG("Generated XML_CONSTRUCTED parser successfully");
        } else {
            // Treat as content only
            arena       = build_chat_peg_native_parser([&](common_chat_peg_native_builder & p) {
                auto content = p.content(p.rest());
                if (thinking_forced_open && !local_pattern.special_markers.at("reasoning_end_marker").empty()) {
                    return p.reasoning_block(
                               p.reasoning(p.until(local_pattern.special_markers.at("reasoning_end_marker"))) +
                               p.literal(local_pattern.special_markers.at("reasoning_end_marker"))) +
                           content;
                }
                return content;
            });
            data.format = COMMON_CHAT_FORMAT_PEG_SIMPLE;
            LOG_DBG("Generated CONTENT_ONLY parser successfully");
        }

        data.parser = arena.save();

        // Determine trigger word for lazy grammar
        std::string trigger_word;
        if (!local_pattern.special_markers.at("tool_call_start_marker").empty()) {
            trigger_word = local_pattern.special_markers.at("tool_call_start_marker");
        } else if (!local_pattern.special_markers.at("function_opener").empty()) {
            trigger_word = local_pattern.special_markers.at("function_opener");
        }

        // Build grammar for tool calls based on discovered patterns
        data.grammar_lazy = inputs.tools.is_array() && !inputs.tools.empty();

        // If thinking forced open, we must constrain from the start (reasoning content)
        if (data.thinking_forced_open) {
            data.grammar_lazy = false;
        }

        if (data.grammar_lazy && !trigger_word.empty()) {
            data.grammar_triggers.push_back({ COMMON_GRAMMAR_TRIGGER_TYPE_WORD, trigger_word });
        }

        // Always build grammar (full or lazy)
        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            // Add tool definitions if any
            if (inputs.tools.is_array()) {
                for (const auto & tool : inputs.tools) {
                    if (!tool.contains("type") || tool.at("type") != "function" || !tool.contains("function")) {
                        continue;
                    }
                    const auto & function = tool.at("function");
                    if (function.contains("parameters")) {
                        auto params = function.at("parameters");
                        builder.resolve_refs(params);
                    }
                }
            }
            arena.build_grammar(builder, data.grammar_lazy);
        });

        // Set preserved tokens - include all discovered markers plus trigger words
        std::vector<std::string> preserved;
        
        // Add trigger word if present
        if (!trigger_word.empty()) {
            preserved.push_back(trigger_word);
        }
        
        // Add all non-empty special markers
        for (const auto & [key, value] : local_pattern.special_markers) {
            if (!value.empty()) {
                // Avoid duplicates
                if (std::find(preserved.begin(), preserved.end(), value) == preserved.end()) {
                    preserved.push_back(value);
                }
            }
        }
        
        // Add any from the original pattern
        for (const auto & token : local_pattern.preserved_tokens) {
            if (!token.empty() && std::find(preserved.begin(), preserved.end(), token) == preserved.end()) {
                preserved.push_back(token);
            }
        }
        
        data.preserved_tokens = preserved;

        // data.prompt was already set

        LOG_DBG("=== PEG PARSER GENERATION COMPLETED ===");

    } catch (const std::exception & e) {
        LOG_DBG("Automatic parser generation failed: %s", e.what());
        throw;
    }

    return data;
}

common_peg_arena UniversalPEGGenerator::build_native_parser(const TemplatePattern &         pattern,
                                                            const minja::chat_template &    tmpl,
                                                            const struct templates_params & inputs,
                                                            bool                            thinking_forced_open) {
    GGML_UNUSED(tmpl);

    auto has_tools = inputs.tools.is_array() && !inputs.tools.empty();

    auto parser = build_chat_peg_native_parser([&](common_chat_peg_native_builder & p) {
        // Reasoning parser - only include if we have valid reasoning markers
        auto reasoning       = p.eps();
        auto reasoning_start = pattern.special_markers.at("reasoning_start_marker");
        auto reasoning_end   = pattern.special_markers.at("reasoning_end_marker");
        auto tool_call_start = pattern.special_markers.at("tool_call_start_marker");
        auto tool_call_end   = pattern.special_markers.at("tool_call_end_marker");
        if ((inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE || thinking_forced_open) &&
            !reasoning_start.empty() && !reasoning_end.empty()) {
            if (thinking_forced_open) {
                LOG_DBG("Building mandatory reasoning block with end marker '%s'", reasoning_start.c_str());
                reasoning = p.reasoning(p.until(reasoning_end)) + reasoning_end;
            } else {
                LOG_DBG("Building optional reasoning block with start '%s' and end '%s'", reasoning_start.c_str(),
                        reasoning_end.c_str());
                reasoning = p.optional(reasoning_start + p.reasoning(p.until(reasoning_end)) + reasoning_end);
            }
        }

        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            if (pattern.format == TemplatePattern::JSON_NATIVE) {
                bool force_calls = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;
                auto tool_calls  = p.standard_json_tools(tool_call_start, tool_call_end, inputs.tools,
                                                         inputs.parallel_tool_calls, force_calls);

                auto content_before_tools = tool_call_start.empty() ? p.eps() : p.content(p.until(tool_call_start));

                return p.sequence({ reasoning, content_before_tools, p.space(), tool_calls, p.space(), p.end() });
            }
            throw std::runtime_error("Native parser requires JSON tool format");
        }

        return p.sequence({ reasoning, p.rule("content", p.content(p.rest())), p.end() });
    });

    return parser;
}

common_peg_arena UniversalPEGGenerator::build_constructed_parser(const TemplatePattern &         pattern,
                                                                 const minja::chat_template &    tmpl,
                                                                 const struct templates_params & inputs,
                                                                 bool                            thinking_forced_open) {
    (void) tmpl;    // Suppress unused parameter warning
    (void) inputs;  // Suppress unused parameter warning

    auto parser = build_chat_peg_constructed_parser([&](common_chat_peg_constructed_builder & p) {
        auto reasoning_start = pattern.special_markers.at("reasoning_start_marker");
        auto reasoning_end   = pattern.special_markers.at("reasoning_end_marker");
        auto tool_call_start = pattern.special_markers.at("tool_call_start_marker");

        // Reasoning parser
        auto reasoning = p.eps();
        if ((inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE || thinking_forced_open) &&
            !reasoning_start.empty() && !reasoning_end.empty()) {
            if (thinking_forced_open) {
                LOG_DBG("Building mandatory reasoning block for constructed parser");
                reasoning = p.rule("reasoning",
                                   p.reasoning_block(p.reasoning(p.until(reasoning_end)) + p.literal(reasoning_end)));
            } else {
                LOG_DBG("Building optional reasoning block for constructed parser");
                reasoning = p.optional(p.rule(
                    "reasoning", p.reasoning_block(p.literal(reasoning_start) + p.reasoning(p.until(reasoning_end)) +
                                                   p.literal(reasoning_end))));
            }
        }

        bool force_calls = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        auto tool_calls  = p.standard_constructed_tools(pattern.special_markers, inputs.tools,
                                                        inputs.parallel_tool_calls, force_calls);

        if (!tool_call_start.empty()) {
            return reasoning + p.space() + p.tag_with_safe_content("content", tool_call_start, tool_calls) +
                   p.optional(p.rule("content", p.content(p.rest())));
        }

        return reasoning + p.rule("content", p.content(p.rest()));
    });

    return parser;
}
