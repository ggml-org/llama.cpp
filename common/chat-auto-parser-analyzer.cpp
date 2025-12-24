#include "chat-auto-parser.h"
#include "chat-auto-parser-helpers.h"

#include "log.h"

#include <minja/chat-template.hpp>
#include <minja/minja.hpp>

using json = nlohmann::ordered_json;

static void foreach_function(const json & tools, const std::function<void(const json &)> & callback) {
    if (tools.is_array()) {
        for (const auto & tool : tools) {
            if (tool.contains("type") && tool.at("type") == "function" && tool.contains("function")) {
                callback(tool);
            }
        }
    }
}

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

static std::string extract_json_field_name(const std::string &              opener,
                                           const std::string &              default_name,
                                           const std::vector<std::string> & candidates) {
    for (const auto & candidate : candidates) {
        std::string pattern = "\"" + candidate + "\"";
        if (opener.find(pattern) != std::string::npos) {
            LOG_DBG("Found JSON field name '%s' in opener\n", candidate.c_str());
            return candidate;
        }
    }
    return default_name;
}

static std::string find_string_difference(const std::string & base, const std::string & extended) {
    size_t common_prefix = 0;
    while (common_prefix < base.length() && common_prefix < extended.length() &&
           base[common_prefix] == extended[common_prefix]) {
        common_prefix++;
    }
    return extended.substr(common_prefix);
}

// ============================================================================
// TemplateAnalyzer Main Analysis Methods
// ============================================================================

TemplatePattern TemplateAnalyzer::analyze_template(const minja::chat_template & tmpl) {
    TemplatePattern pattern;

    auto discovered = analyze_by_differential(tmpl);

    pattern.format = determine_format_from_patterns(discovered);

    if (!discovered.reasoning_start_marker.empty()) {
        pattern.has_reasoning_support = true;
    }

    trim_whitespace(discovered.reasoning_start_marker);
    trim_whitespace(discovered.reasoning_end_marker);

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
        { "reasoning_end_marker",   discovered.reasoning_end_marker   },
        { "content_start_marker",   discovered.content_start_marker   },
        { "content_end_marker",     discovered.content_end_marker     },
        { "tool_name_field",        discovered.tool_name_field        },
        { "tool_args_field",        discovered.tool_args_field        },
        { "tool_id_field",          discovered.tool_id_field          }
    };

    return pattern;
}

DiscoveredPattern TemplateAnalyzer::analyze_by_differential(const minja::chat_template & tmpl) {
    DiscoveredPattern patterns;

    try {
        LOG_DBG("=== STARTING TEMPLATE DIFFERENTIAL ANALYSIS ===\n");

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
                    LOG_DBG("Heuristic: Overriding format to JSON_NATIVE due to JSON signature\n");
                    fmt = TemplatePattern::JSON_NATIVE;
                }

                size_t ff1 = t_out.rfind("test_function_name");
                size_t fbr = (ff1 != std::string::npos) ? t_out.rfind('{', ff1) : std::string::npos;

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

                if (p.tool_call_end_marker.empty()) {
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
                }

                LOG_DBG("Heuristic markers refined: start='%s', end='%s'\n", p.tool_call_start_marker.c_str(),
                        p.tool_call_end_marker.c_str());
            }
        };

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
        LOG_DBG("Base output: %s\n", base_output.c_str());

        inputs.messages   = { tool_msg1 };
        auto tool1_output = tmpl.apply(inputs);
        LOG_DBG("Tool1 output: %s\n", tool1_output.c_str());

        inputs.messages   = { tool_msg2 };
        auto tool2_output = tmpl.apply(inputs);
        LOG_DBG("Tool2 output: %s\n", tool2_output.c_str());

        inputs.messages   = { tool_msg3 };
        auto tool3_output = tmpl.apply(inputs);
        LOG_DBG("Tool3 output: %s\n", tool3_output.c_str());

        std::string tool1_diff = find_string_difference(base_output, tool1_output);
        std::string tool2_diff = find_string_difference(base_output, tool2_output);
        std::string tool3_diff = find_string_difference(base_output, tool3_output);

        LOG_DBG("Tool1 diff length: %zu\n", tool1_diff.length());
        LOG_DBG("Tool2 diff length: %zu\n", tool2_diff.length());
        LOG_DBG("Tool3 diff length: %zu\n", tool3_diff.length());

        if (tool1_diff.empty() && tool2_diff.empty() && tool3_diff.empty()) {
            LOG_DBG("All diffs are empty, trying alternative approach for tool call detection\n");
            json alternative_base_msg = {
                { "role",    "assistant" },
                { "content", "MARKER"    }
            };
            json alternative_tool_msg = {
                { "role",       "assistant"                                                                         },
                { "content",    ""                                                                                  },
                { "tool_calls",
                 json::array({ { { "type", "function" },
                                  { "function",
                                    { { "name", "test_function_name" },
                                      { "arguments",
                                        json::object({ { "param1", "value1" }, { "param2", "value2" } }) } } } } }) }
            };

            auto get_alternative_diffs = [&](bool gen_prompt, std::string & b_out, std::string & t1_out,
                                             std::string & t1_diff, std::string & t2_diff, std::string & t3_diff) {
                minja::chat_template_inputs b_inputs;
                b_inputs.tools                 = inputs.tools;
                b_inputs.messages              = { alternative_base_msg };
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

            std::string n_base_false;
            std::string n_t1_out_false;
            std::string n_t1_diff_false;
            std::string n_t2_diff_false;
            std::string n_t3_diff_false;
            std::string n_base_true;
            std::string n_t1_out_true;
            std::string n_t1_diff_true;
            std::string n_t2_diff_true;
            std::string n_t3_diff_true;

            bool false_ok = get_alternative_diffs(false, n_base_false, n_t1_out_false, n_t1_diff_false, n_t2_diff_false,
                                                  n_t3_diff_false);
            bool true_ok =
                get_alternative_diffs(true, n_base_true, n_t1_out_true, n_t1_diff_true, n_t2_diff_true, n_t3_diff_true);

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

        patterns = extract_patterns_from_differences(tool1_diff, tool2_diff, tool3_diff);

        analyze_reasoning(tmpl, patterns);

        analyze_content_markers(tmpl, patterns);

        LOG_DBG("=== DISCOVERED PATTERNS ===\n");
        LOG_DBG("tool_call_opener: '%s'\n", patterns.tool_call_opener.c_str());
        LOG_DBG("tool_call_closer: '%s'\n", patterns.tool_call_closer.c_str());
        LOG_DBG("function_opener: '%s'\n", patterns.function_opener.c_str());
        LOG_DBG("function_closer: '%s'\n", patterns.function_closer.c_str());
        LOG_DBG("parameter_opener: '%s'\n", patterns.parameter_opener.c_str());
        LOG_DBG("parameter_closer: '%s'\n", patterns.parameter_closer.c_str());
        LOG_DBG("argument_separator: '%s'\n", patterns.argument_separator.c_str());
        LOG_DBG("tool_call_start_marker: '%s'\n", patterns.tool_call_start_marker.c_str());
        LOG_DBG("tool_call_end_marker: '%s'\n", patterns.tool_call_end_marker.c_str());

        auto detected_format = determine_format_from_patterns(patterns);

        refine_json_native(patterns, detected_format, base_output, tool1_output, tool1_diff);

        if (!patterns.tool_call_start_marker.empty() || !patterns.tool_call_opener.empty()) {
            std::vector<std::tuple<std::string, std::string, std::string, std::string>> thinking_patterns = {
                { "<|START_THINKING|>", "THINKING|>", "<|END_THINKING|>", "<|START_THINKING|>" },
                { "<think>",            "think>",     "</think>",         "<think>"            },
                { "[THINKING]",         "THINKING]",  "[/THINKING]",      "[THINKING]"         },
                { "<thinking>",         "thinking>",  "</thinking>",      "<thinking>"         }
            };

            for (const auto & [full_start, partial_start, end_tag, actual_start] : thinking_patterns) {
                bool found_in_marker = false;

                if (!patterns.tool_call_start_marker.empty()) {
                    bool has_start = (patterns.tool_call_start_marker.find(full_start) != std::string::npos ||
                                      patterns.tool_call_start_marker.find(partial_start) != std::string::npos);
                    bool has_end   = patterns.tool_call_start_marker.find(end_tag) != std::string::npos;
                    if (has_start && has_end) {
                        found_in_marker = true;
                    }
                }

                if (!found_in_marker && !patterns.tool_call_opener.empty()) {
                    bool has_start = (patterns.tool_call_opener.find(full_start) != std::string::npos ||
                                      patterns.tool_call_opener.find(partial_start) != std::string::npos);
                    bool has_end   = patterns.tool_call_opener.find(end_tag) != std::string::npos;
                    if (has_start && has_end) {
                        found_in_marker = true;
                    }
                }

                if (found_in_marker) {
                    LOG_DBG("Post-refinement: Detected closed reasoning section in tool call markers\n");
                    LOG_DBG("Clearing potentially incorrect reasoning markers (was: start='%s', end='%s')\n",
                            patterns.reasoning_start_marker.c_str(), patterns.reasoning_end_marker.c_str());

                    patterns.reasoning_start_marker = actual_start;
                    patterns.reasoning_end_marker   = end_tag;
                    LOG_DBG("Set reasoning markers to: start='%s', end='%s'\n", patterns.reasoning_start_marker.c_str(),
                            patterns.reasoning_end_marker.c_str());

                    strip_markers(patterns.tool_call_start_marker, partial_start, end_tag);
                    strip_markers(patterns.tool_call_opener, partial_start, end_tag);
                    LOG_DBG("After stripping: tool_call_start_marker='%s'\n", patterns.tool_call_start_marker.c_str());

                    break;
                }
            }
        }

        LOG_DBG("=== DETECTED FORMAT ===\n");
        LOG_DBG("Format: %s\n", TemplatePattern::format_to_str(detected_format));

        if (detected_format != TemplatePattern::UNKNOWN) {
            bool has_meaningful_markers = false;

            if (detected_format == TemplatePattern::JSON_NATIVE) {
                has_meaningful_markers =
                    (!patterns.tool_call_start_marker.empty() &&
                     (patterns.tool_call_start_marker.find('{') != std::string::npos ||
                      patterns.tool_call_start_marker.find('[') != std::string::npos)) ||
                    (!patterns.function_opener.empty() &&
                     patterns.function_opener.find("{\"name\"") != std::string::npos);
            } else if (detected_format == TemplatePattern::XML_CONSTRUCTED) {
                has_meaningful_markers =
                    !patterns.tool_call_start_marker.empty() &&
                    (patterns.tool_call_start_marker.find('<') != std::string::npos ||
                     patterns.function_opener.find('<') != std::string::npos);
            }

            if (!has_meaningful_markers) {
                LOG_DBG(
                    "Detected format %s but no meaningful tool call markers found - falling back to generic parser\n",
                    TemplatePattern::format_to_str(detected_format));
                detected_format = TemplatePattern::UNKNOWN;
            }
        }

        if (detected_format == TemplatePattern::UNKNOWN) {
            if (!tool1_diff.empty()) {
                LOG_DBG("Format is still UNKNOWN but we have diffs - assuming content only\n");
            } else {
                LOG_DBG("No tool call patterns detected - assuming content only\n");
            }
        }

    } catch (const std::exception & e) {
        LOG_DBG("Template differential analysis failed: %s\n", e.what());
    }

    LOG_DBG("=== ENDING TEMPLATE DIFFERENTIAL ANALYSIS ===\n");

    return patterns;
}

DiscoveredPattern TemplateAnalyzer::extract_patterns_from_differences(const std::string & tool1_diff,
                                                                      const std::string & tool2_diff,
                                                                      const std::string & tool3_diff) {
    LOG_DBG("=== EXTRACTING PATTERNS FROM DIFFERENCES ===\n");

    DiscoveredPattern patterns;

    size_t func1_pos  = tool1_diff.find("test_function_name");
    size_t func2_pos  = tool2_diff.find("test_function_name");
    size_t func3_pos1 = tool3_diff.find("test_function_name");
    size_t func3_pos2 = tool3_diff.find("another_test_function");

    LOG_DBG("Function name positions - func1_pos: %zu, func2_pos: %zu, func3_pos1: %zu, func3_pos2: %zu\n", func1_pos,
            func2_pos, func3_pos1, func3_pos2);

    if (func1_pos != std::string::npos && func2_pos != std::string::npos) {
        LOG_DBG("Found function names, extracting patterns...\n");

        patterns.tool_call_opener = tool1_diff.substr(0, func1_pos);

        patterns.tool_name_field = extract_json_field_name(patterns.tool_call_opener,
                                                           "name",  // default
                                                           { "tool_name", "name", "function_name", "function" });

        patterns.tool_args_field = extract_json_field_name(
            patterns.tool_call_opener + tool1_diff.substr(func1_pos),  // Check full context including parameters
            "arguments",                                               // default
            { "parameters", "arguments", "args", "params", "input" });

        patterns.tool_id_field = extract_json_field_name(patterns.tool_call_opener,
                                                         "",  // default (empty means no ID field)
                                                         { "tool_call_id", "tool_id", "id", "call_id" });

        LOG_DBG("Extracted JSON field names: name_field='%s', args_field='%s', id_field='%s'\n",
                patterns.tool_name_field.c_str(), patterns.tool_args_field.c_str(), patterns.tool_id_field.c_str());

        size_t param1_pos       = tool2_diff.find("\"param1\"");
        bool   param_has_quotes = (param1_pos != std::string::npos);

        size_t param2_pos = tool2_diff.find("\"param2\"");
        size_t value1_pos = tool2_diff.find("\"value1\"");
        size_t value2_pos = tool2_diff.find("\"value2\"");

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




            size_t      search_start = (param1_pos > 20) ? param1_pos - 20 : 0;
            std::string pre_param    = tool2_diff.substr(search_start, param1_pos - search_start);

            size_t delim_pos = pre_param.find_last_of('\n');
            if (delim_pos == std::string::npos) {
                delim_pos = pre_param.find_last_of('>');
            }

            if (delim_pos != std::string::npos) {
                patterns.parameter_key_prefix = pre_param.substr(delim_pos + 1);
            } else {

                size_t start_marker = pre_param.find_last_of("<{[ \"");
                if (start_marker != std::string::npos) {
                    patterns.parameter_key_prefix = pre_param.substr(start_marker);
                } else {
                    patterns.parameter_key_prefix = pre_param;
                }
            }

            trim_whitespace(patterns.parameter_key_prefix);

            size_t key_end = param1_pos + std::string("param1").length();

            if (value1_pos > key_end) {
                patterns.parameter_key_suffix = tool2_diff.substr(key_end, value1_pos - key_end);
                // Don't trim whitespace from parameter_key_suffix as it may contain structural newlines
                // that are needed to properly separate parameter keys from values in XML-style formats
            }

            size_t value1_len = 6;  // "value1"
            (void) value1_len;
            size_t actual_val1_pos = value1_pos;
            (void) actual_val1_pos;
            if (value1_pos > 0 && tool2_diff[value1_pos - 1] == '"' && value1_pos + 6 < tool2_diff.length() &&
                tool2_diff[value1_pos + 6] == '"') {



            }
            size_t end_of_val1 = value1_pos + 6;

            if (param2_pos != std::string::npos && value2_pos != std::string::npos) {
                if (param2_pos > end_of_val1) {
                    std::string gap = tool2_diff.substr(end_of_val1, param2_pos - end_of_val1);


                    if (!patterns.parameter_key_prefix.empty() &&
                        gap.length() >= patterns.parameter_key_prefix.length() &&
                        gap.substr(gap.length() - patterns.parameter_key_prefix.length()) ==
                            patterns.parameter_key_prefix) {
                        std::string closer_and_sep =
                            gap.substr(0, gap.length() - patterns.parameter_key_prefix.length());


                        size_t last_non_sep = closer_and_sep.find_last_not_of(" \n\t,");
                        if (last_non_sep != std::string::npos && last_non_sep < closer_and_sep.length() - 1) {
                            patterns.parameter_closer   = closer_and_sep.substr(0, last_non_sep + 1);
                            patterns.argument_separator = closer_and_sep.substr(last_non_sep + 1);
                        } else {
                            patterns.parameter_closer   = closer_and_sep;
                            patterns.argument_separator = "";
                        }
                    } else {
                        patterns.parameter_closer = gap;
                    }

                    trim_whitespace(patterns.parameter_closer);
                }
            } else {
            }
        }

        const std::string & func_context = tool1_diff;
        size_t              open_pos     = func_context.rfind('<', func1_pos);
        if (open_pos != std::string::npos && open_pos < func1_pos) {
            size_t close_pos = func_context.find('>', open_pos);
            if (close_pos != std::string::npos && close_pos < func1_pos) {
                patterns.function_opener = func_context.substr(open_pos, close_pos - open_pos + 1);
            } else {
                patterns.function_opener = func_context.substr(open_pos, func1_pos - open_pos);
            }
        } else {
            size_t start_pos = 0;
            for (int i = (int) func1_pos - 1; i >= 0; i--) {
                if (func_context[i] == '{' || func_context[i] == '[' || func_context[i] == '(' ||
                    func_context[i] == '<') {
                    start_pos                = i;
                    patterns.function_opener = func_context.substr(start_pos, func1_pos - start_pos);
                    break;
                }
            }
        }


        size_t func_name_end = func1_pos + std::string("test_function_name").length();

        if (func_name_end < func_context.length()) {
            char next_char = func_context[func_name_end];
            if (next_char == '>' || next_char == ']' || next_char == '}') {
                patterns.function_name_suffix = std::string(1, next_char);
            } else if (next_char == '"') {
                if (func_name_end + 1 < func_context.length() && func_context[func_name_end + 1] == '>') {
                    patterns.function_name_suffix = "\">";
                } else {
                    patterns.function_name_suffix = "\"";
                }
            }
        }
        LOG_DBG("Extracted function_name_suffix: '%s'\n", patterns.function_name_suffix.c_str());

        size_t search_start = func_name_end;
        if (!patterns.function_name_suffix.empty()) {
            search_start += patterns.function_name_suffix.length();
        }
        patterns.function_closer = find_closing_pattern(func_context, search_start);

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

        if (patterns.function_closer.empty()) {
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

        LOG_DBG("After processing function context:\n");
        LOG_DBG("  function_opener: '%s'\n", patterns.function_opener.c_str());
        LOG_DBG("  function_closer: '%s'\n", patterns.function_closer.c_str());
        LOG_DBG("  tool_call_opener: '%s'\n", patterns.tool_call_opener.c_str());

        if (patterns.function_opener.length() > 0 &&
            patterns.tool_call_opener.length() > patterns.function_opener.length()) {
            if (!patterns.tool_call_opener.empty() && !patterns.function_opener.empty()) {
                size_t opener_start = patterns.tool_call_opener.length() - patterns.function_opener.length();

                if (opener_start > 0) {
                    std::string before_func = patterns.tool_call_opener.substr(0, opener_start);

                    size_t last_bracket   = before_func.find_last_of('[');
                    size_t tool_obj_brace = std::string::npos;
                    if (last_bracket != std::string::npos && last_bracket + 1 < before_func.length()) {
                        tool_obj_brace = before_func.find('{', last_bracket + 1);
                    }

                    if (tool_obj_brace != std::string::npos) {
                        patterns.tool_call_start_marker = before_func.substr(0, tool_obj_brace);
                    } else if (last_bracket != std::string::npos) {
                        patterns.tool_call_start_marker = before_func.substr(0, last_bracket + 1);
                    } else {
                        patterns.tool_call_start_marker = before_func;
                    }
                } else {
                    patterns.tool_call_start_marker = patterns.tool_call_opener;
                }
            }
        } else {
            patterns.tool_call_start_marker = find_tool_call_start(tool1_diff);
        }

        patterns.tool_call_end_marker = find_tool_call_end(func_context, func1_pos);

        if (!patterns.tool_call_end_marker.empty()) {
            size_t first = patterns.tool_call_end_marker.find_first_not_of(" \n\t");
            size_t last  = patterns.tool_call_end_marker.find_last_not_of(" \n\t");
            if (first != std::string::npos && last != std::string::npos) {
                patterns.tool_call_end_marker = patterns.tool_call_end_marker.substr(first, (last - first + 1));
            }
        }

        LOG_DBG("After finding tool call markers:\n");
        LOG_DBG("  tool_call_start_marker: '%s'\n", patterns.tool_call_start_marker.c_str());
        LOG_DBG("  tool_call_end_marker: '%s'\n", patterns.tool_call_end_marker.c_str());

        if (patterns.tool_call_opener.empty()) {
            patterns.tool_call_opener = infer_tool_call_opener(tool1_diff, tool2_diff, tool3_diff);

            patterns.tool_name_field = extract_json_field_name(patterns.tool_call_opener,
                                                               "name",  // default
                                                               { "tool_name", "name", "function_name", "function" });

            patterns.tool_args_field =
                extract_json_field_name(patterns.tool_call_opener + tool1_diff,
                                        "arguments",  // default
                                        { "parameters", "arguments", "args", "params", "input" });

            patterns.tool_id_field = extract_json_field_name(patterns.tool_call_opener,
                                                             "",  // default (empty means no ID field)
                                                             { "tool_call_id", "tool_id", "id", "call_id" });

            LOG_DBG("Extracted JSON field names (inferred): name_field='%s', args_field='%s', id_field='%s'\n",
                    patterns.tool_name_field.c_str(), patterns.tool_args_field.c_str(), patterns.tool_id_field.c_str());
            if (func1_pos != std::string::npos && patterns.tool_call_opener.length() > func1_pos) {
                patterns.tool_call_opener = patterns.tool_call_opener.substr(0, func1_pos);
            }
        }
        if (patterns.tool_call_closer.empty()) {
            patterns.tool_call_closer = infer_tool_call_closer(tool1_diff, tool2_diff, tool3_diff);
        }

        if (patterns.tool_call_start_marker.empty()) {
            patterns.tool_call_start_marker = find_common_start_pattern(tool1_diff, tool2_diff, tool3_diff);
        }

        fprintf(stderr, "TRUNCATE DEBUG: func1_pos=%zu, marker_len=%zu, marker='%s'\n", func1_pos,
                patterns.tool_call_start_marker.length(), patterns.tool_call_start_marker.c_str());

        if (func1_pos != std::string::npos && patterns.tool_call_start_marker.length() > func1_pos) {
            std::string candidate = patterns.tool_call_start_marker.substr(0, func1_pos);
            fprintf(stderr, "TRUNCATE DEBUG: candidate='%s'\n", candidate.c_str());

            size_t last_opener = candidate.find_last_of("{[");
            fprintf(stderr, "TRUNCATE DEBUG: last_opener=%zu\n", last_opener);

            if (last_opener != std::string::npos) {
                patterns.tool_call_start_marker = candidate.substr(0, last_opener);
                fprintf(stderr, "TRUNCATE DEBUG: after truncate='%s'\n", patterns.tool_call_start_marker.c_str());
            } else {
                patterns.tool_call_start_marker = candidate;
            }
        }

        if (!patterns.tool_call_start_marker.empty()) {
            size_t first = patterns.tool_call_start_marker.find_first_not_of(" \n\t\r");
            size_t last  = patterns.tool_call_start_marker.find_last_not_of(" \n\t\r");
            if (first != std::string::npos && last != std::string::npos) {
                patterns.tool_call_start_marker = patterns.tool_call_start_marker.substr(first, (last - first + 1));
            }
        }

        if (patterns.tool_call_end_marker.empty()) {
            patterns.tool_call_end_marker = find_common_end_pattern(tool1_diff, tool2_diff, tool3_diff);
        }
    }

    return patterns;
}

std::string TemplateAnalyzer::find_closing_pattern(const std::string & diff, size_t func_pos) {
    std::vector<std::string> closers = { "</", "}", "]", ">", " " };

    std::string best_pattern;
    size_t      best_pos = std::string::npos;

    for (const auto & pattern : closers) {
        size_t pos = diff.find(pattern, func_pos);
        if (pos != std::string::npos) {
            if (pos < best_pos) {
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

    std::vector<std::string> start_patterns = { "<", "[", "{", "call", "func", "tool", "TOOL" };
    for (const auto & pattern : start_patterns) {
        size_t pos = diff.find(pattern);
        if (pos < 5) {
            if (pattern == "<") {
                size_t end_pos = diff.find('>', pos);
                if (end_pos != std::string::npos) {
                    return diff.substr(pos, end_pos - pos + 1);
                }
            }
            if (pattern == "[" || pattern == "{") {
                size_t chunk_len = std::min(diff.length() - pos, (size_t) 60);
                return diff.substr(pos, chunk_len);
            }

            size_t end_pos = diff.find_first_of(">]} \n", pos);
            if (end_pos != std::string::npos) {
                if (diff[end_pos] == '>' || diff[end_pos] == ']' || diff[end_pos] == '}') {
                    return diff.substr(pos, end_pos - pos + 1);
                }
                return diff.substr(pos, end_pos - pos);
            }
            return diff.substr(pos, pattern.length());
        }
    }
    return "";
}

std::string TemplateAnalyzer::find_tool_call_end(const std::string & diff, size_t func_pos) {
    char        opener_char = 0;
    std::string start_tag_name;

    std::string openers         = "[{<";
    size_t      last_opener_pos = std::string::npos;
    for (char c : openers) {
        size_t p = diff.rfind(c, func_pos);
        if (p != std::string::npos) {
            if (last_opener_pos == std::string::npos || p > last_opener_pos) {
                last_opener_pos = p;
                opener_char     = c;
            }
        }
    }

    size_t unclosed_bracket = diff.rfind('[', func_pos);
    if (unclosed_bracket != std::string::npos) {
        size_t closer = diff.find(']', unclosed_bracket);
        if (closer == std::string::npos || closer > func_pos) {
            opener_char = '[';
        }
    }

    if (opener_char == '<') {
        size_t tag_start = diff.find('<', last_opener_pos);
        if (tag_start != std::string::npos) {
            size_t tag_end = diff.find_first_of(" >\n", tag_start);
            if (tag_end != std::string::npos) {
                start_tag_name = diff.substr(tag_start + 1, tag_end - (tag_start + 1));
            }
        }
    }

    if (!start_tag_name.empty()) {
        std::string expected_closer = "</" + start_tag_name + ">";
        size_t      pos             = diff.find(expected_closer, func_pos);
        if (pos != std::string::npos) {
            if (opener_char == '[') {
                size_t bracket_pos = diff.rfind(']', pos);
                if (bracket_pos != std::string::npos && bracket_pos > func_pos) {
                    return diff.substr(bracket_pos, (pos + expected_closer.length()) - bracket_pos);
                }
            }
            return expected_closer;
        }
    }

    std::vector<std::string> end_patterns = { "</", "]", "}", ">", "\n", " " };
    std::string              best_pattern;
    size_t                   best_pos = std::string::npos;

    auto is_structural = [](const std::string & s) {
        if (s.empty()) {
            return false;
        }
        return s[0] == ']' || s[0] == '}' || s[0] == '>' || (s.size() >= 2 && s.substr(0, 2) == "</");
    };

    for (const auto & pattern : end_patterns) {
        size_t pos = diff.find(pattern, func_pos);
        if (pos == std::string::npos) {
            continue;
        }

        bool current_is_struct = is_structural(pattern);
        bool best_is_struct    = is_structural(best_pattern);

        bool better = false;
        if (best_pattern.empty()) {
            better = true;
        } else if (pos < best_pos) {
            better = !(best_is_struct && !current_is_struct) &&
                     !(opener_char == '[' && best_pattern[0] == ']' && pattern[0] == '}');
        } else {
            if (!best_is_struct && current_is_struct && pos < best_pos + 400) {
                better = true;
            } else if (best_is_struct && current_is_struct && opener_char == '[' && pattern[0] == ']' &&
                       best_pattern[0] == '}') {
                if (pos < best_pos + 100) {  // Object ends then list ends closely after
                    better = true;
                }
            }
        }

        if (better) {
            best_pattern = pattern;
            best_pos     = pos;

            if (current_is_struct && (pattern == "]" || pattern == "}")) {
                size_t tag_start = diff.find('<', best_pos + pattern.length());
                if (tag_start != std::string::npos && tag_start < best_pos + pattern.length() + 5) {
                    size_t tag_end = diff.find('>', tag_start);
                    if (tag_end != std::string::npos) {
                        best_pattern = diff.substr(best_pos, tag_end - best_pos + 1);
                    }
                }
            }
        }
    }

    return best_pattern;
}

std::string TemplateAnalyzer::infer_tool_call_opener(const std::string & diff1,
                                                     const std::string & diff2,
                                                     const std::string & diff3) {
    std::vector<std::string> differences    = { diff1, diff2, diff3 };
    std::string              common_pattern = find_common_prefix(differences);
    return common_pattern;
}

std::string TemplateAnalyzer::infer_tool_call_closer(const std::string & diff1,
                                                     const std::string & diff2,
                                                     const std::string & diff3) {
    std::vector<std::string> differences    = { diff1, diff2, diff3 };
    std::string              common_pattern = find_common_suffix_generic(differences);
    return common_pattern;
}

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
                temp_common.insert(0, 1, common[common.length() - 1 - j]);
            } else {
                break;
            }
        }
        common = temp_common;
    }
    return common;
}

std::string TemplateAnalyzer::find_common_start_pattern(const std::string & diff1,
                                                        const std::string & diff2,
                                                        const std::string & diff3) {
    std::vector<std::string> diffs = { diff1, diff2, diff3 };
    return find_common_substring_limited(diffs, 20, " \n\t<[{");
}

std::string TemplateAnalyzer::find_common_end_pattern(const std::string & diff1,
                                                      const std::string & diff2,
                                                      const std::string & diff3) {
    std::vector<std::string> diffs = { diff1, diff2, diff3 };
    std::string common_suffix = find_common_suffix_generic(diffs);

    if (common_suffix.length() > 20) {
        size_t pos = common_suffix.find_first_of(" \n\t>}]");
        if (pos != std::string::npos) {
            return common_suffix.substr(0, pos);  // Return up to the delimiter
        }
        return common_suffix.substr(0, 20);       // Limit length
    }
    return common_suffix;
}

TemplatePattern::ToolCallFormat TemplateAnalyzer::determine_format_from_patterns(const DiscoveredPattern & patterns) {
    LOG_DBG("=== DETERMINING FORMAT FROM PATTERNS ===\n");
    LOG_DBG("Checking patterns:\n");
    LOG_DBG("  function_opener: '%s'\n", patterns.function_opener.c_str());
    LOG_DBG("  tool_call_opener: '%s'\n", patterns.tool_call_opener.c_str());
    LOG_DBG("  tool_call_start_marker: '%s'\n", patterns.tool_call_start_marker.c_str());

    if (patterns.tool_call_opener.empty() && patterns.tool_call_closer.empty() && patterns.function_opener.empty() &&
        patterns.function_closer.empty() && patterns.parameter_opener.empty() && patterns.parameter_closer.empty() &&
        patterns.argument_separator.empty() && patterns.tool_call_start_marker.empty() &&
        patterns.tool_call_end_marker.empty()) {
        LOG_DBG("All patterns are empty - template doesn't support tool calls\n");
        return TemplatePattern::UNKNOWN;
    }

    if (!patterns.tool_call_opener.empty()) {
        if (patterns.tool_call_opener.find("{\"name\":") != std::string::npos ||
            patterns.tool_call_opener.find("{\"name\":") != std::string::npos ||
            patterns.tool_call_opener.find("{&quot;name&quot;:") != std::string::npos) {
            LOG_DBG("Detected JSON_NATIVE format from tool_call_opener JSON structure\n");
            return TemplatePattern::JSON_NATIVE;
        }
    }

    if (!patterns.function_opener.empty() && patterns.function_opener.find('<') == 0) {
        bool has_substantial_param_markers = false;
        if (!patterns.parameter_opener.empty()) {
            has_substantial_param_markers = (count_non_whitespace(patterns.parameter_opener) > 1);
        }
        if (!has_substantial_param_markers && !patterns.parameter_closer.empty()) {
            has_substantial_param_markers = (count_non_whitespace(patterns.parameter_closer) > 1);
        }

        if (!has_substantial_param_markers) {
            LOG_DBG("Function opener is XML-like but parameter markers are minimal\n");
            if ((!patterns.tool_call_opener.empty() && (patterns.tool_call_opener.find('[') != std::string::npos ||
                                                        patterns.tool_call_opener.find('{') != std::string::npos)) ||
                (!patterns.tool_call_start_marker.empty() &&
                 (patterns.tool_call_start_marker.find('[') != std::string::npos ||
                  patterns.tool_call_start_marker.find('{') != std::string::npos))) {
                LOG_DBG("Detected JSON_NATIVE format (XML markers but JSON structure)\n");
                return TemplatePattern::JSON_NATIVE;
            }
        }

        LOG_DBG("Detected XML_CONSTRUCTED format from function_opener\n");
        return TemplatePattern::XML_CONSTRUCTED;
    }

    if (!patterns.function_opener.empty() && patterns.function_opener.find('{') == 0) {
        LOG_DBG("Detected JSON_NATIVE format from function_opener\n");
        return TemplatePattern::JSON_NATIVE;
    }

    if (!patterns.tool_call_start_marker.empty() &&
        (patterns.tool_call_start_marker.find('<') == 0 || patterns.tool_call_start_marker.find('[') == 0)) {
        LOG_DBG("Detected XML_CONSTRUCTED format from tool_call_start_marker\n");
        return TemplatePattern::XML_CONSTRUCTED;
    }

    if (!patterns.tool_call_opener.empty() && (patterns.tool_call_opener.find("<function=") != std::string::npos ||
                                               patterns.tool_call_opener.find('<') != std::string::npos)) {
        LOG_DBG("Detected XML_CONSTRUCTED format from tool_call_opener\n");
        return TemplatePattern::XML_CONSTRUCTED;
    }

    if (!patterns.tool_call_opener.empty() &&
        (patterns.tool_call_opener.find('{') != std::string::npos ||
         patterns.tool_call_opener.find('[') != std::string::npos ||
         patterns.tool_call_opener.find("TOOLCALL") != std::string::npos)) {  // For Nemotron-style
        LOG_DBG("Detected JSON_NATIVE format from tool_call_opener\n");
        return TemplatePattern::JSON_NATIVE;
    }

    if (!patterns.tool_call_start_marker.empty() &&
        (patterns.tool_call_start_marker.find('{') != std::string::npos ||
         patterns.tool_call_start_marker.find('[') != std::string::npos ||
         patterns.tool_call_start_marker.find("TOOLCALL") != std::string::npos)) {
        LOG_DBG("Detected JSON_NATIVE format from tool_call_start_marker structure\n");
        return TemplatePattern::JSON_NATIVE;
    }

    if (patterns.tool_call_opener.find("TOOLCALL") != std::string::npos ||
        patterns.tool_call_closer.find("TOOLCALL") != std::string::npos ||
        patterns.tool_call_start_marker.find("TOOLCALL") != std::string::npos) {
        LOG_DBG("Detected JSON_NATIVE format from TOOLCALL pattern\n");
        return TemplatePattern::JSON_NATIVE;
    }

    if (patterns.tool_call_opener.find("\"name\"") != std::string::npos &&
        patterns.tool_call_opener.find("\"arguments\"") != std::string::npos) {
        LOG_DBG("Detected JSON_NATIVE format from JSON structure in arguments\n");
        return TemplatePattern::JSON_NATIVE;
    }

    bool has_substantial_patterns = false;

    if (!patterns.function_opener.empty() || !patterns.tool_call_start_marker.empty()) {
        has_substantial_patterns = true;
        LOG_DBG("Found substantial patterns but couldn't determine specific format\n");
    }

    if (!has_substantial_patterns) {
        LOG_DBG("No substantial patterns found - falling back to UNKNOWN\n");
        return TemplatePattern::UNKNOWN;
    }

    LOG_DBG("Could not determine format from patterns - returning UNKNOWN\n");
    return TemplatePattern::UNKNOWN;
}

void TemplateAnalyzer::analyze_reasoning(const minja::chat_template & tmpl, DiscoveredPattern & patterns) {
    LOG_DBG("=== ANALYZING REASONING ===\n");

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

    if (reasoning_output != base_output) {
        size_t thought_pos = reasoning_output.find("THOUGHT");
        if (thought_pos != std::string::npos) {


            size_t marker_pos = reasoning_output.find("MARKER");
            if (marker_pos != std::string::npos && marker_pos > thought_pos) {
                size_t thought_end            = thought_pos + 7;
                patterns.reasoning_end_marker = reasoning_output.substr(thought_end, marker_pos - thought_end);

                std::string diff             = find_string_difference(base_output, reasoning_output);
                size_t      diff_thought_pos = diff.find("THOUGHT");
                if (diff_thought_pos != std::string::npos) {
                    patterns.reasoning_start_marker = diff.substr(0, diff_thought_pos);
                }
            }
        }
    }

    if (patterns.reasoning_start_marker.empty()) {
        minja::chat_template_inputs inputs_prompt;
        inputs_prompt.messages                         = { base_msg };
        inputs_prompt.add_generation_prompt            = true;
        inputs_prompt.extra_context["enable_thinking"] = false;
        auto prompt_no_think                           = tmpl.apply(inputs_prompt);

        inputs_prompt.extra_context["enable_thinking"] = true;
        auto prompt_think                              = tmpl.apply(inputs_prompt);

        if (prompt_think != prompt_no_think) {
            std::string diff = find_string_difference(prompt_no_think, prompt_think);
            if (!diff.empty()) {
                patterns.reasoning_start_marker = diff;
                LOG_DBG("Detected reasoning marker by add_generation_prompt: %s\n", diff.c_str());
                patterns.reasoning_end_marker = create_closing_tag(diff);
            }
        }

        bool has_closed_reasoning_in_tool_markers = false;
        if (!patterns.tool_call_start_marker.empty() || !patterns.tool_call_opener.empty()) {
            std::vector<std::pair<std::string, std::string>> thinking_patterns = {
                { "<|START_THINKING|>", "<|END_THINKING|>" },
                { "<think>",            "</think>"         },
                { "[THINKING]",         "[/THINKING]"      },
                { "<thinking>",         "</thinking>"      }
            };

            for (const auto & [start_tag, end_tag] : thinking_patterns) {
                if ((patterns.tool_call_start_marker.find(start_tag) != std::string::npos &&
                     patterns.tool_call_start_marker.find(end_tag) != std::string::npos) ||
                    (patterns.tool_call_opener.find(start_tag) != std::string::npos &&
                     patterns.tool_call_opener.find(end_tag) != std::string::npos)) {
                    has_closed_reasoning_in_tool_markers = true;
                    LOG_DBG("Detected closed reasoning section in tool call markers, skipping Method 3\n");
                    break;
                }
            }
        }

        if (patterns.reasoning_start_marker.empty() && !has_closed_reasoning_in_tool_markers) {
            std::string prompt = prompt_think;  // Reuse calculated prompt
            trim_trailing_newlines(prompt);

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
                std::string tag = prompt.substr(last_open);

                std::vector<std::string> blacklisted_tags = { "<|CHATBOT_TOKEN|>", "<|SYSTEM_TOKEN|>",
                                                              "<|USER_TOKEN|>",    "<|ASSISTANT_TOKEN|>",
                                                              "<|im_start|>",      "<|im_end|>",
                                                              "<assistant>",       "<user>",
                                                              "<system>" };

                bool is_blacklisted = false;
                for (const auto & blacklisted : blacklisted_tags) {
                    if (tag == blacklisted) {
                        is_blacklisted = true;
                        LOG_DBG("Tag '%s' is blacklisted as a role marker, not using as reasoning marker\n",
                                tag.c_str());
                        break;
                    }
                }

                if (!is_blacklisted) {
                    patterns.reasoning_start_marker = tag;
                    patterns.reasoning_end_marker = create_closing_tag(tag);
                }
            }
        }
    }

    LOG_DBG("Reasoning markers: start='%s', end='%s'", patterns.reasoning_start_marker.c_str(),
            patterns.reasoning_end_marker.c_str());
}

void TemplateAnalyzer::analyze_content_markers(const minja::chat_template & tmpl, DiscoveredPattern & patterns) {
    LOG_DBG("=== ANALYZING CONTENT MARKERS ===\n");

    json base_msg = {
        { "role",    "assistant"            },
        { "content", "CONTENT_MARKER_12345" }  // Unique marker unlikely to appear elsewhere
    };

    minja::chat_template_inputs inputs;
    inputs.messages = { base_msg };
    std::string output;
    try {
        output = tmpl.apply(inputs);
    } catch (...) {
        LOG_DBG("Failed to render template for content marker analysis\n");
        return;
    }

    size_t marker_pos = output.find("CONTENT_MARKER_12345");
    if (marker_pos == std::string::npos) {
        LOG_DBG("Content marker not found in output\n");
        return;
    }

    std::vector<std::pair<std::string, std::string>> content_patterns = {
        { "<|START_RESPONSE|>", "<|END_RESPONSE|>" },
        { "<|response|>",       "<|/response|>"    },
        { "<response>",         "</response>"      },
        { "<output>",           "</output>"        },
        { "<answer>",           "</answer>"        },
    };

    for (const auto & [start_pattern, end_pattern] : content_patterns) {
        size_t start_pos = output.rfind(start_pattern, marker_pos);
        if (start_pos != std::string::npos) {
            std::string between =
                output.substr(start_pos + start_pattern.length(), marker_pos - start_pos - start_pattern.length());
            size_t first_non_ws = between.find_first_not_of(" \t\n\r");
            if (first_non_ws == std::string::npos || first_non_ws == between.length()) {
                size_t marker_end = marker_pos + strlen("CONTENT_MARKER_12345");
                size_t end_pos    = output.find(end_pattern, marker_end);
                if (end_pos != std::string::npos) {
                    std::string after              = output.substr(marker_end, end_pos - marker_end);
                    size_t      first_non_ws_after = after.find_first_not_of(" \t\n\r");
                    if (first_non_ws_after == std::string::npos || first_non_ws_after == after.length()) {
                        patterns.content_start_marker = start_pattern;
                        patterns.content_end_marker   = end_pattern;
                        LOG_DBG("Found content markers: start='%s', end='%s'\n", patterns.content_start_marker.c_str(),
                                patterns.content_end_marker.c_str());
                        return;
                    }
                }
            }
        }
    }

    if (marker_pos > 0) {
        size_t      search_start = (marker_pos > 50) ? marker_pos - 50 : 0;
        std::string before       = output.substr(search_start, marker_pos - search_start);

        size_t last_gt = before.rfind('>');
        if (last_gt != std::string::npos) {
            size_t tag_start = before.rfind('<', last_gt);
            if (tag_start != std::string::npos) {
                std::string potential_start = before.substr(tag_start, last_gt - tag_start + 1);

                std::string lower_tag = potential_start;
                std::transform(lower_tag.begin(), lower_tag.end(), lower_tag.begin(), ::tolower);
                if (lower_tag.find("response") != std::string::npos || lower_tag.find("content") != std::string::npos ||
                    lower_tag.find("output") != std::string::npos || lower_tag.find("answer") != std::string::npos) {
                    std::string end_tag;
                    if (potential_start.find("|>") != std::string::npos) {
                        std::string tag_name     = potential_start;
                        size_t      start_prefix = tag_name.find("START_");
                        if (start_prefix != std::string::npos) {
                            tag_name.replace(start_prefix, 6, "END_");
                            end_tag = tag_name;
                        }
                    } else if (potential_start[0] == '<' && potential_start.back() == '>') {
                        std::string tag_name = potential_start.substr(1, potential_start.length() - 2);
                        size_t      space    = tag_name.find(' ');
                        if (space != std::string::npos) {
                            tag_name = tag_name.substr(0, space);
                        }
                        end_tag = "</" + tag_name + ">";
                    }

                    if (!end_tag.empty()) {
                        size_t marker_end = marker_pos + strlen("CONTENT_MARKER_12345");
                        if (output.find(end_tag, marker_end) != std::string::npos) {
                            patterns.content_start_marker = potential_start;
                            patterns.content_end_marker   = end_tag;
                            LOG_DBG("Inferred content markers: start='%s', end='%s'\n",
                                    patterns.content_start_marker.c_str(), patterns.content_end_marker.c_str());
                            return;
                        }
                    }
                }
            }
        }
    }

    LOG_DBG("No content markers found\n");
}

std::vector<std::string> TemplateAnalyzer::extract_preserved_tokens(const minja::chat_template & tmpl) {
    auto                     discovered = analyze_by_differential(tmpl);
    std::vector<std::string> tokens;
    collect_non_empty_tokens(discovered, tokens);
    return tokens;
}

