#include "server-chat.h"
#include "server-common.h"

#include <sstream>

json server_chat_convert_responses_to_chatcmpl(const json & response_body) {
    if (!response_body.contains("input")) {
        throw std::invalid_argument("'input' is required");
    }
    if (!json_value(response_body, "previous_response_id", std::string{}).empty()) {
        SRV_WRN("%s\n", "ignoring previous_response_id in Responses request");
    }

    const json input_value = response_body.at("input");
    json chatcmpl_body = response_body;
    chatcmpl_body.erase("input");
    if (chatcmpl_body.contains("previous_response_id")) {
        chatcmpl_body.erase("previous_response_id");
    }
    std::vector<json> chatcmpl_messages;

    if (response_body.contains("instructions")) {
        chatcmpl_messages.push_back({
            {"role",    "system"},
            {"content", json_value(response_body, "instructions", std::string())},
        });
        chatcmpl_body.erase("instructions");
    }

    if (input_value.is_string()) {
        chatcmpl_messages.push_back({
            {"role",    "user"},
            {"content", input_value.get<std::string>()},
        });
    } else if (input_value.is_array()) {
        for (json item : input_value) {
            if (item.is_string()) {
                chatcmpl_messages.push_back({
                    {"role",    "user"},
                    {"content", item.get<std::string>()},
                });
                continue;
            }
            if (!item.is_object()) {
                continue;
            }

            bool merge_prev = !chatcmpl_messages.empty() && chatcmpl_messages.back().value("role", "") == "assistant";
            std::string item_type = json_value(item, "type", std::string{});
            std::string item_role = json_value(item, "role", std::string{});

            // Handle reasoning items (e.g. from Codex history)
            if (item_type == "reasoning" || item.contains("reasoning_content")) {
                std::string reasoning_text = "";
                if (item.contains("reasoning_content") && item.at("reasoning_content").is_string()) {
                    reasoning_text = item.at("reasoning_content").get<std::string>();
                } else if (item.contains("content")) {
                    if (item.at("content").is_string()) {
                        reasoning_text = item.at("content").get<std::string>();
                    } else if (item.at("content").is_array()) {
                        for (const auto & part : item.at("content")) {
                            if (part.is_string()) {
                                reasoning_text += part.get<std::string>();
                            } else if (part.is_object() && part.contains("text") && part.at("text").is_string()) {
                                reasoning_text += part.at("text").get<std::string>();
                            }
                        }
                    }
                }

                if (merge_prev) {
                    auto & prev_msg = chatcmpl_messages.back();
                    prev_msg["reasoning_content"] = reasoning_text;
                } else {
                    chatcmpl_messages.push_back(json {
                        {"role", "assistant"},
                        {"content", json::array()},
                        {"reasoning_content", reasoning_text},
                    });
                }
                continue;
            }

            // Handle function_call / tool_call
            if (item_type == "function_call" || item_type == "tool_call" || (item.contains("call_id") && item.contains("arguments"))) {
                std::string call_id = json_value(item, "call_id", json_value(item, "id", std::string{}));
                std::string name = json_value(item, "name", std::string{});
                std::string args_str = "{}";
                if (item.contains("arguments")) {
                    if (item.at("arguments").is_string()) {
                        args_str = item.at("arguments").get<std::string>();
                    } else {
                        args_str = item.at("arguments").dump();
                    }
                }

                json tool_call = {
                    {"function", json {
                        {"arguments", args_str},
                        {"name",      name},
                    }},
                    {"id",   call_id},
                    {"type", "function"},
                };

                if (merge_prev) {
                    auto & prev_msg = chatcmpl_messages.back();
                    if (!prev_msg.contains("tool_calls") || !prev_msg.at("tool_calls").is_array()) {
                        prev_msg["tool_calls"] = json::array();
                    }
                    prev_msg["tool_calls"].push_back(tool_call);
                } else {
                    chatcmpl_messages.push_back(json {
                        {"role",       "assistant"},
                        {"tool_calls", json::array({tool_call})}
                    });
                }
                continue;
            }

            // Handle function_call_output / tool_result
            if (item_type == "function_call_output" || item_type == "tool_result" || item_role == "tool" || (item.contains("call_id") && item.contains("output"))) {
                std::string call_id = json_value(item, "call_id", json_value(item, "tool_call_id", std::string{}));
                json output_val = item.value("output", item.value("content", json("")));

                if (output_val.is_string()) {
                    chatcmpl_messages.push_back(json {
                        {"content",      output_val.get<std::string>()},
                        {"role",         "tool"},
                        {"tool_call_id", call_id},
                    });
                } else if (output_val.is_array()) {
                    std::vector<json> chatcmpl_outputs;
                    for (const auto & chatcmpl_output : output_val) {
                        if (chatcmpl_output.is_string()) {
                            chatcmpl_outputs.push_back({{"type", "text"}, {"text", chatcmpl_output.get<std::string>()}});
                        } else if (chatcmpl_output.is_object()) {
                            std::string t = json_value(chatcmpl_output, "type", std::string{"text"});
                            if (t == "input_text" || t == "output_text" || t == "text") {
                                chatcmpl_outputs.push_back({{"type", "text"}, {"text", json_value(chatcmpl_output, "text", std::string{})}});
                            } else {
                                chatcmpl_outputs.push_back(chatcmpl_output);
                            }
                        }
                    }
                    chatcmpl_messages.push_back(json {
                        {"content",      chatcmpl_outputs},
                        {"role",         "tool"},
                        {"tool_call_id", call_id},
                    });
                } else {
                    chatcmpl_messages.push_back(json {
                        {"content",      output_val.dump()},
                        {"role",         "tool"},
                        {"tool_call_id", call_id},
                    });
                }
                continue;
            }

            // Determine role
            if (item_role.empty()) {
                if (item_type == "message") {
                    item_role = "user";
                } else {
                    item_role = "user";
                }
            }

            json chatcmpl_content = json::array();
            if (item.contains("content")) {
                const auto & content_obj = item.at("content");
                if (content_obj.is_string()) {
                    chatcmpl_content.push_back({
                        {"type", "text"},
                        {"text", content_obj.get<std::string>()}
                    });
                } else if (content_obj.is_array()) {
                    for (const auto & part : content_obj) {
                        if (part.is_string()) {
                            chatcmpl_content.push_back({
                                {"type", "text"},
                                {"text", part.get<std::string>()}
                            });
                        } else if (part.is_object()) {
                            std::string part_type = json_value(part, "type", std::string{});
                            if (part_type == "input_text" || part_type == "output_text" || part_type == "text" || (part_type.empty() && part.contains("text"))) {
                                chatcmpl_content.push_back({
                                    {"type", "text"},
                                    {"text", json_value(part, "text", std::string{})}
                                });
                            } else if (part_type == "input_image" || part_type == "image_url") {
                                if (part.contains("image_url")) {
                                    chatcmpl_content.push_back({
                                        {"type", "image_url"},
                                        {"image_url", part.at("image_url").is_string() ? json{{"url", part.at("image_url")}} : part.at("image_url")}
                                    });
                                }
                            } else if (part_type == "refusal") {
                                chatcmpl_content.push_back({
                                    {"type", "refusal"},
                                    {"refusal", json_value(part, "refusal", std::string{})}
                                });
                            } else if (part.contains("text")) {
                                chatcmpl_content.push_back({
                                    {"type", "text"},
                                    {"text", json_value(part, "text", std::string{})}
                                });
                            } else {
                                chatcmpl_content.push_back(part);
                            }
                        }
                    }
                }
            }

            if (item_role == "assistant" && merge_prev) {
                auto & prev_msg = chatcmpl_messages.back();
                if (!prev_msg.contains("content") || !prev_msg.at("content").is_array()) {
                    prev_msg["content"] = json::array();
                }
                for (const auto & c : chatcmpl_content) {
                    prev_msg["content"].push_back(c);
                }
            } else {
                json msg = json::object();
                msg["role"] = item_role;
                msg["content"] = chatcmpl_content;
                if (item.contains("tool_calls")) {
                    msg["tool_calls"] = item.at("tool_calls");
                }
                chatcmpl_messages.push_back(msg);
            }
        }
    } else {
        throw std::invalid_argument("'input' must be a string or array of objects");
    }

    chatcmpl_body["messages"] = chatcmpl_messages;

    if (response_body.contains("tools")) {
        if (!response_body.at("tools").is_array()) {
            throw std::invalid_argument("'tools' must be an array of objects");
        }
        std::vector<json> chatcmpl_tools;
        for (json resp_tool : response_body.at("tools")) {
            json chatcmpl_tool;

            const std::string type = json_value(resp_tool, "type", std::string());
            if (type != "function") {
                SRV_WRN("unsupported Responses tool type '%s' skipped\n", type.c_str());
                continue;
            }
            resp_tool.erase("type");
            chatcmpl_tool["type"] = "function";

            if (!resp_tool.contains("strict")) {
                resp_tool["strict"] = true;
            }
            chatcmpl_tool["function"] = resp_tool;
            chatcmpl_tools.push_back(chatcmpl_tool);
        }
        chatcmpl_body.erase("tools");
        if (!chatcmpl_tools.empty()) {
            chatcmpl_body["tools"] = chatcmpl_tools;
        }
    }

    if (response_body.contains("max_output_tokens")) {
        chatcmpl_body.erase("max_output_tokens");
        chatcmpl_body["max_tokens"] = response_body["max_output_tokens"];
    }

    if (response_body.contains("reasoning")) {
        const json & reasoning = response_body.at("reasoning");
        if (reasoning.contains("effort")) {
            chatcmpl_body["reasoning_effort"] = reasoning.at("effort");
        }
        chatcmpl_body.erase("reasoning");
    }

    return chatcmpl_body;
}

// Edits the cch section of an "x-anthropic-billing-header" system prompt.
// Does nothing to any other prompt.
//
// This is a claude message with a "cch=ef01a" attribute that breaks prefix caching.
// The cch stamp is a whitebox end-to-end integrity hint. It's not meaningful as a
// system prompt data, particularly to llama.cpp, but its presence means the prefix
// cache will not get past it: It changes on each request.
//
// Reference: https://github.com/ggml-org/llama.cpp/pull/21793
// Example header:
// ```
// x-anthropic-billing-header: cc_version=2.1.101.e51; cc_entrypoint=cli; cch=a5145;You are Claude Code, Anthropic's official CLI for Claude.
//                                                                            ^^^^^
// ```
static void normalize_anthropic_billing_header(std::string & system_text) {
    if (system_text.rfind("x-anthropic-billing-header:", 0) != 0) {
        return;
    }

    const size_t header_prefix_length = strlen("x-anthropic-billing-header:");
    const size_t cch_length = 5;
    const size_t index_cch = system_text.find("cch=", header_prefix_length);
    if (index_cch == std::string::npos) {
        return;
    }

    const size_t index_replace = index_cch + 4;
    if (index_replace + cch_length < system_text.length() && system_text[index_replace + cch_length] == ';') {
        for (size_t i = 0; i < cch_length; ++i) {
            system_text[index_replace + i] = 'f';
        }
    } else {
        LOG_ERR("anthropic string not as expected: %s", system_text.c_str());
    }
}

json server_chat_convert_anthropic_to_oai(const json & body) {
    json oai_body;

    // Convert system prompt
    json oai_messages = json::array();
    auto system_param = json_value(body, "system", json());
    if (!system_param.is_null()) {
        std::string system_content;

        if (system_param.is_string()) {
            system_content = system_param.get<std::string>();
            normalize_anthropic_billing_header(system_content);
        } else if (system_param.is_array()) {
            for (const auto & block : system_param) {
                if (json_value(block, "type", std::string()) == "text") {
                    auto system_text = json_value(block, "text", std::string());
                    normalize_anthropic_billing_header(system_text);
                    system_content += system_text;
                }
            }
        }

        oai_messages.push_back({
            {"role", "system"},
            {"content", system_content}
        });
    }

    // Convert messages
    if (!body.contains("messages")) {
        throw std::runtime_error("'messages' is required");
    }
    const json & messages = body.at("messages");
    if (messages.is_array()) {
        for (const auto & msg : messages) {
            std::string role = json_value(msg, "role", std::string());

            if (!msg.contains("content")) {
                if (role == "assistant") {
                    continue;
                }
                oai_messages.push_back(msg);
                continue;
            }

            const json & content = msg.at("content");

            if (content.is_string()) {
                oai_messages.push_back(msg);
                continue;
            }

            if (!content.is_array()) {
                oai_messages.push_back(msg);
                continue;
            }

            json tool_calls = json::array();
            json converted_content = json::array();
            json tool_results = json::array();
            std::string reasoning_content;
            bool has_tool_calls = false;

            for (const auto & block : content) {
                std::string type = json_value(block, "type", std::string());

                if (type == "text") {
                    converted_content.push_back(block);
                } else if (type == "thinking") {
                    reasoning_content += json_value(block, "thinking", std::string());
                } else if (type == "image") {
                    json source = json_value(block, "source", json::object());
                    std::string source_type = json_value(source, "type", std::string());

                    if (source_type == "base64") {
                        std::string media_type = json_value(source, "media_type", std::string("image/jpeg"));
                        std::string data = json_value(source, "data", std::string());
                        std::ostringstream ss;
                        ss << "data:" << media_type << ";base64," << data;

                        converted_content.push_back({
                            {"type", "image_url"},
                            {"image_url", {
                                {"url", ss.str()}
                            }}
                        });
                    } else if (source_type == "url") {
                        std::string url = json_value(source, "url", std::string());
                        converted_content.push_back({
                            {"type", "image_url"},
                            {"image_url", {
                                {"url", url}
                            }}
                        });
                    }
                } else if (type == "tool_use") {
                    tool_calls.push_back({
                        {"id", json_value(block, "id", std::string())},
                        {"type", "function"},
                        {"function", {
                            {"name", json_value(block, "name", std::string())},
                            {"arguments", json_value(block, "input", json::object()).dump()}
                        }}
                    });
                    has_tool_calls = true;
                } else if (type == "tool_result") {
                    std::string tool_use_id = json_value(block, "tool_use_id", std::string());

                    auto result_content = json_value(block, "content", json());
                    if (result_content.is_string()) {
                        tool_results.push_back({
                            {"role", "tool"},
                            {"tool_call_id", tool_use_id},
                            {"content", result_content.get<std::string>()}
                        });
                    } else if (result_content.is_array()) {
                        // Single-pass: build both text and content_parts, decide format at the end
                        std::string result_text;
                        json content_parts = json::array();
                        bool has_images = false;

                        for (const auto & c : result_content) {
                            std::string c_type = json_value(c, "type", std::string());
                            if (c_type == "text") {
                                std::string text = json_value(c, "text", std::string());
                                result_text += text;
                                content_parts.push_back({
                                    {"type", "text"},
                                    {"text", text}
                                });
                            } else if (c_type == "image") {
                                has_images = true;
                                json source = json_value(c, "source", json::object());
                                std::string source_type = json_value(source, "type", std::string());
                                if (source_type == "base64") {
                                    std::string media_type = json_value(source, "media_type", std::string("image/jpeg"));
                                    std::string data = json_value(source, "data", std::string());
                                    std::string url = "data:" + media_type + ";base64," + data;
                                    content_parts.push_back({
                                        {"type", "image_url"},
                                        {"image_url", {{"url", url}}}
                                    });
                                } else if (source_type == "url") {
                                    content_parts.push_back({
                                        {"type", "image_url"},
                                        {"image_url", {{"url", json_value(source, "url", std::string())}}}
                                    });
                                }
                            }
                        }

                        if (!has_images) {
                            // Text-only: collapse to a plain string for maximum compatibility
                            tool_results.push_back({
                                {"role", "tool"},
                                {"tool_call_id", tool_use_id},
                                {"content", result_text}
                            });
                        } else {
                            // Mixed or image-only: use array content parts (OpenAI multimodal tool format)
                            tool_results.push_back({
                                {"role", "tool"},
                                {"tool_call_id", tool_use_id},
                                {"content", content_parts}
                            });
                        }
                    } else {
                        tool_results.push_back({
                            {"role", "tool"},
                            {"tool_call_id", tool_use_id},
                            {"content", ""}
                        });
                    }
                }
            }

            if (!converted_content.empty() || has_tool_calls || !reasoning_content.empty()) {
                json new_msg = {{"role", role}};
                if (!converted_content.empty()) {
                    new_msg["content"] = converted_content;
                } else if (has_tool_calls || !reasoning_content.empty()) {
                    new_msg["content"] = "";
                }
                if (!tool_calls.empty()) {
                    new_msg["tool_calls"] = tool_calls;
                }
                if (!reasoning_content.empty()) {
                    new_msg["reasoning_content"] = reasoning_content;
                }
                oai_messages.push_back(new_msg);
            }

            for (const auto & tool_msg : tool_results) {
                oai_messages.push_back(tool_msg);
            }
        }
    }

    oai_body["messages"] = oai_messages;

    // Convert tools
    if (body.contains("tools")) {
        const json & tools = body.at("tools");
        if (tools.is_array()) {
            json oai_tools = json::array();
            for (const auto & tool : tools) {
                oai_tools.push_back({
                    {"type", "function"},
                    {"function", {
                        {"name", json_value(tool, "name", std::string())},
                        {"description", json_value(tool, "description", std::string())},
                        {"parameters", tool.contains("input_schema") ? tool.at("input_schema") : json::object()}
                    }}
                });
            }
            oai_body["tools"] = oai_tools;
        }
    }

    // Convert tool_choice
    if (body.contains("tool_choice")) {
        const json & tc = body.at("tool_choice");
        if (tc.is_object()) {
            std::string type = json_value(tc, "type", std::string());
            if (type == "auto") {
                oai_body["tool_choice"] = "auto";
            } else if (type == "any" || type == "tool") {
                oai_body["tool_choice"] = "required";
            }
        }
    }

    // Convert stop_sequences to stop
    if (body.contains("stop_sequences")) {
        oai_body["stop"] = body.at("stop_sequences");
    }

    // Handle max_tokens (required in Anthropic, but we're permissive)
    if (body.contains("max_tokens")) {
        oai_body["max_tokens"] = body.at("max_tokens");
    } else {
        oai_body["max_tokens"] = 4096;
    }

    // Pass through common params
    for (const auto & key : {"temperature", "top_p", "top_k", "stream", "chat_template_kwargs"}) {
        if (body.contains(key)) {
            oai_body[key] = body.at(key);
        }
    }

    // Handle Anthropic-specific thinking param
    if (body.contains("thinking")) {
        json thinking = json_value(body, "thinking", json::object());
        std::string thinking_type = json_value(thinking, "type", std::string());
        if (thinking_type == "enabled") {
            int budget_tokens = json_value(thinking, "budget_tokens", 10000);
            oai_body["thinking_budget_tokens"] = budget_tokens;
        }
    }

    // Handle Anthropic-specific metadata param
    if (body.contains("metadata")) {
        json metadata = json_value(body, "metadata", json::object());
        std::string user_id = json_value(metadata, "user_id", std::string());
        if (!user_id.empty()) {
            oai_body["__metadata_user_id"] = user_id;
        }
    }

    return oai_body;
}

json server_chat_msg_diff_to_json_oaicompat(const common_chat_msg_diff & diff) {
    json delta = json::object();
    if (!diff.reasoning_content_delta.empty()) {
        delta["reasoning_content"] = diff.reasoning_content_delta;
    }
    if (!diff.content_delta.empty()) {
        delta["content"] = diff.content_delta;
    }
    if (diff.tool_call_index != std::string::npos) {
        json tool_call;
        tool_call["index"] = diff.tool_call_index;
        if (!diff.tool_call_delta.id.empty()) {
            tool_call["id"]   = diff.tool_call_delta.id;
            tool_call["type"] = "function";
        }
        if (!diff.tool_call_delta.name.empty() || !diff.tool_call_delta.arguments.empty()) {
            json function = json::object();
            if (!diff.tool_call_delta.name.empty()) {
                function["name"] = diff.tool_call_delta.name;
            }
            if (!diff.tool_call_delta.arguments.empty()) {
                function["arguments"] = diff.tool_call_delta.arguments;
            }
            tool_call["function"] = function;
        }
        delta["tool_calls"] = json::array({ tool_call });
    }
    return delta;
}

json convert_transcriptions_to_chatcmpl(
        const json & inp_body,
        const common_chat_templates * tmpls,
        const std::map<std::string, uploaded_file> & in_files,
        std::vector<raw_buffer> & out_files) {
    // TODO @ngxson : this function may need to be improved in the future
    // handle input files
    out_files.clear();
    auto it = in_files.find("file");
    if (it != in_files.end()) {
        out_files.push_back(it->second.data);
    } else {
        throw std::invalid_argument("No input file found for transcription");
    }

    // handle input data
    std::string prompt          = json_value(inp_body, "prompt", std::string());
    std::string language        = json_value(inp_body, "language", std::string());
    std::string response_format = json_value(inp_body, "response_format", std::string("json"));
    if (response_format != "json") {
        throw std::invalid_argument("Only 'json' response_format is supported for transcription");
    }
    const common_chat_prompt_preset preset = common_chat_get_asr_prompt(tmpls);
    if (prompt.empty()) {
        prompt = preset.user;
    }
    if (!language.empty()) {
        prompt += string_format(" (language: %s)", language.c_str());
    }
    prompt += get_media_marker();

    json messages = json::array();
    if (!preset.system.empty()) {
        messages.push_back({{"role", "system"}, {"content", preset.system}});
    }
    messages.push_back({{"role", "user"}, {"content", prompt}});

    json chatcmpl_body = inp_body; // copy all fields
    chatcmpl_body["messages"] = messages;

    // because input from form-data, everything is string, we need to correct the types here
    std::string stream = json_value(inp_body, "stream", std::string("false"));
    chatcmpl_body["stream"] = stream == "true";

    if (inp_body.contains("max_tokens")) {
        std::string inp = inp_body["max_tokens"].get<std::string>();
        chatcmpl_body["max_tokens"] = std::stoul(inp);
    }

    if (inp_body.contains("temperature")) {
        std::string inp = inp_body["temperature"].get<std::string>();
        chatcmpl_body["temperature"] = std::stof(inp);
    }

    return chatcmpl_body;
}
