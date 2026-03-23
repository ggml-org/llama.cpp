#include "server-chat.h"
#include "server-common.h"

#include <sstream>

json common_chat_convert_responses_to_chatcmpl(const json & response_body) {
    if (!response_body.contains("input")) {
        throw std::invalid_argument("'input' is required");
    }
    if (!json_value(response_body, "previous_response_id", std::string{}).empty()) {
        throw std::invalid_argument("llama.cpp does not support 'previous_response_id'.");
    }

    const json input_value = response_body.at("input");
    json chatcmpl_body = response_body;
    chatcmpl_body.erase("input");
    std::vector<json> chatcmpl_messages;

    if (response_body.contains("instructions")) {
        chatcmpl_messages.push_back({
            {"role",    "system"},
            {"content", json_value(response_body, "instructions", std::string())},
        });
        chatcmpl_body.erase("instructions");
    }

    if (input_value.is_string()) {
        // #responses_create-input-text_input
        chatcmpl_messages.push_back({
            {"role",    "user"},
            {"content", input_value},
        });
    } else if (input_value.is_array()) {
        // #responses_create-input-input_item_list

        static auto exists_and_is_array = [](const json & j, const char * key) -> bool {
            return j.contains(key) && j.at(key).is_array();
        };
        static auto exists_and_is_string = [](const json & j, const char * key) -> bool {
            return j.contains(key) && j.at(key).is_string();
        };

        for (json item : input_value) {
            bool merge_prev = !chatcmpl_messages.empty() && chatcmpl_messages.back().value("role", "") == "assistant";

            if (exists_and_is_string(item, "content")) {
                // #responses_create-input-input_item_list-input_message-content-text_input
                // Only "Input message" contains item["content"]::string
                // After converting item["content"]::string to item["content"]::array,
                // we can treat "Input message" as sum of "Item-Input message" and "Item-Output message"
                item["content"] = json::array({
                    json {
                        {"text", item.at("content")},
                        {"type", "input_text"}
                    }
                });
            }

            if (exists_and_is_array(item, "content") &&
                exists_and_is_string(item, "role") &&
                (item.at("role") == "user" ||
                    item.at("role") == "system" ||
                    item.at("role") == "developer")
            ) {
                // #responses_create-input-input_item_list-item-input_message
                std::vector<json> chatcmpl_content;

                for (const json & input_item : item.at("content")) {
                    const std::string type = json_value(input_item, "type", std::string());

                    if (type == "input_text") {
                        if (!input_item.contains("text")) {
                            throw std::invalid_argument("'Input text' requires 'text'");
                        }
                        chatcmpl_content.push_back({
                            {"text", input_item.at("text")},
                            {"type", "text"},
                        });
                    } else if (type == "input_image") {
                        // While `detail` is marked as required,
                        // it has default value("auto") and can be omitted.

                        if (!input_item.contains("image_url")) {
                            throw std::invalid_argument("'image_url' is required");
                        }
                        chatcmpl_content.push_back({
                            {"image_url", json {
                                {"url", input_item.at("image_url")}
                            }},
                            {"type", "image_url"},
                        });
                    } else if (type == "input_file") {
                        throw std::invalid_argument("'input_file' is not supported by llamacpp at this moment");
                    } else {
                        throw std::invalid_argument("'type' must be one of 'input_text', 'input_image', or 'input_file'");
                    }
                }

                if (item.contains("type")) {
                    item.erase("type");
                }
                if (item.contains("status")) {
                    item.erase("status");
                }
                item["content"] = chatcmpl_content;

                chatcmpl_messages.push_back(item);
            } else if (exists_and_is_string(item, "role") &&
                item.at("role") == "assistant" &&
                exists_and_is_string(item, "type") &&
                item.at("type") == "message"
            ) {
                // #responses_create-input-input_item_list-item-output_message
                auto chatcmpl_content = json::array();

                // Handle both string content and array content
                if (item.contains("content") && item.at("content").is_string()) {
                    // String content - convert to text content part
                    chatcmpl_content.push_back({
                        {"text", item.at("content")},
                        {"type", "text"},
                    });
                } else if (exists_and_is_array(item, "content")) {
                    // Array content - process each item
                    for (const auto & output_text : item.at("content")) {
                        const std::string type = json_value(output_text, "type", std::string());
                        if (type == "output_text" || type == "input_text") {
                            // Accept both output_text and input_text (string content gets converted to input_text)
                            if (!exists_and_is_string(output_text, "text")) {
                                throw std::invalid_argument("'Output text' requires 'text'");
                            }
                            chatcmpl_content.push_back({
                                {"text", output_text.at("text")},
                                {"type", "text"},
                            });
                        } else if (type == "refusal") {
                            if (!exists_and_is_string(output_text, "refusal")) {
                                throw std::invalid_argument("'Refusal' requires 'refusal'");
                            }
                            chatcmpl_content.push_back({
                                {"refusal", output_text.at("refusal")},
                                {"type", "refusal"},
                            });
                        } else {
                            throw std::invalid_argument("'type' must be one of 'output_text' or 'refusal'");
                        }
                    }
                }

                if (merge_prev) {
                    auto & prev_msg = chatcmpl_messages.back();
                    if (!exists_and_is_array(prev_msg, "content")) {
                        prev_msg["content"] = json::array();
                    }
                    auto & prev_content = prev_msg["content"];
                    prev_content.insert(prev_content.end(), chatcmpl_content.begin(), chatcmpl_content.end());
                } else {
                    item.erase("status");
                    item.erase("type");
                    item["content"] = chatcmpl_content;
                    chatcmpl_messages.push_back(item);
                }
            } else if (exists_and_is_string(item, "arguments") &&
                exists_and_is_string(item, "call_id") &&
                exists_and_is_string(item, "name") &&
                exists_and_is_string(item, "type") &&
                item.at("type") == "function_call"
            ) {
                // #responses_create-input-input_item_list-item-function_tool_call
                json tool_call = {
                    {"function", json {
                        {"arguments", item.at("arguments")},
                        {"name",      item.at("name")},
                    }},
                    {"id",   item.at("call_id")},
                    {"type", "function"},
                };

                if (merge_prev) {
                    auto & prev_msg = chatcmpl_messages.back();
                    if (!exists_and_is_array(prev_msg, "tool_calls")) {
                        prev_msg["tool_calls"] = json::array();
                    }
                    prev_msg["tool_calls"].push_back(tool_call);
                } else {
                    chatcmpl_messages.push_back(json {
                        {"role",       "assistant"},
                        {"tool_calls", json::array({tool_call})}
                    });
                }
            } else if (exists_and_is_string(item, "call_id") &&
                (exists_and_is_string(item, "output") || exists_and_is_array(item, "output")) &&
                exists_and_is_string(item, "type") &&
                item.at("type") == "function_call_output"
            ) {
                // #responses_create-input-input_item_list-item-function_tool_call_output
                if (item.at("output").is_string()) {
                    chatcmpl_messages.push_back(json {
                        {"content",      item.at("output")},
                        {"role",         "tool"},
                        {"tool_call_id", item.at("call_id")},
                    });
                } else {
                    json chatcmpl_outputs = item.at("output");
                    for (json & chatcmpl_output : chatcmpl_outputs) {
                        if (!chatcmpl_output.contains("type") || chatcmpl_output.at("type") != "input_text") {
                            throw std::invalid_argument("Output of tool call should be 'Input text'");
                        }
                        chatcmpl_output["type"] = "text";
                    }
                    chatcmpl_messages.push_back(json {
                        {"content",      chatcmpl_outputs},
                        {"role",         "tool"},
                        {"tool_call_id", item.at("call_id")},
                    });
                }
            } else if (exists_and_is_array(item, "summary") &&
                exists_and_is_string(item, "type") &&
                item.at("type") == "reasoning") {
                // #responses_create-input-input_item_list-item-reasoning

                if (!exists_and_is_array(item, "content")) {
                    throw std::invalid_argument("item['content'] is not an array");
                }
                if (item.at("content").empty()) {
                    throw std::invalid_argument("item['content'] is empty");
                }
                if (!exists_and_is_string(item.at("content")[0], "text")) {
                    throw std::invalid_argument("item['content']['text'] is not a string");
                }

                if (merge_prev) {
                    auto & prev_msg = chatcmpl_messages.back();
                    prev_msg["reasoning_content"] = item.at("content")[0].at("text");
                } else {
                    chatcmpl_messages.push_back(json {
                        {"role", "assistant"},
                        {"content", json::array()},
                        {"reasoning_content", item.at("content")[0].at("text")},
                    });
                }
            } else {
                throw std::invalid_argument("Cannot determine type of 'item'");
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

            if (json_value(resp_tool, "type", std::string()) != "function") {
                throw std::invalid_argument("'type' of tool must be 'function'");
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
        chatcmpl_body["tools"] = chatcmpl_tools;
    }

    if (response_body.contains("max_output_tokens")) {
        chatcmpl_body.erase("max_output_tokens");
        chatcmpl_body["max_tokens"] = response_body["max_output_tokens"];
    }

    return chatcmpl_body;
}

json common_chat_convert_anthropic_to_oai(const json & body) {
    json oai_body;

    // Convert system prompt
    json oai_messages = json::array();
    auto system_param = json_value(body, "system", json());
    if (!system_param.is_null()) {
        std::string system_content;

        if (system_param.is_string()) {
            system_content = system_param.get<std::string>();
        } else if (system_param.is_array()) {
            for (const auto & block : system_param) {
                if (json_value(block, "type", std::string()) == "text") {
                    system_content += json_value(block, "text", std::string());
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
                    std::string result_text;
                    if (result_content.is_string()) {
                        result_text = result_content.get<std::string>();
                    } else if (result_content.is_array()) {
                        for (const auto & c : result_content) {
                            if (json_value(c, "type", std::string()) == "text") {
                                result_text += json_value(c, "text", std::string());
                            }
                        }
                    }

                    tool_results.push_back({
                        {"role", "tool"},
                        {"tool_call_id", tool_use_id},
                        {"content", result_text}
                    });
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
    for (const auto & key : {"temperature", "top_p", "top_k", "stream"}) {
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

json common_chat_msg_diff_to_json_oaicompat(const common_chat_msg_diff & diff) {
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
