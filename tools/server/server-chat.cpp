#include "server-chat.h"
#include "server-common.h"

#include <cctype>
#include <sstream>
#include <unordered_set>

static bool exists_and_is_array(const json & j, const char * key) { return j.contains(key) && j.at(key).is_array(); }
static bool exists_and_is_string(const json & j, const char * key) { return j.contains(key) && j.at(key).is_string(); }

static std::string truncate_for_prompt(const std::string & text, size_t max_len = 64 * 1024) {
    return text.size() <= max_len ? text : text.substr(0, max_len) + "\n\n[truncated " + std::to_string(text.size() - max_len) + " bytes]";
}

static json responses_make_text_content(const std::string & text) {
    return json {
        {"text", text},
        {"type", "text"},
    };
}

static void responses_append_recovery_text(
        std::vector<json> & chatcmpl_content,
        const std::string & role,
        const std::string & item_type,
        const std::string & detail) {
    SRV_WRN(
            "responses recovery: role='%s', item_type='%s', action='inject_text', detail='%s'\n",
            role.c_str(),
            item_type.c_str(),
            detail.c_str());
    chatcmpl_content.push_back(responses_make_text_content("[responses recovery: " + detail + "]"));
}

static std::string sanitize_tool_name(const std::string & name, const std::string & fallback = "tool") {
    std::string out;
    out.reserve(name.size());
    for (unsigned char ch : name) {
        if (std::isalnum(ch) || ch == '_' || ch == '-') {
            out.push_back(static_cast<char>(ch));
        } else {
            out.push_back('_');
        }
    }
    while (!out.empty() && out.front() == '_') {
        out.erase(out.begin());
    }
    if (out.empty()) {
        out = fallback;
    }
    if (out.size() > 64) {
        out.resize(64);
    }
    return out;
}

static json responses_object_schema(json properties, json required = json::array()) {
    json schema = {
        {"type", "object"}, {"properties", std::move(properties)}, {"additionalProperties", true},
    };
    if (!required.empty()) {
        schema["required"] = std::move(required);
    }
    return schema;
}

static json responses_function_tool(
        const std::string & name,
        const std::string & description,
        const json & parameters,
        bool strict = false) {
    return json {
        {"name", name}, {"description", description},
        {"strict", strict}, {"parameters", parameters},
    };
}

static std::string responses_custom_tool_description(const json & resp_tool, const std::string & name) {
    std::string description = json_value(resp_tool, "description", std::string("Freeform Codex tool input."));

    if (name == "apply_patch") {
        description +=
            "\n\nPass only the raw apply_patch body in the input argument. "
            "The body must begin with '*** Begin Patch' and end with '*** End Patch'. "
            "Use '*** Add File: path' to create a file and prefix every new file content line with '+'. "
            "Do not update a file before it exists. "
            "For update hunks, use '@@' with optional semantic context, not unified diff range headers.";
    }

    if (resp_tool.contains("format") && resp_tool.at("format").is_object()) {
        const json & format = resp_tool.at("format");
        const std::string syntax = json_value(format, "syntax", std::string());
        const std::string definition = json_value(format, "definition", std::string());
        if (!definition.empty()) {
            description += "\n\nFreeform input format";
            if (!syntax.empty()) {
                description += " (" + syntax + ")";
            }
            description += ":\n" + definition;
        }
    }

    return description;
}

static json responses_custom_tool_parameters(const std::string & name) {
    std::string input_description = "Freeform tool input.";
    if (name == "apply_patch") {
        input_description = "Complete raw apply_patch body, beginning with '*** Begin Patch' and ending with '*** End Patch'.";
    }

    return responses_object_schema(
            json {{"input", json {{"type", "string"}, {"description", input_description}}}},
            json::array({"input"}));
}

static json parse_arguments_best_effort(const json & value) {
    if (value.is_object() || value.is_array()) {
        return value;
    }
    if (value.is_string()) {
        const std::string raw = value.get<std::string>();
        try {
            return json::parse(raw);
        } catch (const std::exception &) {
            return raw;
        }
    }
    if (value.is_null()) {
        return json::object();
    }
    return value;
}

static json make_tool_call(const std::string & name, const std::string & call_id, const json & arguments) {
    return json {
        {"function", json {
            {"arguments", arguments.is_string() ? arguments.get<std::string>() : arguments.dump()},
            {"name",      name},
        }},
        {"id",   call_id.empty() ? "call_" + random_string() : call_id},
        {"type", "function"},
    };
}

static void append_assistant_tool_call(std::vector<json> & messages, const json & tool_call) {
    if (!messages.empty() && messages.back().value("role", "") == "assistant") {
        auto & prev_msg = messages.back();
        if (!exists_and_is_array(prev_msg, "tool_calls")) {
            prev_msg["tool_calls"] = json::array();
        }
        prev_msg["tool_calls"].push_back(tool_call);
        return;
    }

    messages.push_back(json {
        {"role",       "assistant"},
        {"content",    json::array()},
        {"tool_calls", json::array({tool_call})},
    });
}

static std::string mcp_image_to_data_url(const json & image) {
    const std::string data = json_value(image, "data", std::string());
    if (data.empty()) {
        return "";
    }
    std::string mime_type = json_value(image, "mimeType", std::string());
    if (mime_type.empty()) {
        mime_type = json_value(image, "mime_type", std::string("image/png"));
    }
    return "data:" + mime_type + ";base64," + data;
}

static json encode_tool_output_content_item(const json & item) {
    const std::string type = json_value(item, "type", std::string());
    if (type == "input_text" || type == "output_text" || type == "text") {
        return responses_make_text_content(exists_and_is_string(item, "text")
            ? item.at("text").get<std::string>()
            : "[text output item missing text]");
    }
    if (type == "input_image") {
        if (exists_and_is_string(item, "image_url")) {
            return json {
                {"image_url", json {{"url", item.at("image_url")}}},
                {"type", "image_url"},
            };
        }
        return responses_make_text_content("[image output item missing image_url]");
    }
    if (type == "image") {
        const std::string data_url = mcp_image_to_data_url(item);
        if (!data_url.empty()) {
            return json {
                {"image_url", json {{"url", data_url}}},
                {"type", "image_url"},
            };
        }
        return responses_make_text_content("[image output item missing data]");
    }
    return responses_make_text_content("[unsupported tool output item: " + item.dump() + "]");
}

static json encode_tool_output_content(const json & output) {
    if (output.is_string()) {
        return output.get<std::string>();
    }
    if (output.is_array()) {
        json content = json::array();
        for (const auto & item : output) {
            if (item.is_object()) {
                content.push_back(encode_tool_output_content_item(item));
            } else {
                content.push_back(responses_make_text_content(item.is_string() ? item.get<std::string>() : item.is_null() ? "" : item.dump()));
            }
        }
        return content;
    }
    if (output.is_object()) {
        if (exists_and_is_array(output, "content")) {
            return encode_tool_output_content(output.at("content"));
        }
        if (exists_and_is_string(output, "body")) {
            return output.at("body").get<std::string>();
        }
        if (exists_and_is_string(output, "output")) {
            return output.at("output").get<std::string>();
        }
    }
    return output.dump();
}

static bool responses_shell_bridge_to_web_search(
        const json & action,
        const std::string & wrapper,
        json & arguments) {
    if (wrapper.empty() || !action.is_object() || json_value(action, "type", std::string()) != "exec") {
        return false;
    }
    if (!action.contains("command") || !action.at("command").is_array()) {
        return false;
    }
    const json & command = action.at("command");
    if (command.size() < 3 || !command.at(0).is_string() || command.at(0).get<std::string>() != wrapper ||
        !command.at(1).is_string() || !command.at(2).is_string()) {
        return false;
    }

    const std::string subcommand = command.at(1).get<std::string>();
    if (subcommand == "search") {
        arguments = json {
            {"type",  "search"},
            {"query", command.at(2).get<std::string>()},
        };
        return true;
    }
    if (subcommand == "extract") {
        arguments = json {
            {"type", "open_page"},
            {"url",  command.at(2).get<std::string>()},
        };
        for (size_t i = 3; i + 1 < command.size(); ++i) {
            if (command.at(i).is_string() && command.at(i).get<std::string>() == "--query" && command.at(i + 1).is_string()) {
                arguments["type"] = "find_in_page";
                arguments["pattern"] = command.at(i + 1).get<std::string>();
                return true;
            }
        }
        return true;
    }
    return false;
}

static std::string strip_wrapping_stars(const std::string & value) {
    // This is for replaying file_search shell bridge back into a file_search call, Codex will replay this later
    const size_t first = value.find_first_not_of('*');
    const size_t last = value.find_last_not_of('*');
    return first == std::string::npos ? "" : value.substr(first, last - first + 1);
}

static bool responses_shell_bridge_to_file_search(
        const json & action,
        const std::string & wrapper,
        json & arguments) {
    if (wrapper.empty() || !action.is_object() || json_value(action, "type", std::string()) != "exec") {
        return false;
    }
    if (!action.contains("command") || !action.at("command").is_array()) {
        return false;
    }
    const json & command = action.at("command");
    if (command.size() < 2 || !command.at(0).is_string() || command.at(0).get<std::string>() != wrapper) {
        return false;
    }

    bool files_mode = false;
    std::string query;
    std::string path;
    for (size_t i = 1; i < command.size(); ++i) {
        if (!command.at(i).is_string()) {
            continue;
        }
        const std::string arg = command.at(i).get<std::string>();
        if (arg == "--files") {
            files_mode = true;
        } else if (arg == "--" && i + 1 < command.size() && command.at(i + 1).is_string()) {
            query = command.at(i + 1).get<std::string>();
            if (i + 2 < command.size() && command.at(i + 2).is_string()) {
                path = command.at(i + 2).get<std::string>();
            }
            break;
        } else if (files_mode && arg == "--glob" && i + 1 < command.size() && command.at(i + 1).is_string()) {
            const std::string glob = command.at(i + 1).get<std::string>();
            if (!glob.empty() && glob[0] != '!') {
                query = strip_wrapping_stars(glob);
            }
            i++;
        }
    }
    if (query.empty()) {
        return false;
    }

    arguments = json {
        {"query", query},
        {"mode",  files_mode ? "files" : "content"},
    };
    if (!path.empty() && path != ".") {
        arguments["path"] = path;
    }
    return true;
}

static void append_tool_output_message(std::vector<json> & messages, const json & item) {
    const std::string call_id = json_value(item, "call_id", std::string());
    if (call_id.empty()) {
        messages.push_back(json {
            {"role", "assistant"},
            {"content", json::array({responses_make_text_content("[tool output missing call_id: " + item.dump() + "]")})},
        });
        return;
    }

    json content = json("[tool output missing output]");
    if (item.contains("output")) {
        content = encode_tool_output_content(item.at("output"));
    } else if (item.contains("tools")) {
        content = item.at("tools").dump();
    }
    messages.push_back(json {
        {"content",      content},
        {"role",         "tool"},
        {"tool_call_id", call_id},
    });
}

static bool strip_shell_output_metadata_text(const std::string & text, std::string & stripped) {
    for (const std::string marker : {
            "\nOutput:\n", "\r\nOutput:\r\n", "\nOutput:\r\n", "\r\nOutput:\n",
            "Output:\n", "Output:\r\n",
        }) {
        const size_t pos = text.find(marker);
        if (pos != std::string::npos) {
            stripped = text.substr(pos + marker.size());
            return true;
        }
    }
    return false;
}

static bool extract_web_search_bridge_output_text(const json & output, std::string & text) {
    if (output.is_string()) {
        const std::string raw = output.get<std::string>();
        if (strip_shell_output_metadata_text(raw, text)) {
            return true;
        }
        text = raw;
        return true;
    }
    if (output.is_array()) {
        std::string combined;
        for (const auto & item : output) {
            std::string part;
            if (item.is_string()) {
                part = item.get<std::string>();
            } else if (item.is_object() && exists_and_is_string(item, "text")) {
                part = item.at("text").get<std::string>();
            }
            if (!part.empty()) {
                if (!combined.empty()) {
                    combined += "\n";
                }
                combined += part;
            }
        }
        if (combined.empty()) {
            return false;
        }
        if (strip_shell_output_metadata_text(combined, text)) {
            return true;
        }
        text = combined;
        return true;
    }
    if (output.is_object()) {
        for (const char * key : {"output", "body", "content"}) {
            if (output.contains(key) && extract_web_search_bridge_output_text(output.at(key), text)) {
                return true;
            }
        }
    }
    return false;
}

static void append_web_search_bridge_output_message(std::vector<json> & messages, const json & item) {
    json normalized = item;
    if (normalized.contains("output")) {
        std::string text;
        if (extract_web_search_bridge_output_text(normalized.at("output"), text)) {
            normalized["output"] = text;
        }
    }
    append_tool_output_message(messages, normalized);
}

static std::string input_file_text(const json & input_item) {
    const std::string filename = json_value(input_item, "filename", std::string());
    const std::string file_id  = json_value(input_item, "file_id", std::string());
    const std::string file_url = json_value(input_item, "file_url", std::string());
    const std::string file_data = json_value(input_item, "file_data", std::string());
    const std::string label = filename.empty() ? (file_id.empty() ? "file" : file_id) : filename;

    if (!file_url.empty()) {
        return "[file: " + label + "] " + file_url;
    }
    if (!file_data.empty()) {
        if (file_data.rfind("data:image/", 0) == 0) {
            return "[image file: " + label + " omitted; use a vision-capable model to inspect image attachments]";
        }
        return "[file: " + label + "]\n" + truncate_for_prompt(file_data);
    }
    return "[file: " + label + " data unavailable]";
}

static std::string compaction_summary_text(const json & item) {
    if (exists_and_is_string(item, "encrypted_content")) {
        return item.at("encrypted_content").get<std::string>();
    }
    if (exists_and_is_string(item, "summary")) {
        return item.at("summary").get<std::string>();
    }
    if (exists_and_is_string(item, "content")) {
        return item.at("content").get<std::string>();
    }
    if (exists_and_is_array(item, "summary")) {
        std::string out;
        for (const auto & part : item.at("summary")) {
            if (exists_and_is_string(part, "text")) {
                if (!out.empty()) {
                    out += "\n";
                }
                out += part.at("text").get<std::string>();
            }
        }
        return out;
    }
    return "";
}

static void remember_responses_tool(json & metadata, const std::string & name, const json & resp_tool, const std::string & tool_type) {
    if (metadata.contains(name) && json_value(metadata.at(name), "type", std::string()) == "web_search" && tool_type == "function") {
        return;
    }
    json meta = resp_tool.is_object() ? resp_tool : json::object();
    meta["name"] = name;
    meta["type"] = tool_type;
    metadata[name] = meta;
}

static bool has_function_tool_named(const json & tools, const std::string & name) {
    if (!tools.is_array()) {
        return false;
    }
    for (const auto & tool : tools) {
        if (!tool.is_object() || json_value(tool, "type", std::string()) != "function") {
            continue;
        }
        if (json_value(tool, "name", std::string()) == name) {
            return true;
        }
    }
    return false;
}

static bool responses_tool_choice_requires_tool(const json & response_body, const std::string & tool_type) {
    if (!response_body.contains("tool_choice")) {
        return false;
    }

    const json & tool_choice = response_body.at("tool_choice");
    if (tool_choice.is_string()) {
        const std::string choice = tool_choice.get<std::string>();
        return choice == "required" || choice == "any";
    }

    if (!tool_choice.is_object()) {
        return false;
    }

    const std::string choice_type = json_value(tool_choice, "type", std::string());
    if (choice_type == "required" || choice_type == "any") {
        return true;
    }
    return choice_type == tool_type;
}

static json responses_tool_to_chatcmpl_tool(
        const json & resp_tool,
        json & metadata,
        const json & all_tools,
        const bool expose_hosted_web_search,
        const bool expose_hosted_file_search) {
    if (!resp_tool.is_object()) {
        SRV_WRN("skipping malformed Responses tool: %s\n", resp_tool.dump().c_str());
        return nullptr;
    }

    const std::string tool_type = json_value(resp_tool, "type", std::string("function"));
    json chat_tool;
    json fn;
    std::string name;

    if (tool_type == "function") {
        name = sanitize_tool_name(json_value(resp_tool, "name", std::string()), "function");
        fn = resp_tool;
        fn.erase("type");
        fn["name"] = name;
        if (!fn.contains("parameters") || !fn.at("parameters").is_object()) {
            fn["parameters"] = json {
                {"type", "object"},
                {"properties", json::object()},
                {"additionalProperties", true},
            };
        }
        if (!fn.contains("strict")) {
            fn["strict"] = true;
        }
    } else if (tool_type == "custom") {
        name = sanitize_tool_name(json_value(resp_tool, "name", std::string("custom_tool")), "custom_tool");
        fn = responses_function_tool(
                name,
                responses_custom_tool_description(resp_tool, name),
                responses_custom_tool_parameters(name));
    } else if (tool_type == "local_shell") {
        name = sanitize_tool_name(json_value(resp_tool, "name", std::string("local_shell")), "local_shell");
        fn = responses_function_tool(
                name,
                "Run a local shell command.",
                responses_object_schema(json {
                    {"command", json {{"type", "array"}, {"items", json {{"type", "string"}}}}},
                    {"cmd", json {{"type", "string"}}},
                    {"working_directory", json {{"type", "string"}}},
                    {"timeout_ms", json {{"type", "integer"}}},
                }));
    } else if (tool_type == "tool_search") {
        name = sanitize_tool_name(json_value(resp_tool, "name", std::string("tool_search")), "tool_search");
        fn = responses_function_tool(
                name,
                json_value(resp_tool, "description", std::string("Search for tools available to this Codex session.")),
                resp_tool.contains("parameters") && resp_tool.at("parameters").is_object()
                    ? resp_tool.at("parameters")
                    : responses_object_schema(
                            json {
                                {"query", json {{"type", "string"}}},
                            },
                            json::array({"query"})));
    } else if (tool_type == "task_list" || tool_type == "todo_list") {
        name = sanitize_tool_name(json_value(resp_tool, "name", std::string("update_plan")), "update_plan");
        fn = responses_function_tool(
                name,
                json_value(resp_tool, "description", std::string("Updates the task plan.")),
                resp_tool.contains("parameters") && resp_tool.at("parameters").is_object()
                    ? resp_tool.at("parameters")
                    : json {
                        {"type", "object"},
                        {"properties", json {
                            {"explanation", json {{"type", "string"}}},
                            {"plan", json {
                                {"type", "array"},
                                {"items", json {
                                    {"type", "object"},
                                    {"properties", json {
                                        {"step", json {{"type", "string"}}},
                                        {"status", json {{"type", "string"}, {"enum", json::array({"pending", "in_progress", "completed"})}}},
                                    }},
                                    {"required", json::array({"step", "status"})},
                                    {"additionalProperties", false},
                                }},
                            }},
                        }},
                        {"required", json::array({"plan"})},
                        {"additionalProperties", false},
                    });
    } else if (tool_type == "web_search") {
        name = "web_search";
        if (!expose_hosted_web_search) {
            return nullptr;
        }
        // Only expose hosted web_search when explicitly requested by tool_choice
        // or when the request configured a bridge for the emitted call.
        if (has_function_tool_named(all_tools, name)) {
            remember_responses_tool(metadata, name, resp_tool, tool_type);
            return nullptr;
        }
        fn = responses_function_tool(
                name,
                "Search or open web pages for current information. For search, provide a non-empty query.",
                responses_object_schema(json {
                    {"query", json {{"type", "string"}}},
                    {"queries", json {{"type", "array"}, {"items", json {{"type", "string"}}}}},
                    {"search_query", json {{"type", "array"}, {"items", json {{"type", "string"}}}}},
                    {"input", json {{"type", "string"}}},
                    {"url", json {{"type", "string"}}},
                    {"pattern", json {{"type", "string"}}},
                    {"action", json {{"type", "string"}, {"enum", json::array({"search", "open_page", "find_in_page"})}}},
                }));
    } else if (tool_type == "file_search") {
        name = "file_search";
        if (!expose_hosted_file_search) {
            return nullptr;
        }
        if (has_function_tool_named(all_tools, name)) {
            remember_responses_tool(metadata, name, resp_tool, tool_type);
            return nullptr;
        }
        fn = responses_function_tool(
                name,
                "Search files in the current workspace. Use mode='files' to find paths by name, or mode='content' to search file contents. Prefer this over shell commands such as find.",
                responses_object_schema(json {
                    {"query", json {{"type", "string"}}},
                    {"mode", json {{"type", "string"}, {"enum", json::array({"files", "content"})}}},
                    {"path", json {{"type", "string"}, {"description", "Optional relative workspace path to search."}}},
                }, json::array({"query"})));
    } else if (tool_type == "image_generation") {
        name = sanitize_tool_name(json_value(resp_tool, "name", std::string("image_generation")), "image_generation");
        fn = responses_function_tool(
                name,
                "Generate an image from a prompt.",
                responses_object_schema(
                        json {
                            {"prompt", json {{"type", "string"}}},
                        },
                        json::array({"prompt"})));
    } else {
        SRV_DBG("responses compat: item_type='%s', action='skip_unsupported_tool'\n", tool_type.c_str());
        return nullptr;
    }

    remember_responses_tool(metadata, name, resp_tool, tool_type);
    chat_tool["type"] = "function";
    chat_tool["function"] = fn;
    return chat_tool;
}

json server_chat_convert_responses_to_chatcmpl(
        const json & response_body,
        const std::string & hosted_web_search_wrapper,
        const std::string & hosted_file_search_wrapper) {
    if (!response_body.contains("input")) {
        throw std::invalid_argument("'input' is required");
    }
    if (!json_value(response_body, "previous_response_id", std::string{}).empty()) {
        SRV_WRN("%s", "Responses previous_response_id is accepted as replay-only state; callers must include prior input items\n");
    }

    json input_value = response_body.at("input");
    json chatcmpl_body = response_body;
    chatcmpl_body.erase("input");
    chatcmpl_body.erase("previous_response_id");
    std::vector<json> chatcmpl_messages;
    std::unordered_set<std::string> web_search_bridge_call_ids;
    std::unordered_set<std::string> file_search_bridge_call_ids;

    if (response_body.contains("instructions")) {
        chatcmpl_messages.push_back({
            {"role",    "system"},
            {"content", json_value(response_body, "instructions", std::string())},
        });
        chatcmpl_body.erase("instructions");
    }

    if (input_value.is_object()) {
        input_value = json::array({input_value});
    }

    if (input_value.is_string()) {
        // #responses_create-input-text_input
        chatcmpl_messages.push_back({
            {"role",    "user"},
            {"content", input_value},
        });
    } else if (input_value.is_array()) {
        // #responses_create-input-input_item_list
        for (json item : input_value) {
            bool merge_prev = !chatcmpl_messages.empty() && chatcmpl_messages.back().value("role", "") == "assistant";

            if (!item.is_object()) {
                chatcmpl_messages.push_back(json {
                    {"role", "user"},
                    {"content", json::array({responses_make_text_content("[unsupported Responses input item: " + item.dump() + "]")})},
                });
                continue;
            }

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
                const std::string role = item.at("role").get<std::string>();

                for (const json & input_item : item.at("content")) {
                    const std::string type = json_value(input_item, "type", std::string());

                    if (type == "input_text") {
                        if (!input_item.contains("text")) {
                            responses_append_recovery_text(chatcmpl_content, role, type, "input_text item missing text");
                            continue;
                        }
                        chatcmpl_content.push_back({
                            {"text", input_item.at("text")},
                            {"type", "text"},
                        });
                    } else if (type == "input_image") {
                        // While `detail` is marked as required,
                        // it has default value("auto") and can be omitted.
                        if (!input_item.contains("image_url")) {
                            responses_append_recovery_text(chatcmpl_content, role, type, "input_image item missing image_url");
                            continue;
                        }
                        chatcmpl_content.push_back({
                            {"image_url", json {
                                {"url", input_item.at("image_url")}
                            }},
                            {"type", "image_url"},
                        });
                    } else if (type == "input_file") {
                        chatcmpl_content.push_back(responses_make_text_content(input_file_text(input_item)));
                    } else {
                        const std::string item_type = type.empty() ? std::string("unknown") : type;
                        responses_append_recovery_text(
                                chatcmpl_content,
                                role,
                                item_type,
                                "unsupported input content type " + item_type);
                    }
                }

                if (item.contains("type")) {
                    item.erase("type");
                }
                if (item.contains("status")) {
                    item.erase("status");
                }
                // Merge system/developer messages into the first system message.
                // Many model templates (e.g. Qwen) require all system content at
                // position 0 and reject system messages elsewhere in the conversation.
                if (item.at("role") == "system" || item.at("role") == "developer") {
                    if (!chatcmpl_messages.empty() && chatcmpl_messages[0].value("role", "") == "system") {
                        auto & first_msg = chatcmpl_messages[0];
                        if (first_msg["content"].is_string()) {
                            std::string old_text = first_msg["content"].get<std::string>();
                            first_msg["content"] = json::array({json{{"text", old_text}, {"type", "text"}}});
                        }
                        auto & first_content = first_msg["content"];
                        for (const auto & part : chatcmpl_content) {
                            first_content.push_back(part);
                        }
                        continue;
                    }
                    item["role"] = "system";
                }
                item["content"] = chatcmpl_content;

                chatcmpl_messages.push_back(item);
            } else if (exists_and_is_string(item, "role") &&
                item.at("role") == "assistant" &&
                // status not checked (not always present, e.g. codex-cli omits it)
                // type == "message" for OutputMessage, absent for EasyInputMessage
                (!item.contains("type") || item.at("type") == "message")
            ) {
                // #responses_create-input-input_item_list-item-output_message
                std::vector<json> chatcmpl_content;
                const std::string role = item.at("role").get<std::string>();

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
                                responses_append_recovery_text(chatcmpl_content, role, type, type + " item missing text");
                                continue;
                            }
                            chatcmpl_content.push_back({
                                {"text", output_text.at("text")},
                                {"type", "text"},
                            });
                        } else if (type == "refusal") {
                            if (!exists_and_is_string(output_text, "refusal")) {
                                responses_append_recovery_text(chatcmpl_content, role, type, "refusal item missing refusal");
                                continue;
                            }
                            chatcmpl_content.push_back(responses_make_text_content(
                                "[assistant refusal] " + output_text.at("refusal").get<std::string>()));
                        } else {
                            const std::string item_type = type.empty() ? std::string("unknown") : type;
                            responses_append_recovery_text(
                                    chatcmpl_content,
                                    role,
                                    item_type,
                                    "unsupported assistant content type " + item_type);
                        }
                    }
                }

                if (merge_prev) {
                    auto & prev_msg = chatcmpl_messages.back();
                    if (!exists_and_is_array(prev_msg, "content")) {
                        prev_msg["content"] = json::array();
                    }
                    auto & prev_content = prev_msg["content"];
                    for (const auto & part : chatcmpl_content) {
                        prev_content.push_back(part);
                    }
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
                append_assistant_tool_call(chatcmpl_messages, make_tool_call(
                    item.at("name").get<std::string>(),
                    item.at("call_id").get<std::string>(),
                    parse_arguments_best_effort(item.at("arguments"))));
            } else if (exists_and_is_string(item, "type") &&
                (item.at("type") == "function_call" ||
                 item.at("type") == "custom_tool_call" ||
                 item.at("type") == "local_shell_call" ||
                 item.at("type") == "tool_search_call" ||
                 item.at("type") == "web_search_call" ||
                 item.at("type") == "file_search_call" ||
                 item.at("type") == "image_generation_call")
            ) {
                const std::string type = item.at("type").get<std::string>();
                std::string name = json_value(item, "name", std::string());
                json arguments = json::object();

                if (type == "custom_tool_call") {
                    if (name.empty()) {
                        name = "custom_tool";
                    }
                    arguments["input"] = json_value(item, "input", std::string());
                } else if (type == "local_shell_call") {
                    arguments = json_value(item, "action", json::object());
                    json bridged_arguments;
                    if (responses_shell_bridge_to_web_search(arguments, hosted_web_search_wrapper, bridged_arguments)) {
                        name = "web_search";
                        arguments = bridged_arguments;
                        const std::string call_id = json_value(item, "call_id", json_value(item, "id", std::string()));
                        if (!call_id.empty()) {
                            web_search_bridge_call_ids.insert(call_id);
                        }
                    } else if (responses_shell_bridge_to_file_search(arguments, hosted_file_search_wrapper, bridged_arguments)) {
                        name = "file_search";
                        arguments = bridged_arguments;
                        const std::string call_id = json_value(item, "call_id", json_value(item, "id", std::string()));
                        if (!call_id.empty()) {
                            file_search_bridge_call_ids.insert(call_id);
                        }
                    } else {
                        name = "local_shell";
                    }
                } else if (type == "tool_search_call") {
                    name = "tool_search";
                    arguments = json_value(item, "arguments", json::object());
                } else if (type == "web_search_call") {
                    name = "web_search";
                    arguments = json_value(item, "action", json::object());
                } else if (type == "file_search_call") {
                    name = "file_search";
                    arguments["query"] = json_value(item, "query", std::string());
                    if (arguments.at("query").get<std::string>().empty() &&
                        item.contains("queries") && item.at("queries").is_array() && !item.at("queries").empty() &&
                        item.at("queries").front().is_string()) {
                        arguments["query"] = item.at("queries").front();
                    }
                } else if (type == "image_generation_call") {
                    name = "image_generation";
                    arguments["prompt"] = json_value(item, "revised_prompt", std::string());
                } else if (type == "function_call") {
                    if (name.empty()) {
                        name = "function";
                    }
                    arguments = parse_arguments_best_effort(json_value(item, "arguments", json::object()));
                }

                append_assistant_tool_call(chatcmpl_messages, make_tool_call(
                    name,
                    json_value(item, "call_id", json_value(item, "id", std::string())),
                    arguments));
            } else if (exists_and_is_string(item, "call_id") &&
                exists_and_is_string(item, "type") &&
                (item.at("type") == "function_call_output" || item.at("type") == "custom_tool_call_output" ||
                 item.at("type") == "mcp_tool_call_output" || item.at("type") == "web_search_output" ||
                 item.at("type") == "file_search_output"   || item.at("type") == "tool_search_output")
            ) {
                // #responses_create-input-input_item_list-item-function_tool_call_output
                const std::string call_id = json_value(item, "call_id", std::string());
                if (!call_id.empty() &&
                    (web_search_bridge_call_ids.find(call_id) != web_search_bridge_call_ids.end() ||
                     file_search_bridge_call_ids.find(call_id) != file_search_bridge_call_ids.end())) {
                    append_web_search_bridge_output_message(chatcmpl_messages, item);
                } else {
                    append_tool_output_message(chatcmpl_messages, item);
                }
            } else if (exists_and_is_array(item, "summary") &&
                exists_and_is_string(item, "type") &&
                item.at("type") == "reasoning") {
                // #responses_create-input-input_item_list-item-reasoning

                // content can be: null, omitted, a string, or array of {type, text} objects.
                // Codex may send content:null or omit it entirely (issue openai/codex#11834).
                // OpenCode may send content as a plain string.
                // The spec uses array format: [{"type":"reasoning_text","text":"..."}].
                // encrypted_content (opaque string) is accepted but ignored for local models.
                std::string reasoning_text;
                if (!item.contains("content") || item.at("content").is_null()) {
                    // null or missing content - skip (encrypted_content only, or empty reasoning)
                } else if (item.at("content").is_string()) {
                    reasoning_text = item.at("content").get<std::string>();
                } else if (item.at("content").is_array() && !item.at("content").empty()
                           && exists_and_is_string(item.at("content")[0], "text")) {
                    reasoning_text = item.at("content")[0].at("text").get<std::string>();
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
            } else if (exists_and_is_string(item, "type") &&
                (item.at("type") == "compaction" || item.at("type") == "compaction_summary")) {
                const std::string summary = compaction_summary_text(item);
                if (!summary.empty()) {
                    chatcmpl_messages.push_back(json {
                        {"role", "user"},
                        {"content", "Previous conversation summary:\n\n" + summary},
                    });
                }
            } else if (exists_and_is_string(item, "type") && item.at("type") == "ghost_snapshot") {
                // Ghost snapshots are IDE side, so should not affect the prompt.
            } else {
                chatcmpl_messages.push_back(json {
                    {"role", "assistant"},
                    {"content", json::array({responses_make_text_content("[unsupported Responses item: " + item.dump() + "]")})},
                });
            }
        }
    } else {
        throw std::invalid_argument("'input' must be a string, object, or array of objects");
    }

    chatcmpl_body["messages"] = chatcmpl_messages;

    json response_tools = json::array();
    if (response_body.contains("tools")) {
        chatcmpl_body.erase("tools");
        if (!response_body.at("tools").is_array()) {
            SRV_WRN("%s", "'tools' must be an array of objects; ignoring malformed tools field\n");
        } else {
            response_tools = response_body.at("tools");
        }
    }
    if (!hosted_file_search_wrapper.empty() && !has_function_tool_named(response_tools, "file_search")) {
        response_tools.push_back(json{{"type", "file_search"}});
    }
    if (!response_tools.empty()) {
        json responses_tool_metadata = json::object();
        std::vector<json> chatcmpl_tools;
        const bool expose_web_search =
            !hosted_web_search_wrapper.empty() || responses_tool_choice_requires_tool(response_body, "web_search");
        const bool expose_file_search = !hosted_file_search_wrapper.empty();
        for (const json & resp_tool : response_tools) {
            json chatcmpl_tool = responses_tool_to_chatcmpl_tool(
                    resp_tool,
                    responses_tool_metadata,
                    response_tools,
                    expose_web_search,
                    expose_file_search);
            if (!chatcmpl_tool.is_null()) {
                chatcmpl_tools.push_back(chatcmpl_tool);
            }
        }
        if (!chatcmpl_tools.empty()) {
            chatcmpl_body["tools"] = chatcmpl_tools;
        }
        if (!responses_tool_metadata.empty()) {
            chatcmpl_body["__responses_tool_metadata"] = responses_tool_metadata;
        }
    }

    if (chatcmpl_body.contains("tool_choice") && chatcmpl_body.at("tool_choice").is_object()) {
        json choice = chatcmpl_body.at("tool_choice");
        const std::string choice_type = json_value(choice, "type", std::string());
        if (choice_type == "auto" || choice_type == "none" || choice_type == "required" || choice_type == "any") {
            chatcmpl_body["tool_choice"] = choice_type == "any" ? "required" : choice_type;
        } else {
            std::string name = choice_type == "web_search" || choice_type == "file_search"
                ? choice_type
                : json_value(choice, "name", std::string());
            for (const char * key : {"function", "tool"}) {
                if (name.empty() && choice.contains(key) && choice.at(key).is_object()) {
                    name = json_value(choice.at(key), "name", std::string());
                }
            }
            if (name.empty() && choice_type != "function") {
                name = choice_type.empty() ? "tool" : choice_type;
            }
            if (!name.empty() && chatcmpl_body.contains("tools") && chatcmpl_body.at("tools").is_array()) {
                const std::string selected_name = sanitize_tool_name(name, "tool");
                json selected_tools = json::array();
                for (const auto & tool : chatcmpl_body.at("tools")) {
                    if (tool.is_object() && tool.contains("function") && tool.at("function").is_object() &&
                        json_value(tool.at("function"), "name", std::string()) == selected_name) {
                        selected_tools.push_back(tool);
                    }
                }
                if (!selected_tools.empty()) {
                    chatcmpl_body["tools"] = selected_tools;
                }
            }
            chatcmpl_body["tool_choice"] = "required";
        }
    }

    if (response_body.contains("max_output_tokens")) {
        chatcmpl_body.erase("max_output_tokens");
        chatcmpl_body["max_tokens"] = response_body["max_output_tokens"];
    }

    // Strip Responses-only keys that have no chat completions equivalent
    // (e.g. Codex CLI sends store, include, prompt_cache_key, web_search)
    for (const char * key : {
        "store", "include", "prompt_cache_key", "web_search",
        "text", "truncation", "metadata", "reasoning", "background", 
        "service_tier", "safety_identifier", "max_tool_calls",
    }) {
        chatcmpl_body.erase(key);
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
