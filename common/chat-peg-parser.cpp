#include "chat-peg-parser.h"

#include <nlohmann/json.hpp>
#include <stdexcept>

using json = nlohmann::json;
using Tag = common_chat_peg_tag;

static std::string_view trim_trailing_space(std::string_view sv, int max = -1) {
    int count = 0;
    while (!sv.empty() && std::isspace(static_cast<unsigned char>(sv.back()))) {
        if (max != -1 && count <= max) {
            break;
        }
        sv.remove_suffix(1);
        count++;
    }
    return sv;
}

static std::string_view trim_space(std::string_view sv) {
    // Trim leading whitespace
    while (!sv.empty() && std::isspace(static_cast<unsigned char>(sv.front()))) {
        sv.remove_prefix(1);
    }
    // Trim trailing whitespace
    while (!sv.empty() && std::isspace(static_cast<unsigned char>(sv.back()))) {
        sv.remove_suffix(1);
    }
    return sv;
}

void common_chat_peg_mapper::from_ast(const common_peg_ast_arena & arena, const common_peg_parse_result & result) {
    arena.visit(result, [this](const common_peg_ast_node & node) {
        map(node);
    });
}

void common_chat_peg_mapper::map(const common_peg_ast_node & node) {
    auto tag = static_cast<Tag>(node.tag_id);
    if (tag == Tag::REASONING) {
        // Concatenate to handle multiple REASONING tags (trim trailing space like functional mapper)
        auto text = std::string(trim_trailing_space(node.text));
        if (!text.empty()) {
            result.reasoning_content += text;
        }
    } else if (tag == Tag::CONTENT) {
        // Concatenate to handle multiple CONTENT tags (no trimming, like functional mapper)
        result.content += std::string(node.text);
    } else if (tag != Tag::NONE) {
        throw std::runtime_error("Unexpected tag for this mapper: " + std::to_string(static_cast<int>(tag)));
    }
}

void common_chat_peg_native_mapper::map(const common_peg_ast_node & node) {
    auto tag = static_cast<Tag>(node.tag_id);
    switch (tag) {
        case Tag::TOOL:
        case Tag::TOOL_CLOSE:
        case Tag::REASONING_BLOCK:
            // Structural wrappers - do nothing.
            break;
        case Tag::TOOL_OPEN:
            // Be lazy: don't create tool call here, wait for TOOL_NAME.
            // This avoids creating spurious tool calls during partial parsing.
            current_tool = nullptr;
            pending_tool_id.clear();
            break;
        case Tag::TOOL_ID:
            // Skip partial nodes - the ID isn't complete yet
            if (node.is_partial) {
                break;
            }
            {
                auto text = std::string(trim_trailing_space(node.text));
                // Strip surrounding quotes if present (JSON string value)
                if (text.size() >= 2 && text.front() == '"' && text.back() == '"') {
                    text = text.substr(1, text.size() - 2);
                }
                if (current_tool) {
                    current_tool->id = text;
                } else {
                    // Buffer ID - TOOL_ID may come before TOOL_NAME (e.g., Command R7B)
                    pending_tool_id = text;
                }
            }
            break;
        case Tag::TOOL_NAME:
            // Skip partial nodes - the name isn't complete yet.
            // Note: Using p.atomic(p.literal_tag(Tag::TOOL_NAME, name)) in parsers would
            // achieve the same effect by preventing partial nodes from being created,
            // but this mapper-level check is more defensive and handles all parsers uniformly.
            if (node.is_partial) {
                break;
            }
            // Create tool call lazily on TOOL_NAME, not on TOOL_OPEN.
            result.tool_calls.emplace_back();
            current_tool = &result.tool_calls.back();
            current_tool->name = std::string(trim_trailing_space(node.text));
            // Apply pending ID if any
            if (!pending_tool_id.empty()) {
                current_tool->id = pending_tool_id;
                pending_tool_id.clear();
            }
            break;
        case Tag::TOOL_ARGS:
            if (current_tool) {
                current_tool->arguments = std::string(trim_trailing_space(node.text));
            }
            break;
        case Tag::REASONING:
        case Tag::CONTENT:
        case Tag::NONE:
            common_chat_peg_mapper::map(node);
            break;
        default:
            throw std::runtime_error("Unexpected tag for this mapper: " + std::to_string(static_cast<int>(tag)));
    }
}

void common_chat_peg_constructed_mapper::map(const common_peg_ast_node & node) {
    auto tag = static_cast<Tag>(node.tag_id);
    switch (tag) {
        case Tag::TOOL:
        case Tag::TOOL_ARG:
            // Structural wrappers - do nothing.
            break;
        case Tag::TOOL_OPEN:
            current_tool = nullptr;
            arg_count = 0;
            break;
        case Tag::TOOL_NAME:
            // Skip partial nodes - the name isn't complete yet.
            // Note: Using p.atomic(p.literal_tag(Tag::TOOL_NAME, name)) in parsers would
            // achieve the same effect by preventing partial nodes from being created,
            // but this mapper-level check is more defensive and handles all parsers uniformly.
            if (node.is_partial) {
                break;
            }
            if (current_tool) {
                throw std::runtime_error("bad state");
            }
            result.tool_calls.emplace_back();
            current_tool = &result.tool_calls.back();
            current_tool->name = std::string(node.text);
            current_tool->arguments = "{";
            break;
        case Tag::TOOL_ARG_OPEN:
            needs_closing_quote = false;
            break;
        case Tag::TOOL_ARG_NAME:
            // Skip partial nodes - the name isn't complete yet
            if (node.is_partial) {
                break;
            }
            if (!current_tool) {
                throw std::runtime_error("bad state");
            }
            if (current_tool) {
                if (arg_count > 0) {
                    current_tool->arguments += ",";
                }
                current_tool->arguments += json(trim_trailing_space(node.text)).dump() + ":";
                ++arg_count;
            }
            break;
        case Tag::TOOL_ARG_STRING_VALUE:
            if (!current_tool) {
                throw std::runtime_error("bad state");
            }
            if (current_tool) {
                // Serialize to JSON, but exclude the end quote
                // Use trim_space to remove leading/trailing whitespace from raw string values
                std::string dumped = json(trim_space(node.text)).dump();
                current_tool->arguments += dumped.substr(0, dumped.size() - 1);
                needs_closing_quote = true;
            }
            break;
        case Tag::TOOL_ARG_CLOSE:
            if (!current_tool) {
                throw std::runtime_error("bad state");
            }
            if (current_tool && needs_closing_quote) {
                current_tool->arguments += "\"";
                needs_closing_quote = false;
            }
            break;
        case Tag::TOOL_ARG_JSON_VALUE:
            if (!current_tool) {
                throw std::runtime_error("bad state");
            }
            if (current_tool) {
                current_tool->arguments += std::string(trim_trailing_space(node.text));
            }
            break;
        case Tag::TOOL_CLOSE:
            // Skip partial nodes - we shouldn't close arguments until we've seen
            // the full closing tag.
            if (node.is_partial) {
                break;
            }
            if (!current_tool) {
                throw std::runtime_error("bad state");
            }
            if (current_tool) {
                if (needs_closing_quote) {
                    current_tool->arguments += "\"";
                    needs_closing_quote = false;
                }
                current_tool->arguments += "}";
                current_tool = nullptr;
            }
            break;
        case Tag::REASONING:
        case Tag::CONTENT:
        case Tag::NONE:
            common_chat_peg_mapper::map(node);
            break;
        default:
            throw std::runtime_error("Unexpected tag for this mapper: " + std::to_string(static_cast<int>(tag)));
    }
}

// ============================================================================
// Functional mapper implementations (used by experimental new PEG parsers in chat-parsers/)
// ============================================================================

// Helper: Convert JSON value to arguments string (handles object, string, null cases)
static std::string json_to_arguments(const json & j) {
    if (j.is_object()) {
        return j.dump();
    }
    if (j.is_string()) {
        return j.get<std::string>();
    }
    if (!j.is_null()) {
        return j.dump();
    }
    return "{}";
}

// Helper: Populate tool call from JSON object with configurable field names
static void populate_tool_from_json(
    common_chat_tool_call & tool,
    const json & item,
    const char * name_key,
    const char * id_key,
    const char * args_key
) {
    if (item.contains(name_key)) {
        tool.name = item.at(name_key).get<std::string>();
    }
    if (id_key && item.contains(id_key)) {
        const auto & id = item.at(id_key);
        tool.id = id.is_string() ? id.get<std::string>() : std::to_string(id.get<int>());
    }
    if (item.contains(args_key)) {
        tool.arguments = json_to_arguments(item.at(args_key));
    } else {
        tool.arguments = "{}";
    }
}

// Helper: Handle base content tags (REASONING, CONTENT)
static void handle_base_tags(common_chat_msg & result, const common_peg_ast_node & node) {
    switch (static_cast<Tag>(node.tag_id)) {
        case Tag::REASONING:
            result.reasoning_content += std::string(trim_trailing_space(node.text));
            break;
        case Tag::CONTENT:
            // Don't trim content - preserve trailing whitespace for interleaved content
            result.content += std::string(node.text);
            break;
        default:
            break;
    }
}
