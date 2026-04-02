#include "compaction.h"

#include <sstream>

static const int TOOL_RESULT_MAX_CHARS = 2000;

// Count chars in a JSON value that may be a string or array of content blocks.
// Image blocks are estimated at ~1200 tokens (4800 chars at chars/4 heuristic).
static int32_t count_content_chars(const json & content) {
    if (content.is_string()) {
        return (int32_t) content.get<std::string>().size();
    }
    if (content.is_array()) {
        int32_t chars = 0;
        for (const auto & block : content) {
            if (block.contains("type") && block["type"] == "image_url") {
                chars += 4800;  // ~1200 tokens * 4 chars/token
            } else if (block.contains("text") && block["text"].is_string()) {
                chars += (int32_t) block["text"].get<std::string>().size();
            }
        }
        return chars;
    }
    return 0;
}

int32_t compaction_estimate_tokens(const json & msg) {
    int32_t chars = 0;

    if (msg.contains("content")) {
        chars += count_content_chars(msg["content"]);
    }

    if (msg.contains("reasoning_content") && msg["reasoning_content"].is_string()) {
        chars += (int32_t) msg["reasoning_content"].get<std::string>().size();
    }

    // Count tool call names and arguments
    if (msg.contains("tool_calls") && msg["tool_calls"].is_array()) {
        for (const auto & tc : msg["tool_calls"]) {
            if (tc.contains("function")) {
                const auto & fn = tc["function"];
                if (fn.contains("name") && fn["name"].is_string()) {
                    chars += (int32_t) fn["name"].get<std::string>().size();
                }
                if (fn.contains("arguments") && fn["arguments"].is_string()) {
                    chars += (int32_t) fn["arguments"].get<std::string>().size();
                }
            }
        }
    }

    return (chars + 3) / 4;
}

int32_t compaction_estimate_total_tokens(const json & messages) {
    int32_t total = 0;
    for (const auto & m : messages) {
        total += compaction_estimate_tokens(m);
    }
    return total;
}

size_t compaction_find_cut_point(const json & messages, int32_t keep_recent_tokens) {
    if (messages.size() <= 2) {
        return 1; // only system + at most one message, nothing to cut
    }

    // Collect valid cut points: user messages and assistant messages at turn boundaries.
    // A turn boundary is where the previous message is a tool result or user message.
    // Tool result messages are NEVER valid cut points (they must stay with their tool call).
    std::vector<size_t> valid_cuts;
    for (size_t i = 1; i < messages.size(); i++) {
        if (!messages[i].contains("role")) {
            continue;
        }
        const std::string & role = messages[i]["role"];
        if (role == "user") {
            valid_cuts.push_back(i);
        } else if (role == "assistant" && i > 1) {
            // Assistant at a turn boundary: previous message is tool result or user
            const std::string & prev_role = messages[i - 1].value("role", "");
            if (prev_role == "tool" || prev_role == "user") {
                valid_cuts.push_back(i);
            }
        }
    }

    if (valid_cuts.empty()) {
        return 1; // no user messages to cut at
    }

    // Walk backward from end, accumulating token estimates
    int32_t accumulated = 0;
    size_t cut_idx = valid_cuts[0]; // fallback: keep everything

    for (size_t i = messages.size() - 1; i >= 1; i--) {
        accumulated += compaction_estimate_tokens(messages[i]);
        if (accumulated >= keep_recent_tokens) {
            // Find the first valid cut point >= i
            for (size_t c : valid_cuts) {
                if (c >= i) {
                    cut_idx = c;
                    goto found;
                }
            }
            // No valid cut point at or after i — use the last one before i
            for (auto it = valid_cuts.rbegin(); it != valid_cuts.rend(); ++it) {
                if (*it < i) {
                    cut_idx = *it;
                    goto found;
                }
            }
            break;
        }
    }
found:

    return cut_idx;
}

static std::string truncate_for_summary(const std::string & text, int max_chars) {
    if ((int) text.size() <= max_chars) {
        return text;
    }
    return text.substr(0, max_chars) + "\n\n[... " + std::to_string(text.size() - max_chars) + " more characters truncated]";
}

std::string compaction_serialize_conversation(const json & messages, size_t start, size_t end) {
    std::ostringstream out;

    for (size_t i = start; i < end && i < messages.size(); i++) {
        const auto & msg = messages[i];
        if (!msg.contains("role")) {
            continue;
        }

        const std::string & role = msg["role"];

        if (role == "user" || role == "system") {
            std::string content;
            if (msg.contains("content")) {
                if (msg["content"].is_string()) {
                    content = msg["content"].get<std::string>();
                } else if (msg["content"].is_array()) {
                    for (const auto & block : msg["content"]) {
                        if (block.contains("text") && block["text"].is_string()) {
                            content += block["text"].get<std::string>();
                        }
                    }
                }
            }
            out << "[User]: " << content << "\n\n";
        } else if (role == "assistant") {
            // Reasoning
            if (msg.contains("reasoning_content") && msg["reasoning_content"].is_string() && !msg["reasoning_content"].get<std::string>().empty()) {
                out << "[Assistant thinking]: " << msg["reasoning_content"].get<std::string>() << "\n\n";
            }

            // Content
            if (msg.contains("content") && msg["content"].is_string() && !msg["content"].get<std::string>().empty()) {
                out << "[Assistant]: " << msg["content"].get<std::string>() << "\n\n";
            }

            // Tool calls
            if (msg.contains("tool_calls") && msg["tool_calls"].is_array()) {
                for (const auto & tc : msg["tool_calls"]) {
                    if (!tc.contains("function")) {
                        continue;
                    }
                    const auto & fn = tc["function"];
                    std::string name = fn.value("name", "unknown");
                    std::string args_str = fn.value("arguments", "");

                    // Try to parse arguments as JSON for key=value format
                    std::string formatted_args;
                    try {
                        auto args = json::parse(args_str);
                        bool first = true;
                        for (auto it = args.begin(); it != args.end(); ++it) {
                            if (!first) {
                                formatted_args += ", ";
                            }
                            formatted_args += it.key() + "=" + it.value().dump();
                            first = false;
                        }
                    } catch (...) {
                        formatted_args = args_str;
                    }

                    out << "[Assistant tool calls]: " << name << "(" << formatted_args << ")\n\n";
                }
            }
        } else if (role == "tool") {
            std::string content;
            if (msg.contains("content")) {
                if (msg["content"].is_string()) {
                    content = msg["content"].get<std::string>();
                } else if (msg["content"].is_array()) {
                    // Extract text blocks only (drop images for summarization)
                    for (const auto & block : msg["content"]) {
                        if (block.contains("text") && block["text"].is_string()) {
                            content += block["text"].get<std::string>();
                        }
                    }
                }
            }
            if (!content.empty()) {
                out << "[Tool result]: " << truncate_for_summary(content, TOOL_RESULT_MAX_CHARS) << "\n\n";
            }
        }
    }

    return out.str();
}

const char * SUMMARIZATION_SYSTEM_PROMPT =
    "You are a context summarization assistant. Your task is to read a conversation "
    "between a user and an AI coding assistant, then produce a structured summary "
    "following the exact format specified.\n\n"
    "Do NOT continue the conversation. Do NOT respond to any questions in the "
    "conversation. ONLY output the structured summary.";

const char * SUMMARIZATION_PROMPT =
    "The messages above are a conversation to summarize. Create a structured context "
    "checkpoint summary that another LLM will use to continue the work.\n\n"
    "Use this EXACT format:\n\n"
    "## Goal\n"
    "[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]\n\n"
    "## Constraints & Preferences\n"
    "- [Any constraints, preferences, or requirements mentioned by user]\n"
    "- [Or \"(none)\" if none were mentioned]\n\n"
    "## Progress\n"
    "### Done\n"
    "- [x] [Completed tasks/changes]\n\n"
    "### In Progress\n"
    "- [ ] [Current work]\n\n"
    "### Blocked\n"
    "- [Issues preventing progress, if any]\n\n"
    "## Key Decisions\n"
    "- **[Decision]**: [Brief rationale]\n\n"
    "## Next Steps\n"
    "1. [Ordered list of what should happen next]\n\n"
    "## Critical Context\n"
    "- [Any data, examples, or references needed to continue]\n"
    "- [Or \"(none)\" if not applicable]\n\n"
    "Keep each section concise. Preserve exact file paths, function names, and error messages.";

const char * UPDATE_SUMMARIZATION_PROMPT =
    "The messages above are NEW conversation messages to incorporate into the existing "
    "summary provided in <previous-summary> tags.\n\n"
    "Update the existing structured summary with new information. RULES:\n"
    "- PRESERVE all existing information from the previous summary\n"
    "- ADD new progress, decisions, and context from the new messages\n"
    "- UPDATE the Progress section: move items from \"In Progress\" to \"Done\" when completed\n"
    "- UPDATE \"Next Steps\" based on what was accomplished\n"
    "- PRESERVE exact file paths, function names, and error messages\n"
    "- If something is no longer relevant, you may remove it\n\n"
    "Use this EXACT format:\n\n"
    "## Goal\n"
    "[Preserve existing goals, add new ones if the task expanded]\n\n"
    "## Constraints & Preferences\n"
    "- [Preserve existing, add new ones discovered]\n\n"
    "## Progress\n"
    "### Done\n"
    "- [x] [Include previously done items AND newly completed items]\n\n"
    "### In Progress\n"
    "- [ ] [Current work - update based on progress]\n\n"
    "### Blocked\n"
    "- [Current blockers - remove if resolved]\n\n"
    "## Key Decisions\n"
    "- **[Decision]**: [Brief rationale] (preserve all previous, add new)\n\n"
    "## Next Steps\n"
    "1. [Update based on current state]\n\n"
    "## Critical Context\n"
    "- [Preserve important context, add new if needed]\n\n"
    "Keep each section concise. Preserve exact file paths, function names, and error messages.";
