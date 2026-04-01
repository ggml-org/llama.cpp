#include "compaction.h"

#include <cassert>
#include <cstdio>
#include <string>

static void test_estimate_tokens() {
    // Simple user message
    json msg = {{"role", "user"}, {"content", "Hello world"}};
    int32_t tokens = compaction_estimate_tokens(msg);
    assert(tokens == 3); // 11 chars / 4 = 2.75, ceil = 3

    // Assistant with tool calls
    json assistant = {
        {"role", "assistant"},
        {"content", "Let me read that file."},
        {"tool_calls", json::array({
            {{"function", {{"name", "read"}, {"arguments", R"({"file_path":"/foo/bar.txt","limit":100})"}}}}
        })}
    };
    int32_t t = compaction_estimate_tokens(assistant);
    assert(t > 0);

    // Empty message
    json empty = {{"role", "user"}, {"content", ""}};
    assert(compaction_estimate_tokens(empty) == 0);

    printf("  PASS: estimate_tokens\n");
}

static void test_estimate_total_tokens() {
    json messages = json::array({
        {{"role", "system"}, {"content", std::string(400, 'x')}},  // 100 tokens
        {{"role", "user"}, {"content", std::string(200, 'y')}},    // 50 tokens
        {{"role", "assistant"}, {"content", std::string(100, 'z')}} // 25 tokens
    });
    int32_t total = compaction_estimate_total_tokens(messages);
    assert(total == 175); // 100 + 50 + 25
    printf("  PASS: estimate_total_tokens\n");
}

static void test_find_cut_point_basic() {
    // Build a conversation: system, user1, assistant1, tool1, user2, assistant2
    json messages = json::array({
        {{"role", "system"}, {"content", std::string(400, 'x')}},     // idx 0
        {{"role", "user"}, {"content", std::string(200, 'a')}},       // idx 1 (valid cut)
        {{"role", "assistant"}, {"content", std::string(200, 'b')}},   // idx 2
        {{"role", "tool"}, {"content", std::string(200, 'c')}},       // idx 3 (NOT valid)
        {{"role", "user"}, {"content", std::string(200, 'd')}},       // idx 4 (valid cut)
        {{"role", "assistant"}, {"content", std::string(200, 'e')}},   // idx 5
    });

    // With keep_recent_tokens = 50 (200 chars / 4), should keep just the last message
    // Walking back: idx5=50, accumulated >= 50, snap to nearest user msg at or after idx 5
    // No user msg at idx 5, so it should snap back. The valid cuts are idx 1 and idx 4.
    size_t cut = compaction_find_cut_point(messages, 50);
    // Should keep idx 4+ (the last user message and assistant)
    assert(cut >= 1);
    printf("  PASS: find_cut_point basic (cut=%zu)\n", cut);

    // With keep_recent_tokens = 1000 (way more than total), should keep everything
    size_t cut2 = compaction_find_cut_point(messages, 1000);
    assert(cut2 == 1); // fallback: first valid cut = keep everything from idx 1
    printf("  PASS: find_cut_point large keep (cut=%zu)\n", cut2);
}

static void test_find_cut_point_never_cuts_tool() {
    json messages = json::array({
        {{"role", "system"}, {"content", "sys"}},
        {{"role", "user"}, {"content", std::string(100, 'a')}},
        {{"role", "assistant"}, {"content", std::string(100, 'b')}},
        {{"role", "tool"}, {"content", std::string(100, 'c')}},
        {{"role", "user"}, {"content", std::string(100, 'd')}},
    });

    size_t cut = compaction_find_cut_point(messages, 30);
    // Valid cuts: idx 1 (user), idx 2 (assistant after user), idx 4 (user)
    std::string role = messages[cut]["role"];
    assert(role == "user" || role == "assistant");
    assert(role != "tool");
    printf("  PASS: find_cut_point never cuts at tool (cut=%zu)\n", cut);
}

static void test_find_cut_point_single_user_multi_tool() {
    // Common agent pattern: one user prompt, many assistant/tool iterations
    json messages = json::array({
        {{"role", "system"}, {"content", std::string(400, 'x')}},     // idx 0
        {{"role", "user"}, {"content", std::string(100, 'a')}},       // idx 1 (valid)
        {{"role", "assistant"}, {"content", std::string(200, 'b')}},   // idx 2 (valid: after user)
        {{"role", "tool"}, {"content", std::string(200, 'c')}},       // idx 3 (NOT valid)
        {{"role", "assistant"}, {"content", std::string(200, 'd')}},   // idx 4 (valid: after tool)
        {{"role", "tool"}, {"content", std::string(200, 'e')}},       // idx 5 (NOT valid)
        {{"role", "assistant"}, {"content", std::string(200, 'f')}},   // idx 6 (valid: after tool)
        {{"role", "tool"}, {"content", std::string(200, 'g')}},       // idx 7 (NOT valid)
        {{"role", "assistant"}, {"content", std::string(200, 'h')}},   // idx 8 (valid: after tool)
    });

    // With small keep, should be able to cut somewhere after idx 1
    size_t cut = compaction_find_cut_point(messages, 100);
    assert(cut > 1); // Must be able to cut within the single-user-prompt session
    std::string role = messages[cut]["role"];
    assert(role == "user" || role == "assistant");
    printf("  PASS: find_cut_point single user multi tool (cut=%zu, role=%s)\n", cut, role.c_str());
}

static void test_serialize_conversation() {
    json messages = json::array({
        {{"role", "system"}, {"content", "You are helpful."}},
        {{"role", "user"}, {"content", "Read foo.txt"}},
        {{"role", "assistant"}, {"content", ""},
         {"tool_calls", json::array({
             {{"function", {{"name", "read"}, {"arguments", R"({"file_path":"foo.txt"})"}}}}
         })}},
        {{"role", "tool"}, {"content", "file contents here"}},
        {{"role", "assistant"}, {"content", "The file contains: file contents here"}},
    });

    std::string serialized = compaction_serialize_conversation(messages, 1, 5);

    assert(serialized.find("[User]: Read foo.txt") != std::string::npos);
    assert(serialized.find("[Assistant tool calls]: read(") != std::string::npos);
    assert(serialized.find("[Tool result]: file contents here") != std::string::npos);
    assert(serialized.find("[Assistant]: The file contains") != std::string::npos);

    printf("  PASS: serialize_conversation\n");
}

static void test_serialize_truncates_tool_output() {
    json messages = json::array({
        {{"role", "tool"}, {"content", std::string(5000, 'x')}},
    });

    std::string serialized = compaction_serialize_conversation(messages, 0, 1);
    // Should contain truncation marker
    assert(serialized.find("truncated") != std::string::npos);
    // Should be much shorter than 5000 chars
    assert(serialized.size() < 3000);

    printf("  PASS: serialize truncates tool output\n");
}

static void test_find_cut_point_minimal() {
    // Only system + one user message — nothing to cut
    json messages = json::array({
        {{"role", "system"}, {"content", "sys"}},
        {{"role", "user"}, {"content", "hello"}},
    });
    size_t cut = compaction_find_cut_point(messages, 10);
    assert(cut == 1);
    printf("  PASS: find_cut_point minimal\n");
}

int main() {
    printf("Running compaction tests...\n");
    test_estimate_tokens();
    test_estimate_total_tokens();
    test_find_cut_point_basic();
    test_find_cut_point_never_cuts_tool();
    test_find_cut_point_single_user_multi_tool();
    test_find_cut_point_minimal();
    test_serialize_conversation();
    test_serialize_truncates_tool_output();
    printf("\nAll compaction tests passed!\n");
    return 0;
}
