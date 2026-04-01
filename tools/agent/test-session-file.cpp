#include "session-file.h"

#include <cassert>
#include <cstdio>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

static const char * TEST_FILE = "/tmp/test-llama-session.jsonl";

static void cleanup() {
    fs::remove(TEST_FILE);
}

static void test_write_and_load() {
    cleanup();

    // Write a session
    {
        session_file sf;
        assert(sf.open(TEST_FILE));
        sf.write_header("/tmp/project");

        sf.append_message({{"role", "user"}, {"content", "hello"}});
        sf.append_message({{"role", "assistant"}, {"content", "hi there"}});
        sf.append_message({{"role", "user"}, {"content", "read foo.txt"}});
        sf.append_message({{"role", "assistant"}, {"content", ""},
                           {"tool_calls", json::array({{{"function", {{"name", "read"}, {"arguments", "{}"}}}}}) }});
        sf.append_message({{"role", "tool"}, {"tool_call_id", "call_1"}, {"name", "read"}, {"content", "file contents"}});

        assert(sf.message_count() == 5);
    }

    // Load it back
    auto loaded = session_file::load(TEST_FILE);
    assert(loaded.has_value());
    assert(loaded->messages.size() == 5);
    assert(loaded->previous_summary.empty());
    assert(loaded->messages[0]["role"] == "user");
    assert(loaded->messages[0]["content"] == "hello");
    assert(loaded->messages[4]["role"] == "tool");

    printf("  PASS: write_and_load\n");
    cleanup();
}

static void test_compaction_and_load() {
    cleanup();

    // Write a session with compaction
    {
        session_file sf;
        assert(sf.open(TEST_FILE));
        sf.write_header("/tmp/project");

        // 6 messages (indices 0-5)
        sf.append_message({{"role", "user"}, {"content", "msg0"}});
        sf.append_message({{"role", "assistant"}, {"content", "msg1"}});
        sf.append_message({{"role", "tool"}, {"content", "msg2"}});
        sf.append_message({{"role", "assistant"}, {"content", "msg3"}});
        sf.append_message({{"role", "tool"}, {"content", "msg4"}});
        sf.append_message({{"role", "assistant"}, {"content", "msg5"}});

        // Compact: keep from index 3 (msg3, msg4, msg5)
        sf.append_compaction("## Goal\nTest summary", 3);

        // New messages after compaction
        sf.append_message({{"role", "user"}, {"content", "msg6"}});
        sf.append_message({{"role", "assistant"}, {"content", "msg7"}});
    }

    auto loaded = session_file::load(TEST_FILE);
    assert(loaded.has_value());
    // Should have: msg3, msg4, msg5 (kept) + msg6, msg7 (new) = 5 messages
    assert(loaded->messages.size() == 5);
    assert(loaded->messages[0]["content"] == "msg3");
    assert(loaded->messages[2]["content"] == "msg5");
    assert(loaded->messages[3]["content"] == "msg6");
    assert(loaded->messages[4]["content"] == "msg7");
    assert(loaded->previous_summary == "## Goal\nTest summary");

    printf("  PASS: compaction_and_load\n");
    cleanup();
}

static void test_load_nonexistent() {
    auto loaded = session_file::load("/tmp/nonexistent-session-file.jsonl");
    assert(!loaded.has_value());
    printf("  PASS: load_nonexistent\n");
}

static void test_reopen() {
    cleanup();

    session_file sf;
    assert(sf.open(TEST_FILE));
    sf.write_header("/tmp/project");
    sf.append_message({{"role", "user"}, {"content", "hello"}});
    assert(sf.message_count() == 1);

    // Reopen (truncate)
    sf.reopen();
    sf.write_header("/tmp/project");
    assert(sf.message_count() == 0);

    sf.append_message({{"role", "user"}, {"content", "fresh start"}});

    auto loaded = session_file::load(TEST_FILE);
    assert(loaded.has_value());
    assert(loaded->messages.size() == 1);
    assert(loaded->messages[0]["content"] == "fresh start");

    printf("  PASS: reopen\n");
    cleanup();
}

static void test_resume_append() {
    cleanup();

    // First session run
    {
        session_file sf;
        sf.open(TEST_FILE);
        sf.write_header("/tmp/project");
        sf.append_message({{"role", "user"}, {"content", "first"}});
        sf.append_message({{"role", "assistant"}, {"content", "response1"}});
    }

    // Resume: load, then open for append
    auto loaded = session_file::load(TEST_FILE);
    assert(loaded.has_value());
    assert(loaded->messages.size() == 2);

    {
        session_file sf;
        sf.open(TEST_FILE);  // opens for append
        sf.append_message({{"role", "user"}, {"content", "second"}});
        sf.append_message({{"role", "assistant"}, {"content", "response2"}});
    }

    // Load again — should have all 4 messages
    auto loaded2 = session_file::load(TEST_FILE);
    assert(loaded2.has_value());
    assert(loaded2->messages.size() == 4);
    assert(loaded2->messages[0]["content"] == "first");
    assert(loaded2->messages[3]["content"] == "response2");

    printf("  PASS: resume_append\n");
    cleanup();
}

int main() {
    printf("Running session file tests...\n");
    test_write_and_load();
    test_compaction_and_load();
    test_load_nonexistent();
    test_reopen();
    test_resume_append();
    printf("\nAll session file tests passed!\n");
    return 0;
}
