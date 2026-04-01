#pragma once

#include <nlohmann/json.hpp>

#include <fstream>
#include <optional>
#include <string>

using json = nlohmann::ordered_json;

struct loaded_session {
    json        messages;              // conversation messages (no system prompt)
    std::string previous_summary;      // from last compaction entry, empty if none
    size_t      total_messages_in_file = 0;  // total message entries in the file (for message_count sync)
};

// Append-only JSONL session file for persisting conversation state.
class session_file {
public:
    // Open (or create) a session file for appending.
    bool open(const std::string & path);

    // Reopen the file with truncation (used by /clear).
    void reopen();

    // Write the session header (first line, call once for new sessions).
    void write_header(const std::string & working_dir);

    // Append a single OAI-compat message entry.
    void append_message(const json & msg);

    // Append a compaction entry.
    // kept_from: index into the message entry sequence where kept messages start.
    void append_compaction(const std::string & summary, size_t kept_from);

    // Number of message entries tracked so far (written + loaded).
    size_t message_count() const { return messages_written_; }

    // Sync the message counter after loading an existing session.
    void set_message_count(size_t count) { messages_written_ = count; }

    // Load a session from an existing JSONL file.
    // Returns nullopt if the file doesn't exist or is invalid.
    // Load a session from an existing JSONL file.
    static std::optional<loaded_session> load(const std::string & path);

    // Get the sessions directory for a given config dir and working directory.
    // Returns: {config_dir}/sessions/{encoded_cwd}/
    static std::string get_session_dir(const std::string & config_dir, const std::string & working_dir);

    // Generate a new session file path with timestamp.
    static std::string new_session_path(const std::string & session_dir);

    // Find the most recent .jsonl file in a session directory.
    // Returns empty string if none found.
    static std::string find_latest_session(const std::string & session_dir);

private:
    void write_line(const json & entry);

    std::ofstream file_;
    std::string   path_;
    size_t        messages_written_ = 0;
};
