#pragma once

// Private shared internals for agent-loop-*.cpp files.
// Not part of the public interface — do not include from outside tools/agent/.

#include <mutex>
#include <string>

// Serializes the prompt-formatting + task-posting path without serializing
// the full generation loop.  Used by generate_completion,
// generate_completion_streaming, and generate_summary.
extern std::mutex g_completion_mutex;

// Per-tool-call state used during streaming to display args incrementally
struct tool_stream_state {
    std::string accumulated_args;  // full JSON args text accumulated so far
    std::string name;              // tool name (set on first delta)
    bool header_printed = false;   // whether "› tool_name path" was shown
    std::string displayed_path;    // file_path already printed
    size_t displayed_lines = 0;    // number of content lines already printed
    size_t displayed_bytes = 0;    // byte offset in content_buffer past last printed line
    size_t content_scan_pos = 0;   // byte offset in accumulated_args where content value starts (0 = not found)
    std::string content_buffer;    // decoded content accumulated so far (avoids re-decoding from start)
    size_t content_raw_end = 0;    // how many bytes of accumulated_args have been decoded into content_buffer
    bool content_complete = false; // true once the closing quote of the content field was found
};

// Check for ESC key press without blocking (cross-platform)
bool check_escape_key();

// Extract a string-valued field from a partial (possibly incomplete) JSON object.
// Returns the decoded string value, or empty string if the key is not yet present.
// `complete` is set to true if the closing quote of the value was found.
std::string extract_partial_json_string(
    const std::string & json_fragment,
    const std::string & key,
    bool & complete);

// Incrementally decode more of a JSON string field into tcs.content_buffer.
// Returns true if the field's closing quote was found.
bool decode_field_incremental(
    tool_stream_state & tcs,
    const std::string & key);
