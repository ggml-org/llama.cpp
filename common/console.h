// Console functions

#pragma once

#include "common.h"

#include <functional>
#include <string>
#include <vector>

enum display_type {
    DISPLAY_TYPE_RESET = 0,
    DISPLAY_TYPE_INFO,
    DISPLAY_TYPE_PROMPT,
    DISPLAY_TYPE_REASONING,
    DISPLAY_TYPE_USER_INPUT,
    DISPLAY_TYPE_ERROR,
    DISPLAY_TYPE_TOOL_STREAM,
};

namespace console {
    void init(bool use_simple_io, bool use_advanced_display);
    void cleanup();
    void set_display(display_type display);
    bool readline(std::string & line, bool multiline_input);

    using completion_callback = std::function<std::vector<std::pair<std::string, size_t>>(std::string_view, size_t)>;
    void set_completion_callback(completion_callback cb);

    // Clipboard image paste support (Ctrl+V).
    // The callback should attempt to read an image from the OS clipboard.
    // On success, write bytes and mime type to the output params and return true.
    using paste_image_callback = std::function<bool(std::vector<uint8_t> & bytes, std::string & mime)>;
    void set_paste_image_callback(paste_image_callback cb);

    // Returns and clears images accumulated during the last readline call.
    // Each pair is (bytes, mime_type).
    std::vector<std::pair<std::vector<uint8_t>, std::string>> take_pending_images();

    namespace spinner {
        void start();
        void stop();
    }

    // note: the logging API below output directly to stdout
    // it can negatively impact performance if used on inference thread
    // only use in in a dedicated CLI thread
    // for logging in inference thread, use log.h instead

    LLAMA_COMMON_ATTRIBUTE_FORMAT(1, 2)
    void log(const char * fmt, ...);

    LLAMA_COMMON_ATTRIBUTE_FORMAT(1, 2)
    void error(const char * fmt, ...);

    void flush();

    // RAII guard for atomic multi-part console output
    // Holds the console mutex for the lifetime of the object
    // Use this when you need to output multiple lines/parts atomically
    class output_guard {
    public:
        output_guard();
        ~output_guard();

        // Non-copyable, non-movable
        output_guard(const output_guard &) = delete;
        output_guard & operator=(const output_guard &) = delete;

        // Write to console (mutex already held)
        LLAMA_COMMON_ATTRIBUTE_FORMAT(2, 3)
        void write(const char * fmt, ...);

        // Write raw bytes to console (binary-safe, mutex already held)
        void write_raw(const char * data, size_t len);

        // Change display type (mutex already held)
        void set_display(display_type type);

        // Flush output (mutex already held)
        void flush();
    };
}
