// Single instance console_t class

#pragma once

#include "common.h"

#include <memory>
#include <string>

enum display_type {
    DISPLAY_TYPE_RESET = 0,
    DISPLAY_TYPE_INFO,
    DISPLAY_TYPE_PROMPT,
    DISPLAY_TYPE_REASONING,
    DISPLAY_TYPE_USER_INPUT,
    DISPLAY_TYPE_ERROR
};

class console_t {
private:
    console_t();

public:
    ~console_t();

    static console_t& get_instance();

    void init(bool use_simple_io, bool use_advanced_display);
    void cleanup();

    void set_display(display_type display);
    bool readline(std::string & line, bool multiline_input);

    struct spinner {
        static void start();
        static void stop();
    };
    friend spinner;


    // note: the logging API below output directly to stdout
    // it can negatively impact performance if used on inference thread
    // only use in in a dedicated CLI thread
    // for logging in inference thread, use log.h instead

    LLAMA_COMMON_ATTRIBUTE_FORMAT(1, 2)
    static void log(const char * fmt, ...);

    LLAMA_COMMON_ATTRIBUTE_FORMAT(1, 2)
    static void error(const char * fmt, ...);

    void flush();

private:
    void pop_cursor();
    int  put_codepoint(const char* utf8_codepoint, size_t length, int expectedWidth);
    void replace_last(char ch);
    void move_cursor(int delta);
    void move_word_left(size_t & char_pos, size_t & byte_pos, const std::vector<int> & widths, const std::string & line);
    void move_word_right(size_t & char_pos, size_t & byte_pos, const std::vector<int> & widths, const std::string & line);
    void move_to_line_start(size_t & char_pos, size_t & byte_pos, const std::vector<int> & widths);
    void move_to_line_end(size_t & char_pos, size_t & byte_pos, const std::vector<int> & widths, const std::string & line);
    void delete_at_cursor(std::string & line, std::vector<int> & widths, size_t & char_pos, size_t & byte_pos);
    void clear_current_line(const std::vector<int> & widths);
    void set_line_contents(std::string new_line, std::string & line, std::vector<int> & widths, size_t & char_pos,
                           size_t & byte_pos);
    bool readline_advanced(std::string & line, bool multiline_input);
    void draw_next_frame();

private:
    struct state_t;
    std::unique_ptr<state_t> state;

    struct history_t;
    std::unique_ptr<history_t> history;

    FILE*        out;

#if defined (_WIN32)
    void*        hConsole;
#else
    FILE*        tty;
    termios      initial_state;
#endif

};
