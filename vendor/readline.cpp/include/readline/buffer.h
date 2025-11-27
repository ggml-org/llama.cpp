#pragma once

#include <string>
#include <vector>
#include <locale>
#include <codecvt>

namespace readline {

struct Prompt {
    std::string prompt = "> ";
    std::string alt_prompt = ". ";
    std::string placeholder = "";
    std::string alt_placeholder = "";
    bool use_alt = false;

    std::string get_prompt() const {
        return use_alt ? alt_prompt : prompt;
    }

    std::string get_placeholder() const {
        return use_alt ? alt_placeholder : placeholder;
    }
};

class Buffer {
public:
    explicit Buffer(const Prompt& prompt);
    ~Buffer() = default;

    void add(char32_t c);
    void remove();
    void delete_char();
    void delete_before();
    void delete_remaining();
    void delete_word();

    void move_left();
    void move_right();
    void move_left_word();
    void move_right_word();
    void move_to_start();
    void move_to_end();

    void replace(const std::u32string& text);
    void clear_screen();

    bool is_empty() const { return buffer_.empty(); }
    std::string to_string() const;
    size_t display_size() const;

private:
    void add_char(char32_t c, bool insert);
    void draw_remaining();
    int count_remaining_line_width(int place);
    bool get_line_spacing(int line) const;
    int char_width(char32_t c) const;
    std::string to_utf8(char32_t c) const;
    std::string to_utf8(const std::u32string& str) const;

    std::u32string buffer_;
    std::vector<bool> line_has_space_;
    Prompt prompt_;
    size_t pos_ = 0;
    size_t display_pos_ = 0;
    int width_ = 80;
    int height_ = 24;
    int line_width_ = 70;
};

} // namespace readline
