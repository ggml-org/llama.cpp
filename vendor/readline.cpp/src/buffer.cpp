#include "readline/buffer.h"
#include "readline/types.h"
#include <iostream>
#include <algorithm>
#include <cstring>

#ifdef _WIN32
    #define NOMINMAX
    #include <windows.h>
    #include <io.h>
    #define STDOUT_FILENO _fileno(stdout)
#else
    #include <sys/ioctl.h>
    #include <unistd.h>
#endif

namespace readline {

Buffer::Buffer(const Prompt& prompt)
    : prompt_(prompt) {

    // Get terminal size
#ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi)) {
        width_ = csbi.srWindow.Right - csbi.srWindow.Left + 1;
        height_ = csbi.srWindow.Bottom - csbi.srWindow.Top + 1;
    }
#else
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0) {
        width_ = ws.ws_col;
        height_ = ws.ws_row;
    }
#endif

    line_width_ = width_ - static_cast<int>(prompt_.get_prompt().length());
}

int Buffer::char_width(char32_t c) const {
    // Simplified width calculation
    // Full CJK width detection would require ICU or similar library
    if (c >= 0x1100 && c <= 0x115F) return 2;  // Hangul Jamo
    if (c >= 0x2E80 && c <= 0x9FFF) return 2;  // CJK
    if (c >= 0xAC00 && c <= 0xD7A3) return 2;  // Hangul Syllables
    if (c >= 0xF900 && c <= 0xFAFF) return 2;  // CJK Compatibility Ideographs
    if (c >= 0xFE10 && c <= 0xFE19) return 2;  // Vertical forms
    if (c >= 0xFE30 && c <= 0xFE6F) return 2;  // CJK Compatibility Forms
    if (c >= 0xFF00 && c <= 0xFF60) return 2;  // Fullwidth Forms
    if (c >= 0xFFE0 && c <= 0xFFE6) return 2;  // Fullwidth Forms
    if (c >= 0x20000 && c <= 0x2FFFD) return 2; // CJK Extensions
    if (c >= 0x30000 && c <= 0x3FFFD) return 2; // CJK Extensions
    return 1;
}

std::string Buffer::to_utf8(char32_t c) const {
    std::string result;
    if (c <= 0x7F) {
        result += static_cast<char>(c);
    } else if (c <= 0x7FF) {
        result += static_cast<char>(0xC0 | ((c >> 6) & 0x1F));
        result += static_cast<char>(0x80 | (c & 0x3F));
    } else if (c <= 0xFFFF) {
        result += static_cast<char>(0xE0 | ((c >> 12) & 0x0F));
        result += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
        result += static_cast<char>(0x80 | (c & 0x3F));
    } else if (c <= 0x10FFFF) {
        result += static_cast<char>(0xF0 | ((c >> 18) & 0x07));
        result += static_cast<char>(0x80 | ((c >> 12) & 0x3F));
        result += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
        result += static_cast<char>(0x80 | (c & 0x3F));
    }
    return result;
}

std::string Buffer::to_utf8(const std::u32string& str) const {
    std::string result;
    for (char32_t c : str) {
        result += to_utf8(c);
    }
    return result;
}

std::string Buffer::to_string() const {
    return to_utf8(buffer_);
}

size_t Buffer::display_size() const {
    size_t sum = 0;
    for (char32_t c : buffer_) {
        sum += char_width(c);
    }
    return sum;
}

bool Buffer::get_line_spacing(int line) const {
    if (line >= 0 && line < static_cast<int>(line_has_space_.size())) {
        return line_has_space_[line];
    }
    return false;
}

void Buffer::move_left() {
    if (pos_ > 0) {
        char32_t r = buffer_[pos_ - 1];
        int r_length = char_width(r);

        if (display_pos_ % line_width_ == 0) {
            std::cout << cursor_up_n(1) << CURSOR_BOL << cursor_right_n(width_);
            if (r_length == 2) {
                std::cout << cursor_left_n(1);
            }

            int line = static_cast<int>(display_pos_ / line_width_) - 1;
            bool has_space = get_line_spacing(line);
            if (has_space) {
                display_pos_ -= 1;
                std::cout << cursor_left_n(1);
            }
        } else {
            std::cout << cursor_left_n(r_length);
        }

        pos_ -= 1;
        display_pos_ -= r_length;
    }
}

void Buffer::move_right() {
    if (pos_ < buffer_.size()) {
        char32_t r = buffer_[pos_];
        int r_length = char_width(r);
        pos_ += 1;
        bool has_space = get_line_spacing(display_pos_ / line_width_);
        display_pos_ += r_length;

        if (display_pos_ % line_width_ == 0) {
            std::cout << cursor_down_n(1) << CURSOR_BOL
                     << cursor_right_n(static_cast<int>(prompt_.get_prompt().length()));
        } else if ((display_pos_ - r_length) % line_width_ == static_cast<size_t>(line_width_ - 1) && has_space) {
            std::cout << cursor_down_n(1) << CURSOR_BOL
                     << cursor_right_n(static_cast<int>(prompt_.get_prompt().length()) + r_length);
            display_pos_ += 1;
        } else if (!line_has_space_.empty() && display_pos_ % line_width_ == static_cast<size_t>(line_width_ - 1) && has_space) {
            std::cout << cursor_down_n(1) << CURSOR_BOL
                     << cursor_right_n(static_cast<int>(prompt_.get_prompt().length()));
            display_pos_ += 1;
        } else {
            std::cout << cursor_right_n(r_length);
        }
    }
}

void Buffer::move_left_word() {
    if (pos_ > 0) {
        bool found_nonspace = false;
        while (pos_ > 0) {
            char32_t v = buffer_[pos_ - 1];
            if (v == U' ') {
                if (found_nonspace) {
                    break;
                }
            } else {
                found_nonspace = true;
            }
            move_left();
        }
    }
}

void Buffer::move_right_word() {
    if (pos_ < buffer_.size()) {
        while (pos_ < buffer_.size()) {
            move_right();
            if (pos_ < buffer_.size() && buffer_[pos_] == U' ') {
                break;
            }
        }
    }
}

void Buffer::move_to_start() {
    if (pos_ > 0) {
        int curr_line = static_cast<int>(display_pos_ / line_width_);
        if (curr_line > 0) {
            std::cout << cursor_up_n(curr_line);
        }
        std::cout << CURSOR_BOL << cursor_right_n(static_cast<int>(prompt_.get_prompt().length()));
        pos_ = 0;
        display_pos_ = 0;
    }
}

void Buffer::move_to_end() {
    if (pos_ < buffer_.size()) {
        int curr_line = static_cast<int>(display_pos_ / line_width_);
        int total_lines = static_cast<int>(display_size() / line_width_);
        if (curr_line < total_lines) {
            std::cout << cursor_down_n(total_lines - curr_line);
            int remainder = static_cast<int>(display_size() % line_width_);
            std::cout << CURSOR_BOL
                     << cursor_right_n(static_cast<int>(prompt_.get_prompt().length()) + remainder);
        } else {
            std::cout << cursor_right_n(static_cast<int>(display_size() - display_pos_));
        }

        pos_ = buffer_.size();
        display_pos_ = display_size();
    }
}

void Buffer::add(char32_t c) {
    if (pos_ == buffer_.size()) {
        add_char(c, false);
    } else {
        add_char(c, true);
    }
}

void Buffer::add_char(char32_t c, bool insert) {
    int r_length = char_width(c);
    display_pos_ += r_length;

    if (pos_ > 0) {
        if (display_pos_ % line_width_ == 0) {
            std::cout << to_utf8(c) << "\n" << prompt_.alt_prompt;
            if (insert) {
                if (display_pos_ / line_width_ - 1 < line_has_space_.size()) {
                    line_has_space_[display_pos_ / line_width_ - 1] = false;
                }
            } else {
                line_has_space_.push_back(false);
            }
        } else if (display_pos_ % line_width_ < (display_pos_ - r_length) % line_width_) {
            if (insert) {
                std::cout << CLEAR_TO_EOL;
            }
            std::cout << "\n" << prompt_.alt_prompt;
            display_pos_ += 1;
            std::cout << to_utf8(c);
            if (insert) {
                if (display_pos_ / line_width_ - 1 < line_has_space_.size()) {
                    line_has_space_[display_pos_ / line_width_ - 1] = true;
                }
            } else {
                line_has_space_.push_back(true);
            }
        } else {
            std::cout << to_utf8(c);
        }
    } else {
        std::cout << to_utf8(c);
    }

    if (insert) {
        buffer_.insert(buffer_.begin() + pos_, c);
    } else {
        buffer_.push_back(c);
    }

    pos_ += 1;

    if (insert) {
        draw_remaining();
    }
}

int Buffer::count_remaining_line_width(int place) {
    int sum = 0;
    int counter = -1;
    int prev_len = 0;

    while (place <= line_width_) {
        counter += 1;
        sum += prev_len;
        if (pos_ + counter < buffer_.size()) {
            char32_t r = buffer_[pos_ + counter];
            place += char_width(r);
            prev_len = static_cast<int>(to_utf8(r).length());
        } else {
            break;
        }
    }

    return sum;
}

void Buffer::draw_remaining() {
    int place = 0;
    std::string remaining_text = to_utf8(buffer_.substr(pos_));
    if (pos_ > 0) {
        place = display_pos_ % line_width_;
    }
    std::cout << CURSOR_HIDE;

    int curr_line_length = count_remaining_line_width(place);
    std::string curr_line = remaining_text.substr(0, std::min(static_cast<size_t>(curr_line_length),
                                                               remaining_text.length()));

    if (!curr_line.empty()) {
        std::cout << CLEAR_TO_EOL << curr_line << cursor_left_n(static_cast<int>(curr_line.length()));
    } else {
        std::cout << CLEAR_TO_EOL;
    }

    std::cout << CURSOR_SHOW;
}

void Buffer::remove() {
    if (!buffer_.empty() && pos_ > 0) {
        char32_t r = buffer_[pos_ - 1];
        int r_length = char_width(r);

        if (display_pos_ % line_width_ == 0) {
            std::cout << CURSOR_BOL << CLEAR_TO_EOL << cursor_up_n(1)
                     << CURSOR_BOL << cursor_right_n(width_);

            bool has_space = get_line_spacing(display_pos_ / line_width_ - 1);
            if (has_space) {
                display_pos_ -= 1;
                std::cout << cursor_left_n(1);
            }

            if (r_length == 2) {
                std::cout << cursor_left_n(1) << "  " << cursor_left_n(2);
            } else {
                std::cout << " " << cursor_left_n(1);
            }
        } else {
            std::cout << cursor_left_n(r_length);
            for (int i = 0; i < r_length; ++i) {
                std::cout << " ";
            }
            std::cout << cursor_left_n(r_length);
        }

        pos_ -= 1;
        display_pos_ -= r_length;
        buffer_.erase(buffer_.begin() + pos_);

        if (pos_ < buffer_.size()) {
            draw_remaining();
        }
    }
}

void Buffer::delete_char() {
    if (!buffer_.empty() && pos_ < buffer_.size()) {
        buffer_.erase(buffer_.begin() + pos_);
        draw_remaining();
    }
}

void Buffer::delete_before() {
    while (pos_ > 0) {
        remove();
    }
}

void Buffer::delete_remaining() {
    while (pos_ < buffer_.size()) {
        delete_char();
    }
}

void Buffer::delete_word() {
    if (!buffer_.empty() && pos_ > 0) {
        bool found_nonspace = false;
        while (pos_ > 0) {
            char32_t v = buffer_[pos_ - 1];
            if (v == U' ') {
                if (!found_nonspace) {
                    remove();
                } else {
                    break;
                }
            } else {
                found_nonspace = true;
                remove();
            }
        }
    }
}

void Buffer::replace(const std::u32string& text) {
    display_pos_ = 0;
    pos_ = 0;
    int line_nums = static_cast<int>(display_size() / line_width_);

    buffer_.clear();

    std::cout << CURSOR_BOL << CLEAR_TO_EOL;

    for (int i = 0; i < line_nums; ++i) {
        std::cout << cursor_up_n(1) << CURSOR_BOL << CLEAR_TO_EOL;
    }

    std::cout << CURSOR_BOL << prompt_.get_prompt();

    for (char32_t c : text) {
        add(c);
    }
}

void Buffer::clear_screen() {
    std::cout << CLEAR_SCREEN << CURSOR_RESET << prompt_.get_prompt();
    if (is_empty()) {
        std::string ph = prompt_.get_placeholder();
        std::cout << COLOR_GREY << ph << cursor_left_n(static_cast<int>(ph.length())) << COLOR_DEFAULT;
    } else {
        size_t curr_pos = display_pos_;
        size_t curr_index = pos_;
        pos_ = 0;
        display_pos_ = 0;
        draw_remaining();
        std::cout << CURSOR_RESET << cursor_right_n(static_cast<int>(prompt_.get_prompt().length()));
        if (curr_pos > 0) {
            int target_line = static_cast<int>(curr_pos / line_width_);
            if (target_line > 0) {
                std::cout << cursor_down_n(target_line);
            }
            int remainder = static_cast<int>(curr_pos % line_width_);
            if (remainder > 0) {
                std::cout << cursor_right_n(remainder);
            }
            if (curr_pos % line_width_ == 0) {
                std::cout << CURSOR_BOL << prompt_.alt_prompt;
            }
        }
        pos_ = curr_index;
        display_pos_ = curr_pos;
    }
}

} // namespace readline
