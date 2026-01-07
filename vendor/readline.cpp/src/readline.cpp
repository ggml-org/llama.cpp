#include "readline/readline.h"
#include "readline/types.h"
#include <iostream>
#include <signal.h>

namespace readline {

Readline::Readline(const Prompt& prompt)
    : prompt_(prompt),
      terminal_(std::make_unique<Terminal>()),
      history_(std::make_unique<History>()) {
}

std::u32string Readline::utf8_to_utf32(const std::string& str) {
    std::u32string result;
    size_t i = 0;
    while (i < str.length()) {
        char32_t codepoint = 0;
        unsigned char c = str[i];

        if (c <= 0x7F) {
            codepoint = c;
            i += 1;
        } else if ((c & 0xE0) == 0xC0) {
            if (i + 1 >= str.length()) break;
            codepoint = ((c & 0x1F) << 6) | (str[i + 1] & 0x3F);
            i += 2;
        } else if ((c & 0xF0) == 0xE0) {
            if (i + 2 >= str.length()) break;
            codepoint = ((c & 0x0F) << 12) | ((str[i + 1] & 0x3F) << 6) | (str[i + 2] & 0x3F);
            i += 3;
        } else if ((c & 0xF8) == 0xF0) {
            if (i + 3 >= str.length()) break;
            codepoint = ((c & 0x07) << 18) | ((str[i + 1] & 0x3F) << 12) |
                       ((str[i + 2] & 0x3F) << 6) | (str[i + 3] & 0x3F);
            i += 4;
        } else {
            i += 1;
            continue;
        }

        result += codepoint;
    }
    return result;
}

std::string Readline::utf32_to_utf8(const std::u32string& str) {
    std::string result;
    for (char32_t c : str) {
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
    }
    return result;
}

bool Readline::check_interrupt() {
    // Ensure raw mode is set
    if (!terminal_->is_raw_mode()) {
        terminal_->set_raw_mode();
    }

    // Check if there's input available without blocking
    auto opt_r = terminal_->try_read();
    if (opt_r && *opt_r == CHAR_INTERRUPT) {
        return true;
    }
    return false;
}

std::string Readline::readline() {
    // Ensure raw mode is set and I/O thread is running
    if (!terminal_->is_raw_mode()) {
        terminal_->set_raw_mode();
    }

    std::string prompt = prompt_.get_prompt();
    if (pasting_) {
        prompt = prompt_.alt_prompt;
    }
    std::cout << prompt << std::flush;

    Buffer buf(prompt_);

    bool esc = false;
    bool escex = false;
    bool meta_del = false;
    std::u32string current_line_buf;

    while (true) {
        bool show_placeholder = !pasting_ || prompt_.use_alt;
        if (buf.is_empty() && show_placeholder) {
            std::string ph = prompt_.get_placeholder();
            std::cout << COLOR_GREY << ph << cursor_left_n(static_cast<int>(ph.length()))
                     << COLOR_DEFAULT << std::flush;
        }

        auto opt_r = terminal_->read();
        if (!opt_r) {
            throw eof_error();
        }

        char r = *opt_r;

        if (buf.is_empty()) {
            std::cout << CLEAR_TO_EOL << std::flush;
        }

        if (escex) {
            escex = false;

            switch (r) {
            case KEY_UP:
                history_prev(&buf, current_line_buf);
                break;
            case KEY_DOWN:
                history_next(&buf, current_line_buf);
                break;
            case KEY_LEFT:
                buf.move_left();
                break;
            case KEY_RIGHT:
                buf.move_right();
                break;
            case CHAR_BRACKETED_PASTE: {
                std::string code;
                for (int i = 0; i < 3; ++i) {
                    auto c = terminal_->read();
                    if (c) {
                        code += *c;
                    }
                }
                if (code == CHAR_BRACKETED_PASTE_START) {
                    pasting_ = true;
                } else if (code == CHAR_BRACKETED_PASTE_END) {
                    pasting_ = false;
                }
                break;
            }
            case KEY_DEL:
                if (buf.display_size() > 0) {
                    buf.delete_char();
                }
                meta_del = true;
                break;
            case META_START:
                buf.move_to_start();
                break;
            case META_END:
                buf.move_to_end();
                break;
            default:
                continue;
            }
            continue;
        } else if (esc) {
            esc = false;

            switch (r) {
            case 'b':
                buf.move_left_word();
                break;
            case 'f':
                buf.move_right_word();
                break;
            case CHAR_BACKSPACE:
                buf.delete_word();
                break;
            case CHAR_ESCAPE_EX:
                escex = true;
                break;
            }
            continue;
        }

        switch (r) {
        case CHAR_NULL:
            continue;
        case CHAR_ESC:
            esc = true;
            break;
        case CHAR_INTERRUPT:
            throw interrupt_error();
        case CHAR_PREV:
            history_prev(&buf, current_line_buf);
            break;
        case CHAR_NEXT:
            history_next(&buf, current_line_buf);
            break;
        case CHAR_LINE_START:
            buf.move_to_start();
            break;
        case CHAR_LINE_END:
            buf.move_to_end();
            break;
        case CHAR_BACKWARD:
            buf.move_left();
            break;
        case CHAR_FORWARD:
            buf.move_right();
            break;
        case CHAR_BACKSPACE:
        case CHAR_CTRL_H:
            buf.remove();
            break;
        case CHAR_TAB:
            for (int i = 0; i < 8; ++i) {
                buf.add(U' ');
            }
            break;
        case CHAR_DELETE:
            if (buf.display_size() > 0) {
                buf.delete_char();
            } else {
                throw eof_error();
            }
            break;
        case CHAR_KILL:
            buf.delete_remaining();
            break;
        case CHAR_CTRL_U:
            buf.delete_before();
            break;
        case CHAR_CTRL_L:
            buf.clear_screen();
            break;
        case CHAR_CTRL_W:
            buf.delete_word();
            break;
        case CHAR_CTRL_Z:
#ifndef _WIN32
            kill(0, SIGSTOP);
#endif
            return "";
        case CHAR_ENTER:
        case CHAR_CTRL_J: {
            std::string output = buf.to_string();
            if (!output.empty()) {
                history_->add(output);
            }
            buf.move_to_end();
            std::cout << std::endl;
            return output;
        }
        default:
            if (meta_del) {
                meta_del = false;
                continue;
            }
            if (r >= CHAR_SPACE || r == CHAR_ENTER || r == CHAR_CTRL_J) {
                buf.add(static_cast<char32_t>(static_cast<unsigned char>(r)));
            }
        }
    }
}

void Readline::history_prev(Buffer* buf, std::u32string& current_line_buf) {
    if (history_->pos > 0) {
        if (history_->pos == history_->size()) {
            current_line_buf = utf8_to_utf32(buf->to_string());
        }
        buf->replace(utf8_to_utf32(history_->prev()));
    }
}

void Readline::history_next(Buffer* buf, std::u32string& current_line_buf) {
    if (history_->pos < history_->size()) {
        buf->replace(utf8_to_utf32(history_->next()));
        if (history_->pos == history_->size()) {
            buf->replace(current_line_buf);
        }
    }
}

} // namespace readline
