#pragma once

#include <string>

namespace readline {

// Control characters
constexpr char CHAR_NULL = 0;
constexpr char CHAR_LINE_START = 1;
constexpr char CHAR_BACKWARD = 2;
constexpr char CHAR_INTERRUPT = 3;
constexpr char CHAR_DELETE = 4;
constexpr char CHAR_LINE_END = 5;
constexpr char CHAR_FORWARD = 6;
constexpr char CHAR_BELL = 7;
constexpr char CHAR_CTRL_H = 8;
constexpr char CHAR_TAB = 9;
constexpr char CHAR_CTRL_J = 10;
constexpr char CHAR_KILL = 11;
constexpr char CHAR_CTRL_L = 12;
constexpr char CHAR_ENTER = 13;
constexpr char CHAR_NEXT = 14;
constexpr char CHAR_PREV = 16;
constexpr char CHAR_BCK_SEARCH = 18;
constexpr char CHAR_FWD_SEARCH = 19;
constexpr char CHAR_TRANSPOSE = 20;
constexpr char CHAR_CTRL_U = 21;
constexpr char CHAR_CTRL_W = 23;
constexpr char CHAR_CTRL_Y = 25;
constexpr char CHAR_CTRL_Z = 26;
constexpr char CHAR_ESC = 27;
constexpr char CHAR_SPACE = 32;
constexpr char CHAR_ESCAPE_EX = 91;
constexpr char CHAR_BACKSPACE = 127;

// Special keys
constexpr char KEY_DEL = 51;
constexpr char KEY_UP = 65;
constexpr char KEY_DOWN = 66;
constexpr char KEY_RIGHT = 67;
constexpr char KEY_LEFT = 68;
constexpr char META_END = 70;
constexpr char META_START = 72;

// ANSI escape sequences
constexpr const char* ESC = "\x1b";
constexpr const char* CURSOR_SAVE = "\x1b[s";
constexpr const char* CURSOR_RESTORE = "\x1b[u";
constexpr const char* CURSOR_EOL = "\x1b[E";
constexpr const char* CURSOR_BOL = "\x1b[1G";
constexpr const char* CURSOR_HIDE = "\x1b[?25l";
constexpr const char* CURSOR_SHOW = "\x1b[?25h";
constexpr const char* CLEAR_TO_EOL = "\x1b[K";
constexpr const char* CLEAR_LINE = "\x1b[2K";
constexpr const char* CLEAR_SCREEN = "\x1b[2J";
constexpr const char* CURSOR_RESET = "\x1b[0;0f";
constexpr const char* COLOR_GREY = "\x1b[38;5;245m";
constexpr const char* COLOR_DEFAULT = "\x1b[0m";
constexpr const char* COLOR_BOLD = "\x1b[1m";
constexpr const char* START_BRACKETED_PASTE = "\x1b[?2004h";
constexpr const char* END_BRACKETED_PASTE = "\x1b[?2004l";

// Cursor movement functions
inline std::string cursor_up_n(int n) {
    return std::string(ESC) + "[" + std::to_string(n) + "A";
}

inline std::string cursor_down_n(int n) {
    return std::string(ESC) + "[" + std::to_string(n) + "B";
}

inline std::string cursor_right_n(int n) {
    return std::string(ESC) + "[" + std::to_string(n) + "C";
}

inline std::string cursor_left_n(int n) {
    return std::string(ESC) + "[" + std::to_string(n) + "D";
}

// Bracketed paste
constexpr char CHAR_BRACKETED_PASTE = 50;
constexpr const char* CHAR_BRACKETED_PASTE_START = "00~";
constexpr const char* CHAR_BRACKETED_PASTE_END = "01~";

} // namespace readline
