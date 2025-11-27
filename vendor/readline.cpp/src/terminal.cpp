#include "readline/terminal.h"
#include "readline/errors.h"
#include <stdexcept>
#include <iostream>

#ifdef _WIN32
    #include <io.h>
    #include <cstdio>
    #define STDIN_FILENO _fileno(stdin)
#else
    #include <signal.h>
    #include <cstdio>
    #include <unistd.h>
    #include <cerrno>
#endif

namespace readline {

Terminal::Terminal()
    : raw_mode_(false), stop_io_loop_(false) {

#ifdef _WIN32
    input_handle_ = GetStdHandle(STD_INPUT_HANDLE);
    output_handle_ = GetStdHandle(STD_OUTPUT_HANDLE);

    if (input_handle_ == INVALID_HANDLE_VALUE || output_handle_ == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("Failed to get console handles");
    }

    if (!is_terminal(STDIN_FILENO)) {
        throw std::runtime_error("stdin is not a terminal");
    }
#else
    fd_ = STDIN_FILENO;

    if (!is_terminal(fd_)) {
        throw std::runtime_error("stdin is not a terminal");
    }
#endif

    // Don't start I/O thread yet - will be started when needed
}

Terminal::~Terminal() {
    if (raw_mode_) {
        unset_raw_mode();
    }

    stop_io_loop_ = true;
    queue_cv_.notify_all();

    // Detach the I/O thread - it will be terminated when the process exits
    // We can't safely join it because it may be blocked on read()
    if (io_thread_.joinable()) {
        io_thread_.detach();
    }
}

void Terminal::set_raw_mode() {
    if (raw_mode_) {
        return;
    }

#ifdef _WIN32
    // Get current console mode
    if (!GetConsoleMode(input_handle_, &original_input_mode_)) {
        throw std::runtime_error("Failed to get console input mode");
    }
    if (!GetConsoleMode(output_handle_, &original_output_mode_)) {
        throw std::runtime_error("Failed to get console output mode");
    }

    // Set raw mode for input
    DWORD input_mode = original_input_mode_;
    input_mode &= ~(ENABLE_ECHO_INPUT | ENABLE_LINE_INPUT | ENABLE_PROCESSED_INPUT);
    input_mode |= ENABLE_VIRTUAL_TERMINAL_INPUT;

    if (!SetConsoleMode(input_handle_, input_mode)) {
        throw std::runtime_error("Failed to set console to raw mode");
    }

    // Enable virtual terminal processing for output (for ANSI escape sequences)
    DWORD output_mode = original_output_mode_;
    output_mode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING | DISABLE_NEWLINE_AUTO_RETURN;

    if (!SetConsoleMode(output_handle_, output_mode)) {
        // Restore input mode if output mode fails
        SetConsoleMode(input_handle_, original_input_mode_);
        throw std::runtime_error("Failed to enable virtual terminal processing");
    }
#else
    // Get current terminal settings
    if (tcgetattr(fd_, &original_termios_) < 0) {
        throw std::runtime_error("Failed to get terminal attributes");
    }

    struct termios raw = original_termios_;

    // Set raw mode flags
    raw.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL | IXON);
    raw.c_lflag &= ~(ECHO | ECHONL | ICANON | ISIG | IEXTEN);
    raw.c_cflag &= ~(CSIZE | PARENB);
    raw.c_cflag |= CS8;
    raw.c_cc[VMIN] = 1;
    raw.c_cc[VTIME] = 0;

    if (tcsetattr(fd_, TCSAFLUSH, &raw) < 0) {
        throw std::runtime_error("Failed to set terminal to raw mode");
    }

    // Disable stdout buffering for immediate character display
    std::setvbuf(stdout, nullptr, _IONBF, 0);
#endif

    raw_mode_ = true;

    // Start I/O thread now that raw mode is set
    if (!io_thread_.joinable()) {
        io_thread_ = std::thread(&Terminal::io_loop, this);
    }
}

void Terminal::unset_raw_mode() {
    if (!raw_mode_) {
        return;
    }

#ifdef _WIN32
    if (!SetConsoleMode(input_handle_, original_input_mode_)) {
        throw std::runtime_error("Failed to restore console input mode");
    }
    if (!SetConsoleMode(output_handle_, original_output_mode_)) {
        throw std::runtime_error("Failed to restore console output mode");
    }
#else
    if (tcsetattr(fd_, TCSANOW, &original_termios_) < 0) {
        throw std::runtime_error("Failed to restore terminal settings");
    }
#endif

    raw_mode_ = false;
}

bool Terminal::is_terminal(int fd) {
#ifdef _WIN32
    return _isatty(fd) != 0;
#else
    return isatty(fd) != 0;
#endif
}

void Terminal::io_loop() {
#ifdef _WIN32
    while (!stop_io_loop_) {
        DWORD num_events = 0;
        if (!GetNumberOfConsoleInputEvents(input_handle_, &num_events)) {
            break;
        }

        if (num_events == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        INPUT_RECORD input_record;
        DWORD num_read = 0;

        if (!ReadConsoleInput(input_handle_, &input_record, 1, &num_read)) {
            break;
        }

        if (num_read == 0) {
            continue;
        }

        // Only process key events
        if (input_record.EventType == KEY_EVENT && input_record.Event.KeyEvent.bKeyDown) {
            char c = input_record.Event.KeyEvent.uChar.AsciiChar;
            if (c != 0) {
                {
                    std::lock_guard<std::mutex> lock(queue_mutex_);
                    char_queue_.push(c);
                }
                queue_cv_.notify_one();
            }
        }
    }
#else
    while (!stop_io_loop_) {
        char c;
        ssize_t n = ::read(fd_, &c, 1);

        if (n < 0) {
            if (errno == EINTR || errno == EAGAIN) {
                continue;
            }
            break;
        }

        if (n == 0) {
            break;
        }

        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            char_queue_.push(c);
        }
        queue_cv_.notify_one();
    }
#endif
}

std::optional<char> Terminal::read() {
    std::unique_lock<std::mutex> lock(queue_mutex_);

    queue_cv_.wait(lock, [this] {
        return !char_queue_.empty() || stop_io_loop_;
    });

    if (stop_io_loop_ && char_queue_.empty()) {
        return std::nullopt;
    }

    char c = char_queue_.front();
    char_queue_.pop();
    return c;
}

std::optional<char> Terminal::try_read() {
    std::lock_guard<std::mutex> lock(queue_mutex_);

    if (char_queue_.empty()) {
        return std::nullopt;
    }

    char c = char_queue_.front();
    char_queue_.pop();
    return c;
}

} // namespace readline
