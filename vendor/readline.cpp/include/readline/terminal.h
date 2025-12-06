#pragma once

#include <optional>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <termios.h>
    #include <unistd.h>
#endif

namespace readline {

class Terminal {
public:
    Terminal();
    ~Terminal();

    void set_raw_mode();
    void unset_raw_mode();
    bool is_raw_mode() const { return raw_mode_; }
    std::optional<char> read();
    std::optional<char> try_read();
    bool is_terminal(int fd);

private:
    void io_loop();

#ifdef _WIN32
    HANDLE input_handle_;
    HANDLE output_handle_;
    DWORD original_input_mode_;
    DWORD original_output_mode_;
#else
    int fd_;
    struct termios original_termios_;
#endif
    bool raw_mode_;
    std::thread io_thread_;
    std::queue<char> char_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> stop_io_loop_;
};

} // namespace readline
