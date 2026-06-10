#include "common.h"
#include "log.h"

#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>
#include <algorithm>

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#       define NOMINMAX
#    endif
#    include <io.h>
#    include <windows.h>
#    define isatty _isatty
#    define fileno _fileno
#else
#    include <unistd.h>
#endif // defined(_WIN32)

int common_log_verbosity_thold = LOG_DEFAULT_LLAMA;

int common_log_get_verbosity_thold(void) {
    return common_log_verbosity_thold;
}

void common_log_set_verbosity_thold(int verbosity) {
    common_log_verbosity_thold = verbosity;
}

static int64_t t_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

// colors
enum common_log_col : int {
    COMMON_LOG_COL_DEFAULT = 0,
    COMMON_LOG_COL_BOLD,
    COMMON_LOG_COL_RED,
    COMMON_LOG_COL_GREEN,
    COMMON_LOG_COL_YELLOW,
    COMMON_LOG_COL_BLUE,
    COMMON_LOG_COL_MAGENTA,
    COMMON_LOG_COL_CYAN,
    COMMON_LOG_COL_WHITE,
};

// disable colors by default
static const char* g_col[] = {
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
};

struct common_log_entry {
    enum ggml_log_level level {GGML_LOG_LEVEL_INFO};

    std::vector<char> msg;

    int64_t timestamp { 0 };
    bool is_end       { false }; // signals the worker thread to stop
    bool prefix       { false };

    common_log_entry(size_t size = 256) : msg(size) { }

    void print(FILE * file = nullptr) const {
        FILE * fcur = file;
        if (!fcur) {
            // stderr displays DBG messages only when their verbosity level is not higher than the threshold
            // these messages will still be logged to a file
            if (level == GGML_LOG_LEVEL_DEBUG && common_log_verbosity_thold < LOG_DEFAULT_DEBUG) {
                return;
            }

            fcur = stdout;

            if (level != GGML_LOG_LEVEL_NONE) {
                fcur = stderr;
            }
        }

        if (level != GGML_LOG_LEVEL_NONE && level != GGML_LOG_LEVEL_CONT && prefix) {
            if (timestamp) {
                // [M.s.ms.us]
                fprintf(fcur, "%s%d.%02d.%03d.%03d%s ",
                        g_col[COMMON_LOG_COL_BLUE],
                        (int) (timestamp / 1000000 / 60),
                        (int) (timestamp / 1000000 % 60),
                        (int) (timestamp / 1000 % 1000),
                        (int) (timestamp % 1000),
                        g_col[COMMON_LOG_COL_DEFAULT]);
            }

            switch (level) {
                case GGML_LOG_LEVEL_INFO:  fprintf(fcur, "%sI %s", g_col[COMMON_LOG_COL_GREEN],   g_col[COMMON_LOG_COL_DEFAULT]); break;
                case GGML_LOG_LEVEL_WARN:  fprintf(fcur, "%sW %s", g_col[COMMON_LOG_COL_MAGENTA], ""                        ); break;
                case GGML_LOG_LEVEL_ERROR: fprintf(fcur, "%sE %s", g_col[COMMON_LOG_COL_RED],     ""                        ); break;
                case GGML_LOG_LEVEL_DEBUG: fprintf(fcur, "%sD %s", g_col[COMMON_LOG_COL_YELLOW],  ""                        ); break;
                default:
                    break;
            }
        }

        fprintf(fcur, "%s", msg.data());

        if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_DEBUG) {
            fprintf(fcur, "%s", g_col[COMMON_LOG_COL_DEFAULT]);
        }

        fflush(fcur);
    }
};

struct common_log {
    // sane limit to avoid out-of-memory under heavy profiling
    const size_t max_capacity = 4096;

    // default capacity - will be expanded if needed
    common_log(size_t capacity = 256) {
        file       = nullptr;
        prefix     = false;
        timestamps = false;
        running    = false;
        t_start    = t_us();

        queue.resize(capacity, common_log_entry(256));
        head = 0;
        tail = 0;

        resume();
    }

    ~common_log() {
        pause();
        if (file) {
            fclose(file);
        }
    }

private:
    std::mutex              mtx;
    std::thread             thrd;
    std::condition_variable cv_new;  // new entry
    std::condition_variable cv_full; // wait on full

    FILE * file;

    bool prefix;
    bool timestamps;
    bool running;

    int64_t t_start;

    // queue of entries
    std::vector<common_log_entry> queue;
    size_t head;
    size_t tail;

    bool print_entry(const common_log_entry & e) const {
        if (e.is_end) return true;

        e.print();
        if (file) {
            e.print(file);
        }
        return false;
    }

    bool flush_queue(const std::vector<common_log_entry> & old_queue, size_t old_head, size_t old_tail) const {
        bool stop = false;
        size_t h = old_head;
        size_t size = old_queue.size();
        while (h != old_tail && !stop) {
            stop = print_entry(old_queue[h]);
            h = (h + 1) % size;
        }
        return stop;
    }

    bool expand_and_flush(std::unique_lock<std::mutex> & lock) {
        size_t old_head = head;
        size_t old_tail = tail;
        size_t old_size = queue.size();

        size_t new_capacity = std::min(old_size * 2, max_capacity);
        std::vector<common_log_entry> new_queue(new_capacity, common_log_entry(256));

        queue.swap(new_queue);
        auto & old_queue = new_queue; // alias to make it clear we are flushing the old queue
        head = 0;
        tail = 0;
        cv_full.notify_all(); // wake up blocked producers

        lock.unlock(); // drop the lock during flushing

        bool stop = flush_queue(old_queue, old_head, old_tail);

        lock.lock();
        return stop;
    }

public:
    bool is_full() const {
        return ((tail + 1) % queue.size()) == head;
    }

    bool is_empty() const {
        return head == tail;
    }

    void add(enum ggml_log_level level, const char * fmt, va_list args) {
        std::unique_lock<std::mutex> lock(mtx);

        // block if the queue is full
        cv_full.wait(lock, [this]() { return !running || !is_full(); });

        if (!running) {
            // discard messages while the worker thread is paused
            return;
        }

        auto & entry = queue[tail];

        {
            // cannot use args twice, so make a copy in case we need to expand the buffer
            va_list args_copy;
            va_copy(args_copy, args);

#if 1
            const size_t n = vsnprintf(entry.msg.data(), entry.msg.size(), fmt, args);
            if (n >= entry.msg.size()) {
                entry.msg.resize(n + 1);
                vsnprintf(entry.msg.data(), entry.msg.size(), fmt, args_copy);
            }
#else
            // hack for bolding arguments

            std::stringstream ss;
            for (int i = 0; fmt[i] != 0; i++) {
                if (fmt[i] == '%') {
                    ss << LOG_COL_BOLD;
                    while (fmt[i] != ' ' && fmt[i] != ')' && fmt[i] != ']' && fmt[i] != 0) ss << fmt[i++];
                    ss << LOG_COL_DEFAULT;
                    if (fmt[i] == 0) break;
                }
                ss << fmt[i];
            }
            const size_t n = vsnprintf(entry.msg.data(), entry.msg.size(), ss.str().c_str(), args);
            if (n >= entry.msg.size()) {
                entry.msg.resize(n + 1);
                vsnprintf(entry.msg.data(), entry.msg.size(), ss.str().c_str(), args_copy);
            }
#endif
            va_end(args_copy);
        }

        entry.is_end    = false;
        entry.level     = level;
        entry.prefix    = prefix;
        entry.timestamp = 0;
        if (timestamps) {
            entry.timestamp = t_us() - t_start;
        }

        tail = (tail + 1) % queue.size();
        cv_new.notify_one();
    }

    void resume() {
        std::lock_guard<std::mutex> lock(mtx);

        if (running) {
            return;
        }

        running = true;

        thrd = std::thread([this]() {
            while (true) {
                std::unique_lock<std::mutex> lock(mtx);
                cv_new.wait(lock, [this]() { return !is_empty(); });

                bool stop = false;

                if (is_full() && queue.size() < max_capacity) {
                    stop = expand_and_flush(lock);
                } else {
                    auto & cur = queue[head];

                    lock.unlock(); // drop the lock while printing
                    stop = print_entry(cur);
                    lock.lock();

                    head = (head + 1) % queue.size();
                    cv_full.notify_one();
                }

                if (stop) {
                    break;
                }
            }
        });
    }

    void pause() {
        {
            std::lock_guard<std::mutex> lock(mtx);

            if (!running) {
                return;
            }

            running = false;

            // push an entry to signal the worker thread to stop
            auto & entry = queue[tail];
            entry.is_end = true;
            tail = (tail + 1) % queue.size();

            // wakeup everyone
            cv_new.notify_one();
            cv_full.notify_all();
        }

        thrd.join();
    }

    void set_file(const char * path) {
        pause();

        if (file) {
            fclose(file);
        }

        if (path) {
            file = fopen(path, "w");
        } else {
            file = nullptr;
        }

        resume();
    }

    void set_colors(bool colors) {
        pause();

        if (colors) {
            g_col[COMMON_LOG_COL_DEFAULT] = LOG_COL_DEFAULT;
            g_col[COMMON_LOG_COL_BOLD]    = LOG_COL_BOLD;
            g_col[COMMON_LOG_COL_RED]     = LOG_COL_RED;
            g_col[COMMON_LOG_COL_GREEN]   = LOG_COL_GREEN;
            g_col[COMMON_LOG_COL_YELLOW]  = LOG_COL_YELLOW;
            g_col[COMMON_LOG_COL_BLUE]    = LOG_COL_BLUE;
            g_col[COMMON_LOG_COL_MAGENTA] = LOG_COL_MAGENTA;
            g_col[COMMON_LOG_COL_CYAN]    = LOG_COL_CYAN;
            g_col[COMMON_LOG_COL_WHITE]   = LOG_COL_WHITE;
        } else {
            for (size_t i = 0; i < std::size(g_col); i++) {
                g_col[i] = "";
            }
        }

        resume();
    }

    void set_prefix(bool prefix) {
        std::lock_guard<std::mutex> lock(mtx);

        this->prefix = prefix;
    }

    void set_timestamps(bool timestamps) {
        std::lock_guard<std::mutex> lock(mtx);

        this->timestamps = timestamps;
    }
};

//
// public API
//

struct common_log * common_log_init() {
    return new common_log;
}

struct common_log * common_log_main() {
    // We intentionally leak (i.e. do not delete) the logger singleton because
    // common_log destructor called at DLL teardown phase will cause hanging on Windows.
    // OS will release resources anyway so it should not be a significant issue,
    // though this design may cause logs to be lost if not flushed before the program exits.
    // Refer to https://github.com/ggml-org/llama.cpp/issues/22142 for details.
    static struct common_log * log;
    static std::once_flag    init_flag;
    std::call_once(init_flag, [&]() {
        log = new common_log;
        // Set default to auto-detect colors
        log->set_colors(tty_can_use_colors());
    });

    return log;
}

void common_log_pause(struct common_log * log) {
    log->pause();
}

void common_log_resume(struct common_log * log) {
    log->resume();
}

void common_log_free(struct common_log * log) {
    delete log;
}

void common_log_add(struct common_log * log, enum ggml_log_level level, const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log->add(level, fmt, args);
    va_end(args);
}

void common_log_set_file(struct common_log * log, const char * file) {
    log->set_file(file);
}

void common_log_set_colors(struct common_log * log, log_colors colors) {
    if (colors == LOG_COLORS_AUTO) {
        log->set_colors(tty_can_use_colors());
        return;
    }

    if (colors == LOG_COLORS_DISABLED) {
        log->set_colors(false);
        return;
    }

    GGML_ASSERT(colors == LOG_COLORS_ENABLED);
    log->set_colors(true);
}

void common_log_set_prefix(struct common_log * log, bool prefix) {
    log->set_prefix(prefix);
}

void common_log_set_timestamps(struct common_log * log, bool timestamps) {
    log->set_timestamps(timestamps);
}

void common_log_flush(struct common_log * log) {
    log->pause();
    log->resume();
}

static int common_get_verbosity(enum ggml_log_level level) {
    switch (level) {
        case GGML_LOG_LEVEL_DEBUG: return LOG_LEVEL_DEBUG;
        case GGML_LOG_LEVEL_INFO:  return LOG_LEVEL_TRACE;
        case GGML_LOG_LEVEL_WARN:  return LOG_LEVEL_WARN;
        case GGML_LOG_LEVEL_ERROR: return LOG_LEVEL_ERROR;
        case GGML_LOG_LEVEL_CONT:  return LOG_LEVEL_TRACE;
        case GGML_LOG_LEVEL_NONE:
        default:
            return LOG_LEVEL_OUTPUT;
    }
}

void common_log_default_callback(enum ggml_log_level level, const char * text, void * /*user_data*/) {
    auto verbosity = common_get_verbosity(level);
    if (verbosity <= common_log_verbosity_thold) {
        common_log_add(common_log_main(), level, "%s", text);
    }
}
