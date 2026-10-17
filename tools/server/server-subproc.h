#pragma once

#include "subproc.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <mutex>
#include <queue>
#include <vector>

// simple implementation of a pipe
// used for streaming data between threads
template<typename T>
struct server_pipe {
    std::mutex mutex;
    std::condition_variable cv;
    std::queue<T> queue;
    std::atomic<bool> writer_closed{false};
    std::atomic<bool> reader_closed{false};

    // 0 = unbounded (default)
    // > 0, write() drops the oldest item once the queue is full
    size_t max_size = 0;

    void close_write() {
        writer_closed.store(true, std::memory_order_relaxed);
        cv.notify_all();
    }

    void close_read() {
        reader_closed.store(true, std::memory_order_relaxed);
        cv.notify_all();
    }

    // close_on_stop = true: should_stop means the reader is gone for good, so the writer is told the pipe is broken.
    // close_on_stop = false: should_stop is a per-read deadline and further reads still come, so the pipe stays usable.
    bool read(T & output, const std::function<bool()> & should_stop, bool close_on_stop = true) {
        std::unique_lock<std::mutex> lk(mutex);
        constexpr auto poll_interval = std::chrono::milliseconds(500);
        while (true) {
            if (!queue.empty()) {
                output = std::move(queue.front());
                queue.pop();
                return true;
            }
            if (writer_closed.load()) {
                return false; // clean EOF
            }
            if (should_stop && should_stop()) { // a null should_stop means "never stop"
                if (close_on_stop) {
                    close_read(); // signal broken pipe to writer
                }
                return false; // cancelled / deadline reached
            }
            cv.wait_for(lk, poll_interval);
        }
    }

    bool write(T && data) {
        std::lock_guard<std::mutex> lk(mutex);
        if (reader_closed.load()) {
            return false; // broken pipe
        }
        if (max_size > 0) {
            while (queue.size() >= max_size) {
                queue.pop(); // drop oldest to stay bounded
            }
        }
        queue.push(std::move(data));
        cv.notify_one();
        return true;
    }
};

// a child server writes its state commands to stdout and its logs to stderr
enum server_subproc_stream {
    SERVER_SUBPROC_STDOUT,
    SERVER_SUBPROC_STDERR,
    SERVER_SUBPROC_STREAMS,
};

// gives the current stdout to the caller as a stream of its own, and sends everything else written to stdout to stderr
FILE * server_reserve_stdout();

// wrapper around common_subproc to manage a child server process
// mainly used by router mode
struct server_subproc {
    common_subproc sproc;
    std::atomic<bool> stopped{false}; // set by the monitor once the process exited and was reaped

    bool is_alive() { return sproc.alive(); }
    void terminate() { sproc.terminate(); }
    int  join() { return sproc.join(); }

    // true if both output pipes of the child are available (call after create())
    bool has_output();

    // true once both output pipes are closed
    bool output_closed() const;

    // non-blocking read from one output pipe
    // returns the number of bytes read, 0 when nothing is available, -1 when the pipe is closed or broken
    int read_output(server_subproc_stream stream, char * buf, size_t len);

    // wait until one of a set of children has output, wake() is called, or a timeout passes
    struct waiter {
        waiter();
        ~waiter();

        // thread-safe; on Windows this is a no-op, wait() returns within 50 ms anyway
        void wake();

        // timeout_ms < 0 waits until data or wake(); ready[i] is set for each child with data on an open pipe (or a broken pipe)
        void wait(const std::vector<server_subproc *> & procs, std::vector<bool> & ready, int64_t timeout_ms);

    private:
#ifndef _WIN32
        intptr_t wake_fd[2] = { -1, -1 }; // POSIX self-pipe
#endif
    };

private:
    intptr_t out_handles[SERVER_SUBPROC_STREAMS] = { -1, -1 }; // fd on POSIX, HANDLE on Windows; taken lazily from sproc
    bool     out_closed [SERVER_SUBPROC_STREAMS] = { false, false };
};
