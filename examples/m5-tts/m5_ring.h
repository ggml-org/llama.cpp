// Shared-memory transport between the engine and its stage subprocesses.
//
// The stages used to exchange one request and one reply per call over a pair of
// pipes: a 4-byte header plus 2 KB of floats in each direction, twice per call.
// The depth stage makes sixteen such calls per audio frame, so the engine paid
// a couple of thousand context switches per utterance for the privilege. The
// block below replaces both pipes with one anonymous mapping inherited across
// exec: the parent writes the request, bumps a sequence counter and waits; the
// child bumps the other counter once the reply is in place. The exchange itself
// needs no syscall.
//
// Layout: a fixed header followed by `capacity` request rows and one reply row,
// each `width` floats. A request with zero rows means "clear state" and is
// acknowledged without data, which is how a stage resets between segments.
#pragma once

#include <atomic>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <linux/futex.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

namespace m5 {
namespace ring {

struct Header {
    std::atomic<uint32_t> seq_in;   // producer side: last submitted request
    std::atomic<uint32_t> seq_out;  // consumer side: last published reply
    std::atomic<int32_t> rows;      // rows in the current request
    std::atomic<int32_t> quit;      // set by the parent during shutdown
    int32_t capacity;
    int32_t width;
};

inline size_t bytes(int capacity, int width) {
    return sizeof(Header) + ((size_t) capacity + 1) * (size_t) width * sizeof(float);
}

inline float * request_data(Header * header) {
    return reinterpret_cast<float *>(header + 1);
}

inline float * reply_data(Header * header) {
    return request_data(header) + (size_t) header->capacity * (size_t) header->width;
}

inline double now_ms() {
    struct timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec * 1e3 + (double) ts.tv_nsec / 1e6;
}

// A short spin covers the case where the other side is already inside the
// exchange; after that the waiter has to sleep, because a spinning peer steals
// cycles from the very thread that owes it an answer. On this SoC there are two
// big cores and the codec thread wants one of them, so a pure spin loop made
// the ring slower than the pipe it replaced.
constexpr unsigned kSpinIterations = 64;

inline void cpu_relax() {
#if defined(__aarch64__)
    asm volatile("yield" ::: "memory");
#else
    __builtin_ia32_pause();
#endif
}

inline int futex_wait(std::atomic<uint32_t> * word, uint32_t expected, int timeout_ms) {
    struct timespec ts{};
    ts.tv_sec = timeout_ms / 1000;
    ts.tv_nsec = (long) (timeout_ms % 1000) * 1000000L;
    return (int) syscall(SYS_futex, reinterpret_cast<uint32_t *>(word), FUTEX_WAIT,
                         expected, &ts, nullptr, 0);
}

// Sleeps until `done()` holds. The futex is armed against the value just
// observed, so a peer that publishes in between simply fails the wait with
// EAGAIN and the loop re-checks instead of blocking forever.
template <typename Pred, typename Stop>
inline bool wait_for(std::atomic<uint32_t> * word, Pred done, int timeout_ms, Stop stop) {
    for (unsigned i = 0; i < kSpinIterations; i++) {
        if (done()) return true;
        cpu_relax();
    }
    const double deadline = now_ms() + timeout_ms;
    for (;;) {
        if (done()) return true;
        if (stop()) return false;
        const uint32_t current = word->load(std::memory_order_acquire);
        if (done()) return true;
        const double left = deadline - now_ms();
        if (left <= 0) return false;
        // 20 ms slices keep the stop flag responsive without waking needlessly.
        futex_wait(word, current, left > 20 ? 20 : (int) left + 1);
    }
}

inline void futex_wake(std::atomic<uint32_t> * word) {
    syscall(SYS_futex, reinterpret_cast<uint32_t *>(word), FUTEX_WAKE, INT_MAX, nullptr, nullptr, 0);
}

// Parent side. Owns the mapping and the fd that is handed to the child.
class Client {
public:
    Client() = default;
    ~Client() { close(); }
    Client(const Client &) = delete;
    Client & operator=(const Client &) = delete;

    // Creates the block and returns the fd to pass as --ring-fd, or -1.
    int create(int capacity, int width) {
        capacity_ = capacity;
        width_ = width;
        bytes_ = bytes(capacity, width);
#ifdef SYS_memfd_create
        fd_ = (int) syscall(SYS_memfd_create, "m5-stage-ring", 0);
#endif
        if (fd_ < 0) return -1;
        if (ftruncate(fd_, (off_t) bytes_) != 0) { ::close(fd_); fd_ = -1; return -1; }
        void * p = mmap(nullptr, bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (p == MAP_FAILED) { ::close(fd_); fd_ = -1; return -1; }
        header_ = static_cast<Header *>(p);
        memset(header_, 0, sizeof(Header));
        header_->capacity = capacity;
        header_->width = width;
        return fd_;
    }

    float * request() const { return request_data(header_); }
    float * reply() const { return reply_data(header_); }

    // Waits for the reply matching `seq`; false if the child died or timed out.
    bool wait(uint32_t seq, int pid, int timeout_ms = 5000) {
        const double deadline = now_ms() + timeout_ms;
        auto done = [&] { return header_->seq_out.load(std::memory_order_acquire) >= seq; };
        auto stop = [&] {
            if (header_->quit.load(std::memory_order_relaxed)) return true;
            if (pid > 0 && waitpid(pid, nullptr, WNOHANG) == pid) return true;
            return now_ms() > deadline;
        };
        return wait_for(&header_->seq_out, done, timeout_ms, stop);
    }

    bool reset(int pid) {
        if (!header_) return false;
        header_->rows.store(0, std::memory_order_relaxed);
        const uint32_t seq = header_->seq_in.load(std::memory_order_relaxed) + 1;
        header_->seq_in.store(seq, std::memory_order_release);
        futex_wake(&header_->seq_in);
        return wait(seq, pid);
    }

    // Submits `rows` rows and copies the reply into `out`.
    bool call(const float * data, int rows, float * out, int pid) {
        if (!header_ || rows < 0 || rows > capacity_) return false;
        memcpy(request(), data, sizeof(float) * (size_t) rows * (size_t) width_);
        header_->rows.store(rows, std::memory_order_relaxed);
        const uint32_t seq = header_->seq_in.load(std::memory_order_relaxed) + 1;
        header_->seq_in.store(seq, std::memory_order_release);
        futex_wake(&header_->seq_in);
        if (!wait(seq, pid)) return false;
        memcpy(out, reply(), sizeof(float) * (size_t) width_);
        return true;
    }

    void close() {
        if (header_) {
            header_->quit.store(1, std::memory_order_relaxed);
            // Bump the sequence so a waiting child re-checks the quit flag.
            header_->seq_in.store(header_->seq_in.load(std::memory_order_relaxed) + 1,
                                  std::memory_order_release);
            futex_wake(&header_->seq_in);
            munmap(header_, bytes_);
            header_ = nullptr;
        }
        if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
    }

private:
    Header * header_ = nullptr;
    int fd_ = -1;
    int capacity_ = 0;
    int width_ = 0;
    size_t bytes_ = 0;
};

// Child side: attaches to an inherited fd and answers requests.
class Server {
public:
    bool attach(int fd) {
        if (fd < 0) return false;
        struct stat st {};
        if (fstat(fd, &st) != 0 || st.st_size < (off_t) sizeof(Header)) return false;
        void * p = mmap(nullptr, (size_t) st.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (p == MAP_FAILED) return false;
        header_ = static_cast<Header *>(p);
        if (header_->capacity <= 0 || header_->width <= 0) return false;
        bytes_ = (size_t) st.st_size;
        return true;
    }

    int width() const { return header_ ? header_->width : 0; }
    int capacity() const { return header_ ? header_->capacity : 0; }
    float * request() const { return request_data(header_); }
    float * reply() const { return reply_data(header_); }

    // Blocks until the next request. `rows` is 0 for a reset, negative on quit.
    int next(int & rows) {
        if (!header_) return -1;
        auto done = [&] { return header_->seq_in.load(std::memory_order_acquire) != seen_; };
        auto stop = [&] { return header_->quit.load(std::memory_order_relaxed) != 0; };
        if (!wait_for(&header_->seq_in, done, 1 << 30, stop)) return -1;
        // Shutdown also bumps the sequence, so this check has to come after the
        // wait: otherwise the parent's farewell looks like a request.
        if (header_->quit.load(std::memory_order_relaxed)) return -1;
        seen_ = header_->seq_in.load(std::memory_order_acquire);
        rows = header_->rows.load(std::memory_order_relaxed);
        return 0;
    }

    void publish() {
        if (!header_) return;
        header_->seq_out.store(seen_, std::memory_order_release);
        futex_wake(&header_->seq_out);
    }

    void detach() {
        if (header_) { munmap(header_, bytes_); header_ = nullptr; }
    }

private:
    Header * header_ = nullptr;
    size_t bytes_ = 0;
    uint32_t seen_ = 0;
};

}  // namespace ring
}  // namespace m5
