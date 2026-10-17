#include "server-subproc.h"

#include "log.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <thread>

#ifdef _WIN32
// windows.h defines min and max as macros, which breaks std::min and std::max
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#else
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <unistd.h>
#endif

//
// server_subproc
//

FILE * server_reserve_stdout() {
    fflush(stdout);
    // the reserved stream is not inherited, grandchildren get stdout and stderr of their own
#ifdef _WIN32
    int fd = _dup(_fileno(stdout));
    GGML_ASSERT(fd >= 0);
    SetHandleInformation((HANDLE) _get_osfhandle(fd), HANDLE_FLAG_INHERIT, 0);
    _dup2(_fileno(stderr), _fileno(stdout));
    SetStdHandle(STD_OUTPUT_HANDLE, GetStdHandle(STD_ERROR_HANDLE));
    FILE * f = _fdopen(fd, "w");
#else
    int fd = fcntl(fileno(stdout), F_DUPFD_CLOEXEC, 0);
    GGML_ASSERT(fd >= 0);
    dup2(fileno(stderr), fileno(stdout));
    FILE * f = fdopen(fd, "w");
#endif
    GGML_ASSERT(f);
    return f;
}

bool server_subproc::has_output() {
    FILE * files[SERVER_SUBPROC_STREAMS] = { sproc.stdout_file(), sproc.stderr_file() };
    for (int i = 0; i < SERVER_SUBPROC_STREAMS; i++) {
        if (out_handles[i] >= 0) {
            continue;
        }
        if (!files[i]) {
            return false;
        }
#ifdef _WIN32
        HANDLE h = (HANDLE) _get_osfhandle(_fileno(files[i]));
        if (h == INVALID_HANDLE_VALUE) {
            return false;
        }
        out_handles[i] = (intptr_t) h;
#else
        int fd = fileno(files[i]);
        if (fd < 0) {
            return false;
        }
        fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, 0) | O_NONBLOCK);
        out_handles[i] = fd;
#endif
    }
    return true;
}

bool server_subproc::output_closed() const {
    return out_closed[SERVER_SUBPROC_STDOUT] && out_closed[SERVER_SUBPROC_STDERR];
}

int server_subproc::read_output(server_subproc_stream stream, char * buf, size_t len) {
    if (out_closed[stream]) {
        return -1;
    }
    if (!has_output()) {
        out_closed[stream] = true;
        return -1;
    }
#ifdef _WIN32
    HANDLE h     = (HANDLE) out_handles[stream];
    DWORD  avail = 0;
    if (!PeekNamedPipe(h, NULL, 0, NULL, &avail, NULL)) {
        out_closed[stream] = true; // pipe broken, child gone
        return -1;
    }
    if (avail == 0) {
        return 0;
    }
    DWORD to_read = avail < (DWORD) len ? avail : (DWORD) len;
    DWORD got     = 0;
    if (!ReadFile(h, buf, to_read, &got, NULL) || got == 0) {
        out_closed[stream] = true;
        return -1;
    }
    return (int) got;
#else
    while (true) {
        ssize_t r = read((int) out_handles[stream], buf, len);
        if (r > 0) {
            return (int) r;
        }
        if (r < 0 && errno == EINTR) {
            continue;
        }
        if (r < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            return 0;
        }
        out_closed[stream] = true; // EOF or error
        return -1;
    }
#endif
}

server_subproc::waiter::waiter() {
#ifndef _WIN32
    int fds[2];
    GGML_ASSERT(pipe(fds) == 0);
    for (int fd : fds) {
        fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, 0) | O_NONBLOCK);
    }
    wake_fd[0] = fds[0];
    wake_fd[1] = fds[1];
#endif
}

server_subproc::waiter::~waiter() {
#ifndef _WIN32
    close((int) wake_fd[0]);
    close((int) wake_fd[1]);
#endif
}

void server_subproc::waiter::wake() {
#ifndef _WIN32
    char c = 1;
    (void) !write((int) wake_fd[1], &c, 1);
#endif
}

void server_subproc::waiter::wait(const std::vector<server_subproc *> & procs, std::vector<bool> & ready, int64_t timeout_ms) {
    ready.assign(procs.size(), false);
#ifdef _WIN32
    // no waitable wait exists for anonymous pipes, so poll them in 50 ms steps
    bool any = false;
    for (size_t i = 0; i < procs.size(); i++) {
        server_subproc * p = procs[i];
        ready[i] = !p->has_output();
        for (int s = 0; s < SERVER_SUBPROC_STREAMS && !ready[i]; s++) {
            DWORD avail = 0;
            if (!p->out_closed[s] && (!PeekNamedPipe((HANDLE) p->out_handles[s], NULL, 0, NULL, &avail, NULL) || avail > 0)) {
                ready[i] = true; // data or broken pipe, read_output() tells which
            }
        }
        any = any || ready[i];
    }
    if (!any) {
        int64_t step = timeout_ms < 0 ? 50 : std::min<int64_t>(timeout_ms, 50);
        std::this_thread::sleep_for(std::chrono::milliseconds(step));
    }
#else
    std::vector<pollfd> pfds;
    pfds.reserve(procs.size() * SERVER_SUBPROC_STREAMS + 1);
    pfds.push_back({ (int) wake_fd[0], POLLIN, 0 });
    for (auto * p : procs) {
        const bool open = p->has_output();
        for (int s = 0; s < SERVER_SUBPROC_STREAMS; s++) {
            pfds.push_back({ open && !p->out_closed[s] ? (int) p->out_handles[s] : -1, POLLIN, 0 }); // poll() skips negative fds
        }
    }
    int timeout = timeout_ms < 0 ? -1 : (int) std::min<int64_t>(timeout_ms, std::numeric_limits<int>::max());
    int r = poll(pfds.data(), pfds.size(), timeout);
    if (r < 0 && errno != EINTR) {
        LOG_ERR("%s: poll() failed: %s\n", __func__, strerror(errno));
    }
    if (pfds[0].revents) {
        char buf[64];
        while (read((int) wake_fd[0], buf, sizeof(buf)) > 0) {}
    }
    for (size_t i = 0; i < procs.size(); i++) {
        ready[i] = !procs[i]->has_output();
        for (int s = 0; s < SERVER_SUBPROC_STREAMS; s++) {
            ready[i] = ready[i] || pfds[1 + i * SERVER_SUBPROC_STREAMS + s].revents != 0;
        }
    }
#endif
}
