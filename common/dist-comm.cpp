#include "dist-comm.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cctype>
#include <cstdlib>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <cerrno>
#endif


class dist_comm_socket_context final : public dist_comm_context {
public:
    static std::unique_ptr<dist_comm_socket_context> create(const dist_comm_config & config, std::string & err);

    ~dist_comm_socket_context() override;

    bool exchange_ring_f32(int32_t src_rank, const float * send_data, size_t n,
                           int32_t & recv_src_rank, float * recv_data, std::string & err) override;
    bool broadcast_i32(int32_t & value, int root, std::string & err) override;
    bool broadcast_vector_i32(std::vector<int32_t> & values, int root, std::string & err) override;
    bool broadcast_vector_u8(std::vector<uint8_t> & values, int root, std::string & err) override;

    dist_comm_stats get_stats() const override;

private:
    struct impl;
    explicit dist_comm_socket_context(std::unique_ptr<impl> impl_);

    std::unique_ptr<impl> impl_;
};

namespace {

constexpr uint32_t DIST_COMM_MAGIC = 0x4443434c; // DCCL
constexpr uint32_t OP_ALLREDUCE_F32 = 1;
constexpr uint32_t OP_BCAST_I32     = 2;
constexpr uint32_t OP_BCAST_VEC_I32 = 3;
constexpr uint32_t OP_BCAST_VEC_U8  = 4;
constexpr int SOCKET_BUFSIZE = 50 * 1024 * 1024;

#ifdef _WIN32
using socket_t = SOCKET;
using addrlen_t = int;
constexpr socket_t DIST_CCL_INVALID_SOCKET = INVALID_SOCKET;
#else
using socket_t = int;
using addrlen_t = socklen_t;
constexpr socket_t DIST_CCL_INVALID_SOCKET = -1;
#endif

struct reg_msg {
    uint32_t magic;
    int32_t  rank;
    int32_t  listen_port;
};

struct member_wire {
    int32_t port;
    char    host[64];
};

struct op_hdr {
    uint32_t op;
    uint32_t count;
    int32_t  src_rank;
    int32_t  reserved;
};

struct peer_info {
    std::string host;
    int         port = 0;
};

bool ensure_socket_runtime(std::string & err) {
#ifdef _WIN32
    static std::once_flag wsa_once;
    static bool wsa_ok = false;
    static std::string wsa_err;

    std::call_once(wsa_once, [&]() {
        WSADATA wsa_data{};
        const int rc = WSAStartup(MAKEWORD(2, 2), &wsa_data);
        if (rc != 0) {
            wsa_err = "WSAStartup() failed: code " + std::to_string(rc);
            return;
        }
        wsa_ok = true;
    });

    if (!wsa_ok) {
        err = wsa_err;
        return false;
    }
#else
    (void) err;
#endif
    return true;
}

std::string socket_error_string() {
#ifdef _WIN32
    return "WSA error code " + std::to_string(WSAGetLastError());
#else
    return std::string(strerror(errno));
#endif
}

std::string addrinfo_error_string(int rc) {
#ifdef _WIN32
    const char * msg = gai_strerrorA(rc);
#else
    const char * msg = gai_strerror(rc);
#endif
    if (msg != nullptr) {
        return msg;
    }
    return "code " + std::to_string(rc);
}

bool send_all(socket_t fd, const void * data, size_t size) {
    const char * p = static_cast<const char *>(data);
    size_t sent = 0;
    while (sent < size) {
        const size_t remaining = size - sent;
        const int chunk = static_cast<int>(std::min(remaining, static_cast<size_t>(std::numeric_limits<int>::max())));
        const int n = send(fd, p + sent, chunk, 0);
#ifdef _WIN32
        if (n == SOCKET_ERROR || n <= 0) {
#else
        if (n <= 0) {
#endif
            return false;
        }
        sent += static_cast<size_t>(n);
    }
    return true;
}

bool recv_all(socket_t fd, void * data, size_t size) {
    char * p = static_cast<char *>(data);
    size_t recvd = 0;
    while (recvd < size) {
        const size_t remaining = size - recvd;
        const int chunk = static_cast<int>(std::min(remaining, static_cast<size_t>(std::numeric_limits<int>::max())));
        const int n = recv(fd, p + recvd, chunk, 0);
#ifdef _WIN32
        if (n == 0) {
            WSASetLastError(0);
            return false;
        }
        if (n == SOCKET_ERROR) {
#else
        if (n == 0) {
            errno = 0;
            return false;
        }
        if (n < 0) {
#endif
            return false;
        }
        recvd += static_cast<size_t>(n);
    }
    return true;
}

void close_fd(socket_t & fd) {
    if (fd != DIST_CCL_INVALID_SOCKET) {
#ifdef _WIN32
        closesocket(fd);
#else
        close(fd);
#endif
        fd = DIST_CCL_INVALID_SOCKET;
    }
}

bool set_no_delay(socket_t fd, std::string & err) {
#ifdef _WIN32
    const char one = 1;
    if (setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)) != 0) {
#else
    const int one = 1;
    if (setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)) != 0) {
#endif
        err = "setsockopt(TCP_NODELAY) failed: " + socket_error_string();
        return false;
    }

    return true;
}

bool set_buf_size(socket_t fd) {
#ifdef _WIN32
    const char * optval = reinterpret_cast<const char *>(&SOCKET_BUFSIZE);
    const int ret1 = setsockopt(fd, SOL_SOCKET, SO_SNDBUF, optval, sizeof(SOCKET_BUFSIZE));
    const int ret2 = setsockopt(fd, SOL_SOCKET, SO_RCVBUF, optval, sizeof(SOCKET_BUFSIZE));
#else
    const int ret1 = setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &SOCKET_BUFSIZE, sizeof(SOCKET_BUFSIZE));
    const int ret2 = setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &SOCKET_BUFSIZE, sizeof(SOCKET_BUFSIZE));
#endif
    return ret1 == 0 && ret2 == 0;
}

bool set_reuse_addr(socket_t fd, std::string & err) {
#ifdef _WIN32
    const char one = 1;
    if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) != 0) {
#else
    const int one = 1;
    if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) != 0) {
#endif
        err = "setsockopt(SO_REUSEADDR) failed: " + socket_error_string();
        return false;
    }

    return true;
}

bool set_socket_opt_best_effort(socket_t fd, int level, int optname, const void * optval, addrlen_t optlen) {
#ifdef _WIN32
    return setsockopt(fd, level, optname, static_cast<const char *>(optval), static_cast<int>(optlen)) == 0;
#else
    return setsockopt(fd, level, optname, optval, optlen) == 0;
#endif
}

void tune_socket_common(socket_t fd) {
    const int one = 1;
    (void) set_buf_size(fd);
    (void) set_socket_opt_best_effort(fd, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof(one));
}

void tune_stream_socket_performance(socket_t fd) {
    tune_socket_common(fd);
    const int one = 1;

#ifdef __linux__
#ifdef SO_BUSY_POLL
    const int busy_poll_us = 50;
    (void) set_socket_opt_best_effort(fd, SOL_SOCKET, SO_BUSY_POLL, &busy_poll_us, sizeof(busy_poll_us));
#endif

#ifdef TCP_NOTSENT_LOWAT
    const int notsent_lowat = 256 * 1024;
    (void) set_socket_opt_best_effort(fd, IPPROTO_TCP, TCP_NOTSENT_LOWAT, &notsent_lowat, sizeof(notsent_lowat));
#endif

#ifdef SO_ZEROCOPY
    (void) set_socket_opt_best_effort(fd, SOL_SOCKET, SO_ZEROCOPY, &one, sizeof(one));
#endif
#endif
}

socket_t create_listen_socket(int port, std::string & err) {
    socket_t fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd == DIST_CCL_INVALID_SOCKET) {
        err = "socket() failed: " + socket_error_string();
        return DIST_CCL_INVALID_SOCKET;
    }

    if (!set_reuse_addr(fd, err)) {
        close_fd(fd);
        return DIST_CCL_INVALID_SOCKET;
    }

    tune_socket_common(fd);

    sockaddr_in addr {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    addr.sin_addr.s_addr = htonl(INADDR_ANY);

    if (bind(fd, reinterpret_cast<const sockaddr *>(&addr), sizeof(addr)) != 0) {
        err = "bind() failed on port " + std::to_string(port) + ": " + socket_error_string();
        close_fd(fd);
        return DIST_CCL_INVALID_SOCKET;
    }

    if (listen(fd, SOMAXCONN) != 0) {
        err = "listen() failed on port " + std::to_string(port) + ": " + socket_error_string();
        close_fd(fd);
        return DIST_CCL_INVALID_SOCKET;
    }

    return fd;
}

socket_t accept_socket(socket_t listen_fd, std::string & peer_host, std::string & err) {
    sockaddr_in addr {};
    addrlen_t addr_len = sizeof(addr);
    socket_t fd = accept(listen_fd, reinterpret_cast<sockaddr *>(&addr), &addr_len);
    if (fd == DIST_CCL_INVALID_SOCKET) {
        err = "accept() failed: " + socket_error_string();
        return DIST_CCL_INVALID_SOCKET;
    }

    if (!set_no_delay(fd, err)) {
        close_fd(fd);
        return DIST_CCL_INVALID_SOCKET;
    }

    tune_stream_socket_performance(fd);

    char host_buf[INET_ADDRSTRLEN] = {0};
    if (inet_ntop(AF_INET, &addr.sin_addr, host_buf, sizeof(host_buf)) == nullptr) {
        err = "inet_ntop() failed";
        close_fd(fd);
        return DIST_CCL_INVALID_SOCKET;
    }

    peer_host = host_buf;
    return fd;
}

socket_t connect_with_retry(const std::string & host, int port, std::string & err, int max_attempts = 6000, int delay_ms = 50) {
    struct addrinfo hints {};
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    struct addrinfo * result = nullptr;
    const std::string port_str = std::to_string(port);
    const int gai_rc = getaddrinfo(host.c_str(), port_str.c_str(), &hints, &result);
    if (gai_rc != 0) {
        err = "getaddrinfo() failed for " + host + ":" + port_str + ": " + addrinfo_error_string(gai_rc);
        return DIST_CCL_INVALID_SOCKET;
    }

    socket_t fd = DIST_CCL_INVALID_SOCKET;
    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        for (addrinfo * rp = result; rp != nullptr; rp = rp->ai_next) {
            fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
            if (fd == DIST_CCL_INVALID_SOCKET) {
                continue;
            }

            if (connect(fd, rp->ai_addr, static_cast<addrlen_t>(rp->ai_addrlen)) == 0) {
                if (!set_no_delay(fd, err)) {
                    close_fd(fd);
                    continue;
                }
                tune_stream_socket_performance(fd);
                freeaddrinfo(result);
                return fd;
            }

            close_fd(fd);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    }

    freeaddrinfo(result);
    err = "connect() failed to " + host + ":" + std::to_string(port) +
            " after " + std::to_string(max_attempts) + " attempts";
    return DIST_CCL_INVALID_SOCKET;
}

} // namespace

struct dist_comm_socket_context::impl {
    dist_comm_config cfg;
    socket_t listen_fd = DIST_CCL_INVALID_SOCKET;
    socket_t ring_in_fd = DIST_CCL_INVALID_SOCKET;
    socket_t ring_out_fd = DIST_CCL_INVALID_SOCKET;
    std::vector<peer_info> members;
    dist_comm_stats stats;

    ~impl() {
        close_fd(ring_in_fd);
        close_fd(ring_out_fd);
        close_fd(listen_fd);
    }

    bool write_msg(socket_t fd, const void * data, size_t size, std::string & err) {
        if (!send_all(fd, data, size)) {
            err = "send() failed: " + socket_error_string();
            return false;
        }
        stats.bytes_sent += size;
        return true;
    }

    bool read_msg(socket_t fd, void * data, size_t size, std::string & err) {
        if (!recv_all(fd, data, size)) {
            err = socket_error_string();
            if (err == "WSA error code 0" || err == "Success") {
                err = "peer closed connection";
            } else {
                err = "recv() failed: " + err;
            }
            return false;
        }
        stats.bytes_recv += size;
        return true;
    }

    bool exchange_ring(uint32_t op, int32_t src_rank, const void * send_data_ptr, size_t send_size,
            int32_t & recv_src_rank, void * recv_data_ptr, size_t recv_size, std::string & err) {
        op_hdr send_hdr {
            op,
            static_cast<uint32_t>(send_size),
            src_rank,
            0,
        };
        op_hdr recv_hdr {};

        auto send_payload = [&]() -> bool {
            if (!write_msg(ring_out_fd, &send_hdr, sizeof(send_hdr), err)) {
                return false;
            }
            if (send_size > 0 && !write_msg(ring_out_fd, send_data_ptr, send_size, err)) {
                return false;
            }
            return true;
        };

        auto recv_payload = [&]() -> bool {
            if (!read_msg(ring_in_fd, &recv_hdr, sizeof(recv_hdr), err)) {
                return false;
            }
            if (recv_hdr.op != op) {
                err = "unexpected op code on ring";
                return false;
            }
            if (recv_hdr.count != recv_size) {
                err = "unexpected payload size on ring";
                return false;
            }
            if (recv_size > 0 && !read_msg(ring_in_fd, recv_data_ptr, recv_size, err)) {
                return false;
            }
            recv_src_rank = recv_hdr.src_rank;
            return true;
        };

        // Rank parity avoids blocking send/send for large payloads.
        if ((cfg.rank & 1) == 0) {
            return send_payload() && recv_payload();
        }
        return recv_payload() && send_payload();
    }

    bool bootstrap_master(std::string & err) {
        members.assign(cfg.world_size, {});
        members[0] = {cfg.master_host, cfg.listen_port};

        std::vector<socket_t> reg_fds;
        reg_fds.reserve(cfg.world_size - 1);

        while ((int) reg_fds.size() < cfg.world_size - 1) {
            std::string peer_host;
            socket_t fd = accept_socket(listen_fd, peer_host, err);
            if (fd == DIST_CCL_INVALID_SOCKET) {
                return false;
            }

            reg_msg msg {};
            if (!read_msg(fd, &msg, sizeof(msg), err)) {
                close_fd(fd);
                return false;
            }
            if (msg.magic != DIST_COMM_MAGIC) {
                err = "invalid registration magic";
                close_fd(fd);
                return false;
            }
            if (msg.rank <= 0 || msg.rank >= cfg.world_size) {
                err = "invalid rank in registration";
                close_fd(fd);
                return false;
            }
            members[msg.rank] = {peer_host, msg.listen_port};
            reg_fds.push_back(fd);
        }

        std::vector<member_wire> payload(cfg.world_size);
        for (int i = 0; i < cfg.world_size; ++i) {
            payload[i].port = members[i].port;
            std::memset(payload[i].host, 0, sizeof(payload[i].host));
            std::snprintf(payload[i].host, sizeof(payload[i].host), "%s", members[i].host.c_str());
        }

        for (socket_t fd : reg_fds) {
            if (!write_msg(fd, payload.data(), payload.size() * sizeof(member_wire), err)) {
                close_fd(fd);
                return false;
            }
            close_fd(fd);
        }

        return true;
    }

    bool bootstrap_worker(std::string & err) {
        socket_t fd = connect_with_retry(cfg.master_host, cfg.master_port, err);
        if (fd == DIST_CCL_INVALID_SOCKET) {
            return false;
        }

        reg_msg msg {
            DIST_COMM_MAGIC,
            cfg.rank,
            cfg.listen_port,
        };
        if (!write_msg(fd, &msg, sizeof(msg), err)) {
            close_fd(fd);
            return false;
        }

        std::vector<member_wire> payload(cfg.world_size);
        if (!read_msg(fd, payload.data(), payload.size() * sizeof(member_wire), err)) {
            close_fd(fd);
            return false;
        }
        close_fd(fd);

        members.assign(cfg.world_size, {});
        for (int i = 0; i < cfg.world_size; ++i) {
            members[i].host = payload[i].host;
            members[i].port = payload[i].port;
        }

        return true;
    }

    bool establish_ring(std::string & err) {
        if (cfg.world_size == 1) {
            return true;
        }

        const int next = (cfg.rank + 1) % cfg.world_size;
        const int prev = (cfg.rank + cfg.world_size - 1) % cfg.world_size;

        ring_out_fd = connect_with_retry(members[next].host, members[next].port, err);
        if (ring_out_fd == DIST_CCL_INVALID_SOCKET) {
            return false;
        }

        std::string peer_host;
        ring_in_fd = accept_socket(listen_fd, peer_host, err);
        if (ring_in_fd == DIST_CCL_INVALID_SOCKET) {
            return false;
        }

        int32_t out_rank = cfg.rank;
        if (!write_msg(ring_out_fd, &out_rank, sizeof(out_rank), err)) {
            return false;
        }

        int32_t in_rank = -1;
        if (!read_msg(ring_in_fd, &in_rank, sizeof(in_rank), err)) {
            return false;
        }
        if (in_rank != prev) {
            err = "ring peer rank mismatch";
            return false;
        }

        close_fd(listen_fd);
        return true;
    }
};

dist_comm_socket_context::dist_comm_socket_context(std::unique_ptr<impl> impl_) : impl_(std::move(impl_)) {}

std::unique_ptr<dist_comm_socket_context> dist_comm_socket_context::create(const dist_comm_config & config, std::string & err) {
    if (!ensure_socket_runtime(err)) {
        return nullptr;
    }

    if (config.world_size <= 0) {
        err = "world_size must be > 0";
        return nullptr;
    }
    if (config.rank < 0 || config.rank >= config.world_size) {
        err = "rank must be in [0, world_size)";
        return nullptr;
    }
    if (config.listen_port <= 0 || config.master_port <= 0) {
        err = "listen_port and master_port must be > 0";
        return nullptr;
    }
    if (config.master_host.empty()) {
        err = "master_host is empty";
        return nullptr;
    }

    auto p = std::unique_ptr<impl>(new impl());
    p->cfg = config;
    p->listen_fd = create_listen_socket(config.listen_port, err);
    if (p->listen_fd == DIST_CCL_INVALID_SOCKET) {
        return nullptr;
    }

    if (config.rank == 0) {
        if (!p->bootstrap_master(err)) {
            return nullptr;
        }
    } else {
        if (!p->bootstrap_worker(err)) {
            return nullptr;
        }
    }

    if (!p->establish_ring(err)) {
        return nullptr;
    }

    return std::unique_ptr<dist_comm_socket_context>(new dist_comm_socket_context(std::move(p)));
}

dist_comm_socket_context::~dist_comm_socket_context() = default;

bool dist_comm_socket_context::exchange_ring_f32(int32_t src_rank, const float * send_data, size_t n,
        int32_t & recv_src_rank, float * recv_data, std::string & err) {
    if (impl_->cfg.world_size == 1 || n == 0) {
        recv_src_rank = src_rank;
        if (n > 0 && send_data != recv_data) {
            std::copy(send_data, send_data + n, recv_data);
        }
        return true;
    }

    if (send_data == nullptr || recv_data == nullptr) {
        err = "exchange_ring_f32: null payload";
        return false;
    }

    return impl_->exchange_ring(
        OP_ALLREDUCE_F32,
        src_rank,
        send_data,
        n * sizeof(float),
        recv_src_rank,
        recv_data,
        n * sizeof(float),
        err);
}

bool dist_comm_socket_context::broadcast_i32(int32_t & value, int root, std::string & err) {
    if (impl_->cfg.world_size == 1) {
        return true;
    }
    if (root < 0 || root >= impl_->cfg.world_size) {
        err = "invalid root rank";
        return false;
    }

    op_hdr hdr {
        OP_BCAST_I32,
        static_cast<uint32_t>(sizeof(int32_t)),
        root,
        0,
    };

    op_hdr recv_hdr {};
    int32_t tmp = value;

    if (impl_->cfg.rank == root) {
        if (!impl_->write_msg(impl_->ring_out_fd, &hdr, sizeof(hdr), err)) {
            return false;
        }
        if (!impl_->write_msg(impl_->ring_out_fd, &tmp, sizeof(tmp), err)) {
            return false;
        }
        if (!impl_->read_msg(impl_->ring_in_fd, &recv_hdr, sizeof(recv_hdr), err)) {
            return false;
        }
        if (recv_hdr.op != OP_BCAST_I32 || recv_hdr.count != sizeof(int32_t) || recv_hdr.src_rank != root) {
            err = "invalid broadcast header";
            return false;
        }
        if (!impl_->read_msg(impl_->ring_in_fd, &tmp, sizeof(tmp), err)) {
            return false;
        }
    } else {
        if (!impl_->read_msg(impl_->ring_in_fd, &recv_hdr, sizeof(recv_hdr), err)) {
            return false;
        }
        if (recv_hdr.op != OP_BCAST_I32 || recv_hdr.count != sizeof(int32_t)) {
            err = "invalid broadcast header";
            return false;
        }
        if (!impl_->read_msg(impl_->ring_in_fd, &tmp, sizeof(tmp), err)) {
            return false;
        }
        if (!impl_->write_msg(impl_->ring_out_fd, &recv_hdr, sizeof(recv_hdr), err)) {
            return false;
        }
        if (!impl_->write_msg(impl_->ring_out_fd, &tmp, sizeof(tmp), err)) {
            return false;
        }
    }

    value = tmp;
    impl_->stats.broadcast_calls++;
    return true;
}

bool dist_comm_socket_context::broadcast_vector_i32(std::vector<int32_t> & values, int root, std::string & err) {
    if (impl_->cfg.world_size == 1) {
        return true;
    }
    if (root < 0 || root >= impl_->cfg.world_size) {
        err = "invalid root rank";
        return false;
    }

    op_hdr hdr {
        OP_BCAST_VEC_I32,
        static_cast<uint32_t>(values.size() * sizeof(int32_t)),
        root,
        0,
    };
    op_hdr recv_hdr {};

    if (impl_->cfg.rank == root) {
        if (!impl_->write_msg(impl_->ring_out_fd, &hdr, sizeof(hdr), err)) {
            return false;
        }
        if (hdr.count > 0 && !impl_->write_msg(impl_->ring_out_fd, values.data(), hdr.count, err)) {
            return false;
        }
        if (!impl_->read_msg(impl_->ring_in_fd, &recv_hdr, sizeof(recv_hdr), err)) {
            return false;
        }
        if (recv_hdr.op != OP_BCAST_VEC_I32 || recv_hdr.src_rank != root || recv_hdr.count != hdr.count) {
            err = "invalid vector broadcast header";
            return false;
        }
        if (recv_hdr.count > 0) {
            if (recv_hdr.count % sizeof(int32_t) != 0) {
                err = "invalid vector broadcast payload size";
                return false;
            }
            std::vector<int32_t> sink(recv_hdr.count / sizeof(int32_t));
            if (!impl_->read_msg(impl_->ring_in_fd, sink.data(), recv_hdr.count, err)) {
                return false;
            }
        }
    } else {
        if (!impl_->read_msg(impl_->ring_in_fd, &recv_hdr, sizeof(recv_hdr), err)) {
            return false;
        }
        if (recv_hdr.op != OP_BCAST_VEC_I32) {
            err = "invalid vector broadcast header";
            return false;
        }

        if (recv_hdr.count % sizeof(int32_t) != 0) {
            err = "invalid vector broadcast payload size";
            return false;
        }

        values.resize(recv_hdr.count / sizeof(int32_t));
        if (recv_hdr.count > 0 && !impl_->read_msg(impl_->ring_in_fd, values.data(), recv_hdr.count, err)) {
            return false;
        }

        if (!impl_->write_msg(impl_->ring_out_fd, &recv_hdr, sizeof(recv_hdr), err)) {
            return false;
        }
        if (recv_hdr.count > 0 && !impl_->write_msg(impl_->ring_out_fd, values.data(), recv_hdr.count, err)) {
            return false;
        }
    }

    impl_->stats.broadcast_calls++;
    return true;
}

bool dist_comm_socket_context::broadcast_vector_u8(std::vector<uint8_t> & values, int root, std::string & err) {
    if (impl_->cfg.world_size == 1) {
        return true;
    }
    if (root < 0 || root >= impl_->cfg.world_size) {
        err = "invalid root rank";
        return false;
    }

    op_hdr hdr {
        OP_BCAST_VEC_U8,
        static_cast<uint32_t>(values.size()),
        root,
        0,
    };
    op_hdr recv_hdr {};

    if (impl_->cfg.rank == root) {
        if (!impl_->write_msg(impl_->ring_out_fd, &hdr, sizeof(hdr), err)) {
            return false;
        }
        if (hdr.count > 0 && !impl_->write_msg(impl_->ring_out_fd, values.data(), hdr.count, err)) {
            return false;
        }
        if (!impl_->read_msg(impl_->ring_in_fd, &recv_hdr, sizeof(recv_hdr), err)) {
            return false;
        }
        if (recv_hdr.op != OP_BCAST_VEC_U8 || recv_hdr.src_rank != root || recv_hdr.count != hdr.count) {
            err = "invalid byte broadcast header";
            return false;
        }
        if (recv_hdr.count > 0) {
            std::vector<uint8_t> sink(recv_hdr.count);
            if (!impl_->read_msg(impl_->ring_in_fd, sink.data(), recv_hdr.count, err)) {
                return false;
            }
        }
    } else {
        if (!impl_->read_msg(impl_->ring_in_fd, &recv_hdr, sizeof(recv_hdr), err)) {
            return false;
        }
        if (recv_hdr.op != OP_BCAST_VEC_U8) {
            err = "invalid byte broadcast header";
            return false;
        }

        values.resize(recv_hdr.count);
        if (recv_hdr.count > 0 && !impl_->read_msg(impl_->ring_in_fd, values.data(), recv_hdr.count, err)) {
            return false;
        }

        if (!impl_->write_msg(impl_->ring_out_fd, &recv_hdr, sizeof(recv_hdr), err)) {
            return false;
        }
        if (recv_hdr.count > 0 && !impl_->write_msg(impl_->ring_out_fd, values.data(), recv_hdr.count, err)) {
            return false;
        }
    }

    impl_->stats.broadcast_calls++;
    return true;
}

dist_comm_stats dist_comm_socket_context::get_stats() const {
    return impl_->stats;
}

std::unique_ptr<dist_comm_context> dist_comm_create(const dist_comm_config & config, std::string & err) {
    return std::unique_ptr<dist_comm_context>(dist_comm_socket_context::create(config, err).release());
}

namespace {

bool env_true(const char * key, bool def = false) {
    const char * v = std::getenv(key);
    if (!v) {
        return def;
    }
    std::string s(v);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return s == "1" || s == "true" || s == "yes" || s == "on";
}

bool parse_int_env(const char * key, int & out) {
    const char * v = std::getenv(key);
    if (!v || v[0] == '\0') {
        return false;
    }
    try {
        out = std::stoi(v);
    } catch (...) {
        return false;
    }
    return true;
}

bool parse_host_port(const std::string & input, std::string & host, int & port) {
    const size_t pos = input.rfind(':');
    if (pos == std::string::npos || pos == 0 || pos + 1 >= input.size()) {
        return false;
    }
    host = input.substr(0, pos);
    try {
        port = std::stoi(input.substr(pos + 1));
    } catch (...) {
        return false;
    }
    return port > 0;
}

bool derive_default_listen_port(int master_port, int world_rank, int & listen_port, std::string & err) {
    const int64_t candidate = (int64_t) master_port + world_rank;
    if (candidate <= 0 || candidate > 65535) {
        err = "dist TP listen port must be set explicitly when master_port + rank is outside [1, 65535]";
        return false;
    }
    listen_port = (int) candidate;
    return true;
}

bool parse_split_mode_env(const char * key, llama_split_mode & out, std::string & err) {
    const char * v = std::getenv(key);
    if (!v || v[0] == '\0') {
        return false;
    }

    std::string s(v);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char) std::tolower(c); });

    if (s == "none") {
        out = LLAMA_SPLIT_MODE_NONE;
        return true;
    }
    if (s == "layer") {
        out = LLAMA_SPLIT_MODE_LAYER;
        return true;
    }
    if (s == "row") {
        out = LLAMA_SPLIT_MODE_ROW;
        return true;
    }
    if (s == "tensor") {
        out = LLAMA_SPLIT_MODE_TENSOR;
        return true;
    }

    err = std::string(key) + " must be one of: none, layer, row, tensor";
    return false;
}

bool parse_tensor_split_env(const char * key, float * out, size_t out_len, std::string & err) {
    const char * v = std::getenv(key);
    if (!v || v[0] == '\0') {
        return false;
    }

    std::fill(out, out + out_len, 0.0f);

    std::string s(v);
    size_t start = 0;
    size_t idx = 0;

    while (start <= s.size()) {
        const size_t end = s.find(',', start);
        const std::string tok = s.substr(start, end == std::string::npos ? std::string::npos : end - start);

        if (idx >= out_len) {
            err = std::string(key) + " has too many entries (max " + std::to_string(out_len) + ")";
            return false;
        }

        if (!tok.empty()) {
            try {
                const float val = std::stof(tok);
                if (!std::isfinite(val) || val < 0.0f) {
                    err = std::string(key) + " entries must be finite and >= 0";
                    return false;
                }
                out[idx] = val;
            } catch (...) {
                err = std::string(key) + " contains an invalid float: '" + tok + "'";
                return false;
            }
        }

        idx++;
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }

    return true;
}

struct dist_tp_eval_state {
    dist_comm_context * comm = nullptr;
    int world_rank = 0;
    int world_size = 1;
    bool allow_attn_output_collectives = true;
};

struct dist_tp_decode_sync_state {
    dist_comm_context * comm = nullptr;
    int world_rank = 0;
    int world_size = 1;
    bool seen_sync_batch = false;
    bool transport_ended = false;
    std::string last_transport_error;
    // Rank0/server slot ids are not guaranteed to fit worker n_seq_max; map them to local ids.
    std::unordered_map<llama_seq_id, llama_seq_id> seq_id_remap;
    // Local ids reassigned to a new remote id must be cleared from KV before decode.
    std::vector<llama_seq_id> force_seq_reset;
    // Tracks the latest decoded position per local sequence to detect rewinds cheaply.
    std::unordered_map<llama_seq_id, llama_pos> seq_last_pos;

    std::vector<uint8_t> wire;
    std::vector<llama_token> tokens;
    std::vector<float> embd;
    std::vector<llama_pos> pos;
    std::vector<int32_t> n_seq_id;
    std::vector<llama_seq_id> seq_id_data;
    std::vector<llama_seq_id *> seq_id_ptrs;
    std::vector<int8_t> logits;
};

dist_tp_decode_sync_state * g_worker_decode_sync_state = nullptr;
struct common_dist_tp_runtime * g_worker_runtime = nullptr;

enum : uint32_t {
    DIST_TP_BATCH_HAS_TOKEN    = 1u << 0,
    DIST_TP_BATCH_HAS_EMBD     = 1u << 1,
    DIST_TP_BATCH_HAS_POS      = 1u << 2,
    DIST_TP_BATCH_HAS_N_SEQ_ID = 1u << 3,
    DIST_TP_BATCH_HAS_SEQ_ID   = 1u << 4,
    DIST_TP_BATCH_HAS_LOGITS   = 1u << 5,
};

struct dist_tp_batch_wire_header {
    uint32_t version;
    uint32_t flags;
    int32_t n_tokens;
    int32_t n_embd;
    int32_t seq_id_count;
};

template<typename T>
void wire_append(std::vector<uint8_t> & wire, const T * data, size_t count) {
    if (count == 0) {
        return;
    }
    const size_t offset = wire.size();
    wire.resize(offset + sizeof(T) * count);
    std::memcpy(wire.data() + offset, data, sizeof(T) * count);
}

template<typename T>
bool wire_read(const std::vector<uint8_t> & wire, size_t & offset, T * data, size_t count) {
    const size_t bytes = sizeof(T) * count;
    if (offset + bytes > wire.size()) {
        return false;
    }
    if (bytes > 0) {
        std::memcpy(data, wire.data() + offset, bytes);
    }
    offset += bytes;
    return true;
}

bool pack_decode_batch(llama_context * ctx, const llama_batch & batch, std::vector<uint8_t> & wire, std::string & err) {
    wire.clear();
    if (batch.n_tokens <= 0) {
        err = "empty decode batch";
        return false;
    }

    dist_tp_batch_wire_header hdr {};
    hdr.version = 1;
    hdr.n_tokens = batch.n_tokens;
    hdr.n_embd = 0;
    hdr.seq_id_count = 0;

    if (batch.token != nullptr) {
        hdr.flags |= DIST_TP_BATCH_HAS_TOKEN;
    }
    if (batch.embd != nullptr) {
        hdr.flags |= DIST_TP_BATCH_HAS_EMBD;
        hdr.n_embd = llama_model_n_embd_inp(llama_get_model(ctx));
        if (hdr.n_embd <= 0) {
            err = "invalid embedding width for decode batch";
            return false;
        }
    }
    if (batch.pos != nullptr) {
        hdr.flags |= DIST_TP_BATCH_HAS_POS;
    }
    if (batch.n_seq_id != nullptr) {
        hdr.flags |= DIST_TP_BATCH_HAS_N_SEQ_ID;
        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            hdr.seq_id_count += batch.n_seq_id[i];
        }
    }
    if (batch.seq_id != nullptr && batch.n_seq_id != nullptr) {
        hdr.flags |= DIST_TP_BATCH_HAS_SEQ_ID;
    }
    if (batch.logits != nullptr) {
        hdr.flags |= DIST_TP_BATCH_HAS_LOGITS;
    }

    if ((hdr.flags & (DIST_TP_BATCH_HAS_TOKEN | DIST_TP_BATCH_HAS_EMBD)) == 0) {
        err = "decode batch must contain token or embd";
        return false;
    }

    wire_append(wire, &hdr, 1);
    if (batch.token != nullptr) {
        wire_append(wire, batch.token, (size_t) batch.n_tokens);
    }
    if (batch.embd != nullptr) {
        wire_append(wire, batch.embd, (size_t) batch.n_tokens * (size_t) hdr.n_embd);
    }
    if (batch.pos != nullptr) {
        wire_append(wire, batch.pos, (size_t) batch.n_tokens);
    }
    if (batch.n_seq_id != nullptr) {
        wire_append(wire, batch.n_seq_id, (size_t) batch.n_tokens);
    }
    if ((hdr.flags & DIST_TP_BATCH_HAS_SEQ_ID) != 0) {
        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            if (batch.n_seq_id[i] > 0) {
                wire_append(wire, batch.seq_id[i], (size_t) batch.n_seq_id[i]);
            }
        }
    }
    if (batch.logits != nullptr) {
        wire_append(wire, batch.logits, (size_t) batch.n_tokens);
    }

    return true;
}

bool unpack_decode_batch(const std::vector<uint8_t> & wire, llama_context * ctx, dist_tp_decode_sync_state * st, llama_batch * batch, std::string & err) {
    size_t offset = 0;
    dist_tp_batch_wire_header hdr {};
    if (!wire_read(wire, offset, &hdr, 1)) {
        err = "failed to read batch header";
        return false;
    }
    if (hdr.version != 1 || hdr.n_tokens <= 0) {
        err = "invalid batch header";
        return false;
    }

    st->tokens.clear();
    st->embd.clear();
    st->pos.clear();
    st->n_seq_id.clear();
    st->seq_id_data.clear();
    st->seq_id_ptrs.clear();
    st->logits.clear();

    if ((hdr.flags & DIST_TP_BATCH_HAS_TOKEN) != 0) {
        st->tokens.resize((size_t) hdr.n_tokens);
        if (!wire_read(wire, offset, st->tokens.data(), st->tokens.size())) {
            err = "failed to read token payload";
            return false;
        }
    }
    if ((hdr.flags & DIST_TP_BATCH_HAS_EMBD) != 0) {
        st->embd.resize((size_t) hdr.n_tokens * (size_t) hdr.n_embd);
        if (!wire_read(wire, offset, st->embd.data(), st->embd.size())) {
            err = "failed to read embd payload";
            return false;
        }
    }
    if ((hdr.flags & DIST_TP_BATCH_HAS_POS) != 0) {
        st->pos.resize((size_t) hdr.n_tokens);
        if (!wire_read(wire, offset, st->pos.data(), st->pos.size())) {
            err = "failed to read pos payload";
            return false;
        }
    }
    if ((hdr.flags & DIST_TP_BATCH_HAS_N_SEQ_ID) != 0) {
        st->n_seq_id.resize((size_t) hdr.n_tokens);
        if (!wire_read(wire, offset, st->n_seq_id.data(), st->n_seq_id.size())) {
            err = "failed to read n_seq_id payload";
            return false;
        }
    }
    if ((hdr.flags & DIST_TP_BATCH_HAS_SEQ_ID) != 0) {
        st->seq_id_data.resize((size_t) hdr.seq_id_count);
        st->seq_id_ptrs.resize((size_t) hdr.n_tokens);
        if (!wire_read(wire, offset, st->seq_id_data.data(), st->seq_id_data.size())) {
            err = "failed to read seq_id payload";
            return false;
        }

        if (st->world_rank != 0 && ctx != nullptr) {
            const uint32_t n_seq_max = std::max<uint32_t>(1, llama_n_seq_max(ctx));

            if (n_seq_max == 1) {
                // Fast path for the common worker setup: only local seq 0 is valid.
                llama_seq_id batch_remote_sid = -1;
                for (const llama_seq_id sid : st->seq_id_data) {
                    if (batch_remote_sid < 0) {
                        batch_remote_sid = sid;
                    } else if (sid != batch_remote_sid) {
                        err = "worker seq_id remap exhausted: unique seq ids exceed llama_n_seq_max";
                        return false;
                    }
                }

                auto it_prev = st->seq_id_remap.find(batch_remote_sid);
                if (batch_remote_sid >= 0 && (it_prev == st->seq_id_remap.end() || it_prev->second != 0)) {
                    st->seq_id_remap.clear();
                    st->seq_id_remap.emplace(batch_remote_sid, 0);
                    st->force_seq_reset.push_back(0);
                }

                for (llama_seq_id & sid : st->seq_id_data) {
                    sid = 0;
                }
            } else {

            std::unordered_set<llama_seq_id> batch_remote_ids(st->seq_id_data.begin(), st->seq_id_data.end());
            if (batch_remote_ids.size() > n_seq_max) {
                err = "worker seq_id remap exhausted: unique seq ids exceed llama_n_seq_max";
                return false;
            }

            std::vector<llama_seq_id> local_owner(n_seq_max, (llama_seq_id) -1);
            for (const auto & kv : st->seq_id_remap) {
                const llama_seq_id local_sid = kv.second;
                if (local_sid >= 0 && (uint32_t) local_sid < n_seq_max) {
                    local_owner[(uint32_t) local_sid] = kv.first;
                }
            }

            for (llama_seq_id & sid : st->seq_id_data) {
                const llama_seq_id remote_sid = sid;
                auto it = st->seq_id_remap.find(remote_sid);
                if (it == st->seq_id_remap.end()) {
                    llama_seq_id chosen_local = -1;

                    for (uint32_t local_sid = 0; local_sid < n_seq_max; ++local_sid) {
                        if (local_owner[local_sid] == (llama_seq_id) -1) {
                            chosen_local = (llama_seq_id) local_sid;
                            break;
                        }
                    }

                    if (chosen_local < 0) {
                        // Reuse a local id whose owner is not used by this batch.
                        for (uint32_t local_sid = 0; local_sid < n_seq_max; ++local_sid) {
                            const llama_seq_id owner = local_owner[local_sid];
                            if (batch_remote_ids.find(owner) == batch_remote_ids.end()) {
                                chosen_local = (llama_seq_id) local_sid;
                                st->seq_id_remap.erase(owner);
                                local_owner[local_sid] = (llama_seq_id) -1;
                                break;
                            }
                        }
                    }

                    if (chosen_local < 0) {
                        err = "worker seq_id remap exhausted: unique seq ids exceed llama_n_seq_max";
                        return false;
                    }

                    it = st->seq_id_remap.emplace(remote_sid, chosen_local).first;
                    local_owner[(uint32_t) chosen_local] = remote_sid;
                    // Ensure stale KV for this local id is dropped before using it for a new remote id.
                    st->force_seq_reset.push_back(chosen_local);
                }
                sid = it->second;
            }
            }
        }

        size_t seq_offset = 0;
        for (int32_t i = 0; i < hdr.n_tokens; ++i) {
            const int32_t n_ids = st->n_seq_id.empty() ? 0 : st->n_seq_id[i];
            st->seq_id_ptrs[i] = n_ids > 0 ? st->seq_id_data.data() + seq_offset : nullptr;
            seq_offset += (size_t) std::max<int32_t>(0, n_ids);
        }
    }
    if ((hdr.flags & DIST_TP_BATCH_HAS_LOGITS) != 0) {
        st->logits.resize((size_t) hdr.n_tokens);
        if (!wire_read(wire, offset, st->logits.data(), st->logits.size())) {
            err = "failed to read logits payload";
            return false;
        }
    }
    if (offset != wire.size()) {
        err = "unexpected trailing batch bytes";
        return false;
    }

    batch->n_tokens = hdr.n_tokens;
    batch->token    = st->tokens.empty() ? nullptr : st->tokens.data();
    batch->embd     = st->embd.empty() ? nullptr : st->embd.data();
    batch->pos      = st->pos.empty() ? nullptr : st->pos.data();
    batch->n_seq_id = st->n_seq_id.empty() ? nullptr : st->n_seq_id.data();
    batch->seq_id   = st->seq_id_ptrs.empty() ? nullptr : st->seq_id_ptrs.data();
    batch->logits   = st->logits.empty() ? nullptr : st->logits.data();
    return true;
}

void worker_reset_sequences_for_batch_if_needed(llama_context * ctx, dist_tp_decode_sync_state * st) {
    if (ctx == nullptr || st == nullptr || st->world_rank == 0 || st->pos.empty()) {
        return;
    }

    llama_memory_t mem = llama_get_memory(ctx);
    if (mem == nullptr) {
        return;
    }

    // First clear local ids that were reassigned by remap in this batch.
    for (llama_seq_id sid : st->force_seq_reset) {
        (void) llama_memory_seq_rm(mem, sid, -1, -1);
        st->seq_last_pos.erase(sid);
    }
    st->force_seq_reset.clear();

    std::unordered_map<llama_seq_id, llama_pos> seq_min_pos;
    std::unordered_map<llama_seq_id, llama_pos> seq_max_pos;

    if (!st->n_seq_id.empty() && !st->seq_id_ptrs.empty()) {
        for (size_t i = 0; i < st->pos.size(); ++i) {
            const int32_t n_ids = i < st->n_seq_id.size() ? st->n_seq_id[i] : 0;
            const llama_seq_id * ids = i < st->seq_id_ptrs.size() ? st->seq_id_ptrs[i] : nullptr;
            if (n_ids <= 0 || ids == nullptr) {
                continue;
            }
            for (int32_t j = 0; j < n_ids; ++j) {
                const llama_seq_id sid = ids[j];
                const llama_pos pos = st->pos[i];
                auto it = seq_min_pos.find(sid);
                if (it == seq_min_pos.end() || pos < it->second) {
                    seq_min_pos[sid] = pos;
                }
                auto it_max = seq_max_pos.find(sid);
                if (it_max == seq_max_pos.end() || pos > it_max->second) {
                    seq_max_pos[sid] = pos;
                }
            }
        }
    } else {
        // If sequence ids are omitted, the batch implicitly targets sequence 0.
        const llama_pos min_pos = *std::min_element(st->pos.begin(), st->pos.end());
        const llama_pos max_pos = *std::max_element(st->pos.begin(), st->pos.end());
        seq_min_pos[0] = min_pos;
        seq_max_pos[0] = max_pos;
    }

    for (const auto & [sid, min_pos] : seq_min_pos) {
        const auto it_last = st->seq_last_pos.find(sid);
        // Reset only when a sequence position truly rewinds relative to prior batch.
        if (it_last != st->seq_last_pos.end() && it_last->second > min_pos) {
            (void) llama_memory_seq_rm(mem, sid, -1, -1);
            st->seq_last_pos.erase(sid);
        }

        const auto it_max = seq_max_pos.find(sid);
        if (it_max != seq_max_pos.end()) {
            st->seq_last_pos[sid] = it_max->second;
        }
    }
}

bool starts_with(std::string_view value, std::string_view prefix) {
    return value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix;
}

const std::vector<std::string> & dist_tp_collective_prefix_overrides() {
    static const std::vector<std::string> prefixes = []() {
        std::vector<std::string> out;
        const char * raw = std::getenv("LLAMA_DIST_TP_COLLECT_PREFIXES");
        if (!raw || raw[0] == '\0') {
            return out;
        }

        std::string s(raw);
        size_t start = 0;
        while (start <= s.size()) {
            const size_t end = s.find(',', start);
            std::string tok = s.substr(start, end == std::string::npos ? std::string::npos : end - start);

            const size_t first = tok.find_first_not_of(" \t\r\n");
            if (first != std::string::npos) {
                const size_t last = tok.find_last_not_of(" \t\r\n");
                out.emplace_back(tok.substr(first, last - first + 1));
            }

            if (end == std::string::npos) {
                break;
            }
            start = end + 1;
        }

        return out;
    }();
    return prefixes;
}

bool is_transport_end_error(std::string_view err) {
    return err.find("dist-peer closed connection") != std::string_view::npos ||
           err.find("peer closed connection") != std::string_view::npos ||
           err.find("Timer expired") != std::string_view::npos ||
           err.find("Operation timed out") != std::string_view::npos ||
           err.find("Connection reset") != std::string_view::npos ||
           err.find("Broken pipe") != std::string_view::npos ||
           err.find("Connection timed out") != std::string_view::npos ||
           err.find("Connection refused") != std::string_view::npos ||
           err.find("WSA error code 10053") != std::string_view::npos || // software caused connection abort
           err.find("WSA error code 10054") != std::string_view::npos || // connection reset by peer
           err.find("WSA error code 10058") != std::string_view::npos || // socket shutdown
           err.find("WSA error code 10060") != std::string_view::npos || // timed out
           err.find("WSA error code 10061") != std::string_view::npos;   // connection refused
}

bool mark_worker_transport_end_if_needed(int rank, std::string_view err) {
    if (rank == 0 || g_worker_decode_sync_state == nullptr) {
        return false;
    }

    if (!g_worker_decode_sync_state->seen_sync_batch || !is_transport_end_error(err)) {
        return false;
    }

    // Mark graceful session end so worker loop can reconnect instead of exiting fatally.
    g_worker_decode_sync_state->transport_ended = true;
    g_worker_decode_sync_state->last_transport_error = std::string(err);
    return true;
}

bool dist_tp_collect_attn_output_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        // Some architectures (e.g. mixed linear/full attention variants) may expose
        // attn_output-* as an internal intermediate. Keep enabled by default and
        // allow targeted A/B testing via env override.
        enabled = env_true("LLAMA_DIST_TP_COLLECT_ATTN_OUTPUT", true) ? 1 : 0;
    }
    return enabled == 1;
}

bool dist_tp_collect_attn_output_enabled(const dist_tp_eval_state * st) {
    if (st != nullptr && !st->allow_attn_output_collectives) {
        return false;
    }
    return dist_tp_collect_attn_output_enabled();
}

bool is_dist_tp_collective_node(const ggml_tensor * t, const dist_tp_eval_state * st = nullptr, bool allow_reshaped_attn_output = false) {
    if (t == nullptr || t->name[0] == '\0') {
        return false;
    }

    const std::string_view name(t->name);
    if (name.find(" (reshaped)") != std::string_view::npos) {
        // Meta-boundary allreduce can surface reshaped attention outputs.
        // Keep legacy exclusion for other reshaped intermediates.
        if (!(allow_reshaped_attn_output && starts_with(name, "attn_output-"))) {
            return false;
        }
    }

    // Debug/triage knob: allow forcing an explicit prefix whitelist.
    // Example: LLAMA_DIST_TP_COLLECT_PREFIXES=linear_attn_out-,ffn_down-
    const auto & override_prefixes = dist_tp_collective_prefix_overrides();
    if (!override_prefixes.empty()) {
        for (const std::string & prefix : override_prefixes) {
            if (starts_with(name, prefix)) {
                return true;
            }
        }
        return false;
    }

    return (dist_tp_collect_attn_output_enabled(st) && starts_with(name, "attn_output-")) ||
           starts_with(name, "linear_attn_out-")      ||
           starts_with(name, "ffn_down-")             ||
           starts_with(name, "ffn_out-")              ||
           starts_with(name, "ffn_moe_out-");
}

bool dist_tp_meta_allreduce_tensors(void * comm_ctx, const float * send_data, size_t n,
        int32_t src_rank, float * recv_data, int32_t * recv_src_rank) {
    auto * st = static_cast<dist_tp_eval_state *>(comm_ctx);
    if (!st || st->comm == nullptr) {
        return true;
    }

    if (n == 0) {
        if (recv_src_rank != nullptr) {
            *recv_src_rank = src_rank;
        }
        return true;
    }

    if (send_data == nullptr || recv_data == nullptr || recv_src_rank == nullptr) {
        return false;
    }

    std::string err;
    if (!st->comm->exchange_ring_f32(src_rank, send_data, n, *recv_src_rank, recv_data, err)) {
        if (mark_worker_transport_end_if_needed(st->world_rank, err)) {
            return false;
        }
        std::fprintf(stderr, "error: meta exchange failed on rank %d: %s\n", st->world_rank, err.c_str());
        return false;
    }

    return true;
}

bool dist_tp_decode_batch_callback(llama_context * ctx, llama_batch * batch, void * user_data) {
    auto * st = static_cast<dist_tp_decode_sync_state *>(user_data);
    if (!st || st->world_size <= 1 || st->comm == nullptr) {
        return true;
    }

    st->transport_ended = false;
    st->last_transport_error.clear();

    st->wire.clear();
    if (st->world_rank == 0) {
        if (batch == nullptr || batch->n_tokens <= 0) {
            std::fprintf(stderr, "error: rank 0 must provide a non-empty decode batch for distributed llama_decode\n");
            return false;
        }
        std::string err;
        if (!pack_decode_batch(ctx, *batch, st->wire, err)) {
            std::fprintf(stderr, "error: failed to pack decode batch on rank 0: %s\n", err.c_str());
            return false;
        }
    }

    std::string err;
    if (!st->comm->broadcast_vector_u8(st->wire, 0, err)) {
        if (st->world_rank != 0 && st->seen_sync_batch && is_transport_end_error(err)) {
            st->transport_ended = true;
            st->last_transport_error = err;
            return false;
        }
        std::fprintf(stderr, "error: decode batch broadcast failed on rank %d: %s\n", st->world_rank, err.c_str());
        return false;
    }

    st->seen_sync_batch = true;

    if (st->world_rank != 0) {
        if (st->wire.empty()) {
            std::fprintf(stderr, "error: received empty decode batch on rank %d\n", st->world_rank);
            return false;
        }
        if (!unpack_decode_batch(st->wire, ctx, st, batch, err)) {
            std::fprintf(stderr, "error: failed to unpack decode batch on rank %d: %s\n", st->world_rank, err.c_str());
            return false;
        }

        worker_reset_sequences_for_batch_if_needed(ctx, st);
    }

    return true;
}

} // namespace

struct common_dist_tp_runtime {
    common_dist_tp_env env;
    std::unique_ptr<dist_comm_context> comm;
    dist_tp_eval_state eval_state;
    dist_tp_decode_sync_state decode_sync_state;
};

static bool common_dist_tp_worker_reconnect(common_dist_tp_runtime * runtime, std::string & err) {
    if (runtime == nullptr) {
        err = "missing worker runtime";
        return false;
    }

    dist_comm_config comm_cfg;
    comm_cfg.rank        = runtime->env.world_rank;
    comm_cfg.world_size  = runtime->env.world_size;
    comm_cfg.master_host = runtime->env.master_host;
    comm_cfg.master_port = runtime->env.master_port;
    comm_cfg.listen_port = runtime->env.listen_port;

    auto comm = dist_comm_create(comm_cfg, err);
    if (!comm) {
        return false;
    }

    runtime->comm = std::move(comm);
    runtime->eval_state.comm = runtime->comm.get();
    runtime->decode_sync_state.comm = runtime->comm.get();
    runtime->decode_sync_state.seen_sync_batch = false;
    runtime->decode_sync_state.transport_ended = false;
    runtime->decode_sync_state.last_transport_error.clear();
    runtime->decode_sync_state.seq_id_remap.clear();
    runtime->decode_sync_state.force_seq_reset.clear();
    runtime->decode_sync_state.seq_last_pos.clear();

    g_worker_decode_sync_state = &runtime->decode_sync_state;
    return true;
}

bool common_dist_tp_env_load(common_dist_tp_env & out, std::string & err) {
    out = {};
    parse_int_env("LLAMA_DIST_TP_MASTER_RANK", out.master_rank);

    const bool has_world_size = parse_int_env("LLAMA_DIST_TP_WORLD_SIZE", out.world_size);
    if (has_world_size && out.world_size <= 0) {
        err = "LLAMA_DIST_TP_WORLD_SIZE must be a positive integer";
        return false;
    }

    // Distributed mode is inferred from world size.
    out.enabled = out.world_size >= 2;
    if (!out.enabled) {
        return true;
    }

    if (!has_world_size || out.world_size < 2) {
        err = "LLAMA_DIST_TP_WORLD_SIZE must be >= 2 when distributed TP is enabled";
        return false;
    }
    if (!parse_int_env("LLAMA_DIST_TP_WORLD_RANK", out.world_rank) || out.world_rank < 0 || out.world_rank >= out.world_size) {
        err = "LLAMA_DIST_TP_WORLD_RANK must be in [0, LLAMA_DIST_TP_WORLD_SIZE)";
        return false;
    }
    const char * master = std::getenv("LLAMA_DIST_TP_MASTER");
    if (!master || !parse_host_port(master, out.master_host, out.master_port)) {
        err = "LLAMA_DIST_TP_MASTER must be set as host:port";
        return false;
    }
    if (!parse_int_env("LLAMA_DIST_TP_LISTEN_PORT", out.listen_port)) {
        if (!derive_default_listen_port(out.master_port, out.world_rank, out.listen_port, err)) {
            return false;
        }
    }
    if (out.listen_port <= 0) {
        err = "LLAMA_DIST_TP_LISTEN_PORT must be a positive integer";
        return false;
    }
    if (out.master_rank != 0) {
        err = "LLAMA_DIST_TP_MASTER_RANK must currently be 0";
        return false;
    }

    return true;
}

bool common_dist_tp_resolve(const common_params & params, common_dist_tp_env & out, std::string & err) {
    out = {};
    const bool has_param_config =
        params.dist_tp_world_size != 1 ||
        params.dist_tp_rank != 0 ||
        params.dist_tp_master_rank != 0 ||
        params.dist_tp_master_host != "127.0.0.1" ||
        params.dist_tp_master_port > 0 ||
        params.dist_tp_listen_port > 0;

    common_dist_tp_env env_cfg;
    std::string env_err;
    const bool have_env_cfg = common_dist_tp_env_load(env_cfg, env_err);
    if (!have_env_cfg && !has_param_config) {
        err = env_err;
        return false;
    }

    out.enabled = has_param_config ? (params.dist_tp_world_size >= 2) : env_cfg.enabled;
    if (!out.enabled) {
        return true;
    }

    out.world_size = params.dist_tp_world_size;
    out.world_rank = params.dist_tp_rank;
    out.master_rank = params.dist_tp_master_rank;
    out.master_host = params.dist_tp_master_host;
    out.master_port = params.dist_tp_master_port;
    out.listen_port = params.dist_tp_listen_port;

    if (have_env_cfg && env_cfg.enabled) {
        out.enabled = true;
        if (!has_param_config || params.dist_tp_world_size == 1) out.world_size = env_cfg.world_size;
        if (!has_param_config || params.dist_tp_rank == 0) out.world_rank = env_cfg.world_rank;
        if (!has_param_config || params.dist_tp_master_rank == 0) out.master_rank = env_cfg.master_rank;
        if (!has_param_config || params.dist_tp_master_port <= 0) out.master_port = env_cfg.master_port;
        if (!has_param_config || params.dist_tp_listen_port <= 0) out.listen_port = env_cfg.listen_port;
        if (!has_param_config || params.dist_tp_master_host == "127.0.0.1") out.master_host = env_cfg.master_host;
    }

    if (out.master_rank != 0) {
        err = "only master_rank=0 is supported currently";
        return false;
    }
    if (out.world_size < 2) {
        err = "dist TP world size must be >= 2";
        return false;
    }
    if (out.world_rank < 0 || out.world_rank >= out.world_size) {
        err = "dist TP rank must be in [0, world_size)";
        return false;
    }
    if (out.master_port <= 0) {
        err = "dist TP master port must be > 0";
        return false;
    }
    if (out.listen_port <= 0 && !derive_default_listen_port(out.master_port, out.world_rank, out.listen_port, err)) {
        return false;
    }
    if (out.listen_port <= 0) {
        err = "dist TP listen port must be > 0";
        return false;
    }
    return true;
}

bool common_dist_tp_apply_env_overrides(common_params & params, const common_dist_tp_env & env, std::string & err) {
    if (!env.enabled) {
        return true;
    }

    auto export_env = [&](const char * key, const std::string & value) -> bool {
#ifdef _WIN32
        return _putenv_s(key, value.c_str()) == 0;
#else
        return setenv(key, value.c_str(), 1) == 0;
#endif
    };

    // Keep loader/runtime env in sync with CLI-resolved config.
    if (!export_env("LLAMA_DIST_TP_WORLD_SIZE", std::to_string(env.world_size)) ||
        !export_env("LLAMA_DIST_TP_WORLD_RANK", std::to_string(env.world_rank)) ||
        !export_env("LLAMA_DIST_TP_MASTER_RANK", std::to_string(env.master_rank)) ||
        !export_env("LLAMA_DIST_TP_MASTER", env.master_host + ":" + std::to_string(env.master_port)) ||
        !export_env("LLAMA_DIST_TP_LISTEN_PORT", std::to_string(env.listen_port))) {
        err = "failed to export distributed TP process environment";
        return false;
    }

    llama_split_mode split_mode_override = params.split_mode;
    if (!parse_split_mode_env("LLAMA_DIST_TP_MODEL_SPLIT_MODE", split_mode_override, err)) {
        if (!err.empty()) {
            return false;
        }
    } else {
        params.split_mode = split_mode_override;
    }

    if (!parse_tensor_split_env("LLAMA_DIST_TP_MODEL_TENSOR_SPLIT", params.tensor_split, sizeof(params.tensor_split)/sizeof(params.tensor_split[0]), err)) {
        if (!err.empty()) {
            return false;
        }
    }

    if (env_true("LLAMA_DIST_TP_FAST_STARTUP", true)) {
        params.fit_params = false;
        params.warmup = false;
    }

    const char * use_mmap_env = std::getenv("LLAMA_DIST_TP_USE_MMAP");
    if (use_mmap_env != nullptr) {
        params.use_mmap = env_true("LLAMA_DIST_TP_USE_MMAP", params.use_mmap);
    } else if (params.split_mode == LLAMA_SPLIT_MODE_TENSOR && env.world_size > 1) {
        params.use_mmap = false;
    }

    if (env_true("LLAMA_DIST_TP_VERBOSE", true)) {
        std::fprintf(stderr,
            "[dist-tp][rank=%d] split_mode=%d use_mmap=%s warmup=%s\n",
            env.world_rank, (int) params.split_mode, params.use_mmap ? "true" : "false", params.warmup ? "true" : "false");
    }

    return true;
}

common_dist_tp_runtime * common_dist_tp_runtime_create(common_params & params, const common_dist_tp_env & env, std::string & err) {
    (void) params;

    if (!env.enabled) {
        return nullptr;
    }

    // Cross-rank distributed TP communication is implemented here (dist_comm + callbacks).
    // This path is independent from ggml-backend-meta internals.
    dist_comm_config comm_cfg;
    comm_cfg.rank        = env.world_rank;
    comm_cfg.world_size  = env.world_size;
    comm_cfg.master_host = env.master_host;
    comm_cfg.master_port = env.master_port;
    comm_cfg.listen_port = env.listen_port;

    auto comm = dist_comm_create(comm_cfg, err);
    if (!comm) {
        return nullptr;
    }

    auto * runtime = new common_dist_tp_runtime();
    runtime->env = env;
    runtime->comm = std::move(comm);

    runtime->eval_state.comm = runtime->comm.get();
    runtime->eval_state.world_rank = env.world_rank;
    runtime->eval_state.world_size = env.world_size;

    const bool unsupported_topology = env_true("LLAMA_DIST_TP_ALLOW_UNSUPPORTED_TOPOLOGY", false);

    runtime->eval_state.allow_attn_output_collectives = true;

    runtime->decode_sync_state.comm = runtime->comm.get();
    runtime->decode_sync_state.world_rank = env.world_rank;
    runtime->decode_sync_state.world_size = env.world_size;

    const bool use_meta_allreduce =
        runtime->eval_state.world_size > 1 &&
        runtime->eval_state.comm != nullptr;

    if (unsupported_topology && runtime->eval_state.world_rank == 0) {
        std::fprintf(stderr,
                "[dist-tp] unsupported topology: keeping meta-boundary allreduce enabled\n");
    }

    if (use_meta_allreduce) {
        void * meta_ctx = nullptr;
        ggml_backend_meta_dist_allreduce_tensor_t meta_allreduce = nullptr;
        ggml_backend_meta_get_dist_allreduce(&meta_ctx, &meta_allreduce);
        if (meta_allreduce != nullptr &&
                (meta_ctx != &runtime->eval_state || meta_allreduce != dist_tp_meta_allreduce_tensors)) {
            err = "another distributed TP runtime already owns the process-global meta allreduce hook";
            delete runtime;
            return nullptr;
        }
        ggml_backend_meta_set_dist_allreduce(&runtime->eval_state, dist_tp_meta_allreduce_tensors);
    } else {
        ggml_backend_meta_set_dist_allreduce(nullptr, nullptr);
    }

    return runtime;
}

void common_dist_tp_runtime_destroy(common_dist_tp_runtime * runtime) {
    if (runtime != nullptr && g_worker_runtime == runtime) {
        g_worker_runtime = nullptr;
    }

    if (runtime != nullptr && g_worker_decode_sync_state == &runtime->decode_sync_state) {
        g_worker_decode_sync_state = nullptr;
    }

    void * meta_ctx = nullptr;
    ggml_backend_meta_dist_allreduce_tensor_t meta_allreduce = nullptr;
    ggml_backend_meta_get_dist_allreduce(&meta_ctx, &meta_allreduce);
    if (meta_ctx == &runtime->eval_state && meta_allreduce == dist_tp_meta_allreduce_tensors) {
        ggml_backend_meta_set_dist_allreduce(nullptr, nullptr);
    }

    delete runtime;
}

bool common_dist_tp_runtime_attach_context(common_dist_tp_runtime * runtime, llama_context * ctx, std::string & /*err*/) {
    if (runtime == nullptr || ctx == nullptr) {
        return false;
    }

    const llama_model * model = llama_get_model(ctx);
    const int n_head_kv = model != nullptr ? llama_model_n_head_kv(model) : 0;
    runtime->eval_state.allow_attn_output_collectives =
        n_head_kv <= 0 || runtime->eval_state.world_size <= n_head_kv;

    if (runtime->eval_state.world_rank == 0 && n_head_kv > 0 && runtime->eval_state.world_size > n_head_kv) {
        std::fprintf(stderr,
            "[dist-tp] world_size=%d exceeds model n_head_kv=%d: disabling attn_output collectives by default\n",
            runtime->eval_state.world_size,
            n_head_kv);
    }

    llama_set_decode_batch_callback(ctx, dist_tp_decode_batch_callback, &runtime->decode_sync_state);
    if (runtime->decode_sync_state.world_rank != 0) {
        g_worker_runtime = runtime;
        g_worker_decode_sync_state = &runtime->decode_sync_state;
    }
    return true;
}

int common_dist_tp_run_worker(common_init_result & llama_init, const common_dist_tp_env & env) {
    if (!env.enabled) {
        std::fprintf(stderr, "error: distributed TP is not enabled\n");
        return 1;
    }
    if (env.world_rank == 0) {
        std::fprintf(stderr, "error: dist-peer worker mode requires rank > 0\n");
        return 1;
    }

    llama_context * ctx = llama_init.context();
    if (ctx == nullptr || llama_init.model() == nullptr) {
        std::fprintf(stderr, "error: failed to initialize model/context\n");
        return 1;
    }

    llama_batch worker_batch = {};
    while (true) {
        const int rc = llama_decode(ctx, worker_batch);
        if (rc != 0) {
            if (g_worker_decode_sync_state != nullptr &&
                    g_worker_decode_sync_state->transport_ended) {
                std::fprintf(stderr,
                    "info: worker rank %d transport ended, waiting for next rank0 session: %s\n",
                    env.world_rank,
                    g_worker_decode_sync_state->last_transport_error.c_str());

                if (auto * mem = llama_get_memory(ctx)) {
                    llama_memory_clear(mem, true);
                }

                std::string reconnect_err;
                if (!common_dist_tp_worker_reconnect(g_worker_runtime, reconnect_err)) {
                    std::fprintf(stderr,
                        "info: worker rank %d reconnect attempt failed: %s\n",
                        env.world_rank,
                        reconnect_err.c_str());
                    std::this_thread::sleep_for(std::chrono::milliseconds(250));
                }
                continue;
            }
            std::fprintf(stderr, "error: worker llama_decode failed on rank %d\n", env.world_rank);
            return 1;
        }
    }
}
