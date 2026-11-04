#include "dist-comm.h"
#include "dist-comm-nd.h"
#include "ggml-trace.h"
#include "gguf.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
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

constexpr uint32_t DIST_COMM_MAGIC = 0x4443434d; // DCCL + 1: the registration carries the link table
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

// A rank's ND capable addresses and their on-link prefix lengths, sent at registration and
// broadcast by the master, so that every rank can work out the point-to-point link between every
// pair of ranks (link_host).
constexpr uint32_t DIST_MAX_LINK_ADDRS = 8;
struct link_addrs_wire {
    uint32_t rdma;                         // 1 when this rank runs its linked edges on ND
    uint32_t count;
    uint32_t ipv4[DIST_MAX_LINK_ADDRS];    // network byte order
    uint32_t prefix[DIST_MAX_LINK_ADDRS];  // on-link prefix length in bits
};

struct reg_msg {
    uint32_t        magic;
    int32_t         rank;
    int32_t         listen_port;
    link_addrs_wire links;
};

struct member_wire {
    int32_t         port;
    char            host[64];
    link_addrs_wire links;
};

struct op_hdr {
    uint32_t op;
    uint32_t count;
    int32_t  src_rank;
    int32_t  reserved;
};

struct peer_info {
    std::string     host;
    int             port  = 0;
    link_addrs_wire links = {};
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

// same truthiness rule as env_true() below, duplicated here because the transport selection
// runs before that anonymous namespace is in scope
bool dist_transport_env_true(const char * key, bool def) {
    const char * v = std::getenv(key);
    if (!v || v[0] == '\0') {
        return def;
    }
    std::string s(v);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return s == "1" || s == "true" || s == "yes" || s == "on";
}

// This rank's entry for the link table. LLAMA_DIST_TP_RDMA=0 clears rdma, which keeps every edge of
// this rank on TCP while the ring still dials over the point-to-point links.
link_addrs_wire dist_local_links() {
    link_addrs_wire out = {};
    out.rdma = dist_nd_available() && dist_transport_env_true("LLAMA_DIST_TP_RDMA", true) ? 1 : 0;
    for (const dist_nd_local_addr & a : dist_nd_local_addrs()) {
        if (out.count == DIST_MAX_LINK_ADDRS) {
            break;
        }
        out.ipv4[out.count]   = a.ipv4;
        out.prefix[out.count] = a.prefix;
        out.count++;
    }
    return out;
}

} // namespace

struct dist_comm_socket_context::impl {
    dist_comm_config cfg;
    socket_t listen_fd = DIST_CCL_INVALID_SOCKET;
    socket_t ring_in_fd = DIST_CCL_INVALID_SOCKET;
    socket_t ring_out_fd = DIST_CCL_INVALID_SOCKET;
    std::vector<peer_info> members;
    dist_comm_stats stats;

    // Optional NetworkDirect data path for the collective payload. Bootstrap, framing, ring
    // ordering and the broadcasts all stay on the TCP sockets above; only exchange_ring()
    // moves its bytes over ND when these are non-null.
    //
    // There is one connection per ring edge, not one per rank. At world_size 2 next and prev
    // are the same peer, so a single bidirectional connection serves both halves and the two
    // pointers alias it (nd_shared). From world_size 3 up they are different hosts and each
    // edge is negotiated separately, so one half can be on ND while the other is on TCP. That
    // is the normal case on a fabric that is a line rather than a ring: the edge that closes
    // the ring has no point-to-point cable behind it.
    dist_nd_conn * nd_send  = nullptr;   // outbound half, towards next
    dist_nd_conn * nd_recv  = nullptr;   // inbound half, from prev
    bool           nd_shared = false;    // nd_send and nd_recv are the same object; free once

    // Full-mesh all-reduce state, see exchange_mesh(). mesh_max_bytes 0 means the mesh is off.
    size_t               mesh_max_bytes    = 0;
    bool                 mesh_pending      = false;   // the first ring step ran as a mesh exchange
    size_t               mesh_pending_size = 0;
    const void *         mesh_pending_ptr  = nullptr; // where that step left prev's payload
    int32_t              mesh_prev_src     = -1;
    int32_t              mesh_next_src     = -1;
    std::vector<uint8_t> mesh_from_next;              // handed back by the second ring step

    // 4-rank full mesh, when both diagonals have a point-to-point link (want_mesh4), see exchange_mesh4().
    // The diagonal joins this rank and rank + 2 over its own socket and ND connection. mesh4_pending
    // counts the ring steps still to be served from the last exchange: 2 after it, then 1, then 0.
    socket_t             diag_fd            = DIST_CCL_INVALID_SOCKET;
    dist_nd_conn *       nd_diag            = nullptr;
    bool                 mesh4              = false;
    int                  mesh4_pending      = 0;
    size_t               mesh4_pending_size = 0;
    const void *         mesh4_pending_ptr  = nullptr; // where the step before left its payload
    int32_t              mesh4_expect_src   = -1;      // and the rank that payload came from
    int32_t              mesh_diag_src      = -1;
    std::vector<uint8_t> mesh_from_diag;               // handed back by the second ring step

    ~impl() {
        // at world_size 2 both pointers are the same object, so freeing twice would be a
        // double free
        if (nd_send != nullptr) {
            dist_nd_free(nd_send);
            if (nd_shared) {
                nd_recv = nullptr;
            }
            nd_send = nullptr;
        }
        if (nd_recv != nullptr) {
            dist_nd_free(nd_recv);
            nd_recv = nullptr;
        }
        if (nd_diag != nullptr) {
            dist_nd_free(nd_diag);
            nd_diag = nullptr;
        }
        close_fd(ring_in_fd);
        close_fd(ring_out_fd);
        close_fd(diag_fd);
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

        // Decided per edge, not per rank: an edge is on ND when its connection came up, which both
        // of its ends agreed on at setup. From world_size 3 up the two edges are different links
        // and may legitimately differ: one on ND, the other on TCP.
        const bool use_nd_send = nd_send != nullptr;
        const bool use_nd_recv = nd_recv != nullptr;

        auto send_payload = [&]() -> bool {
            if (use_nd_send) {
                // header and payload leave as one ND chunk
                if (!dist_nd_send2(nd_send, &send_hdr, sizeof(send_hdr), send_data_ptr, send_size, err)) {
                    return false;
                }
                stats.bytes_sent += sizeof(send_hdr) + send_size;
                return true;
            }
            if (!write_msg(ring_out_fd, &send_hdr, sizeof(send_hdr), err)) {
                return false;
            }
            if (send_size > 0 && !write_msg(ring_out_fd, send_data_ptr, send_size, err)) {
                return false;
            }
            return true;
        };

        auto recv_payload = [&]() -> bool {
            if (use_nd_recv) {
                if (!dist_nd_recv(nd_recv, &recv_hdr, sizeof(recv_hdr), err)) {
                    return false;
                }
                stats.bytes_recv += sizeof(recv_hdr);
            } else if (!read_msg(ring_in_fd, &recv_hdr, sizeof(recv_hdr), err)) {
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
            if (recv_size > 0) {
                if (use_nd_recv) {
                    if (!dist_nd_recv(nd_recv, recv_data_ptr, recv_size, err)) {
                        return false;
                    }
                    stats.bytes_recv += recv_size;
                } else if (!read_msg(ring_in_fd, recv_data_ptr, recv_size, err)) {
                    return false;
                }
            }
            recv_src_rank = recv_hdr.src_rank;
            return true;
        };

        // Full-duplex exchange. A decode collective is almost pure latency, and decode runs one per
        // subgraph boundary for every token. dist_nd_send2 returns once the chunk is posted, so
        // posting first and draining after the receive puts both directions on the wire together,
        // one transfer's time where send-then-receive would take two.
        //
        // Taken only when this edge's send is actually on ND. A TCP send blocks in the socket
        // buffer and gains nothing, and a mixed edge (ND one way, TCP the other, which happens
        // from world_size 3 up) keeps the ordering it was reasoned about with.
        bool ok;
        if (use_nd_send) {
            ok = send_payload() && recv_payload() && dist_nd_send_drain(nd_send, err);
        } else {
            // Rank parity avoids blocking send/send for large payloads.
            //
            // Parity is a 2-colouring of the ring, and an odd-length cycle cannot be 2-coloured,
            // so from world_size 3 up exactly one edge has both ends calling send_payload()
            // first. That is a property of the topology, not a bug to schedule around: no
            // assignment of send-first / recv-first to ranks can do better than one such edge on
            // an odd ring, and parity already achieves that minimum. Do not "fix" this by
            // switching to a rank < peer rule; a rank sits on two edges and one sequential
            // ordering cannot satisfy both.
            //
            // It is safe because neither transport requires the peer to be inside recv_payload()
            // before a send can make progress. TCP has SOCKET_BUFSIZE of socket buffer to absorb
            // the payload, and ND completes a send into the receives the peer posted at setup
            // time. It costs latency on that one edge, not correctness.
            ok = ((cfg.rank & 1) == 0)
                ? (send_payload() && recv_payload())
                : (recv_payload() && send_payload());
        }

        if (ok) {
            // Recorded so the trace report accounts for the bytes this collective actually put
            // on the wire, tagged with the transport that carried them. An edge whose ND
            // connection did not come up stays on TCP even in an RDMA run, so the endpoint is
            // read from use_nd rather than from whether ND is configured.
            const int64_t ne[GGML_MAX_DIMS] = { (int64_t) send_size, 1, 1, 1 };
            // the two halves can ride different transports once the ring spans more than two
            // hosts, so a mixed step is reported as such rather than being rounded to one label
            const char * endpoint = use_nd_send ? (use_nd_recv ? "nd" : "nd/tcp")
                                                : (use_nd_recv ? "tcp/nd" : "tcp");
            ggml_trace_rpc(endpoint, cfg.rank, "EXCHANGE_RING", "dist-tp", ne,
                           sizeof(send_hdr) + send_size, sizeof(recv_hdr) + recv_size);
        }
        return ok;
    }

    // Whether a collective of this size runs as a mesh exchange. Every rank must answer the
    // same way for the same size, so this reads only values every rank derives alike, never
    // whether this rank's own ND connections happen to be up: a rank that fell back to TCP must
    // not go to the mesh while its peers stay on the ring.
    bool mesh_takes(size_t size) const {
        return mesh_max_bytes != 0 && size <= mesh_max_bytes;
    }

    // Full-mesh exchange for world_size 3.
    //
    // The ring all-reduce costs world_size - 1 sequential exchanges per collective. A decode
    // collective is a 16 KB payload, so each of those exchanges is almost pure wire latency, and
    // at world_size 3 the second one doubles the network cost of every collective. Both other
    // ranks are ring neighbours at world_size 3, so the connections a mesh needs already exist:
    // the edge towards next and the edge from prev each carry traffic in both directions. ND
    // connections are bidirectional by construction (each end runs both halves of the flow
    // control, exactly as at world_size 2), and the ring sockets are full-duplex TCP whose
    // reverse direction is otherwise unused. One mesh exchange sends this rank's payload to both
    // peers and receives both of theirs, so a collective costs one round of latency, not two.
    //
    // It is transparent to the caller. Both callers in ggml-backend-meta.cpp run a store and
    // forward ring: the second step sends exactly the buffer the first step received, tagged
    // with the rank it came from. The first call does the whole exchange and returns prev's
    // payload; the second returns next's payload without touching the wire. At world_size 3
    // that is the same (payload, source) sequence the ring delivers, so the caller's sum
    // (own + prev) + next is bit-identical to the ring's. The second call checks that it really
    // is forwarding the first call's result and fails loudly if a caller ever stops doing that.
    //
    // Only payloads up to mesh_max_bytes take the mesh (mesh_takes). Both sends are posted before
    // either receive, which is safe while a payload fits one ND chunk and the TCP socket buffer.
    // Larger payloads (prefill) keep the ring. Every rank sees the same payload size, so every
    // rank takes the same path.
    bool exchange_mesh(uint32_t op, int32_t src_rank, const void * send_data_ptr, size_t size,
            int32_t & recv_src_rank, void * recv_data_ptr, std::string & err) {
        if (mesh_pending) {
            mesh_pending = false;
            if (size != mesh_pending_size || send_data_ptr != mesh_pending_ptr || src_rank != mesh_prev_src) {
                err = "mesh exchange: second ring step is not forwarding the first step's payload";
                return false;
            }
            std::memcpy(recv_data_ptr, mesh_from_next.data(), size);
            recv_src_rank = mesh_next_src;
            return true;
        }

        // Each ND connection carries both directions of its edge, under the ring's per-edge rule.
        const bool nd_next = nd_send != nullptr;
        const bool nd_prev = nd_recv != nullptr;

        const op_hdr send_hdr {
            op,
            static_cast<uint32_t>(size),
            src_rank,
            0,
        };

        auto send_to = [&](bool use_nd, dist_nd_conn * c, socket_t fd) -> bool {
            if (use_nd) {
                if (!dist_nd_send2(c, &send_hdr, sizeof(send_hdr), send_data_ptr, size, err)) {
                    return false;
                }
                stats.bytes_sent += sizeof(send_hdr) + size;
                return true;
            }
            if (!write_msg(fd, &send_hdr, sizeof(send_hdr), err)) {
                return false;
            }
            return size == 0 || write_msg(fd, send_data_ptr, size, err);
        };

        auto recv_from = [&](bool use_nd, dist_nd_conn * c, socket_t fd, void * dst, int32_t & src_out) -> bool {
            op_hdr hdr {};
            if (use_nd) {
                if (!dist_nd_recv(c, &hdr, sizeof(hdr), err)) {
                    return false;
                }
                stats.bytes_recv += sizeof(hdr);
            } else if (!read_msg(fd, &hdr, sizeof(hdr), err)) {
                return false;
            }
            if (hdr.op != op) {
                err = "unexpected op code on mesh";
                return false;
            }
            if (hdr.count != size) {
                err = "unexpected payload size on mesh";
                return false;
            }
            if (size > 0) {
                if (use_nd) {
                    if (!dist_nd_recv(c, dst, size, err)) {
                        return false;
                    }
                    stats.bytes_recv += size;
                } else if (!read_msg(fd, dst, size, err)) {
                    return false;
                }
            }
            src_out = hdr.src_rank;
            return true;
        };

        mesh_from_next.resize(size);

        // Both sends are posted before either receive, so the two links carry this rank's payload
        // at the same time. An ND send returns once it is posted, and the drains below retire
        // both.
        bool ok = send_to(nd_next, nd_send, ring_out_fd) &&
                  send_to(nd_prev, nd_recv, ring_in_fd) &&
                  recv_from(nd_prev, nd_recv, ring_in_fd, recv_data_ptr, mesh_prev_src) &&
                  recv_from(nd_next, nd_send, ring_out_fd, mesh_from_next.data(), mesh_next_src);
        if (ok && nd_next) {
            ok = dist_nd_send_drain(nd_send, err);
        }
        if (ok && nd_prev) {
            ok = dist_nd_send_drain(nd_recv, err);
        }
        if (!ok) {
            return false;
        }

        recv_src_rank     = mesh_prev_src;
        mesh_pending      = true;
        mesh_pending_size = size;
        mesh_pending_ptr  = recv_data_ptr;

        const int64_t ne[GGML_MAX_DIMS] = { (int64_t) size, 1, 1, 1 };
        const char * endpoint = (nd_next && nd_prev) ? "nd" : (nd_next || nd_prev) ? "nd/tcp" : "tcp";
        ggml_trace_rpc(endpoint, cfg.rank, "EXCHANGE_MESH", "dist-tp", ne,
                       2 * (sizeof(send_hdr) + size), 2 * (sizeof(send_hdr) + size));
        return true;
    }

    // p's address on the point-to-point link between ranks a and p: a subnet on which both have an
    // ND capable address, the longest prefix if there are several. Empty when there is none. Every
    // rank holds the same table, so every rank gets the same answer for every pair.
    std::string link_host(int a, int p) const {
        const link_addrs_wire & la = members[a].links;
        const link_addrs_wire & lp = members[p].links;
        uint32_t best        = 0;
        uint32_t best_prefix = 0;
        for (uint32_t i = 0; i < la.count; ++i) {
            for (uint32_t j = 0; j < lp.count; ++j) {
                const uint32_t pre = la.prefix[i];
                if (pre == 0 || pre > 32 || pre != lp.prefix[j] || pre <= best_prefix || la.ipv4[i] == lp.ipv4[j]) {
                    continue;
                }
                const uint32_t mask = htonl(pre == 32 ? 0xffffffffu : ~(0xffffffffu >> pre));
                if (((la.ipv4[i] ^ lp.ipv4[j]) & mask) == 0) {
                    best        = lp.ipv4[j];
                    best_prefix = pre;
                }
            }
        }
        if (best_prefix == 0) {
            return "";
        }
        in_addr addr {};
        addr.s_addr = best;
        char buf[INET_ADDRSTRLEN] = {0};
        return inet_ntop(AF_INET, &addr, buf, sizeof(buf)) != nullptr ? std::string(buf) : std::string();
    }

    // Whether the edge between ranks a and b runs on ND: both ranks want it and they share a link.
    bool edge_on_nd(int a, int b) const {
        return members[a].links.rdma != 0 && members[b].links.rdma != 0 && !link_host(a, b).empty();
    }

    // world_size 4 with a point-to-point link on both diagonals (0-2 and 1-3), so a cable between
    // every pair of ranks
    bool want_mesh4() const {
        return cfg.world_size == 4 && !link_host(0, 2).empty() && !link_host(1, 3).empty();
    }

    // establish_ring() for the 4-rank mesh: the ring edges as usual plus the two diagonals, 0-2 and 1-3.
    // Every rank dials next and, on ranks 2 and 3, its diagonal, and names itself on each socket at
    // once; then it accepts prev and, on ranks 0 and 1, the diagonal, telling them apart by the rank
    // each one names. No rank reads before it has written everything it dials, so the order in which
    // the connections arrive does not matter.
    bool establish_ring_mesh4(const std::string & next_host, int next_port, std::string & err) {
        const int  prev      = (cfg.rank + 3) % 4;
        const int  diag      = (cfg.rank + 2) % 4;
        const bool dial_diag = cfg.rank > diag;
        const int32_t me = cfg.rank;

        ring_out_fd = connect_with_retry(next_host, next_port, err);
        if (ring_out_fd == DIST_CCL_INVALID_SOCKET) {
            return false;
        }
        if (!write_msg(ring_out_fd, &me, sizeof(me), err)) {
            return false;
        }
        if (dial_diag) {
            const std::string host = link_host(cfg.rank, diag);
            const int         port = members[diag].port;
            std::fprintf(stderr, "dist-tp: diagonal %d->%d dials %s:%d (point-to-point link)\n",
                         cfg.rank, diag, host.c_str(), port);
            std::fflush(stderr);
            diag_fd = connect_with_retry(host, port, err);
            if (diag_fd == DIST_CCL_INVALID_SOCKET) {
                err = "diagonal " + std::to_string(cfg.rank) + "->" + std::to_string(diag) + ": " + err;
                return false;
            }
            if (!write_msg(diag_fd, &me, sizeof(me), err)) {
                return false;
            }
        }

        for (int left = dial_diag ? 1 : 2; left > 0; --left) {
            std::string peer_host;
            socket_t fd = accept_socket(listen_fd, peer_host, err);
            if (fd == DIST_CCL_INVALID_SOCKET) {
                return false;
            }
            int32_t r = -1;
            if (!read_msg(fd, &r, sizeof(r), err)) {
                close_fd(fd);
                return false;
            }
            if (r == prev && ring_in_fd == DIST_CCL_INVALID_SOCKET) {
                ring_in_fd = fd;
            } else if (r == diag && !dial_diag && diag_fd == DIST_CCL_INVALID_SOCKET) {
                diag_fd = fd;
            } else {
                close_fd(fd);
                err = "ring peer rank mismatch (rank " + std::to_string(r) + " from " + peer_host + ")";
                return false;
            }
        }

        close_fd(listen_fd);
        return true;
    }

    // The diagonal's ND connection, after every ring edge: each rank is on exactly one diagonal, and
    // both of its ends get here once their own ring edges are done (establish_nd_per_edge), so no rank
    // waits on a peer that is still busy with another edge. The lower rank listens, as on the ring.
    void establish_nd_diag() {
        const int diag = (cfg.rank + 2) % 4;
        if (diag_fd == DIST_CCL_INVALID_SOCKET || !edge_on_nd(cfg.rank, diag)) {
            return;
        }
        std::string nd_err;
        nd_diag = dist_nd_establish((uintptr_t) diag_fd, (uintptr_t) diag_fd, cfg.rank < diag,
                                    nd_err, /* strict_local_addr */ true);
        if (nd_diag == nullptr) {
            std::fprintf(stderr,
                         "dist-tp: NetworkDirect not established on diagonal %d<->%d "
                         "(rank %d, %s); the diagonal stays on TCP\n",
                         cfg.rank < diag ? cfg.rank : diag, cfg.rank < diag ? diag : cfg.rank, cfg.rank, nd_err.c_str());
        }
    }

    bool mesh4_takes(size_t size) const {
        return mesh4 && size <= mesh_max_bytes;
    }

    // Full-mesh exchange for world_size 4, with one cable between every pair of ranks (want_mesh4).
    //
    // The ring all-reduce takes world_size - 1 = 3 dependent exchanges per collective. Here every
    // rank sends its payload to all three peers at once and receives theirs: next and prev over the
    // two ring edges, each used in both directions as in the 3-rank mesh, and rank + 2 over the
    // diagonal. A collective costs one round of latency, not three.
    //
    // Transparent to the caller, as at world_size 3: the ring's three steps hand over prev's payload,
    // then diag's, then next's, each step forwarding the buffer the step before it received. The
    // first call does the whole exchange and returns prev's payload; the second and third return
    // diag's and next's without touching the wire, after checking that they really forward what the
    // step before returned. The caller's sum is therefore bit-identical to the ring's.
    //
    // Payloads up to mesh_max_bytes (mesh4_takes) post all three sends, then take the three
    // receives. Larger ones (prefill) keep the ring.
    bool exchange_mesh4(uint32_t op, int32_t src_rank, const void * send_data_ptr, size_t size,
            int32_t & recv_src_rank, void * recv_data_ptr, std::string & err) {
        if (mesh4_pending > 0) {
            if (size != mesh4_pending_size || send_data_ptr != mesh4_pending_ptr || src_rank != mesh4_expect_src) {
                mesh4_pending = 0;
                err = "mesh exchange: a ring step is not forwarding the step before's payload (4 ranks)";
                return false;
            }
            const bool diag_step = mesh4_pending == 2;
            std::memcpy(recv_data_ptr, diag_step ? mesh_from_diag.data() : mesh_from_next.data(), size);
            recv_src_rank     = diag_step ? mesh_diag_src : mesh_next_src;
            mesh4_pending    -= 1;
            mesh4_pending_ptr = recv_data_ptr;
            mesh4_expect_src  = recv_src_rank;
            return true;
        }

        const int next = (cfg.rank + 1) % 4;
        const int prev = (cfg.rank + 3) % 4;
        const int diag = (cfg.rank + 2) % 4;

        // One rule per connection, the same at both of its ends: ND when the connection came up.
        const bool nd_next = nd_send != nullptr;
        const bool nd_prev = nd_recv != nullptr;
        const bool nd_dg   = nd_diag != nullptr;

        const op_hdr send_hdr {
            op,
            static_cast<uint32_t>(size),
            src_rank,
            0,
        };

        auto send_to = [&](bool use_nd, dist_nd_conn * c, socket_t fd) -> bool {
            if (use_nd) {
                if (!dist_nd_send2(c, &send_hdr, sizeof(send_hdr), send_data_ptr, size, err)) {
                    return false;
                }
                stats.bytes_sent += sizeof(send_hdr) + size;
                return true;
            }
            if (!write_msg(fd, &send_hdr, sizeof(send_hdr), err)) {
                return false;
            }
            return size == 0 || write_msg(fd, send_data_ptr, size, err);
        };

        auto check_hdr = [&](const op_hdr & hdr) -> bool {
            if (hdr.op != op) {
                err = "unexpected op code on mesh";
                return false;
            }
            if (hdr.count != size) {
                err = "unexpected payload size on mesh";
                return false;
            }
            return true;
        };

        auto recv_from = [&](bool use_nd, dist_nd_conn * c, socket_t fd, void * dst, int32_t & src_out) -> bool {
            op_hdr hdr {};
            if (use_nd) {
                if (!dist_nd_recv(c, &hdr, sizeof(hdr), err)) {
                    return false;
                }
                stats.bytes_recv += sizeof(hdr);
            } else if (!read_msg(fd, &hdr, sizeof(hdr), err)) {
                return false;
            }
            if (!check_hdr(hdr)) {
                return false;
            }
            if (size > 0) {
                if (use_nd) {
                    if (!dist_nd_recv(c, dst, size, err)) {
                        return false;
                    }
                    stats.bytes_recv += size;
                } else if (!read_msg(fd, dst, size, err)) {
                    return false;
                }
            }
            src_out = hdr.src_rank;
            return true;
        };

        mesh_from_next.resize(size);
        mesh_from_diag.resize(size);

        // All three sends are posted before any receive, so the three links carry this rank's
        // payload at the same time. An ND send returns once it is posted, and the drains below
        // retire them.
        bool ok = send_to(nd_next, nd_send, ring_out_fd) &&
                  send_to(nd_prev, nd_recv, ring_in_fd) &&
                  send_to(nd_dg, nd_diag, diag_fd) &&
                  recv_from(nd_prev, nd_recv, ring_in_fd, recv_data_ptr, mesh_prev_src) &&
                  recv_from(nd_dg, nd_diag, diag_fd, mesh_from_diag.data(), mesh_diag_src) &&
                  recv_from(nd_next, nd_send, ring_out_fd, mesh_from_next.data(), mesh_next_src);
        if (ok && nd_next) {
            ok = dist_nd_send_drain(nd_send, err);
        }
        if (ok && nd_prev) {
            ok = dist_nd_send_drain(nd_recv, err);
        }
        if (ok && nd_dg) {
            ok = dist_nd_send_drain(nd_diag, err);
        }
        if (!ok) {
            return false;
        }
        if (mesh_prev_src != prev || mesh_diag_src != diag || mesh_next_src != next) {
            err = "mesh exchange: a payload arrived from an unexpected rank (4 ranks)";
            return false;
        }

        recv_src_rank      = mesh_prev_src;
        mesh4_pending      = 2;
        mesh4_pending_size = size;
        mesh4_pending_ptr  = recv_data_ptr;
        mesh4_expect_src   = mesh_prev_src;

        const int64_t ne[GGML_MAX_DIMS] = { (int64_t) size, 1, 1, 1 };
        const char * endpoint = (nd_next && nd_prev && nd_dg) ? "nd"
                              : (nd_next || nd_prev || nd_dg) ? "nd/tcp" : "tcp";
        ggml_trace_rpc(endpoint, cfg.rank, "EXCHANGE_MESH", "dist-tp", ne,
                       3 * (sizeof(send_hdr) + size), 3 * (sizeof(send_hdr) + size));
        return true;
    }

    // select_allreduce() at world_size 4 with the diagonals up (establish_ring_mesh4).
    void select_allreduce_mesh4() {
        constexpr size_t cap = 256 * 1024 - sizeof(op_hdr);
        mesh_max_bytes = 64 * 1024;
        const char * v = std::getenv("LLAMA_DIST_TP_MESH_MAX_BYTES");
        if (v != nullptr && v[0] != '\0') {
            try {
                const long long n = std::stoll(v);
                if (n > 0) {
                    mesh_max_bytes = std::min((size_t) n, cap);
                }
            } catch (...) {
                // keep the default
            }
        }
        mesh4 = true;

        const int next = (cfg.rank + 1) % 4;
        const int prev = (cfg.rank + 3) % 4;
        const int diag = (cfg.rank + 2) % 4;
        std::fprintf(stderr,
                     "dist-tp all-reduce: full mesh up to %zu B, ring above [rank %d/%d] "
                     "(<->%d %s, <->%d %s, <->%d %s)\n",
                     mesh_max_bytes, cfg.rank, cfg.world_size,
                     next, nd_send != nullptr ? "ND" : "TCP",
                     prev, nd_recv != nullptr ? "ND" : "TCP",
                     diag, nd_diag != nullptr ? "ND" : "TCP");
        std::fflush(stderr);
    }

    // Full-mesh all-reduce at world_size 3, and at 4 with a cable between every pair of ranks; the
    // ring otherwise. See exchange_mesh(). Runs after select_transport() so the banner below can
    // say which transports the mesh will use.
    void select_allreduce() {
        if (cfg.world_size == 4 && diag_fd != DIST_CCL_INVALID_SOCKET) {
            select_allreduce_mesh4();
            return;
        }
        if (cfg.world_size != 3) {
            return;
        }

        // A decode collective is 16 KB. The cap keeps a mesh payload inside one ND chunk
        // (ND_PAYLOAD in dist-comm-nd.cpp, 256 KiB including this header).
        constexpr size_t cap = 256 * 1024 - sizeof(op_hdr);
        mesh_max_bytes = 64 * 1024;
        const char * v = std::getenv("LLAMA_DIST_TP_MESH_MAX_BYTES");
        if (v != nullptr && v[0] != '\0') {
            try {
                const long long n = std::stoll(v);
                if (n > 0) {
                    mesh_max_bytes = std::min((size_t) n, cap);
                }
            } catch (...) {
                // keep the default
            }
        }

        const int next = (cfg.rank + 1) % cfg.world_size;
        const int prev = (cfg.rank + cfg.world_size - 1) % cfg.world_size;
        std::fprintf(stderr,
                     "dist-tp all-reduce: full mesh up to %zu B, ring above [rank %d/%d] "
                     "(<->%d %s, <->%d %s)\n",
                     mesh_max_bytes, cfg.rank, cfg.world_size,
                     next, nd_send != nullptr ? "ND" : "TCP",
                     prev, nd_recv != nullptr ? "ND" : "TCP");
        std::fflush(stderr);
    }

    bool bootstrap_master(std::string & err) {
        members.assign(cfg.world_size, {});
        members[0] = {cfg.master_host, cfg.listen_port, dist_local_links()};

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
            if (msg.links.count > DIST_MAX_LINK_ADDRS) {
                err = "invalid address count in registration";
                close_fd(fd);
                return false;
            }
            members[msg.rank] = {peer_host, msg.listen_port, msg.links};
            reg_fds.push_back(fd);
        }

        std::vector<member_wire> payload(cfg.world_size);
        for (int i = 0; i < cfg.world_size; ++i) {
            payload[i].port  = members[i].port;
            payload[i].links = members[i].links;
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
            dist_local_links(),
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
            members[i].host  = payload[i].host;
            members[i].port  = payload[i].port;
            members[i].links = payload[i].links;
            if (members[i].links.count > DIST_MAX_LINK_ADDRS) {
                err = "invalid address count in the member table";
                return false;
            }
        }

        return true;
    }

    bool establish_ring(std::string & err) {
        if (cfg.world_size == 1) {
            return true;
        }

        const int next = (cfg.rank + 1) % cfg.world_size;
        const int prev = (cfg.rank + cfg.world_size - 1) % cfg.world_size;

        // Where to dial for this rank's outbound ring edge. The master recorded every rank at the
        // source address of its registration, on the one network every rank shares. When this
        // rank and next both have an ND capable address on one subnet, that subnet is the
        // point-to-point link between them (a USB4 cable is a subnet of its own, with no route to
        // the others), and the edge dials next's end of it instead. That is not only for speed:
        // dist-comm-nd takes the local RDMA address from the ring socket, so an edge dialed over
        // the shared network can only ever run TCP. The inbound side needs nothing, since the
        // listener is bound to INADDR_ANY. A pair without a link, such as the edge that closes the
        // ring on a fabric shaped like a line, keeps the registered address.
        const std::string link      = link_host(cfg.rank, next);
        const std::string next_host = link.empty() ? members[next].host : link;
        const int         next_port = members[next].port;
        std::fprintf(stderr, "dist-tp: ring edge %d->%d dials %s:%d (%s)\n", cfg.rank, next, next_host.c_str(),
                     next_port, link.empty() ? "registered address" : "point-to-point link");

        if (want_mesh4()) {
            return establish_ring_mesh4(next_host, next_port, err);
        }

        ring_out_fd = connect_with_retry(next_host, next_port, err);
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

    // world_size >= 3: next and prev are different hosts, so each ring edge needs its own ND
    // connection. Three things make this work that did not arise at world_size 2.
    //
    //  - Edges are negotiated in one fixed global order, e = 0 .. world_size-1, where edge e
    //    joins rank e to rank e+1. Every rank walks the same list and acts only on the two edges
    //    it belongs to. Both endpoints of an edge therefore handshake at the same moment, and no
    //    rank is left blocked reading a peer that is busy negotiating a different edge. Get this
    //    order wrong and the run hangs for ND_CONNECT_TIMEOUT_S per edge instead of failing.
    //
    //  - On each edge the lower rank listens. The provider's connection manager sits on a fixed
    //    port (ND_CM_PORT) and its listener binds one specific local address, so two listeners
    //    clash only when they land on the same address. Lower-rank-listens leaves every host
    //    with at most one listener here, which sidesteps the question rather than relying on the
    //    two USB4 adapters on the hub node having distinct addresses.
    //
    //  - The negotiation is handed the same socket for both halves. dist_nd_establish() writes
    //    on tcp_out and reads on tcp_in; at world_size 2 those are two sockets to one peer, but
    //    per edge there is exactly one socket and it already faces the right host.
    //
    // An edge whose two ranks share no link stays on TCP (edge_on_nd), which is the expected
    // outcome for the edge that closes the ring on a line-shaped fabric: it has no point-to-point
    // cable behind it and runs over whatever ordinary network carries it.
    void establish_nd_per_edge() {
        for (int e = 0; e < cfg.world_size; ++e) {
            const int u = e;
            const int v = (e + 1) % cfg.world_size;
            if ((cfg.rank != u && cfg.rank != v) || !edge_on_nd(u, v)) {
                continue;
            }
            const bool i_am_u = (cfg.rank == u);
            // the socket already connected to this edge's peer: outbound for u, inbound for v
            const socket_t fd = i_am_u ? ring_out_fd : ring_in_fd;
            // computed identically on both ends, so exactly one side listens
            const bool listen_side = (cfg.rank == (u < v ? u : v));

            std::string nd_err;
            dist_nd_conn * c = dist_nd_establish((uintptr_t) fd, (uintptr_t) fd, listen_side,
                                                 nd_err, /* strict_local_addr */ true);
            if (c == nullptr) {
                std::fprintf(stderr,
                             "dist-tp: NetworkDirect not established on ring edge %d->%d "
                             "(rank %d, %s); that edge stays on TCP\n",
                             u, v, cfg.rank, nd_err.c_str());
            }
            if (i_am_u) {
                nd_send = c;
            } else {
                nd_recv = c;
            }
        }
    }

    // Selects the data path for exchange_ring(). An edge runs on ND when both of its ranks have
    // an ND capable address on one subnet and neither sets LLAMA_DIST_TP_RDMA=0 (edge_on_nd);
    // any failure here is logged and leaves that edge on TCP rather than aborting the run.
    void select_transport() {
        if (cfg.world_size == 2 && edge_on_nd(0, 1)) {
            std::string nd_err;
            // next and prev are the same peer, so one bidirectional connection carries both
            // halves. rank 0 listens and advertises, rank 1 dials in; rank 0 makes the final
            // call.
            nd_send = dist_nd_establish((uintptr_t) ring_out_fd, (uintptr_t) ring_in_fd,
                                        cfg.rank == 0, nd_err);
            if (nd_send == nullptr) {
                std::fprintf(stderr,
                             "dist-tp: NetworkDirect setup failed on rank %d (%s); "
                             "falling back to TCP\n", cfg.rank, nd_err.c_str());
            } else {
                nd_recv   = nd_send;
                nd_shared = true;
            }
        } else if (cfg.world_size > 2) {
            establish_nd_per_edge();
            establish_nd_diag();
        }

        // The harness greps for this exact prefix to prove which transport a benchmark actually
        // used, so the leading "dist-tp transport: " and the "[rank i/N]" field must stay put.
        const bool any_nd = nd_send != nullptr || nd_recv != nullptr;
        std::fprintf(stderr, "dist-tp transport: %s [rank %d/%d]",
                     any_nd ? "RDMA (ND)" : "TCP", cfg.rank, cfg.world_size);
        if (cfg.world_size > 2) {
            // From 3 ranks up the two edges are separate links and a partial result is the
            // expected outcome on a fabric that is a line, so report each edge rather than
            // letting one word imply both are on ND.
            const int next = (cfg.rank + 1) % cfg.world_size;
            const int prev = (cfg.rank + cfg.world_size - 1) % cfg.world_size;
            std::fprintf(stderr, " (->%d %s, %d-> %s)",
                         next, nd_send != nullptr ? "ND" : "TCP",
                         prev, nd_recv != nullptr ? "ND" : "TCP");
            if (diag_fd != DIST_CCL_INVALID_SOCKET) {
                std::fprintf(stderr, " (<->%d %s)", (cfg.rank + 2) % cfg.world_size, nd_diag != nullptr ? "ND" : "TCP");
            }
        }
        if (any_nd && diag_fd != DIST_CCL_INVALID_SOCKET && nd_diag == nullptr) {
            // the harness reads "all payloads over ND" as proof that nothing rode TCP; a diagonal on
            // TCP must not print it
            std::fprintf(stderr, " (diagonal on TCP)");
        } else if (any_nd) {
            std::fprintf(stderr, " (all payloads over ND)");
        }
        std::fprintf(stderr, "\n");
        std::fflush(stderr);
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

    // one decode exchange (op header and payload) takes the small ND connection
    dist_nd_set_small_message(config.decode_floats > 0 ? sizeof(op_hdr) + config.decode_floats * sizeof(float) : 0);

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

    p->select_transport();
    p->select_allreduce();

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

    // A pending mesh step must be served by the mesh whatever its size check would say now.
    const size_t bytes = n * sizeof(float);
    if (impl_->mesh4) {
        if (impl_->mesh4_pending > 0 || impl_->mesh4_takes(bytes)) {
            return impl_->exchange_mesh4(OP_ALLREDUCE_F32, src_rank, send_data, bytes, recv_src_rank, recv_data, err);
        }
    } else if (impl_->mesh_pending || impl_->mesh_takes(bytes)) {
        return impl_->exchange_mesh(OP_ALLREDUCE_F32, src_rank, send_data, bytes, recv_src_rank, recv_data, err);
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

// Floats in the exchange of one decode token: one row of the model's n_embd in the boundary
// all-reduce's wire format. 0 when the model header does not give n_embd.
size_t dist_tp_decode_floats(const std::string & model_path) {
    gguf_init_params gp = { /* no_alloc = */ true, /* ctx = */ nullptr };
    gguf_context * g = gguf_init_from_file(model_path.c_str(), gp);
    if (g == nullptr) {
        return 0;
    }
    size_t n_embd = 0;
    const int64_t arch = gguf_find_key(g, "general.architecture");
    if (arch >= 0 && gguf_get_kv_type(g, arch) == GGUF_TYPE_STRING) {
        const std::string key = std::string(gguf_get_val_str(g, arch)) + ".embedding_length";
        const int64_t k = gguf_find_key(g, key.c_str());
        if (k >= 0 && gguf_get_kv_type(g, k) == GGUF_TYPE_UINT32) {
            n_embd = gguf_get_val_u32(g, k);
        }
    }
    gguf_free(g);
    return n_embd > 0 ? ggml_backend_meta_dist_wire_floats(n_embd) : 0;
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
    size_t decode_floats = 0; // dist_comm_config::decode_floats, kept for a worker reconnect
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
    comm_cfg.rank          = runtime->env.world_rank;
    comm_cfg.world_size    = runtime->env.world_size;
    comm_cfg.master_host   = runtime->env.master_host;
    comm_cfg.master_port   = runtime->env.master_port;
    comm_cfg.listen_port   = runtime->env.listen_port;
    comm_cfg.decode_floats = runtime->decode_floats;

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
    if (!env.enabled) {
        return nullptr;
    }

    // Cross-rank distributed TP communication is implemented here (dist_comm + callbacks).
    // This path is independent from ggml-backend-meta internals.
    dist_comm_config comm_cfg;
    comm_cfg.rank          = env.world_rank;
    comm_cfg.world_size    = env.world_size;
    comm_cfg.master_host   = env.master_host;
    comm_cfg.master_port   = env.master_port;
    comm_cfg.listen_port   = env.listen_port;
    comm_cfg.decode_floats = dist_tp_decode_floats(params.model.path);

    auto comm = dist_comm_create(comm_cfg, err);
    if (!comm) {
        return nullptr;
    }

    auto * runtime = new common_dist_tp_runtime();
    runtime->env = env;
    runtime->decode_floats = comm_cfg.decode_floats;
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
