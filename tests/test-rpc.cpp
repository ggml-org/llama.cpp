// Losing an RPC server must not take the local process down with it.
// A real RPC server runs in a background thread, with a small TCP proxy between it and the
// client. Closing the proxy is the fault: the socket goes away in the middle of an operation,
// which is what the client sees when a remote worker dies.
//
// Not covered: the server-side GGML_ASSERT removal in rpc_server::graph_compute(). That needs
// a backend whose graph compute fails, and the CPU backend has no such path.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-rpc.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

static int g_failures = 0;

static void check(bool ok, const char * what) {
    printf("  %-62s %s\n", what, ok ? "OK" : "FAIL");
    if (!ok) {
        g_failures++;
    }
}

static void begin(const char * name) {
    printf("%s\n", name);
}

// reaching the call is the assertion: an abort would have taken the process down first
static void reached(const char * what) {
    check(true, what);
}

// the proxy must not die of SIGPIPE itself when it writes to a socket the client already
// closed. It is per socket on purpose: ignoring SIGPIPE process wide would also cover the
// RPC client, which is one of the things under test here.
#if defined(SO_NOSIGPIPE)
#  define PROXY_SEND_FLAGS 0
#else
#  define PROXY_SEND_FLAGS MSG_NOSIGNAL
#endif

static void no_sigpipe(int fd) {
#ifdef SO_NOSIGPIPE
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#else
    (void) fd;
#endif
}

// the RPC server cannot report the port it got, so it must be told one up front
static int find_free_port() {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return -1;
    }
    sockaddr_in addr = {};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port        = 0;
    socklen_t len = sizeof(addr);
    if (bind(fd, (sockaddr *) &addr, sizeof(addr)) < 0 || getsockname(fd, (sockaddr *) &addr, &len) < 0) {
        close(fd);
        return -1;
    }
    int port = ntohs(addr.sin_port);
    close(fd);
    return port;
}

static int connect_loopback(int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return -1;
    }
    sockaddr_in addr = {};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port        = htons(port);
    if (connect(fd, (sockaddr *) &addr, sizeof(addr)) < 0) {
        close(fd);
        return -1;
    }
    no_sigpipe(fd);
    return fd;
}

// Loopback TCP proxy with a kill switch, between the RPC client and the RPC server.
// The loops poll with a short timeout instead of blocking in accept()/recv(), because waking
// a thread parked in a syscall by closing its socket is not portable.
class rpc_proxy {
public:
    ~rpc_proxy() { kill(); }

    bool start(int upstream_port) {
        upstream = upstream_port;

        listen_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (listen_fd < 0) {
            return false;
        }
        int one = 1;
        setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

        sockaddr_in addr = {};
        addr.sin_family      = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port        = 0;
        socklen_t len = sizeof(addr);
        if (bind(listen_fd, (sockaddr *) &addr, sizeof(addr)) < 0 ||
            listen(listen_fd, 4) < 0 ||
            getsockname(listen_fd, (sockaddr *) &addr, &len) < 0) {
            close(listen_fd);
            listen_fd = -1;
            return false;
        }
        listen_port = ntohs(addr.sin_port);

        accept_thread = std::thread(&rpc_proxy::accept_loop, this);
        return true;
    }

    std::string endpoint() const { return "127.0.0.1:" + std::to_string(listen_port); }

    // the fault: drop every socket, like the remote worker process going away
    void kill() {
        if (stopping.exchange(true)) {
            return;
        }
        if (accept_thread.joinable()) {
            accept_thread.join();
        }
        for (auto & t : pump_threads) {
            if (t.joinable()) {
                t.join();
            }
        }
        pump_threads.clear();

        // closed only after every thread that could touch them is joined
        std::lock_guard<std::mutex> lock(mutex);
        for (int fd : conn_fds) {
            close(fd);
        }
        conn_fds.clear();
        if (listen_fd >= 0) {
            close(listen_fd);
            listen_fd = -1;
        }
    }

private:
    static bool wait_readable(int fd) {
        pollfd pfd = { fd, POLLIN, 0 };
        return poll(&pfd, 1, 20) > 0 && (pfd.revents & (POLLIN | POLLHUP | POLLERR)) != 0;
    }

    static bool send_all(int fd, const char * data, size_t size) {
        size_t sent = 0;
        while (sent < size) {
            ssize_t n = send(fd, data + sent, size - sent, PROXY_SEND_FLAGS);
            if (n <= 0) {
                return false;
            }
            sent += (size_t) n;
        }
        return true;
    }

    void accept_loop() {
        while (!stopping) {
            if (!wait_readable(listen_fd)) {
                continue;
            }
            int client_fd = accept(listen_fd, nullptr, nullptr);
            if (client_fd < 0) {
                continue;
            }
            no_sigpipe(client_fd);
            int upstream_fd = connect_loopback(upstream);
            if (upstream_fd < 0) {
                close(client_fd);
                continue;
            }
            std::lock_guard<std::mutex> lock(mutex);
            if (stopping) {
                close(client_fd);
                close(upstream_fd);
                return;
            }
            conn_fds.push_back(client_fd);
            conn_fds.push_back(upstream_fd);
            pump_threads.emplace_back(&rpc_proxy::pump, this, client_fd, upstream_fd);
            pump_threads.emplace_back(&rpc_proxy::pump, this, upstream_fd, client_fd);
        }
    }

    void pump(int from, int to) {
        std::vector<char> buf(64 * 1024);
        while (!stopping) {
            if (!wait_readable(from)) {
                continue;
            }
            ssize_t n = recv(from, buf.data(), buf.size(), 0);
            if (n <= 0) {
                break;
            }
            if (!send_all(to, buf.data(), (size_t) n)) {
                break;
            }
        }
        // the RPC server serves one client at a time, so the end of stream has to reach it or
        // it stays parked in recv() and never accepts the next connection
        shutdown(to, SHUT_WR);
    }

    int                      upstream    = 0;
    int                      listen_fd   = -1;
    int                      listen_port = 0;
    std::atomic<bool>        stopping{false};
    std::thread              accept_thread;
    std::vector<std::thread> pump_threads;
    std::mutex               mutex;
    std::vector<int>         conn_fds;
};

static const int64_t N = 32;

struct rpc_client {
    ggml_backend_t             backend = nullptr;
    ggml_backend_buffer_type_t buft    = nullptr;
    ggml_backend_buffer_t      buffer  = nullptr;
    ggml_context *             ctx     = nullptr;
    ggml_tensor *              a       = nullptr;
    ggml_tensor *              b       = nullptr;
    ggml_tensor *              c       = nullptr;
    ggml_cgraph *              graph   = nullptr;

    bool init(const std::string & endpoint) {
        backend = ggml_backend_rpc_init(endpoint.c_str(), 0);
        buft    = ggml_backend_rpc_buffer_type(endpoint.c_str(), 0);
        if (backend == nullptr || buft == nullptr) {
            return false;
        }
        ggml_init_params params = {
            /* .mem_size   = */ ggml_tensor_overhead() * 8 + ggml_graph_overhead(),
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true,
        };
        ctx = ggml_init(params);
        a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, N);
        b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, N);
        c = ggml_add(ctx, a, b);

        buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (buffer == nullptr) {
            return false;
        }
        graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, c);
        return true;
    }

    void upload() const {
        std::vector<float> data(N * N);
        for (size_t i = 0; i < data.size(); i++) {
            data[i] = (float) i;
        }
        ggml_backend_tensor_set(a, data.data(), 0, data.size() * sizeof(float));
        for (size_t i = 0; i < data.size(); i++) {
            data[i] = (float) (2 * i);
        }
        ggml_backend_tensor_set(b, data.data(), 0, data.size() * sizeof(float));
    }

    ggml_status compute() const { return ggml_backend_graph_compute(backend, graph); }

    bool download_and_verify() const {
        std::vector<float> out(N * N, -1.0f);
        ggml_backend_tensor_get(c, out.data(), 0, out.size() * sizeof(float));
        for (size_t i = 0; i < out.size(); i++) {
            if (out[i] != (float) (3 * i)) {
                return false;
            }
        }
        return true;
    }

    void free() {
        if (buffer) {
            ggml_backend_buffer_free(buffer);
            buffer = nullptr;
        }
        if (ctx) {
            ggml_free(ctx);
            ctx = nullptr;
        }
        if (backend) {
            ggml_backend_free(backend);
            backend = nullptr;
        }
    }
};

int main() {
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu_dev == nullptr) {
        printf("no CPU device, skipping\n");
        return 0;
    }

    const int server_port = find_free_port();
    if (server_port < 0) {
        printf("could not reserve a port, skipping\n");
        return 0;
    }
    const std::string server_endpoint = "127.0.0.1:" + std::to_string(server_port);

    // the server loops forever; it is detached and the process outlives it
    std::thread server([cpu_dev, server_endpoint]() {
        ggml_backend_dev_t devs[1] = { cpu_dev };
        ggml_backend_rpc_start_server(server_endpoint.c_str(), nullptr, 2, 1, devs);
    });
    server.detach();

    // wait for the listener to come up
    for (int i = 0; i < 200; i++) {
        int fd = connect_loopback(server_port);
        if (fd >= 0) {
            close(fd);
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Each scenario needs its own proxy, so its own endpoint string: a failed endpoint is
    // latched in a process-wide set that is never cleared.
    rpc_proxy proxy_a;
    if (!proxy_a.start(server_port)) {
        printf("could not start the proxy, skipping\n");
        return 0;
    }

    rpc_client client;
    if (!client.init(proxy_a.endpoint())) {
        printf("could not reach the RPC server, skipping\n");
        return 0;
    }

    // if this fails the harness is broken and nothing below means anything
    begin("control: a healthy endpoint computes correctly");
    client.upload();
    check(client.compute() == GGML_STATUS_SUCCESS, "graph compute succeeds");
    check(client.download_and_verify(), "result is a + b");

    begin("a read from a dead endpoint returns, and does not leave stale data");
    proxy_a.kill();

    // neither the correct answer nor zero, so a read that does nothing is visible
    std::vector<float> out(N * N);
    memset(out.data(), 0xA5, out.size() * sizeof(float));

    // before the fix this aborted the process through RPC_STATUS_ASSERT
    ggml_backend_tensor_get(client.c, out.data(), 0, out.size() * sizeof(float));
    reached("the process survives the read");

    bool zeroed = true;
    for (float v : out) {
        if (v != 0.0f) {
            zeroed = false;
            break;
        }
    }
    check(zeroed, "the destination is zeroed rather than left stale");

    begin("the failed endpoint latches");
    check(client.compute() == GGML_STATUS_FAILED, "graph compute reports GGML_STATUS_FAILED");
    check(client.compute() == GGML_STATUS_FAILED, "and keeps reporting it");

    begin("the other entry points on a dead endpoint fail without aborting");
    size_t free_mem  = 12345;
    size_t total_mem = 12345;
    ggml_backend_rpc_get_device_memory(proxy_a.endpoint().c_str(), 0, &free_mem, &total_mem);
    check(free_mem == 0 && total_mem == 0, "get_device_memory reports zero, not a stale value");
    check(ggml_backend_buft_alloc_buffer(client.buft, 1024) == nullptr, "alloc_buffer returns nullptr");

    begin("freeing a buffer on a dead endpoint does not abort");
    client.free();
    reached("the process survives the teardown");

    // Nothing reads from the socket here, so the send path must notice the failure.
    // GRAPH_COMPUTE has no response, so the first send after the peer closes can still land in
    // the socket buffer and look like a success. A bounded number of tries must reach
    // GGML_STATUS_FAILED, and none of them may take the process down.
    begin("a compute issued straight after the worker dies reports a failure");
    rpc_proxy proxy_b;
    if (!proxy_b.start(server_port)) {
        printf("  could not start the proxy, skipping\n");
    } else {
        rpc_client client_b;
        if (!client_b.init(proxy_b.endpoint())) {
            printf("  could not reach the RPC server, skipping\n");
        } else {
            client_b.upload();
            check(client_b.compute() == GGML_STATUS_SUCCESS, "control: compute succeeds while healthy");

            proxy_b.kill();

            ggml_status status = GGML_STATUS_SUCCESS;
            int         tries  = 0;
            while (status != GGML_STATUS_FAILED && tries < 8) {
                status = client_b.compute();
                tries++;
            }
            check(status == GGML_STATUS_FAILED, "compute reports GGML_STATUS_FAILED within 8 attempts");
            client_b.free();
        }
    }

    // two clients have vanished on the server by now
    begin("the server keeps serving after a client dies mid-operation");
    rpc_proxy proxy_c;
    if (!proxy_c.start(server_port)) {
        printf("  could not start the proxy, skipping\n");
    } else {
        rpc_client client_c;
        check(client_c.init(proxy_c.endpoint()), "a new client connects");
        if (client_c.backend != nullptr && client_c.buffer != nullptr) {
            client_c.upload();
            check(client_c.compute() == GGML_STATUS_SUCCESS, "graph compute succeeds");
            check(client_c.download_and_verify(), "result is a + b");
        }
        client_c.free();
        proxy_c.kill();
    }

    printf("\n%s (%d failure%s)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures, g_failures == 1 ? "" : "s");
    return g_failures == 0 ? 0 : 1;
}
