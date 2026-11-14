#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml-rpc.h"
#include "ggml.h"

#include <cstdlib>
#include <cstring>
#include <string>

#ifdef _WIN32
#    include <winsock2.h>
#    include <ws2tcpip.h>
typedef SOCKET sockfd_t;
#else
#    include <arpa/inet.h>
#    include <netdb.h>
#    include <sys/socket.h>
#    include <sys/time.h>
#    include <unistd.h>
typedef int sockfd_t;
#endif

// wire structs mirrored from ggml/src/ggml-rpc/ggml-rpc.cpp (packed)
#pragma pack(push, 1)
struct rpc_tensor_probe {
    uint64_t id;
    uint32_t type;
    uint64_t buffer;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
    char     name[GGML_MAX_NAME];
    int32_t  use_count;
};

struct rpc_msg_get_alloc_size_req_probe {
    uint32_t          device;
    rpc_tensor_probe  tensor;
    rpc_tensor_probe  srcs[GGML_MAX_SRC];
};
#pragma pack(pop)

static const uint8_t RPC_CMD_GET_ALLOC_SIZE = 13;
static const uint8_t RPC_CMD_HELLO          = 14;

static void probe_close(sockfd_t fd) {
#ifdef _WIN32
    closesocket(fd);
#else
    close(fd);
#endif
}

static bool probe_connect(const char * host, int port, sockfd_t * fd_out) {
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(port);
    GGML_ASSERT(inet_pton(AF_INET, host, &addr.sin_addr) == 1);

    sockfd_t fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd == (sockfd_t) -1) {
        return false;
    }
    struct timeval tv = { 5, 0 };
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, (const char *) &tv, sizeof(tv));
    if (connect(fd, (struct sockaddr *) &addr, sizeof(addr)) != 0) {
        probe_close(fd);
        return false;
    }
    *fd_out = fd;
    return true;
}

static bool probe_send_all(sockfd_t fd, const void * buf, size_t len) {
    const char * p = (const char *) buf;
    while (len > 0) {
        const auto n = send(fd, p, (int) len, 0);
        if (n <= 0) {
            return false;
        }
        p += n;
        len -= n;
    }
    return true;
}

static bool probe_recv_all(sockfd_t fd, void * buf, size_t len) {
    char * p = (char *) buf;
    while (len > 0) {
        const auto n = recv(fd, p, (int) len, 0);
        if (n <= 0) {
            return false;
        }
        p += n;
        len -= n;
    }
    return true;
}

static bool probe_hello(sockfd_t fd) {
    const uint8_t cmd = RPC_CMD_HELLO;
    const uint64_t req_size = 24;
    uint8_t caps[24] = {};
    uint8_t rsp[28];
    uint64_t rsp_size = 0;
    return probe_send_all(fd, &cmd, sizeof(cmd)) &&
           probe_send_all(fd, &req_size, sizeof(req_size)) &&
           probe_send_all(fd, caps, sizeof(caps)) &&
           probe_recv_all(fd, &rsp_size, sizeof(rsp_size)) &&
           rsp_size == sizeof(rsp) &&
           probe_recv_all(fd, rsp, sizeof(rsp));
}

// a malformed tensor must be rejected, not kill the whole server
// ref: https://github.com/ggml-org/llama.cpp/issues/30458
static void test_malformed_tensor_rejected(const char * endpoint) {
    const char * colon = strrchr(endpoint, ':');
    GGML_ASSERT(colon != nullptr);
    const std::string host(endpoint, colon);
    const int port = atoi(colon + 1);

    struct malformed_case {
        const char * what;
        uint32_t     type;
        uint32_t     ne[GGML_MAX_DIMS];
        uint32_t     op;
    };
    const malformed_case cases[] = {
        // element count overflows int64_t
        { "ne overflow",      GGML_TYPE_F32,  { 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF }, 0 },
        // ne[0] not a multiple of the Q4_0 block size (32)
        { "misaligned ne0",   GGML_TYPE_Q4_0, { 33, 1, 1, 1 }, 0 },
        // op out of range
        { "invalid op",       GGML_TYPE_F32,  { 4, 1, 1, 1 }, 0xFFFFFF },
    };

    for (const auto & tcase : cases) {
        sockfd_t fd;
        GGML_ASSERT(probe_connect(host.c_str(), port, &fd));
        GGML_ASSERT(probe_hello(fd));

        rpc_msg_get_alloc_size_req_probe req = {};
        req.tensor.type = tcase.type;
        req.tensor.op   = tcase.op;
        for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
            req.tensor.ne[i] = tcase.ne[i];
        }

        const uint8_t cmd = RPC_CMD_GET_ALLOC_SIZE;
        const uint64_t req_size = sizeof(req);
        GGML_ASSERT(probe_send_all(fd, &cmd, sizeof(cmd)));
        GGML_ASSERT(probe_send_all(fd, &req_size, sizeof(req_size)));
        GGML_ASSERT(probe_send_all(fd, &req, sizeof(req)));

        // no response: the server must close the connection
        char byte;
        const auto n = recv(fd, &byte, 1, 0);
        GGML_ASSERT(n <= 0);
        probe_close(fd);

        // the server must still serve new connections
        sockfd_t fd2;
        GGML_ASSERT(probe_connect(host.c_str(), port, &fd2));
        GGML_ASSERT(probe_hello(fd2));
        probe_close(fd2);
    }
}

int main(int argc, char ** argv) {
    GGML_ASSERT(argc == 3);
    ggml_backend_load_all();

#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#endif

    const char * endpoint_a = argv[1];
    const char * endpoint_b = argv[2];

    // must run before ggml_backend_rpc_init(): the dispatcher connections
    // below would block the single-threaded server
    test_malformed_tensor_rejected(endpoint_a);

    ggml_backend_t backend_a = ggml_backend_rpc_init(endpoint_a, 0);
    ggml_backend_t backend_b = ggml_backend_rpc_init(endpoint_b, 0);
    GGML_ASSERT(backend_a != nullptr);
    GGML_ASSERT(backend_b != nullptr);

    ggml_init_params params = {
        /* .mem_size   = */ 3*ggml_tensor_overhead() + ggml_graph_overhead_custom(1, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);

    ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend_a);
    GGML_ASSERT(buffer != nullptr);

    // A remote pointer allocated by server A is not meaningful to server B.
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 1, false);
    graph->nodes[0] = tensor;
    graph->n_nodes = 1;

    GGML_ASSERT(ggml_backend_graph_compute(backend_b, graph) == GGML_STATUS_SUCCESS);
    // Wait for server B to finish the graph before the script checks its log.
    size_t free_mem;
    size_t total_mem;
    ggml_backend_rpc_get_device_memory(endpoint_b, 0, &free_mem, &total_mem);
    GGML_ASSERT(total_mem > 0);
    ggml_backend_buffer_free(buffer);

    // Two tensors with the same ne[] but different nb[] must not share a cached alloc size.
    // ref: https://github.com/ggml-org/llama.cpp/issues/28360
    ggml_backend_buffer_type_t buft = ggml_backend_rpc_buffer_type(endpoint_a, 0);
    GGML_ASSERT(buft != nullptr);

    // MUL_MAT may need extra memory, so the size is read from the server [TAG_ALLOC_SIZE_EXPAND]
    ggml_tensor * packed = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
    packed->op = GGML_OP_MUL_MAT;

    // same ne[], twice the row stride
    ggml_tensor * strided = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
    strided->op = GGML_OP_MUL_MAT;
    strided->nb[1] = 2*strided->nb[1];
    strided->nb[2] = strided->ne[1]*strided->nb[1];
    strided->nb[3] = strided->nb[2];
    GGML_ASSERT(ggml_nbytes(strided) > ggml_nbytes(packed));

    // ask for the packed tensor first, so a cache keyed without nb[] holds the smaller size
    GGML_ASSERT(ggml_backend_buft_get_alloc_size(buft, packed)  >= ggml_nbytes(packed));
    GGML_ASSERT(ggml_backend_buft_get_alloc_size(buft, strided) >= ggml_nbytes(strided));

    ggml_free(ctx);
    ggml_backend_free(backend_b);
    ggml_backend_free(backend_a);

#ifdef _WIN32
    WSACleanup();
#endif
    return 0;
}
