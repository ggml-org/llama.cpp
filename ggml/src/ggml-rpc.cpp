#include "ggml-rpc.h"
#include "ggml.h"
#include "ggml-backend-impl.h"

#include <cinttypes>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#     define NOMINMAX
#  endif
#  include <windows.h>
#  include <winsock2.h>
#else
#  include <arpa/inet.h>
#  include <sys/socket.h>
#  include <sys/types.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <netdb.h>
#  include <unistd.h>
#endif
#include <string.h>

#define UNUSED GGML_UNUSED

#define GGML_DEBUG 0
#if (GGML_DEBUG >= 1)
#define GGML_PRINT_DEBUG(...) printf(__VA_ARGS__)
#else
#define GGML_PRINT_DEBUG(...)
#endif

#ifdef _WIN32
typedef SOCKET sockfd_t;
using ssize_t = __int64;
#else
typedef int sockfd_t;
#endif

// cross-platform socket
struct socket_t {
    sockfd_t fd;
    socket_t(sockfd_t fd) : fd(fd) {}
    ~socket_t() {
        GGML_PRINT_DEBUG("[%s] closing socket %d\n", __func__, this->fd);
#ifdef _WIN32
        closesocket(this->fd);
#else
        close(this->fd);
#endif
    }
};

// ggml_tensor is serialized into rpc_tensor
#pragma pack(push, 1)
struct rpc_tensor {
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
    char name[GGML_MAX_NAME];

    char padding[4];
};
#pragma pack(pop)

static_assert(sizeof(rpc_tensor) % 8 == 0, "rpc_tensor size must be multiple of 8");

// RPC commands
enum rpc_cmd {
    RPC_CMD_ALLOC_BUFFER = 0,
    RPC_CMD_GET_ALIGNMENT,
    RPC_CMD_GET_MAX_SIZE,
    RPC_CMD_BUFFER_GET_BASE,
    RPC_CMD_FREE_BUFFER,
    RPC_CMD_BUFFER_CLEAR,
    RPC_CMD_SET_TENSOR,
    RPC_CMD_GET_TENSOR,
    RPC_CMD_COPY_TENSOR,
    RPC_CMD_GRAPH_COMPUTE,
    RPC_CMD_GET_DEVICE_MEMORY,
    RPC_CMD_COUNT,
};

// RPC data structures

static ggml_guid_t ggml_backend_rpc_guid() {
    static ggml_guid guid = {0x99, 0x68, 0x5b, 0x6c, 0xd2, 0x83, 0x3d, 0x24, 0x25, 0x36, 0x72, 0xe1, 0x5b, 0x0e, 0x14, 0x03};
    return &guid;
}

struct ggml_backend_rpc_buffer_type_context {
    std::string endpoint;
    std::string name;
    size_t alignment;
    size_t max_size;
};

struct ggml_backend_rpc_context {
    std::string endpoint;
    std::string name;
};

struct ggml_backend_rpc_buffer_context {
    std::shared_ptr<socket_t> sock;
    std::unordered_map<ggml_backend_buffer_t, void *> base_cache;
    uint64_t remote_ptr;
    std::string name;
};

// RPC helper functions

static std::shared_ptr<socket_t> make_socket(sockfd_t fd) {
#ifdef _WIN32
    if (fd == INVALID_SOCKET) {
        return nullptr;
    }
#else
    if (fd < 0) {
        return nullptr;
    }
#endif
    return std::make_shared<socket_t>(fd);
}

static bool set_no_delay(sockfd_t sockfd) {
    int flag = 1;
    // set TCP_NODELAY to disable Nagle's algorithm
    int ret = setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (char *)&flag, sizeof(int));
    return ret == 0;
}

static bool set_reuse_addr(sockfd_t sockfd) {
    int flag = 1;
    int ret = setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (char *)&flag, sizeof(int));
    return ret == 0;
}

static std::shared_ptr<socket_t> socket_connect(const char * host, int port) {
    struct sockaddr_in addr;
    auto sockfd = socket(AF_INET, SOCK_STREAM, 0);
    auto sock_ptr = make_socket(sockfd);
    if (sock_ptr == nullptr) {
        return nullptr;
    }
    if (!set_no_delay(sockfd)) {
        fprintf(stderr, "Failed to set TCP_NODELAY\n");
        return nullptr;
    }
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    struct hostent * server = gethostbyname(host);
    if (server == NULL) {
        fprintf(stderr, "Cannot resolve host '%s'\n", host);
        return nullptr;
    }
    memcpy(&addr.sin_addr.s_addr, server->h_addr, server->h_length);
    if (connect(sock_ptr->fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        return nullptr;
    }
    return sock_ptr;
}

static std::shared_ptr<socket_t> socket_accept(sockfd_t srv_sockfd) {
    auto client_socket_fd = accept(srv_sockfd, NULL, NULL);
    auto client_socket = make_socket(client_socket_fd);
    if (client_socket == nullptr) {
        return nullptr;
    }
    if (!set_no_delay(client_socket_fd)) {
        fprintf(stderr, "Failed to set TCP_NODELAY\n");
        return nullptr;
    }
    return client_socket;
}

static std::shared_ptr<socket_t> create_server_socket(const char * host, int port) {
    auto sockfd = socket(AF_INET, SOCK_STREAM, 0);
    auto sock = make_socket(sockfd);
    if (sock == nullptr) {
        return nullptr;
    }
    if (!set_reuse_addr(sockfd)) {
        fprintf(stderr, "Failed to set SO_REUSEADDR\n");
        return nullptr;
    }
    if (inet_addr(host) == INADDR_NONE) {
        fprintf(stderr, "Invalid host address: %s\n", host);
        return nullptr;
    }
    struct sockaddr_in serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr(host);
    serv_addr.sin_port = htons(port);

    if (bind(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {
        return nullptr;
    }
    if (listen(sockfd, 1) < 0) {
        return nullptr;
    }
    return sock;
}

static bool send_data(sockfd_t sockfd, const void * data, size_t size) {
    size_t bytes_sent = 0;
    while (bytes_sent < size) {
        ssize_t n = send(sockfd, (const char *)data + bytes_sent, size - bytes_sent, 0);
        if (n < 0) {
            return false;
        }
        bytes_sent += n;
    }
    return true;
}

static bool recv_data(sockfd_t sockfd, void * data, size_t size) {
    size_t bytes_recv = 0;
    while (bytes_recv < size) {
        ssize_t n = recv(sockfd, (char *)data + bytes_recv, size - bytes_recv, 0);
        if (n <= 0) {
            return false;
        }
        bytes_recv += n;
    }
    return true;
}

static bool parse_endpoint(const std::string & endpoint, std::string & host, int & port) {
    size_t pos = endpoint.find(':');
    if (pos == std::string::npos) {
        return false;
    }
    host = endpoint.substr(0, pos);
    port = std::stoi(endpoint.substr(pos + 1));
    return true;
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// RPC response: | response_size (8 bytes) | response_data (response_size bytes) |
static bool send_rpc_cmd(const std::shared_ptr<socket_t> & sock, enum rpc_cmd cmd, const std::vector<uint8_t> & input, std::vector<uint8_t> & output) {
    uint8_t cmd_byte = cmd;
    if (!send_data(sock->fd, &cmd_byte, sizeof(cmd_byte))) {
        return false;
    }
    uint64_t input_size = input.size();
    if (!send_data(sock->fd, &input_size, sizeof(input_size))) {
        return false;
    }
    if (!send_data(sock->fd, input.data(), input.size())) {
        return false;
    }
    uint64_t output_size;
    if (!recv_data(sock->fd, &output_size, sizeof(output_size))) {
        return false;
    }
    if (output_size == 0) {
        output.clear();
        return true;
    }
    output.resize(output_size);
    if (!recv_data(sock->fd, output.data(), output_size)) {
        return false;
    }
    return true;
}

// RPC client-side implementation

static std::shared_ptr<socket_t> get_socket(const std::string & endpoint) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    static std::unordered_map<std::string, std::weak_ptr<socket_t>> sockets;
    static bool initialized = false;

    auto it = sockets.find(endpoint);
    if (it != sockets.end()) {
        if (auto sock = it->second.lock()) {
            return sock;
        }
    }
    std::string host;
    int port;
    if (!parse_endpoint(endpoint, host, port)) {
        return nullptr;
    }
#ifdef _WIN32
    if (!initialized) {
        WSADATA wsaData;
        int res = WSAStartup(MAKEWORD(2, 2), &wsaData);
        if (res != 0) {
            return nullptr;
        }
        initialized = true;
    }
#else
    UNUSED(initialized);
#endif
    auto sock = socket_connect(host.c_str(), port);
    if (sock == nullptr) {
        return nullptr;
    }
    GGML_PRINT_DEBUG("[%s] connected to %s, sockfd=%d\n", __func__, endpoint.c_str(), sock->fd);
    sockets[endpoint] = sock;
    return sock;
}

GGML_CALL static const char * ggml_backend_rpc_buffer_get_name(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    return ctx->name.c_str();
}

GGML_CALL static void ggml_backend_rpc_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    // input serialization format: | remote_ptr (8 bytes) |
    std::vector<uint8_t> input(sizeof(uint64_t), 0);
    uint64_t remote_ptr = ctx->remote_ptr;
    memcpy(input.data(), &remote_ptr, sizeof(remote_ptr));
    std::vector<uint8_t> output;
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_FREE_BUFFER, input, output);
    GGML_ASSERT(status);
    GGML_ASSERT(output.empty());
    delete ctx;
}

GGML_CALL static void * ggml_backend_rpc_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    if (ctx->base_cache.find(buffer) != ctx->base_cache.end()) {
        return ctx->base_cache[buffer];
    }
    // input serialization format: | remote_ptr (8 bytes) |
    std::vector<uint8_t> input(sizeof(uint64_t), 0);
    uint64_t remote_ptr = ctx->remote_ptr;
    memcpy(input.data(), &remote_ptr, sizeof(remote_ptr));
    std::vector<uint8_t> output;
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_BUFFER_GET_BASE, input, output);
    GGML_ASSERT(status);
    GGML_ASSERT(output.size() == sizeof(uint64_t));
    // output serialization format: | base_ptr (8 bytes) |
    uint64_t base_ptr;
    memcpy(&base_ptr, output.data(), sizeof(base_ptr));
    void * base = reinterpret_cast<void *>(base_ptr);
    ctx->base_cache[buffer] = base;
    return base;
}

static rpc_tensor serialize_tensor(const ggml_tensor * tensor) {
    rpc_tensor result;
    result.id = reinterpret_cast<uint64_t>(tensor);
    result.type = tensor->type;
    if (tensor->buffer) {
        ggml_backend_buffer_t buffer = tensor->buffer;
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        result.buffer = ctx->remote_ptr;
    } else {
        result.buffer = 0;
    }
    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result.ne[i] = tensor->ne[i];
        result.nb[i] = tensor->nb[i];
    }
    result.op = tensor->op;
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result.op_params[i] = tensor->op_params[i];
    }
    result.flags = tensor->flags;
    for (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        result.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    result.view_src = reinterpret_cast<uint64_t>(tensor->view_src);
    result.view_offs = tensor->view_offs;
    result.data = reinterpret_cast<uint64_t>(tensor->data);
    snprintf(result.name, GGML_MAX_NAME, "%s", tensor->name);
    return result;
}

GGML_CALL static void ggml_backend_rpc_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    UNUSED(buffer);
    if (ggml_is_quantized(tensor->type)) {
        // TODO: this check is due to MATRIX_ROW_PADDING in CUDA and should be generalized
        GGML_ASSERT(tensor->ne[0] % 512 == 0 && "unsupported quantized tensor");
    }
}

GGML_CALL static void ggml_backend_rpc_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    // input serialization format: | rpc_tensor | offset (8 bytes) | data (size bytes) |
    size_t input_size = sizeof(rpc_tensor) + sizeof(uint64_t) + size;
    std::vector<uint8_t> input(input_size, 0);
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    memcpy(input.data(), &rpc_tensor, sizeof(rpc_tensor));
    memcpy(input.data() + sizeof(rpc_tensor), &offset, sizeof(offset));
    memcpy(input.data() + sizeof(rpc_tensor) + sizeof(offset), data, size);
    std::vector<uint8_t> output;
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_SET_TENSOR, input, output);
    GGML_ASSERT(status);
}

GGML_CALL static void ggml_backend_rpc_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    // input serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) |
    int input_size = sizeof(rpc_tensor) + 2*sizeof(uint64_t);
    std::vector<uint8_t> input(input_size, 0);
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    memcpy(input.data(), &rpc_tensor, sizeof(rpc_tensor));
    memcpy(input.data() + sizeof(rpc_tensor), &offset, sizeof(offset));
    memcpy(input.data() + sizeof(rpc_tensor) + sizeof(offset), &size, sizeof(size));
    std::vector<uint8_t> output;
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_GET_TENSOR, input, output);
    GGML_ASSERT(status);
    GGML_ASSERT(output.size() == size);
    // output serialization format: | data (size bytes) |
    memcpy(data, output.data(), size);
}

GGML_CALL static bool ggml_backend_rpc_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    // check if src and dst are on the same server
    ggml_backend_buffer_t src_buffer = src->buffer;
    ggml_backend_rpc_buffer_context * src_ctx = (ggml_backend_rpc_buffer_context *)src_buffer->context;
    ggml_backend_buffer_t dst_buffer = dst->buffer;
    ggml_backend_rpc_buffer_context * dst_ctx = (ggml_backend_rpc_buffer_context *)dst_buffer->context;
    if (src_ctx->sock != dst_ctx->sock) {
        return false;
    }
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    // input serialization format: | rpc_tensor src | rpc_tensor dst |
    int input_size = 2*sizeof(rpc_tensor);
    std::vector<uint8_t> input(input_size, 0);
    rpc_tensor rpc_src = serialize_tensor(src);
    rpc_tensor rpc_dst = serialize_tensor(dst);
    memcpy(input.data(), &rpc_src, sizeof(rpc_src));
    memcpy(input.data() + sizeof(rpc_src), &rpc_dst, sizeof(rpc_dst));
    std::vector<uint8_t> output;
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_COPY_TENSOR, input, output);
    GGML_ASSERT(status);
    // output serialization format: | result (1 byte) |
    GGML_ASSERT(output.size() == 1);
    return output[0];
}

GGML_CALL static void ggml_backend_rpc_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    // serialization format: | bufptr (8 bytes) | value (1 byte) |
    int input_size = sizeof(uint64_t) + sizeof(uint8_t);
    std::vector<uint8_t> input(input_size, 0);
    memcpy(input.data(), &ctx->remote_ptr, sizeof(ctx->remote_ptr));
    memcpy(input.data() + sizeof(ctx->remote_ptr), &value, sizeof(value));
    std::vector<uint8_t> output;
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_BUFFER_CLEAR, input, output);
    GGML_ASSERT(status);
}

static ggml_backend_buffer_i ggml_backend_rpc_buffer_interface = {
    /* .get_name        = */ ggml_backend_rpc_buffer_get_name,
    /* .free_buffer     = */ ggml_backend_rpc_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_rpc_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_rpc_buffer_init_tensor,
    /* .set_tensor      = */ ggml_backend_rpc_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_rpc_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_rpc_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_rpc_buffer_clear,
    /* .reset           = */ NULL,
};

GGML_CALL static const char * ggml_backend_rpc_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->name.c_str();
}

GGML_CALL static ggml_backend_buffer_t ggml_backend_rpc_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    // input serialization format: | size (8 bytes) |
    int input_size = sizeof(uint64_t);
    std::vector<uint8_t> input(input_size, 0);
    memcpy(input.data(), &size, sizeof(size));
    std::vector<uint8_t> output;
    auto sock = get_socket(buft_ctx->endpoint);
    bool status = send_rpc_cmd(sock, RPC_CMD_ALLOC_BUFFER, input, output);
    GGML_ASSERT(status);
    GGML_ASSERT(output.size() == 2*sizeof(uint64_t));
    // output serialization format: | remote_ptr (8 bytes) | remote_size (8 bytes) |
    uint64_t remote_ptr;
    memcpy(&remote_ptr, output.data(), sizeof(remote_ptr));
    size_t remote_size;
    memcpy(&remote_size, output.data() + sizeof(uint64_t), sizeof(remote_size));
    if (remote_ptr != 0) {
        ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft,
            ggml_backend_rpc_buffer_interface,
            new ggml_backend_rpc_buffer_context{sock, {}, remote_ptr, "RPC[" + std::string(buft_ctx->endpoint) + "]"},
            remote_size);
        return buffer;
    } else {
        return nullptr;
    }
}

static size_t get_alignment(const std::shared_ptr<socket_t> & sock) {
    // input serialization format: | 0 bytes |
    std::vector<uint8_t> input;
    std::vector<uint8_t> output;
    bool status = send_rpc_cmd(sock, RPC_CMD_GET_ALIGNMENT, input, output);
    GGML_ASSERT(status);
    GGML_ASSERT(output.size() == sizeof(uint64_t));
    // output serialization format: | alignment (8 bytes) |
    uint64_t alignment;
    memcpy(&alignment, output.data(), sizeof(alignment));
    return alignment;
}

GGML_CALL static size_t ggml_backend_rpc_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->alignment;
}

static size_t get_max_size(const std::shared_ptr<socket_t> & sock) {
    // input serialization format: | 0 bytes |
    std::vector<uint8_t> input;
    std::vector<uint8_t> output;
    bool status = send_rpc_cmd(sock, RPC_CMD_GET_MAX_SIZE, input, output);
    GGML_ASSERT(status);
    GGML_ASSERT(output.size() == sizeof(uint64_t));
    // output serialization format: | max_size (8 bytes) |
    uint64_t max_size;
    memcpy(&max_size, output.data(), sizeof(max_size));
    return max_size;
}

GGML_CALL static size_t ggml_backend_rpc_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->max_size;
}

GGML_CALL static size_t ggml_backend_rpc_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    UNUSED(buft);
    return ggml_nbytes(tensor);
}

static ggml_backend_buffer_type_i ggml_backend_rpc_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_rpc_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_rpc_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_rpc_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_rpc_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_rpc_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

GGML_CALL static const char * ggml_backend_rpc_name(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;

    return rpc_ctx->name.c_str();
}

GGML_CALL static void ggml_backend_rpc_free(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    delete rpc_ctx;
    delete backend;
}

GGML_CALL static ggml_backend_buffer_type_t ggml_backend_rpc_get_default_buffer_type(ggml_backend_t backend) {
    ggml_backend_rpc_context * ctx = (ggml_backend_rpc_context *)backend->context;
    return ggml_backend_rpc_buffer_type(ctx->endpoint.c_str());
}

GGML_CALL static void ggml_backend_rpc_synchronize(ggml_backend_t backend) {
    UNUSED(backend);
    // this is no-op because we don't have any async operations
}

static void add_tensor(ggml_tensor * tensor, std::vector<rpc_tensor> & tensors, std::unordered_set<ggml_tensor*> & visited) {
    if (tensor == nullptr) {
        return;
    }
    if (visited.find(tensor) != visited.end()) {
        return;
    }
    visited.insert(tensor);
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        add_tensor(tensor->src[i], tensors, visited);
    }
    add_tensor(tensor->view_src, tensors, visited);
    tensors.push_back(serialize_tensor(tensor));
}

static void serialize_graph(const ggml_cgraph * cgraph, std::vector<uint8_t> & output) {
    uint32_t n_nodes = cgraph->n_nodes;
    std::vector<rpc_tensor> tensors;
    std::unordered_set<ggml_tensor*> visited;
    for (uint32_t i = 0; i < n_nodes; i++) {
        add_tensor(cgraph->nodes[i], tensors, visited);
    }
    // serialization format:
    // | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    int output_size = sizeof(uint32_t) + n_nodes * sizeof(uint64_t) + sizeof(uint32_t) + n_tensors * sizeof(rpc_tensor);
    output.resize(output_size, 0);
    memcpy(output.data(), &n_nodes, sizeof(n_nodes));
    for (uint32_t i = 0; i < n_nodes; i++) {
        memcpy(output.data() + sizeof(n_nodes) + i * sizeof(uint64_t), &cgraph->nodes[i], sizeof(uint64_t));
    }
    uint32_t * out_ntensors = (uint32_t *)(output.data() + sizeof(n_nodes) + n_nodes * sizeof(uint64_t));
    *out_ntensors = n_tensors;
    rpc_tensor * out_tensors = (rpc_tensor *)(output.data() + sizeof(n_nodes) + n_nodes * sizeof(uint64_t) + sizeof(uint32_t));
    memcpy(out_tensors, tensors.data(), n_tensors * sizeof(rpc_tensor));
}

GGML_CALL static enum ggml_status ggml_backend_rpc_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    std::vector<uint8_t> input;
    serialize_graph(cgraph, input);
    std::vector<uint8_t> output;
    auto sock = get_socket(rpc_ctx->endpoint);
    bool status = send_rpc_cmd(sock, RPC_CMD_GRAPH_COMPUTE, input, output);
    GGML_ASSERT(status);
    GGML_ASSERT(output.size() == 1);
    return (enum ggml_status)output[0];
}

GGML_CALL static bool ggml_backend_rpc_supports_op(ggml_backend_t backend, const ggml_tensor * op) {
    UNUSED(backend);
    UNUSED(op);
    //TODO: call the remote backend and cache the results
    return true;
}

GGML_CALL static bool ggml_backend_rpc_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    if (!buft || buft->iface.get_name != ggml_backend_rpc_buffer_type_name) {
        return false;
    }
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    return buft_ctx->endpoint == rpc_ctx->endpoint;
}

static ggml_backend_i ggml_backend_rpc_interface = {
    /* .get_name                = */ ggml_backend_rpc_name,
    /* .free                    = */ ggml_backend_rpc_free,
    /* .get_default_buffer_type = */ ggml_backend_rpc_get_default_buffer_type,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ggml_backend_rpc_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_rpc_graph_compute,
    /* .supports_op             = */ ggml_backend_rpc_supports_op,
    /* .supports_buft           = */ ggml_backend_rpc_supports_buft,
    /* .offload_op              = */ NULL,
    /* .event_new               = */ NULL,
    /* .event_free              = */ NULL,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .event_synchronize       = */ NULL,
};

GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    // NOTE: buffer types are allocated and never freed; this is by design
    static std::unordered_map<std::string, ggml_backend_buffer_type_t> buft_map;
    auto it = buft_map.find(endpoint);
    if (it != buft_map.end()) {
        return it->second;
    }
    auto sock = get_socket(endpoint);
    if (sock == nullptr) {
        fprintf(stderr, "Failed to connect to %s\n", endpoint);
        return nullptr;
    }
    size_t alignment = get_alignment(sock);
    size_t max_size = get_max_size(sock);
    ggml_backend_rpc_buffer_type_context * buft_ctx = new ggml_backend_rpc_buffer_type_context {
        /* .endpoint  = */ endpoint,
        /* .name      = */ "RPC[" + std::string(endpoint) + "]",
        /* .alignment = */ alignment,
        /* .max_size  = */ max_size
    };

    ggml_backend_buffer_type_t buft = new ggml_backend_buffer_type {
        /* .iface   = */ ggml_backend_rpc_buffer_type_interface,
        /* .context = */ buft_ctx
    };
    buft_map[endpoint] = buft;
    return buft;
}

GGML_CALL ggml_backend_t ggml_backend_rpc_init(const char * endpoint) {
    ggml_backend_rpc_context * ctx = new ggml_backend_rpc_context {
        /* .endpoint  = */ endpoint,
        /* .name      = */ "RPC[" + std::string(endpoint) + "]",
    };

    ggml_backend_t backend = new ggml_backend {
        /* .guid      = */ ggml_backend_rpc_guid(),
        /* .interface = */ ggml_backend_rpc_interface,
        /* .context   = */ ctx
    };
    return backend;
}

GGML_API GGML_CALL bool ggml_backend_is_rpc(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_rpc_guid());
}

static void get_device_memory(const std::shared_ptr<socket_t> & sock, size_t * free, size_t * total) {
    // input serialization format: | 0 bytes |
    std::vector<uint8_t> input;
    std::vector<uint8_t> output;
    bool status = send_rpc_cmd(sock, RPC_CMD_GET_DEVICE_MEMORY, input, output);
    GGML_ASSERT(status);
    GGML_ASSERT(output.size() == 2*sizeof(uint64_t));
    // output serialization format: | free (8 bytes) | total (8 bytes) |
    uint64_t free_mem;
    memcpy(&free_mem, output.data(), sizeof(free_mem));
    uint64_t total_mem;
    memcpy(&total_mem, output.data() + sizeof(uint64_t), sizeof(total_mem));
    *free = free_mem;
    *total = total_mem;
}

GGML_API GGML_CALL void ggml_backend_rpc_get_device_memory(const char * endpoint, size_t * free, size_t * total) {
    auto sock = get_socket(endpoint);
    if (sock == nullptr) {
        *free = 0;
        *total = 0;
        return;
    }
    get_device_memory(sock, free, total);
}

// RPC server-side implementation

class rpc_server {
public:
    rpc_server(ggml_backend_t backend) : backend(backend) {}
    ~rpc_server();

    bool alloc_buffer(const std::vector<uint8_t> & input, std::vector<uint8_t> & output);
    void get_alignment(std::vector<uint8_t> & output);
    void get_max_size(std::vector<uint8_t> & output);
    bool buffer_get_base(const std::vector<uint8_t> & input, std::vector<uint8_t> & output);
    bool free_buffer(const std::vector<uint8_t> & input);
    bool buffer_clear(const std::vector<uint8_t> & input);
    bool set_tensor(const std::vector<uint8_t> & input);
    bool get_tensor(const std::vector<uint8_t> & input, std::vector<uint8_t> & output);
    bool copy_tensor(const std::vector<uint8_t> & input, std::vector<uint8_t> & output);
    bool graph_compute(const std::vector<uint8_t> & input, std::vector<uint8_t> & output);

private:
    ggml_tensor * deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor);
    ggml_tensor * create_node(uint64_t id,
                              struct ggml_context * ctx,
                              const std::unordered_map<uint64_t, const rpc_tensor*> & tensor_ptrs,
                              std::unordered_map<uint64_t, struct ggml_tensor*> & tensor_map);


    ggml_backend_t backend;
    std::unordered_set<ggml_backend_buffer_t> buffers;
};

bool rpc_server::alloc_buffer(const std::vector<uint8_t> & input, std::vector<uint8_t> & output) {
    // input serialization format: | size (8 bytes) |
    if (input.size() != sizeof(uint64_t)) {
        return false;
    }
    uint64_t size;
    memcpy(&size, input.data(), sizeof(size));
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, size);
    uint64_t remote_ptr = 0;
    uint64_t remote_size = 0;
    if (buffer != nullptr) {
        remote_ptr = reinterpret_cast<uint64_t>(buffer);
        remote_size = buffer->size;
        GGML_PRINT_DEBUG("[%s] size: %" PRIu64 " -> remote_ptr: %" PRIx64 ", remote_size: %" PRIu64 "\n", __func__, size, remote_ptr, remote_size);
        buffers.insert(buffer);
    } else {
        GGML_PRINT_DEBUG("[%s] size: %" PRIu64 " -> failed\n", __func__, size);
    }
    // output serialization format: | remote_ptr (8 bytes) | remote_size (8 bytes) |
    output.resize(2*sizeof(uint64_t), 0);
    memcpy(output.data(), &remote_ptr, sizeof(remote_ptr));
    memcpy(output.data() + sizeof(uint64_t), &remote_size, sizeof(remote_size));
    return true;
}

void rpc_server::get_alignment(std::vector<uint8_t> & output) {
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    size_t alignment = ggml_backend_buft_get_alignment(buft);
    GGML_PRINT_DEBUG("[%s] alignment: %lu\n", __func__, alignment);
    // output serialization format: | alignment (8 bytes) |
    output.resize(sizeof(uint64_t), 0);
    memcpy(output.data(), &alignment, sizeof(alignment));
}

void rpc_server::get_max_size(std::vector<uint8_t> & output) {
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    size_t max_size = ggml_backend_buft_get_max_size(buft);
    GGML_PRINT_DEBUG("[%s] max_size: %lu\n", __func__, max_size);
    // output serialization format: | max_size (8 bytes) |
    output.resize(sizeof(uint64_t), 0);
    memcpy(output.data(), &max_size, sizeof(max_size));
}

bool rpc_server::buffer_get_base(const std::vector<uint8_t> & input, std::vector<uint8_t> & output) {
    // input serialization format: | remote_ptr (8 bytes) |
    if (input.size() != sizeof(uint64_t)) {
        return false;
    }
    uint64_t remote_ptr;
    memcpy(&remote_ptr, input.data(), sizeof(remote_ptr));
    GGML_PRINT_DEBUG("[%s] remote_ptr: %" PRIx64 "\n", __func__, remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_PRINT_DEBUG("[%s] buffer not found\n", __func__);
        return false;
    }
    void * base = ggml_backend_buffer_get_base(buffer);
    // output serialization format: | base_ptr (8 bytes) |
    uint64_t base_ptr = reinterpret_cast<uint64_t>(base);
    output.resize(sizeof(uint64_t), 0);
    memcpy(output.data(), &base_ptr, sizeof(base_ptr));
    return true;
}

bool rpc_server::free_buffer(const std::vector<uint8_t> & input) {
    // input serialization format: | remote_ptr (8 bytes) |
    if (input.size() != sizeof(uint64_t)) {
        return false;
    }
    uint64_t remote_ptr;
    memcpy(&remote_ptr, input.data(), sizeof(remote_ptr));
    GGML_PRINT_DEBUG("[%s] remote_ptr: %" PRIx64 "\n", __func__, remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_PRINT_DEBUG("[%s] buffer not found\n", __func__);
        return false;
    }
    ggml_backend_buffer_free(buffer);
    buffers.erase(buffer);
    return true;
}

bool rpc_server::buffer_clear(const std::vector<uint8_t> & input) {
    // input serialization format: | remote_ptr (8 bytes) | value (1 byte) |
    if (input.size() != sizeof(uint64_t) + sizeof(uint8_t)) {
        return false;
    }
    uint64_t remote_ptr;
    memcpy(&remote_ptr, input.data(), sizeof(remote_ptr));
    uint8_t value;
    memcpy(&value, input.data() + sizeof(uint64_t), sizeof(value));
    GGML_PRINT_DEBUG("[%s] remote_ptr: %" PRIx64 ", value: %u\n", __func__, remote_ptr, value);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_PRINT_DEBUG("[%s] buffer not found\n", __func__);
        return false;
    }
    ggml_backend_buffer_clear(buffer, value);
    return true;
}

ggml_tensor * rpc_server::deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor) {
    ggml_tensor * result = ggml_new_tensor_4d(ctx, (ggml_type) tensor->type,
        tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = tensor->nb[i];
    }
    result->buffer = reinterpret_cast<ggml_backend_buffer_t>(tensor->buffer);
    if (result->buffer && buffers.find(result->buffer) == buffers.end()) {
        return nullptr;
    }

    // require that the tensor data does not go beyond the buffer end
    uint64_t tensor_size = (uint64_t) ggml_nbytes(result);
    uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(result->buffer);
    uint64_t buffer_size = (uint64_t) ggml_backend_buffer_get_size(result->buffer);
    GGML_ASSERT(tensor->data + tensor_size >= tensor->data); // check for overflow
    GGML_ASSERT(tensor->data >= buffer_start && tensor->data + tensor_size <= buffer_start + buffer_size);

    result->op = (ggml_op) tensor->op;
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result->op_params[i] = tensor->op_params[i];
    }
    result->flags = tensor->flags;
    result->data = reinterpret_cast<void *>(tensor->data);
    ggml_set_name(result, tensor->name);
    return result;
}


bool rpc_server::set_tensor(const std::vector<uint8_t> & input) {
    // serialization format: | rpc_tensor | offset (8 bytes) | data (size bytes) |
    if (input.size() < sizeof(rpc_tensor) + sizeof(uint64_t)) {
        return false;
    }
    const rpc_tensor * in_tensor = (const rpc_tensor *)input.data();
    uint64_t offset;
    memcpy(&offset, input.data() + sizeof(rpc_tensor), sizeof(offset));
    const size_t size = input.size() - sizeof(rpc_tensor) - sizeof(offset);

    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    if (tensor == nullptr) {
        GGML_PRINT_DEBUG("[%s] error deserializing tensor\n", __func__);
        ggml_free(ctx);
        return false;
    }
    GGML_PRINT_DEBUG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %zu\n", __func__, (void*)tensor->buffer, tensor->data, offset, size);

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (in_tensor->data + offset < p0 || in_tensor->data + offset >= p1 || size > (p1 - in_tensor->data - offset)) {
            GGML_ABORT("[%s] tensor->data out of bounds\n", __func__);
        }
    }

    const void * data = input.data() + sizeof(rpc_tensor) + sizeof(offset);
    ggml_backend_tensor_set(tensor, data, offset, size);
    ggml_free(ctx);
    return true;
}

bool rpc_server::get_tensor(const std::vector<uint8_t> & input, std::vector<uint8_t> & output) {
    // serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) |
    if (input.size() != sizeof(rpc_tensor) + 2*sizeof(uint64_t)) {
        return false;
    }
    const rpc_tensor * in_tensor = (const rpc_tensor *)input.data();
    uint64_t offset;
    memcpy(&offset, input.data() + sizeof(rpc_tensor), sizeof(offset));
    uint64_t size;
    memcpy(&size, input.data() + sizeof(rpc_tensor) + sizeof(offset), sizeof(size));

    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    if (tensor == nullptr) {
        GGML_PRINT_DEBUG("[%s] error deserializing tensor\n", __func__);
        ggml_free(ctx);
        return false;
    }
    GGML_PRINT_DEBUG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %" PRIu64 "\n", __func__, (void*)tensor->buffer, tensor->data, offset, size);

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (in_tensor->data + offset < p0 || in_tensor->data + offset >= p1 || size > (p1 - in_tensor->data - offset)) {
            GGML_ABORT("[%s] tensor->data out of bounds\n", __func__);
        }
    }

    // output serialization format: | data (size bytes) |
    output.resize(size, 0);
    ggml_backend_tensor_get(tensor, output.data(), offset, size);
    ggml_free(ctx);
    return true;
}

bool rpc_server::copy_tensor(const std::vector<uint8_t> & input, std::vector<uint8_t> & output) {
    // serialization format: | rpc_tensor src | rpc_tensor dst |
    if (input.size() != 2*sizeof(rpc_tensor)) {
        return false;
    }
    const rpc_tensor * rpc_src = (const rpc_tensor *)input.data();
    const rpc_tensor * rpc_dst = (const rpc_tensor *)(input.data() + sizeof(rpc_src));

    struct ggml_init_params params {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    ggml_tensor * src = deserialize_tensor(ctx, rpc_src);
    ggml_tensor * dst = deserialize_tensor(ctx, rpc_dst);
    if (src == nullptr || dst == nullptr) {
        GGML_PRINT_DEBUG("[%s] error deserializing tensors\n", __func__);
        ggml_free(ctx);
        return false;
    }
    GGML_PRINT_DEBUG("[%s] src->buffer: %p, dst->buffer: %p\n", __func__, (void*)src->buffer, (void*)dst->buffer);
    bool result = ggml_backend_buffer_copy_tensor(src, dst);
    // output serialization format: | result (1 byte) |
    output.resize(1, 0);
    output[0] = result;
    ggml_free(ctx);
    return true;
}

ggml_tensor * rpc_server::create_node(uint64_t id,
                                      struct ggml_context * ctx,
                                      const std::unordered_map<uint64_t, const rpc_tensor*> & tensor_ptrs,
                                      std::unordered_map<uint64_t, struct ggml_tensor*> & tensor_map) {
    if (id == 0) {
        return nullptr;
    }
    if (tensor_map.find(id) != tensor_map.end()) {
        return tensor_map[id];
    }
    const rpc_tensor * tensor = tensor_ptrs.at(id);
    struct ggml_tensor * result = deserialize_tensor(ctx, tensor);
    if (result == nullptr) {
        return nullptr;
    }
    tensor_map[id] = result;
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        result->src[i] = create_node(tensor->src[i], ctx, tensor_ptrs, tensor_map);
    }
    result->view_src = create_node(tensor->view_src, ctx, tensor_ptrs, tensor_map);
    result->view_offs = tensor->view_offs;
    return result;
}

bool rpc_server::graph_compute(const std::vector<uint8_t> & input, std::vector<uint8_t> & output) {
    // serialization format:
    // | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    if (input.size() < sizeof(uint32_t)) {
        return false;
    }
    uint32_t n_nodes;
    memcpy(&n_nodes, input.data(), sizeof(n_nodes));
    if (input.size() < sizeof(uint32_t) + n_nodes*sizeof(uint64_t) + sizeof(uint32_t)) {
        return false;
    }
    const uint64_t * nodes = (const uint64_t *)(input.data() + sizeof(n_nodes));
    uint32_t n_tensors;
    memcpy(&n_tensors, input.data() + sizeof(n_nodes) + n_nodes*sizeof(uint64_t), sizeof(n_tensors));
    if (input.size() < sizeof(uint32_t) + n_nodes*sizeof(uint64_t) + sizeof(uint32_t) + n_tensors*sizeof(rpc_tensor)) {
        return false;
    }
    const rpc_tensor * tensors = (const rpc_tensor *)(input.data() + sizeof(n_nodes) + n_nodes*sizeof(uint64_t) + sizeof(n_tensors));
    GGML_PRINT_DEBUG("[%s] n_nodes: %u, n_tensors: %u\n", __func__, n_nodes, n_tensors);

    static size_t buf_size = ggml_tensor_overhead()*(n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, false);
    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, n_nodes, false);
    graph->n_nodes = n_nodes;
    std::unordered_map<uint64_t, const rpc_tensor*> tensor_ptrs;
    for (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs[tensors[i].id] = &tensors[i];
    }
    std::unordered_map<uint64_t, ggml_tensor*> tensor_map;
    for (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        memcpy(&id, &nodes[i], sizeof(id));
        graph->nodes[i] = create_node(id, ctx, tensor_ptrs, tensor_map);
    }
    ggml_status status = ggml_backend_graph_compute(backend, graph);
    // output serialization format: | status (1 byte) |
    output.resize(1, 0);
    output[0] = status;
    ggml_free(ctx);
    return true;
}

rpc_server::~rpc_server() {
    for (auto buffer : buffers) {
        ggml_backend_buffer_free(buffer);
    }
}

static void rpc_serve_client(ggml_backend_t backend, sockfd_t sockfd, size_t free_mem, size_t total_mem) {
    rpc_server server(backend);
    while (true) {
        uint8_t cmd;
        if (!recv_data(sockfd, &cmd, 1)) {
            break;
        }
        if (cmd >= RPC_CMD_COUNT) {
            // fail fast if the command is invalid
            fprintf(stderr, "Unknown command: %d\n", cmd);
            break;
        }
        std::vector<uint8_t> input;
        std::vector<uint8_t> output;
        uint64_t input_size;
        if (!recv_data(sockfd, &input_size, sizeof(input_size))) {
            break;
        }
        try {
            input.resize(input_size);
        } catch (const std::bad_alloc & e) {
            fprintf(stderr, "Failed to allocate input buffer of size %" PRIu64 "\n", input_size);
            break;
        }
        if (!recv_data(sockfd, input.data(), input_size)) {
            break;
        }
        bool ok = true;
        switch (cmd) {
            case RPC_CMD_ALLOC_BUFFER: {
                ok = server.alloc_buffer(input, output);
                break;
            }
            case RPC_CMD_GET_ALIGNMENT: {
                server.get_alignment(output);
                break;
            }
            case RPC_CMD_GET_MAX_SIZE: {
                server.get_max_size(output);
                break;
            }
            case RPC_CMD_BUFFER_GET_BASE: {
                ok = server.buffer_get_base(input, output);
                break;
            }
            case RPC_CMD_FREE_BUFFER: {
                ok = server.free_buffer(input);
                break;
            }
            case RPC_CMD_BUFFER_CLEAR: {
                ok = server.buffer_clear(input);
                break;
            }
            case RPC_CMD_SET_TENSOR: {
                ok = server.set_tensor(input);
                break;
            }
            case RPC_CMD_GET_TENSOR: {
                ok = server.get_tensor(input, output);
                break;
            }
            case RPC_CMD_COPY_TENSOR: {
                ok = server.copy_tensor(input, output);
                break;
            }
            case RPC_CMD_GRAPH_COMPUTE: {
                ok = server.graph_compute(input, output);
                break;
            }
            case RPC_CMD_GET_DEVICE_MEMORY: {
                // output serialization format: | free (8 bytes) | total (8 bytes) |
                output.resize(2*sizeof(uint64_t), 0);
                memcpy(output.data(), &free_mem, sizeof(free_mem));
                memcpy(output.data() + sizeof(uint64_t), &total_mem, sizeof(total_mem));
                break;
            }
            default: {
                fprintf(stderr, "Unknown command: %d\n", cmd);
                ok = false;
            }
        }
        if (!ok) {
            break;
        }
        uint64_t output_size = output.size();
        if (!send_data(sockfd, &output_size, sizeof(output_size))) {
            break;
        }
        if (!send_data(sockfd, output.data(), output_size)) {
            break;
        }
    }
}

void start_rpc_server(ggml_backend_t backend, const char * endpoint, size_t free_mem, size_t total_mem) {
    std::string host;
    int port;
    if (!parse_endpoint(endpoint, host, port)) {
        return;
    }
#ifdef _WIN32
    {
        WSADATA wsaData;
        int res = WSAStartup(MAKEWORD(2, 2), &wsaData);
        if (res != 0) {
            fprintf(stderr, "WSAStartup failed: %d\n", res);
            return;
        }
    }
#endif
    auto server_socket = create_server_socket(host.c_str(), port);
    if (server_socket == nullptr) {
        fprintf(stderr, "Failed to create server socket\n");
        return;
    }
    while (true) {
        auto client_socket = socket_accept(server_socket->fd);
        if (client_socket == nullptr) {
            fprintf(stderr, "Failed to accept client connection\n");
            return;
        }
        printf("Accepted client connection, free_mem=%zu, total_mem=%zu\n", free_mem, total_mem);
        fflush(stdout);
        rpc_serve_client(backend, client_socket->fd, free_mem, total_mem);
        printf("Client connection closed\n");
        fflush(stdout);
    }
#ifdef _WIN32
    WSACleanup();
#endif
}
