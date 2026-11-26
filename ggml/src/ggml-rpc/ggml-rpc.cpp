#include "ggml-rpc.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-cpp.h"
#include "transport.h"
#include "log.h"

#include <array>
#include <chrono>
#include <cinttypes>
#include <optional>
#include <string>
#include <thread>
#include <vector>
#include <queue>
#include <type_traits>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <atomic>

namespace fs = std::filesystem;

// macro for nicer error messages on server crash
#define RPC_STATUS_ASSERT(x) if (!(x)) GGML_ABORT("Remote RPC server crashed or returned malformed response")

// all RPC structures must be packed
#pragma pack(push, 1)
// ggml_tensor is serialized into rpc_tensor
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

    int32_t use_count;
};

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
    RPC_CMD_SET_TENSOR_HASH,
    RPC_CMD_GET_TENSOR,
    RPC_CMD_COPY_TENSOR,
    RPC_CMD_GRAPH_COMPUTE,
    RPC_CMD_GET_DEVICE_MEMORY,
    RPC_CMD_INIT_TENSOR,
    RPC_CMD_GET_ALLOC_SIZE,
    RPC_CMD_HELLO,
    RPC_CMD_DEVICE_COUNT,
    RPC_CMD_GRAPH_RECOMPUTE,
    RPC_CMD_MEMSET_TENSOR,
    RPC_CMD_SET_TENSOR_2D,
    RPC_CMD_GET_TENSOR_2D,
    RPC_CMD_COMM_INIT,
    RPC_CMD_COMM_ALLREDUCE,
    RPC_CMD_COMM_FREE,
    RPC_CMD_NONE,
    RPC_CMD_COUNT,
};

static_assert(RPC_CMD_HELLO == 14, "RPC_CMD_HELLO must be always 14");

static const char * rpc_cmd_name(enum rpc_cmd cmd) {
    switch (cmd) {
        case RPC_CMD_ALLOC_BUFFER:      return "ALLOC_BUFFER";
        case RPC_CMD_GET_ALIGNMENT:     return "GET_ALIGNMENT";
        case RPC_CMD_GET_MAX_SIZE:      return "GET_MAX_SIZE";
        case RPC_CMD_BUFFER_GET_BASE:   return "BUFFER_GET_BASE";
        case RPC_CMD_FREE_BUFFER:       return "FREE_BUFFER";
        case RPC_CMD_BUFFER_CLEAR:      return "BUFFER_CLEAR";
        case RPC_CMD_SET_TENSOR:        return "SET_TENSOR";
        case RPC_CMD_SET_TENSOR_HASH:   return "SET_TENSOR_HASH";
        case RPC_CMD_GET_TENSOR:        return "GET_TENSOR";
        case RPC_CMD_COPY_TENSOR:       return "COPY_TENSOR";
        case RPC_CMD_GRAPH_COMPUTE:     return "GRAPH_COMPUTE";
        case RPC_CMD_GET_DEVICE_MEMORY: return "GET_DEVICE_MEMORY";
        case RPC_CMD_INIT_TENSOR:       return "INIT_TENSOR";
        case RPC_CMD_GET_ALLOC_SIZE:    return "GET_ALLOC_SIZE";
        case RPC_CMD_HELLO:             return "HELLO";
        case RPC_CMD_DEVICE_COUNT:      return "DEVICE_COUNT";
        case RPC_CMD_GRAPH_RECOMPUTE:   return "GRAPH_RECOMPUTE";
        case RPC_CMD_MEMSET_TENSOR:     return "MEMSET_TENSOR";
        case RPC_CMD_SET_TENSOR_2D:     return "SET_TENSOR_2D";
        case RPC_CMD_GET_TENSOR_2D:     return "GET_TENSOR_2D";
        case RPC_CMD_COMM_INIT:         return "COMM_INIT";
        case RPC_CMD_COMM_ALLREDUCE:    return "COMM_ALLREDUCE";
        case RPC_CMD_COMM_FREE:         return "COMM_FREE";
        case RPC_CMD_NONE:              return "NONE";
        default:                        return "UNKNOWN";
    }
}

// Try RPC_CMD_SET_TENSOR_HASH first when data size is larger than this threshold
const size_t HASH_THRESHOLD = 10 * 1024 * 1024;

// Maximum number of graphs cached per device; client and server must use the same value
// so that both sides clear their caches at the same point in the message stream
const size_t GRAPH_CACHE_MAX = 1024;

struct rpc_msg_hello_req {
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
};

// server feature flags in rpc_msg_hello_rsp::flags
enum rpc_dispatcher_flags : uint8_t {
    RPC_DISPATCHER_FLAGS_SEND_TENSOR_HASH = 1 << 0, // client should try SET_TENSOR_HASH before sending tensor data
};

struct rpc_msg_hello_rsp {
    uint8_t major;
    uint8_t minor;
    uint8_t patch;
    uint8_t flags;
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
};

struct rpc_msg_device_count_rsp {
    uint32_t device_count;
};

struct rpc_msg_get_alloc_size_req {
    uint32_t   device;
    rpc_tensor tensor;
    rpc_tensor srcs[GGML_MAX_SRC];
};

struct rpc_msg_get_alloc_size_rsp {
    uint64_t alloc_size;
};

struct rpc_msg_init_tensor_req {
    rpc_tensor tensor;
};

struct rpc_msg_alloc_buffer_req {
    uint32_t device;
    uint64_t size;
};

struct rpc_msg_alloc_buffer_rsp {
    uint64_t remote_ptr;
    uint64_t remote_size;
};

struct rpc_msg_get_alignment_req {
    uint32_t device;
};

struct rpc_msg_get_alignment_rsp {
    uint64_t alignment;
};

struct rpc_msg_get_max_size_req {
    uint32_t device;
};

struct rpc_msg_get_max_size_rsp {
    uint64_t max_size;
};

struct rpc_msg_buffer_get_base_req {
    uint64_t remote_ptr;
};

struct rpc_msg_buffer_get_base_rsp {
    uint64_t base_ptr;
};

struct rpc_msg_free_buffer_req {
    uint64_t remote_ptr;
};

struct rpc_msg_buffer_clear_req {
    uint64_t remote_ptr;
    uint8_t value;
};

struct rpc_msg_memset_tensor_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
    uint8_t value;
};

struct rpc_msg_set_tensor_hash_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t hash;
};

struct rpc_msg_set_tensor_hash_rsp {
    uint8_t result;
};

struct rpc_msg_get_tensor_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
};

struct rpc_msg_copy_tensor_req {
    rpc_tensor src;
    rpc_tensor dst;
};

struct rpc_msg_copy_tensor_rsp {
    uint8_t result;
};

struct rpc_msg_get_device_memory_req {
    uint32_t device;
};

struct rpc_msg_get_device_memory_rsp {
    uint64_t free_mem;
    uint64_t total_mem;
};

struct rpc_msg_graph_recompute_req {
    uint32_t device;
    uint64_t uid;
};

// GET_TENSOR_2D request and SET_TENSOR_2D header
struct rpc_msg_tensor_2d_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
    uint64_t n_copies;
    uint64_t stride;
};

struct rpc_msg_comm_init_req {
    uint32_t device;
    uint32_t rank;
    uint32_t world;
    uint32_t port;      // rank 0: port to listen on; rank > 0: rank 0's comm port
    char     host[64];  // rank > 0: rank 0's host
};

struct rpc_msg_comm_init_rsp {
    uint8_t ok;
};

struct rpc_msg_comm_allreduce_req {
    uint32_t   device;
    rpc_tensor tensor;
};

struct rpc_msg_comm_free_req {
    uint32_t device;
};

// SET_TENSOR header, followed by the data
struct rpc_msg_set_tensor_req {
    rpc_tensor tensor;
    uint8_t    cache_flag;
    uint64_t   offset;
};

// GRAPH_COMPUTE header, followed by the node ids and the tensors
struct rpc_msg_graph_compute_req {
    uint32_t device;
    uint64_t uid;
    uint32_t n_nodes;
};

#pragma pack(pop)

// request and response types of the commands with fixed-size requests, void when there is no response
template <rpc_cmd cmd> struct rpc_msg_types;

#define RPC_MSG_TYPES(cmd, req_t, rsp_t) \
    template <> struct rpc_msg_types<cmd> { using req = req_t; using rsp = rsp_t; }

RPC_MSG_TYPES(RPC_CMD_ALLOC_BUFFER,      rpc_msg_alloc_buffer_req,      rpc_msg_alloc_buffer_rsp);
RPC_MSG_TYPES(RPC_CMD_GET_ALIGNMENT,     rpc_msg_get_alignment_req,     rpc_msg_get_alignment_rsp);
RPC_MSG_TYPES(RPC_CMD_GET_MAX_SIZE,      rpc_msg_get_max_size_req,      rpc_msg_get_max_size_rsp);
RPC_MSG_TYPES(RPC_CMD_GET_ALLOC_SIZE,    rpc_msg_get_alloc_size_req,    rpc_msg_get_alloc_size_rsp);
RPC_MSG_TYPES(RPC_CMD_GET_DEVICE_MEMORY, rpc_msg_get_device_memory_req, rpc_msg_get_device_memory_rsp);
RPC_MSG_TYPES(RPC_CMD_BUFFER_GET_BASE,   rpc_msg_buffer_get_base_req,   rpc_msg_buffer_get_base_rsp);
RPC_MSG_TYPES(RPC_CMD_FREE_BUFFER,       rpc_msg_free_buffer_req,       void);
RPC_MSG_TYPES(RPC_CMD_BUFFER_CLEAR,      rpc_msg_buffer_clear_req,      void);
RPC_MSG_TYPES(RPC_CMD_INIT_TENSOR,       rpc_msg_init_tensor_req,       void);
RPC_MSG_TYPES(RPC_CMD_MEMSET_TENSOR,     rpc_msg_memset_tensor_req,     void);
RPC_MSG_TYPES(RPC_CMD_SET_TENSOR_HASH,   rpc_msg_set_tensor_hash_req,   rpc_msg_set_tensor_hash_rsp);
RPC_MSG_TYPES(RPC_CMD_GET_TENSOR,        rpc_msg_get_tensor_req,        std::vector<uint8_t>);
RPC_MSG_TYPES(RPC_CMD_GET_TENSOR_2D,     rpc_msg_tensor_2d_req,         std::vector<uint8_t>);
RPC_MSG_TYPES(RPC_CMD_COPY_TENSOR,       rpc_msg_copy_tensor_req,       rpc_msg_copy_tensor_rsp);
RPC_MSG_TYPES(RPC_CMD_GRAPH_RECOMPUTE,   rpc_msg_graph_recompute_req,   void);
RPC_MSG_TYPES(RPC_CMD_COMM_INIT,         rpc_msg_comm_init_req,         rpc_msg_comm_init_rsp);
RPC_MSG_TYPES(RPC_CMD_COMM_ALLREDUCE,    rpc_msg_comm_allreduce_req,    void);
RPC_MSG_TYPES(RPC_CMD_COMM_FREE,         rpc_msg_comm_free_req,         void);

#undef RPC_MSG_TYPES

// RPC data structures

static ggml_guid_t ggml_backend_rpc_guid() {
    static ggml_guid guid = {0x99, 0x68, 0x5b, 0x6c, 0xd2, 0x83, 0x3d, 0x24, 0x25, 0x36, 0x72, 0xe1, 0x5b, 0x0e, 0x14, 0x03};
    return &guid;
}

struct ggml_backend_rpc_device_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    std::string description;
};

struct ggml_backend_rpc_buffer_type_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    size_t      alignment;
    size_t      max_size;
};

class rpc_dispatcher;
struct ggml_backend_rpc_context {
    std::shared_ptr<rpc_dispatcher> dispatcher;
    std::string                     endpoint;
    uint32_t                        device;
    std::string                     name;
};

struct ggml_backend_rpc_buffer_context {
    std::shared_ptr<rpc_dispatcher>   dispatcher;
    void                            * base_ptr;
    uint64_t                          remote_ptr;
};

// RPC helper functions

// Computes FNV-1a hash of the data
static uint64_t fnv_hash(const uint8_t * data, size_t len, uint64_t hash = 0xcbf29ce484222325ULL) {
    const uint64_t fnv_prime = 0x100000001b3ULL;

    for (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= fnv_prime;
    }
    return hash;
}

// builds a variable-size request; the buffer must not change once queued
struct rpc_writer {
    std::shared_ptr<std::vector<uint8_t>> buf = std::make_shared<std::vector<uint8_t>>();

    explicit rpc_writer(size_t capacity) { buf->reserve(capacity); }

    void write_bytes(const void * data, size_t size) {
        const uint8_t * p = (const uint8_t *) data;
        buf->insert(buf->end(), p, p + size);
    }

    template <typename T>
    void write(const T & value) {
        static_assert(std::is_trivially_copyable_v<T>);
        write_bytes(&value, sizeof(value));
    }

    std::shared_ptr<const void> data() const { return std::shared_ptr<const void>(buf, buf->data()); }
    size_t size() const { return buf->size(); }
};

// bounds-checked cursor over a received request
struct rpc_reader {
    const uint8_t * ptr;
    size_t          left;

    explicit rpc_reader(const std::vector<uint8_t> & input) : ptr(input.data()), left(input.size()) {}

    // n elements of elem_size bytes, returned in place: the pointer has no alignment guarantee
    bool read_bytes(const uint8_t *& out, size_t n, size_t elem_size = 1) {
        if (elem_size != 0 && n > left / elem_size) {
            return false;
        }
        out   = ptr;
        ptr  += n*elem_size;
        left -= n*elem_size;
        return true;
    }

    template <typename T>
    bool read(T & value) {
        static_assert(std::is_trivially_copyable_v<T>);
        const uint8_t * p;
        if (!read_bytes(p, sizeof(T))) {
            return false;
        }
        memcpy(&value, p, sizeof(T));
        return true;
    }

    bool empty() const { return left == 0; }
};

static bool send_msg(socket_ptr sock, const void * msg, size_t msg_size) {
    if (!sock->send_data(&msg_size, sizeof(msg_size))) {
        return false;
    }
    if (!sock->send_data(msg, msg_size)) {
        return false;
    }
    return sock->flush();
}

static bool recv_msg(socket_ptr sock, void * msg, size_t msg_size) {
    uint64_t size;
    if (!sock->recv_data(&size, sizeof(size))) {
        return false;
    }
    if (size != msg_size) {
        return false;
    }
    return sock->recv_data(msg, msg_size);
}

static bool recv_msg(socket_ptr sock, std::vector<uint8_t> & input) {
    uint64_t size;
    if (!sock->recv_data(&size, sizeof(size))) {
        return false;
    }
    try {
        input.resize(size);
    } catch (const std::length_error &) {
        LOG_ERROR("Failed to allocate input buffer of size %" PRIu64 "\n", size);
        return false;
    } catch (const std::bad_alloc &) {
        LOG_ERROR("Failed to allocate input buffer of size %" PRIu64 "\n", size);
        return false;
    }
    return sock->recv_data(input.data(), size);
}

static bool parse_endpoint(const std::string & endpoint, std::string & host, int & port) {
    size_t pos = endpoint.find(':');
    if (pos == std::string::npos) {
        return false;
    }
    host = endpoint.substr(0, pos);
    try {
        port = std::stoi(endpoint.substr(pos + 1));
    } catch (...) {
        return false;
    }
    return true;
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// request_data is input followed by an optional payload, so large data does not need to be copied into input
// No response
static bool send_rpc_cmd_payload(socket_ptr sock, enum rpc_cmd cmd, const void * input, size_t input_size, const void * payload, size_t payload_size) {
    uint8_t cmd_byte = cmd;
    if (!sock->send_data(&cmd_byte, sizeof(cmd_byte))) {
        return false;
    }
    const size_t request_size = input_size + payload_size;
    if (!sock->send_data(&request_size, sizeof(request_size))) {
        return false;
    }
    if (!sock->send_data(input, input_size)) {
        return false;
    }
    if (payload_size > 0 && !sock->send_data(payload, payload_size)) {
        return false;
    }
    return sock->flush();
}

static bool send_rpc_cmd(socket_ptr sock, enum rpc_cmd cmd, const void * input, size_t input_size) {
    return send_rpc_cmd_payload(sock, cmd, input, input_size, nullptr, 0);
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// RPC response: | response_size (8 bytes) | response_data (response_size bytes) |
static bool send_rpc_cmd(socket_ptr sock, enum rpc_cmd cmd, const void * input, size_t input_size, void * output, size_t output_size) {
    if (!send_rpc_cmd(sock, cmd, input, input_size)) {
        return false;
    }
    uint64_t out_size;
    if (!sock->recv_data(&out_size, sizeof(out_size))) {
        return false;
    }
    if (out_size != output_size) {
        return false;
    }
    if (!sock->recv_data(output, output_size)) {
        return false;
    }
    return true;
}

// RPC client-side implementation

// with busy spinning on, the dispatcher still blocks on its queue after this long without commands
static constexpr auto RPC_BUSY_SPIN_IDLE_TIME = std::chrono::milliseconds(100);

static inline void rpc_cpu_relax() {
#if defined(__aarch64__) && (defined(__clang__) || defined(__GNUC__))
    __asm__ volatile("yield" ::: "memory");
#else
    std::this_thread::yield();
#endif
}

// Performs HELLO handshake with transport auto-negotiation.
// Advertises local capabilities via conn_caps; if the server responds with
// matching capabilities, the socket is upgraded transparently.
static bool negotiate_hello(const std::shared_ptr<socket_t> & sock, rpc_dispatcher_flags & flags) {
    rpc_msg_hello_req request = {};
    rpc_msg_hello_rsp response = {};

    sock->get_caps(request.conn_caps);

    bool status = send_rpc_cmd(sock, RPC_CMD_HELLO, &request, sizeof(request), &response, sizeof(response));
    if (!status) {
        LOG_ERROR("RPC hello exchange with the server failed\n");
        return false;
    }

    if (response.major != RPC_PROTO_MAJOR_VERSION || response.minor > RPC_PROTO_MINOR_VERSION) {
        LOG_ERROR("RPC server version mismatch: %d.%d.%d\n",
                       response.major, response.minor, response.patch);
        return false;
    }

    sock->update_caps(response.conn_caps);
    flags = (rpc_dispatcher_flags) response.flags;
    LOG_DBG("[%s] handshake ok: server version %d.%d.%d, transport: %s\n",
            __func__, response.major, response.minor, response.patch, sock->transport_name());
    return true;
}

template <typename T>
class message_queue {
public:
    message_queue() {}

    bool push(const T &value) {
        std::unique_lock<std::mutex> lock(mutex);
        if (interrupted) {
            return false;
        }
        queue.push(value);
        cvar.notify_all();
        return true;
    }

    bool pop(T* out) {
        std::unique_lock<std::mutex> lock(mutex);
        cvar.wait(lock, [this] { return !queue.empty() || interrupted; });
        if (interrupted) {
            return false;
        }
        *out = queue.front();
        queue.pop();
        return true;
    }

    bool try_pop(T* out) {
        std::unique_lock<std::mutex> lock(mutex);
        if (interrupted || queue.empty()) {
            return false;
        }
        *out = queue.front();
        queue.pop();
        return true;
    }

    void interrupt() {
        std::unique_lock<std::mutex> lock(mutex);
        interrupted = true;
        lock.unlock();
        cvar.notify_all();
    }

private:
    bool interrupted = false;
    std::queue<T> queue;
    std::mutex mutex;
    std::condition_variable cvar;
};

class rpc_dispatcher {
public:
    rpc_dispatcher() {
    }

    void send(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size, const void * payload, size_t payload_size, void * output, size_t output_size);
    void send_async(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size, void * output = nullptr, size_t output_size = 0);

    // sends the request of `cmd` and waits for the response, if the command has one
    template <rpc_cmd cmd>
    typename rpc_msg_types<cmd>::rsp call(const typename rpc_msg_types<cmd>::req & request) {
        using req_t = typename rpc_msg_types<cmd>::req;
        using rsp_t = typename rpc_msg_types<cmd>::rsp;
        auto input = std::make_shared<req_t>(request);
        if constexpr (std::is_void_v<rsp_t>) {
            send(cmd, input, sizeof(req_t), nullptr, 0, nullptr, 0);
        } else {
            static_assert(std::is_trivially_copyable_v<rsp_t>, "variable-size responses are read into caller memory with send()");
            rsp_t response = {};
            send(cmd, input, sizeof(req_t), nullptr, 0, &response, sizeof(response));
            return response;
        }
    }

    // queues the request of `cmd`; *response, if given, is written when the command completes
    template <rpc_cmd cmd>
    void call_async(const typename rpc_msg_types<cmd>::req & request, typename rpc_msg_types<cmd>::rsp * response = nullptr) {
        using req_t = typename rpc_msg_types<cmd>::req;
        using rsp_t = typename rpc_msg_types<cmd>::rsp;
        auto input = std::make_shared<req_t>(request);
        if constexpr (std::is_void_v<rsp_t>) {
            send_async(cmd, input, sizeof(req_t));
        } else {
            static_assert(std::is_trivially_copyable_v<rsp_t>, "variable-size responses are read into caller memory with send_async()");
            send_async(cmd, input, sizeof(req_t), response, sizeof(rsp_t));
        }
    }

    ggml_backend_event_t event_new(ggml_backend_dev_t dev);
    void event_free(ggml_backend_event_t event);
    void event_synchronize(ggml_backend_event_t event);
    void event_record(ggml_backend_event_t event);
    void synchronize();
    void busy_spin_acquire();
    void busy_spin_release();
    void graph_compute(uint32_t device, const ggml_cgraph * cgraph);

    bool start(const std::string & endpoint);
    void work();

    bool get_send_tensor_hash() const { return flags & RPC_DISPATCHER_FLAGS_SEND_TENSOR_HASH; }

    ~rpc_dispatcher();

private:
    struct rpc_msg {
        rpc_cmd                       cmd;
        std::shared_ptr<const void>   input;
        size_t                        input_size;
        void                        * output;
        size_t                        output_size;
        const void                  * payload      = nullptr;
        size_t                        payload_size = 0;
        std::promise<void>            completion;
    };
    using rpc_msg_ptr   = std::shared_ptr<rpc_msg>;
    using rpc_msg_queue = message_queue<rpc_msg_ptr>;

    struct rpc_event {
        rpc_msg_ptr              msg;
        std::shared_future<void> sf;
    };

    std::mutex graph_mutex;
    std::unordered_map<uint32_t, std::unordered_set<uint64_t>> graph_uids;

    rpc_msg_queue queue;
    socket_ptr    sock;

    rpc_dispatcher_flags flags = {};

    std::atomic_uint busy_spin_users = 0;
    std::atomic_bool running;
    std::thread      thread;
};

static void rpc_dispatcher_trampoline(rpc_dispatcher * dispatcher)
{
    dispatcher->work();
}

void rpc_dispatcher::send(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size, const void * payload, size_t payload_size, void * output, size_t output_size) {
    auto msg = std::make_shared<rpc_msg>();
    msg->cmd = cmd;
    msg->input = std::move(input);
    msg->input_size = input_size;
    msg->output = output;
    msg->output_size = output_size;
    msg->payload = payload;
    msg->payload_size = payload_size;
    GGML_ASSERT(queue.push(msg));
    auto future = msg->completion.get_future();
    future.wait();
}

void rpc_dispatcher::send_async(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size, void * output, size_t output_size) {
    auto msg = std::make_shared<rpc_msg>();
    msg->cmd = cmd;
    msg->input = std::move(input);
    msg->input_size = input_size;
    msg->output = output;
    msg->output_size = output_size;
    GGML_ASSERT(queue.push(msg));
}

ggml_backend_event_t rpc_dispatcher::event_new(ggml_backend_dev_t dev) {
    rpc_event * ev = new rpc_event;
    ev->msg = std::make_shared<rpc_msg>();
    ev->msg->cmd = RPC_CMD_NONE;
    ev->sf = ev->msg->completion.get_future().share();
    GGML_ASSERT(queue.push(ev->msg));
    return new ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ ev,
    };
}

void rpc_dispatcher::event_free(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    delete ev;
}

void rpc_dispatcher::event_synchronize(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    ev->sf.wait();
}

void rpc_dispatcher::event_record(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    ev->msg = std::make_shared<rpc_msg>();
    ev->msg->cmd = RPC_CMD_NONE;
    ev->sf = ev->msg->completion.get_future().share();
    GGML_ASSERT(queue.push(ev->msg));
}

void rpc_dispatcher::synchronize() {
    // to ensure all messages are processed, submit dummy message and wait for it to complete
    auto msg = std::make_shared<rpc_msg>();
    msg->cmd = RPC_CMD_NONE;
    GGML_ASSERT(queue.push(msg));
    msg->completion.get_future().wait();
}

void rpc_dispatcher::busy_spin_acquire() {
    busy_spin_users.fetch_add(1, std::memory_order_relaxed);
}

void rpc_dispatcher::busy_spin_release() {
    const unsigned previous = busy_spin_users.fetch_sub(1, std::memory_order_relaxed);
    GGML_ASSERT(previous > 0);
}

bool rpc_dispatcher::start(const std::string & endpoint) {
    std::string host;
    int port;
    if (!parse_endpoint(endpoint, host, port)) {
        LOG_ERROR("Failed to parse endpoint: %s\n", endpoint.c_str());
        return false;
    }
    if (!rpc_transport_init()) {
        LOG_ERROR("RPC transport initialization failed\n");
        return false;
    }

    sock = socket_t::connect(host.c_str(), port);
    if (sock == nullptr) {
        LOG_ERROR("Failed to connect to %s\n", endpoint.c_str());
        return false;
    }
    if (!negotiate_hello(sock, flags)) {
        LOG_ERROR("RPC handshake failed for %s\n", endpoint.c_str());
        return false;
    }
    LOG_DBG("[%s] connected to %s\n", __func__, endpoint.c_str());
    running = true;
    thread = std::thread(rpc_dispatcher_trampoline, this);
    return true;
}

void rpc_dispatcher::work() {
    auto last_cmd = std::chrono::steady_clock::now();
    while (running) {
        rpc_msg_ptr msg_ptr;
        // spin only while commands keep coming, so an idle dispatcher does not keep a core busy
        const bool spin = busy_spin_users.load(std::memory_order_relaxed) != 0 &&
                          std::chrono::steady_clock::now() - last_cmd < RPC_BUSY_SPIN_IDLE_TIME;
        if (spin) {
            if (!queue.try_pop(&msg_ptr)) {
                rpc_cpu_relax();
                continue;
            }
        } else if (!queue.pop(&msg_ptr)) {
            break;
        }
        if (msg_ptr->cmd != RPC_CMD_NONE) {
            LOG_DBG2("[%s] %s (in: %s, out: %s)\n", __func__, rpc_cmd_name(msg_ptr->cmd),
                     rpc_log_format_size(msg_ptr->input_size).c_str(), rpc_log_format_size(msg_ptr->output_size).c_str());
            const bool trace = rpc_log_debug_level() >= 3;
            const auto t0 = std::chrono::steady_clock::now();
            if (msg_ptr->output) {
                bool status = send_rpc_cmd(sock, msg_ptr->cmd, msg_ptr->input.get(), msg_ptr->input_size, msg_ptr->output, msg_ptr->output_size);
                RPC_STATUS_ASSERT(status);
            } else {
                bool status = send_rpc_cmd_payload(sock, msg_ptr->cmd, msg_ptr->input.get(), msg_ptr->input_size, msg_ptr->payload, msg_ptr->payload_size);
                RPC_STATUS_ASSERT(status);
            }
            if (trace) {
                const std::chrono::duration<double, std::milli> elapsed = std::chrono::steady_clock::now() - t0;
                LOG_DBG3("[%s] %s done in %.3f ms\n", __func__, rpc_cmd_name(msg_ptr->cmd), elapsed.count());
            }
        } else {
            LOG_DBG3("[%s] sync barrier\n", __func__);
        }
        msg_ptr->completion.set_value();
        last_cmd = std::chrono::steady_clock::now();
    }
}

rpc_dispatcher::~rpc_dispatcher() {
    running = false;
    queue.interrupt();
    sock = nullptr;
    if (thread.joinable()) {
        thread.join();
    }
}

static std::shared_ptr<rpc_dispatcher> get_dispatcher(const std::string & endpoint) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    static std::unordered_map<std::string, std::weak_ptr<rpc_dispatcher>> dispatchers;

    auto it = dispatchers.find(endpoint);
    if (it != dispatchers.end()) {
        if (auto dispatcher = it->second.lock()) {
            return dispatcher;
        }
    }

    auto dispatcher = std::make_shared<rpc_dispatcher>();
    if (!dispatcher->start(endpoint)) {
        return nullptr;
    }
    dispatchers[endpoint] = dispatcher;
    return dispatcher;
}

static void ggml_backend_rpc_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, ctx->remote_ptr);
    ctx->dispatcher->call<RPC_CMD_FREE_BUFFER>({ ctx->remote_ptr });
    delete ctx;
}

static void * ggml_backend_rpc_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    if (ctx->base_ptr != nullptr) {
        return ctx->base_ptr;
    }
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, ctx->remote_ptr);
    const auto response = ctx->dispatcher->call<RPC_CMD_BUFFER_GET_BASE>({ ctx->remote_ptr });
    ctx->base_ptr = reinterpret_cast<void *>(response.base_ptr);
    return ctx->base_ptr;
}

static bool ggml_backend_buffer_is_rpc(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_rpc_buffer_free_buffer;
}

static rpc_tensor serialize_tensor(const ggml_tensor * tensor, const rpc_dispatcher * dispatcher = nullptr) {
    rpc_tensor result;
    if (!tensor) {
        memset(&result, 0, sizeof(result));
        return result;
    }

    result.id = reinterpret_cast<uint64_t>(tensor);
    result.type = tensor->type;
    if (tensor->buffer && ggml_backend_buffer_is_rpc(tensor->buffer)) {
        ggml_backend_buffer_t buffer = tensor->buffer;
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        // ref: https://github.com/ggml-org/llama.cpp/pull/26500
        if (ctx != nullptr && (dispatcher == nullptr || ctx->dispatcher.get() == dispatcher)) {
            result.buffer = ctx->remote_ptr;
            result.data = reinterpret_cast<uint64_t>(tensor->data);
        } else {
            result.buffer = 0;
            result.data = 0;
        }
    } else {
        result.buffer = 0;
        result.data   = 0;
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

    // Avoid sending uninitialized data over the wire
    memset(result.name, 0, sizeof(result.name));
    result.use_count = 0;

    snprintf(result.name, GGML_MAX_NAME, "%s", tensor->name);
    return result;
}

static enum ggml_status ggml_backend_rpc_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;

    // CUDA backend on the server pads everything to 512 due to CUDA limitations.
    // Due to bandwidth constraints, we only call the server init tensor functions if necessary.
    // In particular, only quantized tensors need padding
    if (ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr)) {
        LOG_DBG2("[%s] tensor: %s\n", __func__, rpc_log_format_tensor(tensor).c_str());
        ctx->dispatcher->call<RPC_CMD_INIT_TENSOR>({ serialize_tensor(tensor) });
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_rpc_buffer_memset_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, value: %u\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str(), value);
    ctx->dispatcher->call<RPC_CMD_MEMSET_TENSOR>({ serialize_tensor(tensor), offset, size, value });
}

// the hash cache is meant for weights, so that a model reload can skip re-sending them.
// compute-buffer inputs (the activations ggml_backend_sched copies between backends) must not
// take this path, otherwise with `rpc-server -c` every ubatch above the threshold is written
// to the cache directory and later served from there.
// hashing is slow (~1 GB/s), so only do it when the server asks for tensor hashes.
static bool rpc_use_hash_cache(const rpc_dispatcher & dispatcher, const ggml_tensor * tensor, size_t size) {
    return dispatcher.get_send_tensor_hash() && size > HASH_THRESHOLD && tensor->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

// true if the server loaded the data from its cache; on a miss, cache_flag asks it to cache the data sent next
static bool rpc_set_tensor_from_cache(rpc_dispatcher & dispatcher, const ggml_tensor * tensor, const rpc_tensor & wire,
                                      const void * data, size_t offset, size_t size, uint8_t & cache_flag) {
    cache_flag = 0;
    if (!rpc_use_hash_cache(dispatcher, tensor, size)) {
        return false;
    }
    // TODO: make this async
    const rpc_msg_set_tensor_hash_req request = { wire, offset, fnv_hash((const uint8_t *) data, size) };
    const auto response = dispatcher.call<RPC_CMD_SET_TENSOR_HASH>(request);
    LOG_DBG2("[%s] tensor: %s, hash: 0x%" PRIx64 ", cache: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
             request.hash, response.result ? "hit" : "miss");
    cache_flag = response.result ? 0 : 1;
    return response.result != 0;
}

static void ggml_backend_rpc_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    const rpc_tensor wire = serialize_tensor(tensor);
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    uint8_t cache_flag;
    if (rpc_set_tensor_from_cache(*ctx->dispatcher, tensor, wire, data, offset, size, cache_flag)) {
        return;
    }
    // | rpc_msg_set_tensor_req | data (size bytes) |
    auto request = std::make_shared<rpc_msg_set_tensor_req>(rpc_msg_set_tensor_req{ wire, cache_flag, offset });
    ctx->dispatcher->send(RPC_CMD_SET_TENSOR, request, sizeof(*request), data, size, nullptr, 0);
}

static void ggml_backend_rpc_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, n_copies: %zu, stride: %zu\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), offset, rpc_log_format_size(size).c_str(), n_copies, stride_tensor);
    // | rpc_msg_tensor_2d_req | data (size * n_copies bytes) |
    rpc_writer writer(sizeof(rpc_msg_tensor_2d_req) + size*n_copies);
    writer.write(rpc_msg_tensor_2d_req{ serialize_tensor(tensor), offset, size, n_copies, stride_tensor });
    for (size_t i = 0; i < n_copies; i++) {
        writer.write_bytes((const char *) data + i*stride_data, size);
    }
    ctx->dispatcher->send(RPC_CMD_SET_TENSOR_2D, writer.data(), writer.size(), nullptr, 0, nullptr, 0);
}

static void ggml_backend_rpc_buffer_get_tensor_2d(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, n_copies: %zu, stride: %zu\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), offset, rpc_log_format_size(size).c_str(), n_copies, stride_tensor);
    auto request = std::make_shared<rpc_msg_tensor_2d_req>();
    request->tensor   = serialize_tensor(tensor);
    request->offset   = offset;
    request->size     = size;
    request->n_copies = n_copies;
    request->stride   = stride_tensor;
    if (stride_data == size) {
        ctx->dispatcher->send(RPC_CMD_GET_TENSOR_2D, request, sizeof(*request), nullptr, 0, data, size*n_copies);
    } else {
        std::vector<uint8_t> packed(size*n_copies);
        ctx->dispatcher->send(RPC_CMD_GET_TENSOR_2D, request, sizeof(*request), nullptr, 0, packed.data(), packed.size());
        for (size_t i = 0; i < n_copies; i++) {
            memcpy((char *)data + i*stride_data, packed.data() + i*size, size);
        }
    }
}

static void ggml_backend_rpc_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    auto request = std::make_shared<rpc_msg_get_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size = size;
    ctx->dispatcher->send(RPC_CMD_GET_TENSOR, request, sizeof(*request), nullptr, 0, data, size);
}

static bool ggml_backend_rpc_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_rpc(src->buffer)) {
        // check if src and dst are on the same server
        ggml_backend_buffer_t src_buffer = src->buffer;
        ggml_backend_rpc_buffer_context * src_ctx = (ggml_backend_rpc_buffer_context *)src_buffer->context;
        ggml_backend_buffer_t dst_buffer = dst->buffer;
        ggml_backend_rpc_buffer_context * dst_ctx = (ggml_backend_rpc_buffer_context *)dst_buffer->context;
        if (src_ctx->dispatcher != dst_ctx->dispatcher) {
            LOG_DBG2("[%s] src and dst are on different servers, falling back to a host copy\n", __func__);
            return false;
        }
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        LOG_DBG2("[%s] src: %s -> dst: %s\n", __func__, rpc_log_format_tensor(src).c_str(), rpc_log_format_tensor(dst).c_str());
        return ctx->dispatcher->call<RPC_CMD_COPY_TENSOR>({ serialize_tensor(src), serialize_tensor(dst) }).result;
    }
    return false;
}

static void ggml_backend_rpc_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 ", value: %u\n", __func__, ctx->remote_ptr, value);
    ctx->dispatcher->call<RPC_CMD_BUFFER_CLEAR>({ ctx->remote_ptr, value });
}

static ggml_backend_buffer_i ggml_backend_rpc_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_rpc_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_rpc_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_rpc_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_rpc_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_rpc_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_rpc_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_rpc_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_rpc_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_rpc_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_rpc_buffer_clear,
    /* .reset           = */ NULL,
    /* .set_tensor_n    = */ NULL,
    /* .get_tensor_n    = */ NULL,
    /* .cpy_tensor_n    = */ NULL,
};

static const char * ggml_backend_rpc_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_rpc_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    auto dispatcher = get_dispatcher(buft_ctx->endpoint);
    if (dispatcher == nullptr) {
        return nullptr;
    }
    const auto response = dispatcher->call<RPC_CMD_ALLOC_BUFFER>({ buft_ctx->device, size });
    if (response.remote_ptr != 0) {
        LOG_DBG("[%s] endpoint: %s, device: %u, size: %s -> remote_ptr: 0x%" PRIx64 ", remote_size: %s\n",
                __func__, buft_ctx->endpoint.c_str(), buft_ctx->device, rpc_log_format_size(size).c_str(),
                response.remote_ptr, rpc_log_format_size(response.remote_size).c_str());
        ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft,
            ggml_backend_rpc_buffer_interface,
            new ggml_backend_rpc_buffer_context{dispatcher, nullptr, response.remote_ptr},
            response.remote_size);
        return buffer;
    } else {
        LOG_WARN("[%s] endpoint: %s, device: %u, size: %s -> failed\n",
                __func__, buft_ctx->endpoint.c_str(), buft_ctx->device, rpc_log_format_size(size).c_str());
        return nullptr;
    }
}

static size_t get_alignment(const std::shared_ptr<rpc_dispatcher> & dispatcher, uint32_t device) {
    const auto response = dispatcher->call<RPC_CMD_GET_ALIGNMENT>({ device });
    LOG_DBG2("[%s] device: %u, alignment: %s\n", __func__, device, rpc_log_format_size(response.alignment).c_str());
    return response.alignment;
}

static size_t ggml_backend_rpc_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->alignment;
}

static size_t get_max_size(const std::shared_ptr<rpc_dispatcher> & dispatcher, uint32_t device) {
    const auto response = dispatcher->call<RPC_CMD_GET_MAX_SIZE>({ device });
    LOG_DBG2("[%s] device: %u, max_size: %s\n", __func__, device, rpc_log_format_size(response.max_size).c_str());
    return response.max_size;
}

static size_t ggml_backend_rpc_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->max_size;
}

static size_t ggml_backend_rpc_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    // should we query the remote server for the actual size
    bool rpc_get = false;

    // See comments in init_tensor.
    rpc_get |= ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr);

    // [TAG_ALLOC_SIZE_EXPAND]
    // ops that may require additional memory for fleeting data on certain backends
    // ref: https://github.com/ggml-org/llama.cpp/pull/15966
    rpc_get |= ggml_op_alloc_size_may_expand(tensor->op);

    if (rpc_get) {
        ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;

        // the reported size must never be below ggml_nbytes: rpc_tensor stores nb[] as uint32_t,
        // so a stride over 4 GiB is truncated on the wire and the remote size comes back too small
        // TODO: change rpc_tensor nb to 64-bit int
        const size_t min_size = ggml_nbytes(tensor);

        // Cache key for calls to read the alloc_size.
        // We deliberately exclude src tensor dimensions from the key because:
        // 1. For CPU backends, alloc_size = ggml_nbytes(output) regardless of src shapes
        // 2. For GPU backends, the reservation graph uses max dimensions, so the
        //    cached value from reservation is always >= any subsequent request
        // 3. Including src dims causes cache misses per-ubatch (e.g. growing KV cache)
        //    which blocks the main thread behind in-flight GRAPH_COMPUTE commands
        struct alloc_size_cache_key {
            uint32_t device;
            uint32_t type;
            uint32_t op;
            int32_t  op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
            uint32_t ne[GGML_MAX_DIMS];
            uint64_t nb[GGML_MAX_DIMS];
        };

        alloc_size_cache_key key = {};
        key.device = buft_ctx->device;
        key.type = tensor->type;
        key.op = tensor->op;
        memcpy(key.op_params, tensor->op_params, sizeof(key.op_params));
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            key.ne[i] = (uint32_t)tensor->ne[i];
            key.nb[i] = (uint64_t)tensor->nb[i];
        }

        uint64_t cache_hash = fnv_hash((const uint8_t *)&key, sizeof(key));
        cache_hash = fnv_hash((const uint8_t *)buft_ctx->endpoint.data(), buft_ctx->endpoint.size(), cache_hash);

        // alloc sizes are immutable for a given tensor configuration
        static std::mutex cache_mutex;
        static std::unordered_map<uint64_t, size_t> cache;

        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            auto it = cache.find(cache_hash);
            if (it != cache.end()) {
                LOG_DBG3("[%s] cache hit: %s -> %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                         rpc_log_format_size(it->second).c_str());
                return std::max<size_t>(it->second, min_size);
            }
        }

        rpc_msg_get_alloc_size_req request = {};
        request.device = buft_ctx->device;
        request.tensor = serialize_tensor(tensor);

        // .get_alloc_size could be a function of the tensor's srcs, so we must serialize them as well
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            request.srcs[i] = serialize_tensor(tensor->src[i]);
        }

        LOG_DBG2("[%s] cache miss: %s, querying server\n", __func__, rpc_log_format_tensor(tensor).c_str());
        auto dispatcher = get_dispatcher(buft_ctx->endpoint);
        if (dispatcher == nullptr) {
            return min_size;
        }
        const auto response = dispatcher->call<RPC_CMD_GET_ALLOC_SIZE>(request);

        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            cache[cache_hash] = response.alloc_size;
        }

        LOG_DBG2("[%s] %s -> alloc_size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 rpc_log_format_size(response.alloc_size).c_str());

        return std::max<size_t>(response.alloc_size, min_size);
    }

    return ggml_nbytes(tensor);
}

static ggml_backend_buffer_type_i ggml_backend_rpc_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_rpc_buffer_type_name,
    /* .alloc_buffer        = */ ggml_backend_rpc_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ NULL,
    /* .get_alignment       = */ ggml_backend_rpc_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_rpc_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_rpc_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ NULL,
    /* .is_host             = */ NULL,
};

static const char * ggml_backend_rpc_name(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;

    return rpc_ctx->name.c_str();
}

static void ggml_backend_rpc_free(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    delete rpc_ctx;
    delete backend;
}

static void ggml_backend_rpc_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_rpc_context * ctx = (ggml_backend_rpc_context *)backend->context;
    const rpc_tensor wire = serialize_tensor(tensor);
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    uint8_t cache_flag;
    if (rpc_set_tensor_from_cache(*ctx->dispatcher, tensor, wire, data, offset, size, cache_flag)) {
        return;
    }
    // the caller may reuse data once this returns, so it is copied into the request
    rpc_writer writer(sizeof(rpc_msg_set_tensor_req) + size);
    writer.write(rpc_msg_set_tensor_req{ wire, cache_flag, offset });
    writer.write_bytes(data, size);
    ctx->dispatcher->send_async(RPC_CMD_SET_TENSOR, writer.data(), writer.size());
}

static void ggml_backend_rpc_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_rpc_context * ctx = (ggml_backend_rpc_context *)backend->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    auto request = std::make_shared<rpc_msg_get_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size = size;
    ctx->dispatcher->send_async(RPC_CMD_GET_TENSOR, request, sizeof(*request), data, size);
}

static void ggml_backend_rpc_synchronize(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->synchronize();
}

// adds the tensor after its sources and view_src, so that the server can build the graph in one pass
static void add_tensor(ggml_tensor * tensor, const ggml_cgraph * cgraph, const rpc_dispatcher * dispatcher, std::vector<rpc_tensor> & tensors, std::unordered_set<ggml_tensor*> & visited) {
    if (tensor == nullptr) {
        return;
    }
    if (visited.find(tensor) != visited.end()) {
        return;
    }
    visited.insert(tensor);
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        add_tensor(tensor->src[i], cgraph, dispatcher, tensors, visited);
    }
    add_tensor(tensor->view_src, cgraph, dispatcher, tensors, visited);
    rpc_tensor result = serialize_tensor(tensor, dispatcher);
    const size_t hash_pos = ggml_hash_find(&cgraph->visited_hash_set, tensor);
    if (hash_pos != GGML_HASHSET_FULL && ggml_bitset_get(cgraph->visited_hash_set.used, hash_pos)) {
        result.use_count = cgraph->use_counts[hash_pos];
    }
    tensors.push_back(result);
}

// | rpc_msg_graph_compute_req | nodes (n_nodes * uint64_t) | n_tensors (uint32_t) | tensors (n_tensors * rpc_tensor) |
static rpc_writer serialize_graph(uint32_t device, const ggml_cgraph * cgraph, const rpc_dispatcher * dispatcher) {
    const uint32_t n_nodes = cgraph->n_nodes;
    std::vector<rpc_tensor> tensors;
    std::unordered_set<ggml_tensor*> visited;
    for (uint32_t i = 0; i < n_nodes; i++) {
        add_tensor(cgraph->nodes[i], cgraph, dispatcher, tensors, visited);
    }
    const uint32_t n_tensors = tensors.size();
    rpc_writer writer(sizeof(rpc_msg_graph_compute_req) + n_nodes*sizeof(uint64_t) + sizeof(n_tensors) + n_tensors*sizeof(rpc_tensor));
    writer.write(rpc_msg_graph_compute_req{ device, cgraph->uid, n_nodes });
    for (uint32_t i = 0; i < n_nodes; i++) {
        writer.write(reinterpret_cast<uint64_t>(cgraph->nodes[i]));
    }
    writer.write(n_tensors);
    writer.write_bytes(tensors.data(), n_tensors*sizeof(rpc_tensor));
    return writer;
}

void rpc_dispatcher::graph_compute(uint32_t device, const ggml_cgraph * cgraph) {
    std::lock_guard<std::mutex> lock(graph_mutex);
    GGML_ASSERT(cgraph->n_nodes > 0);
    auto & device_graph_uids = graph_uids[device];
    bool reuse = cgraph->uid != 0 && device_graph_uids.count(cgraph->uid) > 0;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 ", n_nodes: %u, reuse: %s\n", __func__, device, cgraph->uid, cgraph->n_nodes, reuse ? "yes" : "no");
    if (reuse) {
        call_async<RPC_CMD_GRAPH_RECOMPUTE>({ device, cgraph->uid });
    } else {
        if (cgraph->uid != 0) {
            if (device_graph_uids.size() >= GRAPH_CACHE_MAX) {
                LOG_DBG("[%s] clearing graph cache for device %u (%zu entries)\n",
                        __func__, device, device_graph_uids.size());
                device_graph_uids.clear();
            }
            device_graph_uids.insert(cgraph->uid);
        }
        rpc_writer writer = serialize_graph(device, cgraph, this);
        LOG_DBG2("[%s] serialized graph: %s\n", __func__, rpc_log_format_size(writer.size()).c_str());
        send_async(RPC_CMD_GRAPH_COMPUTE, writer.data(), writer.size());
    }
}

static enum ggml_status ggml_backend_rpc_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->graph_compute(rpc_ctx->device, cgraph);
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_rpc_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->event_record(event);
}

static void ggml_backend_rpc_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    // this is noop for RPC as we have a single stream
    GGML_UNUSED(backend);
    GGML_UNUSED(event);
}

static void ggml_backend_rpc_set_tensor_2d_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_tensor_set_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
    GGML_UNUSED(backend);
}

static void ggml_backend_rpc_get_tensor_2d_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_tensor_get_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
    GGML_UNUSED(backend);
}

static ggml_backend_i ggml_backend_rpc_interface = {
    /* .get_name                = */ ggml_backend_rpc_name,
    /* .free                    = */ ggml_backend_rpc_free,
    /* .set_tensor_async        = */ ggml_backend_rpc_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_rpc_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_rpc_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_rpc_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ggml_backend_rpc_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_rpc_graph_compute,
    /* .event_record            = */ ggml_backend_rpc_event_record,
    /* .event_wait              = */ ggml_backend_rpc_event_wait,
    /* .graph_optimize          = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint, uint32_t device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    std::string buft_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    // NOTE: buffer types are allocated and never freed; this is by design
    static std::unordered_map<std::string, ggml_backend_buffer_type_t> buft_map;
    auto it = buft_map.find(buft_name);
    if (it != buft_map.end()) {
        return it->second;
    }
    auto dispatcher = get_dispatcher(endpoint);
    if (dispatcher == nullptr) {
        return nullptr;
    }
    size_t alignment = get_alignment(dispatcher, device);
    size_t max_size = get_max_size(dispatcher, device);
    ggml_backend_rpc_buffer_type_context * buft_ctx = new ggml_backend_rpc_buffer_type_context {
        /* .endpoint  = */ endpoint,
        /* .device    = */ device,
        /* .name      = */ buft_name,
        /* .alignment = */ alignment,
        /* .max_size  = */ max_size
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
    ggml_backend_buffer_type_t buft = new ggml_backend_buffer_type {
        /* .iface   = */ ggml_backend_rpc_buffer_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ buft_ctx
    };
    buft_map[buft_name] = buft;
    LOG_DBG("[%s] created %s (alignment: %s, max_size: %s)\n", __func__, buft_name.c_str(),
            rpc_log_format_size(alignment).c_str(), rpc_log_format_size(max_size).c_str());
    return buft;
}

ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device) {
    std::string dev_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    auto dispatcher = get_dispatcher(endpoint);
    if (dispatcher == nullptr) {
        return nullptr;
    }
    ggml_backend_rpc_context * ctx = new ggml_backend_rpc_context {
        /* .dispatcher = */ dispatcher,
        /* .endpoint   = */ endpoint,
        /* .device     = */ device,
        /* .name       = */ dev_name,
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
    LOG_DBG("[%s] %s\n", __func__, dev_name.c_str());
    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_rpc_guid(),
        /* .iface   = */ ggml_backend_rpc_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ ctx
    };
    return backend;
}

bool ggml_backend_is_rpc(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_rpc_guid());
}

void ggml_backend_rpc_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total) {
    auto dispatcher = get_dispatcher(endpoint);
    if (dispatcher == nullptr) {
        *free  = 0;
        *total = 0;
        return;
    }
    const auto response = dispatcher->call<RPC_CMD_GET_DEVICE_MEMORY>({ device });
    LOG_DBG2("[%s] device: %u, free: %s, total: %s\n", __func__, device,
             rpc_log_format_size(response.free_mem).c_str(), rpc_log_format_size(response.total_mem).c_str());
    *free = response.free_mem;
    *total = response.total_mem;
}

// RPC server-side implementation

// ggml_nbytes on the wire values, false if it overflows
static bool rpc_tensor_nbytes(const rpc_tensor & tensor, uint64_t & nbytes) {
    nbytes = 0;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (tensor.ne[i] == 0) {
            return true;
        }
    }
    // ne and nb are 32-bit on the wire, so every product fits in 64 bits and only the sums can overflow
    const ggml_type type      = (ggml_type) tensor.type;
    const uint64_t  blck_size = ggml_blck_size(type);
    uint64_t n = blck_size == 1 ? ggml_type_size(type) + (uint64_t) (tensor.ne[0] - 1)*tensor.nb[0]
                                : (uint64_t) tensor.ne[0]*tensor.nb[0]/blck_size;
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        const uint64_t term = (uint64_t) (tensor.ne[i] - 1)*tensor.nb[i];
        if (term > UINT64_MAX - n) {
            return false;
        }
        n += term;
    }
    if (n != (size_t) n) {
        return false;
    }
    nbytes = n;
    return true;
}

static bool rpc_in_buffer(ggml_backend_buffer_t buffer, uint64_t addr, uint64_t size) {
    const uint64_t base     = (uint64_t) ggml_backend_buffer_get_base(buffer);
    const uint64_t buf_size = (uint64_t) ggml_backend_buffer_get_size(buffer);
    return addr >= base && addr - base <= buf_size && size <= buf_size - (addr - base);
}

// checks [offset, offset + (n_copies - 1)*stride + size) against the tensor size, which ggml_backend_tensor_set/get assert on
static bool rpc_check_region(const char * func, const ggml_tensor * tensor, uint64_t offset, uint64_t size,
                             uint64_t n_copies = 1, uint64_t stride = 0) {
    if (tensor == nullptr || tensor->buffer == nullptr || tensor->data == nullptr) {
        LOG_ERROR("[%s] invalid or unallocated tensor\n", func);
        return false;
    }
    // rows must not overlap, so size * n_copies <= span
    if (n_copies == 0 || (n_copies > 1 && stride < size) ||
        (n_copies > 1 && stride != 0 && n_copies - 1 > (UINT64_MAX - size) / stride)) {
        LOG_ERROR("[%s] invalid region: size=%" PRIu64 ", n_copies=%" PRIu64 ", stride=%" PRIu64 "\n", func, size, n_copies, stride);
        return false;
    }
    const uint64_t span   = (n_copies - 1)*stride + size;
    const uint64_t nbytes = ggml_nbytes(tensor);
    if (offset > nbytes || span > nbytes - offset) {
        LOG_ERROR("[%s] region (offset=%" PRIu64 ", span=%" PRIu64 ") out of tensor '%s' bounds (%" PRIu64 ")\n",
                  func, offset, span, tensor->name, nbytes);
        return false;
    }
    return true;
}

// some backends address a view through view_src->data + view_offs, so that range must lie inside view_src's buffer
static bool rpc_check_view(const ggml_tensor * tensor) {
    const ggml_tensor * view_src = tensor->view_src;
    if (view_src == nullptr || tensor->buffer == nullptr || ggml_nbytes(tensor) == 0) {
        return true;
    }
    if (view_src->buffer == nullptr) {
        return false;
    }
    const uint64_t addr = (uint64_t) view_src->data;
    return tensor->view_offs <= UINT64_MAX - addr && rpc_in_buffer(view_src->buffer, addr + tensor->view_offs, ggml_nbytes(tensor));
}

// a computed node and its sources must be allocated on this server
static bool rpc_check_node(const ggml_tensor * node) {
    if (ggml_op_is_empty(node->op)) {
        return true;
    }
    if (ggml_nelements(node) > 0 && node->buffer == nullptr) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        const ggml_tensor * src = node->src[i];
        if (src != nullptr && ggml_nelements(src) > 0 && src->buffer == nullptr) {
            return false;
        }
    }
    return true;
}

class rpc_server {
public:
    rpc_server(std::vector<ggml_backend_t> all_backends, const char * cache_dir)
        : backends(std::move(all_backends)), cache_dir(cache_dir) {
        stored_graphs.resize(backends.size());
        comm_states.resize(backends.size());
        ggml_init_params params {
            /*.mem_size   =*/ ggml_tensor_overhead()*(1 + GGML_MAX_SRC),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        req_ctx.reset(ggml_init(params));
        GGML_ASSERT(req_ctx != nullptr);
    }
    ~rpc_server();

    void hello(rpc_msg_hello_rsp & response);
    bool alloc_buffer(const rpc_msg_alloc_buffer_req & request, rpc_msg_alloc_buffer_rsp & response);
    bool get_alignment(const rpc_msg_get_alignment_req & request, rpc_msg_get_alignment_rsp & response);
    bool get_max_size(const rpc_msg_get_max_size_req & request, rpc_msg_get_max_size_rsp & response);
    bool buffer_get_base(const rpc_msg_buffer_get_base_req & request, rpc_msg_buffer_get_base_rsp & response);
    bool free_buffer(const rpc_msg_free_buffer_req & request);
    bool buffer_clear(const rpc_msg_buffer_clear_req & request);
    bool memset_tensor(const rpc_msg_memset_tensor_req & request);
    bool set_tensor(socket_ptr sock, uint64_t input_size);
    bool set_tensor_2d(const std::vector<uint8_t> & input);
    bool set_tensor_hash(const rpc_msg_set_tensor_hash_req & request, rpc_msg_set_tensor_hash_rsp & response);
    bool get_tensor(const rpc_msg_get_tensor_req & request, std::vector<uint8_t> & response);
    bool get_tensor_2d(const rpc_msg_tensor_2d_req & request, std::vector<uint8_t> & response);
    bool copy_tensor(const rpc_msg_copy_tensor_req & request, rpc_msg_copy_tensor_rsp & response);
    bool graph_compute(const std::vector<uint8_t> & input);
    bool graph_recompute(const rpc_msg_graph_recompute_req & request);
    bool comm_init(const rpc_msg_comm_init_req & request, rpc_msg_comm_init_rsp & response);
    bool comm_allreduce(const rpc_msg_comm_allreduce_req & request);
    bool comm_free(const rpc_msg_comm_free_req & request);
    bool init_tensor(const rpc_msg_init_tensor_req & request);
    bool get_alloc_size(const rpc_msg_get_alloc_size_req & request, rpc_msg_get_alloc_size_rsp & response);
    bool get_device_memory(const rpc_msg_get_device_memory_req & request, rpc_msg_get_device_memory_rsp & response);

    struct stored_graph {
        std::vector<uint8_t>   buffer;
        ggml_cgraph          * graph;
    };

private:
    // empty context for the tensors of one request, valid until the next call
    ggml_context * request_ctx() {
        ggml_reset(req_ctx.get());
        return req_ctx.get();
    }
    // nullptr if the buffer was not allocated by this server
    ggml_backend_buffer_t find_buffer(uint64_t remote_ptr) const {
        ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(remote_ptr);
        return buffers.find(buffer) != buffers.end() ? buffer : nullptr;
    }
    void sync_all_backends();
    void comm_free_device(uint32_t dev_id);
    static bool comm_allreduce_pairwise(void * comm_ctx_v, ggml_tensor ** tensors);
    static bool comm_bootstrap_exchange(void * user_data, const void * send_buf, void * recv_buf, size_t size);
    bool get_cached_file(uint64_t hash, std::vector<uint8_t> & data);
    ggml_tensor * deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor);


    // Per-device state for one communicator.
    struct comm_state {
        // Sockets to the other servers. Rank 0 holds one per peer in rank order, so peers[i] is
        // rank i + 1. Every other rank holds one to rank 0. The pairwise path reduces over peers[0].
        std::vector<socket_ptr> peers;
        uint32_t                rank = 0;
        uint32_t                world = 0;
        ggml_backend_t          pairwise_backend = nullptr;
        ggml_backend_buffer_ptr scratch;
        size_t                  scratch_size = 0;
        std::vector<uint8_t>    send_buf;
        std::vector<uint8_t>    recv_buf;

        // The backend's communicator. Null on the pairwise path, which has none.
        void *                              backend_comm_ctx = nullptr;

        // The chosen reduction, set once a path is agreed. Receives backend_comm_ctx, or
        // comm_state on the pairwise path. A communicator covers one device, so the caller
        // passes a one-element tensor array.
        ggml_backend_comm_allreduce_tensor_t allreduce_fn    = nullptr;
    };

    std::vector<ggml_backend_t> backends;
    const char * cache_dir;
    std::unordered_set<ggml_backend_buffer_t> buffers;
    // computed graphs cached per backend, keyed by uid
    std::vector<std::unordered_map<uint64_t, stored_graph>> stored_graphs;
    std::vector<comm_state> comm_states;
    // SET_TENSOR data is received in chunks through this buffer
    std::vector<uint8_t> set_tensor_buf;
    ggml_context_ptr req_ctx;
};

void rpc_server::hello(rpc_msg_hello_rsp & response) {
    response.major = RPC_PROTO_MAJOR_VERSION;
    response.minor = RPC_PROTO_MINOR_VERSION;
    response.patch = RPC_PROTO_PATCH_VERSION;
    response.flags = cache_dir ? RPC_DISPATCHER_FLAGS_SEND_TENSOR_HASH : 0;
    LOG_DBG("[%s] version: %d.%d.%d\n", __func__, response.major, response.minor, response.patch);
}

bool rpc_server::get_alloc_size(const rpc_msg_get_alloc_size_req & request, rpc_msg_get_alloc_size_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft;
    ggml_context * ctx = request_ctx();

    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr) {
        LOG_ERROR("Null tensor pointer passed to server get_alloc_size function.\n");
        return false;
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (request.srcs[i].id != 0) {
            tensor->src[i] = deserialize_tensor(ctx, &request.srcs[i]);
            if (tensor->src[i] == nullptr) {
                return false;
            }
        }
    }

    LOG_DBG2("[%s] device: %u, tensor: %s, buffer: %p, data: %p\n", __func__, dev_id, rpc_log_format_tensor(tensor).c_str(),
             (void*)tensor->buffer, tensor->data);
    if (tensor->buffer == nullptr) {
        //No buffer allocated.
        buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    } else {
        buft = tensor->buffer->buft;
    }

    response.alloc_size = ggml_backend_buft_get_alloc_size(buft, tensor);

    return true;
}

bool rpc_server::alloc_buffer(const rpc_msg_alloc_buffer_req & request, rpc_msg_alloc_buffer_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, request.size);
    response.remote_ptr = 0;
    response.remote_size = 0;
    if (buffer != nullptr) {
        response.remote_ptr = reinterpret_cast<uint64_t>(buffer);
        response.remote_size = buffer->size;
        LOG_DBG("[%s] device: %u, size: %s -> remote_ptr: 0x%" PRIx64 ", remote_size: %s\n",
                __func__, dev_id, rpc_log_format_size(request.size).c_str(), response.remote_ptr,
                rpc_log_format_size(response.remote_size).c_str());
        buffers.insert(buffer);
    } else {
        LOG_WARN("[%s] device: %u, size: %s -> failed\n", __func__, dev_id, rpc_log_format_size(request.size).c_str());
    }
    return true;
}

bool rpc_server::get_alignment(const rpc_msg_get_alignment_req & request, rpc_msg_get_alignment_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    size_t alignment = ggml_backend_buft_get_alignment(buft);
    LOG_DBG2("[%s] device: %u, alignment: %s\n", __func__, dev_id, rpc_log_format_size(alignment).c_str());
    response.alignment = alignment;
    return true;
}

bool rpc_server::get_max_size(const rpc_msg_get_max_size_req & request, rpc_msg_get_max_size_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    size_t max_size = ggml_backend_buft_get_max_size(buft);
    LOG_DBG2("[%s] device: %u, max_size: %s\n", __func__, dev_id, rpc_log_format_size(max_size).c_str());
    response.max_size = max_size;
    return true;
}

bool rpc_server::buffer_get_base(const rpc_msg_buffer_get_base_req & request, rpc_msg_buffer_get_base_rsp & response) {
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = find_buffer(request.remote_ptr);
    if (buffer == nullptr) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    void * base = ggml_backend_buffer_get_base(buffer);
    response.base_ptr = reinterpret_cast<uint64_t>(base);
    return true;
}

bool rpc_server::free_buffer(const rpc_msg_free_buffer_req & request) {
    sync_all_backends();
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = find_buffer(request.remote_ptr);
    if (buffer == nullptr) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    // Discard all cached graphs to avoid use-after-free in graph_recompute,
    // since their nodes may hold pointers to the buffer being freed. The client
    // will have to re-serialize its graphs from now on.
    for (auto & sgs : stored_graphs) {
        for (auto & sg : sgs) {
            sg.second.graph = nullptr;
        }
    }
    ggml_backend_buffer_free(buffer);
    buffers.erase(buffer);
    return true;
}

bool rpc_server::buffer_clear(const rpc_msg_buffer_clear_req & request) {
    sync_all_backends();
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 ", value: %u\n", __func__, request.remote_ptr, request.value);
    ggml_backend_buffer_t buffer = find_buffer(request.remote_ptr);
    if (buffer == nullptr) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    ggml_backend_buffer_clear(buffer, request.value);
    return true;
}

bool rpc_server::memset_tensor(const rpc_msg_memset_tensor_req & request) {
    sync_all_backends();
    ggml_tensor * tensor = deserialize_tensor(request_ctx(), &request.tensor);
    if (!rpc_check_region(__func__, tensor, request.offset, request.size)) {
        return false;
    }

    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, value: %u\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (void *) tensor->buffer, tensor->data, request.offset,
            rpc_log_format_size(request.size).c_str(), request.value);
    ggml_backend_tensor_memset(tensor, request.value, request.offset, request.size);
    return true;
}

// rejects the values that ggml_new_tensor or the backends would assert on
ggml_tensor * rpc_server::deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor) {
    if (tensor->type >= GGML_TYPE_COUNT || ggml_blck_size((ggml_type) tensor->type) == 0) {
        LOG_ERROR("[%s] invalid tensor type: %u\n", __func__, tensor->type);
        return nullptr;
    }
    const ggml_type type = (ggml_type) tensor->type;
    if (tensor->op >= GGML_OP_COUNT) {
        LOG_ERROR("[%s] invalid tensor op: %u\n", __func__, tensor->op);
        return nullptr;
    }
    if (tensor->ne[0] % ggml_blck_size(type) != 0) {
        LOG_ERROR("[%s] ne[0] = %u is not a multiple of the %s block size\n", __func__, tensor->ne[0], ggml_type_name(type));
        return nullptr;
    }
    // same element count check as ggml_new_tensor
    uint64_t nelements = tensor->ne[0];
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        if (tensor->ne[i] > 1) {
            if ((uint64_t) INT64_MAX / tensor->ne[i] <= nelements) {
                LOG_ERROR("[%s] element count overflows int64_t\n", __func__);
                return nullptr;
            }
            nelements *= tensor->ne[i];
        }
    }
    uint64_t nbytes;
    if (!rpc_tensor_nbytes(*tensor, nbytes)) {
        LOG_ERROR("[%s] tensor size overflows\n", __func__);
        return nullptr;
    }

    ggml_backend_buffer_t buffer = find_buffer(tensor->buffer);
    if (buffer && nbytes > 0 && !rpc_in_buffer(buffer, tensor->data, nbytes)) {
        LOG_ERROR("[%s] tensor '%.*s' (op %s, type %s) data 0x%" PRIx64 " with size %" PRIu64 " out of buffer bounds "
                  "(base %p, size %zu)\n", __func__, (int) sizeof(tensor->name), tensor->name, ggml_op_name((ggml_op) tensor->op),
                  ggml_type_name(type), tensor->data, nbytes, ggml_backend_buffer_get_base(buffer), ggml_backend_buffer_get_size(buffer));
        return nullptr;
    }

    ggml_tensor * result = ggml_new_tensor_4d(ctx, type, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = tensor->nb[i];
    }
    result->buffer = buffer;
    result->op = (ggml_op) tensor->op;
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result->op_params[i] = tensor->op_params[i];
    }
    result->flags = tensor->flags;
    result->data = reinterpret_cast<void *>(tensor->data);
    ggml_set_name(result, tensor->name);
    return result;
}


bool rpc_server::set_tensor(socket_ptr sock, uint64_t input_size) {
    sync_all_backends();
    // | rpc_msg_set_tensor_req | data (size bytes) |
    rpc_msg_set_tensor_req request;
    if (input_size < sizeof(request) || !sock->recv_data(&request, sizeof(request))) {
        return false;
    }
    const uint8_t  cache_flag = request.cache_flag;
    const uint64_t offset     = request.offset;
    const size_t   size       = input_size - sizeof(request);

    ggml_tensor * tensor = deserialize_tensor(request_ctx(), &request.tensor);
    if (!rpc_check_region(__func__, tensor, offset, size)) {
        return false;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (void*)tensor->buffer, tensor->data, offset,
            rpc_log_format_size(size).c_str());

    if (cache_dir && cache_flag) {
        // the whole data is needed to compute the hash
        std::vector<uint8_t> data(size);
        if (!sock->recv_data(data.data(), size)) {
            return false;
        }
        uint64_t hash = fnv_hash(data.data(), size);
        char hash_str[17];
        snprintf(hash_str, sizeof(hash_str), "%016" PRIx64, hash);
        // save to cache_dir/hash_str
        fs::path cache_file = fs::path(cache_dir) / hash_str;
        std::ofstream ofs(cache_file, std::ios::binary);
        ofs.write((const char *)data.data(), size);
        LOG_INFO("[%s] saved to '%s'\n", __func__, cache_file.string().c_str());
        ggml_backend_tensor_set(tensor, data.data(), offset, size);
        return true;
    }

    // a buffer of the tensor size would be allocated and page faulted for every tensor, which is slow
    const size_t chunk_size = 16*1024*1024;
    if (set_tensor_buf.size() < std::min(size, chunk_size)) {
        set_tensor_buf.resize(std::min(size, chunk_size));
    }
    for (size_t done = 0; done < size; ) {
        const size_t n = std::min(size - done, chunk_size);
        if (!sock->recv_data(set_tensor_buf.data(), n)) {
            return false;
        }
        ggml_backend_tensor_set(tensor, set_tensor_buf.data(), offset + done, n);
        done += n;
    }
    return true;
}

bool rpc_server::set_tensor_2d(const std::vector<uint8_t> & input) {
    sync_all_backends();
    // | rpc_msg_tensor_2d_req | data (size * n_copies bytes) |
    rpc_reader reader(input);
    rpc_msg_tensor_2d_req request;
    if (!reader.read(request)) {
        return false;
    }

    ggml_tensor * tensor = deserialize_tensor(request_ctx(), &request.tensor);
    if (!rpc_check_region(__func__, tensor, request.offset, request.size, request.n_copies, request.stride)) {
        return false;
    }
    const uint8_t * data;
    if (!reader.read_bytes(data, request.n_copies, request.size) || !reader.empty()) {
        LOG_ERROR("[%s] data size does not match size * n_copies\n", __func__);
        return false;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, n_copies: %" PRIu64 ", stride: %" PRIu64 "\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (void*)tensor->buffer, tensor->data, request.offset,
            rpc_log_format_size(request.size).c_str(), request.n_copies, request.stride);

    ggml_backend_tensor_set_2d(tensor, data, request.offset, request.size, request.n_copies, request.stride, request.size);
    return true;
}

bool rpc_server::get_cached_file(uint64_t hash, std::vector<uint8_t> & data) {
    if (!cache_dir) {
        return false;
    }
    char hash_str[17];
    snprintf(hash_str, sizeof(hash_str), "%016" PRIx64, hash);
    fs::path cache_file = fs::path(cache_dir) / hash_str;
    std::error_code ec;
    if (!fs::exists(cache_file, ec)) {
        return false;
    }
    std::ifstream ifs(cache_file, std::ios::binary);
    ifs.seekg(0, std::ios::end);
    size_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    data.resize(size);
    ifs.read((char *)data.data(), size);
    return true;
}

bool rpc_server::set_tensor_hash(const rpc_msg_set_tensor_hash_req & request, rpc_msg_set_tensor_hash_rsp & response)
{
    sync_all_backends();
    ggml_tensor * tensor = deserialize_tensor(request_ctx(), &request.tensor);
    if (!rpc_check_region(__func__, tensor, request.offset, 0)) {
        return false;
    }

    std::vector<uint8_t> cached_file;
    if (!get_cached_file(request.hash, cached_file)) {
        LOG_DBG2("[%s] tensor: %s, offset: %" PRIu64 ", hash: 0x%" PRIx64 ", cache miss\n",
                __func__, rpc_log_format_tensor(tensor).c_str(), request.offset, request.hash);
        response.result = 0;
        return true;
    }
    size_t size = cached_file.size();
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, hash: 0x%" PRIx64 ", cache hit\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (void*)tensor->buffer, tensor->data, request.offset,
            rpc_log_format_size(size).c_str(), request.hash);

    if (!rpc_check_region(__func__, tensor, request.offset, size)) {
        return false;
    }
    ggml_backend_tensor_set(tensor, cached_file.data(), request.offset, size);
    response.result = 1;
    return true;
}

bool rpc_server::init_tensor(const rpc_msg_init_tensor_req & request) {
    sync_all_backends();
    ggml_tensor * tensor = deserialize_tensor(request_ctx(), &request.tensor);
    if (tensor == nullptr) {
        LOG_ERROR("Null tensor pointer passed to server init_tensor function.\n");
        return false;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p\n", __func__, rpc_log_format_tensor(tensor).c_str(),
            (void*)tensor->buffer, tensor->data);
    // Call the backend's buffer_init_tensor function
    ggml_backend_buffer_t buffer = tensor->buffer;
    if (buffer && buffer->iface.init_tensor) {
        buffer->iface.init_tensor(buffer, tensor);
    } else {
        if (!buffer) {
            LOG_ERROR("Tensor with null buffer passed to init_tensor function\n");
        }
    }

    if (tensor->extra != nullptr) {
        // This pointer can either be passed around client/server, or probably better stored server-side and kept track of.
        // Currently unimplemented.
        LOG_ERROR("tensor->extra populated by the backend, this is currently unsupported.\n");
        return false;
    }

    return true;
}

bool rpc_server::get_tensor(const rpc_msg_get_tensor_req & request, std::vector<uint8_t> & response) {
    sync_all_backends();
    ggml_tensor * tensor = deserialize_tensor(request_ctx(), &request.tensor);
    if (!rpc_check_region(__func__, tensor, request.offset, request.size)) {
        return false;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), (void*)tensor->buffer, tensor->data, request.offset,
             rpc_log_format_size(request.size).c_str());

    response.resize(request.size, 0);
    ggml_backend_tensor_get(tensor, response.data(), request.offset, request.size);
    return true;
}

bool rpc_server::get_tensor_2d(const rpc_msg_tensor_2d_req & request, std::vector<uint8_t> & response) {
    sync_all_backends();
    ggml_tensor * tensor = deserialize_tensor(request_ctx(), &request.tensor);
    if (!rpc_check_region(__func__, tensor, request.offset, request.size, request.n_copies, request.stride)) {
        return false;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, n_copies: %" PRIu64 ", stride: %" PRIu64 "\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (void*)tensor->buffer, tensor->data, request.offset,
            rpc_log_format_size(request.size).c_str(), request.n_copies, request.stride);

    response.resize(request.size * request.n_copies, 0);
    ggml_backend_tensor_get_2d(tensor, response.data(), request.offset, request.size, request.n_copies, request.stride, request.size);
    return true;
}

bool rpc_server::copy_tensor(const rpc_msg_copy_tensor_req & request, rpc_msg_copy_tensor_rsp & response) {
    sync_all_backends();
    ggml_context * ctx = request_ctx();

    ggml_tensor * src = deserialize_tensor(ctx, &request.src);
    ggml_tensor * dst = deserialize_tensor(ctx, &request.dst);
    if (src == nullptr || src->buffer == nullptr || !rpc_check_region(__func__, dst, 0, ggml_nbytes(src))) {
        LOG_ERROR("[%s] invalid tensors\n", __func__);
        return false;
    }

    LOG_DBG2("[%s] src: %s -> dst: %s\n", __func__, rpc_log_format_tensor(src).c_str(), rpc_log_format_tensor(dst).c_str());

    response.result = ggml_backend_buffer_copy_tensor(src, dst);
    return true;
}

bool rpc_server::graph_compute(const std::vector<uint8_t> & input) {
    // | rpc_msg_graph_compute_req | nodes (n_nodes * uint64_t) | n_tensors (uint32_t) | tensors (n_tensors * rpc_tensor) |
    rpc_reader reader(input);
    rpc_msg_graph_compute_req request;
    const uint8_t * node_ids;
    uint32_t        n_tensors;
    const uint8_t * tensor_data;
    if (!reader.read(request) || !reader.read_bytes(node_ids, request.n_nodes, sizeof(uint64_t)) ||
        !reader.read(n_tensors) || !reader.read_bytes(tensor_data, n_tensors, sizeof(rpc_tensor)) || !reader.empty()) {
        LOG_ERROR("[%s] malformed request\n", __func__);
        return false;
    }
    const uint32_t device  = request.device;
    const uint64_t uid     = request.uid;
    const uint32_t n_nodes = request.n_nodes;
    if (device >= backends.size()) {
        return false;
    }
    const rpc_tensor * tensors = (const rpc_tensor *) tensor_data;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 ", n_nodes: %u, n_tensors: %u\n", __func__, device, uid, n_nodes, n_tensors);

    // graphs with uid == 0 are not cached, see GRAPH_CACHE_MAX for the eviction policy
    if (uid != 0 && stored_graphs[device].size() >= GRAPH_CACHE_MAX) {
        LOG_DBG("[%s] clearing graph cache for device %u (%zu entries)\n",
                __func__, device, stored_graphs[device].size());
        stored_graphs[device].clear();
    }
    stored_graph sg_tmp;
    stored_graph & sg = uid != 0 ? stored_graphs[device][uid] : sg_tmp;

    size_t buf_size = ggml_tensor_overhead()*(n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, false);
    if (sg.buffer.size() < buf_size) {
        sg.buffer.resize(buf_size);
    }
    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ sg.buffer.data(),
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, n_nodes, false);
    graph->n_nodes = n_nodes;

    // the compute itself is asynchronous, so building the graph is the only server-side cost
    // observable here - the round trip is timed on the client
    const auto t_deserialize = std::chrono::steady_clock::now();

    // tensors come in topological order (see add_tensor), so the graph is built in one pass and a source sent
    // after its user is rejected; a tensor may be its own source (ggml_cast)
    std::unordered_map<uint64_t, uint32_t> tensor_idx;
    tensor_idx.reserve(n_tensors);
    std::vector<ggml_tensor *> created(n_tensors);
    auto lookup = [&](uint64_t id, ggml_tensor ** out) {
        if (id == 0) {
            *out = nullptr;
            return true;
        }
        auto it = tensor_idx.find(id);
        if (it == tensor_idx.end()) {
            return false;
        }
        *out = created[it->second];
        return true;
    };
    for (uint32_t i = 0; i < n_tensors; i++) {
        const rpc_tensor & wire = tensors[i];
        ggml_tensor * result = deserialize_tensor(ctx, &wire);
        if (result == nullptr) {
            return false;
        }
        if (result->buffer == nullptr && result->data != nullptr) {
            LOG_ERROR("[%s] tensor %u (id=%" PRIu64 ") has a data pointer but no buffer\n", __func__, i, wire.id);
            return false;
        }
        if (wire.id == 0 || !tensor_idx.emplace(wire.id, i).second) {
            LOG_ERROR("[%s] tensor %u has a zero or duplicate id %" PRIu64 "\n", __func__, i, wire.id);
            return false;
        }
        created[i] = result;
        bool found = lookup(wire.view_src, &result->view_src);
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            found = found && lookup(wire.src[s], &result->src[s]);
        }
        if (!found) {
            LOG_ERROR("[%s] tensor %u (id=%" PRIu64 ") references a tensor that was not sent before it\n", __func__, i, wire.id);
            return false;
        }
        result->view_offs = wire.view_offs;
        if (!rpc_check_view(result)) {
            LOG_ERROR("[%s] view %u (id=%" PRIu64 ", view_offs=%" PRIu64 ") out of view_src bounds\n", __func__, i, wire.id, wire.view_offs);
            return false;
        }
    }
    for (uint32_t i = 0; i < n_nodes; i++) {
        uint64_t id;
        memcpy(&id, node_ids + i*sizeof(id), sizeof(id));
        auto it = tensor_idx.find(id);
        if (it == tensor_idx.end()) {
            LOG_ERROR("[%s] graph node %u (id=%" PRIu64 ") is not in the tensor list\n", __func__, i, id);
            return false;
        }
        ggml_tensor * node = created[it->second];
        if (!rpc_check_node(node)) {
            LOG_ERROR("[%s] graph node %u (%s) or one of its sources is not allocated on this server\n", __func__, i, node->name);
            return false;
        }
        const size_t hash_pos = ggml_hash_insert(&graph->visited_hash_set, node);
        if (hash_pos == GGML_HASHSET_ALREADY_EXISTS) {
            LOG_ERROR("[%s] graph node %u (id=%" PRIu64 ") is a duplicate\n", __func__, i, id);
            return false;
        }
        graph->nodes[i] = node;
        graph->use_counts[hash_pos] = tensors[it->second].use_count;
        LOG_DBG3("[%s] node %u: id=%" PRIu64 ", %s\n", __func__, i, id, rpc_log_format_tensor(node).c_str());
    }
    const std::chrono::duration<double, std::milli> deserialize_ms = std::chrono::steady_clock::now() - t_deserialize;

    ggml_status status = ggml_backend_graph_compute_async(backends[device], graph);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    sg.graph = graph;
    LOG_DBG3("[%s] device: %u, graph build: %.3f ms\n", __func__, device, deserialize_ms.count());
    return true;
}

bool rpc_server::graph_recompute(const rpc_msg_graph_recompute_req & request) {
    uint32_t device = request.device;
    if (device >= backends.size()) {
        return false;
    }
    auto it = stored_graphs[device].find(request.uid);
    if (it == stored_graphs[device].end() || it->second.graph == nullptr) {
        LOG_ERROR("[%s] device: %u, graph with uid %" PRIu64 " not found\n", __func__, device, request.uid);
        return false;
    }
    ggml_cgraph * graph = it->second.graph;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 "\n", __func__, device, request.uid);
    ggml_status status = ggml_backend_graph_compute_async(backends[device], graph);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    return true;
}

// graph compute is asynchronous; commands that read or write buffer data synchronize first
void rpc_server::sync_all_backends() {
    for (ggml_backend_t backend : backends) {
        ggml_backend_synchronize(backend);
    }
}

void rpc_server::comm_free_device(uint32_t dev_id) {
    comm_state & state = comm_states[dev_id];
    if (state.backend_comm_ctx != nullptr) {
        auto comm_free = (ggml_backend_comm_free_t) ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(ggml_backend_get_device(backends[dev_id])), "ggml_backend_comm_free");
        if (comm_free != nullptr) {
            comm_free(state.backend_comm_ctx);
        }
    }
    state = comm_state();
}

// Exchange RPC transport capabilities over a peer connection. recv_first selects which side
// receives first. Both sides receiving first would deadlock.
static bool comm_caps_handshake(const socket_ptr & peer, bool recv_first) {
    uint8_t local_caps[RPC_CONN_CAPS_SIZE]  = {};
    uint8_t remote_caps[RPC_CONN_CAPS_SIZE] = {};
    peer->get_caps(local_caps);
    if (recv_first) {
        if (!peer->recv_data(remote_caps, sizeof(remote_caps)) ||
            !peer->send_data(local_caps, sizeof(local_caps))) {
            return false;
        }
    } else {
        if (!peer->send_data(local_caps, sizeof(local_caps)) ||
            !peer->recv_data(remote_caps, sizeof(remote_caps))) {
            return false;
        }
    }
    peer->update_caps(remote_caps);
    return true;
}

// The comm link between servers uses the same caps negotiation as the client HELLO,
// so it gets the same transport upgrades (e.g. RDMA).
bool rpc_server::comm_init(const rpc_msg_comm_init_req & request, rpc_msg_comm_init_rsp & response) {
    response.ok = 0;
    if (request.device >= backends.size() || request.world < 2 || request.rank >= request.world) {
        return true;
    }
    const uint32_t dev_id = request.device;
    comm_state & state = comm_states[dev_id];
    if (state.allreduce_fn != nullptr) {
        response.ok = 1;
        return true;
    }

    // Joining a communicator blocks until every rank arrives, so drain outstanding work first.
    sync_all_backends();

    // Rank 0 listens and accepts world-1 peers; every other rank connects back to it.
    if (request.rank == 0) {
        socket_ptr srv = socket_t::create_server("0.0.0.0", request.port);
        if (srv == nullptr) {
            LOG_ERROR("[%s] failed to listen on comm port %u\n", __func__, request.port);
            return true;
        }
        // Store peers[i] as rank i + 1, as comm_bootstrap_exchange needs.
        // Each peer sends its rank because peers may connect in any order.
        state.peers.resize(request.world - 1);
        for (uint32_t i = 1; i < request.world; i++) {
            socket_ptr peer = srv->accept();
            uint32_t peer_rank = 0;
            if (peer == nullptr || !comm_caps_handshake(peer, /*recv_first =*/ true) ||
                !peer->recv_data(&peer_rank, sizeof(peer_rank)) ||
                peer_rank == 0 || peer_rank >= request.world || state.peers[peer_rank - 1] != nullptr) {
                LOG_ERROR("[%s] bad peer connection (rank %u)\n", __func__, peer_rank);
                state.peers.clear();
                return true;
            }
            state.peers[peer_rank - 1] = std::move(peer);
        }
    } else {
        const std::string host(request.host, strnlen(request.host, sizeof(request.host)));
        socket_ptr peer;
        // rank 0 may not be listening yet, retry for a few seconds
        for (int i = 0; i < 100 && peer == nullptr; i++) {
            peer = socket_t::connect(host.c_str(), request.port);
            if (peer == nullptr) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }
        if (peer == nullptr) {
            LOG_ERROR("[%s] failed to connect to peer %s:%u\n", __func__, host.c_str(), request.port);
            return true;
        }
        if (!comm_caps_handshake(peer, /*recv_first =*/ false) ||
            !peer->send_data(&request.rank, sizeof(request.rank)) || !peer->flush()) {
            return true;
        }
        state.peers.push_back(std::move(peer));
    }

    state.rank  = request.rank;
    state.world = request.world;

    // Forces the pairwise path, for comparing the two on one build. Setting it on any one rank
    // is enough, because the negotiation spreads the decision.
    const bool no_backend_communicator = std::getenv("GGML_RPC_NO_BACKEND_COMMUNICATOR") != nullptr;

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backends[dev_id]));
    auto init_rank    = (ggml_backend_comm_init_rank_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_init_rank");
    auto allreduce_fn = (ggml_backend_comm_allreduce_tensor_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_allreduce_tensor");

    // The ranks must agree on backend communicator or pairwise. Every rank shares whether its
    // backend provides a communicator, and all ranks use it only if every rank can.
    const uint8_t self_can_use_backend_comm = !no_backend_communicator && init_rank != nullptr && allreduce_fn != nullptr;
    std::vector<uint8_t> all_can_use_backend_comm(request.world);
    if (!comm_bootstrap_exchange(&state, &self_can_use_backend_comm, all_can_use_backend_comm.data(),
                                 sizeof(self_can_use_backend_comm))) {
        state.peers.clear();
        return true;
    }
    const bool use_backend_communicator = std::all_of(all_can_use_backend_comm.begin(), all_can_use_backend_comm.end(),
                                                      [](uint8_t c) { return c != 0; });

    if (use_backend_communicator) {
        void * backend_ctx = init_rank(backends[dev_id], (int) request.rank, (int) request.world, comm_bootstrap_exchange, &state);
        if (backend_ctx == nullptr) {
            LOG_ERROR("[%s] failed to join communicator as rank %u/%u\n", __func__, request.rank, request.world);
            state.peers.clear();
            return true;
        }
        state.backend_comm_ctx = backend_ctx;
        state.allreduce_fn     = allreduce_fn;
        // The sockets have done their job; the collective runs between the backends from here on.
        state.peers.clear();
        LOG_INFO("[%s] device %u using backend communicator, rank %u/%u\n", __func__, dev_id, request.rank, request.world);
        response.ok = 1;
        return true;
    }

    // No backend communicator, so fall back to the pairwise exchange, which only covers two ranks.
    if (request.world != 2) {
        LOG_ERROR("[%s] world size %u requires a backend communicator\n", __func__, request.world);
        state.peers.clear();
        return true;
    }
    state.pairwise_backend = backends[dev_id];
    state.allreduce_fn     = comm_allreduce_pairwise;
    LOG_INFO("[%s] device %u using pairwise communicator, rank %u\n", __func__, dev_id, request.rank);
    response.ok = 1;
    return true;
}

// Bootstrap callback for ggml_backend_comm_init_rank_t, over the peer sockets. Rank 0 collects
// size bytes from every rank in rank order, then sends the whole array back to every rank.
// user_data is the comm_state.
bool rpc_server::comm_bootstrap_exchange(void * user_data, const void * send_buf, void * recv_buf, size_t size) {
    comm_state & state = *(comm_state *) user_data;
    uint8_t * recv = (uint8_t *) recv_buf;
    const size_t total = (size_t) state.world * size;

    if (state.rank == 0) {
        memcpy(recv, send_buf, size);
        for (size_t i = 0; i < state.peers.size(); i++) {
            if (!state.peers[i]->recv_data(recv + (i + 1) * size, size)) {
                return false;
            }
        }
        for (const socket_ptr & peer : state.peers) {
            if (!peer->send_data(recv, total) || !peer->flush()) {
                return false;
            }
        }
        return true;
    }
    return state.peers[0]->send_data(send_buf, size) && state.peers[0]->flush() &&
           state.peers[0]->recv_data(recv, total);
}

// Reduces this rank's contribution in place, exchanging partials with the peer over peers[0].
// Used when the backend provides no communicator. comm_ctx_v is the comm_state.
bool rpc_server::comm_allreduce_pairwise(void * comm_ctx_v, ggml_tensor ** tensors) {
    comm_state & state = *(comm_state *) comm_ctx_v;
    ggml_tensor * t_dst = tensors[0];
    ggml_backend_t backend = state.pairwise_backend;

    const size_t  nbytes = ggml_nbytes(t_dst);
    const int64_t ne     = ggml_nelements(t_dst);

    size_t ctx_size = 16*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(8, false);
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    // reduce large partials in bf16 to halve the wire bytes; small (decode-sized) ones
    // stay f32 since the extra casts and sync cost more than the bytes saved
    const bool   wire_bf16  = t_dst->type == GGML_TYPE_F32 && ne >= 32768;
    const size_t wire_bytes = wire_bf16 ? (size_t) ne*2 : nbytes;
    const size_t need       = wire_bf16 ? 2*nbytes : nbytes;
    if (state.scratch_size < need) {
        state.scratch.reset(ggml_backend_alloc_buffer(backend, need));
        state.scratch_size = need;
    }
    char * scratch_base = (char *) ggml_backend_buffer_get_base(state.scratch.get());
    state.send_buf.resize(wire_bytes);
    state.recv_buf.resize(wire_bytes);

    auto new_scratch_tensor = [&](ggml_type type, size_t offset) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, type, t_dst->ne[0], t_dst->ne[1], t_dst->ne[2], t_dst->ne[3]);
        t->buffer = state.scratch.get();
        t->data   = scratch_base + offset;
        return t;
    };
    auto new_cpy_node = [&](ggml_tensor * src, ggml_tensor * dst) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, dst->type, dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
        t->op     = GGML_OP_CPY;
        t->src[0] = src;
        t->src[1] = dst;
        t->buffer = dst->buffer;
        t->data   = dst->data;
        t->flags |= GGML_TENSOR_FLAG_COMPUTE;
        return t;
    };
    auto compute_nodes = [&](ggml_tensor * n0, ggml_tensor * n1) {
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 2, false);
        graph->nodes[0] = n0;
        graph->nodes[1] = n1;
        graph->n_nodes  = n1 != nullptr ? 2 : 1;
        ggml_status status = ggml_backend_graph_compute_async(backend, graph);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    };

    // wait for the pending subgraph that produced this partial
    ggml_backend_synchronize(backend);

    ggml_tensor * t_wire_send = nullptr;
    ggml_tensor * t_wire_recv = nullptr;
    if (wire_bf16) {
        t_wire_send = new_scratch_tensor(GGML_TYPE_BF16, 0);
        t_wire_recv = new_scratch_tensor(GGML_TYPE_BF16, ne*2);
        compute_nodes(new_cpy_node(t_dst, t_wire_send), nullptr);
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(t_wire_send, state.send_buf.data(), 0, wire_bytes);
    } else {
        ggml_backend_tensor_get(t_dst, state.send_buf.data(), 0, wire_bytes);
    }

    // rank 0 sends first, rank 1 receives first, so large payloads cannot deadlock
    if (state.rank == 0) {
        if (!state.peers[0]->send_data(state.send_buf.data(), wire_bytes) || !state.peers[0]->flush() ||
            !state.peers[0]->recv_data(state.recv_buf.data(), wire_bytes)) {
            return false;
        }
    } else {
        if (!state.peers[0]->recv_data(state.recv_buf.data(), wire_bytes) ||
            !state.peers[0]->send_data(state.send_buf.data(), wire_bytes) || !state.peers[0]->flush()) {
            return false;
        }
    }

    ggml_tensor * t_peer = new_scratch_tensor(t_dst->type, wire_bf16 ? (size_t) ne*4 : 0);
    ggml_tensor * t_cast = nullptr;
    if (wire_bf16) {
        ggml_backend_tensor_set(t_wire_recv, state.recv_buf.data(), 0, wire_bytes);
        t_cast = new_cpy_node(t_wire_recv, t_peer);
    } else {
        ggml_backend_tensor_set(t_peer, state.recv_buf.data(), 0, wire_bytes);
    }

    ggml_tensor * t_red = ggml_new_tensor_4d(ctx, t_dst->type, t_dst->ne[0], t_dst->ne[1], t_dst->ne[2], t_dst->ne[3]);
    t_red->op     = GGML_OP_ADD;
    t_red->src[0] = t_dst;
    t_red->src[1] = t_peer;
    t_red->buffer = t_dst->buffer;
    t_red->data   = t_dst->data;
    t_red->flags |= GGML_TENSOR_FLAG_COMPUTE;

    if (t_cast != nullptr) {
        compute_nodes(t_cast, t_red);
    } else {
        compute_nodes(t_red, nullptr);
    }
    return true;
}

bool rpc_server::comm_allreduce(const rpc_msg_comm_allreduce_req & request) {
    if (request.device >= backends.size()) {
        return false;
    }
    comm_state & state = comm_states[request.device];
    if (state.allreduce_fn == nullptr) {
        LOG_ERROR("[%s] no communicator for device %u\n", __func__, request.device);
        return false;
    }

    ggml_tensor * t_dst = deserialize_tensor(request_ctx(), &request.tensor);
    if (t_dst == nullptr || t_dst->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    if (ggml_nbytes(t_dst) == 0) {
        return true;
    }

    // Neither path needs a wait here. The pairwise path waits internally, and the backend orders
    // its reduction after the work that produced the partial sum.
    void * comm_ctx = state.backend_comm_ctx != nullptr ? state.backend_comm_ctx : &state;
    if (!state.allreduce_fn(comm_ctx, &t_dst)) {
        LOG_ERROR("[%s] allreduce failed for tensor %s\n", __func__, t_dst->name);
        return false;
    }
    return true;
}

bool rpc_server::comm_free(const rpc_msg_comm_free_req & request) {
    if (request.device >= backends.size()) {
        return false;
    }
    // Wait for outstanding work. Destroying a communicator while its kernels are still running
    // is unsafe.
    sync_all_backends();
    comm_free_device(request.device);
    return true;
}

bool rpc_server::get_device_memory(const rpc_msg_get_device_memory_req & request, rpc_msg_get_device_memory_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    size_t free, total;
    ggml_backend_dev_t dev = ggml_backend_get_device(backends[dev_id]);
    ggml_backend_dev_memory(dev, &free, &total);
    response.free_mem = free;
    response.total_mem = total;
    LOG_DBG2("[%s] device: %u, free: %s, total: %s\n", __func__, dev_id,
             rpc_log_format_size(response.free_mem).c_str(), rpc_log_format_size(response.total_mem).c_str());
    return true;
}

rpc_server::~rpc_server() {
    // Wait for outstanding work. Destroying a communicator while its kernels are still running
    // is unsafe.
    sync_all_backends();

    // Free any communicator still open. A client can disconnect without sending COMM_FREE.
    for (uint32_t dev_id = 0; dev_id < comm_states.size(); dev_id++) {
        comm_free_device(dev_id);
    }
    for (auto buffer : buffers) {
        ggml_backend_buffer_free(buffer);
    }
}

template <typename T>
static bool rpc_send_rsp(const socket_ptr & sock, const T & response) {
    return send_msg(sock, &response, sizeof(response));
}

static bool rpc_send_rsp(const socket_ptr & sock, const std::vector<uint8_t> & response) {
    return send_msg(sock, response.data(), response.size());
}

// receives the request of `cmd`, runs the handler and sends the response, if the command has one
template <rpc_cmd cmd, typename handler_t>
static bool rpc_serve(const socket_ptr & sock, rpc_server & server, handler_t handler) {
    using req_t = typename rpc_msg_types<cmd>::req;
    using rsp_t = typename rpc_msg_types<cmd>::rsp;
    req_t request;
    if (!recv_msg(sock, &request, sizeof(request))) {
        return false;
    }
    if constexpr (std::is_void_v<rsp_t>) {
        return (server.*handler)(request);
    } else {
        rsp_t response = {};
        return (server.*handler)(request, response) && rpc_send_rsp(sock, response);
    }
}

static void rpc_serve_client(const std::vector<ggml_backend_t> & backends, const char * cache_dir,
                             socket_ptr sock) {
    rpc_server server(backends, cache_dir);
    uint8_t cmd;
    if (!sock->recv_data(&cmd, 1)) {
        return;
    }
    if (cmd != RPC_CMD_HELLO) {
        LOG_ERROR("Expected HELLO command, update client\n");
        return;
    }

    // Read input_size and validate protocol version
    uint64_t hello_input_size;
    if (!sock->recv_data(&hello_input_size, sizeof(hello_input_size))) {
        return;
    }

    if (hello_input_size != sizeof(rpc_msg_hello_req)) {
        LOG_ERROR("HELLO request size mismatch (%zu vs %zu) — client needs upgrade to protocol v%d.x\n",
                       (size_t)hello_input_size, sizeof(rpc_msg_hello_req), RPC_PROTO_MAJOR_VERSION);
        return;
    }

    rpc_msg_hello_req req = {};
    if (!sock->recv_data(&req, sizeof(req))) {
        return;
    }

    rpc_msg_hello_rsp rsp = {};
    server.hello(rsp);
    // Advertise server transport capabilities based on client's caps
    sock->get_caps(rsp.conn_caps);
    if (!send_msg(sock, &rsp, sizeof(rsp))) {
        return;
    }

    // Activate transport upgrade using client's caps
    sock->update_caps(req.conn_caps);
    LOG_DBG("[%s] client connected, transport: %s\n", __func__, sock->transport_name());
    std::vector<uint8_t> input;
    while (true) {
        if (!sock->recv_data(&cmd, 1)) {
            break;
        }
        if (cmd >= RPC_CMD_COUNT) {
            // fail fast if the command is invalid
            LOG_ERROR("Unknown command: %d\n", cmd);
            break;
        }
        LOG_DBG2("[%s] %s\n", __func__, rpc_cmd_name((enum rpc_cmd)cmd));
        bool ok = false;
        try {
            switch (cmd) {
                case RPC_CMD_DEVICE_COUNT: {
                    const rpc_msg_device_count_rsp response = { (uint32_t) backends.size() };
                    ok = recv_msg(sock, nullptr, 0) && send_msg(sock, &response, sizeof(response));
                    break;
                }
                case RPC_CMD_SET_TENSOR: {
                    // the handler streams the data from the socket
                    uint64_t input_size;
                    ok = sock->recv_data(&input_size, sizeof(input_size)) && server.set_tensor(sock, input_size);
                    break;
                }
                case RPC_CMD_SET_TENSOR_2D:     ok = recv_msg(sock, input) && server.set_tensor_2d(input); break;
                case RPC_CMD_GRAPH_COMPUTE:     ok = recv_msg(sock, input) && server.graph_compute(input); break;
                case RPC_CMD_ALLOC_BUFFER:      ok = rpc_serve<RPC_CMD_ALLOC_BUFFER>     (sock, server, &rpc_server::alloc_buffer);      break;
                case RPC_CMD_GET_ALIGNMENT:     ok = rpc_serve<RPC_CMD_GET_ALIGNMENT>    (sock, server, &rpc_server::get_alignment);     break;
                case RPC_CMD_GET_MAX_SIZE:      ok = rpc_serve<RPC_CMD_GET_MAX_SIZE>     (sock, server, &rpc_server::get_max_size);      break;
                case RPC_CMD_GET_ALLOC_SIZE:    ok = rpc_serve<RPC_CMD_GET_ALLOC_SIZE>   (sock, server, &rpc_server::get_alloc_size);    break;
                case RPC_CMD_GET_DEVICE_MEMORY: ok = rpc_serve<RPC_CMD_GET_DEVICE_MEMORY>(sock, server, &rpc_server::get_device_memory); break;
                case RPC_CMD_BUFFER_GET_BASE:   ok = rpc_serve<RPC_CMD_BUFFER_GET_BASE>  (sock, server, &rpc_server::buffer_get_base);   break;
                case RPC_CMD_FREE_BUFFER:       ok = rpc_serve<RPC_CMD_FREE_BUFFER>      (sock, server, &rpc_server::free_buffer);       break;
                case RPC_CMD_BUFFER_CLEAR:      ok = rpc_serve<RPC_CMD_BUFFER_CLEAR>     (sock, server, &rpc_server::buffer_clear);      break;
                case RPC_CMD_INIT_TENSOR:       ok = rpc_serve<RPC_CMD_INIT_TENSOR>      (sock, server, &rpc_server::init_tensor);       break;
                case RPC_CMD_MEMSET_TENSOR:     ok = rpc_serve<RPC_CMD_MEMSET_TENSOR>    (sock, server, &rpc_server::memset_tensor);     break;
                case RPC_CMD_SET_TENSOR_HASH:   ok = rpc_serve<RPC_CMD_SET_TENSOR_HASH>  (sock, server, &rpc_server::set_tensor_hash);   break;
                case RPC_CMD_GET_TENSOR:        ok = rpc_serve<RPC_CMD_GET_TENSOR>       (sock, server, &rpc_server::get_tensor);        break;
                case RPC_CMD_GET_TENSOR_2D:     ok = rpc_serve<RPC_CMD_GET_TENSOR_2D>    (sock, server, &rpc_server::get_tensor_2d);     break;
                case RPC_CMD_COPY_TENSOR:       ok = rpc_serve<RPC_CMD_COPY_TENSOR>      (sock, server, &rpc_server::copy_tensor);       break;
                case RPC_CMD_GRAPH_RECOMPUTE:   ok = rpc_serve<RPC_CMD_GRAPH_RECOMPUTE>  (sock, server, &rpc_server::graph_recompute);   break;
                case RPC_CMD_COMM_INIT:         ok = rpc_serve<RPC_CMD_COMM_INIT>        (sock, server, &rpc_server::comm_init);         break;
                case RPC_CMD_COMM_ALLREDUCE:    ok = rpc_serve<RPC_CMD_COMM_ALLREDUCE>   (sock, server, &rpc_server::comm_allreduce);    break;
                case RPC_CMD_COMM_FREE:         ok = rpc_serve<RPC_CMD_COMM_FREE>        (sock, server, &rpc_server::comm_free);         break;
                default:
                    // HELLO is only valid as the first command
                    LOG_ERROR("Unhandled command: %s\n", rpc_cmd_name((enum rpc_cmd)cmd));
                    break;
            }
        } catch (const std::bad_alloc &) {
            // allocation sizes come from the client, e.g. a large GET_TENSOR response
            LOG_ERROR("[%s] %s: out of memory\n", __func__, rpc_cmd_name((enum rpc_cmd)cmd));
            ok = false;
        }
        if (!ok) {
            return;
        }
    }
}

void ggml_backend_rpc_start_server(const char * endpoint, const char * cache_dir,
                                   size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices) {
    if (n_devices == 0 || devices == nullptr) {
        LOG_ERROR("Invalid arguments to ggml_backend_rpc_start_server\n");
        return;
    }
    std::vector<ggml_backend_t> backends;
    LOG_INFO("Starting RPC server v%d.%d.%d\n",
        RPC_PROTO_MAJOR_VERSION,
        RPC_PROTO_MINOR_VERSION,
        RPC_PROTO_PATCH_VERSION);
    LOG_INFO("  endpoint       : %s\n", endpoint);
    LOG_INFO("  local cache    : %s\n", cache_dir ? cache_dir : "n/a");
    LOG_INFO("Devices:\n");
    for (size_t i = 0; i < n_devices; i++) {
        auto dev = devices[i];
        size_t free, total;
        ggml_backend_dev_memory(dev, &free, &total);
        LOG_INFO("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
                 total / 1024 / 1024, free / 1024 / 1024);
        auto backend = ggml_backend_dev_init(dev, nullptr);
        if (!backend) {
            LOG_ERROR("Failed to create backend for device %s\n", dev->iface.get_name(dev));
            return;
        }
        backends.push_back(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (reg) {
            auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            if (ggml_backend_set_n_threads_fn) {
                ggml_backend_set_n_threads_fn(backend, n_threads);
            }
        }
    }

    std::string host;
    int port;
    if (!parse_endpoint(endpoint, host, port)) {
        return;
    }

#ifdef GGML_RPC_RDMA
    LOG_INFO("  transport      : TCP (RDMA auto-negotiate enabled)\n");
#else
    LOG_INFO("  transport      : TCP\n");
#endif // GGML_RPC_RDMA
    if (!rpc_transport_init()) {
        LOG_ERROR("Failed to initialize RPC transport\n");
        return;
    }
    auto server_socket = socket_t::create_server(host.c_str(), port);
    if (server_socket == nullptr) {
        LOG_ERROR("Failed to create server socket\n");
        return;
    }
    while (true) {
        auto client_socket = server_socket->accept();
        if (client_socket == nullptr) {
            LOG_ERROR("Failed to accept client connection\n");
            return;
        }
        LOG_INFO("Accepted client connection\n");
        rpc_serve_client(backends, cache_dir, client_socket);
        LOG_INFO("Client connection closed\n");
    }
    rpc_transport_shutdown();
    for (auto backend : backends) {
        ggml_backend_free(backend);
    }
}

static const char * ggml_backend_rpc_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ctx->name.c_str();
}

static const char * ggml_backend_rpc_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ctx->description.c_str();
}

static void ggml_backend_rpc_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ggml_backend_rpc_get_device_memory(ctx->endpoint.c_str(), ctx->device, free, total);
}

static enum ggml_backend_dev_type ggml_backend_rpc_device_get_type(ggml_backend_dev_t dev) {
    // TODO: obtain value from the server
    return GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_rpc_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_rpc_device_get_name(dev);
    props->description = ggml_backend_rpc_device_get_description(dev);
    props->type        = ggml_backend_rpc_device_get_type(dev);
    ggml_backend_rpc_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ true,
        /* .mmap_support          = */ true,
        /* .fast_async_host_input = */ false,
    };
}

static ggml_backend_t ggml_backend_rpc_device_init(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ggml_backend_rpc_init(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_rpc_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ggml_backend_rpc_buffer_type(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(dev);
}

static bool ggml_backend_rpc_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);
    GGML_UNUSED(op);
    //TODO: call the remote backend and cache the results
    return true;
}

static bool ggml_backend_rpc_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (!buft || buft->iface.get_name != ggml_backend_rpc_buffer_type_name) {
        return false;
    }
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    ggml_backend_rpc_device_context * dev_ctx = (ggml_backend_rpc_device_context *)dev->context;
    return buft_ctx->endpoint == dev_ctx->endpoint && buft_ctx->device == dev_ctx->device;
}

static ggml_backend_event_t ggml_backend_rpc_device_event_new(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    auto dispatcher = get_dispatcher(ctx->endpoint);
    if (dispatcher == nullptr) {
        return nullptr;
    }
    return dispatcher->event_new(dev);
}

static void ggml_backend_rpc_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    auto dispatcher = get_dispatcher(ctx->endpoint);
    if (dispatcher == nullptr) {
        return;
    }
    dispatcher->event_free(event);
}

static void ggml_backend_rpc_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    auto dispatcher = get_dispatcher(ctx->endpoint);
    if (dispatcher == nullptr) {
        return;
    }
    dispatcher->event_synchronize(event);
}

static const struct ggml_backend_device_i ggml_backend_rpc_device_i = {
    /* .get_name             = */ ggml_backend_rpc_device_get_name,
    /* .get_description      = */ ggml_backend_rpc_device_get_description,
    /* .get_memory           = */ ggml_backend_rpc_device_get_memory,
    /* .get_type             = */ ggml_backend_rpc_device_get_type,
    /* .get_props            = */ ggml_backend_rpc_device_get_props,
    /* .init_backend         = */ ggml_backend_rpc_device_init,
    /* .get_buffer_type      = */ ggml_backend_rpc_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_rpc_device_supports_op,
    /* .supports_buft        = */ ggml_backend_rpc_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ ggml_backend_rpc_device_event_new,
    /* .event_free           = */ ggml_backend_rpc_device_event_free,
    /* .event_synchronize    = */ ggml_backend_rpc_device_event_synchronize,
};

// backend reg interface

struct ggml_backend_rpc_reg_context {
    std::string                     name;
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_rpc_reg_get_name(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    return ctx ? ctx->name.c_str() : "RPC";
}

static size_t ggml_backend_rpc_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    return ctx ? ctx->devices.size() : 0;
}

static ggml_backend_dev_t ggml_backend_rpc_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    if (ctx == nullptr) {
        GGML_ABORT("The RPC backend does not have enumerated devices - use ggml_backend_rpc_add_server instead");
    } else {
        GGML_ASSERT(index < ctx->devices.size());
        return ctx->devices[index];
    }
}

// Pairwise allreduce between two RPC servers over a direct server-to-server connection.
// The client only sends fire-and-forget COMM_ALLREDUCE commands; the tensor data is
// exchanged between the servers and never passes through the client.
struct ggml_backend_rpc_comm_context {
    struct rank_info {
        std::string                     endpoint;
        uint32_t                        device;
        std::shared_ptr<rpc_dispatcher> dispatcher;
    };
    std::vector<rank_info> ranks;
};

static void ggml_backend_rpc_comm_free(void * comm_ctx_v) {
    ggml_backend_rpc_comm_context * comm_ctx = (ggml_backend_rpc_comm_context *) comm_ctx_v;
    if (comm_ctx == nullptr) {
        return;
    }
    for (const auto & rank : comm_ctx->ranks) {
        rank.dispatcher->call<RPC_CMD_COMM_FREE>({ rank.device });
        rank.dispatcher->busy_spin_release();
    }
    delete comm_ctx;
}

static void * ggml_backend_rpc_comm_init(ggml_backend_t * backends, size_t n_backends) {
    if (n_backends < 2 || std::getenv("GGML_RPC_NO_COMM") != nullptr) {
        return nullptr;
    }
    std::vector<ggml_backend_rpc_comm_context::rank_info> ranks;
    ranks.reserve(n_backends);
    for (size_t i = 0; i < n_backends; i++) {
        if (!ggml_backend_is_rpc(backends[i])) {
            return nullptr;
        }
        ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *) backends[i]->context;
        // one rank per endpoint: a server processes its socket sequentially, so a second
        // COMM_INIT on the same connection would deadlock behind the first
        for (const auto & rank : ranks) {
            if (rank.endpoint == rpc_ctx->endpoint) {
                LOG_WARN("%s: multiple ranks on endpoint %s are not supported\n", __func__, rpc_ctx->endpoint.c_str());
                return nullptr;
            }
        }
        ranks.push_back({rpc_ctx->endpoint, rpc_ctx->device, rpc_ctx->dispatcher});
    }

    // TODO: simplify this logic
    // rank 1 connects to rank 0 on its serving host; endpoints must be mutually reachable
    // (e.g. do not bind the servers to 127.0.0.1 when they run on different machines)
    std::string host0;
    int port0;
    if (!parse_endpoint(ranks[0].endpoint, host0, port0)) {
        return nullptr;
    }
    const uint32_t comm_port = (uint32_t) port0 + 1000;
    if (host0.size() >= 64) {
        return nullptr;
    }

    for (const auto & rank : ranks) {
        rank.dispatcher->busy_spin_acquire();
    }

    // Send all init requests before reading any response: rank 0 blocks in accept
    // until rank 1 has connected.
    std::vector<rpc_msg_comm_init_rsp> responses(n_backends);
    for (size_t i = 0; i < n_backends; i++) {
        rpc_msg_comm_init_req request = {};
        request.device = ranks[i].device;
        request.rank   = (uint32_t) i;
        request.world  = (uint32_t) n_backends;
        request.port   = comm_port;
        if (i > 0) {
            memcpy(request.host, host0.c_str(), host0.size());
        }
        ranks[i].dispatcher->call_async<RPC_CMD_COMM_INIT>(request, &responses[i]);
    }
    for (size_t i = 0; i < n_backends; i++) {
        ranks[i].dispatcher->synchronize();
    }
    bool ok = true;
    for (size_t i = 0; i < n_backends; i++) {
        if (!responses[i].ok) {
            LOG_WARN("%s: rank %zu (%s) failed to initialize\n", __func__, i, ranks[i].endpoint.c_str());
            ok = false;
        }
    }
    if (!ok) {
        for (const auto & rank : ranks) {
            rank.dispatcher->busy_spin_release();
        }
        return nullptr;
    }
    LOG_INFO("%s: pairwise communicator initialized (%s <-> %s)\n", __func__,
             ranks[0].endpoint.c_str(), ranks[1].endpoint.c_str());
    return new ggml_backend_rpc_comm_context{std::move(ranks)};
}

static bool ggml_backend_rpc_comm_allreduce_tensor(void * comm_ctx_v, ggml_tensor ** tensors) {
    ggml_backend_rpc_comm_context * comm_ctx = (ggml_backend_rpc_comm_context *) comm_ctx_v;
    if (comm_ctx == nullptr) {
        return false;
    }
    const size_t n_ranks = comm_ctx->ranks.size();
    const int64_t ne = ggml_nelements(tensors[0]);
    if (ne == 0) {
        return true;
    }
    for (size_t i = 0; i < n_ranks; i++) {
        if (tensors[i] == nullptr || tensors[i]->type != GGML_TYPE_F32 || ggml_nelements(tensors[i]) != ne ||
                !ggml_is_contiguously_allocated(tensors[i]) ||
                tensors[i]->buffer == nullptr || !ggml_backend_buffer_is_rpc(tensors[i]->buffer)) {
            return false;
        }
        // a rank with a disabled node has garbage in its partial and must contribute zeros,
        // which only the fallback path handles
        if ((tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            return false;
        }
    }
    for (size_t i = 0; i < n_ranks; i++) {
        comm_ctx->ranks[i].dispatcher->call_async<RPC_CMD_COMM_ALLREDUCE>({ comm_ctx->ranks[i].device, serialize_tensor(tensors[i]) });
    }
    return true;
}

static void * ggml_backend_rpc_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (std::strcmp(name, "ggml_backend_rpc_add_server") == 0) {
        return (void *)ggml_backend_rpc_add_server;
    }
    if (std::strcmp(name, "ggml_backend_rpc_start_server") == 0) {
        return (void *)ggml_backend_rpc_start_server;
    }
    if (std::strcmp(name, "ggml_backend_comm_init") == 0) {
        return (void *)ggml_backend_rpc_comm_init;
    }
    if (std::strcmp(name, "ggml_backend_comm_free") == 0) {
        return (void *)ggml_backend_rpc_comm_free;
    }
    if (std::strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        return (void *)ggml_backend_rpc_comm_allreduce_tensor;
    }
    return NULL;

    GGML_UNUSED(reg);
}

static const struct ggml_backend_reg_i ggml_backend_rpc_reg_i = {
    /* .get_name         = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count = */ ggml_backend_rpc_reg_get_device_count,
    /* .get_device       = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_reg(void) {
    static struct ggml_backend_reg ggml_backend_rpc_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_rpc_reg;
}

static uint32_t ggml_backend_rpc_get_device_count(const char * endpoint) {
    auto dispatcher = get_dispatcher(endpoint);
    if (dispatcher == nullptr) {
        return 0;
    }
    rpc_msg_device_count_rsp response;
    dispatcher->send(RPC_CMD_DEVICE_COUNT, nullptr, 0, nullptr, 0, &response, sizeof(response));
    return response.device_count;
}

static const ggml_backend_reg_i ggml_backend_rpc_reg_interface = {
    /* .get_name          = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count  = */ ggml_backend_rpc_reg_get_device_count,
    /* .get_device        = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_add_server(const char * endpoint) {
    static std::unordered_map<std::string, ggml_backend_reg_t> reg_map;
    static std::mutex mutex;
    static uint32_t dev_id = 0;
    std::lock_guard<std::mutex> lock(mutex);
    if (reg_map.find(endpoint) != reg_map.end()) {
        LOG_DBG3("[%s] endpoint %s already registered\n", __func__, endpoint);
        return reg_map[endpoint];
    }
    uint32_t dev_count = ggml_backend_rpc_get_device_count(endpoint);
    LOG_DBG2("[%s] endpoint: %s, device count: %u\n", __func__, endpoint, dev_count);
    if (dev_count == 0) {
        return nullptr;
    }
    ggml_backend_rpc_reg_context * ctx = new ggml_backend_rpc_reg_context;
    ctx->name = "RPC[" + std::string(endpoint) + "]";
    for (uint32_t ind = 0; ind < dev_count; ind++) {
        std::string dev_name = "RPC" + std::to_string(dev_id);
        std::string dev_desc = std::string(endpoint);
        ggml_backend_rpc_device_context * dev_ctx = new ggml_backend_rpc_device_context {
            /* .endpoint    = */    endpoint,
            /* .device      = */    ind,
            /* .name        = */    dev_name,
            /* .description = */    dev_desc,
        };

        ggml_backend_dev_t dev = new ggml_backend_device {
            /* .iface   = */ ggml_backend_rpc_device_i,
            /* .reg     = */ ggml_backend_rpc_reg(),
            /* .context = */ dev_ctx,
        };
        ctx->devices.push_back(dev);
        dev_id++;
    }
    ggml_backend_reg_t reg = new ggml_backend_reg {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_interface,
        /* .context     = */ ctx
    };
    reg_map[endpoint] = reg;
    return reg;
}


GGML_BACKEND_DL_IMPL(ggml_backend_rpc_reg)
