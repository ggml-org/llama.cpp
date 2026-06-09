#include "ggml-rpc.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-cpp.h"
#include "transport.h"

#include <array>
#include <atomic>
#include <cinttypes>
#include <optional>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <cmath>
#include <exception>
#include <limits>

#ifdef GGML_RPC_ZLIB
#include <zlib.h>
#endif

static const char * RPC_DEBUG = std::getenv("GGML_RPC_DEBUG");

#define LOG_DBG(...) \
    do { if (RPC_DEBUG) GGML_LOG_DEBUG(__VA_ARGS__); } while (0)


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

    char padding[4];
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
    RPC_CMD_GET_DEVICE_TYPE,
    RPC_CMD_SET_TENSOR_ZLIB,
    RPC_CMD_COUNT,
};

static_assert(RPC_CMD_HELLO == 14, "RPC_CMD_HELLO must be always 14");

// Try RPC_CMD_SET_TENSOR_HASH first when data size is larger than this threshold.
const size_t HASH_THRESHOLD_DEFAULT = 10 * 1024 * 1024;

struct rpc_msg_hello_req {
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
};

enum rpc_hello_flags : uint8_t {
    RPC_HELLO_FLAG_NO_CACHE = 1 << 0,
    RPC_HELLO_FLAG_DEVICE_TYPE = 1 << 1,
    RPC_HELLO_FLAG_SET_TENSOR_ZLIB = 1 << 2,
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

struct rpc_msg_get_device_type_req {
    uint32_t device;
};

struct rpc_msg_get_device_type_rsp {
    uint32_t type;
};

struct rpc_msg_graph_recompute_req {
    uint32_t device;
};

#pragma pack(pop)

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
    uint64_t    last_graph_uid;
};

struct ggml_backend_rpc_buffer_type_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    size_t      alignment;
    size_t      max_size;
    std::mutex  alloc_size_cache_mutex;
    std::unordered_map<std::string, size_t> alloc_size_cache;
};

struct ggml_backend_rpc_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
};

struct ggml_backend_rpc_buffer_context {
    std::shared_ptr<socket_t> sock;
    void * base_ptr;
    uint64_t remote_ptr;
};

// RPC helper functions

// Computes FNV-1a hash of the data
static uint64_t fnv_hash(const uint8_t * data, size_t len) {
    const uint64_t fnv_prime = 0x100000001b3ULL;
    uint64_t hash = 0xcbf29ce484222325ULL;

    for (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= fnv_prime;
    }
    return hash;
}

static size_t rpc_cache_min_size() {
    static const size_t cache_min_size = []() {
        const char * env = std::getenv("GGML_RPC_CACHE_MIN_SIZE");
        if (env == nullptr || env[0] == '\0') {
            return HASH_THRESHOLD_DEFAULT;
        }

        char * end = nullptr;
        errno = 0;
        unsigned long long parsed = std::strtoull(env, &end, 10);
        if (errno != 0 || end == env || *end != '\0' || parsed > std::numeric_limits<size_t>::max()) {
            GGML_LOG_WARN("Ignoring invalid GGML_RPC_CACHE_MIN_SIZE='%s'\n", env);
            return HASH_THRESHOLD_DEFAULT;
        }

        return static_cast<size_t>(parsed);
    }();

    return cache_min_size;
}

static bool rpc_env_truthy(const char * name) {
    const char * env = std::getenv(name);
    return env != nullptr && env[0] != '\0' && strcmp(env, "0") != 0;
}

static size_t rpc_parse_size_env(const char * name, size_t default_value, size_t max_value) {
    const char * env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') {
        return default_value;
    }

    char * end = nullptr;
    errno = 0;
    unsigned long long parsed = std::strtoull(env, &end, 10);
    if (env[0] == '-' || env[0] == '+' || errno != 0 || end == env || *end != '\0' ||
            parsed > std::numeric_limits<size_t>::max()) {
        GGML_LOG_WARN("Ignoring invalid %s='%s'\n", name, env);
        return default_value;
    }
    if (parsed > max_value) {
        GGML_LOG_WARN("Clamping %s='%s' to %zu bytes\n", name, env, max_value);
        return max_value;
    }

    return static_cast<size_t>(parsed);
}

static constexpr size_t RPC_SET_TENSOR_ZLIB_DEFAULT_MIN_SIZE    = 1024*1024;
static constexpr size_t RPC_SET_TENSOR_ZLIB_DEFAULT_SAMPLE_SIZE = 64*1024;
static constexpr size_t RPC_SET_TENSOR_ZLIB_MAX_SAMPLE_SIZE     = 16*1024*1024;
static constexpr double RPC_SET_TENSOR_ZLIB_DEFAULT_RATIO       = 0.70;
static constexpr int    RPC_SET_TENSOR_ZLIB_DEFAULT_LEVEL       = 1;

static bool rpc_set_tensor_zlib_client_enabled() {
    static const bool enabled = []() {
        const bool requested = rpc_env_truthy("GGML_RPC_SET_TENSOR_COMPRESS");
#ifndef GGML_RPC_ZLIB
        if (requested) {
            GGML_LOG_WARN("Ignoring GGML_RPC_SET_TENSOR_COMPRESS because RPC was built without zlib support\n");
        }
        return false;
#else
        return requested;
#endif
    }();

    return enabled;
}

static size_t rpc_set_tensor_zlib_min_size() {
    static const size_t min_size = rpc_parse_size_env(
            "GGML_RPC_SET_TENSOR_COMPRESS_MIN_SIZE",
            RPC_SET_TENSOR_ZLIB_DEFAULT_MIN_SIZE,
            std::numeric_limits<size_t>::max());
    return min_size;
}

static size_t rpc_set_tensor_zlib_sample_size() {
    static const size_t sample_size = rpc_parse_size_env(
            "GGML_RPC_SET_TENSOR_COMPRESS_SAMPLE_SIZE",
            RPC_SET_TENSOR_ZLIB_DEFAULT_SAMPLE_SIZE,
            RPC_SET_TENSOR_ZLIB_MAX_SAMPLE_SIZE);
    return sample_size;
}

static double rpc_set_tensor_zlib_sample_ratio() {
    static const double ratio = []() {
        const char * env = std::getenv("GGML_RPC_SET_TENSOR_COMPRESS_SAMPLE_RATIO");
        if (env == nullptr || env[0] == '\0') {
            return RPC_SET_TENSOR_ZLIB_DEFAULT_RATIO;
        }

        char * end = nullptr;
        errno = 0;
        const double parsed = std::strtod(env, &end);
        if (errno != 0 || end == env || *end != '\0' || parsed <= 0.0 || parsed >= 1.0) {
            GGML_LOG_WARN("Ignoring invalid GGML_RPC_SET_TENSOR_COMPRESS_SAMPLE_RATIO='%s'\n", env);
            return RPC_SET_TENSOR_ZLIB_DEFAULT_RATIO;
        }

        return parsed;
    }();

    return ratio;
}

static int rpc_set_tensor_zlib_level() {
    static const int level = []() {
        const char * env = std::getenv("GGML_RPC_SET_TENSOR_COMPRESS_LEVEL");
        if (env == nullptr || env[0] == '\0') {
            return RPC_SET_TENSOR_ZLIB_DEFAULT_LEVEL;
        }

        char * end = nullptr;
        errno = 0;
        const long parsed = std::strtol(env, &end, 10);
        if (errno != 0 || end == env || *end != '\0' || parsed < 1 || parsed > 9) {
            GGML_LOG_WARN("Ignoring invalid GGML_RPC_SET_TENSOR_COMPRESS_LEVEL='%s'\n", env);
            return RPC_SET_TENSOR_ZLIB_DEFAULT_LEVEL;
        }

        return (int) parsed;
    }();

    return level;
}

static bool rpc_compression_probe_enabled() {
    static const bool enabled = []() {
        const char * env = std::getenv("GGML_RPC_COMPRESSION_PROBE");
        if (env != nullptr && env[0] != '\0' && strcmp(env, "0") != 0) {
            return true;
        }

        const char * dump_dir = std::getenv("GGML_RPC_COMPRESSION_PROBE_DUMP_DIR");
        return dump_dir != nullptr && dump_dir[0] != '\0';
    }();

    return enabled;
}

static constexpr size_t RPC_COMPRESSION_PROBE_DEFAULT_SAMPLE = 256*1024;
static constexpr size_t RPC_COMPRESSION_PROBE_MAX_SAMPLE     = 16*1024*1024;

static size_t rpc_compression_probe_max_sample() {
    static const size_t max_sample = []() {
        const char * env = std::getenv("GGML_RPC_COMPRESSION_PROBE_MAX_SAMPLE");
        if (env == nullptr || env[0] == '\0') {
            return RPC_COMPRESSION_PROBE_DEFAULT_SAMPLE;
        }

        char * end = nullptr;
        errno = 0;
        unsigned long long parsed = std::strtoull(env, &end, 10);
        if (env[0] == '-' || env[0] == '+' || errno != 0 || end == env || *end != '\0' ||
                parsed > std::numeric_limits<size_t>::max()) {
            GGML_LOG_WARN("Ignoring invalid GGML_RPC_COMPRESSION_PROBE_MAX_SAMPLE='%s'\n", env);
            return RPC_COMPRESSION_PROBE_DEFAULT_SAMPLE;
        }
        if (parsed > RPC_COMPRESSION_PROBE_MAX_SAMPLE) {
            GGML_LOG_WARN("Clamping GGML_RPC_COMPRESSION_PROBE_MAX_SAMPLE='%s' to %zu bytes\n",
                    env, RPC_COMPRESSION_PROBE_MAX_SAMPLE);
            return RPC_COMPRESSION_PROBE_MAX_SAMPLE;
        }

        return static_cast<size_t>(parsed);
    }();

    return max_sample;
}

static const std::string & rpc_compression_probe_dump_dir() {
    static const std::string dump_dir = []() {
        const char * env = std::getenv("GGML_RPC_COMPRESSION_PROBE_DUMP_DIR");
        return env == nullptr ? std::string() : std::string(env);
    }();

    return dump_dir;
}

static bool rpc_compression_probe_dump_enabled() {
    return !rpc_compression_probe_dump_dir().empty();
}

static size_t rpc_compression_probe_dump_max_files() {
    static const size_t max_files = []() {
        const char * env = std::getenv("GGML_RPC_COMPRESSION_PROBE_DUMP_MAX_FILES");
        if (env == nullptr || env[0] == '\0') {
            return (size_t) 64;
        }

        char * end = nullptr;
        errno = 0;
        unsigned long long parsed = std::strtoull(env, &end, 10);
        if (env[0] == '-' || env[0] == '+' || errno != 0 || end == env || *end != '\0' ||
                parsed > std::numeric_limits<size_t>::max()) {
            GGML_LOG_WARN("Ignoring invalid GGML_RPC_COMPRESSION_PROBE_DUMP_MAX_FILES='%s'\n", env);
            return (size_t) 64;
        }

        return static_cast<size_t>(parsed);
    }();

    return max_files;
}

static constexpr size_t RPC_WIRE_SIZE_SIZE   = sizeof(uint64_t);
static constexpr size_t RPC_WIRE_CMD_SIZE    = sizeof(uint8_t);
static constexpr size_t RPC_WIRE_HEADER_SIZE = RPC_WIRE_CMD_SIZE + RPC_WIRE_SIZE_SIZE;
static constexpr size_t RPC_COALESCE_MAX     = 4096;
static constexpr size_t RPC_ALLOC_SIZE_CACHE_MAX = 4096;

struct rpc_trace_latency_samples {
    uint64_t seen = 0;
    uint64_t max_ns = 0;
    std::vector<uint64_t> samples_ns;
};

struct rpc_trace_cmd_stat {
    uint64_t calls        = 0;
    uint64_t input_bytes  = 0;
    uint64_t output_bytes = 0;
    uint64_t one_way_ns   = 0;
    uint64_t response_ns  = 0;
    uint64_t server_ns    = 0;
};

struct rpc_trace_copy_stat {
    uint64_t calls = 0;
    uint64_t bytes = 0;
};

struct rpc_trace_tensor_stat {
    uint64_t calls = 0;
    uint64_t bytes = 0;
    uint64_t elapsed_ns = 0;
    rpc_trace_latency_samples elapsed_samples;
};

struct rpc_trace_pending_one_way_stat {
    uint64_t calls = 0;
    uint64_t bytes = 0;
};

struct rpc_trace_pending_drain_stat {
    uint64_t sync_calls = 0;
    uint64_t failed_sync_calls = 0;
    uint64_t pending_calls = 0;
    uint64_t pending_bytes = 0;
    uint64_t wait_ns = 0;
    uint64_t failed_wait_ns = 0;
    rpc_trace_latency_samples wait_samples;
};

struct rpc_compression_probe_stat {
    uint64_t calls = 0;
    uint64_t bytes = 0;
    uint64_t sampled_bytes = 0;
    uint64_t zero_bytes = 0;
    uint64_t rle_estimated_bytes = 0;
    uint64_t longest_run = 0;
    uint64_t probe_ns = 0;
    std::array<uint64_t, 256> hist = {};
};

struct rpc_trace_state {
    std::mutex mutex;
    bool has_activity = false;
    uint64_t active_rpc_backends = 0;
    rpc_trace_cmd_stat client[RPC_CMD_COUNT] = {};
    rpc_trace_cmd_stat server[RPC_CMD_COUNT] = {};
    std::unordered_map<std::string, std::array<rpc_trace_cmd_stat, RPC_CMD_COUNT>> client_by_endpoint;
    std::unordered_map<std::string, rpc_trace_tensor_stat> tensor_ops;
    std::unordered_map<std::string, rpc_trace_copy_stat> cross_endpoint_copies;
    std::unordered_map<std::string, rpc_trace_copy_stat> cross_endpoint_tensor_copies;
    std::unordered_map<std::string, rpc_compression_probe_stat> compression_probe;
    std::unordered_map<std::string, std::array<rpc_trace_pending_one_way_stat, RPC_CMD_COUNT>> pending_one_way_by_connection;
    std::unordered_map<std::string, rpc_trace_pending_drain_stat> pending_one_way_drains;
};

struct rpc_trace_server_span;
static thread_local rpc_trace_server_span * rpc_trace_current_server_span = nullptr;

static bool rpc_trace_enabled() {
    static const bool enabled = []() {
        const char * env = std::getenv("GGML_RPC_TRACE");
        return env != nullptr && env[0] != '\0' && strcmp(env, "0") != 0;
    }();

    return enabled;
}

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
        case RPC_CMD_GET_DEVICE_TYPE:   return "GET_DEVICE_TYPE";
        case RPC_CMD_SET_TENSOR_ZLIB:   return "SET_TENSOR_ZLIB";
        case RPC_CMD_COUNT:             break;
    }

    return "UNKNOWN";
}

static uint64_t rpc_trace_now_ns() {
    using clock = std::chrono::steady_clock;
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now().time_since_epoch()).count();
}

static rpc_trace_state & rpc_trace_get_state() {
    static rpc_trace_state * state = new rpc_trace_state();
    return *state;
}

static size_t rpc_trace_latency_sample_limit() {
    static const size_t limit = []() {
        const char * env = std::getenv("GGML_RPC_TRACE_LATENCY_SAMPLE_LIMIT");
        if (env == nullptr || env[0] == '\0') {
            return (size_t) 8192;
        }

        char * end = nullptr;
        errno = 0;
        const unsigned long long parsed = std::strtoull(env, &end, 10);
        if (errno != 0 || end == env || *end != '\0') {
            GGML_LOG_WARN("Ignoring invalid GGML_RPC_TRACE_LATENCY_SAMPLE_LIMIT='%s'\n", env);
            return (size_t) 8192;
        }

        return (size_t) parsed;
    }();

    return limit;
}

static uint64_t rpc_trace_sample_hash(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30))*0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27))*0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

static void rpc_trace_record_latency_sample(rpc_trace_latency_samples & samples, uint64_t elapsed_ns) {
    samples.seen++;
    samples.max_ns = std::max(samples.max_ns, elapsed_ns);

    const size_t limit = rpc_trace_latency_sample_limit();
    if (limit == 0) {
        return;
    }

    if (samples.samples_ns.size() < limit) {
        samples.samples_ns.push_back(elapsed_ns);
        return;
    }

    const uint64_t slot = rpc_trace_sample_hash(samples.seen)%samples.seen;
    if (slot < limit) {
        samples.samples_ns[(size_t) slot] = elapsed_ns;
    }
}

static double rpc_trace_ns_to_ms(uint64_t ns) {
    return (double) ns/1000000.0;
}

static double rpc_trace_percentile_sorted_ms(const std::vector<uint64_t> & sorted_ns, double percentile) {
    if (sorted_ns.empty()) {
        return 0.0;
    }
    if (sorted_ns.size() == 1) {
        return rpc_trace_ns_to_ms(sorted_ns[0]);
    }

    const double position = percentile*(double) (sorted_ns.size() - 1);
    const size_t lower = (size_t) std::floor(position);
    const size_t upper = (size_t) std::ceil(position);
    const double fraction = position - (double) lower;
    const double lower_ms = rpc_trace_ns_to_ms(sorted_ns[lower]);
    const double upper_ms = rpc_trace_ns_to_ms(sorted_ns[upper]);
    return lower_ms + (upper_ms - lower_ms)*fraction;
}

static void rpc_trace_add_cmd_stat(
        rpc_trace_cmd_stat & stat, uint64_t input_bytes, uint64_t output_bytes,
        uint64_t one_way_ns, uint64_t response_ns, uint64_t server_ns) {
    stat.calls++;
    stat.input_bytes += input_bytes;
    stat.output_bytes += output_bytes;
    stat.one_way_ns += one_way_ns;
    stat.response_ns += response_ns;
    stat.server_ns += server_ns;
}

static std::string rpc_trace_key3(const char * first, const std::string & second, const char * third) {
    std::string key = first ? first : "";
    key.push_back('\t');
    key += second;
    key.push_back('\t');
    key += third ? third : "";
    return key;
}

static void rpc_trace_split_key3(
        const std::string & key, std::string & first, std::string & second, std::string & third) {
    const size_t p0 = key.find('\t');
    const size_t p1 = p0 == std::string::npos ? std::string::npos : key.find('\t', p0 + 1);
    if (p0 == std::string::npos || p1 == std::string::npos) {
        first = key;
        second.clear();
        third.clear();
        return;
    }
    first  = key.substr(0, p0);
    second = key.substr(p0 + 1, p1 - p0 - 1);
    third  = key.substr(p1 + 1);
}

static std::string rpc_trace_connection_key(const socket_ptr & sock) {
    char ptr[32];
    snprintf(ptr, sizeof(ptr), "%p", (const void *) sock.get());
    std::string key = sock->label();
    key.push_back('#');
    key += ptr;
    return key;
}

static void rpc_trace_record_client(
        enum rpc_cmd cmd, const std::string & endpoint, uint64_t input_bytes, uint64_t output_bytes,
        uint64_t elapsed_ns, bool has_response) {
    if (!rpc_trace_enabled()) {
        return;
    }

    auto & state = rpc_trace_get_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.has_activity = true;
    rpc_trace_add_cmd_stat(
        state.client[cmd], input_bytes, output_bytes,
        has_response ? 0 : elapsed_ns, has_response ? elapsed_ns : 0, 0);
    if (!endpoint.empty()) {
        rpc_trace_add_cmd_stat(
            state.client_by_endpoint[endpoint][cmd], input_bytes, output_bytes,
            has_response ? 0 : elapsed_ns, has_response ? elapsed_ns : 0, 0);
    }
}

static bool rpc_trace_is_pending_one_way(enum rpc_cmd cmd) {
    switch (cmd) {
        case RPC_CMD_SET_TENSOR:
        case RPC_CMD_SET_TENSOR_ZLIB:
        case RPC_CMD_GRAPH_COMPUTE:
        case RPC_CMD_GRAPH_RECOMPUTE:
            return true;
        default:
            return false;
    }
}

static void rpc_trace_record_pending_one_way(
        enum rpc_cmd cmd, const std::string & connection_key, uint64_t input_bytes) {
    if (!rpc_trace_enabled() || connection_key.empty() || !rpc_trace_is_pending_one_way(cmd)) {
        return;
    }

    auto & state = rpc_trace_get_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.has_activity = true;
    auto & stat = state.pending_one_way_by_connection[connection_key][cmd];
    stat.calls++;
    stat.bytes += input_bytes;
}

static void rpc_trace_record_pending_one_way_drain(
        enum rpc_cmd response_cmd, const std::string & endpoint, const std::string & connection_key,
        uint64_t wait_ns, bool response_ok) {
    if (!rpc_trace_enabled() || endpoint.empty() || connection_key.empty()) {
        return;
    }

    auto & state = rpc_trace_get_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto it = state.pending_one_way_by_connection.find(connection_key);
    if (it == state.pending_one_way_by_connection.end()) {
        return;
    }

    bool has_pending = false;
    for (int i = 0; i < RPC_CMD_COUNT; ++i) {
        const auto & pending = it->second[i];
        if (pending.calls == 0) {
            continue;
        }

        has_pending = true;
        auto & drain = state.pending_one_way_drains[
            rpc_trace_key3(endpoint.c_str(), rpc_cmd_name(response_cmd), rpc_cmd_name((rpc_cmd) i))];
        if (response_ok) {
            drain.sync_calls++;
            drain.wait_ns += wait_ns;
            rpc_trace_record_latency_sample(drain.wait_samples, wait_ns);
        } else {
            drain.failed_sync_calls++;
            drain.failed_wait_ns += wait_ns;
        }
        drain.pending_calls += pending.calls;
        drain.pending_bytes += pending.bytes;
    }

    if (has_pending) {
        it->second = {};
        state.has_activity = true;
    }
}

static void rpc_trace_record_server(enum rpc_cmd cmd, uint64_t input_bytes, uint64_t output_bytes, uint64_t elapsed_ns) {
    if (!rpc_trace_enabled()) {
        return;
    }

    auto & state = rpc_trace_get_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.has_activity = true;
    rpc_trace_add_cmd_stat(state.server[cmd], input_bytes, output_bytes, 0, 0, elapsed_ns);
}

static void rpc_trace_record_cross_endpoint_copy(const std::string & src_endpoint, const std::string & dst_endpoint, uint64_t bytes) {
    if (!rpc_trace_enabled()) {
        return;
    }

    auto & state = rpc_trace_get_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.has_activity = true;
    auto & stat = state.cross_endpoint_copies[src_endpoint + " -> " + dst_endpoint];
    stat.calls++;
    stat.bytes += bytes;
}

static void rpc_trace_record_cross_endpoint_tensor_copy(
        const std::string & src_endpoint, const std::string & dst_endpoint, const char * tensor_name, uint64_t bytes) {
    if (!rpc_trace_enabled()) {
        return;
    }

    auto & state = rpc_trace_get_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.has_activity = true;
    auto & stat = state.cross_endpoint_tensor_copies[
        rpc_trace_key3((src_endpoint + " -> " + dst_endpoint).c_str(), "", tensor_name)];
    stat.calls++;
    stat.bytes += bytes;
}

static void rpc_trace_record_tensor_op(
        const char * op, const std::string & endpoint, const char * tensor_name, uint64_t bytes, uint64_t elapsed_ns) {
    if (!rpc_trace_enabled()) {
        return;
    }

    auto & state = rpc_trace_get_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.has_activity = true;
    auto & stat = state.tensor_ops[rpc_trace_key3(op, endpoint, tensor_name)];
    stat.calls++;
    stat.bytes += bytes;
    stat.elapsed_ns += elapsed_ns;
    rpc_trace_record_latency_sample(stat.elapsed_samples, elapsed_ns);
}

static void rpc_compression_probe_scan_bytes(
        const uint8_t * bytes, size_t size,
        std::array<uint64_t, 256> & hist, uint64_t & zero_bytes,
        uint64_t & rle_estimated_bytes, uint64_t & longest_run) {
    size_t i = 0;
    while (i < size) {
        const uint8_t value = bytes[i];
        size_t run = 1;
        while (i + run < size && bytes[i + run] == value) {
            run++;
        }

        hist[value] += run;
        if (value == 0) {
            zero_bytes += run;
        }
        longest_run = std::max<uint64_t>(longest_run, run);
        rle_estimated_bytes += 2*((run + 254)/255);
        i += run;
    }
}

struct rpc_compression_probe_sample_window {
    size_t offset = 0;
    size_t size = 0;
};

static std::vector<rpc_compression_probe_sample_window> rpc_compression_probe_sample_windows(
        size_t size, size_t sample_budget) {
    std::vector<rpc_compression_probe_sample_window> windows;
    sample_budget = std::min(size, sample_budget);
    if (sample_budget == 0) {
        return windows;
    }

    if (sample_budget == size) {
        windows.push_back({0, size});
        return windows;
    }

    const size_t window_count = std::min<size_t>(3, sample_budget);
    size_t consumed = 0;
    windows.reserve(window_count);
    for (size_t i = 0; i < window_count; ++i) {
        const size_t remaining_windows = window_count - i;
        const size_t window_size = (sample_budget - consumed + remaining_windows - 1)/remaining_windows;
        size_t offset = 0;
        if (window_count == 1 || i == 0) {
            offset = 0;
        } else if (i + 1 == window_count) {
            offset = size - window_size;
        } else {
            offset = (size - window_size)/2;
        }

        windows.push_back({offset, window_size});
        consumed += window_size;
    }

    return windows;
}

#ifdef GGML_RPC_ZLIB
static bool rpc_zlib_size_fits(size_t size, const char * what) {
    if (size > (size_t) std::numeric_limits<uLong>::max()) {
        GGML_LOG_WARN("Skipping RPC zlib compression: %s size %zu exceeds zlib limit\n", what, size);
        return false;
    }
    return true;
}

static bool rpc_zlib_compress_buffer(const uint8_t * data, size_t size, int level, std::vector<uint8_t> & compressed) {
    if (!rpc_zlib_size_fits(size, "source")) {
        return false;
    }

    uLongf compressed_bound = compressBound((uLong) size);
    if (compressed_bound > (uLongf) std::numeric_limits<size_t>::max()) {
        GGML_LOG_WARN("Skipping RPC zlib compression: compressed bound exceeds size_t\n");
        return false;
    }

    compressed.resize((size_t) compressed_bound);
    int rc = compress2(compressed.data(), &compressed_bound, data, (uLong) size, level);
    if (rc != Z_OK) {
        GGML_LOG_WARN("Skipping RPC zlib compression: zlib returned %d\n", rc);
        compressed.clear();
        return false;
    }

    compressed.resize((size_t) compressed_bound);
    return true;
}

static bool rpc_set_tensor_zlib_sample_passes(const void * data, size_t size, int level) {
    const size_t sample_budget = std::min(size, rpc_set_tensor_zlib_sample_size());
    if (sample_budget == 0) {
        return false;
    }

    const uint8_t * bytes = (const uint8_t *) data;
    const auto windows = rpc_compression_probe_sample_windows(size, sample_budget);
    std::vector<uint8_t> sample;
    sample.reserve(sample_budget);
    for (const auto & window : windows) {
        sample.insert(sample.end(), bytes + window.offset, bytes + window.offset + window.size);
    }
    if (sample.empty()) {
        return false;
    }

    std::vector<uint8_t> compressed_sample;
    if (!rpc_zlib_compress_buffer(sample.data(), sample.size(), level, compressed_sample)) {
        return false;
    }

    const double sample_ratio = (double) compressed_sample.size()/(double) sample.size();
    return sample_ratio <= rpc_set_tensor_zlib_sample_ratio();
}
#endif

static std::string rpc_compression_probe_safe_token(const char * text) {
    std::string out;
    if (text == nullptr || text[0] == '\0') {
        return "_";
    }

    for (const unsigned char c : std::string(text)) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
                c == '.' || c == '-' || c == '_') {
            out.push_back((char) c);
        } else {
            out.push_back('_');
        }
        if (out.size() >= 80) {
            break;
        }
    }

    return out.empty() ? "_" : out;
}

static std::string rpc_compression_probe_safe_token(const std::string & text) {
    return rpc_compression_probe_safe_token(text.c_str());
}

static std::string rpc_compression_probe_json_escape(const std::string & text) {
    std::string out;
    out.reserve(text.size() + 8);
    for (const unsigned char c : text) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[7];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back((char) c);
                }
                break;
        }
    }
    return out;
}

static std::mutex & rpc_compression_probe_dump_mutex() {
    static std::mutex mutex;
    return mutex;
}

static uint64_t rpc_compression_probe_next_dump_index() {
    static std::atomic<uint64_t> dump_index{0};
    return dump_index.fetch_add(1, std::memory_order_relaxed);
}

static void rpc_compression_probe_dump_sample(
        const char * op, const std::string & endpoint, const char * tensor_name,
        const uint8_t * bytes, size_t size,
        const std::vector<rpc_compression_probe_sample_window> & windows) {
    if (!rpc_compression_probe_dump_enabled() || windows.empty()) {
        return;
    }

    const uint64_t dump_index = rpc_compression_probe_next_dump_index();
    if (dump_index >= rpc_compression_probe_dump_max_files()) {
        if (dump_index == rpc_compression_probe_dump_max_files()) {
            GGML_LOG_WARN("GGML_RPC_COMPRESSION_PROBE_DUMP_MAX_FILES reached; skipping further sample dumps\n");
        }
        return;
    }

    const fs::path dump_dir = rpc_compression_probe_dump_dir();
    try {
        fs::create_directories(dump_dir);
    } catch (const std::exception & e) {
        GGML_LOG_WARN("Failed to create GGML_RPC_COMPRESSION_PROBE_DUMP_DIR='%s': %s\n",
                dump_dir.string().c_str(), e.what());
        return;
    }

    uint64_t sampled_bytes = 0;
    for (const auto & window : windows) {
        sampled_bytes += window.size;
    }

    char index_buf[32];
    snprintf(index_buf, sizeof(index_buf), "%06" PRIu64, dump_index);
    const std::string file_name = std::string(index_buf) + "_" +
        rpc_compression_probe_safe_token(op) + "_" +
        rpc_compression_probe_safe_token(endpoint) + "_" +
        rpc_compression_probe_safe_token(tensor_name) + ".bin";
    const fs::path sample_path = dump_dir / file_name;

    {
        std::ofstream sample(sample_path, std::ios::binary);
        if (!sample) {
            GGML_LOG_WARN("Failed to open RPC compression sample '%s' for writing\n", sample_path.string().c_str());
            return;
        }
        for (const auto & window : windows) {
            sample.write((const char *) bytes + window.offset, window.size);
        }
    }

    std::string window_text;
    for (size_t i = 0; i < windows.size(); ++i) {
        if (i != 0) {
            window_text += ",";
        }
        window_text += std::to_string(windows[i].offset);
        window_text += ":";
        window_text += std::to_string(windows[i].size);
    }

    std::lock_guard<std::mutex> lock(rpc_compression_probe_dump_mutex());
    std::ofstream meta(dump_dir / "samples.jsonl", std::ios::app);
    if (!meta) {
        GGML_LOG_WARN("Failed to append RPC compression sample metadata in '%s'\n", dump_dir.string().c_str());
        return;
    }

    meta << "{\"file\":\"" << rpc_compression_probe_json_escape(file_name)
         << "\",\"operation\":\"" << rpc_compression_probe_json_escape(op == nullptr ? "" : op)
         << "\",\"endpoint\":\"" << rpc_compression_probe_json_escape(endpoint)
         << "\",\"tensor\":\"" << rpc_compression_probe_json_escape(tensor_name == nullptr ? "" : tensor_name)
         << "\",\"bytes\":" << size
         << ",\"sampled_bytes\":" << sampled_bytes
         << ",\"windows\":\"" << rpc_compression_probe_json_escape(window_text)
         << "\"}\n";
}

static void rpc_compression_probe_record(
        const char * op, const std::string & endpoint, const char * tensor_name, const void * data, size_t size) {
    if (!rpc_compression_probe_enabled() || data == nullptr || size == 0) {
        return;
    }

    const uint64_t start_ns = rpc_trace_now_ns();
    const uint8_t * bytes = (const uint8_t *) data;
    const size_t sample_budget = std::min(size, rpc_compression_probe_max_sample());
    if (sample_budget == 0) {
        return;
    }

    const auto windows = rpc_compression_probe_sample_windows(size, sample_budget);
    std::array<uint64_t, 256> hist = {};
    uint64_t zero_bytes = 0;
    uint64_t rle_estimated_bytes = 0;
    uint64_t longest_run = 0;
    uint64_t sampled_bytes = 0;

    for (const auto & window : windows) {
        rpc_compression_probe_scan_bytes(bytes + window.offset, window.size,
                hist, zero_bytes, rle_estimated_bytes, longest_run);
        sampled_bytes += window.size;
    }
    rpc_compression_probe_dump_sample(op, endpoint, tensor_name, bytes, size, windows);

    auto & state = rpc_trace_get_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.has_activity = true;
    auto & stat = state.compression_probe[rpc_trace_key3(op, endpoint, tensor_name)];
    stat.calls++;
    stat.bytes += size;
    stat.sampled_bytes += sampled_bytes;
    stat.zero_bytes += zero_bytes;
    stat.rle_estimated_bytes += rle_estimated_bytes;
    stat.longest_run = std::max(stat.longest_run, longest_run);
    stat.probe_ns += rpc_trace_now_ns() - start_ns;
    for (size_t b = 0; b < hist.size(); ++b) {
        stat.hist[b] += hist[b];
    }
}

static void rpc_trace_client_backend_init() {
    if (!rpc_trace_enabled() && !rpc_compression_probe_enabled()) {
        return;
    }

    auto & state = rpc_trace_get_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.active_rpc_backends++;
}

static bool rpc_trace_client_backend_free() {
    if (!rpc_trace_enabled() && !rpc_compression_probe_enabled()) {
        return false;
    }

    auto & state = rpc_trace_get_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.active_rpc_backends > 0) {
        state.active_rpc_backends--;
    }
    return state.active_rpc_backends == 0;
}

static void rpc_trace_print_cmd_table(FILE * stream, const char * label, const rpc_trace_cmd_stat * stats) {
    fprintf(stream, "ggml-rpc trace %s commands:\n", label);
    fprintf(stream, "  %-18s %12s %14s %14s %12s %12s %12s\n",
            "command", "calls", "in_bytes", "out_bytes", "send_ms", "wait_ms", "server_ms");
    for (int i = 0; i < RPC_CMD_COUNT; ++i) {
        const auto & stat = stats[i];
        if (stat.calls == 0) {
            continue;
        }
        const double send_ms = (double) stat.one_way_ns / 1000000.0;
        const double wait_ms = (double) stat.response_ns / 1000000.0;
        const double server_ms = (double) stat.server_ns / 1000000.0;
        fprintf(stream, "  %-18s %12" PRIu64 " %14" PRIu64 " %14" PRIu64 " %12.3f %12.3f %12.3f\n",
                rpc_cmd_name((rpc_cmd) i), stat.calls, stat.input_bytes, stat.output_bytes, send_ms, wait_ms, server_ms);
    }
}

static void rpc_trace_print_endpoint_cmd_table(
        FILE * stream, const std::unordered_map<std::string, std::array<rpc_trace_cmd_stat, RPC_CMD_COUNT>> & stats) {
    std::vector<std::string> endpoints;
    endpoints.reserve(stats.size());
    for (const auto & endpoint_stats : stats) {
        endpoints.push_back(endpoint_stats.first);
    }
    std::sort(endpoints.begin(), endpoints.end());

    fprintf(stream, "ggml-rpc trace client commands by endpoint:\n");
    fprintf(stream, "  %-24s %-18s %12s %14s %14s %12s %12s\n",
            "endpoint", "command", "calls", "in_bytes", "out_bytes", "send_ms", "wait_ms");
    for (const auto & endpoint : endpoints) {
        const auto & endpoint_stats = stats.at(endpoint);
        for (int i = 0; i < RPC_CMD_COUNT; ++i) {
            const auto & stat = endpoint_stats[i];
            if (stat.calls == 0) {
                continue;
            }
            const double send_ms = (double) stat.one_way_ns / 1000000.0;
            const double wait_ms = (double) stat.response_ns / 1000000.0;
            fprintf(stream, "  %-24s %-18s %12" PRIu64 " %14" PRIu64 " %14" PRIu64 " %12.3f %12.3f\n",
                    endpoint.c_str(), rpc_cmd_name((rpc_cmd) i), stat.calls,
                    stat.input_bytes, stat.output_bytes, send_ms, wait_ms);
        }
    }
}

static void rpc_trace_print_tensor_ops(
        FILE * stream, const std::unordered_map<std::string, rpc_trace_tensor_stat> & stats) {
    std::vector<const std::unordered_map<std::string, rpc_trace_tensor_stat>::value_type *> rows;
    rows.reserve(stats.size());
    for (const auto & entry : stats) {
        if (entry.second.calls != 0) {
            rows.push_back(&entry);
        }
    }
    std::sort(rows.begin(), rows.end(), [](const auto & lhs, const auto & rhs) {
        if (lhs->second.elapsed_ns != rhs->second.elapsed_ns) {
            return lhs->second.elapsed_ns > rhs->second.elapsed_ns;
        }
        return lhs->second.bytes > rhs->second.bytes;
    });

    fprintf(stream, "ggml-rpc trace tensor operations (top by elapsed):\n");
    fprintf(stream, "  %-16s %-24s %-42s %12s %12s %14s %12s %12s %12s %12s %12s\n",
            "operation", "endpoint", "tensor", "calls", "samples", "bytes", "elapsed_ms",
            "avg_ms", "p50_ms", "p95_ms", "max_ms");
    const size_t limit = std::min<size_t>(rows.size(), 24);
    for (size_t i = 0; i < limit; ++i) {
        std::string op;
        std::string endpoint;
        std::string tensor;
        rpc_trace_split_key3(rows[i]->first, op, endpoint, tensor);
        const auto & stat = rows[i]->second;
        const double elapsed_ms = rpc_trace_ns_to_ms(stat.elapsed_ns);
        const double avg_ms = stat.calls == 0 ? 0.0 : elapsed_ms/(double) stat.calls;
        std::vector<uint64_t> sorted_samples = stat.elapsed_samples.samples_ns;
        std::sort(sorted_samples.begin(), sorted_samples.end());
        const double p50_ms = rpc_trace_percentile_sorted_ms(sorted_samples, 0.50);
        const double p95_ms = rpc_trace_percentile_sorted_ms(sorted_samples, 0.95);
        const double max_ms = rpc_trace_ns_to_ms(stat.elapsed_samples.max_ns);
        fprintf(stream, "  %-16s %-24s %-42s %12" PRIu64 " %12zu %14" PRIu64
                        " %12.3f %12.3f %12.3f %12.3f %12.3f\n",
                op.c_str(), endpoint.c_str(), tensor.c_str(), stat.calls,
                stat.elapsed_samples.samples_ns.size(), stat.bytes,
                elapsed_ms, avg_ms, p50_ms, p95_ms, max_ms);
    }
}

static double rpc_compression_probe_entropy_bits_per_byte(const rpc_compression_probe_stat & stat) {
    if (stat.sampled_bytes == 0) {
        return 0.0;
    }

    double entropy = 0.0;
    const double total = (double) stat.sampled_bytes;
    for (uint64_t count : stat.hist) {
        if (count == 0) {
            continue;
        }
        const double p = (double) count/total;
        entropy -= p*std::log2(p);
    }
    return entropy;
}

static void rpc_compression_probe_print(
        FILE * stream, const std::unordered_map<std::string, rpc_compression_probe_stat> & stats) {
    std::vector<std::pair<std::string, rpc_compression_probe_stat>> rows;
    rows.reserve(stats.size());
    for (const auto & entry : stats) {
        if (entry.second.calls != 0) {
            rows.push_back(entry);
        }
    }
    std::sort(rows.begin(), rows.end(), [](const auto & lhs, const auto & rhs) {
        if (lhs.second.bytes != rhs.second.bytes) {
            return lhs.second.bytes > rhs.second.bytes;
        }
        return lhs.second.sampled_bytes > rhs.second.sampled_bytes;
    });

    fprintf(stream, "ggml-rpc compression probe tensor payloads (top by bytes):\n");
    fprintf(stream, "  %-16s %-24s %-42s %8s %14s %14s %10s %11s %12s %10s %10s %10s %12s\n",
            "operation", "endpoint", "tensor", "calls", "bytes", "sampled",
            "coverage", "entropy_bpb", "ideal_ratio", "rle_ratio", "zero_pct", "max_run", "probe_ms");
    const size_t limit = std::min<size_t>(rows.size(), 24);
    for (size_t i = 0; i < limit; ++i) {
        std::string op;
        std::string endpoint;
        std::string tensor;
        rpc_trace_split_key3(rows[i].first, op, endpoint, tensor);
        const auto & stat = rows[i].second;
        const double entropy = rpc_compression_probe_entropy_bits_per_byte(stat);
        const double coverage = stat.bytes == 0 ? 0.0 : 100.0*(double) stat.sampled_bytes/(double) stat.bytes;
        const double ideal_ratio = stat.sampled_bytes == 0 ? 0.0 : entropy/8.0;
        const double rle_ratio =
            stat.sampled_bytes == 0 ? 0.0 : (double) stat.rle_estimated_bytes/(double) stat.sampled_bytes;
        const double zero_pct =
            stat.sampled_bytes == 0 ? 0.0 : 100.0*(double) stat.zero_bytes/(double) stat.sampled_bytes;
        const double probe_ms = (double) stat.probe_ns/1000000.0;
        fprintf(stream, "  %-16s %-24s %-42s %8" PRIu64 " %14" PRIu64 " %14" PRIu64
                        " %9.2f%% %11.3f %12.3f %10.3f %10.2f %10" PRIu64 " %12.3f\n",
                op.c_str(), endpoint.c_str(), tensor.c_str(),
                stat.calls, stat.bytes, stat.sampled_bytes,
                coverage, entropy, ideal_ratio, rle_ratio, zero_pct, stat.longest_run, probe_ms);
    }
}

static bool rpc_trace_has_cmd_stats(const rpc_trace_cmd_stat * stats) {
    for (int i = 0; i < RPC_CMD_COUNT; ++i) {
        if (stats[i].calls != 0) {
            return true;
        }
    }
    return false;
}

static bool rpc_trace_has_endpoint_cmd_stats(
        const std::unordered_map<std::string, std::array<rpc_trace_cmd_stat, RPC_CMD_COUNT>> & stats) {
    for (const auto & endpoint_stats : stats) {
        if (rpc_trace_has_cmd_stats(endpoint_stats.second.data())) {
            return true;
        }
    }
    return false;
}

static bool rpc_trace_has_copy_stats(const std::unordered_map<std::string, rpc_trace_copy_stat> & stats) {
    for (const auto & entry : stats) {
        if (entry.second.calls != 0) {
            return true;
        }
    }
    return false;
}

static bool rpc_trace_has_tensor_stats(const std::unordered_map<std::string, rpc_trace_tensor_stat> & stats) {
    for (const auto & entry : stats) {
        if (entry.second.calls != 0) {
            return true;
        }
    }
    return false;
}

static bool rpc_compression_probe_has_stats(const std::unordered_map<std::string, rpc_compression_probe_stat> & stats) {
    for (const auto & entry : stats) {
        if (entry.second.calls != 0) {
            return true;
        }
    }
    return false;
}

static void rpc_trace_print_cross_endpoint_tensor_copies(
        FILE * stream, const std::unordered_map<std::string, rpc_trace_copy_stat> & stats) {
    std::vector<std::pair<std::string, rpc_trace_copy_stat>> rows;
    rows.reserve(stats.size());
    for (const auto & entry : stats) {
        if (entry.second.calls != 0) {
            rows.push_back(entry);
        }
    }
    std::sort(rows.begin(), rows.end(), [](const auto & lhs, const auto & rhs) {
        return lhs.second.bytes > rhs.second.bytes;
    });

    fprintf(stream, "ggml-rpc trace cross-endpoint copy tensors (top by bytes):\n");
    fprintf(stream, "  %-48s %-42s %12s %14s\n", "endpoints", "tensor", "calls", "bytes");
    const size_t limit = std::min<size_t>(rows.size(), 24);
    for (size_t i = 0; i < limit; ++i) {
        std::string endpoints;
        std::string unused;
        std::string tensor;
        rpc_trace_split_key3(rows[i].first, endpoints, unused, tensor);
        fprintf(stream, "  %-48s %-42s %12" PRIu64 " %14" PRIu64 "\n",
                endpoints.c_str(), tensor.c_str(), rows[i].second.calls, rows[i].second.bytes);
    }
}

static bool rpc_trace_has_pending_drain_stats(const std::unordered_map<std::string, rpc_trace_pending_drain_stat> & stats) {
    for (const auto & entry : stats) {
        if (entry.second.sync_calls != 0 || entry.second.failed_sync_calls != 0) {
            return true;
        }
    }
    return false;
}

static void rpc_trace_print_pending_one_way_drains(
        FILE * stream, const std::unordered_map<std::string, rpc_trace_pending_drain_stat> & stats) {
    std::vector<const std::unordered_map<std::string, rpc_trace_pending_drain_stat>::value_type *> rows;
    rows.reserve(stats.size());
    for (const auto & entry : stats) {
        if (entry.second.sync_calls != 0 || entry.second.failed_sync_calls != 0) {
            rows.push_back(&entry);
        }
    }
    std::sort(rows.begin(), rows.end(), [](const auto & lhs, const auto & rhs) {
        const uint64_t lhs_wait = lhs->second.wait_ns + lhs->second.failed_wait_ns;
        const uint64_t rhs_wait = rhs->second.wait_ns + rhs->second.failed_wait_ns;
        if (lhs_wait != rhs_wait) {
            return lhs_wait > rhs_wait;
        }
        return lhs->second.pending_bytes > rhs->second.pending_bytes;
    });

    fprintf(stream, "ggml-rpc trace sync waits after one-way commands:\n");
    fprintf(stream, "  %-24s %-18s %-18s %12s %12s %12s %14s %14s %12s %12s %12s %12s %12s %12s\n",
            "endpoint", "response_cmd", "pending_cmd", "ok_syncs", "samples", "fail_syncs",
            "pending_calls", "pending_bytes", "wait_ms", "avg_wait_ms", "p50_wait_ms",
            "p95_wait_ms", "max_wait_ms", "fail_wait_ms");
    for (const auto & row : rows) {
        std::string endpoint;
        std::string response_cmd;
        std::string pending_cmd;
        rpc_trace_split_key3(row->first, endpoint, response_cmd, pending_cmd);
        const auto & stat = row->second;
        const double wait_ms = rpc_trace_ns_to_ms(stat.wait_ns);
        const double avg_wait_ms = stat.sync_calls == 0 ? 0.0 : wait_ms/(double) stat.sync_calls;
        std::vector<uint64_t> sorted_samples = stat.wait_samples.samples_ns;
        std::sort(sorted_samples.begin(), sorted_samples.end());
        const double p50_wait_ms = rpc_trace_percentile_sorted_ms(sorted_samples, 0.50);
        const double p95_wait_ms = rpc_trace_percentile_sorted_ms(sorted_samples, 0.95);
        const double max_wait_ms = rpc_trace_ns_to_ms(stat.wait_samples.max_ns);
        const double failed_wait_ms = rpc_trace_ns_to_ms(stat.failed_wait_ns);
        fprintf(stream, "  %-24s %-18s %-18s %12" PRIu64 " %12zu"
                        " %12" PRIu64 " %14" PRIu64 " %14" PRIu64
                        " %12.3f %12.3f %12.3f %12.3f %12.3f %12.3f\n",
                endpoint.c_str(), response_cmd.c_str(), pending_cmd.c_str(),
                stat.sync_calls, stat.wait_samples.samples_ns.size(), stat.failed_sync_calls,
                stat.pending_calls, stat.pending_bytes,
                wait_ms, avg_wait_ms, p50_wait_ms, p95_wait_ms, max_wait_ms, failed_wait_ms);
    }
}

static void rpc_trace_report() {
    if (!rpc_trace_enabled() && !rpc_compression_probe_enabled()) {
        return;
    }

    auto & state = rpc_trace_get_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.has_activity) {
        return;
    }
    const bool has_client = rpc_trace_has_cmd_stats(state.client);
    const bool has_server = rpc_trace_has_cmd_stats(state.server);
    const bool has_client_by_endpoint = rpc_trace_has_endpoint_cmd_stats(state.client_by_endpoint);
    const bool has_tensor_ops = rpc_trace_has_tensor_stats(state.tensor_ops);
    const bool has_copies = rpc_trace_has_copy_stats(state.cross_endpoint_copies);
    const bool has_tensor_copies = rpc_trace_has_copy_stats(state.cross_endpoint_tensor_copies);
    const bool has_compression_probe = rpc_compression_probe_has_stats(state.compression_probe);
    const bool has_pending_drains = rpc_trace_has_pending_drain_stats(state.pending_one_way_drains);
    if (!has_client && !has_server && !has_client_by_endpoint && !has_tensor_ops && !has_copies && !has_tensor_copies &&
            !has_compression_probe && !has_pending_drains) {
        return;
    }
    fprintf(stderr, "\n");
    if (has_client) {
        rpc_trace_print_cmd_table(stderr, "client", state.client);
    }
    if (has_client_by_endpoint) {
        rpc_trace_print_endpoint_cmd_table(stderr, state.client_by_endpoint);
    }
    if (has_tensor_ops) {
        rpc_trace_print_tensor_ops(stderr, state.tensor_ops);
    }
    if (has_pending_drains) {
        rpc_trace_print_pending_one_way_drains(stderr, state.pending_one_way_drains);
    }
    if (has_server) {
        rpc_trace_print_cmd_table(stderr, "server", state.server);
    }
    if (has_copies) {
        fprintf(stderr, "ggml-rpc trace cross-endpoint copy fallbacks:\n");
        fprintf(stderr, "  %-48s %12s %14s\n", "endpoints", "calls", "bytes");
        for (const auto & entry : state.cross_endpoint_copies) {
            fprintf(stderr, "  %-48s %12" PRIu64 " %14" PRIu64 "\n",
                    entry.first.c_str(), entry.second.calls, entry.second.bytes);
        }
    }
    if (has_tensor_copies) {
        rpc_trace_print_cross_endpoint_tensor_copies(stderr, state.cross_endpoint_tensor_copies);
    }
    if (has_compression_probe) {
        rpc_compression_probe_print(stderr, state.compression_probe);
    }
    for (int i = 0; i < RPC_CMD_COUNT; ++i) {
        state.client[i] = {};
        state.server[i] = {};
    }
    state.client_by_endpoint.clear();
    state.tensor_ops.clear();
    state.cross_endpoint_copies.clear();
    state.cross_endpoint_tensor_copies.clear();
    state.compression_probe.clear();
    state.pending_one_way_by_connection.clear();
    state.pending_one_way_drains.clear();
    state.has_activity = false;
    fflush(stderr);
}

static void rpc_trace_report_server_connection() {
    if (!rpc_trace_enabled()) {
        return;
    }

    auto & state = rpc_trace_get_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!rpc_trace_has_cmd_stats(state.server)) {
        return;
    }
    fprintf(stderr, "\n");
    rpc_trace_print_cmd_table(stderr, "server", state.server);
    for (int i = 0; i < RPC_CMD_COUNT; ++i) {
        state.server[i] = {};
    }
    fflush(stderr);
}

struct rpc_trace_server_connection_report {
    ~rpc_trace_server_connection_report() {
        rpc_trace_report_server_connection();
    }
};

struct rpc_trace_server_span {
    explicit rpc_trace_server_span(enum rpc_cmd cmd)
        : cmd(cmd), previous(rpc_trace_current_server_span), start_ns(0), input_bytes(0), output_bytes(0), active(rpc_trace_enabled()) {
        if (active) {
            start_ns = rpc_trace_now_ns();
            rpc_trace_current_server_span = this;
        }
    }

    ~rpc_trace_server_span() {
        if (!active) {
            return;
        }

        rpc_trace_current_server_span = previous;
        rpc_trace_record_server(cmd, input_bytes, output_bytes, rpc_trace_now_ns() - start_ns);
    }

    enum rpc_cmd cmd;
    rpc_trace_server_span * previous;
    uint64_t start_ns;
    uint64_t input_bytes;
    uint64_t output_bytes;
    bool active;
};

static void rpc_trace_server_add_input(uint64_t bytes) {
    if (rpc_trace_current_server_span != nullptr) {
        rpc_trace_current_server_span->input_bytes += bytes;
    }
}

static void rpc_trace_server_add_output(uint64_t bytes) {
    if (rpc_trace_current_server_span != nullptr) {
        rpc_trace_current_server_span->output_bytes += bytes;
    }
}

static bool send_msg(socket_ptr sock, const void * msg, size_t msg_size) {
    rpc_trace_server_add_output(msg_size);
    uint64_t wire_size = msg_size;
    if (msg_size <= RPC_COALESCE_MAX) {
        std::array<uint8_t, RPC_WIRE_SIZE_SIZE + RPC_COALESCE_MAX> frame;
        memcpy(frame.data(), &wire_size, RPC_WIRE_SIZE_SIZE);
        if (msg_size > 0) {
            memcpy(frame.data() + RPC_WIRE_SIZE_SIZE, msg, msg_size);
        }
        return sock->send_data(frame.data(), RPC_WIRE_SIZE_SIZE + msg_size);
    }
    if (!sock->send_data(&wire_size, RPC_WIRE_SIZE_SIZE)) {
        return false;
    }
    return sock->send_data(msg, msg_size);
}

static bool recv_msg(socket_ptr sock, void * msg, size_t msg_size) {
    uint64_t size;
    if (!sock->recv_data(&size, sizeof(size))) {
        return false;
    }
    if (size != msg_size) {
        return false;
    }
    rpc_trace_server_add_input(size);
    return sock->recv_data(msg, msg_size);
}

static bool recv_msg(socket_ptr sock, std::vector<uint8_t> & input) {
    uint64_t size;
    if (!sock->recv_data(&size, sizeof(size))) {
        return false;
    }
    try {
        input.resize(size);
    } catch (const std::bad_alloc & e) {
        GGML_LOG_ERROR("Failed to allocate input buffer of size %" PRIu64 "\n", size);
        return false;
    }
    rpc_trace_server_add_input(size);
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

static bool send_rpc_cmd_request(socket_ptr sock, enum rpc_cmd cmd, const void * input, size_t input_size) {
    std::array<uint8_t, RPC_WIRE_HEADER_SIZE + RPC_COALESCE_MAX> frame;
    frame[0] = static_cast<uint8_t>(cmd);
    uint64_t wire_input_size = input_size;
    memcpy(frame.data() + RPC_WIRE_CMD_SIZE, &wire_input_size, RPC_WIRE_SIZE_SIZE);
    if (input_size <= RPC_COALESCE_MAX) {
        if (input_size > 0) {
            memcpy(frame.data() + RPC_WIRE_HEADER_SIZE, input, input_size);
        }
        return sock->send_data(frame.data(), RPC_WIRE_HEADER_SIZE + input_size);
    }
    if (!sock->send_data(frame.data(), RPC_WIRE_HEADER_SIZE)) {
        return false;
    }
    return sock->send_data(input, input_size);
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// No response
static bool send_rpc_cmd(socket_ptr sock, enum rpc_cmd cmd, const void * input, size_t input_size) {
    const uint64_t start_ns = rpc_trace_enabled() ? rpc_trace_now_ns() : 0;
    bool status = send_rpc_cmd_request(sock, cmd, input, input_size);
    if (status && rpc_trace_enabled()) {
        rpc_trace_record_pending_one_way(cmd, rpc_trace_connection_key(sock), input_size);
    }
    if (rpc_trace_enabled()) {
        rpc_trace_record_client(cmd, sock->label(), input_size, 0, rpc_trace_now_ns() - start_ns, false);
    }
    return status;
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// RPC response: | response_size (8 bytes) | response_data (response_size bytes) |
static bool send_rpc_cmd(socket_ptr sock, enum rpc_cmd cmd, const void * input, size_t input_size, void * output, size_t output_size) {
    const uint64_t start_ns = rpc_trace_enabled() ? rpc_trace_now_ns() : 0;
    bool status = send_rpc_cmd_request(sock, cmd, input, input_size);
    const uint64_t wait_start_ns = rpc_trace_enabled() ? rpc_trace_now_ns() : 0;
    if (status) {
        uint64_t out_size;
        status = sock->recv_data(&out_size, sizeof(out_size));
        if (status && out_size != output_size) {
            status = false;
        }
        if (status) {
            status = sock->recv_data(output, output_size);
        }
    }
    if (rpc_trace_enabled()) {
        rpc_trace_record_pending_one_way_drain(
            cmd, sock->label(), rpc_trace_connection_key(sock), rpc_trace_now_ns() - wait_start_ns, status);
    }
    if (rpc_trace_enabled()) {
        rpc_trace_record_client(cmd, sock->label(), input_size, status ? output_size : 0, rpc_trace_now_ns() - start_ns, true);
    }
    return status;
}

// RPC client-side implementation

// Performs HELLO handshake with transport auto-negotiation.
// Advertises local capabilities via conn_caps; if the server responds with
// matching capabilities, the socket is upgraded transparently.
static bool negotiate_hello(const std::shared_ptr<socket_t> & sock) {
    rpc_msg_hello_req request = {};
    rpc_msg_hello_rsp response = {};

    sock->get_caps(request.conn_caps);

    bool status = send_rpc_cmd(sock, RPC_CMD_HELLO, &request, sizeof(request), &response, sizeof(response));
    if (!status) {
        GGML_LOG_ERROR("Failed to complete RPC hello handshake\n");
        return false;
    }

    if (response.major != RPC_PROTO_MAJOR_VERSION || response.minor > RPC_PROTO_MINOR_VERSION) {
        GGML_LOG_ERROR("RPC server version mismatch: %d.%d.%d\n",
                       response.major, response.minor, response.patch);
        return false;
    }

    sock->update_caps(response.conn_caps);
    sock->set_skip_tensor_hash((response.flags & RPC_HELLO_FLAG_NO_CACHE) != 0);
    sock->set_supports_device_type((response.flags & RPC_HELLO_FLAG_DEVICE_TYPE) != 0);
    sock->set_supports_set_tensor_zlib((response.flags & RPC_HELLO_FLAG_SET_TENSOR_ZLIB) != 0);
    return true;
}

static std::shared_ptr<socket_t> get_socket(const std::string & endpoint) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    static std::unordered_map<std::string, std::weak_ptr<socket_t>> sockets;

    auto it = sockets.find(endpoint);
    if (it != sockets.end()) {
        if (auto sock = it->second.lock()) {
            return sock;
        }
    }
    std::string host;
    int port;
    if (!parse_endpoint(endpoint, host, port)) {
        GGML_LOG_ERROR("Failed to parse endpoint: %s\n", endpoint.c_str());
        return nullptr;
    }

    if (!rpc_transport_init()) {
        return nullptr;
    }
    auto sock = socket_t::connect(host.c_str(), port);
    if (sock == nullptr) {
        return nullptr;
    }
    if (rpc_trace_enabled()) {
        sock->set_label(endpoint.c_str());
    }
    if (!negotiate_hello(sock)) {
        return nullptr;
    }
    LOG_DBG("[%s] connected to %s\n", __func__, endpoint.c_str());
    sockets[endpoint] = sock;
    return sock;
}

static void ggml_backend_rpc_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_msg_free_buffer_req request = {ctx->remote_ptr};
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_FREE_BUFFER, &request, sizeof(request), nullptr, 0);
    RPC_STATUS_ASSERT(status);
    delete ctx;
}

static void * ggml_backend_rpc_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    if (ctx->base_ptr != nullptr) {
        return ctx->base_ptr;
    }
    rpc_msg_buffer_get_base_req request = {ctx->remote_ptr};
    rpc_msg_buffer_get_base_rsp response;
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_BUFFER_GET_BASE, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    ctx->base_ptr = reinterpret_cast<void *>(response.base_ptr);
    return ctx->base_ptr;
}

static bool ggml_backend_buffer_is_rpc(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_rpc_buffer_free_buffer;
}

static rpc_tensor serialize_tensor(const ggml_tensor * tensor) {
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
        result.buffer = ctx != nullptr ? ctx->remote_ptr : 0;
        result.data = reinterpret_cast<uint64_t>(tensor->data);
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
    memset(result.padding, 0, sizeof(result.padding));

    snprintf(result.name, GGML_MAX_NAME, "%s", tensor->name);
    return result;
}

static enum ggml_status ggml_backend_rpc_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;

    // CUDA backend on the server pads everything to 512 due to CUDA limitations.
    // Due to bandwidth constraints, we only call the server init tensor functions if necessary.
    // In particular, only quantized tensors need padding
    if (ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr)) {
        rpc_msg_init_tensor_req request;

        request.tensor = serialize_tensor(tensor);

        bool status = send_rpc_cmd(ctx->sock, RPC_CMD_INIT_TENSOR, &request, sizeof(request), nullptr, 0);
        RPC_STATUS_ASSERT(status);
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_rpc_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    if (size > rpc_cache_min_size() && !ctx->sock->skip_tensor_hash()) {
        rpc_msg_set_tensor_hash_req request;
        request.tensor = rpc_tensor;
        request.offset = offset;
        request.hash = fnv_hash((const uint8_t*)data, size);
        rpc_msg_set_tensor_hash_rsp response;
        const uint64_t trace_start_ns = rpc_trace_enabled() ? rpc_trace_now_ns() : 0;
        bool status = send_rpc_cmd(ctx->sock, RPC_CMD_SET_TENSOR_HASH, &request, sizeof(request), &response, sizeof(response));
        RPC_STATUS_ASSERT(status);
        if (rpc_trace_enabled()) {
            ggml_backend_rpc_buffer_type_context * buft_ctx =
                (ggml_backend_rpc_buffer_type_context *)ggml_backend_buffer_get_type(buffer)->context;
            rpc_trace_record_tensor_op(
                "SET_TENSOR_HASH", buft_ctx->endpoint, tensor->name, size, rpc_trace_now_ns() - trace_start_ns);
        }
        if (response.result) {
            // the server has the same data, no need to send it
            return;
        }
    }

#ifdef GGML_RPC_ZLIB
    if (rpc_set_tensor_zlib_client_enabled() && ctx->sock->supports_set_tensor_zlib() &&
            size >= rpc_set_tensor_zlib_min_size()) {
        const int level = rpc_set_tensor_zlib_level();
        const uint64_t trace_start_ns = rpc_trace_enabled() ? rpc_trace_now_ns() : 0;
        if (rpc_set_tensor_zlib_sample_passes(data, size, level)) {
            std::vector<uint8_t> compressed;
            if (rpc_zlib_compress_buffer((const uint8_t *) data, size, level, compressed) &&
                    (double) compressed.size()/(double) size <= rpc_set_tensor_zlib_sample_ratio()) {
                const size_t header_size = sizeof(rpc_tensor) + sizeof(uint64_t) + sizeof(uint64_t);
                if (compressed.size() <= std::numeric_limits<size_t>::max() - header_size) {
                    const uint64_t uncompressed_size = size;
                    const size_t input_size = header_size + compressed.size();
                    std::vector<uint8_t> input(input_size, 0);
                    memcpy(input.data(), &rpc_tensor, sizeof(rpc_tensor));
                    memcpy(input.data() + sizeof(rpc_tensor), &offset, sizeof(offset));
                    memcpy(input.data() + sizeof(rpc_tensor) + sizeof(offset),
                            &uncompressed_size, sizeof(uncompressed_size));
                    memcpy(input.data() + header_size, compressed.data(), compressed.size());

                    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_SET_TENSOR_ZLIB, input.data(), input.size());
                    RPC_STATUS_ASSERT(status);
                    if (rpc_trace_enabled() || rpc_compression_probe_enabled()) {
                        ggml_backend_rpc_buffer_type_context * buft_ctx =
                            (ggml_backend_rpc_buffer_type_context *)ggml_backend_buffer_get_type(buffer)->context;
                        if (rpc_trace_enabled()) {
                            rpc_trace_record_tensor_op(
                                    "SET_TENSOR_ZLIB", buft_ctx->endpoint, tensor->name, size,
                                    rpc_trace_now_ns() - trace_start_ns);
                        }
                        if (rpc_compression_probe_enabled()) {
                            rpc_compression_probe_record("SET_TENSOR_ZLIB", buft_ctx->endpoint, tensor->name, data, size);
                        }
                    }
                    return;
                }
                GGML_LOG_WARN("Skipping RPC zlib compression: compressed input size overflow\n");
            }
        }
    }
#endif

    // input serialization format: | rpc_tensor | offset (8 bytes) | data (size bytes)
    size_t input_size = sizeof(rpc_tensor) + sizeof(uint64_t) + size;
    std::vector<uint8_t> input(input_size, 0);
    memcpy(input.data(), &rpc_tensor, sizeof(rpc_tensor));
    memcpy(input.data() + sizeof(rpc_tensor), &offset, sizeof(offset));
    memcpy(input.data() + sizeof(rpc_tensor) + sizeof(offset), data, size);
    const uint64_t trace_start_ns = rpc_trace_enabled() ? rpc_trace_now_ns() : 0;
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_SET_TENSOR, input.data(), input.size());
    RPC_STATUS_ASSERT(status);
    if (rpc_trace_enabled() || rpc_compression_probe_enabled()) {
        ggml_backend_rpc_buffer_type_context * buft_ctx =
            (ggml_backend_rpc_buffer_type_context *)ggml_backend_buffer_get_type(buffer)->context;
        if (rpc_trace_enabled()) {
            rpc_trace_record_tensor_op("SET_TENSOR", buft_ctx->endpoint, tensor->name, size, rpc_trace_now_ns() - trace_start_ns);
        }
        if (rpc_compression_probe_enabled()) {
            rpc_compression_probe_record("SET_TENSOR", buft_ctx->endpoint, tensor->name, data, size);
        }
    }
}

static void ggml_backend_rpc_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_msg_get_tensor_req request;
    request.tensor = serialize_tensor(tensor);
    request.offset = offset;
    request.size = size;
    const uint64_t trace_start_ns = rpc_trace_enabled() ? rpc_trace_now_ns() : 0;
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_GET_TENSOR, &request, sizeof(request), data, size);
    RPC_STATUS_ASSERT(status);
    if (rpc_trace_enabled() || rpc_compression_probe_enabled()) {
        ggml_backend_rpc_buffer_type_context * buft_ctx =
            (ggml_backend_rpc_buffer_type_context *)ggml_backend_buffer_get_type(buffer)->context;
        if (rpc_trace_enabled()) {
            rpc_trace_record_tensor_op("GET_TENSOR", buft_ctx->endpoint, tensor->name, size, rpc_trace_now_ns() - trace_start_ns);
        }
        if (rpc_compression_probe_enabled()) {
            rpc_compression_probe_record("GET_TENSOR", buft_ctx->endpoint, tensor->name, data, size);
        }
    }
}

static bool ggml_backend_rpc_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_rpc(src->buffer)) {
        // check if src and dst are on the same server
        ggml_backend_buffer_t src_buffer = src->buffer;
        ggml_backend_rpc_buffer_context * src_ctx = (ggml_backend_rpc_buffer_context *)src_buffer->context;
        ggml_backend_buffer_t dst_buffer = dst->buffer;
        ggml_backend_rpc_buffer_context * dst_ctx = (ggml_backend_rpc_buffer_context *)dst_buffer->context;
        if (src_ctx->sock != dst_ctx->sock) {
            if (rpc_trace_enabled()) {
                ggml_backend_rpc_buffer_type_context * src_buft_ctx =
                    (ggml_backend_rpc_buffer_type_context *)ggml_backend_buffer_get_type(src_buffer)->context;
                ggml_backend_rpc_buffer_type_context * dst_buft_ctx =
                    (ggml_backend_rpc_buffer_type_context *)ggml_backend_buffer_get_type(dst_buffer)->context;
                rpc_trace_record_cross_endpoint_copy(src_buft_ctx->endpoint, dst_buft_ctx->endpoint, ggml_nbytes(src));
                rpc_trace_record_cross_endpoint_tensor_copy(
                    src_buft_ctx->endpoint, dst_buft_ctx->endpoint, src->name, ggml_nbytes(src));
            }
            return false;
        }
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        rpc_msg_copy_tensor_req request;
        request.src = serialize_tensor(src);
        request.dst = serialize_tensor(dst);
        rpc_msg_copy_tensor_rsp response;
        const uint64_t trace_start_ns = rpc_trace_enabled() ? rpc_trace_now_ns() : 0;
        bool status = send_rpc_cmd(ctx->sock, RPC_CMD_COPY_TENSOR, &request, sizeof(request), &response, sizeof(response));
        RPC_STATUS_ASSERT(status);
        if (rpc_trace_enabled()) {
            ggml_backend_rpc_buffer_type_context * dst_buft_ctx =
                (ggml_backend_rpc_buffer_type_context *)ggml_backend_buffer_get_type(dst_buffer)->context;
            rpc_trace_record_tensor_op(
                "COPY_TENSOR", dst_buft_ctx->endpoint, src->name, ggml_nbytes(src), rpc_trace_now_ns() - trace_start_ns);
        }
        return response.result;
    }
    return false;
}

static void ggml_backend_rpc_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_msg_buffer_clear_req request = {ctx->remote_ptr, value};
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_BUFFER_CLEAR, &request, sizeof(request), nullptr, 0);
    RPC_STATUS_ASSERT(status);
}

static ggml_backend_buffer_i ggml_backend_rpc_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_rpc_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_rpc_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_rpc_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_rpc_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_rpc_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_rpc_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_rpc_buffer_clear,
    /* .reset           = */ NULL,
};

static const char * ggml_backend_rpc_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_rpc_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    rpc_msg_alloc_buffer_req request = {buft_ctx->device, size};
    rpc_msg_alloc_buffer_rsp response;
    auto sock = get_socket(buft_ctx->endpoint);
    bool status = send_rpc_cmd(sock, RPC_CMD_ALLOC_BUFFER, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    if (response.remote_ptr != 0) {
        ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft,
            ggml_backend_rpc_buffer_interface,
            new ggml_backend_rpc_buffer_context{sock, nullptr, response.remote_ptr},
            response.remote_size);
        return buffer;
    } else {
        return nullptr;
    }
}

static size_t get_alignment(const std::shared_ptr<socket_t> & sock, uint32_t device) {
    rpc_msg_get_alignment_req request = {device};
    rpc_msg_get_alignment_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_GET_ALIGNMENT, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    return response.alignment;
}

static size_t ggml_backend_rpc_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->alignment;
}

static size_t get_max_size(const std::shared_ptr<socket_t> & sock, uint32_t device) {
    rpc_msg_get_max_size_req request = {device};
    rpc_msg_get_max_size_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_GET_MAX_SIZE, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
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

    // ops that require additional memory for fleeting data on certain backends
    // ref: https://github.com/ggml-org/llama.cpp/pull/15966
    rpc_get |= tensor->op == GGML_OP_FLASH_ATTN_EXT;
    rpc_get |= tensor->op == GGML_OP_MUL_MAT_ID;

    if (rpc_get) {
        ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
        auto sock = get_socket(buft_ctx->endpoint);

        rpc_msg_get_alloc_size_req request = {
            /*.device =*/ buft_ctx->device,
            /*.tensor =*/ serialize_tensor(tensor),
            /*.srcs   =*/ {},
        };

        // .get_alloc_size could be a function of the tensor's srcs, so we must serialize them as well
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            request.srcs[i] = serialize_tensor(tensor->src[i]);
        }

        const std::string cache_key(reinterpret_cast<const char *>(&request), sizeof(request));
        {
            std::lock_guard<std::mutex> lock(buft_ctx->alloc_size_cache_mutex);
            auto it = buft_ctx->alloc_size_cache.find(cache_key);
            if (it != buft_ctx->alloc_size_cache.end()) {
                return it->second;
            }
        }

        rpc_msg_get_alloc_size_rsp response;
        bool status = send_rpc_cmd(sock, RPC_CMD_GET_ALLOC_SIZE, &request, sizeof(request), &response, sizeof(response));
        RPC_STATUS_ASSERT(status);

        const size_t alloc_size = response.alloc_size;
        {
            std::lock_guard<std::mutex> lock(buft_ctx->alloc_size_cache_mutex);
            if (buft_ctx->alloc_size_cache.size() >= RPC_ALLOC_SIZE_CACHE_MAX) {
                buft_ctx->alloc_size_cache.clear();
            }
            buft_ctx->alloc_size_cache.emplace(cache_key, alloc_size);
        }

        return alloc_size;
    }

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

static const char * ggml_backend_rpc_name(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;

    return rpc_ctx->name.c_str();
}

static void ggml_backend_rpc_free(ggml_backend_t backend) {
    const bool report_trace = rpc_trace_client_backend_free();
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    delete rpc_ctx;
    delete backend;
    if (report_trace) {
        rpc_trace_report();
    }
}

static void ggml_backend_rpc_synchronize(ggml_backend_t backend) {
    GGML_UNUSED(backend);
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

static void serialize_graph(uint32_t device, const ggml_cgraph * cgraph, std::vector<uint8_t> & output) {
    uint32_t n_nodes = cgraph->n_nodes;
    std::vector<rpc_tensor> tensors;
    std::unordered_set<ggml_tensor*> visited;
    for (uint32_t i = 0; i < n_nodes; i++) {
        add_tensor(cgraph->nodes[i], tensors, visited);
    }
    // serialization format:
    // | device (4 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    int output_size = 2*sizeof(uint32_t) + n_nodes * sizeof(uint64_t) + sizeof(uint32_t) + n_tensors * sizeof(rpc_tensor);
    output.resize(output_size, 0);
    uint8_t * dest = output.data();
    memcpy(dest, &device, sizeof(device));
    dest += sizeof(device);
    memcpy(dest, &n_nodes, sizeof(n_nodes));
    dest += sizeof(n_nodes);
    for (uint32_t i = 0; i < n_nodes; i++) {
        memcpy(dest + i * sizeof(uint64_t), &cgraph->nodes[i], sizeof(uint64_t));
    }
    dest += n_nodes * sizeof(uint64_t);
    memcpy(dest, &n_tensors, sizeof(n_tensors));
    dest += sizeof(n_tensors);
    rpc_tensor * out_tensors = (rpc_tensor *)dest;
    memcpy(out_tensors, tensors.data(), n_tensors * sizeof(rpc_tensor));
}

static enum ggml_status ggml_backend_rpc_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    ggml_backend_dev_t rpc_dev = ggml_backend_get_device(backend);
    ggml_backend_rpc_device_context * rpc_dev_ctx = (ggml_backend_rpc_device_context *)rpc_dev->context;

    GGML_ASSERT(cgraph->n_nodes > 0);
    bool reuse = cgraph->uid != 0 && rpc_dev_ctx->last_graph_uid == cgraph->uid;
    if (reuse) {
        rpc_msg_graph_recompute_req request;
        request.device = rpc_ctx->device;
        auto sock = get_socket(rpc_ctx->endpoint);
        bool status = send_rpc_cmd(sock, RPC_CMD_GRAPH_RECOMPUTE, &request, sizeof(request));
        RPC_STATUS_ASSERT(status);
    } else {
        rpc_dev_ctx->last_graph_uid = cgraph->uid;
        std::vector<uint8_t> input;
        serialize_graph(rpc_ctx->device, cgraph, input);
        auto sock = get_socket(rpc_ctx->endpoint);
        bool status = send_rpc_cmd(sock, RPC_CMD_GRAPH_COMPUTE, input.data(), input.size());
        RPC_STATUS_ASSERT(status);
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_backend_i ggml_backend_rpc_interface = {
    /* .get_name                = */ ggml_backend_rpc_name,
    /* .free                    = */ ggml_backend_rpc_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ggml_backend_rpc_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_rpc_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
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
    auto sock = get_socket(endpoint);
    if (sock == nullptr) {
        GGML_LOG_ERROR("Failed to connect to %s\n", endpoint);
        return nullptr;
    }
    size_t alignment = get_alignment(sock, device);
    size_t max_size = get_max_size(sock, device);
    ggml_backend_rpc_buffer_type_context * buft_ctx = new ggml_backend_rpc_buffer_type_context {
        /* .endpoint  = */ endpoint,
        /* .device    = */ device,
        /* .name      = */ buft_name,
        /* .alignment = */ alignment,
        /* .max_size  = */ max_size,
        /* .alloc_size_cache_mutex = */ {},
        /* .alloc_size_cache       = */ {}
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
    ggml_backend_buffer_type_t buft = new ggml_backend_buffer_type {
        /* .iface   = */ ggml_backend_rpc_buffer_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ buft_ctx
    };
    buft_map[buft_name] = buft;
    return buft;
}

ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device) {
    rpc_trace_client_backend_init();
    std::string dev_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    ggml_backend_rpc_context * ctx = new ggml_backend_rpc_context {
        /* .endpoint       = */ endpoint,
        /* .device         = */ device,
        /* .name           = */ dev_name,
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
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

static void get_device_memory(const std::shared_ptr<socket_t> & sock, uint32_t device, size_t * free, size_t * total) {
    rpc_msg_get_device_memory_req request;
    request.device = device;
    rpc_msg_get_device_memory_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_GET_DEVICE_MEMORY, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    *free = response.free_mem;
    *total = response.total_mem;
}

void ggml_backend_rpc_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total) {
    auto sock = get_socket(endpoint);
    if (sock == nullptr) {
        *free = 0;
        *total = 0;
        return;
    }
    get_device_memory(sock, device, free, total);
}

static bool is_valid_backend_device_type(uint32_t type) {
    switch ((enum ggml_backend_dev_type) type) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:
        case GGML_BACKEND_DEVICE_TYPE_GPU:
        case GGML_BACKEND_DEVICE_TYPE_IGPU:
        case GGML_BACKEND_DEVICE_TYPE_ACCEL:
        case GGML_BACKEND_DEVICE_TYPE_META:
            return true;
    }
    return false;
}

static const char * backend_device_type_name(enum ggml_backend_dev_type type) {
    switch (type) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:   return "CPU";
        case GGML_BACKEND_DEVICE_TYPE_GPU:   return "GPU";
        case GGML_BACKEND_DEVICE_TYPE_IGPU:  return "IGPU";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "ACCEL";
        case GGML_BACKEND_DEVICE_TYPE_META:  return "META";
    }
    return "unknown";
}

static bool get_remote_device_type(const std::shared_ptr<socket_t> & sock, uint32_t device, enum ggml_backend_dev_type * type) {
    if (!sock->supports_device_type()) {
        return false;
    }

    rpc_msg_get_device_type_req request;
    request.device = device;
    rpc_msg_get_device_type_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_GET_DEVICE_TYPE, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);

    if (!is_valid_backend_device_type(response.type)) {
        GGML_LOG_WARN("RPC server returned invalid device type %u\n", response.type);
        return false;
    }

    *type = (enum ggml_backend_dev_type) response.type;
    return true;
}

// RPC server-side implementation

class rpc_server {
public:
    rpc_server(std::vector<ggml_backend_t> all_backends, const char * cache_dir)
        : backends(std::move(all_backends)), cache_dir(cache_dir) {
        stored_graphs.resize(backends.size());
    }
    ~rpc_server();

    void hello(rpc_msg_hello_rsp & response);
    bool alloc_buffer(const rpc_msg_alloc_buffer_req & request, rpc_msg_alloc_buffer_rsp & response);
    bool get_alignment(const rpc_msg_get_alignment_req & request, rpc_msg_get_alignment_rsp & response);
    bool get_max_size(const rpc_msg_get_max_size_req & request, rpc_msg_get_max_size_rsp & response);
    bool buffer_get_base(const rpc_msg_buffer_get_base_req & request, rpc_msg_buffer_get_base_rsp & response);
    bool free_buffer(const rpc_msg_free_buffer_req & request);
    bool buffer_clear(const rpc_msg_buffer_clear_req & request);
    bool set_tensor(const std::vector<uint8_t> & input);
    bool set_tensor_zlib(const std::vector<uint8_t> & input);
    bool set_tensor_hash(const rpc_msg_set_tensor_hash_req & request, rpc_msg_set_tensor_hash_rsp & response);
    bool get_tensor(const rpc_msg_get_tensor_req & request, std::vector<uint8_t> & response);
    bool copy_tensor(const rpc_msg_copy_tensor_req & request, rpc_msg_copy_tensor_rsp & response);
    bool graph_compute(const std::vector<uint8_t> & input);
    bool graph_recompute(const rpc_msg_graph_recompute_req & request);
    bool init_tensor(const rpc_msg_init_tensor_req & request);
    bool get_alloc_size(const rpc_msg_get_alloc_size_req & request, rpc_msg_get_alloc_size_rsp & response);
    bool get_device_memory(const rpc_msg_get_device_memory_req & request, rpc_msg_get_device_memory_rsp & response);
    bool get_device_type(const rpc_msg_get_device_type_req & request, rpc_msg_get_device_type_rsp & response);

    struct stored_graph {
        std::vector<uint8_t>   buffer;
        ggml_cgraph          * graph;
    };

private:
    bool get_cached_file(uint64_t hash, std::vector<uint8_t> & data);
    void cache_tensor_data(const void * data, size_t size);
    ggml_tensor * deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor);
    ggml_tensor * create_node(uint64_t id,
                              struct ggml_context * ctx,
                              const std::unordered_map<uint64_t, const rpc_tensor*> & tensor_ptrs,
                              std::unordered_map<uint64_t, struct ggml_tensor*> & tensor_map);


    std::vector<ggml_backend_t> backends;
    const char * cache_dir;
    std::unordered_set<ggml_backend_buffer_t> buffers;
    // store the last computed graph for each backend
    std::vector<stored_graph> stored_graphs;
};

void rpc_server::hello(rpc_msg_hello_rsp & response) {
    response.major = RPC_PROTO_MAJOR_VERSION;
    response.minor = RPC_PROTO_MINOR_VERSION;
    response.patch = RPC_PROTO_PATCH_VERSION;
    response.flags |= RPC_HELLO_FLAG_DEVICE_TYPE;
    if (cache_dir == nullptr) {
        response.flags |= RPC_HELLO_FLAG_NO_CACHE;
    }
#ifdef GGML_RPC_ZLIB
    response.flags |= RPC_HELLO_FLAG_SET_TENSOR_ZLIB;
#endif
    LOG_DBG("[%s] version: %d.%d.%d, flags: 0x%x\n",
            __func__, response.major, response.minor, response.patch, response.flags);
}

bool rpc_server::get_alloc_size(const rpc_msg_get_alloc_size_req & request, rpc_msg_get_alloc_size_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft;
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead()*(1 + GGML_MAX_SRC),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr) {
        GGML_LOG_ERROR("Null tensor pointer passed to server get_alloc_size function.\n");
        return false;
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (request.srcs[i].id != 0) {
            tensor->src[i] = deserialize_tensor(ctx, &request.srcs[i]);
        }
    }

    LOG_DBG("[%s] device: %d, buffer: %p, data: %p\n", __func__, dev_id, (void*)tensor->buffer, tensor->data);
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
        LOG_DBG("[%s] device: %d, size: %" PRIu64 " -> remote_ptr: %" PRIx64 ", remote_size: %" PRIu64 "\n",
            __func__, dev_id, request.size, response.remote_ptr, response.remote_size);
        buffers.insert(buffer);
    } else {
        LOG_DBG("[%s] device: %d, size: %" PRIu64 " -> failed\n", __func__, dev_id, request.size);
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
    LOG_DBG("[%s] device: %d, alignment: %lu\n", __func__, dev_id, alignment);
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
    LOG_DBG("[%s] device: %d, max_size: %lu\n", __func__, dev_id, max_size);
    response.max_size = max_size;
    return true;
}

bool rpc_server::buffer_get_base(const rpc_msg_buffer_get_base_req & request, rpc_msg_buffer_get_base_rsp & response) {
    LOG_DBG("[%s] remote_ptr: %" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    void * base = ggml_backend_buffer_get_base(buffer);
    response.base_ptr = reinterpret_cast<uint64_t>(base);
    return true;
}

bool rpc_server::free_buffer(const rpc_msg_free_buffer_req & request) {
    LOG_DBG("[%s] remote_ptr: %" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    ggml_backend_buffer_free(buffer);
    buffers.erase(buffer);
    return true;
}

bool rpc_server::buffer_clear(const rpc_msg_buffer_clear_req & request) {
    LOG_DBG("[%s] remote_ptr: %" PRIx64 ", value: %u\n", __func__, request.remote_ptr, request.value);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    ggml_backend_buffer_clear(buffer, request.value);
    return true;
}

ggml_tensor * rpc_server::deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor) {
    // Validate tensor type before using it
    if (tensor->type >= GGML_TYPE_COUNT) {
        GGML_LOG_ERROR("[%s] invalid tensor type received: %u\n", __func__, tensor->type);
        return nullptr;
    }

    // Fix: Prevent division by zero if blck_size is 0 (e.g., deprecated types)
    if (ggml_blck_size((enum ggml_type)tensor->type) == 0) {
        GGML_LOG_ERROR("[%s] invalid tensor type received (blck_size is 0): %u\n", __func__, tensor->type);
        return nullptr;
    }

    ggml_tensor * result = ggml_new_tensor_4d(ctx, (ggml_type) tensor->type,
        tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);

    // ggml_new_tensor_4d might fail if dimensions are invalid, although less likely to crash than invalid type
    if (result == nullptr) {
        GGML_LOG_ERROR("[%s] ggml_new_tensor_4d failed for type %u\n", __func__, tensor->type);
        return nullptr;
    }

    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = tensor->nb[i];
    }
    result->buffer = reinterpret_cast<ggml_backend_buffer_t>(tensor->buffer);
    if (result->buffer && buffers.find(result->buffer) == buffers.end()) {
        result->buffer = nullptr;
    }

    if (result->buffer) {
        // require that the tensor data does not go beyond the buffer end
        uint64_t tensor_size = (uint64_t) ggml_nbytes(result);
        uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(result->buffer);
        uint64_t buffer_size = (uint64_t) ggml_backend_buffer_get_size(result->buffer);
        GGML_ASSERT(tensor->data + tensor_size >= tensor->data); // check for overflow
        GGML_ASSERT(tensor->data >= buffer_start && tensor->data + tensor_size <= buffer_start + buffer_size);
    }

    result->op = (ggml_op) tensor->op;
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result->op_params[i] = tensor->op_params[i];
    }
    result->flags = tensor->flags;
    result->data = reinterpret_cast<void *>(tensor->data);
    ggml_set_name(result, tensor->name);
    return result;
}

static bool rpc_validate_tensor_data_region(
        const rpc_tensor * in_tensor, const ggml_tensor * tensor, uint64_t offset, size_t size, const char * func) {
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", func);
        return false;
    }

    const uint64_t p0 = (uint64_t) (uintptr_t) ggml_backend_buffer_get_base(tensor->buffer);
    const uint64_t buffer_size = (uint64_t) ggml_backend_buffer_get_size(tensor->buffer);
    if (buffer_size > std::numeric_limits<uint64_t>::max() - p0) {
        GGML_LOG_ERROR("[%s] tensor buffer bounds overflow (base=0x%" PRIx64 ", size=%" PRIu64 ")\n",
                func, p0, buffer_size);
        return false;
    }
    const uint64_t p1 = p0 + buffer_size;
    if (in_tensor->data > std::numeric_limits<uint64_t>::max() - offset) {
        GGML_LOG_ERROR("[%s] tensor data offset overflow (data=0x%" PRIx64 ", offset=%" PRIu64 ")\n",
                func, in_tensor->data, offset);
        return false;
    }

    const uint64_t start = in_tensor->data + offset;
    const uint64_t requested_size = (uint64_t) size;
    if (start < p0 || start >= p1 || requested_size > p1 - start) {
        GGML_LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu) out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                       func, in_tensor->data, offset, size, p0, p1);
        return false;
    }

    return true;
}

void rpc_server::cache_tensor_data(const void * data, size_t size) {
    if (cache_dir == nullptr || size <= rpc_cache_min_size()) {
        return;
    }

    uint64_t hash = fnv_hash((const uint8_t*)data, size);
    char hash_str[17];
    snprintf(hash_str, sizeof(hash_str), "%016" PRIx64, hash);
    // save to cache_dir/hash_str
    fs::path cache_file = fs::path(cache_dir) / hash_str;
    std::ofstream ofs(cache_file, std::ios::binary);
    ofs.write((const char *)data, size);
    GGML_LOG_INFO("[%s] saved to '%s'\n", __func__, cache_file.string().c_str());
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
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    if (!rpc_validate_tensor_data_region(in_tensor, tensor, offset, size, __func__)) {
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %zu\n", __func__, (void*)tensor->buffer, tensor->data, offset, size);

    const void * data = input.data() + sizeof(rpc_tensor) + sizeof(offset);
    cache_tensor_data(data, size);
    ggml_backend_tensor_set(tensor, data, offset, size);
    return true;
}

bool rpc_server::set_tensor_zlib(const std::vector<uint8_t> & input) {
#ifndef GGML_RPC_ZLIB
    (void) input;
    GGML_LOG_ERROR("[%s] RPC server was built without zlib support\n", __func__);
    return false;
#else
    // serialization format: | rpc_tensor | offset (8 bytes) | raw_size (8 bytes) | zlib_data |
    const size_t header_size = sizeof(rpc_tensor) + sizeof(uint64_t) + sizeof(uint64_t);
    if (input.size() < header_size) {
        return false;
    }

    const rpc_tensor * in_tensor = (const rpc_tensor *) input.data();
    uint64_t offset;
    memcpy(&offset, input.data() + sizeof(rpc_tensor), sizeof(offset));
    uint64_t raw_size_u64;
    memcpy(&raw_size_u64, input.data() + sizeof(rpc_tensor) + sizeof(offset), sizeof(raw_size_u64));
    if (raw_size_u64 == 0 || raw_size_u64 > (uint64_t) std::numeric_limits<size_t>::max()) {
        GGML_LOG_ERROR("[%s] invalid uncompressed size: %" PRIu64 "\n", __func__, raw_size_u64);
        return false;
    }

    const size_t raw_size = (size_t) raw_size_u64;
    const size_t compressed_size = input.size() - header_size;
    if (compressed_size == 0 || compressed_size >= raw_size) {
        GGML_LOG_ERROR("[%s] invalid compressed size: %zu for raw size %zu\n",
                __func__, compressed_size, raw_size);
        return false;
    }

    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    if (!rpc_validate_tensor_data_region(in_tensor, tensor, offset, raw_size, __func__)) {
        return false;
    }

    if (!rpc_zlib_size_fits(raw_size, "decompressed") || !rpc_zlib_size_fits(compressed_size, "compressed")) {
        return false;
    }

    std::vector<uint8_t> data(raw_size);
    uLongf dest_len = (uLongf) raw_size;
    int rc = uncompress(data.data(), &dest_len, input.data() + header_size, (uLong) compressed_size);
    if (rc != Z_OK || dest_len != (uLongf) raw_size) {
        GGML_LOG_ERROR("[%s] zlib inflate failed: rc=%d, bytes=%lu/%zu\n",
                __func__, rc, (unsigned long) dest_len, raw_size);
        return false;
    }

    LOG_DBG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", raw_size: %zu, compressed_size: %zu\n",
            __func__, (void*)tensor->buffer, tensor->data, offset, raw_size, compressed_size);
    cache_tensor_data(data.data(), raw_size);
    ggml_backend_tensor_set(tensor, data.data(), offset, raw_size);
    return true;
#endif
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
    std::vector<uint8_t> cached_file;
    if (!get_cached_file(request.hash, cached_file)) {
        response.result = 0;
        return true;
    }
    size_t size = cached_file.size();
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %zu, hash: %" PRIx64 "\n",
            __func__, (void*)tensor->buffer, tensor->data, request.offset, size, request.hash);

    if (!rpc_validate_tensor_data_region(&request.tensor, tensor, request.offset, size, __func__)) {
        return false;
    }
    ggml_backend_tensor_set(tensor, cached_file.data(), request.offset, size);
    response.result = 1;
    return true;
}

bool rpc_server::init_tensor(const rpc_msg_init_tensor_req & request) {
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr) {
        GGML_LOG_ERROR("Null tensor pointer passed to server init_tensor function.\n");
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p\n", __func__, (void*)tensor->buffer, tensor->data);
    // Call the backend's buffer_init_tensor function
    ggml_backend_buffer_t buffer = tensor->buffer;
    if (buffer && buffer->iface.init_tensor) {
        buffer->iface.init_tensor(buffer, tensor);
    } else {
        if (!buffer) {
            GGML_LOG_ERROR("Tensor with null buffer passed to init_tensor function\n");
        }
    }

    if (tensor->extra != nullptr) {
        // This pointer can either be passed around client/server, or probably better stored server-side and kept track of.
        // Currently unimplemented.
        GGML_LOG_ERROR("tensor->extra populated by the backend, this is currently unsupported.\n");
        return false;
    }

    return true;
}

bool rpc_server::get_tensor(const rpc_msg_get_tensor_req & request, std::vector<uint8_t> & response) {
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %" PRIu64 "\n", __func__, (void*)tensor->buffer, tensor->data, request.offset, request.size);

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (request.tensor.data + request.offset < p0 ||
            request.tensor.data + request.offset >= p1 ||
            request.size > (p1 - request.tensor.data - request.offset)) {
                GGML_LOG_ERROR("[%s] requested tensor region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%" PRIu64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                               __func__, request.tensor.data, request.offset, request.size, p0, p1);
                return false;
        }
    }

    response.resize(request.size, 0);
    ggml_backend_tensor_get(tensor, response.data(), request.offset, request.size);
    return true;
}

bool rpc_server::copy_tensor(const rpc_msg_copy_tensor_req & request, rpc_msg_copy_tensor_rsp & response) {
    struct ggml_init_params params {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * src = deserialize_tensor(ctx, &request.src);
    ggml_tensor * dst = deserialize_tensor(ctx, &request.dst);
    if (src == nullptr || dst == nullptr || src->buffer == nullptr || dst->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensors\n", __func__);
        return false;
    }

    uint64_t src_size   = (uint64_t) ggml_nbytes(src);
    uint64_t dst_data   = (uint64_t) dst->data;
    uint64_t dst_base   = (uint64_t) ggml_backend_buffer_get_base(dst->buffer);
    uint64_t dst_buf_sz = (uint64_t) ggml_backend_buffer_get_size(dst->buffer);

    if (dst_data + src_size > dst_base + dst_buf_sz) {
        GGML_LOG_ERROR("[%s] out-of-bounds write in rpc_server::copy_tensor:\n"
                         "    write range : [0x%" PRIx64 ", 0x%" PRIx64 "]\n"
                         "    buffer base: [0x%" PRIx64 ", 0x%" PRIx64 "]\n",
                         __func__,
                         dst_data,
                         dst_data + src_size,
                         dst_base,
                         dst_base + dst_buf_sz);
        return false;
    }

    LOG_DBG("[%s] src->buffer: %p, dst->buffer: %p\n",
            __func__, (void*) src->buffer, (void*) dst->buffer);

    response.result = ggml_backend_buffer_copy_tensor(src, dst);
    if (!response.result) {
        // Keep same-server fallback copies on the RPC server instead of routing
        // tensor bytes back through the RPC client.
        const size_t nbytes = ggml_nbytes(src);
        std::vector<uint8_t> data(nbytes);
        ggml_backend_tensor_get(src, data.data(), 0, nbytes);
        ggml_backend_tensor_set(dst, data.data(), 0, nbytes);
        response.result = true;
    }
    return true;
}

ggml_tensor * rpc_server::create_node(uint64_t id,
                                      struct ggml_context * ctx,
                                      const std::unordered_map<uint64_t, const rpc_tensor*> & tensor_ptrs,
                                      std::unordered_map<uint64_t, struct ggml_tensor*> & tensor_map) {
    if (tensor_map.find(id) != tensor_map.end()) {
        return tensor_map[id];
    }
    // Safely find the tensor pointer
    auto it_ptr = tensor_ptrs.find(id);
    if (it_ptr == tensor_ptrs.end()) {
        return nullptr;
    }
    const rpc_tensor * tensor = it_ptr->second;

    struct ggml_tensor * result = deserialize_tensor(ctx, tensor);
    if (result == nullptr) {
        return nullptr;
    }
    if (result->buffer == nullptr && result->data != nullptr) {
        GGML_LOG_ERROR("[%s] invalid data ptr", __func__);
        return nullptr;
    }
    tensor_map[id] = result;
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        // Check if the source ID is 0 before calling create_node recursively
        if (tensor->src[i] == 0) {
            result->src[i] = nullptr;
        } else {
            result->src[i] = create_node(tensor->src[i], ctx, tensor_ptrs, tensor_map);
            // If the recursive call failed for a non-zero ID, propagate the error
            if (result->src[i] == nullptr) {
                GGML_LOG_ERROR("[%s] failed to create source node %d (src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                               __func__, i, tensor->src[i], id);
                // Must return nullptr to signal failure up the call stack
                return nullptr;
            }
        }
    }

    // Handle view_src similarly
    if (tensor->view_src == 0) {
        result->view_src = nullptr;
    } else {
        result->view_src = create_node(tensor->view_src, ctx, tensor_ptrs, tensor_map);
        // If the recursive call failed for a non-zero ID, propagate the error
        if (result->view_src == nullptr) {
            GGML_LOG_ERROR("[%s] failed to create view_src node (view_src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                           __func__, tensor->view_src, id);
            // Must return nullptr to signal failure up the call stack
            return nullptr;
        }
    }
    result->view_offs = tensor->view_offs;
    return result;
}

bool rpc_server::graph_compute(const std::vector<uint8_t> & input) {
    // serialization format:
    // | device (4 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    if (input.size() < 2*sizeof(uint32_t)) {
        return false;
    }
    const uint8_t * src = input.data();
    uint32_t device;
    memcpy(&device, src, sizeof(device));
    src += sizeof(device);
    if (device >= backends.size()) {
        return false;
    }
    uint32_t n_nodes;
    memcpy(&n_nodes, src, sizeof(n_nodes));
    src += sizeof(n_nodes);
    if (input.size() < 2*sizeof(uint32_t) + n_nodes*sizeof(uint64_t) + sizeof(uint32_t)) {
        return false;
    }
    const uint64_t * nodes = (const uint64_t *)src;
    src += n_nodes*sizeof(uint64_t);
    uint32_t n_tensors;
    memcpy(&n_tensors, src, sizeof(n_tensors));
    src += sizeof(n_tensors);
    if (input.size() < 2*sizeof(uint32_t) + n_nodes*sizeof(uint64_t) + sizeof(uint32_t) + n_tensors*sizeof(rpc_tensor)) {
        return false;
    }
    const rpc_tensor * tensors = (const rpc_tensor *)src;
    LOG_DBG("[%s] device: %u, n_nodes: %u, n_tensors: %u\n", __func__, device, n_nodes, n_tensors);

    size_t buf_size = ggml_tensor_overhead()*(n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, false);
    if (stored_graphs[device].buffer.size() < buf_size) {
        stored_graphs[device].buffer.resize(buf_size);
    }
    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ stored_graphs[device].buffer.data(),
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, n_nodes, false);
    graph->n_nodes = n_nodes;
    std::unordered_map<uint64_t, const rpc_tensor*> tensor_ptrs;
    tensor_ptrs.reserve(n_tensors);
    for (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs.emplace(tensors[i].id, &tensors[i]);
    }
    std::unordered_map<uint64_t, ggml_tensor*> tensor_map;
    tensor_map.reserve(n_nodes);
    for (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        memcpy(&id, &nodes[i], sizeof(id));
        graph->nodes[i] = create_node(id, ctx, tensor_ptrs, tensor_map);

        // Check if create_node failed for a *non-zero* ID.
        // If id was 0, create_node returning nullptr is expected.
        // If id was non-zero and create_node returned nullptr, it indicates a deserialization error.
        if (graph->nodes[i] == nullptr && id != 0) {
            GGML_LOG_ERROR("[%s] failed to create graph node %d (id=%" PRId64 ")\n", __func__, i, id);
            return false;
        }
    }
    ggml_status status = ggml_backend_graph_compute(backends[device], graph);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    stored_graphs[device].graph = graph;
    return true;
}

bool rpc_server::graph_recompute(const rpc_msg_graph_recompute_req & request) {
    uint32_t device = request.device;
    if (device >= backends.size()) {
        return false;
    }
    if (stored_graphs[device].graph == nullptr) {
        return false;
    }
    ggml_cgraph * graph = stored_graphs[device].graph;
    LOG_DBG("[%s] device: %u\n", __func__, device);
    ggml_status status = ggml_backend_graph_compute(backends[device], graph);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
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
    LOG_DBG("[%s] device: %u, free_mem: %" PRIu64 ", total_mem: %" PRIu64 "\n", __func__, dev_id, response.free_mem, response.total_mem);
    return true;
}

bool rpc_server::get_device_type(const rpc_msg_get_device_type_req & request, rpc_msg_get_device_type_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(backends[dev_id]);
    response.type = (uint32_t) ggml_backend_dev_type(dev);
    LOG_DBG("[%s] device: %u, type: %u\n", __func__, dev_id, response.type);
    return true;
}

rpc_server::~rpc_server() {
    for (auto buffer : buffers) {
        ggml_backend_buffer_free(buffer);
    }
}

static void rpc_serve_client(const std::vector<ggml_backend_t> & backends, const char * cache_dir,
                             socket_ptr sock) {
    rpc_trace_server_connection_report trace_connection_report;
    rpc_server server(backends, cache_dir);
    uint8_t cmd;
    if (!sock->recv_data(&cmd, 1)) {
        return;
    }
    if (cmd != RPC_CMD_HELLO) {
        GGML_LOG_ERROR("Expected HELLO command, update client\n");
        return;
    }

    // Read input_size and validate protocol version
    uint64_t hello_input_size;
    if (!sock->recv_data(&hello_input_size, sizeof(hello_input_size))) {
        return;
    }

    if (hello_input_size != sizeof(rpc_msg_hello_req)) {
        GGML_LOG_ERROR("HELLO request size mismatch (%zu vs %zu) — client needs upgrade to protocol v%d.x\n",
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
    while (true) {
        if (!sock->recv_data(&cmd, 1)) {
            break;
        }
        if (cmd >= RPC_CMD_COUNT) {
            // fail fast if the command is invalid
            GGML_LOG_ERROR("Unknown command: %d\n", cmd);
            break;
        }
        rpc_trace_server_span trace_span((rpc_cmd) cmd);
        switch (cmd) {
            case RPC_CMD_HELLO: {
                // HELLO command is handled above
                return;
            }
            case RPC_CMD_DEVICE_COUNT: {
                if (!recv_msg(sock, nullptr, 0)) {
                    return;
                }
                rpc_msg_device_count_rsp response;
                response.device_count = backends.size();
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_ALLOC_BUFFER: {
                rpc_msg_alloc_buffer_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_alloc_buffer_rsp response;
                if (!server.alloc_buffer(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_ALLOC_SIZE: {
                rpc_msg_get_alloc_size_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_alloc_size_rsp response;
                if (!server.get_alloc_size(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_ALIGNMENT: {
                rpc_msg_get_alignment_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_alignment_rsp response;
                if (!server.get_alignment(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_MAX_SIZE: {
                rpc_msg_get_max_size_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_max_size_rsp response;
                if (!server.get_max_size(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_BUFFER_GET_BASE: {
                rpc_msg_buffer_get_base_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_buffer_get_base_rsp response;
                if (!server.buffer_get_base(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_FREE_BUFFER: {
                rpc_msg_free_buffer_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.free_buffer(request)) {
                    return;
                }
                if (!send_msg(sock, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_BUFFER_CLEAR: {
                rpc_msg_buffer_clear_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.buffer_clear(request)) {
                    return;
                }
                if (!send_msg(sock, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                if (!server.set_tensor(input)) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR_ZLIB: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                if (!server.set_tensor_zlib(input)) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR_HASH: {
                rpc_msg_set_tensor_hash_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_set_tensor_hash_rsp response;
                if (!server.set_tensor_hash(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_INIT_TENSOR: {
                rpc_msg_init_tensor_req request;
                if (!recv_msg(sock, &request,sizeof(request))) {
                    return;
                }
                if (!server.init_tensor(request)) {
                    return;
                }
                if (!send_msg(sock, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_TENSOR: {
                rpc_msg_get_tensor_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                std::vector<uint8_t> response;
                if (!server.get_tensor(request, response)) {
                    return;
                }
                if (!send_msg(sock, response.data(), response.size())) {
                    return;
                }
                break;
            }
            case RPC_CMD_COPY_TENSOR: {
                rpc_msg_copy_tensor_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_copy_tensor_rsp response;
                if (!server.copy_tensor(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GRAPH_COMPUTE: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                if (!server.graph_compute(input)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GRAPH_RECOMPUTE: {
                rpc_msg_graph_recompute_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.graph_recompute(request)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_DEVICE_MEMORY: {
                rpc_msg_get_device_memory_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_device_memory_rsp response;
                if (!server.get_device_memory(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_DEVICE_TYPE: {
                rpc_msg_get_device_type_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_device_type_rsp response;
                if (!server.get_device_type(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            default: {
                GGML_LOG_ERROR("Unknown command: %d\n", cmd);
                return;
            }
        }
    }
}

void ggml_backend_rpc_start_server(const char * endpoint, const char * cache_dir,
                                   size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices) {
    if (n_devices == 0 || devices == nullptr) {
        fprintf(stderr, "Invalid arguments to ggml_backend_rpc_start_server\n");
        return;
    }
    std::vector<ggml_backend_t> backends;
    printf("Starting RPC server v%d.%d.%d\n",
        RPC_PROTO_MAJOR_VERSION,
        RPC_PROTO_MINOR_VERSION,
        RPC_PROTO_PATCH_VERSION);
    printf("  endpoint       : %s\n", endpoint);
    printf("  local cache    : %s\n", cache_dir ? cache_dir : "n/a");
    printf("Devices:\n");
    for (size_t i = 0; i < n_devices; i++) {
        auto dev = devices[i];
        size_t free, total;
        ggml_backend_dev_memory(dev, &free, &total);
        printf("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
               total / 1024 / 1024, free / 1024 / 1024);
        auto backend = ggml_backend_dev_init(dev, nullptr);
        if (!backend) {
            fprintf(stderr, "Failed to create backend for device %s\n", dev->iface.get_name(dev));
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
    printf("  transport      : TCP (RDMA auto-negotiate enabled)\n");
#else
    printf("  transport      : TCP\n");
#endif // GGML_RPC_RDMA
    if (!rpc_transport_init()) {
        fprintf(stderr, "Failed to initialize RPC transport\n");
        return;
    }
    auto server_socket = socket_t::create_server(host.c_str(), port);
    if (server_socket == nullptr) {
        fprintf(stderr, "Failed to create server socket\n");
        return;
    }
    while (true) {
        auto client_socket = server_socket->accept();
        if (client_socket == nullptr) {
            fprintf(stderr, "Failed to accept client connection\n");
            return;
        }
        printf("Accepted client connection\n");
        fflush(stdout);
        rpc_serve_client(backends, cache_dir, client_socket);
        printf("Client connection closed\n");
        fflush(stdout);
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
    // RPC devices are classified as GPU for compatibility with existing
    // offload-device selection. The remote underlying type is reported in the
    // device description when the server supports RPC device-type reporting.
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_rpc_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_rpc_device_get_name(dev);
    props->description = ggml_backend_rpc_device_get_description(dev);
    props->type        = ggml_backend_rpc_device_get_type(dev);
    ggml_backend_rpc_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
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
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
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

static void * ggml_backend_rpc_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (std::strcmp(name, "ggml_backend_rpc_add_server") == 0) {
        return (void *)ggml_backend_rpc_add_server;
    }
    if (std::strcmp(name, "ggml_backend_rpc_start_server") == 0) {
        return (void *)ggml_backend_rpc_start_server;
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
    auto sock = get_socket(endpoint);
    if (sock == nullptr) {
        GGML_LOG_ERROR("Failed to connect to %s\n", endpoint);
        return 0;
    }
    rpc_msg_device_count_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_DEVICE_COUNT, nullptr, 0, &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
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
        return reg_map[endpoint];
    }
    uint32_t dev_count = ggml_backend_rpc_get_device_count(endpoint);
    if (dev_count == 0) {
        return nullptr;
    }
    auto sock = get_socket(endpoint);
    if (sock == nullptr) {
        return nullptr;
    }
    ggml_backend_rpc_reg_context * ctx = new ggml_backend_rpc_reg_context;
    ctx->name = "RPC[" + std::string(endpoint) + "]";
    for (uint32_t ind = 0; ind < dev_count; ind++) {
        std::string dev_name = "RPC" + std::to_string(dev_id);
        std::string dev_desc = std::string(endpoint);
        enum ggml_backend_dev_type remote_type;
        if (get_remote_device_type(sock, ind, &remote_type)) {
            dev_desc += " (remote type: ";
            dev_desc += backend_device_type_name(remote_type);
            dev_desc += ")";
        }
        ggml_backend_rpc_device_context * dev_ctx = new ggml_backend_rpc_device_context {
            /* .endpoint    = */    endpoint,
            /* .device      = */    ind,
            /* .name        = */    dev_name,
            /* .description = */    dev_desc,
            /* .last_graph_uid = */ 0,
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
