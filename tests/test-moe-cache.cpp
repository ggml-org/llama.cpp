#include "ggml.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "../ggml/src/ggml-backend-moe-cache.h"

#ifdef GGML_CUDA
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int64_t n_in      = 256;
constexpr int64_t n_out     = 128;
constexpr int64_t n_expert  = 64;
constexpr int64_t n_used    = 2;
constexpr int64_t n_tokens  = 1;
constexpr int64_t multi_n_used   = 6;
constexpr int64_t multi_n_tokens = 6;
constexpr int     max_steps = 160;

struct log_capture {
    std::mutex mutex;
    std::condition_variable cv;
    std::string text;

    void clear() {
        std::lock_guard<std::mutex> lock(mutex);
        text.clear();
    }

    std::string get() {
        std::lock_guard<std::mutex> lock(mutex);
        return text;
    }

    bool wait_for(
            const char * pattern,
            const std::atomic<bool> & stop,
            std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex);
        return cv.wait_for(lock, timeout, [&] {
            return text.find(pattern) != std::string::npos || stop.load();
        }) && text.find(pattern) != std::string::npos;
    }
};

static void log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    log_capture & capture = *static_cast<log_capture *>(user_data);
    {
        std::lock_guard<std::mutex> lock(capture.mutex);
        capture.text += text;
    }
    capture.cv.notify_all();
}

static void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

struct invalidation_record {
    const void * base;
    size_t size;
};

static std::vector<invalidation_record> * invalidation_records = nullptr;

static void record_invalidation(const void * base, size_t size) {
    if (invalidation_records) {
        invalidation_records->push_back({base, size});
    }
}

static void * test_provider_session_create(
        void * const * backends, int n_backends,
        const struct ggml_moe_cache_config * config) {
    (void) backends; (void) n_backends; (void) config;
    return nullptr;
}

static void test_provider_session_destroy(void * session) { (void) session; }
static void test_provider_session_enter(void * session) { (void) session; }
static void test_provider_session_leave(void * session) { (void) session; }

// The tests exercise the selected device's provider directly, so a machine
// with several providers (CUDA + Vulkan, say) always tests a consistent
// pair of session creator and session callbacks. Populated in main().
static ggml_moe_cache_api g_test_api;

static ggml_moe_cache_api & active_api() {
    return g_test_api;
}

struct reset_buffer_context {
    uint8_t data[64];
    bool reset = false;
};

static void * reset_buffer_get_base(ggml_backend_buffer_t buffer) {
    return ((reset_buffer_context *)buffer->context)->data;
}

static void reset_buffer_reset(ggml_backend_buffer_t buffer) {
    ((reset_buffer_context *)buffer->context)->reset = true;
}

static bool run_invalidation_hook_coverage(ggml_backend_t cpu) {
    const ggml_init_params params = {
        4 * ggml_tensor_overhead(),
        nullptr,
        true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "cache-invalidation-hooks: failed to create context\n");
        return false;
    }

    ggml_tensor * src = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
    ggml_tensor * dst = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    if (!buffer) {
        fprintf(stderr, "cache-invalidation-hooks: failed to allocate tensors\n");
        ggml_free(ctx);
        return false;
    }
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // Register a test provider whose invalidate hook records buffer mutations.
    // The provider registry is the supported way to install a hook: the buffer
    // mutation path iterates the registered providers.
    static int dummy_owner = 0;
    ggml_moe_cache_api test_api = {};
    test_api.owner = &dummy_owner;
    test_api.session_create = test_provider_session_create;
    test_api.session_destroy = test_provider_session_destroy;
    test_api.session_enter = test_provider_session_enter;
    test_api.session_leave = test_provider_session_leave;
    test_api.invalidate = record_invalidation;
    ggml_moe_cache_register(&test_api);

    std::vector<float> data(64, 0.25f);
    std::vector<invalidation_record> records;
    invalidation_records = &records;

    bool ok = true;
    auto check = [&](const char * operation,
                     std::initializer_list<invalidation_record> expected) {
        bool matches = records.size() == expected.size();
        size_t index = 0;
        for (const invalidation_record & item : expected) {
            if (index >= records.size() || records[index].base != item.base ||
                records[index].size != item.size) {
                matches = false;
            }
            index++;
        }
        if (!matches) {
            fprintf(stderr,
                    "cache-invalidation-hooks: %s produced %zu records, expected %zu\n",
                    operation, records.size(), expected.size());
        }
        records.clear();
        ok &= matches;
    };

    char * dst_base = (char *)dst->data;
    void * buffer_base = ggml_backend_buffer_get_base(buffer);
    const size_t buffer_size = ggml_backend_buffer_get_size(buffer);

    ggml_backend_tensor_set(dst, data.data(), 4, 16);
    check("tensor set", {{dst_base + 4, 16}});

    ggml_backend_tensor_set_async(cpu, dst, data.data(), 8, 20);
    check("async tensor set", {{dst_base + 8, 20}});

    ggml_backend_tensor_set_2d(
            dst, data.data(), 0, 8, 3, 16, 8);
    check("2D tensor set", {
            {dst_base, 8}, {dst_base + 16, 8}, {dst_base + 32, 8}});

    ggml_backend_tensor_set_2d_async(
            cpu, dst, data.data(), 4, 8, 3, 16, 8);
    check("async 2D tensor set", {
            {dst_base + 4, 8}, {dst_base + 20, 8}, {dst_base + 36, 8}});

    ggml_backend_tensor_memset(dst, 0, 12, 24);
    check("tensor memset", {{dst_base + 12, 24}});

    if (!ggml_backend_buffer_copy_tensor(src, dst)) {
        fprintf(stderr, "cache-invalidation-hooks: direct tensor copy failed\n");
        ok = false;
    }
    check("direct tensor copy", {{dst_base, ggml_nbytes(dst)}});

    ggml_backend_tensor_copy(src, dst);
    check("tensor copy", {{dst_base, ggml_nbytes(dst)}});

    ggml_backend_tensor_copy_async(cpu, cpu, src, dst);
    check("async tensor copy", {{dst_base, ggml_nbytes(dst)}});

    ggml_backend_buffer_clear(buffer, 0);
    check("buffer clear", {{buffer_base, buffer_size}});

    ggml_backend_buffer_set_usage(
            buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    check("weight usage removal", {{buffer_base, buffer_size}});

    ggml_backend_tensor_set(dst, data.data(), 0, 16);
    check("non-weight tensor set", {});

    ggml_backend_buffer_set_usage(
            buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    check("weight usage restoration", {});

    ggml_backend_buffer_reset(buffer);
    check("buffer reset without callback", {});

    reset_buffer_context reset_context;
    ggml_backend_buffer_i reset_iface = {};
    reset_iface.get_base = reset_buffer_get_base;
    reset_iface.reset = reset_buffer_reset;
    ggml_backend_buffer_t reset_buffer = ggml_backend_buffer_init(
            ggml_backend_cpu_buffer_type(), reset_iface,
            &reset_context, sizeof(reset_context.data));
    ggml_backend_buffer_set_usage(
            reset_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_reset(reset_buffer);
    check("buffer reset", {
            {reset_context.data, sizeof(reset_context.data)}});
    if (!reset_context.reset) {
        fprintf(stderr, "cache-invalidation-hooks: reset callback was not called\n");
        ok = false;
    }
    ggml_backend_buffer_free(reset_buffer);
    check("reset buffer free", {
            {reset_context.data, sizeof(reset_context.data)}});

    ggml_backend_buffer_free(buffer);
    buffer = nullptr;
    check("buffer free", {{buffer_base, buffer_size}});

    ggml_moe_cache_unregister(&dummy_owner);
    invalidation_records = nullptr;
    ggml_free(ctx);
    printf("cache-invalidation-hooks: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

static bool has_positive_field(const std::string & text, const char * field) {
    size_t position = 0;
    while ((position = text.find(field, position)) != std::string::npos) {
        position += strlen(field);
        char * end = nullptr;
        const long long value = strtoll(text.c_str() + position, &end, 10);
        if (end != text.c_str() + position && value > 0) {
            return true;
        }
    }
    return false;
}

static long long max_field_value(const std::string & text, const char * field) {
    long long result = -1;
    size_t position = 0;
    while ((position = text.find(field, position)) != std::string::npos) {
        position += strlen(field);
        char * end = nullptr;
        const long long value = strtoll(text.c_str() + position, &end, 10);
        if (end != text.c_str() + position) {
            result = std::max(result, value);
        }
    }
    return result;
}

static size_t count_field_at_least(
        const std::string & text, const char * field, long long minimum) {
    size_t result = 0;
    size_t position = 0;
    while ((position = text.find(field, position)) != std::string::npos) {
        position += strlen(field);
        char * end = nullptr;
        const long long value = strtoll(text.c_str() + position, &end, 10);
        if (end != text.c_str() + position && value >= minimum) {
            result++;
        }
    }
    return result;
}

static size_t count_occurrences(const std::string & text, const char * pattern) {
    size_t result = 0;
    size_t position = 0;
    while ((position = text.find(pattern, position)) != std::string::npos) {
        result++;
        position += strlen(pattern);
    }
    return result;
}

// Any GPU/IGPU device whose backend registered a moe-cache provider. The
// test exercises the selected device's provider (see g_test_api below).
static ggml_backend_dev_t find_provider_device() {
    ggml_backend_load_all();
    for (size_t index = 0; index < ggml_backend_dev_count(); index++) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
        if ((ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_GPU ||
             ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_IGPU) &&
            ggml_moe_cache_get(reg).owner != nullptr) {
            return device;
        }
    }
    return nullptr;
}

static ggml_backend_dev_t find_other_provider_device(
        ggml_backend_dev_t excluded) {
    for (size_t index = 0; index < ggml_backend_dev_count(); index++) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
        if (device != excluded &&
            (ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_GPU ||
             ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_IGPU) &&
            ggml_moe_cache_get(reg).owner != nullptr) {
            return device;
        }
    }
    return nullptr;
}

static long provider_physical_device(ggml_backend_dev_t device) {
    const char * description = ggml_backend_dev_description(device);
    const char * marker = description ? strstr(description, "(physical device ") : nullptr;
    if (!marker) {
        return -1;
    }
    marker += strlen("(physical device ");
    char * end = nullptr;
    const long physical = strtol(marker, &end, 10);
    return end != marker ? physical : -1;
}

static ggml_backend_t init_cpu_backend() {
    for (size_t index = 0; index < ggml_backend_dev_count(); index++) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        if (ggml_backend_dev_type(device) !=
                GGML_BACKEND_DEVICE_TYPE_CPU) {
            continue;
        }
        ggml_backend_t backend =
            ggml_backend_dev_init(device, nullptr);
        if (!backend) {
            continue;
        }
        ggml_backend_reg_t reg =
            ggml_backend_dev_backend_reg(device);
        auto set_n_threads =
            (ggml_backend_set_n_threads_t)
                ggml_backend_reg_get_proc_address(
                    reg, "ggml_backend_set_n_threads");
        if (set_n_threads) {
            set_n_threads(backend, 4);
        }
        return backend;
    }
    return nullptr;
}

static bool compare_output(
        const std::vector<float> & reference,
        const std::vector<float> & actual,
        double max_nmse) {
    double squared_error = 0.0;
    double squared_reference = 0.0;
    for (size_t index = 0; index < reference.size(); index++) {
        if (!std::isfinite(actual[index])) {
            return false;
        }
        const double difference = (double) actual[index] - reference[index];
        squared_error += difference * difference;
        squared_reference += (double) reference[index] * reference[index];
    }
    return squared_error / std::max(squared_reference, 1e-12) <= max_nmse;
}

static void configure_cache(
        const char * fail_stage,
        const char * max_batch = "1",
        const char * dedicated_mmv = "1") {
    set_env("GGML_CUDA_MOE_CACHE", "1");
    set_env("GGML_CUDA_MOE_CACHE_MODE", "on");
    set_env("GGML_CUDA_MOE_CACHE_BUDGET_MB", "4");
    set_env("GGML_CUDA_MOE_CACHE_RESERVE_MB", "0");
    set_env("GGML_CUDA_MOE_CACHE_MIN_EXPERT_KB", "1");
    set_env("GGML_CUDA_MOE_CACHE_MAX_BATCH", max_batch);
    set_env("GGML_CUDA_MOE_CACHE_INSERTS", "4");
    set_env("GGML_CUDA_MOE_CACHE_ADMIT_AFTER", "1");
    set_env("GGML_CUDA_MOE_CACHE_THROTTLE", "1");
    set_env("GGML_CUDA_MOE_CACHE_QUEUE", "16");
    set_env("GGML_CUDA_MOE_CACHE_STATS", "1");
    set_env("GGML_CUDA_MOE_CACHE_NDEV", "1");
    set_env("GGML_CUDA_MOE_CACHE_MIN_CC", "0");
    set_env("GGML_CUDA_MOE_CACHE_SERIAL_FILL", nullptr);
    set_env("GGML_CUDA_MOE_CACHE_DEDICATED_MMV", dedicated_mmv);
    set_env("GGML_CUDA_MOE_CACHE_OVERLAP_CPU_ROWS", "0");
    set_env("GGML_CUDA_MOE_CACHE_FAIL", fail_stage);
}

static bool run_capability_queries(
        ggml_backend_dev_t provider_device, ggml_backend_t cpu) {
    const ggml_moe_cache_api api =
        ggml_moe_cache_get(ggml_backend_dev_backend_reg(provider_device));
    if (!api.query_config || !api.query_device || !api.query_shape) {
        fprintf(stderr, "cache-capabilities: query API is incomplete\n");
        return false;
    }

    configure_cache(nullptr);
    ggml_moe_cache_config config = {};
    bool ok = api.query_config(1, 0, &config) == 1;
    ok &= config.min_devices == 1; // automatic mode now accepts a single device
    ok &= config.budget_bytes == 4u * 1024 * 1024;
    ok &= config.reserve_bytes == 0;
    ok &= config.minimum_slab_bytes == 1024u * 1024 * 1024;
    ok &= config.min_expert_bytes == 1024;
    ok &= config.min_expert_explicit == 1;
    ok &= config.overlap_cpu_rows == 0;

    ggml_moe_cache_device_caps device = {};
    ok &= api.query_device(provider_device, &config, &device) == 1;
    ok &= device.logical_device >= 0;
    ok &= device.physical_device >= 0;
    ok &= device.compute_capability >= config.min_compute_capability;
    ok &= device.min_expert_bytes == 1024;
    ok &= api.query_device(cpu->device, &config, &device) == 0;

    const size_t expert_size = ggml_row_size(GGML_TYPE_Q4_0, n_in) * n_out;
    ggml_moe_cache_shape_caps shape = {};
    ok &= api.query_shape(
            GGML_TYPE_Q4_0, n_in, n_out, 64, expert_size, &shape) == 1;
    ok &= shape.pool_bytes == expert_size * 64;
    ok &= shape.minimum_bytes == shape.scratch_bytes + shape.pool_bytes;
    ok &= api.query_shape(
            GGML_TYPE_F32, n_in, n_out, 64, expert_size, &shape) == 0;
    ok &= api.query_shape(
            GGML_TYPE_Q4_0, n_in, n_out, 64, expert_size - 1, &shape) == 0;
    ok &= api.query_shape(
            GGML_TYPE_Q4_0, n_in, n_out, 1, expert_size, &shape) == 1;
    ok &= shape.pool_bytes == expert_size * 64;
    ok &= api.query_shape(
            GGML_TYPE_Q4_0, n_in, n_out, 0, expert_size, &shape) == 0;

    set_env("GGML_CUDA_MOE_CACHE", "0");
    set_env("GGML_CUDA_MOE_CACHE_MODE", "off");
    ok &= api.query_config(-1, 0, &config) == 0;

    set_env("GGML_CUDA_MOE_CACHE", "1");
    set_env("GGML_CUDA_MOE_CACHE_MODE", "auto");
    set_env("GGML_CUDA_MOE_CACHE_BUDGET_MB", nullptr);
    set_env("GGML_CUDA_MOE_CACHE_MIN_EXPERT_KB", nullptr);
    set_env("GGML_CUDA_MOE_CACHE_MAX_BATCH", nullptr);
    set_env("GGML_CUDA_MOE_CACHE_MIN_CC", nullptr);
    set_env("GGML_CUDA_MOE_CACHE_OVERLAP_CPU_ROWS", nullptr);
    ok &= api.query_config(1, 0, &config) == 1;
    ok &= config.min_devices == 1; // automatic mode now accepts a single device
    ok &= config.minimum_slab_bytes == 1024u * 1024 * 1024;
    ok &= config.min_compute_capability == 800;
    ok &= config.min_expert_bytes == 512u * 1024;
    ok &= config.min_expert_explicit == 0;
    ok &= config.max_batch == 8;
    ok &= config.overlap_cpu_rows == -1;
    ok &= api.query_device(provider_device, &config, &device) == 1;
    ok &= device.min_expert_bytes == 512u * 1024;

    ok &= api.query_config(0, 0, &config) == 1;
    ok &= config.min_devices == 1;
    ok &= config.minimum_slab_bytes == 0;
    ok &= config.min_compute_capability == 700;
    ok &= config.min_expert_bytes == 1024u * 1024;
    ok &= config.min_expert_explicit == 0;
    ok &= config.max_batch == 8;
    ok &= config.overlap_cpu_rows == -1;
    ok &= api.query_device(provider_device, &config, &device) == 1;
    ok &= device.min_expert_bytes == (device.compute_capability >= 800
            ? 512u * 1024 : 1024u * 1024);
    configure_cache(nullptr);

    printf("cache-capabilities: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

struct test_graph {
    ggml_context * ctx = nullptr;
    ggml_tensor * out = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
};

static test_graph make_graph(
        ggml_backend_t cpu,
        ggml_tensor * weights,
        ggml_tensor * activations,
        ggml_tensor * ids) {
    ggml_init_params params = {
        8 * ggml_tensor_overhead() + ggml_graph_overhead(),
        nullptr,
        true,
    };
    test_graph result;
    result.ctx = ggml_init(params);
    if (!result.ctx) {
        return result;
    }
    result.out = ggml_mul_mat_id(result.ctx, weights, activations, ids);
    ggml_set_name(result.out, "moe_cache_test_out");
    result.graph = ggml_new_graph(result.ctx);
    ggml_build_forward_expand(result.graph, result.out);
    result.buffer = ggml_backend_alloc_ctx_tensors(result.ctx, cpu);
    return result;
}

static test_graph make_fused_graph(
        ggml_backend_t cpu,
        ggml_tensor * up_weights,
        ggml_tensor * gate_weights,
        ggml_tensor * activations,
        ggml_tensor * ids) {
    ggml_init_params params = {
        12 * ggml_tensor_overhead() + ggml_graph_overhead(),
        nullptr,
        true,
    };
    test_graph result;
    result.ctx = ggml_init(params);
    if (!result.ctx) {
        return result;
    }
    ggml_tensor * up = ggml_mul_mat_id(
            result.ctx, up_weights, activations, ids);
    ggml_tensor * gate = ggml_mul_mat_id(
            result.ctx, gate_weights, activations, ids);
    result.out = ggml_swiglu_split(result.ctx, gate, up);
    ggml_set_name(up, "moe_cache_fused_up");
    ggml_set_name(gate, "moe_cache_fused_gate");
    ggml_set_name(result.out, "moe_cache_fused_out");
    result.graph = ggml_new_graph(result.ctx);
    ggml_build_forward_expand(result.graph, result.out);
    result.buffer = ggml_backend_alloc_ctx_tensors(result.ctx, cpu);
    return result;
}

static test_graph make_clamped_fused_graph(
        ggml_backend_t cpu,
        ggml_tensor * up_weights,
        ggml_tensor * gate_weights,
        ggml_tensor * activations,
        ggml_tensor * ids) {
    ggml_init_params params = {
        16 * ggml_tensor_overhead() + ggml_graph_overhead(),
        nullptr,
        true,
    };
    test_graph result;
    result.ctx = ggml_init(params);
    if (!result.ctx) {
        return result;
    }
    ggml_tensor * up = ggml_mul_mat_id(
            result.ctx, up_weights, activations, ids);
    ggml_tensor * gate = ggml_mul_mat_id(
            result.ctx, gate_weights, activations, ids);
    ggml_tensor * up_clamped =
        ggml_clamp(result.ctx, up, -0.25f, 0.25f);
    ggml_tensor * gate_clamped = ggml_clamp(
            result.ctx, gate,
            -std::numeric_limits<float>::infinity(), 0.20f);
    result.out = ggml_swiglu_split(
            result.ctx, gate_clamped, up_clamped);
    ggml_set_name(up, "moe_cache_clamped_up");
    ggml_set_name(gate, "moe_cache_clamped_gate");
    ggml_set_name(up_clamped, "moe_cache_clamped_up_limit");
    ggml_set_name(gate_clamped, "moe_cache_clamped_gate_limit");
    ggml_set_name(result.out, "moe_cache_clamped_out");
    result.graph = ggml_new_graph(result.ctx);
    ggml_build_forward_expand(result.graph, result.out);
    result.buffer = ggml_backend_alloc_ctx_tensors(result.ctx, cpu);
    return result;
}

static void free_graph(test_graph & graph) {
    if (graph.buffer) {
        ggml_backend_buffer_free(graph.buffer);
    }
    if (graph.ctx) {
        ggml_free(graph.ctx);
    }
    graph = {};
}

static bool run_scenario(
        const char * name,
        const char * fail_stage,
        ggml_backend_t gpu,
        ggml_backend_t cpu,
        test_graph & graph,
        const std::vector<float> & reference,
        log_capture & capture,
        const char * max_batch = "1",
        const char * required_field = nullptr,
        const char * dedicated_mmv = "1") {
    configure_cache(fail_stage, max_batch, dedicated_mmv);
    capture.clear();

    ggml_backend_t backends[] = { gpu, cpu };
    ggml_backend_sched_t scheduler = ggml_backend_sched_new(
            backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, false);
    if (!scheduler) {
        fprintf(stderr, "%s: failed to create scheduler\n", name);
        return false;
    }
    ggml_backend_sched_set_tensor_backend(scheduler, graph.out, cpu);
    if (!ggml_backend_sched_alloc_graph(scheduler, graph.graph) ||
        ggml_backend_sched_get_tensor_backend(scheduler, graph.out) != cpu) {
        fprintf(stderr, "%s: MUL_MAT_ID was not assigned to CPU\n", name);
        ggml_backend_sched_free(scheduler);
        return false;
    }

    bool output_ok = true;
    bool live_hit = false;
    std::vector<float> actual(reference.size());
    for (int step = 0; step < max_steps; step++) {
        const enum ggml_status status =
            ggml_backend_sched_graph_compute(scheduler, graph.graph);
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "%s: graph compute failed at step %d: %s\n",
                    name, step, ggml_status_to_string(status));
            output_ok = false;
            break;
        }
        ggml_backend_tensor_get(
                graph.out, actual.data(), 0, actual.size() * sizeof(float));
        if (!compare_output(reference, actual, 5e-4)) {
            fprintf(stderr, "%s: output mismatch at step %d\n", name, step);
            output_ok = false;
            break;
        }
        if (!fail_stage && has_positive_field(capture.get(), "hits=")) {
            live_hit = true;
        }
        if (step >= 64) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    ggml_backend_sched_free(scheduler);
    const std::string log = capture.get();
    bool stage_ok = false;
    if (!fail_stage) {
        stage_ok = live_hit &&
            log.find("[moe-cache] enabled:") != std::string::npos &&
            max_field_value(log, "dispatch-fail=") == 0 &&
            max_field_value(log, "collect-fail=") == 0 &&
            (!required_field || has_positive_field(log, required_field));
    } else if (strcmp(fail_stage, "dispatch") == 0) {
        stage_ok = has_positive_field(log, "dispatch-fail=");
    } else if (strcmp(fail_stage, "collect") == 0) {
        stage_ok = has_positive_field(log, "collect-fail=");
    } else if (strcmp(fail_stage, "insert") == 0) {
        stage_ok = has_positive_field(log, "fill-fail=");
    } else if (strcmp(fail_stage, "slab") == 0) {
        stage_ok = log.find("allocation failed") != std::string::npos;
    }
    if (required_field) {
        stage_ok &= has_positive_field(log, required_field);
    }
    if (!stage_ok) {
        fprintf(stderr, "%s: cache stage was not observed\n%s", name, log.c_str());
    }
    printf("%s: %s\n", name, output_ok && stage_ok ? "OK" : "FAIL");
    return output_ok && stage_ok;
}

static bool run_multi_token_scenario(
        ggml_backend_t gpu,
        ggml_backend_t cpu,
        ggml_tensor * weights,
        ggml_tensor * gate_weights,
        log_capture & capture,
        bool fused_enabled,
        bool dedup_enabled) {
    const ggml_init_params params = {
        4 * ggml_tensor_overhead(),
        nullptr,
        true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "cache-multi-token: failed to create tensor context\n");
        return false;
    }

    ggml_tensor * ids = ggml_new_tensor_2d(
            ctx, GGML_TYPE_I32, multi_n_used, multi_n_tokens);
    ggml_tensor * activations = ggml_new_tensor_3d(
            ctx, GGML_TYPE_F32, n_in, 1, multi_n_tokens);
    ggml_set_name(ids, "moe_cache_multi_token_ids");
    ggml_set_name(activations, "moe_cache_multi_token_activations");

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    if (!buffer) {
        fprintf(stderr, "cache-multi-token: failed to allocate CPU tensors\n");
        ggml_free(ctx);
        return false;
    }

    std::vector<int32_t> ids_data(multi_n_used * multi_n_tokens);
    for (int64_t token = 0; token < multi_n_tokens; token++) {
        for (int64_t id = 0; id < multi_n_used; id++) {
            ids_data[token * multi_n_used + id] =
                (int32_t) ((token * 7 + id * 3) % n_expert);
        }
    }
    ggml_backend_tensor_set(
            ids, ids_data.data(), 0,
            ids_data.size() * sizeof(ids_data[0]));

    std::vector<float> activation_data(ggml_nelements(activations));
    for (size_t index = 0; index < activation_data.size(); index++) {
        activation_data[index] =
            0.31f * std::sin((float) index * 0.053f) -
            0.12f * std::cos((float) index * 0.089f);
    }
    ggml_backend_tensor_set(
            activations, activation_data.data(), 0,
            activation_data.size() * sizeof(float));

    test_graph graph = make_graph(cpu, weights, activations, ids);
    test_graph fused_graph = make_fused_graph(
            cpu, weights, gate_weights, activations, ids);
    test_graph clamped_fused_graph = make_clamped_fused_graph(
            cpu, weights, gate_weights, activations, ids);
    bool ok = graph.ctx && graph.buffer;
    if (fused_enabled) {
        ok = ok && fused_graph.ctx && fused_graph.buffer &&
            clamped_fused_graph.ctx && clamped_fused_graph.buffer;
    }
    if (!ok) {
        fprintf(stderr, "cache-multi-token: failed to create graphs\n");
    } else {
        set_env("GGML_CUDA_MOE_CACHE", "0");
        auto make_reference = [&](const char * name, test_graph & test,
                                  std::vector<float> & reference) {
            if (ggml_backend_graph_compute(cpu, test.graph) !=
                    GGML_STATUS_SUCCESS) {
                fprintf(stderr, "%s: CPU reference compute failed\n", name);
                return false;
            }
            reference.resize(ggml_nelements(test.out));
            ggml_backend_tensor_get(
                    test.out, reference.data(), 0,
                    reference.size() * sizeof(float));
            return true;
        };
        std::vector<float> reference;
        std::vector<float> fused_reference;
        std::vector<float> clamped_fused_reference;
        ok = make_reference("cache-multi-token", graph, reference) &&
            make_reference(
                    "cache-fused-multi-token", fused_graph,
                    fused_reference) &&
            make_reference(
                    "cache-fused-clamped-multi-token",
                    clamped_fused_graph, clamped_fused_reference);
        ok = ok && run_scenario(
                "cache-multi-token", nullptr, gpu, cpu,
                graph, reference, capture, nullptr,
                dedup_enabled ? "act-dedup=" : nullptr);
        if (!fused_enabled) {
            printf("cache-fused-multi-token: SKIP (provider-specific)\n");
            printf("cache-fused-clamped-multi-token: SKIP (provider-specific)\n");
        } else {
            ok = ok && run_scenario(
                    "cache-fused-multi-token", nullptr, gpu, cpu,
                    fused_graph, fused_reference, capture,
                    nullptr, "fusion-nodes=");
            ok = ok && run_scenario(
                    "cache-fused-clamped-multi-token", nullptr, gpu, cpu,
                    clamped_fused_graph, clamped_fused_reference, capture,
                    nullptr, "fusion-nodes=");
        }
    }

    free_graph(clamped_fused_graph);
    free_graph(fused_graph);
    free_graph(graph);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

struct mxfp4_fixture {
    ggml_context * ctx = nullptr;
    ggml_tensor * weights = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    test_graph graph;
    std::vector<float> reference;
};

static void free_mxfp4_fixture(mxfp4_fixture & fixture) {
    free_graph(fixture.graph);
    if (fixture.buffer) {
        ggml_backend_buffer_free(fixture.buffer);
    }
    if (fixture.ctx) {
        ggml_free(fixture.ctx);
    }
    fixture = {};
}

static bool init_mxfp4_fixture(
        const char * name,
        const char * weight_name,
        int64_t mxfp4_n_in,
        int64_t mxfp4_n_out,
        ggml_backend_t cpu,
        mxfp4_fixture & fixture) {
    const ggml_init_params params = {
        8 * ggml_tensor_overhead(),
        nullptr,
        true,
    };
    fixture.ctx = ggml_init(params);
    if (!fixture.ctx) {
        fprintf(stderr, "%s: failed to create tensor context\n", name);
        return false;
    }

    fixture.weights = ggml_new_tensor_3d(
            fixture.ctx, GGML_TYPE_MXFP4,
            mxfp4_n_in, mxfp4_n_out, n_expert);
    ggml_tensor * ids = ggml_new_tensor_2d(
            fixture.ctx, GGML_TYPE_I32, n_used, n_tokens);
    ggml_tensor * activations = ggml_new_tensor_3d(
            fixture.ctx, GGML_TYPE_F32, mxfp4_n_in, 1, n_tokens);
    ggml_set_name(fixture.weights, weight_name);
    ggml_set_name(ids, "moe_cache_mxfp4_ids");
    ggml_set_name(activations, "moe_cache_mxfp4_activations");

    fixture.buffer = ggml_backend_alloc_ctx_tensors(fixture.ctx, cpu);
    if (!fixture.buffer) {
        fprintf(stderr, "%s: failed to allocate CPU tensors\n", name);
        free_mxfp4_fixture(fixture);
        return false;
    }
    ggml_backend_buffer_set_usage(
            fixture.buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    std::vector<float> weights_f32(ggml_nelements(fixture.weights));
    for (size_t index = 0; index < weights_f32.size(); index++) {
        weights_f32[index] =
            0.18f * std::sin((float) (index % 991) * 0.023f) +
            0.07f * std::cos((float) (index % 421) * 0.037f);
    }
    std::vector<uint8_t> weights_mxfp4(ggml_nbytes(fixture.weights));
    const size_t quantized = ggml_quantize_chunk(
            GGML_TYPE_MXFP4, weights_f32.data(), weights_mxfp4.data(),
            0, mxfp4_n_out * n_expert, mxfp4_n_in, nullptr);
    if (quantized != weights_mxfp4.size()) {
        fprintf(stderr, "%s: unexpected quantized size: %zu != %zu\n",
                name, quantized, weights_mxfp4.size());
        free_mxfp4_fixture(fixture);
        return false;
    }
    ggml_backend_tensor_set(
            fixture.weights, weights_mxfp4.data(), 0, weights_mxfp4.size());

    const int32_t ids_data[n_used] = { 0, 1 };
    ggml_backend_tensor_set(ids, ids_data, 0, sizeof(ids_data));
    std::vector<float> activation_data(ggml_nelements(activations));
    for (size_t index = 0; index < activation_data.size(); index++) {
        activation_data[index] =
            0.45f * std::sin((float) index * 0.061f) -
            0.16f * std::cos((float) index * 0.097f);
    }
    ggml_backend_tensor_set(
            activations, activation_data.data(), 0,
            activation_data.size() * sizeof(float));

    fixture.graph = make_graph(
            cpu, fixture.weights, activations, ids);
    if (!fixture.graph.ctx || !fixture.graph.buffer) {
        fprintf(stderr, "%s: failed to create graph\n", name);
        free_mxfp4_fixture(fixture);
        return false;
    }

    set_env("GGML_CUDA_MOE_CACHE", "0");
    if (ggml_backend_graph_compute(cpu, fixture.graph.graph) !=
            GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: CPU reference compute failed\n", name);
        free_mxfp4_fixture(fixture);
        return false;
    }
    fixture.reference.resize(ggml_nelements(fixture.graph.out));
    ggml_backend_tensor_get(
            fixture.graph.out, fixture.reference.data(), 0,
            fixture.reference.size() * sizeof(float));
    return true;
}

static bool run_mxfp4_shared_pool(
        ggml_backend_t gpu,
        ggml_backend_t cpu,
        log_capture & capture) {
    mxfp4_fixture up;
    mxfp4_fixture down;
    bool initialized = init_mxfp4_fixture(
            "cache-mxfp4-up", "blk.6.ffn_up_exps.weight",
            n_in, n_out, cpu, up);
    initialized = initialized && init_mxfp4_fixture(
            "cache-mxfp4-down", "blk.6.ffn_down_exps.weight",
            n_out, n_in, cpu, down);
    if (!initialized) {
        free_mxfp4_fixture(up);
        free_mxfp4_fixture(down);
        return false;
    }
    if (ggml_nbytes(up.weights) != ggml_nbytes(down.weights)) {
        fprintf(stderr, "cache-mxfp4-shared-pool: expert sizes differ\n");
        free_mxfp4_fixture(up);
        free_mxfp4_fixture(down);
        return false;
    }

    configure_cache(nullptr);
    capture.clear();
    ggml_backend_t backends[] = { gpu, cpu };
    ggml_backend_sched_t scheduler = ggml_backend_sched_new(
            backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, false);
    if (!scheduler) {
        fprintf(stderr, "cache-mxfp4-shared-pool: failed to create scheduler\n");
        free_mxfp4_fixture(up);
        free_mxfp4_fixture(down);
        return false;
    }

    auto run_orientation = [&](const char * name, mxfp4_fixture & fixture,
                               long long hits_before) {
        ggml_backend_sched_reset(scheduler);
        ggml_backend_sched_set_tensor_backend(
                scheduler, fixture.graph.out, cpu);
        if (!ggml_backend_sched_alloc_graph(
                    scheduler, fixture.graph.graph) ||
            ggml_backend_sched_get_tensor_backend(
                    scheduler, fixture.graph.out) != cpu) {
            fprintf(stderr, "%s: MUL_MAT_ID was not assigned to CPU\n", name);
            return false;
        }

        std::vector<float> actual(fixture.reference.size());
        for (int step = 0; step < max_steps; step++) {
            const enum ggml_status status =
                ggml_backend_sched_graph_compute(
                        scheduler, fixture.graph.graph);
            if (status != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "%s: graph compute failed at step %d: %s\n",
                        name, step, ggml_status_to_string(status));
                return false;
            }
            ggml_backend_tensor_get(
                    fixture.graph.out, actual.data(), 0,
                    actual.size() * sizeof(float));
            if (!compare_output(fixture.reference, actual, 5e-4)) {
                fprintf(stderr, "%s: output mismatch at step %d\n", name, step);
                return false;
            }
            const std::string live_log = capture.get();
            if (max_field_value(live_log, "hits=") > hits_before) {
                return max_field_value(live_log, "dispatch-fail=") == 0 &&
                       max_field_value(live_log, "collect-fail=") == 0;
            }
            if (step >= 64) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        fprintf(stderr, "%s: cache hit was not observed\n", name);
        return false;
    };

    const bool up_ok = run_orientation("cache-mxfp4-up", up, 0);
    const long long up_hits = max_field_value(capture.get(), "hits=");
    const bool down_ok = up_ok &&
        run_orientation("cache-mxfp4-down", down, up_hits);
    ggml_backend_sched_free(scheduler);

    const std::string log = capture.get();
    const bool one_pool = count_occurrences(log, " pool[") == 1;
    const bool no_failures =
        max_field_value(log, "dispatch-fail=") == 0 &&
        max_field_value(log, "collect-fail=") == 0;
    if (!one_pool) {
        fprintf(stderr,
                "cache-mxfp4-shared-pool: expected one shared pool\n%s",
                log.c_str());
    }
    const bool ok = up_ok && down_ok && one_pool && no_failures;
    printf("cache-mxfp4-shared-pool: %s\n", ok ? "OK" : "FAIL");
    free_mxfp4_fixture(up);
    free_mxfp4_fixture(down);
    return ok;
}

static bool run_invalidation_scenario(
        ggml_backend_t gpu,
        ggml_backend_t cpu,
        test_graph & graph,
        ggml_tensor * weights,
        const std::vector<uint8_t> & replacement_row,
        const std::vector<float> & old_reference,
        log_capture & capture) {
    configure_cache(nullptr);
    capture.clear();

    ggml_backend_t backends[] = { gpu, cpu };
    ggml_backend_sched_t scheduler = ggml_backend_sched_new(
            backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, false);
    if (!scheduler) {
        fprintf(stderr, "cache-invalidate: failed to create scheduler\n");
        return false;
    }
    ggml_backend_sched_set_tensor_backend(scheduler, graph.out, cpu);
    if (!ggml_backend_sched_alloc_graph(scheduler, graph.graph) ||
        ggml_backend_sched_get_tensor_backend(scheduler, graph.out) != cpu) {
        fprintf(stderr, "cache-invalidate: MUL_MAT_ID was not assigned to CPU\n");
        ggml_backend_sched_free(scheduler);
        return false;
    }

    std::vector<float> actual(old_reference.size());
    long long hits_before = -1;
    bool output_ok = true;
    for (int step = 0; step < max_steps; step++) {
        const enum ggml_status status =
            ggml_backend_sched_graph_compute(scheduler, graph.graph);
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "cache-invalidate: warmup failed at step %d: %s\n",
                    step, ggml_status_to_string(status));
            output_ok = false;
            break;
        }
        ggml_backend_tensor_get(
                graph.out, actual.data(), 0, actual.size() * sizeof(float));
        if (!compare_output(old_reference, actual, 5e-4)) {
            fprintf(stderr, "cache-invalidate: warmup mismatch at step %d\n", step);
            output_ok = false;
            break;
        }
        hits_before = max_field_value(capture.get(), "hits=");
        if (hits_before > 0) {
            break;
        }
        if (step >= 64) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    if (output_ok && hits_before <= 0) {
        fprintf(stderr, "cache-invalidate: no cache hit before mutation\n");
        output_ok = false;
    }

    std::vector<float> new_reference(old_reference.size());
    if (output_ok) {
        ggml_backend_tensor_set(
                weights, replacement_row.data(), 0, replacement_row.size());
        if (ggml_backend_graph_compute(cpu, graph.graph) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "cache-invalidate: CPU reference compute failed\n");
            output_ok = false;
        } else {
            ggml_backend_tensor_get(
                    graph.out, new_reference.data(), 0,
                    new_reference.size() * sizeof(float));
            float max_change = 0.0f;
            for (size_t index = 0; index < new_reference.size(); index++) {
                max_change = std::max(
                        max_change, std::abs(new_reference[index] - old_reference[index]));
            }
            if (max_change < 0.01f) {
                fprintf(stderr, "cache-invalidate: mutation did not change the reference\n");
                output_ok = false;
            }
        }
    }

    capture.clear();
    bool repopulated = false;
    if (output_ok) {
        for (int step = 0; step < max_steps; step++) {
            const enum ggml_status status =
                ggml_backend_sched_graph_compute(scheduler, graph.graph);
            if (status != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "cache-invalidate: compute failed at step %d: %s\n",
                        step, ggml_status_to_string(status));
                output_ok = false;
                break;
            }
            ggml_backend_tensor_get(
                    graph.out, actual.data(), 0, actual.size() * sizeof(float));
            if (!compare_output(new_reference, actual, 5e-4)) {
                fprintf(stderr, "cache-invalidate: stale output at step %d\n", step);
                output_ok = false;
                break;
            }
            if (max_field_value(capture.get(), "hits=") > hits_before) {
                repopulated = true;
                break;
            }
            if (step >= 64) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }

    ggml_backend_sched_free(scheduler);
    if (output_ok && !repopulated) {
        fprintf(stderr, "cache-invalidate: mutated expert was not repopulated\n%s",
                capture.get().c_str());
        output_ok = false;
    }
    printf("cache-invalidate: %s\n", output_ok ? "OK" : "FAIL");
    return output_ok;
}

constexpr int64_t stress_n_out  = 65;
constexpr int64_t stress_n_used = 64;

struct stress_fixture {
    ggml_context * ctx = nullptr;
    ggml_tensor * weights = nullptr;
    ggml_tensor * ids = nullptr;
    ggml_tensor * activations = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    test_graph graph;
    std::vector<float> reference;
};

static void free_stress_fixture(stress_fixture & fixture) {
    free_graph(fixture.graph);
    if (fixture.buffer) {
        ggml_backend_buffer_free(fixture.buffer);
    }
    if (fixture.ctx) {
        ggml_free(fixture.ctx);
    }
    fixture = {};
}

static bool init_stress_fixture(stress_fixture & fixture, ggml_backend_t cpu) {
    const ggml_init_params params = {
        8 * ggml_tensor_overhead(),
        nullptr,
        true,
    };
    fixture.ctx = ggml_init(params);
    if (!fixture.ctx) {
        return false;
    }

    fixture.weights = ggml_new_tensor_3d(
            fixture.ctx, GGML_TYPE_Q4_0, n_in, stress_n_out, n_expert);
    fixture.ids = ggml_new_tensor_2d(
            fixture.ctx, GGML_TYPE_I32, stress_n_used, n_tokens);
    fixture.activations = ggml_new_tensor_3d(
            fixture.ctx, GGML_TYPE_F32, n_in, stress_n_used, n_tokens);
    ggml_set_name(fixture.weights, "blk.1.ffn_up_exps.weight");
    ggml_set_name(fixture.ids, "moe_cache_stress_ids");
    ggml_set_name(fixture.activations, "moe_cache_stress_activations");

    fixture.buffer = ggml_backend_alloc_ctx_tensors(fixture.ctx, cpu);
    if (!fixture.buffer) {
        free_stress_fixture(fixture);
        return false;
    }
    ggml_backend_buffer_set_usage(
            fixture.buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    std::vector<float> weights_f32(ggml_nelements(fixture.weights));
    for (size_t index = 0; index < weights_f32.size(); index++) {
        weights_f32[index] =
            0.13f * std::sin((float) (index % 983) * 0.019f) -
            0.04f * std::cos((float) (index % 419) * 0.029f);
    }
    std::vector<uint8_t> weights_q4(ggml_nbytes(fixture.weights));
    const size_t quantized = ggml_quantize_chunk(
            GGML_TYPE_Q4_0, weights_f32.data(), weights_q4.data(),
            0, stress_n_out * n_expert, n_in, nullptr);
    if (quantized != weights_q4.size()) {
        fprintf(stderr, "stress: unexpected quantized size\n");
        free_stress_fixture(fixture);
        return false;
    }
    ggml_backend_tensor_set(
            fixture.weights, weights_q4.data(), 0, weights_q4.size());

    std::vector<int32_t> ids_data(stress_n_used);
    for (int32_t index = 0; index < stress_n_used; index++) {
        ids_data[index] = index;
    }
    ggml_backend_tensor_set(
            fixture.ids, ids_data.data(), 0,
            ids_data.size() * sizeof(ids_data[0]));

    std::vector<float> activation_data(
            ggml_nelements(fixture.activations));
    for (size_t index = 0; index < activation_data.size(); index++) {
        activation_data[index] =
            0.35f * std::sin((float) index * 0.067f) +
            0.17f * std::cos((float) index * 0.103f);
    }
    ggml_backend_tensor_set(
            fixture.activations, activation_data.data(), 0,
            activation_data.size() * sizeof(float));

    fixture.graph = make_graph(
            cpu, fixture.weights, fixture.activations, fixture.ids);
    if (!fixture.graph.ctx || !fixture.graph.buffer) {
        fprintf(stderr, "stress: failed to create graph\n");
        free_stress_fixture(fixture);
        return false;
    }

    set_env("GGML_CUDA_MOE_CACHE", "0");
    if (ggml_backend_graph_compute(cpu, fixture.graph.graph) !=
            GGML_STATUS_SUCCESS) {
        fprintf(stderr, "stress: CPU reference compute failed\n");
        free_stress_fixture(fixture);
        return false;
    }
    fixture.reference.resize(ggml_nelements(fixture.graph.out));
    ggml_backend_tensor_get(
            fixture.graph.out, fixture.reference.data(), 0,
            fixture.reference.size() * sizeof(float));
    return true;
}

static ggml_backend_sched_t make_scheduler(
        const char * name,
        ggml_backend_t gpu,
        ggml_backend_t cpu,
        test_graph & graph) {
    ggml_backend_t backends[] = { gpu, cpu };
    ggml_backend_sched_t scheduler = ggml_backend_sched_new(
            backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, false);
    if (!scheduler) {
        fprintf(stderr, "%s: failed to create scheduler\n", name);
        return nullptr;
    }
    ggml_backend_sched_set_tensor_backend(scheduler, graph.out, cpu);
    if (!ggml_backend_sched_alloc_graph(scheduler, graph.graph) ||
        ggml_backend_sched_get_tensor_backend(scheduler, graph.out) != cpu) {
        fprintf(stderr, "%s: MUL_MAT_ID was not assigned to CPU\n", name);
        ggml_backend_sched_free(scheduler);
        return nullptr;
    }
    return scheduler;
}

static bool compute_matches(
        const char * name,
        ggml_backend_sched_t scheduler,
        test_graph & graph,
        const std::vector<float> & reference,
        int step) {
    const enum ggml_status status =
        ggml_backend_sched_graph_compute(scheduler, graph.graph);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: graph compute failed at step %d: %s\n",
                name, step, ggml_status_to_string(status));
        return false;
    }
    std::vector<float> actual(reference.size());
    ggml_backend_tensor_get(
            graph.out, actual.data(), 0, actual.size() * sizeof(float));
    if (!compare_output(reference, actual, 5e-4)) {
        fprintf(stderr, "%s: output mismatch at step %d\n", name, step);
        return false;
    }
    return true;
}

static bool run_precensus_invalidation(
        ggml_backend_t gpu,
        ggml_backend_t cpu,
        test_graph & graph,
        ggml_tensor * weights,
        const uint8_t * unchanged_expert,
        size_t expert_size,
        const std::vector<float> & reference,
        log_capture & capture) {
    configure_cache(nullptr);
    capture.clear();

    ggml_backend_sched_t scheduler = make_scheduler(
            "cache-precensus-invalidate", gpu, cpu, graph);
    if (!scheduler) {
        return false;
    }

    bool output_ok = compute_matches(
            "cache-precensus-invalidate", scheduler, graph, reference, 0);
    if (output_ok) {
        ggml_backend_tensor_set(
                weights, unchanged_expert,
                (n_expert - 1) * expert_size, expert_size);
    }
    for (int step = 1; step < 80 && output_ok; step++) {
        output_ok = compute_matches(
                "cache-precensus-invalidate", scheduler, graph,
                reference, step);
        if (step >= 64) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    ggml_backend_sched_free(scheduler);
    const std::string log = capture.get();
    const bool census_ok =
        log.find("slots=64 ") != std::string::npos;
    if (!census_ok) {
        fprintf(stderr,
                "cache-precensus-invalidate: tensor census was not stable\n%s",
                log.c_str());
    }
    printf("cache-precensus-invalidate: %s\n",
            output_ok && census_ok ? "OK" : "FAIL");
    return output_ok && census_ok;
}

static bool run_concurrent_sessions(
        ggml_backend_dev_t provider_device,
        ggml_backend_t gpu,
        ggml_backend_t cpu,
        stress_fixture & fixture,
        log_capture & capture) {
    configure_cache(nullptr);
    set_env("GGML_CUDA_MOE_CACHE_STATS", "0");
    capture.clear();

    ggml_backend_t gpu_second =
        ggml_backend_dev_init(provider_device, nullptr);
    ggml_backend_t cpu_second = init_cpu_backend();
    if (!gpu_second || !cpu_second) {
        fprintf(stderr, "cache-concurrent: failed to create second backends\n");
        if (gpu_second) {
            ggml_backend_free(gpu_second);
        }
        if (cpu_second) {
            ggml_backend_free(cpu_second);
        }
        return false;
    }
    test_graph second_graph = make_graph(
            cpu_second, fixture.weights, fixture.activations, fixture.ids);
    if (!second_graph.ctx || !second_graph.buffer) {
        fprintf(stderr, "cache-concurrent: failed to create second graph\n");
        free_graph(second_graph);
        ggml_backend_free(gpu_second);
        ggml_backend_free(cpu_second);
        return false;
    }

    ggml_backend_sched_t first = make_scheduler(
            "cache-concurrent-1", gpu, cpu, fixture.graph);
    ggml_backend_sched_t second = make_scheduler(
            "cache-concurrent-2", gpu_second, cpu_second, second_graph);
    if (!first || !second) {
        if (first) {
            ggml_backend_sched_free(first);
        }
        if (second) {
            ggml_backend_sched_free(second);
        }
        free_graph(second_graph);
        ggml_backend_free(gpu_second);
        ggml_backend_free(cpu_second);
        return false;
    }

    constexpr int concurrent_steps = 112;
    std::atomic<int> ready{0};
    std::atomic<bool> start{false};
    std::atomic<bool> output_ok{true};
    auto run = [&](const char * name, ggml_backend_sched_t scheduler,
                   test_graph & graph) {
        ready.fetch_add(1);
        while (!start.load()) {
            std::this_thread::yield();
        }
        for (int step = 0; step < concurrent_steps && output_ok.load(); step++) {
            if (!compute_matches(
                    name, scheduler, graph, fixture.reference, step)) {
                output_ok.store(false);
                break;
            }
            if (step >= 64) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    };

    std::thread first_thread(
            run, "cache-concurrent-1", first, std::ref(fixture.graph));
    std::thread second_thread(
            run, "cache-concurrent-2", second, std::ref(second_graph));
    while (ready.load() != 2) {
        std::this_thread::yield();
    }
    start.store(true);
    first_thread.join();
    second_thread.join();

    ggml_backend_sched_free(first);
    ggml_backend_sched_free(second);
    const std::string log = capture.get();
    const bool cache_ok =
        count_occurrences(log, " pool[") >= 2 &&
        count_field_at_least(log, "hits=", stress_n_used) >= 2 &&
        count_field_at_least(log, "used=", stress_n_used) >= 2;
    if (!cache_ok) {
        fprintf(stderr, "cache-concurrent: full cache use was not observed\n%s",
                log.c_str());
    }

    free_graph(second_graph);
    ggml_backend_free(gpu_second);
    ggml_backend_free(cpu_second);
    printf("cache-concurrent: %s\n",
            output_ok.load() && cache_ok ? "OK" : "FAIL");
    return output_ok.load() && cache_ok;
}

static bool run_fused_concurrent_sessions(
        ggml_backend_dev_t provider_device,
        ggml_backend_t gpu,
        ggml_backend_t cpu,
        ggml_tensor * up_weights,
        ggml_tensor * gate_weights,
        ggml_tensor * activations,
        ggml_tensor * ids,
        test_graph & first_graph,
        const std::vector<float> & reference,
        log_capture & capture) {
    configure_cache(nullptr);
    set_env("GGML_CUDA_MOE_CACHE_BUDGET_MB", "16");
    set_env("GGML_CUDA_MOE_CACHE_STATS", "0");
    capture.clear();

    ggml_backend_t gpu_second =
        ggml_backend_dev_init(provider_device, nullptr);
    ggml_backend_t cpu_second = init_cpu_backend();
    if (!gpu_second || !cpu_second) {
        fprintf(stderr, "cache-fused-concurrent: failed to create second backends\n");
        if (gpu_second) {
            ggml_backend_free(gpu_second);
        }
        if (cpu_second) {
            ggml_backend_free(cpu_second);
        }
        return false;
    }

    test_graph second_graph = make_fused_graph(
            cpu_second, up_weights, gate_weights, activations, ids);
    if (!second_graph.ctx || !second_graph.buffer) {
        fprintf(stderr, "cache-fused-concurrent: failed to create second graph\n");
        free_graph(second_graph);
        ggml_backend_free(gpu_second);
        ggml_backend_free(cpu_second);
        return false;
    }

    ggml_backend_sched_t first = make_scheduler(
            "cache-fused-concurrent-1", gpu, cpu, first_graph);
    ggml_backend_sched_t second = make_scheduler(
            "cache-fused-concurrent-2", gpu_second, cpu_second,
            second_graph);
    if (!first || !second) {
        if (first) {
            ggml_backend_sched_free(first);
        }
        if (second) {
            ggml_backend_sched_free(second);
        }
        free_graph(second_graph);
        ggml_backend_free(gpu_second);
        ggml_backend_free(cpu_second);
        return false;
    }

    std::atomic<int> ready{0};
    std::atomic<bool> start{false};
    std::atomic<bool> output_ok{true};
    auto run = [&](const char * name, ggml_backend_sched_t scheduler,
                   test_graph & graph) {
        ready.fetch_add(1);
        while (!start.load()) {
            std::this_thread::yield();
        }
        for (int step = 0; step < max_steps && output_ok.load(); step++) {
            if (!compute_matches(name, scheduler, graph, reference, step)) {
                output_ok.store(false);
                break;
            }
            if (step >= 64) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    };

    std::thread first_thread(
            run, "cache-fused-concurrent-1", first,
            std::ref(first_graph));
    std::thread second_thread(
            run, "cache-fused-concurrent-2", second,
            std::ref(second_graph));
    while (ready.load() != 2) {
        std::this_thread::yield();
    }
    start.store(true);
    first_thread.join();
    second_thread.join();

    ggml_backend_sched_free(first);
    ggml_backend_sched_free(second);
    const std::string log = capture.get();
    const bool cache_ok =
        count_field_at_least(log, "fusion-nodes=", 1) >= 2 &&
        max_field_value(log, "dispatch-fail=") == 0 &&
        max_field_value(log, "collect-fail=") == 0;
    if (!cache_ok) {
        fprintf(stderr,
                "cache-fused-concurrent: fusion was not observed in both sessions\n%s",
                log.c_str());
    }

    free_graph(second_graph);
    ggml_backend_free(gpu_second);
    ggml_backend_free(cpu_second);
    configure_cache(nullptr);
    printf("cache-fused-concurrent: %s\n",
            output_ok.load() && cache_ok ? "OK" : "FAIL");
    return output_ok.load() && cache_ok;
}

static bool run_fused_repeated_lifecycle(
        ggml_backend_t gpu,
        ggml_backend_t cpu,
        test_graph & graph,
        const std::vector<float> & reference,
        log_capture & capture) {
    configure_cache(nullptr);
    constexpr int cycles = 8;
    bool ok = true;
    for (int cycle = 0; cycle < cycles && ok; cycle++) {
        capture.clear();
        ggml_backend_sched_t scheduler = make_scheduler(
                "cache-fused-lifecycle", gpu, cpu, graph);
        if (!scheduler) {
            ok = false;
            break;
        }

        bool fused = false;
        for (int step = 0; step < max_steps && ok && !fused; step++) {
            ok = compute_matches(
                    "cache-fused-lifecycle", scheduler, graph,
                    reference, step);
            fused = has_positive_field(
                    capture.get(), "fusion-nodes=");
            if (step >= 64 && !fused) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        ok &= fused;
        ggml_backend_sched_free(scheduler);
    }

    printf("cache-fused-lifecycle: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

static bool run_repeated_lifecycle(
        ggml_backend_t gpu,
        ggml_backend_t cpu,
        stress_fixture & fixture,
        log_capture & capture) {
    configure_cache(nullptr);
    capture.clear();

    constexpr int cycles = 8;
    constexpr int census_steps = 65;
    bool output_ok = true;
    for (int cycle = 0; cycle < cycles && output_ok; cycle++) {
        ggml_backend_sched_t scheduler = make_scheduler(
                "cache-lifecycle", gpu, cpu, fixture.graph);
        if (!scheduler) {
            output_ok = false;
            break;
        }
        for (int step = 0; step < census_steps; step++) {
            const enum ggml_status status =
                ggml_backend_sched_graph_compute(scheduler, fixture.graph.graph);
            if (status != GGML_STATUS_SUCCESS) {
                fprintf(stderr,
                        "cache-lifecycle: compute failed in cycle %d step %d: %s\n",
                        cycle, step, ggml_status_to_string(status));
                output_ok = false;
                break;
            }
        }
        if (output_ok) {
            std::vector<float> actual(fixture.reference.size());
            ggml_backend_tensor_get(
                    fixture.graph.out, actual.data(), 0,
                    actual.size() * sizeof(float));
            output_ok = compare_output(
                    fixture.reference, actual, 5e-4);
            if (!output_ok) {
                fprintf(stderr,
                        "cache-lifecycle: output mismatch in cycle %d\n", cycle);
            }
        }
        ggml_backend_sched_free(scheduler);
    }

    const std::string log = capture.get();
    const bool cache_ok =
        count_occurrences(log, " pool[") >= cycles &&
        has_positive_field(log, "enqueued=");
    if (!cache_ok) {
        fprintf(stderr, "cache-lifecycle: fill startup was not observed\n%s",
                log.c_str());
    }
    printf("cache-lifecycle: %s\n",
            output_ok && cache_ok ? "OK" : "FAIL");
    return output_ok && cache_ok;
}

static bool run_fill_invalidation(
        ggml_backend_t gpu,
        ggml_backend_t cpu,
        stress_fixture & fixture,
        log_capture & capture) {
    configure_cache(nullptr);
    capture.clear();

    ggml_backend_sched_t scheduler = make_scheduler(
            "cache-fill-invalidate", gpu, cpu, fixture.graph);
    if (!scheduler) {
        return false;
    }

    std::vector<float> replacement_f32(n_in * stress_n_out);
    for (size_t index = 0; index < replacement_f32.size(); index++) {
        replacement_f32[index] =
            1.1f + 0.2f * std::sin((float) index * 0.043f);
    }
    std::vector<uint8_t> replacement_q4(
            ggml_row_size(GGML_TYPE_Q4_0, n_in) * stress_n_out);
    const size_t replacement_size = ggml_quantize_chunk(
            GGML_TYPE_Q4_0, replacement_f32.data(), replacement_q4.data(),
            0, stress_n_out, n_in, nullptr);
    if (replacement_size != replacement_q4.size()) {
        fprintf(stderr,
                "cache-fill-invalidate: unexpected replacement size\n");
        ggml_backend_sched_free(scheduler);
        return false;
    }

    std::atomic<bool> stop{false};
    std::atomic<bool> mutation_started{false};
    std::atomic<bool> mutation_done{false};
    std::thread mutator([&] {
        if (capture.wait_for(
                " pool[", stop, std::chrono::seconds(5))) {
            mutation_started.store(true);
            ggml_backend_tensor_set(
                    fixture.weights, replacement_q4.data(), 0,
                    replacement_q4.size());
            mutation_done.store(true);
        }
    });

    bool output_ok = true;
    for (int step = 0; step < max_steps && !mutation_done.load(); step++) {
        const enum ggml_status status =
            ggml_backend_sched_graph_compute(scheduler, fixture.graph.graph);
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr,
                    "cache-fill-invalidate: warmup failed at step %d: %s\n",
                    step, ggml_status_to_string(status));
            output_ok = false;
            break;
        }
        if (mutation_started.load()) {
            const auto deadline =
                std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (!mutation_done.load() &&
                   std::chrono::steady_clock::now() < deadline) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            break;
        }
    }
    stop.store(true);
    capture.cv.notify_all();
    mutator.join();
    if (!mutation_started.load() || !mutation_done.load()) {
        fprintf(stderr,
                "cache-fill-invalidate: concurrent mutation did not complete\n");
        output_ok = false;
    }

    std::vector<float> new_reference(fixture.reference.size());
    if (output_ok &&
        ggml_backend_graph_compute(cpu, fixture.graph.graph) !=
            GGML_STATUS_SUCCESS) {
        fprintf(stderr,
                "cache-fill-invalidate: CPU reference compute failed\n");
        output_ok = false;
    }
    if (output_ok) {
        ggml_backend_tensor_get(
                fixture.graph.out, new_reference.data(), 0,
                new_reference.size() * sizeof(float));
        float max_change = 0.0f;
        for (size_t index = 0; index < new_reference.size(); index++) {
            max_change = std::max(
                    max_change,
                    std::abs(new_reference[index] - fixture.reference[index]));
        }
        if (max_change < 0.01f) {
            fprintf(stderr,
                    "cache-fill-invalidate: mutation did not change output\n");
            output_ok = false;
        }
    }

    capture.clear();
    bool repopulated = false;
    if (output_ok) {
        for (int step = 0; step < max_steps; step++) {
            if (!compute_matches(
                    "cache-fill-invalidate", scheduler, fixture.graph,
                    new_reference, step)) {
                output_ok = false;
                break;
            }
            if (has_positive_field(capture.get(), "hits=")) {
                repopulated = true;
                break;
            }
            if (step >= 64) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }
    ggml_backend_sched_free(scheduler);
    if (output_ok && !repopulated) {
        fprintf(stderr,
                "cache-fill-invalidate: cache was not repopulated\n%s",
                capture.get().c_str());
        output_ok = false;
    }
    printf("cache-fill-invalidate: %s\n", output_ok ? "OK" : "FAIL");
    return output_ok;
}

static void * create_direct_session(
        ggml_backend_t gpu, ggml_backend_t cpu) {
    const ggml_moe_cache_api api =
        ggml_moe_cache_get(ggml_backend_dev_backend_reg(ggml_backend_get_device(gpu)));
    if (!api.session_create) {
        return nullptr;
    }
    void * backends[] = { gpu, cpu };
    return api.session_create(backends, 2, nullptr);
}

static bool run_explicit_session_config(
        ggml_backend_t gpu, ggml_backend_t cpu) {
    configure_cache(nullptr);
    const ggml_moe_cache_api api =
        ggml_moe_cache_get(ggml_backend_dev_backend_reg(ggml_backend_get_device(gpu)));
    ggml_moe_cache_config config = {};
    if (!api.query_config ||
        !api.query_config(0, 4, &config)) {
        fprintf(stderr, "cache-explicit-config: failed to query configuration\n");
        return false;
    }

    set_env("GGML_CUDA_MOE_CACHE", "0");
    set_env("GGML_CUDA_MOE_CACHE_MODE", "off");
    void * backends[] = { gpu, cpu };
    void * session = api.session_create(backends, 2, &config);
    const bool ok = session != nullptr;
    if (session) {
        api.session_destroy(session);
    }
    bool invalid_rejected = true;
    for (int overlap_cpu_rows : { -2, 9 }) {
        config.overlap_cpu_rows = overlap_cpu_rows;
        void * invalid = api.session_create(backends, 2, &config);
        invalid_rejected &= invalid == nullptr;
        if (invalid) {
            api.session_destroy(invalid);
        }
    }
    config.overlap_cpu_rows = 0;
    for (int min_expert_explicit : { -1, 2 }) {
        config.min_expert_explicit = min_expert_explicit;
        void * invalid = api.session_create(backends, 2, &config);
        invalid_rejected &= invalid == nullptr;
        if (invalid) {
            api.session_destroy(invalid);
        }
    }
    configure_cache(nullptr);
    printf("cache-explicit-config: %s\n", ok && invalid_rejected ? "OK" : "FAIL");
    return ok && invalid_rejected;
}

static bool direct_begin_ready(
        const char * name, const void * base, size_t expert_size,
        int64_t direct_n_in, int64_t direct_n_out,
        int direct_type, int64_t direct_n_expert) {
    const ggml_moe_cache_api api = ggml_moe_cache_active();
    if (!api.begin || !api.end) {
        return false;
    }
    void * node = api.begin(
            name, base, expert_size, direct_n_in, direct_n_out,
            direct_type, direct_n_expert, 1, 1);
    if (!node) {
        return false;
    }
    api.end(node);
    return true;
}

static bool wait_for_direct_pool(
        const char * name, const void * base, size_t expert_size,
        int64_t direct_n_in, int64_t direct_n_out,
        int direct_type, int64_t direct_n_expert) {
    for (int step = 0; step < 80; step++) {
        if (direct_begin_ready(
                name, base, expert_size, direct_n_in, direct_n_out,
                direct_type, direct_n_expert)) {
            return true;
        }
    }
    return false;
}

static bool run_policy_diagnostics(
        ggml_backend_t gpu, ggml_backend_t cpu,
        ggml_tensor * weights, log_capture & capture) {
    configure_cache(nullptr);
    capture.clear();

    void * session = create_direct_session(gpu, cpu);
    if (!session) {
        fprintf(stderr, "cache-policy-diagnostics: failed to create session\n");
        return false;
    }

    const size_t expert_size = ggml_nbytes(weights) / weights->ne[2];
    const ggml_moe_cache_api api =
        ggml_moe_cache_get(ggml_backend_dev_backend_reg(ggml_backend_get_device(gpu)));
    api.session_enter(session);
    void * first = api.begin(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type,
            weights->ne[2], 1, 1);
    const bool census_waited = first == nullptr;
    if (first) {
        api.end(first);
    }
    void * second = api.begin(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type,
            weights->ne[2], 1, 1);
    const bool census_completed = second != nullptr;
    if (second) {
        api.end(second);
    }
    for (int64_t n_tokens : { 2, 3 }) {
        void * oversize = api.begin(
                weights->name, weights->data, expert_size,
                weights->ne[0], weights->ne[1], weights->type,
                weights->ne[2], n_tokens, n_tokens);
        if (oversize) {
            api.end(oversize);
        }
    }
    void * excess_rows = api.begin(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type,
            weights->ne[2], 1, 65);
    if (excess_rows) {
        api.end(excess_rows);
    }
    api.session_leave(session);
    api.session_destroy(session);

    const std::string log = capture.get();
    const bool configured =
        log.find("configured: mode=on devices=1 budget=4 MiB cap") != std::string::npos &&
        log.find("max-batch=1") != std::string::npos;
    const bool capacity =
        log.find("capacity: cap=4 MiB granted=4 MiB") != std::string::npos &&
        log.find("free=") != std::string::npos &&
        log.find("reserve=0 MiB") != std::string::npos;
    const bool bypass = count_occurrences(log, "above max-batch=1") == 1;
    const bool row_bypass = count_occurrences(log, "above row limit=64") == 1;
    const bool pool = count_occurrences(log, " pool[") == 1;
    const bool ok = census_waited && census_completed && configured &&
        capacity && bypass && row_bypass && pool;
    if (!ok) {
        fprintf(stderr, "cache-policy-diagnostics: expected diagnostics were not observed\n%s", log.c_str());
    }
    printf("cache-policy-diagnostics: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

static int direct_plan_one(
        const char * name, const void * base, size_t expert_size,
        int64_t direct_n_in, int64_t direct_n_out,
        int direct_type, int64_t direct_n_expert, int32_t expert) {
    const ggml_moe_cache_api api = ggml_moe_cache_active();
    if (!api.begin || !api.plan || !api.end) {
        return -1;
    }
    void * node = api.begin(
            name, base, expert_size, direct_n_in, direct_n_out,
            direct_type, direct_n_expert, 1, 1);
    if (!node) {
        return -1;
    }
    int32_t slot = -1;
    const int hits = api.plan(node, &expert, 1, &slot);
    api.end(node);
    return hits;
}

static int direct_plan_many(
        const char * name, const void * base, size_t expert_size,
        int64_t direct_n_in, int64_t direct_n_out,
        int direct_type, int64_t direct_n_expert,
        const int32_t * experts, int n_experts,
        int64_t direct_n_tokens = 1) {
    const ggml_moe_cache_api api = ggml_moe_cache_active();
    if (!api.begin || !api.plan || !api.end) {
        return -1;
    }
    void * node = api.begin(
            name, base, expert_size, direct_n_in, direct_n_out,
            direct_type, direct_n_expert, direct_n_tokens, n_experts);
    if (!node) {
        return -1;
    }
    std::vector<int32_t> slots(n_experts, -1);
    const int hits = api.plan(
            node, experts, n_experts, slots.data());
    api.end(node);
    return hits;
}

static bool wait_for_direct_resident(
        ggml_tensor * weights, int32_t expert) {
    const size_t expert_size = ggml_nbytes(weights) / weights->ne[2];
    (void) direct_plan_one(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type,
            weights->ne[2], expert);
    for (int attempt = 0; attempt < 200; attempt++) {
        if (direct_plan_one(
                weights->name, weights->data, expert_size,
                weights->ne[0], weights->ne[1], weights->type,
                weights->ne[2], expert) == 1) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
}

static bool run_fused_partial_invalidation(
        ggml_backend_t gpu,
        ggml_backend_t cpu,
        ggml_tensor * up_weights,
        ggml_tensor * gate_weights,
        ggml_tensor * activations,
        const uint8_t * up_row,
        const uint8_t * gate_row,
        const std::vector<float> & reference,
        log_capture & capture) {
    const ggml_moe_cache_api api =
        ggml_moe_cache_get(ggml_backend_dev_backend_reg(ggml_backend_get_device(gpu)));
    if (!api.fused_begin || !api.collect || !api.end) {
        fprintf(stderr, "cache-fused-partial: incomplete cache API\n");
        return false;
    }

    configure_cache(nullptr);
    capture.clear();
    void * session = create_direct_session(gpu, cpu);
    if (!session) {
        fprintf(stderr, "cache-fused-partial: failed to create session\n");
        return false;
    }

    const size_t expert_size =
        ggml_nbytes(up_weights) / up_weights->ne[2];
    const ggml_moe_cache_tensor_desc up = {
        up_weights->name, up_weights->data, expert_size,
        up_weights->ne[0], up_weights->ne[1], up_weights->ne[2],
        (int32_t)up_weights->type,
    };
    const ggml_moe_cache_tensor_desc gate = {
        gate_weights->name, gate_weights->data, expert_size,
        gate_weights->ne[0], gate_weights->ne[1], gate_weights->ne[2],
        (int32_t)gate_weights->type,
    };
    const float * act_rows[2] = {
        (const float *)activations->data,
        (const float *)activations->data,
    };

    api.session_enter(session);
    bool pool_ready = false;
    for (int step = 0; step < 80 && !pool_ready; step++) {
        const bool up_ready = direct_begin_ready(
                up_weights->name, up_weights->data, expert_size,
                up_weights->ne[0], up_weights->ne[1], up_weights->type,
                up_weights->ne[2]);
        const bool gate_ready = direct_begin_ready(
                gate_weights->name, gate_weights->data, expert_size,
                gate_weights->ne[0], gate_weights->ne[1], gate_weights->type,
                gate_weights->ne[2]);
        pool_ready = up_ready && gate_ready;
    }

    bool partial_ok = pool_ready &&
        wait_for_direct_resident(up_weights, 0) &&
        wait_for_direct_resident(gate_weights, 0);

    auto execute = [&](const int32_t * experts, int n_ids,
                       uint64_t expected_mask,
                       const int * reference_rows) {
        uint64_t hit_mask = 0;
        void * node = api.fused_begin(
                &up, &gate, GGML_GLU_OP_SWIGLU,
                -std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity(),
                -std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity(),
                experts, n_ids, 1, act_rows, &hit_mask);
        if (!node || hit_mask != expected_mask) {
            if (node) {
                api.end(node);
            }
            return false;
        }

        int n_hits = 0;
        float output[2 * n_out];
        float * rows[2] = { nullptr, nullptr };
        for (int row = 0; row < n_ids; row++) {
            if (hit_mask & (UINT64_C(1) << row)) {
                rows[n_hits] = output + n_hits * n_out;
                n_hits++;
            }
        }
        const bool collected =
            active_api().collect(node, n_hits, rows, n_out) == 1;
        active_api().end(node);
        if (!collected) {
            return false;
        }

        for (int hit = 0; hit < n_hits; hit++) {
            const int reference_row = reference_rows[hit];
            std::vector<float> expected(
                    reference.begin() + reference_row * n_out,
                    reference.begin() + (reference_row + 1) * n_out);
            std::vector<float> actual(
                    output + hit * n_out,
                    output + (hit + 1) * n_out);
            if (!compare_output(expected, actual, 5e-4)) {
                return false;
            }
        }
        return true;
    };

    const int32_t partial_ids[2] = { 0, 1 };
    const int partial_reference[1] = { 0 };
    partial_ok &= execute(
            partial_ids, 2, UINT64_C(1), partial_reference);

    const int32_t duplicate_ids[2] = { 0, 0 };
    const int duplicate_reference[2] = { 0, 0 };
    partial_ok &= execute(
            duplicate_ids, 2, UINT64_C(3), duplicate_reference);

    auto invalidated = [&](ggml_tensor * weights,
                           const uint8_t * unchanged_row) {
        ggml_backend_tensor_set(
                weights, unchanged_row, 0, expert_size);
        const int32_t expert = 0;
        uint64_t hit_mask = 0;
        void * node = active_api().fused_begin(
                &up, &gate, GGML_GLU_OP_SWIGLU,
                -std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity(),
                -std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity(),
                &expert, 1, 1, act_rows, &hit_mask);
        if (node) {
            active_api().end(node);
            return false;
        }
        if (hit_mask != 0 || !wait_for_direct_resident(weights, expert)) {
            return false;
        }
        const int reference_row[1] = { 0 };
        return execute(&expert, 1, UINT64_C(1), reference_row);
    };

    const bool up_invalidation_ok = partial_ok &&
        invalidated(up_weights, up_row);
    const bool gate_invalidation_ok = up_invalidation_ok &&
        invalidated(gate_weights, gate_row);
    active_api().session_leave(session);
    active_api().session_destroy(session);

    const bool stats_ok = has_positive_field(
            capture.get(), "fusion-nodes=");
    printf("cache-fused-partial-duplicate: %s\n",
            partial_ok ? "OK" : "FAIL");
    printf("cache-fused-up-invalidate: %s\n",
            up_invalidation_ok ? "OK" : "FAIL");
    printf("cache-fused-gate-invalidate: %s\n",
            gate_invalidation_ok && stats_ok ? "OK" : "FAIL");
    return gate_invalidation_ok && stats_ok;
}

static bool run_cpu_overlap_policy(
        ggml_backend_t gpu, ggml_backend_t cpu,
        ggml_tensor * weights, log_capture & capture) {
    configure_cache(nullptr);
    set_env("GGML_CUDA_MOE_CACHE_OVERLAP_CPU_ROWS", "1");
    capture.clear();

    void * session = create_direct_session(gpu, cpu);
    if (!session) {
        fprintf(stderr, "cache-cpu-overlap: failed to create session\n");
        configure_cache(nullptr);
        return false;
    }

    const size_t expert_size = ggml_nbytes(weights) / weights->ne[2];
    active_api().session_enter(session);
    bool ok = wait_for_direct_pool(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type, weights->ne[2]);
    const int32_t experts[] = { 0, 1 };
    for (int index = 0; index < 2 && ok; index++) {
        (void) direct_plan_one(
                weights->name, weights->data, expert_size,
                weights->ne[0], weights->ne[1], weights->type,
                weights->ne[2], experts[index]);
        bool resident = false;
        for (int attempt = 0; attempt < 100; attempt++) {
            if (direct_plan_one(
                    weights->name, weights->data, expert_size,
                    weights->ne[0], weights->ne[1], weights->type,
                    weights->ne[2], experts[index]) == 1) {
                resident = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        ok &= resident;
    }
    if (ok) {
        ok = direct_plan_many(
                weights->name, weights->data, expert_size,
                weights->ne[0], weights->ne[1], weights->type,
                weights->ne[2], experts, 2) == 1;
    }
    active_api().session_leave(session);
    active_api().session_destroy(session);

    ok &= has_positive_field(capture.get(), "cpu-overlap=");
    configure_cache(nullptr);
    printf("cache-cpu-overlap: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

static bool run_adaptive_cpu_overlap_policy(
        ggml_backend_t gpu, ggml_backend_t cpu,
        ggml_tensor * weights, log_capture & capture) {
    configure_cache(nullptr, "8");
    set_env("GGML_CUDA_MOE_CACHE_OVERLAP_CPU_ROWS", nullptr);
    capture.clear();

    void * session = create_direct_session(gpu, cpu);
    if (!session) {
        fprintf(stderr, "cache-cpu-overlap-auto: failed to create session\n");
        configure_cache(nullptr);
        return false;
    }

    const size_t expert_size = ggml_nbytes(weights) / weights->ne[2];
    active_api().session_enter(session);
    bool ok = wait_for_direct_pool(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type, weights->ne[2]);
    for (int32_t expert = 0; expert < 8 && ok; expert++) {
        (void) direct_plan_one(
                weights->name, weights->data, expert_size,
                weights->ne[0], weights->ne[1], weights->type,
                weights->ne[2], expert);
        bool resident = false;
        for (int attempt = 0; attempt < 100; attempt++) {
            if (direct_plan_one(
                    weights->name, weights->data, expert_size,
                    weights->ne[0], weights->ne[1], weights->type,
                    weights->ne[2], expert) == 1) {
                resident = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        ok &= resident;
    }

    const int32_t single_token_experts[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    if (ok) {
        ok = direct_plan_many(
                weights->name, weights->data, expert_size,
                weights->ne[0], weights->ne[1], weights->type,
                weights->ne[2], single_token_experts, 8) == 6;
    }

    int32_t multi_token_experts[multi_n_used * multi_n_tokens];
    for (int token = 0; token < multi_n_tokens; token++) {
        for (int id = 0; id < multi_n_used; id++) {
            multi_token_experts[token * multi_n_used + id] = id;
        }
    }
    if (ok) {
        ok = direct_plan_many(
                weights->name, weights->data, expert_size,
                weights->ne[0], weights->ne[1], weights->type,
                weights->ne[2], multi_token_experts,
                multi_n_used * multi_n_tokens, multi_n_tokens) == 30;
    }
    active_api().session_leave(session);
    active_api().session_destroy(session);

    ok &= capture.get().find("cpu-overlap=auto") != std::string::npos;
    ok &= has_positive_field(capture.get(), "cpu-overlap=");
    configure_cache(nullptr);
    printf("cache-cpu-overlap-auto: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

// Drives the adaptive cpu-overlap calibration in moe-cache.cu
// (moe_cache_calibration_on_enter/on_leave) through a full convergence:
// enough single-token decode scopes to exhaust the unmeasured global
// warm-up, then the whole round-robin candidate sweep, then confirms it
// locks in and stops adjusting. Each scope is one session_enter/plan/leave
// cycle over a single already-resident expert -- moe_cache_plan() updates
// the calibration scope stats unconditionally per planned node, so neither
// multiple experts nor a hit are required, only n_tokens == 1 per scope.
static bool run_adaptive_cpu_overlap_calibration(
        ggml_backend_t gpu, ggml_backend_t cpu,
        ggml_tensor * weights, log_capture & capture) {
    configure_cache(nullptr, "8");
    set_env("GGML_CUDA_MOE_CACHE_OVERLAP_CPU_ROWS", nullptr);
    capture.clear();

    void * session = create_direct_session(gpu, cpu);
    if (!session) {
        fprintf(stderr, "cache-cpu-overlap-calibration: failed to create session\n");
        configure_cache(nullptr);
        return false;
    }

    const size_t expert_size = ggml_nbytes(weights) / weights->ne[2];
    active_api().session_enter(session);
    bool ok = wait_for_direct_pool(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type, weights->ne[2]);
    for (int attempt = 0; attempt < 100 && ok; attempt++) {
        if (direct_plan_one(
                weights->name, weights->data, expert_size,
                weights->ne[0], weights->ne[1], weights->type,
                weights->ne[2], /* expert */ 0) == 1) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    active_api().session_leave(session);

    // moe-cache.cu: moe_cache_calib_global_warmup_scopes (64) +
    // moe_cache_calib_rounds (3) x moe_cache_calib_candidate_count (6) x
    // (moe_cache_calib_round_warmup_scopes (1) + _measured_scopes (4)) = 154.
    // Run comfortably past that so calibration has definitely completed,
    // then a few more to confirm it stays locked in afterward.
    constexpr int total_scopes = 170;
    constexpr int expected_round_lines = 18; // 3 rounds x 6 candidates
    for (int step = 0; step < total_scopes && ok; step++) {
        active_api().session_enter(session);
        direct_plan_one(
                weights->name, weights->data, expert_size,
                weights->ne[0], weights->ne[1], weights->type,
                weights->ne[2], /* expert */ 0);
        active_api().session_leave(session);
    }
    active_api().session_destroy(session);

    const std::string log = capture.get();
    ok &= log.find("cpu-overlap calibration: warm-up complete") != std::string::npos;
    ok &= count_occurrences(log, "cpu-overlap calibration: round=") == expected_round_lines;
    ok &= count_occurrences(log, "cpu-overlap calibration complete:") == 1;

    configure_cache(nullptr);
    printf("cache-cpu-overlap-calibration: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

static bool run_scope_isolation(
        ggml_backend_t gpu, ggml_backend_t cpu, ggml_tensor * weights) {
    if (!active_api().session_enter || !active_api().session_leave ||
        !active_api().begin || !active_api().end ||
        !active_api().invalidate || !active_api().session_destroy) {
        fprintf(stderr, "cache-scope: incomplete cache API\n");
        return false;
    }

    configure_cache(nullptr);
    void * outer = create_direct_session(gpu, cpu);
    if (!outer) {
        fprintf(stderr, "cache-scope: failed to create outer session\n");
        return false;
    }

    const size_t expert_size = ggml_nbytes(weights) / weights->ne[2];
    active_api().session_enter(outer);
    const bool warmed = wait_for_direct_pool(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type, weights->ne[2]);
    active_api().session_leave(outer);

    set_env("GGML_CUDA_MOE_CACHE_RESERVE_MB", "1048576");
    void * dormant = create_direct_session(gpu, cpu);
    if (!dormant) {
        fprintf(stderr, "cache-scope: failed to create dormant session\n");
        active_api().session_destroy(outer);
        return false;
    }
    active_api().session_enter(dormant);
    const bool dormant_begin = direct_begin_ready(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type, weights->ne[2]);
    active_api().session_leave(dormant);

    active_api().session_enter(outer);
    const bool outer_before = direct_begin_ready(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type, weights->ne[2]);

    active_api().session_enter(nullptr);
    const bool null_leaked = direct_begin_ready(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type, weights->ne[2]);
    active_api().session_leave(nullptr);
    const bool outer_after_null = direct_begin_ready(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type, weights->ne[2]);

    active_api().session_enter(dormant);
    const bool dormant_leaked = direct_begin_ready(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type, weights->ne[2]);
    active_api().session_leave(dormant);
    const bool outer_after_dormant = direct_begin_ready(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type, weights->ne[2]);
    active_api().session_leave(outer);

    active_api().session_destroy(dormant);
    active_api().session_destroy(outer);
    const bool ok = warmed && !dormant_begin && outer_before &&
        !null_leaked && outer_after_null &&
        !dormant_leaked && outer_after_dormant;
    printf("cache-scope: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

static bool run_shape_liveness(
        ggml_backend_t gpu, ggml_backend_t cpu) {
    configure_cache(nullptr);
    set_env("GGML_CUDA_MOE_CACHE_BUDGET_MB", "16");
    void * session = create_direct_session(gpu, cpu);
    if (!session) {
        fprintf(stderr, "cache-shape-liveness: failed to create session\n");
        return false;
    }

    constexpr int64_t shape_a_in = 256;
    constexpr int64_t shape_b_in = 512;
    constexpr int64_t shape_out = 128;
    constexpr int64_t shape_experts = 64;
    const size_t shape_a_expert =
        ggml_row_size(GGML_TYPE_Q4_0, shape_a_in) * shape_out;
    const size_t shape_b_expert =
        ggml_row_size(GGML_TYPE_Q4_0, shape_b_in) * shape_out;
    std::vector<uint8_t> shape_a(shape_a_expert * shape_experts);
    std::vector<uint8_t> shape_b(shape_b_expert * shape_experts);

    active_api().session_enter(session);
    (void)direct_begin_ready(
            "blk.2.ffn_up_exps.weight", shape_a.data(), shape_a_expert,
            shape_a_in, shape_out, GGML_TYPE_Q4_0, shape_experts);
    active_api().invalidate(shape_a.data(), shape_a.size());
    const bool shape_b_ready = wait_for_direct_pool(
            "blk.3.ffn_up_exps.weight", shape_b.data(), shape_b_expert,
            shape_b_in, shape_out, GGML_TYPE_Q4_0, shape_experts);
    const bool shape_a_ready = wait_for_direct_pool(
            "blk.2.ffn_up_exps.weight", shape_a.data(), shape_a_expert,
            shape_a_in, shape_out, GGML_TYPE_Q4_0, shape_experts);
    active_api().session_leave(session);
    active_api().session_destroy(session);

    const bool ok = shape_b_ready && shape_a_ready;
    printf("cache-shape-liveness: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

static bool run_exact_shape_inventory(
        ggml_backend_t gpu, ggml_backend_t cpu,
        log_capture & capture) {
    configure_cache(nullptr);
    set_env("GGML_CUDA_MOE_CACHE_BUDGET_MB", "16");
    set_env("GGML_CUDA_MOE_CACHE_ADMIT_AFTER", nullptr);
    set_env("GGML_CUDA_MOE_CACHE_THROTTLE", "8");
    set_env("GGML_CUDA_MOE_CACHE_STATS", "0");
    capture.clear();

    void * session = create_direct_session(gpu, cpu);
    if (!session) {
        fprintf(stderr, "cache-shape-inventory: failed to create session\n");
        return false;
    }

    constexpr int64_t direct_n_in = 256;
    constexpr int64_t direct_n_out = 128;
    constexpr int64_t first_n_expert = 16;
    constexpr int64_t second_n_expert = 49;
    const size_t expert_size =
        ggml_row_size(GGML_TYPE_Q4_0, direct_n_in) * direct_n_out;
    std::vector<uint8_t> first(expert_size * first_n_expert);
    std::vector<uint8_t> second(expert_size * second_n_expert);
    std::vector<uint8_t> late(expert_size);

    active_api().session_enter(session);
    (void)direct_begin_ready(
            "blk.7.ffn_up_exps.weight", first.data(), expert_size,
            direct_n_in, direct_n_out, GGML_TYPE_Q4_0, first_n_expert);
    (void)direct_begin_ready(
            "blk.8.ffn_up_exps.weight", second.data(), expert_size,
            direct_n_in, direct_n_out, GGML_TYPE_Q4_0, second_n_expert);
    bool ready = false;
    for (int step = 0; step < 40 && !ready; step++) {
        ready |= direct_begin_ready(
                "blk.7.ffn_up_exps.weight", first.data(), expert_size,
                direct_n_in, direct_n_out, GGML_TYPE_Q4_0, first_n_expert);
        ready |= direct_begin_ready(
                "blk.8.ffn_up_exps.weight", second.data(), expert_size,
                direct_n_in, direct_n_out, GGML_TYPE_Q4_0, second_n_expert);
    }
    const bool complete_first_miss = direct_plan_one(
            "blk.7.ffn_up_exps.weight", first.data(), expert_size,
            direct_n_in, direct_n_out, GGML_TYPE_Q4_0, first_n_expert, 0) == 0;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    const bool complete_second_hit = direct_plan_one(
            "blk.7.ffn_up_exps.weight", first.data(), expert_size,
            direct_n_in, direct_n_out, GGML_TYPE_Q4_0, first_n_expert, 0) == 1;
    const bool late_first_miss = direct_plan_one(
            "blk.9.ffn_up_exps.weight", late.data(), expert_size,
            direct_n_in, direct_n_out, GGML_TYPE_Q4_0, 1, 0) == 0;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    const bool late_second_miss = direct_plan_one(
            "blk.9.ffn_up_exps.weight", late.data(), expert_size,
            direct_n_in, direct_n_out, GGML_TYPE_Q4_0, 1, 0) == 0;
    bool late_hit = false;
    for (int attempt = 0; attempt < 100 && !late_hit; attempt++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        late_hit = direct_plan_one(
                "blk.9.ffn_up_exps.weight", late.data(), expert_size,
                direct_n_in, direct_n_out, GGML_TYPE_Q4_0, 1, 0) == 1;
    }
    active_api().session_leave(session);
    active_api().session_destroy(session);

    const std::string log = capture.get();
    const bool ok = ready &&
        max_field_value(log, "slots=") == first_n_expert + second_n_expert &&
        log.find("entries=65 coverage=complete") != std::string::npos &&
        log.find("admit=1-complete/2-partial/8-replace") != std::string::npos &&
        complete_first_miss && complete_second_hit &&
        late_first_miss && late_second_miss && late_hit &&
        log.find("fills=serial") != std::string::npos;
    if (!ok) {
        fprintf(stderr, "cache-shape-inventory: unexpected pool inventory\n%s",
                log.c_str());
    }
    printf("cache-shape-inventory: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

static bool run_complete_pool_allocation(
        ggml_backend_t gpu, ggml_backend_t cpu) {
    configure_cache(nullptr);
    set_env("GGML_CUDA_MOE_CACHE_BUDGET_MB", "4");
    void * session = create_direct_session(gpu, cpu);
    if (!session) {
        fprintf(stderr, "cache-complete-pools: failed to create session\n");
        return false;
    }

    constexpr int64_t direct_n_in = 512;
    constexpr int64_t common_n_out = 64;
    constexpr int64_t rare_n_out = 128;
    constexpr int64_t direct_n_expert = 64;
    const size_t common_expert =
        ggml_row_size(GGML_TYPE_Q4_0, direct_n_in) * common_n_out;
    const size_t rare_expert =
        ggml_row_size(GGML_TYPE_Q4_0, direct_n_in) * rare_n_out;
    std::vector<std::vector<uint8_t>> common_tensors(
            8, std::vector<uint8_t>(common_expert * direct_n_expert));
    std::vector<uint8_t> rare_tensor(rare_expert * direct_n_expert);

    active_api().session_enter(session);
    for (const auto & tensor : common_tensors) {
        (void)direct_begin_ready(
                "blk.9.ffn_up_exps.weight", tensor.data(), common_expert,
                direct_n_in, common_n_out, GGML_TYPE_Q4_0, direct_n_expert);
    }
    (void)direct_begin_ready(
            "blk.9.ffn_down_exps.weight", rare_tensor.data(), rare_expert,
            direct_n_in, rare_n_out, GGML_TYPE_Q4_0, direct_n_expert);

    const bool common_ready = wait_for_direct_pool(
            "blk.9.ffn_up_exps.weight", common_tensors[0].data(), common_expert,
            direct_n_in, common_n_out, GGML_TYPE_Q4_0, direct_n_expert);
    const bool rare_ready = direct_begin_ready(
            "blk.9.ffn_down_exps.weight", rare_tensor.data(), rare_expert,
            direct_n_in, rare_n_out, GGML_TYPE_Q4_0, direct_n_expert);
    active_api().session_leave(session);
    active_api().session_destroy(session);

    const bool ok = common_ready && rare_ready;
    printf("cache-complete-pools: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

static bool run_shared_budget(
        ggml_backend_t gpu, ggml_backend_t cpu,
        log_capture & capture) {
    configure_cache(nullptr);
    size_t free_bytes = 0;
    size_t total_bytes = 0;
#ifdef GGML_CUDA
    // The cache's budget path queries cudaMemGetInfo, not the backend device
    // memory API. On UMA devices (Grace Blackwell, Jetson/Orin) the backend
    // reports /proc/meminfo while cudaMemGetInfo reports the CUDA view, and
    // the two disagree. Match the cache's query so the reserve leaves the
    // expected remainder on every device class.
    {
        int prev_device = 0;
        cudaError_t err = cudaGetDevice(&prev_device);
        if (err != cudaSuccess) {
            fprintf(stderr, "cache-shared-budget: cudaGetDevice failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        // Restore the device afterwards: run_route_override runs next and
        // exercises routing across two devices, so the current device must
        // not leak out of this test. Fall back to the current device rather
        // than device 0 when the physical device is not reported.
        const long physical = provider_physical_device(gpu->device);
        const int target = physical >= 0 ? (int) physical : prev_device;
        if (target != prev_device) {
            err = cudaSetDevice(target);
            if (err != cudaSuccess) {
                fprintf(stderr, "cache-shared-budget: cudaSetDevice failed: %s\n",
                        cudaGetErrorString(err));
                return false;
            }
        }
        err = cudaMemGetInfo(&free_bytes, &total_bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "cache-shared-budget: cudaMemGetInfo failed: %s\n",
                    cudaGetErrorString(err));
            cudaSetDevice(prev_device);
            return false;
        }
        if (target != prev_device) {
            cudaSetDevice(prev_device);
        }
    }
#else
    ggml_backend_dev_memory(gpu->device, &free_bytes, &total_bytes);
#endif
    const size_t free_mib = free_bytes >> 20;
    if (free_mib < 64) {
        printf("cache-shared-budget: SKIP (insufficient free VRAM)\n");
        return true;
    }

    const std::string reserve = std::to_string(free_mib - 24);
    set_env("GGML_CUDA_MOE_CACHE_BUDGET_MB", nullptr);
    set_env("GGML_CUDA_MOE_CACHE_RESERVE_MB", reserve.c_str());
    capture.clear();

    void * first = create_direct_session(gpu, cpu);
    void * second = create_direct_session(gpu, cpu);
    if (!first || !second) {
        fprintf(stderr, "cache-shared-budget: failed to create sessions\n");
        if (first) {
            active_api().session_destroy(first);
        }
        if (second) {
            active_api().session_destroy(second);
        }
        configure_cache(nullptr);
        return false;
    }

    constexpr int64_t direct_n_in = 256;
    constexpr int64_t direct_n_out = 128;
    constexpr int64_t direct_n_expert = 64;
    const size_t expert_size =
        ggml_row_size(GGML_TYPE_Q4_0, direct_n_in) * direct_n_out;
    std::vector<uint8_t> weights(expert_size * direct_n_expert);

    active_api().session_enter(first);
    const bool first_ready = wait_for_direct_pool(
            "blk.6.ffn_up_exps.weight", weights.data(), expert_size,
            direct_n_in, direct_n_out, GGML_TYPE_Q4_0, direct_n_expert);
    active_api().session_leave(first);

    active_api().session_enter(second);
    const bool second_ready = wait_for_direct_pool(
            "blk.6.ffn_up_exps.weight", weights.data(), expert_size,
            direct_n_in, direct_n_out, GGML_TYPE_Q4_0, direct_n_expert);
    active_api().session_leave(second);

    active_api().session_destroy(second);
    active_api().session_destroy(first);

    const std::string shared_log = capture.get();
    const bool divided = count_field_at_least(shared_log, "granted=", 8) == 2 &&
        max_field_value(shared_log, "granted=") <= 16;

    capture.clear();
    void * replacement = create_direct_session(gpu, cpu);
    bool replacement_ready = false;
    if (replacement) {
        active_api().session_enter(replacement);
        replacement_ready = wait_for_direct_pool(
                "blk.6.ffn_up_exps.weight", weights.data(), expert_size,
                direct_n_in, direct_n_out, GGML_TYPE_Q4_0, direct_n_expert);
        active_api().session_leave(replacement);
        active_api().session_destroy(replacement);
    }
    const std::string replacement_log = capture.get();
    const bool released = max_field_value(replacement_log, "granted=") >= 20;

    capture.clear();
    const std::string high_reserve = std::to_string(free_mib - 8);
    set_env("GGML_CUDA_MOE_CACHE_RESERVE_MB", high_reserve.c_str());
    void * high = create_direct_session(gpu, cpu);
    set_env("GGML_CUDA_MOE_CACHE_RESERVE_MB", reserve.c_str());
    void * low = create_direct_session(gpu, cpu);
    if (high) {
        active_api().session_destroy(high);
    }
    void * after_high = create_direct_session(gpu, cpu);

    bool low_ready = false;
    if (low) {
        active_api().session_enter(low);
        low_ready = wait_for_direct_pool(
                "blk.6.ffn_up_exps.weight", weights.data(), expert_size,
                direct_n_in, direct_n_out, GGML_TYPE_Q4_0, direct_n_expert);
        active_api().session_leave(low);
    }
    bool after_high_ready = false;
    if (after_high) {
        active_api().session_enter(after_high);
        after_high_ready = wait_for_direct_pool(
                "blk.6.ffn_up_exps.weight", weights.data(), expert_size,
                direct_n_in, direct_n_out, GGML_TYPE_Q4_0, direct_n_expert);
        active_api().session_leave(after_high);
    }
    if (after_high) {
        active_api().session_destroy(after_high);
    }
    if (low) {
        active_api().session_destroy(low);
    }
    const std::string reserve_log = capture.get();
    const bool reserve_lowered = high && low && after_high &&
        low_ready && after_high_ready &&
        count_field_at_least(reserve_log, "granted=", 8) == 2;
    configure_cache(nullptr);

    const bool ok = first_ready && second_ready && divided &&
        replacement_ready && released && reserve_lowered;
    if (!ok) {
        fprintf(stderr, "cache-shared-budget: unexpected claim behavior\n%s%s%s",
                shared_log.c_str(), replacement_log.c_str(), reserve_log.c_str());
    }
    printf("cache-shared-budget: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

static bool run_route_override(
        ggml_backend_dev_t first_device,
        ggml_backend_t gpu, ggml_backend_t cpu,
        log_capture & capture) {
    ggml_backend_dev_t second_device =
        find_other_provider_device(first_device);
    if (!second_device) {
        printf("cache-route-override: SKIP (one CUDA device)\n");
        return true;
    }
    const long first_physical = provider_physical_device(first_device);
    if (first_physical >= 0 &&
        first_physical == provider_physical_device(second_device)) {
        printf("cache-route-override: SKIP (one physical CUDA device)\n");
        return true;
    }
    ggml_backend_t second_gpu =
        ggml_backend_dev_init(second_device, nullptr);
    if (!second_gpu) {
        fprintf(stderr,
                "cache-route-override: failed to initialize second device\n");
        return false;
    }

    configure_cache(nullptr);
    set_env("GGML_CUDA_MOE_CACHE_BUDGET_MB", "8");
    set_env("GGML_CUDA_MOE_CACHE_NDEV", "2");
    capture.clear();
    void * backends[] = { gpu, second_gpu, cpu };
    void * session = active_api().session_create(backends, 3, nullptr);
    if (!session) {
        fprintf(stderr,
                "cache-route-override: failed to create session\n");
        ggml_backend_free(second_gpu);
        return false;
    }

    constexpr int64_t shape_a_in = 1024;
    constexpr int64_t shape_a_out = 160;
    constexpr int64_t shape_b_in = 512;
    constexpr int64_t shape_b_out = 384;
    constexpr int64_t shape_experts = 64;
    const size_t shape_a_expert =
        ggml_row_size(GGML_TYPE_Q4_0, shape_a_in) * shape_a_out;
    const size_t shape_b_expert =
        ggml_row_size(GGML_TYPE_Q4_0, shape_b_in) * shape_b_out;
    std::vector<uint8_t> shape_a(shape_a_expert * shape_experts);
    std::vector<uint8_t> shape_b(shape_b_expert * shape_experts);

    active_api().session_enter(session);
    const bool shape_a_ready = wait_for_direct_pool(
            "blk.4.ffn_up_exps.weight", shape_a.data(), shape_a_expert,
            shape_a_in, shape_a_out, GGML_TYPE_Q4_0, shape_experts);
    const bool shape_b_ready = wait_for_direct_pool(
            "blk.4.ffn_down_exps.weight", shape_b.data(), shape_b_expert,
            shape_b_in, shape_b_out, GGML_TYPE_Q4_0, shape_experts);
    active_api().session_leave(session);

    ggml_moe_cache_config config = {};
    ggml_moe_cache_device_caps first_caps = {};
    ggml_moe_cache_device_caps second_caps = {};
    const bool queried = active_api().query_config(0, 8, &config) &&
        active_api().query_device(first_device, &config, &first_caps) &&
        active_api().query_device(second_device, &config, &second_caps);
    const bool expect_parallel = queried && first_caps.compute_capability >= 800 &&
        second_caps.compute_capability >= 800;
    active_api().session_destroy(session);
    const bool fill_policy = capture.get().find(
            expect_parallel ? "fills=parallel" : "fills=serial") != std::string::npos;

    ggml_moe_cache_config automatic = {};
    capture.clear();
    const bool automatic_queried =
        active_api().query_config(1, 4, &automatic);
    void * dormant = automatic_queried
        ? active_api().session_create(backends, 3, &automatic) : nullptr;
    const bool dormant_created = dormant != nullptr;
    bool dormant_begin = false;
    if (dormant) {
        active_api().session_enter(dormant);
        dormant_begin = direct_begin_ready(
                "blk.4.ffn_up_exps.weight", shape_a.data(), shape_a_expert,
                shape_a_in, shape_a_out, GGML_TYPE_Q4_0, shape_experts);
        active_api().session_leave(dormant);
        active_api().session_destroy(dormant);
    }
    const bool slab_floor = dormant_created && !dormant_begin &&
        capture.get().find("automatic slab floor") != std::string::npos;
    ggml_backend_free(second_gpu);

    const bool ok = shape_a_ready && shape_b_ready && queried && fill_policy &&
        automatic_queried && slab_floor;
    printf("cache-route-override: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

static bool run_single_gpu_auto(
        ggml_backend_dev_t provider_device, ggml_backend_t gpu, ggml_backend_t cpu,
        log_capture & capture) {
    const ggml_moe_cache_api api =
        ggml_moe_cache_get(ggml_backend_dev_backend_reg(provider_device));
    if (!api.query_config || !api.session_create || !api.session_destroy) {
        printf("cache-single-gpu-auto: FAIL (incomplete cache API)\n");
        return false;
    }

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    ggml_backend_dev_memory(provider_device, &free_bytes, &total_bytes);
    if (free_bytes < 1024u * 1024 * 1024 + 64u * 1024 * 1024) {
        printf("cache-single-gpu-auto: SKIP (device free memory below the auto slab floor)\n");
        return true;
    }

    configure_cache(nullptr);
    set_env("GGML_CUDA_MOE_CACHE_MODE", "auto");
    set_env("GGML_CUDA_MOE_CACHE_BUDGET_MB", "2048");
    set_env("GGML_CUDA_MOE_CACHE_RESERVE_MB", "0");
    capture.clear();

    ggml_moe_cache_config config = {};
    if (!api.query_config(1, 2048, &config)) {
        printf("cache-single-gpu-auto: FAIL (query_config)\n");
        return false;
    }
    bool ok = config.min_devices == 1 &&
        config.minimum_slab_bytes == 1024u * 1024 * 1024;

    void * backends[] = { gpu, cpu };
    void * session = api.session_create(backends, 2, &config);
    ok &= session != nullptr;
    if (session) {
        constexpr int64_t shape_in = 1024;
        constexpr int64_t shape_out = 160;
        constexpr int64_t shape_experts = 64;
        const size_t expert_size =
            ggml_row_size(GGML_TYPE_Q4_0, shape_in) * shape_out;
        std::vector<uint8_t> data(expert_size * shape_experts);
        active_api().session_enter(session);
        const bool ready = wait_for_direct_pool(
                "blk.4.ffn_up_exps.weight", data.data(), expert_size,
                shape_in, shape_out, GGML_TYPE_Q4_0, shape_experts);
        const bool no_dormant_log =
            capture.get().find("automatic slab floor") == std::string::npos;
        active_api().session_leave(session);
        api.session_destroy(session);
        ok &= ready && no_dormant_log;
        if (!ready) {
            fprintf(stderr, "cache-single-gpu-auto: auto session did not engage\n%s\n",
                    capture.get().c_str());
        }
    }
    configure_cache(nullptr);
    printf("cache-single-gpu-auto: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

static bool run_admission_policy_once(
        ggml_backend_t gpu, ggml_backend_t cpu,
        int new_expert_misses, log_capture & capture) {
    configure_cache(nullptr);
    set_env("GGML_CUDA_MOE_CACHE_BUDGET_MB", "8");
    set_env("GGML_CUDA_MOE_CACHE_ADMIT_AFTER", "2");
    set_env("GGML_CUDA_MOE_CACHE_THROTTLE", "8");
    set_env("GGML_CUDA_MOE_CACHE_STATS", "0");
    capture.clear();

    void * session = create_direct_session(gpu, cpu);
    if (!session) {
        return false;
    }

    constexpr int64_t policy_n_in = 1024;
    constexpr int64_t policy_n_out = 206;
    constexpr int64_t policy_n_expert = 65;
    const size_t expert_size =
        ggml_row_size(GGML_TYPE_Q4_0, policy_n_in) * policy_n_out;
    std::vector<uint8_t> weights(expert_size * policy_n_expert);
    const char * name = "blk.5.ffn_up_exps.weight";

    active_api().session_enter(session);
    bool ok = wait_for_direct_pool(
            name, weights.data(), expert_size,
            policy_n_in, policy_n_out,
            GGML_TYPE_Q4_0, policy_n_expert);
    for (int32_t expert = 0; expert < 64 && ok; expert++) {
        if (direct_plan_one(
                name, weights.data(), expert_size,
                policy_n_in, policy_n_out,
                GGML_TYPE_Q4_0, policy_n_expert, expert) != 0) {
            ok = false;
            break;
        }
        if (expert == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        if (direct_plan_one(
                name, weights.data(), expert_size,
                policy_n_in, policy_n_out,
                GGML_TYPE_Q4_0, policy_n_expert, expert) != 0) {
            ok = false;
            break;
        }

        bool hit = false;
        for (int attempt = 0; attempt < 100; attempt++) {
            if (direct_plan_one(
                    name, weights.data(), expert_size,
                    policy_n_in, policy_n_out,
                    GGML_TYPE_Q4_0, policy_n_expert, expert) == 1) {
                hit = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        ok &= hit;
    }

    for (int miss = 0; miss < new_expert_misses && ok; miss++) {
        ok &= direct_plan_one(
                name, weights.data(), expert_size,
                policy_n_in, policy_n_out,
                GGML_TYPE_Q4_0, policy_n_expert, 64) == 0;
    }
    if (new_expert_misses == 8 && ok) {
        bool hit = false;
        for (int attempt = 0; attempt < 100; attempt++) {
            if (direct_plan_one(
                    name, weights.data(), expert_size,
                    policy_n_in, policy_n_out,
                    GGML_TYPE_Q4_0, policy_n_expert, 64) == 1) {
                hit = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        ok &= hit;
    }

    active_api().session_leave(session);
    active_api().session_destroy(session);
    const std::string log = capture.get();
    const long long expected_enqueued =
        new_expert_misses == 8 ? 65 : 64;
    const long long expected_evictions =
        new_expert_misses == 8 ? 1 : 0;
    return ok &&
        max_field_value(log, "slots=") == 64 &&
        max_field_value(log, "enqueued=") == expected_enqueued &&
        max_field_value(log, "evictions=") == expected_evictions;
}

static bool run_admission_policy(
        ggml_backend_t gpu, ggml_backend_t cpu,
        log_capture & capture) {
    const bool seven = run_admission_policy_once(
            gpu, cpu, 7, capture);
    const bool eight = run_admission_policy_once(
            gpu, cpu, 8, capture);
    const bool ok = seven && eight;
    printf("cache-admission-policy: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

// Run a test only when the provider supports the behavior it checks; on
// other providers print a skip line so CI shows the coverage gap explicitly.
static bool run_optional(
        bool enabled, const char * name, std::function<bool()> test) {
    if (!enabled) {
        printf("%s: SKIP (provider-specific)\n", name);
        return true;
    }
    return test();
}

} // namespace

int main() {
    log_capture capture;
    ggml_log_set(log_callback, &capture);

    ggml_backend_dev_t provider_device = find_provider_device();
    if (!provider_device) {
        printf("SKIP: no moe-cache provider device available\n");
        return 77; // ctest interprets 77 as SKIP_RETURN_CODE
    }
    ggml_backend_reg_t provider_reg =
        ggml_backend_dev_backend_reg(provider_device);
    g_test_api = ggml_moe_cache_get(provider_reg);
    const bool provider_is_cuda =
        strcmp(ggml_backend_reg_name(provider_reg), "CUDA") == 0;
    printf("test-moe-cache: provider=%s\n",
            ggml_backend_reg_name(provider_reg));

    ggml_backend_t gpu = ggml_backend_dev_init(provider_device, nullptr);
    ggml_backend_t cpu = init_cpu_backend();
    if (!gpu || !cpu) {
        fprintf(stderr, "failed to initialize provider and CPU backends\n");
        if (gpu) {
            ggml_backend_free(gpu);
        }
        if (cpu) {
            ggml_backend_free(cpu);
        }
        return 1;
    }
    ggml_init_params static_params = {
        16 * ggml_tensor_overhead(),
        nullptr,
        true,
    };
    ggml_context * static_ctx = ggml_init(static_params);
    if (!static_ctx) {
        fprintf(stderr, "failed to create tensor context\n");
        ggml_backend_free(gpu);
        ggml_backend_free(cpu);
        return 1;
    }

    ggml_tensor * weights = ggml_new_tensor_3d(
            static_ctx, GGML_TYPE_Q4_0, n_in, n_out, n_expert);
    ggml_tensor * gate_weights = ggml_new_tensor_3d(
            static_ctx, GGML_TYPE_Q4_0, n_in, n_out, n_expert);
    ggml_tensor * ids = ggml_new_tensor_2d(
            static_ctx, GGML_TYPE_I32, n_used, n_tokens);
    ggml_tensor * activations = ggml_new_tensor_3d(
            static_ctx, GGML_TYPE_F32, n_in, 1, n_tokens);
    ggml_set_name(weights, "blk.0.ffn_up_exps.weight");
    ggml_set_name(gate_weights, "blk.0.ffn_gate_exps.weight");
    ggml_set_name(ids, "moe_cache_test_ids");
    ggml_set_name(activations, "moe_cache_test_activations");

    ggml_backend_buffer_t static_buffer =
        ggml_backend_alloc_ctx_tensors(static_ctx, cpu);
    if (!static_buffer) {
        fprintf(stderr, "failed to allocate CPU tensors\n");
        ggml_free(static_ctx);
        ggml_backend_free(gpu);
        ggml_backend_free(cpu);
        return 1;
    }
    ggml_backend_buffer_set_usage(
            static_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    std::vector<float> weights_f32(ggml_nelements(weights));
    for (size_t index = 0; index < weights_f32.size(); index++) {
        weights_f32[index] =
            0.15f * std::sin((float) (index % 997) * 0.017f) +
            0.05f * std::cos((float) (index % 431) * 0.031f);
    }
    std::vector<uint8_t> weights_q4(ggml_nbytes(weights));
    const size_t quantized = ggml_quantize_chunk(
            GGML_TYPE_Q4_0, weights_f32.data(), weights_q4.data(),
            0, n_out * n_expert, n_in, nullptr);
    if (quantized != weights_q4.size()) {
        fprintf(stderr, "unexpected quantized size: %zu != %zu\n",
                quantized, weights_q4.size());
        ggml_backend_buffer_free(static_buffer);
        ggml_free(static_ctx);
        ggml_backend_free(gpu);
        ggml_backend_free(cpu);
        return 1;
    }
    ggml_backend_tensor_set(
            weights, weights_q4.data(), 0, weights_q4.size());

    std::vector<float> gate_weights_f32(ggml_nelements(gate_weights));
    for (size_t index = 0; index < gate_weights_f32.size(); index++) {
        gate_weights_f32[index] =
            0.11f * std::sin((float) (index % 881) * 0.021f) -
            0.08f * std::cos((float) (index % 389) * 0.027f);
    }
    std::vector<uint8_t> gate_weights_q4(ggml_nbytes(gate_weights));
    const size_t gate_quantized = ggml_quantize_chunk(
            GGML_TYPE_Q4_0, gate_weights_f32.data(), gate_weights_q4.data(),
            0, n_out * n_expert, n_in, nullptr);
    if (gate_quantized != gate_weights_q4.size()) {
        fprintf(stderr, "unexpected gate quantized size: %zu != %zu\n",
                gate_quantized, gate_weights_q4.size());
        ggml_backend_buffer_free(static_buffer);
        ggml_free(static_ctx);
        ggml_backend_free(gpu);
        ggml_backend_free(cpu);
        return 1;
    }
    ggml_backend_tensor_set(
            gate_weights, gate_weights_q4.data(), 0,
            gate_weights_q4.size());

    const int32_t ids_data[n_used] = { 0, 1 };
    ggml_backend_tensor_set(ids, ids_data, 0, sizeof(ids_data));
    std::vector<float> activation_data(ggml_nelements(activations));
    for (size_t index = 0; index < activation_data.size(); index++) {
        activation_data[index] =
            0.4f * std::sin((float) index * 0.07f) -
            0.2f * std::cos((float) index * 0.11f);
    }
    ggml_backend_tensor_set(
            activations, activation_data.data(), 0,
            activation_data.size() * sizeof(float));

    test_graph graph = make_graph(cpu, weights, activations, ids);
    if (!graph.ctx || !graph.buffer) {
        fprintf(stderr, "failed to create test graph\n");
        free_graph(graph);
        ggml_backend_buffer_free(static_buffer);
        ggml_free(static_ctx);
        ggml_backend_free(gpu);
        ggml_backend_free(cpu);
        return 1;
    }

    set_env("GGML_CUDA_MOE_CACHE", "0");
    if (ggml_backend_graph_compute(cpu, graph.graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "CPU reference compute failed\n");
        free_graph(graph);
        ggml_backend_buffer_free(static_buffer);
        ggml_free(static_ctx);
        ggml_backend_free(gpu);
        ggml_backend_free(cpu);
        return 1;
    }
    std::vector<float> reference(ggml_nelements(graph.out));
    ggml_backend_tensor_get(
            graph.out, reference.data(), 0, reference.size() * sizeof(float));

    test_graph fused_graph = make_fused_graph(
            cpu, weights, gate_weights, activations, ids);
    if (!fused_graph.ctx || !fused_graph.buffer) {
        fprintf(stderr, "failed to create fused test graph\n");
        free_graph(fused_graph);
        free_graph(graph);
        ggml_backend_buffer_free(static_buffer);
        ggml_free(static_ctx);
        ggml_backend_free(gpu);
        ggml_backend_free(cpu);
        return 1;
    }
    set_env("GGML_CUDA_MOE_CACHE", "0");
    if (ggml_backend_graph_compute(cpu, fused_graph.graph) !=
            GGML_STATUS_SUCCESS) {
        fprintf(stderr, "fused CPU reference compute failed\n");
        free_graph(fused_graph);
        free_graph(graph);
        ggml_backend_buffer_free(static_buffer);
        ggml_free(static_ctx);
        ggml_backend_free(gpu);
        ggml_backend_free(cpu);
        return 1;
    }
    std::vector<float> fused_reference(ggml_nelements(fused_graph.out));
    ggml_backend_tensor_get(
            fused_graph.out, fused_reference.data(), 0,
            fused_reference.size() * sizeof(float));

    test_graph clamped_fused_graph = make_clamped_fused_graph(
            cpu, weights, gate_weights, activations, ids);
    if (!clamped_fused_graph.ctx || !clamped_fused_graph.buffer) {
        fprintf(stderr, "failed to create clamped fused test graph\n");
        free_graph(clamped_fused_graph);
        free_graph(fused_graph);
        free_graph(graph);
        ggml_backend_buffer_free(static_buffer);
        ggml_free(static_ctx);
        ggml_backend_free(gpu);
        ggml_backend_free(cpu);
        return 1;
    }
    set_env("GGML_CUDA_MOE_CACHE", "0");
    if (ggml_backend_graph_compute(cpu, clamped_fused_graph.graph) !=
            GGML_STATUS_SUCCESS) {
        fprintf(stderr, "clamped fused CPU reference compute failed\n");
        free_graph(clamped_fused_graph);
        free_graph(fused_graph);
        free_graph(graph);
        ggml_backend_buffer_free(static_buffer);
        ggml_free(static_ctx);
        ggml_backend_free(gpu);
        ggml_backend_free(cpu);
        return 1;
    }
    std::vector<float> clamped_fused_reference(
            ggml_nelements(clamped_fused_graph.out));
    ggml_backend_tensor_get(
            clamped_fused_graph.out, clamped_fused_reference.data(), 0,
            clamped_fused_reference.size() * sizeof(float));

    bool ok = run_capability_queries(provider_device, cpu);
    ok &= run_invalidation_hook_coverage(cpu);
    ok &= run_scenario("cache-hit", nullptr, gpu, cpu, graph, reference, capture);
    ok &= run_scenario(
            "cache-generic-mmv", nullptr, gpu, cpu, graph, reference,
            capture, "1", nullptr, "0");
    ok &= run_scenario("dispatch-fallback", "dispatch", gpu, cpu, graph, reference, capture);
    ok &= run_scenario("collect-fallback", "collect", gpu, cpu, graph, reference, capture);
    ok &= run_scenario("insert-fallback", "insert", gpu, cpu, graph, reference, capture);
    ok &= run_scenario("slab-fallback", "slab", gpu, cpu, graph, reference, capture);
    ok &= run_optional(provider_is_cuda, "cache-fused-swiglu", [&] {
        return run_scenario(
                "cache-fused-swiglu", nullptr, gpu, cpu,
                fused_graph, fused_reference, capture,
                "1", "fusion-nodes=");
    });
    ok &= run_optional(provider_is_cuda, "cache-fused-dispatch-fallback", [&] {
        return run_scenario(
                "cache-fused-dispatch-fallback", "dispatch", gpu, cpu,
                fused_graph, fused_reference, capture,
                "1", "fusion-attempts=");
    });
    ok &= run_optional(provider_is_cuda, "cache-fused-collect-fallback", [&] {
        return run_scenario(
                "cache-fused-collect-fallback", "collect", gpu, cpu,
                fused_graph, fused_reference, capture,
                "1", "fusion-nodes=");
    });
    ok &= run_optional(provider_is_cuda, "cache-fused-clamped-swiglu", [&] {
        return run_scenario(
                "cache-fused-clamped-swiglu", nullptr, gpu, cpu,
                clamped_fused_graph, clamped_fused_reference, capture,
                "1", "fusion-nodes=");
    });
    ok &= run_optional(provider_is_cuda, "cache-fused-clamped-collect-fallback", [&] {
        return run_scenario(
                "cache-fused-clamped-collect-fallback", "collect", gpu, cpu,
                clamped_fused_graph, clamped_fused_reference, capture,
                "1", "fusion-nodes=");
    });
    ok &= run_optional(provider_is_cuda, "cache-fused-partial-duplicate", [&] {
        return run_fused_partial_invalidation(
                gpu, cpu, weights, gate_weights, activations,
                weights_q4.data(), gate_weights_q4.data(),
                fused_reference, capture);
    });
    ok &= run_optional(provider_is_cuda, "cache-fused-concurrent", [&] {
        return run_fused_concurrent_sessions(
                provider_device, gpu, cpu, weights, gate_weights,
                activations, ids, fused_graph, fused_reference, capture);
    });
    ok &= run_optional(provider_is_cuda, "cache-fused-lifecycle", [&] {
        return run_fused_repeated_lifecycle(
                gpu, cpu, fused_graph, fused_reference, capture);
    });
    ok &= run_multi_token_scenario(
            gpu, cpu, weights, gate_weights, capture,
            provider_is_cuda, provider_is_cuda);
    ok &= run_mxfp4_shared_pool(gpu, cpu, capture);
    const size_t expert_size = ggml_nbytes(weights) / n_expert;
    ok &= run_precensus_invalidation(
            gpu, cpu, graph, weights,
            weights_q4.data() + (n_expert - 1) * expert_size,
            expert_size, reference, capture);

    const int32_t repeated_ids[n_used] = { 0, 0 };
    ggml_backend_tensor_set(ids, repeated_ids, 0, sizeof(repeated_ids));
    set_env("GGML_CUDA_MOE_CACHE", "0");
    if (ggml_backend_graph_compute(cpu, graph.graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "cache-invalidate: initial CPU reference compute failed\n");
        ok = false;
    } else {
        std::vector<float> old_reference(ggml_nelements(graph.out));
        ggml_backend_tensor_get(
                graph.out, old_reference.data(), 0,
                old_reference.size() * sizeof(float));

        std::vector<float> replacement_f32(n_in);
        for (size_t index = 0; index < replacement_f32.size(); index++) {
            replacement_f32[index] =
                1.25f + 0.3f * std::sin((float) index * 0.041f);
        }
        std::vector<uint8_t> replacement_q4(
                ggml_row_size(GGML_TYPE_Q4_0, n_in));
        const size_t replacement_size = ggml_quantize_chunk(
                GGML_TYPE_Q4_0, replacement_f32.data(), replacement_q4.data(),
                0, 1, n_in, nullptr);
        if (replacement_size != replacement_q4.size()) {
            fprintf(stderr, "cache-invalidate: unexpected replacement size\n");
            ok = false;
        } else {
            ok &= run_invalidation_scenario(
                    gpu, cpu, graph, weights, replacement_q4,
                    old_reference, capture);
        }
    }

    stress_fixture stress;
    if (!init_stress_fixture(stress, cpu)) {
        fprintf(stderr, "failed to initialize cache stress fixture\n");
        ok = false;
    } else {
        ok &= run_concurrent_sessions(
                provider_device, gpu, cpu, stress, capture);
        ok &= run_repeated_lifecycle(gpu, cpu, stress, capture);
        ok &= run_fill_invalidation(gpu, cpu, stress, capture);
    }
    free_stress_fixture(stress);
    ok &= run_optional(provider_is_cuda, "cache-scope", [&] {
        return run_scope_isolation(gpu, cpu, weights);
    });
    ok &= run_explicit_session_config(gpu, cpu);
    ok &= run_optional(provider_is_cuda, "cache-policy-diagnostics", [&] {
        return run_policy_diagnostics(gpu, cpu, weights, capture);
    });
    ok &= run_optional(provider_is_cuda, "cache-cpu-overlap", [&] {
        return run_cpu_overlap_policy(gpu, cpu, weights, capture);
    });
    ok &= run_optional(provider_is_cuda, "cache-cpu-overlap-auto", [&] {
        return run_adaptive_cpu_overlap_policy(gpu, cpu, weights, capture);
    });
    ok &= run_optional(provider_is_cuda, "cache-cpu-overlap-calibration", [&] {
        return run_adaptive_cpu_overlap_calibration(gpu, cpu, weights, capture);
    });
    ok &= run_shape_liveness(gpu, cpu);
    ok &= run_optional(provider_is_cuda, "cache-shape-inventory", [&] {
        return run_exact_shape_inventory(gpu, cpu, capture);
    });
    ok &= run_complete_pool_allocation(gpu, cpu);
    ok &= run_optional(provider_is_cuda, "cache-shared-budget", [&] {
        return run_shared_budget(gpu, cpu, capture);
    });
    ok &= run_optional(provider_is_cuda, "cache-route-override", [&] {
        return run_route_override(provider_device, gpu, cpu, capture);
    });
    ok &= run_single_gpu_auto(provider_device, gpu, cpu, capture);
    ok &= run_optional(provider_is_cuda, "cache-admission-policy", [&] {
        return run_admission_policy(gpu, cpu, capture);
    });

    free_graph(clamped_fused_graph);
    free_graph(fused_graph);
    free_graph(graph);
    ggml_backend_free(gpu);
#ifdef GGML_BACKEND_DL
    ggml_backend_unload(provider_reg);
    ggml_backend_buffer_free(static_buffer);
    static_buffer = nullptr;
    printf("cache-backend-unload: OK\n");

    ggml_backend_dev_t reloaded_device = find_provider_device();
    ggml_backend_t reloaded_gpu = reloaded_device
        ? ggml_backend_dev_init(reloaded_device, nullptr) : nullptr;
    ggml_backend_reg_t reloaded_reg = reloaded_gpu
        ? ggml_backend_dev_backend_reg(reloaded_device) : nullptr;
    configure_cache(nullptr);
    void * reloaded_session = nullptr;
    if (reloaded_gpu) {
        // The registry re-registers on load; use the fresh api so the
        // destroy callback comes from the reloaded library, not the
        // pre-unload snapshot in g_test_api.
        const ggml_moe_cache_api reloaded_api =
            ggml_moe_cache_get(reloaded_reg);
        void * backends[] = { reloaded_gpu, cpu };
        reloaded_session = reloaded_api.session_create
            ? reloaded_api.session_create(backends, 2, nullptr) : nullptr;
        if (reloaded_session) {
            reloaded_api.session_destroy(reloaded_session);
        }
    }
    const bool reload_ok = reloaded_session != nullptr;
    if (reloaded_gpu) {
        ggml_backend_free(reloaded_gpu);
        ggml_backend_unload(reloaded_reg);
    }
    printf("cache-backend-reload: %s\n", reload_ok ? "OK" : "FAIL");
    ok &= reload_ok;
#else
    (void) provider_reg;
    printf("cache-backend-unload: SKIP (static backend)\n");
#endif
    if (static_buffer) {
        ggml_backend_buffer_free(static_buffer);
    }
    ggml_free(static_ctx);
    ggml_quantize_free();
    ggml_backend_free(cpu);
    ggml_log_set(nullptr, nullptr);
    return ok ? 0 : 1;
}
