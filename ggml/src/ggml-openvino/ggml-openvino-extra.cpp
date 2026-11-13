#include "ggml-openvino-extra.h"

#include "device/npu_config.h"
#include "device/npu_quant_policy.h"
#include "ggml-impl.h"
#include "ggml.h"
#include "model-cache.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#include <openvino/runtime/intel_npu/level_zero/level_zero.hpp>
#include <openvino/runtime/properties.hpp>
#include <optional>
#include <sstream>

#if defined(__APPLE__)
#    include <crt_externs.h>
#elif !defined(_WIN32)
extern char ** environ;
#endif

ov::Core & ov_singleton_core() {
    static ov::Core core;
    return core;
}

static bool has_prefix(const std::string & s, const std::string & prefix) {
    return s.size() >= prefix.size() && std::equal(prefix.begin(), prefix.end(), s.begin());
}

static bool is_virtual_routing_device(const std::string & device_name) {
    return has_prefix(device_name, "AUTO") || has_prefix(device_name, "MULTI") || has_prefix(device_name, "HETERO");
}

static std::vector<std::string> ov_enumerate_devices() {
    std::vector<std::string> result;

    for (const auto & device : ov_singleton_core().get_available_devices()) {
        if (!is_virtual_routing_device(device)) {
            result.push_back(device);
        }
    }

    if (result.empty()) {
        result.push_back("CPU");
    }

    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

std::string ggml_openvino_get_device_description(const std::string & device_name) {
    std::string description = device_name;
    try {
        description = ov_singleton_core().get_property(device_name, ov::device::full_name);
    } catch (...) {
        return device_name;
    }

    if (has_prefix(device_name, "NPU")) {
        try {
            const std::string arch =
                ov_singleton_core().get_property(device_name, "DEVICE_ARCHITECTURE").as<std::string>();
            if (!arch.empty()) {
                description += " (NPU " + arch + ")";
            }
        } catch (...) {
        }
    }

    return description;
}

// requested: GGML_OPENVINO_DEVICE, nullptr if unset. available_devices is never empty (see ov_enumerate_devices)
static std::string resolve_openvino_device_name(const std::vector<std::string> & available_devices,
                                                const char * requested) {
    auto available = [&](const std::string & name) {
        return std::find(available_devices.begin(), available_devices.end(), name) != available_devices.end();
    };
    if (requested == nullptr) {
        return available("CPU") ? "CPU" : available_devices.front();
    }
    if (!available(requested)) {
        // No fallback to CPU (easy to miss) and no GPU -> GPU.0 alias (with iGPU + dGPU, GPU.0 is often the
        // wrong one). List the devices here: --list-devices initializes this backend and would abort too.
        std::string list;
        for (const std::string & name : available_devices) {
            list += "\n  " + name + ": " + ggml_openvino_get_device_description(name);
        }
        GGML_ABORT(
            "GGML OpenVINO Backend: GGML_OPENVINO_DEVICE=%s is not available. "
            "Set it to one of the available OpenVINO devices:%s",
            requested, list.c_str());
    }
    return requested;
}

namespace {

enum class env_type {
    STR,    // any text: a device, a path, plugin properties
    ENUM,   // one of env_var::values
    INT,    // a whole number
    FLOAT,  // a number
    BOOL,   // a whole number: positive is on, zero or negative is off
};

struct env_var {
    const char * name;
    env_type type;
    const char * values = nullptr;  // ENUM only: the accepted values, separated by '|'
};

// All recognized GGML_OPENVINO_* env vars. Their values are checked and cached once at backend init time and read
// back via ggml_openvino_getenv_str() (raw string) or ggml_openvino_getenv_int() (integer / boolean toggle).
// A value that does not fit the type stops the backend; another GGML_OPENVINO_* name in the environment is reported.
constexpr env_var env_vars[] = {
    // Text
    {"GGML_OPENVINO_DEVICE", env_type::STR},
    {"GGML_OPENVINO_CACHE_DIR", env_type::STR},
    {"GGML_OPENVINO_SPILL_DIR", env_type::STR},
    {"GGML_OPENVINO_DEBUG_NODE", env_type::STR},
    {"GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR", env_type::STR},
    {"GGML_OPENVINO_NPU_COMPILE_CONFIG", env_type::STR},
    {"GGML_OPENVINO_NPU_COMPILER_TYPE", env_type::STR},
    {"GGML_OPENVINO_NPUW_FUNCALL_FOR_ALL", env_type::STR},
    {"GGML_OPENVINO_NPUW_UNFOLD_IREQS", env_type::STR},
    {"GGML_OPENVINO_COMPILATION_NUM_THREADS", env_type::STR},
    {"GGML_OPENVINO_NPU_CONFIG", env_type::STR},
    {"GGML_OPENVINO_GPU_CONFIG", env_type::STR},
    {"GGML_OPENVINO_CPU_CONFIG", env_type::STR},
    // NPUW dump options require an OpenVINO developer build (NPU_PLUGIN_DEVELOPER_BUILD).
    // DUMP_SUBS accepts YES, MIN (unique function bodies), last, or indices such as 0,2.
    {"GGML_OPENVINO_NPUW_DUMP_SUBS", env_type::STR},
    {"GGML_OPENVINO_NPUW_DUMP_SUBS_DIR", env_type::STR},
    {"GGML_OPENVINO_NPUW_DUMP_SUBS_ON_FAIL", env_type::STR},
    {"GGML_OPENVINO_NPUW_DUMP_IO", env_type::STR},
    // One of a fixed set of values
    {"GGML_OPENVINO_NPU_REQUANT_POLICY", env_type::ENUM, "group-128|channel-wise"},
    {"GGML_OPENVINO_NPU_TOKEN_EMBD_QUANT", env_type::ENUM, "f16|q8_0_c|q4_0_128|q4_0_c"},
    {"GGML_OPENVINO_REQUANT_KQUANT", env_type::ENUM, "q4_sym128|q4_sym128_all|q4_asym64|q4_asym64_all|native"},
    // Numbers
    {"GGML_OPENVINO_PREFILL_CHUNK_SIZE", env_type::INT},
    {"GGML_OPENVINO_LAZY_HOST_THRESHOLD_MB", env_type::INT},
    {"GGML_OPENVINO_PROFILING", env_type::INT},
    {"GGML_OPENVINO_GDN_DECOMPOSE_CHUNK", env_type::INT},
    {"GGML_OPENVINO_GDN_Q_SCALE", env_type::FLOAT},
    // Switches
    {"GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY", env_type::BOOL},
    {"GGML_OPENVINO_NPU_BLOB_PREBUILD", env_type::BOOL},
    {"GGML_OPENVINO_NPU_KV_ON_DEMAND", env_type::BOOL},
    {"GGML_OPENVINO_STATEFUL_EXECUTION", env_type::BOOL},
    {"GGML_OPENVINO_DUMP_CGRAPH", env_type::BOOL},
    {"GGML_OPENVINO_DUMP_IR", env_type::BOOL},
    {"GGML_OPENVINO_DEBUG_INPUT", env_type::BOOL},
    {"GGML_OPENVINO_DEBUG_OUTPUT", env_type::BOOL},
    // Force the static (NPU-shape) compute path on any device, e.g. GGML_OPENVINO_DEVICE=CPU,
    // to test the static-shape translation without NPUW/real NPU hardware in the loop.
    {"GGML_OPENVINO_FORCE_STATIC", env_type::BOOL},
    {"GGML_OPENVINO_PRINT_CGRAPH_TENSOR_ADDRESS", env_type::BOOL},
    {"GGML_OPENVINO_DISABLE_CACHE", env_type::BOOL},
    {"GGML_OPENVINO_DISABLE_FUSED_RECURRENCE", env_type::BOOL},
    {"GGML_OPENVINO_ENABLE_KV_4D", env_type::BOOL},
    {"GGML_OPENVINO_ENABLE_FALLBACK", env_type::BOOL},
    {"GGML_OPENVINO_KV_SCATTER_ELEMENTS", env_type::BOOL},
    {"GGML_OPENVINO_MANUAL_GQA_ATTN", env_type::BOOL},
    {"GGML_OPENVINO_DISABLE_ELTWISE_RANK_ALIGN", env_type::BOOL},
    {"GGML_OPENVINO_MOE_OP", env_type::BOOL},
    {"GGML_OPENVINO_EXACT_ZERO_POINT", env_type::BOOL},
    {"GGML_OPENVINO_MEMORY_OPTIMIZE", env_type::BOOL},
    {"GGML_OPENVINO_RELEASE_WEIGHTS", env_type::BOOL},
    {"GGML_OPENVINO_REDUCE_COMPILE_MEM", env_type::BOOL},
    {"GGML_OPENVINO_LOG_UNSUPPORTED_OPS", env_type::BOOL},
    {"GGML_OPENVINO_LOG_SWA_LAYERS", env_type::BOOL},
    {"GGML_OPENVINO_NATIVE_SOFTPLUS", env_type::BOOL},
    {"GGML_OPENVINO_DISABLE_REMOTE_OUTPUTS", env_type::BOOL},
    {"GGML_OPENVINO_DISABLE_KV_STATE_RELAYOUT", env_type::BOOL},
    {"GGML_OPENVINO_NPU_FAST_MASK", env_type::BOOL},
    {"GGML_OPENVINO_DISABLE_SHAPE_FROM_MASK", env_type::BOOL},
    {"GGML_OPENVINO_GDN_NATIVE_STATE", env_type::BOOL},
    {"GGML_OPENVINO_GDN_DECOMPOSE", env_type::BOOL},
    {"GGML_OPENVINO_NPU_L0_HOST_TENSORS", env_type::BOOL},
    {"GGML_OPENVINO_KV_SLICE", env_type::BOOL},
    {"GGML_OPENVINO_NPU_HFA_PREFILL", env_type::BOOL},
    {"GGML_OPENVINO_NPU_DECODE_UNFOLD", env_type::BOOL},
    {"GGML_OPENVINO_NPU_PARALLEL_COMPILE", env_type::BOOL},
    {"GGML_OPENVINO_SELF_CONTAINED_BLOB", env_type::BOOL},
    {"GGML_OPENVINO_SELF_CONTAINED_MMAP", env_type::BOOL},
    {"GGML_OPENVINO_NPU_SHARED_BANK", env_type::BOOL},
    {"GGML_OPENVINO_NPU_WEIGHTLESS_BANK", env_type::BOOL},
    // Build the precise (but O(n_nodes)) graph cache key. Needed by op tests.
    {"GGML_OPENVINO_FULL_GRAPH_KEY", env_type::BOOL},
};

std::string trim_spaces(const std::string & s) {
    size_t begin = 0;
    size_t end = s.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(s[begin]))) {
        begin++;
    }
    while (end > begin && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        end--;
    }
    return s.substr(begin, end - begin);
}

// Strict parse: digits with optional sign and surrounding spaces, in int range.
// "set X=1 && ..." in cmd keeps a trailing space, so trailing spaces are allowed.
bool parse_env_int(const char * v, int & out) {
    char * end = nullptr;
    errno = 0;
    const long value = std::strtol(v, &end, 10);
    if (end == v || errno == ERANGE || value < INT_MIN || value > INT_MAX) {
        return false;
    }
    while (std::isspace(static_cast<unsigned char>(*end))) {
        end++;
    }
    if (*end != '\0') {
        return false;
    }
    out = static_cast<int>(value);
    return true;
}

// The value to cache for a variable that is set. Stops the backend when the value does not fit the type: a silent
// fallback to the default is easy to miss.
std::string checked_env_value(const env_var & var, const char * value) {
    int number = 0;
    switch (var.type) {
    case env_type::STR:
        break;
    case env_type::ENUM: {
        const std::string wanted = trim_spaces(value);
        const std::string values = var.values;
        std::string list;
        for (size_t pos = 0; pos <= values.size();) {
            const size_t bar = std::min(values.find('|', pos), values.size());
            const std::string accepted = values.substr(pos, bar - pos);
            if (wanted == accepted) {
                return wanted;
            }
            list += (list.empty() ? "" : ", ") + accepted;
            pos = bar + 1;
        }
        GGML_ABORT("ggml-openvino: %s=%s is not valid, use one of: %s", var.name, value, list.c_str());
    }
    case env_type::INT:
        if (!parse_env_int(value, number)) {
            GGML_ABORT("ggml-openvino: %s=%s is not valid, use a whole number", var.name, value);
        }
        return std::to_string(number);
    case env_type::FLOAT: {
        char * end = nullptr;
        (void) std::strtof(value, &end);
        if (end == value || !trim_spaces(end).empty()) {
            GGML_ABORT("ggml-openvino: %s=%s is not valid, use a number", var.name, value);
        }
        break;
    }
    case env_type::BOOL:
        if (!parse_env_int(value, number)) {
            GGML_ABORT("ggml-openvino: %s=%s is not valid, use 1 (on) or 0 (off)", var.name, value);
        }
        return number > 0 ? "1" : "0";
    }
    return value;
}

// A misspelled name would otherwise be ignored without a message.
void warn_unknown_env_vars() {
    static const char prefix[] = "GGML_OPENVINO_";
#if defined(_WIN32)
    char ** env = _environ;
#elif defined(__APPLE__)
    char ** env = *_NSGetEnviron();
#else
    char ** env = environ;
#endif
    for (; env != nullptr && *env != nullptr; ++env) {
        if (strncmp(*env, prefix, sizeof(prefix) - 1) != 0) {
            continue;
        }
        const char * eq = strchr(*env, '=');
        const std::string name(*env, eq ? static_cast<size_t>(eq - *env) : strlen(*env));
        const bool known = std::any_of(std::begin(env_vars), std::end(env_vars),
                                       [&name](const env_var & var) { return name == var.name; });
        if (!known) {
            GGML_LOG_WARN("ggml-openvino: unknown environment variable %s is ignored\n", name.c_str());
        }
    }
}

}  // namespace

// =====================================================
// Device Configuration Implementations
// =====================================================

void ggml_openvino_device_config::init() {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    if (initialized) {
        return;
    }
    // Set up front: a failed OpenCL setup below is not retried on every call
    initialized = true;

    for (const env_var & var : env_vars) {
        const char * value = getenv(var.name);
        // unset and empty both mean the default
        if (value != nullptr && value[0] != '\0') {
            environment_variables[var.name] = checked_env_value(var, value);
        }
    }
    warn_unknown_env_vars();
    if (!environment_variables.empty()) {
        std::string settings;
        for (const env_var & var : env_vars) {
            const auto it = environment_variables.find(var.name);
            if (it != environment_variables.end()) {
                settings += std::string(" ") + var.name + "=" + it->second;
            }
        }
        GGML_LOG_INFO("ggml-openvino: settings from the environment:%s\n", settings.c_str());
    }

    if (ggml_openvino_getenv_int("GGML_OPENVINO_STATEFUL_EXECUTION") != 0) {
        GGML_LOG_WARN(
            "ggml-openvino: GGML_OPENVINO_STATEFUL_EXECUTION is deprecated and will be removed, "
            "use stateless execution (the default)\n");
    }

    available_devices = ov_enumerate_devices();
    if (const char * requested = ggml_openvino_getenv_str("GGML_OPENVINO_DEVICE")) {
        std::stringstream ss(requested);
        std::string name;
        while (std::getline(ss, name, ',')) {
            selected_devices.push_back(resolve_openvino_device_name(available_devices, name.c_str()));
        }
        for (const auto & selected : selected_devices) {
            if (selected_devices.size() > 1 && !has_prefix(selected, "GPU")) {
                GGML_ABORT("GGML OpenVINO Backend: GGML_OPENVINO_DEVICE=%s: only GPUs can be combined", requested);
            }
        }
    } else {
        selected_devices.push_back(resolve_openvino_device_name(available_devices, nullptr));
    }
    device_name = selected_devices.front();
    is_npu = has_prefix(device_name, "NPU");

    ggml_openvino_model_cache_init();

    const char * cache_dir = ggml_openvino_getenv_str("GGML_OPENVINO_CACHE_DIR");
    if (has_prefix(device_name, "NPU")) {
        compile_config = ggml_openvino_npu_build_compile_config(cache_dir);
    } else {
        if (cache_dir && strlen(cache_dir) > 0) {
            compile_config.insert(ov::cache_dir(cache_dir));
            compile_config.insert(ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE));
        }
        // Plugin properties of the selected device, applied last
        const bool gpu = has_prefix(device_name, "GPU");
        ggml_openvino_apply_config_overrides(
            compile_config, ggml_openvino_getenv_str(gpu ? "GGML_OPENVINO_GPU_CONFIG" : "GGML_OPENVINO_CPU_CONFIG"),
            gpu ? "GPU" : "CPU");
    }

    if (ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING") >= 2) {
        compile_config.insert(ov::enable_profiling(true));
    }

    for (const auto & name : selected_devices) {
        // Initialize remote context with queue sharing for GPU
        if (has_prefix(name, "GPU")) {
            auto & gpu = gpu_contexts[name];
            // Use the OpenCL context OpenVINO created for this device, so GPU.N gets its own device
            cl_context cl_ctx;
            try {
                auto ov_ctx = ov_singleton_core().get_default_context(name).as<ov::intel_gpu::ocl::ClContext>();
                cl_ctx = ov_ctx.get();
            } catch (const std::exception & e) {
                // The consumers of the remote context have no host fallback, and OpenVINO
                // already reported the device as present.
                GGML_ABORT("ggml-openvino: failed to get the OpenCL context for %s: %s", name.c_str(), e.what());
            }

            cl_int err;
            cl_device_id cl_device;
            err = clGetContextInfo(cl_ctx, CL_CONTEXT_DEVICES, sizeof(cl_device), &cl_device, nullptr);
            if (err != CL_SUCCESS) {
                GGML_ABORT("ggml-openvino: failed to get the OpenCL device for %s: %d", name.c_str(), err);
            }

            cl_platform_id cl_platform;
            err = clGetDeviceInfo(cl_device, CL_DEVICE_PLATFORM, sizeof(cl_platform), &cl_platform, nullptr);
            if (err != CL_SUCCESS) {
                GGML_ABORT("ggml-openvino: failed to get the OpenCL platform for %s: %d", name.c_str(), err);
            }

            cl_mem_fill_fn = (clEnqueueMemFillINTEL_fn) clGetExtensionFunctionAddressForPlatform(
                cl_platform, "clEnqueueMemFillINTEL");
            cl_mem_cpy_fn =
                (clEnqueueMemcpyINTEL_fn) clGetExtensionFunctionAddressForPlatform(cl_platform, "clEnqueueMemcpyINTEL");

            cl_ulong device_max_alloc = 0;
            err = clGetDeviceInfo(cl_device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(device_max_alloc), &device_max_alloc,
                                  nullptr);
            if (err == CL_SUCCESS) {
                gpu.max_alloc_size = device_max_alloc;
            } else {
                // not fatal, ggml then allocates one buffer
                GGML_LOG_WARN("Failed to get OpenCL max allocation size: %d\n", err);
            }

            const cl_queue_properties profiling_properties[] = {
                CL_QUEUE_PROPERTIES,
                CL_QUEUE_PROFILING_ENABLE,
                0,
            };
            const cl_queue_properties * queue_properties =
                ggml_openvino_getenv_int("GGML_OPENVINO_PROFILING") >= 2 ? profiling_properties : nullptr;
            gpu.cl_queue = clCreateCommandQueueWithProperties(cl_ctx, cl_device, queue_properties, &err);
            if (err != CL_SUCCESS) {
                GGML_ABORT("ggml-openvino: failed to create the OpenCL queue for %s: %d", name.c_str(), err);
            }

            // Create OpenVINO remote context with queue sharing
            gpu.remote_context = ov::intel_gpu::ocl::ClContext(ov_singleton_core(), gpu.cl_queue);
        } else if (has_prefix(name, "NPU")) {
            // remote tensor is not used for NPU yet
            // remote_context = ov_singleton_core().get_default_context(name);
        }
    }
}

ggml_openvino_device_config::~ggml_openvino_device_config() {
    for (auto & [name, gpu] : gpu_contexts) {
        if (gpu.cl_queue != nullptr) {
            clReleaseCommandQueue(gpu.cl_queue);
            gpu.cl_queue = nullptr;
        }
    }
}

// Get the global device config singleton
ggml_openvino_device_config & ggml_openvino_get_device_config() {
    static ggml_openvino_device_config config;
    return config;
}

// Initialize device config (call during backend init)
void ggml_openvino_init_device_config() {
    ggml_openvino_get_device_config().init();
}

// Get the device name
const std::string & ggml_openvino_get_device_name() {
    return ggml_openvino_get_device_config().device_name;
}

const std::vector<std::string> & ggml_openvino_get_available_devices() {
    auto & config = ggml_openvino_get_device_config();
    config.init();
    return config.available_devices;
}

bool ggml_openvino_is_selected_device(const std::string & device) {
    const auto & selected = ggml_openvino_get_device_config().selected_devices;
    return std::find(selected.begin(), selected.end(), device) != selected.end();
}

// Get the value of a GGML_OPENVINO_* env var as a string. Returns
// default_value when the var is unset or set to an empty string.
const char * ggml_openvino_getenv_str(const char * var, const char * default_value) {
    auto & env_map = ggml_openvino_get_device_config().environment_variables;
    auto it = env_map.find(var);
    return (it == env_map.end() || it->second.empty()) ? default_value : it->second.c_str();
}

// Get the value of a GGML_OPENVINO_* env var as an int. Returns default_value
// (0) when the var is unset or empty. Used for both integer settings (e.g.
// GGML_OPENVINO_PREFILL_CHUNK_SIZE) and boolean toggles, which are cached as 1 (on) or 0 (off).
int ggml_openvino_getenv_int(const char * var, int default_value) {
    const char * v = ggml_openvino_getenv_str(var, nullptr);
    int value = default_value;
    return v != nullptr && parse_env_int(v, value) ? value : default_value;
}

void ggml_openvino_apply_config_overrides(ov::AnyMap & config, const char * spec, const char * device) {
    if (spec == nullptr) {
        return;
    }
    // A property name: letters, digits, '_' and '.'
    auto is_key = [](const std::string & s) {
        return !s.empty() &&
               std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isalnum(c) || c == '_' || c == '.'; });
    };
    std::string key;
    std::string value;
    auto apply_pair = [&] {
        if (!key.empty()) {
            value = trim_spaces(value);
            config[key] = value;
            GGML_LOG_INFO("GGML OpenVINO: %s config override %s=%s\n", device, key.c_str(), value.c_str());
        }
    };
    const std::string pairs(spec);
    for (size_t pos = 0; pos <= pairs.size();) {
        const size_t comma = std::min(pairs.find(',', pos), pairs.size());
        const std::string part = pairs.substr(pos, comma - pos);
        const size_t eq = part.find('=');
        const std::string name = eq == std::string::npos ? std::string() : trim_spaces(part.substr(0, eq));
        if (is_key(name)) {
            apply_pair();
            key = name;
            value = part.substr(eq + 1);
        } else if (!key.empty()) {
            // no KEY= after the comma: the comma belongs to the value, as in NPUW_DEVICES=NPU,CPU
            value += "," + part;
        } else if (!trim_spaces(part).empty()) {
            GGML_LOG_WARN("ggml-openvino: GGML_OPENVINO_%s_CONFIG: '%s' is ignored, expected KEY=VALUE\n", device,
                          part.c_str());
        }
        pos = comma + 1;
    }
    apply_pair();
}

std::string ggml_openvino_config_overrides_in_use() {
    const char * name = ggml_openvino_is_npu() ? "GGML_OPENVINO_NPU_CONFIG" :
                        ggml_openvino_is_gpu() ? "GGML_OPENVINO_GPU_CONFIG" :
                                                 "GGML_OPENVINO_CPU_CONFIG";
    const char * value = ggml_openvino_getenv_str(name);
    return value ? std::string(name) + "=" + value : std::string();
}

bool ggml_openvino_gdn_prescale_enabled() {
    // float value, so not getenv_int: "64.0" must enable it. Unset means on.
    const char * v = ggml_openvino_getenv_str("GGML_OPENVINO_GDN_Q_SCALE", nullptr);
    return v == nullptr || std::strtof(v, nullptr) > 0.0f;
}

bool ggml_openvino_gdn_native_state_enabled() {
    // only verified on NPU so far
    return ggml_openvino_is_npu() && ggml_openvino_getenv_int("GGML_OPENVINO_GDN_NATIVE_STATE") > 0;
}

bool ggml_openvino_npu_is_4000() {
    if (!ggml_openvino_is_npu()) {
        return false;
    }
    static const bool is_4000 = [] {
        try {
            return ov_singleton_core().get_property("NPU", "DEVICE_ARCHITECTURE").as<std::string>() == "4000";
        } catch (...) {
            return false;
        }
    }();
    return is_4000;
}

bool ggml_openvino_gdn_decompose_enabled() {
    if (ggml_openvino_getenv_str("GGML_OPENVINO_GDN_DECOMPOSE") != nullptr) {
        return ggml_openvino_getenv_int("GGML_OPENVINO_GDN_DECOMPOSE") > 0;
    }
    return ggml_openvino_npu_is_4000();
}

int ggml_openvino_gdn_decompose_chunk() {
    const int chunk = ggml_openvino_getenv_int("GGML_OPENVINO_GDN_DECOMPOSE_CHUNK");
    if (chunk >= 2 && chunk <= 128 && (chunk & (chunk - 1)) == 0) {
        return chunk;
    }
    return 16;
}

float ggml_openvino_gdn_q_scale() {
    constexpr float default_scale = 64.0f;
    const char * v = ggml_openvino_getenv_str("GGML_OPENVINO_GDN_Q_SCALE", nullptr);
    float scale = v ? std::strtof(v, nullptr) : default_scale;
    if (scale < default_scale) {
        GGML_LOG_WARN("ggml-openvino: GGML_OPENVINO_GDN_Q_SCALE=%s is below the minimum %.1f, clamping\n", v,
                      default_scale);
        scale = default_scale;
    }
    return scale;
}

bool ggml_openvino_reduce_compile_mem_enabled() {
    const char * reduce_compile_mem = ggml_openvino_getenv_str("GGML_OPENVINO_REDUCE_COMPILE_MEM");
    if (reduce_compile_mem != nullptr) {
        return ggml_openvino_getenv_int("GGML_OPENVINO_REDUCE_COMPILE_MEM") != 0;
    }
    return ggml_openvino_getenv_int("GGML_OPENVINO_MEMORY_OPTIMIZE") != 0;
}

bool ggml_openvino_release_weights_enabled() {
    const char * release_weights = ggml_openvino_getenv_str("GGML_OPENVINO_RELEASE_WEIGHTS");
    if (release_weights != nullptr) {
        return ggml_openvino_is_gpu() && ggml_openvino_getenv_int("GGML_OPENVINO_RELEASE_WEIGHTS") != 0;
    }
    return ggml_openvino_is_gpu() && ggml_openvino_getenv_int("GGML_OPENVINO_MEMORY_OPTIMIZE") != 0;
}

// Check if running on NPU
bool ggml_openvino_is_npu() {
    return ggml_openvino_get_device_config().is_npu;
}

bool ggml_openvino_is_gpu() {
    return has_prefix(ggml_openvino_get_device_name(), "GPU");
}

bool ggml_openvino_kv_slice_enabled() {
    return ggml_openvino_getenv_int("GGML_OPENVINO_KV_SLICE", 1) != 0;
}

bool ggml_openvino_npu_fast_mask_enabled() {
    return ggml_openvino_getenv_int("GGML_OPENVINO_NPU_FAST_MASK", 1) != 0;
}

bool ggml_openvino_npu_l0_host_tensors_enabled() {
    return ggml_openvino_getenv_int("GGML_OPENVINO_NPU_L0_HOST_TENSORS", 1) != 0;
}

bool ggml_openvino_npu_hfa_prefill_enabled() {
    return ggml_openvino_is_npu() && ggml_openvino_getenv_int("GGML_OPENVINO_NPU_HFA_PREFILL") != 0;
}

bool ggml_openvino_npu_decode_unfold_enabled() {
    return ggml_openvino_getenv_int("GGML_OPENVINO_NPU_DECODE_UNFOLD") != 0;
}

bool ggml_openvino_npu_parallel_compile_enabled() {
    return ggml_openvino_getenv_int("GGML_OPENVINO_NPU_PARALLEL_COMPILE") != 0;
}

int ggml_openvino_npu_decode_bucket() {
    if (!ggml_openvino_is_npu()) {
        return 0;
    }
    // A self-contained blob embeds its weights bank, and NPUW cannot load a blob into a bank that already holds
    // weights, so a decode model per KV window cannot be imported next to the decode model of the pair.
    if (ggml_openvino_getenv_str("GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR") &&
        ggml_openvino_getenv_int("GGML_OPENVINO_SELF_CONTAINED_BLOB") != 0 &&
        !ggml_openvino_npu_weightless_bank_enabled()) {
        return 0;
    }
    return ggml_openvino_getenv_int("GGML_OPENVINO_NPU_BLOB_PREBUILD", 1) != 0 ? 1024 : 0;
}

int ggml_openvino_static_kv_window(int rows, int ctx) {
    int window = 1;
    while (window < rows && window < ctx) {
        window *= 2;
    }
    return std::min(window, ctx);
}

bool ggml_openvino_npu_kv_on_demand_enabled() {
    return ggml_openvino_is_npu() && ggml_openvino_getenv_int("GGML_OPENVINO_NPU_KV_ON_DEMAND") != 0;
}

bool ggml_openvino_npu_kv_head_major_enabled() {
    return ggml_openvino_npu_kv_on_demand_enabled() && ggml_openvino_npu_decode_bucket() > 0 &&
           ggml_openvino_kv_slice_enabled();
}

bool ggml_openvino_npu_raw_weight_cache_enabled() {
    return ggml_openvino_getenv_int("GGML_OPENVINO_SELF_CONTAINED_BLOB") != 0 ||
           ggml_openvino_npu_weightless_bank_enabled();
}

bool ggml_openvino_npu_weightless_bank_enabled() {
    return ggml_openvino_getenv_int("GGML_OPENVINO_NPU_WEIGHTLESS_BANK") != 0;
}

namespace {
struct kv_storage_entry {
    size_t row_bytes = 0;
    size_t max_rows = 0;
    size_t cap_rows = 0;
    // 0: ggml row layout [rows][heads * head_size]; >0: head-major [heads][cap_rows][head_size].
    size_t heads = 0;
    ov::Tensor storage;
};

struct kv_storage_registry {
    std::mutex mutex;
    // Keyed by the ggml data address of each KV tensor.
    std::map<uintptr_t, kv_storage_entry> entries;
};

kv_storage_registry & kv_registry() {
    static kv_storage_registry reg;
    return reg;
}

kv_storage_entry * kv_find_locked(kv_storage_registry & reg, uintptr_t addr, uintptr_t & base) {
    auto it = reg.entries.upper_bound(addr);
    if (it == reg.entries.begin()) {
        return nullptr;
    }
    --it;
    if (addr >= it->first + it->second.max_rows * it->second.row_bytes) {
        return nullptr;
    }
    base = it->first;
    return &it->second;
}

// Storage offset of byte `off` of the ggml tensor; its row must be below cap_rows.
size_t kv_storage_offset(const kv_storage_entry & e, size_t off) {
    if (e.heads == 0) {
        return off;
    }
    const size_t head_bytes = e.row_bytes / e.heads;
    const size_t row = off / e.row_bytes;
    const size_t in_row = off % e.row_bytes;
    return ((in_row / head_bytes) * e.cap_rows + row) * head_bytes + in_row % head_bytes;
}

bool kv_grow_locked(kv_storage_entry & e, size_t rows, size_t heads) {
    if (heads > 0 && e.heads != heads) {
        if (e.cap_rows > 0) {
            throw std::runtime_error("ggml-openvino: on-demand KV storage layout changed");
        }
        GGML_ASSERT(e.row_bytes % heads == 0);
        e.heads = heads;
    }
    rows = std::min(rows, e.max_rows);
    if (rows <= e.cap_rows) {
        return false;
    }
    const int bucket = ggml_openvino_npu_decode_bucket();
    size_t cap = bucket > 0 ? static_cast<size_t>(bucket) : 1024;
    while (cap < rows) {
        cap *= 2;
    }
    cap = std::min(cap, e.max_rows);
    const size_t bytes = cap * e.row_bytes;
    ov::Tensor grown;
    if (ggml_openvino_npu_l0_host_tensors_enabled()) {
        grown = ov_singleton_core().get_default_context("NPU").create_host_tensor(ov::element::u8, ov::Shape{bytes});
    } else {
        grown = ov::Tensor(ov::element::u8, ov::Shape{bytes});
    }
    auto * dst = static_cast<uint8_t *>(grown.data());
    // Rows past the history are masked out, but must stay finite (0 * NaN is NaN).
    std::memset(dst, 0, bytes);
    if (e.cap_rows > 0) {
        const auto * src = static_cast<const uint8_t *>(e.storage.data());
        if (e.heads == 0) {
            std::memcpy(dst, src, e.cap_rows * e.row_bytes);
        } else {
            const size_t head_bytes = e.row_bytes / e.heads;
            for (size_t h = 0; h < e.heads; ++h) {
                std::memcpy(dst + h * cap * head_bytes, src + h * e.cap_rows * head_bytes, e.cap_rows * head_bytes);
            }
        }
    }
    // Requests still bound to the old storage keep it alive until they are rebound or released.
    e.storage = grown;
    e.cap_rows = cap;
    return true;
}
}  // namespace

void ggml_openvino_kv_storage_register(void * data, size_t row_bytes, size_t rows) {
    auto & reg = kv_registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    auto & e = reg.entries[reinterpret_cast<uintptr_t>(data)];
    e = kv_storage_entry{};
    e.row_bytes = row_bytes;
    e.max_rows = rows;
}

void ggml_openvino_kv_storage_unregister(const void * base, size_t size) {
    auto & reg = kv_registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    const auto begin = reinterpret_cast<uintptr_t>(base);
    reg.entries.erase(reg.entries.lower_bound(begin), reg.entries.lower_bound(begin + size));
}

void * ggml_openvino_kv_storage(const void * ptr, size_t bytes) {
    auto & reg = kv_registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    if (reg.entries.empty()) {
        return nullptr;
    }
    const auto addr = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t base = 0;
    auto * e = kv_find_locked(reg, addr, base);
    if (e == nullptr) {
        return nullptr;
    }
    const size_t offset = addr - base;
    if (e->heads > 0 && offset != 0) {
        throw std::runtime_error("ggml-openvino: head-major KV storage is only bound from the tensor start");
    }
    if (kv_grow_locked(*e, (offset + bytes + e->row_bytes - 1) / e->row_bytes, 0)) {
        GGML_LOG_WARN("%s: KV storage grown during binding to %zu rows\n", __func__, e->cap_rows);
    }
    if (offset + bytes > e->cap_rows * e->row_bytes) {
        throw std::runtime_error("ggml-openvino: on-demand KV access beyond the KV cache");
    }
    return static_cast<uint8_t *>(e->storage.data()) + offset;
}

bool ggml_openvino_kv_storage_reserve(const void * data, size_t rows, size_t heads) {
    auto & reg = kv_registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    auto it = reg.entries.find(reinterpret_cast<uintptr_t>(data));
    if (it == reg.entries.end()) {
        return false;
    }
    const bool had_storage = it->second.cap_rows > 0;
    return kv_grow_locked(it->second, rows, heads) && had_storage;
}

size_t ggml_openvino_kv_storage_capacity(const void * data) {
    auto & reg = kv_registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    auto it = reg.entries.find(reinterpret_cast<uintptr_t>(data));
    return it == reg.entries.end() ? 0 : it->second.cap_rows;
}

bool ggml_openvino_kv_storage_copy(const void * ptr, void * host, size_t size, ggml_openvino_kv_copy_op op) {
    auto & reg = kv_registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    if (reg.entries.empty()) {
        return false;
    }
    uintptr_t base = 0;
    auto * e = kv_find_locked(reg, reinterpret_cast<uintptr_t>(ptr), base);
    if (e == nullptr) {
        return false;
    }
    const size_t begin = reinterpret_cast<uintptr_t>(ptr) - base;
    if (op != GGML_OPENVINO_KV_READ) {
        kv_grow_locked(*e, (begin + size + e->row_bytes - 1) / e->row_bytes, 0);
    }
    auto * storage = static_cast<uint8_t *>(e->storage.data());
    auto * h = static_cast<uint8_t *>(host);
    const size_t head_bytes = e->heads > 0 ? e->row_bytes / e->heads : e->row_bytes;
    for (size_t done = 0; done < size;) {
        const size_t off = begin + done;
        const size_t n = std::min(size - done, head_bytes - off % head_bytes);
        const bool stored = off / e->row_bytes < e->cap_rows;
        uint8_t * s = stored ? storage + kv_storage_offset(*e, off) : nullptr;
        if (op == GGML_OPENVINO_KV_READ) {
            if (s) {
                std::memcpy(h + done, s, n);
            } else {
                std::memset(h + done, 0, n);
            }
        } else if (op == GGML_OPENVINO_KV_WRITE) {
            std::memcpy(s, h + done, n);
        } else {
            std::memset(s, *h, n);
        }
        done += n;
    }
    return true;
}

bool ggml_openvino_kv_storage_write_rows(const void * data, const void * src, size_t src_stride, size_t src_col,
                                         size_t dst_row, size_t n) {
    auto & reg = kv_registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    auto it = reg.entries.find(reinterpret_cast<uintptr_t>(data));
    if (it == reg.entries.end() || it->second.heads == 0) {
        return false;
    }
    auto & e = it->second;
    if (dst_row + n > e.cap_rows || src_col + n > src_stride) {
        throw std::runtime_error("ggml-openvino: head-major KV row write out of range");
    }
    const size_t head_bytes = e.row_bytes / e.heads;
    auto * dst = static_cast<uint8_t *>(e.storage.data());
    const auto * s = static_cast<const uint8_t *>(src);
    for (size_t h = 0; h < e.heads; ++h) {
        std::memcpy(dst + (h * e.cap_rows + dst_row) * head_bytes, s + (h * src_stride + src_col) * head_bytes,
                    n * head_bytes);
    }
    return true;
}

void ggml_openvino_kv_storage_clear(const void * base, size_t size, uint8_t value) {
    auto & reg = kv_registry();
    std::lock_guard<std::mutex> lock(reg.mutex);
    const auto begin = reinterpret_cast<uintptr_t>(base);
    for (auto it = reg.entries.lower_bound(begin); it != reg.entries.end() && it->first < begin + size; ++it) {
        if (it->second.cap_rows > 0) {
            std::memset(it->second.storage.data(), value, it->second.storage.get_byte_size());
        }
    }
}

bool ggml_openvino_shape_from_mask_enabled() {
    return ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_SHAPE_FROM_MASK") == 0;
}

size_t ggml_openvino_max_alloc_size(const std::string & device) {
    const auto & gpu_contexts = ggml_openvino_get_device_config().gpu_contexts;
    auto it = gpu_contexts.find(device);
    return it == gpu_contexts.end() ? SIZE_MAX : it->second.max_alloc_size;
}

// Get the remote context for the device (returns empty optional for CPU)
std::optional<ov::RemoteContext> ggml_openvino_get_remote_context(const std::string & device) {
    const auto & gpu_contexts = ggml_openvino_get_device_config().gpu_contexts;
    auto it = gpu_contexts.find(device);
    if (it == gpu_contexts.end()) {
        return std::nullopt;
    }
    return it->second.remote_context;
}

// Get the compile config for the current device
const ov::AnyMap & ggml_openvino_get_compile_config() {
    return ggml_openvino_get_device_config().compile_config;
}

// Get the OpenCL command queue of the device for GPU operations
cl_command_queue ggml_openvino_get_cl_queue(const std::string & device) {
    const auto & gpu_contexts = ggml_openvino_get_device_config().gpu_contexts;
    auto it = gpu_contexts.find(device);
    return it == gpu_contexts.end() ? nullptr : it->second.cl_queue;
}

// Get the clEnqueueMemFillINTEL function pointer
clEnqueueMemFillINTEL_fn ggml_openvino_get_clEnqueueMemFillINTEL() {
    return ggml_openvino_get_device_config().cl_mem_fill_fn;
}

// Get the clEnqueueMemcpyINTEL function pointer
clEnqueueMemcpyINTEL_fn ggml_openvino_get_clEnqueueMemcpyINTEL() {
    return ggml_openvino_get_device_config().cl_mem_cpy_fn;
}

// Get requantization type for a tensor type (returns nullopt if no requant needed)
std::optional<ExtraQuantType> ggml_openvino_get_requant_type(const ggml_tensor * tensor, bool no_requant) {
    if (no_requant) {
        return std::nullopt;
    }
    if (strncmp(tensor->name, "token_embd.weight", 17) == 0) {
        if (ggml_openvino_is_npu()) {
            if (auto npu_type = ggml_openvino_npu_token_embd_requant_type(tensor)) {
                return npu_type;
            }
        }
        return ExtraQuantType::Q8_0_C;
    }

    if (strncmp(tensor->name, "output.weight", 13) == 0) {
        return ExtraQuantType::Q8_0_C;
    }
    if (ggml_openvino_is_npu()) {
        return ggml_openvino_npu_weight_requant_type(tensor);
    }
    // By default Q6_K/Q5_K are requantized to Q8_0_C, which *inflates* 6- and 5-bit weights to 8
    // while the rest of the model stays at 4 bits, and Q4_K keeps its native group-32 layout
    // (an f16 scale plus an f16 zero point per 32 weights = 0.125 B/weight of metadata).
    // Decode of a large model is bandwidth-bound, so both cost throughput.
    //
    // GGML_OPENVINO_REQUANT_KQUANT selects a 4-bit target instead. Names are
    // q4_<sym|asym><group>[_all]: <sym|asym> says whether a per-group zero point is kept, <group>
    // is the group size. The _all suffix includes Q4_K; q4_asym64_all also includes
    // ordinary Q8_0/Q5_1 linear weights, matching the INT4_ASYM/group-64 primary precision
    // used by NNCF with ratio=1.0. Embedding/output retain the channel-wise INT8 overrides above:
    //   q4_sym128      Q6_K/Q5_K -> Q4_0_128 (u4, group 128, symmetric)
    //   q4_sym128_all  and Q4_K too -- drops Q4_K's per-32 zero point, which costs some accuracy
    //   q4_asym64      Q6_K/Q5_K -> Q4_1_64 (u4, group 64, asymmetric)
    //   q4_asym64_all  Q6_K/Q5_K/Q4_K/Q8_0/Q5_1 -> Q4_1_64 (u4, group 64, asymmetric)
    //   native         no requantization at all (keep Q6_K/Q5_K as they are)
    //
    // q4_asym64 leaves Q4_K at its native group 32. Use q4_asym64_all to keep the group size uniform.
    const char * rq = ggml_openvino_getenv_str("GGML_OPENVINO_REQUANT_KQUANT");
    auto is_opt = [rq](const char * name) {
        return rq && strcmp(rq, name) == 0;
    };
    const bool sym128 = is_opt("q4_sym128");
    const bool sym128_all = is_opt("q4_sym128_all");
    const bool asym64_all = is_opt("q4_asym64_all");
    const bool asym64 = is_opt("q4_asym64");

    // TODO: temporary workaround for slow GPU-plugin GatherMatmul prefill above 16 tokens. Q4_1_64 experts let the plugin fuse the MoE into MOECompressed.
    if (ggml_openvino_is_gpu() && tensor->ne[2] > 1 && strstr(tensor->name, "exps") != nullptr &&
        (tensor->type == GGML_TYPE_Q4_K || tensor->type == GGML_TYPE_Q5_K || tensor->type == GGML_TYPE_Q6_K)) {
        return ExtraQuantType::Q4_1_64;
    }

    if (tensor->type == GGML_TYPE_Q4_K) {
        if (sym128_all) {
            return ExtraQuantType::Q4_0_128;
        }
        if (asym64_all) {
            return ExtraQuantType::Q4_1_64;
        }
    }
    // Include dense Q8_0/Q5_1 in the explicit all-asymmetric policy. Keep the default GPU
    // workaround below restricted to MoE experts: ordinary grouped INT8 MatMul is supported.
    if (asym64_all && (tensor->type == GGML_TYPE_Q5_1 || tensor->type == GGML_TYPE_Q8_0)) {
        // A group must fit within each row. Use backup channel-wise INT8 for incompatible
        // widths (Q8_0/Q5_1 allow rows of 32 weights); every affected Qwen3.6 row is divisible by 64.
        return tensor->ne[0] % 64 == 0 ? ExtraQuantType::Q4_1_64 : ExtraQuantType::Q8_0_C;
    }
    if (tensor->type == GGML_TYPE_Q5_0) {
        //return ExtraQuantType::Q4_0_128;
        return ExtraQuantType::Q4_0_64;
    }
    // MoE expert weights (3D, ne[2] = n_expert) stored as Q5_1/Q8_0 are the expert-side
    // equivalent of Q6_K/Q5_K: kept at 8 bits by default while the rest of the model is at 4
    // (gemma-4 26B-A4B keeps its down projection there). Send them to 4 bits under the same
    // option, at group 64 rather than 128: the down expert has k=704, which 64 divides
    // (704/64 = 11) and 128 does not.
    if (tensor->ne[2] > 1 && (tensor->type == GGML_TYPE_Q5_1 || tensor->type == GGML_TYPE_Q8_0)) {
        if (sym128 || sym128_all) {
            return ExtraQuantType::Q4_0_64;
        }
        if (asym64 || asym64_all) {
            return ExtraQuantType::Q4_1_64;
        }
        // TODO: temporary workaround for a known OpenVINO GPU-plugin bug -- remove once the
        // plugin computes grouped 8-bit GatherMatmulCompressed correctly. This costs accuracy
        // (5/8-bit -> 4-bit) on any model it applies to, so it must not outlive the bug.
        //
        // On GPU these would otherwise stay in their native *grouped 8-bit* layout, which the GPU
        // plugin's GatherMatmulCompressed computes incorrectly -- gemma-4 26B-A4B (whose down
        // projection is Q5_1) produces garbage, while the same graph is correct on CPU. It is
        // specific to grouped 8 bit: the gate/up experts are grouped u4 *with* a zero point and
        // are fine, and Qwen3.5 / granite are fine because their Q5_K/Q6_K down projections
        // already requantize to per-channel Q8_0_C (grouped=0). Sending these to grouped 4 bit
        // avoids the broken layout and restores correct output.
        // Opt out with GGML_OPENVINO_REQUANT_KQUANT=native.
        if (ggml_openvino_is_gpu() && !is_opt("native")) {
            return ExtraQuantType::Q4_0_64;
        }
    }
    switch (tensor->type) {
    case GGML_TYPE_Q6_K:
    case GGML_TYPE_Q5_K:
        if (sym128 || sym128_all) {
            return ExtraQuantType::Q4_0_128;
        }
        if (asym64 || asym64_all) {
            return ExtraQuantType::Q4_1_64;
        }
        if (is_opt("native")) {
            return std::nullopt;
        }
        return ExtraQuantType::Q8_0_C;
    default:
        return std::nullopt;
    }
}

bool ggml_openvino_use_exact_zero_point(const ggml_tensor * tensor) {
    // Other 3D weights, such as MLA's per-head attn_k_b / attn_v_b, feed a batched MatMul, where the GPU plugin fails
    // on u4 weights with an f16 zero point (CL_OUT_OF_RESOURCES), so the opt-in covers 2D weights only.
    const bool is_moe_expert = tensor->ne[2] > 1 && strstr(tensor->name, "exps") != nullptr;
    return is_moe_expert || (tensor->ne[2] == 1 && ggml_openvino_getenv_int("GGML_OPENVINO_EXACT_ZERO_POINT") > 0);
}

// =====================================================
// Extracted Layout Calculation
// =====================================================

ggml_openvino_extracted_layout ggml_openvino_get_extracted_layout(const ggml_tensor * tensor, bool use_bias) {
    ggml_openvino_extracted_layout layout = {};
    layout.is_symmetric = false;

    if (!ggml_is_quantized(tensor->type)) {
        return layout;
    }

    // Most quantized weights use the existing 2D extraction path. 3D expert weights for
    // MUL_MAT_ID (MoE) are also supported, either as MXFP4 (packed, dedicated branch below) or via the
    // generic sizing math below, which is shape-agnostic (based on total element count). Only reject 4D.
    if (tensor->ne[3] != 1) {
        return layout;
    }

    // An exact zero point (see extract_quantized_weights) needs an f16 zp slot instead of the
    // packed integer one. Keep in sync with process_weight_tensor, which fills the slot.
    const bool exact_zp = ggml_openvino_use_exact_zero_point(tensor);

    int64_t n_elements = ggml_nelements(tensor);
    const size_t alignment = 64;  // Good for SIMD

    if (tensor->type == GGML_TYPE_MXFP4 && (tensor->ne[2] > 1 || tensor->ne[3] > 1)) {
        layout.weights_per_block = 32;
        layout.is_symmetric = true;
        layout.weights_size = ggml_nbytes(tensor);
        layout.weights_offset = 0;
        layout.total_size = layout.weights_size;
        return layout;
    }

    // Check if requantization is needed (NPU-specific)
    auto requant_type = ggml_openvino_get_requant_type(tensor, use_bias);
    if (requant_type.has_value()) {
        layout.is_requant = true;
        layout.requant_type = requant_type;

        // Special case: requant to F16 - just store F16 weights, no scales/zp
        if (requant_type.value() == ExtraQuantType::F16) {
            layout.weights_size = n_elements * sizeof(uint16_t);  // F16 = 2 bytes
            layout.total_size = layout.weights_size;
            layout.weights_offset = 0;
            // No scales/zp for F16
            return layout;
        }

        // Requant to different quantized format (e.g., Q4_0_128)
        switch (requant_type.value()) {
        case ExtraQuantType::Q4_0_128:
            layout.is_u4 = true;
            layout.weights_per_block = 128;
            layout.is_symmetric = true;
            break;
        case ExtraQuantType::Q4_1_64:
            layout.is_u4 = true;
            layout.weights_per_block = 64;
            layout.is_symmetric = false;
            break;
        case ExtraQuantType::Q4_0_64:
            layout.is_u4 = true;
            layout.weights_per_block = 64;
            layout.is_symmetric = true;
            break;
        case ExtraQuantType::Q4_0_C:
            layout.is_u4 = true;
            layout.weights_per_block = tensor->ne[0];
            layout.is_symmetric = true;
            break;
        case ExtraQuantType::Q8_0_32:
            layout.is_u4 = false;
            layout.weights_per_block = 32;
            layout.is_symmetric = true;
            break;
        case ExtraQuantType::Q8_0_C:
            layout.is_u4 = false;
            layout.weights_per_block = tensor->ne[0];
            layout.is_symmetric = true;
            break;
        case ExtraQuantType::Q8_1_C:
            layout.is_u4 = false;
            layout.weights_per_block = tensor->ne[0];
            break;
        default:
            GGML_ABORT("Code of re-quantizing to channel-wise is not updated");
        }

        if (layout.is_requant) {
            // Calculate sizes for requantized format
            layout.weights_size = layout.is_u4 ? (n_elements / 2) : n_elements;
            int64_t n_blocks = n_elements / layout.weights_per_block;
            layout.scales_size = n_blocks * sizeof(uint16_t);
            // For symmetric quantization, no zp needed (weights stored as signed)
            if (layout.is_symmetric) {
                layout.zp_size = 0;
            } else {
                layout.zp_size = layout.is_u4 ? ((n_blocks + 1) / 2) : n_blocks;
            }

            layout.weights_offset = 0;
            layout.scales_offset = ((layout.weights_size + alignment - 1) / alignment) * alignment;
            layout.zp_offset = layout.scales_offset + ((layout.scales_size + alignment - 1) / alignment) * alignment;
            layout.total_size = layout.zp_offset + layout.zp_size;
            layout.total_size = std::max(layout.total_size, ggml_nbytes(tensor));
            return layout;
        }
    }

    // Normal extraction (no requant) - determine format based on tensor type
    layout.is_u4 = false;
    layout.weights_per_block = 32;
    layout.is_symmetric = false;

    switch (tensor->type) {
    case GGML_TYPE_MXFP4:
    case GGML_TYPE_Q4_0:
        layout.is_u4 = true;
        layout.is_symmetric = true;
        break;

    case GGML_TYPE_Q4_1:
    case GGML_TYPE_Q4_K:
        layout.is_u4 = true;
        break;

    case GGML_TYPE_Q8_0:
        layout.is_symmetric = true;
        break;

    case GGML_TYPE_Q5_0:
        // i8 weights (5-bit values minus 16), symmetric (scale only)
        layout.is_symmetric = true;
        break;

    case GGML_TYPE_Q5_1:
        // u8 weights (5-bit values), asymmetric (scale + zero point)
        break;

    case GGML_TYPE_Q6_K:
        layout.weights_per_block = 16;
        layout.is_symmetric = true;
        break;

    case GGML_TYPE_Q5_K:
        break;

    default:
        // Unsupported quantization type
        return layout;
    }

    // Calculate sizes
    // Weights: U4 = n_elements/2 bytes, U8 = n_elements bytes
    layout.weights_size = layout.is_u4 ? (n_elements / 2) : n_elements;

    // Scales: F16 per block, except MXFP4 which stores one E8M0 byte per block.
    int64_t n_blocks = n_elements / layout.weights_per_block;
    layout.scales_size = n_blocks * (tensor->type == GGML_TYPE_MXFP4 ? sizeof(uint8_t) : sizeof(uint16_t));
    // For symmetric quantization, no zp needed (weights stored as signed). An exact zero point
    // needs one f16 per block instead of a packed u4/u8 integer zero point.
    if (layout.is_symmetric) {
        layout.zp_size = 0;
    } else if (use_bias || exact_zp) {
        layout.zp_size = n_blocks * sizeof(uint16_t);
    } else {
        layout.zp_size = layout.is_u4 ? ((n_blocks + 1) / 2) : n_blocks;
    }

    // Layout in buffer: [weights | scales | zp] with alignment
    layout.weights_offset = 0;
    layout.scales_offset = ((layout.weights_size + alignment - 1) / alignment) * alignment;
    layout.zp_offset = layout.scales_offset + ((layout.scales_size + alignment - 1) / alignment) * alignment;
    layout.total_size = layout.zp_offset + layout.zp_size;
    layout.total_size = std::max(layout.total_size, ggml_nbytes(tensor));

    return layout;
}

ggml_openvino_tensor_extra * ggml_openvino_create_tensor_extra(const ggml_tensor * tensor,
                                                               bool is_remote,
                                                               const std::string & device) {
    ov::Shape shape;
    for (int i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        shape.push_back(static_cast<size_t>(tensor->ne[i]));
    }

    ov::element::Type element_type;
    switch (tensor->type) {
    case GGML_TYPE_F32:
        element_type = ov::element::f32;
        break;
    case GGML_TYPE_F16:
        element_type = ov::element::f16;
        break;
    case GGML_TYPE_BF16:
        element_type = ov::element::bf16;
        break;
    case GGML_TYPE_I32:
        element_type = ov::element::i32;
        break;
    case GGML_TYPE_I64:
        element_type = ov::element::i64;
        break;
    default:
        // GGML_LOG_WARN("%s: unsupported tensor type for ov::Tensor: %s\n", __func__, ggml_type_name(tensor->type));
        return nullptr;
    }

    auto remote_context = ggml_openvino_get_remote_context(device);

    std::shared_ptr<ov::Tensor> ov_tensor;
    if (is_remote) {
        GGML_ASSERT(ggml_openvino_is_gpu());
        auto gpu_context = remote_context->as<ov::intel_gpu::ocl::ClContext>();
        auto usm_tensor = gpu_context.create_tensor(element_type, shape,
                                                    tensor->data);
        ov_tensor = std::make_shared<ov::intel_gpu::ocl::USMTensor>(std::move(usm_tensor));
    } else {
        ov_tensor = std::make_shared<ov::Tensor>(element_type, shape, tensor->data);
    }

    return new ggml_openvino_tensor_extra(ov_tensor, tensor->data);
}
