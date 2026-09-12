#include "dispatch/dispatch.h"
#include "hrx-interop-utils.h"
#include "kernel-corpus/kernel-corpus.h"
#include "runtime/kernel-executable-cache.h"
#include "runtime/loom-kernel-jit.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#define REQUIRE(condition)                                                                           \
    do {                                                                                             \
        if (!(condition)) {                                                                          \
            std::fprintf(stderr, "%s:%d: requirement failed: %s\n", __FILE__, __LINE__, #condition); \
            std::abort();                                                                            \
        }                                                                                            \
    } while (false)

namespace {

class HrxTestDevice {
  public:
    ~HrxTestDevice() {
        if (device != nullptr) {
            hrx_device_release(device);
        }
    }

    bool open() {
        hrx_status_t init_status = hrx_gpu_initialize(0);
        if (!hrx_status_is_ok(init_status)) {
            if (hrx_status_code(init_status) == HRX_STATUS_ALREADY_EXISTS) {
                hrx_status_ignore(init_status);
            } else {
                hrx_status_ignore(init_status);
                return false;
            }
        }

        int count = 0;
        if (ggml::hrx::ErrorResult error = ggml::hrx::take_status(hrx_gpu_device_count(&count))) {
            std::fprintf(stderr, "skip executable cache materialization: device count failed: %s\n", error->c_str());
            return false;
        }
        for (int i = 0; i < count; ++i) {
            hrx_device_t candidate = nullptr;
            if (ggml::hrx::ErrorResult error = ggml::hrx::take_status(hrx_gpu_device_get(i, &candidate))) {
                std::fprintf(stderr, "skip HRX device %d: %s\n", i, error->c_str());
                continue;
            }
            if (candidate == nullptr) {
                continue;
            }
            hrx_device_retain(candidate);
            std::optional<std::string> candidate_architecture =
                read_string_property(candidate, HRX_DEVICE_PROPERTY_ARCHITECTURE);
            if (!candidate_architecture) {
                hrx_device_release(candidate);
                continue;
            }
            device       = candidate;
            architecture = *candidate_architecture;
            return true;
        }
        return false;
    }

    hrx_device_t device = nullptr;
    std::string  architecture;

  private:
    static std::optional<std::string> read_string_property(hrx_device_t device, hrx_device_property_t property) {
        std::vector<char> buffer(64);
        while (buffer.size() <= 4096) {
            hrx_status_t status = hrx_device_get_property(device, property, buffer.data(), buffer.size());
            if (hrx_status_is_ok(status)) {
                return std::string(buffer.data());
            }
            if (hrx_status_code(status) != HRX_STATUS_OUT_OF_RANGE) {
                if (ggml::hrx::ErrorResult error = ggml::hrx::take_status(status)) {
                    std::fprintf(stderr, "HRX property query failed: %s\n", error->c_str());
                }
                return std::nullopt;
            }
            hrx_status_ignore(status);
            buffer.resize(buffer.size() * 2);
        }
        return std::nullopt;
    }
};

static ggml_hrx_loom_jit_source_format to_jit_source_format(ggml::hrx::KernelSourceFormat format) {
    switch (format) {
        case ggml::hrx::KERNEL_SOURCE_FORMAT_TEXT:
            return GGML_HRX_LOOM_JIT_SOURCE_FORMAT_TEXT;
        case ggml::hrx::KERNEL_SOURCE_FORMAT_BINARY:
            return GGML_HRX_LOOM_JIT_SOURCE_FORMAT_BYTECODE;
    }
    return GGML_HRX_LOOM_JIT_SOURCE_FORMAT_TEXT;
}

static const ggml::hrx::KernelDefinition & find_kernel(const char * name) {
    const ggml::hrx::KernelCorpus & corpus = ggml::hrx::get_qwen_kernel_corpus();
    for (const ggml::hrx::KernelDefinition & kernel : corpus.kernels) {
        if (std::strcmp(kernel.name, name) == 0) {
            return kernel;
        }
    }
    std::fprintf(stderr, "missing test kernel: %s\n", name);
    std::abort();
}

static const ggml::hrx::KernelDefinition * find_targeted_kernel(const char * name, const char * target) {
    const ggml::hrx::KernelCorpus & corpus = ggml::hrx::get_qwen_kernel_corpus();
    for (const ggml::hrx::KernelDefinition & kernel : corpus.kernels) {
        if (std::strcmp(kernel.name, name) == 0 && std::strcmp(kernel.target_selector, target) == 0) {
            return &kernel;
        }
    }
    return nullptr;
}

static ggml::hrx::Dispatch make_add_dispatch(const ggml::hrx::KernelDefinition & definition, int64_t element_count) {
    ggml::hrx::Dispatch dispatch;
    dispatch.kernel.kernel_id = definition.id;
    dispatch.kernel.integer_parameters.emplace("element_count", element_count);
    dispatch.bindings.reserve(definition.bindings.size());
    for (size_t i = 0; i < definition.bindings.size(); ++i) {
        dispatch.bindings.push_back({
            ggml::hrx::ValueId(static_cast<int32_t>(i + 1)),
            0,
            static_cast<size_t>(element_count) * sizeof(float),
        });
    }
    return dispatch;
}

static ggml::hrx::LoomKernelCompileRequest make_compile_request(
    const ggml::hrx::KernelDefinition &        definition,
    const std::map<std::string, int64_t> &     workload,
    const std::map<std::string, std::string> & compile_config = {}) {
    ggml::hrx::LoomKernelCompileRequest request;
    REQUIRE(!definition.compile_recipe.primary_sources.empty());

    const ggml::hrx::KernelSourceRef & primary_source = definition.compile_recipe.primary_sources.front();
    REQUIRE(primary_source.contents != nullptr);
    request.source_data          = primary_source.contents->source.data;
    request.source_size          = primary_source.contents->source.length;
    request.source_format        = to_jit_source_format(primary_source.contents->source.format);
    request.source_identifier    = primary_source.path != nullptr ? primary_source.path : "";
    request.symbol               = definition.symbol != nullptr ? definition.symbol : "";
    request.launch_config_symbol = definition.name != nullptr ? definition.name : "";

    request.dependencies.reserve(definition.compile_recipe.library_sources.size());
    for (const ggml::hrx::KernelSourceRef & dependency_ref : definition.compile_recipe.library_sources) {
        REQUIRE(dependency_ref.contents != nullptr);
        request.dependencies.push_back({
            dependency_ref.contents->source.data,
            dependency_ref.contents->source.length,
            to_jit_source_format(dependency_ref.contents->source.format),
            dependency_ref.path,
        });
    }

    std::map<std::string, std::string> merged_config;
    for (const ggml::hrx::KernelCompileConfig & config : definition.compile_config) {
        merged_config[config.key != nullptr ? config.key : ""] = config.value != nullptr ? config.value : "";
    }
    for (const auto & item : compile_config) {
        merged_config[item.first] = item.second;
    }
    request.config_storage.reserve(merged_config.size());
    for (const auto & item : merged_config) {
        request.config_storage.push_back(item);
    }

    request.workload.reserve(definition.workload_parameters.size());
    for (const ggml::hrx::KernelScalarDefinition & parameter : definition.workload_parameters) {
        REQUIRE(parameter.type != nullptr);
        REQUIRE(std::strcmp(parameter.type, "index") == 0);
        const auto found = workload.find(parameter.name != nullptr ? parameter.name : "");
        REQUIRE(found != workload.end());
        request.workload.push_back(found->second);
    }
    return request;
}

static ggml::hrx::LoomCompiledKernelRef compile_kernel(ggml::hrx::LoomJit &                       jit,
                                                       const std::string &                        key,
                                                       const ggml::hrx::KernelDefinition &        definition,
                                                       const std::map<std::string, int64_t> &     workload,
                                                       const std::map<std::string, std::string> & compile_config = {}) {
    return jit.compile(key, make_compile_request(definition, workload, compile_config));
}

static bool resolve_with_timeout(const ggml::hrx::LoomCompiledKernelRef & ref, std::chrono::seconds timeout) {
    std::atomic<bool> done    = false;
    std::atomic<bool> success = false;
    std::thread       resolver([&] {
        success.store(ref->resolve(), std::memory_order_release);
        done.store(true, std::memory_order_release);
    });

    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!done.load(std::memory_order_acquire)) {
        if (std::chrono::steady_clock::now() >= deadline) {
            std::fprintf(stderr, "timed out waiting for async Loom compile: %s\n", ref->key().c_str());
            std::abort();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    resolver.join();
    return success.load(std::memory_order_acquire);
}

static void require_compiled_kernel(const ggml::hrx::LoomCompiledKernelRef & ref) {
    REQUIRE(ref != nullptr);
    REQUIRE(resolve_with_timeout(ref, std::chrono::seconds(120)));
    ggml_hrx_loom_jit_compile_result compiled = ref->take_result();
    REQUIRE(compiled.hsaco_data != nullptr);
    REQUIRE(compiled.hsaco_size > 0);
    REQUIRE(compiled.launch_config.workgroup_count[0] > 0);
    REQUIRE(compiled.launch_config.workgroup_size[0] > 0);
    compiled.reset();
}

static int64_t elapsed_us(std::chrono::steady_clock::time_point begin, std::chrono::steady_clock::time_point end) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
}

static void run_cache_materialize_case(HrxTestDevice &                     device,
                                       ggml::hrx::LoomJitMode              mode,
                                       const ggml::hrx::KernelDefinition & add) {
    ggml::hrx::KernelExecutablePrepareContext context = {};
    context.device                                    = device.device;
    context.target                                    = device.architecture.c_str();

    ggml::hrx::Dispatch              dispatch = make_add_dispatch(add, 64);
    std::vector<uint8_t>             constants;
    ggml::hrx::KernelExecutableCache cache(mode);

    ggml::hrx::KernelExecutableRef ref = cache.get_or_compile(context, add, dispatch, constants);
    REQUIRE(ref.valid());

    std::shared_ptr<ggml::hrx::KernelExecutable> executable = cache.materialize(context, ref, constants);
    REQUIRE(executable != nullptr);
    REQUIRE(executable->executable != nullptr);
    REQUIRE(executable->export_info.binding_count == dispatch.bindings.size());
    REQUIRE(executable->export_info.constant_byte_length == constants.size());
    REQUIRE(executable->launch.workgroup_count[0] > 0);
    REQUIRE(executable->launch.workgroup_size[0] > 0);

    std::shared_ptr<ggml::hrx::KernelExecutable> loaded_hit = cache.materialize(context, ref, constants);
    REQUIRE(loaded_hit == executable);

    std::vector<uint8_t>                         constants_for_prepare;
    std::shared_ptr<ggml::hrx::KernelExecutable> prepared =
        cache.prepare(context, add, dispatch, constants_for_prepare);
    REQUIRE(prepared == executable);

    const char * mode_name = mode == ggml::hrx::LoomJitMode::Async ? "async" : "sync";
    std::printf("%s KernelExecutableCache materialized ggml_add_f32 for %s\n", mode_name, device.architecture.c_str());
}

static void run_targeted_export_materialize_case(HrxTestDevice & device) {
    const ggml::hrx::KernelDefinition * definition =
        find_targeted_kernel("ggml_linear_q6k_q8_1_x4", device.architecture.c_str());
    if (definition == nullptr) {
        return;
    }

    ggml::hrx::KernelExecutablePrepareContext context = {};
    context.device                                    = device.device;
    context.target                                    = device.architecture.c_str();

    ggml::hrx::Dispatch dispatch;
    dispatch.kernel.kernel_id = definition->id;
    dispatch.kernel.integer_parameters.emplace("token_count", 1);
    dispatch.kernel.integer_parameters.emplace("input_size", 2048);
    dispatch.kernel.integer_parameters.emplace("output_size", 151936);
    dispatch.kernel.compile_parameters.emplace("ggml.linear_q6k_q8_1_x4.token_capacity", "1");
    dispatch.kernel.compile_parameters.emplace("ggml.linear_q6k_q8_1_x4.output_capacity", "151936");
    dispatch.bindings.resize(3);

    ggml::hrx::KernelExecutableCache             cache(ggml::hrx::LoomJitMode::Sync);
    std::vector<uint8_t>                         constants;
    std::shared_ptr<ggml::hrx::KernelExecutable> executable = cache.prepare(context, *definition, dispatch, constants);
    REQUIRE(executable != nullptr);
    REQUIRE(executable->executable != nullptr);
    REQUIRE(executable->launch.workgroup_count[0] == 151936);
    std::printf("KernelExecutableCache materialized targeted %s as export %s for %s\n", definition->symbol,
                definition->name, device.architecture.c_str());
}

}  // namespace

int main() {
    static constexpr const char * kTarget = "gfx1100";

    const ggml::hrx::KernelDefinition & add             = find_kernel("ggml_add_f32");
    const ggml::hrx::KernelDefinition & gather_add      = find_kernel("ggml_gather_add_f32");
    const ggml::hrx::KernelDefinition & rmsnorm         = find_kernel("qwen3_moe_rmsnorm_f32");
    const ggml::hrx::KernelDefinition & router_top8     = find_kernel("qwen3_moe_router_top8_f32");
    const ggml::hrx::KernelDefinition & expert_table    = find_kernel("qwen3_moe_build_expert_table");
    const ggml::hrx::KernelDefinition & partition_table = find_kernel("qwen3_moe_build_expert_partition_table");

    std::string                         sync_error;
    std::unique_ptr<ggml::hrx::LoomJit> sync_jit =
        ggml::hrx::create_loom_jit(kTarget, ggml::hrx::LoomJitMode::Sync, sync_error);
    REQUIRE(sync_jit != nullptr);
    REQUIRE(!sync_jit->async_enabled());

    const auto                       sync_begin = std::chrono::steady_clock::now();
    ggml::hrx::LoomCompiledKernelRef sync_ref   = compile_kernel(*sync_jit, "sync-add-64", add,
                                                                 {
                                                                   { "element_count", 64 }
    });
    require_compiled_kernel(sync_ref);
    const auto    sync_end        = std::chrono::steady_clock::now();
    const int64_t sync_compile_us = elapsed_us(sync_begin, sync_end);
    std::printf("sync Loom compile completed in %ld us\n", static_cast<long>(sync_compile_us));

    std::string                         async_error;
    std::unique_ptr<ggml::hrx::LoomJit> async_jit =
        ggml::hrx::create_loom_jit(kTarget, ggml::hrx::LoomJitMode::Async, async_error);
    REQUIRE(async_jit != nullptr);
    REQUIRE(async_jit->async_enabled());

    std::vector<ggml::hrx::LoomCompiledKernelRef> refs;
    refs.reserve(9);
    const auto enqueue_begin = std::chrono::steady_clock::now();
    refs.push_back(compile_kernel(*async_jit, "async-add-64", add,
                                  {
                                      { "element_count", 64 }
    }));
    refs.push_back(compile_kernel(*async_jit, "async-add-128", add,
                                  {
                                      { "element_count", 128 }
    }));
    refs.push_back(compile_kernel(*async_jit, "async-add-256", add,
                                  {
                                      { "element_count", 256 }
    }));
    refs.push_back(compile_kernel(*async_jit, "async-add-512", add,
                                  {
                                      { "element_count", 512 }
    }));
    refs.push_back(compile_kernel(*async_jit, "async-rmsnorm-1", rmsnorm,
                                  {
                                      { "token_count", 1 }
    },
                                  {
                                      { "qwen3_moe.model.hidden_size", "2048" },
                                      { "qwen3_moe.model.rms_epsilon", "0.000001" },
                                      { "qwen3_moe.workload.token_capacity", "1" },
                                      { "ggml.quantize_q8_1_x4.group_capacity", "256" },
                                  }));
    refs.push_back(compile_kernel(*async_jit, "async-router-top8-1", router_top8,
                                  {
                                      { "token_count",     1 },
                                      { "route_id_stride", 8 },
    },
                                  {
                                      { "qwen3_moe.router.expert_count", "128" },
                                      { "qwen3_moe.router.route_count", "8" },
                                      { "qwen3_moe.workload.token_capacity", "1" },
                                  }));
    refs.push_back(compile_kernel(*async_jit, "async-expert-table-1", expert_table,
                                  {
                                      { "token_count",  1   },
                                      { "route_count",  8   },
                                      { "route_stride", 8   },
                                      { "expert_count", 128 },
    },
                                  {
                                      { "qwen3_moe.routed_gate_up.expert_count", "128" },
                                      { "qwen3_moe.routed_gate_up.route_count", "8" },
                                      { "qwen3_moe.workload.token_capacity", "1" },
                                  }));
    refs.push_back(compile_kernel(*async_jit, "async-partition-table-1", partition_table,
                                  {
                                      { "token_count",  1   },
                                      { "route_count",  8   },
                                      { "expert_count", 128 },
    },
                                  {
                                      { "qwen3_moe.routed_gate_up.expert_count", "128" },
                                      { "qwen3_moe.routed_gate_up.route_count", "8" },
                                      { "qwen3_moe.workload.token_capacity", "1" },
                                  }));
    // Exercise gather-add coverage in the async JIT path.
    refs.push_back(compile_kernel(*async_jit, "async-gather-add-2-to-1", gather_add,
                                  {
                                      { "source_token_count", 2    },
                                      { "output_token_count", 1    },
                                      { "hidden_size",        2048 },
    }));
    const auto    enqueue_end = std::chrono::steady_clock::now();
    const int64_t enqueue_us  = elapsed_us(enqueue_begin, enqueue_end);
    std::printf("async Loom enqueue completed in %ld us\n", static_cast<long>(enqueue_us));

    const int64_t max_expected_enqueue_us = std::max<int64_t>(100000, sync_compile_us / 2);
    REQUIRE(enqueue_us < max_expected_enqueue_us);

    for (const ggml::hrx::LoomCompiledKernelRef & ref : refs) {
        require_compiled_kernel(ref);
    }

    std::printf("async Loom JIT compiled %zu kernels\n", refs.size());

    HrxTestDevice device;
    if (device.open()) {
        run_cache_materialize_case(device, ggml::hrx::LoomJitMode::Sync, add);
        run_cache_materialize_case(device, ggml::hrx::LoomJitMode::Async, add);
        run_targeted_export_materialize_case(device);
    } else {
        std::printf("skipping KernelExecutableCache materialization checks: no HRX device available\n");
    }

    return 0;
}
