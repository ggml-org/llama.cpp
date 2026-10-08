#include "dispatch/dispatch.h"
#include "hrx-interop-utils.h"
#include "kernel-corpus/kernel-corpus.h"
#include "runtime/kernel-executable-cache.h"
#include "runtime/loom-kernel-jit.h"
#include "testing_suite.h"

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

static std::map<std::string, std::string> binary_f32_exact_config(const char * op, int64_t element_count) {
    const std::string count = std::to_string(element_count);
    return {
        { "ggml.binary_f32.op", op },
        { "ggml.binary_f32.ne0", count },
        { "ggml.binary_f32.ne1", "1" },
        { "ggml.binary_f32.ne2", "1" },
        { "ggml.binary_f32.src0_stride1", count },
        { "ggml.binary_f32.src0_stride2", count },
        { "ggml.binary_f32.src0_stride3", count },
        { "ggml.binary_f32.src1_stride1", count },
        { "ggml.binary_f32.src1_stride2", count },
        { "ggml.binary_f32.src1_stride3", count },
        { "ggml.binary_f32.src0_span", count },
        { "ggml.binary_f32.src1_span", count },
    };
}

static ggml::hrx::Dispatch make_binary_add_dispatch(const ggml::hrx::KernelDefinition & definition,
                                                    int64_t                             element_count) {
    ggml::hrx::Dispatch dispatch;
    dispatch.kernel.kernel_id = definition.id;
    dispatch.kernel.integer_parameters.emplace("element_count", element_count);
    dispatch.kernel.compile_parameters = binary_f32_exact_config("0", element_count);
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

static std::map<std::string, int64_t> binary_f32_exact_workload(int64_t element_count) {
    return {
        { "element_count", element_count },
    };
}

static std::map<std::string, int64_t> binary_bc_f32_workload() {
    return {
        { "element_count",      48 },
        { "ne0",                16 },
        { "ne1",                3  },
        { "ne2",                1  },
        { "ne3",                1  },
        { "src0_element_count", 48 },
        { "src1_element_count", 16 },
    };
}

static std::map<std::string, std::string> binary_bc_f32_config(const char * op) {
    return {
        { "ggml.binary_bc_f32.op",                  op  },
        { "ggml.binary_bc_f32.src0_broadcast_dim0", "0" },
        { "ggml.binary_bc_f32.src0_broadcast_dim1", "0" },
        { "ggml.binary_bc_f32.src0_broadcast_dim2", "0" },
        { "ggml.binary_bc_f32.src0_broadcast_dim3", "0" },
        { "ggml.binary_bc_f32.src1_broadcast_dim0", "0" },
        { "ggml.binary_bc_f32.src1_broadcast_dim1", "1" },
        { "ggml.binary_bc_f32.src1_broadcast_dim2", "0" },
        { "ggml.binary_bc_f32.src1_broadcast_dim3", "0" },
    };
}

static std::map<std::string, int64_t> token_count_workload(int64_t token_count) {
    return {
        { "token_count", token_count },
    };
}

static std::map<std::string, std::string> moe_routing_config() {
    return {
        { "ggml.moe_routing.route_count", "8" },
        { "ggml.moe_routing.expert_count", "128" },
        { "ggml.moe_routing.descriptor_expert_mask", "127" },
        { "ggml.moe_routing.descriptor_partition_shift", "7" },
        { "ggml.moe_routing.descriptor_row_count_shift", "13" },
        { "ggml.moe_routing.partition_workgroup_size", "128" },
        { "ggml.workload.token_capacity", "1" },
    };
}

static std::map<std::string, int64_t> flash_attention_workload(int64_t query_token_count,
                                                               int64_t key_value_token_count) {
    return {
        { "query_token_count",     query_token_count     },
        { "key_value_token_count", key_value_token_count },
    };
}

static std::map<std::string, int64_t> flash_attention_decode_workload(int64_t key_value_token_count) {
    return {
        { "key_value_token_count", key_value_token_count },
    };
}

static std::map<std::string, std::string> flash_attention_config(const char * query_head_count,
                                                                 const char * key_value_head_count,
                                                                 const char * head_size,
                                                                 const char * attention_scale) {
    return {
        { "ggml.flash_attention.query_head_count",     query_head_count     },
        { "ggml.flash_attention.key_value_head_count", key_value_head_count },
        { "ggml.flash_attention.qk_head_size",         head_size            },
        { "ggml.flash_attention.value_head_size",      head_size            },
        { "ggml.flash_attention.attention_scale",      attention_scale      },
        { "ggml.flash_attention.apply_gate",           "0"                  },
        { "ggml.flash_attention.gate_stride_head",     "1"                  },
        { "ggml.flash_attention.gate_stride_token",    "1"                  },
    };
}

static std::map<std::string, std::string> flash_attention_decode_base_config(const char * query_head_count,
                                                                             const char * key_value_head_count,
                                                                             const char * head_size,
                                                                             const char * attention_scale) {
    return {
        { "ggml.flash_attention.query_head_count",     query_head_count     },
        { "ggml.flash_attention.key_value_head_count", key_value_head_count },
        { "ggml.flash_attention.qk_head_size",         head_size            },
        { "ggml.flash_attention.value_head_size",      head_size            },
        { "ggml.flash_attention.attention_scale",      attention_scale      },
        { "ggml.flash_attention.apply_gate",           "0"                  },
        { "ggml.flash_attention.gate_stride_head",     head_size            },
        { "ggml.flash_attention.gate_stride_token",    head_size            },
    };
}

static std::map<std::string, std::string> flash_attention_decode_config(const char * query_head_count,
                                                                        const char * key_value_head_count,
                                                                        const char * head_size,
                                                                        const char * attention_scale,
                                                                        const char * key_value_token_capacity) {
    std::map<std::string, std::string> config =
        flash_attention_decode_base_config(query_head_count, key_value_head_count, head_size, attention_scale);
    config["ggml.flash_attention.decode.key_value_token_capacity"] = key_value_token_capacity;
    return config;
}

static std::map<std::string, std::string> mul_mat_id_f16_f16_config(const char * weight_format) {
    return {
        { "ggml.mul_mat_id_f16_f16.input_size",    "768"         },
        { "ggml.mul_mat_id_f16_f16.route_count",   "8"           },
        { "ggml.mul_mat_id_f16_f16.expert_count",  "128"         },
        { "ggml.mul_mat_id_f16_f16.output_size",   "2048"        },
        { "ggml.mul_mat_id_f16_f16.weight_format", weight_format },
        { "ggml.workload.token_capacity",          "4"           },
    };
}

static std::map<std::string, std::string> mul_mat_id_swiglu_f16_f16_config(const char * gate_format,
                                                                           const char * up_format) {
    return {
        { "ggml.mul_mat_id_swiglu_f16_f16.input_size",                 "2048"      },
        { "ggml.mul_mat_id_swiglu_f16_f16.output_size",                "768"       },
        { "ggml.mul_mat_id_swiglu_f16_f16.expert_count",               "128"       },
        { "ggml.mul_mat_id_swiglu_f16_f16.route_count",                "8"         },
        { "ggml.mul_mat_id_swiglu_f16_f16.gate_weight_format",         gate_format },
        { "ggml.mul_mat_id_swiglu_f16_f16.up_weight_format",           up_format   },
        { "ggml.mul_mat_id_swiglu_f16_f16.descriptor_expert_mask",     "127"       },
        { "ggml.mul_mat_id_swiglu_f16_f16.descriptor_partition_shift", "7"         },
        { "ggml.mul_mat_id_swiglu_f16_f16.descriptor_row_count_shift", "13"        },
    };
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
                                       const ggml::hrx::KernelDefinition & binary) {
    ggml::hrx::KernelExecutablePrepareContext context = {};
    context.device                                    = device.device;
    context.target                                    = device.architecture.c_str();

    ggml::hrx::Dispatch              dispatch = make_binary_add_dispatch(binary, 64);
    std::vector<uint8_t>             constants;
    ggml_hrx_loom_jit_launch_config  launch;
    ggml::hrx::KernelExecutableCache cache(mode);

    ggml::hrx::KernelExecutableRef ref = cache.get_or_compile(context, binary, dispatch);
    REQUIRE(ref.valid());

    std::shared_ptr<ggml::hrx::KernelExecutable> executable =
        cache.materialize(context, ref, dispatch.kernel, constants, launch);
    REQUIRE(executable != nullptr);
    REQUIRE(executable->executable != nullptr);
    REQUIRE(executable->export_info.binding_count == dispatch.bindings.size());
    REQUIRE(executable->export_info.constant_byte_length == constants.size());
    REQUIRE(launch.workgroup_count[0] > 0);
    REQUIRE(launch.workgroup_size[0] > 0);

    std::shared_ptr<ggml::hrx::KernelExecutable> loaded_hit =
        cache.materialize(context, ref, dispatch.kernel, constants, launch);
    REQUIRE(loaded_hit == executable);

    std::vector<uint8_t>                         constants_for_prepare;
    ggml_hrx_loom_jit_launch_config              launch_for_prepare;
    std::shared_ptr<ggml::hrx::KernelExecutable> prepared =
        cache.prepare(context, binary, dispatch, constants_for_prepare, launch_for_prepare);
    REQUIRE(prepared == executable);

    const char * mode_name = mode == ggml::hrx::LoomJitMode::Async ? "async" : "sync";
    std::printf("%s KernelExecutableCache materialized ggml_binary_f32 for %s\n", mode_name,
                device.architecture.c_str());
}

static void run_dynamic_cache_materialize_case(HrxTestDevice &                     device,
                                               ggml::hrx::LoomJitMode              mode,
                                               const ggml::hrx::KernelDefinition & binary) {
    ggml::hrx::KernelExecutablePrepareContext context = {};
    context.device                                    = device.device;
    context.target                                    = device.architecture.c_str();

    ggml::hrx::Dispatch first  = make_binary_add_dispatch(binary, 64);
    ggml::hrx::Dispatch second = make_binary_add_dispatch(binary, 257);
    first.kernel.compile_parameters = binary_f32_exact_config("0", 512);
    second.kernel.compile_parameters = first.kernel.compile_parameters;
    first.kernel.workload_specialization  = ggml::hrx::WorkloadSpecialization::Dynamic;
    second.kernel.workload_specialization = ggml::hrx::WorkloadSpecialization::Dynamic;

    ggml::hrx::KernelExecutableCache cache(mode);
    ggml::hrx::KernelExecutableRef   first_ref  = cache.get_or_compile(context, binary, first);
    ggml::hrx::KernelExecutableRef   second_ref = cache.get_or_compile(context, binary, second);
    REQUIRE(first_ref.valid());
    REQUIRE(second_ref.valid());
    REQUIRE(first_ref.entry == second_ref.entry);

    std::vector<uint8_t>            first_constants;
    std::vector<uint8_t>            second_constants;
    ggml_hrx_loom_jit_launch_config first_launch;
    ggml_hrx_loom_jit_launch_config second_launch;
    std::shared_ptr<ggml::hrx::KernelExecutable> first_executable =
        cache.materialize(context, first_ref, first.kernel, first_constants, first_launch);
    std::shared_ptr<ggml::hrx::KernelExecutable> second_executable =
        cache.materialize(context, second_ref, second.kernel, second_constants, second_launch);
    REQUIRE(first_executable != nullptr);
    REQUIRE(second_executable == first_executable);
    REQUIRE(first_launch.workgroup_count[0] == 1);
    REQUIRE(second_launch.workgroup_count[0] == 2);
    REQUIRE(first_constants.size() == second_constants.size());
    REQUIRE(first_constants != second_constants);

    ggml::hrx::Dispatch exact_second = second;
    exact_second.kernel.workload_specialization = ggml::hrx::WorkloadSpecialization::Exact;
    ggml::hrx::KernelExecutableRef exact_ref = cache.get_or_compile(context, binary, exact_second);
    REQUIRE(exact_ref.valid());
    REQUIRE(exact_ref.entry != first_ref.entry);
}

static void run_dynamic_export_reuse_case(
    HrxTestDevice &                           device,
    const char *                              kernel_name,
    const std::map<std::string, int64_t> &    first_workload,
    const std::map<std::string, int64_t> &    second_workload,
    const std::map<std::string, std::string> & compile_config = {}) {
    const ggml::hrx::KernelDefinition & definition = find_kernel(kernel_name);
    ggml::hrx::KernelExecutablePrepareContext context = {};
    context.device                                    = device.device;
    context.target                                    = device.architecture.c_str();

    auto make_dispatch = [&](const std::map<std::string, int64_t> & workload) {
        ggml::hrx::Dispatch dispatch;
        dispatch.kernel.kernel_id               = definition.id;
        dispatch.kernel.integer_parameters      = workload;
        dispatch.kernel.compile_parameters      = compile_config;
        dispatch.kernel.workload_specialization = ggml::hrx::WorkloadSpecialization::Dynamic;
        dispatch.bindings.resize(definition.bindings.size());
        return dispatch;
    };

    ggml::hrx::Dispatch first  = make_dispatch(first_workload);
    ggml::hrx::Dispatch second = make_dispatch(second_workload);
    ggml::hrx::KernelExecutableCache cache(ggml::hrx::LoomJitMode::Sync);
    ggml::hrx::KernelExecutableRef first_ref  = cache.get_or_compile(context, definition, first);
    ggml::hrx::KernelExecutableRef second_ref = cache.get_or_compile(context, definition, second);
    REQUIRE(first_ref.valid());
    REQUIRE(second_ref.valid());
    REQUIRE(first_ref.entry == second_ref.entry);

    std::vector<uint8_t>            first_constants;
    std::vector<uint8_t>            second_constants;
    ggml_hrx_loom_jit_launch_config first_launch;
    ggml_hrx_loom_jit_launch_config second_launch;
    std::shared_ptr<ggml::hrx::KernelExecutable> first_executable =
        cache.materialize(context, first_ref, first.kernel, first_constants, first_launch);
    std::shared_ptr<ggml::hrx::KernelExecutable> second_executable =
        cache.materialize(context, second_ref, second.kernel, second_constants, second_launch);
    REQUIRE(first_executable != nullptr);
    REQUIRE(second_executable == first_executable);
    REQUIRE(first_launch.workgroup_count[0] > 0);
    REQUIRE(second_launch.workgroup_count[0] > 0);
    REQUIRE(first_constants.size() == second_constants.size());
    REQUIRE(first_constants != second_constants);
    std::printf("dynamic executable reuse materialized %s for two workloads\n", kernel_name);
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
    ggml_hrx_loom_jit_launch_config              launch;
    std::shared_ptr<ggml::hrx::KernelExecutable> executable =
        cache.prepare(context, *definition, dispatch, constants, launch);
    REQUIRE(executable != nullptr);
    REQUIRE(executable->executable != nullptr);
    REQUIRE(launch.workgroup_count[0] == 151936);
    std::printf("KernelExecutableCache materialized targeted %s as export %s for %s\n", definition->symbol,
                definition->name, device.architecture.c_str());
}

}  // namespace

static void run_loom_jit_compile_checks() {
    static constexpr const char * kTarget = "gfx1100";

    const ggml::hrx::KernelDefinition & binary          = find_kernel("ggml_binary_f32");
    const ggml::hrx::KernelDefinition & binary_bc       = find_kernel("ggml_binary_bc_f32");
    const ggml::hrx::KernelDefinition & gather_add      = find_kernel("ggml_gather_add_f32");
    const ggml::hrx::KernelDefinition & rmsnorm         = find_kernel("ggml_rmsnorm_f32");
    const ggml::hrx::KernelDefinition & router_top8     = find_kernel("qwen3_moe_router_top8_f32");
    const ggml::hrx::KernelDefinition & expert_table    = find_kernel("ggml_moe_build_expert_table");
    const ggml::hrx::KernelDefinition & partition_table = find_kernel("ggml_moe_build_expert_partition_table");
    const ggml::hrx::KernelDefinition & mul_mat_id_f16  = find_kernel("ggml_mul_mat_id_f16_f16_wmma");
    const ggml::hrx::KernelDefinition & swiglu_f16      = find_kernel("ggml_mul_mat_id_swiglu_f16_f16_wmma");
    const ggml::hrx::KernelDefinition & flash_prefill   = find_kernel("ggml_flash_attention_f32_f16_wmma");
    const ggml::hrx::KernelDefinition & decode_mul_mat  = find_kernel("ggml_mul_mat_f32_f32_decode_wave64");
    const ggml::hrx::KernelDefinition & flash_decode =
        find_kernel("ggml_flash_attention_decode_split_f32_f16_wmma_next_q8");

    std::string                         sync_error;
    std::unique_ptr<ggml::hrx::LoomJit> sync_jit =
        ggml::hrx::create_loom_jit(kTarget, ggml::hrx::LoomJitMode::Sync, sync_error);
    REQUIRE(sync_jit != nullptr);
    REQUIRE(!sync_jit->async_enabled());

    const auto                       sync_begin = std::chrono::steady_clock::now();
    ggml::hrx::LoomCompiledKernelRef sync_ref   = compile_kernel(
        *sync_jit, "sync-binary-add-64", binary, binary_f32_exact_workload(64), binary_f32_exact_config("0", 64));
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
    refs.reserve(31);
    const auto enqueue_begin = std::chrono::steady_clock::now();
    refs.push_back(compile_kernel(*async_jit, "async-binary-add-64", binary, binary_f32_exact_workload(64),
                                  binary_f32_exact_config("0", 64)));
    refs.push_back(compile_kernel(*async_jit, "async-binary-add-128", binary, binary_f32_exact_workload(128),
                                  binary_f32_exact_config("0", 128)));
    refs.push_back(compile_kernel(*async_jit, "async-binary-add-256", binary, binary_f32_exact_workload(256),
                                  binary_f32_exact_config("0", 256)));
    refs.push_back(compile_kernel(*async_jit, "async-binary-add-512", binary, binary_f32_exact_workload(512),
                                  binary_f32_exact_config("0", 512)));
    refs.push_back(compile_kernel(*async_jit, "async-binary-geglu-256", binary, binary_f32_exact_workload(256),
                                  binary_f32_exact_config("5", 256)));
    refs.push_back(compile_kernel(*async_jit, "async-binary-reglu-256", binary, binary_f32_exact_workload(256),
                                  binary_f32_exact_config("6", 256)));
    refs.push_back(compile_kernel(*async_jit, "async-binary-geglu-erf-256", binary, binary_f32_exact_workload(256),
                                  binary_f32_exact_config("7", 256)));
    refs.push_back(compile_kernel(*async_jit, "async-binary-geglu-quick-256", binary, binary_f32_exact_workload(256),
                                  binary_f32_exact_config("8", 256)));
    refs.push_back(compile_kernel(*async_jit, "async-binary-bc-mul-48", binary_bc, binary_bc_f32_workload(),
                                  binary_bc_f32_config("2")));
    refs.push_back(compile_kernel(*async_jit, "async-rmsnorm-1", rmsnorm,
                                  {
                                      { "token_count", 1 }
    },
                                  {
                                      { "ggml.rmsnorm_f32.hidden_size", "2048" },
                                      { "ggml.rmsnorm_f32.input_stride", "2048" },
                                      { "ggml.rmsnorm_f32.rms_epsilon", "0.000001" },
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
                                  moe_routing_config()));
    refs.push_back(compile_kernel(*async_jit, "async-partition-table-1", partition_table,
                                  {
                                      { "token_count",  1   },
                                      { "route_count",  8   },
                                      { "expert_count", 128 },
    },
                                  moe_routing_config()));
    // Exercise gather-add coverage in the async JIT path.
    refs.push_back(compile_kernel(*async_jit, "async-gather-add-2-to-1", gather_add,
                                  {
                                      { "source_token_count", 2    },
                                      { "output_token_count", 1    },
                                      { "hidden_size",        2048 },
    },
                                  {
                                      { "ggml.gather_add_f32.binary_op", "3" },
    }));
    refs.push_back(compile_kernel(*async_jit, "async-mul-mat-id-f16-q4", mul_mat_id_f16, token_count_workload(4),
                                  mul_mat_id_f16_f16_config("4")));
    refs.push_back(compile_kernel(*async_jit, "async-mul-mat-id-f16-q6", mul_mat_id_f16, token_count_workload(4),
                                  mul_mat_id_f16_f16_config("6")));
    refs.push_back(compile_kernel(*async_jit, "async-mul-mat-id-swiglu-f16-q4k", swiglu_f16, token_count_workload(4),
                                  mul_mat_id_swiglu_f16_f16_config("4", "4")));
    refs.push_back(compile_kernel(*async_jit, "async-mul-mat-id-swiglu-f16-q6k", swiglu_f16, token_count_workload(4),
                                  mul_mat_id_swiglu_f16_f16_config("6", "6")));
    refs.push_back(compile_kernel(*async_jit, "async-mul-mat-id-swiglu-f16-iq4-xs", swiglu_f16, token_count_workload(4),
                                  mul_mat_id_swiglu_f16_f16_config("23", "23")));
    refs.push_back(compile_kernel(*async_jit, "async-mul-mat-id-swiglu-f16-q3k", swiglu_f16, token_count_workload(4),
                                  mul_mat_id_swiglu_f16_f16_config("11", "11")));
    refs.push_back(compile_kernel(*async_jit, "async-mul-mat-id-swiglu-f16-iq4-nl", swiglu_f16, token_count_workload(4),
                                  mul_mat_id_swiglu_f16_f16_config("20", "20")));
    refs.push_back(compile_kernel(*async_jit, "async-mul-mat-id-swiglu-f16-q8-0", swiglu_f16, token_count_workload(4),
                                  mul_mat_id_swiglu_f16_f16_config("80", "80")));
    refs.push_back(compile_kernel(*async_jit, "async-mul-mat-id-swiglu-f16-q8-1", swiglu_f16, token_count_workload(4),
                                  mul_mat_id_swiglu_f16_f16_config("81", "81")));
    refs.push_back(compile_kernel(*async_jit, "async-mul-mat-id-swiglu-f16-f16", swiglu_f16, token_count_workload(4),
                                  mul_mat_id_swiglu_f16_f16_config("16", "16")));
    refs.push_back(compile_kernel(*async_jit, "async-mul-mat-decode-q8-0-gemma-vocab", decode_mul_mat,
                                  {
                                      { "token_count", 1      },
                                      { "input_size",  3840   },
                                      { "output_size", 262208 },
    },
                                  {
                                      { "ggml.mul_mat_f32_f32_decode.token_capacity", "1" },
                                      { "ggml.mul_mat_f32_f32_decode.output_capacity", "262208" },
                                      { "ggml.mul_mat_f32_f32_decode.weight_format", "80" },
                                  }));
    refs.push_back(compile_kernel(*async_jit, "async-flash-prefill-128", flash_prefill,
                                  flash_attention_workload(128, 128),
                                  flash_attention_config("32", "4", "128", "0.0883883461")));
    refs.push_back(compile_kernel(*async_jit, "async-flash-prefill-256", flash_prefill,
                                  flash_attention_workload(128, 128),
                                  flash_attention_config("32", "4", "256", "0.0625")));
    refs.push_back(compile_kernel(*async_jit, "async-flash-decode-128", flash_decode,
                                  flash_attention_decode_workload(64),
                                  flash_attention_decode_config("32", "4", "128", "0.0883883461", "64")));
    const auto    enqueue_end = std::chrono::steady_clock::now();
    const int64_t enqueue_us  = elapsed_us(enqueue_begin, enqueue_end);
    std::printf("async Loom enqueue completed in %ld us\n", static_cast<long>(enqueue_us));

    const int64_t max_expected_enqueue_us = std::max<int64_t>(100000, sync_compile_us / 2);
    REQUIRE(enqueue_us < max_expected_enqueue_us);

    for (const ggml::hrx::LoomCompiledKernelRef & ref : refs) {
        require_compiled_kernel(ref);
    }

    std::printf("async Loom JIT compiled %zu kernels\n", refs.size());

    const auto & q6_prefill = find_kernel("ggml_mul_mat_q6_k_f16_wmma_prefill_wave32");

    const struct {
        int64_t tokens;
        int64_t inputs;
        int64_t outputs;
        int64_t accumulate;
        int64_t workgroup_size;
        int64_t token_tiles;
        int64_t output_tiles;
    } q6_launch_cases[] = {
        { 128,   256,  4096, 0,  256, 1, 32 },
        { 256,   256,  8192, 0,  512, 1, 64 },
        { 512,   256,  1024, 0,  512, 2,  8 },
        { 512,   256,  3840, 0,  512, 2, 30 },
        { 512,   256,  4096, 0, 1024, 1, 32 },
        { 512,   256,  4224, 0, 1024, 1, 33 },
        { 512,   256,  4096, 1,  256, 4, 32 },
        {1024,   256,  2048, 0, 1024, 2, 16 },
        { 512, 17408,  5120, 0,  512, 1, 80 },
        { 512,  5120, 10240, 0, 1024, 1, 80 },
        { 512,  5120,  1024, 0,  512, 2,  8 },
        { 512, 17408,  4096, 0,  512, 1, 64 },
        { 512,  5120,  1984, 0,  256, 4, 16 },
        { 512,  5120,  2048, 0,  512, 1, 32 },
        {1024,  5120,  1024, 0,  512, 2, 16 },
        {2048,  5120,  1024, 0,  512, 4, 16 },
    };

    for (const auto & test : q6_launch_cases) {
        const std::string key = "q6-prefill-" + std::to_string(test.tokens) + "-" + std::to_string(test.inputs) + "-" +
                                std::to_string(test.outputs) + "-" + std::to_string(test.accumulate);
        auto ref =
            compile_kernel(*sync_jit, key, q6_prefill, token_count_workload(test.tokens),
                           {
            { "ggml.mul_mat_q6_k_packed.input_size",          std::to_string(test.inputs) },
            { "ggml.mul_mat_q6_k_packed.output_size",         std::to_string(test.outputs) },
            { "ggml.mul_mat_q6_k_packed.output_accumulation", std::to_string(test.accumulate) },
            { "ggml.mul_mat_q6_k_packed.weight_offset",       "0" },
            { "ggml.mul_mat_q6_k_packed.token_capacity",      std::to_string(test.tokens) },
        });
        REQUIRE(ref != nullptr);
        REQUIRE(resolve_with_timeout(ref, std::chrono::seconds(120)));
        auto compiled = ref->take_result();
        REQUIRE(compiled.hsaco_data != nullptr);
        REQUIRE(compiled.launch_config.workgroup_size[0] == test.workgroup_size);
        REQUIRE(compiled.launch_config.workgroup_count[0] == test.token_tiles);
        REQUIRE(compiled.launch_config.workgroup_count[1] == test.output_tiles);
        compiled.reset();
    }

    const auto & q4_prefill = find_kernel("ggml_mul_mat_q4_k_f16_wmma_prefill_wave32");

    const struct {
        int64_t tokens;
        int64_t inputs;
        int64_t outputs;
        int64_t workgroup_size;
        int64_t token_tiles;
        int64_t output_tiles;
        int64_t input_layout;
    } q4_launch_cases[] = {
        { 256,  8192,  8192,  512, 1,  64, 0 },
        { 512,  8192,  1024,  512, 2,   8, 0 },
        { 512,  8192,  3840,  512, 2,  30, 0 },
        { 512,  8192,  4096,  512, 1,  64, 1 },
        { 512,  8192,  4224,  512, 1,  66, 1 },
        { 512,  8192, 12288,  512, 1, 192, 1 },
        { 512,  8192, 16384,  512, 1, 256, 1 },
        {1024,  8192,  2048, 1024, 4,   8, 0 },
        { 512, 17408,  5120, 1024, 2,  20, 0 },
        { 512,  5120,  6144,  512, 1,  96, 1 },
        { 512,  6144,  5120,  512, 1,  80, 1 },
        { 512,  5120, 10240,  512, 1, 160, 1 },
        { 512,  5120,  8192,  512, 1, 128, 1 },
        {1024,  5120,  4096, 1024, 4,  16, 0 },
        { 512, 17408,  5120,  512, 1,  80, 1 },
    };

    for (const auto & test : q4_launch_cases) {
        const std::string key = "q4-prefill-" + std::to_string(test.tokens) + "-" + std::to_string(test.inputs) + "-" +
                                std::to_string(test.outputs) + "-" + std::to_string(test.input_layout);
        auto ref = compile_kernel(*sync_jit, key, q4_prefill, token_count_workload(test.tokens),
                                  {
            { "ggml.mul_mat.input_size",      std::to_string(test.inputs) },
            { "ggml.mul_mat.output_size",     std::to_string(test.outputs) },
            { "ggml.workload.token_capacity", std::to_string(test.tokens) },
            { "ggml.mul_mat.f16_input_layout", std::to_string(test.input_layout) },
        });
        REQUIRE(ref != nullptr);
        REQUIRE(resolve_with_timeout(ref, std::chrono::seconds(120)));
        auto compiled = ref->take_result();
        REQUIRE(compiled.hsaco_data != nullptr);
        REQUIRE(compiled.launch_config.workgroup_size[0] == test.workgroup_size);
        REQUIRE(compiled.launch_config.workgroup_count[0] == test.token_tiles);
        REQUIRE(compiled.launch_config.workgroup_count[1] == test.output_tiles);
        compiled.reset();
    }

    const auto & packed_f16 = find_kernel("ggml_copy_f16_k16_major");
    for (const auto * input_is_f16 : { "0", "1" }) {
        auto ref =
            compile_kernel(*sync_jit, std::string("packed-f16-") + input_is_f16, packed_f16, token_count_workload(512),
                           {
            { "ggml.copy_f16_k16_major.input_size",   "256" },
            { "ggml.copy_f16_k16_major.token_count",  "512" },
            { "ggml.copy_f16_k16_major.input_is_f16", input_is_f16 },
        });
        require_compiled_kernel(ref);
    }

    const auto & transpose_f16 = find_kernel("ggml_copy_transpose_f16");
    auto         transpose_ref = compile_kernel(*sync_jit, "transpose-f16-8192x1024", transpose_f16,
                                                {
                                                    { "row_count", 8192 },
                                                    { "column_count", 1024 },
    },
                                                {
    });
    require_compiled_kernel(transpose_ref);
    auto transposed_attention_config = flash_attention_config("24", "4", "256", "0.0625");
    transposed_attention_config.emplace("ggml.flash_attention.value_layout", "1");
    auto transposed_attention_ref =
        compile_kernel(*sync_jit, "flash-transposed-value", flash_prefill,
                       {
                           { "query_token_count",     512  },
                           { "key_value_token_count", 8192 }
    },
                                                   transposed_attention_config);
    require_compiled_kernel(transposed_attention_ref);

    const auto & paired_q4 = find_kernel("ggml_mul_mat_swiglu_q4_k_f16_wmma_prefill_wave32");

    const struct {
        int64_t tokens;
        int64_t outputs;
        int64_t threads;
        int64_t token_tiles;
    } paired_q4_cases[] = {
        { 256, 4096,  512, 1 },
        { 512, 3840,  512, 2 },
        { 512, 4096,  512, 1 },
        { 768, 8192,  512, 3 },
        {1024, 2048,  512, 2 },
        {2048,17408,  512, 4 },
    };

    for (const auto & test : paired_q4_cases) {
        const std::string key = "paired-q4-" + std::to_string(test.tokens) + "-" + std::to_string(test.outputs);
        auto              ref = compile_kernel(*sync_jit, key, paired_q4, token_count_workload(test.tokens),
                                               {
            { "ggml.mul_mat_swiglu.input_size",  "5120" },
            { "ggml.mul_mat_swiglu.output_size", std::to_string(test.outputs) },
            { "ggml.mul_mat_swiglu.op",          "4" },
            { "ggml.mul_mat_swiglu.f16_output_layout", "0" },
            { "ggml.workload.token_capacity",   std::to_string(test.tokens) },
        });
        REQUIRE(ref != nullptr);
        REQUIRE(resolve_with_timeout(ref, std::chrono::seconds(120)));
        auto compiled = ref->take_result();
        REQUIRE(compiled.hsaco_data != nullptr);
        REQUIRE(compiled.launch_config.workgroup_size[0] == test.threads);
        REQUIRE(compiled.launch_config.workgroup_count[0] == test.token_tiles);
        const bool packed = test.tokens % 512 == 0 && (test.tokens / 512) * (test.outputs / 64) >= 64;
        REQUIRE(compiled.launch_config.workgroup_count[1] == test.outputs / (packed ? 32 : 64));
        compiled.reset();
    }

    for (const int64_t tokens : { 256, 512, 1024, 2048 }) {
        auto ref = compile_kernel(*sync_jit, "paired-q4-packed-output-" + std::to_string(tokens), paired_q4,
                                  token_count_workload(tokens),
                                  {
            { "ggml.mul_mat_swiglu.input_size",  "5120" },
            { "ggml.mul_mat_swiglu.output_size", "17408" },
            { "ggml.mul_mat_swiglu.op",          "4" },
            { "ggml.mul_mat_swiglu.f16_output_layout", "1" },
            { "ggml.workload.token_capacity",   std::to_string(tokens) },
        });
        require_compiled_kernel(ref);
    }

    const auto & paired_q4_decode = find_kernel("ggml_mul_mat_swiglu_q4_q8_1_x4_lowtoken_dot");

    const struct {
        int64_t tokens;
        int64_t inputs;
        int64_t outputs;
        int64_t threads;
        int64_t row_tiles;
    } paired_q4_decode_cases[] = {
        { 1, 5120, 17408, 128, 2176 },
        { 1, 4096,  8192, 128, 1024 },
        { 1, 4096, 16384, 128, 2048 },
        { 1, 8192,  8192, 128, 1024 },
        { 1, 5376, 17408, 128, 2176 },
        { 2, 5120, 17408, 512, 1088 },
        { 3, 5120, 17408, 512, 1088 },
        { 4, 5120, 17408, 512, 1088 },
        { 5, 5120, 17408, 512, 1088 },
        { 5, 4096,  8192, 512,  256 },
        { 5, 4096, 16384, 512, 1024 },
        { 3, 8192,  8192, 512,  512 },
        { 5, 5632, 12288, 512,  768 },
        { 5, 5376, 17408, 512,  544 },
        { 5, 6144, 17408, 128, 2176 },
        { 5,  256,  4032, 128,  504 },
        { 5,  256,  4096, 512,  128 },
        { 5, 5632,  4096, 512,  128 },
        { 5, 5888,  4096, 128,  512 },
    };

    for (const auto & test : paired_q4_decode_cases) {
        const std::string key = "paired-q4-decode-" + std::to_string(test.tokens) + "-" + std::to_string(test.inputs) +
                                "-" + std::to_string(test.outputs);
        auto ref = compile_kernel(*sync_jit, key, paired_q4_decode, token_count_workload(test.tokens),
                                  {
            { "ggml.mul_mat_swiglu.input_size",  std::to_string(test.inputs) },
            { "ggml.mul_mat_swiglu.output_size", std::to_string(test.outputs) },
            { "ggml.mul_mat_swiglu.op",          "4" },
            { "ggml.workload.token_capacity",   std::to_string(test.tokens) },
        });
        REQUIRE(ref != nullptr);
        REQUIRE(resolve_with_timeout(ref, std::chrono::seconds(120)));
        auto compiled = ref->take_result();
        REQUIRE(compiled.hsaco_data != nullptr);
        REQUIRE(compiled.launch_config.workgroup_size[0] == test.threads);
        REQUIRE(compiled.launch_config.workgroup_count[0] == test.row_tiles);
        REQUIRE(compiled.launch_config.workgroup_count[1] == 1);
        compiled.reset();
    }

    const auto & quant_decode = find_kernel("ggml_mul_mat_f32_f32_wmma");

    const struct {
        int64_t tokens;
        int64_t inputs;
        int64_t outputs;
        int64_t format;
        int64_t threads;
        int64_t row_tiles;
    } quant_decode_cases[] = {
        { 1,  5120,   48,  4,  256,  12 },
        { 2,  5120,   48,  4, 1024,  12 },
        { 3,  5120,   48,  4, 1024,  12 },
        { 4,  5120,   48,  4, 1024,  12 },
        { 5,  5120,   48,  4, 1024,  12 },
        { 5,  5120,   48,  5, 1024,  12 },
        { 5,  5120,   48,  6, 1024,  12 },
        { 5,  4096,    1,  4, 1024,   1 },
        { 5,  4096,   16,  4, 1024,   4 },
        { 5,  8192,   63,  4, 1024,  16 },
        { 5, 32768,   48,  4, 1024,  12 },
        { 5,  3840,   48,  4,  256,  12 },
        { 5,  4352,   48,  4,  256,  12 },
        { 5,  4096,   64,  4,  128,   8 },
        { 6,  5120,   48,  4,  128,   1 },
        { 1, 17408, 5120, 46, 128, 640 },
        { 4, 17408, 5120, 46, 128, 640 },
        { 5, 17408, 5120, 46, 512, 160 },
        { 5,  8192, 4096, 46, 512, 128 },
        { 5,  8448, 4160, 46, 512, 130 },
        { 5,  7936, 4096, 46, 128, 512 },
        { 5,  8192, 4032, 46, 128, 504 },
        { 5,  8192, 8256, 46, 128, 1032 },
        { 1, 17408, 5120, 44,  128, 640 },
        { 4, 17408, 5120, 44,  128, 640 },
        { 5, 17408, 5120, 44, 1024,  80 },
        { 5,  8192, 4096, 44,  128, 1024 },
        { 1,  6144, 5120, 44,  128,  640 },
        { 4,  6144, 5120, 44,  128,  640 },
        { 5,  6144, 5120, 44,  128, 1280 },
        { 5,  6400, 5120, 44,  128,  640 },
        { 5,  6656, 5120, 44,  128, 1280 },
        { 5,  6144, 4096, 44,  128,  512 },
        { 5,  6144, 6144, 44,  128,  768 },
        { 5,  5120, 6144, 44,  128,  768 },
        { 5,  8448, 4160, 44,  128, 520 },
        { 5,  9216, 9216, 44,  128, 1152 },
        { 5, 12288, 8192, 44,  128, 1024 },
        { 5, 16384, 4096, 44,  128, 512 },
        { 5, 16384, 5056, 44,  128, 632 },
        { 5, 16384, 5120, 44, 1024,  80 },
        { 5, 20480, 4096, 44, 1024,  64 },
        { 5, 20736, 4160, 44, 1024,  65 },
        { 5,  7936, 4096, 44,  128, 512 },
        { 5,  8192, 4032, 44,  128, 504 },
        { 5,  8192, 8256, 44,  128, 1032 },
    };

    for (const auto & test : quant_decode_cases) {
        const std::string key = "quant-decode-" + std::to_string(test.tokens) + "-" + std::to_string(test.inputs) +
                                "-" + std::to_string(test.outputs) + "-" + std::to_string(test.format);
        auto ref =
            compile_kernel(*sync_jit, key, quant_decode, token_count_workload(test.tokens),
                           {
            { "ggml.mul_mat.input_size",          std::to_string(test.inputs) },
            { "ggml.mul_mat.output_size",         std::to_string(test.outputs) },
            { "ggml.mul_mat.weight_format",       std::to_string(test.format) },
            { "ggml.mul_mat.activation_format",   test.format == 44 || test.format == 46 ? "9" : "0" },
            { "ggml.mul_mat.output_accumulation", "0" },
            { "ggml.mul_mat.output_unary_op",     "23" },
            { "ggml.workload.token_capacity",     std::to_string(test.tokens) },
        });
        REQUIRE(ref != nullptr);
        REQUIRE(resolve_with_timeout(ref, std::chrono::seconds(120)));
        auto compiled = ref->take_result();
        REQUIRE(compiled.hsaco_data != nullptr);
        REQUIRE(compiled.launch_config.workgroup_size[0] == test.threads);
        REQUIRE(compiled.launch_config.workgroup_count[0] == test.row_tiles);
        REQUIRE(compiled.launch_config.workgroup_count[1] == 1);
        compiled.reset();
    }


    for (const char * name : { "ggml_mul_mat_quantized_f16_wmma_prefill_conv4",
                              "ggml_mul_mat_quantized_f16_wmma_prefill_conv4_interior" }) {
        const auto & quantized_conv4 = find_kernel(name);
        REQUIRE(quantized_conv4.bindings.size() == (std::string(name).find("interior") != std::string::npos ? 5 : 6));
        for (const int format : {4, 6}) {
            for (const int64_t channels : {2112, 8192, 10240}) {
                auto ref = compile_kernel(*sync_jit, std::string(name) + "-" + std::to_string(format) + "-" +
                                          std::to_string(channels), quantized_conv4, token_count_workload(512), {
                    {"ggml.mul_mat.input_size", "5120"},
                    {"ggml.mul_mat.output_size", std::to_string(channels)},
                    {"ggml.mul_mat.weight_format", std::to_string(format)},
                });
                REQUIRE(ref != nullptr);
                REQUIRE(resolve_with_timeout(ref, std::chrono::seconds(120)));
                auto compiled = ref->take_result();
                REQUIRE(compiled.hsaco_data != nullptr);
                REQUIRE(compiled.launch_config.workgroup_size[0] == 512);
                REQUIRE(compiled.launch_config.workgroup_count[0] == 1);
                REQUIRE(compiled.launch_config.workgroup_count[1] == channels / 64);
                compiled.reset();
            }
        }
    }

    const auto & conv_finish = find_kernel("llm_ssm_conv_dconv4_silu_prefill_finish_f32");
    REQUIRE(conv_finish.bindings.size() == 5);
    for (const int64_t channels : { 2112, 8192, 8256, 10240 }) {
        auto ref = compile_kernel(*sync_jit, "conv-finish-" + std::to_string(channels), conv_finish,
                                  {
        },
                                  {
            { "llm.ssm_conv.generic.d_inner", std::to_string(channels) },
        });
        REQUIRE(ref != nullptr);
        REQUIRE(resolve_with_timeout(ref, std::chrono::seconds(120)));
        auto compiled = ref->take_result();
        REQUIRE(compiled.hsaco_data != nullptr);
        REQUIRE(compiled.launch_config.workgroup_size[0] == 256);
        REQUIRE(compiled.launch_config.workgroup_count[0] == (channels + 255) / 256);
    }

    const auto & ssm_prefill = find_kernel("llm_ssm_conv_dconv4_silu_prefill_512_wg1024");
    for (const int64_t channels : { 8192, 8224, 8256, 10208, 10240 }) {
        auto ref = compile_kernel(*sync_jit, "ssm-prefill-" + std::to_string(channels), ssm_prefill,
                                  {
        },
                                  {
            { "llm.ssm_conv.prefill.d_conv",           "4" },
            { "llm.ssm_conv.prefill.d_inner",          std::to_string(channels) },
            { "llm.ssm_conv.prefill.n_t",              "512" },
            { "llm.ssm_conv.prefill.n_s",              "1" },
            { "llm.ssm_conv.prefill.state_row_stride", std::to_string(channels) },
            { "llm.ssm_conv.prefill.x_row_stride",     std::to_string(channels) },
        });
        REQUIRE(ref != nullptr);
        REQUIRE(resolve_with_timeout(ref, std::chrono::seconds(120)));
        auto compiled = ref->take_result();
        REQUIRE(compiled.hsaco_data != nullptr);
        REQUIRE(compiled.launch_config.workgroup_size[0] == 1024);
        REQUIRE(compiled.launch_config.workgroup_count[0] == channels / (channels % 64 == 0 ? 64 : 32));
        REQUIRE(compiled.launch_config.workgroup_count[1] == 1);
        compiled.reset();
    }

    const struct {
        const char * name;
        int64_t tokens;
        size_t bindings;
    } gdn_cases[] = {
        { "llm_gated_delta_net_f32_wmma_head128", 17, 7 },
        { "llm_gated_delta_net_f32_wmma_head128_rmsnorm_gate", 17, 11 },
        { "llm_gated_delta_net_f32_wmma_head128_projection_epilogue", 16, 9 },
        { "llm_gated_delta_net_f32_wmma_head128_inplace", 1, 7 },
        { "llm_gated_delta_net_f32_wmma_head128_inplace_projection_epilogue", 1, 9 },
        { "llm_gated_delta_net_f32_wmma_head128_snapshot", 5, 8 },
        { "llm_gated_delta_net_f32_wmma_head128_snapshot_projection_epilogue", 5, 10 },
    };

    for (const auto & test : gdn_cases) {
        const auto & definition = find_kernel(test.name);
        REQUIRE(definition.bindings.size() == test.bindings);
        auto ref = compile_kernel(*sync_jit, test.name, definition,
                                  {
        },
                                  {
            { "llm.gated_delta_net.head_width", "128" },
            { "llm.gated_delta_net.head_count", "3" },
            { "llm.gated_delta_net.token_count", std::to_string(test.tokens) },
            { "llm.gated_delta_net.sequence_count", "1" },
            { "llm.gated_delta_net.qk_stride1", "128" },
            { "llm.gated_delta_net.qk_stride2", "640" },
            { "llm.gated_delta_net.qk_stride3", std::to_string(test.tokens * 640) },
            { "llm.gated_delta_net.value_stride1", "128" },
            { "llm.gated_delta_net.value_stride2", "640" },
            { "llm.gated_delta_net.value_stride3", std::to_string(test.tokens * 640) },
            { "llm.gated_delta_net.scalar_stride1", "1" },
            { "llm.gated_delta_net.scalar_stride2", "3" },
            { "llm.gated_delta_net.scalar_stride3", std::to_string(test.tokens * 3) },
            { "llm.gated_delta_net.query_head_count", "1" },
            { "llm.gated_delta_net.query_sequence_ratio", "1" },
            { "llm.gated_delta_net.snapshot_stride", "49216" },
            { "llm.gated_delta_net.l2_epsilon", "0.000001" },
            { "ggml.rmsnorm_gate_f32.rms_epsilon", "0.000001" },
            { "ggml.rmsnorm_gate_f32.gate_op", "15" },
        });
        REQUIRE(ref != nullptr);
        REQUIRE(resolve_with_timeout(ref, std::chrono::seconds(120)));
        auto compiled = ref->take_result();
        REQUIRE(compiled.hsaco_data != nullptr);
        REQUIRE(compiled.launch_config.workgroup_size[0] == 256);
        REQUIRE(compiled.launch_config.workgroup_count[0] == 3);
        REQUIRE(compiled.launch_config.workgroup_count[1] == 1);
        compiled.reset();
    }
}

static void run_moe_routing_compile_checks() {
    static constexpr const char * kTarget = "gfx1100";

    std::string                         error;
    std::unique_ptr<ggml::hrx::LoomJit> jit = ggml::hrx::create_loom_jit(kTarget, ggml::hrx::LoomJitMode::Sync, error);
    REQUIRE(jit != nullptr);

    const std::map<std::string, int64_t> expert_table_workload = {
        { "token_count", 1 },
        { "route_count", 8 },
        { "route_stride", 8 },
        { "expert_count", 128 },
    };
    const auto & expert_table = find_kernel("ggml_moe_build_expert_table");
    require_compiled_kernel(
        compile_kernel(*jit, "moe-expert-table", expert_table, expert_table_workload, moe_routing_config()));

    const std::map<std::string, int64_t> partition_table_workload = {
        { "token_count", 1 },
        { "route_count", 8 },
        { "expert_count", 128 },
    };
    const auto & partition_table = find_kernel("ggml_moe_build_expert_partition_table");
    require_compiled_kernel(compile_kernel(*jit, "moe-expert-partition-table", partition_table,
                                           partition_table_workload, moe_routing_config()));

    const auto & weighted_reduce_publish_q8 =
        find_kernel("qwen3_moe_routed_down_weighted_reduce_next_rmsnorm_f32_publish_q8");
    REQUIRE(weighted_reduce_publish_q8.bindings.size() == 6);
    require_compiled_kernel(compile_kernel(*jit, "moe-weighted-reduce-next-rmsnorm-publish-q8",
                                           weighted_reduce_publish_q8, token_count_workload(2), {
                                               { "qwen3_moe.model.hidden_size", "2048" },
                                               { "qwen3_moe.model.rms_epsilon", "0.000001" },
                                               { "qwen3_moe.routed_down.output_size", "2048" },
                                               { "qwen3_moe.routed_down.route_count", "8" },
                                               { "qwen3_moe.workload.token_capacity", "2" },
                                               { "ggml.quantize_q8_1_x4.group_capacity", "32" },
                                           }));
}

static void run_prefill_fusion_compile_checks() {
    static constexpr const char * kTarget = "gfx1100";

    std::string                         error;
    std::unique_ptr<ggml::hrx::LoomJit> jit = ggml::hrx::create_loom_jit(kTarget, ggml::hrx::LoomJitMode::Sync, error);
    REQUIRE(jit != nullptr);

    const auto & rmsnorm_f32_f16 = find_kernel("ggml_rmsnorm_binary_f32_f16");
    require_compiled_kernel(compile_kernel(*jit, "prefill-rmsnorm-f32-f16", rmsnorm_f32_f16, token_count_workload(512),
                                           {
                                               { "ggml.rmsnorm_binary_f32.hidden_size", "2048" },
                                               { "ggml.rmsnorm_binary_f32.rms_epsilon", "0.00001" },
                                               { "ggml.rmsnorm_binary_f32.op", "2" },
                                           }));

    const auto & rmsnorm_lowtoken_striped = find_kernel("ggml_rmsnorm_binary_f32_lowtoken_striped");
    require_compiled_kernel(compile_kernel(*jit, "rmsnorm-lowtoken-striped", rmsnorm_lowtoken_striped,
                                           token_count_workload(1), {
                                               { "ggml.rmsnorm_binary_f32.hidden_size", "5120" },
                                               { "ggml.rmsnorm_binary_f32.rms_epsilon", "0.000001" },
                                               { "ggml.rmsnorm_binary_f32.op", "2" },
                                           }));

    const auto & rmsnorm_strided = find_kernel("ggml_rmsnorm_binary_strided_f32");
    require_compiled_kernel(compile_kernel(*jit, "prefill-rmsnorm-strided", rmsnorm_strided, token_count_workload(64),
                                           {
                                               { "ggml.rmsnorm_binary_f32.hidden_size", "256" },
                                               { "ggml.rmsnorm_binary_f32.input_stride", "288" },
                                               { "ggml.rmsnorm_binary_f32.rms_epsilon", "0.00001" },
                                               { "ggml.rmsnorm_binary_f32.op", "2" },
                                           }));

    const auto & rmsnorm_q8_f16 = find_kernel("ggml_rmsnorm_binary_q8_1_x4_publish");
    require_compiled_kernel(compile_kernel(*jit, "prefill-rmsnorm-q8-f16", rmsnorm_q8_f16, token_count_workload(512),
                                           {
                                               { "ggml.rmsnorm_binary_q8_1_x4.hidden_size", "2048" },
                                               { "ggml.rmsnorm_binary_q8_1_x4.rms_epsilon", "0.00001" },
                                               { "ggml.rmsnorm_binary_q8_1_x4.op", "2" },
                                               { "ggml.rmsnorm_binary_q8_1_x4.publish_f16", "1" },
                                           }));

    const auto & flash_f16 = find_kernel("ggml_flash_attention_f32_f16_wmma_publish_f16");
    require_compiled_kernel(compile_kernel(*jit, "prefill-flash-f16", flash_f16, flash_attention_workload(512, 512),
                                           flash_attention_config("32", "8", "64", "0.125")));

    const auto & swiglu_k16 = find_kernel("ggml_mul_mat_tiled_pair_input_f32_binary_publish_f32_k16");
    require_compiled_kernel(compile_kernel(*jit, "prefill-swiglu-k16", swiglu_k16, token_count_workload(512),
                                           {
                                               { "ggml.mul_mat_swiglu.input_size", "2048" },
                                               { "ggml.mul_mat_swiglu.output_size", "8192" },
                                               { "ggml.mul_mat_swiglu.gate_weight_format", "4" },
                                               { "ggml.mul_mat_swiglu.up_weight_format", "4" },
                                               { "ggml.mul_mat_swiglu.op", "4" },
                                               { "ggml.workload.token_capacity", "512" },
                                           }));

    const auto & get_rows_rmsnorm = find_kernel("ggml_get_rows_rmsnorm_binary_q8_1_x4_f16");
    require_compiled_kernel(compile_kernel(*jit, "prefill-get-rows-rmsnorm", get_rows_rmsnorm,
                                           {
                                               { "token_count", 512 },
                                               { "row_count", 128256 },
                                               { "hidden_size", 2048 },
    },
                                           {
                                               { "ggml.get_rows_f32.token_capacity", "512" },
                                               { "ggml.get_rows_f32.hidden_capacity", "2048" },
                                               { "ggml.get_rows_f32.weight_format", "6" },
                                               { "ggml.get_rows_rmsnorm.rms_epsilon", "0.00001" },
                                           }));

    const auto & get_rows_scale = find_kernel("ggml_get_rows_scale_f32");
    require_compiled_kernel(compile_kernel(*jit, "prefill-get-rows-scale", get_rows_scale,
                                           {
                                               { "token_count", 64 },
                                               { "row_count", 122753 },
                                               { "hidden_size", 2560 },
    },
                                           {
                                               { "ggml.get_rows_f32.token_capacity", "64" },
                                               { "ggml.get_rows_f32.hidden_capacity", "2560" },
                                               { "ggml.get_rows_f32.weight_format", "32" },
                                               { "ggml.get_rows_scale_f32.scale", "0.177800179" },
                                               { "ggml.get_rows_scale_f32.bias", "0" },
                                           }));
    require_compiled_kernel(compile_kernel(*jit, "gemma-prefill-get-rows-scale-q5_1", get_rows_scale,
                                           {
                                               { "token_count", 64 },
                                               { "row_count", 262144 },
                                               { "hidden_size", 640 },
    },
                                           {
                                               { "ggml.get_rows_f32.token_capacity", "64" },
                                               { "ggml.get_rows_f32.hidden_capacity", "640" },
                                               { "ggml.get_rows_f32.weight_format", "51" },
                                               { "ggml.get_rows_scale_f32.scale", "25.2982216" },
                                               { "ggml.get_rows_scale_f32.bias", "0" },
                                           }));
    require_compiled_kernel(compile_kernel(*jit, "gemma-decode-get-rows-scale-q5_1", get_rows_scale,
                                           {
                                               { "token_count", 1 },
                                               { "row_count", 262144 },
                                               { "hidden_size", 640 },
    },
                                           {
                                               { "ggml.get_rows_f32.token_capacity", "1" },
                                               { "ggml.get_rows_f32.hidden_capacity", "640" },
                                               { "ggml.get_rows_f32.weight_format", "51" },
                                               { "ggml.get_rows_scale_f32.scale", "25.2982216" },
                                               { "ggml.get_rows_scale_f32.bias", "0" },
                                           }));

    const auto & lfm_dconv3 = find_kernel("llm_ssm_conv_lfm_dconv3_state_f32");
    REQUIRE(lfm_dconv3.bindings.size() == 5);
    for (const int64_t tokens : {1, 64}) {
        auto ref = compile_kernel(*jit, "lfm-dconv3-" + std::to_string(tokens), lfm_dconv3, {}, {
            {"llm.ssm_conv.lfm_dconv3.n_t", std::to_string(tokens)},
        });
        REQUIRE(ref != nullptr);
        REQUIRE(resolve_with_timeout(ref, std::chrono::seconds(120)));
        auto compiled = ref->take_result();
        REQUIRE(compiled.hsaco_data != nullptr);
        REQUIRE(compiled.launch_config.workgroup_size[0] == 256);
        REQUIRE(compiled.launch_config.workgroup_count[0] == static_cast<uint32_t>(8 * tokens));
    }
}

static void run_binary_fusion_compile_checks() {
    std::string                         error;
    std::unique_ptr<ggml::hrx::LoomJit> jit =
        ggml::hrx::create_loom_jit("gfx1100", ggml::hrx::LoomJitMode::Sync, error);
    REQUIRE(jit != nullptr);

    const auto & gather = find_kernel("ggml_gather_add_f32");
    require_compiled_kernel(compile_kernel(*jit, "binary-fusion-gather-div", gather,
                                           {
                                               { "source_token_count", 2    },
                                               { "output_token_count", 1    },
                                               { "hidden_size",        2048 },
    },
                                           {
                                               { "ggml.gather_add_f32.binary_op", "3" },
                                           }));

    const auto & scale = find_kernel("ggml_scale_add_f32");
    require_compiled_kernel(compile_kernel(*jit, "binary-fusion-scale-sub", scale,
                                           {
                                               { "element_count", 128 },
    },
                                           {
                                               { "ggml.scale_add_f32.ne0", "128" },
                                               { "ggml.scale_add_f32.ne1", "1" },
                                               { "ggml.scale_add_f32.ne2", "1" },
                                               { "ggml.scale_add_f32.input_stride1", "128" },
                                               { "ggml.scale_add_f32.input_stride2", "128" },
                                               { "ggml.scale_add_f32.input_stride3", "128" },
                                               { "ggml.scale_add_f32.input_span", "128" },
                                               { "ggml.scale_add_f32.residual_stride1", "128" },
                                               { "ggml.scale_add_f32.residual_stride2", "128" },
                                               { "ggml.scale_add_f32.residual_stride3", "128" },
                                               { "ggml.scale_add_f32.residual_span", "128" },
                                               { "ggml.scale_add_f32.scale", "0.5" },
                                               { "ggml.scale_add_f32.bias", "0" },
                                               { "ggml.scale_add_f32.binary_op", "1" },
                                               { "ggml.scale_add_f32.scaled_lhs", "0" },
                                           }));

    const auto & rms = find_kernel("ggml_rmsnorm_mul_add_f32");
    require_compiled_kernel(compile_kernel(*jit, "binary-fusion-rms-div-sub", rms,
                                           {
                                               { "token_count", 1 },
    },
                                           {
                                               { "ggml.rmsnorm_mul_add_f32.hidden_size", "128" },
                                               { "ggml.rmsnorm_mul_add_f32.rms_epsilon", "0.000001" },
                                               { "ggml.rmsnorm_mul_add_f32.normalized_op", "3" },
                                               { "ggml.rmsnorm_mul_add_f32.output_op", "1" },
                                               { "ggml.rmsnorm_mul_add_f32.binary_lhs", "0" },
                                           }));

    const auto & gather_rms = find_kernel("ggml_gather_add_rmsnorm_binary_f32");
    require_compiled_kernel(compile_kernel(*jit, "binary-fusion-gather-rms-mul", gather_rms,
                                           {
                                               { "source_token_count", 2 },
                                               { "output_token_count", 1 },
    },
                                           {
                                               { "ggml.gather_add_rmsnorm_binary_f32.hidden_size", "128" },
                                               { "ggml.gather_add_rmsnorm_binary_f32.rms_epsilon", "0.000001" },
                                               { "ggml.gather_add_rmsnorm_binary_f32.gather_op", "2" },
                                           }));

    const auto & ssm = find_kernel("llm_ssm_conv_binary_f32");
    require_compiled_kernel(compile_kernel(*jit, "binary-fusion-ssm-rhs-div", ssm,
                                           {
                                               { "n_t", 8 },
                                               { "n_s", 1 },
    },
                                           {
                                               { "llm.ssm_conv.generic.d_conv", "4" },
                                               { "llm.ssm_conv.generic.d_inner", "64" },
                                               { "llm.ssm_conv.generic.unary_op", "23" },
                                               { "llm.ssm_conv.generic.binary_op", "3" },
                                               { "llm.ssm_conv.generic.binary_lhs", "0" },
                                               { "llm.ssm_conv.generic.workgroup_size", "256" },
                                           }));

    const auto & vector_postops = find_kernel("ggml_mul_mat_vector_bias_residual_f32_f32");
    for (const int64_t epilogue : { int64_t{ 1 }, int64_t{ 2 }, int64_t{ 3 } }) {
        require_compiled_kernel(compile_kernel(
            *jit, "binary-fusion-vector-postops-" + std::to_string(epilogue), vector_postops, token_count_workload(1),
            {
                { "ggml.matmul.vector_postops.input_size",     "2048"                   },
                { "ggml.matmul.vector_postops.output_size",    "2048"                   },
                { "ggml.matmul.vector_postops.weight_format",  "4"                      },
                { "ggml.matmul.vector_postops.apply_bias",     epilogue & 1 ? "1" : "0" },
                { "ggml.matmul.vector_postops.apply_residual", epilogue & 2 ? "1" : "0" },
                { "ggml.workload.token_capacity",              "1"                      },
        }));
    }

    const auto compile_postops = [&](const char * kernel_name, const char * key_prefix, int64_t token_count,
                                     int64_t token_capacity) {
        const auto & kernel = find_kernel(kernel_name);
        for (const int64_t epilogue : { int64_t{ 1 }, int64_t{ 2 }, int64_t{ 3 } }) {
            require_compiled_kernel(compile_kernel(
                *jit, std::string(key_prefix) + std::to_string(epilogue), kernel, token_count_workload(token_count),
                {
                    { "ggml.mul_mat_postops.input_size",    "2048"                         },
                    { "ggml.mul_mat_postops.output_size",   "2048"                         },
                    { "ggml.mul_mat_postops.weight_format", "4"                            },
                    { "ggml.mul_mat_postops.epilogue",      std::to_string(epilogue)       },
                    { "ggml.workload.token_capacity",       std::to_string(token_capacity) },
            }));
        }
    };
    compile_postops("ggml_mul_mat_skinny_input_f32_bias_residual_publish_f32", "binary-fusion-skinny-postops-", 2, 2);
    compile_postops("ggml_mul_mat_tiled_input_f32_bias_residual_publish_f32", "binary-fusion-tiled-postops-", 32, 32);

    const auto & pair_postops = find_kernel("ggml_mul_mat_tiled_pair_input_f32_binary_bias_residual_publish_f32");
    for (const int64_t epilogue : { int64_t{ 1 }, int64_t{ 2 }, int64_t{ 3 } }) {
        require_compiled_kernel(compile_kernel(*jit, "binary-fusion-pair-postops-" + std::to_string(epilogue),
                                               pair_postops, token_count_workload(32),
                                               {
                                                   { "ggml.mul_mat_swiglu.input_size",         "2048"                   },
                                                   { "ggml.mul_mat_swiglu.output_size",        "2048"                   },
                                                   { "ggml.mul_mat_swiglu.gate_weight_format", "4"                      },
                                                   { "ggml.mul_mat_swiglu.up_weight_format",   "4"                      },
                                                   { "ggml.mul_mat_swiglu.op",                 "4"                      },
                                                   { "ggml.mul_mat_swiglu.epilogue",           std::to_string(epilogue) },
                                                   { "ggml.workload.token_capacity",           "32"                     },
        }));
    }
}

static void run_loom_jit_materialize_checks() {
    const ggml::hrx::KernelDefinition & binary = find_kernel("ggml_binary_f32");

    HrxTestDevice device;
    REQUIRE(device.open());
    run_cache_materialize_case(device, ggml::hrx::LoomJitMode::Sync, binary);
    run_cache_materialize_case(device, ggml::hrx::LoomJitMode::Async, binary);
    run_dynamic_cache_materialize_case(device, ggml::hrx::LoomJitMode::Sync, binary);
    run_dynamic_cache_materialize_case(device, ggml::hrx::LoomJitMode::Async, binary);
    run_dynamic_export_reuse_case(device, "ggml_unary_f32", { { "element_count", 64 } },
                                  { { "element_count", 257 } }, { { "ggml.unary_f32.op", "4" } });
    run_dynamic_export_reuse_case(device, "ggml_copy_f32", { { "element_count", 64 } },
                                  { { "element_count", 257 } });
    run_dynamic_export_reuse_case(
        device, "ggml_concat_dim0_f32",
        { { "row_count", 3 }, { "lhs_span", 192 }, { "rhs_span", 96 } },
        { { "row_count", 7 }, { "lhs_span", 448 }, { "rhs_span", 224 } },
        { { "ggml.concat_dim0_f32.lhs_width", "64" }, { "ggml.concat_dim0_f32.rhs_width", "32" } });
    run_dynamic_export_reuse_case(
        device, "ggml_rmsnorm_f32", { { "token_count", 1 } }, { { "token_count", 4 } },
        { { "ggml.rmsnorm_f32.hidden_size", "256" }, { "ggml.rmsnorm_f32.input_stride", "256" },
          { "ggml.rmsnorm_f32.rms_epsilon", "1e-06" } });
    run_dynamic_export_reuse_case(
        device, "ggml_get_rows_f32",
        { { "token_count", 3 }, { "row_count", 64 }, { "hidden_size", 256 } },
        { { "token_count", 7 }, { "row_count", 64 }, { "hidden_size", 256 } },
        { { "ggml.get_rows_f32.token_capacity", "8" }, { "ggml.get_rows_f32.hidden_capacity", "256" },
          { "ggml.get_rows_f32.weight_format", "32" } });
    run_dynamic_export_reuse_case(
        device, "ggml_rope_f32", { { "token_count", 3 }, { "input_span", 768 } },
        { { "token_count", 7 }, { "input_span", 1792 } },
        { { "ggml.rope_f32.head_size", "64" }, { "ggml.rope_f32.n_dims", "64" },
          { "ggml.rope_f32.head_count", "4" }, { "ggml.rope_f32.token_capacity", "8" },
          { "ggml.rope_f32.input_stride1", "64" }, { "ggml.rope_f32.input_stride2", "256" },
          { "ggml.rope_f32.mscale", "1" }, { "ggml.rope_f32.mode", "0" } });
    run_dynamic_export_reuse_case(
        device, "ggml_mul_mat_tiled_input_f32_publish_f32_aligned", { { "token_count", 32 } },
        { { "token_count", 64 } },
        { { "ggml.mul_mat.input_size", "256" }, { "ggml.mul_mat.output_size", "64" },
          { "ggml.mul_mat.output_accumulation", "0" }, { "ggml.mul_mat.output_unary_op", "23" },
          { "ggml.mul_mat.weight_format", "4" } });
    run_dynamic_export_reuse_case(
        device, "ggml_mul_mat_tiled_pair_input_f32_binary_publish_f32_aligned", { { "token_count", 32 } },
        { { "token_count", 64 } },
        { { "ggml.mul_mat_swiglu.input_size", "256" }, { "ggml.mul_mat_swiglu.output_size", "64" },
          { "ggml.mul_mat_swiglu.gate_weight_format", "4" },
          { "ggml.mul_mat_swiglu.up_weight_format", "4" }, { "ggml.mul_mat_swiglu.op", "4" } });
    run_dynamic_export_reuse_case(
        device, "llm_ssm_conv_f32", { { "n_t", 8 }, { "n_s", 1 } }, { { "n_t", 16 }, { "n_s", 2 } },
        { { "llm.ssm_conv.generic.d_conv", "4" }, { "llm.ssm_conv.generic.d_inner", "64" },
          { "llm.ssm_conv.generic.unary_op", "23" }, { "llm.ssm_conv.generic.workgroup_size", "256" } });
    run_dynamic_export_reuse_case(
        device, "llm_ssm_conv_binary_f32", { { "n_t", 8 }, { "n_s", 1 } },
        { { "n_t", 16 }, { "n_s", 2 } },
        { { "llm.ssm_conv.generic.d_conv", "4" }, { "llm.ssm_conv.generic.d_inner", "64" },
          { "llm.ssm_conv.generic.unary_op", "23" }, { "llm.ssm_conv.generic.binary_op", "3" },
          { "llm.ssm_conv.generic.binary_lhs", "0" },
          { "llm.ssm_conv.generic.workgroup_size", "256" } });
    run_dynamic_export_reuse_case(device, "ggml_copy_transpose_f16",
                                  { { "row_count", 512 }, { "column_count", 256 } },
                                  { { "row_count", 1024 }, { "column_count", 512 } });
    run_dynamic_export_reuse_case(
        device, "ggml_flash_attention_f32_f16_wmma", flash_attention_workload(16, 64),
        flash_attention_workload(17, 65), flash_attention_config("32", "4", "128", "0.0883883461"));
    run_dynamic_export_reuse_case(
        device, "ggml_flash_attention_f32_f16_wmma_publish_f16", flash_attention_workload(16, 64),
        flash_attention_workload(17, 65), flash_attention_config("32", "4", "128", "0.0883883461"));
    run_dynamic_export_reuse_case(
        device, "ggml_flash_attention_decode_split_f32_f16_wmma_next_q8",
        flash_attention_decode_workload(65), flash_attention_decode_workload(127),
        flash_attention_decode_config("32", "4", "128", "0.0883883461", "128"));
    run_dynamic_export_reuse_case(
        device, "llm_gated_delta_net_projection_epilogue_f32", { { "element_count", 48 } },
        { { "element_count", 192 } },
        { { "llm.gated_delta_net.epilogue_head_count", "48" },
          { "llm.gated_delta_net.epilogue_workgroup_size", "256" } });
    run_targeted_export_materialize_case(device);
}

static void run_binary_publication_compile_checks() {
    std::string error;
    std::unique_ptr<ggml::hrx::LoomJit> jit =
        ggml::hrx::create_loom_jit("gfx1100", ggml::hrx::LoomJitMode::Sync, error);
    REQUIRE(jit != nullptr);

    const auto & binary_q8 = find_kernel("ggml_binary_f32_publish_q8_1_x4");
    require_compiled_kernel(compile_kernel(*jit, "binary-mul-q8", binary_q8, {
                                               { "token_count", 2 },
                                           }, {
                                               { "ggml.binary_f32.op", "2" },
                                               { "ggml.binary_f32.ne0", "2048" },
                                               { "ggml.binary_f32.src0_stride1", "2048" },
                                               { "ggml.binary_f32.src1_stride1", "2048" },
                                               { "ggml.binary_f32.src0_span", "4096" },
                                               { "ggml.binary_f32.src1_span", "4096" },
                                           }));

    const auto & binary_bc_q8 = find_kernel("ggml_binary_bc_f32_publish_q8_1_x4");
    require_compiled_kernel(compile_kernel(*jit, "binary-bc-mul-q8", binary_bc_q8, {
                                               { "token_count", 2 },
                                               { "hidden_size", 256 },
                                               { "src0_element_count", 512 },
                                               { "src1_element_count", 256 },
                                           }, {
                                               { "ggml.binary_bc_f32.op", "2" },
                                               { "ggml.binary_bc_f32.src0_broadcast_dim0", "0" },
                                               { "ggml.binary_bc_f32.src0_broadcast_dim1", "0" },
                                               { "ggml.binary_bc_f32.src1_broadcast_dim0", "0" },
                                               { "ggml.binary_bc_f32.src1_broadcast_dim1", "1" },
                                           }));
}

static bool has_hrx_test_device() {
    HrxTestDevice device;
    return device.open();
}

static void register_hrx_loom_jit_cases(test_runner::Suite & suite) {
    suite.host_case("compile", [] { run_loom_jit_compile_checks(); });
    suite.host_case("binary-publication", [] { run_binary_publication_compile_checks(); });
    suite.host_case("moe-routing", [] { run_moe_routing_compile_checks(); });
    suite.host_case("prefill-fusions", [] { run_prefill_fusion_compile_checks(); });
    suite.host_case("binary-fusions", [] { run_binary_fusion_compile_checks(); });
    suite.device_case("materialize", [] { run_loom_jit_materialize_checks(); });
}

int main(int argc, char ** argv) {
    test_runner::Suite suite(
        test_runner::Config::with_prefix("HRX Loom JIT test", "hrx-loom-jit", "GGML_HRX_LOOM_JIT_TEST", 2));
    register_hrx_loom_jit_cases(suite);

    const bool has_device = has_hrx_test_device();
    return suite.run(argc, argv, has_device);
}
