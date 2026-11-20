#include "loom-jit.h"

#include "loomc/launch_config.h"
#include "loomc/loomc.h"
#include "loomc/target/amdgpu.h"

#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <vector>

static constexpr size_t GGML_HRX_LOOM_MAX_LAUNCH_RESULTS = 1024;

struct ggml_hrx_loom_launch_program {
    loomc_launch_config_program_t * program  = nullptr;
    loomc_launch_config_function_t  function = loomc_launch_config_function_invalid();
    std::mutex                      mutex;
    // The result depends only on the workload. The program call is slow.
    std::map<std::vector<uint64_t>, ggml_hrx_loom_launch_config> results;

    ~ggml_hrx_loom_launch_program() {
        if (program != nullptr) {
            loomc_launch_config_program_release(program);
        }
    }
};

struct ggml_hrx_loom_jit {
    loomc_target_environment_t * target_environment = nullptr;
    loomc_context_t *            context            = nullptr;
    loomc_target_profile_t *     target_profile     = nullptr;
    loomc_compiler_t *           compiler           = nullptr;
    loomc_pass_program_t *       pass_program       = nullptr;
    std::string                  targets; // Loom source that defines the kernel targets.

    ~ggml_hrx_loom_jit() {
        loomc_pass_program_release(pass_program);
        loomc_compiler_release(compiler);
        loomc_target_profile_release(target_profile);
        loomc_context_release(context);
        loomc_target_environment_release(target_environment);
    }
};

namespace {

template <typename T, void (*Release)(T *)> class loom_handle {
  public:
    loom_handle() = default;
    loom_handle(const loom_handle &) = delete;
    loom_handle & operator=(const loom_handle &) = delete;
    ~loom_handle() { reset(); }

    T * get() const { return value; }

    T ** out() {
        reset();
        return &value;
    }

    void reset() {
        if (value != nullptr) {
            Release(value);
        }
        value = nullptr;
    }

  private:
    T * value = nullptr;
};

using loom_workspace          = loom_handle<loomc_workspace_t, loomc_workspace_release>;
using loom_source             = loom_handle<loomc_source_t, loomc_source_release>;
using loom_module             = loom_handle<loomc_module_t, loomc_module_release>;
using loom_result             = loom_handle<loomc_result_t, loomc_result_release>;
using loom_link_index_builder = loom_handle<loomc_link_index_builder_t, loomc_link_index_builder_free>;
using loom_link_index         = loom_handle<loomc_link_index_t, loomc_link_index_release>;
using loom_linker             = loom_handle<loomc_linker_t, loomc_linker_release>;

#if defined(LOOMC_ARTIFACT_ROLE_COMPILE_REPORT)
using artifact_selector = loomc_string_view_t;
#define LOOM_ARTIFACT_LAUNCH_CONFIG loomc_make_cstring_view(LOOMC_ARTIFACT_ROLE_LAUNCH_CONFIG)
#define LOOM_ARTIFACT_KERNEL        loomc_make_cstring_view(LOOMC_ARTIFACT_ROLE_KERNEL)
static bool artifact_matches(const loomc_artifact_t * artifact, artifact_selector selector) {
    return loomc_string_view_equal(artifact->role, selector);
}
#else  // defined(LOOMC_ARTIFACT_ROLE_COMPILE_REPORT)
using artifact_selector = loomc_artifact_kind_t;
#define LOOM_ARTIFACT_LAUNCH_CONFIG LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG
#define LOOM_ARTIFACT_KERNEL        LOOMC_ARTIFACT_KIND_EXECUTABLE
static bool artifact_matches(const loomc_artifact_t * artifact, artifact_selector selector) {
    return artifact->kind == selector;
}
#endif // defined(LOOMC_ARTIFACT_ROLE_COMPILE_REPORT)

static hrx_status_code_t status_code_from_loom(loomc_status_code_t code) {
    switch (code) {
        case LOOMC_STATUS_OK:                  return HRX_STATUS_OK;
        case LOOMC_STATUS_CANCELLED:           return HRX_STATUS_CANCELLED;
        case LOOMC_STATUS_INVALID_ARGUMENT:    return HRX_STATUS_INVALID_ARGUMENT;
        case LOOMC_STATUS_DEADLINE_EXCEEDED:   return HRX_STATUS_DEADLINE_EXCEEDED;
        case LOOMC_STATUS_NOT_FOUND:           return HRX_STATUS_NOT_FOUND;
        case LOOMC_STATUS_ALREADY_EXISTS:      return HRX_STATUS_ALREADY_EXISTS;
        case LOOMC_STATUS_PERMISSION_DENIED:   return HRX_STATUS_PERMISSION_DENIED;
        case LOOMC_STATUS_RESOURCE_EXHAUSTED:  return HRX_STATUS_OUT_OF_MEMORY;
        case LOOMC_STATUS_FAILED_PRECONDITION: return HRX_STATUS_FAILED_PRECONDITION;
        case LOOMC_STATUS_ABORTED:             return HRX_STATUS_ABORTED;
        case LOOMC_STATUS_OUT_OF_RANGE:        return HRX_STATUS_OUT_OF_RANGE;
        case LOOMC_STATUS_UNIMPLEMENTED:       return HRX_STATUS_UNIMPLEMENTED;
        case LOOMC_STATUS_UNAVAILABLE:         return HRX_STATUS_UNAVAILABLE;
        case LOOMC_STATUS_DATA_LOSS:           return HRX_STATUS_DATA_LOSS;
        default:                               return HRX_STATUS_INTERNAL;
    }
}

static std::string format_loom_status(loomc_status_t status) {
    loomc_host_size_t length = 0;
    loomc_status_format(status, 0, nullptr, &length);
    std::string text(length + 1, '\0');
    loomc_host_size_t actual = 0;
    loomc_status_format(status, text.size(), text.data(), &actual);
    text.resize(actual);
    return text;
}

static hrx_status_t make_status(hrx_status_code_t code, const std::string & message) {
    return hrx_make_status(code, message.c_str());
}

// Takes ownership of the status.
static hrx_status_t status_from_loom(loomc_status_t status, const char * what) {
    if (loomc_status_is_ok(status)) {
        return hrx_ok_status();
    }
    const hrx_status_code_t code    = status_code_from_loom(loomc_status_code(status));
    const std::string       message = std::string(what) + ": " + format_loom_status(status);
    loomc_status_free(status);
    return make_status(code, message);
}

static hrx_status_t status_from_result(const loomc_result_t * result, const char * what) {
    if (result != nullptr && loomc_result_succeeded(result)) {
        return hrx_ok_status();
    }
    std::string message = what;
    if (result != nullptr) {
        for (loomc_host_size_t i = 0; i < loomc_result_diagnostic_count(result); ++i) {
            const loomc_diagnostic_t * diag = loomc_result_diagnostic_at(result, i);
            if (diag == nullptr) {
                continue;
            }
            message += "\n  ";
            message.append(diag->code.data, diag->code.size);
            message += ": ";
            message.append(diag->message.data, diag->message.size);
            if (diag->range.start_line || diag->range.start_column) {
                message += " @ " + std::to_string(diag->range.start_line) + ":" + std::to_string(diag->range.start_column);
            }
        }
    }
    return make_status(HRX_STATUS_FAILED_PRECONDITION, message);
}

static const loomc_artifact_t * find_artifact(const loomc_result_t * result, artifact_selector selector, const char * format) {
    const loomc_string_view_t fmt = loomc_make_cstring_view(format);
    for (loomc_host_size_t i = 0; i < loomc_result_artifact_count(result); ++i) {
        const loomc_artifact_t * artifact = loomc_result_artifact_at(result, i);
        if (artifact != nullptr && artifact_matches(artifact, selector) && loomc_string_view_equal(artifact->format, fmt)) {
            return artifact;
        }
    }
    return nullptr;
}

static hrx_status_t copy_artifact(const loomc_artifact_t * artifact, std::vector<uint8_t> & out) {
    loomc_byte_span_t contents = loomc_byte_span_empty();
    const bool borrowed = loomc_byte_sequence_try_get_contiguous_span(artifact->contents, &contents);
    if (!borrowed) {
        loomc_status_t status = loomc_byte_sequence_clone(artifact->contents, loomc_allocator_system(), &contents);
        if (!loomc_status_is_ok(status)) {
            return status_from_loom(status, "copy Loom artifact");
        }
    }
    out.assign(contents.data, contents.data + contents.data_length);
    if (!borrowed) {
        loomc_allocator_free(loomc_allocator_system(), const_cast<uint8_t *>(contents.data));
    }
    return hrx_ok_status();
}

static hrx_status_t create_source(const ggml_hrx_loom_source & src, loomc_source_t ** out) {
    loomc_source_options_t options = {};
    options.type           = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS;
    options.structure_size = sizeof(options);
    options.format         = LOOMC_SOURCE_FORMAT_TEXT;
    options.identifier     = loomc_make_cstring_view(src.identifier);
    options.contents       = loomc_make_byte_span(src.data, src.size);
    options.storage        = LOOMC_SOURCE_STORAGE_BORROWED;
    return status_from_loom(loomc_source_create(&options, loomc_allocator_system(), out), "create Loom source");
}

static hrx_status_t index_source(loomc_link_index_builder_t * builder, loomc_source_t * source, const char * name) {
    loomc_link_index_source_options_t options = {};
    options.provider_name = loomc_make_cstring_view(name);
    options.role          = LOOMC_LINK_PROVIDER_ROLE_INPUT;
    return status_from_loom(loomc_link_index_builder_add_source(builder, source, &options, nullptr), "index Loom source");
}

static hrx_status_t finish_index(loomc_link_index_builder_t * builder, loom_link_index & index) {
    loom_result  result;
    hrx_status_t status = status_from_loom(loomc_link_index_builder_finish(builder, index.out(), result.out()), "finish Loom link index");
    if (!hrx_status_is_ok(status)) {
        return status;
    }
    return status_from_result(result.get(), "Loom source indexing failed");
}

// config.decl ops are not call edges. A selective link drops them. Thus link all library sources into one archive first, then write it back as text.
static hrx_status_t archive_sources(ggml_hrx_loom_jit * jit, loomc_linker_t * linker, loomc_workspace_t * workspace,
                                   const char * name, loom_link_index & index, loom_source & archived) {
    loom_module          module;
    loom_result          result;
    loomc_link_options_t options = {};
    options.type           = LOOMC_STRUCTURE_TYPE_LINK_OPTIONS;
    options.structure_size = sizeof(options);
    options.link_index     = index.get();
    options.module_name    = loomc_make_cstring_view(name);
    options.flags          = LOOMC_LINK_FLAG_STRIP_TEST_SYMBOLS;
    hrx_status_t status = status_from_loom(loomc_link_module(linker, workspace, &options, module.out(), result.out()), "archive Loom module");
    if (!hrx_status_is_ok(status)) {
        return status;
    }
    status = status_from_result(result.get(), "Loom archive link failed");
    if (!hrx_status_is_ok(status)) {
        return status;
    }

    loomc_module_serialize_options_t serialize = {};
    serialize.type           = LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS;
    serialize.structure_size = sizeof(serialize);
    serialize.format         = LOOMC_SOURCE_FORMAT_TEXT;
    serialize.identifier     = loomc_make_cstring_view(name);
    status = status_from_loom(loomc_module_serialize_to_source(module.get(), &serialize, loomc_allocator_system(), archived.out()),
                              "serialize Loom archive");
    if (!hrx_status_is_ok(status)) {
        return status;
    }

    loom_link_index_builder builder;
    status = status_from_loom(loomc_link_index_builder_allocate(jit->context, nullptr, loomc_allocator_system(), builder.out()),
                              "allocate Loom archive index");
    if (!hrx_status_is_ok(status)) {
        return status;
    }
    status = index_source(builder.get(), archived.get(), name);
    if (!hrx_status_is_ok(status)) {
        return status;
    }
    return finish_index(builder.get(), index);
}

} // namespace

hrx_status_t ggml_hrx_loom_jit_create(const char * processor, const char * kernel_target, ggml_hrx_loom_jit ** out_jit) {
    *out_jit = nullptr;
    if (processor == nullptr || processor[0] == '\0' || kernel_target == nullptr || kernel_target[0] == '\0') {
        return make_status(HRX_STATUS_INVALID_ARGUMENT, "Loom JIT needs a target processor");
    }
    std::unique_ptr<ggml_hrx_loom_jit> jit(new (std::nothrow) ggml_hrx_loom_jit());
    if (!jit) {
        return make_status(HRX_STATUS_OUT_OF_MEMORY, "allocate Loom JIT");
    }
    // Kernels and library providers use target.decl. The family records are requirements and do not follow the device.
    jit->targets = std::string("amdgpu.target<") + kernel_target + "> @ggml_wave32 {subgroup_size = 32}\n" +
                   "amdgpu.target<" + kernel_target + "> @ggml_wave64 {subgroup_size = 64}\n" +
                   "amdgpu.target<gfx11-generic> @ggml_gfx11\n" +
                   "amdgpu.target<gfx12-generic> @ggml_gfx12\n";

    hrx_status_t status = status_from_loom(loomc_target_environment_create_amdgpu(loomc_allocator_system(), &jit->target_environment),
                                           "create AMDGPU target environment");
    if (!hrx_status_is_ok(status)) {
        return status;
    }

    loomc_context_target_options_t target_options = {};
    target_options.type               = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
    target_options.structure_size     = sizeof(target_options);
    target_options.target_environment = jit->target_environment;
    loomc_context_options_t context_options = {};
    context_options.type           = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS;
    context_options.structure_size = sizeof(context_options);
    context_options.next           = &target_options;
    status = status_from_loom(loomc_context_create(&context_options, loomc_allocator_system(), &jit->context), "create Loom context");
    if (!hrx_status_is_ok(status)) {
        return status;
    }

    loomc_amdgpu_profile_options_t profile = {};
    profile.type            = LOOMC_STRUCTURE_TYPE_AMDGPU_PROFILE_OPTIONS;
    profile.structure_size  = sizeof(profile);
    profile.identifier      = loomc_make_cstring_view(processor);
    profile.identity.target = loomc_make_cstring_view(processor);
    status = status_from_loom(loomc_target_profile_create_amdgpu(jit->target_environment, &profile, loomc_allocator_system(), &jit->target_profile),
                              "create AMDGPU target profile");
    if (!hrx_status_is_ok(status)) {
        return status;
    }
    status = status_from_loom(loomc_compiler_create(jit->context, nullptr, loomc_allocator_system(), &jit->compiler), "create Loom compiler");
    if (!hrx_status_is_ok(status)) {
        return status;
    }

    loomc_target_pipeline_options_t pipeline = {};
    pipeline.type                     = LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS;
    pipeline.structure_size           = sizeof(pipeline);
    pipeline.identifier               = loomc_make_cstring_view("ggml-hrx-amdgpu");
    pipeline.kind                     = LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW;
    pipeline.control_flow_lowering    = LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG;
    pipeline.source_to_low_max_errors = 20;
    loom_result result;
    status = status_from_loom(loomc_pass_program_create_from_target_pipeline(jit->context, &pipeline, loomc_allocator_system(),
                                                                             &jit->pass_program, result.out()),
                              "create target pass program");
    if (!hrx_status_is_ok(status)) {
        return status;
    }
    status = status_from_result(result.get(), "target pass program failed");
    if (!hrx_status_is_ok(status)) {
        return status;
    }

    *out_jit = jit.release();
    return hrx_ok_status();
}

void ggml_hrx_loom_jit_free(ggml_hrx_loom_jit * jit) {
    delete jit;
}

hrx_status_t ggml_hrx_loom_compile(ggml_hrx_loom_jit * jit, const ggml_hrx_loom_compile_options * options,
                                   ggml_hrx_loom_compile_result * out_result) {
    out_result->hsaco.clear();
    out_result->launch = nullptr;
    if (options->source.data == nullptr || options->source.size == 0 || options->symbol == nullptr) {
        return make_status(HRX_STATUS_INVALID_ARGUMENT, "Loom compile needs a source and a symbol");
    }
    const char * symbol = options->symbol;

    loom_workspace workspace;
    hrx_status_t   status = status_from_loom(loomc_workspace_create(nullptr, loomc_allocator_system(), workspace.out()),
                                             "create Loom workspace");
    if (!hrx_status_is_ok(status)) {
        return status;
    }

    loom_link_index_builder builder;
    status = status_from_loom(loomc_link_index_builder_allocate(jit->context, nullptr, loomc_allocator_system(), builder.out()),
                              "allocate Loom link index");
    if (!hrx_status_is_ok(status)) {
        return status;
    }

    std::vector<ggml_hrx_loom_source> inputs = { options->source };
    inputs.insert(inputs.end(), options->deps, options->deps + options->n_deps);
    inputs.push_back({ jit->targets.data(), jit->targets.size(), "ggml-hrx-targets.loom" });

    // The sources must stay alive until the index is complete.
    std::vector<std::unique_ptr<loom_source>> sources;
    for (const ggml_hrx_loom_source & src : inputs) {
        sources.emplace_back(new loom_source());
        status = create_source(src, sources.back()->out());
        if (!hrx_status_is_ok(status)) {
            return status;
        }
        status = index_source(builder.get(), sources.back()->get(), src.identifier);
        if (!hrx_status_is_ok(status)) {
            return status;
        }
    }
    loom_link_index index;
    status = finish_index(builder.get(), index);
    if (!hrx_status_is_ok(status)) {
        return status;
    }

    loom_linker linker;
    status = status_from_loom(loomc_linker_create(jit->context, nullptr, loomc_allocator_system(), linker.out()), "create Loom linker");
    if (!hrx_status_is_ok(status)) {
        return status;
    }

    loom_source archived;
    status = archive_sources(jit, linker.get(), workspace.get(), options->source.identifier, index, archived);
    if (!hrx_status_is_ok(status)) {
        return status;
    }

    std::vector<loomc_config_binding_t> config(options->n_config);
    for (size_t i = 0; i < options->n_config; ++i) {
        config[i].key   = loomc_make_cstring_view(options->config[i].key);
        config[i].value = loomc_make_cstring_view(options->config[i].value);
    }

    const loomc_target_specialization_t specialization = { loomc_make_cstring_view(symbol), jit->target_profile };
    loomc_target_specialization_options_t target_options = {};
    target_options.type                 = LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS;
    target_options.structure_size       = sizeof(target_options);
    target_options.specializations      = &specialization;
    target_options.specialization_count = 1;

    const loomc_string_view_t roots[] = { loomc_make_cstring_view(symbol) };
    loomc_link_options_t link = {};
    link.type                  = LOOMC_STRUCTURE_TYPE_LINK_OPTIONS;
    link.structure_size        = sizeof(link);
    link.next                  = &target_options;
    link.mode                  = LOOMC_LINK_MODE_LINK;
    link.link_index            = index.get();
    link.module_name           = loomc_make_cstring_view(symbol);
    link.root_symbols          = roots;
    link.root_symbol_count     = 1;
    link.flags                 = LOOMC_LINK_FLAG_STRIP_TEST_SYMBOLS;
    link.config.bindings       = config.empty() ? nullptr : config.data();
    link.config.binding_count  = config.size();

    loom_module module;
    loom_result result;
    status = status_from_loom(loomc_link_module(linker.get(), workspace.get(), &link, module.out(), result.out()), "link Loom root");
    if (!hrx_status_is_ok(status)) {
        return status;
    }
    status = status_from_result(result.get(), "Loom root link failed");
    if (!hrx_status_is_ok(status)) {
        return status;
    }

    loomc_compile_options_t compile = {};
    compile.type           = LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS;
    compile.structure_size = sizeof(compile);
    compile.next           = &target_options;
    compile.module_name    = loomc_make_cstring_view(symbol);
    compile.artifact_flags = LOOMC_COMPILE_ARTIFACT_FLAG_LAUNCH_CONFIG;
    compile.config_flags   = LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED;
    status = status_from_loom(loomc_compile_module(jit->compiler, workspace.get(), jit->pass_program, module.get(), &compile,
                                                   loomc_allocator_system(), result.out()),
                              "compile Loom module");
    if (!hrx_status_is_ok(status)) {
        return status;
    }
    status = status_from_result(result.get(), "Loom compilation failed");
    if (!hrx_status_is_ok(status)) {
        return status;
    }

    const loomc_artifact_t * launch_artifact = find_artifact(result.get(), LOOM_ARTIFACT_LAUNCH_CONFIG, LOOMC_ARTIFACT_FORMAT_LOOM_BYTECODE);
    if (launch_artifact == nullptr) {
        return make_status(HRX_STATUS_NOT_FOUND, "Loom did not return a launch config");
    }
    std::unique_ptr<ggml_hrx_loom_launch_program> launch(new (std::nothrow) ggml_hrx_loom_launch_program());
    if (!launch) {
        return make_status(HRX_STATUS_OUT_OF_MEMORY, "allocate Loom launch program");
    }
    status = status_from_loom(loomc_launch_config_program_load(launch_artifact, loomc_allocator_system(), &launch->program),
                              "load Loom launch config");
    if (!hrx_status_is_ok(status)) {
        return status;
    }
    status = status_from_loom(loomc_launch_config_program_lookup_function(launch->program, loomc_make_cstring_view(symbol), &launch->function),
                              "find Loom launch config function");
    if (!hrx_status_is_ok(status)) {
        return status;
    }

    loomc_emit_options_t emit = {};
    emit.type            = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS;
    emit.structure_size  = sizeof(emit);
    emit.artifact_format = loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO);
    emit.identifier      = loomc_make_cstring_view(symbol);
    emit.artifact_flags  = LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY;
    status = status_from_loom(loomc_emit_module(jit->target_environment, workspace.get(), module.get(), &emit,
                                                loomc_allocator_system(), result.out()),
                              "emit AMDGPU code object");
    if (!hrx_status_is_ok(status)) {
        return status;
    }
    status = status_from_result(result.get(), "AMDGPU code object emission failed");
    if (!hrx_status_is_ok(status)) {
        return status;
    }
    const loomc_artifact_t * hsaco = find_artifact(result.get(), LOOM_ARTIFACT_KERNEL, LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO);
    if (hsaco == nullptr) {
        return make_status(HRX_STATUS_NOT_FOUND, "Loom did not return an AMDGPU code object");
    }
    status = copy_artifact(hsaco, out_result->hsaco);
    if (!hrx_status_is_ok(status)) {
        return status;
    }

    out_result->launch = launch.release();
    return hrx_ok_status();
}

void ggml_hrx_loom_launch_program_free(ggml_hrx_loom_launch_program * program) {
    delete program;
}

hrx_status_t ggml_hrx_loom_launch_program_eval(ggml_hrx_loom_launch_program * program, const int64_t * workload,
                                               size_t n_workload, ggml_hrx_loom_launch_config * out_config) {
    std::vector<uint64_t> bits(workload, workload + n_workload);

    loomc_launch_config_t config = {};
    config.type           = LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG;
    config.structure_size = sizeof(config);

    std::lock_guard<std::mutex> lock(program->mutex);
    auto it = program->results.find(bits);
    if (it != program->results.end()) {
        *out_config = it->second;
        return hrx_ok_status();
    }
    hrx_status_t status = status_from_loom(loomc_launch_config_program_invoke(program->program, program->function,
                                                                              bits.empty() ? nullptr : bits.data(), bits.size(), &config),
                                           "evaluate Loom launch config");
    if (!hrx_status_is_ok(status)) {
        return status;
    }
    if (!config.workgroup_count.x || !config.workgroup_count.y || !config.workgroup_count.z ||
        !config.workgroup_size.x  || !config.workgroup_size.y  || !config.workgroup_size.z) {
        return make_status(HRX_STATUS_FAILED_PRECONDITION, "Loom launch config has an empty workgroup count or size");
    }
    out_config->workgroup_count[0] = config.workgroup_count.x;
    out_config->workgroup_count[1] = config.workgroup_count.y;
    out_config->workgroup_count[2] = config.workgroup_count.z;
    out_config->workgroup_size[0]  = config.workgroup_size.x;
    out_config->workgroup_size[1]  = config.workgroup_size.y;
    out_config->workgroup_size[2]  = config.workgroup_size.z;
    out_config->subgroup_size      = config.subgroup_size;
    if (program->results.size() >= GGML_HRX_LOOM_MAX_LAUNCH_RESULTS) {
        program->results.clear();
    }
    program->results.emplace(std::move(bits), *out_config);
    return hrx_ok_status();
}
