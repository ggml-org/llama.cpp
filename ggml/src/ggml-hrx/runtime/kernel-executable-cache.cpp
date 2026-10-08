#include "kernel-executable-cache.h"

#include "ggml-impl.h"
#include "hrx-interop-utils.h"

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <map>
#include <sstream>
#include <utility>

namespace ggml::hrx {
namespace {

static ggml_hrx_loom_jit_source_format to_jit_source_format(KernelSourceFormat format) {
    switch (format) {
        case KERNEL_SOURCE_FORMAT_TEXT:
            return GGML_HRX_LOOM_JIT_SOURCE_FORMAT_TEXT;
        case KERNEL_SOURCE_FORMAT_BINARY:
            return GGML_HRX_LOOM_JIT_SOURCE_FORMAT_BYTECODE;
    }
    return GGML_HRX_LOOM_JIT_SOURCE_FORMAT_TEXT;
}

static void append_u32(std::vector<uint8_t> & bytes, uint32_t value) {
    const size_t offset = bytes.size();
    bytes.resize(offset + sizeof(value));
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

static void append_u64(std::vector<uint8_t> & bytes, uint64_t value) {
    const size_t offset = bytes.size();
    bytes.resize(offset + sizeof(value));
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

static bool pack_kernel_constants(const KernelDefinition & definition,
                                  const KernelSpecialization & specialization,
                                  size_t                       expected_size,
                                  std::vector<uint8_t> &       constants) {
    constants.clear();
    const size_t parameter_count = definition.launch_parameters.size();
    if (parameter_count == 0) {
        return expected_size == 0;
    }
    if (expected_size % parameter_count != 0) {
        GGML_LOG_ERROR("%s: launch scalar ABI size does not match manifest for %s\n", __func__,
                       kernel_definition_name(definition).c_str());
        return false;
    }
    const size_t scalar_size = expected_size / parameter_count;
    if (scalar_size != sizeof(uint32_t) && scalar_size != sizeof(uint64_t)) {
        GGML_LOG_ERROR("%s: unsupported launch scalar ABI width %zu for %s\n", __func__, scalar_size,
                       kernel_definition_name(definition).c_str());
        return false;
    }
    for (const KernelScalarDefinition & parameter : definition.launch_parameters) {
        const char * name = parameter.name != nullptr ? parameter.name : "";
        const char * type = parameter.type != nullptr ? parameter.type : "";
        const auto   item = specialization.integer_parameters.find(name);
        if (item == specialization.integer_parameters.end() || std::strcmp(type, "index") != 0 || item->second < 0 ||
            (scalar_size == sizeof(uint32_t) &&
             static_cast<uint64_t>(item->second) > std::numeric_limits<uint32_t>::max())) {
            constants.clear();
            GGML_LOG_ERROR("%s: invalid launch scalar %s for %s\n", __func__, name,
                           kernel_definition_name(definition).c_str());
            return false;
        }
        if (scalar_size == sizeof(uint32_t)) {
            append_u32(constants, static_cast<uint32_t>(item->second));
        } else {
            append_u64(constants, static_cast<uint64_t>(item->second));
        }
    }
    return true;
}

static bool build_workload_arguments(const KernelDefinition &       definition,
                                     const KernelSpecialization &   specialization,
                                     std::vector<int64_t> &           workload) {
    workload.clear();
    workload.reserve(definition.workload_parameters.size());
    for (const KernelScalarDefinition & parameter : definition.workload_parameters) {
        const char * name = parameter.name != nullptr ? parameter.name : "";
        const char * type = parameter.type != nullptr ? parameter.type : "";
        const auto   item = specialization.integer_parameters.find(name);
        if (item == specialization.integer_parameters.end() || std::strcmp(type, "index") != 0) {
            GGML_LOG_ERROR("%s: invalid workload scalar %s for %s\n", __func__, name,
                           kernel_definition_name(definition).c_str());
            workload.clear();
            return false;
        }
        workload.push_back(item->second);
    }
    return true;
}

static std::string kernel_executable_key(const KernelDefinition & definition,
                                         const Dispatch &         dispatch,
                                         const char *             target) {
    std::ostringstream out;
    out << (target != nullptr ? target : "") << '|' << definition.source_digest << '|' << definition.symbol
        << "|recipe=" << definition.compile_recipe.mode << "|workload="
        << (dispatch.kernel.workload_specialization == WorkloadSpecialization::Dynamic ? "dynamic" : "exact");
    if (dispatch.kernel.workload_specialization == WorkloadSpecialization::Exact) {
        for (const KernelScalarDefinition & parameter : definition.workload_parameters) {
            const char * name = parameter.name != nullptr ? parameter.name : "";
            const auto   item = dispatch.kernel.integer_parameters.find(name);
            out << '|' << name << '=';
            if (item == dispatch.kernel.integer_parameters.end()) {
                out << "<missing>";
            } else {
                out << item->second;
            }
        }
    }
    for (const KernelCompileConfig & config : definition.compile_config) {
        out << '|' << (config.key != nullptr ? config.key : "") << '=' << (config.value != nullptr ? config.value : "");
    }
    for (const auto & config : dispatch.kernel.compile_parameters) {
        out << '|' << config.first << '=' << config.second;
    }
    return out.str();
}

static bool build_compile_request(const KernelDefinition &   definition,
                                  const Dispatch &           dispatch,
                                  LoomKernelCompileRequest & request) {
    if (definition.compile_recipe.primary_sources.empty()) {
        GGML_LOG_ERROR("%s: kernel %s has no primary source\n", __func__, kernel_definition_name(definition).c_str());
        return false;
    }
    const KernelSourceRef & primary_source = definition.compile_recipe.primary_sources.front();
    const KernelSource *    source         = primary_source.contents;
    if (source == nullptr) {
        GGML_LOG_ERROR("%s: missing embedded source for %s\n", __func__, primary_source.path);
        return false;
    }

    request.source_data          = source->source.data;
    request.source_size          = source->source.length;
    request.source_format        = to_jit_source_format(source->source.format);
    request.source_identifier    = primary_source.path != nullptr ? primary_source.path : "";
    request.symbol               = definition.symbol != nullptr ? definition.symbol : "";
    request.launch_config_symbol = definition.name != nullptr ? definition.name : "";

    request.dependencies.reserve(definition.compile_recipe.library_sources.size());
    for (const KernelSourceRef & dependency_ref : definition.compile_recipe.library_sources) {
        const KernelSource * dependency = dependency_ref.contents;
        if (dependency == nullptr) {
            GGML_LOG_ERROR("%s: missing embedded dependency for %s\n", __func__, dependency_ref.path);
            return false;
        }
        request.dependencies.push_back({
            dependency->source.data,
            dependency->source.length,
            to_jit_source_format(dependency->source.format),
            dependency_ref.path,
        });
    }

    std::map<std::string, std::string> merged_configs;
    for (const KernelCompileConfig & config : definition.compile_config) {
        merged_configs[config.key != nullptr ? config.key : ""] = config.value != nullptr ? config.value : "";
    }
    for (const auto & config : dispatch.kernel.compile_parameters) {
        merged_configs[config.first] = config.second;
    }
    request.config_storage.reserve(merged_configs.size());
    for (const auto & config : merged_configs) {
        request.config_storage.push_back(config);
    }

    request.specialize_workload = dispatch.kernel.workload_specialization == WorkloadSpecialization::Exact;
    if (!build_workload_arguments(definition, dispatch.kernel, request.workload)) {
        return false;
    }
    return true;
}

static std::shared_ptr<KernelExecutable> load_kernel_executable(const KernelExecutablePrepareContext & context,
                                                                const KernelDefinition &               definition,
                                                                const Dispatch &                       dispatch,
                                                                const std::string &                    key,
                                                                ggml_hrx_loom_jit_compile_result &     compiled,
                                                                std::string &                          error_message) {
    if (context.device == nullptr) {
        error_message = "missing HRX device";
        GGML_LOG_ERROR("%s: load %s: %s\n", __func__, key.c_str(), error_message.c_str());
        return nullptr;
    }
    if (context.target == nullptr) {
        error_message = "missing HRX target";
        GGML_LOG_ERROR("%s: load %s: %s\n", __func__, key.c_str(), error_message.c_str());
        return nullptr;
    }

    auto executable            = std::make_shared<KernelExecutable>();
    executable->launch_program = std::exchange(compiled.launch_program, nullptr);
    if (ErrorResult error =
            take_status(hrx_executable_load_data(context.device, compiled.hsaco_data, compiled.hsaco_size, "amdgpu",
                                                 context.target, &executable->executable))) {
        error_message = "load " + key + ": " + *error;
        GGML_LOG_ERROR("%s: %s\n", __func__, error_message.c_str());
        return nullptr;
    }
    if (ErrorResult error = take_status(hrx_executable_lookup_export_by_name(executable->executable, definition.name,
                                                                             &executable->export_ordinal))) {
        error_message = "lookup " + key + ": " + *error;
        GGML_LOG_ERROR("%s: %s\n", __func__, error_message.c_str());
        return nullptr;
    }
    if (ErrorResult error = take_status(
            hrx_executable_export_info(executable->executable, executable->export_ordinal, &executable->export_info))) {
        error_message = "inspect " + key + ": " + *error;
        GGML_LOG_ERROR("%s: %s\n", __func__, error_message.c_str());
        return nullptr;
    }
    const size_t launch_parameter_count = definition.launch_parameters.size();
    const bool constant_abi_valid = launch_parameter_count == 0 ? executable->export_info.constant_byte_length == 0 :
                                                                  executable->export_info.constant_byte_length == launch_parameter_count * sizeof(uint32_t) ||
                                                                      executable->export_info.constant_byte_length == launch_parameter_count * sizeof(uint64_t);
    if (executable->launch_program == nullptr || executable->export_info.binding_count != dispatch.bindings.size() ||
        !constant_abi_valid ||
        executable->export_info.parameter_count != dispatch.bindings.size() + definition.launch_parameters.size()) {
        error_message = "compiled ABI does not match manifest for " + key;
        GGML_LOG_ERROR("%s: %s\n", __func__, error_message.c_str());
        return nullptr;
    }
    return executable;
}

}  // namespace

class KernelExecutableCacheEntry {
  public:
    KernelExecutableCacheEntry(std::string              key,
                               const KernelDefinition & definition,
                               const Dispatch &         dispatch,
                               LoomCompiledKernelRef    compiled_ref) :
        key(std::move(key)),
        definition(&definition),
        dispatch(dispatch),
        compiled_ref(std::move(compiled_ref)) {}

    std::mutex                        mutex;
    std::condition_variable           complete;
    std::string                       key;
    const KernelDefinition *          definition = nullptr;
    Dispatch                          dispatch;
    LoomCompiledKernelRef             compiled_ref;
    std::shared_ptr<KernelExecutable> executable;
    std::string                       error;

    enum class LoadState {
        Unloaded,
        Loading,
        Loaded,
        Failed,
    };

    std::atomic<LoadState> load_state = LoadState::Unloaded;
};

static bool prepare_kernel_invocation(const KernelExecutableCacheEntry & entry,
                                      const KernelExecutable &           executable,
                                      const KernelSpecialization &       specialization,
                                      std::vector<uint8_t> &             constants,
                                      ggml_hrx_loom_jit_launch_config &  launch) {
    std::vector<int64_t> workload;
    if (!build_workload_arguments(*entry.definition, specialization, workload)) {
        return false;
    }
    if (ErrorResult error = take_status(ggml_hrx_loom_jit_launch_program_evaluate(
            executable.launch_program, workload.empty() ? nullptr : workload.data(), workload.size(), &launch))) {
        GGML_LOG_ERROR("%s: evaluate launch %s: %s\n", __func__, entry.key.c_str(), error->c_str());
        return false;
    }
    return pack_kernel_constants(*entry.definition, specialization, executable.export_info.constant_byte_length,
                                 constants);
}

KernelExecutable::~KernelExecutable() {
    ggml_hrx_loom_jit_launch_program_release(launch_program);
    if (executable != nullptr) {
        hrx_executable_release(executable);
    }
}

KernelExecutableCache::KernelExecutableCache(LoomJitMode mode) : mode_(mode), mode_is_forced_(true) {}

KernelExecutableCache::~KernelExecutableCache() {
    clear();
}

bool KernelExecutableCache::ensure_jit_locked(const char * target, std::string & error_message) {
    if (target == nullptr || target[0] == '\0') {
        error_message = "missing HRX target";
        GGML_LOG_ERROR("%s: %s\n", __func__, error_message.c_str());
        return false;
    }
    if (jit_ != nullptr) {
        if (target_ != target) {
            error_message =
                "HRX kernel executable cache target mismatch: existing " + target_ + ", requested " + target;
            GGML_LOG_ERROR("%s: %s\n", __func__, error_message.c_str());
            return false;
        }
        return true;
    }

    jit_ = mode_is_forced_ ? create_loom_jit(target, mode_, error_message) : create_loom_jit(target, error_message);
    if (jit_ == nullptr) {
        if (error_message.empty()) {
            error_message = "create Loom JIT failed";
        }
        return false;
    }
    target_ = target;
    return true;
}

KernelExecutableRef KernelExecutableCache::get_or_compile(const KernelExecutablePrepareContext & context,
                                                          const KernelDefinition &               definition,
                                                          const Dispatch &                       dispatch) {
    KernelExecutableRef ref;

    const std::string        key = kernel_executable_key(definition, dispatch, context.target);
    LoomKernelCompileRequest request;
    LoomCompiledKernelRef    compiled_ref;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto                  found = cache_.find(key);
        if (found != cache_.end()) {
            ref.entry = found->second;
            return ref;
        }
        std::string error_message;
        if (!ensure_jit_locked(context.target, error_message)) {
            return ref;
        }
        if (!build_compile_request(definition, dispatch, request)) {
            return ref;
        }

        compiled_ref = jit_->compile(key, std::move(request));
        auto entry   = std::make_shared<KernelExecutableCacheEntry>(key, definition, dispatch, compiled_ref);
        cache_.emplace(key, entry);
        ref.entry = std::move(entry);
    }
    return ref;
}

std::shared_ptr<KernelExecutable> KernelExecutableCache::materialize(const KernelExecutablePrepareContext & context,
                                                                     const KernelExecutableRef &            ref,
                                                                     const KernelSpecialization & specialization,
                                                                     std::vector<uint8_t> &       constants,
                                                                     ggml_hrx_loom_jit_launch_config & launch) {
    if (!ref.valid()) {
        return nullptr;
    }

    KernelExecutableCacheEntry &          entry      = *ref.entry;
    KernelExecutableCacheEntry::LoadState load_state = entry.load_state.load(std::memory_order_acquire);
    if (load_state == KernelExecutableCacheEntry::LoadState::Loaded) {
        std::shared_ptr<KernelExecutable> executable =
            std::atomic_load_explicit(&entry.executable, std::memory_order_acquire);
        return executable != nullptr && prepare_kernel_invocation(entry, *executable, specialization, constants, launch)
                   ? executable
                   : nullptr;
    }
    if (load_state == KernelExecutableCacheEntry::LoadState::Failed) {
        std::lock_guard<std::mutex> entry_lock(entry.mutex);
        GGML_LOG_ERROR("%s: %s\n", __func__, entry.error.c_str());
        return nullptr;
    }

    KernelExecutableCacheEntry::LoadState expected = KernelExecutableCacheEntry::LoadState::Unloaded;
    if (!entry.load_state.compare_exchange_strong(expected, KernelExecutableCacheEntry::LoadState::Loading,
                                                  std::memory_order_acq_rel, std::memory_order_acquire)) {
        std::unique_lock<std::mutex> entry_lock(entry.mutex);
        entry.complete.wait(entry_lock, [&] {
            const KernelExecutableCacheEntry::LoadState current = entry.load_state.load(std::memory_order_acquire);
            return current == KernelExecutableCacheEntry::LoadState::Loaded ||
                   current == KernelExecutableCacheEntry::LoadState::Failed;
        });
        if (entry.load_state.load(std::memory_order_acquire) == KernelExecutableCacheEntry::LoadState::Failed) {
            GGML_LOG_ERROR("%s: %s\n", __func__, entry.error.c_str());
            return nullptr;
        }
        std::shared_ptr<KernelExecutable> executable =
            std::atomic_load_explicit(&entry.executable, std::memory_order_acquire);
        return executable != nullptr && prepare_kernel_invocation(entry, *executable, specialization, constants, launch)
                   ? executable
                   : nullptr;
    }

    LoomCompiledKernelRef compiled_ref;
    {
        std::lock_guard<std::mutex> entry_lock(entry.mutex);
        compiled_ref = entry.compiled_ref;
    }

    std::string error_message;
    if (compiled_ref == nullptr || !compiled_ref->resolve()) {
        error_message = compiled_ref ? compiled_ref->error_message() : "missing compiled kernel";
        GGML_LOG_ERROR("%s: %s\n", __func__, error_message.c_str());

        {
            std::lock_guard<std::mutex> entry_lock(entry.mutex);
            entry.error = error_message;
            entry.load_state.store(KernelExecutableCacheEntry::LoadState::Failed, std::memory_order_release);
        }
        entry.complete.notify_all();

        std::lock_guard<std::mutex> lock(mutex_);
        const auto                  found = cache_.find(entry.key);
        if (found != cache_.end() && found->second == ref.entry) {
            cache_.erase(found);
        }
        return nullptr;
    }

    ggml_hrx_loom_jit_compile_result  compiled   = compiled_ref->take_result();
    std::shared_ptr<KernelExecutable> executable = load_kernel_executable(
        context, *entry.definition, entry.dispatch, entry.key, compiled, error_message);

    if (executable != nullptr) {
        compiled.reset();
        {
            std::lock_guard<std::mutex> entry_lock(entry.mutex);
            entry.compiled_ref.reset();
            std::atomic_store_explicit(&entry.executable, executable, std::memory_order_release);
            entry.load_state.store(KernelExecutableCacheEntry::LoadState::Loaded, std::memory_order_release);
        }
    } else {
        {
            std::lock_guard<std::mutex> entry_lock(entry.mutex);
            entry.error = std::move(error_message);
            entry.load_state.store(KernelExecutableCacheEntry::LoadState::Failed, std::memory_order_release);
        }
    }
    entry.complete.notify_all();

    if (executable == nullptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto                  found = cache_.find(entry.key);
        if (found != cache_.end() && found->second == ref.entry) {
            cache_.erase(found);
        }
    }
    if (executable != nullptr && !prepare_kernel_invocation(entry, *executable, specialization, constants, launch)) {
        return nullptr;
    }
    return executable;
}

std::shared_ptr<KernelExecutable> KernelExecutableCache::prepare(const KernelExecutablePrepareContext & context,
                                                                 const KernelDefinition &               definition,
                                                                 const Dispatch &                       dispatch,
                                                                 std::vector<uint8_t> &                 constants,
                                                                 ggml_hrx_loom_jit_launch_config &      launch) {
    const KernelExecutableRef ref = get_or_compile(context, definition, dispatch);
    return materialize(context, ref, dispatch.kernel, constants, launch);
}

void KernelExecutableCache::clear() {
    if (jit_ != nullptr) {
        jit_->clear();
    }
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
    jit_.reset();
    target_.clear();
}

}  // namespace ggml::hrx
