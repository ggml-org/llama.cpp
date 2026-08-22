#include "loom-kernel-jit.h"

#include "ggml-impl.h"
#include "hrx-interop-utils.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <thread>
#include <utility>

namespace ggml::hrx {
namespace {

class LoomAmdgpuJit {
  public:
    LoomAmdgpuJit()                                  = default;
    LoomAmdgpuJit(const LoomAmdgpuJit &)             = delete;
    LoomAmdgpuJit & operator=(const LoomAmdgpuJit &) = delete;

    ~LoomAmdgpuJit() { reset(); }

    bool create(const char * target, std::string & error_message) {
        reset();
        ggml_hrx_loom_jit_amdgpu_options options = {};
        options.processor                        = target;
        options.identifier                       = target;
        if (ErrorResult error = take_status(ggml_hrx_loom_jit_amdgpu_create(&options, &jit_))) {
            error_message = "create Loom JIT: " + *error;
            GGML_LOG_ERROR("%s: %s\n", __func__, error_message.c_str());
            return false;
        }
        return true;
    }

    ggml_hrx_loom_jit_amdgpu * get() const { return jit_; }

  private:
    void reset() {
        if (jit_ != nullptr) {
            ggml_hrx_loom_jit_amdgpu_release(jit_);
            jit_ = nullptr;
        }
    }

    ggml_hrx_loom_jit_amdgpu * jit_ = nullptr;
};

static size_t default_worker_count() {
    const unsigned int hardware_threads = std::thread::hardware_concurrency();
    if (hardware_threads == 0) {
        return 1;
    }
    return std::min<size_t>(hardware_threads, 4);
}

static bool compile_kernel(ggml_hrx_loom_jit_amdgpu *         jit,
                           const LoomKernelCompileRequest &   request,
                           const std::string &                key,
                           ggml_hrx_loom_jit_compile_result & compiled,
                           std::string &                      error_message) {
    std::vector<ggml_hrx_loom_jit_config_binding> configs;
    configs.reserve(request.config_storage.size());
    for (const auto & config : request.config_storage) {
        configs.push_back({ config.first.c_str(), config.second.c_str() });
    }

    ggml_hrx_loom_jit_compile_options compile_options = {};
    compile_options.source_data                       = request.source_data;
    compile_options.source_size                       = request.source_size;
    compile_options.source_format                     = request.source_format;
    compile_options.source_identifier                 = request.source_identifier.c_str();
    compile_options.root_symbol                       = request.symbol.c_str();
    compile_options.launch_config_symbol              = request.launch_config_symbol.c_str();
    compile_options.module_name                       = request.symbol.c_str();
    compile_options.artifact_identifier               = request.symbol.c_str();
    compile_options.dependencies                      = request.dependencies.data();
    compile_options.dependency_count                  = request.dependencies.size();
    compile_options.config_bindings                   = configs.data();
    compile_options.config_binding_count              = configs.size();
    compile_options.workload_arguments                = request.workload.data();
    compile_options.workload_argument_count           = request.workload.size();
    compile_options.evaluate_launch_config            = true;

    if (ErrorResult error = take_status(ggml_hrx_loom_jit_amdgpu_compile(jit, &compile_options, &compiled))) {
        error_message = "compile " + key + ": " + *error;
        GGML_LOG_ERROR("%s: %s\n", __func__, error_message.c_str());
        return false;
    }
    return true;
}

static bool is_disabled_value(const char * value) {
    if (value == nullptr) {
        return false;
    }
    return std::strcmp(value, "0") == 0 || std::strcmp(value, "false") == 0 || std::strcmp(value, "FALSE") == 0 ||
           std::strcmp(value, "off") == 0 || std::strcmp(value, "OFF") == 0;
}

}  // namespace

class LoomSyncJit final : public LoomJit {
  public:
    LoomSyncJit(const char * target, std::string & error_message) { valid_ = jit_.create(target, error_message); }

    LoomCompiledKernelRef compile(std::string key, LoomKernelCompileRequest request) override {
        auto compiled_ref = std::make_shared<LoomCompiledKernel>(std::move(key), std::move(request));
        if (!valid_) {
            compiled_ref->complete({}, false, "Loom JIT is not initialized");
            return compiled_ref;
        }
        compile_ref(*compiled_ref, jit_.get());
        return compiled_ref;
    }

    bool async_enabled() const override { return false; }

  private:
    static void compile_ref(LoomCompiledKernel & compiled_ref, ggml_hrx_loom_jit_amdgpu * jit) {
        ggml_hrx_loom_jit_compile_result compiled;
        std::string                      error_message;
        const bool success = compile_kernel(jit, compiled_ref.request(), compiled_ref.key(), compiled, error_message);
        compiled_ref.complete(std::move(compiled), success, std::move(error_message));
    }

    LoomAmdgpuJit jit_;
    bool          valid_ = false;
};

class LoomAsyncJit final : public LoomJit {
  public:
    LoomAsyncJit(const char * target, std::string & error_message) : target_(target != nullptr ? target : "") {
        if (target_.empty()) {
            error_message = "missing HRX target";
            return;
        }
        valid_ = true;
    }

    ~LoomAsyncJit() override { clear(); }

    LoomCompiledKernelRef compile(std::string key, LoomKernelCompileRequest request) override {
        auto compiled_ref = std::make_shared<LoomCompiledKernel>(std::move(key), std::move(request));
        if (!valid_) {
            compiled_ref->complete({}, false, "Loom JIT is not initialized");
            return compiled_ref;
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            start_workers_locked();
            pending_.push_back(compiled_ref);
        }
        work_available_.notify_one();
        return compiled_ref;
    }

    void clear() override { stop_workers(); }

    bool async_enabled() const override { return true; }

  private:
    void start_workers_locked() {
        if (!workers_.empty()) {
            return;
        }
        shutdown_          = false;
        const size_t count = default_worker_count();
        workers_.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    void stop_workers() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (workers_.empty()) {
                return;
            }
            shutdown_ = true;
        }
        work_available_.notify_all();
        for (std::thread & worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            pending_.clear();
            workers_.clear();
            shutdown_ = false;
        }
    }

    void worker_loop() {
        LoomAmdgpuJit worker_jit;
        std::string   jit_error;
        const bool    jit_ready = worker_jit.create(target_.c_str(), jit_error);
        for (;;) {
            LoomCompiledKernelRef compiled_ref;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                work_available_.wait(lock, [&] { return shutdown_ || !pending_.empty(); });
                if (shutdown_ && pending_.empty()) {
                    return;
                }
                compiled_ref = std::move(pending_.front());
                pending_.pop_front();
            }

            if (!jit_ready) {
                compiled_ref->complete({}, false, jit_error);
                continue;
            }
            compile_ref(*compiled_ref, worker_jit.get());
        }
    }

    static void compile_ref(LoomCompiledKernel & compiled_ref, ggml_hrx_loom_jit_amdgpu * jit) {
        ggml_hrx_loom_jit_compile_result compiled;
        std::string                      error_message;
        const bool success = compile_kernel(jit, compiled_ref.request(), compiled_ref.key(), compiled, error_message);
        compiled_ref.complete(std::move(compiled), success, std::move(error_message));
    }

    const std::string                 target_;
    bool                              valid_ = false;
    std::mutex                        mutex_;
    std::condition_variable           work_available_;
    std::deque<LoomCompiledKernelRef> pending_;
    std::vector<std::thread>          workers_;
    bool                              shutdown_ = false;
};

bool loom_async_jit_enabled_from_environment() {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_HRX_ASYNC_JIT");
        return !is_disabled_value(value);
    }();
    return enabled;
}

std::unique_ptr<LoomJit> create_loom_jit(const char * target, LoomJitMode mode, std::string & error_message) {
    if (target == nullptr || target[0] == '\0') {
        error_message = "missing HRX target";
        GGML_LOG_ERROR("%s: %s\n", __func__, error_message.c_str());
        return nullptr;
    }
    if (mode == LoomJitMode::Async) {
        auto jit = std::make_unique<LoomAsyncJit>(target, error_message);
        if (!error_message.empty()) {
            GGML_LOG_ERROR("%s: %s\n", __func__, error_message.c_str());
            return nullptr;
        }
        return jit;
    }
    auto jit = std::make_unique<LoomSyncJit>(target, error_message);
    if (!error_message.empty()) {
        return nullptr;
    }
    return jit;
}

std::unique_ptr<LoomJit> create_loom_jit(const char * target, std::string & error_message) {
    return create_loom_jit(target, loom_async_jit_enabled_from_environment() ? LoomJitMode::Async : LoomJitMode::Sync,
                           error_message);
}

LoomCompiledKernel::LoomCompiledKernel(std::string key, LoomKernelCompileRequest request) :
    key_(std::move(key)),
    request_(std::move(request)) {}

bool LoomCompiledKernel::resolve() const {
    const State current = state();
    if (current != State::Pending) {
        return current == State::Succeeded;
    }

    std::unique_lock<std::mutex> lock(mutex_);
    complete_.wait(lock, [&] { return state() != State::Pending; });
    return state() == State::Succeeded;
}

std::string LoomCompiledKernel::error_message() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return error_;
}

ggml_hrx_loom_jit_compile_result LoomCompiledKernel::take_result() {
    std::lock_guard<std::mutex> lock(mutex_);
    return std::move(compiled_);
}

void LoomCompiledKernel::complete(ggml_hrx_loom_jit_compile_result compiled, bool success, std::string error) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        compiled_ = std::move(compiled);
        error_    = std::move(error);
        state_.store(success ? State::Succeeded : State::Failed, std::memory_order_release);
    }
    complete_.notify_all();
}

}  // namespace ggml::hrx
