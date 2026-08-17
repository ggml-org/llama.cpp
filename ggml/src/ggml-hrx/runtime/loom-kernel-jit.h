#pragma once

#include "loom-jit.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace ggml::hrx {

struct LoomKernelCompileRequest {
    const void *                                     source_data   = nullptr;
    size_t                                           source_size   = 0;
    ggml_hrx_loom_jit_source_format                  source_format = GGML_HRX_LOOM_JIT_SOURCE_FORMAT_TEXT;
    std::string                                      source_identifier;
    std::string                                      symbol;
    std::string                                      launch_config_symbol;
    std::vector<ggml_hrx_loom_jit_source>            dependencies;
    std::vector<std::pair<std::string, std::string>> config_storage;
    std::vector<int64_t>                             workload;
};

class LoomCompiledKernel {
  public:
    LoomCompiledKernel(std::string key, LoomKernelCompileRequest request);

    LoomCompiledKernel(const LoomCompiledKernel &)             = delete;
    LoomCompiledKernel & operator=(const LoomCompiledKernel &) = delete;

    const std::string & key() const { return key_; }

    bool                             resolve() const;
    std::string                      error_message() const;
    ggml_hrx_loom_jit_compile_result take_result();

  private:
    friend class LoomSyncJit;
    friend class LoomAsyncJit;

    const LoomKernelCompileRequest & request() const { return request_; }

    void complete(ggml_hrx_loom_jit_compile_result compiled, bool success, std::string error);

    enum class State {
        Pending,
        Succeeded,
        Failed,
    };

    State state() const { return state_.load(std::memory_order_acquire); }

    mutable std::mutex               mutex_;
    mutable std::condition_variable  complete_;
    std::string                      key_;
    LoomKernelCompileRequest         request_;
    ggml_hrx_loom_jit_compile_result compiled_;
    std::string                      error_;
    std::atomic<State>               state_ = State::Pending;
};

using LoomCompiledKernelRef = std::shared_ptr<LoomCompiledKernel>;

class LoomJit {
  public:
    virtual ~LoomJit() = default;

    LoomJit(const LoomJit &)             = delete;
    LoomJit & operator=(const LoomJit &) = delete;

    virtual LoomCompiledKernelRef compile(std::string key, LoomKernelCompileRequest request) = 0;

    virtual void clear() {}

    virtual bool async_enabled() const = 0;

  protected:
    LoomJit() = default;
};

enum class LoomJitMode {
    Sync,
    Async,
};

std::unique_ptr<LoomJit> create_loom_jit(const char * target, LoomJitMode mode, std::string & error_message);
std::unique_ptr<LoomJit> create_loom_jit(const char * target, std::string & error_message);
bool                     loom_async_jit_enabled_from_environment();

}  // namespace ggml::hrx
