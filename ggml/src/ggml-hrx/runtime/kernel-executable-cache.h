#pragma once

#include "dispatch/dispatch.h"
#include "hrx_runtime.h"
#include "kernel-corpus/kernel-corpus.h"
#include "runtime/loom-kernel-jit.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace ggml::hrx {

class KernelExecutableCacheEntry;

struct KernelExecutable {
    ~KernelExecutable();

    hrx_executable_t                executable     = nullptr;
    uint32_t                        export_ordinal = 0;
    hrx_executable_export_info_t    export_info    = {};
    ggml_hrx_loom_jit_launch_config launch;
};

struct KernelExecutablePrepareContext {
    hrx_device_t device = nullptr;
    const char * target = nullptr;
};

struct KernelExecutableRef {
    std::shared_ptr<KernelExecutableCacheEntry> entry;

    bool valid() const { return entry != nullptr; }
};

class KernelExecutableCache {
  public:
    KernelExecutableCache() = default;
    explicit KernelExecutableCache(LoomJitMode mode);
    ~KernelExecutableCache();

    KernelExecutableRef get_or_compile(const KernelExecutablePrepareContext & context,
                                       const KernelDefinition &               definition,
                                       const Dispatch &                       dispatch,
                                       std::vector<uint8_t> &                 constants);

    std::shared_ptr<KernelExecutable> materialize(const KernelExecutablePrepareContext & context,
                                                  const KernelExecutableRef &            ref,
                                                  const std::vector<uint8_t> &           constants);

    std::shared_ptr<KernelExecutable> prepare(const KernelExecutablePrepareContext & context,
                                              const KernelDefinition &               definition,
                                              const Dispatch &                       dispatch,
                                              std::vector<uint8_t> &                 constants);

    void clear();

  private:
    bool ensure_jit_locked(const char * target, std::string & error_message);

    std::mutex                                                                   mutex_;
    std::unordered_map<std::string, std::shared_ptr<KernelExecutableCacheEntry>> cache_;
    std::unique_ptr<LoomJit>                                                     jit_;
    std::string                                                                  target_;
    LoomJitMode                                                                  mode_           = LoomJitMode::Async;
    bool                                                                         mode_is_forced_ = false;
};

}  // namespace ggml::hrx
