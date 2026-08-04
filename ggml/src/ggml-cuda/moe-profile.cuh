#pragma once

#include <cstdlib>
#include <cstring>

#if defined(GGML_CUDA_MOE_PROFILE_NVTX3)
#    include <nvtx3/nvToolsExt.h>
#elif defined(GGML_CUDA_MOE_PROFILE_NVTOOLSEXT)
#    include <nvToolsExt.h>
#else
#    error "GGML_CUDA_MOE_PROFILE requires an NVTX header selection"
#endif

static inline bool ggml_cuda_moe_profile_enabled() {
    static const bool enabled = []() {
        const char * value = std::getenv("GGML_CUDA_MOE_PROFILE");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

class ggml_cuda_moe_profile_scope {
  public:
    explicit ggml_cuda_moe_profile_scope(const char * name, bool condition = true) :
        active(condition && ggml_cuda_moe_profile_enabled()) {
        if (active) {
            nvtxRangePushA(name);
        }
    }

    ~ggml_cuda_moe_profile_scope() {
        if (active) {
            nvtxRangePop();
        }
    }

    ggml_cuda_moe_profile_scope(const ggml_cuda_moe_profile_scope &)             = delete;
    ggml_cuda_moe_profile_scope & operator=(const ggml_cuda_moe_profile_scope &) = delete;

  private:
    bool active;
};
