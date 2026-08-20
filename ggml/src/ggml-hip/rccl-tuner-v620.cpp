#include "rccl-tuner-v6.h"
#include "rccl-tuner-policy.h"

#include <hip/hip_runtime_api.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

struct context {
    bool eligible = false;
    bool applied = false;
    size_t ranks = 0;
    int channels = 0;
};

bool env_present(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && value[0] != '\0';
}

bool v620_topology(size_t ranks) {
    int count = 0;
    if (ranks < 2 || ranks > 8 || hipGetDeviceCount(&count) != hipSuccess || count < static_cast<int>(ranks) || count > 8) {
        return false;
    }

    for (int i = 0; i < count; ++i) {
        hipDeviceProp_t prop{};
        if (hipGetDeviceProperties(&prop, i) != hipSuccess) {
            return false;
        }
        if (std::strncmp(prop.gcnArchName, "gfx1030", 7) != 0 ||
                std::strncmp(prop.name, "AMD Radeon Pro V620", 20) != 0) {
            return false;
        }
    }
    return true;
}

ggml_rdna2_rccl_tune_mode get_mode() {
    const char * mode = std::getenv("GGML_HIP_RCCL_TUNE");
    if (mode != nullptr && std::strcmp(mode, "force") == 0) {
        return ggml_rdna2_rccl_tune_mode::force;
    }
    if (mode == nullptr || std::strcmp(mode, "auto") == 0) {
        return ggml_rdna2_rccl_tune_mode::automatic;
    }
    return ggml_rdna2_rccl_tune_mode::off;
}

bool certified_pcie_hop2_topology(size_t ranks) {
    if (ranks != 4) {
        return false;
    }
    for (int src = 0; src < 4; ++src) {
        for (int dst = 0; dst < 4; ++dst) {
            if (src == dst) {
                continue;
            }
            int can_access = 0;
            unsigned link_type = 0;
            unsigned hop_count = 0;
            if (hipDeviceCanAccessPeer(&can_access, src, dst) != hipSuccess || !can_access ||
                    hipExtGetLinkTypeAndHopCount(src, dst, &link_type, &hop_count) != hipSuccess ||
                    link_type != 2 || hop_count != 2) {
                return false;
            }
        }
    }
    return true;
}

ncclResult_t plugin_init(void ** out, uint64_t, size_t ranks, size_t nodes,
                         ncclDebugLogger_t, ncclNvlDomainInfo_v5_t *, ncclTunerConstants_v6_t *) {
    auto * ctx = new context;
    const bool conflicting_env = env_present("NCCL_ALGO") || env_present("NCCL_PROTO") ||
        env_present("NCCL_MIN_NCHANNELS") || env_present("NCCL_MAX_NCHANNELS") ||
        env_present("NCCL_NTHREADS");
    ctx->ranks = ranks;
    ctx->channels = static_cast<int>(ranks < 3 ? ranks : 3);
    const ggml_rdna2_rccl_tune_mode mode = get_mode();
    const bool all_v620 = v620_topology(ranks);
    const bool pcie_hop2 = certified_pcie_hop2_topology(ranks);
    ctx->eligible = ggml_rdna2_rccl_policy_eligible({
        mode, ranks, nodes, all_v620, pcie_hop2, conflicting_env,
    });

    const char * mode_name = mode == ggml_rdna2_rccl_tune_mode::force ? "force" :
        mode == ggml_rdna2_rccl_tune_mode::automatic ? "auto" : "off";
    std::fprintf(stderr, "[rdna2-tuner] v6 loaded mode=%s ranks=%zu nodes=%zu channels=%d topology=%s eligible=%d\n",
                 mode_name, ranks, nodes, ctx->channels,
                 pcie_hop2 ? "pcie-hop2" : "other", ctx->eligible ? 1 : 0);
    *out = ctx;
    return ncclSuccess;
}

ncclResult_t plugin_get_coll_info(void * opaque, ncclFunc_t coll_type, size_t bytes,
                                  int, float ** costs, int num_algo, int num_proto, int,
                                  int * channels) {
    auto * ctx = static_cast<context *>(opaque);
    if (ctx == nullptr || channels == nullptr || costs == nullptr) {
        return ncclInternalError;
    }

    *channels = 0;
    if (!ctx->eligible || coll_type != ncclFuncAllReduce || bytes != 20480 ||
            num_algo <= NCCL_ALGO_RING || num_proto <= NCCL_PROTO_LL) {
        return ncclSuccess;
    }

    auto table = reinterpret_cast<float (*)[NCCL_NUM_PROTOCOLS_V5]>(costs);
    if (table[NCCL_ALGO_RING][NCCL_PROTO_LL] == NCCL_ALGO_PROTO_IGNORE) {
        return ncclSuccess;
    }

    table[NCCL_ALGO_RING][NCCL_PROTO_LL] = 0.0f;
    *channels = ctx->channels;
    if (!ctx->applied) {
        std::fprintf(stderr, "[rdna2-tuner] applied allreduce bytes=%zu algo=Ring proto=LL channels=%d ranks=%zu\n", bytes, ctx->channels, ctx->ranks);
        ctx->applied = true;
    }
    return ncclSuccess;
}

ncclResult_t plugin_finalize(void * opaque) {
    delete static_cast<context *>(opaque);
    return ncclSuccess;
}

} // namespace

extern "C" const ncclTuner_v6_t ncclTunerPlugin_v6 = {
    "rdna2-v620-hot-size",
    plugin_init,
    plugin_get_coll_info,
    plugin_finalize,
    nullptr,
};
