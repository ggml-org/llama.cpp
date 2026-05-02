#include "fit.h"

#include "llama.h"
#include "log.h"

#include "../src/llama-ext.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <cinttypes>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#   ifndef WIN32_LEAN_AND_MEAN
#       define WIN32_LEAN_AND_MEAN
#   endif
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#elif defined(__linux__)
#   include <unistd.h>
#elif defined(__APPLE__) && defined(__MACH__)
#   include <mach/mach.h>
#   include <sys/sysctl.h>
#endif

// this enum is only used in llama_params_fit_impl but needs to be defined outside of it to fix a Windows compilation issue
// enum to identify part of a layer for distributing its tensors:
enum common_layer_fraction_t {
    LAYER_FRACTION_NONE = 0, // nothing
    LAYER_FRACTION_ATTN = 1, // attention
    LAYER_FRACTION_UP   = 2, // attention + up
    LAYER_FRACTION_GATE = 3, // attention + up + gate
    LAYER_FRACTION_MOE  = 4, // everything but sparse MoE weights
};

class common_params_fit_exception : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

static constexpr int64_t COMMON_FIT_MIB = 1024LL*1024LL;
static constexpr int64_t COMMON_FIT_GIB = 1024LL*COMMON_FIT_MIB;

static std::string common_fit_lower(std::string s) {
    for (char & c : s) {
        c = char(std::tolower((unsigned char) c));
    }
    return s;
}

static bool common_fit_contains(const std::string & text, const char * needle) {
    return text.find(needle) != std::string::npos;
}

static bool common_fit_env_flag_enabled(const char * name) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return false;
    }

    const std::string normalized = common_fit_lower(value);
    return normalized != "0" && normalized != "false" && normalized != "no" && normalized != "off";
}

static bool common_fit_totals_are_close(const size_t a, const size_t b) {
    if (a == 0 || b == 0) {
        return false;
    }

    const long double lo = std::min(a, b);
    const long double hi = std::max(a, b);
    return lo / hi >= 0.90L;
}

static std::string common_fit_dev_text(ggml_backend_dev_t dev) {
    if (dev == nullptr) {
        return "";
    }

    const char * name_c = ggml_backend_dev_name(dev);
    const char * desc_c = ggml_backend_dev_description(dev);
    const std::string name = common_fit_lower(name_c ? name_c : "");
    const std::string desc = common_fit_lower(desc_c ? desc_c : "");
    return name + " " + desc;
}

static bool common_fit_dev_overlaps_host_budget(
        ggml_backend_dev_t dev,
        const llama_device_memory_data & dmd_dev,
        const llama_device_memory_data & dmd_host) {
    if (dev == nullptr || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        return false;
    }

    // CUDA unified memory intentionally makes device allocations compete with the
    // system-memory pool. Treat it like an overlapping budget even on discrete GPUs.
    if (common_fit_env_flag_enabled("GGML_CUDA_ENABLE_UNIFIED_MEMORY")) {
        return true;
    }

    const std::string text = common_fit_dev_text(dev);

    // Backends do not currently expose a portable "this heap is system backed"
    // flag here, so keep the explicit part conservative and vendor-neutral.
    // The generic "Graphics" suffix alone is not enough: some discrete GPUs use
    // broad marketing names. It becomes a UMA signal only when it is paired with
    // an APU/iGPU string or a device budget that is essentially the host budget.
    const bool text_declares_uma =
        common_fit_contains(text, "integrated") ||
        common_fit_contains(text, "igpu")       ||
        common_fit_contains(text, "apu")        ||
        common_fit_contains(text, "uma")        ||
        common_fit_contains(text, "unified")    ||
        common_fit_contains(text, "shared")     ||
        common_fit_contains(text, "strx_halo")  ||
        common_fit_contains(text, "strix_halo") ||
        common_fit_contains(text, "strix halo") ||
        common_fit_contains(text, "uhd graphics") ||
        common_fit_contains(text, "iris")       ||
        common_fit_contains(text, "xe graphics") ||
        common_fit_contains(text, "radeon 8060s") ||
        common_fit_contains(text, "radeon 8050s") ||
        common_fit_contains(text, "radeon 890m")  ||
        common_fit_contains(text, "radeon 880m")  ||
        common_fit_contains(text, "radeon 780m")  ||
        common_fit_contains(text, "radeon 680m");

    if (text_declares_uma) {
        return true;
    }

    const bool device_budget_looks_like_host_budget = common_fit_totals_are_close(dmd_dev.total, dmd_host.total);
    const bool generic_integrated_name =
        common_fit_contains(text, "graphics") &&
        (common_fit_contains(text, "intel") || common_fit_contains(text, "radeon") || common_fit_contains(text, "amd"));

    return device_budget_looks_like_host_budget && generic_integrated_name;
}

static int64_t common_fit_memory_total_i64(const llama_memory_breakdown_data & mb) {
    return int64_t(mb.total());
}

static int64_t common_fit_clamp_i64(const int64_t value, const int64_t lo, const int64_t hi) {
    return std::max(lo, std::min(value, hi));
}

struct common_fit_host_memory_status {
    size_t available = 0;
    size_t total     = 0;
    size_t watermark = 0;
    const char * source = "";
};

#if defined(__linux__)
static size_t common_fit_get_linux_zone_high_watermark() {
    std::ifstream zoneinfo("/proc/zoneinfo");
    if (!zoneinfo.is_open()) {
        return 0;
    }

    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        return 0;
    }

    uint64_t high_pages = 0;
    std::string key;
    uint64_t value = 0;
    while (zoneinfo >> key) {
        if (key == "high" && (zoneinfo >> value)) {
            high_pages += value;
        }
    }

    return size_t(high_pages * uint64_t(page_size));
}
#endif

static bool common_fit_get_os_host_memory(common_fit_host_memory_status & status) {
#if defined(_WIN32)
    MEMORYSTATUSEX statex;
    statex.dwLength = sizeof(statex);
    if (!GlobalMemoryStatusEx(&statex)) {
        return false;
    }
    status.available = size_t(statex.ullAvailPhys);
    status.total     = size_t(statex.ullTotalPhys);
    status.source    = "GlobalMemoryStatusEx";
    return status.available > 0 && status.total > 0;
#elif defined(__linux__)
    std::ifstream meminfo("/proc/meminfo");
    if (!meminfo.is_open()) {
        return false;
    }

    uint64_t mem_total_kib = 0;
    uint64_t mem_available_kib = 0;
    std::string key;
    uint64_t value = 0;
    std::string unit;
    while (meminfo >> key >> value >> unit) {
        if (key == "MemTotal:") {
            mem_total_kib = value;
        } else if (key == "MemAvailable:") {
            mem_available_kib = value;
        }
    }
    if (mem_total_kib == 0 || mem_available_kib == 0) {
        return false;
    }

    status.available = size_t(mem_available_kib * 1024ULL);
    status.total     = size_t(mem_total_kib     * 1024ULL);
    status.watermark = common_fit_get_linux_zone_high_watermark();
    status.source    = "/proc/meminfo MemAvailable";
    return true;
#elif defined(__APPLE__) && defined(__MACH__)
    uint64_t mem_total = 0;
    size_t mem_total_len = sizeof(mem_total);
    if (sysctlbyname("hw.memsize", &mem_total, &mem_total_len, nullptr, 0) != 0 || mem_total == 0) {
        return false;
    }

    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    vm_statistics64_data_t vmstat;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vmstat), &count) != KERN_SUCCESS) {
        return false;
    }

    vm_size_t page_size = 0;
    if (host_page_size(mach_host_self(), &page_size) != KERN_SUCCESS || page_size == 0) {
        return false;
    }

    const uint64_t reusable_pages = uint64_t(vmstat.free_count)
        + uint64_t(vmstat.inactive_count)
        + uint64_t(vmstat.speculative_count);
    status.available = size_t(reusable_pages * uint64_t(page_size));
    status.total     = size_t(mem_total);
    status.source    = "host_statistics64";
    return status.available > 0 && status.total > 0;
#else
    (void) status;
    return false;
#endif
}

static bool common_fit_get_file_size(const char * path, uint64_t & size) {
    size = 0;
    if (path == nullptr || path[0] == '\0') {
        return false;
    }

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return false;
    }

    const std::streampos end = file.tellg();
    if (end <= 0) {
        return false;
    }

    size = static_cast<uint64_t>(end);
    return true;
}

static void common_fit_probe_load_precheck(const char * path_model, const llama_model_params * mparams) {
    if (mparams == nullptr || mparams->use_mmap) {
        return;
    }

    common_fit_host_memory_status os_host_memory;
    if (!common_fit_get_os_host_memory(os_host_memory) || os_host_memory.available == 0) {
        LOG_WRN("%s: cannot pre-check --no-mmap probe-load risk because OS host-memory availability is unknown\n", __func__);
        return;
    }

    uint64_t model_size = 0;
    if (!common_fit_get_file_size(path_model, model_size) || model_size == 0) {
        LOG_WRN("%s: cannot pre-check --no-mmap probe-load risk because model file size is unknown\n", __func__);
        return;
    }

    const uint64_t available = static_cast<uint64_t>(os_host_memory.available);
    const uint64_t guard = static_cast<uint64_t>(std::min<size_t>(
        std::max<size_t>(os_host_memory.available / 16, size_t(512)*COMMON_FIT_MIB),
        size_t(4)*COMMON_FIT_GIB));

    if (model_size > available || guard > available - model_size) {
        throw common_params_fit_exception(std::string("--fit probe-load precheck failed: --no-mmap model file is ")
            + std::to_string(model_size/COMMON_FIT_MIB) + " MiB, available host memory is "
            + std::to_string(available/COMMON_FIT_MIB) + " MiB, required preflight guard is "
            + std::to_string(guard/COMMON_FIT_MIB)
            + " MiB; refusing to make the fit probe the first unsafe allocation point");
    }

    LOG_DBG("%s: --no-mmap probe-load precheck passed: model=%" PRIu64 " MiB, host available=%" PRIu64 " MiB, guard=%" PRIu64 " MiB (%s)\n",
        __func__, model_size/COMMON_FIT_MIB, available/COMMON_FIT_MIB, guard/COMMON_FIT_MIB, os_host_memory.source);
}

struct common_fit_shared_margin_result {
    int64_t margin           = 0;
    int64_t guard            = 0;
    int64_t kernel_watermark = 0;
    const char * profile     = "generic";
};

static common_fit_shared_margin_result common_fit_shared_host_memory_margin(
        int64_t shared_total,
        int64_t requested_margin,
        int64_t os_watermark,
        bool has_rocm_overlap,
        bool has_vulkan_overlap) {

    const int64_t GiB = COMMON_FIT_GIB;

    common_fit_shared_margin_result result;
    result.margin = requested_margin;

    if (shared_total <= 0) {
        return result;
    }

    // Budget group 2 is the overlapping UMA / unified-memory pool. Keep the
    // base reserve backend-neutral and easy to reason about: choose a small
    // system-RAM reserve by host-memory class, then add the kernel watermark
    // when the OS exposes one. This avoids both extremes:
    //   - no UMA headroom on small shared-memory systems
    //   - a huge shared_total/N penalty on large UMA systems
    // The final steady-state memory and startup transient reserve are modeled
    // separately below.
    const char * profile =
        has_rocm_overlap   ? "rocm-uma-system-reserve-tiered" :
        has_vulkan_overlap ? "vulkan-uma-system-reserve-tiered" :
                             "uma-system-reserve-tiered";

    int64_t shared_budget_guard = 3*GiB;
    if (shared_total >= 192*GiB) {
        shared_budget_guard = 9*GiB;
    } else if (shared_total >= 96*GiB) {
        shared_budget_guard = 6*GiB;
    }

    // Linux zone high watermarks are added on top when available, because those
    // pages are not usable allocation headroom. Users can still request a larger
    // target explicitly via --fit-target.
    const int64_t kernel_watermark = os_watermark > 0 ? 2*os_watermark : 0;

    result.guard            = shared_budget_guard;
    result.kernel_watermark = kernel_watermark;
    result.profile          = profile;
    result.margin           = std::max(requested_margin, shared_budget_guard + kernel_watermark);
    return result;
}

struct common_fit_shared_memory_components {
    int64_t model   = 0;
    int64_t context = 0;
    int64_t compute = 0;

    int64_t total() const {
        return model + context + compute;
    }
};

struct common_fit_shared_peak_guard_result {
    // Additional headroom required while the real loader reaches the steady
    // state measured by memory_breakdown. This is a reserve, not an assertion
    // that final steady memory is larger.
    int64_t guard         = 0;
    int64_t compute_guard = 0;
    int64_t context_guard = 0;
    int64_t model_guard   = 0;
    int64_t min_guard     = 0;
    int64_t max_guard     = 0;
    int64_t context_div   = 0;
    int64_t model_div     = 0;
    const char * model_profile = "steady-model-fragmentation";
};

static common_fit_shared_peak_guard_result common_fit_shared_peak_guard(
        const common_fit_shared_memory_components & c,
        int64_t shared_total,
        const llama_model_params * mparams) {
    const int64_t GiB = COMMON_FIT_GIB;

    common_fit_shared_peak_guard_result result;

    // The no_alloc probe gives a steady-state memory breakdown. The real startup
    // has an additional load/upload phase before it reaches that steady state.
    // On UMA this phase is charged to the same shared RAM budget as the final
    // device buffers, so fit needs an explicit transient headroom model.
    //
    // Keep the steady-state prediction exact and account for a small generic
    // loader/upload window separately. This avoids treating the transient load
    // phase as permanent model memory while still protecting UMA budgets where
    // upload/staging pressure competes with the final device buffers.
    int64_t default_model_div = 32;
    result.model_profile = "mmap-loader-window";
    if (mparams != nullptr && !mparams->use_mmap) {
        default_model_div = mparams->use_direct_io ? 20 : 16;
        result.model_profile = mparams->use_direct_io ? "direct-io-loader-window" : "non-mmap-loader-window";
    } else if (mparams != nullptr && mparams->use_direct_io) {
        default_model_div = 24;
        result.model_profile = "direct-io-mmap-loader-window";
    }

    result.context_div = 16;
    result.model_div   = default_model_div;

    result.min_guard = 512*COMMON_FIT_MIB;
    result.max_guard = shared_total > 0 ? std::min<int64_t>(shared_total / 8, 16*GiB) : 16*GiB;
    result.max_guard = std::max(result.max_guard, result.min_guard);

    result.compute_guard = std::max<int64_t>(0, c.compute);
    result.context_guard = std::max<int64_t>(0, c.context / result.context_div);
    result.model_guard   = std::max<int64_t>(0, c.model   / result.model_div);

    int64_t guard = std::max(result.min_guard, std::max(result.compute_guard, std::max(result.context_guard, result.model_guard)));


    result.guard = common_fit_clamp_i64(guard, result.min_guard, result.max_guard);
    return result;
}

static void common_fit_reserve_probe_graph(llama_model * model, llama_context * ctx, const llama_context_params * cparams) {
    uint32_t n_tokens = 1;
    if (llama_model_has_decoder(model)) {
        const uint32_t n_batch = cparams->n_batch == 0 ? 1 : cparams->n_batch;
        n_tokens = std::max<uint32_t>(1, std::min<uint32_t>(2, n_batch));
    }

    if (llama_graph_reserve(ctx, n_tokens, 1, n_tokens) == nullptr) {
        throw common_params_fit_exception("failed to reserve compute graph for fit probe");
    }
}

// The fit probe must measure the same steady-state buffers that real startup will
// require, but it must not execute a decode. Reserving the graph is enough to
// materialize the compute-buffer reservation in the memory breakdown without
// mutating generation state; startup/load transients are still modeled separately
// for shared-memory UMA budgets below.

static std::vector<llama_device_memory_data> common_get_device_memory_data(
        const char * path_model,
        const llama_model_params * mparams,
        const llama_context_params * cparams,
        std::vector<ggml_backend_dev_t> & devs,
        uint32_t & hp_ngl,
        uint32_t & hp_n_ctx_train,
        uint32_t & hp_n_expert,
        ggml_log_level log_level) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } original_logger;
        ggml_log_level min_level; // prints below this log level go to debug log
    };
    user_data_t ud;
    llama_log_get(&ud.original_logger.callback, &ud.original_logger.user_data);
    ud.min_level = log_level;

    struct log_restore_t {
        ggml_log_callback callback;
        void * user_data;

        ~log_restore_t() {
            llama_log_set(callback, user_data);
        }
    } log_restore { ud.original_logger.callback, ud.original_logger.user_data };

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        const ggml_log_level level_eff = level >= ud->min_level ? level : GGML_LOG_LEVEL_DEBUG;
        ud->original_logger.callback(level_eff, text, ud->original_logger.user_data);
    }, &ud);

    common_fit_probe_load_precheck(path_model, mparams);

    llama_model_params mparams_copy = *mparams;
    mparams_copy.no_alloc  = true;
    mparams_copy.use_mmap  = mparams->use_mmap;
    mparams_copy.use_mlock = false;

    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(path_model, mparams_copy), llama_model_free);
    if (model == nullptr) {
        throw std::runtime_error("failed to load model");
    }

    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), *cparams), llama_free);
    if (ctx == nullptr) {
        throw std::runtime_error("failed to create llama_context from model");
    }

    common_fit_reserve_probe_graph(model.get(), ctx.get(), cparams);

    const size_t nd = llama_model_n_devices(model.get());
    std::vector<llama_device_memory_data> ret(nd + 1);

    llama_memory_breakdown memory_breakdown = llama_get_memory_breakdown(ctx.get());

    for (const auto & [buft, mb] : memory_breakdown) {
        if (ggml_backend_buft_is_host(buft)) {
            ret.back().mb.model   += mb.model;
            ret.back().mb.context += mb.context;
            ret.back().mb.compute += mb.compute;
            continue;
        }

        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (!dev) {
            continue;
        }
        for (size_t i = 0; i < nd; i++) {
            if (dev == llama_model_get_device(model.get(), i)) {
                ret[i].mb.model   += mb.model;
                ret[i].mb.context += mb.context;
                ret[i].mb.compute += mb.compute;
                break;
            }
        }
    }

    {
        ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu_dev == nullptr) {
            throw std::runtime_error("no CPU backend found");
        }
        size_t free;
        size_t total;
        ggml_backend_dev_memory(cpu_dev, &free, &total);

        common_fit_host_memory_status os_host_memory;
        if (common_fit_get_os_host_memory(os_host_memory)) {
            if (free == 0 || os_host_memory.available < free) {
                LOG_INF("%s: host available memory from %s limits fit budget to %zu MiB (backend reported %zu MiB)\n",
                    __func__, os_host_memory.source, os_host_memory.available/COMMON_FIT_MIB, free/COMMON_FIT_MIB);
                free = os_host_memory.available;
            } else {
                LOG_DBG("%s: host available memory from %s is %zu MiB (backend reported %zu MiB)\n",
                    __func__, os_host_memory.source, os_host_memory.available/COMMON_FIT_MIB, free/COMMON_FIT_MIB);
            }

            if (total == 0 || os_host_memory.total < total) {
                total = os_host_memory.total;
            }
            if (os_host_memory.watermark > 0) {
                LOG_DBG("%s: OS memory watermark from %s is %zu MiB\n",
                    __func__, os_host_memory.source, os_host_memory.watermark/COMMON_FIT_MIB);
            }
        }

        ret.back().free  = free;
        ret.back().total = total;
    }
    for (size_t i = 0; i < nd; i++) {
        size_t free;
        size_t total;
        ggml_backend_dev_t dev = llama_model_get_device(model.get(), i);
        ggml_backend_dev_memory(dev, &free, &total);

        if (free == 0 && total == 0) {
            throw common_params_fit_exception(std::string("device ") + ggml_backend_dev_name(dev)
                + " did not report memory; cannot safely fit to an unknown device budget");
        }
        ret[i].free  = free;
        ret[i].total = total;
    }

    devs.clear();
    for (int i = 0; i < llama_model_n_devices(model.get()); i++) {
        devs.push_back(llama_model_get_device(model.get(), i));
    }

    hp_ngl         = llama_model_n_layer(model.get());
    hp_n_ctx_train = llama_model_n_ctx_train(model.get());
    hp_n_expert    = llama_model_n_expert(model.get());

    common_memory_breakdown_print(ctx.get());

    return ret;
}

static void common_params_fit_impl(
        const char * path_model, struct llama_model_params * mparams, struct llama_context_params * cparams,
        float * tensor_split, struct llama_model_tensor_buft_override * tensor_buft_overrides,
        size_t * margins_s, uint32_t n_ctx_min, enum ggml_log_level log_level) {
    if (mparams->split_mode == LLAMA_SPLIT_MODE_TENSOR) {
        throw common_params_fit_exception("llama_params_fit is not implemented for SPLIT_MODE_TENSOR, abort");
    }
    constexpr int64_t MiB = 1024*1024;
    typedef std::vector<llama_device_memory_data> dmds_t;
    const llama_model_params default_mparams = llama_model_default_params();

    std::vector<ggml_backend_dev_t> devs;
    uint32_t hp_ngl = 0; // hparams.n_gpu_layers
    uint32_t hp_nct = 0; // hparams.n_ctx_train
    uint32_t hp_nex = 0; // hparams.n_expert

    // step 1: get data for default parameters and check whether any changes are necessary in the first place

    LOG_INF("%s: getting device memory data for initial parameters:\n", __func__);
    LOG_INF("%s: fitting uses memory budget groups 0/1/2 with frozen/min-observed budgets\n", __func__);
    const dmds_t dmds_full = common_get_device_memory_data(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);
    const size_t nd = devs.size(); // number of devices

    std::vector<int64_t> margins; // this function uses int64_t rather than size_t for memory sizes to more conveniently handle deficits
    margins.reserve(nd);
    if (nd == 0) {
        margins.push_back(margins_s[0]);
    } else {
        for (size_t id = 0; id < nd; id++) {
            margins.push_back(margins_s[id]);
        }
    }

    const char * const log_fn = __func__;

    std::vector<bool> dev_overlaps_host_budget;
    dev_overlaps_host_budget.reserve(nd);
    bool has_overlapping_budget = false;
    bool has_rocm_overlap = false;
    bool has_vulkan_overlap = false;
    for (size_t id = 0; id < nd; id++) {
        const bool overlaps = common_fit_dev_overlaps_host_budget(devs[id], dmds_full[id], dmds_full.back());
        const std::string dev_text = common_fit_dev_text(devs[id]);
        dev_overlaps_host_budget.push_back(overlaps);
        has_overlapping_budget = has_overlapping_budget || overlaps;
        if (overlaps) {
            has_rocm_overlap   = has_rocm_overlap   || common_fit_contains(dev_text, "rocm") || common_fit_contains(dev_text, "hip");
            has_vulkan_overlap = has_vulkan_overlap || common_fit_contains(dev_text, "vulkan") || common_fit_contains(dev_text, "radv");
            LOG_INF("%s: budget group 2 enabled: device %zu (%s) overlaps the host/system-memory budget\n",
                log_fn, id, ggml_backend_dev_name(devs[id]));
        }
    }

    const int64_t requested_host_margin = margins.empty() ? int64_t(margins_s[0]) : *std::max_element(margins.begin(), margins.end());
    int64_t shared_overlap_margin = requested_host_margin;
    if (has_overlapping_budget) {
        common_fit_host_memory_status os_host_memory;
        const bool have_os_host_memory = common_fit_get_os_host_memory(os_host_memory);

        int64_t shared_total = dmds_full.back().total;
        int64_t requested_shared_margin = requested_host_margin;
        for (size_t id = 0; id < nd; id++) {
            if (!dev_overlaps_host_budget[id]) {
                continue;
            }
            if (dmds_full[id].total > 0) {
                shared_total = shared_total > 0 ? std::min(shared_total, int64_t(dmds_full[id].total)) : int64_t(dmds_full[id].total);
            }
            requested_shared_margin = std::max(requested_shared_margin, margins[id]);
        }

        const common_fit_shared_margin_result shared_margin = common_fit_shared_host_memory_margin(
            shared_total, requested_shared_margin, have_os_host_memory ? int64_t(os_host_memory.watermark) : 0,
            has_rocm_overlap, has_vulkan_overlap);
        shared_overlap_margin = shared_margin.margin;

        LOG_INF("%s: budget group 0 = physical/API-reported device budget; group 1 = host budget; group 2 = shared/overlapping budget\n", log_fn);
        if (shared_overlap_margin > requested_shared_margin) {
            LOG_INF("%s: budget group 2 margin is %" PRId64 " MiB (requested %" PRId64 " MiB, shared total %" PRId64 " MiB, guard %" PRId64 " MiB, watermark %" PRId64 " MiB, profile %s)\n",
                log_fn, shared_overlap_margin/MiB, requested_shared_margin/MiB, shared_total/MiB,
                shared_margin.guard/MiB, shared_margin.kernel_watermark/MiB, shared_margin.profile);
            LOG_INF("%s: budget group 2 guard uses tiered UMA reserve: 3 GiB below 96 GiB, 6 GiB from 96 GiB, 9 GiB from 192 GiB shared memory\n",
                log_fn);
            if (have_os_host_memory && os_host_memory.watermark > 0) {
                LOG_INF("%s: OS-reported kernel watermark is %zu MiB\n",
                    log_fn, os_host_memory.watermark/COMMON_FIT_MIB);
            }
            if (has_rocm_overlap || has_vulkan_overlap) {
                LOG_INF("%s: budget group 2 also applies a candidate-dependent startup transient reserve for UMA load/upload/allocator pressure\n",
                    log_fn);
            }
        } else {
            LOG_INF("%s: budget group 2 using requested shared margin of %" PRId64 " MiB\n",
                log_fn, shared_overlap_margin/MiB);
        }
    }

    enum common_fit_budget_group_kind {
        COMMON_FIT_BUDGET_GROUP_DEVICE_API = 0,
        COMMON_FIT_BUDGET_GROUP_HOST       = 1,
        COMMON_FIT_BUDGET_GROUP_SHARED     = 2,
    };

    struct common_fit_budget_group_budget_floor {
        common_fit_budget_group_kind kind;
        int device = -1;
        int64_t free = 0;
    };

    std::vector<common_fit_budget_group_budget_floor> budget_free_floors;

    auto stabilized_budget_free = [&](common_fit_budget_group_kind kind, int device, const std::string & name, int64_t free) -> int64_t {
        if (free <= 0) {
            return free;
        }

        for (auto & floor : budget_free_floors) {
            if (floor.kind != kind || floor.device != device) {
                continue;
            }

            if (free < floor.free) {
                LOG_INF("%s: budget stabilization: observed lower budget for %s: %" PRId64 " -> %" PRId64 " MiB; using the lower value for this fit decision\n",
                    log_fn, name.c_str(), floor.free/COMMON_FIT_MIB, free/COMMON_FIT_MIB);
                floor.free = free;
            } else if (free > floor.free) {
                const int64_t delta = free - floor.free;
                if (delta >= 128*COMMON_FIT_MIB) {
                    LOG_INF("%s: budget stabilization: current budget for %s is %" PRId64 " MiB, but frozen fit budget is %" PRId64 " MiB; ignoring +%" PRId64 " MiB drift\n",
                        log_fn, name.c_str(), free/COMMON_FIT_MIB, floor.free/COMMON_FIT_MIB, delta/COMMON_FIT_MIB);
                }
            }
            return floor.free;
        }

        budget_free_floors.push_back({kind, device, free});
        LOG_INF("%s: budget stabilization: frozen initial budget for %s at %" PRId64 " MiB\n",
            log_fn, name.c_str(), free/COMMON_FIT_MIB);
        return free;
    };

    struct common_fit_budget_group_status {
        common_fit_budget_group_kind kind;
        int device = -1;
        std::string name;
        int64_t free        = 0;
        int64_t used        = 0; // charged use: final steady-state use plus any startup transient reserve
        int64_t steady_used = 0;
        int64_t peak_guard  = 0;
        int64_t margin      = 0;
        common_fit_shared_memory_components shared_components;

        int64_t surplus() const {
            return free - used - margin;
        }

        int64_t required_margin() const {
            return margin;
        }
    };

    auto usable_budget_free = [&](const llama_device_memory_data & dmd, common_fit_budget_group_kind kind, int device) -> int64_t {
        if (dmd.free > 0) {
            return int64_t(dmd.free);
        }

        if (kind == COMMON_FIT_BUDGET_GROUP_HOST && dmd.total > 0) {
            LOG_WRN("%s: host backend did not report free memory; using total host memory as a last-resort fit budget\n",
                log_fn);
            return int64_t(dmd.total);
        }

        throw common_params_fit_exception("device " + std::to_string(device)
            + " did not report free memory; cannot safely fit to an unknown device budget");
    };

    auto collect_budget_groups = [&](const dmds_t & dmds) {
        std::vector<common_fit_budget_group_status> groups;
        groups.reserve(nd + 2);

        auto make_group = [&](
                common_fit_budget_group_kind kind,
                int device,
                std::string name,
                int64_t free,
                int64_t steady_used,
                int64_t peak_guard,
                int64_t margin,
                common_fit_shared_memory_components shared_components = {}) {
            common_fit_budget_group_status group;
            group.kind              = kind;
            group.device            = device;
            const int64_t stable_free = stabilized_budget_free(kind, device, name, free);
            group.name              = std::move(name);
            group.free              = stable_free;
            group.steady_used        = steady_used;
            group.peak_guard         = peak_guard;
            group.used               = steady_used + peak_guard;
            group.margin             = margin;
            group.shared_components  = shared_components;
            return group;
        };

        if (nd == 0) {
            const int64_t host_used = common_fit_memory_total_i64(dmds.back().mb);
            groups.push_back(make_group(
                COMMON_FIT_BUDGET_GROUP_HOST,
                -1,
                "budget group 1 / host",
                usable_budget_free(dmds.back(), COMMON_FIT_BUDGET_GROUP_HOST, -1),
                host_used,
                0,
                margins[0]));
            return groups;
        }

        for (size_t id = 0; id < nd; id++) {
            const int64_t dev_used = common_fit_memory_total_i64(dmds[id].mb);
            groups.push_back(make_group(
                COMMON_FIT_BUDGET_GROUP_DEVICE_API,
                int(id),
                "budget group 0 / device " + std::to_string(id),
                usable_budget_free(dmds[id], COMMON_FIT_BUDGET_GROUP_DEVICE_API, int(id)),
                dev_used,
                0,
                margins[id]));
        }

        const int64_t host_used = common_fit_memory_total_i64(dmds.back().mb);
        groups.push_back(make_group(
            COMMON_FIT_BUDGET_GROUP_HOST,
            -1,
            "budget group 1 / host",
            usable_budget_free(dmds.back(), COMMON_FIT_BUDGET_GROUP_HOST, -1),
            host_used,
            0,
            requested_host_margin));

        if (has_overlapping_budget) {
            int64_t shared_free = usable_budget_free(dmds.back(), COMMON_FIT_BUDGET_GROUP_HOST, -1);
            common_fit_shared_memory_components shared;
            shared.model   += int64_t(dmds.back().mb.model);
            shared.context += int64_t(dmds.back().mb.context);
            shared.compute += int64_t(dmds.back().mb.compute);

            for (size_t id = 0; id < nd; id++) {
                if (!dev_overlaps_host_budget[id]) {
                    continue;
                }
                const int64_t dev_free = usable_budget_free(dmds[id], COMMON_FIT_BUDGET_GROUP_DEVICE_API, int(id));
                shared_free = shared_free > 0 ? std::min(shared_free, dev_free) : dev_free;
                shared.model   += int64_t(dmds[id].mb.model);
                shared.context += int64_t(dmds[id].mb.context);
                shared.compute += int64_t(dmds[id].mb.compute);
            }

            const common_fit_shared_peak_guard_result peak = common_fit_shared_peak_guard(shared, shared_free, mparams);
            groups.push_back(make_group(
                COMMON_FIT_BUDGET_GROUP_SHARED,
                -1,
                "budget group 2 / shared-overlap",
                shared_free,
                shared.total(),
                peak.guard,
                shared_overlap_margin,
                shared));
        }

        return groups;
    };

    auto find_matching_budget_group = [&](const dmds_t & dmds, const common_fit_budget_group_status & ref) {
        const std::vector<common_fit_budget_group_status> groups = collect_budget_groups(dmds);
        for (const auto & group : groups) {
            if (group.kind == ref.kind && group.device == ref.device) {
                return group;
            }
        }
        throw std::runtime_error("internal fit error: matching memory budget group not found");
    };

    auto limiting_budget_group = [&](const dmds_t & dmds) {
        const std::vector<common_fit_budget_group_status> groups = collect_budget_groups(dmds);
        assert(!groups.empty());
        common_fit_budget_group_status worst = groups.front();
        for (const auto & group : groups) {
            if (group.surplus() < worst.surplus()) {
                worst = group;
            }
        }
        return worst;
    };

    auto log_budget_group = [&](const common_fit_budget_group_status & group, const char * reason) {
        const int64_t headroom = group.free - group.used;
        if (group.peak_guard > 0) {
            const common_fit_shared_peak_guard_result peak = common_fit_shared_peak_guard(group.shared_components, group.free, mparams);
            LOG_INF("%s: %s: %s charged steady+transient %" PRId64 " MiB (final steady %" PRId64 " MiB + startup reserve %" PRId64 " MiB), budget %" PRId64 " MiB, headroom %" PRId64 " MiB, target %" PRId64 " MiB, surplus %" PRId64 " MiB\n",
                log_fn, reason, group.name.c_str(), group.used/MiB, group.steady_used/MiB, group.peak_guard/MiB,
                group.free/COMMON_FIT_MIB, headroom/MiB, group.margin/MiB, group.surplus()/MiB);
            LOG_INF("%s: %s startup reserve detail: compute=%" PRId64 " MiB, context/%" PRId64 "=%" PRId64 " MiB, model/%" PRId64 "=%" PRId64 " MiB (%s), min=%" PRId64 " MiB, max=%" PRId64 " MiB\n",
                log_fn, group.name.c_str(),
                peak.compute_guard/MiB,
                peak.context_div, peak.context_guard/MiB,
                peak.model_div, peak.model_guard/MiB, peak.model_profile,
                peak.min_guard/MiB, peak.max_guard/MiB);
        } else {
            LOG_INF("%s: %s: %s used %" PRId64 " MiB, budget %" PRId64 " MiB, headroom %" PRId64 " MiB, target %" PRId64 " MiB, surplus %" PRId64 " MiB\n",
                log_fn, reason, group.name.c_str(), group.used/MiB, group.free/COMMON_FIT_MIB, headroom/MiB, group.margin/MiB, group.surplus()/MiB);
        }
    };

    auto targets_met = [&](const dmds_t & dmds, bool verbose) -> bool {
        bool ok = true;
        const std::vector<common_fit_budget_group_status> groups = collect_budget_groups(dmds);
        for (const auto & group : groups) {
            if (verbose) {
                log_budget_group(group, "budget check");
            }

            const int64_t deficit = group.surplus();
            if (deficit < 0) {
                if (verbose) {
                    LOG_INF("%s: %s is still short by %" PRId64 " MiB against target\n",
                        log_fn, group.name.c_str(), -deficit/MiB);
                }
                ok = false;
            }
        }
        return ok;
    };

    auto verify_current_fit = [&](const char * stage) {
        const dmds_t dmds_verify = common_get_device_memory_data(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);
        if (!targets_met(dmds_verify, true)) {
            throw common_params_fit_exception(std::string(stage) + " did not pass budget-group verification, abort");
        }
        LOG_INF("%s: %s verified against budget groups 0/1%s\n",
            __func__, stage, has_overlapping_budget ? "/2" : "");
    };

    std::vector<std::string> dev_names;
    {
        dev_names.reserve(nd);
        size_t max_length = 0;
        for (const auto & dev : devs) {
            std::string name = ggml_backend_dev_name(dev);
            name += " (";
            name += ggml_backend_dev_description(dev);
            name += ")";
            dev_names.push_back(name);
            max_length = std::max(max_length, name.length());
        }
        for (std::string & dn : dev_names) {
            dn.insert(dn.end(), max_length - dn.length(), ' ');
        }
    }

    int64_t sum_free           = 0;
    int64_t sum_projected_used = 0;
    std::vector<int64_t> projected_free_per_device;
    projected_free_per_device.reserve(nd);

    if (nd == 0) {
        sum_projected_used = dmds_full.back().mb.total();
        sum_free           = usable_budget_free(dmds_full.back(), COMMON_FIT_BUDGET_GROUP_HOST, -1);
        LOG_INF("%s: projected to use %" PRId64 " MiB of host memory vs. %" PRId64 " MiB of available host budget\n",
            __func__, sum_projected_used/MiB, sum_free/COMMON_FIT_MIB);
        if (targets_met(dmds_full, false)) {
            LOG_INF("%s: host budget target can be met, no changes needed\n", __func__);
            return;
        }
    } else {
        if (nd > 1) {
            LOG_INF("%s: projected memory use with initial parameters [MiB]:\n", __func__);
        }
        for (size_t id = 0; id < nd; id++) {
            const llama_device_memory_data & dmd = dmds_full[id];

            const int64_t projected_used = dmd.mb.total();
            const int64_t projected_free = int64_t(dmd.free) - projected_used;
            projected_free_per_device.push_back(projected_free);

            sum_free           += dmd.free;
            sum_projected_used += projected_used;

            if (nd > 1) {
                LOG_INF("%s:   - %s: %6" PRId64 " total, %6" PRId64 " used, %6" PRId64 " free vs. target of %6" PRId64 "\n",
                    __func__, dev_names[id].c_str(), dmd.total/MiB, projected_used/MiB, projected_free/COMMON_FIT_MIB, margins[id]/MiB);
            }
        }
        assert(sum_free >= 0 && sum_projected_used >= 0);
        LOG_INF("%s: projected to use %" PRId64 " MiB of device memory vs. %" PRId64 " MiB of free device memory\n",
            __func__, sum_projected_used/MiB, sum_free/COMMON_FIT_MIB);
        if (targets_met(dmds_full, false)) {
            if (nd == 1) {
                LOG_INF("%s: will leave %" PRId64 " >= %" PRId64 " MiB of free device memory, no changes needed\n",
                    __func__, projected_free_per_device[0]/MiB, margins[0]/MiB);
            } else {
                LOG_INF("%s: targets for free memory can be met on all devices, no changes needed\n", __func__);
            }
            if (has_overlapping_budget) {
                LOG_INF("%s: shared/overlapping budget target can also be met, no changes needed\n", __func__);
            }
            return;
        }
    }

    // step 2: try reducing memory use by reducing the context size

    {
        const common_fit_budget_group_status limiting_full = limiting_budget_group(dmds_full);
        if (limiting_full.surplus() < 0) {
            log_budget_group(limiting_full, "limiting budget");
            LOG_INF("%s: cannot meet %s target, need to reduce memory by %" PRId64 " MiB\n",
                __func__, limiting_full.name.c_str(), -limiting_full.surplus()/MiB);

            if (cparams->n_ctx == 0) {
                if (hp_nct > n_ctx_min) {
                    const int64_t limiting_used_target = limiting_full.free - limiting_full.required_margin();

                    cparams->n_ctx = n_ctx_min;
                    const dmds_t dmds_min_ctx = common_get_device_memory_data(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);
                    const common_fit_budget_group_status limiting_min_ctx = find_matching_budget_group(dmds_min_ctx, limiting_full);

                    const int64_t used_full = limiting_full.used;
                    const int64_t used_min  = limiting_min_ctx.used;
                    const int64_t ctx_fit_denom = used_full - used_min;

                    if (limiting_used_target > used_min && ctx_fit_denom > 0) {
                        // Build a measured demand model from two real probes of the same
                        // limiting budget group. The result is only a candidate; every
                        // enabled budget group is re-probed and verified before success.
                        const int64_t bytes_per_ctx = ctx_fit_denom / std::max<uint32_t>(1, hp_nct - n_ctx_min);
                        const uint32_t ctx_candidate = n_ctx_min
                            + uint32_t((uint64_t(hp_nct - n_ctx_min) * uint64_t(limiting_used_target - used_min))
                                / uint64_t(ctx_fit_denom));

                        cparams->n_ctx = std::max(ctx_candidate - ctx_candidate % 256, n_ctx_min); // round down context for CUDA backend

                        const int64_t memory_reduction = (hp_nct - cparams->n_ctx) * bytes_per_ctx;
                        LOG_INF("%s: measured steady+transient context demand for %s: min_ctx=%" PRIu32 " steady+transient %" PRId64 " MiB, full_ctx=%" PRIu32 " steady+transient %" PRId64 " MiB, slope=%" PRId64 " KiB/token\n",
                            __func__, limiting_full.name.c_str(), n_ctx_min, used_min/MiB, hp_nct, used_full/MiB, bytes_per_ctx/1024);
                        LOG_INF("%s: demand-based context candidate: target=%" PRId64 " MiB -> n_ctx=%" PRIu32 " (%" PRId64 " MiB less than full context)\n",
                            __func__, limiting_used_target/MiB, cparams->n_ctx, memory_reduction/MiB);
                    } else {
                        const int64_t memory_reduction = used_full - used_min;
                        LOG_INF("%s: only minimum context can be tried for %s -> n_ctx=%" PRIu32 ", %" PRId64 " MiB less than full context\n",
                            __func__, limiting_full.name.c_str(), cparams->n_ctx, memory_reduction/MiB);
                    }

                    dmds_t dmds_reduced_ctx = cparams->n_ctx == n_ctx_min
                        ? dmds_min_ctx
                        : common_get_device_memory_data(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);

                    if (!targets_met(dmds_reduced_ctx, true) && cparams->n_ctx > n_ctx_min && targets_met(dmds_min_ctx, false)) {
                        const uint32_t ctx_initial = cparams->n_ctx;
                        LOG_INF("%s: measured context candidate did not pass all budget groups, binary-searching lower verified context\n", __func__);

                        uint32_t ctx_lo = n_ctx_min;
                        uint32_t ctx_hi = ctx_initial;
                        dmds_t dmds_lo = dmds_min_ctx;

                        while (ctx_hi > ctx_lo + 256) {
                            uint32_t ctx_mid = ctx_lo + ((ctx_hi - ctx_lo) / 512) * 256;
                            if (ctx_mid <= ctx_lo) {
                                ctx_mid = ctx_lo + 256;
                            }
                            if (ctx_mid >= ctx_hi) {
                                ctx_mid = ctx_hi - 256;
                            }

                            cparams->n_ctx = ctx_mid;
                            const dmds_t dmds_mid = common_get_device_memory_data(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);
                            if (targets_met(dmds_mid, false)) {
                                ctx_lo  = ctx_mid;
                                dmds_lo = dmds_mid;
                            } else {
                                ctx_hi = ctx_mid;
                            }
                        }

                        cparams->n_ctx = ctx_lo;
                        dmds_reduced_ctx = dmds_lo;
                        LOG_INF("%s: context size verified after backoff from %" PRIu32 " to %" PRIu32 "\n",
                            __func__, ctx_initial, cparams->n_ctx);
                    }

                    if (targets_met(dmds_reduced_ctx, true)) {
                        // The demand interpolation above is intentionally conservative: it uses two
                        // real probes and then rounds down. With UMA budgets the candidate can become
                        // even more conservative because the startup transient reserve is nonlinear
                        // (it is max(compute, context/N, model/M, min)). Do not accept the first
                        // passing point as final when a higher context can be verified directly.
                        // Instead, binary-search upward and keep only re-probed candidates that pass
                        // every active budget group. This recovers context generically
                        // without weakening group 0/1/2 accounting or relying on
                        // backend-specific constants.
                        const uint32_t ctx_first_verified = cparams->n_ctx;
                        if (cparams->n_ctx + 256 <= hp_nct) {
                            const int max_upward_probes = 12;

                            LOG_INF("%s: first context candidate passed; searching upward for highest verified context using frozen/min-observed budgets (max %d probes)\n",
                                __func__, max_upward_probes);

                            uint32_t ctx_lo = cparams->n_ctx;
                            uint32_t ctx_hi = hp_nct;
                            dmds_t dmds_lo = dmds_reduced_ctx;
                            int probe_count = 0;

                            while (ctx_hi > ctx_lo + 256 && probe_count < max_upward_probes) {
                                uint32_t ctx_mid = ctx_lo + ((ctx_hi - ctx_lo) / 512) * 256;
                                if (ctx_mid <= ctx_lo) {
                                    ctx_mid = ctx_lo + 256;
                                }
                                if (ctx_mid >= ctx_hi) {
                                    ctx_mid = ctx_hi - 256;
                                }

                                cparams->n_ctx = ctx_mid;
                                const dmds_t dmds_mid = common_get_device_memory_data(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);
                                probe_count++;

                                if (targets_met(dmds_mid, false)) {
                                    ctx_lo  = ctx_mid;
                                    dmds_lo = dmds_mid;
                                } else {
                                    ctx_hi = ctx_mid;
                                }
                            }

                            cparams->n_ctx = ctx_lo;
                            dmds_reduced_ctx = dmds_lo;

                            if (ctx_lo > ctx_first_verified) {
                                LOG_INF("%s: reclaimed context by verified upward search: n_ctx %" PRIu32 " -> %" PRIu32 " after %d probes\n",
                                    __func__, ctx_first_verified, ctx_lo, probe_count);
                            } else {
                                LOG_INF("%s: upward search kept first verified context n_ctx=%" PRIu32 " after %d probes\n",
                                    __func__, ctx_lo, probe_count);
                            }
                            if (probe_count >= max_upward_probes && ctx_hi > ctx_lo + 256) {
                                LOG_INF("%s: upward context search stopped at probe limit with remaining interval [%" PRIu32 ", %" PRIu32 "]\n",
                                    __func__, ctx_lo, ctx_hi);
                            }

                            // Print the final accepted candidate after the upward search, because this
                            // is the value that will be committed to the real init path. It should still
                            // have positive surplus.
                            targets_met(dmds_reduced_ctx, true);
                        }

                        if (nd <= 1) {
                            LOG_INF("%s: entire model can be fit by reducing context (highest verified against frozen/min-observed budget groups)\n", __func__);
                        } else {
                            LOG_INF("%s: entire model can be fit across devices by reducing context (highest verified against frozen/min-observed budget groups)\n", __func__);
                        }
                        return;
                    }
                    LOG_INF("%s: context reduction alone did not pass budget-group verification, trying weight placement changes\n", __func__);
                } else {
                    if (n_ctx_min == UINT32_MAX) {
                        LOG_INF("%s: user has requested full context size of %" PRIu32 " -> no change\n", __func__, hp_nct);
                    } else {
                        LOG_INF("%s: default model context size is %" PRIu32 " which is <= the min. context size of %" PRIu32 " -> no change\n",
                            __func__, hp_nct, n_ctx_min);
                    }
                }
            } else {
                LOG_INF("%s: context size set by user to %" PRIu32 " -> no change\n", __func__, cparams->n_ctx);
            }
        }
    }
    if (nd == 0) {
        throw common_params_fit_exception("was unable to fit model into system memory by reducing context, abort");
    }

    if (mparams->n_gpu_layers != default_mparams.n_gpu_layers) {
        throw common_params_fit_exception("n_gpu_layers already set by user to " + std::to_string(mparams->n_gpu_layers) + ", abort");
    }
    if (nd > 1) {
        if (!tensor_split) {
            throw common_params_fit_exception("did not provide a buffer to write the tensor_split to, abort");
        }
        if (mparams->tensor_split) {
            for (size_t id = 0; id < nd; id++) {
                if (mparams->tensor_split[id] != 0.0f) {
                    throw common_params_fit_exception("model_params::tensor_split already set by user, abort");
                }
            }
        }
        if (mparams->split_mode == LLAMA_SPLIT_MODE_ROW) {
            throw common_params_fit_exception("changing weight allocation for LLAMA_SPLIT_MODE_ROW not implemented, abort");
        }
    }
    if (!tensor_buft_overrides) {
        throw common_params_fit_exception("did not provide buffer to set tensor_buft_overrides, abort");
    }
    if (mparams->tensor_buft_overrides && (mparams->tensor_buft_overrides->pattern || mparams->tensor_buft_overrides->buft)) {
        throw common_params_fit_exception("model_params::tensor_buft_overrides already set by user, abort");
    }

    // step 3: iteratively fill the back to front with "dense" layers
    //   - for a dense model simply fill full layers, giving each device a contiguous slice of the model
    //   - for a MoE model, same as dense model but with all MoE tensors in system memory

    // utility function that returns a static C string matching the tensors for a specific layer index and layer fraction:
    auto get_overflow_pattern = [&](const size_t il, const common_layer_fraction_t lf) -> const char * {
        constexpr size_t n_strings = 1000;
        if (il >= n_strings) {
            throw std::runtime_error("at most " + std::to_string(n_strings) + " model layers are supported");
        }
        switch (lf) {
            case LAYER_FRACTION_ATTN: {
                static std::array<std::string, n_strings> patterns;
                if (patterns[il].empty()) {
                    patterns[il] = "blk\\." + std::to_string(il) + "\\.ffn_(gate|up|gate_up|down).*";
                }
                return patterns[il].c_str();
            }
            case LAYER_FRACTION_UP: {
                static std::array<std::string, n_strings> patterns;
                if (patterns[il].empty()) {
                    patterns[il] = "blk\\." + std::to_string(il) + "\\.ffn_(gate|gate_up|down).*";
                }
                return patterns[il].c_str();
            }
            case LAYER_FRACTION_GATE: {
                static std::array<std::string, n_strings> patterns;
                if (patterns[il].empty()) {
                    patterns[il] = "blk\\." + std::to_string(il) + "\\.ffn_down.*";
                }
                return patterns[il].c_str();
            }
            case LAYER_FRACTION_MOE: {
                static std::array<std::string, n_strings> patterns;
                if (patterns[il].empty()) {
                    patterns[il] = "blk\\." + std::to_string(il) + "\\.ffn_(up|down|gate_up|gate)_(ch|)exps";
                }
                return patterns[il].c_str();
            }
            default:
                GGML_ABORT("fatal error");
        }
    };

    struct ngl_t {
        uint32_t n_layer = 0; // number of total layers
        uint32_t n_part  = 0; // number of partial layers, <= n_layer

        // for the first partial layer varying parts can overflow, all further layers use LAYER_FRACTION_MOE:
        common_layer_fraction_t overflow_type = LAYER_FRACTION_MOE;

        uint32_t n_full() const {
            assert(n_layer >= n_part);
            return n_layer - n_part;
        }
    };

    const size_t ntbo = llama_max_tensor_buft_overrides();

    // utility function to set n_gpu_layers and tensor_split
    auto set_ngl_tensor_split_tbo = [&](
            const std::vector<ngl_t> & ngl_per_device,
            const std::vector<ggml_backend_buffer_type_t> & overflow_bufts,
            llama_model_params & mparams) {
        mparams.n_gpu_layers = 0;
        for (size_t id = 0; id < nd; id++) {
            mparams.n_gpu_layers += ngl_per_device[id].n_layer;
            if (nd > 1) {
                tensor_split[id] = ngl_per_device[id].n_layer;
            }
        }
        assert(uint32_t(mparams.n_gpu_layers) <= hp_ngl + 1);
        uint32_t il0 = hp_ngl + 1 - mparams.n_gpu_layers; // start index for tensor buft overrides

        mparams.tensor_split = tensor_split;

        size_t itbo = 0;
        for (size_t id = 0; id < nd; id++) {
            il0 += ngl_per_device[id].n_full();
            for (uint32_t il = il0; il < il0 + ngl_per_device[id].n_part; il++) {
                if (itbo + 1 >= ntbo) {
                    tensor_buft_overrides[itbo].pattern = nullptr;
                    tensor_buft_overrides[itbo].buft    = nullptr;
                    itbo++;
                    mparams.tensor_buft_overrides = tensor_buft_overrides;
                    throw common_params_fit_exception("llama_max_tensor_buft_overrides() == "
                        + std::to_string(ntbo) + " is insufficient for model");
                }
                tensor_buft_overrides[itbo].pattern = get_overflow_pattern(il, il == il0 ? ngl_per_device[id].overflow_type : LAYER_FRACTION_MOE);
                tensor_buft_overrides[itbo].buft = il == il0 ? overflow_bufts[id] : ggml_backend_cpu_buffer_type();
                itbo++;
            }
            il0 += ngl_per_device[id].n_part;
        }
        tensor_buft_overrides[itbo].pattern = nullptr;
        tensor_buft_overrides[itbo].buft    = nullptr;
        itbo++;
        mparams.tensor_buft_overrides = tensor_buft_overrides;
    };

    // utility function that returns the memory use per device for given numbers of layers per device
    auto get_memory_for_layers = [&](
            const char * func_name,
            const std::vector<ngl_t> & ngl_per_device,
            const std::vector<ggml_backend_buffer_type_t> & overflow_bufts) -> std::vector<int64_t> {
        llama_model_params mparams_copy = *mparams;
        set_ngl_tensor_split_tbo(ngl_per_device, overflow_bufts, mparams_copy);

        const dmds_t dmd_nl = common_get_device_memory_data(
            path_model, &mparams_copy, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);

        LOG_INF("%s: memory for test allocation by device:\n", func_name);
        for (size_t id = 0; id < nd; id++) {
            const ngl_t & n = ngl_per_device[id];
            LOG_INF(
                "%s: id=%zu, n_layer=%2" PRIu32 ", n_part=%2" PRIu32 ", overflow_type=%d, mem=%6" PRId64 " MiB\n",
                func_name, id, n.n_layer, n.n_part, int(n.overflow_type), dmd_nl[id].mb.total()/MiB);
        }

        std::vector<int64_t> ret;
        ret.reserve(nd);
        for (size_t id = 0; id < nd; id++) {
            ret.push_back(dmd_nl[id].mb.total());
        }
        return ret;
    };

    int64_t global_surplus_cpu_moe = 0;
    if (hp_nex > 0) {
        const static std::string pattern_moe_all = "blk\\.\\d+\\.ffn_(up|down|gate_up|gate)_(ch|)exps"; // matches all MoE tensors
        ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
        tensor_buft_overrides[0] = {pattern_moe_all.c_str(), cpu_buft};
        tensor_buft_overrides[1] = {nullptr, nullptr};
        mparams->tensor_buft_overrides = tensor_buft_overrides;

        LOG_INF("%s: getting device memory data with all MoE tensors moved to system memory:\n", __func__);
        const dmds_t dmds_cpu_moe = common_get_device_memory_data(
            path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);

        for (size_t id = 0; id < nd; id++) {
            global_surplus_cpu_moe += dmds_cpu_moe[id].free;
            global_surplus_cpu_moe -= int64_t(dmds_cpu_moe[id].mb.total()) + margins[id];
        }

        if (global_surplus_cpu_moe > 0) {
            LOG_INF("%s: with only dense weights in device memory there is a total surplus of %" PRId64 " MiB\n",
                __func__, global_surplus_cpu_moe/MiB);
        } else {
            LOG_INF("%s: with only dense weights in device memory there is still a total deficit of %" PRId64 " MiB\n",
                __func__, -global_surplus_cpu_moe/MiB);
        }

        // reset
        tensor_buft_overrides[0] = {nullptr, nullptr};
        mparams->tensor_buft_overrides = tensor_buft_overrides;
    }

    std::vector<int64_t> targets; // maximum acceptable memory use per device
    targets.reserve(nd);
    for (size_t id = 0; id < nd; id++) {
        targets.push_back(int64_t(dmds_full[id].free) - margins[id]);
        LOG_INF("%s: id=%zu, target=%" PRId64 " MiB\n", __func__, id, targets[id]/MiB);
    }

    std::vector<ggml_backend_buffer_type_t> overflow_bufts; // which bufts the first partial layer of a device overflows to:
    overflow_bufts.reserve(nd);
    for (size_t id = 0; id < nd; id++) {
        overflow_bufts.push_back(ggml_backend_cpu_buffer_type());
    }

    std::vector<ngl_t> ngl_per_device(nd);
    std::vector<int64_t> mem = get_memory_for_layers(__func__, ngl_per_device, overflow_bufts);
    for (size_t id = 0; id < nd; id++) {
        if (mem[id] > targets[id]) {
            throw common_params_fit_exception("device " + std::to_string(id)
                + " already exceeds the free-memory target before assigning model layers; cannot safely fit");
        }
    }

    // optimize the number of layers per device using the method of false position:
    //   - ngl_per_device has 0 layers for each device, lower bound
    //   - try a "high" configuration where a device is given all unassigned layers
    //   - interpolate the memory use / layer between low and high linearly to get a guess where it meets our target
    //   - check memory use of our guess, replace either the low or high bound
    //   - once we only have a difference of a single layer, stop and return the lower bound that just barely still fits
    //   - the last device has the output layer, which cannot be a partial layer
    if (hp_nex == 0) {
        LOG_INF("%s: filling dense layers back-to-front:\n", __func__);
    } else {
        LOG_INF("%s: filling dense-only layers back-to-front:\n", __func__);
    }
    for (size_t id_rev = nd; id_rev-- > 0;) {
        const size_t id = id_rev;
        uint32_t n_unassigned = hp_ngl + 1;
        for (size_t jd = id + 1; jd < nd; ++jd) {
            assert(n_unassigned >= ngl_per_device[jd].n_layer);
            n_unassigned -= ngl_per_device[jd].n_layer;
        }

        std::vector<ngl_t> ngl_per_device_high = ngl_per_device;
        ngl_per_device_high[id].n_layer = n_unassigned;
        if (hp_nex > 0) {
            ngl_per_device_high[id].n_part = id < nd - 1 ? ngl_per_device_high[id].n_layer : ngl_per_device_high[id].n_layer - 1;
        }
        if (ngl_per_device_high[id].n_layer > 0) {
            std::vector<int64_t> mem_high = get_memory_for_layers(__func__, ngl_per_device_high, overflow_bufts);
            if (mem_high[id] > targets[id]) {
                assert(ngl_per_device_high[id].n_layer > ngl_per_device[id].n_layer);
                uint32_t delta = ngl_per_device_high[id].n_layer - ngl_per_device[id].n_layer;
                LOG_INF("%s: start filling device %zu, delta=%" PRIu32 "\n", __func__, id, delta);
                while (delta > 1) {
                    uint32_t step_size = int64_t(delta) * (targets[id] - mem[id]) / (mem_high[id] - mem[id]);
                    step_size = std::max(step_size, uint32_t(1));
                    step_size = std::min(step_size, delta - 1);

                    std::vector<ngl_t> ngl_per_device_test = ngl_per_device;
                    ngl_per_device_test[id].n_layer += step_size;
                    if (hp_nex) {
                        ngl_per_device_test[id].n_part += size_t(id) == nd - 1 && ngl_per_device_test[id].n_part == 0 ?
                            step_size - 1 : step_size; // the first layer is the output layer which must always be full
                    }
                    const std::vector<int64_t> mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts);

                    if (mem_test[id] <= targets[id]) {
                        ngl_per_device = ngl_per_device_test;
                        mem            = mem_test;
                        LOG_INF("%s: set ngl_per_device[%zu].n_layer=%" PRIu32 "\n", __func__, id, ngl_per_device[id].n_layer);
                    } else {
                        ngl_per_device_high = ngl_per_device_test;
                        mem_high            = mem_test;
                        LOG_INF("%s: set ngl_per_device_high[%zu].n_layer=%" PRIu32 "\n", __func__, id, ngl_per_device_high[id].n_layer);
                    }
                    delta = ngl_per_device_high[id].n_layer - ngl_per_device[id].n_layer;
                }
            } else {
                assert(ngl_per_device_high[id].n_layer == n_unassigned);
                ngl_per_device = ngl_per_device_high;
                mem            = mem_high;
                LOG_INF("%s: set ngl_per_device[%zu].n_layer=%" PRIu32 "\n", __func__, id, ngl_per_device[id].n_layer);
            }
        }

        const int64_t projected_margin = int64_t(dmds_full[id].free) - mem[id];
        LOG_INF(
            "%s:   - %s: %2" PRIu32 " layers, %6" PRId64 " MiB used, %6" PRId64 " MiB free\n",
            __func__, dev_names[id].c_str(), ngl_per_device[id].n_layer, mem[id]/MiB, projected_margin/MiB);
    }
    if (hp_nex == 0 || global_surplus_cpu_moe <= 0) {
        set_ngl_tensor_split_tbo(ngl_per_device, overflow_bufts, *mparams);
        verify_current_fit("layer placement");
        return;
    }

    // step 4: for a MoE model where all dense tensors fit,
    //     convert the dense-only layers in the back to full layers in the front until all devices are full
    // essentially the same procedure as for the dense-only layers except front-to-back
    // also, try fitting at least part of one more layer to reduce waste for "small" GPUs with e.g. 24 GiB VRAM

    size_t id_dense_start = nd;
    for (size_t id_rev = nd; id_rev-- > 0;) {
        const size_t id = id_rev;
        if (ngl_per_device[id].n_layer > 0) {
            id_dense_start = id;
            continue;
        }
        break;
    }
    assert(id_dense_start < nd);

    LOG_INF("%s: converting dense-only layers to full layers and filling them front-to-back with overflow to next device/system memory:\n", __func__);
    for (size_t id = 0; id <= id_dense_start && id_dense_start < nd; id++) {
        std::vector<ngl_t> ngl_per_device_high = ngl_per_device;
        for (size_t jd = id_dense_start; jd < nd; jd++) {
            const uint32_t n_layer_move = jd < nd - 1 ? ngl_per_device_high[jd].n_layer : ngl_per_device_high[jd].n_layer - 1;
            ngl_per_device_high[id].n_layer += n_layer_move;
            ngl_per_device_high[jd].n_layer -= n_layer_move;
            ngl_per_device_high[jd].n_part = 0;
        }
        size_t id_dense_start_high = nd - 1;
        std::vector<int64_t> mem_high = get_memory_for_layers(__func__, ngl_per_device_high, overflow_bufts);

        if (mem_high[id] > targets[id]) {
            assert(ngl_per_device_high[id].n_full() >= ngl_per_device[id].n_full());
            uint32_t delta = ngl_per_device_high[id].n_full() - ngl_per_device[id].n_full();
            while (delta > 1) {
                uint32_t step_size = int64_t(delta) * (targets[id] - mem[id]) / (mem_high[id] - mem[id]);
                step_size = std::max(step_size, uint32_t(1));
                step_size = std::min(step_size, delta - 1);

                std::vector<ngl_t> ngl_per_device_test = ngl_per_device;
                size_t id_dense_start_test = id_dense_start;
                uint32_t n_converted_test = 0;
                for (;id_dense_start_test < nd; id_dense_start_test++) {
                    const uint32_t n_convert_jd = std::min(step_size - n_converted_test, ngl_per_device_test[id_dense_start_test].n_part);
                    ngl_per_device_test[id_dense_start_test].n_layer -= n_convert_jd;
                    ngl_per_device_test[id_dense_start_test].n_part -= n_convert_jd;
                    ngl_per_device_test[id].n_layer += n_convert_jd;
                    n_converted_test += n_convert_jd;

                    if (ngl_per_device_test[id_dense_start_test].n_part > 0) {
                        break;
                    }
                }
                const std::vector<int64_t> mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts);

                if (mem_test[id] <= targets[id]) {
                    ngl_per_device = ngl_per_device_test;
                    mem            = mem_test;
                    id_dense_start = id_dense_start_test;
                    LOG_INF("%s: set ngl_per_device[%zu].(n_layer, n_part)=(%" PRIu32 ", %" PRIu32 "), id_dense_start=%zu\n",
                        __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);
                } else {
                    ngl_per_device_high = ngl_per_device_test;
                    mem_high            = mem_test;
                    id_dense_start_high = id_dense_start_test;
                    LOG_INF("%s: set ngl_per_device_high[%zu].(n_layer, n_part)=(%" PRIu32 ", %" PRIu32 "), id_dense_start_high=%zu\n",
                        __func__, id, ngl_per_device_high[id].n_layer, ngl_per_device_high[id].n_part, id_dense_start_high);
                }
                assert(ngl_per_device_high[id].n_full() >= ngl_per_device[id].n_full());
                delta = ngl_per_device_high[id].n_full() - ngl_per_device[id].n_full();
            }
        } else {
            ngl_per_device = ngl_per_device_high;
            mem            = mem_high;
            id_dense_start = id_dense_start_high;
            LOG_INF("%s: set ngl_per_device[%zu].(n_layer, n_part)=(%" PRIu32 ", %" PRIu32 "), id_dense_start=%zu\n",
                __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);
        }

        // try to fit at least part of one more layer
        if (ngl_per_device[id_dense_start].n_layer > (id < nd - 1 ? 0 : 1)) {
            std::vector<ngl_t> ngl_per_device_test = ngl_per_device;
            size_t id_dense_start_test = id_dense_start;
            ngl_per_device_test[id_dense_start_test].n_layer--;
            ngl_per_device_test[id_dense_start_test].n_part--;
            ngl_per_device_test[id].n_layer++;
            ngl_per_device_test[id].n_part++;
            if (ngl_per_device_test[id_dense_start_test].n_part == 0) {
                id_dense_start_test++;
            }
            ngl_per_device_test[id].overflow_type = LAYER_FRACTION_UP;
            std::vector<ggml_backend_buffer_type_t> overflow_bufts_test = overflow_bufts;
            if (id < nd - 1) {
                overflow_bufts_test[id] = ggml_backend_dev_buffer_type(devs[id + 1]);
            }
            LOG_INF("%s: trying to fit one extra layer with overflow_type=LAYER_FRACTION_UP\n", __func__);
            std::vector<int64_t> mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts_test);
            if (mem_test[id] < targets[id] && (id + 1 == nd || mem_test[id + 1] < targets[id + 1])) {
                ngl_per_device = ngl_per_device_test;
                overflow_bufts = overflow_bufts_test;
                mem            = mem_test;
                id_dense_start = id_dense_start_test;
                LOG_INF("%s: set ngl_per_device[%zu].(n_layer, n_part, overflow_type)=(%" PRIu32 ", %" PRIu32 ", UP), id_dense_start=%zu\n",
                    __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);

                ngl_per_device_test[id].overflow_type = LAYER_FRACTION_GATE;
                LOG_INF("%s: trying to fit one extra layer with overflow_type=LAYER_FRACTION_GATE\n", __func__);
                mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts_test);
                if (mem_test[id] < targets[id] && (id + 1 == nd || mem_test[id + 1] < targets[id + 1])) {
                    ngl_per_device = ngl_per_device_test;
                    overflow_bufts = overflow_bufts_test;
                    mem            = mem_test;
                    id_dense_start = id_dense_start_test;
                    LOG_INF("%s: set ngl_per_device[%zu].(n_layer, n_part, overflow_type)=(%" PRIu32 ", %" PRIu32 ", GATE), id_dense_start=%zu\n",
                        __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);
                }
            } else {
                ngl_per_device_test[id].overflow_type = LAYER_FRACTION_ATTN;
                LOG_INF("%s: trying to fit one extra layer with overflow_type=LAYER_FRACTION_ATTN\n", __func__);
                mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts_test);
                if (mem_test[id] < targets[id] && (id + 1 == nd || mem_test[id + 1] < targets[id + 1])) {
                    ngl_per_device = ngl_per_device_test;
                    overflow_bufts = overflow_bufts_test;
                    mem            = mem_test;
                    id_dense_start = id_dense_start_test;
                    LOG_INF("%s: set ngl_per_device[%zu].(n_layer, n_part, overflow_type)=(%" PRIu32 ", %" PRIu32 ", ATTN), id_dense_start=%zu\n",
                        __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);
                }
            }
        }

        const int64_t projected_margin = int64_t(dmds_full[id].free) - mem[id];
        LOG_INF(
            "%s:   - %s: %2" PRIu32 " layers (%2" PRIu32 " overflowing), %6" PRId64 " MiB used, %6" PRId64 " MiB free\n",
            __func__, dev_names[id].c_str(), ngl_per_device[id].n_layer, ngl_per_device[id].n_part, mem[id]/MiB, projected_margin/MiB);
    }

    // print info for devices that were not changed during the conversion from dense only to full layers:
    for (size_t id = id_dense_start + 1; id < nd; id++) {
        const int64_t projected_margin = int64_t(dmds_full[id].free) - mem[id];
        LOG_INF(
            "%s:   - %s: %2" PRIu32 " layers (%2" PRIu32 " overflowing), %6" PRId64 " MiB used, %6" PRId64 " MiB free\n",
            __func__, dev_names[id].c_str(), ngl_per_device[id].n_layer, ngl_per_device[id].n_part, mem[id]/MiB, projected_margin/MiB);
    }

    set_ngl_tensor_split_tbo(ngl_per_device, overflow_bufts, *mparams);
    verify_current_fit("layer placement");
}

enum common_params_fit_status common_fit_params(
        const char * path_model,
        llama_model_params * mparams,
        llama_context_params * cparams,
        float * tensor_split,
        llama_model_tensor_buft_override * tensor_buft_overrides,
        size_t * margins,
        uint32_t n_ctx_min,
        ggml_log_level log_level) {
    const int64_t t0_us = llama_time_us();
    common_params_fit_status status = COMMON_PARAMS_FIT_STATUS_SUCCESS;
    if (path_model == nullptr || path_model[0] == '\0' || mparams == nullptr || cparams == nullptr || margins == nullptr) {
        LOG_ERR("%s: invalid arguments passed to --fit; aborting fit before mutating parameters\n", __func__);
        const int64_t t1_us = llama_time_us();
        LOG_INF("%s: fitting params to free memory took %.2f seconds\n", __func__, (t1_us - t0_us) * 1e-6);
        return COMMON_PARAMS_FIT_STATUS_ERROR;
    }
    try {
        llama_model_params   mparams_fit = *mparams;
        llama_context_params cparams_fit = *cparams;

        std::vector<float> tensor_split_fit;
        float * tensor_split_ptr = tensor_split;
        if (tensor_split != nullptr) {
            tensor_split_fit.assign(tensor_split, tensor_split + llama_max_devices());
            tensor_split_ptr = tensor_split_fit.data();
        }

        std::vector<llama_model_tensor_buft_override> tensor_buft_overrides_fit;
        llama_model_tensor_buft_override * tensor_buft_overrides_ptr = tensor_buft_overrides;
        if (tensor_buft_overrides != nullptr) {
            tensor_buft_overrides_fit.assign(tensor_buft_overrides, tensor_buft_overrides + llama_max_tensor_buft_overrides());
            tensor_buft_overrides_ptr = tensor_buft_overrides_fit.data();
        }

        common_params_fit_impl(path_model, &mparams_fit, &cparams_fit,
            tensor_split_ptr, tensor_buft_overrides_ptr, margins, n_ctx_min, log_level);

        if (!tensor_split_fit.empty() && mparams_fit.tensor_split == tensor_split_fit.data()) {
            std::copy(tensor_split_fit.begin(), tensor_split_fit.end(), tensor_split);
            mparams_fit.tensor_split = tensor_split;
        }
        if (!tensor_buft_overrides_fit.empty() && mparams_fit.tensor_buft_overrides == tensor_buft_overrides_fit.data()) {
            std::copy(tensor_buft_overrides_fit.begin(), tensor_buft_overrides_fit.end(), tensor_buft_overrides);
            mparams_fit.tensor_buft_overrides = tensor_buft_overrides;
        }

        *mparams = mparams_fit;
        *cparams = cparams_fit;
        LOG_INF("%s: successfully fit params to free device memory\n", __func__);
    } catch (const common_params_fit_exception & e) {
        LOG_WRN("%s: failed to fit params to free device memory: %s\n", __func__, e.what());
        status = COMMON_PARAMS_FIT_STATUS_FAILURE;
    } catch (const std::runtime_error & e) {
        LOG_ERR("%s: encountered an error while trying to fit params to free device memory: %s\n", __func__, e.what());
        status = COMMON_PARAMS_FIT_STATUS_ERROR;
    }
    const int64_t t1_us = llama_time_us();
    LOG_INF("%s: fitting params to free memory took %.2f seconds\n", __func__, (t1_us - t0_us) * 1e-6);
    return status;
}

void common_memory_breakdown_print(const struct llama_context * ctx) {
    //const auto & devices = ctx->get_model().devices;
    const auto * model = llama_get_model(ctx);

    std::vector<ggml_backend_dev_t> devices;
    for (int i = 0; i < llama_model_n_devices(model); i++) {
        devices.push_back(llama_model_get_device(model, i));
    }

    llama_memory_breakdown memory_breakdown = llama_get_memory_breakdown(ctx);

    std::vector<std::array<std::string, 9>> table_data;
    table_data.reserve(devices.size());
    const std::string template_header = "%s: | %s | %s   %s    %s   %s   %s   %s    %s |\n";
    const std::string template_gpu    = "%s: | %s | %s = %s + (%s = %s + %s + %s) + %s |\n";
    const std::string template_other  = "%s: | %s | %s   %s    %s = %s + %s + %s    %s |\n";

    table_data.push_back({template_header, "memory breakdown [MiB]", "total", "free", "self", "model", "context", "compute", "unaccounted"});

    constexpr size_t MiB = 1024 * 1024;
    const std::vector<std::string> desc_prefixes_strip = {"NVIDIA ", "GeForce ", "Tesla ", "AMD ", "Radeon ", "Instinct "};

    // track seen buffer types to avoid double counting:
    std::set<ggml_backend_buffer_type_t> seen_buffer_types;

    // accumulative memory breakdown for each device and for host:
    std::vector<llama_memory_breakdown_data> mb_dev(devices.size());
    llama_memory_breakdown_data              mb_host;

    for (const auto & buft_mb : memory_breakdown) {
        ggml_backend_buffer_type_t          buft = buft_mb.first;
        const llama_memory_breakdown_data & mb   = buft_mb.second;
        if (ggml_backend_buft_is_host(buft)) {
            mb_host.model   += mb.model;
            mb_host.context += mb.context;
            mb_host.compute += mb.compute;
            seen_buffer_types.insert(buft);
            continue;
        }
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (dev) {
            int i_dev = -1;
            for (size_t i = 0; i < devices.size(); i++) {
                if (devices[i] == dev) {
                    i_dev = i;
                    break;
                }
            }
            if (i_dev != -1) {
                mb_dev[i_dev].model   += mb.model;
                mb_dev[i_dev].context += mb.context;
                mb_dev[i_dev].compute += mb.compute;
                seen_buffer_types.insert(buft);
                continue;
            }
        }
    }

    // print memory breakdown for each device:
    for (size_t i = 0; i < devices.size(); i++) {
        ggml_backend_dev_t dev = devices[i];
        llama_memory_breakdown_data mb = mb_dev[i];

        const std::string name = ggml_backend_dev_name(dev);
        std::string desc = ggml_backend_dev_description(dev);
        for (const std::string & prefix : desc_prefixes_strip) {
            if (desc.length() >= prefix.length() && desc.substr(0, prefix.length()) == prefix) {
                desc = desc.substr(prefix.length());
            }
        }

        size_t free, total;
        ggml_backend_dev_memory(dev, &free, &total);

        const size_t self = mb.model + mb.context + mb.compute;
        const int64_t unaccounted = static_cast<int64_t>(total) - static_cast<int64_t>(free) - static_cast<int64_t>(self);

        table_data.push_back({
            template_gpu,
            "  - " + name + " (" + desc + ")",
            std::to_string(total / MiB),
            std::to_string(free / MiB),
            std::to_string(self / MiB),
            std::to_string(mb.model / MiB),
            std::to_string(mb.context / MiB),
            std::to_string(mb.compute / MiB),
            std::to_string(unaccounted / static_cast<int64_t>(MiB))});
    }

    // print memory breakdown for host:
    {
        const size_t self = mb_host.model + mb_host.context + mb_host.compute;
        table_data.push_back({
            template_other,
            "  - Host",
            "", // total
            "", // free
            std::to_string(self / MiB),
            std::to_string(mb_host.model / MiB),
            std::to_string(mb_host.context / MiB),
            std::to_string(mb_host.compute / MiB),
            ""}); // unaccounted
    }

    // print memory breakdown for all remaining buffer types:
    for (const auto & buft_mb : memory_breakdown) {
        ggml_backend_buffer_type_t          buft = buft_mb.first;
        const llama_memory_breakdown_data & mb   = buft_mb.second;
        if (seen_buffer_types.count(buft) == 1) {
            continue;
        }
        const std::string name = ggml_backend_buft_name(buft);
        const size_t self = mb.model + mb.context + mb.compute;
        table_data.push_back({
            template_other,
            "  - " + name,
            "", // total
            "", // free
            std::to_string(self / MiB),
            std::to_string(mb.model / MiB),
            std::to_string(mb.context / MiB),
            std::to_string(mb.compute / MiB),
            ""}); // unaccounted
        seen_buffer_types.insert(buft);
    }

    for (size_t j = 1; j < table_data[0].size(); j++) {
        size_t max_len = 0;
        for (const auto & td : table_data) {
            max_len = std::max(max_len, td[j].length());
        }
        for (auto & td : table_data) {
            td[j].insert(j == 1 ? td[j].length() : 0, max_len - td[j].length(), ' ');
        }
    }
    for (const auto & td : table_data) {
        LOG_INF(td[0].c_str(),
            __func__, td[1].c_str(), td[2].c_str(), td[3].c_str(), td[4].c_str(), td[5].c_str(),
            td[6].c_str(), td[7].c_str(), td[8].c_str());
    }
}

void common_fit_print(
        const char * path_model,
        llama_model_params * mparams,
        llama_context_params * cparams) {
    std::vector<ggml_backend_dev_t> devs;
    uint32_t hp_ngl = 0; // hparams.n_gpu_layers
    uint32_t hp_nct = 0; // hparams.n_ctx_train
    uint32_t hp_nex = 0; // hparams.n_expert

    auto dmd = common_get_device_memory_data(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, GGML_LOG_LEVEL_ERROR);
    GGML_ASSERT(dmd.size() == devs.size() + 1);

    for (size_t id = 0; id < devs.size(); id++) {
        printf("%s ",  ggml_backend_dev_name(devs[id]));
        printf("%zu ", dmd[id].mb.model/1024/1024);
        printf("%zu ", dmd[id].mb.context/1024/1024);
        printf("%zu ", dmd[id].mb.compute/1024/1024);
        printf("\n");
    }

    printf("Host ");
    printf("%zu ", dmd.back().mb.model/1024/1024);
    printf("%zu ", dmd.back().mb.context/1024/1024);
    printf("%zu ", dmd.back().mb.compute/1024/1024);
    printf("\n");
}
