#include "llama-tiered.h"

#include "llama-impl.h"

#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct llama_tiered_model {
    llama_model * model = nullptr;
    llama_tiered_memory_stats stats = {};

    std::vector<ggml_backend_dev_t> devices;
    std::vector<llama_model_tensor_buft_override> overrides;
};

namespace {

thread_local std::string tiered_last_error;

struct parsed_tensor {
    std::string name;
    uint64_t size = 0;
    uint64_t n_expert = 0;
    double active_fraction = 1.0;
    bool ssd_eligible = false;
    ggml_cuda_tiered_memory_tier tier = GGML_CUDA_TIERED_MEMORY_VRAM;
};

struct model_metadata {
    std::string architecture;
    uint64_t expert_count = 0;
    uint64_t expert_used_count = 0;
    uint64_t split_count = 1;
    uint64_t split_no = 0;
    std::vector<parsed_tensor> tensors;
};

using tiered_plan_begin_fn = ggml_backend_buffer_type_t (*)(
        ggml_backend_dev_t,
        const ggml_cuda_tiered_tensor_plan *,
        size_t,
        ggml_cuda_tiered_plan_options);
using tiered_plan_end_fn = void (*)(ggml_backend_dev_t);

struct tiered_plan_guard {
    ggml_backend_dev_t dev = nullptr;
    tiered_plan_end_fn end = nullptr;

    ~tiered_plan_guard() {
        if (dev && end) {
            end(dev);
        }
    }
};

struct writable_mmap_guard {
    bool previous = llama_mmap_get_writable();

    writable_mmap_guard() {
        llama_mmap_set_writable(true);
    }

    ~writable_mmap_guard() {
        llama_mmap_set_writable(previous);
    }
};

bool is_expert_stack(const std::string & name) {
    return name.find("_exps") != std::string::npos ||
           name.find(".experts.") != std::string::npos;
}

bool is_streamable_expert_weight(const std::string & name) {
    const bool expert = name.find("_exps.weight") != std::string::npos ||
                        name.find(".experts.") != std::string::npos;
    const bool weight = name.size() >= 7 &&
                        name.compare(name.size() - 7, 7, ".weight") == 0;
    return expert && weight;
}

uint64_t get_unsigned(const gguf_context * ctx, const char * key, uint64_t fallback = 0) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        return fallback;
    }

    switch (gguf_get_kv_type(ctx, id)) {
        case GGUF_TYPE_UINT8:  return gguf_get_val_u8(ctx, id);
        case GGUF_TYPE_UINT16: return gguf_get_val_u16(ctx, id);
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(ctx, id);
        case GGUF_TYPE_UINT64: return gguf_get_val_u64(ctx, id);
        case GGUF_TYPE_INT8:   return std::max<int64_t>(0, gguf_get_val_i8(ctx, id));
        case GGUF_TYPE_INT16:  return std::max<int64_t>(0, gguf_get_val_i16(ctx, id));
        case GGUF_TYPE_INT32:  return std::max<int64_t>(0, gguf_get_val_i32(ctx, id));
        case GGUF_TYPE_INT64:  return std::max<int64_t>(0, gguf_get_val_i64(ctx, id));
        default:
            throw std::runtime_error(std::string("GGUF key has a non-integer type: ") + key);
    }
}

std::string get_string(const gguf_context * ctx, const char * key) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        return {};
    }
    if (gguf_get_kv_type(ctx, id) != GGUF_TYPE_STRING) {
        throw std::runtime_error(std::string("GGUF key has a non-string type: ") + key);
    }
    return gguf_get_val_str(ctx, id);
}

void append_gguf_file(
        const std::string & path,
        bool first,
        model_metadata & result,
        std::set<std::string> & tensor_names) {
    ggml_context * tensor_ctx = nullptr;
    gguf_init_params params = {
        /* .no_alloc = */ true,
        /* .ctx      = */ &tensor_ctx,
    };

    gguf_context * raw_gguf = gguf_init_from_file(path.c_str(), params);
    if (!raw_gguf) {
        throw std::runtime_error("failed to read GGUF metadata from " + path);
    }

    struct metadata_guard {
        gguf_context * gguf;
        ggml_context * tensors;
        ~metadata_guard() {
            gguf_free(gguf);
            ggml_free(tensors);
        }
    } guard { raw_gguf, tensor_ctx };

    if (first) {
        result.architecture = get_string(raw_gguf, "general.architecture");
        if (result.architecture.empty()) {
            throw std::runtime_error("GGUF is missing general.architecture");
        }

        const std::string expert_count_key = result.architecture + ".expert_count";
        const std::string expert_used_key = result.architecture + ".expert_used_count";
        result.expert_count = get_unsigned(raw_gguf, expert_count_key.c_str(), 0);
        result.expert_used_count = get_unsigned(raw_gguf, expert_used_key.c_str(), 0);
        result.split_count = std::max<uint64_t>(1, get_unsigned(raw_gguf, "split.count", 1));
        result.split_no = get_unsigned(raw_gguf, "split.no", 0);
    }

    const int64_t n_tensors = gguf_get_n_tensors(raw_gguf);
    result.tensors.reserve(result.tensors.size() + static_cast<size_t>(n_tensors));

    for (int64_t index = 0; index < n_tensors; ++index) {
        const char * name_ptr = gguf_get_tensor_name(raw_gguf, index);
        if (!name_ptr) {
            throw std::runtime_error("GGUF tensor without a name");
        }
        const std::string name(name_ptr);
        if (!tensor_names.insert(name).second) {
            throw std::runtime_error("duplicate GGUF tensor: " + name);
        }

        parsed_tensor tensor;
        tensor.name = name;
        tensor.size = gguf_get_tensor_size(raw_gguf, index);
        tensor.ssd_eligible = is_streamable_expert_weight(name);

        if (is_expert_stack(name)) {
            const int64_t * ne = gguf_get_tensor_ne(raw_gguf, index);
            const uint64_t tensor_experts = ne && ne[2] > 1 ? static_cast<uint64_t>(ne[2]) : 0;
            tensor.n_expert = result.expert_count ? result.expert_count : tensor_experts;
            if (tensor.n_expert == 0 || result.expert_used_count == 0) {
                throw std::runtime_error(
                        "expert tensor requires expert_count and expert_used_count metadata: " + name);
            }
            tensor.active_fraction = std::min(
                    1.0,
                    static_cast<double>(result.expert_used_count) /
                    static_cast<double>(tensor.n_expert));
        }

        result.tensors.push_back(std::move(tensor));
    }
}

std::vector<std::string> resolve_model_files(const char * path_model, model_metadata & metadata) {
    if (!path_model || path_model[0] == '\0') {
        throw std::invalid_argument("path_model is empty");
    }

    std::vector<std::string> paths = { path_model };
    std::set<std::string> tensor_names;
    append_gguf_file(paths.front(), true, metadata, tensor_names);

    if (metadata.split_count <= 1) {
        return paths;
    }
    if (metadata.split_no != 0) {
        throw std::runtime_error("tiered loading must start from split 0");
    }
    if (metadata.split_count > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error("GGUF split count is too large");
    }

    std::vector<char> prefix(llama_path_max(), 0);
    const int prefix_length = llama_split_prefix(
            prefix.data(), prefix.size(), path_model,
            0, static_cast<int32_t>(metadata.split_count));
    if (prefix_length <= 0) {
        throw std::runtime_error(
                "split GGUF does not use the conventional -00001-of-NNNNN naming scheme");
    }

    paths.clear();
    paths.reserve(static_cast<size_t>(metadata.split_count));
    std::vector<char> split_path(llama_path_max(), 0);
    for (int32_t index = 0; index < static_cast<int32_t>(metadata.split_count); ++index) {
        const int length = llama_split_path(
                split_path.data(), split_path.size(), prefix.data(),
                index, static_cast<int32_t>(metadata.split_count));
        if (length <= 0) {
            throw std::runtime_error("failed to construct GGUF split path");
        }
        paths.emplace_back(split_path.data(), static_cast<size_t>(length));
    }

    metadata.tensors.clear();
    tensor_names.clear();
    for (size_t index = 0; index < paths.size(); ++index) {
        append_gguf_file(paths[index], index == 0, metadata, tensor_names);
    }
    return paths;
}

void assign_tiers(
        model_metadata & metadata,
        uint64_t vram_budget,
        uint64_t dram_budget,
        llama_tiered_memory_stats & stats) {
    std::vector<size_t> priority(metadata.tensors.size());
    for (size_t i = 0; i < priority.size(); ++i) {
        priority[i] = i;
    }

    std::stable_sort(priority.begin(), priority.end(), [&](size_t lhs, size_t rhs) {
        const parsed_tensor & a = metadata.tensors[lhs];
        const parsed_tensor & b = metadata.tensors[rhs];
        if (a.active_fraction != b.active_fraction) {
            return a.active_fraction > b.active_fraction;
        }
        if (a.ssd_eligible != b.ssd_eligible) {
            // Keep routers, scales, norms, and other non-streamable tensors in a
            // resident tier before consuming DRAM with expert matrices.
            return !a.ssd_eligible;
        }
        return a.name < b.name;
    });

    uint64_t vram_remaining = vram_budget;
    for (const size_t index : priority) {
        parsed_tensor & tensor = metadata.tensors[index];
        if (tensor.size <= vram_remaining) {
            tensor.tier = GGML_CUDA_TIERED_MEMORY_VRAM;
            vram_remaining -= tensor.size;
        } else {
            tensor.tier = GGML_CUDA_TIERED_MEMORY_SSD;
        }
    }

    uint64_t dram_remaining = dram_budget;
    for (const size_t index : priority) {
        parsed_tensor & tensor = metadata.tensors[index];
        if (tensor.tier != GGML_CUDA_TIERED_MEMORY_SSD) {
            continue;
        }
        if (tensor.size <= dram_remaining) {
            tensor.tier = GGML_CUDA_TIERED_MEMORY_DRAM;
            dram_remaining -= tensor.size;
        }
    }

    for (const parsed_tensor & tensor : metadata.tensors) {
        const double active_bytes = static_cast<double>(tensor.size) * tensor.active_fraction;
        switch (tensor.tier) {
            case GGML_CUDA_TIERED_MEMORY_VRAM:
                stats.vram_bytes += tensor.size;
                stats.active_vram_bytes_per_token += active_bytes;
                break;
            case GGML_CUDA_TIERED_MEMORY_DRAM:
                stats.dram_bytes += tensor.size;
                stats.active_dram_bytes_per_token += active_bytes;
                break;
            case GGML_CUDA_TIERED_MEMORY_SSD:
                if (!tensor.ssd_eligible) {
                    throw std::runtime_error(
                            "VRAM + DRAM budgets are too small for non-streamable tensor: " + tensor.name);
                }
                stats.ssd_bytes += tensor.size;
                stats.active_ssd_bytes_per_token += active_bytes;
                stats.ssd_tensor_count++;
                break;
        }
    }
    stats.tensor_count = static_cast<uint32_t>(metadata.tensors.size());
}

void copy_user_overrides(
        const llama_model_tensor_buft_override * source,
        std::vector<llama_model_tensor_buft_override> & destination) {
    if (!source) {
        return;
    }
    for (const llama_model_tensor_buft_override * current = source;
            current->pattern != nullptr; ++current) {
        destination.push_back(*current);
    }
}

} // namespace

extern "C" llama_tiered_memory_params llama_tiered_memory_default_params(void) {
    llama_tiered_memory_params result = {};
    result.vram_budget_bytes = 0;
    result.dram_budget_bytes = 0;
    result.vram_reserve_bytes = 1024ull * 1024ull * 1024ull;
    result.ssd_cache_bytes = 0;
    result.main_gpu = 0;
    result.strict = true;
    return result;
}

extern "C" llama_tiered_model * llama_tiered_model_load_from_file(
        const char * path_model,
        llama_model_params model_params,
        llama_tiered_memory_params tiered_params) {
    tiered_last_error.clear();

    try {
        if (tiered_params.main_gpu < 0) {
            throw std::invalid_argument("main_gpu must be non-negative");
        }
        if (tiered_params.ssd_cache_bytes > std::numeric_limits<size_t>::max()) {
            throw std::invalid_argument("ssd_cache_bytes exceeds the platform size limit");
        }

        ggml_backend_load_all();

#if defined(GGML_USE_CUDA) && !defined(GGML_BACKEND_DL)
        ggml_backend_cuda_tiered_register();
#endif

        ggml_backend_reg_t tiered_reg = ggml_backend_reg_by_name("CUDA_TIERED");
        if (!tiered_reg) {
            throw std::runtime_error(
                    "CUDA_TIERED backend is unavailable; build with GGML_CUDA=ON and GGML_BACKEND_DL=OFF");
        }
        if (static_cast<size_t>(tiered_params.main_gpu) >=
                ggml_backend_reg_dev_count(tiered_reg)) {
            throw std::out_of_range("main_gpu is outside the CUDA device range");
        }

        ggml_backend_dev_t tiered_dev = ggml_backend_reg_dev_get(
                tiered_reg, static_cast<size_t>(tiered_params.main_gpu));

        size_t free_vram = 0;
        size_t total_vram = 0;
        ggml_backend_dev_memory(tiered_dev, &free_vram, &total_vram);
        if (free_vram == 0 && tiered_params.vram_budget_bytes == 0) {
            throw std::runtime_error("CUDA backend did not report free VRAM");
        }

        uint64_t vram_budget = tiered_params.vram_budget_bytes;
        if (vram_budget == 0) {
            vram_budget = free_vram > tiered_params.vram_reserve_bytes ?
                    static_cast<uint64_t>(free_vram) - tiered_params.vram_reserve_bytes : 0;
        }
        if (tiered_params.ssd_cache_bytes > vram_budget) {
            throw std::invalid_argument("ssd_cache_bytes exceeds the available VRAM budget");
        }
        vram_budget -= tiered_params.ssd_cache_bytes;

        model_metadata metadata;
        std::vector<std::string> paths = resolve_model_files(path_model, metadata);

        std::unique_ptr<llama_tiered_model> owner(new llama_tiered_model);
        assign_tiers(metadata, vram_budget, tiered_params.dram_budget_bytes, owner->stats);

        // Every weight fits in VRAM, so there is nothing to stage. Load through
        // the plain CUDA device instead: the tiered buffer type would allocate
        // each weight separately and route compute through an extra wrapper for
        // no benefit.
        if (owner->stats.dram_bytes == 0 && owner->stats.ssd_bytes == 0) {
            ggml_backend_reg_t cuda_reg = ggml_backend_reg_by_name("CUDA");
            if (!cuda_reg) {
                throw std::runtime_error("CUDA backend is unavailable");
            }
            if (static_cast<size_t>(tiered_params.main_gpu) >= ggml_backend_reg_dev_count(cuda_reg)) {
                throw std::out_of_range("main_gpu is outside the CUDA device range");
            }

            LLAMA_LOG_INFO("%s: all %.2f MiB of weights fit in VRAM; using the CUDA backend\n",
                    __func__, owner->stats.vram_bytes / 1024.0 / 1024.0);

            owner->devices = { ggml_backend_reg_dev_get(cuda_reg, static_cast<size_t>(tiered_params.main_gpu)), nullptr };
            copy_user_overrides(model_params.tensor_buft_overrides, owner->overrides);
            owner->overrides.push_back({ nullptr, nullptr });

            model_params.devices = owner->devices.data();
            model_params.tensor_buft_overrides = owner->overrides.data();
            model_params.n_gpu_layers = -1;
            model_params.main_gpu = 0;
            model_params.split_mode = LLAMA_SPLIT_MODE_NONE;
            model_params.tensor_split = nullptr;

            if (paths.size() == 1) {
                owner->model = llama_model_load_from_file(paths[0].c_str(), model_params);
            } else {
                std::vector<const char *> path_ptrs;
                path_ptrs.reserve(paths.size());
                for (const std::string & path : paths) {
                    path_ptrs.push_back(path.c_str());
                }
                owner->model = llama_model_load_from_splits(
                        path_ptrs.data(), path_ptrs.size(), model_params);
            }
            if (!owner->model) {
                throw std::runtime_error("llama_model loading failed");
            }
            return owner.release();
        }

        std::vector<ggml_cuda_tiered_tensor_plan> entries;
        entries.reserve(metadata.tensors.size());
        for (const parsed_tensor & tensor : metadata.tensors) {
            entries.push_back({ tensor.name.c_str(), tensor.tier });
        }

        auto plan_begin = reinterpret_cast<tiered_plan_begin_fn>(
                ggml_backend_reg_get_proc_address(
                        tiered_reg, "ggml_backend_cuda_tiered_plan_begin"));
        auto plan_end = reinterpret_cast<tiered_plan_end_fn>(
                ggml_backend_reg_get_proc_address(
                        tiered_reg, "ggml_backend_cuda_tiered_plan_end"));
        if (!plan_begin || !plan_end) {
            throw std::runtime_error("CUDA_TIERED backend does not expose its plan API");
        }

        ggml_cuda_tiered_plan_options backend_options = {};
        backend_options.ssd_cache_bytes = static_cast<size_t>(tiered_params.ssd_cache_bytes);
        backend_options.strict = tiered_params.strict;

        ggml_backend_buffer_type_t tiered_buft = plan_begin(
                tiered_dev, entries.data(), entries.size(), backend_options);
        if (!tiered_buft) {
            throw std::runtime_error("CUDA_TIERED backend rejected the generated plan");
        }
        tiered_plan_guard plan_guard { tiered_dev, plan_end };

        owner->devices = { tiered_dev, nullptr };
        copy_user_overrides(model_params.tensor_buft_overrides, owner->overrides);
        owner->overrides.push_back({ ".*", tiered_buft });
        owner->overrides.push_back({ nullptr, nullptr });

        model_params.devices = owner->devices.data();
        model_params.tensor_buft_overrides = owner->overrides.data();
        model_params.n_gpu_layers = -2;
        model_params.split_mode = LLAMA_SPLIT_MODE_NONE;
        model_params.load_mode = LLAMA_LOAD_MODE_MMAP;
        model_params.mmap_prefetch = false;
        model_params.main_gpu = 0;
        model_params.tensor_split = nullptr;
        model_params.use_extra_bufts = false;
        model_params.no_host = false;
        model_params.no_alloc = false;

        writable_mmap_guard mmap_guard;
        if (paths.size() == 1) {
            owner->model = llama_model_load_from_file(paths[0].c_str(), model_params);
        } else {
            std::vector<const char *> path_ptrs;
            path_ptrs.reserve(paths.size());
            for (const std::string & path : paths) {
                path_ptrs.push_back(path.c_str());
            }
            owner->model = llama_model_load_from_splits(
                    path_ptrs.data(), path_ptrs.size(), model_params);
        }

        if (!owner->model) {
            throw std::runtime_error("llama_model loading failed");
        }

        return owner.release();
    } catch (const std::exception & error) {
        tiered_last_error = error.what();
        return nullptr;
    } catch (...) {
        tiered_last_error = "unknown tiered-memory loading error";
        return nullptr;
    }
}

extern "C" llama_model * llama_tiered_model_get_model(llama_tiered_model * tiered_model) {
    return tiered_model ? tiered_model->model : nullptr;
}

extern "C" const llama_tiered_memory_stats * llama_tiered_model_get_stats(
        const llama_tiered_model * tiered_model) {
    return tiered_model ? &tiered_model->stats : nullptr;
}

extern "C" void llama_tiered_model_free(llama_tiered_model * tiered_model) {
    if (!tiered_model) {
        return;
    }
    llama_model_free(tiered_model->model);
    delete tiered_model;
}

extern "C" const char * llama_tiered_last_error(void) {
    return tiered_last_error.c_str();
}
