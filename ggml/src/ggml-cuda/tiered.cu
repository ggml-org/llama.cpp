#include "ggml-cuda.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__linux__) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <unistd.h>

namespace {

#define TIERED_CUDA_CHECK(expr) do {                                                   \
    const cudaError_t tiered_cuda_status = (expr);                                     \
    if (tiered_cuda_status != cudaSuccess) {                                           \
        throw std::runtime_error(std::string(#expr) + ": " +                         \
                cudaGetErrorString(tiered_cuda_status));                               \
    }                                                                                  \
} while (0)

#define TIERED_CU_CHECK(expr) do {                                                     \
    const CUresult tiered_cu_status = (expr);                                          \
    if (tiered_cu_status != CUDA_SUCCESS) {                                            \
        const char * tiered_cu_name = nullptr;                                         \
        const char * tiered_cu_text = nullptr;                                         \
        cuGetErrorName(tiered_cu_status, &tiered_cu_name);                             \
        cuGetErrorString(tiered_cu_status, &tiered_cu_text);                           \
        throw std::runtime_error(std::string(#expr) + ": " +                         \
                (tiered_cu_name ? tiered_cu_name : "CUDA_ERROR") + " (" +             \
                (tiered_cu_text ? tiered_cu_text : "unknown") + ")");                 \
    }                                                                                  \
} while (0)

struct tiered_plan {
    std::unordered_map<std::string, ggml_cuda_tiered_memory_tier> tensors;
    ggml_cuda_tiered_plan_options options = {};
};

thread_local std::unordered_map<int, std::shared_ptr<tiered_plan>> tls_plans;

struct host_registration {
    uintptr_t begin = 0;
    uintptr_t end = 0;
    bool owned = false;
};

struct expert_cache_slot {
    int32_t expert = -1;
    uint64_t last_hit_epoch = 0;
    size_t device_offset = 0;
};

struct expert_cache_state {
    size_t slot_size = 0;
    std::vector<uint16_t> heat;
    std::vector<expert_cache_slot> slots;
    std::vector<int32_t> slot_of_expert; // -1 when the expert is not resident
};

struct tensor_state {
    ggml_cuda_tiered_memory_tier tier = GGML_CUDA_TIERED_MEMORY_VRAM;
    void * host_ptr = nullptr;
    void * device_ptr = nullptr;
    bool host_ptr_owned = false;
    size_t size = 0;
    size_t alloc_size = 0;

    CUdeviceptr sparse_address = 0;
    size_t sparse_size = 0;
    size_t sparse_granularity = 0;
    std::unique_ptr<expert_cache_state> cache;
};

struct tiered_buffer_context {
    int device = 0;
    void * address_space = nullptr;
    size_t address_space_size = 0;
    std::shared_ptr<tiered_plan> plan;
    std::unordered_map<const ggml_tensor *, std::unique_ptr<tensor_state>> tensors;
    std::vector<host_registration> registrations;
    void * expert_staging = nullptr;
    size_t expert_staging_size = 0;
    void * dram_staging = nullptr;
    size_t dram_staging_size = 0;
    void * expert_cache = nullptr;
    size_t expert_cache_size = 0;
    bool expert_cache_initialized = false;
    cudaStream_t copy_stream = nullptr;
    uint64_t cache_epoch = 0;
    uint64_t cache_requests = 0;
    uint64_t cache_hits = 0;
    uint64_t cache_hit_bytes = 0;
    uint64_t cache_miss_bytes = 0;
    uint64_t cache_h2d_bytes = 0;
    uint64_t cache_d2d_bytes = 0;
    uint64_t cache_admissions = 0;
    uint64_t cache_evictions = 0;
    size_t host_tensors = 0;
    const ggml_tensor * ids_cache_tensor = nullptr;
    uint64_t ids_cache_run = 0;
    std::vector<int32_t> ids_cache_values;
    // Identity ids for the packed layout. Decode always stages the same number
    // of experts in router order, so [0, 1, ... k-1] never changes and is
    // uploaded once.
    void * packed_ids = nullptr;
    size_t packed_ids_count = 0;
    std::mutex compute_mutex;
};

// Bumped once per graph, so a cached ids read is only reused inside the graph
// that produced it.
std::atomic<uint64_t> tiered_graph_run{1};

// Host-tier tensors across all live tiered buffers. While this is zero every
// weight is VRAM resident, so graph compute needs no staging at all and can
// skip its per-node scan.
std::atomic<size_t> tiered_host_tensors{0};

struct tiered_buffer_type_context {
    int device = 0;
    ggml_backend_dev_t device_handle = nullptr;
    std::string name;
};

struct tiered_backend_context {
    int device = 0;
    ggml_backend_t inner = nullptr;
    std::string name;
};

struct tiered_device_context {
    int device = 0;
    ggml_backend_dev_t inner_device = nullptr;
    ggml_backend_buffer_type_t tiered_buft = nullptr;
    std::string name;
    std::string description;
    std::string device_id;
};

struct tiered_registry_context {
    std::vector<ggml_backend_dev_t> devices;
};

static ggml_backend_reg tiered_registry;
static tiered_registry_context tiered_registry_ctx;
static std::once_flag tiered_register_once;

static size_t page_size() {
    static const size_t value = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    return value;
}

static uintptr_t align_down(uintptr_t value, size_t alignment) {
    return value & ~(static_cast<uintptr_t>(alignment) - 1);
}

static uintptr_t align_up(uintptr_t value, size_t alignment) {
    return (value + alignment - 1) & ~(static_cast<uintptr_t>(alignment) - 1);
}

static size_t align_up_size(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

static const ggml_tensor * tiered_view_base(const ggml_tensor * tensor, size_t * offset) {
    *offset = 0;
    while (tensor && tensor->view_src) {
        *offset += tensor->view_offs;
        tensor = tensor->view_src;
    }
    return tensor;
}

static ggml_status tiered_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor);

// Identifies tiered buffers by init_tensor's function pointer rather than a
// ggml_backend_buft_name() string compare, since this runs for every src of
// every graph node, every token.
static tiered_buffer_context * buffer_context(const ggml_tensor * tensor) {
    if (!tensor) {
        return nullptr;
    }
    size_t view_offset = 0;
    tensor = tiered_view_base(tensor, &view_offset);
    GGML_UNUSED(view_offset);
    ggml_backend_buffer_t buffer = tensor ? tensor->buffer : nullptr;
    if (!buffer || !buffer->context || buffer->iface.init_tensor != tiered_buffer_init_tensor) {
        return nullptr;
    }
    return static_cast<tiered_buffer_context *>(buffer->context);
}

// tensor->extra caches the resolved tensor_state, set in
// tiered_buffer_init_tensor, so the graph-compute hot path is a field read
// instead of an unordered_map lookup.
static tensor_state * state_for(const ggml_tensor * tensor) {
    if (!tensor) {
        return nullptr;
    }
    size_t view_offset = 0;
    const ggml_tensor * base = tiered_view_base(tensor, &view_offset);
    if (!buffer_context(tensor)) {
        return nullptr;
    }
    return base ? static_cast<tensor_state *>(base->extra) : nullptr;
}

static void set_device(int device) {
    TIERED_CUDA_CHECK(cudaSetDevice(device));
}

static void add_registration(tiered_buffer_context * ctx, uintptr_t begin, uintptr_t end, bool owned) {
    ctx->registrations.push_back({begin, end, owned});
    std::sort(ctx->registrations.begin(), ctx->registrations.end(), [](const auto & lhs, const auto & rhs) {
        return lhs.begin < rhs.begin;
    });
}

static cudaError_t register_host_range(void * ptr, size_t size) {
    return cudaHostRegister(ptr, size, cudaHostRegisterPortable | cudaHostRegisterMapped);
}

// Staging runs on its own stream so transfers do not serialize against the
// compute stream of the inner CUDA backend.
static cudaStream_t ensure_copy_stream(tiered_buffer_context * ctx) {
    if (!ctx->copy_stream) {
        TIERED_CUDA_CHECK(cudaStreamCreateWithFlags(&ctx->copy_stream, cudaStreamNonBlocking));
    }
    return ctx->copy_stream;
}

static void ensure_registered(tiered_buffer_context * ctx, void * ptr, size_t size) {
    const size_t page = page_size();
    const uintptr_t wanted_begin = align_down(reinterpret_cast<uintptr_t>(ptr), page);
    const uintptr_t wanted_end = align_up(reinterpret_cast<uintptr_t>(ptr) + size, page);

    uintptr_t cursor = wanted_begin;
    const auto existing = ctx->registrations;
    for (const auto & registration : existing) {
        if (registration.end <= cursor) {
            continue;
        }
        if (registration.begin >= wanted_end) {
            break;
        }
        if (registration.begin > cursor) {
            const uintptr_t gap_end = std::min(registration.begin, wanted_end);
            const size_t gap_size = gap_end - cursor;
            const cudaError_t status = register_host_range(reinterpret_cast<void *>(cursor), gap_size);
            if (status == cudaErrorHostMemoryAlreadyRegistered) {
                (void) cudaGetLastError();
                add_registration(ctx, cursor, gap_end, false);
            } else {
                TIERED_CUDA_CHECK(status);
                add_registration(ctx, cursor, gap_end, true);
            }
            cursor = gap_end;
        }
        cursor = std::max(cursor, registration.end);
        if (cursor >= wanted_end) {
            return;
        }
    }

    if (cursor < wanted_end) {
        const size_t gap_size = wanted_end - cursor;
        const cudaError_t status = register_host_range(reinterpret_cast<void *>(cursor), gap_size);
        if (status == cudaErrorHostMemoryAlreadyRegistered) {
            (void) cudaGetLastError();
            add_registration(ctx, cursor, wanted_end, false);
        } else {
            TIERED_CUDA_CHECK(status);
            add_registration(ctx, cursor, wanted_end, true);
        }
    }
}

static void copy_host_to_device(tiered_buffer_context * ctx, void * dst, const void * src, size_t size) {
    if (size == 0) {
        return;
    }

    const uintptr_t begin = reinterpret_cast<uintptr_t>(src);
    const uintptr_t end = begin + size;
    if (end < begin) {
        throw std::runtime_error("host-to-device copy range overflow");
    }

    std::vector<uintptr_t> cuts = { begin, end };
    for (const auto & registration : ctx->registrations) {
        if (registration.end <= begin) {
            continue;
        }
        if (registration.begin >= end) {
            break;
        }
        if (registration.begin > begin && registration.begin < end) {
            cuts.push_back(registration.begin);
        }
        if (registration.end > begin && registration.end < end) {
            cuts.push_back(registration.end);
        }
    }

    std::sort(cuts.begin(), cuts.end());
    cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
    const cudaStream_t stream = ensure_copy_stream(ctx);
    for (size_t i = 1; i < cuts.size(); ++i) {
        const size_t offset = cuts[i - 1] - begin;
        const size_t copy_size = cuts[i] - cuts[i - 1];
        TIERED_CUDA_CHECK(cudaMemcpyAsync(
                static_cast<char *>(dst) + offset,
                reinterpret_cast<const void *>(cuts[i - 1]),
                copy_size,
                cudaMemcpyHostToDevice,
                stream));
    }
}

static size_t vmm_granularity(int device) {
#if defined(GGML_CUDA_NO_VMM)
    GGML_UNUSED(device);
    return 0;
#else
    TIERED_CU_CHECK(cuInit(0));
    CUmemAllocationProp property = {};
    property.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    property.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    property.location.id = device;

    size_t granularity = 0;
    TIERED_CU_CHECK(cuMemGetAllocationGranularity(
            &granularity, &property, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    return granularity;
#endif
}

static bool is_ssd_eligible(const ggml_tensor * tensor) {
    if (!tensor || tensor->ne[2] <= 1) {
        return false;
    }
    const std::string name = tensor->name;
    const bool expert = name.find("_exps.weight") != std::string::npos ||
                        name.find(".experts.") != std::string::npos;
    const bool weight = name.size() >= 7 &&
                        name.compare(name.size() - 7, 7, ".weight") == 0;
    return expert && weight;
}

static void init_sparse_tensor(tiered_buffer_context * ctx, ggml_tensor * tensor, tensor_state * state) {
#if defined(GGML_CUDA_NO_VMM)
    GGML_UNUSED(ctx);
    GGML_UNUSED(tensor);
    GGML_UNUSED(state);
    throw std::runtime_error("tiered SSD streaming requires CUDA VMM support");
#else
    if (!is_ssd_eligible(tensor)) {
        throw std::runtime_error(std::string("SSD tier only supports stacked MoE expert tensors: ") + tensor->name);
    }
    state->sparse_granularity = vmm_granularity(ctx->device);
    state->sparse_size = align_up_size(state->size, state->sparse_granularity);
    TIERED_CU_CHECK(cuMemAddressReserve(
            &state->sparse_address, state->sparse_size,
            state->sparse_granularity, 0, 0));
    tensor->data = reinterpret_cast<void *>(state->sparse_address);
#endif
}

static void free_sparse_tensor(tensor_state * state) {
#if !defined(GGML_CUDA_NO_VMM)
    if (state->sparse_address) {
        (void) cuMemAddressFree(state->sparse_address, state->sparse_size);
        state->sparse_address = 0;
    }
#else
    GGML_UNUSED(state);
#endif
}

static void disable_expert_cache(tiered_buffer_context * ctx) {
    // admissions may still be reading/writing the cache on the copy stream
    if (ctx->copy_stream) {
        (void) cudaStreamSynchronize(ctx->copy_stream);
    }
    for (auto & entry : ctx->tensors) {
        entry.second->cache.reset();
    }
    if (ctx->expert_cache) {
        (void) cudaFree(ctx->expert_cache);
        ctx->expert_cache = nullptr;
    }
    ctx->expert_cache_size = 0;
}

static void initialize_expert_cache(tiered_buffer_context * ctx) {
    if (ctx->expert_cache_initialized) {
        return;
    }
    ctx->expert_cache_initialized = true;

    const size_t requested_budget = ctx->plan->options.ssd_cache_bytes;
    if (requested_budget == 0) {
        return;
    }

    struct cache_candidate {
        ggml_tensor * tensor = nullptr;
        tensor_state * state = nullptr;
        size_t slot_size = 0;
        size_t n_slots = 0;
    };

    try {
        std::vector<cache_candidate> candidates;
        for (auto & entry : ctx->tensors) {
            ggml_tensor * tensor = const_cast<ggml_tensor *>(entry.first);
            tensor_state * state = entry.second.get();
            if (state->tier == GGML_CUDA_TIERED_MEMORY_VRAM ||
                    !state->host_ptr || !is_ssd_eligible(tensor) ||
                    tensor->ne[2] <= 1 || tensor->ne[3] != 1 ||
                    state->size != ggml_nbytes(tensor)) {
                continue;
            }

            const size_t guard = std::min<size_t>(tensor->nb[2], 512);
            if (tensor->nb[2] > std::numeric_limits<size_t>::max() - guard) {
                continue;
            }
            const size_t raw_slot_size = tensor->nb[2] + guard;
            if (raw_slot_size > std::numeric_limits<size_t>::max() - 255) {
                continue;
            }
            candidates.push_back({ tensor, state, align_up_size(raw_slot_size, 256), 0 });
        }

        std::sort(candidates.begin(), candidates.end(), [](const cache_candidate & lhs, const cache_candidate & rhs) {
            return std::strcmp(lhs.tensor->name, rhs.tensor->name) < 0;
        });

        // Losing the cache costs about half the decode rate, so back the budget
        // off and retry instead of giving up on the first allocation failure.
        set_device(ctx->device);
        size_t budget = requested_budget;
        size_t used = 0;
        cudaError_t allocation_status = cudaErrorMemoryAllocation;
        for (int attempt = 0; attempt < 8 && !ctx->expert_cache; ++attempt) {
            for (cache_candidate & candidate : candidates) {
                candidate.n_slots = 0;
            }
            used = 0;
            bool made_progress = true;
            while (made_progress) {
                made_progress = false;
                for (cache_candidate & candidate : candidates) {
                    if (candidate.n_slots >= static_cast<size_t>(candidate.tensor->ne[2]) ||
                            candidate.slot_size > budget - used) {
                        continue;
                    }
                    ++candidate.n_slots;
                    used += candidate.slot_size;
                    made_progress = true;
                }
            }

            if (used == 0) {
                GGML_LOG_WARN("tiered-memory: expert cache budget %.2f MiB is too small for one slot\n",
                        budget / 1024.0 / 1024.0);
                return;
            }

            allocation_status = cudaMalloc(&ctx->expert_cache, used);
            if (allocation_status != cudaSuccess) {
                (void) cudaGetLastError();
                ctx->expert_cache = nullptr;
                budget = budget / 4 * 3;
            }
        }

        if (!ctx->expert_cache) {
            GGML_LOG_WARN("tiered-memory: could not allocate an expert cache within %.2f MiB (%s); cache disabled\n",
                    requested_budget / 1024.0 / 1024.0, cudaGetErrorString(allocation_status));
            return;
        }
        if (budget != requested_budget) {
            GGML_LOG_WARN("tiered-memory: expert cache reduced to %.2f MiB of the requested %.2f MiB to fit VRAM\n",
                    used / 1024.0 / 1024.0, requested_budget / 1024.0 / 1024.0);
        }
        ctx->expert_cache_size = used;

        struct prepared_cache {
            tensor_state * state = nullptr;
            std::unique_ptr<expert_cache_state> cache;
        };
        std::vector<prepared_cache> prepared;
        size_t offset = 0;
        size_t total_slots = 0;
        for (const cache_candidate & candidate : candidates) {
            if (candidate.n_slots == 0) {
                continue;
            }
            auto cache = std::make_unique<expert_cache_state>();
            cache->slot_size = candidate.slot_size;
            cache->heat.resize(static_cast<size_t>(candidate.tensor->ne[2]), 0);
            cache->slot_of_expert.assign(static_cast<size_t>(candidate.tensor->ne[2]), -1);
            cache->slots.resize(candidate.n_slots);
            for (expert_cache_slot & slot : cache->slots) {
                slot.device_offset = offset;
                offset += candidate.slot_size;
            }
            total_slots += candidate.n_slots;
            prepared.push_back({ candidate.state, std::move(cache) });
        }

        for (prepared_cache & item : prepared) {
            item.state->cache = std::move(item.cache);
        }

        GGML_LOG_INFO("tiered-memory: expert cache %.2f/%.2f MiB, %zu slots across %zu tensors\n",
                used / 1024.0 / 1024.0,
                requested_budget / 1024.0 / 1024.0,
                total_slots,
                prepared.size());
    } catch (const std::exception & error) {
        disable_expert_cache(ctx);
        GGML_LOG_WARN("tiered-memory: expert cache initialization failed (%s); cache disabled\n", error.what());
    }
}

// Staging must cover the CUDA row-padding guard, not just the tensor bytes.
static size_t staging_bytes(const tensor_state * state) {
    return std::max(state->sparse_size, align_up_size(state->alloc_size, 256));
}

// Dense DRAM weights are re-staged on every graph, so the scratch is kept and
// grown instead of being allocated per node.
static void * ensure_dram_staging(tiered_buffer_context * ctx, size_t size) {
    if (ctx->dram_staging_size < size) {
        if (ctx->dram_staging) {
            TIERED_CUDA_CHECK(cudaStreamSynchronize(ensure_copy_stream(ctx)));
            TIERED_CUDA_CHECK(cudaFree(ctx->dram_staging));
            ctx->dram_staging = nullptr;
            ctx->dram_staging_size = 0;
        }
        TIERED_CUDA_CHECK(cudaMalloc(&ctx->dram_staging, size));
        ctx->dram_staging_size = size;
    }
    return ctx->dram_staging;
}

static void expert_slab_range(
        const ggml_tensor * tensor,
        const tensor_state * state,
        size_t tensor_offset,
        int32_t expert,
        size_t * slab_begin,
        size_t * slab_end) {
    *slab_begin = tensor_offset + static_cast<size_t>(expert) * tensor->nb[2];
    *slab_end = tensor_offset + static_cast<size_t>(expert + 1) * tensor->nb[2];
    if (expert < tensor->ne[2] - 1) {
        *slab_end += std::min<size_t>(tensor->nb[2], 512);
    }
    *slab_end = std::min(state->size, *slab_end);
    if (*slab_begin >= *slab_end || *slab_end > state->size) {
        throw std::runtime_error("MUL_MAT_ID expert slab exceeds its base tensor");
    }
}

// Decode stages the selected experts packed at the tensor's own nb[2] stride,
// so MUL_MAT_ID can address them with ids [0, 1, ... k-1]. The kernel reads a
// little past the last expert, hence the trailing guard.
static size_t packed_staging_bytes(const ggml_tensor * tensor, size_t n_experts) {
    return align_up_size(n_experts * tensor->nb[2] + 512, 256);
}

static const int32_t * ensure_packed_ids(tiered_buffer_context * ctx, size_t count) {
    if (ctx->packed_ids_count < count) {
        if (ctx->packed_ids) {
            TIERED_CUDA_CHECK(cudaStreamSynchronize(ensure_copy_stream(ctx)));
            TIERED_CUDA_CHECK(cudaFree(ctx->packed_ids));
            ctx->packed_ids = nullptr;
            ctx->packed_ids_count = 0;
        }
        std::vector<int32_t> identity(count);
        for (size_t i = 0; i < count; ++i) {
            identity[i] = static_cast<int32_t>(i);
        }
        TIERED_CUDA_CHECK(cudaMalloc(&ctx->packed_ids, count * sizeof(int32_t)));
        TIERED_CUDA_CHECK(cudaMemcpyAsync(ctx->packed_ids, identity.data(),
                count * sizeof(int32_t), cudaMemcpyHostToDevice, ensure_copy_stream(ctx)));
        TIERED_CUDA_CHECK(cudaStreamSynchronize(ensure_copy_stream(ctx)));
        ctx->packed_ids_count = count;
    }
    return static_cast<const int32_t *>(ctx->packed_ids);
}

static bool stage_cached_experts(
        tiered_buffer_context * ctx,
        const ggml_tensor * tensor,
        tensor_state * state,
        size_t tensor_offset,
        const std::vector<int32_t> & ranked_ids,
        const std::vector<int32_t> & sorted_ids,
        size_t ids_per_row,
        size_t n_rows) {
    expert_cache_state * cache = state->cache.get();
    if (!cache || n_rows != 1 || tensor_offset != 0 || tensor->view_src || tensor->ne[3] != 1 ||
            cache->heat.size() != static_cast<size_t>(tensor->ne[2]) ||
            cache->slot_of_expert.size() != static_cast<size_t>(tensor->ne[2])) {
        return false;
    }

    for (uint16_t & value : cache->heat) {
        value = static_cast<uint16_t>((static_cast<uint32_t>(value) * 7) / 8);
    }
    for (size_t rank = 0; rank < ranked_ids.size(); ++rank) {
        const size_t row_rank = rank % ids_per_row;
        const uint64_t increment = static_cast<uint64_t>(ids_per_row - row_rank) * 8;
        uint16_t & value = cache->heat[static_cast<size_t>(ranked_ids[rank])];
        value = static_cast<uint16_t>(std::min<uint64_t>(
                std::numeric_limits<uint16_t>::max(), static_cast<uint64_t>(value) + increment));
    }

    if (++ctx->cache_epoch == 0) {
        ++ctx->cache_epoch;
    }

    struct cache_miss {
        int32_t expert = -1;
        size_t slab_begin = 0;
        size_t copy_size = 0;
    };
    std::vector<cache_miss> misses;
    std::vector<uint8_t> protected_slots(cache->slots.size(), 0);
    const cudaStream_t stream = ensure_copy_stream(ctx);
    GGML_UNUSED(sorted_ids);

    // Rank order is the packed order, so slab j lands at j*nb[2] and the ids
    // handed to MUL_MAT_ID become the identity.
    for (size_t rank = 0; rank < ranked_ids.size(); ++rank) {
        const int32_t expert = ranked_ids[rank];
        const size_t slab_begin = tensor_offset + static_cast<size_t>(expert) * tensor->nb[2];
        if (slab_begin >= state->size) {
            throw std::runtime_error("MUL_MAT_ID expert slab exceeds its base tensor");
        }
        const size_t copy_size = std::min<size_t>(tensor->nb[2], state->size - slab_begin);
        const size_t packed_offset = rank * tensor->nb[2];
        ++ctx->cache_requests;

        const int32_t hit_slot = cache->slot_of_expert[static_cast<size_t>(expert)];

        if (hit_slot >= 0) {
            expert_cache_slot & slot = cache->slots[static_cast<size_t>(hit_slot)];
            TIERED_CUDA_CHECK(cudaMemcpyAsync(
                    static_cast<char *>(ctx->expert_staging) + packed_offset,
                    static_cast<const char *>(ctx->expert_cache) + slot.device_offset,
                    copy_size,
                    cudaMemcpyDeviceToDevice,
                    stream));
            slot.last_hit_epoch = ctx->cache_epoch;
            protected_slots[static_cast<size_t>(hit_slot)] = 1;
            ++ctx->cache_hits;
            ctx->cache_hit_bytes += copy_size;
            ctx->cache_d2d_bytes += copy_size;
        } else {
            copy_host_to_device(ctx,
                    static_cast<char *>(ctx->expert_staging) + packed_offset,
                    static_cast<const char *>(state->host_ptr) + slab_begin,
                    copy_size);
            misses.push_back({ expert, packed_offset, copy_size });
            ctx->cache_miss_bytes += copy_size;
            ctx->cache_h2d_bytes += copy_size;
        }
    }

    std::sort(misses.begin(), misses.end(), [&](const cache_miss & lhs, const cache_miss & rhs) {
        const uint16_t lhs_heat = cache->heat[static_cast<size_t>(lhs.expert)];
        const uint16_t rhs_heat = cache->heat[static_cast<size_t>(rhs.expert)];
        return lhs_heat != rhs_heat ? lhs_heat > rhs_heat : lhs.expert < rhs.expert;
    });

    // Staging is complete here, so the caller can enqueue compute. The admission
    // copies below only read staging, so they stay async and overlap with it.
    TIERED_CUDA_CHECK(cudaStreamSynchronize(stream));

    for (const cache_miss & miss : misses) {
        size_t victim = cache->slots.size();
        for (size_t slot_index = 0; slot_index < cache->slots.size(); ++slot_index) {
            const expert_cache_slot & slot = cache->slots[slot_index];
            if (protected_slots[slot_index]) {
                continue;
            }
            if (slot.expert < 0) {
                victim = slot_index;
                break;
            }
            if (victim == cache->slots.size()) {
                victim = slot_index;
                continue;
            }
            const expert_cache_slot & current = cache->slots[victim];
            const uint16_t slot_heat = cache->heat[static_cast<size_t>(slot.expert)];
            const uint16_t current_heat = cache->heat[static_cast<size_t>(current.expert)];
            if (slot_heat < current_heat ||
                    (slot_heat == current_heat && slot.last_hit_epoch < current.last_hit_epoch)) {
                victim = slot_index;
            }
        }

        if (victim == cache->slots.size()) {
            continue;
        }
        expert_cache_slot & slot = cache->slots[victim];
        if (slot.expert >= 0 &&
                cache->heat[static_cast<size_t>(miss.expert)] <= cache->heat[static_cast<size_t>(slot.expert)]) {
            continue;
        }

        TIERED_CUDA_CHECK(cudaMemcpyAsync(
                static_cast<char *>(ctx->expert_cache) + slot.device_offset,
                static_cast<const char *>(ctx->expert_staging) + miss.slab_begin,
                miss.copy_size,
                cudaMemcpyDeviceToDevice,
                stream));
        if (slot.expert >= 0) {
            cache->slot_of_expert[static_cast<size_t>(slot.expert)] = -1;
            ++ctx->cache_evictions;
        }
        slot.expert = miss.expert;
        cache->slot_of_expert[static_cast<size_t>(miss.expert)] = static_cast<int32_t>(victim);
        slot.last_hit_epoch = ctx->cache_epoch;
        protected_slots[victim] = 1;
        ++ctx->cache_admissions;
        ctx->cache_d2d_bytes += miss.copy_size;
    }

    return true;
}

static void * stage_tiered_experts(
        tiered_buffer_context * ctx,
        const ggml_tensor * tensor,
        tensor_state * state,
        const ggml_tensor * ids,
        int64_t & out_packed_experts,
        const int32_t * & out_packed_ids) {
    out_packed_experts = 0;
    out_packed_ids = nullptr;
    if (!ids || ids->type != GGML_TYPE_I32) {
        throw std::runtime_error("MUL_MAT_ID expert ids must be I32");
    }

    if (ids->ne[0] <= 0 || ids->ne[1] <= 0 || ids->ne[2] != 1 || ids->ne[3] != 1 || ids->nb[0] != sizeof(int32_t)) {
        throw std::runtime_error("MUL_MAT_ID expert ids have an unsupported layout");
    }

    const size_t ids_per_row = static_cast<size_t>(ids->ne[0]);
    const size_t n_rows = static_cast<size_t>(ids->ne[1]);
    if (ids_per_row > std::numeric_limits<size_t>::max() / sizeof(int32_t) ||
            n_rows > std::numeric_limits<size_t>::max() / ids_per_row) {
        throw std::runtime_error("MUL_MAT_ID expert ids size overflow");
    }
    const size_t row_size = ids_per_row * sizeof(int32_t);
    if (ids->nb[1] < row_size) {
        throw std::runtime_error("MUL_MAT_ID expert ids have an invalid row stride");
    }

    // The gate, up and down projections of one layer share a router output, so
    // reading it back once per graph saves two blocking device-to-host copies.
    const uint64_t run = tiered_graph_run.load(std::memory_order_relaxed);
    if (ctx->ids_cache_tensor != ids || ctx->ids_cache_run != run) {
        ctx->ids_cache_values.resize(ids_per_row * n_rows);
        ggml_backend_tensor_get_2d(ids, ctx->ids_cache_values.data(), 0, row_size, n_rows, ids->nb[1], row_size);
        ctx->ids_cache_tensor = ids;
        ctx->ids_cache_run = run;
    }
    const std::vector<int32_t> & ranked_ids = ctx->ids_cache_values;

    size_t tensor_offset = 0;
    (void) tiered_view_base(tensor, &tensor_offset);
    if (tensor_offset > state->size || ggml_nbytes(tensor) > state->size - tensor_offset) {
        throw std::runtime_error("tiered expert tensor view exceeds its base tensor");
    }

    for (const int32_t expert : ranked_ids) {
        if (expert < 0 || expert >= tensor->ne[2]) {
            throw std::runtime_error("MUL_MAT_ID produced an out-of-range expert id");
        }
    }

    std::vector<int32_t> host_ids = ranked_ids;
    std::sort(host_ids.begin(), host_ids.end());
    host_ids.erase(std::unique(host_ids.begin(), host_ids.end()), host_ids.end());

    set_device(ctx->device);

    // Decode selects a fixed number of experts, so it only needs room for those
    // rather than the whole stack. Prompt batches can touch every expert and
    // keep the original layout.
    const bool packed = (n_rows == 1) && tensor_offset == 0 && !tensor->view_src && tensor->ne[3] == 1;

    size_t staging_size = packed ? packed_staging_bytes(tensor, ids_per_row) : staging_bytes(state);
    if (!ctx->expert_cache_initialized && ctx->plan->options.ssd_cache_bytes > 0) {
        for (const auto & entry : ctx->tensors) {
            const ggml_tensor * candidate_tensor = entry.first;
            const tensor_state * candidate_state = entry.second.get();
            if (candidate_state->tier == GGML_CUDA_TIERED_MEMORY_VRAM || !is_ssd_eligible(candidate_tensor)) {
                continue;
            }
            staging_size = std::max(staging_size, packed
                    ? packed_staging_bytes(candidate_tensor, ids_per_row)
                    : staging_bytes(candidate_state));
        }
    }
    // Give the buffer back when a prompt-sized allocation is no longer needed.
    const bool oversized = ctx->expert_staging_size > 4 * staging_size;
    if (ctx->expert_staging_size < staging_size || oversized) {
        if (ctx->expert_staging) {
            TIERED_CUDA_CHECK(cudaStreamSynchronize(ensure_copy_stream(ctx)));
            TIERED_CUDA_CHECK(cudaFree(ctx->expert_staging));
            ctx->expert_staging = nullptr;
            ctx->expert_staging_size = 0;
        }
        cudaError_t allocation_status = cudaMalloc(&ctx->expert_staging, staging_size);
        if (allocation_status != cudaSuccess && ctx->expert_cache) {
            GGML_LOG_WARN("tiered-memory: disabling expert cache to grow staging scratch to %.2f MiB\n",
                    staging_size / 1024.0 / 1024.0);
            (void) cudaGetLastError();
            disable_expert_cache(ctx);
            allocation_status = cudaMalloc(&ctx->expert_staging, staging_size);
        }
        TIERED_CUDA_CHECK(allocation_status);
        // slab copies never write the guard tail, so define it once here
        TIERED_CUDA_CHECK(cudaMemsetAsync(ctx->expert_staging, 0, staging_size, ensure_copy_stream(ctx)));
        ctx->expert_staging_size = staging_size;
    }

    initialize_expert_cache(ctx);

    if (!stage_cached_experts(ctx, tensor, state, tensor_offset, ranked_ids, host_ids, ids_per_row, n_rows)) {
        if (packed) {
            for (size_t rank = 0; rank < ranked_ids.size(); ++rank) {
                const size_t slab_begin = static_cast<size_t>(ranked_ids[rank]) * tensor->nb[2];
                if (slab_begin >= state->size) {
                    throw std::runtime_error("MUL_MAT_ID expert slab exceeds its base tensor");
                }
                const size_t copy_size = std::min<size_t>(tensor->nb[2], state->size - slab_begin);
                copy_host_to_device(ctx,
                        static_cast<char *>(ctx->expert_staging) + rank * tensor->nb[2],
                        static_cast<const char *>(state->host_ptr) + slab_begin,
                        copy_size);
                ctx->cache_h2d_bytes += copy_size;
            }
        } else {
            for (int64_t i3 = 0; i3 < tensor->ne[3]; ++i3) {
                size_t first = 0;
                while (first < host_ids.size()) {
                    size_t last = first;
                    while (last + 1 < host_ids.size() && host_ids[last + 1] == host_ids[last] + 1) {
                        ++last;
                    }

                    const size_t slab_begin = tensor_offset +
                            static_cast<size_t>(i3) * tensor->nb[3] +
                            static_cast<size_t>(host_ids[first]) * tensor->nb[2];
                    size_t slab_end = tensor_offset +
                            static_cast<size_t>(i3) * tensor->nb[3] +
                            static_cast<size_t>(host_ids[last] + 1) * tensor->nb[2];
                    if (host_ids[last] < tensor->ne[2] - 1) {
                        slab_end += std::min<size_t>(tensor->nb[2], 512);
                    }
                    slab_end = std::min(state->size, slab_end);
                    if (slab_begin >= slab_end || slab_end > state->size) {
                        throw std::runtime_error("MUL_MAT_ID expert slab exceeds its base tensor");
                    }

                    const size_t copy_size = slab_end - slab_begin;
                    copy_host_to_device(ctx,
                            static_cast<char *>(ctx->expert_staging) + slab_begin,
                            static_cast<const char *>(state->host_ptr) + slab_begin,
                            copy_size);
                    ctx->cache_h2d_bytes += copy_size;
                    first = last + 1;
                }
            }
        }
        TIERED_CUDA_CHECK(cudaStreamSynchronize(ensure_copy_stream(ctx)));
    }

    if (packed) {
        out_packed_experts = static_cast<int64_t>(ids_per_row);
        out_packed_ids = ensure_packed_ids(ctx, ids_per_row);
        return ctx->expert_staging;
    }
    return static_cast<char *>(ctx->expert_staging) + tensor_offset;
}

static const char * tiered_buft_name(ggml_backend_buffer_type_t buft) {
    auto * ctx = static_cast<tiered_buffer_type_context *>(buft->context);
    return ctx->name.c_str();
}

static void tiered_buffer_free(ggml_backend_buffer_t buffer) {
    auto * ctx = static_cast<tiered_buffer_context *>(buffer->context);
    set_device(ctx->device);

    for (auto & entry : ctx->tensors) {
        tensor_state * state = entry.second.get();
        if (state->tier == GGML_CUDA_TIERED_MEMORY_VRAM && state->device_ptr) {
            (void) cudaFree(state->device_ptr);
            state->device_ptr = nullptr;
        }
        if (state->host_ptr_owned && state->host_ptr) {
            (void) cudaFreeHost(state->host_ptr);
            state->host_ptr = nullptr;
            state->host_ptr_owned = false;
        }
        if (state->tier == GGML_CUDA_TIERED_MEMORY_SSD) {
            free_sparse_tensor(state);
        }
    }

    tiered_host_tensors.fetch_sub(ctx->host_tensors, std::memory_order_relaxed);
    ctx->host_tensors = 0;

    if (ctx->plan->options.ssd_cache_bytes > 0) {
        const uint64_t cacheable_bytes = ctx->cache_hit_bytes + ctx->cache_miss_bytes;
        const double request_hit_rate = ctx->cache_requests > 0 ?
                100.0 * static_cast<double>(ctx->cache_hits) / static_cast<double>(ctx->cache_requests) : 0.0;
        const double byte_hit_rate = cacheable_bytes > 0 ?
                100.0 * static_cast<double>(ctx->cache_hit_bytes) / static_cast<double>(cacheable_bytes) : 0.0;
        GGML_LOG_INFO(
                "tiered-memory: expert cache hits %llu/%llu (%.2f%% requests, %.2f%% bytes), "
                "H2D %.2f MiB, D2D %.2f MiB, admissions %llu, evictions %llu\n",
                static_cast<unsigned long long>(ctx->cache_hits),
                static_cast<unsigned long long>(ctx->cache_requests),
                request_hit_rate,
                byte_hit_rate,
                ctx->cache_h2d_bytes / 1024.0 / 1024.0,
                ctx->cache_d2d_bytes / 1024.0 / 1024.0,
                static_cast<unsigned long long>(ctx->cache_admissions),
                static_cast<unsigned long long>(ctx->cache_evictions));
    }
    disable_expert_cache(ctx);

    if (ctx->expert_staging) {
        (void) cudaFree(ctx->expert_staging);
        ctx->expert_staging = nullptr;
        ctx->expert_staging_size = 0;
    }

    if (ctx->dram_staging) {
        (void) cudaFree(ctx->dram_staging);
        ctx->dram_staging = nullptr;
        ctx->dram_staging_size = 0;
    }

    if (ctx->copy_stream) {
        (void) cudaStreamDestroy(ctx->copy_stream);
        ctx->copy_stream = nullptr;
    }

    for (auto it = ctx->registrations.rbegin(); it != ctx->registrations.rend(); ++it) {
        if (it->owned) {
            (void) cudaHostUnregister(reinterpret_cast<void *>(it->begin));
        }
    }

    if (ctx->address_space) {
        (void) munmap(ctx->address_space, ctx->address_space_size);
    }
    delete ctx;
}

static void * tiered_buffer_base(ggml_backend_buffer_t buffer) {
    auto * ctx = static_cast<tiered_buffer_context *>(buffer->context);
    return ctx->address_space;
}

static ggml_status tiered_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    auto * ctx = static_cast<tiered_buffer_context *>(buffer->context);
    if (!tensor->view_src) {
        auto & state_ptr = ctx->tensors[tensor];
        if (!state_ptr) {
            state_ptr = std::make_unique<tensor_state>();
        }
        tensor->extra = state_ptr.get();
    }
    return GGML_STATUS_SUCCESS;
}

static void tiered_buffer_set_tensor(
        ggml_backend_buffer_t buffer,
        ggml_tensor * tensor,
        const void * data,
        size_t offset,
        size_t size) {
    auto * ctx = static_cast<tiered_buffer_context *>(buffer->context);
    set_device(ctx->device);

    if (offset != 0 || size != ggml_nbytes(tensor)) {
        throw std::runtime_error(std::string("tiered weights require full-tensor initialization: ") + tensor->name);
    }

    if (tensor->view_src) {
        if (!tensor->view_src->data) {
            throw std::runtime_error(std::string("tiered view initialized before its source tensor: ") + tensor->name);
        }
        tensor->data = static_cast<char *>(tensor->view_src->data) + tensor->view_offs;
        size_t view_offset = 0;
        tiered_view_base(tensor, &view_offset);
        tensor_state * state = state_for(tensor);
        if (!state || !state->host_ptr || !tensor->data) {
            throw std::runtime_error(std::string("tiered view initialized before its base tensor: ") + tensor->name);
        }
        if (state->tier == GGML_CUDA_TIERED_MEMORY_VRAM) {
            copy_host_to_device(ctx, tensor->data, data, size);
            TIERED_CUDA_CHECK(cudaStreamSynchronize(ensure_copy_stream(ctx)));
        } else {
            const void * expected = static_cast<const char *>(state->host_ptr) + view_offset;
            if (data != expected && std::memcmp(data, expected, size) != 0) {
                throw std::runtime_error(std::string("host-tier view data differs from its base tensor: ") + tensor->name);
            }
        }
        return;
    }

    auto & state_ptr = ctx->tensors[tensor];
    if (!state_ptr) {
        state_ptr = std::make_unique<tensor_state>();
    }
    tensor_state * state = state_ptr.get();
    if (state->host_ptr || state->device_ptr || state->sparse_address) {
        throw std::runtime_error(std::string("tiered tensor initialized more than once: ") + tensor->name);
    }

    state->host_ptr = const_cast<void *>(data);
    state->size = size;
    state->alloc_size = ggml_backend_buft_get_alloc_size(
            ggml_backend_cuda_buffer_type(ctx->device), tensor);

    const auto found = ctx->plan->tensors.find(tensor->name);
    if (found == ctx->plan->tensors.end()) {
        if (ctx->plan->options.strict) {
            throw std::runtime_error(std::string("tiered plan is missing tensor: ") + tensor->name);
        }
        state->tier = GGML_CUDA_TIERED_MEMORY_VRAM;
    } else {
        state->tier = found->second;
    }

    // CUDA pads quantized rows to a guard boundary. A directly mapped DRAM
    // tensor has no room for the guard, but a streamed one lands in staging
    // scratch that we size ourselves, so stream it instead of rejecting it.
    if (state->tier == GGML_CUDA_TIERED_MEMORY_DRAM && state->alloc_size != state->size &&
            is_ssd_eligible(tensor)) {
        state->tier = GGML_CUDA_TIERED_MEMORY_SSD;
    }

    if (state->tier == GGML_CUDA_TIERED_MEMORY_DRAM && state->alloc_size != state->size) {
        if (ctx->plan->options.strict) {
            throw std::runtime_error(std::string("host tier does not support CUDA padding for tensor: ") + tensor->name);
        }
        state->tier = GGML_CUDA_TIERED_MEMORY_VRAM;
    }

    switch (state->tier) {
        case GGML_CUDA_TIERED_MEMORY_VRAM: {
            TIERED_CUDA_CHECK(cudaMalloc(&state->device_ptr, state->alloc_size));
            copy_host_to_device(ctx, state->device_ptr, data, state->size);
            TIERED_CUDA_CHECK(cudaStreamSynchronize(ensure_copy_stream(ctx)));
            if (state->alloc_size > state->size) {
                TIERED_CUDA_CHECK(cudaMemset(
                        static_cast<char *>(state->device_ptr) + state->size,
                        0,
                        state->alloc_size - state->size));
            }
            tensor->data = state->device_ptr;
        } break;

        case GGML_CUDA_TIERED_MEMORY_DRAM: {
            cudaDeviceProp properties = {};
            TIERED_CUDA_CHECK(cudaGetDeviceProperties(&properties, ctx->device));
            if (!properties.canMapHostMemory) {
                throw std::runtime_error("CUDA device cannot map host memory");
            }

            // GGUF weights normally come from a read-only file mmap. Some
            // pre-Ampere GPUs/drivers reject cudaHostRegisterMapped for that
            // range. Fall back to an owned mapped pinned copy.
            try {
                ensure_registered(ctx, state->host_ptr, state->size);
            } catch (const std::exception & error) {
                (void) cudaGetLastError();
                GGML_LOG_WARN(
                        "tiered-memory: direct DRAM registration failed for %s (%s); "
                        "using a mapped pinned copy\n",
                        tensor->name, error.what());

                void * pinned = nullptr;
                TIERED_CUDA_CHECK(cudaHostAlloc(
                        &pinned, state->size,
                        cudaHostAllocPortable | cudaHostAllocMapped));
                std::memcpy(pinned, data, state->size);
                state->host_ptr = pinned;
                state->host_ptr_owned = true;
            }

            TIERED_CUDA_CHECK(cudaHostGetDevicePointer(
                    &state->device_ptr, state->host_ptr, 0));
            tensor->data = state->device_ptr;
        } break;

        case GGML_CUDA_TIERED_MEMORY_SSD:
            init_sparse_tensor(ctx, tensor, state);
            try {
                ensure_registered(ctx, state->host_ptr, state->size);
            } catch (const std::exception & error) {
                (void) cudaGetLastError();
                GGML_LOG_WARN("tiered-memory: could not page-lock %s (%s); using pageable transfers\n",
                        tensor->name, error.what());
            }
            break;
    }

    if (state->tier != GGML_CUDA_TIERED_MEMORY_VRAM) {
        ++ctx->host_tensors;
        tiered_host_tensors.fetch_add(1, std::memory_order_relaxed);
    }

    GGML_LOG_INFO("tiered-memory: %-4s %8.2f MiB %s\n",
            state->tier == GGML_CUDA_TIERED_MEMORY_VRAM ? "VRAM" :
            state->tier == GGML_CUDA_TIERED_MEMORY_DRAM ? "DRAM" : "SSD",
            state->size / 1024.0 / 1024.0,
            tensor->name);
}

static void tiered_buffer_get_tensor(
        ggml_backend_buffer_t buffer,
        const ggml_tensor * tensor,
        void * data,
        size_t offset,
        size_t size) {
    auto * ctx = static_cast<tiered_buffer_context *>(buffer->context);
    tensor_state * state = state_for(tensor);
    if (!state) {
        throw std::runtime_error("tiered tensor state not found");
    }
    if (offset + size > ggml_nbytes(tensor)) {
        throw std::runtime_error("tiered tensor read is out of bounds");
    }
    if (state->tier == GGML_CUDA_TIERED_MEMORY_VRAM) {
        set_device(ctx->device);
        TIERED_CUDA_CHECK(cudaMemcpy(
                data, static_cast<const char *>(tensor->data) + offset,
                size, cudaMemcpyDeviceToHost));
    } else {
        size_t view_offset = 0;
        tiered_view_base(tensor, &view_offset);
        std::memcpy(data, static_cast<const char *>(state->host_ptr) + view_offset + offset, size);
    }
}

static void tiered_buffer_memset_tensor(
        ggml_backend_buffer_t buffer,
        ggml_tensor * tensor,
        uint8_t value,
        size_t offset,
        size_t size) {
    auto * ctx = static_cast<tiered_buffer_context *>(buffer->context);
    tensor_state * state = state_for(tensor);
    if (!state) {
        return;
    }
    if (state->tier == GGML_CUDA_TIERED_MEMORY_VRAM) {
        set_device(ctx->device);
        TIERED_CUDA_CHECK(cudaMemset(
                static_cast<char *>(tensor->data) + offset, value, size));
    } else if (state->host_ptr) {
        throw std::runtime_error("attempted to mutate a read-only tiered weight tensor");
    }
}

static void tiered_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(value);
}

static const ggml_backend_buffer_i tiered_buffer_interface = {
    /* .free_buffer     = */ tiered_buffer_free,
    /* .get_base        = */ tiered_buffer_base,
    /* .init_tensor     = */ tiered_buffer_init_tensor,
    /* .memset_tensor   = */ tiered_buffer_memset_tensor,
    /* .set_tensor      = */ tiered_buffer_set_tensor,
    /* .get_tensor      = */ tiered_buffer_get_tensor,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ tiered_buffer_clear,
    /* .reset           = */ nullptr,
};

static ggml_backend_buffer_t tiered_buft_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    auto * buft_ctx = static_cast<tiered_buffer_type_context *>(buft->context);
    auto plan_it = tls_plans.find(buft_ctx->device);
    if (plan_it == tls_plans.end() || !plan_it->second) {
        GGML_LOG_ERROR("tiered-memory: no active plan for device %d\n", buft_ctx->device);
        return nullptr;
    }

    void * address_space = mmap(nullptr, size, PROT_NONE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (address_space == MAP_FAILED) {
        GGML_LOG_ERROR("tiered-memory: failed to reserve %zu bytes of virtual address space\n", size);
        return nullptr;
    }

    auto * ctx = new tiered_buffer_context;
    ctx->device = buft_ctx->device;
    ctx->address_space = address_space;
    ctx->address_space_size = size;
    ctx->plan = plan_it->second;

    return ggml_backend_buffer_init(buft, tiered_buffer_interface, ctx, size);
}

static size_t tiered_buft_alignment(ggml_backend_buffer_type_t buft) {
    auto * ctx = static_cast<tiered_buffer_type_context *>(buft->context);
    return ggml_backend_buft_get_alignment(ggml_backend_cuda_buffer_type(ctx->device));
}

static size_t tiered_buft_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    auto * ctx = static_cast<tiered_buffer_type_context *>(buft->context);
    return ggml_backend_buft_get_alloc_size(
            ggml_backend_cuda_buffer_type(ctx->device), tensor);
}

static const ggml_backend_buffer_type_i tiered_buft_interface = {
    /* .get_name       = */ tiered_buft_name,
    /* .alloc_buffer   = */ tiered_buft_alloc,
    /* .get_alignment  = */ tiered_buft_alignment,
    /* .get_max_size   = */ nullptr,
    /* .get_alloc_size = */ tiered_buft_alloc_size,
    /* .is_host        = */ nullptr,
};

static const char * tiered_backend_name(ggml_backend_t backend) {
    auto * ctx = static_cast<tiered_backend_context *>(backend->context);
    return ctx->name.c_str();
}

static void tiered_backend_free(ggml_backend_t backend) {
    auto * ctx = static_cast<tiered_backend_context *>(backend->context);
    ggml_backend_free(ctx->inner);
    delete ctx;
    delete backend;
}

static void tiered_backend_set_tensor_async(
        ggml_backend_t backend, ggml_tensor * tensor,
        const void * data, size_t offset, size_t size) {
    auto * ctx = static_cast<tiered_backend_context *>(backend->context);
    ggml_backend_tensor_set_async(ctx->inner, tensor, data, offset, size);
}

static void tiered_backend_get_tensor_async(
        ggml_backend_t backend, const ggml_tensor * tensor,
        void * data, size_t offset, size_t size) {
    auto * ctx = static_cast<tiered_backend_context *>(backend->context);
    ggml_backend_tensor_get_async(ctx->inner, tensor, data, offset, size);
}

static void tiered_backend_set_tensor_2d_async(
        ggml_backend_t backend, ggml_tensor * tensor,
        const void * data, size_t offset, size_t size,
        size_t n_copies, size_t stride_tensor, size_t stride_data) {
    auto * ctx = static_cast<tiered_backend_context *>(backend->context);
    ggml_backend_tensor_set_2d_async(ctx->inner, tensor, data, offset, size,
            n_copies, stride_tensor, stride_data);
}

static void tiered_backend_get_tensor_2d_async(
        ggml_backend_t backend, const ggml_tensor * tensor,
        void * data, size_t offset, size_t size,
        size_t n_copies, size_t stride_tensor, size_t stride_data) {
    auto * ctx = static_cast<tiered_backend_context *>(backend->context);
    ggml_backend_tensor_get_2d_async(ctx->inner, tensor, data, offset, size,
            n_copies, stride_tensor, stride_data);
}

static void tiered_backend_synchronize(ggml_backend_t backend) {
    auto * ctx = static_cast<tiered_backend_context *>(backend->context);
    ggml_backend_synchronize(ctx->inner);
}

static ggml_status compute_view(tiered_backend_context * ctx, ggml_cgraph * graph, int begin, int end) {
    if (begin >= end) {
        return GGML_STATUS_SUCCESS;
    }
    ggml_cgraph view = ggml_graph_view(graph, begin, end);
    // Propagate the caller's graph uid instead of minting a fresh one: the
    // inner CUDA backend keys its captured-graph cache off the view's first
    // node pointer (stable per segment across tokens), and only replays
    // without recapturing when the uid also matches the last capture. A
    // fresh uid every call defeated that cache and forced a recapture every
    // token even when nothing about this segment changed.
    view.uid = graph->uid;
    return ggml_backend_graph_compute(ctx->inner, &view);
}

static ggml_status tiered_backend_graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    auto * ctx = static_cast<tiered_backend_context *>(backend->context);

    // Nothing to stage while every weight is VRAM resident, so run the whole
    // graph in one piece and skip the per-node source scan below.
    if (tiered_host_tensors.load(std::memory_order_relaxed) == 0) {
        return ggml_backend_graph_compute(ctx->inner, graph);
    }

    tiered_graph_run.fetch_add(1, std::memory_order_relaxed);

    std::unique_lock<std::mutex> compute_lock;
    for (int i = 0; i < graph->n_nodes && !compute_lock.owns_lock(); ++i) {
        for (int src_index = 0; src_index < GGML_MAX_SRC; ++src_index) {
            tiered_buffer_context * weight_ctx = buffer_context(graph->nodes[i]->src[src_index]);
            if (weight_ctx) {
                compute_lock = std::unique_lock<std::mutex>(weight_ctx->compute_mutex);
                break;
            }
        }
    }
    int segment_begin = 0;

    for (int i = 0; i < graph->n_nodes; ++i) {
        ggml_tensor * node = graph->nodes[i];
        tensor_state * streamed_state = nullptr;
        tiered_buffer_context * weight_ctx = nullptr;

        tensor_state * dram_state = nullptr;
        tiered_buffer_context * dram_ctx = nullptr;
        size_t dram_view_offset = 0;

        for (int src_index = 0; src_index < GGML_MAX_SRC; ++src_index) {
            ggml_tensor * src = node->src[src_index];
            tensor_state * state = state_for(src);
            if (!state) {
                continue;
            }

            if (state->tier == GGML_CUDA_TIERED_MEMORY_SSD) {
                if (node->op != GGML_OP_MUL_MAT_ID || src_index != 0) {
                    GGML_LOG_ERROR("tiered-memory: SSD tensor %s is used by unsupported op %s\n",
                            src->name, ggml_op_name(node->op));
                    return GGML_STATUS_FAILED;
                }
                streamed_state = state;
                weight_ctx = buffer_context(src);
                continue;
            }

            if (state->tier == GGML_CUDA_TIERED_MEMORY_DRAM &&
                    node->op == GGML_OP_MUL_MAT_ID && src_index == 0) {
                streamed_state = state;
                weight_ctx = buffer_context(src);
                continue;
            }

            // Stage DRAM MUL_MAT weights because cuBLAS cannot use mapped host aliases reliably on Turing.
            if (state->tier == GGML_CUDA_TIERED_MEMORY_DRAM &&
                    node->op == GGML_OP_MUL_MAT && src_index == 0) {
                size_t view_offset = 0;
                (void) tiered_view_base(src, &view_offset);
                dram_state = state;
                dram_ctx = buffer_context(src);
                dram_view_offset = view_offset;
            }
        }

        if (!streamed_state && !dram_state) {
            continue;
        }

        ggml_status status = compute_view(ctx, graph, segment_begin, i);
        if (status != GGML_STATUS_SUCCESS) {
            return status;
        }
        ggml_backend_synchronize(ctx->inner);

        if (dram_state) {
            void * staged = nullptr;
            ggml_tensor * original_weight = node->src[0];
            ggml_tensor staged_weight = {};

            try {
                if (dram_view_offset > dram_state->size ||
                        ggml_nbytes(original_weight) > dram_state->size - dram_view_offset) {
                    throw std::runtime_error("tiered DRAM weight view exceeds its base tensor");
                }
                set_device(ctx->device);
                staged = ensure_dram_staging(dram_ctx, dram_state->alloc_size);
                copy_host_to_device(dram_ctx,
                        staged,
                        dram_state->host_ptr,
                        dram_state->size);
                if (dram_state->alloc_size > dram_state->size) {
                    TIERED_CUDA_CHECK(cudaMemsetAsync(
                            static_cast<char *>(staged) + dram_state->size,
                            0,
                            dram_state->alloc_size - dram_state->size,
                            ensure_copy_stream(dram_ctx)));
                }
                TIERED_CUDA_CHECK(cudaStreamSynchronize(ensure_copy_stream(dram_ctx)));

                staged_weight = *original_weight;
                staged_weight.data = static_cast<char *>(staged) + dram_view_offset;
                staged_weight.view_src = nullptr;
                staged_weight.view_offs = 0;
                node->src[0] = &staged_weight;

                status = compute_view(ctx, graph, i, i + 1);
                ggml_backend_synchronize(ctx->inner);

                node->src[0] = original_weight;
            } catch (const std::exception & error) {
                node->src[0] = original_weight;
                GGML_LOG_ERROR("tiered-memory: failed to stage DRAM weight %s: %s\n",
                        original_weight ? original_weight->name : "unknown", error.what());
                return GGML_STATUS_FAILED;
            }
        } else {
            ggml_tensor * weight = node->src[0];
            ggml_tensor * original_ids = node->src[2];
            ggml_tensor staged_weight = {};
            ggml_tensor staged_ids = {};
            try {
                int64_t packed_experts = 0;
                const int32_t * packed_ids = nullptr;
                void * staged_data = stage_tiered_experts(
                        weight_ctx, weight, streamed_state, original_ids, packed_experts, packed_ids);
                staged_weight = *weight;
                staged_weight.data = staged_data;
                staged_weight.view_src = nullptr;
                staged_weight.view_offs = 0;
                node->src[0] = &staged_weight;

                // Experts were packed in router order, so the ids that select
                // them become the identity.
                if (packed_experts > 0) {
                    staged_weight.ne[2] = packed_experts;
                    staged_ids = *original_ids;
                    staged_ids.data = const_cast<int32_t *>(packed_ids);
                    staged_ids.view_src = nullptr;
                    staged_ids.view_offs = 0;
                    node->src[2] = &staged_ids;
                }

                status = compute_view(ctx, graph, i, i + 1);
                ggml_backend_synchronize(ctx->inner);
                node->src[0] = weight;
                node->src[2] = original_ids;
            } catch (const std::exception & error) {
                node->src[0] = weight;
                node->src[2] = original_ids;
                GGML_LOG_ERROR("tiered-memory: failed to stream %s: %s\n",
                        weight->name, error.what());
                return GGML_STATUS_FAILED;
            }
        }

        if (status != GGML_STATUS_SUCCESS) {
            return status;
        }
        segment_begin = i + 1;
    }

    return compute_view(ctx, graph, segment_begin, graph->n_nodes);
}

static void tiered_backend_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    auto * ctx = static_cast<tiered_backend_context *>(backend->context);
    ggml_backend_event_record(event, ctx->inner);
}

static void tiered_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    auto * ctx = static_cast<tiered_backend_context *>(backend->context);
    ggml_backend_event_wait(ctx->inner, event);
}

static const ggml_backend_i tiered_backend_interface = {
    /* .get_name            = */ tiered_backend_name,
    /* .free                = */ tiered_backend_free,
    /* .set_tensor_async    = */ tiered_backend_set_tensor_async,
    /* .get_tensor_async    = */ tiered_backend_get_tensor_async,
    /* .set_tensor_2d_async = */ tiered_backend_set_tensor_2d_async,
    /* .get_tensor_2d_async = */ tiered_backend_get_tensor_2d_async,
    /* .cpy_tensor_async    = */ nullptr,
    /* .synchronize         = */ tiered_backend_synchronize,
    /* .graph_plan_create   = */ nullptr,
    /* .graph_plan_free     = */ nullptr,
    /* .graph_plan_update   = */ nullptr,
    /* .graph_plan_compute  = */ nullptr,
    /* .graph_compute       = */ tiered_backend_graph_compute,
    /* .event_record        = */ tiered_backend_event_record,
    /* .event_wait          = */ tiered_backend_event_wait,
    /* .graph_optimize      = */ nullptr,
};

static ggml_guid_t tiered_backend_guid() {
    static ggml_guid guid = { 0x6c, 0x6c, 0x61, 0x6d, 0x61, 0x79, 0x2d, 0x74,
                              0x69, 0x65, 0x72, 0x65, 0x64, 0x2d, 0x30, 0x31 };
    return &guid;
}

static const char * tiered_device_name(ggml_backend_dev_t dev) {
    return static_cast<tiered_device_context *>(dev->context)->name.c_str();
}

static const char * tiered_device_description(ggml_backend_dev_t dev) {
    return static_cast<tiered_device_context *>(dev->context)->description.c_str();
}

static void tiered_device_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    auto * ctx = static_cast<tiered_device_context *>(dev->context);
    ggml_backend_dev_memory(ctx->inner_device, free, total);
}

static enum ggml_backend_dev_type tiered_device_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void tiered_device_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    auto * ctx = static_cast<tiered_device_context *>(dev->context);
    ggml_backend_dev_get_props(ctx->inner_device, props);
    props->name = ctx->name.c_str();
    props->description = ctx->description.c_str();
    props->type = GGML_BACKEND_DEVICE_TYPE_ACCEL;
    props->device_id = ctx->device_id.c_str();
    props->caps.buffer_from_host_ptr = false;
}

static ggml_backend_t tiered_device_init(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    auto * dev_ctx = static_cast<tiered_device_context *>(dev->context);
    ggml_backend_t inner = ggml_backend_dev_init(dev_ctx->inner_device, nullptr);
    if (!inner) {
        return nullptr;
    }
    auto * ctx = new tiered_backend_context;
    ctx->device = dev_ctx->device;
    ctx->inner = inner;
    ctx->name = dev_ctx->name;

    return new ggml_backend {
        /* .guid    = */ tiered_backend_guid(),
        /* .iface   = */ tiered_backend_interface,
        /* .device  = */ dev,
        /* .context = */ ctx,
    };
}

static ggml_backend_buffer_type_t tiered_device_buffer_type(ggml_backend_dev_t dev) {
    auto * ctx = static_cast<tiered_device_context *>(dev->context);
    return ggml_backend_cuda_buffer_type(ctx->device);
}

static ggml_backend_buffer_type_t tiered_device_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_cuda_host_buffer_type();
}

static bool tiered_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    auto * ctx = static_cast<tiered_device_context *>(dev->context);
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        tensor_state * state = state_for(op->src[i]);
        if (state && state->tier == GGML_CUDA_TIERED_MEMORY_SSD &&
                (op->op != GGML_OP_MUL_MAT_ID || i != 0)) {
            return false;
        }
    }
    return ggml_backend_dev_supports_op(ctx->inner_device, op);
}

static bool tiered_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    auto * ctx = static_cast<tiered_device_context *>(dev->context);
    return buft == ctx->tiered_buft ||
           ggml_backend_dev_supports_buft(ctx->inner_device, buft);
}

static bool tiered_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    auto * ctx = static_cast<tiered_device_context *>(dev->context);
    return ggml_backend_dev_offload_op(ctx->inner_device, op);
}

static ggml_backend_event_t tiered_device_event_new(ggml_backend_dev_t dev) {
    auto * ctx = static_cast<tiered_device_context *>(dev->context);
    return ggml_backend_event_new(ctx->inner_device);
}

static void tiered_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    ggml_backend_event_free(event);
}

static void tiered_device_event_sync(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    ggml_backend_event_synchronize(event);
}

static const ggml_backend_device_i tiered_device_interface = {
    /* .get_name             = */ tiered_device_name,
    /* .get_description      = */ tiered_device_description,
    /* .get_memory           = */ tiered_device_memory,
    /* .get_type             = */ tiered_device_type,
    /* .get_props            = */ tiered_device_props,
    /* .init_backend         = */ tiered_device_init,
    /* .get_buffer_type      = */ tiered_device_buffer_type,
    /* .get_host_buffer_type = */ tiered_device_host_buffer_type,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ tiered_device_supports_op,
    /* .supports_buft        = */ tiered_device_supports_buft,
    /* .offload_op           = */ tiered_device_offload_op,
    /* .event_new            = */ tiered_device_event_new,
    /* .event_free           = */ tiered_device_event_free,
    /* .event_synchronize    = */ tiered_device_event_sync,
};

static const char * tiered_reg_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return "CUDA_TIERED";
}

static size_t tiered_reg_device_count(ggml_backend_reg_t reg) {
    auto * ctx = static_cast<tiered_registry_context *>(reg->context);
    return ctx->devices.size();
}

static ggml_backend_dev_t tiered_reg_device(ggml_backend_reg_t reg, size_t index) {
    auto * ctx = static_cast<tiered_registry_context *>(reg->context);
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static void * tiered_reg_proc(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (std::strcmp(name, "ggml_backend_cuda_tiered_plan_begin") == 0) {
        return reinterpret_cast<void *>(ggml_backend_cuda_tiered_plan_begin);
    }
    if (std::strcmp(name, "ggml_backend_cuda_tiered_plan_end") == 0) {
        return reinterpret_cast<void *>(ggml_backend_cuda_tiered_plan_end);
    }
    return nullptr;
}

static const ggml_backend_reg_i tiered_reg_interface = {
    /* .get_name         = */ tiered_reg_name,
    /* .get_device_count = */ tiered_reg_device_count,
    /* .get_device       = */ tiered_reg_device,
    /* .get_proc_address = */ tiered_reg_proc,
};

} // namespace

extern "C" ggml_backend_buffer_type_t ggml_backend_cuda_tiered_plan_begin(
        ggml_backend_dev_t dev,
        const ggml_cuda_tiered_tensor_plan * entries,
        size_t n_entries,
        ggml_cuda_tiered_plan_options options) {
    if (!dev || !entries || n_entries == 0) {
        return nullptr;
    }
    auto * dev_ctx = static_cast<tiered_device_context *>(dev->context);
    auto plan = std::make_shared<tiered_plan>();
    plan->options = options;
    plan->tensors.reserve(n_entries);
    for (size_t i = 0; i < n_entries; ++i) {
        if (!entries[i].name) {
            return nullptr;
        }
        plan->tensors.emplace(entries[i].name, entries[i].tier);
    }
    tls_plans[dev_ctx->device] = std::move(plan);
    return dev_ctx->tiered_buft;
}

extern "C" void ggml_backend_cuda_tiered_plan_end(ggml_backend_dev_t dev) {
    if (!dev) {
        return;
    }
    auto * dev_ctx = static_cast<tiered_device_context *>(dev->context);
    tls_plans.erase(dev_ctx->device);
}

extern "C" void ggml_backend_cuda_tiered_register(void) {
    std::call_once(tiered_register_once, [] {
        const int count = ggml_backend_cuda_get_device_count();
        tiered_registry_ctx.devices.reserve(static_cast<size_t>(count));

        for (int device = 0; device < count; ++device) {
            ggml_backend_dev_t inner_dev = ggml_backend_reg_dev_get(
                    ggml_backend_cuda_reg(), static_cast<size_t>(device));

            auto * dev_ctx = new tiered_device_context;
            dev_ctx->device = device;
            dev_ctx->inner_device = inner_dev;
            dev_ctx->name = "CUDA_TIERED" + std::to_string(device);
            dev_ctx->description = std::string("tiered-memory wrapper for ") +
                    ggml_backend_dev_description(inner_dev);

            ggml_backend_dev_props props = {};
            ggml_backend_dev_get_props(inner_dev, &props);
            dev_ctx->device_id = props.device_id ? props.device_id : dev_ctx->name;
            dev_ctx->device_id += "-tiered";

            auto * buft_ctx = new tiered_buffer_type_context;
            buft_ctx->device = device;
            buft_ctx->name = dev_ctx->name + "_Weights";

            auto * buft = new ggml_backend_buffer_type {
                /* .iface   = */ tiered_buft_interface,
                /* .device  = */ nullptr,
                /* .context = */ buft_ctx,
            };

            ggml_backend_dev_t dev = new ggml_backend_device {
                /* .iface   = */ tiered_device_interface,
                /* .reg     = */ &tiered_registry,
                /* .context = */ dev_ctx,
            };
            buft->device = dev;
            buft_ctx->device_handle = dev;
            dev_ctx->tiered_buft = buft;
            tiered_registry_ctx.devices.push_back(dev);
        }

        tiered_registry = ggml_backend_reg {
            /* .api_version = */ GGML_BACKEND_API_VERSION,
            /* .iface       = */ tiered_reg_interface,
            /* .context     = */ &tiered_registry_ctx,
        };
        ggml_backend_register(&tiered_registry);
    });
}

#if defined(GGML_BACKEND_DL)
namespace {
struct tiered_dynamic_registrar {
    tiered_dynamic_registrar() {
        ggml_backend_cuda_tiered_register();
    }
};
static tiered_dynamic_registrar tiered_dynamic_registration;
} // namespace
#endif

#else

extern "C" ggml_backend_buffer_type_t ggml_backend_cuda_tiered_plan_begin(
        ggml_backend_dev_t dev,
        const ggml_cuda_tiered_tensor_plan * entries,
        size_t n_entries,
        ggml_cuda_tiered_plan_options options) {
    GGML_UNUSED(dev);
    GGML_UNUSED(entries);
    GGML_UNUSED(n_entries);
    GGML_UNUSED(options);
    return nullptr;
}

extern "C" void ggml_backend_cuda_tiered_plan_end(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
}

extern "C" void ggml_backend_cuda_tiered_register(void) {
}

#endif
