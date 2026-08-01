#include "ggml-cuda.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
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

struct sparse_mapping {
    CUdeviceptr address = 0;
    size_t size = 0;
    CUmemGenericAllocationHandle handle = 0;
};

struct tensor_state {
    ggml_cuda_tiered_memory_tier tier = GGML_CUDA_TIERED_MEMORY_VRAM;
    void * host_ptr = nullptr;
    void * device_ptr = nullptr;
    size_t size = 0;
    size_t alloc_size = 0;

    CUdeviceptr sparse_address = 0;
    size_t sparse_size = 0;
    size_t sparse_granularity = 0;
    std::vector<sparse_mapping> active_mappings;
};

struct tiered_buffer_context {
    int device = 0;
    void * address_space = nullptr;
    size_t address_space_size = 0;
    std::shared_ptr<tiered_plan> plan;
    std::unordered_map<const ggml_tensor *, std::unique_ptr<tensor_state>> tensors;
    std::vector<ggml_tensor *> all_tensors;
    std::vector<host_registration> registrations;
};

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

static tiered_buffer_context * buffer_context(const ggml_tensor * tensor) {
    if (!tensor) {
        return nullptr;
    }
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (!buffer || !buffer->context) {
        return nullptr;
    }
    const char * name = ggml_backend_buft_name(buffer->buft);
    if (!name || std::strncmp(name, "CUDA_TIERED", 12) != 0) {
        return nullptr;
    }
    return static_cast<tiered_buffer_context *>(buffer->context);
}

static tensor_state * state_for(const ggml_tensor * tensor) {
    tiered_buffer_context * ctx = buffer_context(tensor);
    if (!ctx) {
        return nullptr;
    }
    auto found = ctx->tensors.find(tensor);
    if (found != ctx->tensors.end()) {
        return found->second.get();
    }
    if (tensor->view_src) {
        found = ctx->tensors.find(tensor->view_src);
        if (found != ctx->tensors.end()) {
            return found->second.get();
        }
    }
    return nullptr;
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
            unsigned int flags = cudaHostRegisterPortable | cudaHostRegisterMapped;
#if CUDART_VERSION >= 11010
            flags |= cudaHostRegisterReadOnly;
#endif
            const cudaError_t status = cudaHostRegister(reinterpret_cast<void *>(cursor), gap_size, flags);
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
        unsigned int flags = cudaHostRegisterPortable | cudaHostRegisterMapped;
#if CUDART_VERSION >= 11010
        flags |= cudaHostRegisterReadOnly;
#endif
        const cudaError_t status = cudaHostRegister(reinterpret_cast<void *>(cursor), gap_size, flags);
        if (status == cudaErrorHostMemoryAlreadyRegistered) {
            (void) cudaGetLastError();
            add_registration(ctx, cursor, wanted_end, false);
        } else {
            TIERED_CUDA_CHECK(status);
            add_registration(ctx, cursor, wanted_end, true);
        }
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
    for (const auto & mapping : state->active_mappings) {
        (void) cuMemUnmap(mapping.address, mapping.size);
        (void) cuMemRelease(mapping.handle);
    }
    state->active_mappings.clear();
    if (state->sparse_address) {
        (void) cuMemAddressFree(state->sparse_address, state->sparse_size);
        state->sparse_address = 0;
    }
#else
    GGML_UNUSED(state);
#endif
}

static void stage_sparse_experts(tiered_buffer_context * ctx, const ggml_tensor * tensor, tensor_state * state, const ggml_tensor * ids) {
#if defined(GGML_CUDA_NO_VMM)
    GGML_UNUSED(ctx);
    GGML_UNUSED(tensor);
    GGML_UNUSED(state);
    GGML_UNUSED(ids);
    throw std::runtime_error("tiered SSD streaming requires CUDA VMM support");
#else
    if (!state->active_mappings.empty()) {
        throw std::runtime_error("tiered SSD tensor already has active mappings");
    }
    if (!ids || ids->type != GGML_TYPE_I32) {
        throw std::runtime_error("MUL_MAT_ID expert ids must be I32");
    }

    std::vector<int32_t> host_ids(static_cast<size_t>(ggml_nelements(ids)));
    ggml_backend_tensor_get(ids, host_ids.data(), 0, ggml_nbytes(ids));

    std::vector<size_t> chunks;
    chunks.reserve(host_ids.size());
    const size_t granularity = state->sparse_granularity;

    for (const int32_t expert : host_ids) {
        if (expert < 0 || expert >= tensor->ne[2]) {
            throw std::runtime_error("MUL_MAT_ID produced an out-of-range expert id");
        }
        for (int64_t i3 = 0; i3 < tensor->ne[3]; ++i3) {
            const size_t slab_begin = static_cast<size_t>(i3) * tensor->nb[3] +
                    static_cast<size_t>(expert) * tensor->nb[2];
            const size_t slab_end = std::min(state->size, slab_begin + tensor->nb[2]);
            const size_t first = slab_begin / granularity * granularity;
            const size_t last = align_up_size(slab_end, granularity);
            for (size_t offset = first; offset < last; offset += granularity) {
                chunks.push_back(offset);
            }
        }
    }

    std::sort(chunks.begin(), chunks.end());
    chunks.erase(std::unique(chunks.begin(), chunks.end()), chunks.end());

    CUmemAllocationProp property = {};
    property.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    property.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    property.location.id = ctx->device;

    CUmemAccessDesc access = {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = ctx->device;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    try {
        for (const size_t offset : chunks) {
            const size_t mapping_size = std::min(granularity, state->sparse_size - offset);
            CUmemGenericAllocationHandle handle = 0;
            TIERED_CU_CHECK(cuMemCreate(&handle, mapping_size, &property, 0));
            const CUdeviceptr address = state->sparse_address + offset;
            try {
                TIERED_CU_CHECK(cuMemMap(address, mapping_size, 0, handle, 0));
                TIERED_CU_CHECK(cuMemSetAccess(address, mapping_size, &access, 1));
            } catch (...) {
                (void) cuMemRelease(handle);
                throw;
            }

            state->active_mappings.push_back({address, mapping_size, handle});

            const size_t copy_size = offset < state->size ?
                    std::min(mapping_size, state->size - offset) : 0;
            if (copy_size > 0) {
                TIERED_CUDA_CHECK(cudaMemcpy(
                        reinterpret_cast<void *>(address),
                        static_cast<const char *>(state->host_ptr) + offset,
                        copy_size,
                        cudaMemcpyHostToDevice));
            }
            if (copy_size < mapping_size) {
                TIERED_CUDA_CHECK(cudaMemset(
                        reinterpret_cast<char *>(address) + copy_size,
                        0,
                        mapping_size - copy_size));
            }
        }
    } catch (...) {
        free_sparse_tensor(state);
        state->sparse_size = align_up_size(state->size, granularity);
        TIERED_CU_CHECK(cuMemAddressReserve(
                &state->sparse_address, state->sparse_size,
                granularity, 0, 0));
        throw;
    }
#endif
}

static void unstage_sparse_experts(tensor_state * state) {
#if !defined(GGML_CUDA_NO_VMM)
    for (const auto & mapping : state->active_mappings) {
        TIERED_CU_CHECK(cuMemUnmap(mapping.address, mapping.size));
        TIERED_CU_CHECK(cuMemRelease(mapping.handle));
    }
    state->active_mappings.clear();
#else
    GGML_UNUSED(state);
#endif
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
        if (state->tier == GGML_CUDA_TIERED_MEMORY_SSD) {
            free_sparse_tensor(state);
        }
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
    if (ctx->tensors.find(tensor) == ctx->tensors.end()) {
        ctx->tensors.emplace(tensor, std::make_unique<tensor_state>());
        ctx->all_tensors.push_back(tensor);
    }
    return GGML_STATUS_SUCCESS;
}

static void tiered_refresh_views(tiered_buffer_context * ctx) {
    for (size_t pass = 0; pass < ctx->all_tensors.size(); ++pass) {
        bool changed = false;
        for (ggml_tensor * tensor : ctx->all_tensors) {
            if (!tensor->view_src || !tensor->view_src->data) {
                continue;
            }
            void * expected = static_cast<char *>(tensor->view_src->data) + tensor->view_offs;
            if (tensor->data != expected) {
                tensor->data = expected;
                changed = true;
            }
        }
        if (!changed) {
            break;
        }
    }
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

    auto & state_ptr = ctx->tensors[tensor];
    if (!state_ptr) {
        state_ptr = std::make_unique<tensor_state>();
        ctx->all_tensors.push_back(tensor);
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

    if (state->tier != GGML_CUDA_TIERED_MEMORY_VRAM && state->alloc_size != state->size) {
        if (ctx->plan->options.strict) {
            throw std::runtime_error(std::string("host tier does not support CUDA padding for tensor: ") + tensor->name);
        }
        state->tier = GGML_CUDA_TIERED_MEMORY_VRAM;
    }

    switch (state->tier) {
        case GGML_CUDA_TIERED_MEMORY_VRAM: {
            TIERED_CUDA_CHECK(cudaMalloc(&state->device_ptr, state->alloc_size));
            TIERED_CUDA_CHECK(cudaMemcpy(
                    state->device_ptr, data, state->size, cudaMemcpyHostToDevice));
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
            ensure_registered(ctx, state->host_ptr, state->size);
            TIERED_CUDA_CHECK(cudaHostGetDevicePointer(
                    &state->device_ptr, state->host_ptr, 0));
            tensor->data = state->device_ptr;
        } break;

        case GGML_CUDA_TIERED_MEMORY_SSD:
            init_sparse_tensor(ctx, tensor, state);
            break;
    }

    tiered_refresh_views(ctx);

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
    auto found = ctx->tensors.find(tensor);
    if (found == ctx->tensors.end()) {
        throw std::runtime_error("tiered tensor state not found");
    }
    tensor_state * state = found->second.get();
    if (offset + size > state->size) {
        throw std::runtime_error("tiered tensor read is out of bounds");
    }
    if (state->tier == GGML_CUDA_TIERED_MEMORY_VRAM) {
        set_device(ctx->device);
        TIERED_CUDA_CHECK(cudaMemcpy(
                data, static_cast<const char *>(state->device_ptr) + offset,
                size, cudaMemcpyDeviceToHost));
    } else {
        std::memcpy(data, static_cast<const char *>(state->host_ptr) + offset, size);
    }
}

static void tiered_buffer_memset_tensor(
        ggml_backend_buffer_t buffer,
        ggml_tensor * tensor,
        uint8_t value,
        size_t offset,
        size_t size) {
    auto * ctx = static_cast<tiered_buffer_context *>(buffer->context);
    auto found = ctx->tensors.find(tensor);
    if (found == ctx->tensors.end()) {
        return;
    }
    tensor_state * state = found->second.get();
    if (state->tier == GGML_CUDA_TIERED_MEMORY_VRAM) {
        set_device(ctx->device);
        TIERED_CUDA_CHECK(cudaMemset(
                static_cast<char *>(state->device_ptr) + offset, value, size));
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
    view.uid = ggml_graph_next_uid();
    return ggml_backend_graph_compute(ctx->inner, &view);
}

static ggml_status tiered_backend_graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    auto * ctx = static_cast<tiered_backend_context *>(backend->context);
    int segment_begin = 0;

    for (int i = 0; i < graph->n_nodes; ++i) {
        ggml_tensor * node = graph->nodes[i];
        tensor_state * ssd_state = nullptr;
        tiered_buffer_context * weight_ctx = nullptr;

        for (int src_index = 0; src_index < GGML_MAX_SRC; ++src_index) {
            ggml_tensor * src = node->src[src_index];
            tensor_state * state = state_for(src);
            if (!state || state->tier != GGML_CUDA_TIERED_MEMORY_SSD) {
                continue;
            }
            if (node->op != GGML_OP_MUL_MAT_ID || src_index != 0) {
                GGML_LOG_ERROR("tiered-memory: SSD tensor %s is used by unsupported op %s\n",
                        src->name, ggml_op_name(node->op));
                return GGML_STATUS_FAILED;
            }
            ssd_state = state;
            weight_ctx = buffer_context(src);
        }

        if (!ssd_state) {
            continue;
        }

        ggml_status status = compute_view(ctx, graph, segment_begin, i);
        if (status != GGML_STATUS_SUCCESS) {
            return status;
        }
        ggml_backend_synchronize(ctx->inner);

        ggml_tensor * weight = node->src[0];
        try {
            stage_sparse_experts(weight_ctx, weight, ssd_state, node->src[2]);
            status = compute_view(ctx, graph, i, i + 1);
            ggml_backend_synchronize(ctx->inner);
            unstage_sparse_experts(ssd_state);
        } catch (const std::exception & error) {
            GGML_LOG_ERROR("tiered-memory: failed to stream %s: %s\n",
                    weight->name, error.what());
            try {
                unstage_sparse_experts(ssd_state);
            } catch (...) {
            }
            return GGML_STATUS_FAILED;
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
