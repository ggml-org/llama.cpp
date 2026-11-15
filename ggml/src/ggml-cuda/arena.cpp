#include "arena.h"
#include "ggml-cuda.h"
#include "ggml-impl.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <memory>
#include <new>
#include <utility>
#include <vector>

static const char * arena_error(cudaError_t err) {
    return err == cudaSuccess ? nullptr : cudaGetErrorString(err);
}

static const char * arena_error(CUresult err) {
    if (err == CUDA_SUCCESS) {
        return nullptr;
    }
    const char * msg = nullptr;
    cuGetErrorString(err, &msg);
    return msg ? msg : "CUDA driver error";
}

#define ARENA_CHECK(call) do { const char * err = arena_error(call); if (err) return err; } while (0)

struct ggml_cuda_arena {
    int device;
    CUmemAllocationProp prop = {};
    CUmemAccessDesc access[GGML_CUDA_MAX_DEVICES] = {};
    size_t n_access = 0;
    CUdeviceptr base = 0;
    size_t capacity = 0;
    size_t mapped = 0;
    size_t granularity = 0;
    cudaStream_t stream = nullptr;
    std::vector<std::pair<size_t, size_t>> mappings;

    ~ggml_cuda_arena() {
        cudaSetDevice(device);
        if (stream) {
            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
        }
        for (auto mapping : mappings) {
            cuMemUnmap(base + mapping.first, mapping.second);
        }
        if (base) {
            cuMemAddressFree(base, capacity);
        }
    }
};

static const char * arena_create(int device, size_t reserved, ggml_cuda_arena ** out, uintptr_t * base) {
    ARENA_CHECK(cudaSetDevice(device));
    std::unique_ptr<ggml_cuda_arena> arena(new (std::nothrow) ggml_cuda_arena);
    if (!arena) {
        return "out of host memory for CUDA arena";
    }
    arena->device = device;
    size_t free, total;
    ARENA_CHECK(cudaMemGetInfo(&free, &total));
    const size_t margin = 512ull << 20;
    if (reserved > free || free - reserved <= margin) {
        return "not enough GPU memory for arena";
    }
    arena->prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    arena->prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    arena->prop.location.id = device;
    for (int peer = 0; peer < ggml_backend_cuda_get_device_count(); peer++) {
        if (peer != device) {
            int can_access = 0;
            ARENA_CHECK(cudaDeviceCanAccessPeer(&can_access, peer, device));
            if (!can_access) continue;
        }
        auto & access = arena->access[arena->n_access++];
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = peer;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }
    ARENA_CHECK(cuMemGetAllocationGranularity(&arena->granularity, &arena->prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    arena->capacity = (free - reserved - margin) / arena->granularity * arena->granularity;
    ARENA_CHECK(cuMemAddressReserve(&arena->base, arena->capacity, 0, 0, 0));
    ARENA_CHECK(cudaStreamCreateWithFlags(&arena->stream, cudaStreamNonBlocking));
    *base = arena->base;
    *out = arena.release();
    return nullptr;
}

static const char * arena_grow(ggml_cuda_arena * arena, size_t end, size_t * mapped) {
    if (end > arena->capacity) {
        GGML_LOG_ERROR("GPU%d: requested arena extent %.2f MiB exceeds capacity %.2f MiB\n",
                arena->device, end / 1048576.0, arena->capacity / 1048576.0);
        return "hybrid GPU arena capacity exceeded";
    }
    if (end <= arena->mapped) {
        *mapped = arena->mapped;
        return nullptr;
    }
    ARENA_CHECK(cudaSetDevice(arena->device));
    size_t extent = std::max(size_t(64ull << 20), end - arena->mapped);
    extent = (extent + arena->granularity - 1) / arena->granularity * arena->granularity;
    extent = std::min(extent, arena->capacity - arena->mapped);
    CUmemGenericAllocationHandle handle;
    ARENA_CHECK(cuMemCreate(&handle, extent, &arena->prop, 0));
    CUdeviceptr ptr = arena->base + arena->mapped;
    CUresult result = cuMemMap(ptr, extent, 0, handle, 0);
    cuMemRelease(handle);
    ARENA_CHECK(result);
    try {
        arena->mappings.emplace_back(arena->mapped, extent);
    } catch (const std::bad_alloc &) {
        cuMemUnmap(ptr, extent);
        return "out of host memory for CUDA arena mapping";
    }
    // VMM allocations need explicit peer access, including when NCCL enables P2P.
    const char * err = arena_error(cuMemSetAccess(ptr, extent, arena->access, arena->n_access));
    if (!err) {
        err = arena_error(cudaMemset((void *) ptr, 0, extent));
    }
    if (err) {
        cuMemUnmap(ptr, extent);
        arena->mappings.pop_back();
        return err;
    }
    arena->mapped += extent;
    *mapped = arena->mapped;
    return nullptr;
}

static const char * arena_snapshot(ggml_cuda_arena * arena, void * dst, size_t size) {
    ARENA_CHECK(cudaSetDevice(arena->device));
    return arena_error(cudaMemcpy(dst, (void *) arena->base, size, cudaMemcpyDeviceToHost));
}

static cudaError_t copy_rows(void * dst, size_t dst_pitch, const void * src, size_t src_pitch,
        size_t width, size_t rows, cudaMemcpyKind kind, cudaStream_t stream) {
    if (rows == 1) {
        return cudaMemcpyAsync(dst, src, width, kind, stream);
    }
    return cudaMemcpy2DAsync(dst, dst_pitch, src, src_pitch, width, rows, kind, stream);
}

static const char * arena_restore(ggml_cuda_arena * arena, size_t scratch_offset,
        const ggml_cuda_arena_copy * local, size_t n_local, const void * uploads,
        const ggml_cuda_arena_copy * upload, size_t n_upload, const ggml_cuda_arena_span * gaps, size_t n_gaps) {
    ARENA_CHECK(cudaSetDevice(arena->device));
    CUdeviceptr scratch = arena->base + scratch_offset;
    for (size_t i = 0; i < n_local; ++i) {
        const auto & copy = local[i];
        ARENA_CHECK(copy_rows((void *) (scratch + copy.scratch), copy.width, (void *) (arena->base + copy.src),
                    copy.src_pitch, copy.width, copy.rows, cudaMemcpyDeviceToDevice, arena->stream));
    }
    for (size_t i = 0; i < n_local; ++i) {
        const auto & copy = local[i];
        ARENA_CHECK(copy_rows((void *) (arena->base + copy.dst), copy.dst_pitch, (void *) (scratch + copy.scratch),
                    copy.width, copy.width, copy.rows, cudaMemcpyDeviceToDevice, arena->stream));
    }
    for (size_t i = 0; i < n_upload; ++i) {
        const auto & copy = upload[i];
        ARENA_CHECK(copy_rows((void *) (arena->base + copy.dst), copy.dst_pitch, (const char *) uploads + copy.src,
                    copy.src_pitch, copy.width, copy.rows, cudaMemcpyHostToDevice, arena->stream));
    }
    for (size_t i = 0; i < n_gaps; ++i) {
        ARENA_CHECK(cudaMemsetAsync((void *) (arena->base + gaps[i].offset), 0, gaps[i].size, arena->stream));
    }
    return arena_error(cudaStreamSynchronize(arena->stream));
}

const ggml_cuda_arena_api * ggml_backend_cuda_get_arena_api() {
    static const ggml_cuda_arena_api api = {
        arena_create,
        [](ggml_cuda_arena * arena) { delete arena; },
        arena_grow,
        [](void ** ptr, size_t size) { return arena_error(cudaHostAlloc(ptr, size, cudaHostAllocPortable | cudaHostAllocMapped)); },
        [](void * ptr) { cudaFreeHost(ptr); },
        arena_snapshot,
        arena_restore,
    };
    return &api;
}
