#include "batch-get.cuh"
#include "common.cuh"
#include "ggml-cpp.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
struct ggml_cuda_batch_get_descriptor {
    const unsigned char * src;
    unsigned char * dst;
    size_t size;
};

static __global__ void ggml_cuda_batch_get_kernel(const ggml_cuda_batch_get_descriptor * copies, size_t count) {
    constexpr int warp = ggml_cuda_get_physical_warp_size();
    const size_t index = ((size_t) blockIdx.x * blockDim.x + threadIdx.x) / warp;
    if (index >= count) {
        return;
    }
    const int lane = threadIdx.x % warp;
    const auto copy = copies[index];
    const uintptr_t alignment = (uintptr_t) copy.src | (uintptr_t) copy.dst;
    size_t tail = 0;
    if (alignment % 16 == 0) {
        for (size_t i = lane; i < copy.size / 16; i += warp) {
            ggml_cuda_memcpy_1<16>(copy.dst + 16*i, copy.src + 16*i);
        }
        tail = copy.size / 16 * 16;
    } else if (alignment % 4 == 0) {
        for (size_t i = lane; i < copy.size / 4; i += warp) {
            ggml_cuda_memcpy_1<4>(copy.dst + 4*i, copy.src + 4*i);
        }
        tail = copy.size / 4 * 4;
    } else if (alignment % 2 == 0) {
        for (size_t i = lane; i < copy.size / 2; i += warp) {
            ggml_cuda_memcpy_1<2>(copy.dst + 2*i, copy.src + 2*i);
        }
        tail = copy.size / 2 * 2;
    }
    for (size_t i = tail + lane; i < copy.size; i += warp) {
        ggml_cuda_memcpy_1<1>(copy.dst + i, copy.src + i);
    }
}

struct ggml_cuda_batch_get_mapping {
    void * host = nullptr;

    ~ggml_cuda_batch_get_mapping() {
        if (host) {
            CUDA_CHECK(cudaHostUnregister(host));
        }
    }
};
#endif

bool ggml_cuda_get_tensor_n(int device,
        const ggml_tensor ** tensors, void * data, const size_t * offsets,
        const size_t * sizes, const size_t * data_offsets, size_t n_tensors) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(device);
    GGML_UNUSED(tensors);
    GGML_UNUSED(data);
    GGML_UNUSED(offsets);
    GGML_UNUSED(sizes);
    GGML_UNUSED(data_offsets);
    GGML_UNUSED(n_tensors);
    return false;
#else
    constexpr size_t max_copies = 1024 * 1024;
    constexpr size_t tile_bytes = 64 * 1024;
    std::vector<std::pair<size_t, size_t>> ranges;
    ranges.reserve(n_tensors);
    size_t tiles = 0;
    for (size_t i = 0; i < n_tensors; ++i) {
        if (sizes[i] == 0) {
            continue;
        }
        if (data_offsets[i] > SIZE_MAX - sizes[i]) {
            return false;
        }
        ranges.emplace_back(data_offsets[i], data_offsets[i] + sizes[i]);
        const size_t count = sizes[i] / tile_bytes + (sizes[i] % tile_bytes != 0);
        tiles += std::min(count, max_copies - tiles);
    }
    if (ranges.empty()) {
        return true;
    }
    std::sort(ranges.begin(), ranges.end());
    for (size_t i = 1; i < ranges.size(); ++i) {
        // Preserve the ordered copy path when CPU destinations overlap.
        if (ranges[i].first < ranges[i - 1].second) {
            return false;
        }
    }
#if defined(_WIN32)
    SYSTEM_INFO system_info;
    GetSystemInfo(&system_info);
    const uintptr_t page = system_info.dwPageSize;
#else
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        return false;
    }
    const uintptr_t page = (uintptr_t) page_size;
#endif
    const uintptr_t base = (uintptr_t) data;
    if (base > UINTPTR_MAX - ranges.back().second || base + ranges.back().second > UINTPTR_MAX - (page - 1)) {
        return false;
    }
    const uintptr_t begin = (base + ranges.front().first) / page * page;
    const uintptr_t end = (base + ranges.back().second + page - 1) / page * page;
    const size_t bytes = tiles * sizeof(ggml_cuda_batch_get_descriptor);
    ggml_backend_buffer_ptr descriptors(ggml_backend_buft_alloc_buffer(ggml_backend_cuda_buffer_type(device), bytes));
    if (!descriptors) {
        return false;
    }
    const auto host_type = ggml_backend_cuda_host_buffer_type();
    ggml_backend_buffer_ptr host_descriptors(ggml_backend_buft_alloc_buffer(host_type, bytes));
    if (!host_descriptors || ggml_backend_buffer_get_type(host_descriptors.get()) != host_type) {
        return false;
    }
    ggml_cuda_batch_get_mapping mapping;
    if (cudaHostRegister((void *) begin, end - begin, cudaHostRegisterMapped | cudaHostRegisterPortable) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    mapping.host = (void *) begin;
    unsigned char * destination = nullptr;
    if (cudaHostGetDevicePointer((void **) &destination, mapping.host, 0) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    auto * copies = static_cast<ggml_cuda_batch_get_descriptor *>(ggml_backend_buffer_get_base(host_descriptors.get()));
    auto * device_copies = static_cast<ggml_cuda_batch_get_descriptor *>(ggml_backend_buffer_get_base(descriptors.get()));
    size_t index = 0;
    size_t consumed = 0;
    while (index < n_tensors) {
        size_t count = 0;
        while (index < n_tensors && count < tiles) {
            const size_t size = std::min(sizes[index] - consumed, tile_bytes);
            if (size) {
                copies[count++] = {(const unsigned char *) tensors[index]->data + offsets[index] + consumed,
                        destination + (base + data_offsets[index] - begin) + consumed, size};
            }
            consumed += size;
            if (consumed == sizes[index]) {
                ++index;
                consumed = 0;
            }
        }
        if (count) {
            CUDA_CHECK(cudaMemcpyAsync(device_copies, copies, count * sizeof(copies[0]), cudaMemcpyHostToDevice, cudaStreamPerThread));
            const int threads = 256;
            const int warp = ggml_cuda_info().devices[device].warp_size;
            const unsigned int blocks = (unsigned int) ((count + threads / warp - 1) / (threads / warp));
            ggml_cuda_batch_get_kernel<<<blocks, threads, 0, cudaStreamPerThread>>>(device_copies, count);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        }
    }
    return true;
#endif
}
