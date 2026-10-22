#include "batch-get.cuh"
#include "common.cuh"
#include "ggml-cpp.h"

#include <climits>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
struct ggml_cuda_batch_get_descriptor {
    const unsigned char * src;
    unsigned char * dst;
    size_t size;
};

static __global__ void ggml_cuda_batch_get_kernel(const ggml_cuda_batch_get_descriptor * copies, size_t count) {
    // Blocks along y cooperate on each descriptor.
    const size_t index = blockIdx.x;
    if (index >= count) {
        return;
    }
    const size_t first = (size_t) blockIdx.y * blockDim.x + threadIdx.x;
    const size_t stride = (size_t) gridDim.y * blockDim.x;
    const auto copy = copies[index];
    // Use 16-, 4-, or 2-byte copies when both pointers meet that alignment; threads copy separate chunks.
    const uintptr_t alignment = (uintptr_t) copy.src | (uintptr_t) copy.dst;
    size_t tail = 0;
    if (alignment % 16 == 0) {
        for (size_t i = first; i < copy.size / 16; i += stride) {
            ggml_cuda_memcpy_1<16>(copy.dst + 16*i, copy.src + 16*i);
        }
        tail = copy.size / 16 * 16;
    } else if (alignment % 4 == 0) {
        for (size_t i = first; i < copy.size / 4; i += stride) {
            ggml_cuda_memcpy_1<4>(copy.dst + 4*i, copy.src + 4*i);
        }
        tail = copy.size / 4 * 4;
    } else if (alignment % 2 == 0) {
        for (size_t i = first; i < copy.size / 2; i += stride) {
            ggml_cuda_memcpy_1<2>(copy.dst + 2*i, copy.src + 2*i);
        }
        tail = copy.size / 2 * 2;
    }
    // Copy trailing bytes, or the entire range when no wider alignment is available.
    for (size_t i = tail + first; i < copy.size; i += stride) {
        ggml_cuda_memcpy_1<1>(copy.dst + i, copy.src + i);
    }
}

struct ggml_cuda_batch_get_mapping {
    void * host = nullptr;

    ~ggml_cuda_batch_get_mapping() {
        if (host) {
            // Unregister and unpin the CPU destination on scope exit; the caller still owns its storage.
            CUDA_CHECK(cudaHostUnregister(host));
        }
    }
};
#endif

bool ggml_cuda_get_tensor_n(int device,
        const ggml_tensor * const * tensors, void * const * data, const size_t * offsets,
        const size_t * sizes, size_t n_tensors) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(device);
    GGML_UNUSED(tensors);
    GGML_UNUSED(data);
    GGML_UNUSED(offsets);
    GGML_UNUSED(sizes);
    GGML_UNUSED(n_tensors);
    return false;
#else
    // Provisional dispatch values based on experiments.
    constexpr size_t min_batch_reads = 128;
    constexpr size_t max_bytes_per_read = 128 * 1024;
    // Collect non-empty CPU destination ranges.
    std::vector<std::pair<uintptr_t, uintptr_t>> ranges;
    ranges.reserve(n_tensors);
    size_t payload_bytes = 0;
    for (size_t i = 0; i < n_tensors; ++i) {
        if (sizes[i] == 0) {
            continue;
        }
        const uintptr_t address = (uintptr_t) data[i];
        if (address > UINTPTR_MAX - sizes[i] || payload_bytes > SIZE_MAX - sizes[i]) {
            return false;
        }
        payload_bytes += sizes[i];
        ranges.emplace_back(address, address + sizes[i]);
    }
    if (ranges.empty()) {
        return true;
    }
    const size_t n_copies = ranges.size();
    if (n_copies > SIZE_MAX / sizeof(ggml_cuda_batch_get_descriptor) || n_copies > INT_MAX) {
        return false;
    }
    if (n_copies < min_batch_reads) {
        return false;
    }
    const int threads = 256;
    const size_t nsm = ggml_cuda_info().devices[device].nsm;
    const size_t blocks_per_copy = std::min<size_t>(UINT16_MAX, std::max<size_t>(1, (nsm + n_copies - 1) / n_copies));
    const dim3 blocks((unsigned int) n_copies, (unsigned int) blocks_per_copy);
    std::sort(ranges.begin(), ranges.end());
    for (size_t i = 1; i < ranges.size(); ++i) {
        // Preserve the ordered copy path when CPU destinations overlap.
        if (ranges[i].first < ranges[i - 1].second) {
            return false;
        }
    }
    // Find the CPU span to register, including gaps between destinations.
    const uintptr_t begin = ranges.front().first;
    const uintptr_t end = ranges.back().second;
    const size_t bytes_per_read = payload_bytes / n_copies + (payload_bytes % n_copies != 0);
    if (bytes_per_read > max_bytes_per_read || end - begin - payload_bytes > payload_bytes) {
        return false;
    }
    // Allocate per-call descriptor storage on the GPU and in pinned CPU memory; both buffers are freed on scope exit.
    const size_t bytes = n_copies * sizeof(ggml_cuda_batch_get_descriptor);
    ggml_backend_buffer_ptr descriptors(ggml_backend_buft_alloc_buffer(ggml_backend_cuda_buffer_type(device), bytes));
    if (!descriptors) {
        return false;
    }
    const auto host_type = ggml_backend_cuda_host_buffer_type();
    ggml_backend_buffer_ptr host_descriptors(ggml_backend_buft_alloc_buffer(host_type, bytes));
    if (!host_descriptors || ggml_backend_buffer_get_type(host_descriptors.get()) != host_type) {
        return false;
    }
    // Register and map the existing CPU destination so the kernel can write to it directly.
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
    size_t count = 0;
    // Fill one descriptor for each non-empty read.
    for (size_t i = 0; i < n_tensors; ++i) {
        if (sizes[i] == 0) {
            continue;
        }
        copies[count++] = {(const unsigned char *) tensors[i]->data + offsets[i],
                destination + ((uintptr_t) data[i] - begin), sizes[i]};
    }
    // Upload descriptors, then launch the kernel to copy payload bytes into mapped CPU memory.
    CUDA_CHECK(cudaMemcpyAsync(device_copies, copies, bytes, cudaMemcpyHostToDevice, cudaStreamPerThread));
    ggml_cuda_batch_get_kernel<<<blocks, threads, 0, cudaStreamPerThread>>>(device_copies, count);
    CUDA_CHECK(cudaGetLastError());
    // Wait for GPU writes before releasing the CPU mapping and descriptor buffers.
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    return true;
#endif
}
