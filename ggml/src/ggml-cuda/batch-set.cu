#include "batch-set.cuh"
#include "common.cuh"
#include "ggml-cpp.h"

#include <climits>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
struct ggml_cuda_batch_set_descriptor {
    const unsigned char * src;
    unsigned char * dst;
    size_t size;
};

static __global__ void ggml_cuda_batch_set_kernel(const ggml_cuda_batch_set_descriptor * copies, size_t count) {
    // Blocks along y cooperate on each descriptor.
    const size_t index = blockIdx.x;
    if (index >= count) {
        return;
    }
    const size_t first = (size_t) blockIdx.y * blockDim.x + threadIdx.x;
    const size_t stride = (size_t) gridDim.y * blockDim.x;
    const auto copy = copies[index];
    // Use 16-, 8-, 4-, or 2-byte copies when both pointers meet that alignment; threads copy separate chunks.
    const uintptr_t alignment = (uintptr_t) copy.src | (uintptr_t) copy.dst;
    size_t tail = 0;
    if (alignment % 16 == 0) {
        for (size_t i = first; i < copy.size / 16; i += stride) {
            ggml_cuda_memcpy_1<16>(copy.dst + 16*i, copy.src + 16*i);
        }
        tail = copy.size / 16 * 16;
    } else if (alignment % 8 == 0) {
        for (size_t i = first; i < copy.size / 8; i += stride) {
            ggml_cuda_memcpy_1<8>(copy.dst + 8*i, copy.src + 8*i);
        }
        tail = copy.size / 8 * 8;
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

struct ggml_cuda_batch_set_mapping {
    void * host = nullptr;
    unsigned char * device = nullptr;

    ggml_cuda_batch_set_mapping(void * host, size_t size) {
        const cudaError_t err = cudaHostRegister(host, size, cudaHostRegisterMapped | cudaHostRegisterPortable);
        if (err != cudaSuccess) {
            (void) cudaGetLastError();
            GGML_LOG_DEBUG("%s: failed to register %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
            return;
        }
        this->host = host;
        if (cudaHostGetDevicePointer((void **) &device, host, 0) != cudaSuccess) {
            (void) cudaGetLastError();
            device = nullptr;
            return;
        }
    }

    ggml_cuda_batch_set_mapping(const ggml_cuda_batch_set_mapping &) = delete;
    ggml_cuda_batch_set_mapping & operator=(const ggml_cuda_batch_set_mapping &) = delete;

    ~ggml_cuda_batch_set_mapping() {
        if (host) {
            // Release only the registration acquired by this object.
            CUDA_CHECK(cudaHostUnregister(host));
        }
    }
};
#endif

bool ggml_cuda_set_tensor_n(int device,
        ggml_tensor * const * tensors, const void * const * data, const size_t * offsets,
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
    // Use the same dispatch limits as batch-get.
    constexpr size_t min_batch_reads = 128;
    constexpr size_t max_scatter_read_bytes = 128 * 1024;
    // Collect GPU destination ranges; repeated CPU sources are allowed.
    std::vector<std::pair<uintptr_t, uintptr_t>> ranges;
    ranges.reserve(n_tensors);
    size_t payload_bytes = 0;
    uintptr_t begin = UINTPTR_MAX;
    uintptr_t end = 0;
    // Number of non-empty reads whose size is at most max_scatter_read_bytes.
    size_t n_reads_small = 0;
    for (size_t i = 0; i < n_tensors; ++i) {
        if (sizes[i] == 0) {
            continue;
        }
        const uintptr_t address = (uintptr_t) data[i];
        GGML_ASSERT(address <= UINTPTR_MAX - sizes[i]);
        GGML_ASSERT(payload_bytes <= SIZE_MAX - sizes[i]);
        payload_bytes += sizes[i];
        n_reads_small += sizes[i] <= max_scatter_read_bytes;
        begin = std::min(begin, address);
        end = std::max(end, address + sizes[i]);
        const uintptr_t destination = (uintptr_t) tensors[i]->data + offsets[i];
        GGML_ASSERT(destination <= UINTPTR_MAX - sizes[i]);
        ranges.emplace_back(destination, destination + sizes[i]);
    }
    if (ranges.empty()) {
        return true;
    }
    // Total number of non-empty reads.
    const size_t n_reads = ranges.size();
    if (n_reads > SIZE_MAX / sizeof(ggml_cuda_batch_set_descriptor) || n_reads > INT_MAX) {
        return false;
    }
    if (n_reads < 2 || (n_reads < min_batch_reads && n_reads_small == n_reads)) {
        return false;
    }
    std::sort(ranges.begin(), ranges.end());
    for (size_t i = 1; i < ranges.size(); ++i) {
        // Preserve the ordered copy path when GPU destinations overlap.
        if (ranges[i].first < ranges[i - 1].second) {
            return false;
        }
    }
    // Limit gaps in the CPU span without underflow when sources overlap.
    if (end - begin > payload_bytes && end - begin - payload_bytes > payload_bytes) {
        return false;
    }
    // Register and map the CPU source so the kernel can read it directly.
    ggml_cuda_batch_set_mapping mapping((void *) begin, end - begin);
    if (!mapping.host) {
        return false;
    }
    // Reads selected for scatter; zero means all reads use DMA.
    size_t n_reads_scatter = mapping.device && n_reads_small >= min_batch_reads ? n_reads_small : 0;
    // Keep descriptor storage alive until the final stream synchronization.
    constexpr size_t max_batch_descriptors = 128 * 1024;
    const size_t bytes = std::min(n_reads_scatter, max_batch_descriptors) * sizeof(ggml_cuda_batch_set_descriptor);
    ggml_backend_buffer_ptr descriptors(n_reads_scatter ? ggml_backend_buft_alloc_buffer(ggml_backend_cuda_buffer_type(device), bytes) : nullptr);
    if (!descriptors) {
        n_reads_scatter = 0;
    }
    const auto host_type = ggml_backend_cuda_host_buffer_type();
    ggml_backend_buffer_ptr host_descriptors(n_reads_scatter ? ggml_backend_buft_alloc_buffer(host_type, bytes) : nullptr);
    if (!host_descriptors || ggml_backend_buffer_get_type(host_descriptors.get()) != host_type) {
        n_reads_scatter = 0;
    }
    if (n_reads_scatter) {
        auto * copies = static_cast<ggml_cuda_batch_set_descriptor *>(ggml_backend_buffer_get_base(host_descriptors.get()));
        auto * device_copies = static_cast<ggml_cuda_batch_set_descriptor *>(ggml_backend_buffer_get_base(descriptors.get()));
        const int threads = 256;
        const size_t nsm = ggml_cuda_info().devices[device].nsm;
        size_t i = 0;
        for (size_t first = 0; first < n_reads_scatter; first += max_batch_descriptors) {
            const size_t n_batch = std::min(n_reads_scatter - first, max_batch_descriptors);
            size_t count = 0;
            for (; i < n_tensors && count < n_batch; ++i) {
                if (sizes[i] == 0 || sizes[i] > max_scatter_read_bytes) {
                    continue;
                }
                copies[count++] = {mapping.device + ((uintptr_t) data[i] - begin),
                        (unsigned char *) tensors[i]->data + offsets[i], sizes[i]};
            }
            const size_t blocks_per_copy = std::min<size_t>(UINT16_MAX, std::max<size_t>(1, (nsm + n_batch - 1) / n_batch));
            const dim3 blocks((unsigned int) n_batch, (unsigned int) blocks_per_copy);
            // Stream ordering keeps each kernel ahead of the next descriptor overwrite.
            CUDA_CHECK(cudaMemcpyAsync(device_copies, copies, n_batch * sizeof(copies[0]), cudaMemcpyHostToDevice, cudaStreamPerThread));
            ggml_cuda_batch_set_kernel<<<blocks, threads, 0, cudaStreamPerThread>>>(device_copies, n_batch);
            CUDA_CHECK(cudaGetLastError());
            if (first + n_batch < n_reads_scatter) {
                // Wait before refilling the pinned host descriptor buffer.
                CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
            }
        }
    }
    // Keep the source pinned for DMA too; overlapping destinations already declined above.
    for (size_t i = 0; i < n_tensors; ++i) {
        if (sizes[i] && (!n_reads_scatter || sizes[i] > max_scatter_read_bytes)) {
            CUDA_CHECK(cudaMemcpyAsync((char *) tensors[i]->data + offsets[i], data[i], sizes[i], cudaMemcpyHostToDevice, cudaStreamPerThread));
        }
    }
    // Wait for GPU reads before releasing the CPU mapping and descriptor buffers.
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    return true;
#endif
}
