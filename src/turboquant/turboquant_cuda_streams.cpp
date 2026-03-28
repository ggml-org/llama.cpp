// TurboQuant CUDA Stream Implementation
// Async operations and stream management

#include "turboquant_cuda_streams.h"
#include <iostream>

namespace turboquant {
namespace cuda {

// Stream pool implementation
StreamPool::StreamPool(int num_streams) : current_stream_(0) {
    streams_.reserve(num_streams);
    for (int i = 0; i < num_streams; ++i) {
        cudaStream_t stream;
        cudaError_t err = cudaStreamCreate(&stream);
        if (err != cudaSuccess) {
            std::cerr << "Failed to create CUDA stream " << i << ": " 
                      << cudaGetErrorString(err) << std::endl;
            break;
        }
        streams_.push_back(stream);
    }
}

StreamPool::~StreamPool() {
    for (auto& stream : streams_) {
        if (stream) {
            cudaStreamDestroy(stream);
        }
    }
}

cudaStream_t StreamPool::get_stream() {
    std::lock_guard<std::mutex> lock(mutex_);
    cudaStream_t stream = streams_[current_stream_];
    current_stream_ = (current_stream_ + 1) % streams_.size();
    return stream;
}

void StreamPool::synchronize_all() {
    for (auto& stream : streams_) {
        if (stream) {
            cudaStreamSynchronize(stream);
        }
    }
}

// Global stream pool
StreamPool& get_stream_pool() {
    static StreamPool pool(4);
    return pool;
}

// Async memory operations
cudaError_t malloc_async(void** ptr, size_t size, cudaStream_t stream) {
    // Use cudaMalloc for now (cudaMallocAsync requires newer CUDA)
    cudaError_t err = cudaMalloc(ptr, size);
    return err;
}

cudaError_t free_async(void* ptr, cudaStream_t stream) {
    // Use cudaFree for now
    cudaError_t err = cudaFree(ptr);
    return err;
}

cudaError_t memcpy_async(void* dst, const void* src, size_t size, 
                         cudaMemcpyKind kind, cudaStream_t stream) {
    return cudaMemcpyAsync(dst, src, size, kind, stream);
}

cudaError_t memcpy_to_device_async(void* dst, const void* src, size_t size, cudaStream_t stream) {
    cudaError_t err = memcpy_async(dst, src, size, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        return err;
    }
    // Synchronize to ensure data is on device before kernel launch
    // This prevents race conditions in quantization pipelines
    err = cudaStreamSynchronize(stream);
    return err;
}

cudaError_t memcpy_to_host_async(void* dst, const void* src, size_t size, cudaStream_t stream) {
    cudaError_t err = memcpy_async(dst, src, size, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        return err;
    }
    // Synchronize to ensure data is available on host
    // TODO: Make this optional for batched operations where latency matters
    err = cudaStreamSynchronize(stream);
    return err;
}

} // namespace cuda
} // namespace turboquant
