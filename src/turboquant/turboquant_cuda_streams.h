// TurboQuant CUDA Stream Management
// Async operations and stream pool for GPU optimization

#ifndef TURBOQUANT_CUDA_STREAMS_H
#define TURBOQUANT_CUDA_STREAMS_H

#include <cuda_runtime.h>
#include <vector>
#include <mutex>

namespace turboquant {
namespace cuda {

// Stream pool for parallel execution
class StreamPool {
public:
    StreamPool(int num_streams = 4);
    ~StreamPool();
    
    // Get next available stream (round-robin)
    cudaStream_t get_stream();
    
    // Synchronize all streams
    void synchronize_all();
    
    // Get number of streams
    int get_num_streams() const { return streams_.size(); }
    
private:
    std::vector<cudaStream_t> streams_;
    int current_stream_;
    std::mutex mutex_;
};

// Global stream pool instance
StreamPool& get_stream_pool();

// Async memory operations
cudaError_t malloc_async(void** ptr, size_t size, cudaStream_t stream = 0);
cudaError_t free_async(void* ptr, cudaStream_t stream = 0);
cudaError_t memcpy_async(void* dst, const void* src, size_t size, 
                         cudaMemcpyKind kind, cudaStream_t stream = 0);

// Convenience functions
cudaError_t memcpy_to_device_async(void* dst, const void* src, size_t size, cudaStream_t stream = 0);
cudaError_t memcpy_to_host_async(void* dst, const void* src, size_t size, cudaStream_t stream = 0);

} // namespace cuda
} // namespace turboquant

#endif // TURBOQUANT_CUDA_STREAMS_H
