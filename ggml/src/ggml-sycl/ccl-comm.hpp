#pragma once

#include "common.hpp"

#if GGML_SYCL_CCL
#include <oneapi/ccl.hpp>
#endif

//
// oneCCL-based collective communication for tensor parallelism
//
// This provides optimized ALL_REDUCE operations using Intel's oneCCL library,
// which offers better performance than manual host-staged reductions.
//

#if GGML_SYCL_CCL

// CCL context for tensor parallelism
struct ggml_sycl_ccl_context {
    std::vector<ccl::communicator> comms;  // One per device (or one for multi-process)
    std::vector<ccl::stream> streams;       // CCL streams wrapping SYCL queues
    int world_size;
    int rank;  // This process's rank (for multi-process mode)
    bool initialized;

    ggml_sycl_ccl_context() : world_size(0), rank(0), initialized(false) {}
};

// Global CCL context
extern ggml_sycl_ccl_context g_ccl_ctx;

// Initialize CCL for tensor parallelism (single-process multi-device mode)
// Called from ggml_sycl_tp_init() when TP is enabled
// world_size: number of devices in the TP group
// queues: array of SYCL queues, one per device
// NOTE: This is disabled in single-process mode - CCL requires multi-process
void ggml_sycl_ccl_init(int world_size, queue_ptr* queues);

// Initialize CCL for multi-process tensor parallelism
// Called when running under mpirun with one GPU per process
// rank: this process's MPI rank (0 to world_size-1)
// world_size: total number of MPI processes
// queue: this process's SYCL queue (for its assigned GPU)
void ggml_sycl_ccl_init_multiprocess(int rank, int world_size, queue_ptr queue);

// Perform in-place ALL_REDUCE sum on a float buffer
// buf: device buffer to reduce (in-place)
// count: number of float elements
// device: which device the buffer is on
void ggml_sycl_ccl_allreduce_sum_f32(float* buf, size_t count, int device);

// Perform ALL_REDUCE sum with separate send/recv buffers
// send_buf: source buffer on device
// recv_buf: destination buffer on device (can be same as send_buf for in-place)
// count: number of float elements
// device: which device the buffers are on
void ggml_sycl_ccl_allreduce_sum_f32(const float* send_buf, float* recv_buf, size_t count, int device);

// Check if CCL is initialized
bool ggml_sycl_ccl_is_initialized();

// Cleanup CCL resources
void ggml_sycl_ccl_free();

#else // !GGML_SYCL_CCL

// Stub implementations when CCL is not available
inline void ggml_sycl_ccl_init(int world_size, queue_ptr* queues) {
    GGML_UNUSED(world_size);
    GGML_UNUSED(queues);
}

inline void ggml_sycl_ccl_init_multiprocess(int rank, int world_size, queue_ptr queue) {
    GGML_UNUSED(rank);
    GGML_UNUSED(world_size);
    GGML_UNUSED(queue);
    fprintf(stderr, "SYCL CCL: Multi-process CCL not available (compiled without oneCCL)\n");
}

inline void ggml_sycl_ccl_allreduce_sum_f32(float* buf, size_t count, int device) {
    GGML_UNUSED(buf);
    GGML_UNUSED(count);
    GGML_UNUSED(device);
}

inline void ggml_sycl_ccl_allreduce_sum_f32(const float* send_buf, float* recv_buf, size_t count, int device) {
    GGML_UNUSED(send_buf);
    GGML_UNUSED(recv_buf);
    GGML_UNUSED(count);
    GGML_UNUSED(device);
}

inline bool ggml_sycl_ccl_is_initialized() { return false; }

inline void ggml_sycl_ccl_free() {}

#endif // GGML_SYCL_CCL
