#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

// A kernel of the user in a GGML_OP_CUSTOM node:
//   ggml_custom_4d(ctx, type, ne0, ne1, ne2, ne3, args, n_args, fun, 1, kernel)
// fun is ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_kernel_op") (NULL if not supported), kernel is a ggml_cuda_kernel
// The backend launches: name(dst->data, args[0]->data, ..., args[n_args - 1]->data, *params)
//
// - the kernel and all it points to must outlive the graph, with CUDA graphs its values are captured
// - the tensors must be contiguous
// - the kernels can read and write the memory of ggml_backend_cuda_host_buffer_type() at its host address
// - a kernel on a side stream runs next to the nodes that follow, until a kernel with join
//   what it reads and writes is valid for the other nodes after the join, the backend keeps its tensors allocated until then
struct ggml_cuda_kernel {
    const char * image;    // module of the kernel as output by nvcc: PTX (NUL-terminated), cubin or fatbin
    const char * name;     // name of the kernel in the module (declare it extern "C")
    unsigned int grid[3];
    unsigned int block[3];
    unsigned int smem;     // bytes of dynamic shared memory
    const void * params;   // last argument of the kernel, a struct passed by value, NULL for none
    int          stream;   // 0: the stream of the graph, 1..3: a side stream
    bool         join;     // stream 0 only: wait for the kernels on the side streams first
};

// MUL_MAT_ID skips negative ids (their rows of the result are zero) for up to this many tokens with weights of this type, 0 for none
// TODO: [TAG_MUL_MAT_ID_SKIP] remove once all the MUL_MAT_ID kernels skip negative ids
GGML_BACKEND_API int ggml_backend_cuda_mul_mat_id_skip_max_batch(ggml_backend_t backend, enum ggml_type type);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

#ifdef  __cplusplus
}
#endif
