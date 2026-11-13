#pragma once

#include "common.cuh"

// GGML_OP_CUSTOM nodes that run a kernel of the user, see ggml_cuda_kernel in ggml-cuda.h

#define GGML_CUDA_KERNEL_OP_MAX_SIDE 3

// the fun of the kernel ops, nullptr if not supported
ggml_custom_op_t ggml_cuda_kernel_op_fun();

// op is a kernel op that the device can run
bool ggml_cuda_kernel_op_supported(int device, const ggml_tensor * op);

// the side stream of a kernel op (0 for the stream of the graph, also for other nodes), *join: it waits for the side streams
int ggml_cuda_kernel_op_stream(const ggml_tensor * op, bool * join);

void ggml_cuda_op_kernel(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
