#include "common.cuh"

// the mailbox ops (ggml-mailbox.h): GGML_OP_CUSTOM nodes that move a layer's data to and from page-locked host memory
bool ggml_cuda_mailbox_op_supported(const ggml_tensor * op);

void ggml_cuda_op_mailbox(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
