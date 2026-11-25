#include "../cl-common.h"
#include "../ops.h"

void ggml_cl_load_kernels_reshape(ggml_backend_opencl_context * backend_ctx) {
    GGML_UNUSED(backend_ctx);
}

void ggml_cl_reshape(
        ggml_backend_t backend, const ggml_tensor * src0,
        const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_cl_nop(backend, src0, src1, dst);
}
