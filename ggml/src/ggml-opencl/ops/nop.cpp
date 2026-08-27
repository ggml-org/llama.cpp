#include "../cl-common.h"
#include "../ops.h"

void ggml_cl_load_kernels_nop(ggml_backend_opencl_context * backend_ctx) {
    cl_int err;
    const std::string & compile_opts = backend_ctx->kernel_compile_opts;
    GGML_UNUSED(backend_ctx);
    GGML_UNUSED(err);
    GGML_UNUSED(compile_opts);
}

void ggml_cl_nop(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    UNUSED(backend);
    UNUSED(src0);
    UNUSED(src1);
    UNUSED(dst);
}
