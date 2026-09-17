#include "../cl-common.h"
#include "../ops.h"

void ggml_cl_load_kernels_fill(ggml_backend_opencl_context * backend_ctx) {
    cl_int err;
    const std::string & compile_opts = backend_ctx->kernel_compile_opts;
    // fill
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "fill.cl.h"
        };
#else
        const std::string kernel_src = read_file("fill.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->fill.kernel_fill = clCreateKernel(prog, "kernel_fill_f32", &err), err));
        GGML_LOG_CONT(".");

        CL_CHECK(clReleaseProgram(prog));
    }
}

void ggml_cl_fill(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    UNUSED(src0);
    UNUSED(src1);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    float v = 0.0f;
    memcpy(&v, ((int32_t *) dst->op_params), sizeof(float));

    const int64_t n = ggml_nelements(dst);

    cl_kernel kernel = backend_ctx->fill.kernel_fill;

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(float),    &v));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(float),    &n));

    size_t local_work_size[1] = { 256 };
    size_t global_work_size[1] = { ((size_t)n + local_work_size[0] - 1) / local_work_size[0] * local_work_size[0] };

    backend_ctx->enqueue_ndrange_kernel(kernel, 1, global_work_size, local_work_size, dst);
}
