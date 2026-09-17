#include "../cl-common.h"
#include "../ops.h"

void ggml_cl_load_kernels_relu(ggml_backend_opencl_context * backend_ctx) {
    cl_int err;
    const std::string & compile_opts = backend_ctx->kernel_compile_opts;
    // relu
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "relu.cl.h"
        };
#else
        const std::string kernel_src = read_file("relu.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->relu.kernel_relu = clCreateKernel(prog, "kernel_relu", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }
}

void ggml_cl_relu(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    UNUSED(src1);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    cl_kernel kernel = backend_ctx->relu.kernel_relu;

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offsetd));

    const int64_t n = ggml_nelements(dst);

    size_t global_work_size[] = {(size_t)n, 1, 1};
    size_t local_work_size[] = {64, 1, 1};

    size_t * local_work_size_ptr = local_work_size;
    if (n % 64 != 0 && !backend_ctx->non_uniform_workgroups) {
        local_work_size_ptr = nullptr;  // Let driver choose the work-group sizes.
    }

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size_ptr, dst);
}
