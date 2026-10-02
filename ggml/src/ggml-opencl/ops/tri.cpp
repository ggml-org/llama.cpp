#include "../cl-common.h"
#include "../ops.h"

void ggml_cl_load_kernels_tri(ggml_backend_opencl_context * backend_ctx) {
    cl_int err;
    const std::string & compile_opts = backend_ctx->kernel_compile_opts;
    // tri
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "tri.cl.h"
        };
#else
        const std::string kernel_src = read_file("tri.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->tri.kernel_tri = clCreateKernel(prog, "kernel_tri_f32", &err), err));
        GGML_LOG_CONT(".");

        CL_CHECK(clReleaseProgram(prog));
    }
}

void ggml_cl_tri(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
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

    const int tri_type = ggml_get_op_params_i32(dst, 0);
    const int64_t n = ggml_nelements(dst);
    const int     ne0  = dst->ne[0];
    const int     ne1  = dst->ne[1];

    cl_kernel kernel = backend_ctx->tri.kernel_tri;

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int),      &n));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int),      &ne0));
    CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int),      &ne1));
    CL_CHECK(clSetKernelArg(kernel, 7, sizeof(int),      &tri_type));

    size_t local_work_size[1] = { 256 };
    size_t global_work_size[1] = { ((size_t)n + local_work_size[0] - 1) / local_work_size[0] * local_work_size[0] };

    backend_ctx->enqueue_ndrange_kernel(kernel, 1, global_work_size, local_work_size, dst);
}
