#include "../cl-common.h"
#include "../ops.h"

void ggml_cl_load_kernels_diag_mask_inf(ggml_backend_opencl_context * backend_ctx) {
    cl_int err;
    const std::string & compile_opts = backend_ctx->kernel_compile_opts;
    // diag_mask_inf
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "diag_mask_inf.cl.h"
        };
#else
        const std::string kernel_src = read_file("diag_mask_inf.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->diag_mask_inf.kernel_diag_mask_inf_8 = clCreateKernel(prog, "kernel_diag_mask_inf_8", &err), err));
        CL_CHECK((backend_ctx->diag_mask_inf.kernel_diag_mask_inf   = clCreateKernel(prog, "kernel_diag_mask_inf", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }
}

void ggml_cl_diag_mask_inf(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    UNUSED(src1);

    int n_past = ((int32_t *)(dst->op_params))[0];

    const int  ne00 = src0 ? src0->ne[0] : 0;
    const int  ne01 = src0 ? src0->ne[1] : 0;
    const int  ne02 = src0 ? src0->ne[2] : 0;

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    cl_kernel kernel;

    if (ne00%8 == 0) {
        kernel = backend_ctx->diag_mask_inf.kernel_diag_mask_inf_8;

        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extrad->data_device));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offsetd));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int),      &ne00));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int),      &ne01));
        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int),      &n_past));

        size_t global_work_size[] = {(size_t)ne00*ne01*ne02/8, 1, 1};
        size_t local_work_size[] = {64, 1, 1};

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
    } else {
        kernel = backend_ctx->diag_mask_inf.kernel_diag_mask_inf;

        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extrad->data_device));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offsetd));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int),      &ne00));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int),      &ne01));
        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int),      &n_past));

        size_t global_work_size[] = {(size_t)ne00, (size_t)ne01, (size_t)ne02};
        size_t local_work_size[] = {64, 1, 1};

        size_t * local_work_size_ptr = local_work_size;
        if (ne00 % 64 != 0 && !backend_ctx->non_uniform_workgroups) {
            local_work_size_ptr = nullptr;  // Let driver choose the work-group sizes.
        }

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size_ptr, dst);
    }
}
