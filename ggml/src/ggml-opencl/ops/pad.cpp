#include "../cl-common.h"
#include "../ops.h"

void ggml_cl_load_kernels_pad(ggml_backend_opencl_context * backend_ctx) {
    cl_int err;
    const std::string & compile_opts = backend_ctx->kernel_compile_opts;
    // pad
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "pad.cl.h"
        };
#else
        const std::string kernel_src = read_file("pad.cl");
#endif
        if (!kernel_src.empty()) {
            cl_program prog =
                build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);
            CL_CHECK((backend_ctx->pad.kernel_pad = clCreateKernel(prog, "kernel_pad", &err), err));
            CL_CHECK(clReleaseProgram(prog));
            GGML_LOG_CONT(".");
        } else {
            GGML_LOG_WARN("ggml_opencl: pad kernel source not found or empty. Pad operations will not be available.\n");
            backend_ctx->pad.kernel_pad = nullptr;
        }
    }
}

void ggml_cl_pad(ggml_backend_t backend, const ggml_tensor * src0, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    if (backend_ctx->pad.kernel_pad == nullptr) {
        GGML_LOG_WARN("%s: pad kernel not available, skipping OpenCL execution.\n", __func__);
        return;
    }

    ggml_tensor_extra_cl * extra_src0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra_dst  = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong off_src0 = extra_src0->offset + src0->view_offs;
    cl_ulong off_dst  = extra_dst->offset  + dst->view_offs;

    const int s_ne0 = src0->ne[0];
    const int s_ne1 = src0->ne[1];
    const int s_ne2 = src0->ne[2];
    const int s_ne3 = src0->ne[3];

    const int s_nb0 = src0->nb[0];
    const int s_nb1 = src0->nb[1];
    const int s_nb2 = src0->nb[2];
    const int s_nb3 = src0->nb[3];

    const int d_ne0 = dst->ne[0];
    const int d_ne1 = dst->ne[1];
    const int d_ne2 = dst->ne[2];
    const int d_ne3 = dst->ne[3];

    const int d_nb0 = dst->nb[0];
    const int d_nb1 = dst->nb[1];
    const int d_nb2 = dst->nb[2];
    const int d_nb3 = dst->nb[3];

    const int lp0 = ((const int*)(dst->op_params))[0];
    const int rp0 = ((const int*)(dst->op_params))[1];
    const int lp1 = ((const int*)(dst->op_params))[2];
    const int rp1 = ((const int*)(dst->op_params))[3];
    const int lp2 = ((const int*)(dst->op_params))[4];
    const int rp2 = ((const int*)(dst->op_params))[5];
    const int lp3 = ((const int*)(dst->op_params))[6];
    const int rp3 = ((const int*)(dst->op_params))[7];

    cl_kernel kernel = backend_ctx->pad.kernel_pad;

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),    &extra_src0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong),  &off_src0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),    &extra_dst->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong),  &off_dst));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(int),       &s_ne0));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(int),       &s_ne1));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),       &s_ne2));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),       &s_ne3));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong),  &s_nb0));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong),  &s_nb1));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong),  &s_nb2));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong),  &s_nb3));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),       &d_ne0));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),       &d_ne1));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),       &d_ne2));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),       &d_ne3));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong),  &d_nb0));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong),  &d_nb1));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong),  &d_nb2));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong),  &d_nb3));
    CL_CHECK(clSetKernelArg(kernel, 20, sizeof(int),       &lp0));
    CL_CHECK(clSetKernelArg(kernel, 21, sizeof(int),       &rp0));
    CL_CHECK(clSetKernelArg(kernel, 22, sizeof(int),       &lp1));
    CL_CHECK(clSetKernelArg(kernel, 23, sizeof(int),       &rp1));
    CL_CHECK(clSetKernelArg(kernel, 24, sizeof(int),       &lp2));
    CL_CHECK(clSetKernelArg(kernel, 25, sizeof(int),       &rp2));
    CL_CHECK(clSetKernelArg(kernel, 26, sizeof(int),       &lp3));
    CL_CHECK(clSetKernelArg(kernel, 27, sizeof(int),       &rp3));

    size_t lws0 = 64;
    size_t gws0 = (( (size_t)d_ne0 + lws0 - 1 ) / lws0) * lws0;

    size_t global_work_size[] = { gws0, (size_t)d_ne1, (size_t)d_ne2*d_ne3 };
    size_t local_work_size[]  = { lws0, 1, 1 };

    size_t * local_work_size_ptr = local_work_size;
    if (d_ne0 % lws0 != 0 && !backend_ctx->non_uniform_workgroups) {
        local_work_size_ptr = nullptr;
    }

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size_ptr, dst);
}
