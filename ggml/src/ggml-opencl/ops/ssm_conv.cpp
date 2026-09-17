#include "../cl-common.h"
#include "../ops.h"

void ggml_cl_load_kernels_ssm_conv(ggml_backend_opencl_context * backend_ctx) {
    cl_int err;
    const std::string & compile_opts = backend_ctx->kernel_compile_opts;
    // ssm_conv
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "ssm_conv.cl.h"
        };
#else
        const std::string kernel_src = read_file("ssm_conv.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->ssm_conv.kernel_ssm_conv_f32_f32   = clCreateKernel(prog, "kernel_ssm_conv_f32_f32", &err), err));
        CL_CHECK((backend_ctx->ssm_conv.kernel_ssm_conv_f32_f32_4 = clCreateKernel(prog, "kernel_ssm_conv_f32_f32_4", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }
}

void ggml_cl_ssm_conv(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    int ne01 = src0->ne[1];
    cl_ulong nb00 = src0->nb[0];
    cl_ulong nb01 = src0->nb[1];
    cl_ulong nb02 = src0->nb[2];

    int ne10 = src1->ne[0];
    cl_ulong nb11 = src1->nb[1];

    int ne1  = dst->ne[1];
    int ne2  = dst->ne[2];
    cl_ulong nb0 = dst->nb[0];
    cl_ulong nb1 = dst->nb[1];
    cl_ulong nb2 = dst->nb[2];

    cl_kernel kernel = backend_ctx->ssm_conv.kernel_ssm_conv_f32_f32;

    if (ne10 % 4 == 0) {
        kernel = backend_ctx->ssm_conv.kernel_ssm_conv_f32_f32_4;
    }

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_ulong), &nb00));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &nb02));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne10));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb11));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb0));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb1));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &nb2));

    size_t global_work_size[] = {(size_t)ne01, (size_t)ne1, (size_t)ne2};
    size_t local_work_size[]  = {64, 1, 1};

    size_t * local_work_size_ptr = local_work_size;
    if (ne01 % 64 != 0 && !backend_ctx->non_uniform_workgroups) {
        local_work_size_ptr = nullptr;
    }

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size_ptr, dst);
}
