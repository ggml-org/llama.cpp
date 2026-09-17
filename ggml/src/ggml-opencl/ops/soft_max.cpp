#include "../cl-common.h"
#include "../ops.h"

void ggml_cl_load_kernels_soft_max(ggml_backend_opencl_context * backend_ctx) {
    cl_int err;
    const std::string & compile_opts = backend_ctx->kernel_compile_opts;
    // softmax_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "softmax_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("softmax_f32.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->soft_max.kernel_soft_max = clCreateKernel(prog, "kernel_soft_max", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // softmax_f16
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "softmax_f16.cl.h"
        };
#else
        const std::string kernel_src = read_file("softmax_f16.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->soft_max.kernel_soft_max_f16 = clCreateKernel(prog, "kernel_soft_max_f16", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // softmax_4_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "softmax_4_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("softmax_4_f32.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->soft_max.kernel_soft_max_4 = clCreateKernel(prog, "kernel_soft_max_4", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // softmax_4_f16
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "softmax_4_f16.cl.h"
        };
#else
        const std::string kernel_src = read_file("softmax_4_f16.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->soft_max.kernel_soft_max_4_f16 = clCreateKernel(prog, "kernel_soft_max_4_f16", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }
}

void ggml_cl_soft_max(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    // Softmax can now fuse KQ mask and KQ scale, which used to be two additional
    // ops before softmax. It now also fuses alibi if `max_bias > 0`. For llama,
    // alibi is not used; however, for some other models, it is used.
    // KQ_mask
    if (src1) {
        GGML_ASSERT(src1);
        GGML_ASSERT(src1->extra);
    }

    const ggml_tensor * src2 = dst->src[2];
    if (src2) {
        GGML_ASSERT(src2->extra);
    }

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    ggml_tensor_extra_cl * extra1 = src1 ? (ggml_tensor_extra_cl *)src1->extra : nullptr;
    ggml_tensor_extra_cl * extra2 = src2 ? (ggml_tensor_extra_cl *)src2->extra : nullptr;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    cl_ulong offset1 = extra1 ? extra1->offset + src1->view_offs : offset0;
    cl_ulong offset2 = extra2 ? extra2->offset + src2->view_offs : offset0;

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const cl_long nb01 = src0->nb[1];
    const cl_long nb02 = src0->nb[2];
    const cl_long nb03 = src0->nb[3];

    const int ne12 = src1 ? src1->ne[2] : 0;
    const int ne13 = src1 ? src1->ne[3] : 0;

    const cl_long nb11 = src1 ? src1->nb[1] : 0;
    const cl_long nb12 = src1 ? src1->nb[2] : 0;
    const cl_long nb13 = src1 ? src1->nb[3] : 0;

    const cl_long nb1 = dst->nb[1];
    const cl_long nb2 = dst->nb[2];
    const cl_long nb3 = dst->nb[3];

    float scale, max_bias;
    memcpy(&scale,    dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, dst->op_params + 1, sizeof(float));

    const int n_head      = src0->ne[2];
    const int n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    const bool use_f16 = (src1 && src1->type == GGML_TYPE_F16);

    // Local size must be wave size. Each workgroup is a wave, working on a row,
    // where a row corresponds to leading dimension.
    int nth = MIN(32, ne00);

    if (backend_ctx->gpu_family == INTEL) {
        // This is the same as the initial value.
        nth = MIN(32, ne00);
    }
    else if (backend_ctx->gpu_family == ADRENO) {
        nth = 64;
    } else {
        GGML_ASSERT(false && "TODO: Unknown GPU");
    }

    cl_kernel kernel;

    if (ne00%4 == 0) {
        if (use_f16) {
            kernel = backend_ctx->soft_max.kernel_soft_max_4_f16;
        } else {
            kernel = backend_ctx->soft_max.kernel_soft_max_4;
        }
    } else {
        if (use_f16) {
            kernel = backend_ctx->soft_max.kernel_soft_max_f16;
        } else {
            kernel = backend_ctx->soft_max.kernel_soft_max;
        }
    }

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   extra1 ? &extra1->data_device : &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   extra2 ? &extra2->data_device : &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offset2));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb02));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb03));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne12));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne13));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &nb11));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(cl_ulong), &nb12));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nb13));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong), &nb1));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong), &nb2));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &nb3));
    CL_CHECK(clSetKernelArg(kernel, 20, sizeof(float),    &scale));
    CL_CHECK(clSetKernelArg(kernel, 21, sizeof(float),    &max_bias));
    CL_CHECK(clSetKernelArg(kernel, 22, sizeof(float),    &m0));
    CL_CHECK(clSetKernelArg(kernel, 23, sizeof(float),    &m1));
    CL_CHECK(clSetKernelArg(kernel, 24, sizeof(int),      &n_head_log2));

    size_t global_work_size[] = {(size_t)ne01*nth, (size_t)ne02, (size_t)ne03};
    size_t local_work_size[] = {(size_t)nth, 1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}
