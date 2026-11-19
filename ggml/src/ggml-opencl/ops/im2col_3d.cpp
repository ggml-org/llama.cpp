#include "../cl-common.h"
#include "../ops.h"

void ggml_cl_load_kernels_im2col_3d(ggml_backend_opencl_context * backend_ctx) {
    cl_int err;
    const std::string & compile_opts = backend_ctx->kernel_compile_opts;

    // im2col_3d_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "im2col_3d_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("im2col_3d_f32.cl");
#endif
        cl_program prog = build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->im2col_3d.kernel_im2col_3d_f32 = clCreateKernel(prog, "kernel_im2col_3d_f32", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // im2col_3d_f16
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "im2col_3d_f16.cl.h"
        };
#else
        const std::string kernel_src = read_file("im2col_3d_f16.cl");
#endif
        cl_program prog = build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->im2col_3d.kernel_im2col_3d_f16 = clCreateKernel(prog, "kernel_im2col_3d_f16", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }
}

void ggml_cl_im2col_3d(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32);

    ggml_backend_opencl_context * backend_ctx = (ggml_backend_opencl_context *) backend->context;

    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *) src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *) dst->extra;

    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    const int32_t s0 = ((const int32_t *)(dst->op_params))[0];
    const int32_t s1 = ((const int32_t *)(dst->op_params))[1];
    const int32_t s2 = ((const int32_t *)(dst->op_params))[2];
    const int32_t p0 = ((const int32_t *)(dst->op_params))[3];
    const int32_t p1 = ((const int32_t *)(dst->op_params))[4];
    const int32_t p2 = ((const int32_t *)(dst->op_params))[5];
    const int32_t d0 = ((const int32_t *)(dst->op_params))[6];
    const int32_t d1 = ((const int32_t *)(dst->op_params))[7];
    const int32_t d2 = ((const int32_t *)(dst->op_params))[8];
    const cl_long IC = ((const int32_t *)(dst->op_params))[9];

    const cl_long N  = src1->ne[3] / IC;
    const cl_long ID = src1->ne[2];
    const cl_long IH = src1->ne[1];
    const cl_long IW = src1->ne[0];

    const cl_long KD = src0->ne[2];
    const cl_long KH = src0->ne[1];
    const cl_long KW = src0->ne[0];

    const cl_long OD = dst->ne[3] / N;
    const cl_long OH = dst->ne[2];
    const cl_long OW = dst->ne[1];

    const size_t  es       = ggml_element_size(src1);
    const cl_long stride_x = src1->nb[0] / es;
    const cl_long stride_y = src1->nb[1] / es;
    const cl_long stride_z = src1->nb[2] / es;
    const cl_long stride_q = src1->nb[3] / es;

    const cl_long IC_KD_KH_KW = IC * KD * KH * KW;

    cl_kernel kernel;

    if (dst->type == GGML_TYPE_F16) {
        kernel = backend_ctx->im2col_3d.kernel_im2col_3d_f16;
    } else {
        kernel = backend_ctx->im2col_3d.kernel_im2col_3d_f32;
    }

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_long),  &IC));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_long),  &ID));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_long),  &IH));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_long),  &IW));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_long),  &KD));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_long),  &KH));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_long),  &KW));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_long),  &OD));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_long),  &OH));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_long),  &OW));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_long),  &N));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(cl_long),  &stride_x));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_long),  &stride_y));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_long),  &stride_z));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_long),  &stride_q));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(int),      &s0));
    CL_CHECK(clSetKernelArg(kernel, 20, sizeof(int),      &s1));
    CL_CHECK(clSetKernelArg(kernel, 21, sizeof(int),      &s2));
    CL_CHECK(clSetKernelArg(kernel, 22, sizeof(int),      &p0));
    CL_CHECK(clSetKernelArg(kernel, 23, sizeof(int),      &p1));
    CL_CHECK(clSetKernelArg(kernel, 24, sizeof(int),      &p2));
    CL_CHECK(clSetKernelArg(kernel, 25, sizeof(int),      &d0));
    CL_CHECK(clSetKernelArg(kernel, 26, sizeof(int),      &d1));
    CL_CHECK(clSetKernelArg(kernel, 27, sizeof(int),      &d2));

    const size_t local_col = 64;
    const size_t num_blocks = (IC_KD_KH_KW + local_col - 1) / local_col;
    size_t global_work_size[] = { num_blocks * local_col, (size_t) OW, (size_t) (N * OD * OH) };
    size_t local_work_size[] = { local_col, 1, 1 };

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}
