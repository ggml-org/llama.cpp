#include "../cl-common.h"
#include "../ops.h"

void ggml_cl_load_kernels_im2col(ggml_backend_opencl_context * backend_ctx) {
    cl_int err;
    const std::string & compile_opts = backend_ctx->kernel_compile_opts;
    // im2col_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "im2col_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("im2col_f32.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->im2col.kernel_im2col_f32 = clCreateKernel(prog, "kernel_im2col_f32", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // im2col_f16
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "im2col_f16.cl.h"
        };
#else
        const std::string kernel_src = read_file("im2col_f16.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->im2col.kernel_im2col_f16 = clCreateKernel(prog, "kernel_im2col_f16", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }
}

void ggml_cl_im2col(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    // src0 - filter, src1 - input
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    const int32_t s0 = ((const int32_t*)(dst->op_params))[0];
    const int32_t s1 = ((const int32_t*)(dst->op_params))[1];
    const int32_t p0 = ((const int32_t*)(dst->op_params))[2];
    const int32_t p1 = ((const int32_t*)(dst->op_params))[3];
    const int32_t d0 = ((const int32_t*)(dst->op_params))[4];
    const int32_t d1 = ((const int32_t*)(dst->op_params))[5];

    const bool is_2D = ((const int32_t*)(dst->op_params))[6] == 1;

    const cl_long IC = src1->ne[is_2D ? 2 : 1];
    const cl_long IH = is_2D ? src1->ne[1] : 1;
    const cl_long IW =         src1->ne[0];

    const cl_long KH = is_2D ? src0->ne[1] : 1;
    const cl_long KW =         src0->ne[0];

    const cl_long OH = is_2D ? dst->ne[2] : 1;
    const cl_long OW =         dst->ne[1];

    // nb is byte offset, src is type float32
    const cl_ulong delta_offset = src1->nb[is_2D ? 2 : 1]/4;
    const cl_long  batch        = src1->ne[is_2D ? 3 : 2];
    const cl_ulong batch_offset = src1->nb[is_2D ? 3 : 2]/4;

    const cl_long pelements = OW*KW*KH;
    const cl_long CHW       = IC*KH*KW;

    cl_kernel kernel;

    if(dst->type == GGML_TYPE_F16) {
        kernel = backend_ctx->im2col.kernel_im2col_f16;
    } else {
        kernel = backend_ctx->im2col.kernel_im2col_f32;
    }

    CL_CHECK(clSetKernelArg(kernel,   0, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,   1, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,   2, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,   3, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,   4, sizeof(cl_ulong), &batch_offset));
    CL_CHECK(clSetKernelArg(kernel,   5, sizeof(cl_ulong), &delta_offset));
    CL_CHECK(clSetKernelArg(kernel,   6, sizeof(cl_long),  &IW));
    CL_CHECK(clSetKernelArg(kernel,   7, sizeof(cl_long),  &IH));
    CL_CHECK(clSetKernelArg(kernel,   8, sizeof(cl_long),  &IC));
    CL_CHECK(clSetKernelArg(kernel,   9, sizeof(cl_long),  &OW));
    CL_CHECK(clSetKernelArg(kernel,  10, sizeof(cl_long),  &OH));
    CL_CHECK(clSetKernelArg(kernel,  11, sizeof(cl_long),  &KW));
    CL_CHECK(clSetKernelArg(kernel,  12, sizeof(cl_long),  &KH));
    CL_CHECK(clSetKernelArg(kernel,  13, sizeof(cl_long),  &pelements));
    CL_CHECK(clSetKernelArg(kernel,  14, sizeof(cl_long),  &CHW));
    CL_CHECK(clSetKernelArg(kernel,  15, sizeof(int),      &s0));
    CL_CHECK(clSetKernelArg(kernel,  16, sizeof(int),      &s1));
    CL_CHECK(clSetKernelArg(kernel,  17, sizeof(int),      &p0));
    CL_CHECK(clSetKernelArg(kernel,  18, sizeof(int),      &p1));
    CL_CHECK(clSetKernelArg(kernel,  19, sizeof(int),      &d0));
    CL_CHECK(clSetKernelArg(kernel,  20, sizeof(int),      &d1));

    const int num_blocks = (pelements + 256 - 1) / 256;
    size_t global_work_size[] = {(size_t)num_blocks*256, (size_t)OH, (size_t)batch*IC};
    size_t local_work_size[] = {256, 1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}
