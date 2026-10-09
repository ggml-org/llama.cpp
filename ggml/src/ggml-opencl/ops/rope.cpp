#include "../cl-common.h"
#include "../ops.h"

void ggml_cl_load_kernels_rope(ggml_backend_opencl_context * backend_ctx) {
    cl_int err;
    const std::string & compile_opts = backend_ctx->kernel_compile_opts;
    // rope
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "rope.cl.h"
        };
#else
        const std::string kernel_src = read_file("rope.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->rope.kernel_rope_norm_f32   = clCreateKernel(prog, "kernel_rope_norm_f32", &err), err));
        CL_CHECK((backend_ctx->rope.kernel_rope_norm_f16   = clCreateKernel(prog, "kernel_rope_norm_f16", &err), err));
        CL_CHECK((backend_ctx->rope.kernel_rope_neox_f32   = clCreateKernel(prog, "kernel_rope_neox_f32", &err), err));
        CL_CHECK((backend_ctx->rope.kernel_rope_neox_f16   = clCreateKernel(prog, "kernel_rope_neox_f16", &err), err));
        CL_CHECK((backend_ctx->rope.kernel_rope_multi_f32  = clCreateKernel(prog, "kernel_rope_multi_f32", &err), err));
        CL_CHECK((backend_ctx->rope.kernel_rope_multi_f16  = clCreateKernel(prog, "kernel_rope_multi_f16", &err), err));
        CL_CHECK((backend_ctx->rope.kernel_rope_vision_f32 = clCreateKernel(prog, "kernel_rope_vision_f32", &err), err));
        CL_CHECK((backend_ctx->rope.kernel_rope_vision_f16 = clCreateKernel(prog, "kernel_rope_vision_f16", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }
}

void ggml_cl_rope(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
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

    ggml_tensor * src2 = dst->src[2];
    ggml_tensor_extra_cl * extra2 = src2 ? (ggml_tensor_extra_cl *)src2->extra : nullptr;

    cl_ulong offset2 = extra2 ? extra2->offset + src2->view_offs : offset0;

    const int  ne00 = src0 ? src0->ne[0] : 0;
    const int  ne01 = src0 ? src0->ne[1] : 0;
    const int  ne02 = src0 ? src0->ne[2] : 0;
    const int  ne03 = src0 ? src0->ne[3] : 0;

    const cl_ulong  nb00 = src0 ? src0->nb[0] : 0;
    const cl_ulong  nb01 = src0 ? src0->nb[1] : 0;
    const cl_ulong  nb02 = src0 ? src0->nb[2] : 0;
    const cl_ulong  nb03 = src0 ? src0->nb[3] : 0;

    const int ne10 = src1 ? src1->ne[0] : 0;
    const int ne11 = src1 ? src1->ne[1] : 0; UNUSED(ne11);
    const int ne12 = src1 ? src1->ne[2] : 0; UNUSED(ne12);
    const int ne13 = src1 ? src1->ne[3] : 0; UNUSED(ne13);

    const int  ne0 = dst ? dst->ne[0] : 0;
    const int  ne1 = dst ? dst->ne[1] : 0;
    const int  ne2 = dst ? dst->ne[2] : 0;
    const int  ne3 = dst ? dst->ne[3] : 0;

    const cl_ulong  nb0 = dst ? dst->nb[0] : 0;
    const cl_ulong  nb1 = dst ? dst->nb[1] : 0;
    const cl_ulong  nb2 = dst ? dst->nb[2] : 0;
    const cl_ulong  nb3 = dst ? dst->nb[3] : 0;

    GGML_ASSERT(ne10 % ne02 == 0);
    GGML_ASSERT(ne10 >= ne02);

    int nth = MIN(64, ne00);

    const int n_past     = ((int *) dst->op_params)[0];
    const int n_dims     = ((int *) dst->op_params)[1];
    const int mode       = ((int *) dst->op_params)[2];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];
    const int n_offs     = ((int32_t *) dst->op_params)[15];

    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
    int32_t sections[4];

    memcpy(&freq_base,   (int32_t *) dst->op_params + 5, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params + 6, sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params + 7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params + 8, sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params + 9, sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));
    memcpy(&sections,    (int32_t *) dst->op_params + 11, sizeof(int32_t)*4);

    const bool is_neox = mode & 2;
    const bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;
    const bool is_vision = mode == GGML_ROPE_TYPE_VISION;
    const int  is_imrope = mode == GGML_ROPE_TYPE_IMROPE;

    if (is_mrope) {
        GGML_ASSERT(sections[0] > 0 || sections[1] > 0 || sections[2] > 0);
    }

    if (is_vision) {
        GGML_ASSERT(n_dims == ne00/2);
        GGML_ASSERT(n_offs == 0);
    }

    cl_kernel kernel;

    if (is_neox) {
        switch (src0->type) {
            case GGML_TYPE_F32:
                kernel = backend_ctx->rope.kernel_rope_neox_f32;
                break;
            case GGML_TYPE_F16:
                kernel = backend_ctx->rope.kernel_rope_neox_f16;
                break;
            default:
                GGML_ASSERT(false);
        };
    } else if (is_mrope && !is_vision) {
        switch (src0->type) {
            case GGML_TYPE_F32:
                kernel = backend_ctx->rope.kernel_rope_multi_f32;
                break;
            case GGML_TYPE_F16:
                kernel = backend_ctx->rope.kernel_rope_multi_f16;
                break;
            default:
                GGML_ASSERT(false);
        };
    } else if (is_vision) {
        switch (src0->type) {
            case GGML_TYPE_F32:
                kernel = backend_ctx->rope.kernel_rope_vision_f32;
                break;
            case GGML_TYPE_F16:
                kernel = backend_ctx->rope.kernel_rope_vision_f16;
                break;
            default:
                GGML_ASSERT(false);
        }
    } else {
        switch (src0->type) {
            case GGML_TYPE_F32:
                kernel = backend_ctx->rope.kernel_rope_norm_f32;
                break;
            case GGML_TYPE_F16:
                kernel = backend_ctx->rope.kernel_rope_norm_f16;
                break;
            default:
                GGML_ASSERT(false);
        };
    }

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   extra2 ? &extra2->data_device : &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offset2));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne02));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne03));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb00));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &nb02));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(cl_ulong), &nb03));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &ne0));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &ne1));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &ne2));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(int),      &ne3));
    CL_CHECK(clSetKernelArg(kernel, 20, sizeof(cl_ulong), &nb0));
    CL_CHECK(clSetKernelArg(kernel, 21, sizeof(cl_ulong), &nb1));
    CL_CHECK(clSetKernelArg(kernel, 22, sizeof(cl_ulong), &nb2));
    CL_CHECK(clSetKernelArg(kernel, 23, sizeof(cl_ulong), &nb3));
    CL_CHECK(clSetKernelArg(kernel, 24, sizeof(int),      &n_past));
    CL_CHECK(clSetKernelArg(kernel, 25, sizeof(int),      &n_dims));
    CL_CHECK(clSetKernelArg(kernel, 26, sizeof(int),      &n_ctx_orig));
    CL_CHECK(clSetKernelArg(kernel, 27, sizeof(float),    &freq_base));
    CL_CHECK(clSetKernelArg(kernel, 28, sizeof(float),    &freq_scale));
    CL_CHECK(clSetKernelArg(kernel, 29, sizeof(float),    &ext_factor));
    CL_CHECK(clSetKernelArg(kernel, 30, sizeof(float),    &attn_factor));
    CL_CHECK(clSetKernelArg(kernel, 31, sizeof(float),    &beta_fast));
    CL_CHECK(clSetKernelArg(kernel, 32, sizeof(float),    &beta_slow));
    // both mrope and vision kernels have sections
    if (is_mrope || is_vision) {
        CL_CHECK(clSetKernelArg(kernel, 33, sizeof(int32_t)*4, &sections));
    }
    // only mrope has is_imrope
    if (is_mrope && !is_vision) {
        CL_CHECK(clSetKernelArg(kernel, 34, sizeof(int), &is_imrope));
    }
    if (!is_mrope && !is_vision) {
        CL_CHECK(clSetKernelArg(kernel, 33, sizeof(int), &n_offs));
    } else if (is_mrope && !is_vision) {
        CL_CHECK(clSetKernelArg(kernel, 35, sizeof(int), &n_offs));
    }

    size_t global_work_size[] = {(size_t)ne01*nth, (size_t)ne02, (size_t)ne03};
    size_t local_work_size[] = {(size_t)nth, 1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}
