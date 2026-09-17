#include "../cl-common.h"
#include "../ops.h"

void ggml_cl_load_kernels_glu(ggml_backend_opencl_context * backend_ctx) {
    cl_int err;
    const std::string & compile_opts = backend_ctx->kernel_compile_opts;
    // glu
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "glu.cl.h"
        };
#else
        const std::string kernel_src = read_file("glu.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->glu.kernel_geglu           = clCreateKernel(prog, "kernel_geglu", &err), err));
        CL_CHECK((backend_ctx->glu.kernel_reglu           = clCreateKernel(prog, "kernel_reglu", &err), err));
        CL_CHECK((backend_ctx->glu.kernel_swiglu          = clCreateKernel(prog, "kernel_swiglu", &err), err));
        CL_CHECK((backend_ctx->glu.kernel_swiglu_oai      = clCreateKernel(prog, "kernel_swiglu_oai", &err), err));
        CL_CHECK((backend_ctx->glu.kernel_swiglu_clamp    = clCreateKernel(prog, "kernel_swiglu_clamp", &err), err));
        CL_CHECK((backend_ctx->glu.kernel_geglu_erf       = clCreateKernel(prog, "kernel_geglu_erf", &err), err));
        CL_CHECK((backend_ctx->glu.kernel_geglu_quick     = clCreateKernel(prog, "kernel_geglu_quick", &err), err));
        CL_CHECK((backend_ctx->glu.kernel_geglu_f16       = clCreateKernel(prog, "kernel_geglu_f16", &err), err));
        CL_CHECK((backend_ctx->glu.kernel_reglu_f16       = clCreateKernel(prog, "kernel_reglu_f16", &err), err));
        CL_CHECK((backend_ctx->glu.kernel_swiglu_f16      = clCreateKernel(prog, "kernel_swiglu_f16", &err), err));
        CL_CHECK((backend_ctx->glu.kernel_swiglu_clamp_f16 = clCreateKernel(prog, "kernel_swiglu_clamp_f16", &err), err));
        CL_CHECK((backend_ctx->glu.kernel_geglu_erf_f16   = clCreateKernel(prog, "kernel_geglu_erf_f16", &err), err));
        CL_CHECK((backend_ctx->glu.kernel_geglu_quick_f16 = clCreateKernel(prog, "kernel_geglu_quick_f16", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }
}

void ggml_cl_glu(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    GGML_ASSERT(ggml_is_contiguous_1(src0));

    if (src1) {
        GGML_ASSERT(src1);
        GGML_ASSERT(src1->extra);
        GGML_ASSERT(ggml_are_same_shape(src0, src1));
    }

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    cl_kernel kernel;
    switch (ggml_get_glu_op(dst)) {
        case GGML_GLU_OP_GEGLU:
            if (dst->type == GGML_TYPE_F32) {
                kernel = backend_ctx->glu.kernel_geglu;
            } else {
                kernel = backend_ctx->glu.kernel_geglu_f16;
            }
            break;
        case GGML_GLU_OP_REGLU:
            if (dst->type == GGML_TYPE_F32) {
                kernel = backend_ctx->glu.kernel_reglu;
            } else {
                kernel = backend_ctx->glu.kernel_reglu_f16;
            }
            break;
        case GGML_GLU_OP_SWIGLU:
            if (dst->type == GGML_TYPE_F32) {
                kernel = backend_ctx->glu.kernel_swiglu;
            } else {
                kernel = backend_ctx->glu.kernel_swiglu_f16;
            }
            break;
        case GGML_GLU_OP_SWIGLU_OAI:
            kernel = backend_ctx->glu.kernel_swiglu_oai;
            break;
        case GGML_GLU_OP_SWIGLU_CLAMP:
            if (dst->type == GGML_TYPE_F32) {
                kernel = backend_ctx->glu.kernel_swiglu_clamp;
            } else {
                kernel = backend_ctx->glu.kernel_swiglu_clamp_f16;
            }
            break;
        case GGML_GLU_OP_GEGLU_ERF:
            if (dst->type == GGML_TYPE_F32) {
                kernel = backend_ctx->glu.kernel_geglu_erf;
            } else {
                kernel = backend_ctx->glu.kernel_geglu_erf_f16;
            }
            break;
        case GGML_GLU_OP_GEGLU_QUICK:
            if (dst->type == GGML_TYPE_F32) {
                kernel = backend_ctx->glu.kernel_geglu_quick;
            } else {
                kernel = backend_ctx->glu.kernel_geglu_quick_f16;
            }
            break;
        default:
            GGML_ABORT("Unsupported glu op");
    }

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    ggml_tensor_extra_cl * extra1 = src1 ? (ggml_tensor_extra_cl *)src1->extra : nullptr;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    cl_ulong offset1 = extra1 ? extra1->offset + src1->view_offs : offset0;

    const int ne0       = dst->ne[0];

    const cl_ulong nb01 = src0->nb[1];
    const cl_ulong nb11 = src1 ? src1->nb[1] : nb01;

    const cl_ulong nb1  = dst->nb[1];

    const int   swp   = ggml_get_op_params_i32(dst, 1);
    const float alpha = ggml_get_op_params_f32(dst, 2);
    const float limit = ggml_get_op_params_f32(dst, 3);

    const int ne00_off = src1 ? 0 : (swp ? ne0 : 0);
    const int ne10_off = src1 ? 0 : (swp ? 0 : ne0);

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   src1 ? &extra1->data_device : &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &nb11));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne0));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb1));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne00_off));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne10_off));

    if (ggml_get_glu_op(dst) == GGML_GLU_OP_SWIGLU_OAI || ggml_get_glu_op(dst) == GGML_GLU_OP_SWIGLU_CLAMP) {
        CL_CHECK(clSetKernelArg(kernel, 12, sizeof(float), &limit));
    }
    if (ggml_get_glu_op(dst) == GGML_GLU_OP_SWIGLU_OAI) {
        CL_CHECK(clSetKernelArg(kernel, 13, sizeof(float), &alpha));
    }

    const size_t nrows = ggml_nrows(src0);
    size_t nth = backend_ctx->max_workgroup_size < 512 ? backend_ctx->max_workgroup_size : 512;
    size_t global_work_size[] = {nrows*nth, 1, 1};
    size_t local_work_size[] = {nth, 1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}
