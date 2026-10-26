#include "../cl-common.h"
#include "../ops.h"

void ggml_cl_load_kernels_unary_ext(ggml_backend_opencl_context * backend_ctx) {
    cl_int err;
    const std::string & compile_opts = backend_ctx->kernel_compile_opts;
    // unary_ext (sgn, step, elu, hardswish, hardsigmoid, floor, ceil, round, trunc)
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "unary_ext.cl.h"
        };
#else
        const std::string kernel_src = read_file("unary_ext.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);
#define CL_UNARY_EXT_K(op) \
        CL_CHECK((backend_ctx->unary_ext.kernel_##op##_f32    = clCreateKernel(prog, "kernel_" #op "_f32",    &err), err)); \
        CL_CHECK((backend_ctx->unary_ext.kernel_##op##_f32_4  = clCreateKernel(prog, "kernel_" #op "_f32_4",  &err), err)); \
        CL_CHECK((backend_ctx->unary_ext.kernel_##op##_f32_nc = clCreateKernel(prog, "kernel_" #op "_f32_nc", &err), err)); \
        CL_CHECK((backend_ctx->unary_ext.kernel_##op##_f16    = clCreateKernel(prog, "kernel_" #op "_f16",    &err), err)); \
        CL_CHECK((backend_ctx->unary_ext.kernel_##op##_f16_4  = clCreateKernel(prog, "kernel_" #op "_f16_4",  &err), err)); \
        CL_CHECK((backend_ctx->unary_ext.kernel_##op##_f16_nc = clCreateKernel(prog, "kernel_" #op "_f16_nc", &err), err));
        CL_UNARY_EXT_K(sgn)
        CL_UNARY_EXT_K(step)
        CL_UNARY_EXT_K(elu)
        CL_UNARY_EXT_K(hardswish)
        CL_UNARY_EXT_K(hardsigmoid)
        CL_UNARY_EXT_K(floor)
        CL_UNARY_EXT_K(ceil)
        CL_UNARY_EXT_K(round)
        CL_UNARY_EXT_K(trunc)
#undef CL_UNARY_EXT_K
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

}

// Shared driver for the extended unary ops (unary_ext.cl), same selection as
// ggml_cl_abs: contiguous picks the vec4 kernel when the element count is a
// multiple of 4 (else scalar); non-contiguous uses the stride-addressed kernel.
static void ggml_cl_unary_ext(ggml_backend_t backend, const ggml_tensor * src0, ggml_tensor * dst,
                              cl_kernel k_f32, cl_kernel k_f32_4, cl_kernel k_f32_nc,
                              cl_kernel k_f16, cl_kernel k_f16_4, cl_kernel k_f16_nc) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    const int     ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
    const cl_ulong nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    const cl_ulong nb0 = dst->nb[0], nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    const bool is_f16 = (src0->type == GGML_TYPE_F16);
    cl_kernel kernel;

    if (ggml_is_contiguous(src0)) {
        int n = ggml_nelements(dst);
        if (n % 4 == 0) {
            kernel = is_f16 ? k_f16_4 : k_f32_4;
            n /= 4;
        } else {
            kernel = is_f16 ? k_f16 : k_f32;
        }

        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extrad->data_device));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offsetd));

        size_t global_work_size[] = {(size_t)n, 1, 1};
        size_t local_work_size[]  = {64, 1, 1};
        size_t * local_work_size_ptr = local_work_size;
        if (n % 64 != 0 && !backend_ctx->non_uniform_workgroups) {
            local_work_size_ptr = nullptr;
        }
        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size_ptr, dst);
    } else {
        kernel = is_f16 ? k_f16_nc : k_f32_nc;

        CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
        CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
        CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extrad->data_device));
        CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offsetd));
        CL_CHECK(clSetKernelArg(kernel,  4, sizeof(int),      &ne00));
        CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &nb00));
        CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_ulong), &nb01));
        CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &nb02));
        CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &nb03));
        CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb0));
        CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb1));
        CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb2));
        CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb3));

        int nth = 64;
        size_t global_work_size[] = {(size_t)ne01*nth, (size_t)ne02, (size_t)ne03};
        size_t local_work_size[]  = {(size_t)nth, 1, 1};
        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
    }
}

#define GGML_CL_UNARY_EXT_WRAP(FN, OP)                                                                 \
void FN(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) { \
    UNUSED(src1);                                                                                      \
    ggml_backend_opencl_context *c = (ggml_backend_opencl_context *)backend->context;                  \
    ggml_cl_unary_ext(backend, src0, dst, c->unary_ext.kernel_##OP##_f32, c->unary_ext.kernel_##OP##_f32_4,      \
                      c->unary_ext.kernel_##OP##_f32_nc, c->unary_ext.kernel_##OP##_f16,                         \
                      c->unary_ext.kernel_##OP##_f16_4, c->unary_ext.kernel_##OP##_f16_nc);                      \
}

GGML_CL_UNARY_EXT_WRAP(ggml_cl_sgn,         sgn)
GGML_CL_UNARY_EXT_WRAP(ggml_cl_step,        step)
GGML_CL_UNARY_EXT_WRAP(ggml_cl_elu,         elu)
GGML_CL_UNARY_EXT_WRAP(ggml_cl_hardswish,   hardswish)
GGML_CL_UNARY_EXT_WRAP(ggml_cl_hardsigmoid, hardsigmoid)
GGML_CL_UNARY_EXT_WRAP(ggml_cl_floor,       floor)
GGML_CL_UNARY_EXT_WRAP(ggml_cl_ceil,        ceil)
GGML_CL_UNARY_EXT_WRAP(ggml_cl_round,       round)
GGML_CL_UNARY_EXT_WRAP(ggml_cl_trunc,       trunc)

#undef GGML_CL_UNARY_EXT_WRAP
