#include "../cl-common.h"
#include "../ops.h"

void ggml_cl_load_kernels_set(ggml_backend_opencl_context * backend_ctx) {
    cl_int err;
    const std::string & compile_opts = backend_ctx->kernel_compile_opts;
    GGML_UNUSED(backend_ctx);
    GGML_UNUSED(err);
    GGML_UNUSED(compile_opts);
}

void ggml_cl_set(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    GGML_ASSERT((src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_I32) &&
        src1->type == src0->type && dst->type == src0->type);

    GGML_TENSOR_LOCALS(int,      ne0, src0, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb0, src0, nb);
    GGML_TENSOR_LOCALS(int,      ne1, src1, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb1, src1, nb);
    GGML_TENSOR_LOCALS(int,      ne,  dst,  ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb,  dst,  nb);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    const cl_ulong pnb1    = ((const int32_t *)dst->op_params)[0];
    const cl_ulong pnb2    = ((const int32_t *)dst->op_params)[1];
    const cl_ulong pnb3    = ((const int32_t *)dst->op_params)[2];
    const cl_ulong offs    = ((const int32_t *)dst->op_params)[3];
    const bool     inplace = (bool)((const int32_t *)dst->op_params)[4];

    cl_kernel kernel = nullptr;

    // for inplace case, dst is a view of src0 and is updated on top of it
    // so for non-inplace case, copy src0 to dst first
    if (!inplace) {
        ggml_cl_cpy(backend, src0, dst, nullptr);
    }

    // then copy src1 to dst with specified offset
    if (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        kernel = backend_ctx->cpy.kernel_cpy_f32_f32;
    } else if (src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_I32) {
        kernel = backend_ctx->cpy.kernel_cpy_i32_i32;
    } else {
        GGML_ASSERT(false && "not implemented");
    }

    offsetd += offs;
    cl_ulong nb = ggml_element_size(dst);

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(int),      &ne10));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(int),      &ne11));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne12));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne13));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &nb10));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb11));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb12));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb13));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne10));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne11));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &ne12));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne13));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nb));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong), &pnb1));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong), &pnb2));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &pnb3));

    int max_local_size = backend_ctx->get_kernel_workgroup_size(kernel);

    const int nth = MIN(max_local_size, ne00);

    size_t global_work_size[] = {(size_t)ne11*nth, (size_t)ne12, (size_t)ne13};
    size_t local_work_size[] = {(size_t)nth, 1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}
