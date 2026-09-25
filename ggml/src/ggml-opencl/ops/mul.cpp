#include "../cl-common.h"
#include "../ops.h"

void ggml_cl_load_kernels_mul(ggml_backend_opencl_context * backend_ctx) {
    cl_int err;
    const std::string & compile_opts = backend_ctx->kernel_compile_opts;
    // mul
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul.kernel_mul         = clCreateKernel(prog, "kernel_mul", &err), err));
        CL_CHECK((backend_ctx->mul.kernel_mul_row     = clCreateKernel(prog, "kernel_mul_row", &err), err));
        CL_CHECK((backend_ctx->mul.kernel_mul_f16     = clCreateKernel(prog, "kernel_mul_f16", &err), err));
        CL_CHECK((backend_ctx->mul.kernel_mul_row_f16 = clCreateKernel(prog, "kernel_mul_row_f16", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }
}

void ggml_cl_mul(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    GGML_ASSERT(src0->type == src1->type);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const cl_ulong nb00 = src0->nb[0];
    const cl_ulong nb01 = src0->nb[1];
    const cl_ulong nb02 = src0->nb[2];
    const cl_ulong nb03 = src0->nb[3];

    const int ne10 = src1->ne[0];
    const int ne11 = src1->ne[1];
    const int ne12 = src1->ne[2];
    const int ne13 = src1->ne[3]; UNUSED(ne13);

    const cl_ulong nb10 = src1->nb[0];
    const cl_ulong nb11 = src1->nb[1];
    const cl_ulong nb12 = src1->nb[2];
    const cl_ulong nb13 = src1->nb[3]; UNUSED(nb13);

    const int ne0  = dst->ne[0];
    const int ne1  = dst->ne[1];
    const int ne2  = dst->ne[2];
    const int ne3  = dst->ne[3];

    const cl_ulong nb0  = dst->nb[0];
    const cl_ulong nb1  = dst->nb[1];
    const cl_ulong nb2  = dst->nb[2];
    const cl_ulong nb3  = dst->nb[3];

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    bool bcast_row = false;
    cl_kernel kernel;

    if (ggml_nelements(src1) == ne10 && ggml_is_contiguous(src1) && ne00 % 4 == 0 && ne10 % 4 == 0) {
        GGML_ASSERT(ggml_is_contiguous(src0));

        // src1 is a row
        GGML_ASSERT(ne11 == 1);

        bcast_row = true;
        int ne = ne00 / 4;

        if (src0->type == GGML_TYPE_F32) {
            kernel = backend_ctx->mul.kernel_mul_row;
        } else {
            kernel = backend_ctx->mul.kernel_mul_row_f16;
        }

        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extra1->data_device));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offset1));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem),   &extrad->data_device));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_ulong), &offsetd));
        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int),      &ne));
    } else {
        if (src0->type == GGML_TYPE_F32) {
            kernel = backend_ctx->mul.kernel_mul;
        } else {
            kernel = backend_ctx->mul.kernel_mul_f16;
        }

        CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
        CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
        CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
        CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
        CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
        CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
        CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
        CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
        CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne02));
        CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne03));
        CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb00));
        CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb01));
        CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb02));
        CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &nb03));
        CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &ne10));
        CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne11));
        CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &ne12));
        CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &ne13));
        CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong), &nb10));
        CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &nb11));
        CL_CHECK(clSetKernelArg(kernel, 20, sizeof(cl_ulong), &nb12));
        CL_CHECK(clSetKernelArg(kernel, 21, sizeof(cl_ulong), &nb13));
        CL_CHECK(clSetKernelArg(kernel, 22, sizeof(int),      &ne0));
        CL_CHECK(clSetKernelArg(kernel, 23, sizeof(int),      &ne1));
        CL_CHECK(clSetKernelArg(kernel, 24, sizeof(int),      &ne2));
        CL_CHECK(clSetKernelArg(kernel, 25, sizeof(int),      &ne3));
        CL_CHECK(clSetKernelArg(kernel, 26, sizeof(cl_ulong), &nb0));
        CL_CHECK(clSetKernelArg(kernel, 27, sizeof(cl_ulong), &nb1));
        CL_CHECK(clSetKernelArg(kernel, 28, sizeof(cl_ulong), &nb2));
        CL_CHECK(clSetKernelArg(kernel, 29, sizeof(cl_ulong), &nb3));
    }

    if (bcast_row) {
        int n = ggml_nelements(dst)/4;
        size_t global_work_size[] = {(size_t)n, 1, 1};
        size_t local_work_size[] = {64, 1, 1};

        size_t * local_work_size_ptr = local_work_size;
        if (n % 64 != 0 && !backend_ctx->non_uniform_workgroups) {
            local_work_size_ptr = nullptr;  // Let driver choose the work-group sizes.
        }

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size_ptr, dst);
    } else {
        unsigned int nth = MIN(64, ne0);
        size_t global_work_size[] = {ne01*nth, (size_t)ne02, (size_t)ne03};
        size_t local_work_size[] = {nth, 1, 1};

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
    }
}
