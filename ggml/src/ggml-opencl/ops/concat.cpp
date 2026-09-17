#include "../cl-common.h"
#include "../ops.h"

void ggml_cl_load_kernels_concat(ggml_backend_opencl_context * backend_ctx) {
    cl_int err;
    const std::string & compile_opts = backend_ctx->kernel_compile_opts;
    // concat
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "concat.cl.h"
        };
#else
        const std::string kernel_src = read_file("concat.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);
        CL_CHECK((backend_ctx->concat.kernel_concat_b1 = clCreateKernel(prog, "kernel_concat_b1", &err), err));
        CL_CHECK((backend_ctx->concat.kernel_concat_b2 = clCreateKernel(prog, "kernel_concat_b2", &err), err));
        CL_CHECK((backend_ctx->concat.kernel_concat_b4 = clCreateKernel(prog, "kernel_concat_b4", &err), err));
        CL_CHECK((backend_ctx->concat.kernel_concat_b8 = clCreateKernel(prog, "kernel_concat_b8", &err), err));
        CL_CHECK((backend_ctx->concat.kernel_concat_b4_pack = clCreateKernel(prog, "kernel_concat_b4_pack", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

}

void ggml_cl_concat(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);
    GGML_ASSERT(src0->type == src1->type);
    GGML_ASSERT(src0->type == dst->type);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd  = extrad->offset + dst->view_offs;

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const cl_ulong nb00 = src0->nb[0];
    const cl_ulong nb01 = src0->nb[1];
    const cl_ulong nb02 = src0->nb[2];
    const cl_ulong nb03 = src0->nb[3];

    const cl_ulong nb10 = src1->nb[0];
    const cl_ulong nb11 = src1->nb[1];
    const cl_ulong nb12 = src1->nb[2];
    const cl_ulong nb13 = src1->nb[3];

    const int ne0 = dst->ne[0];
    const int ne1 = dst->ne[1];
    const int ne2 = dst->ne[2];
    const int ne3 = dst->ne[3];

    const cl_ulong nb0 = dst->nb[0];
    const cl_ulong nb1 = dst->nb[1];
    const cl_ulong nb2 = dst->nb[2];
    const cl_ulong nb3 = dst->nb[3];

    const cl_int dim = ((const int32_t *) dst->op_params)[0];
    GGML_ASSERT(dim >= 0 && dim <= 3);

    int nth = MIN(64, ne0);

    const size_t ts = ggml_type_size(dst->type);
    // the pack kernel copies 4-byte elements, so it is only valid for those.
    const bool concat_pack = (dim == 0 && ne0 < 32 && ts == 4);
    cl_kernel kernel;
    if (concat_pack) {
        kernel = backend_ctx->concat.kernel_concat_b4_pack;
    } else {
        switch (ts) {
            case 1:  kernel = backend_ctx->concat.kernel_concat_b1; break;
            case 2:  kernel = backend_ctx->concat.kernel_concat_b2; break;
            case 4:  kernel = backend_ctx->concat.kernel_concat_b4; break;
            case 8:  kernel = backend_ctx->concat.kernel_concat_b8; break;
            default: GGML_ABORT("unsupported concat element size: %zu", ts);
        }
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
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &nb10));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(cl_ulong), &nb11));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nb12));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong), &nb13));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &ne0));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &nb0));
    CL_CHECK(clSetKernelArg(kernel, 20, sizeof(cl_ulong), &nb1));
    CL_CHECK(clSetKernelArg(kernel, 21, sizeof(cl_ulong), &nb2));
    CL_CHECK(clSetKernelArg(kernel, 22, sizeof(cl_ulong), &nb3));
    CL_CHECK(clSetKernelArg(kernel, 23, sizeof(cl_int),   &dim));

    if (concat_pack) {
        // packed kernel needs the dst dims to unflatten its 1-D row index.
        CL_CHECK(clSetKernelArg(kernel, 24, sizeof(int), &ne1));
        CL_CHECK(clSetKernelArg(kernel, 25, sizeof(int), &ne2));
        CL_CHECK(clSetKernelArg(kernel, 26, sizeof(int), &ne3));

        const int maxwg = (int)backend_ctx->get_kernel_workgroup_size(kernel);
        const int base  = MIN(64, maxwg);
        const int tpr   = MIN(ne0, base);                 // threads per row
        const int rpw   = MAX(1, base / tpr);             // rows per workgroup
        const int lsz   = tpr * rpw;
        const int nrows = ne1*ne2*ne3;
        const int nwg   = (nrows + rpw - 1) / rpw;
        size_t global_work_size[] = {(size_t)nwg*lsz, 1, 1};
        size_t local_work_size[]  = {(size_t)lsz, 1, 1};
        backend_ctx->enqueue_ndrange_kernel(kernel, 1, global_work_size, local_work_size, dst);
    } else {
        size_t global_work_size[] = {(size_t)ne1*nth, (size_t)ne2, (size_t)ne3};
        size_t local_work_size[] = {(size_t)nth, 1, 1};

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
    }
}
