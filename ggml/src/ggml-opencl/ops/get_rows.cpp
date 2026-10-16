#include "../cl-common.h"
#include "../ops.h"

void ggml_cl_load_kernels_get_rows(ggml_backend_opencl_context * backend_ctx) {
    cl_int err;
    const std::string & compile_opts = backend_ctx->kernel_compile_opts;
    // get_rows
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "get_rows.cl.h"
        };
#else
        const std::string kernel_src = read_file("get_rows.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->get_rows.kernel_get_rows_f32  = clCreateKernel(prog, "kernel_get_rows_f32", &err), err));
        CL_CHECK((backend_ctx->get_rows.kernel_get_rows_f16  = clCreateKernel(prog, "kernel_get_rows_f16", &err), err));
        CL_CHECK((backend_ctx->get_rows.kernel_get_rows_q4_0 = clCreateKernel(prog, "kernel_get_rows_q4_0", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }
}

void ggml_cl_get_rows(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    GGML_TENSOR_LOCALS(int,      ne0, src0, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb0, src0, nb);
    GGML_TENSOR_LOCALS(int,      ne1, src1, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb1, src1, nb);
    GGML_TENSOR_LOCALS(int,      ne,  dst,  ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb,  dst,  nb);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    cl_kernel kernel;

    switch (src0->type) {
        case GGML_TYPE_F32:
            kernel = backend_ctx->get_rows.kernel_get_rows_f32;
            break;
        case GGML_TYPE_F16:
            kernel = backend_ctx->get_rows.kernel_get_rows_f16;
            break;
        case GGML_TYPE_Q4_0:
            kernel = backend_ctx->get_rows.kernel_get_rows_q4_0;
            break;
        default:
            GGML_ASSERT(false && "not implemented");
    }

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &nb02));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb03));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne10));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb10));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb11));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &nb12));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &nb1));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(cl_ulong), &nb2));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nb3));

    int max_workgroup_size = backend_ctx->get_kernel_workgroup_size(kernel);
    int nth = 1;
    while (nth < ne00 && 2*nth <= max_workgroup_size) {
        nth *= 2;
    }

    int nchunks = 1;
    if (src0->type == GGML_TYPE_F32) {
        const int chunk_target = nth * 4;
        nchunks = (ne00 + chunk_target - 1) / chunk_target;
        nchunks = MAX(1, MIN(nchunks, 64));
    }

    size_t global_work_size[] = {(size_t)ne10*nth*nchunks, (size_t)ne11, (size_t)ne12};
    size_t local_work_size[] = {(size_t)nth, 1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}
