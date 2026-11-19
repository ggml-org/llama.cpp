#include "../cl-common.h"
#include "../ops.h"

void ggml_cl_load_kernels_cumsum(ggml_backend_opencl_context * backend_ctx) {
    cl_int err;
    const std::string & compile_opts = backend_ctx->kernel_compile_opts;
    // cumsum
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "cumsum.cl.h"
        };
#else
        const std::string kernel_src = read_file("cumsum.cl");
#endif
        cl_program prog;
        prog = build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->cumsum.kernel_cumsum_blk = clCreateKernel(prog, "kernel_cumsum_blk", &err), err));
        CL_CHECK((backend_ctx->cumsum.kernel_cumsum_add = clCreateKernel(prog, "kernel_cumsum_add", &err), err));
        GGML_LOG_CONT(".");
        CL_CHECK(clReleaseProgram(prog));
    }
}

void ggml_cl_cumsum(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);
    GGML_UNUSED(src1);

    GGML_ASSERT(src0->nb[0] == ggml_type_size(src0->type));
    GGML_ASSERT(ggml_is_contiguous(src0));

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    GGML_TENSOR_LOCALS(int,      ne0, src0, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb0, src0, nb);

    cl_kernel kernel = backend_ctx->cumsum.kernel_cumsum_blk;

    int max_workgroup_size = backend_ctx->get_kernel_workgroup_size(kernel);
    int nth = 1;
    while (nth < ne00 && 2*nth <= max_workgroup_size) {
        nth *= 2;
    }

    GGML_ASSERT(ne00 <= nth*nth);

    const int net0 = CEIL_DIV(ne00, nth);
    const int net1 = ne01;
    const int net2 = ne02;
    const int net3 = ne03;

    const cl_ulong nbt0 = sizeof(float);
    const cl_ulong nbt1 = net0*nbt0;
    const cl_ulong nbt2 = net1*nbt1;
    const cl_ulong nbt3 = net2*nbt2;

    static ggml_cl_buffer tmp_buffer;
    tmp_buffer.allocate(backend_ctx->context, net0*ne01*ne02*ne03*sizeof(float));

    CL_CHECK(clSetKernelArg(kernel,   0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,   1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel,   2, sizeof(cl_mem),   &tmp_buffer.buffer));
    CL_CHECK(clSetKernelArg(kernel,   3, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,   4, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,   5, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,   6, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel,   7, sizeof(int),      &ne02));
    CL_CHECK(clSetKernelArg(kernel,   8, sizeof(int),      &ne03));
    CL_CHECK(clSetKernelArg(kernel,   9, sizeof(cl_ulong), &nb00));
    CL_CHECK(clSetKernelArg(kernel,  10, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel,  11, sizeof(cl_ulong), &nb02));
    CL_CHECK(clSetKernelArg(kernel,  12, sizeof(cl_ulong), &nb03));
    CL_CHECK(clSetKernelArg(kernel,  13, sizeof(int),      &net0));
    CL_CHECK(clSetKernelArg(kernel,  14, sizeof(int),      &net1));
    CL_CHECK(clSetKernelArg(kernel,  15, sizeof(int),      &net2));

    size_t global_work_size[] = { (size_t)(nth*net0*ne01), (size_t)ne02, (size_t)ne03};
    size_t local_work_size[] = { (size_t)nth, 1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);

    if(ne00 > nth) {
        // if a single workgroup cannot handle an entire row, each workgroup
        // computes a partial sum and stores to dst, tmp_buffer contains the sum
        // of the each workgroup; cumsum this buffer and add to the partial sums in dst
        cl_ulong offsett = 0;
        kernel = backend_ctx->cumsum.kernel_cumsum_blk;
        CL_CHECK(clSetKernelArg(kernel,   0, sizeof(cl_mem),   &tmp_buffer.buffer));
        CL_CHECK(clSetKernelArg(kernel,   1, sizeof(cl_ulong), &offsett));
        CL_CHECK(clSetKernelArg(kernel,   2, sizeof(cl_mem),   &tmp_buffer.buffer));
        CL_CHECK(clSetKernelArg(kernel,   3, sizeof(cl_mem),   &tmp_buffer.buffer));
        CL_CHECK(clSetKernelArg(kernel,   4, sizeof(cl_ulong), &offsett));
        CL_CHECK(clSetKernelArg(kernel,   5, sizeof(int),      &net0));
        CL_CHECK(clSetKernelArg(kernel,   6, sizeof(int),      &ne01));
        CL_CHECK(clSetKernelArg(kernel,   7, sizeof(int),      &ne02));
        CL_CHECK(clSetKernelArg(kernel,   8, sizeof(int),      &ne03));
        CL_CHECK(clSetKernelArg(kernel,   9, sizeof(cl_ulong), &nbt0));
        CL_CHECK(clSetKernelArg(kernel,  10, sizeof(cl_ulong), &nbt1));
        CL_CHECK(clSetKernelArg(kernel,  11, sizeof(cl_ulong), &nbt2));
        CL_CHECK(clSetKernelArg(kernel,  12, sizeof(cl_ulong), &nbt3));
        CL_CHECK(clSetKernelArg(kernel,  13, sizeof(int),      &net0));
        CL_CHECK(clSetKernelArg(kernel,  14, sizeof(int),      &net1));
        CL_CHECK(clSetKernelArg(kernel,  15, sizeof(int),      &net2));

        size_t global_work_size_1[] = { (size_t)net1*nth, (size_t)net2, (size_t)net3};
        size_t local_work_size_1[] = { (size_t)nth, 1, 1};
        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size_1, local_work_size_1, dst);

        kernel = backend_ctx->cumsum.kernel_cumsum_add;
        CL_CHECK(clSetKernelArg(kernel,   0, sizeof(cl_mem),   &tmp_buffer.buffer));
        CL_CHECK(clSetKernelArg(kernel,   1, sizeof(cl_mem),   &extrad->data_device));
        CL_CHECK(clSetKernelArg(kernel,   2, sizeof(cl_ulong), &offsetd));
        CL_CHECK(clSetKernelArg(kernel,   3, sizeof(int),      &ne00));
        CL_CHECK(clSetKernelArg(kernel,   4, sizeof(int),      &ne01));
        CL_CHECK(clSetKernelArg(kernel,   5, sizeof(int),      &ne02));
        CL_CHECK(clSetKernelArg(kernel,   6, sizeof(int),      &ne03));
        CL_CHECK(clSetKernelArg(kernel,   7, sizeof(int),      &nbt0));
        CL_CHECK(clSetKernelArg(kernel,   8, sizeof(int),      &nbt1));
        CL_CHECK(clSetKernelArg(kernel,   9, sizeof(int),      &nbt2));
        CL_CHECK(clSetKernelArg(kernel,  10, sizeof(int),      &nbt3));

        size_t global_work_size_2[] = { (size_t)(nth*net0*ne01), (size_t)ne02, (size_t)ne03};
        size_t local_work_size_2[] = { (size_t)nth, 1, 1};
        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size_2, local_work_size_2, dst);
    }
}
