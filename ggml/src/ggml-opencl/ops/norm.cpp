#include "../cl-common.h"
#include "../ops.h"

void ggml_cl_load_kernels_norm(ggml_backend_opencl_context * backend_ctx) {
    cl_int err;
    const std::string & compile_opts = backend_ctx->kernel_compile_opts;
    // norm
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "norm.cl.h"
        };
#else
        const std::string kernel_src = read_file("norm.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->norm.kernel_norm         = clCreateKernel(prog, "kernel_norm", &err), err));
        CL_CHECK((backend_ctx->norm.kernel_norm_mul_add = clCreateKernel(prog, "kernel_norm_mul_add", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }
}

void ggml_cl_norm(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    UNUSED(src1);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    GGML_TENSOR_LOCALS(int,      ne0, src0, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb0, src0, nb);

    int nth = 1;
    while (nth < ne00 && nth < 64) {
        nth *= 2;
    }

    cl_kernel kernel = backend_ctx->norm.kernel_norm;

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),    &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong),  &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),    &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong),  &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(int),       &ne00));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(int),       &ne01));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),       &ne02));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),       &ne03));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong),  &nb00));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong),  &nb01));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong),  &nb02));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong),  &nb03));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(float),     &eps));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(float)*nth, NULL));

    size_t global_work_size[] = {(size_t)ne01*nth, (size_t)ne02, (size_t)ne03};
    size_t local_work_size[] = {(size_t)nth, 1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

void ggml_opencl_op_norm_fused(ggml_backend_t backend, ggml_tensor * norm_tensor, ggml_tensor * mul_tensor, ggml_tensor * add_tensor) {
    GGML_ASSERT(norm_tensor && mul_tensor && add_tensor);

    const ggml_tensor * src0 = norm_tensor->src[0];
    const ggml_tensor * src1 = mul_tensor->src[0] == norm_tensor ? mul_tensor->src[1] : mul_tensor->src[0];
    const ggml_tensor * src2 = add_tensor->src[0] == mul_tensor ? add_tensor->src[1] : add_tensor->src[0];
    const ggml_tensor * dst = add_tensor;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extra2 = (ggml_tensor_extra_cl *)src2->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offset2 = extra2->offset + src2->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    float eps;
    memcpy(&eps, norm_tensor->op_params, sizeof(float));

    const int ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
    const cl_ulong nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    const int ne10 = src1->ne[0], ne11 = src1->ne[1], ne12 = src1->ne[2], ne13 = src1->ne[3];
    const cl_ulong nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    const int ne20 = src2->ne[0], ne21 = src2->ne[1], ne22 = src2->ne[2], ne23 = src2->ne[3];
    const cl_ulong nb21 = src2->nb[1], nb22 = src2->nb[2], nb23 = src2->nb[3];
    const cl_ulong nbd1 = dst->nb[1], nbd2 = dst->nb[2], nbd3 = dst->nb[3];

    size_t sgs;
    if (backend_ctx->gpu_family == ADRENO) sgs = 64;
    else if (backend_ctx->gpu_family == INTEL) sgs = 32;
    else GGML_ASSERT(false && "Unsupported GPU");

    cl_kernel kernel = backend_ctx->norm.kernel_norm_mul_add;

    int nth = sgs;
    int max_workgroup_size = backend_ctx->get_kernel_workgroup_size(kernel);
    while (nth < ne00/4 && nth < max_workgroup_size) nth *= 2;
    nth = MIN(nth, max_workgroup_size);
    nth = MIN(nth, ne00/4);

    size_t gws[] = {(size_t)ne01*nth, (size_t)ne02, (size_t)ne03};
    size_t lws[] = {(size_t)nth, 1, 1};
    size_t num_subgroups = (nth + sgs - 1) / sgs;

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), &extra2->data_device));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_ulong), &offset2));
    CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_mem), &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel, 8, sizeof(int), &ne00));
    CL_CHECK(clSetKernelArg(kernel, 9, sizeof(int), &ne01));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int), &ne02));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int), &ne03));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &nb02));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &nb03));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int), &ne10));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int), &ne11));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int), &ne12));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int), &ne13));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &nb11));
    CL_CHECK(clSetKernelArg(kernel, 20, sizeof(cl_ulong), &nb12));
    CL_CHECK(clSetKernelArg(kernel, 21, sizeof(cl_ulong), &nb13));
    CL_CHECK(clSetKernelArg(kernel, 22, sizeof(int), &ne20));
    CL_CHECK(clSetKernelArg(kernel, 23, sizeof(int), &ne21));
    CL_CHECK(clSetKernelArg(kernel, 24, sizeof(int), &ne22));
    CL_CHECK(clSetKernelArg(kernel, 25, sizeof(int), &ne23));
    CL_CHECK(clSetKernelArg(kernel, 26, sizeof(cl_ulong), &nb21));
    CL_CHECK(clSetKernelArg(kernel, 27, sizeof(cl_ulong), &nb22));
    CL_CHECK(clSetKernelArg(kernel, 28, sizeof(cl_ulong), &nb23));
    CL_CHECK(clSetKernelArg(kernel, 29, sizeof(cl_ulong), &nbd1));
    CL_CHECK(clSetKernelArg(kernel, 30, sizeof(cl_ulong), &nbd2));
    CL_CHECK(clSetKernelArg(kernel, 31, sizeof(cl_ulong), &nbd3));
    CL_CHECK(clSetKernelArg(kernel, 32, sizeof(float), &eps));
    CL_CHECK(clSetKernelArg(kernel, 33, sizeof(cl_float2) * num_subgroups, NULL));

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, gws, lws, dst);
}
