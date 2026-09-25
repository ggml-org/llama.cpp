#include "../cl-common.h"
#include "../ops.h"

void ggml_cl_load_kernels_rms_norm(ggml_backend_opencl_context * backend_ctx) {
    cl_int err;
    const std::string & compile_opts = backend_ctx->kernel_compile_opts;
    // rms_norm
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "rms_norm.cl.h"
        };
#else
        const std::string kernel_src = read_file("rms_norm.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->rms_norm.kernel_rms_norm     = clCreateKernel(prog, "kernel_rms_norm", &err), err));
        CL_CHECK((backend_ctx->rms_norm.kernel_rms_norm_mul = clCreateKernel(prog, "kernel_rms_norm_mul", &err), err));
        CL_CHECK((backend_ctx->rms_norm.kernel_rms_norm_mul_add = clCreateKernel(prog, "kernel_rms_norm_mul_add", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

}

void ggml_opencl_op_rms_norm_mul_add_fused(ggml_backend_t backend, ggml_tensor * rms_norm_tensor, ggml_tensor * mul_tensor, ggml_tensor * add_tensor) {
    GGML_ASSERT(rms_norm_tensor && mul_tensor && add_tensor);

    const ggml_tensor * src0 = rms_norm_tensor->src[0];
    const ggml_tensor * src1 = mul_tensor->src[0] == rms_norm_tensor ? mul_tensor->src[1] : mul_tensor->src[0];
    const ggml_tensor * src2 = add_tensor->src[0] == mul_tensor ? add_tensor->src[1] : add_tensor->src[0];
    const ggml_tensor * dst  = add_tensor;

    GGML_ASSERT(src0 && src0->extra);
    GGML_ASSERT(src1 && src1->extra);
    GGML_ASSERT(src2 && src2->extra);
    GGML_ASSERT(dst  && dst->extra);

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
    memcpy(&eps, rms_norm_tensor->op_params, sizeof(float));

    const int ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
    const cl_ulong nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    const int ne10 = src1->ne[0], ne11 = src1->ne[1], ne12 = src1->ne[2], ne13 = src1->ne[3];
    const cl_ulong nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    const int ne20 = src2->ne[0], ne21 = src2->ne[1], ne22 = src2->ne[2], ne23 = src2->ne[3];
    const cl_ulong nb21 = src2->nb[1], nb22 = src2->nb[2], nb23 = src2->nb[3];
    const cl_ulong nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    GGML_ASSERT(ne00 % 4 == 0);

    size_t sgs;
    if (backend_ctx->gpu_family == ADRENO) sgs = 64;
    else if (backend_ctx->gpu_family == INTEL) sgs = 32;
    else GGML_ASSERT(false && "Unsupported GPU");

    cl_kernel kernel = backend_ctx->rms_norm.kernel_rms_norm_mul_add;

    int nth = sgs;
    int max_workgroup_size = backend_ctx->get_kernel_workgroup_size(kernel);
    while (nth < ne00 && nth < max_workgroup_size) nth *= 2;
    nth = MIN(nth, max_workgroup_size);
    nth = MIN(nth, ne00);

    size_t global_work_size[] = {(size_t)ne01*nth, (size_t)ne02, (size_t)ne03};
    size_t local_work_size[]  = {(size_t)nth, 1, 1};

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extra2->data_device));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offset2));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne02));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne03));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &nb02));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &nb03));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne10));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &ne11));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &ne12));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &ne13));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &nb11));
    CL_CHECK(clSetKernelArg(kernel, 20, sizeof(cl_ulong), &nb12));
    CL_CHECK(clSetKernelArg(kernel, 21, sizeof(cl_ulong), &nb13));
    CL_CHECK(clSetKernelArg(kernel, 22, sizeof(int),      &ne20));
    CL_CHECK(clSetKernelArg(kernel, 23, sizeof(int),      &ne21));
    CL_CHECK(clSetKernelArg(kernel, 24, sizeof(int),      &ne22));
    CL_CHECK(clSetKernelArg(kernel, 25, sizeof(int),      &ne23));
    CL_CHECK(clSetKernelArg(kernel, 26, sizeof(cl_ulong), &nb21));
    CL_CHECK(clSetKernelArg(kernel, 27, sizeof(cl_ulong), &nb22));
    CL_CHECK(clSetKernelArg(kernel, 28, sizeof(cl_ulong), &nb23));
    CL_CHECK(clSetKernelArg(kernel, 29, sizeof(cl_ulong), &nb1));
    CL_CHECK(clSetKernelArg(kernel, 30, sizeof(cl_ulong), &nb2));
    CL_CHECK(clSetKernelArg(kernel, 31, sizeof(cl_ulong), &nb3));
    CL_CHECK(clSetKernelArg(kernel, 32, sizeof(float),    &eps));
    CL_CHECK(clSetKernelArg(kernel, 33, sizeof(float)*sgs, NULL));

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

void ggml_cl_rms_norm(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    UNUSED(src1);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    //ggml_backend_opencl_device_context * dev_ctx =
    //    (ggml_backend_opencl_device_context *)backend->device->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    const int ne00 = src0 ? src0->ne[0] : 0;
    const int ne01 = src0 ? src0->ne[1] : 0;
    const int ne02 = src0 ? src0->ne[2] : 0;
    const int ne03 = src0 ? src0->ne[3] : 0;

    const cl_ulong nb01 = src0 ? src0->nb[1] : 0;
    const cl_ulong nb02 = src0 ? src0->nb[2] : 0;
    const cl_ulong nb03 = src0 ? src0->nb[3] : 0;

    GGML_ASSERT(ne00 % 4 == 0);

    const int nth = MIN(64, ne00);

    size_t global_work_size[] = {(size_t)ne01*nth, (size_t)ne02, (size_t)ne03};
    size_t local_work_size[] = {(size_t)nth, 1, 1};

    cl_kernel kernel = backend_ctx->rms_norm.kernel_rms_norm;

    // Note, this kernel declares local memory in kernel args and the size
    // depends on subgroup size.
    // Note, this requires OpenCL 2.1 and above
    // For now we use fixed subgroup size to simplify support for OpenCL 2.0.
    size_t sgs;
    //CL_CHECK(clGetKernelSubGroupInfo(kernel, dev_ctx->device,
    //    CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE,
    //    sizeof(local_work_size), local_work_size,
    //    sizeof(size_t), &sgs, NULL));
    if (backend_ctx->gpu_family == ADRENO) {
        sgs = 64;
    } else if (backend_ctx->gpu_family == INTEL) {
        sgs = 32;
    } else {
        GGML_ASSERT(false && "Unsupported GPU");
    }

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),    &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong),  &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),    &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong),  &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(int),       &ne00));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(int),       &ne01));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),       &ne02));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),       &ne03));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong),  &nb01));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong),  &nb02));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong),  &nb03));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(float),     &eps));
    // This is local memory - the size depends on subgroup size.
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(float)*nth/sgs,  NULL));

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

void ggml_opencl_op_rms_norm_fused(ggml_backend_t backend, ggml_tensor * rms_norm_tensor, ggml_tensor * mul_tensor) {
    GGML_ASSERT(mul_tensor);
    GGML_ASSERT(rms_norm_tensor);

    // src0 is the src of rms_norm, src1 is the other src of mul (one being rms_norm)
    const ggml_tensor * src0 = rms_norm_tensor->src[0];
    const ggml_tensor * src1;
    if (mul_tensor->src[0] == rms_norm_tensor) {
        src1 = mul_tensor->src[1];
    } else if (mul_tensor->src[1] == rms_norm_tensor) {
        src1 = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false && "Invalid args for rms_norm and mul");
    }
    const ggml_tensor * dst = mul_tensor;

    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    float eps;
    memcpy(&eps, rms_norm_tensor->op_params, sizeof(float));

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const cl_ulong nb01 = src0->nb[1];
    const cl_ulong nb02 = src0->nb[2];
    const cl_ulong nb03 = src0->nb[3];

    const int ne10 = src1->ne[0];
    const int ne11 = src1->ne[1];
    const int ne12 = src1->ne[2];
    const int ne13 = src1->ne[3];

    const cl_ulong nb11 = src1->nb[1];
    const cl_ulong nb12 = src1->nb[2];
    const cl_ulong nb13 = src1->nb[3];

    const cl_ulong nb1 = dst->nb[1];
    const cl_ulong nb2 = dst->nb[2];
    const cl_ulong nb3 = dst->nb[3];

    GGML_ASSERT(ne00 % 4 == 0);

    size_t sgs;
    if (backend_ctx->gpu_family == ADRENO) {
        sgs = 64;
    } else if (backend_ctx->gpu_family == INTEL) {
        sgs = 32;
    } else {
        GGML_ASSERT(false && "Unsupported GPU");
    }

    cl_kernel kernel = backend_ctx->rms_norm.kernel_rms_norm_mul;

    int nth = sgs;
    int max_workgroup_size = backend_ctx->get_kernel_workgroup_size(kernel);
    while (nth < ne00 && nth < max_workgroup_size) {
        nth *= 2;
    }
    nth = MIN(nth, max_workgroup_size);
    nth = MIN(nth, ne00);

    size_t global_work_size[] = {(size_t)ne01*nth, (size_t)ne02, (size_t)ne03};
    size_t local_work_size[] = {(size_t)nth, 1, 1};

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),        &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong),      &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),        &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong),      &offset1));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),        &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong),      &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),           &ne00));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),           &ne01));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),           &ne02));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),           &ne03));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong),      &nb01));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong),      &nb02));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong),      &nb03));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),           &ne10));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),           &ne11));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),           &ne12));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),           &ne13));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong),      &nb11));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong),      &nb12));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong),      &nb13));
    CL_CHECK(clSetKernelArg(kernel, 20, sizeof(cl_ulong),      &nb1));
    CL_CHECK(clSetKernelArg(kernel, 21, sizeof(cl_ulong),      &nb2));
    CL_CHECK(clSetKernelArg(kernel, 22, sizeof(cl_ulong),      &nb3));
    CL_CHECK(clSetKernelArg(kernel, 23, sizeof(float),         &eps));
    CL_CHECK(clSetKernelArg(kernel, 24, sizeof(float)*sgs,     NULL));

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}
