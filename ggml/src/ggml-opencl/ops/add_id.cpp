#include "../cl-common.h"
#include "../ops.h"

void ggml_cl_load_kernels_add_id(ggml_backend_opencl_context * backend_ctx) {
    cl_int err;
    const std::string & compile_opts = backend_ctx->kernel_compile_opts;
    // add_id
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "add_id.cl.h"
        };
#else
        const std::string kernel_src = read_file("add_id.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->add_id.kernel_add_id = clCreateKernel(prog, "kernel_add_id", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }
}

void ggml_cl_add_id(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    const ggml_tensor * src2 = dst->src[2];
    GGML_ASSERT(src2);
    GGML_ASSERT(src2->extra);

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src2->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous_rows(src0));

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];

    const cl_ulong nb01 = src0->nb[1];
    const cl_ulong nb02 = src0->nb[2];

    const cl_ulong nb11 = src1->nb[1];

    const cl_ulong nb21 = src2->nb[1];

    const int ne0 = dst->ne[0];
    const int ne1 = dst->ne[1];

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extra2 = (ggml_tensor_extra_cl *)src2->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offset2 = extra2->offset + src2->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    cl_kernel kernel = backend_ctx->add_id.kernel_add_id;

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extra2->data_device));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offset2));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb02));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb11));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb21));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne0));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne1));

    int nth = MIN(ne00, (int) backend_ctx->get_kernel_workgroup_size(kernel));
    size_t global_work_size[] = { (size_t)ne01*nth, (size_t)ne02, 1 };
    size_t local_work_size[] = { (size_t)nth, 1, 1 };

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}
