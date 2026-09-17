#include "../cl-common.h"

void ggml_cl_load_kernels_lightning_indexer(ggml_backend_opencl_context * backend_ctx) {
    cl_int err;
    const std::string & compile_opts = backend_ctx->kernel_compile_opts;
#ifdef GGML_OPENCL_EMBED_KERNELS
    const std::string kernel_src {
        #include "lightning_indexer.cl.h"
    };
#else
    const std::string kernel_src = read_file("lightning_indexer.cl");
#endif
    cl_program prog = build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);
    CL_CHECK((backend_ctx->lightning_indexer.kernel_lightning_indexer = clCreateKernel(prog, "kernel_lightning_indexer", &err), err));
    CL_CHECK(clReleaseProgram(prog));
    GGML_LOG_CONT(".");
}

void ggml_cl_lightning_indexer(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_backend_opencl_context * backend_ctx = (ggml_backend_opencl_context *) backend->context;
    cl_kernel kernel = backend_ctx->lightning_indexer.kernel_lightning_indexer;
    const ggml_tensor * src2 = dst->src[2];
    const ggml_tensor * src3 = dst->src[3];
    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *) src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *) src1->extra;
    ggml_tensor_extra_cl * extra2 = (ggml_tensor_extra_cl *) src2->extra;
    ggml_tensor_extra_cl * extra3 = (ggml_tensor_extra_cl *) src3->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *) dst->extra;

    const cl_ulong offset0 = extra0->offset + src0->view_offs;
    const cl_ulong offset1 = extra1->offset + src1->view_offs;
    const cl_ulong offset2 = extra2->offset + src2->view_offs;
    const cl_ulong offset3 = extra3->offset + src3->view_offs;
    const cl_ulong offsetd = extrad->offset + dst->view_offs;

    const int head_size = src0->ne[0];
    const int n_heads   = src0->ne[1];
    const int n_tokens  = src0->ne[2];
    const int n_keys    = src1->ne[2];
    const int n_masks   = src3->ne[3];
    // Keep the local-memory reduction for small grids or few heads.
    const int use_shuffle = head_size == 128 && backend_ctx->has_khr_subgroup_shuffle && n_heads >= 32 && ggml_nelements(dst) >= 1024;
    // BF16 tensors are stored as F16 on the device.
    const int type_k = src1->type == GGML_TYPE_F32 ? 0 : 1;

    const cl_ulong nbq1 = src0->nb[1];
    const cl_ulong nbq2 = src0->nb[2];
    const cl_ulong nbq3 = src0->nb[3];
    const cl_ulong nbk2 = src1->nb[2];
    const cl_ulong nbk3 = src1->nb[3];
    const cl_ulong nbw1 = src2->nb[1];
    const cl_ulong nbw3 = src2->nb[3];
    const cl_ulong nbm1 = src3->nb[1];
    const cl_ulong nbm3 = src3->nb[3];

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extra2->data_device));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offset2));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_mem),   &extra3->data_device));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &offset3));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &head_size));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &n_heads));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &n_tokens));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &n_keys));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &n_masks));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &type_k));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nbq1));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong), &nbq2));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong), &nbq3));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &nbk2));
    CL_CHECK(clSetKernelArg(kernel, 20, sizeof(cl_ulong), &nbk3));
    CL_CHECK(clSetKernelArg(kernel, 21, sizeof(cl_ulong), &nbw1));
    CL_CHECK(clSetKernelArg(kernel, 22, sizeof(cl_ulong), &nbw3));
    CL_CHECK(clSetKernelArg(kernel, 23, sizeof(cl_ulong), &nbm1));
    CL_CHECK(clSetKernelArg(kernel, 24, sizeof(cl_ulong), &nbm3));
    CL_CHECK(clSetKernelArg(kernel, 25, sizeof(int),      &use_shuffle));

    const int keys_per_group = use_shuffle ? 16 : 4;
    size_t global_work_size[] = { (size_t) ((src1->ne[2] + keys_per_group - 1)/keys_per_group)*64, (size_t) src0->ne[2], (size_t) src0->ne[3] };
    size_t local_work_size[]  = { 64, 1, 1 };
    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}
