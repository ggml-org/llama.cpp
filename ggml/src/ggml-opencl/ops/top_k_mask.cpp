#include "../cl-common.h"

void ggml_cl_load_kernels_top_k_mask(ggml_backend_opencl_context * backend_ctx) {
    cl_int err;
#ifdef GGML_OPENCL_EMBED_KERNELS
    const std::string kernel_src {
        #include "top_k_mask.cl.h"
    };
#else
    const std::string kernel_src = read_file("top_k_mask.cl");
#endif
    // Scores can contain NaN and -inf, so do not use the default fast-math options.
    const std::string opencl_c_std =
        std::string("CL") + std::to_string(backend_ctx->opencl_c_version.major) + "." + std::to_string(backend_ctx->opencl_c_version.minor);
    const std::string opts = "-cl-std=" + opencl_c_std +
        (backend_ctx->adreno_use_large_buffer ? " -qcom-enable-large-buffer" : "");
    for (int i = 0; i < 2; ++i) {
        cl_program prog = build_program_from_source(backend_ctx, kernel_src.c_str(), opts + " -DMASK_F16=" + std::to_string(i == 0));
        CL_CHECK((backend_ctx->top_k_mask.kernel_top_k_mask[i] = clCreateKernel(prog, "kernel_top_k_mask", &err), err));
        CL_CHECK((backend_ctx->top_k_mask.kernel_top_k_mask_fill[i] = clCreateKernel(prog, "kernel_top_k_mask_fill", &err), err));
        CL_CHECK(clReleaseProgram(prog));
    }
    GGML_LOG_CONT(".");
}

void ggml_cl_top_k_mask(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_backend_opencl_context * backend_ctx = (ggml_backend_opencl_context *) backend->context;
    const ggml_tensor * score  = dst->src[2];
    const ggml_tensor * blocks = dst->src[3];
    const ggml_tensor * extra  = dst->src[4];
    const int type = dst->type == GGML_TYPE_F16 ? 0 : 1;
    cl_kernel kernel = backend_ctx->top_k_mask.kernel_top_k_mask[type];

    const ggml_tensor_extra_cl * extram = (const ggml_tensor_extra_cl *) src0->extra;
    const ggml_tensor_extra_cl * extrak = (const ggml_tensor_extra_cl *) src1->extra;
    const ggml_tensor_extra_cl * extras = score  ? (const ggml_tensor_extra_cl *) score->extra  : nullptr;
    const ggml_tensor_extra_cl * extrab = blocks ? (const ggml_tensor_extra_cl *) blocks->extra : nullptr;
    const ggml_tensor_extra_cl * extrae = extra  ? (const ggml_tensor_extra_cl *) extra->extra  : nullptr;
    const ggml_tensor_extra_cl * extrad = (const ggml_tensor_extra_cl *) dst->extra;

    const cl_mem buffer_s = extras ? extras->data_device : nullptr;
    const cl_mem buffer_b = extrab ? extrab->data_device : nullptr;
    const cl_mem buffer_e = extrae ? extrae->data_device : nullptr;

    const cl_ulong offsetm = extram->offset + src0->view_offs;
    const cl_ulong offsetk = extrak->offset + src1->view_offs;
    const cl_ulong offsets = extras ? extras->offset + score->view_offs  : 0;
    const cl_ulong offsetb = extrab ? extrab->offset + blocks->view_offs : 0;
    const cl_ulong offsete = extrae ? extrae->offset + extra->view_offs  : 0;
    const cl_ulong offsetd = extrad->offset + dst->view_offs;

    // single launch: fill == true, two launches: fill = false
    // X2 benefits from a parallel fill for long single rows; mobile Adreno favors one launch.
    const cl_int fill = !(backend_ctx->adreno_gen == ADRENO_GPU_GEN::X2E &&
        ggml_nrows(dst) == 1 && dst->ne[0] >= 65536 && ggml_is_contiguous(dst));

    if (!fill) {
        GGML_ASSERT(ggml_is_contiguous(dst));

        cl_kernel kernel_fill = backend_ctx->top_k_mask.kernel_top_k_mask_fill[type];

        const cl_ulong n = ggml_nelements(dst);
        const size_t nv = 16/ggml_type_size(dst->type);
        const size_t nth = MIN((size_t) 256, backend_ctx->get_kernel_workgroup_size(kernel_fill));

        CL_CHECK(clSetKernelArg(kernel_fill, 0, sizeof(cl_mem), &extrad->data_device));
        CL_CHECK(clSetKernelArg(kernel_fill, 1, sizeof(cl_ulong), &offsetd));
        CL_CHECK(clSetKernelArg(kernel_fill, 2, sizeof(cl_ulong), &n));

        size_t global[] = {((n + nv - 1)/nv + nth - 1)/nth*nth};
        size_t local[] = {nth};
        backend_ctx->enqueue_ndrange_kernel(kernel_fill, 1, global, local, dst);
    }

    GGML_TENSOR_LOCALS(cl_ulong, nem, src0,   ne);
    GGML_TENSOR_LOCALS(cl_ulong, nbm, src0,   nb);
    GGML_TENSOR_LOCALS(cl_ulong, nek, src1,   ne);
    GGML_TENSOR_LOCALS(cl_ulong, nbk, src1,   nb);
    GGML_TENSOR_LOCALS(cl_ulong, nes, score,  ne);
    GGML_TENSOR_LOCALS(cl_ulong, nbs, score,  nb);
    GGML_TENSOR_LOCALS(cl_ulong, neb, blocks, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nbb, blocks, nb);
    GGML_TENSOR_LOCALS(cl_ulong, nee, extra,  ne);
    GGML_TENSOR_LOCALS(cl_ulong, nbe, extra,  nb);
    GGML_TENSOR_LOCALS(cl_ulong, ned, dst,    ne);
    GGML_TENSOR_LOCALS(cl_ulong, nbd, dst,    nb);

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extram->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offsetm));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extrak->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offsetk));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &buffer_s));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsets));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_mem),   &buffer_b));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &offsetb));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_mem),   &buffer_e));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &offsete));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nem0));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &nem1));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &nem2));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(cl_ulong), &nem3));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nbm1));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong), &nbm2));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong), &nbm3));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &nek0));
    CL_CHECK(clSetKernelArg(kernel, 20, sizeof(cl_ulong), &nbk0));
    CL_CHECK(clSetKernelArg(kernel, 21, sizeof(cl_ulong), &nbk1));
    CL_CHECK(clSetKernelArg(kernel, 22, sizeof(cl_ulong), &nbk2));
    CL_CHECK(clSetKernelArg(kernel, 23, sizeof(cl_ulong), &nbk3));
    CL_CHECK(clSetKernelArg(kernel, 24, sizeof(cl_ulong), &nes0));
    CL_CHECK(clSetKernelArg(kernel, 25, sizeof(cl_ulong), &nbs0));
    CL_CHECK(clSetKernelArg(kernel, 26, sizeof(cl_ulong), &nbs1));
    CL_CHECK(clSetKernelArg(kernel, 27, sizeof(cl_ulong), &nbs2));
    CL_CHECK(clSetKernelArg(kernel, 28, sizeof(cl_ulong), &nbs3));
    CL_CHECK(clSetKernelArg(kernel, 29, sizeof(cl_ulong), &neb0));
    CL_CHECK(clSetKernelArg(kernel, 30, sizeof(cl_ulong), &neb1));
    CL_CHECK(clSetKernelArg(kernel, 31, sizeof(cl_ulong), &nbb0));
    CL_CHECK(clSetKernelArg(kernel, 32, sizeof(cl_ulong), &nbb1));
    CL_CHECK(clSetKernelArg(kernel, 33, sizeof(cl_ulong), &nee0));
    CL_CHECK(clSetKernelArg(kernel, 34, sizeof(cl_ulong), &nbe0));
    CL_CHECK(clSetKernelArg(kernel, 35, sizeof(cl_ulong), &nbe1));
    CL_CHECK(clSetKernelArg(kernel, 36, sizeof(cl_ulong), &nbe2));
    CL_CHECK(clSetKernelArg(kernel, 37, sizeof(cl_ulong), &nbe3));
    CL_CHECK(clSetKernelArg(kernel, 38, sizeof(cl_ulong), &ned1));
    CL_CHECK(clSetKernelArg(kernel, 39, sizeof(cl_ulong), &ned2));
    CL_CHECK(clSetKernelArg(kernel, 40, sizeof(cl_ulong), &nbd1));
    CL_CHECK(clSetKernelArg(kernel, 41, sizeof(cl_ulong), &nbd2));
    CL_CHECK(clSetKernelArg(kernel, 42, sizeof(cl_ulong), &nbd3));
    CL_CHECK(clSetKernelArg(kernel, 43, sizeof(cl_int),   &fill));

    const size_t nth = MIN((size_t) 512, backend_ctx->get_kernel_workgroup_size(kernel));
    size_t global[] = {nth, (size_t) ggml_nrows(dst)};
    size_t local[] = {nth, 1};
    backend_ctx->enqueue_ndrange_kernel(kernel, 2, global, local, dst);
}
