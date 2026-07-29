#include "../cl-common.h"

void ggml_cl_load_kernels_top_k(ggml_backend_opencl_context * backend_ctx) {
    cl_int err;
    const std::string & compile_opts = backend_ctx->kernel_compile_opts;
#ifdef GGML_OPENCL_EMBED_KERNELS
    const std::string kernel_src {
        #include "top_k.cl.h"
    };
#else
    const std::string kernel_src = read_file("top_k.cl");
#endif
    cl_program prog = build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);
    CL_CHECK((backend_ctx->kernel_top_k_f32 = clCreateKernel(prog, "kernel_top_k_f32", &err), err));
    CL_CHECK((backend_ctx->kernel_top_k_merge_f32 = clCreateKernel(prog, "kernel_top_k_merge_f32", &err), err));
    CL_CHECK(clReleaseProgram(prog));
    GGML_LOG_CONT(".");
}

void ggml_cl_top_k(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_UNUSED(src1);

    ggml_backend_opencl_context * backend_ctx = (ggml_backend_opencl_context *) backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *) src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *) dst->extra;

    const cl_ulong offset0 = extra0->offset + src0->view_offs;
    const cl_ulong offsetd = extrad->offset + dst->view_offs;

    GGML_TENSOR_LOCALS(int,      ne0, src0, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb0, src0, nb);

    const int k = dst->ne[0];
    cl_kernel kernel = backend_ctx->kernel_top_k_f32;
    const int max_wg = backend_ctx->get_kernel_workgroup_size(kernel);

    int nth = 1;
    const int target = MIN(512, MAX(32, ne00/4));
    while (2*nth <= MIN(target, max_wg)) {
        nth *= 2;
    }

    const size_t n_rows = ggml_nrows(src0);
    const bool tiled = ne00 > 32768 && k > 0 && k <= 256 && n_rows <= 4;
    const int tile_width = tiled ? 4096 : ne00;
    const int n_tiles = tiled ? 1 + (ne00 - 1)/tile_width : 1;
    const int n_candidates = (n_tiles - 1)*k + MIN(k, ne00 - (n_tiles - 1)*tile_width);
    cl_mem candidates = nullptr;
    if (tiled) {
        GGML_ASSERT(n_rows <= SIZE_MAX/sizeof(cl_uint2)/(size_t) n_candidates);
        backend_ctx->prealloc_top_k.allocate(backend_ctx->context, n_rows*(size_t) n_candidates*sizeof(cl_uint2));
        candidates = backend_ctx->prealloc_top_k.buffer;
    }

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne02));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &nb00));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb02));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb03));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &k));
    CL_CHECK(clSetKernelArg(kernel, 12, MAX(259, 2*nth)*sizeof(cl_uint), nullptr));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_mem),   &candidates));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &tile_width));

    size_t global_work_size[] = { (size_t) nth*n_tiles, n_rows, 1 };
    size_t local_work_size[]  = { (size_t) nth, 1, 1 };
    backend_ctx->enqueue_ndrange_kernel(kernel, 2, global_work_size, local_work_size, dst);

    if (tiled) {
        // The global top-k is contained in the top-k of each tile.
        kernel = backend_ctx->kernel_top_k_merge_f32;
        const int max_wg_merge = backend_ctx->get_kernel_workgroup_size(kernel);

        int nth_merge = 1;
        const int target_merge = MIN(512, MAX(32, n_candidates/4));
        while (2*nth_merge <= MIN(target_merge, max_wg_merge)) {
            nth_merge *= 2;
        }

        CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &candidates));
        CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extrad->data_device));
        CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_ulong), &offsetd));
        CL_CHECK(clSetKernelArg(kernel,  3, sizeof(int),      &n_candidates));
        CL_CHECK(clSetKernelArg(kernel,  4, sizeof(int),      &k));
        CL_CHECK(clSetKernelArg(kernel,  5, MAX(259, 2*nth_merge)*sizeof(cl_uint), nullptr));

        global_work_size[0] = nth_merge;
        local_work_size[0] = nth_merge;

        backend_ctx->enqueue_ndrange_kernel(kernel, 2, global_work_size, local_work_size, dst);
    }
}
