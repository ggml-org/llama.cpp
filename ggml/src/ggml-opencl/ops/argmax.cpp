#include "../cl-common.h"
#include "../ops.h"

void ggml_cl_load_kernels_argmax(ggml_backend_opencl_context * backend_ctx) {
    auto opencl_c_std =
        std::string("CL") + std::to_string(backend_ctx->opencl_c_version.major) + "." + std::to_string(backend_ctx->opencl_c_version.minor);
    std::string compile_opts = std::string("-cl-std=") + opencl_c_std;

    if (!backend_ctx->kernels_loaded_argmax) {
        cl_int err;
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "argmax.cl.h"
        };
#else
        const std::string kernel_src = read_file("argmax.cl");
#endif
        cl_program prog = build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);
        CL_CHECK((backend_ctx->kernel_argmax_f32_i32 = clCreateKernel(prog, "kernel_argmax_f32_i32", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        backend_ctx->kernels_loaded_argmax = true;
    }
}

void ggml_cl_argmax(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);
    GGML_UNUSED(src1);

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    ggml_backend_opencl_context * backend_ctx = (ggml_backend_opencl_context *) backend->context;
    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *) src0->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *) dst->extra;

    const cl_ulong offset0 = extra0->offset + src0->view_offs;
    const cl_ulong offsetd = extrad->offset + dst->view_offs;
    const int ne00 = src0->ne[0];
    const int nrows = ggml_nrows(src0);
    constexpr int nth = 64;

    cl_kernel kernel = backend_ctx->kernel_argmax_f32_i32;
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int),      &nrows));

    size_t global_work_size[] = { (size_t) nth, (size_t) CEIL_DIV(nrows, 2), 1 };
    size_t local_work_size[] = { (size_t) nth, 1, 1 };
    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}
