#include "../cl-common.h"
#include "../ops.h"

void ggml_cl_load_kernels_dsv4_hc(ggml_backend_opencl_context * backend_ctx) {
    cl_int err;
    const std::string & compile_opts = backend_ctx->kernel_compile_opts;
#ifdef GGML_OPENCL_EMBED_KERNELS
    const std::string kernel_src {
        #include "dsv4_hc.cl.h"
    };
#else
    const std::string kernel_src = read_file("dsv4_hc.cl");
#endif
    for (int i = 0; i < 2; ++i) {
        cl_program prog = build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts + (i == 0 ? " -DHC_VEC=1" : " -DHC_VEC=4"));
        if (i == 0) {
            CL_CHECK((backend_ctx->dsv4_hc.kernel_dsv4_hc_pre = clCreateKernel(prog, "kernel_dsv4_hc_pre", &err), err));
            backend_ctx->dsv4_hc.local_dsv4_hc_pre = MIN((size_t) 64, backend_ctx->get_kernel_workgroup_size(backend_ctx->dsv4_hc.kernel_dsv4_hc_pre));
        }
        CL_CHECK((backend_ctx->dsv4_hc.kernel_dsv4_hc_post[i] = clCreateKernel(prog, "kernel_dsv4_hc_post", &err), err));
        backend_ctx->dsv4_hc.local_dsv4_hc_post[i] = MIN((size_t) 64, backend_ctx->get_kernel_workgroup_size(backend_ctx->dsv4_hc.kernel_dsv4_hc_post[i]));
        CL_CHECK(clReleaseProgram(prog));
    }
    GGML_LOG_CONT(".");
}

void ggml_cl_dsv4_hc(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    auto * backend_ctx = (ggml_backend_opencl_context *) backend->context;
    const bool pre = dst->op == GGML_OP_DSV4_HC_PRE;
    const bool vec = !pre && src0->ne[0] % 4 == 0 && src0->nb[0] == sizeof(float) && src1->nb[0] == sizeof(float);
    cl_kernel kernel = pre ? backend_ctx->dsv4_hc.kernel_dsv4_hc_pre : backend_ctx->dsv4_hc.kernel_dsv4_hc_post[vec];

    cl_uint arg = 0;
    for (int i = 0; i < (pre ? 2 : 4); ++i) {
        const ggml_tensor * src = dst->src[i];
        auto * extra = src ? (ggml_tensor_extra_cl *) src->extra : nullptr;
        const cl_mem buffer = extra ? extra->data_device : nullptr;
        const cl_ulong offset = src ? extra->offset + src->view_offs : 0;
        CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &buffer));
        CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &offset));
        for (int j = 0; j < 3; ++j) {
            const cl_ulong stride = src ? src->nb[j] : 0;
            CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &stride));
        }
    }

    auto * extra = (ggml_tensor_extra_cl *) dst->extra;
    const cl_ulong offset = extra->offset + dst->view_offs;
    const cl_long n_embd = src0->ne[0];
    const cl_long n_hc = pre ? src0->ne[1] : src1->ne[1];
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &extra->data_device));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &offset));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_long), &n_embd));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_long), &n_hc));
    if (pre) {
        const float scale = ggml_get_op_params_f32(dst, 0);
        const cl_int gated = ggml_get_op_params_i32(dst, 1);
        CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(float), &scale));
        CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_int), &gated));
    }

    const size_t n_work = (size_t) n_embd / (vec ? 4 : 1);
    size_t local_work_size[] = { pre ? backend_ctx->dsv4_hc.local_dsv4_hc_pre : backend_ctx->dsv4_hc.local_dsv4_hc_post[vec], 1, 1 };
    size_t global_work_size[] = { CEIL_DIV(n_work, local_work_size[0]) * local_work_size[0], (size_t) dst->ne[1], (size_t) dst->ne[2] };
    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}
