#include "../cl-common.h"
#include "../ops.h"

void ggml_cl_load_kernels_xielu(ggml_backend_opencl_context * backend_ctx) {
    cl_int err;
    const std::string & compile_opts = backend_ctx->kernel_compile_opts;
#ifdef GGML_OPENCL_EMBED_KERNELS
    const std::string kernel_src {
        #include "xielu.cl.h"
    };
#else
    const std::string kernel_src = read_file("xielu.cl");
#endif
    cl_program prog = build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);
    CL_CHECK((backend_ctx->xielu.kernel_xielu_f32 = clCreateKernel(prog, "kernel_xielu_f32", &err), err));
    CL_CHECK((backend_ctx->xielu.kernel_xielu_f16 = clCreateKernel(prog, "kernel_xielu_f16", &err), err));
    CL_CHECK(clReleaseProgram(prog));
    GGML_LOG_CONT(".");
}

void ggml_cl_xielu(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    UNUSED(src1);

    auto * backend_ctx = (ggml_backend_opencl_context *) backend->context;
    cl_kernel kernel = src0->type == GGML_TYPE_F16 ? backend_ctx->xielu.kernel_xielu_f16 : backend_ctx->xielu.kernel_xielu_f32;

    const auto * extra0 = (const ggml_tensor_extra_cl *) src0->extra;
    const auto * extrad = (const ggml_tensor_extra_cl *) dst->extra;

    const cl_ulong offset0 = extra0->offset + src0->view_offs;
    const cl_ulong offsetd = extrad->offset + dst->view_offs;

    GGML_TENSOR_LOCALS(int,      ne0, src0, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb0, src0, nb);
    GGML_TENSOR_LOCALS(cl_ulong, nb,  dst,  nb);

    const float alpha_n = ggml_get_op_params_f32(dst, 1);
    const float alpha_p = ggml_get_op_params_f32(dst, 2);
    const float beta    = ggml_get_op_params_f32(dst, 3);
    const float eps     = ggml_get_op_params_f32(dst, 4);

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &nb00));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &nb02));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &nb03));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb0));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb1));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb2));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb3));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(float),    &alpha_n));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(float),    &alpha_p));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(float),    &beta));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(float),    &eps));

    const size_t nth = 64;
    size_t global[] = {(size_t) ne01*nth, (size_t) ne02, (size_t) ne03};
    size_t local[]  = {nth, 1, 1};
    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global, local, dst);
}
