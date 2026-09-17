#include "../cl-common.h"

void ggml_cl_load_kernels_conv_2d_dw(ggml_backend_opencl_context * backend_ctx) {
    cl_int err;
    const std::string & compile_opts = backend_ctx->kernel_compile_opts;
#ifdef GGML_OPENCL_EMBED_KERNELS
    const std::string kernel_src {
        #include "conv2_dw.cl.h"
    };
#else
    const std::string kernel_src = read_file("conv2_dw.cl");
#endif
    cl_program prog = build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);
    CL_CHECK((backend_ctx->conv_2d_dw.kernel_conv2d_dw_f32_f32 = clCreateKernel(prog, "kernel_conv2d_dw_f32_f32", &err), err));
    CL_CHECK((backend_ctx->conv_2d_dw.kernel_conv2d_dwf16_f32 = clCreateKernel(prog, "kernel_conv2d_dwf16_f32", &err), err));
    CL_CHECK(clReleaseProgram(prog));
    GGML_LOG_CONT(".");
}

void ggml_cl_conv2_dw(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS;
    ggml_backend_opencl_context * backend_ctx = (ggml_backend_opencl_context *) backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *) src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *) src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *) dst->extra;

    const cl_long offset0 = extra0->offset + src0->view_offs;
    const cl_long offset1 = extra1->offset + src1->view_offs;
    const cl_long offsetd = extrad->offset + dst->view_offs;

    const cl_long C  = ne12;
    const cl_long N  = ne13;
    const cl_long KW = ne00;
    const cl_long KH = ne01;
    const cl_long W  = ne10;
    const cl_long H  = ne11;
    const cl_long OW = ne0;
    const cl_long OH = ne1;

    const cl_int s0 = dst->op_params[0];
    const cl_int s1 = dst->op_params[1];
    const cl_int p0 = dst->op_params[2];
    const cl_int p1 = dst->op_params[3];
    const cl_int d0 = dst->op_params[4];
    const cl_int d1 = dst->op_params[5];

    const cl_long cl_nb01 = nb01/ggml_type_size(src0->type);
    const cl_long cl_nb03 = nb03/ggml_type_size(src0->type);
    const cl_long cl_nb11 = nb11/ggml_type_size(src1->type);
    const cl_long cl_nb12 = nb12/ggml_type_size(src1->type);
    const cl_long cl_nb13 = nb13/ggml_type_size(src1->type);
    const cl_long cl_nb1  = nb1/ggml_type_size(dst->type);
    const cl_long cl_nb2  = nb2/ggml_type_size(dst->type);
    const cl_long cl_nb3  = nb3/ggml_type_size(dst->type);

    cl_kernel kernel;
    if (src0->type == GGML_TYPE_F32) {
        kernel = backend_ctx->conv_2d_dw.kernel_conv2d_dw_f32_f32;
    } else if (src0->type == GGML_TYPE_F16) {
        kernel = backend_ctx->conv_2d_dw.kernel_conv2d_dwf16_f32;
    } else {
        GGML_ASSERT(false && "Unsupported data type combination for conv2_dw");
    }

    cl_uint idx = 0;
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_long),  &offset0));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_long),  &offset1));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_long),  &offsetd));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_long),  &C));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_long),  &N));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_long),  &KW));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_long),  &KH));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_long),  &W));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_long),  &H));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_long),  &OW));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_long),  &OH));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_int),   &s0));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_int),   &s1));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_int),   &p0));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_int),   &p1));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_int),   &d0));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_int),   &d1));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_long),  &cl_nb01));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_long),  &cl_nb03));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_long),  &cl_nb11));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_long),  &cl_nb12));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_long),  &cl_nb13));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_long),  &cl_nb1));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_long),  &cl_nb2));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_long),  &cl_nb3));

    const size_t total = (size_t) N * C * OH * OW;
    const size_t local_size = MIN((size_t) 256, backend_ctx->get_kernel_workgroup_size(kernel));
    size_t global_work_size[] = { CEIL_DIV(total, local_size) * local_size };
    size_t local_work_size[] = { local_size };

    backend_ctx->enqueue_ndrange_kernel(kernel, 1, global_work_size, local_work_size, dst);
}
