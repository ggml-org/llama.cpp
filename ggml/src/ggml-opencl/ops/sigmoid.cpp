#include "../cl-common.h"
#include "../ops.h"

void ggml_cl_load_kernels_sigmoid(ggml_backend_opencl_context * backend_ctx) {
    cl_int err;
    const std::string & compile_opts = backend_ctx->kernel_compile_opts;
    // sigmoid
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "sigmoid.cl.h"
        };
#else
        const std::string kernel_src = read_file("sigmoid.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->sigmoid.kernel_sigmoid_f32 = clCreateKernel(prog, "kernel_sigmoid_f32", &err), err));
        CL_CHECK((backend_ctx->sigmoid.kernel_sigmoid_f16 = clCreateKernel(prog, "kernel_sigmoid_f16", &err), err));
        CL_CHECK((backend_ctx->sigmoid.kernel_sigmoid_f32_nc = clCreateKernel(prog, "kernel_sigmoid_f32_nc", &err), err));
        CL_CHECK((backend_ctx->sigmoid.kernel_sigmoid_f16_nc = clCreateKernel(prog, "kernel_sigmoid_f16_nc", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }
}

void ggml_cl_sigmoid(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
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

    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        GGML_TENSOR_LOCALS(int,      ne0, src0, ne);
        GGML_TENSOR_LOCALS(cl_ulong, nb0, src0, nb);
        GGML_TENSOR_LOCALS(cl_ulong, nb,  dst,  nb);

        cl_kernel kernel = src0->type == GGML_TYPE_F32 ? backend_ctx->sigmoid.kernel_sigmoid_f32_nc : backend_ctx->sigmoid.kernel_sigmoid_f16_nc;

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

        size_t global_work_size[] = {(size_t)ne01 * 64, (size_t)ne02, (size_t)ne03};
        size_t local_work_size[] = {64, 1, 1};
        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
        return;
    }

    cl_kernel kernel;
    if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        kernel = backend_ctx->sigmoid.kernel_sigmoid_f32;
    } else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
        kernel = backend_ctx->sigmoid.kernel_sigmoid_f16;
    } else if (src0->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) {
        // bf16 converted to f16
        kernel = backend_ctx->sigmoid.kernel_sigmoid_f16;
    } else {
        GGML_ASSERT(false && "Unsupported data types for sigmoid (input and output must be both f32 or f16)");
    }

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offsetd));

    const int64_t n = ggml_nelements(dst);

    size_t global_work_size[] = {(size_t)n, 1, 1};
    size_t local_work_size[] = {64, 1, 1};

    size_t * local_work_size_ptr = local_work_size;
    if (n % 64 != 0 && !backend_ctx->non_uniform_workgroups) {
        local_work_size_ptr = nullptr;  // Let driver choose the work-group sizes.
    }

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size_ptr, dst);
}
