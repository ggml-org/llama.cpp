#include "../cl-common.h"
#include "../ops.h"

void ggml_cl_load_kernels_sqr(ggml_backend_opencl_context * backend_ctx) {
    cl_int err;
    const std::string & compile_opts = backend_ctx->kernel_compile_opts;
    // sqr
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "sqr.cl.h"
        };
#else
        const std::string kernel_src = read_file("sqr.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->sqr.kernel_sqr_cont_f32     = clCreateKernel(prog, "kernel_sqr_cont_f32", &err), err));
        CL_CHECK((backend_ctx->sqr.kernel_sqr_cont_f32_4   = clCreateKernel(prog, "kernel_sqr_cont_f32_4", &err), err));
        CL_CHECK((backend_ctx->sqr.kernel_sqr_cont_f16     = clCreateKernel(prog, "kernel_sqr_cont_f16", &err), err));
        CL_CHECK((backend_ctx->sqr.kernel_sqr_cont_f16_4   = clCreateKernel(prog, "kernel_sqr_cont_f16_4", &err), err));

        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }
}

void ggml_cl_sqr(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
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

    cl_kernel kernel;

    // Currently assumes src0 is contiguous
    int n = ggml_nelements(dst);
    if (n % 4 == 0) {
        if (src0->type == GGML_TYPE_F32) {
            kernel = backend_ctx->sqr.kernel_sqr_cont_f32_4;
        } else {
            kernel = backend_ctx->sqr.kernel_sqr_cont_f16_4;
        }
        n /= 4;
    } else {
        if (src0->type == GGML_TYPE_F32) {
            kernel = backend_ctx->sqr.kernel_sqr_cont_f32;
        } else {
            kernel = backend_ctx->sqr.kernel_sqr_cont_f16;
        }
    }

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offsetd));

    size_t global_work_size[] = {(size_t)n, 1, 1};
    size_t local_work_size[] = {64, 1, 1};

    size_t * local_work_size_ptr = local_work_size;
    if (n % 64 != 0 && !backend_ctx->non_uniform_workgroups) {
        local_work_size_ptr = nullptr;
    }

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size_ptr, dst);
}
