#include "../cl-common.h"
#include "../ops.h"

void ggml_cl_load_kernels_ssm_scan(ggml_backend_opencl_context * backend_ctx) {
    cl_int err;
    const std::string & compile_opts = backend_ctx->kernel_compile_opts;

#ifdef GGML_OPENCL_EMBED_KERNELS
    const std::string kernel_src {
        #include "ssm_scan.cl.h"
    };
#else
    const std::string kernel_src = read_file("ssm_scan.cl");
#endif
    cl_program prog = build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

    CL_CHECK((backend_ctx->ssm_scan.kernel_ssm_scan_f32 = clCreateKernel(prog, "kernel_ssm_scan_f32", &err), err));
    CL_CHECK((backend_ctx->ssm_scan.kernel_ssm_scan_f32_mamba2_d128 = clCreateKernel(prog, "kernel_ssm_scan_f32_mamba2_d128", &err), err));
    CL_CHECK((backend_ctx->ssm_scan.kernel_ssm_scan_f32_mamba2_d256 = clCreateKernel(prog, "kernel_ssm_scan_f32_mamba2_d256", &err), err));

    cl_kernel * kernels[] = {
        &backend_ctx->ssm_scan.kernel_ssm_scan_f32_mamba2_d128,
        &backend_ctx->ssm_scan.kernel_ssm_scan_f32_mamba2_d256
    };

    // specialized kernels use subgroups and assume subgroup size is 64,
    // if device does not support subgroups or subgroup size is not 64,
    // release these kernels
    for (int i = 0; i < 2; ++i) {
        size_t subgroup_size = 0;
#if CL_TARGET_OPENCL_VERSION >= 210
        const size_t local_work_size[] = { 64, 1 };
        const cl_int subgroup_err = clGetKernelSubGroupInfo(*kernels[i], backend_ctx->device, CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE,
                sizeof(local_work_size), local_work_size, sizeof(subgroup_size), &subgroup_size, nullptr);
        if (subgroup_err != CL_SUCCESS) {
            subgroup_size = 0;
        }
#endif
        // The specialized kernels reduce over one 64-lane subgroup.
        if (subgroup_size != 64) {
            CL_CHECK(clReleaseKernel(*kernels[i]));
            *kernels[i] = nullptr;
        }
    }
    CL_CHECK(clReleaseProgram(prog));
    GGML_LOG_CONT(".");
}

void ggml_cl_ssm_scan(ggml_backend_t backend, ggml_tensor * dst) {
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);
    GGML_ASSERT(dst->src[0]);
    GGML_ASSERT(dst->src[0]->extra);
    GGML_ASSERT(dst->src[1]);
    GGML_ASSERT(dst->src[1]->extra);
    GGML_ASSERT(dst->src[2]);
    GGML_ASSERT(dst->src[2]->extra);
    GGML_ASSERT(dst->src[3]);
    GGML_ASSERT(dst->src[3]->extra);
    GGML_ASSERT(dst->src[4]);
    GGML_ASSERT(dst->src[4]->extra);
    GGML_ASSERT(dst->src[5]);
    GGML_ASSERT(dst->src[5]->extra);
    GGML_ASSERT(dst->src[6]);
    GGML_ASSERT(dst->src[6]->extra);

    ggml_backend_opencl_context * backend_ctx = (ggml_backend_opencl_context *) backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *) dst->src[0]->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *) dst->src[1]->extra;
    ggml_tensor_extra_cl * extra2 = (ggml_tensor_extra_cl *) dst->src[2]->extra;
    ggml_tensor_extra_cl * extra3 = (ggml_tensor_extra_cl *) dst->src[3]->extra;
    ggml_tensor_extra_cl * extra4 = (ggml_tensor_extra_cl *) dst->src[4]->extra;
    ggml_tensor_extra_cl * extra5 = (ggml_tensor_extra_cl *) dst->src[5]->extra;
    ggml_tensor_extra_cl * extra6 = (ggml_tensor_extra_cl *) dst->src[6]->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *) dst->extra;

    const cl_ulong offset0 = extra0->offset + dst->src[0]->view_offs;
    const cl_ulong offset1 = extra1->offset + dst->src[1]->view_offs;
    const cl_ulong offset2 = extra2->offset + dst->src[2]->view_offs;
    const cl_ulong offset3 = extra3->offset + dst->src[3]->view_offs;
    const cl_ulong offset4 = extra4->offset + dst->src[4]->view_offs;
    const cl_ulong offset5 = extra5->offset + dst->src[5]->view_offs;
    const cl_ulong offset6 = extra6->offset + dst->src[6]->view_offs;
    const cl_ulong offsetd = extrad->offset + dst->view_offs;

    const ggml_tensor * s   = dst->src[0];
    const ggml_tensor * x   = dst->src[1];
    const ggml_tensor * dt  = dst->src[2];
    const ggml_tensor * A   = dst->src[3];
    const ggml_tensor * B   = dst->src[4];
    const ggml_tensor * C   = dst->src[5];

    const cl_ulong s_nb1  = s->nb[1];
    const cl_ulong s_nb2  = s->nb[2];
    const cl_ulong s_nb3  = s->nb[3];
    const cl_ulong x_nb1  = x->nb[1];
    const cl_ulong x_nb2  = x->nb[2];
    const cl_ulong x_nb3  = x->nb[3];
    const cl_ulong dt_nb1 = dt->nb[1];
    const cl_ulong dt_nb2 = dt->nb[2];
    const cl_ulong A_nb1  = A->nb[1];
    const cl_ulong B_nb1  = B->nb[1];
    const cl_ulong B_nb2  = B->nb[2];
    const cl_ulong B_nb3  = B->nb[3];
    const cl_ulong C_nb1  = C->nb[1];
    const cl_ulong C_nb2  = C->nb[2];
    const cl_ulong C_nb3  = C->nb[3];

    const cl_uint A_ne0     = A->ne[0];
    const cl_uint d_state   = s->ne[0];
    const cl_int  head_dim  = x->ne[0];
    const cl_int  n_head    = x->ne[1];
    const cl_int  n_group   = B->ne[1];
    const cl_int  n_tokens  = x->ne[2];
    const cl_uint n_seqs    = x->ne[3];
    const cl_uint K         = ggml_get_op_params_i32(dst, 0);
    const cl_ulong s_off_bytes = (cl_ulong) ggml_nelements(x) * sizeof(float);

    cl_kernel kernel = backend_ctx->ssm_scan.kernel_ssm_scan_f32;
    size_t nth = d_state;
    if (A_ne0 == 1 && K == 1) {
        cl_kernel kernel_mamba2 = nullptr;
        if (d_state == 128) {
            kernel_mamba2 = backend_ctx->ssm_scan.kernel_ssm_scan_f32_mamba2_d128;
        } else if (d_state == 256) {
            kernel_mamba2 = backend_ctx->ssm_scan.kernel_ssm_scan_f32_mamba2_d256;
        }
        if (kernel_mamba2 != nullptr) {
            kernel = kernel_mamba2;
            nth = 64;
        }
    }

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extra2->data_device));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offset2));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_mem),   &extra3->data_device));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &offset3));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_mem),   &extra4->data_device));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &offset4));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_mem),   &extra5->data_device));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &offset5));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_mem),   &extra6->data_device));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &offset6));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &s_nb2));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong), &s_nb3));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong), &x_nb2));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &x_nb3));
    CL_CHECK(clSetKernelArg(kernel, 20, sizeof(cl_ulong), &dt_nb1));
    CL_CHECK(clSetKernelArg(kernel, 21, sizeof(cl_ulong), &dt_nb2));
    CL_CHECK(clSetKernelArg(kernel, 22, sizeof(cl_ulong), &A_nb1));
    CL_CHECK(clSetKernelArg(kernel, 23, sizeof(cl_ulong), &B_nb2));
    CL_CHECK(clSetKernelArg(kernel, 24, sizeof(cl_ulong), &B_nb3));
    CL_CHECK(clSetKernelArg(kernel, 25, sizeof(cl_ulong), &C_nb2));
    CL_CHECK(clSetKernelArg(kernel, 26, sizeof(cl_ulong), &C_nb3));
    CL_CHECK(clSetKernelArg(kernel, 27, sizeof(cl_ulong), &s_off_bytes));
    CL_CHECK(clSetKernelArg(kernel, 28, sizeof(cl_int),   &head_dim));
    CL_CHECK(clSetKernelArg(kernel, 29, sizeof(cl_int),   &n_head));
    CL_CHECK(clSetKernelArg(kernel, 30, sizeof(cl_int),   &n_group));
    CL_CHECK(clSetKernelArg(kernel, 31, sizeof(cl_int),   &n_tokens));

    if (kernel == backend_ctx->ssm_scan.kernel_ssm_scan_f32) {
        CL_CHECK(clSetKernelArg(kernel, 32, sizeof(cl_ulong), &s_nb1));
        CL_CHECK(clSetKernelArg(kernel, 33, sizeof(cl_ulong), &x_nb1));
        CL_CHECK(clSetKernelArg(kernel, 34, sizeof(cl_ulong), &B_nb1));
        CL_CHECK(clSetKernelArg(kernel, 35, sizeof(cl_ulong), &C_nb1));
        CL_CHECK(clSetKernelArg(kernel, 36, sizeof(cl_uint),  &A_ne0));
        CL_CHECK(clSetKernelArg(kernel, 37, sizeof(cl_uint),  &d_state));
        CL_CHECK(clSetKernelArg(kernel, 38, sizeof(cl_uint),  &n_seqs));
        CL_CHECK(clSetKernelArg(kernel, 39, sizeof(cl_uint),  &K));
        CL_CHECK(clSetKernelArg(kernel, 40, d_state * sizeof(float), nullptr));
    }

    size_t global_work_size[] = {
        (size_t) head_dim * (size_t) n_head * nth,
        (size_t) n_seqs,
    };
    size_t local_work_size[] = { nth, 1 };

    backend_ctx->enqueue_ndrange_kernel(kernel, 2, global_work_size, local_work_size, dst);
}
