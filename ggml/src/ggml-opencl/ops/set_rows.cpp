#include "../cl-common.h"
#include "../ops.h"

void ggml_cl_load_kernels_set_rows(ggml_backend_opencl_context * backend_ctx) {
    cl_int err;
    const std::string & compile_opts = backend_ctx->kernel_compile_opts;
    // set_rows
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "set_rows.cl.h"
        };
#else
        const std::string kernel_src = read_file("set_rows.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->set_rows.kernel_set_rows_f32_i64     = clCreateKernel(prog, "kernel_set_rows_f32_i64",     &err), err));
        CL_CHECK((backend_ctx->set_rows.kernel_set_rows_f32_i32     = clCreateKernel(prog, "kernel_set_rows_f32_i32",     &err), err));
        CL_CHECK((backend_ctx->set_rows.kernel_set_rows_f16_i64     = clCreateKernel(prog, "kernel_set_rows_f16_i64",     &err), err));
        CL_CHECK((backend_ctx->set_rows.kernel_set_rows_f16_i32     = clCreateKernel(prog, "kernel_set_rows_f16_i32",     &err), err));
        CL_CHECK((backend_ctx->set_rows.kernel_set_rows_q8_0_i64    = clCreateKernel(prog, "kernel_set_rows_q8_0_i64",    &err), err));
        CL_CHECK((backend_ctx->set_rows.kernel_set_rows_q8_0_i32    = clCreateKernel(prog, "kernel_set_rows_q8_0_i32",    &err), err));
        CL_CHECK((backend_ctx->set_rows.kernel_set_rows_q8_0_soa_i64 = clCreateKernel(prog, "kernel_set_rows_q8_0_soa_i64", &err), err));
        CL_CHECK((backend_ctx->set_rows.kernel_set_rows_q8_0_soa_i32 = clCreateKernel(prog, "kernel_set_rows_q8_0_soa_i32", &err), err));
        CL_CHECK((backend_ctx->set_rows.kernel_set_rows_q4_0_i64     = clCreateKernel(prog, "kernel_set_rows_q4_0_i64",     &err), err));
        CL_CHECK((backend_ctx->set_rows.kernel_set_rows_q4_0_i32     = clCreateKernel(prog, "kernel_set_rows_q4_0_i32",     &err), err));
        CL_CHECK((backend_ctx->set_rows.kernel_set_rows_q4_0_soa_i64 = clCreateKernel(prog, "kernel_set_rows_q4_0_soa_i64", &err), err));
        CL_CHECK((backend_ctx->set_rows.kernel_set_rows_q4_0_soa_i32 = clCreateKernel(prog, "kernel_set_rows_q4_0_soa_i32", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }
}
void ggml_cl_set_rows(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);
    GGML_ASSERT(src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32);

    // ne0 = ne00
    // ne2 = ne02
    // ne3 = ne03

    GGML_TENSOR_LOCALS(int,      ne0, src0, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb0, src0, nb);

    GGML_TENSOR_LOCALS(int,      ne1, src1, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb1, src1, nb);

    GGML_TENSOR_LOCALS(int,      ne, dst, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb, dst, nb);

    const int nblk0 = ne0/ggml_blck_size(dst->type);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;

    const bool q8_0_soa = dst->type == GGML_TYPE_Q8_0 && ggml_cl_is_q8_0_soa(dst);
    const bool q4_0_soa = dst->type == GGML_TYPE_Q4_0 && ggml_cl_is_q4_0_soa(dst);
    const bool is_soa   = q8_0_soa || q4_0_soa;

    cl_kernel kernel;

    if (q8_0_soa) {
        kernel = (src1->type == GGML_TYPE_I64)
                    ? backend_ctx->set_rows.kernel_set_rows_q8_0_soa_i64
                    : backend_ctx->set_rows.kernel_set_rows_q8_0_soa_i32;
    } else if (q4_0_soa) {
        kernel = (src1->type == GGML_TYPE_I64)
                    ? backend_ctx->set_rows.kernel_set_rows_q4_0_soa_i64
                    : backend_ctx->set_rows.kernel_set_rows_q4_0_soa_i32;
    } else {
        switch (dst->type) {
            case GGML_TYPE_F32:
                kernel = (src1->type == GGML_TYPE_I64)
                            ? backend_ctx->set_rows.kernel_set_rows_f32_i64
                            : backend_ctx->set_rows.kernel_set_rows_f32_i32;
                break;
            case GGML_TYPE_F16:
                kernel = (src1->type == GGML_TYPE_I64)
                            ? backend_ctx->set_rows.kernel_set_rows_f16_i64
                            : backend_ctx->set_rows.kernel_set_rows_f16_i32;
                break;
            case GGML_TYPE_Q8_0:
                kernel = (src1->type == GGML_TYPE_I64)
                            ? backend_ctx->set_rows.kernel_set_rows_q8_0_i64
                            : backend_ctx->set_rows.kernel_set_rows_q8_0_i32;
                break;
            case GGML_TYPE_Q4_0:
                kernel = (src1->type == GGML_TYPE_I64)
                            ? backend_ctx->set_rows.kernel_set_rows_q4_0_i64
                            : backend_ctx->set_rows.kernel_set_rows_q4_0_i32;
                break;
            default:
                GGML_ABORT("not implemented");
        }
    }

    fastdiv_vals ne11_ = init_fastdiv_values(ne11);
    fastdiv_vals ne12_ = init_fastdiv_values(ne12);

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));

    if (is_soa) {
        // The q/d subbuffers in q8_0/q4_0 extras are interchangeable here.
        // For views (e.g. ggml_set_rows' `out`), follow view_src for the SoA extra.
        const ggml_tensor * soa_src = dst->view_src != nullptr ? dst->view_src : dst;
        cl_mem q_mem = nullptr;
        cl_mem d_mem = nullptr;
        if (q8_0_soa) {
            ggml_tensor_extra_cl_q8_0 * e = (ggml_tensor_extra_cl_q8_0 *)soa_src->extra;
            q_mem = e->q;
            d_mem = e->d;
        } else {
            ggml_tensor_extra_cl_q4_0 * e = (ggml_tensor_extra_cl_q4_0 *)soa_src->extra;
            q_mem = e->q;
            d_mem = e->d;
        }
        cl_ulong offset_q = 0;
        cl_ulong offset_d = 0;
        const int ne1_dst = dst->ne[1];
        const int ne2_dst = dst->ne[2];
        const int ne3_dst = dst->ne[3];

        CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &q_mem));
        CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offset_q));
        CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_mem),   &d_mem));
        CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &offset_d));
        CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne01));
        CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb01));
        CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb02));
        CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb03));
        CL_CHECK(clSetKernelArg(kernel, 12, sizeof(fastdiv_vals), &ne11_));
        CL_CHECK(clSetKernelArg(kernel, 13, sizeof(fastdiv_vals), &ne12_));
        CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &nb10));
        CL_CHECK(clSetKernelArg(kernel, 15, sizeof(cl_ulong), &nb11));
        CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nb12));
        CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &nblk0));
        CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &ne1_dst));
        CL_CHECK(clSetKernelArg(kernel, 19, sizeof(int),      &ne2_dst));
        CL_CHECK(clSetKernelArg(kernel, 20, sizeof(int),      &ne3_dst));
    } else {
        ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;
        cl_ulong offsetd = extrad->offset + dst->view_offs;

        CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
        CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
        CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne01));
        CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &nb01));
        CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &nb02));
        CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb03));
        CL_CHECK(clSetKernelArg(kernel, 10, sizeof(fastdiv_vals), &ne11_));
        CL_CHECK(clSetKernelArg(kernel, 11, sizeof(fastdiv_vals), &ne12_));
        CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb10));
        CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &nb11));
        CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &nb12));
        CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &nblk0));
        CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nb1));
        CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong), &nb2));
        CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong), &nb3));
    }

    int nth0 = 64;
    if (backend_ctx->gpu_family == INTEL) {
        nth0 = 32;
    } else if (backend_ctx->gpu_family == ADRENO) {
        nth0 = 64;
    }

    int max_workgroup_size = backend_ctx->get_kernel_workgroup_size(kernel);
    while (nth0 < nblk0 && nth0 < max_workgroup_size) {
        nth0 *= 2;
    }

    int rows_per_workgroup = 1;
    if (nth0 > nblk0) {
        rows_per_workgroup = nth0 / nblk0;
        nth0 = nblk0;
    }

    size_t global_work_size[] = {
        (size_t)(ne01 + rows_per_workgroup - 1)/rows_per_workgroup*nth0,
        (size_t)ne02*rows_per_workgroup,
        (size_t)ne03};
    size_t local_work_size[] = {(size_t)nth0, (size_t)rows_per_workgroup, 1};

    // ne01 == 0 makes global_work_size[0] zero here; enqueue_ndrange_kernel drops the empty range.
    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}
