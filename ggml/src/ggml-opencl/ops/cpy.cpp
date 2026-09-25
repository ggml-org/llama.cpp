#include "../cl-common.h"
#include "../ops.h"

void ggml_cl_load_kernels_cpy(ggml_backend_opencl_context * backend_ctx) {
    cl_int err;
    const std::string & compile_opts = backend_ctx->kernel_compile_opts;
    // cpy
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "cpy.cl.h"
        };
#else
        const std::string kernel_src = read_file("cpy.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->cpy.kernel_cpy_f16_f16 = clCreateKernel(prog, "kernel_cpy_f16_f16", &err), err));
        CL_CHECK((backend_ctx->cpy.kernel_cpy_f16_f32 = clCreateKernel(prog, "kernel_cpy_f16_f32", &err), err));
        CL_CHECK((backend_ctx->cpy.kernel_cpy_f32_f16 = clCreateKernel(prog, "kernel_cpy_f32_f16", &err), err));
        CL_CHECK((backend_ctx->cpy.kernel_cpy_f32_f32 = clCreateKernel(prog, "kernel_cpy_f32_f32", &err), err));
        CL_CHECK((backend_ctx->cpy.kernel_cpy_f32_f32_pack = clCreateKernel(prog, "kernel_cpy_f32_f32_pack", &err), err));
        {   // optional: without it ggml_cl_cpy keeps the row-mapped kernel
            cl_int err_flat = CL_SUCCESS;
            cl_kernel k = clCreateKernel(prog, "kernel_cpy_f32_f32_flat", &err_flat);
            if (err_flat == CL_SUCCESS) {
                backend_ctx->cpy.kernel_cpy_f32_f32_flat = k;
            }
        }
        CL_CHECK((backend_ctx->cpy.kernel_cpy_i32_i32 = clCreateKernel(prog, "kernel_cpy_i32_i32", &err), err));
        GGML_LOG_CONT(".");
    }

}

void ggml_cl_cpy(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);

    // GGML_OP_CPY happens between src0 and src1.
    // GGML_OP_DUP and GGML_OP_CONT happen between src0 and dst.
    UNUSED(dst);

    GGML_TENSOR_LOCALS(int,      ne0, src0, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb0, src0, nb);
    GGML_TENSOR_LOCALS(int,      ne1, src1, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb1, src1, nb);

    const enum ggml_type src0t = src0->type;
    const enum ggml_type src1t = src1->type;

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;

    // A contiguous f32 -> f32 copy is a linear move. The kernel below maps one workgroup to
    // each row, so a tensor with few long rows runs on a single compute unit; dispatch those
    // over the whole device instead. GGML_OPENCL_CPY_FLAT=0 restores the row-mapped path.
    static const bool cpy_flat_on = []{
        const char * e = getenv("GGML_OPENCL_CPY_FLAT");
        return !(e && e[0] == '0');
    }();
    if (cpy_flat_on && backend_ctx->cpy.kernel_cpy_f32_f32_flat != nullptr &&
        src0t == GGML_TYPE_F32 && src1t == GGML_TYPE_F32 &&
        ggml_is_contiguous(src0) && ggml_is_contiguous(src1) &&
        ggml_nelements(src0) == ggml_nelements(src1)) {
        cl_kernel k = backend_ctx->cpy.kernel_cpy_f32_f32_flat;
        const cl_ulong nelem = (cl_ulong) ggml_nelements(src0);
        const cl_ulong n4    = nelem / 4;

        CL_CHECK(clSetKernelArg(k, 0, sizeof(cl_mem),   &extra0->data_device));
        CL_CHECK(clSetKernelArg(k, 1, sizeof(cl_ulong), &offset0));
        CL_CHECK(clSetKernelArg(k, 2, sizeof(cl_mem),   &extra1->data_device));
        CL_CHECK(clSetKernelArg(k, 3, sizeof(cl_ulong), &offset1));
        CL_CHECK(clSetKernelArg(k, 4, sizeof(cl_ulong), &nelem));
        CL_CHECK(clSetKernelArg(k, 5, sizeof(cl_ulong), &n4));

        // one work item per float4, plus one for the trailing scalars
        const size_t items = (size_t) n4 + ((nelem % 4) ? 1 : 0);
        const size_t lsz   = MIN((size_t) 64, backend_ctx->max_workgroup_size);
        size_t global_work_size[] = { ((items + lsz - 1) / lsz) * lsz, 1, 1 };
        size_t local_work_size[]  = { lsz, 1, 1 };

        backend_ctx->enqueue_ndrange_kernel(k, 1, global_work_size, local_work_size, src1);
        return;
    }

    cl_kernel kernel;

    switch (src0t) {
        case GGML_TYPE_F32:
            switch (src1t) {
                case GGML_TYPE_F16:
                    kernel = backend_ctx->cpy.kernel_cpy_f32_f16;
                    break;
                case GGML_TYPE_F32:
                    kernel = ne00 < 32 ? backend_ctx->cpy.kernel_cpy_f32_f32_pack
                                       : backend_ctx->cpy.kernel_cpy_f32_f32;
                    break;
                default:
                    GGML_ASSERT(false && "not implemented");
            }
            break;
        case GGML_TYPE_F16:
            switch (src1t) {
                case GGML_TYPE_F16:
                    kernel = backend_ctx->cpy.kernel_cpy_f16_f16;
                    break;
                case GGML_TYPE_F32:
                    kernel = backend_ctx->cpy.kernel_cpy_f16_f32;
                    break;
                default:
                    GGML_ASSERT(false && "not implemented");
            }
            break;
        case GGML_TYPE_I32:
            switch (src1t) {
                case GGML_TYPE_I32:
                    kernel = backend_ctx->cpy.kernel_cpy_i32_i32;
                    break;
                default:
                    GGML_ASSERT(false && "not implemented");
            }
            break;
        default:
            GGML_ASSERT(false && "not implemented");
    }

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne02));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne03));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &nb00));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb02));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb03));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne10));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne11));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &ne12));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne13));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nb10));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong), &nb11));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong), &nb12));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &nb13));

    if (kernel == backend_ctx->cpy.kernel_cpy_f32_f32_pack) {
        const int maxwg = (int)backend_ctx->get_kernel_workgroup_size(kernel);
        const int base  = MIN(64, maxwg);
        const int tpr   = MIN(ne00, base);                 // threads per row
        const int rpw   = MAX(1, base / tpr);              // rows per workgroup
        const int lsz   = tpr * rpw;                       // <= base <= maxwg
        const int nrows = ne01*ne02*ne03;
        const int nwg   = (nrows + rpw - 1) / rpw;

        size_t global_work_size[] = {(size_t)nwg*lsz, 1, 1};
        size_t local_work_size[]  = {(size_t)lsz, 1, 1};

        backend_ctx->enqueue_ndrange_kernel(kernel, 1, global_work_size, local_work_size, src1);
    } else {
        const int nth = MIN(64, ne00);

        size_t global_work_size[] = {(size_t)ne01*nth, (size_t)ne02, (size_t)ne03};
        size_t local_work_size[] = {(size_t)nth, 1, 1};

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, src1);
    }
}
