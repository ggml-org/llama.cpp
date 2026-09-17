#include "../cl-common.h"
#include "../ops.h"

void ggml_cl_load_kernels_conv_2d(ggml_backend_opencl_context * backend_ctx) {
    cl_int err;
    const std::string & compile_opts = backend_ctx->kernel_compile_opts;
    // conv2d
    {
        #ifdef GGML_OPENCL_EMBED_KERNELS
                const std::string kernel_src {
                    #include "conv2d.cl.h"
                };
                const std::string kernel_src_f16_f32 {
                    #include "conv2d_f16_f32.cl.h"
                };
        #else
                const std::string kernel_src = read_file("conv2d.cl");
                const std::string kernel_src_f16_f32 = read_file("conv2d_f16_f32.cl");
        #endif
                if (!kernel_src.empty()) {
                    cl_program prog_f16 =
                        build_program_from_source(backend_ctx, kernel_src.c_str(), (std::string(compile_opts) + " -DUSE_FP16=1").c_str());
                    CL_CHECK((backend_ctx->conv_2d.kernel_conv_2d_f16 = clCreateKernel(prog_f16, "kernel_conv_2d", &err), err));
                    CL_CHECK(clReleaseProgram(prog_f16));
                    GGML_LOG_CONT(".");
                    cl_program prog_f32 =
                        build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);
                    CL_CHECK((backend_ctx->conv_2d.kernel_conv_2d_f32 = clCreateKernel(prog_f32, "kernel_conv_2d", &err), err));
                    CL_CHECK(clReleaseProgram(prog_f32));
                    GGML_LOG_CONT(".");
                } else {
                    GGML_LOG_WARN("ggml_opencl: conv2d kernel source not found or empty. This op will not be available.\n");
                    backend_ctx->conv_2d.kernel_conv_2d_f16 = nullptr;
                    backend_ctx->conv_2d.kernel_conv_2d_f32 = nullptr;
                }
                if (!kernel_src_f16_f32.empty()) {
                    cl_program prog_f16_f32 =
                        build_program_from_source(backend_ctx, kernel_src_f16_f32.c_str(), compile_opts);
                    CL_CHECK((backend_ctx->conv_2d.kernel_conv_2d_f16_f32 = clCreateKernel(prog_f16_f32, "kernel_conv_2d", &err), err));
                    CL_CHECK(clReleaseProgram(prog_f16_f32));
                    GGML_LOG_CONT(".");
                } else {
                    GGML_LOG_WARN("ggml_opencl: conv2d_f16_f32 kernel source not found or empty. This op will not be available.\n");
                    backend_ctx->conv_2d.kernel_conv_2d_f16_f32 = nullptr;
                }
    }
}

void ggml_cl_conv_2d(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS;
    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    const cl_uint Cout = ne03;
    const cl_uint Cin = ne02;
    const cl_uint N = ne13;
    const cl_uint KW = ne00;
    const cl_uint KH = ne01;
    const cl_uint W = ne10;
    const cl_uint H = ne11;
    const cl_uint OW = ne0;
    const cl_uint OH = ne1;

    const cl_uint s0 = dst->op_params[0];
    const cl_uint s1 = dst->op_params[1];
    const cl_uint p0 = dst->op_params[2];
    const cl_uint p1 = dst->op_params[3];
    const cl_uint d0 = dst->op_params[4];
    const cl_uint d1 = dst->op_params[5];

    const cl_uint cl_nb00 = nb00/ggml_type_size(src0->type);
    const cl_uint cl_nb01 = nb01/ggml_type_size(src0->type);
    const cl_uint cl_nb02 = nb02/ggml_type_size(src0->type);
    const cl_uint cl_nb03 = nb03/ggml_type_size(src0->type);
    const cl_uint cl_nb10 = nb10/ggml_type_size(src1->type);
    const cl_uint cl_nb11 = nb11/ggml_type_size(src1->type);
    const cl_uint cl_nb12 = nb12/ggml_type_size(src1->type);
    const cl_uint cl_nb13 = nb13/ggml_type_size(src1->type);
    const cl_uint cl_nb1 = nb1/ggml_type_size(dst->type);
    const cl_uint cl_nb2 = nb2/ggml_type_size(dst->type);
    const cl_uint cl_nb3 = nb3/ggml_type_size(dst->type);

    const int64_t NPQ = (int64_t)N * OW * OH;

    const uint32_t BS_K = 64;
    const uint32_t BS_NPQ = 64;
    const uint32_t BS_CRS = 16;
    const uint32_t VEC_SIZE = 4;

    const uint32_t TS_K = 4;
    const uint32_t TS_NPQ = 8;

    const uint32_t WG_K = BS_K / TS_K;
    const uint32_t WG_NPQ = BS_NPQ / TS_NPQ;

    auto splitWork = [](uint32_t work_size, uint32_t block_size) { return (block_size + work_size - 1) / block_size; };
    const uint32_t NB_K = splitWork(Cout, BS_K);
    const uint32_t NB_NPQ = splitWork(NPQ, BS_NPQ);

    cl_kernel kernel;
    size_t shmem_size;

    if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16) {
        kernel = backend_ctx->conv_2d.kernel_conv_2d_f16;
        shmem_size = (size_t)(BS_K * BS_CRS * sizeof(cl_half) + BS_CRS * (BS_NPQ / VEC_SIZE) * sizeof(cl_half4));
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        kernel = backend_ctx->conv_2d.kernel_conv_2d_f32;
        shmem_size = (size_t)(BS_K * BS_CRS * sizeof(cl_float) + BS_CRS * (BS_NPQ / VEC_SIZE) * sizeof(cl_float4));
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32) {
        kernel = backend_ctx->conv_2d.kernel_conv_2d_f16_f32;
        shmem_size = (size_t)(BS_K * BS_CRS * sizeof(cl_half) + BS_CRS * (BS_NPQ / VEC_SIZE) * sizeof(cl_float4));
    } else {
        GGML_ASSERT(false && "Unsupported data type combination for conv2d");
    }

    cl_uint idx = 0;
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_mem), &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_mem), &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_mem), &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel, idx++, shmem_size, NULL));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &Cout));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &Cin));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &N));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &KW));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &KH));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &W));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &H));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &OW));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &OH));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &s0));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &s1));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &p0));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &p1));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &d0));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &d1));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &cl_nb00));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &cl_nb01));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &cl_nb02));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &cl_nb03));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &cl_nb10));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &cl_nb11));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &cl_nb12));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &cl_nb13));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &cl_nb1));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &cl_nb2));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint), &cl_nb3));

    size_t global_work_size[] = { (size_t)NB_K * WG_K, (size_t)NB_NPQ * WG_NPQ, 1 };
    size_t local_work_size[] = { (size_t)WG_K, (size_t)WG_NPQ, 1 };

    backend_ctx->enqueue_ndrange_kernel(kernel, 2, global_work_size, local_work_size, dst);
}
