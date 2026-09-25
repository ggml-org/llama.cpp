#include "cl-common.h"

void ggml_cl_load_kernels_repack(ggml_backend_opencl_context * backend_ctx) {
    cl_int err;
    const std::string & compile_opts = backend_ctx->kernel_compile_opts;
    // cvt
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "cvt.cl.h"
        };
#else
        const std::string kernel_src = read_file("cvt.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->repack.kernel_convert_block_q1_0  = clCreateKernel(prog, "kernel_convert_block_q1_0", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_restore_block_q1_0  = clCreateKernel(prog, "kernel_restore_block_q1_0", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_convert_block_q4_0_noshuffle = clCreateKernel(prog, "kernel_convert_block_q4_0_noshuffle", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_restore_block_q4_0_noshuffle = clCreateKernel(prog, "kernel_restore_block_q4_0_noshuffle", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_convert_block_q4_0  = clCreateKernel(prog, "kernel_convert_block_q4_0", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_restore_block_q4_0  = clCreateKernel(prog, "kernel_restore_block_q4_0", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_convert_block_q4_0_trans4_ns = clCreateKernel(prog, "kernel_convert_block_q4_0_trans4_ns", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_restore_block_q4_0_trans4_ns = clCreateKernel(prog, "kernel_restore_block_q4_0_trans4_ns", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_convert_block_q4_1_noshuffle = clCreateKernel(prog, "kernel_convert_block_q4_1_noshuffle", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_restore_block_q4_1_noshuffle = clCreateKernel(prog, "kernel_restore_block_q4_1_noshuffle", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_convert_block_q4_1  = clCreateKernel(prog, "kernel_convert_block_q4_1", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_restore_block_q4_1  = clCreateKernel(prog, "kernel_restore_block_q4_1", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_convert_block_q4_1_trans4_ns = clCreateKernel(prog, "kernel_convert_block_q4_1_trans4_ns", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_restore_block_q4_1_trans4_ns = clCreateKernel(prog, "kernel_restore_block_q4_1_trans4_ns", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_convert_block_q5_0  = clCreateKernel(prog, "kernel_convert_block_q5_0", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_restore_block_q5_0  = clCreateKernel(prog, "kernel_restore_block_q5_0", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_convert_block_q5_0_noshuffle = clCreateKernel(prog, "kernel_convert_block_q5_0_noshuffle", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_restore_block_q5_0_noshuffle = clCreateKernel(prog, "kernel_restore_block_q5_0_noshuffle", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_convert_block_q5_1_noshuffle = clCreateKernel(prog, "kernel_convert_block_q5_1_noshuffle", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_restore_block_q5_1_noshuffle = clCreateKernel(prog, "kernel_restore_block_q5_1_noshuffle", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_convert_block_q5_0_trans4_ns = clCreateKernel(prog, "kernel_convert_block_q5_0_trans4_ns", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_restore_block_q5_0_trans4_ns = clCreateKernel(prog, "kernel_restore_block_q5_0_trans4_ns", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_convert_block_q5_1  = clCreateKernel(prog, "kernel_convert_block_q5_1", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_restore_block_q5_1  = clCreateKernel(prog, "kernel_restore_block_q5_1", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_convert_block_q5_1_trans4_ns = clCreateKernel(prog, "kernel_convert_block_q5_1_trans4_ns", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_restore_block_q5_1_trans4_ns = clCreateKernel(prog, "kernel_restore_block_q5_1_trans4_ns", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_convert_block_q4_k_trans4_ns = clCreateKernel(prog, "kernel_convert_block_q4_k_trans4_ns", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_restore_block_q4_k_trans4_ns = clCreateKernel(prog, "kernel_restore_block_q4_k_trans4_ns", &err), err));
#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        CL_CHECK((backend_ctx->repack.kernel_convert_block_q4_k_tiled_ns = clCreateKernel(prog, "kernel_convert_block_q4_k_tiled_ns", &err), err));
#endif
        CL_CHECK((backend_ctx->repack.kernel_convert_block_q5_k_trans4_ns = clCreateKernel(prog, "kernel_convert_block_q5_k_trans4_ns", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_restore_block_q5_k_trans4_ns = clCreateKernel(prog, "kernel_restore_block_q5_k_trans4_ns", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_convert_block_q6_k_trans4_ns = clCreateKernel(prog, "kernel_convert_block_q6_k_trans4_ns", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_restore_block_q6_k_trans4_ns = clCreateKernel(prog, "kernel_restore_block_q6_k_trans4_ns", &err), err));
#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        CL_CHECK((backend_ctx->repack.kernel_convert_block_q6_k_tiled_ns = clCreateKernel(prog, "kernel_convert_block_q6_k_tiled_ns", &err), err));
#endif
        CL_CHECK((backend_ctx->repack.kernel_convert_block_mxfp4 = clCreateKernel(prog, "kernel_convert_block_mxfp4", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_convert_block_mxfp4_trans = clCreateKernel(prog, "kernel_convert_block_mxfp4_trans", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_convert_block_mxfp4_trans4_ns = clCreateKernel(prog, "kernel_convert_block_mxfp4_trans4_ns", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_restore_block_mxfp4_trans4_ns = clCreateKernel(prog, "kernel_restore_block_mxfp4_trans4_ns", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_restore_block_mxfp4_trans = clCreateKernel(prog, "kernel_restore_block_mxfp4_trans", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_restore_block_mxfp4 = clCreateKernel(prog, "kernel_restore_block_mxfp4", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_convert_block_q8_0  = clCreateKernel(prog, "kernel_convert_block_q8_0", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_restore_block_q8_0  = clCreateKernel(prog, "kernel_restore_block_q8_0", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_restore_block_q8_0_trans  = clCreateKernel(prog, "kernel_restore_block_q8_0_trans", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_dequant_q8_0_f16_view_aos = clCreateKernel(prog, "kernel_dequant_q8_0_f16_view_aos", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_dequant_q8_0_f32_view_aos = clCreateKernel(prog, "kernel_dequant_q8_0_f32_view_aos", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_dequant_q4_0_f16_view_aos = clCreateKernel(prog, "kernel_dequant_q4_0_f16_view_aos", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_dequant_q4_0_f32_view_aos = clCreateKernel(prog, "kernel_dequant_q4_0_f32_view_aos", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_convert_block_q4_K  = clCreateKernel(prog, "kernel_convert_block_q4_K", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_restore_block_q4_K  = clCreateKernel(prog, "kernel_restore_block_q4_K", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_convert_block_q4_K_noshuffle = clCreateKernel(prog, "kernel_convert_block_q4_K_noshuffle", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_restore_block_q4_K_noshuffle = clCreateKernel(prog, "kernel_restore_block_q4_K_noshuffle", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_convert_block_q5_K  = clCreateKernel(prog, "kernel_convert_block_q5_K", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_restore_block_q5_K  = clCreateKernel(prog, "kernel_restore_block_q5_K", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_convert_block_q5_K_noshuffle = clCreateKernel(prog, "kernel_convert_block_q5_K_noshuffle", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_restore_block_q5_K_noshuffle = clCreateKernel(prog, "kernel_restore_block_q5_K_noshuffle", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_convert_block_q6_K  = clCreateKernel(prog, "kernel_convert_block_q6_K", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_restore_block_q6_K  = clCreateKernel(prog, "kernel_restore_block_q6_K", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_convert_block_q6_K_noshuffle  = clCreateKernel(prog, "kernel_convert_block_q6_K_noshuffle", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_restore_block_q6_K_noshuffle  = clCreateKernel(prog, "kernel_restore_block_q6_K_noshuffle", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_convert_block_iq4_nl = clCreateKernel(prog, "kernel_convert_block_iq4_nl", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_restore_block_iq4_nl = clCreateKernel(prog, "kernel_restore_block_iq4_nl", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_convert_block_iq4_nl_noshuffle = clCreateKernel(prog, "kernel_convert_block_iq4_nl_noshuffle", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_restore_block_iq4_nl_noshuffle = clCreateKernel(prog, "kernel_restore_block_iq4_nl_noshuffle", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_convert_bf16_to_f16 = clCreateKernel(prog, "kernel_convert_bf16_to_f16", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_convert_f16_to_bf16 = clCreateKernel(prog, "kernel_convert_f16_to_bf16", &err), err));
#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        CL_CHECK((backend_ctx->repack.kernel_moe_expand_scale_q8_0 = clCreateKernel(prog, "kernel_moe_expand_scale_q8_0", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_moe_expand_scale_q5_0 = clCreateKernel(prog, "kernel_moe_expand_scale_q5_0", &err), err));
        CL_CHECK((backend_ctx->repack.kernel_moe_expand_scale_q5_K = clCreateKernel(prog, "kernel_moe_expand_scale_q5_K", &err), err));
#endif
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

}
static cl_mem ggml_cl_create_temp_upload_buffer(
    cl_context context, cl_command_queue queue,
    size_t nbytes, const void * data,
    const char * tensor_name_for_log)
{
    cl_int err;
    cl_mem buf = clCreateBuffer(context, CL_MEM_READ_WRITE, nbytes, NULL, &err);
    if (err != CL_SUCCESS) {
        clFinish(queue);
        buf = clCreateBuffer(context, CL_MEM_READ_WRITE, nbytes, NULL, &err);
    }
    if (err == CL_SUCCESS) {
        const cl_int werr = clEnqueueWriteBuffer(queue, buf, CL_TRUE, 0, nbytes, data, 0, NULL, NULL);
        if (werr == CL_SUCCESS) {
            return buf;
        }
        clReleaseMemObject(buf);
    }
    buf = clCreateBuffer(context,
        CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR | CL_MEM_HOST_WRITE_ONLY,
        nbytes, NULL, &err);
    if (err != CL_SUCCESS) {
        return NULL;
    }
    void * mapped = clEnqueueMapBuffer(queue, buf, CL_TRUE,
        CL_MAP_WRITE_INVALIDATE_REGION, 0, nbytes, 0, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(buf);
        return NULL;
    }
    memcpy(mapped, data, nbytes);
    const cl_int uerr = clEnqueueUnmapMemObject(queue, buf, mapped, 0, NULL, NULL);
    if (uerr != CL_SUCCESS) {
        clReleaseMemObject(buf);
        return NULL;
    }
    if (tensor_name_for_log) {
        GGML_LOG_INFO("ggml_opencl: %s (%.1f MiB) - device alloc failed, using CL_MEM_ALLOC_HOST_PTR fallback\n",
                      tensor_name_for_log, nbytes / 1024.0 / 1024.0);
    }
    return buf;
}

void ggml_backend_opencl_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_opencl_device_context * dev_ctx = (ggml_backend_opencl_device_context *) buffer->buft->device->context;
    ggml_backend_opencl_context * backend_ctx = dev_ctx->backend_ctx;

    cl_context context = backend_ctx->context;
    cl_command_queue queue = backend_ctx->queue;

#ifdef GGML_OPENCL_SOA_Q
    if (tensor->type == GGML_TYPE_Q1_0) {
        ggml_tensor_extra_cl * extra_orig = (ggml_tensor_extra_cl *)tensor->extra;
        GGML_ASSERT(extra_orig && "Tesnors in OpenCL backend should have been allocated and initialized");

        // Allocate the new extra and create aliases from the original.
        ggml_backend_opencl_buffer_context * ctx = (ggml_backend_opencl_buffer_context *) buffer->context;
        ggml_tensor_extra_cl_q1_0 * extra = ctx->ggml_opencl_alloc_temp_tensor_extra_q1_0();

        // q1_0 block = ggml_half d + (QK1_0/8) quant bytes = 2 + 16 = 18 bytes
        size_t size_d = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*sizeof(ggml_fp16_t);
        size_t size_q = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*(ggml_blck_size(tensor->type)/8);
        GGML_ASSERT(size_d + size_q == ggml_nbytes(tensor) && "Incorrect tensor size");

        cl_int err;
        cl_mem data_device = clCreateBuffer(context, CL_MEM_READ_WRITE,
            ggml_nbytes(tensor), NULL, &err);
        CL_CHECK(err);
        CL_CHECK(clEnqueueWriteBuffer(
            queue, data_device, CL_TRUE, 0,
            ggml_nbytes(tensor), data, 0, NULL, NULL));

        // The original tensor memory is divided into scales and quants, i.e.,
        // we first store scales, then quants.
        cl_buffer_region region;

        // Create subbuffer for scales.
        region.origin = align_to(extra_orig->offset + tensor->view_offs + offset, backend_ctx->alignment);
        region.size = size_d;
        extra->d = clCreateSubBuffer(
            extra_orig->data_device, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);
        auto previous_origin = region.origin;

        // Create subbuffer for quants.
        region.origin = align_to(previous_origin + size_d, backend_ctx->alignment);
        region.size = size_q;
        extra->q = clCreateSubBuffer(
            extra_orig->data_device, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);

        cl_kernel kernel = backend_ctx->repack.kernel_convert_block_q1_0;

        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_device));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->q));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extra->d));

        size_t global_work_size[] = {(size_t)ggml_nelements(tensor)/ggml_blck_size(tensor->type), 1, 1};
        size_t local_work_size[] = {64, 1, 1};

        cl_event evt;
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &evt));
        CL_CHECK(clWaitForEvents(1, &evt));
        CL_CHECK(clReleaseMemObject(data_device));

        tensor->extra = extra;

        // q is uint32 (32 sign bits each); d is one half per 128-block.
#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        if (enable_adreno_trans_weight(backend_ctx, tensor)) {
            int M = tensor->ne[1];   // ne01
            int K = tensor->ne[0];   // ne00

            GGML_ASSERT(K % 128 == 0);
            GGML_ASSERT(M % 4 == 0);
            GGML_ASSERT(tensor->ne[2] == 1);
            GGML_ASSERT(tensor->ne[3] == 1);

            transpose_2d_as_32b(backend_ctx, extra->q, extra->q, size_q, K/32,  M);
            transpose_2d_as_16b(backend_ctx, extra->d, extra->d, size_d, K/128, M);
        } // end transpose
#endif // GGML_OPENCL_USE_ADRENO_KERNELS

        return;
    }
    // We separate the quantized bits and scale from block_q4_0 by using an
    // additional kernel, where each thread handles a block. We first read the
    // original weights into a temporary buffer, then create two separate
    // buffers for quantized bits and scales, which are then populated by the
    // conversion kernel.
    if (tensor->type == GGML_TYPE_Q4_0) {
        // Views can't SoA-ify here - parent owns the layout (see q8_0 guard).
        if (tensor->view_src != nullptr || !ggml_is_contiguous(tensor)) {
            return;
        }
        // Tensors should have been preallocated, therefore they should
        // already have ggml_tensor_extra_cl as extra.
        ggml_tensor_extra_cl * extra_orig = (ggml_tensor_extra_cl *)tensor->extra;
        GGML_ASSERT(extra_orig && "Tesnors in OpenCL backend should have been allocated and initialized");

        // Allocate the new extra and create aliases from the original.
        ggml_backend_opencl_buffer_context * ctx = (ggml_backend_opencl_buffer_context *) buffer->context;
        ggml_tensor_extra_cl_q4_0 * extra = ctx->ggml_opencl_alloc_temp_tensor_extra_q4_0();

        size_t size_d = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*sizeof(ggml_fp16_t);
        size_t size_q = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*ggml_blck_size(tensor->type)/2;
        GGML_ASSERT(size_d + size_q == ggml_nbytes(tensor) && "Incorrect tensor size");

        cl_int err;
        cl_mem data_device = ggml_cl_create_temp_upload_buffer(context, queue, ggml_nbytes(tensor), data, tensor->name);
        GGML_ASSERT(data_device != NULL && "set_tensor: temp upload buffer alloc failed");

        // We consider the specified offset arg as always, although For weights
        // the offset arg should be 0 (we do not assert this).
        //GGML_ASSERT(offset == 0);

        // We create subbuffers from the original tensor buffer for scales and
        // quants - i.e., scales and quants are aliases into the buffer object
        // that backs the original tensor. This is a cleaner way to adapt to the
        // new memory management.
        // In the old code, we allocate new buffers for scales and quants
        // respectively, which could still be done but would result in double
        // allocation; properly deallocating the preallocated buffer that backs
        // the tensors is tricky and would leak the backend specific information
        // into the general backend code.
        // Does this create misaligned subbuffers (alignment is 1024) in certain
        // cases ?
        cl_buffer_region region;

        // The original tensor memory is divided into scales and quants, i.e.,
        // we first store scales, then quants.
        // Create subbuffer for scales.
        region.origin = align_to(extra_orig->offset + tensor->view_offs + offset, backend_ctx->alignment);
        region.size = size_d;
        extra->d = clCreateSubBuffer(
            extra_orig->data_device, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);
        auto previous_origin = region.origin;

        // Create subbuffer for quants.
        region.origin = align_to(previous_origin + size_d, backend_ctx->alignment);
        region.size = size_q;
        extra->q = clCreateSubBuffer(
            extra_orig->data_device, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        // Adreno moe q4_0 kernel needs special transpose and unshuffling
        if (use_adreno_moe_kernels(backend_ctx, tensor)) {
            cl_kernel kernel = backend_ctx->repack.kernel_convert_block_q4_0_trans4_ns;

            int ne00 = tensor->ne[0];
            int ne01 = tensor->ne[1];
            int ne02 = tensor->ne[2];
            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_device));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->q));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extra->d));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int), &ne00));
            CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int), &ne01));

            size_t global_work_size[3] = {static_cast<size_t>(((ne01 + 63) / 64) * 64), static_cast<size_t>(ne00 / 32), static_cast<size_t>(ne02)};
            size_t local_work_size[3] = {64, 2, 1};

            cl_event evt;
            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &evt));
            CL_CHECK(clWaitForEvents(1, &evt));
            CL_CHECK(clReleaseMemObject(data_device));

            // Create image for Q
            cl_image_format img_format_q = {CL_R, CL_UNSIGNED_INT32};
            cl_image_desc img_desc_q = {
                CL_MEM_OBJECT_IMAGE1D_BUFFER,
                static_cast<size_t>(ggml_nelements(tensor) / 8),
                0, 0, 0, 0, 0, 0, 0,
                { extra->q }
            };
            extra->q_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_format_q, &img_desc_q, NULL, &err);
            tensor->extra = extra;
            // MoE tensors are also SOA'ed
            ctx->q4_0_soa_tensors.insert(tensor);

            return;
        }
#endif // GGML_OPENCL_USE_ADRENO_KERNELS

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        cl_kernel kernel = backend_ctx->repack.kernel_convert_block_q4_0;

        // The optimized kernels need weights in natural order, so unshuffle.
        if (use_adreno_kernels(backend_ctx, tensor)) {
            kernel = backend_ctx->repack.kernel_convert_block_q4_0_noshuffle;
        }
#else
        cl_kernel kernel = backend_ctx->repack.kernel_convert_block_q4_0;
#endif // GGML_OPENCL_USE_ADRENO_KERNELS
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_device));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->q));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extra->d));

        size_t global_work_size[] = {(size_t)ggml_nelements(tensor)/ggml_blck_size(tensor->type), 1, 1};
        size_t local_work_size[] = {64, 1, 1};

        cl_event evt;
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &evt));
        CL_CHECK(clWaitForEvents(1, &evt));
        CL_CHECK(clReleaseMemObject(data_device));

        tensor->extra = extra;
        ctx->q4_0_soa_tensors.insert(tensor);

        // transpose the weights and scales
#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        // Only do transpose for large, non batched matrix
        // TODO: use preallocated images instead of sub-buffer then image
        if (use_adreno_kernels(backend_ctx, tensor)) {
        int M = tensor->ne[1];
            int K = tensor->ne[0];

            GGML_ASSERT(K % 32 == 0);

            if (use_q4_0_bin_kernels(backend_ctx, tensor)) {
                cl_int err;
                cl_image_format wimg_fmt;
                cl_image_desc   wimg_desc;

                // transpose quants as 32-bit words (M-first)
                GGML_ASSERT(M % 64 == 0);
                transpose_2d_as_32b(backend_ctx, extra->q, extra->q, size_q, K / 8, M);
                transpose_2d_as_16b(backend_ctx, extra->d, extra->d, size_d, K / 32, M);

                wimg_fmt = { CL_R, CL_UNSIGNED_INT32 };
                memset(&wimg_desc, 0, sizeof(wimg_desc));
                wimg_desc.image_type  = CL_MEM_OBJECT_IMAGE1D_BUFFER;
                wimg_desc.image_width = (size_t)M * K / 8;
                wimg_desc.buffer      = extra->q;
                CL_CHECK((extra->q_img = clCreateImage(context, CL_MEM_READ_ONLY, &wimg_fmt, &wimg_desc, NULL, &err), err));

                wimg_fmt = { CL_R, CL_HALF_FLOAT };
                memset(&wimg_desc, 0, sizeof(wimg_desc));
                wimg_desc.image_type  = CL_MEM_OBJECT_IMAGE1D_BUFFER;
                wimg_desc.image_width = (size_t)M * K / 32;
                wimg_desc.buffer      = extra->d;
                CL_CHECK((extra->d_img = clCreateImage(context, CL_MEM_READ_ONLY, &wimg_fmt, &wimg_desc, NULL, &err), err));
            } else {
                // Transpose q and d as ushort
                transpose_2d_as_16b(backend_ctx, extra->q, extra->q, size_q, K/4, M);
                transpose_2d_as_16b(backend_ctx, extra->d, extra->d, size_d, K/32, M);
            }
        }
#endif // GGML_OPENCL_USE_ADRENO_KERNELS
        return;
    }
    if (tensor->type == GGML_TYPE_Q4_1) {
        ggml_tensor_extra_cl * extra_orig = (ggml_tensor_extra_cl *)tensor->extra;
        GGML_ASSERT(extra_orig && "Tesnors in OpenCL backend should have been allocated and initialized");

        // Allocate the new extra and create aliases from the original.
        ggml_backend_opencl_buffer_context * ctx = (ggml_backend_opencl_buffer_context *) buffer->context;
        ggml_tensor_extra_cl_q4_1 * extra = ctx->ggml_opencl_alloc_temp_tensor_extra_q4_1();

        size_t size_d = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*sizeof(ggml_fp16_t);
        size_t size_m = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*sizeof(ggml_fp16_t);
        size_t size_q = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*ggml_blck_size(tensor->type)/2;
        GGML_ASSERT(size_d + size_m + size_q == ggml_nbytes(tensor) && "Incorrect tensor size");

        cl_int err;
        cl_mem data_device = ggml_cl_create_temp_upload_buffer(context, queue, ggml_nbytes(tensor), data, tensor->name);
        GGML_ASSERT(data_device != NULL && "set_tensor: temp upload buffer alloc failed");

        cl_buffer_region region;

        // The original tensor memory is divided into scales and quants, i.e.,
        // we first store scales, mins, then quants.
        // Create subbuffer for scales.
        region.origin = align_to(extra_orig->offset + tensor->view_offs + offset, backend_ctx->alignment);
        region.size = size_d;
        extra->d = clCreateSubBuffer(
            extra_orig->data_device, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);
        auto previous_origin = region.origin;

        // Create subbuffer for mins.
        region.origin = align_to(previous_origin + size_d, backend_ctx->alignment);
        region.size = size_m;
        extra->m = clCreateSubBuffer(
            extra_orig->data_device, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);
        previous_origin = region.origin;

        // Create subbuffer for quants.
        region.origin = align_to(previous_origin + size_m, backend_ctx->alignment);
        region.size = size_q;
        extra->q = clCreateSubBuffer(
            extra_orig->data_device, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        // Adreno moe q4_1 kernel needs special transpose and unshuffling
        if (use_adreno_moe_kernels(backend_ctx, tensor)) {
            cl_kernel kernel = backend_ctx->repack.kernel_convert_block_q4_1_trans4_ns;

            int ne00 = tensor->ne[0];
            int ne01 = tensor->ne[1];
            int ne02 = tensor->ne[2];
            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_device));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->q));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extra->d));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &extra->m));
            CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int), &ne00));
            CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int), &ne01));

            size_t global_work_size[3] = {static_cast<size_t>(((ne01 + 63) / 64) * 64), static_cast<size_t>(ne00 / 32), static_cast<size_t>(ne02)};
            size_t local_work_size[3] = {64, 2, 1};

            cl_event evt;
            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &evt));
            CL_CHECK(clWaitForEvents(1, &evt));
            CL_CHECK(clReleaseMemObject(data_device));

            // Create image for Q
            cl_image_format img_format_q = {CL_R, CL_UNSIGNED_INT32};
            cl_image_desc img_desc_q = {
                CL_MEM_OBJECT_IMAGE1D_BUFFER,
                static_cast<size_t>(ggml_nelements(tensor) / 8),
                0, 0, 0, 0, 0, 0, 0,
                { extra->q }
            };
            extra->q_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_format_q, &img_desc_q, NULL, &err);
            tensor->extra = extra;

            return;
        }
#endif // GGML_OPENCL_USE_ADRENO_KERNELS

        // normal q4_1 repack
#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        cl_kernel kernel = backend_ctx->repack.kernel_convert_block_q4_1;

        if (use_adreno_kernels(backend_ctx, tensor)) {
            kernel = backend_ctx->repack.kernel_convert_block_q4_1_noshuffle;
        }
#else
        cl_kernel kernel = backend_ctx->repack.kernel_convert_block_q4_1;
#endif // GGML_OPENCL_USE_ADRENO_KERNELS
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_device));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->q));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extra->d));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &extra->m));

        size_t global_work_size[] = {(size_t)ggml_nelements(tensor)/ggml_blck_size(tensor->type), 1, 1};
        size_t local_work_size[] = {64, 1, 1};

        cl_event evt;
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &evt));
        CL_CHECK(clWaitForEvents(1, &evt));
        CL_CHECK(clReleaseMemObject(data_device));

        tensor->extra = extra;

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        if (use_adreno_kernels(backend_ctx, tensor)) {

            int M = tensor->ne[1];
            int K = tensor->ne[0];

            GGML_ASSERT(K % 32 == 0);

            // Transpose q as ushort
            transpose_2d_as_16b(backend_ctx, extra->q, extra->q, size_q, K/4, M);
            // Transpose d as ushort
            transpose_2d_as_16b(backend_ctx, extra->d, extra->d, size_d, K/32, M);
            // Transpose m as ushort
            transpose_2d_as_16b(backend_ctx, extra->m, extra->m, size_m, K/32, M);
        }
#endif // GGML_OPENCL_USE_ADRENO_KERNELS
        return;
    }
    if (tensor->type == GGML_TYPE_Q5_0) {
        ggml_tensor_extra_cl * extra_orig = (ggml_tensor_extra_cl *)tensor->extra;
        GGML_ASSERT(extra_orig && "Tesnors in OpenCL backend should have been allocated and initialized");

        // Allocate the new extra and create aliases from the original.
        ggml_backend_opencl_buffer_context * ctx = (ggml_backend_opencl_buffer_context *) buffer->context;
        ggml_tensor_extra_cl_q5_0 * extra = ctx->ggml_opencl_alloc_temp_tensor_extra_q5_0();

        size_t size_d = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*sizeof(ggml_fp16_t);
        size_t size_qs = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*ggml_blck_size(tensor->type)/2;
        size_t size_qh = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*sizeof(int32_t);
        GGML_ASSERT(size_d + size_qs + size_qh == ggml_nbytes(tensor) && "Incorrect tensor size");

        cl_int err;
        cl_mem data_device = ggml_cl_create_temp_upload_buffer(context, queue, ggml_nbytes(tensor), data, tensor->name);
        GGML_ASSERT(data_device != NULL && "set_tensor: temp upload buffer alloc failed");

        cl_buffer_region region;

        // Create subbuffer for scales.
        region.origin = align_to(extra_orig->offset + tensor->view_offs + offset, backend_ctx->alignment);
        region.size = size_d;
        extra->d = clCreateSubBuffer(
            extra_orig->data_device, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);
        auto previous_origin = region.origin;

        // Create subbuffer for qh.
        region.origin = align_to(previous_origin + size_d, backend_ctx->alignment);
        region.size = size_qh;
        extra->qh = clCreateSubBuffer(
            extra_orig->data_device, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);
        previous_origin = region.origin;

        // Create subbuffer for qs.
        region.origin = align_to(previous_origin + size_qh, backend_ctx->alignment);
        region.size = size_qs;
        extra->qs = clCreateSubBuffer(
            extra_orig->data_device, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        // Adreno moe q5_0 kernel needs special transpose and unshuffling
        if (use_adreno_moe_kernels(backend_ctx, tensor)) {
            cl_kernel kernel = backend_ctx->repack.kernel_convert_block_q5_0_trans4_ns;

            int ne00 = tensor->ne[0];
            int ne01 = tensor->ne[1];
            int ne02 = tensor->ne[2];
            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_device));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->qs));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extra->qh));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &extra->d));
            CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int), &ne00));
            CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int), &ne01));

            size_t global_work_size[3] = {static_cast<size_t>(((ne01 + 63) / 64) * 64), static_cast<size_t>(ne00 / 32), static_cast<size_t>(ne02)};
            size_t local_work_size[3] = {64, 2, 1};

            cl_event evt;
            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &evt));
            CL_CHECK(clWaitForEvents(1, &evt));
            CL_CHECK(clReleaseMemObject(data_device));

            // Create image for Q
            cl_image_format img_format_qs = {CL_R, CL_UNSIGNED_INT32};
            cl_image_desc img_desc_qs = {
                CL_MEM_OBJECT_IMAGE1D_BUFFER,
                static_cast<size_t>(ggml_nelements(tensor) / 8),
                0, 0, 0, 0, 0, 0, 0,
                { extra->qs }
            };
            extra->qs_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_format_qs, &img_desc_qs, NULL, &err);
            tensor->extra = extra;

            // Generic dp4a MoE path
            {
                static const char * q5dp4a_env = getenv("GGML_OPENCL_Q5_MOE_DP4A");
                const bool q5dp4a = q5dp4a_env ? (atoi(q5dp4a_env) != 0)
                                               : (backend_ctx->adreno_gen == ADRENO_GPU_GEN::X2E);
                if (q5dp4a && ne02 > 1 && (ne00 % 32 == 0)) {
                    size_t nb32 = (size_t)ne00 / 32;
                    size_t sc_elems = (size_t)ne02 * ne01 * nb32 * 2;
                    size_t mn_elems = (size_t)ne02 * ne01 * nb32;
                    extra->scale = clCreateBuffer(context, CL_MEM_READ_WRITE, sc_elems * sizeof(cl_half), NULL, &err); CL_CHECK(err);
                    extra->min   = clCreateBuffer(context, CL_MEM_READ_WRITE, mn_elems * sizeof(cl_half), NULL, &err); CL_CHECK(err);
                    cl_kernel ek = backend_ctx->repack.kernel_moe_expand_scale_q5_0;
                    CL_CHECK(clSetKernelArg(ek, 0, sizeof(cl_mem), &extra->d));
                    CL_CHECK(clSetKernelArg(ek, 1, sizeof(cl_mem), &extra->scale));
                    CL_CHECK(clSetKernelArg(ek, 2, sizeof(cl_mem), &extra->min));
                    CL_CHECK(clSetKernelArg(ek, 3, sizeof(int), &ne00));
                    CL_CHECK(clSetKernelArg(ek, 4, sizeof(int), &ne01));
                    size_t eg[3] = { (size_t)(((ne01 + 63) / 64) * 64), nb32, (size_t)ne02 };
                    size_t el[3] = { 64, 1, 1 };
                    cl_event evt;
                    CL_CHECK(clEnqueueNDRangeKernel(queue, ek, 3, NULL, eg, el, 0, NULL, &evt));
                    CL_CHECK(clWaitForEvents(1, &evt));
                }
            }

            return;
        }
#endif // GGML_OPENCL_USE_ADRENO_KERNELS

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        if (use_adreno_kernels(backend_ctx, tensor)) {
            cl_kernel kernel = backend_ctx->repack.kernel_convert_block_q5_0_noshuffle;
            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_device));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->qs));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extra->qh));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &extra->d));

            size_t global_work_size[] = {(size_t)ggml_nelements(tensor)/ggml_blck_size(tensor->type), 1, 1};
            size_t local_work_size[] = {64, 1, 1};

            cl_event evt;
            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &evt));
            CL_CHECK(clWaitForEvents(1, &evt));
            CL_CHECK(clReleaseMemObject(data_device));

            tensor->extra = extra;

            int M = tensor->ne[1];
            int K = tensor->ne[0];
            GGML_ASSERT(K % 32 == 0);

            // Transpose qs as ushort
            transpose_2d_as_16b(backend_ctx, extra->qs, extra->qs, size_qs, K/4, M);
            // Transpose qh as uchar
            transpose_2d_as_8b(backend_ctx, extra->qh, extra->qh, size_qh, K/8, M);
            // Transpose d as ushort
            transpose_2d_as_16b(backend_ctx, extra->d, extra->d, size_d, K/32, M);

            return;
        }
#endif // GGML_OPENCL_USE_ADRENO_KERNELS
        cl_kernel kernel = backend_ctx->repack.kernel_convert_block_q5_0;
        cl_ulong n_blk = ggml_nelements(tensor)/ggml_blck_size(tensor->type);
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_device));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->qs));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extra->qh));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &extra->d));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_ulong), &n_blk));

        size_t global_work_size[] = {(size_t)CEIL_DIV(n_blk, 64) * 64, 1, 1};
        size_t local_work_size[] = {64, 1, 1};

        cl_event evt;
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &evt));
        CL_CHECK(clWaitForEvents(1, &evt));
        CL_CHECK(clReleaseMemObject(data_device));

        tensor->extra = extra;
        return;
    }
    if (tensor->type == GGML_TYPE_Q5_1) {
        ggml_tensor_extra_cl * extra_orig = (ggml_tensor_extra_cl *)tensor->extra;
        GGML_ASSERT(extra_orig && "Tesnors in OpenCL backend should have been allocated and initialized");

        // Allocate the new extra and create aliases from the original.
        ggml_backend_opencl_buffer_context * ctx = (ggml_backend_opencl_buffer_context *) buffer->context;
        ggml_tensor_extra_cl_q5_1 * extra = ctx->ggml_opencl_alloc_temp_tensor_extra_q5_1();

        size_t size_d = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*sizeof(ggml_fp16_t);
        size_t size_m = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*sizeof(ggml_fp16_t);
        size_t size_qs = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*ggml_blck_size(tensor->type)/2;
        size_t size_qh = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*sizeof(int32_t);
        GGML_ASSERT(size_d + size_m + size_qs + size_qh == ggml_nbytes(tensor) && "Incorrect tensor size");

        cl_int err;
        cl_mem data_device = ggml_cl_create_temp_upload_buffer(context, queue, ggml_nbytes(tensor), data, tensor->name);
        GGML_ASSERT(data_device != NULL && "set_tensor: temp upload buffer alloc failed");

        cl_buffer_region region;

        // The original tensor memory is divided into scales and quants, i.e.,
        // we first store scales, mins, then quants.
        // Create subbuffer for scales.
        region.origin = align_to(extra_orig->offset + tensor->view_offs + offset, backend_ctx->alignment);
        region.size = size_d;
        extra->d = clCreateSubBuffer(
            extra_orig->data_device, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);
        auto previous_origin = region.origin;

        // Create subbuffer for mins.
        region.origin = align_to(previous_origin + size_d, backend_ctx->alignment);
        region.size = size_m;
        extra->m = clCreateSubBuffer(
            extra_orig->data_device, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);
        previous_origin = region.origin;

        // Create subbuffer for qh.
        region.origin = align_to(previous_origin + size_m, backend_ctx->alignment);
        region.size = size_qh;
        extra->qh = clCreateSubBuffer(
            extra_orig->data_device, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);
        previous_origin = region.origin;

        // Create subbuffer for qs.
        region.origin = align_to(previous_origin + size_qh, backend_ctx->alignment);
        region.size = size_qs;
        extra->qs = clCreateSubBuffer(
            extra_orig->data_device, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        // Adreno moe q5_1 kernel needs special transpose and unshuffling
        if (use_adreno_moe_kernels(backend_ctx, tensor)) {
            cl_kernel kernel = backend_ctx->repack.kernel_convert_block_q5_1_trans4_ns;

            int ne00 = tensor->ne[0];
            int ne01 = tensor->ne[1];
            int ne02 = tensor->ne[2];
            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_device));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->qs));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extra->qh));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &extra->d));
            CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), &extra->m));
            CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int), &ne00));
            CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int), &ne01));

            size_t global_work_size[3] = {static_cast<size_t>(((ne01 + 63) / 64) * 64), static_cast<size_t>(ne00 / 32), static_cast<size_t>(ne02)};
            size_t local_work_size[3] = {64, 2, 1};

            cl_event evt;
            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &evt));
            CL_CHECK(clWaitForEvents(1, &evt));
            CL_CHECK(clReleaseMemObject(data_device));

            // Create image for Q
            cl_image_format img_format_qs = {CL_R, CL_UNSIGNED_INT32};
            cl_image_desc img_desc_qs = {
                CL_MEM_OBJECT_IMAGE1D_BUFFER,
                static_cast<size_t>(ggml_nelements(tensor) / 8),
                0, 0, 0, 0, 0, 0, 0,
                { extra->qs }
            };
            extra->qs_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_format_qs, &img_desc_qs, NULL, &err);
            tensor->extra = extra;

            return;
        }
#endif // GGML_OPENCL_USE_ADRENO_KERNELS

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        if (use_adreno_kernels(backend_ctx, tensor)) {
            cl_kernel kernel = backend_ctx->repack.kernel_convert_block_q5_1_noshuffle;
            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_device));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->qs));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extra->qh));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &extra->d));
            CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), &extra->m));

            size_t global_work_size[] = {(size_t)ggml_nelements(tensor)/ggml_blck_size(tensor->type), 1, 1};
            size_t local_work_size[] = {64, 1, 1};

            cl_event evt;
            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &evt));
            CL_CHECK(clWaitForEvents(1, &evt));
            CL_CHECK(clReleaseMemObject(data_device));

            tensor->extra = extra;

            int M = tensor->ne[1];
            int K = tensor->ne[0];
            GGML_ASSERT(K % 32 == 0);

            // Transpose qs as ushort
            transpose_2d_as_16b(backend_ctx, extra->qs, extra->qs, size_qs, K/4, M);
            // Transpose qh as uchar
            transpose_2d_as_8b(backend_ctx, extra->qh, extra->qh, size_qh, K/8, M);
            // Transpose d as ushort
            transpose_2d_as_16b(backend_ctx, extra->d, extra->d, size_d, K/32, M);
            // Transpose m as ushort
            transpose_2d_as_16b(backend_ctx, extra->m, extra->m, size_m, K/32, M);

            return;
        }
#endif // GGML_OPENCL_USE_ADRENO_KERNELS
        cl_kernel kernel = backend_ctx->repack.kernel_convert_block_q5_1;
        cl_ulong n_blk = ggml_nelements(tensor)/ggml_blck_size(tensor->type);
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_device));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->qs));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extra->qh));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &extra->d));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), &extra->m));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_ulong), &n_blk));

        size_t global_work_size[] = {(size_t)CEIL_DIV(n_blk, 64) * 64, 1, 1};
        size_t local_work_size[] = {64, 1, 1};

        cl_event evt;
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &evt));
        CL_CHECK(clWaitForEvents(1, &evt));
        CL_CHECK(clReleaseMemObject(data_device));

        tensor->extra = extra;
        return;
    }
    if (tensor->type == GGML_TYPE_MXFP4) {
        ggml_tensor_extra_cl * extra_orig = (ggml_tensor_extra_cl *)tensor->extra;
        GGML_ASSERT(extra_orig && "Tesnors in OpenCL backend should have been allocated and initialized");

        // Allocate the new extra and create aliases from the original.
        ggml_backend_opencl_buffer_context * ctx = (ggml_backend_opencl_buffer_context *) buffer->context;
        ggml_tensor_extra_cl_mxfp4 * extra = ctx->ggml_opencl_alloc_temp_tensor_extra_mxfp4();

        size_t size_e = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*sizeof(char);
        size_t size_q = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*ggml_blck_size(tensor->type)/2;
        GGML_ASSERT(size_e + size_q == ggml_nbytes(tensor) && "Incorrect tensor size");

        cl_int err;
        cl_mem data_device = ggml_cl_create_temp_upload_buffer(context, queue, ggml_nbytes(tensor), data, tensor->name);
        GGML_ASSERT(data_device != NULL && "set_tensor: temp upload buffer alloc failed");

        // The original tensor memory is divided into scales and quants, i.e.,
        // we first store scales, then quants.
        cl_buffer_region region;

        // Create subbuffer for scales.
        region.origin = align_to(extra_orig->offset + tensor->view_offs + offset, backend_ctx->alignment);
        region.size = size_e;
        extra->e = clCreateSubBuffer(
            extra_orig->data_device, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);
        auto previous_origin = region.origin;

        // Create subbuffer for quants.
        region.origin = align_to(previous_origin + size_e, backend_ctx->alignment);
        region.size = size_q;
        extra->q = clCreateSubBuffer(
            extra_orig->data_device, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        // Adreno moe mxfp4 kernel needs special transpose and unshuffling
        if (use_adreno_moe_kernels(backend_ctx, tensor)) {
            cl_kernel kernel = backend_ctx->repack.kernel_convert_block_mxfp4_trans4_ns;

            int ne00 = tensor->ne[0];
            int ne01 = tensor->ne[1];
            int ne02 = tensor->ne[2];
            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_device));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->q));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extra->e));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int), &ne00));
            CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int), &ne01));

            size_t global_work_size[3] = {static_cast<size_t>(((ne01 + 63) / 64) * 64), static_cast<size_t>(ne00 / 32), static_cast<size_t>(ne02)};
            size_t local_work_size[3] = {64, 2, 1};

            cl_event evt;
            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &evt));
            CL_CHECK(clWaitForEvents(1, &evt));
            CL_CHECK(clReleaseMemObject(data_device));
            tensor->extra = extra;

            // Create image for Q
            cl_image_format img_format_q = {CL_R, CL_UNSIGNED_INT32};
            cl_image_desc img_desc_q = {
                CL_MEM_OBJECT_IMAGE1D_BUFFER,
                static_cast<size_t>(ggml_nelements(tensor) / 8),
                0, 0, 0, 0, 0, 0, 0,
                { extra->q }
            };
            extra->q_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_format_q, &img_desc_q, NULL, &err);
            tensor->extra = extra;

            return;
        }

#endif // GGML_OPENCL_USE_ADRENO_KERNELS
        cl_kernel kernel = backend_ctx->repack.kernel_convert_block_mxfp4;

        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_device));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->q));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extra->e));

        size_t global_work_size[3] = {(size_t)ggml_nelements(tensor)/ggml_blck_size(tensor->type), 1, 1};
        size_t local_work_size[3] = {64, 1, 1};

        cl_event evt;
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &evt));
        CL_CHECK(clWaitForEvents(1, &evt));
        CL_CHECK(clReleaseMemObject(data_device));

        // Create image for Q
        cl_image_format img_format_q = {CL_RG, CL_UNSIGNED_INT32};
        cl_image_desc img_desc_q = {
            CL_MEM_OBJECT_IMAGE1D_BUFFER,
            static_cast<size_t>(ggml_nelements(tensor)/32*2),
            0, 0, 0, 0, 0, 0, 0,
            { extra->q }
        };
        extra->q_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_format_q, &img_desc_q, NULL, &err);
        tensor->extra = extra;

        return;
    }
    if (tensor->type == GGML_TYPE_Q8_0) {
        // Views share the parent's buffer; parent owns SoA conversion.
        if (tensor->view_src != nullptr || !ggml_is_contiguous(tensor)) {
            return;
        }

        ggml_tensor_extra_cl * extra_orig = (ggml_tensor_extra_cl *)tensor->extra;
        GGML_ASSERT(extra_orig && "Tesnors in OpenCL backend should have been allocated and initialized");

        // Allocate the new extra and create aliases from the original.
        ggml_backend_opencl_buffer_context * ctx = (ggml_backend_opencl_buffer_context *) buffer->context;
        ggml_tensor_extra_cl_q8_0 * extra = ctx->ggml_opencl_alloc_temp_tensor_extra_q8_0();

        size_t size_d = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*sizeof(ggml_fp16_t);
        size_t size_q = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*(ggml_blck_size(tensor->type)*sizeof(char));
        GGML_ASSERT(size_d + size_q == ggml_nbytes(tensor) && "Incorrect tensor size");

        cl_int err;
        cl_mem data_device = ggml_cl_create_temp_upload_buffer(context, queue, ggml_nbytes(tensor), data, tensor->name);
        GGML_ASSERT(data_device != NULL && "set_tensor: temp upload buffer alloc failed");

        // The original tensor memory is divided into scales and quants, i.e.,
        // we first store scales, then quants.
        cl_buffer_region region;

        // Create subbuffer for scales.
        region.origin = align_to(extra_orig->offset + tensor->view_offs + offset, backend_ctx->alignment);
        region.size = size_d;
        extra->d = clCreateSubBuffer(
            extra_orig->data_device, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);
        auto previous_origin = region.origin;

        // Create subbuffer for quants.
        region.origin = align_to(previous_origin + size_d, backend_ctx->alignment);
        region.size = size_q;
        extra->q = clCreateSubBuffer(
            extra_orig->data_device, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);

        cl_kernel kernel = backend_ctx->repack.kernel_convert_block_q8_0;

        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_device));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->q));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extra->d));

        size_t global_work_size[] = {(size_t)ggml_nelements(tensor)/ggml_blck_size(tensor->type), 1, 1};
        size_t local_work_size[] = {64, 1, 1};

        cl_event evt;
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &evt));
        CL_CHECK(clWaitForEvents(1, &evt));
        CL_CHECK(clReleaseMemObject(data_device));

        tensor->extra = extra;
        ctx->q8_0_soa_tensors.insert(tensor);

        // Generic dp4a MoE path (opt-in GGML_OPENCL_Q8_MOE_DP4A)
#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        {
            static const char * q8dp4a_env = getenv("GGML_OPENCL_Q8_MOE_DP4A");
            const bool q8dp4a = q8dp4a_env ? (atoi(q8dp4a_env) != 0)
                                           : (backend_ctx->adreno_gen == ADRENO_GPU_GEN::X2E);
            if (q8dp4a && tensor->ne[2] > 1 && (tensor->ne[0] % 32 == 0)) {
                int ne00 = (int)tensor->ne[0];
                int ne01 = (int)tensor->ne[1];
                int ne02 = (int)tensor->ne[2];
                size_t nb32 = (size_t)ne00 / 32;
                size_t scale_elems = (size_t)ne02 * ne01 * nb32 * 2;   // 2 per-16-seg scales / 32-block
                extra->scale = clCreateBuffer(context, CL_MEM_READ_WRITE, scale_elems * sizeof(cl_half), NULL, &err);
                CL_CHECK(err);
                cl_kernel ek = backend_ctx->repack.kernel_moe_expand_scale_q8_0;
                CL_CHECK(clSetKernelArg(ek, 0, sizeof(cl_mem), &extra->d));
                CL_CHECK(clSetKernelArg(ek, 1, sizeof(cl_mem), &extra->scale));
                CL_CHECK(clSetKernelArg(ek, 2, sizeof(int), &ne00));
                CL_CHECK(clSetKernelArg(ek, 3, sizeof(int), &ne01));
                size_t eg[3] = { (size_t)(((ne01 + 63) / 64) * 64), nb32, (size_t)ne02 };
                size_t el[3] = { 64, 1, 1 };
                cl_event evt;
                CL_CHECK(clEnqueueNDRangeKernel(queue, ek, 3, NULL, eg, el, 0, NULL, &evt));
                CL_CHECK(clWaitForEvents(1, &evt));
            }
        }
#endif

        // Transpose the weights and scales
#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        if (enable_adreno_trans_weight(backend_ctx, tensor)) {

            int M = tensor->ne[1];   // ne01
            int K = tensor->ne[0];   // ne00

            GGML_ASSERT(K % 32 == 0);
            GGML_ASSERT(M % 4 == 0);
            GGML_ASSERT(tensor->ne[2] == 1);
            GGML_ASSERT(tensor->ne[3] == 1);

            transpose_2d_as_32b(backend_ctx, extra->q, extra->q, size_q, K/4,  M);
            transpose_2d_as_16b(backend_ctx, extra->d, extra->d, size_d, K/32, M);
        } // end transpose
#endif // GGML_OPENCL_USE_ADRENO_KERNELS

        return;
    }
    if (tensor->type == GGML_TYPE_IQ4_NL) {
        ggml_tensor_extra_cl * extra_orig = (ggml_tensor_extra_cl *)tensor->extra;
        GGML_ASSERT(extra_orig && "Tensors in OpenCL backend should have been allocated and initialized");

        ggml_backend_opencl_buffer_context * ctx = (ggml_backend_opencl_buffer_context *) buffer->context;
        ggml_tensor_extra_cl_iq4_nl * extra = ctx->ggml_opencl_alloc_temp_tensor_extra_iq4_nl();

        size_t size_d = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*sizeof(ggml_fp16_t);
        size_t size_q = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*(ggml_blck_size(tensor->type)/2);
        GGML_ASSERT(size_d + size_q == ggml_nbytes(tensor) && "Incorrect tensor size");

        cl_int err;
        cl_mem data_device = ggml_cl_create_temp_upload_buffer(context, queue, ggml_nbytes(tensor), data, tensor->name);
        GGML_ASSERT(data_device != NULL && "set_tensor: temp upload buffer alloc failed");

        cl_buffer_region region;

        // Create subbuffer for scales.
        region.origin = align_to(extra_orig->offset + tensor->view_offs + offset, backend_ctx->alignment);
        region.size = size_d;
        extra->d = clCreateSubBuffer(
            extra_orig->data_device, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);
        auto previous_origin = region.origin;

        // Create subbuffer for quants.
        region.origin = align_to(previous_origin + size_d, backend_ctx->alignment);
        region.size = size_q;
        extra->q = clCreateSubBuffer(
            extra_orig->data_device, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);

    #ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        cl_kernel kernel = backend_ctx->repack.kernel_convert_block_iq4_nl;
        if (use_adreno_kernels(backend_ctx, tensor)) {
            kernel = backend_ctx->repack.kernel_convert_block_iq4_nl_noshuffle;
        }
    #else
        cl_kernel kernel = backend_ctx->repack.kernel_convert_block_iq4_nl;
    #endif
        cl_ulong n_blk = ggml_nelements(tensor)/ggml_blck_size(tensor->type);
        cl_uchar mask_0F = 0x0F;
        cl_uchar mask_F0 = 0xF0;

        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_device));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->q));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extra->d));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_uchar), &mask_0F));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_uchar), &mask_F0));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_ulong), &n_blk));

        size_t global_work_size[] = {(size_t)CEIL_DIV(n_blk, 64)*64, 1, 1};
        size_t local_work_size[] = {64, 1, 1};

        cl_event evt;
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &evt));
        CL_CHECK(clWaitForEvents(1, &evt));
        CL_CHECK(clReleaseMemObject(data_device));

        tensor->extra = extra;

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        if (use_adreno_kernels(backend_ctx, tensor)) {
            int M = tensor->ne[1];
            int K = tensor->ne[0];
            GGML_ASSERT(K % 32 == 0);

            // Transpose q as ushort
            transpose_2d_as_16b(backend_ctx, extra->q, extra->q, size_q, K/4, M);
            // Transpose d as ushort
            transpose_2d_as_16b(backend_ctx, extra->d, extra->d, size_d, K/32, M);
        }
#endif
        return;
    }
    if (tensor->type == GGML_TYPE_Q4_K) {
        ggml_tensor_extra_cl * extra_orig = (ggml_tensor_extra_cl *)tensor->extra;
        GGML_ASSERT(extra_orig && "Tesnors in OpenCL backend should have been allocated and initialized");

        // Allocate the new extra and create aliases from the original.
        ggml_backend_opencl_buffer_context * ctx = (ggml_backend_opencl_buffer_context *) buffer->context;
        ggml_tensor_extra_cl_q4_K * extra = ctx->ggml_opencl_alloc_temp_tensor_extra_q4_K();

        size_t size_d = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*sizeof(ggml_fp16_t);
        size_t size_dm = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*sizeof(ggml_fp16_t);
        size_t size_s = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*(3 * ggml_blck_size(tensor->type) / 64);
        size_t size_q = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*ggml_blck_size(tensor->type)/2;
        GGML_ASSERT(size_d + size_dm + size_s + size_q == ggml_nbytes(tensor) && "Incorrect tensor size");

        cl_int err;
        cl_mem data_device = ggml_cl_create_temp_upload_buffer(context, queue, ggml_nbytes(tensor), data, tensor->name);
        GGML_ASSERT(data_device != NULL && "q4_K set_tensor: temp upload buffer alloc failed");

        cl_buffer_region region;

        // Create subbuffer for d.
        region.origin = align_to(extra_orig->offset + tensor->view_offs + offset, backend_ctx->alignment);
        region.size = size_d;
        extra->d = clCreateSubBuffer(
            extra_orig->data_device, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);
        auto previous_origin = region.origin;

        // Create subbuffer for mins.
        region.origin = align_to(previous_origin + size_d, backend_ctx->alignment);
        region.size = size_dm;
        extra->dm = clCreateSubBuffer(
            extra_orig->data_device, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);
        previous_origin = region.origin;

        // Create subbuffer for s.
        region.origin = align_to(previous_origin + size_dm, backend_ctx->alignment);
        region.size = size_s;
        extra->s = clCreateSubBuffer(
            extra_orig->data_device, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);
        previous_origin = region.origin;

        // Create subbuffer for quants.
        region.origin = align_to(previous_origin + size_s, backend_ctx->alignment);
        region.size = size_q;
        extra->q = clCreateSubBuffer(
            extra_orig->data_device, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        if (use_adreno_moe_kernels(backend_ctx, tensor)) {
            cl_kernel kernel = backend_ctx->repack.kernel_convert_block_q4_k_trans4_ns;

            int ne00 = tensor->ne[0];
            int ne01 = tensor->ne[1];
            int ne02 = tensor->ne[2];

            cl_uchar mask_0F = 0x0F;
            cl_uchar mask_F0 = 0xF0;
            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_device));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->q));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extra->d));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &extra->dm));
            CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), &extra->s));
            CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int), &ne00));
            CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int), &ne01));
            CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_uchar), &mask_0F));
            CL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_uchar), &mask_F0));

            size_t global_work_size[] = {static_cast<size_t>(((ne01 + 63) / 64) * 64), static_cast<size_t>(ne00 / 256), static_cast<size_t>(ne02)};
            size_t local_work_size[] = {64, 1, 1};

            cl_event evt;
            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &evt));
            CL_CHECK(clWaitForEvents(1, &evt));
            CL_CHECK(clReleaseMemObject(data_device));

            cl_image_format img_format_q = {CL_R, CL_UNSIGNED_INT32};
            cl_image_desc img_desc_q = {
                CL_MEM_OBJECT_IMAGE1D_BUFFER,
                static_cast<size_t>(ggml_nelements(tensor) / 8),
                0, 0, 0, 0, 0, 0, 0,
                { extra->q }
            };
            extra->q_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_format_q, &img_desc_q, NULL, &err);
            CL_CHECK(err);
            tensor->extra = extra;

            return;
        }
#endif // GGML_OPENCL_USE_ADRENO_KERNELS

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        // Tiled-wide convert for the long-vocab lm_head/embed (opt-in). The embed/
        // output q4_K weight (token_embd.weight, ne1=vocab) is NOT matched by
        // use_adreno_moe_kernels, so it lands here in the general branch. Produce
        // the final 64-row-tiled canonical layout directly into q/d/dm/s (buffer
        // sizes already match), read back by kernel_gemv_noshuffle_q4_k_f32_tiled.
        if (use_q4k_tiled(backend_ctx, tensor)) {
            cl_kernel tk = backend_ctx->repack.kernel_convert_block_q4_k_tiled_ns;

            int ne00 = tensor->ne[0];
            int ne01 = tensor->ne[1];
            int ne02 = tensor->ne[2];

            CL_CHECK(clSetKernelArg(tk, 0, sizeof(cl_mem), &data_device));
            CL_CHECK(clSetKernelArg(tk, 1, sizeof(cl_mem), &extra->q));
            CL_CHECK(clSetKernelArg(tk, 2, sizeof(cl_mem), &extra->d));
            CL_CHECK(clSetKernelArg(tk, 3, sizeof(cl_mem), &extra->dm));
            CL_CHECK(clSetKernelArg(tk, 4, sizeof(cl_mem), &extra->s));
            CL_CHECK(clSetKernelArg(tk, 5, sizeof(int), &ne00));
            CL_CHECK(clSetKernelArg(tk, 6, sizeof(int), &ne01));

            size_t gws[] = {static_cast<size_t>(((ne01 + 63) / 64) * 64), static_cast<size_t>(ne00 / 256), static_cast<size_t>(ne02)};
            size_t lws[] = {64, 1, 1};

            cl_event tevt;
            CL_CHECK(clEnqueueNDRangeKernel(queue, tk, 3, NULL, gws, lws, 0, NULL, &tevt));
            CL_CHECK(clWaitForEvents(1, &tevt));
            CL_CHECK(clReleaseMemObject(data_device));

            extra->q_img = nullptr;
            tensor->extra = extra;
            return;
        }

        cl_kernel kernel = backend_ctx->repack.kernel_convert_block_q4_K;
        if (use_adreno_kernels(backend_ctx, tensor) && !use_flat_gemv_for_large_m_q4_K(backend_ctx, tensor)) {
            kernel = backend_ctx->repack.kernel_convert_block_q4_K_noshuffle;
        }
#else
        cl_kernel kernel = backend_ctx->repack.kernel_convert_block_q4_K;
#endif // GGML_OPENCL_USE_ADRENO_KERNELS

        cl_uchar mask_0F = 0x0F;
        cl_uchar mask_F0 = 0xF0;

        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_device));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->q));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extra->s));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &extra->d));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), &extra->dm));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_uchar), &mask_0F));
        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_uchar), &mask_F0));

        size_t global_work_size[] = {(size_t)ggml_nelements(tensor)/ggml_blck_size(tensor->type), 1, 1};
        size_t local_work_size[] = {64, 1, 1};

        cl_event evt;
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &evt));
        CL_CHECK(clWaitForEvents(1, &evt));
        CL_CHECK(clReleaseMemObject(data_device));

        tensor->extra  = extra;
#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        if (use_adreno_kernels(backend_ctx, tensor) && !use_flat_gemv_for_large_m_q4_K(backend_ctx, tensor)) {

            int M = tensor->ne[1];
            int K = tensor->ne[0];

            GGML_ASSERT(K % 32 == 0);

            if (use_q4_k_bin_kernels(backend_ctx, tensor)) {
                cl_int err;
                cl_image_format wimg_fmt;
                cl_image_desc   wimg_desc;

                // transpose quants as 32-bit words (M-first)
                GGML_ASSERT(M % 64 == 0);
                transpose_2d_as_32b(backend_ctx, extra->q, extra->q, size_q, K/8, M);

                wimg_fmt = { CL_R, CL_UNSIGNED_INT32 };
                memset(&wimg_desc, 0, sizeof(wimg_desc));
                wimg_desc.image_type  = CL_MEM_OBJECT_IMAGE1D_BUFFER;
                wimg_desc.image_width = (size_t)M * K / 8;
                wimg_desc.buffer      = extra->q;
                CL_CHECK((extra->q_img = clCreateImage(context, CL_MEM_READ_ONLY, &wimg_fmt, &wimg_desc, NULL, &err), err));
            } else {
                // Transpose q as ushort
                transpose_2d_as_16b(backend_ctx, extra->q, extra->q, size_q, K/4, M);
            }
            transpose_2d_as_16b(backend_ctx, extra->d, extra->d, size_d, K/256, M);
            transpose_2d_as_16b(backend_ctx, extra->dm, extra->dm, size_dm, K/256, M);

            // Transpose s as uchar
            transpose_2d_as_8b(backend_ctx, extra->s, extra->s, size_s, K/256*12, M, true, true);
        }
#endif // GGML_OPENCL_USE_ADRENO_KERNELS
        return;
    }
    if (tensor->type == GGML_TYPE_Q5_K) {
        ggml_tensor_extra_cl * extra_orig = (ggml_tensor_extra_cl *)tensor->extra;
        GGML_ASSERT(extra_orig && "Tesnors in OpenCL backend should have been allocated and initialized");

        // Allocate the new extra and create aliases from the original.
        ggml_backend_opencl_buffer_context * ctx = (ggml_backend_opencl_buffer_context *) buffer->context;
        ggml_tensor_extra_cl_q5_K * extra = ctx->ggml_opencl_alloc_temp_tensor_extra_q5_K();

        size_t size_q  = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*ggml_blck_size(tensor->type)/2;
        size_t size_qh = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*ggml_blck_size(tensor->type)/8;
        size_t size_s  = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*(3*ggml_blck_size(tensor->type)/64);
        size_t size_d  = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*sizeof(ggml_fp16_t);
        size_t size_dm = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*sizeof(ggml_fp16_t);
        GGML_ASSERT(size_q + size_qh + size_s + size_d + size_dm == ggml_nbytes(tensor) &&
            "Incorrect tensor size");

        cl_int err;
        cl_mem data_device = ggml_cl_create_temp_upload_buffer(context, queue, ggml_nbytes(tensor), data, tensor->name);
        GGML_ASSERT(data_device != NULL && "q5_K set_tensor: temp upload buffer alloc failed");

        cl_buffer_region region;

        // Create subbuffer for d.
        region.origin = align_to(extra_orig->offset + tensor->view_offs + offset, backend_ctx->alignment);
        region.size = size_d;
        extra->d = clCreateSubBuffer(
            extra_orig->data_device, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);
        auto previous_origin = region.origin;

        // Create subbuffer for dm.
        region.origin = align_to(previous_origin + size_d, backend_ctx->alignment);
        region.size = size_dm;
        extra->dm = clCreateSubBuffer(
            extra_orig->data_device, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);
        previous_origin = region.origin;

        // Create subbuffer for s.
        region.origin = align_to(previous_origin + size_dm, backend_ctx->alignment);
        region.size = size_s;
        extra->s = clCreateSubBuffer(
            extra_orig->data_device, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);
        previous_origin = region.origin;

        // Create subbuffer for q (lower 4 bits)
        region.origin = align_to(previous_origin + size_s, backend_ctx->alignment);
        region.size = size_q;
        extra->q = clCreateSubBuffer(
            extra_orig->data_device, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);
        previous_origin = region.origin;

        // Create subbuffer for qh (upper 1 bit)
        region.origin = align_to(previous_origin + size_q, backend_ctx->alignment);
        region.size = size_qh;
        CL_CHECK((extra->qh = clCreateSubBuffer(extra_orig->data_device, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));
        CL_CHECK(err);

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        if (use_adreno_moe_kernels(backend_ctx, tensor)) {
            cl_kernel kernel = backend_ctx->repack.kernel_convert_block_q5_k_trans4_ns;

            int ne00 = tensor->ne[0];
            int ne01 = tensor->ne[1];
            int ne02 = tensor->ne[2];

            cl_uchar mask_0F = 0x0F;
            cl_uchar mask_F0 = 0xF0;
            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_device));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->q));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extra->qh));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &extra->d));
            CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), &extra->dm));
            CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_mem), &extra->s));
            CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int), &ne00));
            CL_CHECK(clSetKernelArg(kernel, 7, sizeof(int), &ne01));
            CL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_uchar), &mask_0F));
            CL_CHECK(clSetKernelArg(kernel, 9, sizeof(cl_uchar), &mask_F0));

            size_t global_work_size[] = {static_cast<size_t>(((ne01 + 63) / 64) * 64), static_cast<size_t>(ne00 / 256), static_cast<size_t>(ne02)};
            size_t local_work_size[] = {64, 1, 1};

            cl_event evt;
            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &evt));
            CL_CHECK(clWaitForEvents(1, &evt));
            CL_CHECK(clReleaseMemObject(data_device));

            cl_image_format img_format_q = {CL_R, CL_UNSIGNED_INT32};
            cl_image_desc img_desc_q = {
                CL_MEM_OBJECT_IMAGE1D_BUFFER,
                static_cast<size_t>(ggml_nelements(tensor) / 8),
                0, 0, 0, 0, 0, 0, 0,
                { extra->q }
            };
            extra->q_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_format_q, &img_desc_q, NULL, &err);
            CL_CHECK(err);
            tensor->extra = extra;

            // Generic dp4a MoE path
            {
                static const char * q5kdp4a_env = getenv("GGML_OPENCL_Q5K_MOE_DP4A");
                const bool q5kdp4a = q5kdp4a_env ? (atoi(q5kdp4a_env) != 0)
                                                 : (backend_ctx->adreno_gen == ADRENO_GPU_GEN::X2E);
                if (q5kdp4a && ne02 > 1 && (ne00 % 256 == 0)) {
                    size_t nb32     = (size_t)ne00 / 32;
                    size_t sc_elems = (size_t)ne02 * ne01 * nb32 * 2;
                    size_t mn_elems = (size_t)ne02 * ne01 * nb32;
                    extra->scale = clCreateBuffer(context, CL_MEM_READ_WRITE, sc_elems * sizeof(cl_half), NULL, &err); CL_CHECK(err);
                    extra->min   = clCreateBuffer(context, CL_MEM_READ_WRITE, mn_elems * sizeof(cl_half), NULL, &err); CL_CHECK(err);
                    cl_kernel ek = backend_ctx->repack.kernel_moe_expand_scale_q5_K;
                    CL_CHECK(clSetKernelArg(ek, 0, sizeof(cl_mem), &extra->s));
                    CL_CHECK(clSetKernelArg(ek, 1, sizeof(cl_mem), &extra->d));
                    CL_CHECK(clSetKernelArg(ek, 2, sizeof(cl_mem), &extra->dm));
                    CL_CHECK(clSetKernelArg(ek, 3, sizeof(cl_mem), &extra->scale));
                    CL_CHECK(clSetKernelArg(ek, 4, sizeof(cl_mem), &extra->min));
                    CL_CHECK(clSetKernelArg(ek, 5, sizeof(int), &ne00));
                    CL_CHECK(clSetKernelArg(ek, 6, sizeof(int), &ne01));
                    size_t eg[3] = { (size_t)(((ne01 + 63) / 64) * 64), (size_t)(ne00 / 256), (size_t)ne02 };
                    size_t el[3] = { 64, 1, 1 };
                    cl_event evt;
                    CL_CHECK(clEnqueueNDRangeKernel(queue, ek, 3, NULL, eg, el, 0, NULL, &evt));
                    CL_CHECK(clWaitForEvents(1, &evt));
                }
            }

            return;
        }
#endif // GGML_OPENCL_USE_ADRENO_KERNELS

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        cl_kernel kernel = backend_ctx->repack.kernel_convert_block_q5_K;
        if (enable_adreno_trans_weight_q5_K(backend_ctx, tensor)) {
            kernel = backend_ctx->repack.kernel_convert_block_q5_K_noshuffle;
        }
#else
        cl_kernel kernel = backend_ctx->repack.kernel_convert_block_q5_K;
#endif

        cl_uchar mask_0F = 0x0F;
        cl_uchar mask_F0 = 0xF0;

        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &data_device));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),   &extra->q));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extra->qh));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem),   &extra->s));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem),   &extra->d));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_mem),   &extra->dm));
        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_uchar), &mask_0F));
        CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_uchar), &mask_F0));

        size_t global_work_size[] = {(size_t)ggml_nelements(tensor)/ggml_blck_size(tensor->type), 1, 1};
        size_t local_work_size[] = {64, 1, 1};

        cl_event evt;
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &evt));
        CL_CHECK(clWaitForEvents(1, &evt));
        CL_CHECK(clReleaseMemObject(data_device));

        extra->size_q  = size_q;
        extra->size_qh = size_qh;
        extra->size_s  = size_s;
        extra->size_d  = size_d;
        extra->size_dm = size_dm;

        tensor->extra = extra;
#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        if (enable_adreno_trans_weight_q5_K(backend_ctx, tensor)) {

            int M = tensor->ne[1];
            int K = tensor->ne[0];

            GGML_ASSERT(K % 32 == 0);

            // Transpose q, d, dm as ushort, qh as uchar
            transpose_2d_as_16b(backend_ctx, extra->q,  extra->q,  size_q,  K/4,   M);
            transpose_2d_as_8b (backend_ctx, extra->qh, extra->qh, size_qh, K/8,   M);
            transpose_2d_as_16b(backend_ctx, extra->d,  extra->d,  size_d,  K/256, M);
            transpose_2d_as_16b(backend_ctx, extra->dm, extra->dm, size_dm, K/256, M);
        }
#endif // GGML_OPENCL_USE_ADRENO_KERNELS
        return;
    }
    if (tensor->type == GGML_TYPE_Q6_K) {
        ggml_tensor_extra_cl * extra_orig = (ggml_tensor_extra_cl *)tensor->extra;
        GGML_ASSERT(extra_orig && "Tesnors in OpenCL backend should have been allocated and initialized");

        // Allocate the new extra and create aliases from the original.
        ggml_backend_opencl_buffer_context * ctx = (ggml_backend_opencl_buffer_context *) buffer->context;
        ggml_tensor_extra_cl_q6_K * extra = ctx->ggml_opencl_alloc_temp_tensor_extra_q6_K();

        size_t size_ql = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*ggml_blck_size(tensor->type)/2;
        size_t size_qh = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*ggml_blck_size(tensor->type)/4;
        size_t size_s  = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*ggml_blck_size(tensor->type)/16;
        size_t size_d  = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*sizeof(ggml_fp16_t);
        GGML_ASSERT(size_ql + size_qh + size_s + size_d == ggml_nbytes(tensor) &&
            "Incorrect tensor size");

        cl_int err;
        cl_mem data_device = ggml_cl_create_temp_upload_buffer(context, queue, ggml_nbytes(tensor), data, tensor->name);
        GGML_ASSERT(data_device != NULL && "q6_K set_tensor: temp upload buffer alloc failed");

        cl_buffer_region region;

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        // Adreno MoE Q6_K kernel needs special transposed layout
        if (use_adreno_moe_kernels(backend_ctx, tensor)) {
            size_t moe_size_ql = (size_t)(ggml_nelements(tensor) / 8) * sizeof(uint32_t);  // 4 bits per element
            size_t moe_size_qh = (size_t)(ggml_nelements(tensor) / 16) * sizeof(uint32_t); // 2 bits per element
            size_t moe_size_s  = size_s;
            size_t moe_size_d  = size_d;

            // Subbuffer for ql
            region.origin = align_to(extra_orig->offset + tensor->view_offs + offset, backend_ctx->alignment);
            region.size = moe_size_ql;
            CL_CHECK((extra->ql = clCreateSubBuffer(extra_orig->data_device, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));
            auto previous_origin = region.origin;

            // Subbuffer for qh
            region.origin = align_to(previous_origin + moe_size_ql, backend_ctx->alignment);
            region.size = moe_size_qh;
            CL_CHECK((extra->qh = clCreateSubBuffer(extra_orig->data_device, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));
            previous_origin = region.origin;

            // Subbuffer for scales
            region.origin = align_to(previous_origin + moe_size_qh, backend_ctx->alignment);
            region.size = moe_size_s;
            CL_CHECK((extra->s = clCreateSubBuffer(extra_orig->data_device, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));
            previous_origin = region.origin;

            // Subbuffer for d
            region.origin = align_to(previous_origin + moe_size_s, backend_ctx->alignment);
            region.size = moe_size_d;
            CL_CHECK((extra->d = clCreateSubBuffer(extra_orig->data_device, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

            cl_kernel kernel = backend_ctx->repack.kernel_convert_block_q6_k_trans4_ns;

            cl_uchar mask_0F = 0x0F;
            cl_uchar mask_F0 = 0xF0;

            int ne00 = tensor->ne[0];
            int ne01 = tensor->ne[1];
            int ne02 = tensor->ne[2];

            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_device));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->ql));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extra->qh));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &extra->d));
            CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), &extra->s));
            CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int), &ne00));
            CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int), &ne01));
            CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_uchar), &mask_0F));
            CL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_uchar), &mask_F0));

            size_t global_work_size[] = {static_cast<size_t>(((ne01 + 63) / 64) * 64), static_cast<size_t>(ne00 / 256), static_cast<size_t>(ne02)};
            size_t local_work_size[] = {64, 1, 1};

            cl_event evt;
            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &evt));
            CL_CHECK(clWaitForEvents(1, &evt));
            CL_CHECK(clReleaseMemObject(data_device));

            // Create image for ql
            cl_image_format img_format_ql = {CL_R, CL_UNSIGNED_INT32};
            cl_image_desc img_desc_ql = {
                CL_MEM_OBJECT_IMAGE1D_BUFFER,
                static_cast<size_t>(ggml_nelements(tensor) / 8),
                0, 0, 0, 0, 0, 0, 0,
                { extra->ql }
            };
            extra->ql_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_format_ql, &img_desc_ql, NULL, &err);
            tensor->extra = extra;

            return;
        }
#endif // GGML_OPENCL_USE_ADRENO_KERNELS

        // Subbuffer for ql
        region.origin = align_to(extra_orig->offset + tensor->view_offs + offset, backend_ctx->alignment);
        region.size = size_ql;
        CL_CHECK((extra->ql = clCreateSubBuffer(extra_orig->data_device, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));
        auto previous_origin = region.origin;

        // Subbuffer for qh
        region.origin = align_to(previous_origin + size_ql, backend_ctx->alignment);
        region.size = size_qh;
        CL_CHECK((extra->qh = clCreateSubBuffer(extra_orig->data_device, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));
        previous_origin = region.origin;

        // Subbuffer for scales
        region.origin = align_to(previous_origin + size_qh, backend_ctx->alignment);
        region.size = size_s;
        CL_CHECK((extra->s = clCreateSubBuffer(extra_orig->data_device, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));
        previous_origin = region.origin;

        // Create subbuffer for d.
        region.origin = align_to(previous_origin + size_s, backend_ctx->alignment);
        region.size = size_d;
        CL_CHECK((extra->d = clCreateSubBuffer(extra_orig->data_device, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));
        previous_origin = region.origin;

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        // Tiled-wide convert for the long-vocab lm_head/embed (opt-in). The embed
        // /output q6_K weight (e.g. token_embd.weight, ne1=vocab) is NOT matched by
        // use_adreno_moe_kernels, so it lands here in the general branch. Produce
        // the final 64-row-tiled canonical layout directly into ql/qh/s/d (buffer
        // sizes already match), read back by kernel_gemv_noshuffle_q6_K_f32_tiled.
        // Bypasses the plain-SOA convert + per-array transpose below.
        if (use_q6k_tiled(backend_ctx, tensor)) {
            cl_kernel kernel = backend_ctx->repack.kernel_convert_block_q6_k_tiled_ns;

            int ne00 = tensor->ne[0];
            int ne01 = tensor->ne[1];
            int ne02 = tensor->ne[2];

            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_device));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->ql));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extra->qh));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &extra->d));
            CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), &extra->s));
            CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int), &ne00));
            CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int), &ne01));

            size_t global_work_size[] = {static_cast<size_t>(((ne01 + 63) / 64) * 64), static_cast<size_t>(ne00 / 256), static_cast<size_t>(ne02)};
            size_t local_work_size[] = {64, 1, 1};

            cl_event evt;
            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &evt));
            CL_CHECK(clWaitForEvents(1, &evt));
            CL_CHECK(clReleaseMemObject(data_device));

            extra->size_ql = size_ql;
            extra->size_qh = size_qh;
            extra->size_s  = size_s;
            extra->size_d  = size_d;
            tensor->extra  = extra;
            return;
        }
#endif // GGML_OPENCL_USE_ADRENO_KERNELS

        // Flatten the weights
        cl_kernel kernel;
#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        kernel = backend_ctx->repack.kernel_convert_block_q6_K;
        if (use_adreno_kernels(backend_ctx, tensor) && !use_flat_gemv_for_large_m_q6_K(backend_ctx, tensor)) {
            kernel = backend_ctx->repack.kernel_convert_block_q6_K_noshuffle;
        }
#else
        kernel = backend_ctx->repack.kernel_convert_block_q6_K;
#endif // GGML_OPENCL_USE_ADRENO_KERNELS

        cl_uchar mask = 0xff;
        cl_ulong n_blk = ggml_nelements(tensor)/ggml_blck_size(tensor->type);
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &data_device));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),   &extra->ql));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extra->qh));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem),   &extra->s));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem),   &extra->d));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_uchar), &mask));
        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_ulong), &n_blk));

        size_t global_work_size[] = {(size_t)CEIL_DIV(n_blk, 64)*64, 1, 1};
        size_t local_work_size[] = {64, 1, 1};

        cl_event evt;
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &evt));
        CL_CHECK(clWaitForEvents(1, &evt));
        CL_CHECK(clReleaseMemObject(data_device));

        extra->size_ql = size_ql;
        extra->size_qh = size_qh;
        extra->size_s  = size_s;
        extra->size_d  = size_d;

        tensor->extra  = extra;

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        if (use_adreno_kernels(backend_ctx, tensor) && !use_flat_gemv_for_large_m_q6_K(backend_ctx, tensor)) {
            cl_int M = tensor->ne[1];   // ne01
            cl_int K = tensor->ne[0];   // ne00

            if (use_q6_k_bin_kernels(backend_ctx, tensor)) {
                GGML_ASSERT(K % 256 == 0);
                GGML_ASSERT(M % 64 == 0);

                transpose_2d_as_32b(backend_ctx, extra->ql, extra->ql, size_ql, K/8,  M);
                transpose_2d_as_32b(backend_ctx, extra->qh, extra->qh, size_qh, K/16, M);

                cl_image_format wimg_fmt = { CL_R, CL_UNSIGNED_INT32 };
                cl_image_desc   wimg_desc;
                memset(&wimg_desc, 0, sizeof(wimg_desc));
                wimg_desc.image_type  = CL_MEM_OBJECT_IMAGE1D_BUFFER;
                wimg_desc.image_width = static_cast<size_t>(ggml_nelements(tensor) / 8);
                wimg_desc.buffer      = extra->ql;
                CL_CHECK((extra->ql_img = clCreateImage(context, CL_MEM_READ_ONLY, &wimg_fmt, &wimg_desc, NULL, &err), err));

                memset(&wimg_desc, 0, sizeof(wimg_desc));
                wimg_desc.image_type  = CL_MEM_OBJECT_IMAGE1D_BUFFER;
                wimg_desc.image_width = static_cast<size_t>(ggml_nelements(tensor) / 16);
                wimg_desc.buffer      = extra->qh;
                CL_CHECK((extra->qh_img = clCreateImage(context, CL_MEM_READ_ONLY, &wimg_fmt, &wimg_desc, NULL, &err), err));
            } else {
                // Transpose ql as ushort
                transpose_2d_as_16b(backend_ctx,
                    extra->ql, extra->ql, size_ql, K/4, M);

                // Transpose qh as uchar
                transpose_2d_as_8b(backend_ctx,
                    extra->qh, extra->qh, size_qh, K/4, M);

                // Transpose s as ushort
                transpose_2d_as_16b(backend_ctx,
                    extra->s, extra->s, size_s, K/16/2, M);
            }

            // Transpose d as ushort
            transpose_2d_as_16b(backend_ctx,
                extra->d, extra->d, size_d, K/256, M);
        }
#endif // GGML_OPENCL_USE_ADRENO_KERNELS
        return;
    }
#endif // GGML_OPENCL_SOA_Q

    // convert bf16 to f16 and store as f16 in device buffer
    if (tensor->type == GGML_TYPE_BF16) {
        GGML_ASSERT(offset % sizeof(ggml_fp16_t) == 0 && size % sizeof(ggml_fp16_t) == 0
            && "Offset and size must be multiples of 2 for bf16 tensors");

        ggml_tensor_extra_cl * extra = (ggml_tensor_extra_cl *) tensor->extra;
        GGML_ASSERT(extra);

        cl_ulong n_elements = size / sizeof(ggml_fp16_t);
        cl_ulong off_dst = (extra->offset + offset) / sizeof(ggml_fp16_t);

        cl_int err;
        cl_mem data_device = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            size, const_cast<void *>(data), &err);
        CL_CHECK(err);

        cl_kernel kernel = backend_ctx->repack.kernel_convert_bf16_to_f16;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_device));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->data_device));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_ulong), &off_dst));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &n_elements));

        size_t global_work_size[] = { (size_t)CEIL_DIV(n_elements, 64)*64, 1, 1 };
        size_t local_work_size[] = { 64, 1, 1 };

        cl_event evt;
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &evt));
        CL_CHECK(clWaitForEvents(1, &evt));
        CL_CHECK(clReleaseMemObject(data_device));
        CL_CHECK(clReleaseEvent(evt));

        return;
    }

    ggml_tensor_extra_cl * extra = (ggml_tensor_extra_cl *) tensor->extra;
    GGML_ASSERT(extra);

    CL_CHECK(clEnqueueWriteBuffer(
        queue, extra->data_device, CL_TRUE, extra->offset + offset,
        size, data, 0, NULL, NULL));

    GGML_UNUSED(buffer);
}
