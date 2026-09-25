#include "cl-common.h"

void ggml_cl_load_kernels_unpack(ggml_backend_opencl_context * backend_ctx) {
    cl_int err;
    const std::string & compile_opts = backend_ctx->kernel_compile_opts;
    GGML_UNUSED(backend_ctx);
    GGML_UNUSED(err);
    GGML_UNUSED(compile_opts);
}
static cl_mem ggml_cl_create_temp_download_buffer(
    cl_context context, cl_command_queue queue,
    size_t nbytes, const char * tensor_name_for_log)
{
    cl_int err;
    cl_mem buf = clCreateBuffer(context, CL_MEM_READ_WRITE, nbytes, NULL, &err);
    if (err != CL_SUCCESS) {
        clFinish(queue);
        buf = clCreateBuffer(context, CL_MEM_READ_WRITE, nbytes, NULL, &err);
    }
    if (err == CL_SUCCESS) {
        return buf;
    }
    buf = clCreateBuffer(context,
        CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR | CL_MEM_HOST_READ_ONLY,
        nbytes, NULL, &err);
    if (err != CL_SUCCESS) {
        return NULL;
    }
    if (tensor_name_for_log) {
        GGML_LOG_INFO("ggml_opencl: %s download (%.1f MiB) - device alloc failed, using CL_MEM_ALLOC_HOST_PTR fallback\n",
                      tensor_name_for_log, nbytes / 1024.0 / 1024.0);
    }
    return buf;
}

void ggml_backend_opencl_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor->extra);

    ggml_backend_opencl_device_context * dev_ctx = (ggml_backend_opencl_device_context *) buffer->buft->device->context;
    ggml_backend_opencl_context *backend_ctx = dev_ctx->backend_ctx;

    cl_context context = backend_ctx->context;
    cl_command_queue queue = backend_ctx->queue;

    // Make sure all previously submitted commands in other devices are finished.
    sync_with_other_backends(backend_ctx);

#ifdef GGML_OPENCL_SOA_Q
    if (tensor->type == GGML_TYPE_Q1_0) {
        ggml_tensor_extra_cl_q1_0 * extra = (ggml_tensor_extra_cl_q1_0 *)tensor->extra;

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        if (enable_adreno_trans_weight(backend_ctx, tensor)) {
            ggml_cl_buffer buf_trans_q;
            ggml_cl_buffer buf_trans_d;
            ggml_cl_buffer buf_unpacked;

            int M = tensor->ne[1];
            int K = tensor->ne[0];

            size_t size_d = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*sizeof(ggml_fp16_t);
            size_t size_q = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*(ggml_blck_size(tensor->type)/8);

            buf_trans_q.allocate(backend_ctx->context, size_q);
            buf_trans_d.allocate(backend_ctx->context, size_d);
            buf_unpacked.allocate(backend_ctx->context, ggml_nbytes(tensor));

            transpose_2d_as_32b(backend_ctx, extra->q, buf_trans_q.buffer, size_q, M, K/32);
            transpose_2d_as_16b(backend_ctx, extra->d, buf_trans_d.buffer, size_d, M, K/128);

            cl_kernel kernel = backend_ctx->repack.kernel_restore_block_q1_0;
            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_trans_q.buffer));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_trans_d.buffer));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf_unpacked.buffer));

            size_t global_work_size[] = {(size_t)ggml_nelements(tensor)/ggml_blck_size(tensor->type), 1, 1};
            size_t local_work_size[] = {1, 1, 1};

            cl_event evt;
            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &evt));
            CL_CHECK(clWaitForEvents(1, &evt));
            CL_CHECK(clEnqueueReadBuffer(queue, buf_unpacked.buffer, CL_TRUE, offset, size, data, 0, NULL, NULL));
            return;
        }
#endif

        cl_int err;
        cl_mem data_device = clCreateBuffer(context, CL_MEM_READ_WRITE, ggml_nbytes(tensor), NULL, &err);
        CL_CHECK(err);

        cl_kernel kernel = backend_ctx->repack.kernel_restore_block_q1_0;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &extra->q));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->d));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &data_device));

        size_t global_work_size[] = {(size_t)ggml_nelements(tensor)/ggml_blck_size(tensor->type), 1, 1};
        size_t local_work_size[] = {1, 1, 1};

        cl_event evt;
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &evt));
        CL_CHECK(clWaitForEvents(1, &evt));
        CL_CHECK(clEnqueueReadBuffer(queue, data_device, CL_TRUE, offset, size, data, 0, NULL, NULL));
        CL_CHECK(clReleaseMemObject(data_device));
        return;
    }
    // In end-to-end runs, get_tensor is usually used to get back the logits,
    // where we can simply do clEnqueueReadBuffer since they are f32.
    // However, in test-backend-ops, the GPU graph is copied to the CPU backend,
    // which requires reading back quantized weight tensors.
    // To properly support this, we need to restore block_q4_0 struct arrays
    // from the flattened buffers.
    if (tensor->type == GGML_TYPE_Q4_0) {
        // KV-cache q4_0 stays AoS - direct readback, no SoA restore.
        if (!ggml_cl_is_q4_0_soa(tensor)) {
            ggml_tensor_extra_cl * extra_aos = (ggml_tensor_extra_cl *) tensor->extra;
            CL_CHECK(clEnqueueReadBuffer(
                queue, extra_aos->data_device, CL_TRUE,
                extra_aos->offset + tensor->view_offs + offset,
                size, data, 0, NULL, NULL));
            return;
        }
        // SoA extra lives on the parent tensor - follow view_src.
        const ggml_tensor * extra_src = tensor->view_src != nullptr ? tensor->view_src : tensor;
        ggml_tensor_extra_cl_q4_0 * extra = (ggml_tensor_extra_cl_q4_0 *)extra_src->extra;

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        if (use_adreno_moe_kernels(backend_ctx, tensor)) {
            cl_kernel kernel = backend_ctx->repack.kernel_restore_block_q4_0_trans4_ns;

            cl_mem data_device = ggml_cl_create_temp_download_buffer(context, queue, ggml_nbytes(tensor), tensor->name);
            GGML_ASSERT(data_device != NULL && "get_tensor: temp download buffer alloc failed");

            int ne00 = tensor->ne[0];
            int ne01 = tensor->ne[1];
            int ne02 = tensor->ne[2];
            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &extra->q));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->d));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &data_device));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_int), &ne00));
            CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_int), &ne01));

            size_t global_work_size[3] = {static_cast<size_t>(((ne01 + 63) / 64) * 64), static_cast<size_t>(ne00 / 32), static_cast<size_t>(ne02)};
            size_t local_work_size[3] = {64, 2, 1};

            cl_event evt;
            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL,
                global_work_size, local_work_size, 0, NULL, &evt));
            CL_CHECK(clWaitForEvents(1, &evt));
            CL_CHECK(clEnqueueReadBuffer(
                queue, data_device, CL_TRUE, offset,
                size, data, 0, NULL, NULL));
            CL_CHECK(clReleaseMemObject(data_device));
            return;
        }
        if (use_adreno_kernels(backend_ctx, tensor)) {
            ggml_cl_buffer buf_trans_q;
            ggml_cl_buffer buf_trans_d;
            ggml_cl_buffer buf_unpacked;

            cl_int M = tensor->ne[1];   // ne01
            cl_int K = tensor->ne[0];   // ne00

            GGML_ASSERT(K % 32 == 0);
            GGML_ASSERT(M % 4 == 0);

            size_t size_q = (ggml_nelements(tensor)/ggml_blck_size(tensor->type))*ggml_blck_size(tensor->type)/2;
            size_t size_d = (ggml_nelements(tensor)/ggml_blck_size(tensor->type))*sizeof(ggml_fp16_t);
            GGML_ASSERT(size_d + size_q == ggml_nbytes(tensor) && "Incorrect tensor size");

            buf_trans_q.allocate(backend_ctx->context, size_q);
            buf_trans_d.allocate(backend_ctx->context, size_d);
            buf_unpacked.allocate(backend_ctx->context, ggml_nbytes(tensor));

            if (use_q4_0_bin_kernels(backend_ctx, tensor)) {
                transpose_2d_as_32b(backend_ctx, extra->q, buf_trans_q.buffer, size_q, M, K / 8);
            } else {
                transpose_2d_as_16b(backend_ctx, extra->q, buf_trans_q.buffer, size_q, M, K / 4);
            }
            transpose_2d_as_16b(backend_ctx, extra->d, buf_trans_d.buffer, size_d, M, K/32);

            cl_uchar mask_0F = 0x0F;
            cl_uchar mask_F0 = 0xF0;

            size_t global_work_size[] = {(size_t)ggml_nelements(tensor)/ggml_blck_size(tensor->type), 1, 1};
            size_t local_work_size[] = {1, 1, 1};

            cl_kernel kernel = backend_ctx->repack.kernel_restore_block_q4_0_noshuffle;
            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &buf_trans_q.buffer));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),   &buf_trans_d.buffer));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &buf_unpacked.buffer));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_uchar), &mask_0F));
            CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_uchar), &mask_F0));

            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, NULL));
            CL_CHECK(clEnqueueReadBuffer(queue, buf_unpacked.buffer, CL_TRUE, offset, size, data, 0, NULL, NULL));
            return;
        }
#endif

        cl_mem data_device = ggml_cl_create_temp_download_buffer(context, queue, ggml_nbytes(tensor), tensor->name);
        GGML_ASSERT(data_device != NULL && "get_tensor: temp download buffer alloc failed");

        cl_kernel kernel = backend_ctx->repack.kernel_restore_block_q4_0;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &extra->q));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->d));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &data_device));

        size_t global_work_size[] = {(size_t)ggml_nelements(tensor)/ggml_blck_size(tensor->type), 1, 1};
        size_t local_work_size[] = {1, 1, 1};

        cl_event evt;
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL,
            global_work_size, local_work_size, 0, NULL, &evt));
        CL_CHECK(clWaitForEvents(1, &evt));
        CL_CHECK(clEnqueueReadBuffer(
            queue, data_device, CL_TRUE, offset,
            size, data, 0, NULL, NULL));
        CL_CHECK(clReleaseMemObject(data_device));
        return;
    }
    if (tensor->type == GGML_TYPE_Q4_1) {
        ggml_tensor_extra_cl_q4_1 * extra = (ggml_tensor_extra_cl_q4_1 *)tensor->extra;

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        if (use_adreno_moe_kernels(backend_ctx, tensor)) {
            cl_mem data_device = ggml_cl_create_temp_download_buffer(context, queue, ggml_nbytes(tensor), tensor->name);
            GGML_ASSERT(data_device != NULL && "get_tensor: temp download buffer alloc failed");
            cl_kernel kernel = backend_ctx->repack.kernel_restore_block_q4_1_trans4_ns;

            int ne00 = tensor->ne[0];
            int ne01 = tensor->ne[1];
            int ne02 = tensor->ne[2];
            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &extra->q));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->d));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extra->m));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &data_device));
            CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_int), &ne00));
            CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_int), &ne01));

            size_t global_work_size[3] = {static_cast<size_t>(((ne01 + 63) / 64) * 64), static_cast<size_t>(ne00 / 32), static_cast<size_t>(ne02)};
            size_t local_work_size[3] = {64, 2, 1};

            cl_event evt;
            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL,
                global_work_size, local_work_size, 0, NULL, &evt));
            CL_CHECK(clWaitForEvents(1, &evt));
            CL_CHECK(clEnqueueReadBuffer(
                queue, data_device, CL_TRUE, offset,
                size, data, 0, NULL, NULL));
            CL_CHECK(clReleaseMemObject(data_device));
            return;
        }
        if (use_adreno_kernels(backend_ctx, tensor)) {
            static ggml_cl_buffer buf_trans_q;
            static ggml_cl_buffer buf_trans_m;
            static ggml_cl_buffer buf_trans_d;
            static ggml_cl_buffer buf_unpacked;

            cl_int M = tensor->ne[1];
            cl_int K = tensor->ne[0];

            GGML_ASSERT(K % ggml_blck_size(tensor->type) == 0);

            size_t size_q = (ggml_nelements(tensor)/ggml_blck_size(tensor->type))*ggml_blck_size(tensor->type)/2;
            size_t size_d = (ggml_nelements(tensor)/ggml_blck_size(tensor->type))*sizeof(ggml_fp16_t);
            size_t size_m = (ggml_nelements(tensor)/ggml_blck_size(tensor->type))*sizeof(ggml_fp16_t);
            GGML_ASSERT(size_d + size_q + size_m == ggml_nbytes(tensor) && "Incorrect tensor size");

            buf_trans_q.allocate(backend_ctx->context, size_q);
            buf_trans_m.allocate(backend_ctx->context, size_m);
            buf_trans_d.allocate(backend_ctx->context, size_d);
            buf_unpacked.allocate(backend_ctx->context, ggml_nbytes(tensor));

            // transpose q, d, m back
            transpose_2d_as_16b(backend_ctx, extra->q, buf_trans_q.buffer, size_q, M, K/4);
            transpose_2d_as_16b(backend_ctx, extra->d, buf_trans_d.buffer, size_d, M, K/32);
            transpose_2d_as_16b(backend_ctx, extra->m, buf_trans_m.buffer, size_m, M, K/32);

            cl_uchar mask_0F = 0x0F;
            cl_uchar mask_F0 = 0xF0;

            size_t global_work_size[] = {(size_t)ggml_nelements(tensor)/ggml_blck_size(tensor->type), 1, 1};
            size_t local_work_size[] = {1, 1, 1};

            cl_kernel kernel = backend_ctx->repack.kernel_restore_block_q4_1_noshuffle;
            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &buf_trans_q.buffer));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),   &buf_trans_d.buffer));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &buf_trans_m.buffer));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem),   &buf_unpacked.buffer));
            CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_uchar), &mask_0F));
            CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_uchar), &mask_F0));

            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, NULL));
            CL_CHECK(clEnqueueReadBuffer(queue, buf_unpacked.buffer, CL_TRUE, offset, size, data, 0, NULL, NULL));
            return;
        }
#endif

        cl_mem data_device = ggml_cl_create_temp_download_buffer(context, queue, ggml_nbytes(tensor), tensor->name);
        GGML_ASSERT(data_device != NULL && "get_tensor: temp download buffer alloc failed");

        cl_kernel kernel = backend_ctx->repack.kernel_restore_block_q4_1;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &extra->q));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->d));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extra->m));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &data_device));

        size_t global_work_size[] = {(size_t)ggml_nelements(tensor)/ggml_blck_size(tensor->type), 1, 1};
        size_t local_work_size[] = {1, 1, 1};

        cl_event evt;
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL,
            global_work_size, local_work_size, 0, NULL, &evt));
        CL_CHECK(clWaitForEvents(1, &evt));
        CL_CHECK(clEnqueueReadBuffer(
            queue, data_device, CL_TRUE, offset,
            size, data, 0, NULL, NULL));
        CL_CHECK(clReleaseMemObject(data_device));
        return;
    }
    if (tensor->type == GGML_TYPE_Q5_0) {
        ggml_tensor_extra_cl_q5_0 * extra = (ggml_tensor_extra_cl_q5_0 *)tensor->extra;

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        if (use_adreno_moe_kernels(backend_ctx, tensor)) {
            // TODO: use ggml_cl_buffer to manage this temporary buffer
            cl_mem data_device = ggml_cl_create_temp_download_buffer(context, queue, ggml_nbytes(tensor), tensor->name);
            GGML_ASSERT(data_device != NULL && "get_tensor: temp download buffer alloc failed");

            cl_kernel kernel = backend_ctx->repack.kernel_restore_block_q5_0_trans4_ns;

            int ne00 = tensor->ne[0];
            int ne01 = tensor->ne[1];
            int ne02 = tensor->ne[2];
            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &extra->qs));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->qh));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extra->d));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &data_device));
            CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_int), &ne00));
            CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_int), &ne01));

            size_t global_work_size[3] = {static_cast<size_t>(((ne01 + 63) / 64) * 64), static_cast<size_t>(ne00 / 32), static_cast<size_t>(ne02)};
            size_t local_work_size[3] = {64, 2, 1};

            cl_event evt;
            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL,
                global_work_size, local_work_size, 0, NULL, &evt));
            CL_CHECK(clWaitForEvents(1, &evt));
            CL_CHECK(clEnqueueReadBuffer(
                queue, data_device, CL_TRUE, offset,
                size, data, 0, NULL, NULL));
            CL_CHECK(clReleaseMemObject(data_device));
            return;
        }
        if (use_adreno_kernels(backend_ctx, tensor)) {
            ggml_cl_buffer buf_trans_qs;
            ggml_cl_buffer buf_trans_qh;
            ggml_cl_buffer buf_trans_d;
            ggml_cl_buffer buf_unpacked;

            cl_int M = tensor->ne[1];
            cl_int K = tensor->ne[0];

            GGML_ASSERT(K % 32 == 0);

            size_t size_qs = (ggml_nelements(tensor)/ggml_blck_size(tensor->type))*ggml_blck_size(tensor->type)/2;
            size_t size_qh = (ggml_nelements(tensor)/ggml_blck_size(tensor->type))*sizeof(int32_t);
            size_t size_d = (ggml_nelements(tensor)/ggml_blck_size(tensor->type))*sizeof(ggml_fp16_t);

            buf_trans_qs.allocate(backend_ctx->context, size_qs);
            buf_trans_qh.allocate(backend_ctx->context, size_qh);
            buf_trans_d.allocate(backend_ctx->context, size_d);
            buf_unpacked.allocate(backend_ctx->context, ggml_nbytes(tensor));

            transpose_2d_as_16b(backend_ctx, extra->qs, buf_trans_qs.buffer, size_qs, M, K/4);
            transpose_2d_as_8b(backend_ctx, extra->qh, buf_trans_qh.buffer, size_qh, M, K/8);
            transpose_2d_as_16b(backend_ctx, extra->d,  buf_trans_d.buffer,  size_d,  M, K/32);

            cl_uchar mask_0F = 0x0F;
            cl_uchar mask_F0 = 0xF0;

            size_t global_work_size[] = {(size_t)ggml_nelements(tensor)/ggml_blck_size(tensor->type), 1, 1};
            size_t local_work_size[] = {1, 1, 1};

            cl_kernel kernel = backend_ctx->repack.kernel_restore_block_q5_0_noshuffle;
            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &buf_trans_qs.buffer));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),   &buf_trans_qh.buffer));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &buf_trans_d.buffer));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem),   &buf_unpacked.buffer));
            CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_uchar), &mask_0F));
            CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_uchar), &mask_F0));

            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, NULL));
            CL_CHECK(clEnqueueReadBuffer(queue, buf_unpacked.buffer, CL_TRUE, offset, size, data, 0, NULL, NULL));
            return;
        }
#endif // GGML_OPENCL_USE_ADRENO_KERNELS

        cl_int err;
        cl_mem data_device = clCreateBuffer(context, CL_MEM_READ_WRITE,
            ggml_nbytes(tensor), NULL, &err);
        CL_CHECK(err);

        cl_kernel kernel = backend_ctx->repack.kernel_restore_block_q5_0;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &extra->qs));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->qh));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extra->d));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &data_device));

        size_t global_work_size[] = {(size_t)ggml_nelements(tensor)/ggml_blck_size(tensor->type), 1, 1};
        size_t local_work_size[] = {1, 1, 1};

        cl_event evt;
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL,
            global_work_size, local_work_size, 0, NULL, &evt));
        CL_CHECK(clWaitForEvents(1, &evt));
        CL_CHECK(clEnqueueReadBuffer(
            queue, data_device, CL_TRUE, offset,
            size, data, 0, NULL, NULL));
        CL_CHECK(clReleaseMemObject(data_device));
        return;
    }
    if (tensor->type == GGML_TYPE_Q5_1) {
        ggml_tensor_extra_cl_q5_1 * extra = (ggml_tensor_extra_cl_q5_1 *)tensor->extra;

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        if (use_adreno_moe_kernels(backend_ctx, tensor)) {
            // TODO: use ggml_cl_buffer to manage this temporary buffer
            cl_mem data_device = ggml_cl_create_temp_download_buffer(context, queue, ggml_nbytes(tensor), tensor->name);
            GGML_ASSERT(data_device != NULL && "get_tensor: temp download buffer alloc failed");

            cl_kernel kernel = backend_ctx->repack.kernel_restore_block_q5_1_trans4_ns;

            int ne00 = tensor->ne[0];
            int ne01 = tensor->ne[1];
            int ne02 = tensor->ne[2];
            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &extra->qs));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->qh));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extra->d));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &extra->m));
            CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), &data_device));
            CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_int), &ne00));
            CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_int), &ne01));

            size_t global_work_size[3] = {static_cast<size_t>(((ne01 + 63) / 64) * 64), static_cast<size_t>(ne00 / 32), static_cast<size_t>(ne02)};
            size_t local_work_size[3] = {64, 2, 1};

            cl_event evt;
            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL,
                global_work_size, local_work_size, 0, NULL, &evt));
            CL_CHECK(clWaitForEvents(1, &evt));
            CL_CHECK(clEnqueueReadBuffer(
                queue, data_device, CL_TRUE, offset,
                size, data, 0, NULL, NULL));
            CL_CHECK(clReleaseMemObject(data_device));
            return;
        }

        if (use_adreno_kernels(backend_ctx, tensor)) {
            ggml_cl_buffer buf_trans_qs;
            ggml_cl_buffer buf_trans_qh;
            ggml_cl_buffer buf_trans_d;
            ggml_cl_buffer buf_trans_m;
            ggml_cl_buffer buf_unpacked;

            cl_int M = tensor->ne[1];
            cl_int K = tensor->ne[0];
            GGML_ASSERT(K % 32 == 0);

            size_t size_qs = (ggml_nelements(tensor)/ggml_blck_size(tensor->type))*ggml_blck_size(tensor->type)/2;
            size_t size_qh = (ggml_nelements(tensor)/ggml_blck_size(tensor->type))*sizeof(int32_t);
            size_t size_d  = (ggml_nelements(tensor)/ggml_blck_size(tensor->type))*sizeof(ggml_fp16_t);
            size_t size_m  = (ggml_nelements(tensor)/ggml_blck_size(tensor->type))*sizeof(ggml_fp16_t);

            buf_trans_qs.allocate(backend_ctx->context, size_qs);
            buf_trans_qh.allocate(backend_ctx->context, size_qh);
            buf_trans_d.allocate(backend_ctx->context, size_d);
            buf_trans_m.allocate(backend_ctx->context, size_m);
            buf_unpacked.allocate(backend_ctx->context, ggml_nbytes(tensor));

            // Transpose back: from col-major to row-major
            transpose_2d_as_16b(backend_ctx, extra->qs, buf_trans_qs.buffer, size_qs, M, K/4);
            transpose_2d_as_8b(backend_ctx, extra->qh, buf_trans_qh.buffer, size_qh, M, K/8);
            transpose_2d_as_16b(backend_ctx, extra->d,  buf_trans_d.buffer,  size_d,  M, K/32);
            transpose_2d_as_16b(backend_ctx, extra->m,  buf_trans_m.buffer,  size_m,  M, K/32);

            cl_uchar mask_0F = 0x0F;
            cl_uchar mask_F0 = 0xF0;

            size_t global_work_size[] = {(size_t)ggml_nelements(tensor)/ggml_blck_size(tensor->type), 1, 1};
            size_t local_work_size[] = {1, 1, 1};

            cl_kernel kernel = backend_ctx->repack.kernel_restore_block_q5_1_noshuffle;
            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &buf_trans_qs.buffer));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),   &buf_trans_qh.buffer));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &buf_trans_d.buffer));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem),   &buf_trans_m.buffer));
            CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem),   &buf_unpacked.buffer));
            CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_uchar), &mask_0F));
            CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_uchar), &mask_F0));

            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, NULL));
            CL_CHECK(clEnqueueReadBuffer(queue, buf_unpacked.buffer, CL_TRUE, offset, size, data, 0, NULL, NULL));
            return;
        }
#endif // GGML_OPENCL_USE_ADRENO_KERNELS
        cl_int err;
        cl_mem data_device = clCreateBuffer(context, CL_MEM_READ_WRITE,
            ggml_nbytes(tensor), NULL, &err);
        CL_CHECK(err);

        cl_kernel kernel = backend_ctx->repack.kernel_restore_block_q5_1;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &extra->qs));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->qh));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extra->d));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &extra->m));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), &data_device));

        size_t global_work_size[] = {(size_t)ggml_nelements(tensor)/ggml_blck_size(tensor->type), 1, 1};
        size_t local_work_size[] = {1, 1, 1};

        cl_event evt;
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL,
            global_work_size, local_work_size, 0, NULL, &evt));
        CL_CHECK(clWaitForEvents(1, &evt));
        CL_CHECK(clEnqueueReadBuffer(
            queue, data_device, CL_TRUE, offset,
            size, data, 0, NULL, NULL));
        CL_CHECK(clReleaseMemObject(data_device));
        return;
    }
    if (tensor->type == GGML_TYPE_MXFP4) {
        ggml_tensor_extra_cl_mxfp4 * extra = (ggml_tensor_extra_cl_mxfp4 *)tensor->extra;

        cl_mem data_device = ggml_cl_create_temp_download_buffer(context, queue, ggml_nbytes(tensor), tensor->name);
        GGML_ASSERT(data_device != NULL && "get_tensor: temp download buffer alloc failed");

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        if (use_adreno_moe_kernels(backend_ctx, tensor)) {
            cl_kernel kernel = backend_ctx->repack.kernel_restore_block_mxfp4_trans4_ns;

            int ne00 = tensor->ne[0];
            int ne01 = tensor->ne[1];
            int ne02 = tensor->ne[2];
            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &extra->q));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->e));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &data_device));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_int), &ne00));
            CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_int), &ne01));

            size_t global_work_size[3] = {static_cast<size_t>(((ne01 + 63) / 64) * 64), static_cast<size_t>(ne00 / 32), static_cast<size_t>(ne02)};
            size_t local_work_size[3] = {64, 2, 1};

            cl_event evt;
            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL,
                global_work_size, local_work_size, 0, NULL, &evt));
            CL_CHECK(clWaitForEvents(1, &evt));
            CL_CHECK(clEnqueueReadBuffer(
                queue, data_device, CL_TRUE, offset,
                size, data, 0, NULL, NULL));
            CL_CHECK(clReleaseMemObject(data_device));
            return;
        }

#endif // GGML_OPENCL_USE_ADRENO_KERNELS
        cl_kernel kernel = backend_ctx->repack.kernel_restore_block_mxfp4;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &extra->q));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->e));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &data_device));

        size_t global_work_size[] = {(size_t)ggml_nelements(tensor)/ggml_blck_size(tensor->type), 1, 1};
        size_t local_work_size[] = {1, 1, 1};

        cl_event evt;
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL,
            global_work_size, local_work_size, 0, NULL, &evt));
        CL_CHECK(clWaitForEvents(1, &evt));
        CL_CHECK(clEnqueueReadBuffer(
            queue, data_device, CL_TRUE, offset,
            size, data, 0, NULL, NULL));
        CL_CHECK(clReleaseMemObject(data_device));
        return;
    }
    if (tensor->type == GGML_TYPE_Q8_0) {
        // KV-cache q8_0 stays AoS (see Q4_0 branch).
        if (!ggml_cl_is_q8_0_soa(tensor)) {
            ggml_tensor_extra_cl * extra_aos = (ggml_tensor_extra_cl *) tensor->extra;
            CL_CHECK(clEnqueueReadBuffer(
                queue, extra_aos->data_device, CL_TRUE,
                extra_aos->offset + tensor->view_offs + offset,
                size, data, 0, NULL, NULL));
            return;
        }
        // SoA extra lives on the parent - follow view_src.
        const ggml_tensor * extra_src = tensor->view_src != nullptr ? tensor->view_src : tensor;
        ggml_tensor_extra_cl_q8_0 * extra = (ggml_tensor_extra_cl_q8_0 *)extra_src->extra;

        cl_mem data_device = ggml_cl_create_temp_download_buffer(context, queue, ggml_nbytes(tensor), tensor->name);
        GGML_ASSERT(data_device != NULL && "get_tensor: temp download buffer alloc failed");

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        if (enable_adreno_trans_weight(backend_ctx, tensor)) {
            cl_kernel kernel = backend_ctx->repack.kernel_restore_block_q8_0_trans;

            int ne00 = tensor->ne[0];
            int ne01 = tensor->ne[1];
            GGML_ASSERT(tensor->ne[2] == 1);
            GGML_ASSERT(tensor->ne[3] == 1);

            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &extra->q));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->d));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &data_device));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_int), &ne00));
            CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_int), &ne01));

            size_t global_work_size[3] = {static_cast<size_t>(((ne01 + 63) / 64) * 64), 1, 1};
            size_t local_work_size[3] = {64, 1, 1};

            cl_event evt;
            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL,
                global_work_size, local_work_size, 0, NULL, &evt));
            CL_CHECK(clWaitForEvents(1, &evt));

            CL_CHECK(clEnqueueReadBuffer(
                queue, data_device, CL_TRUE, offset,
                size, data, 0, NULL, NULL));
            CL_CHECK(clReleaseMemObject(data_device));
            return;
        }
#endif
        cl_kernel kernel = backend_ctx->repack.kernel_restore_block_q8_0;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &extra->q));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->d));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &data_device));

        size_t global_work_size[] = {(size_t)ggml_nelements(tensor)/ggml_blck_size(tensor->type), 1, 1};
        size_t local_work_size[] = {1, 1, 1};

        cl_event evt;
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL,
            global_work_size, local_work_size, 0, NULL, &evt));
        CL_CHECK(clWaitForEvents(1, &evt));
        CL_CHECK(clEnqueueReadBuffer(
            queue, data_device, CL_TRUE, offset,
            size, data, 0, NULL, NULL));
        CL_CHECK(clReleaseMemObject(data_device));
        return;
    }
    if (tensor->type == GGML_TYPE_IQ4_NL) {
        ggml_tensor_extra_cl_iq4_nl * extra = (ggml_tensor_extra_cl_iq4_nl *)tensor->extra;

        cl_mem data_device = ggml_cl_create_temp_download_buffer(context, queue, ggml_nbytes(tensor), tensor->name);
        GGML_ASSERT(data_device != NULL && "get_tensor: temp download buffer alloc failed");

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        if (use_adreno_kernels(backend_ctx, tensor)) {
            static ggml_cl_buffer buf_trans_q;
            static ggml_cl_buffer buf_trans_d;
            static ggml_cl_buffer buf_unpacked;

            cl_int M = tensor->ne[1];
            cl_int K = tensor->ne[0];
            GGML_ASSERT(K % 32 == 0);

            size_t size_q = (ggml_nelements(tensor)/ggml_blck_size(tensor->type))*(ggml_blck_size(tensor->type)/2);
            size_t size_d = (ggml_nelements(tensor)/ggml_blck_size(tensor->type))*sizeof(ggml_fp16_t);
            GGML_ASSERT(size_d + size_q == ggml_nbytes(tensor) && "Incorrect tensor size");

            buf_trans_q.allocate(backend_ctx->context, size_q);
            buf_trans_d.allocate(backend_ctx->context, size_d);
            buf_unpacked.allocate(backend_ctx->context, ggml_nbytes(tensor));

            // transpose q, d back
            transpose_2d_as_16b(backend_ctx, extra->q, buf_trans_q.buffer, size_q, M, K/4);
            transpose_2d_as_16b(backend_ctx, extra->d, buf_trans_d.buffer, size_d, M, K/32);

            cl_uchar mask_0F = 0x0F;
            cl_uchar mask_F0 = 0xF0;

            cl_kernel kernel = backend_ctx->repack.kernel_restore_block_iq4_nl_noshuffle;
            cl_ulong n_blk = ggml_nelements(tensor)/ggml_blck_size(tensor->type);

            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &buf_trans_q.buffer));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),   &buf_trans_d.buffer));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &buf_unpacked.buffer));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_uchar), &mask_0F));
            CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_uchar), &mask_F0));
            CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_ulong), &n_blk));

            size_t global_work_size[] = {(size_t)n_blk, 1, 1};
            size_t local_work_size[] = {1, 1, 1};

            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, NULL));
            CL_CHECK(clEnqueueReadBuffer(queue, buf_unpacked.buffer, CL_TRUE, offset, size, data, 0, NULL, NULL));
            return;
        }
#endif
        cl_kernel kernel = backend_ctx->repack.kernel_restore_block_iq4_nl;
        cl_ulong n_blk = ggml_nelements(tensor)/ggml_blck_size(tensor->type);

        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &extra->q));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->d));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &data_device));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &n_blk));

        size_t global_work_size[] = {(size_t)n_blk, 1, 1};
        size_t local_work_size[] = {1, 1, 1};

        cl_event evt;
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL,
            global_work_size, local_work_size, 0, NULL, &evt));
        CL_CHECK(clWaitForEvents(1, &evt));
        CL_CHECK(clEnqueueReadBuffer(
            queue, data_device, CL_TRUE, offset,
            size, data, 0, NULL, NULL));
        CL_CHECK(clReleaseMemObject(data_device));
        return;
    }
    if (tensor->type == GGML_TYPE_Q4_K) {
        ggml_tensor_extra_cl_q4_K * extra = (ggml_tensor_extra_cl_q4_K *)tensor->extra;

        cl_mem data_device = ggml_cl_create_temp_download_buffer(context, queue, ggml_nbytes(tensor), tensor->name);
        GGML_ASSERT(data_device != NULL && "get_tensor: temp download buffer alloc failed");

        cl_uchar mask_0F = 0x0F;
        cl_uchar mask_F0 = 0xF0;

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        // Undo the 64-row-tiled canonical pack (kernel_convert_block_q4_k_tiled_ns).
        // Without this, a read-back of a tiled weight returns the tiled bytes
        // reinterpreted as block_q4_K -- which is how test-backend-ops builds its
        // CPU reference (ggml_backend_graph_copy -> tensor_get), so the tiled path
        // "failed" the suite while computing the correct product.
        if (use_q4k_tiled(backend_ctx, tensor)) {
            const int    ne00v = tensor->ne[0];
            const int    ne01v = tensor->ne[1];
            const int    nbv   = ne00v / 256;
            const size_t n_blk = (size_t)nbv * ne01v;

            std::vector<uint32_t> tq(n_blk*32);
            std::vector<uint16_t> td(n_blk), tdm(n_blk);
            std::vector<uint8_t>  ts(n_blk*12);
            CL_CHECK(clEnqueueReadBuffer(queue, extra->q,  CL_TRUE, 0, tq.size()*4,  tq.data(),  0, NULL, NULL));
            CL_CHECK(clEnqueueReadBuffer(queue, extra->d,  CL_TRUE, 0, td.size()*2,  td.data(),  0, NULL, NULL));
            CL_CHECK(clEnqueueReadBuffer(queue, extra->dm, CL_TRUE, 0, tdm.size()*2, tdm.data(), 0, NULL, NULL));
            CL_CHECK(clEnqueueReadBuffer(queue, extra->s,  CL_TRUE, 0, ts.size(),    ts.data(),  0, NULL, NULL));

            std::vector<uint8_t> rebuilt(ggml_nbytes(tensor), 0);
            for (int i01 = 0; i01 < ne01v; ++i01) {
                const int rt = i01/64, rit = i01%64;
                for (int i00 = 0; i00 < nbv; ++i00) {
                    uint8_t * b = rebuilt.data() + ((size_t)i00 + (size_t)i01*nbv)*144;
                    const int tb = rt*nbv + i00;
                    const size_t si = (size_t)tb*64 + rit;

                    memcpy(b + 0, &td [si], 2);
                    memcpy(b + 2, &tdm[si], 2);
                    memcpy(b + 4, &ts[si*12], 12);

                    uint32_t qw[32];
                    for (int gr = 0; gr < 8; ++gr) {
                        const size_t base = ((size_t)tb*8 + gr)*64 + rit;
                        for (int j = 0; j < 4; ++j) qw[gr*4 + j] = tq[base*4 + j];
                    }
                    uint8_t * q = b + 16;
                    for (int e = 0; e < 256; ++e) {
                        const int g = e>>6, w = e&63, h = w>>5, l = w&31;
                        const uint32_t code = (qw[e>>3] >> ((e&7)*4)) & 0xF;
                        q[g*32 + l] |= (uint8_t)(h ? (code << 4) : code);
                    }
                }
            }
            memcpy(data, rebuilt.data() + offset, size);
            CL_CHECK(clReleaseMemObject(data_device));
            return;
        }
        if (use_adreno_moe_kernels(backend_ctx, tensor)) {
            cl_mem data_device = ggml_cl_create_temp_download_buffer(context, queue, ggml_nbytes(tensor), tensor->name);
            GGML_ASSERT(data_device != NULL && "get_tensor: temp download buffer alloc failed");

            cl_kernel kernel = backend_ctx->repack.kernel_restore_block_q4_k_trans4_ns;

            int ne00 = tensor->ne[0];
            int ne01 = tensor->ne[1];
            int ne02 = tensor->ne[2];
            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &extra->q));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->d));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extra->dm));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &extra->s));
            CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), &data_device));
            CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_int), &ne00));
            CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_int), &ne01));
            CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_uchar), &mask_0F));
            CL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_uchar), &mask_F0));

            size_t global_work_size[] = {static_cast<size_t>(((ne01 + 63) / 64) * 64), static_cast<size_t>(ne00 / 256), static_cast<size_t>(ne02)};
            size_t local_work_size[] = {64, 1, 1};

            cl_event evt;
            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL,
                global_work_size, local_work_size, 0, NULL, &evt));
            CL_CHECK(clWaitForEvents(1, &evt));
            CL_CHECK(clEnqueueReadBuffer(
                queue, data_device, CL_TRUE, offset,
                size, data, 0, NULL, NULL));
            CL_CHECK(clReleaseMemObject(data_device));
            return;
        }
        if (use_adreno_kernels(backend_ctx, tensor) && !use_flat_gemv_for_large_m_q4_K(backend_ctx, tensor)) {
            int M = tensor->ne[1];
            int K = tensor->ne[0];

            size_t size_q  = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*ggml_blck_size(tensor->type)/2;
            size_t size_d  = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*sizeof(ggml_fp16_t);
            size_t size_dm = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*sizeof(ggml_fp16_t);
            size_t size_s  = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*12;

            static ggml_cl_buffer buf_trans_q;
            static ggml_cl_buffer buf_trans_d;
            static ggml_cl_buffer buf_trans_dm;
            static ggml_cl_buffer buf_trans_s;

            buf_trans_q.allocate(backend_ctx->context, size_q);
            buf_trans_d.allocate(backend_ctx->context, size_d);
            buf_trans_dm.allocate(backend_ctx->context, size_dm);
            buf_trans_s.allocate(backend_ctx->context, size_s);

            // Transpose q, d, dm, s back
            if (use_q4_k_bin_kernels(backend_ctx, tensor)) {
                transpose_2d_as_32b(backend_ctx, extra->q, buf_trans_q.buffer, size_q, M, K/8);
            } else {
                transpose_2d_as_16b(backend_ctx, extra->q,  buf_trans_q.buffer,  size_q,  M, K/4);
            }
            transpose_2d_as_16b(backend_ctx, extra->d,  buf_trans_d.buffer,  size_d,  M, K/256);
            transpose_2d_as_16b(backend_ctx, extra->dm, buf_trans_dm.buffer, size_dm, M, K/256);
            transpose_2d_as_8b (backend_ctx, extra->s,  buf_trans_s.buffer,  size_s,  M, K/256*12, true, true);

            cl_kernel kernel = backend_ctx->repack.kernel_restore_block_q4_K_noshuffle;
            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_trans_q.buffer));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_trans_s.buffer));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf_trans_d.buffer));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &buf_trans_dm.buffer));
            CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), &data_device));
            CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_uchar), &mask_0F));
            CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_uchar), &mask_F0));

            size_t global_work_size[] = {(size_t)ggml_nelements(tensor)/ggml_blck_size(tensor->type), 1, 1};
            size_t local_work_size[] = {1, 1, 1};

            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL,
                global_work_size, local_work_size, 0, NULL, NULL));
            CL_CHECK(clEnqueueReadBuffer(queue, data_device, CL_TRUE, offset,
                size, data, 0, NULL, NULL));
            CL_CHECK(clReleaseMemObject(data_device));
            return;
        }
#endif // GGML_OPENCL_USE_ADRENO_KERNELS

        cl_kernel kernel = backend_ctx->repack.kernel_restore_block_q4_K;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &extra->q));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->s));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extra->d));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &extra->dm));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), &data_device));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_uchar), &mask_0F));
        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_uchar), &mask_F0));

        size_t global_work_size[] = {(size_t)ggml_nelements(tensor)/ggml_blck_size(tensor->type), 1, 1};
        size_t local_work_size[] = {1, 1, 1};

        cl_event evt;
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL,
            global_work_size, local_work_size, 0, NULL, &evt));
        CL_CHECK(clWaitForEvents(1, &evt));
        CL_CHECK(clEnqueueReadBuffer(
            queue, data_device, CL_TRUE, offset,
            size, data, 0, NULL, NULL));
        CL_CHECK(clReleaseMemObject(data_device));
        return;
    }
    if (tensor->type == GGML_TYPE_Q5_K) {
        ggml_tensor_extra_cl_q5_K * extra = (ggml_tensor_extra_cl_q5_K *)tensor->extra;

        cl_mem data_device = ggml_cl_create_temp_download_buffer(context, queue, ggml_nbytes(tensor), tensor->name);
        GGML_ASSERT(data_device != NULL && "get_tensor: temp download buffer alloc failed");

        cl_uchar mask_0F = 0x0F;
        cl_uchar mask_F0 = 0xF0;

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        if (use_adreno_moe_kernels(backend_ctx, tensor)) {
            cl_mem data_device = ggml_cl_create_temp_download_buffer(context, queue, ggml_nbytes(tensor), tensor->name);
            GGML_ASSERT(data_device != NULL && "get_tensor: temp download buffer alloc failed");
            cl_kernel kernel = backend_ctx->repack.kernel_restore_block_q5_k_trans4_ns;

            int ne00 = tensor->ne[0];
            int ne01 = tensor->ne[1];
            int ne02 = tensor->ne[2];
            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &extra->q));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->qh));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extra->d));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &extra->dm));
            CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), &extra->s));
            CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_mem), &data_device));
            CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_int), &ne00));
            CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_int), &ne01));
            CL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_uchar), &mask_0F));
            CL_CHECK(clSetKernelArg(kernel, 9, sizeof(cl_uchar), &mask_F0));

            size_t global_work_size[] = {static_cast<size_t>(((ne01 + 63) / 64) * 64), static_cast<size_t>(ne00 / 256), static_cast<size_t>(ne02)};
            size_t local_work_size[] = {64, 1, 1};

            cl_event evt;
            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL,
                global_work_size, local_work_size, 0, NULL, &evt));
            CL_CHECK(clWaitForEvents(1, &evt));
            CL_CHECK(clEnqueueReadBuffer(
                queue, data_device, CL_TRUE, offset,
                size, data, 0, NULL, NULL));
            CL_CHECK(clReleaseMemObject(data_device));
            return;
        }
        if (enable_adreno_trans_weight_q5_K(backend_ctx, tensor)) {
            int M = tensor->ne[1];
            int K = tensor->ne[0];

            size_t size_q  = extra->size_q;
            size_t size_qh = extra->size_qh;
            size_t size_d  = extra->size_d;
            size_t size_dm = extra->size_dm;

            static ggml_cl_buffer buf_trans_q;
            static ggml_cl_buffer buf_trans_qh;
            static ggml_cl_buffer buf_trans_d;
            static ggml_cl_buffer buf_trans_dm;

            buf_trans_q.allocate(backend_ctx->context, size_q);
            buf_trans_qh.allocate(backend_ctx->context, size_qh);
            buf_trans_d.allocate(backend_ctx->context, size_d);
            buf_trans_dm.allocate(backend_ctx->context, size_dm);

            // Reverse transpose q, qh, d, dm
            transpose_2d_as_16b(backend_ctx, extra->q,  buf_trans_q.buffer,  size_q,  M, K/4);
            transpose_2d_as_8b (backend_ctx, extra->qh, buf_trans_qh.buffer, size_qh, M, K/8);
            transpose_2d_as_16b(backend_ctx, extra->d,  buf_trans_d.buffer,  size_d,  M, K/256);
            transpose_2d_as_16b(backend_ctx, extra->dm, buf_trans_dm.buffer, size_dm, M, K/256);

            cl_kernel kernel = backend_ctx->repack.kernel_restore_block_q5_K_noshuffle;
            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &buf_trans_q.buffer));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),   &buf_trans_qh.buffer));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extra->s));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem),   &buf_trans_d.buffer));
            CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem),   &buf_trans_dm.buffer));
            CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_mem),   &data_device));
            CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_uchar), &mask_0F));
            CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_uchar), &mask_F0));

            size_t global_work_size[] = {(size_t)ggml_nelements(tensor)/ggml_blck_size(tensor->type), 1, 1};
            size_t local_work_size[] = {1, 1, 1};

            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL,
                global_work_size, local_work_size, 0, NULL, NULL));
            CL_CHECK(clEnqueueReadBuffer(queue, data_device, CL_TRUE, offset,
                size, data, 0, NULL, NULL));
            CL_CHECK(clReleaseMemObject(data_device));
            return;
        }
#endif // GGML_OPENCL_USE_ADRENO_KERNELS

        cl_kernel kernel = backend_ctx->repack.kernel_restore_block_q5_K;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra->q));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),   &extra->qh));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extra->s));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem),   &extra->d));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem),   &extra->dm));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_mem),   &data_device));
        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_uchar), &mask_0F));
        CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_uchar), &mask_F0));

        size_t global_work_size[] = {(size_t)ggml_nelements(tensor)/ggml_blck_size(tensor->type), 1, 1};
        size_t local_work_size[] = {1, 1, 1};

        cl_event evt;
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL,
            global_work_size, local_work_size, 0, NULL, &evt));
        CL_CHECK(clWaitForEvents(1, &evt));
        CL_CHECK(clEnqueueReadBuffer(
            queue, data_device, CL_TRUE, offset,
            size, data, 0, NULL, NULL));
        CL_CHECK(clReleaseMemObject(data_device));
        return;
    }
    if (tensor->type == GGML_TYPE_Q6_K) {
        ggml_tensor_extra_cl_q6_K * extra = (ggml_tensor_extra_cl_q6_K *)tensor->extra;

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        // Undo the 64-row-tiled canonical pack (kernel_convert_block_q6_k_tiled_ns).
        // See the q4_K tiled restore above for why a read-back path is required.
        if (use_q6k_tiled(backend_ctx, tensor)) {
            const int    ne00v = tensor->ne[0];
            const int    ne01v = tensor->ne[1];
            const int    nbv   = ne00v / 256;
            const size_t n_blk = (size_t)nbv * ne01v;

            std::vector<uint32_t> tql(n_blk*32), tqh(n_blk*16);
            std::vector<uint8_t>  ts(n_blk*16);
            std::vector<uint16_t> td(n_blk);
            CL_CHECK(clEnqueueReadBuffer(queue, extra->ql, CL_TRUE, 0, tql.size()*4, tql.data(), 0, NULL, NULL));
            CL_CHECK(clEnqueueReadBuffer(queue, extra->qh, CL_TRUE, 0, tqh.size()*4, tqh.data(), 0, NULL, NULL));
            CL_CHECK(clEnqueueReadBuffer(queue, extra->s,  CL_TRUE, 0, ts.size(),    ts.data(),  0, NULL, NULL));
            CL_CHECK(clEnqueueReadBuffer(queue, extra->d,  CL_TRUE, 0, td.size()*2,  td.data(),  0, NULL, NULL));

            std::vector<uint8_t> rebuilt(ggml_nbytes(tensor), 0);
            for (int i01 = 0; i01 < ne01v; ++i01) {
                const int rt = i01/64, rit = i01%64;
                for (int i00 = 0; i00 < nbv; ++i00) {
                    uint8_t * b = rebuilt.data() + ((size_t)i00 + (size_t)i01*nbv)*210;
                    const int tb = rt*nbv + i00;
                    const size_t si = (size_t)tb*64 + rit;

                    uint32_t qlw[32], qhw[16];
                    for (int g = 0; g < 8; ++g) {
                        const size_t base = ((size_t)tb*8 + g)*64 + rit;
                        for (int j = 0; j < 4; ++j) qlw[g*4 + j] = tql[base*4 + j];
                    }
                    for (int g = 0; g < 4; ++g) {
                        const size_t base = ((size_t)tb*4 + g)*64 + rit;
                        for (int j = 0; j < 4; ++j) qhw[g*4 + j] = tqh[base*4 + j];
                    }

                    uint8_t * ql = b;
                    uint8_t * qh = b + 128;
                    for (int e = 0; e < 256; ++e) {
                        const int n = (e >= 128) ? 1 : 0;
                        const int within = e - n*128, q = within/32, l = within%32;
                        const int off_ql = n*64, off_qh = n*32;
                        const uint8_t low4 = (qlw[e>>3] >> ((e&7)*4)) & 0xF;
                        const uint8_t hi2  = (qhw[e>>4] >> ((e&15)*2)) & 0x3;
                        if      (q == 0) ql[off_ql + l]      |= low4;
                        else if (q == 1) ql[off_ql + l + 32] |= low4;
                        else if (q == 2) ql[off_ql + l]      |= (uint8_t)(low4 << 4);
                        else             ql[off_ql + l + 32] |= (uint8_t)(low4 << 4);
                        qh[off_qh + l] |= (uint8_t)(hi2 << (q*2));
                    }
                    memcpy(b + 192, &ts[si*16], 16);
                    memcpy(b + 208, &td[si], 2);
                }
            }
            memcpy(data, rebuilt.data() + offset, size);
            return;
        }
        if (use_adreno_moe_kernels(backend_ctx, tensor)) {
            cl_mem data_device = ggml_cl_create_temp_download_buffer(context, queue, ggml_nbytes(tensor), tensor->name);
            GGML_ASSERT(data_device != NULL && "get_tensor: temp download buffer alloc failed");

            cl_kernel kernel = backend_ctx->repack.kernel_restore_block_q6_k_trans4_ns;

            cl_uchar mask_0F = 0x0F;
            cl_uchar mask_F0 = 0xF0;

            int ne00 = tensor->ne[0];
            int ne01 = tensor->ne[1];
            int ne02 = tensor->ne[2];
            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &extra->ql));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->qh));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extra->d));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &extra->s));
            CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), &data_device));
            CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_int), &ne00));
            CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_int), &ne01));
            CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_uchar), &mask_0F));
            CL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_uchar), &mask_F0));

            size_t global_work_size[] = {static_cast<size_t>(((ne01 + 63) / 64) * 64), static_cast<size_t>(ne00 / 256), static_cast<size_t>(ne02)};
            size_t local_work_size[] = {64, 1, 1};

            cl_event evt;
            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL,
                global_work_size, local_work_size, 0, NULL, &evt));
            CL_CHECK(clWaitForEvents(1, &evt));
            CL_CHECK(clEnqueueReadBuffer(
                queue, data_device, CL_TRUE, offset,
                size, data, 0, NULL, NULL));
            CL_CHECK(clReleaseMemObject(data_device));
            return;
        }
        if (use_adreno_kernels(backend_ctx, tensor) && !use_flat_gemv_for_large_m_q6_K(backend_ctx, tensor)) {
            static ggml_cl_buffer buf_trans_ql;
            static ggml_cl_buffer buf_trans_qh;
            static ggml_cl_buffer buf_trans_s;
            static ggml_cl_buffer buf_trans_d;
            static ggml_cl_buffer buf_unpacked;

            cl_int M = tensor->ne[1];   // ne01
            cl_int K = tensor->ne[0];   // ne00

            GGML_ASSERT(K % ggml_blck_size(tensor->type) == 0);

            size_t size_ql = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*ggml_blck_size(tensor->type)/2;
            size_t size_qh = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*ggml_blck_size(tensor->type)/4;
            size_t size_s  = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*ggml_blck_size(tensor->type)/16;
            size_t size_d  = ggml_nelements(tensor)/ggml_blck_size(tensor->type)*sizeof(ggml_fp16_t);
            GGML_ASSERT(size_ql + size_qh + size_s + size_d == ggml_nbytes(tensor) && "Incorrect tensor size");

            buf_trans_ql.allocate(backend_ctx->context, size_ql);
            buf_trans_qh.allocate(backend_ctx->context, size_qh);
            buf_trans_d.allocate(backend_ctx->context, size_d);
            buf_unpacked.allocate(backend_ctx->context, ggml_nbytes(tensor));

            cl_mem s_buffer;
            if (use_q6_k_bin_kernels(backend_ctx, tensor)) {
                transpose_2d_as_32b(backend_ctx, extra->ql, buf_trans_ql.buffer, size_ql, M, K/8);
                transpose_2d_as_32b(backend_ctx, extra->qh, buf_trans_qh.buffer, size_qh, M, K/16);
                // s is left row-major, untransposed, for the binary layout.
                s_buffer = extra->s;
            } else {
                // transpose ql, qh, s and d back
                buf_trans_s.allocate(backend_ctx->context, size_s);
                transpose_2d_as_16b(backend_ctx, extra->ql, buf_trans_ql.buffer, size_ql, M, K/4);
                transpose_2d_as_8b(backend_ctx,  extra->qh, buf_trans_qh.buffer, size_qh, M, K/4);
                transpose_2d_as_16b(backend_ctx, extra->s,  buf_trans_s.buffer,  size_s,  M, K/16/2);
                s_buffer = buf_trans_s.buffer;
            }
            transpose_2d_as_16b(backend_ctx, extra->d, buf_trans_d.buffer, size_d, M, K/256);

            // unpack
            cl_uchar mask = 0xFF;
            cl_ulong n_blk = ggml_nelements(tensor)/ggml_blck_size(tensor->type);
            cl_kernel kernel = backend_ctx->repack.kernel_restore_block_q6_K_noshuffle;
            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &buf_trans_ql.buffer));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),   &buf_trans_qh.buffer));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &s_buffer));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem),   &buf_trans_d.buffer));
            CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem),   &buf_unpacked.buffer));
            CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_uchar), &mask));
            CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_ulong), &n_blk));

            size_t global_work_size[] = {(size_t)n_blk, 1, 1};
            size_t local_work_size[] = {1, 1, 1};

            cl_event evt;
            CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &evt));
            CL_CHECK(clWaitForEvents(1, &evt));
            CL_CHECK(clEnqueueReadBuffer(queue, buf_unpacked.buffer, CL_TRUE, offset, size, data, 0, NULL, NULL));

            return;
        }
#endif // GGML_OPENCL_USE_ADRENO_KERNELS

        cl_mem data_device = ggml_cl_create_temp_download_buffer(context, queue, ggml_nbytes(tensor), tensor->name);
        GGML_ASSERT(data_device != NULL && "get_tensor: temp download buffer alloc failed");

        cl_uchar mask = 0xFF;
        cl_ulong n_blk = ggml_nelements(tensor)/ggml_blck_size(tensor->type);
        cl_kernel kernel = backend_ctx->repack.kernel_restore_block_q6_K;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra->ql));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),   &extra->qh));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extra->s));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem),   &extra->d));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem),   &data_device));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_uchar), &mask));
        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_ulong), &n_blk));

        size_t global_work_size[] = {(size_t)n_blk, 1, 1};
        size_t local_work_size[] = {1, 1, 1};

        cl_event evt;
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL,
            global_work_size, local_work_size, 0, NULL, &evt));
        CL_CHECK(clWaitForEvents(1, &evt));
        CL_CHECK(clEnqueueReadBuffer(
            queue, data_device, CL_TRUE, offset,
            size, data, 0, NULL, NULL));
        CL_CHECK(clReleaseMemObject(data_device));
        return;
    }
#endif // GGML_OPENCL_SOA_Q

    if (tensor->type == GGML_TYPE_BF16) {
        GGML_ASSERT(offset % sizeof(ggml_fp16_t) == 0 && size % sizeof(ggml_fp16_t) == 0
            && "Offset and size must be multiples of 2 for bf16 tensors");

        ggml_tensor_extra_cl * extra = (ggml_tensor_extra_cl *) tensor->extra;
        GGML_ASSERT(extra);

        cl_ulong n_elements = size / sizeof(ggml_fp16_t);
        cl_ulong off_src = (extra->offset + tensor->view_offs + offset) / sizeof(ggml_fp16_t);

        cl_int err;
        cl_mem data_device = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &err);
        CL_CHECK(err);

        cl_kernel kernel = backend_ctx->repack.kernel_convert_f16_to_bf16;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &extra->data_device));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &off_src));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &data_device));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &n_elements));

        size_t global_work_size[] = { (size_t)CEIL_DIV(n_elements, 64)*64, 1, 1 };
        size_t local_work_size[] = { 64, 1, 1 };

        cl_event evt;
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &evt));
        CL_CHECK(clWaitForEvents(1, &evt));
        CL_CHECK(clReleaseEvent(evt));

        CL_CHECK(clEnqueueReadBuffer(
            queue, data_device, CL_TRUE, 0, size, data, 0, NULL, NULL));
        CL_CHECK(clReleaseMemObject(data_device));

        return;
    }

    ggml_tensor_extra_cl * extra = (ggml_tensor_extra_cl *) tensor->extra;

    CL_CHECK(clEnqueueReadBuffer(
        queue, extra->data_device, CL_TRUE, extra->offset + tensor->view_offs + offset,
        size, data, 0, NULL, NULL));

    GGML_UNUSED(buffer);
}
