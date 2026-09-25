#include "../cl-common.h"
#include "../ops.h"

void ggml_cl_moe_combine_fused(ggml_backend_t backend, const ggml_tensor * mul, const ggml_tensor * dst) {
    ggml_backend_opencl_context * backend_ctx = (ggml_backend_opencl_context *)backend->context;
    const ggml_tensor * experts = mul->src[0];
    const ggml_tensor * weights = mul->src[1];

    ggml_tensor_extra_cl * ee = (ggml_tensor_extra_cl *)experts->extra;
    ggml_tensor_extra_cl * ew = (ggml_tensor_extra_cl *)weights->extra;
    ggml_tensor_extra_cl * ed = (ggml_tensor_extra_cl *)dst->extra;
    cl_ulong off_e = ee->offset + experts->view_offs;
    cl_ulong off_w = ew->offset + weights->view_offs;
    cl_ulong off_d = ed->offset + dst->view_offs;

    const int n_embd4 = (int)(experts->ne[0] / 4);
    const int k       = (int)experts->ne[1];
    const int nt      = (int)experts->ne[2];
    const cl_uint e1 = (cl_uint)(experts->nb[1] / sizeof(float));
    const cl_uint e2 = (cl_uint)(experts->nb[2] / sizeof(float));
    const cl_uint w1 = (cl_uint)(weights->nb[1] / sizeof(float));
    const cl_uint w2 = (cl_uint)(weights->nb[2] / sizeof(float));
    const cl_uint d1 = (cl_uint)(dst->nb[1] / sizeof(float));

    // The router weights are tiny ([1,k,nt]) and may share a pool buffer with the output;
    // copy them into a private scratch so the fused kernel never reads aliased memory.
    const size_t w_bytes = ggml_nbytes(weights);
    backend_ctx->prealloc_moe_combine_w.allocate(backend_ctx->context, w_bytes);
    CL_CHECK(clEnqueueCopyBuffer(backend_ctx->queue, ew->data_device, backend_ctx->prealloc_moe_combine_w.buffer,
                                 off_w, 0, w_bytes, 0, NULL, NULL));
    cl_mem   w_dev = backend_ctx->prealloc_moe_combine_w.buffer;
    cl_ulong w_off = 0;

    cl_kernel kernel = backend_ctx->mul_mat_id.kernel_moe_combine_f32;
    int a = 0;
    CL_CHECK(clSetKernelArg(kernel, a++, sizeof(cl_mem),   &ee->data_device));
    CL_CHECK(clSetKernelArg(kernel, a++, sizeof(cl_ulong), &off_e));
    CL_CHECK(clSetKernelArg(kernel, a++, sizeof(cl_mem),   &w_dev));
    CL_CHECK(clSetKernelArg(kernel, a++, sizeof(cl_ulong), &w_off));
    CL_CHECK(clSetKernelArg(kernel, a++, sizeof(cl_mem),   &ed->data_device));
    CL_CHECK(clSetKernelArg(kernel, a++, sizeof(cl_ulong), &off_d));
    CL_CHECK(clSetKernelArg(kernel, a++, sizeof(int),      &n_embd4));
    CL_CHECK(clSetKernelArg(kernel, a++, sizeof(int),      &k));
    CL_CHECK(clSetKernelArg(kernel, a++, sizeof(int),      &nt));
    CL_CHECK(clSetKernelArg(kernel, a++, sizeof(cl_uint),  &e1));
    CL_CHECK(clSetKernelArg(kernel, a++, sizeof(cl_uint),  &e2));
    CL_CHECK(clSetKernelArg(kernel, a++, sizeof(cl_uint),  &w1));
    CL_CHECK(clSetKernelArg(kernel, a++, sizeof(cl_uint),  &w2));
    CL_CHECK(clSetKernelArg(kernel, a++, sizeof(cl_uint),  &d1));

    size_t lws[2] = { 64, 1 };
    size_t gws[2] = { (size_t)(((n_embd4 + 63) / 64) * 64), (size_t)nt };
    backend_ctx->enqueue_ndrange_kernel(kernel, 2, gws, lws, dst);
}

void ggml_cl_moe_bias_glu_fused(ggml_backend_t backend, ggml_tensor * gate_mm, const ggml_tensor * gate_add,
                                ggml_tensor * up_mm, const ggml_tensor * up_add, const ggml_tensor * glu) {
    ggml_backend_opencl_context * backend_ctx = (ggml_backend_opencl_context *) backend->context;

    ggml_cl_mul_mat_id(backend, gate_mm->src[0], gate_mm->src[1], gate_mm);
    ggml_cl_mul_mat_id(backend, up_mm->src[0], up_mm->src[1], up_mm);

    const ggml_tensor * gbias = gate_add->src[1];
    const ggml_tensor * ubias = up_add->src[1];
    const ggml_tensor * ids   = gate_add->src[2];

    ggml_tensor_extra_cl * eg  = (ggml_tensor_extra_cl *) gate_mm->extra;
    ggml_tensor_extra_cl * egb = (ggml_tensor_extra_cl *) gbias->extra;
    ggml_tensor_extra_cl * eu  = (ggml_tensor_extra_cl *) up_mm->extra;
    ggml_tensor_extra_cl * eub = (ggml_tensor_extra_cl *) ubias->extra;
    ggml_tensor_extra_cl * ei  = (ggml_tensor_extra_cl *) ids->extra;
    ggml_tensor_extra_cl * ed  = (ggml_tensor_extra_cl *) glu->extra;

    cl_ulong off_g  = eg->offset  + gate_mm->view_offs;
    cl_ulong off_gb = egb->offset + gbias->view_offs;
    cl_ulong off_u  = eu->offset  + up_mm->view_offs;
    cl_ulong off_ub = eub->offset + ubias->view_offs;
    cl_ulong off_i  = ei->offset  + ids->view_offs;
    cl_ulong off_d  = ed->offset  + glu->view_offs;

    const cl_ulong nb01_g = gate_mm->nb[1];
    const cl_ulong nb02_g = gate_mm->nb[2];
    const cl_ulong nb01_u = up_mm->nb[1];
    const cl_ulong nb02_u = up_mm->nb[2];
    const cl_ulong nb11_g = gbias->nb[1];
    const cl_ulong nb11_u = ubias->nb[1];
    const cl_ulong nb21   = ids->nb[1];
    const cl_ulong nbd1   = glu->nb[1];
    const cl_ulong nbd2   = glu->nb[2];

    const int ne0 = (int) glu->ne[0];
    const float alpha = ggml_get_op_params_f32(glu, 2);
    const float limit = ggml_get_op_params_f32(glu, 3);

    cl_kernel kernel = backend_ctx->mul_mat_id.kernel_add_id_add_id_swiglu_oai;

    int i = 0;
    CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_mem),   &eg->data_device));
    CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_ulong), &off_g));
    CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_mem),   &egb->data_device));
    CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_ulong), &off_gb));
    CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_mem),   &eu->data_device));
    CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_ulong), &off_u));
    CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_mem),   &eub->data_device));
    CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_ulong), &off_ub));
    CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_mem),   &ei->data_device));
    CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_ulong), &off_i));
    CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_mem),   &ed->data_device));
    CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_ulong), &off_d));
    CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_ulong), &nb01_g));
    CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_ulong), &nb02_g));
    CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_ulong), &nb01_u));
    CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_ulong), &nb02_u));
    CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_ulong), &nb11_g));
    CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_ulong), &nb11_u));
    CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_ulong), &nb21));
    CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_ulong), &nbd1));
    CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_ulong), &nbd2));
    CL_CHECK(clSetKernelArg(kernel, i++, sizeof(int),      &ne0));
    CL_CHECK(clSetKernelArg(kernel, i++, sizeof(float),    &limit));
    CL_CHECK(clSetKernelArg(kernel, i++, sizeof(float),    &alpha));

    const int nth = MIN(ne0, (int) backend_ctx->get_kernel_workgroup_size(kernel));
    size_t global_work_size[] = { (size_t) glu->ne[1] * nth, (size_t) glu->ne[2], 1 };
    size_t local_work_size[]  = { (size_t) nth, 1, 1 };

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, glu);
}

void ggml_cl_moe_bias_combine_fused(ggml_backend_t backend, const ggml_tensor * add,
                                    const ggml_tensor * mul, const ggml_tensor * dst) {
    ggml_backend_opencl_context * backend_ctx = (ggml_backend_opencl_context *) backend->context;

    const ggml_tensor * experts = add->src[0];
    const ggml_tensor * bias    = add->src[1];
    const ggml_tensor * ids     = add->src[2];
    const ggml_tensor * weights = mul->src[1];

    ggml_tensor_extra_cl * ee = (ggml_tensor_extra_cl *) experts->extra;
    ggml_tensor_extra_cl * eb = (ggml_tensor_extra_cl *) bias->extra;
    ggml_tensor_extra_cl * ei = (ggml_tensor_extra_cl *) ids->extra;
    ggml_tensor_extra_cl * ew = (ggml_tensor_extra_cl *) weights->extra;
    ggml_tensor_extra_cl * ed = (ggml_tensor_extra_cl *) dst->extra;
    cl_ulong off_e = ee->offset + experts->view_offs;
    cl_ulong off_b = eb->offset + bias->view_offs;
    cl_ulong off_i = ei->offset + ids->view_offs;
    cl_ulong off_w = ew->offset + weights->view_offs;
    cl_ulong off_d = ed->offset + dst->view_offs;

    const int n_embd4 = (int) (experts->ne[0] / 4);
    const int k       = (int) experts->ne[1];
    const int nt      = (int) experts->ne[2];
    const cl_uint e1  = (cl_uint) (experts->nb[1] / sizeof(float));
    const cl_uint e2  = (cl_uint) (experts->nb[2] / sizeof(float));
    const cl_uint w1  = (cl_uint) (weights->nb[1] / sizeof(float));
    const cl_uint w2  = (cl_uint) (weights->nb[2] / sizeof(float));
    const cl_uint d1  = (cl_uint) (dst->nb[1] / sizeof(float));
    const cl_ulong nb_b1 = bias->nb[1];
    const cl_ulong nb_i1 = ids->nb[1];

    const size_t w_bytes = ggml_nbytes(weights);
    backend_ctx->prealloc_moe_combine_w.allocate(backend_ctx->context, w_bytes);
    CL_CHECK(clEnqueueCopyBuffer(backend_ctx->queue, ew->data_device, backend_ctx->prealloc_moe_combine_w.buffer,
                                 off_w, 0, w_bytes, 0, NULL, NULL));
    cl_mem w_dev = backend_ctx->prealloc_moe_combine_w.buffer;
    cl_ulong w_off = 0;

    cl_kernel kernel = backend_ctx->mul_mat_id.kernel_moe_combine_bias_f32;
    int i = 0;
    CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_mem),   &ee->data_device));
    CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_ulong), &off_e));
    CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_mem),   &w_dev));
    CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_ulong), &w_off));
    CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_mem),   &eb->data_device));
    CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_ulong), &off_b));
    CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_mem),   &ei->data_device));
    CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_ulong), &off_i));
    CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_mem),   &ed->data_device));
    CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_ulong), &off_d));
    CL_CHECK(clSetKernelArg(kernel, i++, sizeof(int),      &n_embd4));
    CL_CHECK(clSetKernelArg(kernel, i++, sizeof(int),      &k));
    CL_CHECK(clSetKernelArg(kernel, i++, sizeof(int),      &nt));
    CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_uint),  &e1));
    CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_uint),  &e2));
    CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_uint),  &w1));
    CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_uint),  &w2));
    CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_uint),  &d1));
    CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_ulong), &nb_b1));
    CL_CHECK(clSetKernelArg(kernel, i++, sizeof(cl_ulong), &nb_i1));

    size_t local_work_size[]  = { 64, 1 };
    size_t global_work_size[] = { (size_t) (((n_embd4 + 63) / 64) * 64), (size_t) nt };
    backend_ctx->enqueue_ndrange_kernel(kernel, 2, global_work_size, local_work_size, dst);
}

void ggml_cl_load_kernels_mul_mat_id(ggml_backend_opencl_context * backend_ctx) {
    cl_int err;
    const std::string & compile_opts = backend_ctx->kernel_compile_opts;

    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "moe_add_id_glu.cl.h"
        };
#else
        const std::string kernel_src = read_file("moe_add_id_glu.cl");
#endif
        cl_program prog = build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);
        CL_CHECK((backend_ctx->mul_mat_id.kernel_add_id_add_id_swiglu_oai = clCreateKernel(
            prog, "kernel_add_id_add_id_swiglu_oai", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // moe_combine (fused router-weight mul + cross-expert sum)
    {
    #ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "moe_combine.cl.h"
        };
    #else
        const std::string kernel_src = read_file("moe_combine.cl");
    #endif
        cl_program prog = build_program_from_source(
            backend_ctx, kernel_src.c_str(), compile_opts);
        CL_CHECK((backend_ctx->mul_mat_id.kernel_moe_combine_f32 =
                    clCreateKernel(prog, "kernel_moe_combine_f32", &err), err));
        CL_CHECK((backend_ctx->mul_mat_id.kernel_moe_combine_bias_f32 =
                    clCreateKernel(prog, "kernel_moe_combine_bias_f32", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mv_id_q4_0_f32_8x_flat
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_id_q4_0_f32_8x_flat.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_id_q4_0_f32_8x_flat.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat_id.kernel_mul_mv_id_q4_0_f32_8x_flat = clCreateKernel(prog, "kernel_mul_mv_id_q4_0_f32_8x_flat", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mv_id_q8_0_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_id_q8_0_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_id_q8_0_f32.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat_id.kernel_mul_mv_id_q8_0_f32 = clCreateKernel(prog, "kernel_mul_mv_id_q8_0_f32", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mv_id_q8_0_f32_flat
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_id_q8_0_f32_flat.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_id_q8_0_f32_flat.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat_id.kernel_mul_mv_id_q8_0_f32_flat = clCreateKernel(prog, "kernel_mul_mv_id_q8_0_f32_flat", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mv_id_mxfp4_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_id_mxfp4_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_id_mxfp4_f32.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat_id.kernel_mul_mv_id_mxfp4_f32 = clCreateKernel(prog, "kernel_mul_mv_id_mxfp4_f32", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mv_id_mxfp4_f32_flat
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_id_mxfp4_f32_flat.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_id_mxfp4_f32_flat.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat_id.kernel_mul_mv_id_mxfp4_f32_flat = clCreateKernel(prog, "kernel_mul_mv_id_mxfp4_f32_flat", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }
}

static void moe_router_reoerder(ggml_backend_t backend, const ggml_tensor * src, int ne20) {
    cl_int err;
    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra = (ggml_tensor_extra_cl *)src->extra;
    cl_ulong offset = extra->offset + src->view_offs;

    const int ne21 = src->ne[1];
    const int nb21 = src->nb[1];
    const int ne02 = nb21 / src->nb[0];
    const int n_tile_size = 32;
    const int max_post_router_tile = (ne20 * ne21 / n_tile_size) + ne02;

    cl_buffer_region region;
    region.origin = offset;
    region.size = nb21 * ne21;
    cl_mem original_router_buf = clCreateSubBuffer(extra->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
    CL_CHECK(err);

    backend_ctx->prealloc_post_router.allocate(backend_ctx->context, sizeof(int) * max_post_router_tile * n_tile_size);
    region.origin = 0;
    region.size = sizeof(int) * max_post_router_tile * n_tile_size;
    cl_mem post_router_buf = clCreateSubBuffer(backend_ctx->prealloc_post_router.buffer, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
    CL_CHECK(err);

    backend_ctx->prealloc_emap.allocate(backend_ctx->context, sizeof(short) * max_post_router_tile);
    region.origin = 0;
    region.size = sizeof(short) * max_post_router_tile;
    cl_mem emap_buf = clCreateSubBuffer(backend_ctx->prealloc_emap.buffer, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
    CL_CHECK(err);

    backend_ctx->prealloc_hist.allocate(backend_ctx->context, sizeof(int) * ne02);
    region.origin = 0;
    region.size = sizeof(int) * ne02;
    cl_mem hist_buf = clCreateSubBuffer(backend_ctx->prealloc_hist.buffer, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
    CL_CHECK(err);

    backend_ctx->prealloc_tile_offset.allocate(backend_ctx->context, sizeof(int) * ne02);
    region.origin = 0;
    region.size = sizeof(int) * ne02;
    cl_mem tile_offset_buf = clCreateSubBuffer(backend_ctx->prealloc_tile_offset.buffer, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
    CL_CHECK(err);

    backend_ctx->prealloc_slot_counter.allocate(backend_ctx->context, sizeof(int) * ne02);
    region.origin = 0;
    region.size = sizeof(int) * ne02;
    cl_mem slot_counter_buf = clCreateSubBuffer(backend_ctx->prealloc_slot_counter.buffer, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
    CL_CHECK(err);

    backend_ctx->prealloc_total_tiles.allocate(backend_ctx->context, sizeof(int));
    region.origin = 0;
    region.size = sizeof(int);
    cl_mem total_tiles_buf = clCreateSubBuffer(backend_ctx->prealloc_total_tiles.buffer, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
    CL_CHECK(err);

    // Histogram
    cl_kernel kernel = backend_ctx->mul_mat_id.kernel_moe_histogram;
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &original_router_buf));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &hist_buf));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int), &ne21));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int), &ne20));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int), &ne02));

    size_t histogram_global_size[] = {(size_t)(((ne21 + 63) / 64) * 64), static_cast<size_t>(ne20), 1};
    size_t histogram_local_size[] = {64, 1, 1};
    backend_ctx->enqueue_ndrange_kernel(kernel, 3, histogram_global_size, histogram_local_size, src);

    // Scan
    kernel = backend_ctx->mul_mat_id.kernel_moe_scan;
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &hist_buf));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &tile_offset_buf));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &total_tiles_buf));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &slot_counter_buf));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int), &n_tile_size));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int), &ne02));

    size_t scan_global_size[] = {1};
    size_t scan_local_size[] = {1};
    backend_ctx->enqueue_ndrange_kernel(kernel, 1, scan_global_size, scan_local_size, src);

    // Fill
    kernel = backend_ctx->mul_mat_id.kernel_moe_fill;
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &post_router_buf));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &total_tiles_buf));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int), &n_tile_size));

    size_t fill_global_size[] = {(size_t)(((max_post_router_tile + 63) / 64) * 64), n_tile_size, 1};
    size_t fill_local_size[] = {64, 1, 1};
    backend_ctx->enqueue_ndrange_kernel(kernel, 3, fill_global_size, fill_local_size, src);

    static const bool stable_scatter = []{
        const char * env = getenv("GGML_OPENCL_MOE_STABLE_SCATTER");
        return !env || env[0] == '\0' || env[0] != '0';
    }();

    if (stable_scatter) {
        kernel = backend_ctx->mul_mat_id.kernel_moe_scatter_stable;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &original_router_buf));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &post_router_buf));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &emap_buf));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &tile_offset_buf));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int), &ne21));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int), &ne20));
        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int), &ne02));

        size_t scatter_global_size[] = {64, (size_t) ne02};
        size_t scatter_local_size[]  = {64, 1};
        backend_ctx->enqueue_ndrange_kernel(kernel, 2, scatter_global_size, scatter_local_size, src);
    } else {
        kernel = backend_ctx->mul_mat_id.kernel_moe_scatter;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &original_router_buf));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &post_router_buf));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &emap_buf));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &tile_offset_buf));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), &slot_counter_buf));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int), &ne21));
        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int), &ne20));
        CL_CHECK(clSetKernelArg(kernel, 7, sizeof(int), &ne02));

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, histogram_global_size, histogram_local_size, src);
    }

    // [MOE_TILES] env-gated padding probe: read back total_tiles (= Sum_e
    // ceil(k_e/n_tile_size)) and compare to the ideal tile count for the real
    // routing count. Quantifies the per-expert tile-padding waste. Blocking
    // readback perturbs timing -> diagnostic only.
    if (getenv("GGML_OPENCL_MOE_TILES_DEBUG")) {
        int h_total = 0;
        clFinish(backend_ctx->queue);
        CL_CHECK(clEnqueueReadBuffer(backend_ctx->queue, total_tiles_buf, CL_TRUE, 0, sizeof(int), &h_total, 0, NULL, NULL));
        const int routings = ne20 * ne21;
        const int ideal    = (routings + n_tile_size - 1) / n_tile_size;
        const int slots     = h_total * n_tile_size;
        fprintf(stderr, "[MOE_TILES] routings=%d (ne20=%d ne21=%d nexp=%d) total_tiles=%d ideal=%d slots=%d pad=%.1f%%\n",
                routings, ne20, ne21, ne02, h_total, ideal, slots,
                routings > 0 ? 100.0 * (slots - routings) / routings : 0.0);
        fflush(stderr);
    }

    CL_CHECK(clReleaseMemObject(original_router_buf));
    CL_CHECK(clReleaseMemObject(hist_buf));
    CL_CHECK(clReleaseMemObject(tile_offset_buf));
    CL_CHECK(clReleaseMemObject(total_tiles_buf));
    CL_CHECK(clReleaseMemObject(slot_counter_buf));
    CL_CHECK(clReleaseMemObject(post_router_buf));
    CL_CHECK(clReleaseMemObject(emap_buf));
}

void ggml_cl_mul_mat_id(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    const ggml_tensor * src2 = dst->src[2];
    GGML_ASSERT(src2);
    GGML_ASSERT(src2->extra);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extra2 = (ggml_tensor_extra_cl *)src2->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offset2 = extra2->offset + src2->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    GGML_UNUSED(offset0);

#ifdef GGML_OPENCL_SOA_Q
    // SoA extra lives on view_src (view->extra is pre-SoA).
    const ggml_tensor * soa0_src = src0->view_src != nullptr ? src0->view_src : src0;
    ggml_tensor_extra_cl_q4_0 * extra0_q4_0 = (ggml_tensor_extra_cl_q4_0 *)soa0_src->extra;
    ggml_tensor_extra_cl_q4_1 * extra0_q4_1 = (ggml_tensor_extra_cl_q4_1 *)soa0_src->extra;
    ggml_tensor_extra_cl_q5_0 * extra0_q5_0 = (ggml_tensor_extra_cl_q5_0 *)soa0_src->extra;
    ggml_tensor_extra_cl_q5_1 * extra0_q5_1 = (ggml_tensor_extra_cl_q5_1 *)soa0_src->extra;
    ggml_tensor_extra_cl_q4_K * extra0_q4_K = (ggml_tensor_extra_cl_q4_K *)soa0_src->extra;
    ggml_tensor_extra_cl_q5_K * extra0_q5_K = (ggml_tensor_extra_cl_q5_K *)soa0_src->extra;
    ggml_tensor_extra_cl_q6_K * extra0_q6_K = (ggml_tensor_extra_cl_q6_K *)soa0_src->extra;
    ggml_tensor_extra_cl_mxfp4 * extra0_mxfp4 = (ggml_tensor_extra_cl_mxfp4 *)soa0_src->extra;
    ggml_tensor_extra_cl_q8_0 * extra0_q8_0 = (ggml_tensor_extra_cl_q8_0 *)soa0_src->extra;

#endif

    // TODO: general MoE for the following types
    (void)extra0_q4_1;
    (void)extra0_q5_0;
    (void)extra0_q5_1;
    (void)extra0_q4_K;
    (void)extra0_q5_K;
    (void)extra0_q6_K;

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const cl_ulong nb00 = src0->nb[0];
    const cl_ulong nb01 = src0->nb[1];
    const cl_ulong nb02 = src0->nb[2];
    const cl_ulong nb03 = src0->nb[3];

    const int ne10 = src1->ne[0];
    const int ne11 = src1->ne[1];
    const int ne12 = src1->ne[2];
    const int ne13 = src1->ne[3];

    const cl_ulong nb11 = src1->nb[1];
    const cl_ulong nb12 = src1->nb[2];
    const cl_ulong nb13 = src1->nb[3];

    const int ne20 = src2->ne[0];
    const int ne21 = src2->ne[1];

    const cl_ulong nb21 = src2->nb[1];
    const cl_ulong nb20 = src2->nb[0];

    UNUSED(nb20);

    const int ne0 = dst->ne[0];
    const int ne1 = dst->ne[1];
    const int ne2 = dst->ne[2];

    GGML_UNUSED(ne2);

    const int r2 = ne12/ne02;
    const int r3 = ne13/ne03;
    const int dst_rows = ne20*ne21; // ne20 = n_used_experts, ne21 = n_rows

    GGML_ASSERT(ne00 == ne10);

    int sgs   = 32; // subgroup size
    int nsg   = 1;  // number of subgroups
    int nrows = 1;  // number of row in src1
    int ndst  = 4;  // number of values produced by each subgroup

    const int n_tile_size = 32;
    const int max_post_router_tile = (ne20 * ne21 / n_tile_size) + ne02;

    GGML_UNUSED(max_post_router_tile);

    cl_kernel kernel;

    // subgroup mat vec
    switch (src0->type) {
        case GGML_TYPE_Q4_0: {
#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
            if (use_adreno_moe_kernels(backend_ctx, src0)) {
                cl_int status;

                size_t local_size[3] = {64, 2, 1};
                size_t global_size[3] = {64, 2, 1};

                if (ne12 == 1) { // for gemv
                    kernel = backend_ctx->mul_mat_id.kernel_gemv_moe_q4_0_f32_ns;

                    cl_mem src1_sub_buffer, buf_src1_image, buf_src2;

                    // create a sub_buffer for src2
                    cl_buffer_region region;
                    region.origin = offset2;
                    region.size = ne20 * ne21 * sizeof(int);
                    buf_src2 = clCreateSubBuffer(extra2->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                    CL_CHECK(status);

                    // set thread grid
                    global_size[0] = static_cast<size_t>(((ne01 + 63) / 64) * 64);
                    global_size[1] = 4;
                    global_size[2] = static_cast<size_t>(ne20);
                    local_size[1] = 4;

                    // create a sub_buffer for src1
                    region.origin = offset1;
                    region.size = ne10 * ne11 * ne12 * sizeof(float);
                    src1_sub_buffer = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                    CL_CHECK(status);

                    // create image for src1
                    cl_image_format image_format_buf_src1 = {CL_RGBA, CL_FLOAT};
                    cl_image_desc image_desc_buf_src1 = {CL_MEM_OBJECT_IMAGE1D_BUFFER, static_cast<size_t>(ne10 * ne11 * ne12 / 4), 0,0,0,0,0,0,0, {src1_sub_buffer}};
                    buf_src1_image = clCreateImage(backend_ctx->context, CL_MEM_READ_ONLY, &image_format_buf_src1, &image_desc_buf_src1, NULL, &status);
                    CL_CHECK(status);

                    // Set kernel args
                    int arg_idx = 0;
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_q4_0->q));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_q4_0->d));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &buf_src1_image));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &buf_src2));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extrad->data_device));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_ulong),  &offsetd));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),       &ne00));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),       &ne01));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),       &ne11));

                    // launch kernel
                    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_size, local_size, dst);

                    // deallocate sub buffers and images
                    CL_CHECK(clReleaseMemObject(src1_sub_buffer));
                    CL_CHECK(clReleaseMemObject(buf_src1_image));
                    CL_CHECK(clReleaseMemObject(buf_src2));

                } else { // for gemm
                    // dp4a (int8) prefill GEMM variant
                    static const char * q4_0_moe_dp4a_env = getenv("GGML_OPENCL_Q4_0_MOE_DP4A");

                    // It turns out that the prebuilt kernel only outperforms the dp4a variant (on X2-90)
                    // at very large routing counts, so we gate its use accordingly using moe_bin_min,
                    // which can be overridden via the GGML_OPENCL_MOE_BIN_MIN_ROUTINGS environment variable.
                    // The routing count is ne20 * ne21 (n_expert_used * n_tokens).
                    static const char * moe_bin_min_env = getenv("GGML_OPENCL_MOE_BIN_MIN_ROUTINGS");
                    const int  moe_bin_min   = moe_bin_min_env ? atoi(moe_bin_min_env) : 4096;

                    // whether bin kernels are available
                    const bool bin_available = backend_ctx->mul_mat_id.kernel_gemm_moe_q4_0_f32_ns_bin != nullptr;
                    const bool dp4a_bin_available = backend_ctx->mul_mat_id.kernel_gemm_moe_q4_0_q8_1_dp4a_bin != nullptr;

                    bool use_moe_dp4a = q4_0_moe_dp4a_env
                        ? (atoi(q4_0_moe_dp4a_env) != 0)
                        : (backend_ctx->adreno_gen == ADRENO_GPU_GEN::X2E
                           && (dp4a_bin_available || !bin_available
                               || (int)(ne20 * ne21) < moe_bin_min));
                    // dot prod has to be available
                    use_moe_dp4a = backend_ctx->has_integer_dot && use_moe_dp4a;

                    const bool use_bin_kernel = bin_available && !use_moe_dp4a;

                    kernel = use_bin_kernel
                        ? backend_ctx->mul_mat_id.kernel_gemm_moe_q4_0_f32_ns_bin
                        : backend_ctx->mul_mat_id.kernel_gemm_moe_q4_0_f32_ns;

                    // Reorder router if called from test-backend-ops or when new router is generated.
                    // Otherwise reuse the reordered result from previous mul_mat_id call.
                    if ((strstr(src0->name, "as") != NULL) || backend_ctx->toggle_reorder) {
                        moe_router_reoerder(backend, src2, ne20);
                        backend_ctx->toggle_reorder = false;
                    }

                    cl_mem sub_buf_src1_pre, sub_buf_dst, buf_dst_image;
                    cl_mem buf_src1_reordered = nullptr, image_src1_reordered = nullptr;
                    cl_mem buf_src2, buf_src2_emap;

                    cl_buffer_region region;
                    region.origin = 0;
                    region.size = sizeof(int) * max_post_router_tile * n_tile_size;
                    buf_src2 = clCreateSubBuffer(backend_ctx->prealloc_post_router.buffer, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                    CL_CHECK(status);

                    region.origin = 0;
                    region.size = sizeof(short) * max_post_router_tile;
                    buf_src2_emap = clCreateSubBuffer(backend_ctx->prealloc_emap.buffer, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                    CL_CHECK(status);

                    // Reorder activations
                    // create a sub_buffer for src1
                    region.origin = offset1;
                    region.size = ne10 * ne11 * ne12 * sizeof(float);
                    sub_buf_src1_pre = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                    CL_CHECK(status);

                    unsigned short map_ratio = ne20 / ne11;
                    GGML_ASSERT(((map_ratio == 1) || (map_ratio == ne20)) && "Map ratio not supported\n");

                    if (!use_moe_dp4a) {
                        // Create image for reordered src1
                        // Use pre-allocated placeholder
                        region.origin = 0;
                        region.size = ne00 * max_post_router_tile * n_tile_size * sizeof(float);
                        backend_ctx->prealloc_act_trans.allocate(backend_ctx->context, region.size);
                        buf_src1_reordered = clCreateSubBuffer(
                            backend_ctx->prealloc_act_trans.buffer,
                            0,
                            CL_BUFFER_CREATE_TYPE_REGION,
                            &region,
                            &status);
                        CL_CHECK(status);
                        cl_image_format image_format_buf_src1;
                        cl_image_desc image_desc_buf_src1;
                        image_format_buf_src1 = {CL_RGBA, CL_FLOAT};
                        image_desc_buf_src1 = {CL_MEM_OBJECT_IMAGE1D_BUFFER, static_cast<size_t>(ne00 * max_post_router_tile * n_tile_size / 4), 0,0,0,0,0,0,0, {buf_src1_reordered}};
                        if (use_bin_kernel) {
                            // bin kernel uses slightly different image format
                            image_format_buf_src1 = {CL_R, CL_FLOAT};
                            image_desc_buf_src1.image_width = static_cast<size_t>(ne00 * max_post_router_tile * n_tile_size);
                        }
                        image_src1_reordered = clCreateImage(backend_ctx->context, CL_MEM_READ_ONLY, &image_format_buf_src1, &image_desc_buf_src1, NULL, &status);
                        CL_CHECK(status);

                        CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 0, sizeof(cl_mem),        &sub_buf_src1_pre));
                        CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 1, sizeof(cl_mem),        &buf_src2));
                        CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 2, sizeof(cl_mem),        &buf_src1_reordered));
                        CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 3, sizeof(cl_mem),        &(backend_ctx->prealloc_total_tiles.buffer)));
                        CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 4, sizeof(unsigned int),  &ne00));
                        CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 5, sizeof(unsigned short),  &map_ratio));
                        CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 6, sizeof(unsigned int),  &n_tile_size));

                        size_t reorder_b_local_size[3] = {256, 1, 1};
                        size_t reorder_b_global_size[3] = {static_cast<size_t>(((ne00 / 4) + 255) / 256 * 256), static_cast<size_t>(max_post_router_tile * n_tile_size), 1};

                        // Dispatch reorder kernel
                        backend_ctx->enqueue_ndrange_kernel(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 3, reorder_b_global_size, reorder_b_local_size, dst);
                    }

                    // MoE kernel prepare
                    // Create sub buffer for dst
                    region.origin = offsetd;
                    region.size = ne0 * ne1 * ne2 * sizeof(float);
                    sub_buf_dst = clCreateSubBuffer(
                        extrad->data_device,
                        0,
                        CL_BUFFER_CREATE_TYPE_REGION,
                        &region,
                        &status);
                    CL_CHECK(status);
                    // Create image for dst
                    cl_image_format image_format_buf_dst = {CL_R, CL_FLOAT};
                    cl_image_desc image_desc_buf_dst = {CL_MEM_OBJECT_IMAGE1D_BUFFER, static_cast<size_t>(ne0 * ne1 * ne2), 0,0,0,0,0,0,0, {sub_buf_dst}};
                    buf_dst_image = clCreateImage(backend_ctx->context, CL_MEM_WRITE_ONLY, &image_format_buf_dst, &image_desc_buf_dst, NULL, &status);
                    CL_CHECK(status);

                    if (use_moe_dp4a) {
                        const size_t tok_slots = (size_t)max_post_router_tile * n_tile_size;
                        const size_t n_blocks  = tok_slots * (ne00 / 32);
                        backend_ctx->prealloc_moe_qa.allocate(backend_ctx->context, tok_slots * ne00 * sizeof(cl_char));
                        backend_ctx->prealloc_moe_da.allocate(backend_ctx->context, n_blocks * sizeof(cl_half));
                        backend_ctx->prealloc_moe_sa.allocate(backend_ctx->context, n_blocks * sizeof(cl_half));

                        // fused reorder + q8_1 quant straight from the original activations
                        const cl_uint n_kblocks = (cl_uint)(ne00 / 32);
                        cl_kernel rq = backend_ctx->mul_mat_id.kernel_moe_reorder_quant_a_q8_1;
                        CL_CHECK(clSetKernelArg(rq, 0, sizeof(cl_mem),         &sub_buf_src1_pre));
                        CL_CHECK(clSetKernelArg(rq, 1, sizeof(cl_mem),         &buf_src2));
                        CL_CHECK(clSetKernelArg(rq, 2, sizeof(cl_mem),         &backend_ctx->prealloc_moe_qa.buffer));
                        CL_CHECK(clSetKernelArg(rq, 3, sizeof(cl_mem),         &backend_ctx->prealloc_moe_da.buffer));
                        CL_CHECK(clSetKernelArg(rq, 4, sizeof(cl_mem),         &backend_ctx->prealloc_moe_sa.buffer));
                        CL_CHECK(clSetKernelArg(rq, 5, sizeof(cl_mem),         &(backend_ctx->prealloc_total_tiles.buffer)));
                        CL_CHECK(clSetKernelArg(rq, 6, sizeof(cl_uint),        &ne00));
                        CL_CHECK(clSetKernelArg(rq, 7, sizeof(unsigned short), &map_ratio));
                        CL_CHECK(clSetKernelArg(rq, 8, sizeof(cl_uint),        &n_tile_size));
                        CL_CHECK(clSetKernelArg(rq, 9, sizeof(cl_uint),        &n_kblocks));
                        size_t rq_local[2]  = { 32, 1 };
                        size_t rq_global[2] = { (size_t)(((n_kblocks + 31) / 32) * 32), tok_slots };
                        backend_ctx->enqueue_ndrange_kernel(rq, 2, rq_global, rq_local, dst);

                        // dp4a GEMM
                        cl_kernel dk = backend_ctx->mul_mat_id.kernel_gemm_moe_q4_0_q8_1_dp4a;
                        if (backend_ctx->mul_mat_id.kernel_gemm_moe_q4_0_q8_1_dp4a_bin) {
                            dk = backend_ctx->mul_mat_id.kernel_gemm_moe_q4_0_q8_1_dp4a_bin;
                        }
                        int aidx = 0;
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &extra0_q4_0->q_img));
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &extra0_q4_0->d));
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &backend_ctx->prealloc_moe_qa.buffer));
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &backend_ctx->prealloc_moe_da.buffer));
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &backend_ctx->prealloc_moe_sa.buffer));
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &buf_src2));
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &buf_src2_emap));
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &buf_dst_image));
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &(backend_ctx->prealloc_total_tiles.buffer)));
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(int),    &ne00));
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(int),    &ne01));
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(int),    &backend_ctx->adreno_use_moe_ragged_dp4));

                        size_t dp_global[3] = { 64, (size_t)((ne01 + 63) / 64), (size_t)max_post_router_tile };
                        size_t dp_local[3]  = { 64, 1, 1 };
                        backend_ctx->enqueue_ndrange_kernel(dk, 3, dp_global, dp_local, dst);

                        clReleaseMemObject(sub_buf_src1_pre);
                        clReleaseMemObject(buf_src2);
                        clReleaseMemObject(buf_src2_emap);
                        clReleaseMemObject(sub_buf_dst);
                        clReleaseMemObject(buf_dst_image);
                        return;
                    }

                    // Set kernel args
                    int arg_idx = 0;
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_q4_0->q_img));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_q4_0->d));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &image_src1_reordered));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &buf_src2));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &buf_src2_emap));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &buf_dst_image));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &(backend_ctx->prealloc_total_tiles.buffer)));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),       &ne00));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),       &ne01));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_uint),   &backend_ctx->adreno_use_moe_ragged));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_uint),   &backend_ctx->adreno_moe_ragged_skip_gran));

                    // set thread grid
                    global_size[1] = static_cast<size_t>((ne01 + 63) / 64);
                    global_size[2] = static_cast<size_t>(max_post_router_tile);
                    local_size[1] = 1;
                    local_size[2] = 1;

                    // Dispatch kernel
                    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_size, local_size, dst);

                    clReleaseMemObject(sub_buf_src1_pre);
                    clReleaseMemObject(buf_src1_reordered);
                    clReleaseMemObject(image_src1_reordered);
                    clReleaseMemObject(buf_src2);
                    clReleaseMemObject(buf_src2_emap);
                    clReleaseMemObject(sub_buf_dst);
                    clReleaseMemObject(buf_dst_image);
                }
                return;
            } // fallback to generic Q4_0 MoE kernel

#endif // GGML_OPENCL_USE_ADRENO_KERNELS
            kernel = backend_ctx->mul_mat_id.kernel_mul_mv_id_q4_0_f32_8x_flat;

            if (backend_ctx->gpu_family == INTEL) {
                sgs  = 16;
                nsg  = 1;
                ndst = 8;
            } else if (backend_ctx->gpu_family == ADRENO) {
                sgs  = 64;
                nsg  = 1;
                ndst = 8;
            } else {
                GGML_ASSERT(false && "TODO: Unknown GPU");
            }

            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0_q4_0->q));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0_q4_0->d));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extra2->data_device));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offset2));
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne01));
            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne02));
            CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb00));
            CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb02));
            CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne10));
            CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &ne11));
            CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne12));
            CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nb11));
            CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong), &nb12));
            CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &ne20));
            CL_CHECK(clSetKernelArg(kernel, 19, sizeof(int),      &ne21));
            CL_CHECK(clSetKernelArg(kernel, 20, sizeof(cl_ulong), &nb21));
            CL_CHECK(clSetKernelArg(kernel, 21, sizeof(int),      &ne0));
            CL_CHECK(clSetKernelArg(kernel, 22, sizeof(int),      &ne1));
            CL_CHECK(clSetKernelArg(kernel, 23, sizeof(int),      &r2));
            CL_CHECK(clSetKernelArg(kernel, 24, sizeof(int),      &r3));

            break;
        }
        case GGML_TYPE_Q4_1: {
#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
            if (use_adreno_moe_kernels(backend_ctx, src0)) {
                cl_int status;

                size_t local_size[3] = {64, 2, 1};
                size_t global_size[3] = {64, 2, 1};

                if (ne12 == 1) { // for gemv
                    kernel = backend_ctx->mul_mat_id.kernel_gemv_moe_q4_1_f32_ns;

                    cl_mem src1_sub_buffer, buf_src1_image, buf_src2;

                    // create a sub_buffer for src2
                    cl_buffer_region region;
                    region.origin = offset2;
                    region.size = ne20 * ne21 * sizeof(int);
                    buf_src2 = clCreateSubBuffer(extra2->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                    CL_CHECK(status);

                    // set thread grid
                    global_size[0] = static_cast<size_t>(((ne01 + 63) / 64) * 64);
                    global_size[1] = 4;
                    global_size[2] = static_cast<size_t>(ne20);
                    local_size[1] = 4;

                    // create a sub_buffer for src1
                    region.origin = offset1;
                    region.size = ne10 * ne11 * ne12 * sizeof(float);
                    src1_sub_buffer = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                    CL_CHECK(status);

                    // create image for src1
                    cl_image_format image_format_buf_src1 = {CL_RGBA, CL_FLOAT};
                    cl_image_desc image_desc_buf_src1 = {CL_MEM_OBJECT_IMAGE1D_BUFFER, static_cast<size_t>(ne10 * ne11 * ne12 / 4), 0,0,0,0,0,0,0, {src1_sub_buffer}};
                    buf_src1_image = clCreateImage(backend_ctx->context, CL_MEM_READ_ONLY, &image_format_buf_src1, &image_desc_buf_src1, NULL, &status);
                    CL_CHECK(status);

                    // Set kernel args
                    int arg_idx = 0;
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_q4_1->q));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_q4_1->d));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_q4_1->m));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &buf_src1_image));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &buf_src2));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extrad->data_device));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_ulong),  &offsetd));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),       &ne00));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),       &ne01));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),       &ne11));

                    // launch kernel
                    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_size, local_size, dst);

                    // deallocate sub buffers and images
                    CL_CHECK(clReleaseMemObject(src1_sub_buffer));
                    CL_CHECK(clReleaseMemObject(buf_src1_image));
                    CL_CHECK(clReleaseMemObject(buf_src2));

                } else { // for gemm
                    kernel = backend_ctx->mul_mat_id.kernel_gemm_moe_q4_1_f32_ns;
                    if (backend_ctx->mul_mat_id.kernel_gemm_moe_q4_1_f32_ns_bin) {
                        kernel = backend_ctx->mul_mat_id.kernel_gemm_moe_q4_1_f32_ns_bin;
                    }

                    // Reorder router if called from test-backend-ops or when new router is generated.
                    // Otherwise reuse the reordered result from previous mul_mat_id call.
                    if ((strstr(src0->name, "as") != NULL) || backend_ctx->toggle_reorder) {
                        moe_router_reoerder(backend, src2, ne20);
                        backend_ctx->toggle_reorder = false;
                    }

                    cl_mem sub_buf_src1_pre, buf_src1_reordered, image_src1_reordered, sub_buf_dst, buf_dst_image;
                    cl_mem buf_src2, buf_src2_emap;

                    cl_buffer_region region;
                    region.origin = 0;
                    region.size = sizeof(int) * max_post_router_tile * n_tile_size;
                    buf_src2 = clCreateSubBuffer(backend_ctx->prealloc_post_router.buffer, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                    CL_CHECK(status);

                    region.origin = 0;
                    region.size = sizeof(short) * max_post_router_tile;
                    buf_src2_emap = clCreateSubBuffer(backend_ctx->prealloc_emap.buffer, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                    CL_CHECK(status);

                    // Reorder activations
                    // create a sub_buffer for src1
                    region.origin = offset1;
                    region.size = ne10 * ne11 * ne12 * sizeof(float);
                    sub_buf_src1_pre = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                    CL_CHECK(status);

                    // Create image for reordered src1
                    // Use pre-allocated placeholder
                    region.origin = 0;
                    region.size = ne00 * max_post_router_tile * n_tile_size * sizeof(float);
                    backend_ctx->prealloc_act_trans.allocate(backend_ctx->context, region.size);
                    buf_src1_reordered = clCreateSubBuffer(
                        backend_ctx->prealloc_act_trans.buffer,
                        0,
                        CL_BUFFER_CREATE_TYPE_REGION,
                        &region,
                        &status);
                    CL_CHECK(status);
                    cl_image_format image_format_buf_src1;
                    cl_image_desc image_desc_buf_src1;
                    image_format_buf_src1 = {CL_RGBA, CL_FLOAT};
                    image_desc_buf_src1 = {CL_MEM_OBJECT_IMAGE1D_BUFFER, static_cast<size_t>(ne00 * max_post_router_tile * n_tile_size / 4), 0,0,0,0,0,0,0, {buf_src1_reordered}};
                    if (backend_ctx->mul_mat_id.kernel_gemm_moe_q4_1_f32_ns_bin) {
                        // bin kernel uses slightly different image format
                        image_format_buf_src1 = {CL_R, CL_FLOAT};
                        image_desc_buf_src1.image_width = static_cast<size_t>(ne00 * max_post_router_tile * n_tile_size);
                    }
                    image_src1_reordered = clCreateImage(backend_ctx->context, CL_MEM_READ_ONLY, &image_format_buf_src1, &image_desc_buf_src1, NULL, &status);
                    CL_CHECK(status);

                    unsigned short map_ratio = ne20 / ne11;
                    GGML_ASSERT(((map_ratio == 1) || (map_ratio == ne20)) && "Map ratio not supported\n");
                    CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 0, sizeof(cl_mem),        &sub_buf_src1_pre));
                    CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 1, sizeof(cl_mem),        &buf_src2));
                    CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 2, sizeof(cl_mem),        &buf_src1_reordered));
                    CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 3, sizeof(cl_mem),        &(backend_ctx->prealloc_total_tiles.buffer)));
                    CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 4, sizeof(unsigned int),  &ne00));
                    CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 5, sizeof(unsigned short),  &map_ratio));
                    CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 6, sizeof(unsigned int),  &n_tile_size));

                    size_t reorder_b_local_size[3] = {256, 1, 1};
                    size_t reorder_b_global_size[3] = {static_cast<size_t>(((ne00 / 4) + 255) / 256 * 256), static_cast<size_t>(max_post_router_tile * n_tile_size), 1};

                    // Dispatch reorder kernel
                    backend_ctx->enqueue_ndrange_kernel(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 3, reorder_b_global_size, reorder_b_local_size, dst);

                    // MoE kernel prepare
                    // Create sub buffer for dst
                    region.origin = offsetd;
                    region.size = ne0 * ne1 * ne2 * sizeof(float);
                    sub_buf_dst = clCreateSubBuffer(
                        extrad->data_device,
                        0,
                        CL_BUFFER_CREATE_TYPE_REGION,
                        &region,
                        &status);
                    CL_CHECK(status);
                    // Create image for dst
                    cl_image_format image_format_buf_dst = {CL_R, CL_FLOAT};
                    cl_image_desc image_desc_buf_dst = {CL_MEM_OBJECT_IMAGE1D_BUFFER, static_cast<size_t>(ne0 * ne1 * ne2), 0,0,0,0,0,0,0, {sub_buf_dst}};
                    buf_dst_image = clCreateImage(backend_ctx->context, CL_MEM_WRITE_ONLY, &image_format_buf_dst, &image_desc_buf_dst, NULL, &status);
                    CL_CHECK(status);

                    // Set kernel args
                    int arg_idx = 0;
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_q4_1->q_img));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_q4_1->d));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_q4_1->m));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &image_src1_reordered));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &buf_src2));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &buf_src2_emap));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &buf_dst_image));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &(backend_ctx->prealloc_total_tiles.buffer)));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),       &ne00));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),       &ne01));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_uint),   &backend_ctx->adreno_use_moe_ragged));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_uint),   &backend_ctx->adreno_moe_ragged_skip_gran));

                    // set thread grid
                    global_size[1] = static_cast<size_t>((ne01 + 63) / 64);
                    global_size[2] = static_cast<size_t>(max_post_router_tile);
                    local_size[1] = 1;
                    local_size[2] = 1;

                    // Dispatch kernel
                    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_size, local_size, dst);

                    clReleaseMemObject(sub_buf_src1_pre);
                    clReleaseMemObject(buf_src1_reordered);
                    clReleaseMemObject(image_src1_reordered);
                    clReleaseMemObject(buf_src2);
                    clReleaseMemObject(buf_src2_emap);
                    clReleaseMemObject(sub_buf_dst);
                    clReleaseMemObject(buf_dst_image);
                }
                return;
            }
#endif //GGML_OPENCL_USE_ADRENO_KERNELS
        }
        case GGML_TYPE_Q5_0: {
#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
            if (use_adreno_moe_kernels(backend_ctx, src0)) {
                cl_int status;

                size_t local_size[3] = {64, 2, 1};
                size_t global_size[3] = {64, 2, 1};

                if (ne12 == 1) { // for gemv
                    kernel = backend_ctx->mul_mat_id.kernel_gemv_moe_q5_0_f32_ns;

                    cl_mem src1_sub_buffer, buf_src1_image, buf_src2;

                    // create a sub_buffer for src2
                    cl_buffer_region region;
                    region.origin = offset2;
                    region.size = ne20 * ne21 * sizeof(int);
                    buf_src2 = clCreateSubBuffer(extra2->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                    CL_CHECK(status);

                    // set thread grid
                    global_size[0] = static_cast<size_t>(((ne01 + 63) / 64) * 64);
                    global_size[1] = 4;
                    global_size[2] = static_cast<size_t>(ne20);
                    local_size[1] = 4;

                    // create a sub_buffer for src1
                    region.origin = offset1;
                    region.size = ne10 * ne11 * ne12 * sizeof(float);
                    src1_sub_buffer = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                    CL_CHECK(status);

                    // create image for src1
                    cl_image_format image_format_buf_src1 = {CL_RGBA, CL_FLOAT};
                    cl_image_desc image_desc_buf_src1 = {CL_MEM_OBJECT_IMAGE1D_BUFFER, static_cast<size_t>(ne10 * ne11 * ne12 / 4), 0,0,0,0,0,0,0, {src1_sub_buffer}};
                    buf_src1_image = clCreateImage(backend_ctx->context, CL_MEM_READ_ONLY, &image_format_buf_src1, &image_desc_buf_src1, NULL, &status);
                    CL_CHECK(status);

                    // Set kernel args
                    int arg_idx = 0;
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_q5_0->qs));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_q5_0->qh));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_q5_0->d));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &buf_src1_image));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &buf_src2));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extrad->data_device));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_ulong),  &offsetd));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),       &ne00));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),       &ne01));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),       &ne11));

                    // launch kernel
                    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_size, local_size, dst);

                    // deallocate sub buffers and images
                    CL_CHECK(clReleaseMemObject(src1_sub_buffer));
                    CL_CHECK(clReleaseMemObject(buf_src1_image));
                    CL_CHECK(clReleaseMemObject(buf_src2));

                } else { // for gemm
                    kernel = backend_ctx->mul_mat_id.kernel_gemm_moe_q5_0_f32_ns;

                    // Reorder router if called from test-backend-ops or when new router is generated.
                    // Otherwise reuse the reordered result from previous mul_mat_id call.
                    if ((strstr(src0->name, "as") != NULL) || backend_ctx->toggle_reorder) {
                        moe_router_reoerder(backend, src2, ne20);
                        backend_ctx->toggle_reorder = false;
                    }

                    cl_mem sub_buf_src1_pre, buf_src1_reordered, image_src1_reordered, sub_buf_dst, buf_dst_image;
                    cl_mem buf_src2, buf_src2_emap;

                    cl_buffer_region region;
                    region.origin = 0;
                    region.size = sizeof(int) * max_post_router_tile * n_tile_size;
                    buf_src2 = clCreateSubBuffer(backend_ctx->prealloc_post_router.buffer, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                    CL_CHECK(status);

                    region.origin = 0;
                    region.size = sizeof(short) * max_post_router_tile;
                    buf_src2_emap = clCreateSubBuffer(backend_ctx->prealloc_emap.buffer, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                    CL_CHECK(status);

                    // Reorder activations
                    // create a sub_buffer for src1
                    region.origin = offset1;
                    region.size = ne10 * ne11 * ne12 * sizeof(float);
                    sub_buf_src1_pre = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                    CL_CHECK(status);

                    // Generic dp4a MoE GEMM
                    {
                        static const char * q5mdp4a_env = getenv("GGML_OPENCL_Q5_MOE_DP4A");
                        const bool q5mdp4a_on = q5mdp4a_env ? (atoi(q5mdp4a_env) != 0)
                                                            : (backend_ctx->adreno_gen == ADRENO_GPU_GEN::X2E);
                        const bool use_q5_moe_dp4a = q5mdp4a_on
                            && backend_ctx->mul_mat_id.kernel_gemm_moe_q8_1_dp4a_q50 != nullptr
                            && extra0_q5_0->scale != nullptr;

                        if (use_q5_moe_dp4a) {
                            const size_t tok_slots = (size_t)max_post_router_tile * n_tile_size;
                            const size_t n_blocks  = tok_slots * (ne00 / 32);
                            backend_ctx->prealloc_moe_qa.allocate(backend_ctx->context, tok_slots * ne00 * sizeof(cl_char));
                            backend_ctx->prealloc_moe_da.allocate(backend_ctx->context, n_blocks * sizeof(cl_half));
                            backend_ctx->prealloc_moe_sa.allocate(backend_ctx->context, n_blocks * sizeof(cl_half));

                            const cl_uint n_kblocks = (cl_uint)(ne00 / 32);
                            unsigned short map_ratio_q5 = ne20 / ne11;
                            cl_kernel rq = backend_ctx->mul_mat_id.kernel_moe_reorder_quant_a_q8_1;
                            CL_CHECK(clSetKernelArg(rq, 0, sizeof(cl_mem),         &sub_buf_src1_pre));
                            CL_CHECK(clSetKernelArg(rq, 1, sizeof(cl_mem),         &buf_src2));
                            CL_CHECK(clSetKernelArg(rq, 2, sizeof(cl_mem),         &backend_ctx->prealloc_moe_qa.buffer));
                            CL_CHECK(clSetKernelArg(rq, 3, sizeof(cl_mem),         &backend_ctx->prealloc_moe_da.buffer));
                            CL_CHECK(clSetKernelArg(rq, 4, sizeof(cl_mem),         &backend_ctx->prealloc_moe_sa.buffer));
                            CL_CHECK(clSetKernelArg(rq, 5, sizeof(cl_mem),         &(backend_ctx->prealloc_total_tiles.buffer)));
                            CL_CHECK(clSetKernelArg(rq, 6, sizeof(cl_uint),        &ne00));
                            CL_CHECK(clSetKernelArg(rq, 7, sizeof(unsigned short), &map_ratio_q5));
                            CL_CHECK(clSetKernelArg(rq, 8, sizeof(cl_uint),        &n_tile_size));
                            CL_CHECK(clSetKernelArg(rq, 9, sizeof(cl_uint),        &n_kblocks));
                            size_t rq_local[2]  = { 32, 1 };
                            size_t rq_global[2] = { (size_t)(((n_kblocks + 31) / 32) * 32), tok_slots };
                            backend_ctx->enqueue_ndrange_kernel(rq, 2, rq_global, rq_local, dst);

                            region.origin = offsetd;
                            region.size = ne0 * ne1 * ne2 * sizeof(float);
                            cl_mem dp_sub_buf_dst = clCreateSubBuffer(extrad->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                            CL_CHECK(status);
                            cl_image_format dp_ifd = {CL_R, CL_FLOAT};
                            cl_image_desc dp_idd = {CL_MEM_OBJECT_IMAGE1D_BUFFER, static_cast<size_t>(ne0 * ne1 * ne2), 0,0,0,0,0,0,0, {dp_sub_buf_dst}};
                            cl_mem dp_buf_dst_image = clCreateImage(backend_ctx->context, CL_MEM_WRITE_ONLY, &dp_ifd, &dp_idd, NULL, &status);
                            CL_CHECK(status);

                            int ne00i = (int)ne00, ne01i = (int)ne01;
                            cl_kernel dk = backend_ctx->mul_mat_id.kernel_gemm_moe_q8_1_dp4a_q50;
                            int has_min_q5 = 1;
                            int aidx = 0;
                            CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &extra0_q5_0->qs_img));
                            CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &extra0_q5_0->qh));
                            CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &extra0_q5_0->scale));
                            CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &extra0_q5_0->min));
                            CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &backend_ctx->prealloc_moe_qa.buffer));
                            CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &backend_ctx->prealloc_moe_da.buffer));
                            CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &backend_ctx->prealloc_moe_sa.buffer));
                            CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &buf_src2));
                            CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &buf_src2_emap));
                            CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &dp_buf_dst_image));
                            CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &(backend_ctx->prealloc_total_tiles.buffer)));
                            CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(int),    &ne00i));
                            CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(int),    &ne01i));
                            CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(int),    &backend_ctx->adreno_use_moe_ragged_dp4));
                            CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(int),    &has_min_q5));

                            size_t dp_global[3] = { 64, (size_t)((ne01 + 63) / 64), (size_t)max_post_router_tile };
                            size_t dp_local[3]  = { 64, 1, 1 };
                            backend_ctx->enqueue_ndrange_kernel(dk, 3, dp_global, dp_local, dst);

                            clReleaseMemObject(sub_buf_src1_pre);
                            clReleaseMemObject(buf_src2);
                            clReleaseMemObject(buf_src2_emap);
                            clReleaseMemObject(dp_sub_buf_dst);
                            clReleaseMemObject(dp_buf_dst_image);
                            return;
                        }
                    }

                    // Create image for reordered src1
                    // Use pre-allocated placeholder
                    region.origin = 0;
                    region.size = ne00 * max_post_router_tile * n_tile_size * sizeof(float);
                    backend_ctx->prealloc_act_trans.allocate(backend_ctx->context, region.size);
                    buf_src1_reordered = clCreateSubBuffer(
                        backend_ctx->prealloc_act_trans.buffer,
                        0,
                        CL_BUFFER_CREATE_TYPE_REGION,
                        &region,
                        &status);
                    CL_CHECK(status);
                    cl_image_format image_format_buf_src1;
                    cl_image_desc image_desc_buf_src1;
                    image_format_buf_src1 = {CL_RGBA, CL_FLOAT};
                    image_desc_buf_src1 = {CL_MEM_OBJECT_IMAGE1D_BUFFER, static_cast<size_t>(ne00 * max_post_router_tile * n_tile_size / 4), 0,0,0,0,0,0,0, {buf_src1_reordered}};
                    image_src1_reordered = clCreateImage(backend_ctx->context, CL_MEM_READ_ONLY, &image_format_buf_src1, &image_desc_buf_src1, NULL, &status);
                    CL_CHECK(status);

                    unsigned short map_ratio = ne20 / ne11;
                    GGML_ASSERT(((map_ratio == 1) || (map_ratio == ne20)) && "Map ratio not supported\n");
                    CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 0, sizeof(cl_mem),        &sub_buf_src1_pre));
                    CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 1, sizeof(cl_mem),        &buf_src2));
                    CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 2, sizeof(cl_mem),        &buf_src1_reordered));
                    CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 3, sizeof(cl_mem),        &(backend_ctx->prealloc_total_tiles.buffer)));
                    CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 4, sizeof(unsigned int),  &ne00));
                    CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 5, sizeof(unsigned short),  &map_ratio));
                    CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 6, sizeof(unsigned int),  &n_tile_size));

                    size_t reorder_b_local_size[3] = {256, 1, 1};
                    size_t reorder_b_global_size[3] = {static_cast<size_t>(((ne00 / 4) + 255) / 256 * 256), static_cast<size_t>(max_post_router_tile * n_tile_size), 1};

                    // Dispatch reorder kernel
                    backend_ctx->enqueue_ndrange_kernel(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 3, reorder_b_global_size, reorder_b_local_size, dst);

                    // MoE kernel prepare
                    // Create sub buffer for dst
                    region.origin = offsetd;
                    region.size = ne0 * ne1 * ne2 * sizeof(float);
                    sub_buf_dst = clCreateSubBuffer(
                        extrad->data_device,
                        0,
                        CL_BUFFER_CREATE_TYPE_REGION,
                        &region,
                        &status);
                    CL_CHECK(status);
                    // Create image for dst
                    cl_image_format image_format_buf_dst = {CL_R, CL_FLOAT};
                    cl_image_desc image_desc_buf_dst = {CL_MEM_OBJECT_IMAGE1D_BUFFER, static_cast<size_t>(ne0 * ne1 * ne2), 0,0,0,0,0,0,0, {sub_buf_dst}};
                    buf_dst_image = clCreateImage(backend_ctx->context, CL_MEM_WRITE_ONLY, &image_format_buf_dst, &image_desc_buf_dst, NULL, &status);
                    CL_CHECK(status);

                    // Set kernel args
                    int arg_idx = 0;
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_q5_0->qs_img));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_q5_0->qh));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_q5_0->d));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &image_src1_reordered));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &buf_src2));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &buf_src2_emap));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &buf_dst_image));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &(backend_ctx->prealloc_total_tiles.buffer)));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),       &ne00));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),       &ne01));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_uint),   &backend_ctx->adreno_use_moe_ragged));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_uint),   &backend_ctx->adreno_moe_ragged_skip_gran));

                    // set thread grid
                    global_size[1] = static_cast<size_t>((ne01 + 63) / 64);
                    global_size[2] = static_cast<size_t>(max_post_router_tile);
                    local_size[1] = 1;
                    local_size[2] = 1;

                    // Dispatch kernel
                    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_size, local_size, dst);

                    clReleaseMemObject(sub_buf_src1_pre);
                    clReleaseMemObject(buf_src1_reordered);
                    clReleaseMemObject(image_src1_reordered);
                    clReleaseMemObject(buf_src2);
                    clReleaseMemObject(buf_src2_emap);
                    clReleaseMemObject(sub_buf_dst);
                    clReleaseMemObject(buf_dst_image);
                }
                return;
            }
#endif //GGML_OPENCL_USE_ADRENO_KERNELS
        }
        case GGML_TYPE_Q5_1: {
#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
            if (use_adreno_moe_kernels(backend_ctx, src0)) {
                cl_int status;

                size_t local_size[3] = {64, 2, 1};
                size_t global_size[3] = {64, 2, 1};

                if (ne12 == 1) { // for gemv
                    kernel = backend_ctx->mul_mat_id.kernel_gemv_moe_q5_1_f32_ns;

                    cl_mem src1_sub_buffer, buf_src1_image, buf_src2;

                    // create a sub_buffer for src2
                    cl_buffer_region region;
                    region.origin = offset2;
                    region.size = ne20 * ne21 * sizeof(int);
                    buf_src2 = clCreateSubBuffer(extra2->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                    CL_CHECK(status);

                    // set thread grid
                    global_size[0] = static_cast<size_t>(((ne01 + 63) / 64) * 64);
                    global_size[1] = 4;
                    global_size[2] = static_cast<size_t>(ne20);
                    local_size[1] = 4;

                    // create a sub_buffer for src1
                    region.origin = offset1;
                    region.size = ne10 * ne11 * ne12 * sizeof(float);
                    src1_sub_buffer = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                    CL_CHECK(status);

                    // create image for src1
                    cl_image_format image_format_buf_src1 = {CL_RGBA, CL_FLOAT};
                    cl_image_desc image_desc_buf_src1 = {CL_MEM_OBJECT_IMAGE1D_BUFFER, static_cast<size_t>(ne10 * ne11 * ne12 / 4), 0,0,0,0,0,0,0, {src1_sub_buffer}};
                    buf_src1_image = clCreateImage(backend_ctx->context, CL_MEM_READ_ONLY, &image_format_buf_src1, &image_desc_buf_src1, NULL, &status);
                    CL_CHECK(status);

                    // Set kernel args
                    int arg_idx = 0;
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_q5_1->qs));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_q5_1->qh));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_q5_1->d));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_q5_1->m));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &buf_src1_image));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &buf_src2));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extrad->data_device));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_ulong),  &offsetd));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),       &ne00));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),       &ne01));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),       &ne11));

                    // launch kernel
                    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_size, local_size, dst);

                    // deallocate sub buffers and images
                    CL_CHECK(clReleaseMemObject(src1_sub_buffer));
                    CL_CHECK(clReleaseMemObject(buf_src1_image));
                    CL_CHECK(clReleaseMemObject(buf_src2));
                } else { // for gemm
                    kernel = backend_ctx->mul_mat_id.kernel_gemm_moe_q5_1_f32_ns;

                    // Reorder router if called from test-backend-ops or when new router is generated.
                    // Otherwise reuse the reordered result from previous mul_mat_id call.
                    if ((strstr(src0->name, "as") != NULL) || backend_ctx->toggle_reorder) {
                        moe_router_reoerder(backend, src2, ne20);
                        backend_ctx->toggle_reorder = false;
                    }

                    cl_mem sub_buf_src1_pre, buf_src1_reordered, image_src1_reordered, sub_buf_dst, buf_dst_image;
                    cl_mem buf_src2, buf_src2_emap;

                    cl_buffer_region region;
                    region.origin = 0;
                    region.size = sizeof(int) * max_post_router_tile * n_tile_size;
                    buf_src2 = clCreateSubBuffer(backend_ctx->prealloc_post_router.buffer, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                    CL_CHECK(status);

                    region.origin = 0;
                    region.size = sizeof(short) * max_post_router_tile;
                    buf_src2_emap = clCreateSubBuffer(backend_ctx->prealloc_emap.buffer, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                    CL_CHECK(status);

                    // Reorder activations
                    // create a sub_buffer for src1
                    region.origin = offset1;
                    region.size = ne10 * ne11 * ne12 * sizeof(float);
                    sub_buf_src1_pre = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                    CL_CHECK(status);

                    // Create image for reordered src1
                    // Use pre-allocated placeholder
                    region.origin = 0;
                    region.size = ne00 * max_post_router_tile * n_tile_size * sizeof(float);
                    backend_ctx->prealloc_act_trans.allocate(backend_ctx->context, region.size);
                    buf_src1_reordered = clCreateSubBuffer(
                        backend_ctx->prealloc_act_trans.buffer,
                        0,
                        CL_BUFFER_CREATE_TYPE_REGION,
                        &region,
                        &status);
                    CL_CHECK(status);
                    cl_image_format image_format_buf_src1;
                    cl_image_desc image_desc_buf_src1;
                    image_format_buf_src1 = {CL_RGBA, CL_FLOAT};
                    image_desc_buf_src1 = {CL_MEM_OBJECT_IMAGE1D_BUFFER, static_cast<size_t>(ne00 * max_post_router_tile * n_tile_size / 4), 0,0,0,0,0,0,0, {buf_src1_reordered}};
                    image_src1_reordered = clCreateImage(backend_ctx->context, CL_MEM_READ_ONLY, &image_format_buf_src1, &image_desc_buf_src1, NULL, &status);
                    CL_CHECK(status);

                    unsigned short map_ratio = ne20 / ne11;
                    GGML_ASSERT(((map_ratio == 1) || (map_ratio == ne20)) && "Map ratio not supported\n");
                    CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 0, sizeof(cl_mem),        &sub_buf_src1_pre));
                    CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 1, sizeof(cl_mem),        &buf_src2));
                    CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 2, sizeof(cl_mem),        &buf_src1_reordered));
                    CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 3, sizeof(cl_mem),        &(backend_ctx->prealloc_total_tiles.buffer)));
                    CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 4, sizeof(unsigned int),  &ne00));
                    CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 5, sizeof(unsigned short),  &map_ratio));
                    CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 6, sizeof(unsigned int),  &n_tile_size));

                    size_t reorder_b_local_size[3] = {256, 1, 1};
                    size_t reorder_b_global_size[3] = {static_cast<size_t>(((ne00 / 4) + 255) / 256 * 256), static_cast<size_t>(max_post_router_tile * n_tile_size), 1};

                    // Dispatch reorder kernel
                    backend_ctx->enqueue_ndrange_kernel(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 3, reorder_b_global_size, reorder_b_local_size, dst);

                    // MoE kernel prepare
                    // Create sub buffer for dst
                    region.origin = offsetd;
                    region.size = ne0 * ne1 * ne2 * sizeof(float);
                    sub_buf_dst = clCreateSubBuffer(
                        extrad->data_device,
                        0,
                        CL_BUFFER_CREATE_TYPE_REGION,
                        &region,
                        &status);
                    CL_CHECK(status);
                    // Create image for dst
                    cl_image_format image_format_buf_dst = {CL_R, CL_FLOAT};
                    cl_image_desc image_desc_buf_dst = {CL_MEM_OBJECT_IMAGE1D_BUFFER, static_cast<size_t>(ne0 * ne1 * ne2), 0,0,0,0,0,0,0, {sub_buf_dst}};
                    buf_dst_image = clCreateImage(backend_ctx->context, CL_MEM_WRITE_ONLY, &image_format_buf_dst, &image_desc_buf_dst, NULL, &status);
                    CL_CHECK(status);

                    // Set kernel args
                    int arg_idx = 0;
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_q5_1->qs_img));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_q5_1->qh));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_q5_1->d));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_q5_1->m));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &image_src1_reordered));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &buf_src2));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &buf_src2_emap));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &buf_dst_image));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &(backend_ctx->prealloc_total_tiles.buffer)));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),       &ne00));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),       &ne01));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_uint),       &backend_ctx->adreno_use_moe_ragged));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_uint),       &backend_ctx->adreno_moe_ragged_skip_gran));

                    // set thread grid
                    global_size[1] = static_cast<size_t>((ne01 + 63) / 64);
                    global_size[2] = static_cast<size_t>(max_post_router_tile);
                    local_size[1] = 1;
                    local_size[2] = 1;

                    // Dispatch kernel
                    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_size, local_size, dst);

                    clReleaseMemObject(sub_buf_src1_pre);
                    clReleaseMemObject(buf_src1_reordered);
                    clReleaseMemObject(image_src1_reordered);
                    clReleaseMemObject(buf_src2);
                    clReleaseMemObject(buf_src2_emap);
                    clReleaseMemObject(sub_buf_dst);
                    clReleaseMemObject(buf_dst_image);
                }
                return;
            }
#endif //GGML_OPENCL_USE_ADRENO_KERNELS
        }
        case GGML_TYPE_Q8_0: {
#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
            // MoE GEMM for q8_0 at prefill (ne12>1)
            // There is no corresponding gemv_moe, so the code path is different here
            static const char * moe_gemm_q8_env = getenv("GGML_OPENCL_MOE_GEMM_Q8");
            const bool          moe_gemm_q8     = moe_gemm_q8_env
                ? (atoi(moe_gemm_q8_env) != 0)
                : (backend_ctx->adreno_gen == ADRENO_GPU_GEN::X2E);
            if (moe_gemm_q8 && use_adreno_moe_kernels(backend_ctx, src0) && ne12 > 1) {
                cl_int status;

                size_t local_size[3]  = {64, 2, 1};
                size_t global_size[3] = {64, 2, 1};

                kernel = backend_ctx->mul_mat_id.kernel_gemm_moe_q8_0_f32_ns;

                if ((strstr(src0->name, "as") != NULL) || backend_ctx->toggle_reorder) {
                    moe_router_reoerder(backend, src2, ne20);
                    backend_ctx->toggle_reorder = false;
                }

                cl_mem sub_buf_src1_pre, buf_src1_reordered, image_src1_reordered, sub_buf_dst, buf_dst_image;
                cl_mem buf_src2, buf_src2_emap;

                cl_buffer_region region;
                region.origin = 0;
                region.size = sizeof(int) * max_post_router_tile * n_tile_size;
                buf_src2 = clCreateSubBuffer(backend_ctx->prealloc_post_router.buffer, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                CL_CHECK(status);

                region.origin = 0;
                region.size = sizeof(short) * max_post_router_tile;
                buf_src2_emap = clCreateSubBuffer(backend_ctx->prealloc_emap.buffer, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                CL_CHECK(status);

                // Reorder activations (group tokens by expert into tiles of 32)
                region.origin = offset1;
                region.size = ne10 * ne11 * ne12 * sizeof(float);
                sub_buf_src1_pre = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                CL_CHECK(status);

                // Generic dp4a MoE GEMM
                {
                    static const char * q8mdp4a_env = getenv("GGML_OPENCL_Q8_MOE_DP4A");
                    const bool q8mdp4a_on = q8mdp4a_env ? (atoi(q8mdp4a_env) != 0)
                                                        : (backend_ctx->adreno_gen == ADRENO_GPU_GEN::X2E);
                    const bool use_q8_moe_dp4a = q8mdp4a_on
                        && backend_ctx->mul_mat_id.kernel_gemm_moe_q8_1_dp4a_q80 != nullptr
                        && extra0_q8_0->scale != nullptr;
                    if (use_q8_moe_dp4a) {
                        const size_t tok_slots = (size_t)max_post_router_tile * n_tile_size;
                        const size_t n_blocks  = tok_slots * (ne00 / 32);
                        backend_ctx->prealloc_moe_qa.allocate(backend_ctx->context, tok_slots * ne00 * sizeof(cl_char));
                        backend_ctx->prealloc_moe_da.allocate(backend_ctx->context, n_blocks * sizeof(cl_half));
                        backend_ctx->prealloc_moe_sa.allocate(backend_ctx->context, n_blocks * sizeof(cl_half));

                        const cl_uint n_kblocks = (cl_uint)(ne00 / 32);
                        unsigned short map_ratio_q8 = ne20 / ne11;
                        cl_kernel rq = backend_ctx->mul_mat_id.kernel_moe_reorder_quant_a_q8_1;
                        CL_CHECK(clSetKernelArg(rq, 0, sizeof(cl_mem),         &sub_buf_src1_pre));
                        CL_CHECK(clSetKernelArg(rq, 1, sizeof(cl_mem),         &buf_src2));
                        CL_CHECK(clSetKernelArg(rq, 2, sizeof(cl_mem),         &backend_ctx->prealloc_moe_qa.buffer));
                        CL_CHECK(clSetKernelArg(rq, 3, sizeof(cl_mem),         &backend_ctx->prealloc_moe_da.buffer));
                        CL_CHECK(clSetKernelArg(rq, 4, sizeof(cl_mem),         &backend_ctx->prealloc_moe_sa.buffer));
                        CL_CHECK(clSetKernelArg(rq, 5, sizeof(cl_mem),         &(backend_ctx->prealloc_total_tiles.buffer)));
                        CL_CHECK(clSetKernelArg(rq, 6, sizeof(cl_uint),        &ne00));
                        CL_CHECK(clSetKernelArg(rq, 7, sizeof(unsigned short), &map_ratio_q8));
                        CL_CHECK(clSetKernelArg(rq, 8, sizeof(cl_uint),        &n_tile_size));
                        CL_CHECK(clSetKernelArg(rq, 9, sizeof(cl_uint),        &n_kblocks));
                        size_t rq_local[2]  = { 32, 1 };
                        size_t rq_global[2] = { (size_t)(((n_kblocks + 31) / 32) * 32), tok_slots };
                        backend_ctx->enqueue_ndrange_kernel(rq, 2, rq_global, rq_local, dst);

                        // dst image
                        region.origin = offsetd;
                        region.size = ne0 * ne1 * ne2 * sizeof(float);
                        cl_mem dp_sub_buf_dst = clCreateSubBuffer(extrad->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                        CL_CHECK(status);
                        cl_image_format dp_ifd = {CL_R, CL_FLOAT};
                        cl_image_desc dp_idd = {CL_MEM_OBJECT_IMAGE1D_BUFFER, static_cast<size_t>(ne0 * ne1 * ne2), 0,0,0,0,0,0,0, {dp_sub_buf_dst}};
                        cl_mem dp_buf_dst_image = clCreateImage(backend_ctx->context, CL_MEM_WRITE_ONLY, &dp_ifd, &dp_idd, NULL, &status);
                        CL_CHECK(status);

                        int ne00i = (int)ne00, ne01i = (int)ne01;
                        cl_kernel dk = backend_ctx->mul_mat_id.kernel_gemm_moe_q8_1_dp4a_q80;
                        int has_min_q8 = 0;
                        int aidx = 0;
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &extra0_q8_0->q));      // flat int8 codes [expert][row][K]
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &extra0_q8_0->scale));  // uniform scale[16]
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &extra0_q8_0->scale));  // dummy min (has_min=0, unread)
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &backend_ctx->prealloc_moe_qa.buffer));
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &backend_ctx->prealloc_moe_da.buffer));
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &backend_ctx->prealloc_moe_sa.buffer));
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &buf_src2));
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &buf_src2_emap));
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &dp_buf_dst_image));
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &(backend_ctx->prealloc_total_tiles.buffer)));
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(int),    &ne00i));
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(int),    &ne01i));
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(int),    &backend_ctx->adreno_use_moe_ragged_dp4));
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(int),    &has_min_q8));

                        size_t dp_global[3] = { 64, (size_t)((ne01 + 63) / 64), (size_t)max_post_router_tile };
                        size_t dp_local[3]  = { 64, 1, 1 };
                        backend_ctx->enqueue_ndrange_kernel(dk, 3, dp_global, dp_local, dst);

                        clReleaseMemObject(sub_buf_src1_pre);
                        clReleaseMemObject(buf_src2);
                        clReleaseMemObject(buf_src2_emap);
                        clReleaseMemObject(dp_sub_buf_dst);
                        clReleaseMemObject(dp_buf_dst_image);
                        return;
                    }
                }

                region.origin = 0;
                region.size = ne00 * max_post_router_tile * n_tile_size * sizeof(float);
                backend_ctx->prealloc_act_trans.allocate(backend_ctx->context, region.size);
                buf_src1_reordered = clCreateSubBuffer(
                    backend_ctx->prealloc_act_trans.buffer, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                CL_CHECK(status);
                cl_image_format image_format_buf_src1 = {CL_RGBA, CL_FLOAT};
                cl_image_desc image_desc_buf_src1 = {CL_MEM_OBJECT_IMAGE1D_BUFFER, static_cast<size_t>(ne00 * max_post_router_tile * n_tile_size / 4), 0,0,0,0,0,0,0, {buf_src1_reordered}};
                image_src1_reordered = clCreateImage(backend_ctx->context, CL_MEM_READ_ONLY, &image_format_buf_src1, &image_desc_buf_src1, NULL, &status);
                CL_CHECK(status);

                unsigned short map_ratio = ne20 / ne11;
                GGML_ASSERT(((map_ratio == 1) || (map_ratio == ne20)) && "Map ratio not supported\n");
                CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 0, sizeof(cl_mem),         &sub_buf_src1_pre));
                CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 1, sizeof(cl_mem),         &buf_src2));
                CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 2, sizeof(cl_mem),         &buf_src1_reordered));
                CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 3, sizeof(cl_mem),         &(backend_ctx->prealloc_total_tiles.buffer)));
                CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 4, sizeof(unsigned int),   &ne00));
                CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 5, sizeof(unsigned short), &map_ratio));
                CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 6, sizeof(unsigned int),   &n_tile_size));

                size_t reorder_b_local_size[3]  = {256, 1, 1};
                size_t reorder_b_global_size[3] = {static_cast<size_t>(((ne00 / 4) + 255) / 256 * 256), static_cast<size_t>(max_post_router_tile * n_tile_size), 1};
                backend_ctx->enqueue_ndrange_kernel(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 3, reorder_b_global_size, reorder_b_local_size, dst);

                // dst image
                region.origin = offsetd;
                region.size = ne0 * ne1 * ne2 * sizeof(float);
                sub_buf_dst = clCreateSubBuffer(extrad->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                CL_CHECK(status);
                cl_image_format image_format_buf_dst = {CL_R, CL_FLOAT};
                cl_image_desc image_desc_buf_dst = {CL_MEM_OBJECT_IMAGE1D_BUFFER, static_cast<size_t>(ne0 * ne1 * ne2), 0,0,0,0,0,0,0, {sub_buf_dst}};
                buf_dst_image = clCreateImage(backend_ctx->context, CL_MEM_WRITE_ONLY, &image_format_buf_dst, &image_desc_buf_dst, NULL, &status);
                CL_CHECK(status);

                int arg_idx = 0;
                CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &extra0_q8_0->q));   // flat q8_0 quants
                CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &extra0_q8_0->d));   // flat q8_0 scales
                CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &image_src1_reordered));
                CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &buf_src2));
                CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &buf_src2_emap));
                CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &buf_dst_image));
                CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &(backend_ctx->prealloc_total_tiles.buffer)));
                CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),    &ne00));
                CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),    &ne01));

                global_size[1] = static_cast<size_t>((ne01 + 63) / 64);
                global_size[2] = static_cast<size_t>(max_post_router_tile);
                local_size[1]  = 1;
                local_size[2]  = 1;

                backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_size, local_size, dst);

                clReleaseMemObject(sub_buf_src1_pre);
                clReleaseMemObject(buf_src1_reordered);
                clReleaseMemObject(image_src1_reordered);
                clReleaseMemObject(buf_src2);
                clReleaseMemObject(buf_src2_emap);
                clReleaseMemObject(sub_buf_dst);
                clReleaseMemObject(buf_dst_image);
                return;
            }
#endif // GGML_OPENCL_USE_ADRENO_KERNELS
#ifdef GGML_OPENCL_SOA_Q
            kernel = backend_ctx->mul_mat_id.kernel_mul_mv_id_q8_0_f32_flat;

            if (backend_ctx->gpu_family == INTEL) {
                sgs  = 16;
                nsg  = 2;
                ndst = 4;
            } else if (backend_ctx->gpu_family == ADRENO) {
                sgs  = 64;
                nsg  = 2;
                ndst = 4;
            } else {
                GGML_ASSERT(false && "TODO: Unknown GPU");
            }

            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0_q8_0->q));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0_q8_0->d));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extra2->data_device));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offset2));
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne01));
            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb01));
            CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb02));
            CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne11));
            CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne12));
            CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &nb11));
            CL_CHECK(clSetKernelArg(kernel, 15, sizeof(cl_ulong), &nb12));
            CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &ne20));
            CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &ne21));
            CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong), &nb21));
            CL_CHECK(clSetKernelArg(kernel, 19, sizeof(int),      &ne0));
            CL_CHECK(clSetKernelArg(kernel, 20, sizeof(int),      &ne1));
#else
            kernel = backend_ctx->mul_mat_id.kernel_mul_mv_id_q8_0_f32;

            if (backend_ctx->gpu_family == INTEL) {
                sgs  = 16;
                nsg  = 2;
                ndst = 4;
            } else if (backend_ctx->gpu_family == ADRENO) {
                sgs  = 64;
                nsg  = 2;
                ndst = 4;
            } else {
                GGML_ASSERT(false && "TODO: Unknown GPU");
            }

            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extra2->data_device));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offset2));
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne01));
            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb01));
            CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb02));
            CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne11));
            CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne12));
            CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &nb11));
            CL_CHECK(clSetKernelArg(kernel, 15, sizeof(cl_ulong), &nb12));
            CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &ne20));
            CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &ne21));
            CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong), &nb21));
            CL_CHECK(clSetKernelArg(kernel, 19, sizeof(int),      &ne0));
            CL_CHECK(clSetKernelArg(kernel, 20, sizeof(int),      &ne1));
#endif // GGML_OPENCL_SOA_Q
            break;
        }
        case GGML_TYPE_Q4_K: {
#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
            if (use_adreno_moe_kernels(backend_ctx, src0)) {
                cl_int status;

                size_t local_size[3] = {64, 2, 1};
                size_t global_size[3] = {64, 2, 1};

                if (ne12 == 1) { // for gemv
                    kernel = backend_ctx->mul_mat_id.kernel_gemv_moe_q4_k_f32_ns;

                    // Weight-as-texture MoE decode GEMV
                    static const char * moe_decode_wimg_env = getenv("GGML_OPENCL_MOE_DECODE_WIMG");
                    const bool moe_decode_wimg_on = moe_decode_wimg_env
                        ? (atoi(moe_decode_wimg_env) != 0)
                        : (backend_ctx->adreno_gen == ADRENO_GPU_GEN::X2E);
                    const bool use_moe_decode_wimg = moe_decode_wimg_on
                        && backend_ctx->mul_mat_id.kernel_gemv_moe_q4_k_f32_ns_wimg != nullptr
                        && extra0_q4_K->q_img != nullptr;
                    if (use_moe_decode_wimg) {
                        kernel = backend_ctx->mul_mat_id.kernel_gemv_moe_q4_k_f32_ns_wimg;
                    }

                    cl_mem src1_sub_buffer, buf_src1_image, buf_src2;

                    // create a sub_buffer for src2
                    cl_buffer_region region;
                    region.origin = offset2;
                    region.size = ne20 * ne21 * sizeof(int);
                    buf_src2 = clCreateSubBuffer(extra2->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                    CL_CHECK(status);

                    // set thread grid
                    global_size[0] = static_cast<size_t>(((ne01 + 63) / 64) * 64);
                    global_size[1] = 4;
                    global_size[2] = static_cast<size_t>(ne20);
                    local_size[1] = 4;

                    // create a sub_buffer for src1
                    region.origin = offset1;
                    region.size = ne10 * ne11 * ne12 * sizeof(float);
                    src1_sub_buffer = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                    CL_CHECK(status);

                    // create image for src1
                    cl_image_format image_format_buf_src1 = {CL_RGBA, CL_FLOAT};
                    cl_image_desc image_desc_buf_src1 = {CL_MEM_OBJECT_IMAGE1D_BUFFER, static_cast<size_t>(ne10 * ne11 * ne12 / 4), 0,0,0,0,0,0,0, {src1_sub_buffer}};
                    buf_src1_image = clCreateImage(backend_ctx->context, CL_MEM_READ_ONLY, &image_format_buf_src1, &image_desc_buf_src1, NULL, &status);
                    CL_CHECK(status);

                    // Set kernel args
                    int arg_idx = 0;
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    use_moe_decode_wimg ? &extra0_q4_K->q_img : &extra0_q4_K->q));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_q4_K->d));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_q4_K->dm));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_q4_K->s));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &buf_src1_image));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &buf_src2));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extrad->data_device));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_ulong),  &offsetd));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),       &ne00));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),       &ne01));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),       &ne11));

                    // launch kernel
                    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_size, local_size, dst);

                    // deallocate sub buffers and images
                    CL_CHECK(clReleaseMemObject(src1_sub_buffer));
                    CL_CHECK(clReleaseMemObject(buf_src1_image));
                    CL_CHECK(clReleaseMemObject(buf_src2));

                } else { // for gemm
                    kernel = backend_ctx->mul_mat_id.kernel_gemm_moe_q4_k_f32_ns;
                    if (backend_ctx->mul_mat_id.kernel_gemm_moe_q4_k_f32_ns_bin) {
                        kernel = backend_ctx->mul_mat_id.kernel_gemm_moe_q4_k_f32_ns_bin;
                    }

                    // Reorder router if called from test-backend-ops or when new router is generated.
                    // Otherwise reuse the reordered result from previous mul_mat_id call.
                    if ((strstr(src0->name, "as") != NULL) || backend_ctx->toggle_reorder) {
                        moe_router_reoerder(backend, src2, ne20);
                        backend_ctx->toggle_reorder = false;
                    }

                    cl_mem sub_buf_src1_pre, sub_buf_dst, buf_dst_image;
                    cl_mem buf_src1_reordered = nullptr, image_src1_reordered = nullptr;
                    cl_mem buf_src2, buf_src2_emap;

                    // dp4a (int8) prefill GEMM variant
                    static const char * q4k_moe_dp4a_env = getenv("GGML_OPENCL_Q4K_MOE_DP4A");
                    bool  use_moe_dp4a = (q4k_moe_dp4a_env != nullptr)
                                         ? (atoi(q4k_moe_dp4a_env) != 0)
                                         : (backend_ctx->adreno_gen == ADRENO_GPU_GEN::X2E || backend_ctx->adreno_gen == ADRENO_GPU_GEN::X1E);
                    // dot prod has to be available
                    use_moe_dp4a = backend_ctx->has_integer_dot && use_moe_dp4a;
                    // bin kernel takes precedence
                    use_moe_dp4a = use_moe_dp4a && backend_ctx->mul_mat_id.kernel_gemm_moe_q4_k_f32_ns_bin == nullptr;

                    cl_buffer_region region;
                    region.origin = 0;
                    region.size = sizeof(int) * max_post_router_tile * n_tile_size;
                    buf_src2 = clCreateSubBuffer(backend_ctx->prealloc_post_router.buffer, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                    CL_CHECK(status);

                    region.origin = 0;
                    region.size = sizeof(short) * max_post_router_tile;
                    buf_src2_emap = clCreateSubBuffer(backend_ctx->prealloc_emap.buffer, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                    CL_CHECK(status);

                    // Reorder activations
                    region.origin = offset1;
                    region.size = ne10 * ne11 * ne12 * sizeof(float);
                    sub_buf_src1_pre = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                    CL_CHECK(status);

                    unsigned short map_ratio = ne20 / ne11;
                    GGML_ASSERT(((map_ratio == 1) || (map_ratio == ne20)) && "Map ratio not supported\n");

                    if (!use_moe_dp4a) {
                        // Create image for reordered src1
                        region.origin = 0;
                        region.size = ne00 * max_post_router_tile * n_tile_size * sizeof(float);
                        backend_ctx->prealloc_act_trans.allocate(backend_ctx->context, region.size);
                        buf_src1_reordered = clCreateSubBuffer(
                            backend_ctx->prealloc_act_trans.buffer,
                            0,
                            CL_BUFFER_CREATE_TYPE_REGION,
                            &region,
                            &status);
                        CL_CHECK(status);
                        cl_image_format image_format_buf_src1 = {CL_RGBA, CL_FLOAT};
                        cl_image_desc image_desc_buf_src1 = {CL_MEM_OBJECT_IMAGE1D_BUFFER, static_cast<size_t>(ne00 * max_post_router_tile * n_tile_size / 4), 0,0,0,0,0,0,0, {buf_src1_reordered}};
                        if (backend_ctx->mul_mat_id.kernel_gemm_moe_q4_k_f32_ns_bin) {
                            // bin kernel uses slightly different image format
                            image_format_buf_src1 = {CL_R, CL_FLOAT};
                            image_desc_buf_src1.image_width = static_cast<size_t>(ne00 * max_post_router_tile * n_tile_size);
                        }
                        image_src1_reordered = clCreateImage(backend_ctx->context, CL_MEM_READ_ONLY, &image_format_buf_src1, &image_desc_buf_src1, NULL, &status);
                        CL_CHECK(status);

                        CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 0, sizeof(cl_mem),        &sub_buf_src1_pre));
                        CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 1, sizeof(cl_mem),        &buf_src2));
                        CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 2, sizeof(cl_mem),        &buf_src1_reordered));
                        CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 3, sizeof(cl_mem),        &(backend_ctx->prealloc_total_tiles.buffer)));
                        CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 4, sizeof(unsigned int),  &ne00));
                        CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 5, sizeof(unsigned short),  &map_ratio));
                        CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 6, sizeof(unsigned int),  &n_tile_size));

                        size_t reorder_b_local_size[3] = {256, 1, 1};
                        size_t reorder_b_global_size[3] = {static_cast<size_t>(((ne00 / 4) + 255) / 256 * 256), static_cast<size_t>(max_post_router_tile * n_tile_size), 1};

                        // Dispatch reorder kernel
                        backend_ctx->enqueue_ndrange_kernel(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 3, reorder_b_global_size, reorder_b_local_size, dst);
                    }

                    // MoE kernel prepare
                    region.origin = offsetd;
                    region.size = ne0 * ne1 * ne2 * sizeof(float);
                    sub_buf_dst = clCreateSubBuffer(
                        extrad->data_device,
                        0,
                        CL_BUFFER_CREATE_TYPE_REGION,
                        &region,
                        &status);
                    CL_CHECK(status);
                    // Create image for dst
                    cl_image_format image_format_buf_dst = {CL_R, CL_FLOAT};
                    cl_image_desc image_desc_buf_dst = {CL_MEM_OBJECT_IMAGE1D_BUFFER, static_cast<size_t>(ne0 * ne1 * ne2), 0,0,0,0,0,0,0, {sub_buf_dst}};
                    buf_dst_image = clCreateImage(backend_ctx->context, CL_MEM_WRITE_ONLY, &image_format_buf_dst, &image_desc_buf_dst, NULL, &status);
                    CL_CHECK(status);

                    if (use_moe_dp4a) {
                        const size_t tok_slots = (size_t)max_post_router_tile * n_tile_size;
                        const size_t n_blocks  = tok_slots * (ne00 / 32);
                        backend_ctx->prealloc_moe_qa.allocate(backend_ctx->context, tok_slots * ne00 * sizeof(cl_char));
                        backend_ctx->prealloc_moe_da.allocate(backend_ctx->context, n_blocks * sizeof(cl_half));
                        backend_ctx->prealloc_moe_sa.allocate(backend_ctx->context, n_blocks * sizeof(cl_half));

                        // fused reorder + q8_1 quant straight from the original
                        // activations (no intermediate f32 reorder buffer)
                        const cl_uint n_kblocks = (cl_uint)(ne00 / 32);
                        cl_kernel rq = backend_ctx->mul_mat_id.kernel_moe_reorder_quant_a_q8_1;
                        CL_CHECK(clSetKernelArg(rq, 0, sizeof(cl_mem),         &sub_buf_src1_pre));
                        CL_CHECK(clSetKernelArg(rq, 1, sizeof(cl_mem),         &buf_src2));
                        CL_CHECK(clSetKernelArg(rq, 2, sizeof(cl_mem),         &backend_ctx->prealloc_moe_qa.buffer));
                        CL_CHECK(clSetKernelArg(rq, 3, sizeof(cl_mem),         &backend_ctx->prealloc_moe_da.buffer));
                        CL_CHECK(clSetKernelArg(rq, 4, sizeof(cl_mem),         &backend_ctx->prealloc_moe_sa.buffer));
                        CL_CHECK(clSetKernelArg(rq, 5, sizeof(cl_mem),         &(backend_ctx->prealloc_total_tiles.buffer)));
                        CL_CHECK(clSetKernelArg(rq, 6, sizeof(cl_uint),        &ne00));
                        CL_CHECK(clSetKernelArg(rq, 7, sizeof(unsigned short), &map_ratio));
                        CL_CHECK(clSetKernelArg(rq, 8, sizeof(cl_uint),        &n_tile_size));
                        CL_CHECK(clSetKernelArg(rq, 9, sizeof(cl_uint),        &n_kblocks));
                        size_t rq_local[2]  = { 32, 1 };
                        size_t rq_global[2] = { (size_t)(((n_kblocks + 31) / 32) * 32), tok_slots };
                        backend_ctx->enqueue_ndrange_kernel(rq, 2, rq_global, rq_local, dst);

                        // dp4a GEMM
                        cl_kernel dk = backend_ctx->mul_mat_id.kernel_gemm_moe_q4_k_q8_1_dp4a;
                        int aidx = 0;
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &extra0_q4_K->q_img));
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &extra0_q4_K->d));
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &extra0_q4_K->dm));
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &extra0_q4_K->s));
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &backend_ctx->prealloc_moe_qa.buffer));
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &backend_ctx->prealloc_moe_da.buffer));
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &backend_ctx->prealloc_moe_sa.buffer));
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &buf_src2));
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &buf_src2_emap));
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &buf_dst_image));
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &(backend_ctx->prealloc_total_tiles.buffer)));
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(int),    &ne00));
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(int),    &ne01));
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(int),    &backend_ctx->adreno_use_moe_ragged_dp4));

                        size_t dp_global[3] = { 64, (size_t)((ne01 + 63) / 64), (size_t)max_post_router_tile };
                        size_t dp_local[3]  = { 64, 1, 1 };
                        backend_ctx->enqueue_ndrange_kernel(dk, 3, dp_global, dp_local, dst);

                        clReleaseMemObject(sub_buf_src1_pre);
                        clReleaseMemObject(buf_src2);
                        clReleaseMemObject(buf_src2_emap);
                        clReleaseMemObject(sub_buf_dst);
                        clReleaseMemObject(buf_dst_image);
                        return;
                    }

                    // Set kernel args
                    int arg_idx = 0;
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_q4_K->q_img));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_q4_K->d));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_q4_K->dm));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_q4_K->s));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &image_src1_reordered));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &buf_src2));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &buf_src2_emap));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &buf_dst_image));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &(backend_ctx->prealloc_total_tiles.buffer)));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),       &ne00));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),       &ne01));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_uint),   &backend_ctx->adreno_use_moe_ragged));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_uint),   &backend_ctx->adreno_moe_ragged_skip_gran));

                    // set thread grid
                    global_size[1] = static_cast<size_t>((ne01 + 63) / 64);
                    global_size[2] = static_cast<size_t>(max_post_router_tile);
                    local_size[1] = 1;
                    local_size[2] = 1;

                    // Dispatch kernel
                    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_size, local_size, dst);

                    clReleaseMemObject(sub_buf_src1_pre);
                    clReleaseMemObject(buf_src1_reordered);
                    clReleaseMemObject(image_src1_reordered);
                    clReleaseMemObject(buf_src2);
                    clReleaseMemObject(buf_src2_emap);
                    clReleaseMemObject(sub_buf_dst);
                    clReleaseMemObject(buf_dst_image);
                }
                return;
            }
#endif //GGML_OPENCL_USE_ADRENO_KERNELS
        }
        case GGML_TYPE_Q5_K: {
#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
            if (use_adreno_moe_kernels(backend_ctx, src0)) {
                cl_int status;

                size_t local_size[3] = {64, 2, 1};
                size_t global_size[3] = {64, 2, 1};

                if (ne12 == 1) { // for gemv
                    kernel = backend_ctx->mul_mat_id.kernel_gemv_moe_q5_k_f32_ns;

                    cl_mem src1_sub_buffer, buf_src1_image, buf_src2;

                    // create a sub_buffer for src2
                    cl_buffer_region region;
                    region.origin = offset2;
                    region.size = ne20 * ne21 * sizeof(int);
                    buf_src2 = clCreateSubBuffer(extra2->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                    CL_CHECK(status);

                    // set thread grid
                    global_size[0] = static_cast<size_t>(((ne01 + 63) / 64) * 64);
                    global_size[1] = 4;
                    global_size[2] = static_cast<size_t>(ne20);
                    local_size[1] = 4;

                    // create a sub_buffer for src1
                    region.origin = offset1;
                    region.size = ne10 * ne11 * ne12 * sizeof(float);
                    src1_sub_buffer = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                    CL_CHECK(status);

                    // create image for src1
                    cl_image_format image_format_buf_src1 = {CL_RGBA, CL_FLOAT};
                    cl_image_desc image_desc_buf_src1 = {CL_MEM_OBJECT_IMAGE1D_BUFFER, static_cast<size_t>(ne10 * ne11 * ne12 / 4), 0,0,0,0,0,0,0, {src1_sub_buffer}};
                    buf_src1_image = clCreateImage(backend_ctx->context, CL_MEM_READ_ONLY, &image_format_buf_src1, &image_desc_buf_src1, NULL, &status);
                    CL_CHECK(status);

                    // Set kernel args
                    int arg_idx = 0;
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_q5_K->q));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_q5_K->qh));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_q5_K->d));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_q5_K->dm));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_q5_K->s));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &buf_src1_image));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &buf_src2));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extrad->data_device));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_ulong),  &offsetd));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),       &ne00));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),       &ne01));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),       &ne11));

                    // launch kernel
                    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_size, local_size, dst);

                    // deallocate sub buffers and images
                    CL_CHECK(clReleaseMemObject(src1_sub_buffer));
                    CL_CHECK(clReleaseMemObject(buf_src1_image));
                    CL_CHECK(clReleaseMemObject(buf_src2));

                } else { // for gemm
                    kernel = backend_ctx->mul_mat_id.kernel_gemm_moe_q5_k_f32_ns;

                    // Reorder router if called from test-backend-ops or when new router is generated.
                    // Otherwise reuse the reordered result from previous mul_mat_id call.
                    if ((strstr(src0->name, "as") != NULL) || backend_ctx->toggle_reorder) {
                        moe_router_reoerder(backend, src2, ne20);
                        backend_ctx->toggle_reorder = false;
                    }

                    cl_mem sub_buf_src1_pre, buf_src1_reordered, image_src1_reordered, sub_buf_dst, buf_dst_image;
                    cl_mem buf_src2, buf_src2_emap;

                    cl_buffer_region region;
                    region.origin = 0;
                    region.size = sizeof(int) * max_post_router_tile * n_tile_size;
                    buf_src2 = clCreateSubBuffer(backend_ctx->prealloc_post_router.buffer, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                    CL_CHECK(status);

                    region.origin = 0;
                    region.size = sizeof(short) * max_post_router_tile;
                    buf_src2_emap = clCreateSubBuffer(backend_ctx->prealloc_emap.buffer, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                    CL_CHECK(status);

                    // Reorder activations
                    // create a sub_buffer for src1
                    region.origin = offset1;
                    region.size = ne10 * ne11 * ne12 * sizeof(float);
                    sub_buf_src1_pre = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                    CL_CHECK(status);

                    // Generic dp4a MoE GEMM
                    {
                        static const char * q5kmdp4a_env = getenv("GGML_OPENCL_Q5K_MOE_DP4A");
                        const bool q5kmdp4a_on = q5kmdp4a_env ? (atoi(q5kmdp4a_env) != 0)
                                                              : (backend_ctx->adreno_gen == ADRENO_GPU_GEN::X2E);
                        bool use_moe_dp4a = q5kmdp4a_on
                            && backend_ctx->mul_mat_id.kernel_gemm_moe_q8_1_dp4a_q5k != nullptr
                            && extra0_q5_K->scale != nullptr;
                        // dot prod has to be available
                        use_moe_dp4a = backend_ctx->has_integer_dot && use_moe_dp4a;

                        if (use_moe_dp4a) {
                            const size_t tok_slots = (size_t)max_post_router_tile * n_tile_size;
                            const size_t n_blocks  = tok_slots * (ne00 / 32);
                            backend_ctx->prealloc_moe_qa.allocate(backend_ctx->context, tok_slots * ne00 * sizeof(cl_char));
                            backend_ctx->prealloc_moe_da.allocate(backend_ctx->context, n_blocks * sizeof(cl_half));
                            backend_ctx->prealloc_moe_sa.allocate(backend_ctx->context, n_blocks * sizeof(cl_half));

                            const cl_uint n_kblocks = (cl_uint)(ne00 / 32);
                            unsigned short map_ratio_q5k = ne20 / ne11;
                            cl_kernel rq = backend_ctx->mul_mat_id.kernel_moe_reorder_quant_a_q8_1;
                            CL_CHECK(clSetKernelArg(rq, 0, sizeof(cl_mem),         &sub_buf_src1_pre));
                            CL_CHECK(clSetKernelArg(rq, 1, sizeof(cl_mem),         &buf_src2));
                            CL_CHECK(clSetKernelArg(rq, 2, sizeof(cl_mem),         &backend_ctx->prealloc_moe_qa.buffer));
                            CL_CHECK(clSetKernelArg(rq, 3, sizeof(cl_mem),         &backend_ctx->prealloc_moe_da.buffer));
                            CL_CHECK(clSetKernelArg(rq, 4, sizeof(cl_mem),         &backend_ctx->prealloc_moe_sa.buffer));
                            CL_CHECK(clSetKernelArg(rq, 5, sizeof(cl_mem),         &(backend_ctx->prealloc_total_tiles.buffer)));
                            CL_CHECK(clSetKernelArg(rq, 6, sizeof(cl_uint),        &ne00));
                            CL_CHECK(clSetKernelArg(rq, 7, sizeof(unsigned short), &map_ratio_q5k));
                            CL_CHECK(clSetKernelArg(rq, 8, sizeof(cl_uint),        &n_tile_size));
                            CL_CHECK(clSetKernelArg(rq, 9, sizeof(cl_uint),        &n_kblocks));
                            size_t rq_local[2]  = { 32, 1 };
                            size_t rq_global[2] = { (size_t)(((n_kblocks + 31) / 32) * 32), tok_slots };
                            backend_ctx->enqueue_ndrange_kernel(rq, 2, rq_global, rq_local, dst);

                            region.origin = offsetd;
                            region.size = ne0 * ne1 * ne2 * sizeof(float);
                            cl_mem dp_sub_buf_dst = clCreateSubBuffer(extrad->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                            CL_CHECK(status);
                            cl_image_format dp_ifd = {CL_R, CL_FLOAT};
                            cl_image_desc dp_idd = {CL_MEM_OBJECT_IMAGE1D_BUFFER, static_cast<size_t>(ne0 * ne1 * ne2), 0,0,0,0,0,0,0, {dp_sub_buf_dst}};
                            cl_mem dp_buf_dst_image = clCreateImage(backend_ctx->context, CL_MEM_WRITE_ONLY, &dp_ifd, &dp_idd, NULL, &status);
                            CL_CHECK(status);

                            int ne00i = (int)ne00, ne01i = (int)ne01;
                            cl_kernel dk = backend_ctx->mul_mat_id.kernel_gemm_moe_q8_1_dp4a_q5k;
                            int has_min_q5k = 1;
                            int aidx = 0;
                            CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &extra0_q5_K->q_img));
                            CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &extra0_q5_K->qh));
                            CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &extra0_q5_K->scale));
                            CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &extra0_q5_K->min));
                            CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &backend_ctx->prealloc_moe_qa.buffer));
                            CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &backend_ctx->prealloc_moe_da.buffer));
                            CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &backend_ctx->prealloc_moe_sa.buffer));
                            CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &buf_src2));
                            CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &buf_src2_emap));
                            CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &dp_buf_dst_image));
                            CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &(backend_ctx->prealloc_total_tiles.buffer)));
                            CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(int),    &ne00i));
                            CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(int),    &ne01i));
                            CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(int),    &backend_ctx->adreno_use_moe_ragged_dp4));
                            CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(int),    &has_min_q5k));

                            size_t dp_global[3] = { 64, (size_t)((ne01 + 63) / 64), (size_t)max_post_router_tile };
                            size_t dp_local[3]  = { 64, 1, 1 };
                            backend_ctx->enqueue_ndrange_kernel(dk, 3, dp_global, dp_local, dst);

                            clReleaseMemObject(sub_buf_src1_pre);
                            clReleaseMemObject(buf_src2);
                            clReleaseMemObject(buf_src2_emap);
                            clReleaseMemObject(dp_sub_buf_dst);
                            clReleaseMemObject(dp_buf_dst_image);
                            return;
                        }
                    }

                    // Create image for reordered src1
                    // Use pre-allocated placeholder
                    region.origin = 0;
                    region.size = ne00 * max_post_router_tile * n_tile_size * sizeof(float);
                    backend_ctx->prealloc_act_trans.allocate(backend_ctx->context, region.size);
                    buf_src1_reordered = clCreateSubBuffer(
                        backend_ctx->prealloc_act_trans.buffer,
                        0,
                        CL_BUFFER_CREATE_TYPE_REGION,
                        &region,
                        &status);
                    CL_CHECK(status);
                    cl_image_format image_format_buf_src1 = {CL_RGBA, CL_FLOAT};
                    cl_image_desc image_desc_buf_src1 = {CL_MEM_OBJECT_IMAGE1D_BUFFER, static_cast<size_t>(ne00 * max_post_router_tile * n_tile_size / 4), 0,0,0,0,0,0,0, {buf_src1_reordered}};
                    image_src1_reordered = clCreateImage(backend_ctx->context, CL_MEM_READ_ONLY, &image_format_buf_src1, &image_desc_buf_src1, NULL, &status);
                    CL_CHECK(status);

                    unsigned short map_ratio = ne20 / ne11;
                    GGML_ASSERT(((map_ratio == 1) || (map_ratio == ne20)) && "Map ratio not supported\n");
                    CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 0, sizeof(cl_mem),        &sub_buf_src1_pre));
                    CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 1, sizeof(cl_mem),        &buf_src2));
                    CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 2, sizeof(cl_mem),        &buf_src1_reordered));
                    CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 3, sizeof(cl_mem),        &(backend_ctx->prealloc_total_tiles.buffer)));
                    CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 4, sizeof(unsigned int),  &ne00));
                    CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 5, sizeof(unsigned short),  &map_ratio));
                    CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 6, sizeof(unsigned int),  &n_tile_size));

                    size_t reorder_b_local_size[3] = {256, 1, 1};
                    size_t reorder_b_global_size[3] = {static_cast<size_t>(((ne00 / 4) + 255) / 256 * 256), static_cast<size_t>(max_post_router_tile * n_tile_size), 1};

                    // Dispatch reorder kernel
                    backend_ctx->enqueue_ndrange_kernel(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 3, reorder_b_global_size, reorder_b_local_size, dst);

                    // MoE kernel prepare
                    // Create sub buffer for dst
                    region.origin = offsetd;
                    region.size = ne0 * ne1 * ne2 * sizeof(float);
                    sub_buf_dst = clCreateSubBuffer(
                        extrad->data_device,
                        0,
                        CL_BUFFER_CREATE_TYPE_REGION,
                        &region,
                        &status);
                    CL_CHECK(status);
                    // Create image for dst
                    cl_image_format image_format_buf_dst = {CL_R, CL_FLOAT};
                    cl_image_desc image_desc_buf_dst = {CL_MEM_OBJECT_IMAGE1D_BUFFER, static_cast<size_t>(ne0 * ne1 * ne2), 0,0,0,0,0,0,0, {sub_buf_dst}};
                    buf_dst_image = clCreateImage(backend_ctx->context, CL_MEM_WRITE_ONLY, &image_format_buf_dst, &image_desc_buf_dst, NULL, &status);
                    CL_CHECK(status);

                    // Set kernel args
                    int arg_idx = 0;
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_q5_K->q_img));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_q5_K->qh));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_q5_K->s));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_q5_K->d));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_q5_K->dm));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &image_src1_reordered));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &buf_src2));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &buf_src2_emap));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &buf_dst_image));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &(backend_ctx->prealloc_total_tiles.buffer)));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),       &ne00));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),       &ne01));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_uint),   &backend_ctx->adreno_use_moe_ragged));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_uint),   &backend_ctx->adreno_moe_ragged_skip_gran));

                    // set thread grid
                    global_size[1] = static_cast<size_t>((ne01 + 63) / 64);
                    global_size[2] = static_cast<size_t>(max_post_router_tile);
                    local_size[1] = 1;
                    local_size[2] = 1;

                    // Dispatch kernel
                    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_size, local_size, dst);

                    clReleaseMemObject(sub_buf_src1_pre);
                    clReleaseMemObject(buf_src1_reordered);
                    clReleaseMemObject(image_src1_reordered);
                    clReleaseMemObject(buf_src2);
                    clReleaseMemObject(buf_src2_emap);
                    clReleaseMemObject(sub_buf_dst);
                    clReleaseMemObject(buf_dst_image);
                }
                return;
            }
#endif //GGML_OPENCL_USE_ADRENO_KERNELS
        }
        case GGML_TYPE_Q6_K: {
#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
            if (use_adreno_moe_kernels(backend_ctx, src0)) {
                cl_int status;

                size_t local_size[3] = {64, 2, 1};
                size_t global_size[3] = {64, 2, 1};

                if (ne12 == 1) { // for gemv
                    kernel = backend_ctx->mul_mat_id.kernel_gemv_moe_q6_k_f32_ns;

                    cl_mem src1_sub_buffer, buf_src1_image, buf_src2;

                    // create a sub_buffer for src2
                    cl_buffer_region region;
                    region.origin = offset2;
                    region.size = ne20 * ne21 * sizeof(int);
                    buf_src2 = clCreateSubBuffer(extra2->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                    CL_CHECK(status);

                    // set thread grid
                    global_size[0] = static_cast<size_t>(((ne01 + 63) / 64) * 64);
                    global_size[1] = 4;
                    global_size[2] = static_cast<size_t>(ne20);
                    local_size[1] = 4;

                    // create a sub_buffer for src1
                    region.origin = offset1;
                    region.size = ne10 * ne11 * ne12 * sizeof(float);
                    src1_sub_buffer = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                    CL_CHECK(status);

                    // create image for src1
                    cl_image_format image_format_buf_src1 = {CL_RGBA, CL_FLOAT};
                    cl_image_desc image_desc_buf_src1 = {CL_MEM_OBJECT_IMAGE1D_BUFFER, static_cast<size_t>(ne10 * ne11 * ne12 / 4), 0,0,0,0,0,0,0, {src1_sub_buffer}};
                    buf_src1_image = clCreateImage(backend_ctx->context, CL_MEM_READ_ONLY, &image_format_buf_src1, &image_desc_buf_src1, NULL, &status);
                    CL_CHECK(status);

                    // Set kernel args
                    int arg_idx = 0;
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_q6_K->ql));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_q6_K->qh));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_q6_K->s));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_q6_K->d));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &buf_src1_image));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &buf_src2));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extrad->data_device));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_ulong),  &offsetd));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),       &ne00));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),       &ne01));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),       &ne11));

                    // launch kernel
                    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_size, local_size, dst);

                    // deallocate sub buffers and images
                    CL_CHECK(clReleaseMemObject(src1_sub_buffer));
                    CL_CHECK(clReleaseMemObject(buf_src1_image));
                    CL_CHECK(clReleaseMemObject(buf_src2));

                } else { // for gemm
                    kernel = backend_ctx->mul_mat_id.kernel_gemm_moe_q6_k_f32_ns;
                    if (backend_ctx->mul_mat_id.kernel_gemm_moe_q6_k_f32_ns_bin) {
                        kernel = backend_ctx->mul_mat_id.kernel_gemm_moe_q6_k_f32_ns_bin;
                    }

                    // Reorder router if called from test-backend-ops or when new router is generated.
                    // Otherwise reuse the reordered result from previous mul_mat_id call.
                    if ((strstr(src0->name, "as") != NULL) || backend_ctx->toggle_reorder) {
                        moe_router_reoerder(backend, src2, ne20);
                        backend_ctx->toggle_reorder = false;
                    }

                    cl_mem sub_buf_src1_pre, sub_buf_dst, buf_dst_image;
                    cl_mem buf_src1_reordered = nullptr, image_src1_reordered = nullptr;
                    cl_mem buf_src2, buf_src2_emap;

                    // dp4a (int8) q6_K MoE prefill GEMM
                    static const char * q6k_moe_dp4a_env = getenv("GGML_OPENCL_Q6K_MOE_DP4A");
                                 bool   use_moe_dp4a = (q6k_moe_dp4a_env != nullptr)
                                                         ? (atoi(q6k_moe_dp4a_env) != 0)
                                                         : (backend_ctx->adreno_gen == ADRENO_GPU_GEN::X2E
                                                            || backend_ctx->adreno_gen == ADRENO_GPU_GEN::X1E);
                    // dot prod has to be available
                    use_moe_dp4a = backend_ctx->has_integer_dot && use_moe_dp4a;
                    // bin kernel takes precedence
                    use_moe_dp4a = use_moe_dp4a && backend_ctx->mul_mat_id.kernel_gemm_moe_q6_k_f32_ns_bin == nullptr;

                    cl_buffer_region region;
                    region.origin = 0;
                    region.size = sizeof(int) * max_post_router_tile * n_tile_size;
                    buf_src2 = clCreateSubBuffer(backend_ctx->prealloc_post_router.buffer, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                    CL_CHECK(status);

                    region.origin = 0;
                    region.size = sizeof(short) * max_post_router_tile;
                    buf_src2_emap = clCreateSubBuffer(backend_ctx->prealloc_emap.buffer, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                    CL_CHECK(status);

                    // Reorder activations
                    // create a sub_buffer for src1
                    region.origin = offset1;
                    region.size = ne10 * ne11 * ne12 * sizeof(float);
                    sub_buf_src1_pre = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                    CL_CHECK(status);

                    unsigned short map_ratio = ne20 / ne11;
                    GGML_ASSERT(((map_ratio == 1) || (map_ratio == ne20)) && "Map ratio not supported\n");

                    if (!use_moe_dp4a) {
                        // Create image for reordered src1
                        region.origin = 0;
                        region.size = ne00 * max_post_router_tile * n_tile_size * sizeof(float);
                        backend_ctx->prealloc_act_trans.allocate(backend_ctx->context, region.size);
                        buf_src1_reordered = clCreateSubBuffer(
                            backend_ctx->prealloc_act_trans.buffer,
                            0,
                            CL_BUFFER_CREATE_TYPE_REGION,
                            &region,
                            &status);
                        CL_CHECK(status);
                        cl_image_format image_format_buf_src1 = {CL_RGBA, CL_FLOAT};
                        cl_image_desc image_desc_buf_src1 = {CL_MEM_OBJECT_IMAGE1D_BUFFER, static_cast<size_t>(ne00 * max_post_router_tile * n_tile_size / 4), 0,0,0,0,0,0,0, {buf_src1_reordered}};
                        if (backend_ctx->mul_mat_id.kernel_gemm_moe_q6_k_f32_ns_bin) {
                            // bin kernel uses slightly different image format
                            image_format_buf_src1 = {CL_R, CL_FLOAT};
                            image_desc_buf_src1.image_width = static_cast<size_t>(ne00 * max_post_router_tile * n_tile_size);
                        }
                        image_src1_reordered = clCreateImage(backend_ctx->context, CL_MEM_READ_ONLY, &image_format_buf_src1, &image_desc_buf_src1, NULL, &status);
                        CL_CHECK(status);

                        CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 0, sizeof(cl_mem),        &sub_buf_src1_pre));
                        CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 1, sizeof(cl_mem),        &buf_src2));
                        CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 2, sizeof(cl_mem),        &buf_src1_reordered));
                        CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 3, sizeof(cl_mem),        &(backend_ctx->prealloc_total_tiles.buffer)));
                        CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 4, sizeof(unsigned int),  &ne00));
                        CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 5, sizeof(unsigned short),  &map_ratio));
                        CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 6, sizeof(unsigned int),  &n_tile_size));

                        size_t reorder_b_local_size[3] = {256, 1, 1};
                        size_t reorder_b_global_size[3] = {static_cast<size_t>(((ne00 / 4) + 255) / 256 * 256), static_cast<size_t>(max_post_router_tile * n_tile_size), 1};

                        // Dispatch reorder kernel
                        backend_ctx->enqueue_ndrange_kernel(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 3, reorder_b_global_size, reorder_b_local_size, dst);
                    }

                    // MoE kernel prepare
                    // Create sub buffer for dst
                    region.origin = offsetd;
                    region.size = ne0 * ne1 * ne2 * sizeof(float);
                    sub_buf_dst = clCreateSubBuffer(
                        extrad->data_device,
                        0,
                        CL_BUFFER_CREATE_TYPE_REGION,
                        &region,
                        &status);
                    CL_CHECK(status);
                    // Create image for dst
                    cl_image_format image_format_buf_dst = {CL_R, CL_FLOAT};
                    cl_image_desc image_desc_buf_dst = {CL_MEM_OBJECT_IMAGE1D_BUFFER, static_cast<size_t>(ne0 * ne1 * ne2), 0,0,0,0,0,0,0, {sub_buf_dst}};
                    buf_dst_image = clCreateImage(backend_ctx->context, CL_MEM_WRITE_ONLY, &image_format_buf_dst, &image_desc_buf_dst, NULL, &status);
                    CL_CHECK(status);

                    if (use_moe_dp4a) {
                        const size_t tok_slots = (size_t)max_post_router_tile * n_tile_size;
                        const size_t n_blocks  = tok_slots * (ne00 / 32);
                        backend_ctx->prealloc_moe_qa.allocate(backend_ctx->context, tok_slots * ne00 * sizeof(cl_char));
                        backend_ctx->prealloc_moe_da.allocate(backend_ctx->context, n_blocks * sizeof(cl_half));
                        backend_ctx->prealloc_moe_sa.allocate(backend_ctx->context, n_blocks * sizeof(cl_half));

                        // fused reorder + q8_1 quant from the original activations
                        const cl_uint n_kblocks = (cl_uint)(ne00 / 32);
                        cl_kernel rq = backend_ctx->mul_mat_id.kernel_moe_reorder_quant_a_q8_1;
                        CL_CHECK(clSetKernelArg(rq, 0, sizeof(cl_mem),         &sub_buf_src1_pre));
                        CL_CHECK(clSetKernelArg(rq, 1, sizeof(cl_mem),         &buf_src2));
                        CL_CHECK(clSetKernelArg(rq, 2, sizeof(cl_mem),         &backend_ctx->prealloc_moe_qa.buffer));
                        CL_CHECK(clSetKernelArg(rq, 3, sizeof(cl_mem),         &backend_ctx->prealloc_moe_da.buffer));
                        CL_CHECK(clSetKernelArg(rq, 4, sizeof(cl_mem),         &backend_ctx->prealloc_moe_sa.buffer));
                        CL_CHECK(clSetKernelArg(rq, 5, sizeof(cl_mem),         &(backend_ctx->prealloc_total_tiles.buffer)));
                        CL_CHECK(clSetKernelArg(rq, 6, sizeof(cl_uint),        &ne00));
                        CL_CHECK(clSetKernelArg(rq, 7, sizeof(unsigned short), &map_ratio));
                        CL_CHECK(clSetKernelArg(rq, 8, sizeof(cl_uint),        &n_tile_size));
                        CL_CHECK(clSetKernelArg(rq, 9, sizeof(cl_uint),        &n_kblocks));
                        size_t rq_local[2]  = { 32, 1 };
                        size_t rq_global[2] = { (size_t)(((n_kblocks + 31) / 32) * 32), tok_slots };
                        backend_ctx->enqueue_ndrange_kernel(rq, 2, rq_global, rq_local, dst);

                        cl_kernel dk = backend_ctx->mul_mat_id.kernel_gemm_moe_q6_k_q8_1_dp4a;
                        int qi = 0;
                        CL_CHECK(clSetKernelArg(dk, qi++, sizeof(cl_mem), &extra0_q6_K->ql_img));
                        CL_CHECK(clSetKernelArg(dk, qi++, sizeof(cl_mem), &extra0_q6_K->qh));
                        CL_CHECK(clSetKernelArg(dk, qi++, sizeof(cl_mem), &extra0_q6_K->s));
                        CL_CHECK(clSetKernelArg(dk, qi++, sizeof(cl_mem), &extra0_q6_K->d));
                        CL_CHECK(clSetKernelArg(dk, qi++, sizeof(cl_mem), &backend_ctx->prealloc_moe_qa.buffer));
                        CL_CHECK(clSetKernelArg(dk, qi++, sizeof(cl_mem), &backend_ctx->prealloc_moe_da.buffer));
                        CL_CHECK(clSetKernelArg(dk, qi++, sizeof(cl_mem), &buf_src2));
                        CL_CHECK(clSetKernelArg(dk, qi++, sizeof(cl_mem), &buf_src2_emap));
                        CL_CHECK(clSetKernelArg(dk, qi++, sizeof(cl_mem), &buf_dst_image));
                        CL_CHECK(clSetKernelArg(dk, qi++, sizeof(cl_mem), &(backend_ctx->prealloc_total_tiles.buffer)));
                        CL_CHECK(clSetKernelArg(dk, qi++, sizeof(int),    &ne00));
                        CL_CHECK(clSetKernelArg(dk, qi++, sizeof(int),    &ne01));
                        CL_CHECK(clSetKernelArg(dk, qi++, sizeof(int),    &backend_ctx->adreno_use_moe_ragged_dp4));

                        size_t dp_global[3] = { 64, (size_t)((ne01 + 63) / 64), (size_t)max_post_router_tile };
                        size_t dp_local[3]  = { 64, 1, 1 };
                        backend_ctx->enqueue_ndrange_kernel(dk, 3, dp_global, dp_local, dst);

                        clReleaseMemObject(sub_buf_src1_pre);
                        clReleaseMemObject(buf_src2);
                        clReleaseMemObject(buf_src2_emap);
                        clReleaseMemObject(sub_buf_dst);
                        clReleaseMemObject(buf_dst_image);
                        return;
                    }

                    // Set kernel args
                    int arg_idx = 0;
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_q6_K->ql_img));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_q6_K->qh));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_q6_K->s));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_q6_K->d));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &image_src1_reordered));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &buf_src2));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &buf_src2_emap));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &buf_dst_image));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &(backend_ctx->prealloc_total_tiles.buffer)));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),       &ne00));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),       &ne01));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_uint),   &backend_ctx->adreno_use_moe_ragged));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_uint),   &backend_ctx->adreno_moe_ragged_skip_gran));

                    // set thread grid
                    global_size[1] = static_cast<size_t>((ne01 + 63) / 64);
                    global_size[2] = static_cast<size_t>(max_post_router_tile);
                    local_size[1] = 1;
                    local_size[2] = 1;

                    // Dispatch kernel
                    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_size, local_size, dst);

                    clReleaseMemObject(sub_buf_src1_pre);
                    clReleaseMemObject(buf_src1_reordered);
                    clReleaseMemObject(image_src1_reordered);
                    clReleaseMemObject(buf_src2);
                    clReleaseMemObject(buf_src2_emap);
                    clReleaseMemObject(sub_buf_dst);
                    clReleaseMemObject(buf_dst_image);
                }
                return;
            }
#endif //GGML_OPENCL_USE_ADRENO_KERNELS
        }
        case GGML_TYPE_MXFP4: {
#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
            if (use_adreno_moe_kernels(backend_ctx, src0)) {
                cl_int status;

                size_t local_size[3] = {64, 2, 1};
                size_t global_size[3] = {64, 2, 1};

                if (ne12 == 1) { // for gemv
                    kernel = backend_ctx->mul_mat_id.kernel_gemv_moe_mxfp4_f32_ns;

                    // Weight-as-texture MoE decode GEMV (see q4_K _wimg)
                    static const char * moe_decode_wimg_env = getenv("GGML_OPENCL_MOE_DECODE_WIMG");
                    const bool use_moe_decode_wimg = (moe_decode_wimg_env && (atoi(moe_decode_wimg_env) != 0))
                        && backend_ctx->mul_mat_id.kernel_gemv_moe_mxfp4_f32_ns_wimg != nullptr
                        && extra0_mxfp4->q_img != nullptr;
                    if (use_moe_decode_wimg) {
                        kernel = backend_ctx->mul_mat_id.kernel_gemv_moe_mxfp4_f32_ns_wimg;
                    }

                    cl_mem src1_sub_buffer, buf_src1_image, buf_src2;

                    // create a sub_buffer for src2
                    cl_buffer_region region;
                    region.origin = offset2;
                    region.size = ne20 * ne21 * sizeof(int);
                    buf_src2 = clCreateSubBuffer(extra2->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                    CL_CHECK(status);

                    // set thread grid
                    global_size[0] = static_cast<size_t>(((ne01 + 63) / 64) * 64);
                    global_size[1] = 4;
                    global_size[2] = static_cast<size_t>(ne20);
                    local_size[1] = 4;

                    // create a sub_buffer for src1
                    region.origin = offset1;
                    region.size = ne10 * ne11 * ne12 * sizeof(float);
                    src1_sub_buffer = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                    CL_CHECK(status);

                    // create image for src1
                    cl_image_format image_format_buf_src1 = {CL_RGBA, CL_FLOAT};
                    cl_image_desc image_desc_buf_src1 = {CL_MEM_OBJECT_IMAGE1D_BUFFER, static_cast<size_t>(ne10 * ne11 * ne12 / 4), 0,0,0,0,0,0,0, {src1_sub_buffer}};
                    buf_src1_image = clCreateImage(backend_ctx->context, CL_MEM_READ_ONLY, &image_format_buf_src1, &image_desc_buf_src1, NULL, &status);
                    CL_CHECK(status);

                    // Set kernel args
                    int arg_idx = 0;
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    use_moe_decode_wimg ? &extra0_mxfp4->q_img : &extra0_mxfp4->q));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_mxfp4->e));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &buf_src1_image));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &buf_src2));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extrad->data_device));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_ulong),  &offsetd));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),       &ne00));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),       &ne01));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),       &ne11));

                    // launch kernel
                    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_size, local_size, dst);

                    // deallocate sub buffers and images
                    CL_CHECK(clReleaseMemObject(src1_sub_buffer));
                    CL_CHECK(clReleaseMemObject(buf_src1_image));
                    CL_CHECK(clReleaseMemObject(buf_src2));

                } else { // for gemm
                    kernel = backend_ctx->mul_mat_id.kernel_gemm_moe_mxfp4_f32_ns;
                    if (backend_ctx->mul_mat_id.kernel_gemm_moe_mxfp4_f32_ns_bin) {
                        kernel = backend_ctx->mul_mat_id.kernel_gemm_moe_mxfp4_f32_ns_bin;
                    }

                    // Reorder router if called from test-backend-ops or when new router is generated.
                    // Otherwise reuse the reordered result from previous mul_mat_id call.
                    if ((strstr(src0->name, "as") != NULL) || backend_ctx->toggle_reorder) {
                        moe_router_reoerder(backend, src2, ne20);
                        backend_ctx->toggle_reorder = false;
                    }

                    cl_mem sub_buf_src1_pre, sub_buf_dst, buf_dst_image;
                    cl_mem buf_src1_reordered = nullptr, image_src1_reordered = nullptr;
                    cl_mem buf_src2, buf_src2_emap;

                    // dp4a (int8) prefill GEMM variant
                    static const char * mxfp4_moe_dp4a_env = getenv("GGML_OPENCL_MXFP4_MOE_DP4A");
                    bool use_moe_dp4a = mxfp4_moe_dp4a_env
                        ? (atoi(mxfp4_moe_dp4a_env) != 0)
                        : (backend_ctx->adreno_gen == ADRENO_GPU_GEN::X2E);
                    // dot prod has to be available
                    use_moe_dp4a = backend_ctx->has_integer_dot && use_moe_dp4a;
                    if (backend_ctx->mul_mat_id.kernel_gemm_moe_mxfp4_q8_1_dp4a_bin == nullptr) {
                        use_moe_dp4a = use_moe_dp4a && backend_ctx->mul_mat_id.kernel_gemm_moe_mxfp4_f32_ns_bin == nullptr;
                    }

                    cl_buffer_region region;
                    region.origin = 0;
                    region.size = sizeof(int) * max_post_router_tile * n_tile_size;
                    GGML_ASSERT(backend_ctx->prealloc_post_router.buffer);
                    buf_src2 = clCreateSubBuffer(backend_ctx->prealloc_post_router.buffer, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                    CL_CHECK(status);

                    region.origin = 0;
                    region.size = sizeof(short) * max_post_router_tile;
                    buf_src2_emap = clCreateSubBuffer(backend_ctx->prealloc_emap.buffer, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                    CL_CHECK(status);

                    // Reorder activations
                    // create a sub_buffer for src1
                    region.origin = offset1;
                    region.size = ne10 * ne11 * ne12 * sizeof(float);
                    sub_buf_src1_pre = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                    CL_CHECK(status);

                    unsigned short map_ratio = ne20 / ne11;
                    GGML_ASSERT(((map_ratio == 1) || (map_ratio == ne20)) && "Map ratio not supported\n");

                    if (!use_moe_dp4a) {
                        // Create image for reordered src1
                        // Use pre-allocated placeholder
                        region.origin = 0;
                        region.size = ne00 * max_post_router_tile * n_tile_size * sizeof(float);
                        backend_ctx->prealloc_act_trans.allocate(backend_ctx->context, region.size);
                        buf_src1_reordered = clCreateSubBuffer(
                            backend_ctx->prealloc_act_trans.buffer,
                            0,
                            CL_BUFFER_CREATE_TYPE_REGION,
                            &region,
                            &status);
                        CL_CHECK(status);
                        cl_image_format image_format_buf_src1;
                        cl_image_desc image_desc_buf_src1;
                        image_format_buf_src1 = {CL_RGBA, CL_FLOAT};
                        image_desc_buf_src1 = {CL_MEM_OBJECT_IMAGE1D_BUFFER, static_cast<size_t>(ne00 * max_post_router_tile * n_tile_size / 4), 0,0,0,0,0,0,0, {buf_src1_reordered}};
                        if (backend_ctx->mul_mat_id.kernel_gemm_moe_mxfp4_f32_ns_bin) {
                            // bin kernel uses slightly different image format
                            image_format_buf_src1 = {CL_R, CL_FLOAT};
                            image_desc_buf_src1.image_width = static_cast<size_t>(ne00 * max_post_router_tile * n_tile_size);
                        }
                        image_src1_reordered = clCreateImage(backend_ctx->context, CL_MEM_READ_ONLY, &image_format_buf_src1, &image_desc_buf_src1, NULL, &status);
                        CL_CHECK(status);

                        CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 0, sizeof(cl_mem),        &sub_buf_src1_pre));
                        CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 1, sizeof(cl_mem),        &buf_src2));
                        CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 2, sizeof(cl_mem),        &buf_src1_reordered));
                        CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 3, sizeof(cl_mem),        &(backend_ctx->prealloc_total_tiles.buffer)));
                        CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 4, sizeof(unsigned int),  &ne00));
                        CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 5, sizeof(unsigned short), &map_ratio));
                        CL_CHECK(clSetKernelArg(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 6, sizeof(unsigned int),  &n_tile_size));

                        size_t reorder_b_local_size[3] = {256, 1, 1};
                        size_t reorder_b_global_size[3] = {static_cast<size_t>(((ne00 / 4) + 255) / 256 * 256), static_cast<size_t>(max_post_router_tile * n_tile_size), 1};

                        // Dispatch reorder kernel
                        backend_ctx->enqueue_ndrange_kernel(backend_ctx->mul_mat_id.kernel_moe_reorder_b, 3, reorder_b_global_size, reorder_b_local_size, dst);
                    }

                    // MoE kernel prepare
                    // Create sub buffer for dst
                    region.origin = offsetd;
                    region.size = ne0 * ne1 * ne2 * sizeof(float);
                    sub_buf_dst = clCreateSubBuffer(
                        extrad->data_device,
                        0,
                        CL_BUFFER_CREATE_TYPE_REGION,
                        &region,
                        &status);
                    CL_CHECK(status);
                    // Create image for dst
                    cl_image_format image_format_buf_dst = {CL_R, CL_FLOAT};
                    cl_image_desc image_desc_buf_dst = {CL_MEM_OBJECT_IMAGE1D_BUFFER, static_cast<size_t>(ne0 * ne1 * ne2), 0,0,0,0,0,0,0, {sub_buf_dst}};
                    buf_dst_image = clCreateImage(backend_ctx->context, CL_MEM_WRITE_ONLY, &image_format_buf_dst, &image_desc_buf_dst, NULL, &status);
                    CL_CHECK(status);

                    if (use_moe_dp4a) {
                        const size_t tok_slots = (size_t)max_post_router_tile * n_tile_size;
                        const size_t n_blocks  = tok_slots * (ne00 / 32);
                        backend_ctx->prealloc_moe_qa.allocate(backend_ctx->context, tok_slots * ne00 * sizeof(cl_char));
                        backend_ctx->prealloc_moe_da.allocate(backend_ctx->context, n_blocks * sizeof(cl_half));
                        backend_ctx->prealloc_moe_sa.allocate(backend_ctx->context, n_blocks * sizeof(cl_half));

                        // fused reorder + q8_1 quant straight from the original
                        // activations (no intermediate f32 reorder buffer). mxfp4 has no
                        // min term so the GEMM ignores sa, but reorder_quant still writes it.
                        const cl_uint n_kblocks = (cl_uint)(ne00 / 32);
                        cl_kernel rq = backend_ctx->mul_mat_id.kernel_moe_reorder_quant_a_q8_1;
                        CL_CHECK(clSetKernelArg(rq, 0, sizeof(cl_mem),         &sub_buf_src1_pre));
                        CL_CHECK(clSetKernelArg(rq, 1, sizeof(cl_mem),         &buf_src2));
                        CL_CHECK(clSetKernelArg(rq, 2, sizeof(cl_mem),         &backend_ctx->prealloc_moe_qa.buffer));
                        CL_CHECK(clSetKernelArg(rq, 3, sizeof(cl_mem),         &backend_ctx->prealloc_moe_da.buffer));
                        CL_CHECK(clSetKernelArg(rq, 4, sizeof(cl_mem),         &backend_ctx->prealloc_moe_sa.buffer));
                        CL_CHECK(clSetKernelArg(rq, 5, sizeof(cl_mem),         &(backend_ctx->prealloc_total_tiles.buffer)));
                        CL_CHECK(clSetKernelArg(rq, 6, sizeof(cl_uint),        &ne00));
                        CL_CHECK(clSetKernelArg(rq, 7, sizeof(unsigned short), &map_ratio));
                        CL_CHECK(clSetKernelArg(rq, 8, sizeof(cl_uint),        &n_tile_size));
                        CL_CHECK(clSetKernelArg(rq, 9, sizeof(cl_uint),        &n_kblocks));
                        size_t rq_local[2]  = { 32, 1 };
                        size_t rq_global[2] = { (size_t)(((n_kblocks + 31) / 32) * 32), tok_slots };
                        backend_ctx->enqueue_ndrange_kernel(rq, 2, rq_global, rq_local, dst);

                        // dp4a GEMM
                        cl_kernel dk = backend_ctx->mul_mat_id.kernel_gemm_moe_mxfp4_q8_1_dp4a;
                        if (backend_ctx->mul_mat_id.kernel_gemm_moe_mxfp4_q8_1_dp4a_bin) {
                            dk = backend_ctx->mul_mat_id.kernel_gemm_moe_mxfp4_q8_1_dp4a_bin;
                        }
                        int aidx = 0;
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &extra0_mxfp4->q_img));
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &extra0_mxfp4->e));
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &backend_ctx->prealloc_moe_qa.buffer));
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &backend_ctx->prealloc_moe_da.buffer));
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &buf_src2));
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &buf_src2_emap));
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &buf_dst_image));
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(cl_mem), &(backend_ctx->prealloc_total_tiles.buffer)));
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(int),    &ne00));
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(int),    &ne01));
                        CL_CHECK(clSetKernelArg(dk, aidx++, sizeof(int),    &backend_ctx->adreno_use_moe_ragged_dp4));

                        size_t dp_global[3] = { 64, (size_t)((ne01 + 63) / 64), (size_t)max_post_router_tile };
                        size_t dp_local[3]  = { 64, 1, 1 };
                        backend_ctx->enqueue_ndrange_kernel(dk, 3, dp_global, dp_local, dst);

                        clReleaseMemObject(sub_buf_src1_pre);
                        clReleaseMemObject(buf_src2);
                        clReleaseMemObject(buf_src2_emap);
                        clReleaseMemObject(sub_buf_dst);
                        clReleaseMemObject(buf_dst_image);
                        return;
                    }

                    // Set kernel args
                    int arg_idx = 0;
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_mxfp4->q_img));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &extra0_mxfp4->e));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &image_src1_reordered));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &buf_src2));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &buf_src2_emap));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &buf_dst_image));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem),    &(backend_ctx->prealloc_total_tiles.buffer)));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),       &ne00));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(int),       &ne01));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_uint),   &backend_ctx->adreno_use_moe_ragged));
                    CL_CHECK(clSetKernelArg(kernel, arg_idx++, sizeof(cl_uint),   &backend_ctx->adreno_moe_ragged_skip_gran));

                    // set thread grid
                    global_size[1] = static_cast<size_t>((ne01 + 63) / 64);
                    global_size[2] = static_cast<size_t>(max_post_router_tile);
                    local_size[1] = 1;
                    local_size[2] = 1;

                    // Dispatch kernel
                    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_size, local_size, dst);

                    clReleaseMemObject(sub_buf_src1_pre);
                    clReleaseMemObject(buf_src1_reordered);
                    clReleaseMemObject(image_src1_reordered);
                    clReleaseMemObject(buf_src2);
                    clReleaseMemObject(buf_src2_emap);
                    clReleaseMemObject(sub_buf_dst);
                    clReleaseMemObject(buf_dst_image);
                }
                return;
            } // fallback to generic MoE mxfp4 kernel
#endif // GGML_OPENCL_USE_ADRENO_KERNELS

#ifdef GGML_OPENCL_SOA_Q
            kernel = backend_ctx->mul_mat_id.kernel_mul_mv_id_mxfp4_f32_flat;

            cl_mem q;
            if (backend_ctx->gpu_family == INTEL) {
                sgs  = 16;
                nsg  = 2;
                ndst = 2;

                q = extra0_mxfp4->q;
            } else if (backend_ctx->gpu_family == ADRENO) {
                sgs  = 64;
                nsg  = 1;
                ndst = 4;

                q = extra0_mxfp4->q_img;
            } else {
                GGML_ASSERT(false && "TODO: Unknown GPU");
            }

            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &q));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0_mxfp4->e));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extra2->data_device));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offset2));
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb01));
            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb02));
            CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb03));
            CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne11));
            CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne12));
            CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &nb11));
            CL_CHECK(clSetKernelArg(kernel, 15, sizeof(cl_ulong), &nb12));
            CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nb13));
            CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &ne20));
            CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &ne21));
            CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &nb21));
            CL_CHECK(clSetKernelArg(kernel, 20, sizeof(int),      &ne0));
            CL_CHECK(clSetKernelArg(kernel, 21, sizeof(int),      &ne1));
            CL_CHECK(clSetKernelArg(kernel, 22, sizeof(int),      &r2));
            CL_CHECK(clSetKernelArg(kernel, 23, sizeof(int),      &r3));
#else // GGML_OPENCL_SOA_Q
            kernel = backend_ctx->mul_mat_id.kernel_mul_mv_id_mxfp4_f32;

            if (backend_ctx->gpu_family == INTEL) {
                sgs  = 16;
                nsg  = 2;
                ndst = 2;
            } else if (backend_ctx->gpu_family == ADRENO) {
                sgs  = 64;
                nsg  = 2;
                ndst = 2;
            } else {
                GGML_ASSERT(false && "TODO: Unknown GPU");
            }

            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extra2->data_device));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offset2));
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb01));
            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb02));
            CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb03));
            CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne11));
            CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne12));
            CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &nb11));
            CL_CHECK(clSetKernelArg(kernel, 15, sizeof(cl_ulong), &nb12));
            CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nb13));
            CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &ne20));
            CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &ne21));
            CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &nb21));
            CL_CHECK(clSetKernelArg(kernel, 20, sizeof(int),      &ne0));
            CL_CHECK(clSetKernelArg(kernel, 21, sizeof(int),      &ne1));
            CL_CHECK(clSetKernelArg(kernel, 22, sizeof(int),      &r2));
            CL_CHECK(clSetKernelArg(kernel, 23, sizeof(int),      &r3));
            CL_CHECK(clSetKernelArg(kernel, 24, sizeof(float)*sgs,nullptr));
#endif // GGML_OPENCL_SOA_Q
            break;
        }
        default:
            GGML_ASSERT(false && "not implemented");;
    }

    int _ne1 = 1;
    int ne123 = dst_rows;

    size_t global_work_size[] = {(size_t)(ne01+ndst*nsg-1)/(ndst*nsg)*sgs, (size_t)(_ne1+nrows-1)/nrows*nsg, (size_t)ne123};
    size_t local_work_size[] = {(size_t)sgs, (size_t)nsg, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}
