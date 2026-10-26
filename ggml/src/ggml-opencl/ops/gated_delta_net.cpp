#include "../cl-common.h"
#include "../ops.h"

void ggml_cl_load_kernels_gated_delta_net(ggml_backend_opencl_context * backend_ctx) {
    cl_int err;
    const std::string & compile_opts = backend_ctx->kernel_compile_opts;
    // gated_delta_net: one kernel per (S_V, KDA, tgpp) triple.
    {
    #ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gated_delta_net.cl.h"
        };
    #else
        const std::string kernel_src = read_file("gated_delta_net.cl");
    #endif

        const int gdn_sizes[4] = { 16, 32, 64, 128 };
        const int sg_size = backend_ctx->gpu_family == GPU_FAMILY::ADRENO ? 64 : backend_ctx->gpu_family == GPU_FAMILY::INTEL ? 32 : -1;
        if (sg_size < 0) {
            GGML_LOG_ERROR("Unsupported GPU Family: only Adreno and Intel are supported.\n");
            exit(1);
        }

        for (int si = 0; si < 4; si++) {
            const int S_V = gdn_sizes[si];

            // MUST match the dispatcher heuristic in ggml_cl_gated_delta_net exactly.
            int lanes_per_column;
            if (S_V >= 128) {
                lanes_per_column = 8;
            } else {
                lanes_per_column = std::min(S_V, sg_size);
            }

            // Round LANES_PER_COLUMN down until it is:
            //  * power-of-two
            //  * divides both S_V and sg_size
            while (lanes_per_column > 1 &&
                    (((lanes_per_column & (lanes_per_column - 1)) != 0) ||
                    (S_V % lanes_per_column) != 0 ||
                    (sg_size % lanes_per_column) != 0)) {
                lanes_per_column >>= 1;
            }

            GGML_ASSERT(lanes_per_column >= 1);
            GGML_ASSERT(((lanes_per_column & (lanes_per_column - 1)) == 0));
            GGML_ASSERT((S_V % lanes_per_column) == 0);
            GGML_ASSERT((sg_size % lanes_per_column) == 0);

            const bool is_partial_reduce = (lanes_per_column != 1) && (lanes_per_column < sg_size);
            int use_qcom_shuffle = 0;
            if (is_partial_reduce) {
                if (backend_ctx->has_qcom_subgroup_shuffle) {
                    use_qcom_shuffle = 1;
                }
            }
            for (int kda = 0; kda < 2; kda++) {
                for (int tgpp = 0; tgpp < 2; tgpp++) {
                    const int cpl = (tgpp == 0) ? 1 : 4;
                    const int spw  = (tgpp == 0) ? 1 : 1;

                    std::string opts = compile_opts;
                    opts += " -DS_V=" + std::to_string(S_V);
                    opts += " -DKDA=" + std::to_string(kda);
                    opts += " -DSUBGROUP_SIZE=" + std::to_string(sg_size);
                    opts += " -DLANES_PER_COLUMN=" + std::to_string(lanes_per_column);
                    opts += " -DCOLS_PER_LANE_GROUP=" + std::to_string(cpl);
                    opts += " -DUSE_QCOM_SUBGROUP_SHUFFLE=" + std::to_string(use_qcom_shuffle);

                    // Since spw=1 is found to be optimal, SUBGROUPS_PER_WG > 1 code in
                    // the kernel is removed. If you want to experiment with spw > 1,
                    // Please remember to implement code to handle it.
                    opts += " -DSUBGROUPS_PER_WG=" + std::to_string(spw);

                    cl_program prog = build_program_from_source(backend_ctx, kernel_src.c_str(), opts);

                    CL_CHECK((backend_ctx->gated_delta_net.kernel_gated_delta_net_f32[si][kda][tgpp] =
                                clCreateKernel(prog, "kernel_gated_delta_net", &err), err));
                    CL_CHECK(clReleaseProgram(prog));
                }
            }
        }
        GGML_LOG_CONT(".");
    }
}

void ggml_cl_gated_delta_net(ggml_backend_t backend, ggml_tensor * dst) {
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    const ggml_tensor * src_q     = dst->src[0];
    const ggml_tensor * src_k     = dst->src[1];
    const ggml_tensor * src_v     = dst->src[2];
    const ggml_tensor * src_g     = dst->src[3];
    const ggml_tensor * src_beta  = dst->src[4];
    const ggml_tensor * src_state = dst->src[5];

    GGML_ASSERT(src_q && src_q->extra);
    GGML_ASSERT(src_k && src_k->extra);
    GGML_ASSERT(src_v && src_v->extra);
    GGML_ASSERT(src_g && src_g->extra);
    GGML_ASSERT(src_beta && src_beta->extra);
    GGML_ASSERT(src_state && src_state->extra);

    ggml_backend_opencl_context * backend_ctx = (ggml_backend_opencl_context *) backend->context;

    const cl_uint S_v      = (cl_uint) src_v->ne[0];
    const cl_uint H_v      = (cl_uint) src_v->ne[1];
    const cl_uint n_tokens = (cl_uint) src_v->ne[2];
    const cl_uint n_seqs   = (cl_uint) src_v->ne[3];
    const cl_uint K        = (cl_uint) ggml_get_op_params_i32(dst, 0);

    int si;
    switch (S_v) {
        case 16:  si = 0; break;
        case 32:  si = 1; break;
        case 64:  si = 2; break;
        case 128: si = 3; break;
        default:
            GGML_ASSERT(false && "ggml_cl_gated_delta_net: unsupported S_v");
    }

    const int kda = (src_g->ne[0] == (int64_t) S_v) ? 1 : 0;

    // TODO: Optimize when S_v!=128. Not necessary for now as Qwen3.5/6 are all S_v=128
    // token generation mode (tgpp=0):
    // process 1 token at a time, so columns per lane (cpl) == 1
    // prompt processing mode (tgpp=1):
    // cpl=4 to process 4 tokens for single-token. 4 is chosen for Adreno 750 as per
    // work-item/thread has at most 128 registers.
    // All Qwen3.5/6 models are S_v == 128, so LANES_PER_COLUMN == 8
    // such that ROWS_PER_LANE = 128/8 = 16
    // Variables in the kernel:
    // k_reg, q_reg, g_exp are all 16 floats
    // s_shard has cpl*ROWS_PER_LANE = 4*16 = 64 floats
    // Total 112 registers used.
    // subgroups_per_workgroup (spw) can be set to 1,2,4,8,16 for tg and 1,2,4 for pp
    // for S_v=128.
    // Empirically found that when spw=1, we get the best performance for both tg and pp
    const int tgpp = (n_tokens == 1) ? 0 : 1;
    const int cpl  = (tgpp == 0) ? 1 : 4;
    // spw needs adjustment when S_v != 128
    const int spw  = (tgpp == 0) ? 1 : 1;

    cl_kernel kernel = backend_ctx->gated_delta_net.kernel_gated_delta_net_f32[si][kda][tgpp];
    GGML_ASSERT(kernel != nullptr);

    const cl_uint s_off = S_v * H_v * n_tokens * n_seqs;

    const cl_uint sq1 = (cl_uint)(src_q->nb[1]    / sizeof(float));
    const cl_uint sq2 = (cl_uint)(src_q->nb[2]    / sizeof(float));
    const cl_uint sq3 = (cl_uint)(src_q->nb[3]    / sizeof(float));
    const cl_uint sv1 = (cl_uint)(src_v->nb[1]    / sizeof(float));
    const cl_uint sv2 = (cl_uint)(src_v->nb[2]    / sizeof(float));
    const cl_uint sv3 = (cl_uint)(src_v->nb[3]    / sizeof(float));
    const cl_uint sb1 = (cl_uint)(src_beta->nb[1] / sizeof(float));
    const cl_uint sb2 = (cl_uint)(src_beta->nb[2] / sizeof(float));
    const cl_uint sb3 = (cl_uint)(src_beta->nb[3] / sizeof(float));

    const cl_uint H_k = (cl_uint) src_q->ne[1];
    const cl_uint rq3 = (cl_uint)(src_v->ne[3] / src_q->ne[3]);

    const float scale = 1.0f / sqrtf((float) S_v);

    ggml_tensor_extra_cl * extra_q     = (ggml_tensor_extra_cl *) src_q->extra;
    ggml_tensor_extra_cl * extra_k     = (ggml_tensor_extra_cl *) src_k->extra;
    ggml_tensor_extra_cl * extra_v     = (ggml_tensor_extra_cl *) src_v->extra;
    ggml_tensor_extra_cl * extra_g     = (ggml_tensor_extra_cl *) src_g->extra;
    ggml_tensor_extra_cl * extra_beta  = (ggml_tensor_extra_cl *) src_beta->extra;
    ggml_tensor_extra_cl * extra_state = (ggml_tensor_extra_cl *) src_state->extra;
    ggml_tensor_extra_cl * extra_dst   = (ggml_tensor_extra_cl *) dst->extra;

    const cl_ulong off_q     = extra_q->offset     + src_q->view_offs;
    const cl_ulong off_k     = extra_k->offset     + src_k->view_offs;
    const cl_ulong off_v     = extra_v->offset     + src_v->view_offs;
    const cl_ulong off_g     = extra_g->offset     + src_g->view_offs;
    const cl_ulong off_beta  = extra_beta->offset  + src_beta->view_offs;
    const cl_ulong off_state = extra_state->offset + src_state->view_offs;
    const cl_ulong off_dst   = extra_dst->offset   + dst->view_offs;

    int idx = 0;
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_mem),   &extra_q->data_device));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_ulong), &off_q));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_mem),   &extra_k->data_device));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_ulong), &off_k));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_mem),   &extra_v->data_device));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_ulong), &off_v));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_mem),   &extra_g->data_device));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_ulong), &off_g));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_mem),   &extra_beta->data_device));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_ulong), &off_beta));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_mem),   &extra_state->data_device));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_ulong), &off_state));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_mem),   &extra_dst->data_device));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_ulong), &off_dst));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint),  &H_v));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint),  &n_tokens));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint),  &n_seqs));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint),  &s_off));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint),  &sq1));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint),  &sq2));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint),  &sq3));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint),  &sv1));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint),  &sv2));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint),  &sv3));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint),  &sb1));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint),  &sb2));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint),  &sb3));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint),  &H_k));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint),  &rq3));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(float),    &scale));
    CL_CHECK(clSetKernelArg(kernel, idx++, sizeof(cl_uint),    &K));

    // Subgroup size is 64 for Adreno and 32 for Intel
    const int sg_size = backend_ctx->gpu_family == GPU_FAMILY::ADRENO ? 64 : backend_ctx->gpu_family == GPU_FAMILY::INTEL ? 32 : -1;
    if (sg_size < 0) {
        GGML_LOG_ERROR("Unsupported GPU Family: only Adreno and Intel are supported.\n");
        exit(1);
    }

    // For the subgroup-shuffle kernel, we can safely prefer 8 lanes/column for S_v>=128
    // For the subgroup-shuffle kernel:
    //   S_v >= 128  -> prefer 8 lanes/column (good occupancy & register pressure tradeoff)
    //   else        -> min(S_v, subgroup_size)
    int lanes_per_column;
    if ((int)S_v >= 128) {
        lanes_per_column = 8;
    } else {
        lanes_per_column = std::min((int)S_v, sg_size);
    }

    // Max workgroup size for Adreno 750 is 1024
    const int wg_size = sg_size * spw;

    // Ensure lanes_per_column is a power-of-two and divides both S_v and subgroup_size.
    // (Required for lane-group shuffle-xor reduction correctness.)
    while (lanes_per_column > 1 &&
            (((lanes_per_column & (lanes_per_column - 1)) != 0) ||
            (((int)S_v % lanes_per_column) != 0) ||
            (sg_size % lanes_per_column) != 0)) {
        lanes_per_column >>= 1;
    }
    GGML_ASSERT(lanes_per_column >= 1);
    GGML_ASSERT(((lanes_per_column & (lanes_per_column - 1)) == 0));
    GGML_ASSERT(((int)S_v % lanes_per_column) == 0);
    GGML_ASSERT((sg_size % lanes_per_column) == 0);

    const int cols_per_wg = spw * (sg_size / lanes_per_column) * cpl;
    GGML_ASSERT(cols_per_wg > 0);
    GGML_ASSERT(((int)S_v % cols_per_wg) == 0);

    size_t global_work_size[3];
    size_t local_work_size[3];

    global_work_size[0] = (size_t) H_v * (size_t) wg_size;
    global_work_size[1] = (size_t) n_seqs;
    global_work_size[2] = (size_t) S_v / (size_t) cols_per_wg;

    local_work_size[0]  = (size_t) wg_size;
    local_work_size[1]  = 1;
    local_work_size[2]  = 1;

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}
