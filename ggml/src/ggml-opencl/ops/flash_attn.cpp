#include "../cl-common.h"
#include "../ops.h"

struct ggml_cl_flash_attn_temp_buffer {
    cl_mem data = nullptr;

    ~ggml_cl_flash_attn_temp_buffer() {
        if (data != nullptr) {
            CL_CHECK(clReleaseMemObject(data));
            data = nullptr;
        }
    }
};

void ggml_cl_load_kernels_flash_attn(ggml_backend_opencl_context * backend_ctx) {
    cl_int err;
    const std::string & compile_opts = backend_ctx->kernel_compile_opts;
#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
    // Adreno xmem SDPA
    if (backend_ctx->gpu_family == GPU_FAMILY::ADRENO) {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "sdpa_xmem_f32_f16_os8.cl.h"
        };
#else
        const std::string kernel_src = read_file("sdpa_xmem_f32_f16_os8.cl");
#endif
        cl_program program = build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        auto & xmem_attn = backend_ctx->adreno_xmem_attn;
        CL_CHECK((xmem_attn.kernel_q_f32_to_img_scaled =
            clCreateKernel(program, "adreno_xmem_attn_q_f32_to_img_scaled", &err), err));
        CL_CHECK((xmem_attn.kernel_kv_f32_to_img_gqa =
            clCreateKernel(program, "adreno_xmem_attn_kv_f32_to_img_gqa", &err), err));
        CL_CHECK((xmem_attn.kernel_kv_f16_to_img_gqa =
            clCreateKernel(program, "adreno_xmem_attn_kv_f16_to_img_gqa", &err), err));
        CL_CHECK((xmem_attn.kernel_img_to_f32 =
            clCreateKernel(program, "adreno_xmem_attn_img_to_f32", &err), err));
        CL_CHECK((xmem_attn.kernel_k_gather =
            clCreateKernel(program, "adreno_xmem_attn_k_gather", &err), err));
        CL_CHECK((xmem_attn.kernel_pack_k =
            clCreateKernel(program, "adreno_xmem_attn_pack_k", &err), err));
        CL_CHECK((xmem_attn.kernel_qk_gemm =
            clCreateKernel(program, "adreno_xmem_attn_qk_gemm", &err), err));
        CL_CHECK((xmem_attn.kernel_softmax_reduce_basic =
            clCreateKernel(program, "adreno_xmem_attn_softmax_reduce_basic", &err), err));
        CL_CHECK((xmem_attn.kernel_softmax_apply_basic =
            clCreateKernel(program, "adreno_xmem_attn_softmax_apply_basic", &err), err));
        CL_CHECK((xmem_attn.kernel_mask_scores =
            clCreateKernel(program, "adreno_xmem_attn_mask_scores", &err), err));
        CL_CHECK((xmem_attn.kernel_pack_v =
            clCreateKernel(program, "adreno_xmem_attn_pack_v", &err), err));
        CL_CHECK((xmem_attn.kernel_pv_gemm =
            clCreateKernel(program, "adreno_xmem_attn_pv_gemm", &err), err));
        CL_CHECK(clReleaseProgram(program));
        xmem_attn.compiled = true;
        GGML_LOG_CONT(".");
    }

    // repack
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "flash_attn_repack.cl.h"
        };
#else
        const std::string kernel_src = read_file("flash_attn_repack.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->fa.kernel_repack_q_for_wmm = clCreateKernel(prog, "kernel_repack_q_for_wmm", &err), err));
        CL_CHECK((backend_ctx->fa.kernel_repack_k_for_wmm = clCreateKernel(prog, "kernel_repack_k_for_wmm", &err), err));
        CL_CHECK((backend_ctx->fa.kernel_repack_v_for_wmm = clCreateKernel(prog, "kernel_repack_v_for_wmm", &err), err));
        CL_CHECK((backend_ctx->fa.kernel_repack_mask_for_wmm = clCreateKernel(prog, "kernel_repack_mask_for_wmm", &err), err));
        GGML_LOG_CONT(".");
    }

    // kernel_flash_attn_f32_f16_bin
    {
        auto opencl_c_std =
            std::string("CL") + std::to_string(backend_ctx->opencl_c_version.major) + "." + std::to_string(backend_ctx->opencl_c_version.minor);
        std::string bin_compile_opts = std::string("-cl-std=") + opencl_c_std +
                " -cl-mad-enable "
                " -cl-fast-relaxed-math";

        size_t bin_size = 0;
        backend_ctx->fa.kernel_flash_attn_f32_f16_bin = nullptr;

        if (use_adreno_bin_kernels(backend_ctx)) {
            const char * kernel_bin = (const char *)backend_ctx->get_adreno_bin_kernel("flash_attn_f32_f16_wmm", &bin_size);
            if (kernel_bin && bin_size > 0) {
                cl_program prog =
                    build_program_from_binary(backend_ctx->context, backend_ctx->device, kernel_bin, bin_compile_opts, bin_size);

                CL_CHECK((backend_ctx->fa.kernel_flash_attn_f32_f16_bin = clCreateKernel(prog, "flash_attn_f32_f16", &err), err));
                CL_CHECK(clReleaseProgram(prog));
                GGML_LOG_CONT(".");
            }
        }
    }
#endif // GGML_OPENCL_USE_ADRENO_KERNELS
    GGML_UNUSED(backend_ctx);
    GGML_UNUSED(err);
    GGML_UNUSED(compile_opts);
}

bool use_fa_bin_kernels_prefill(
        const ggml_backend_opencl_context * backend_ctx,
        const ggml_tensor * q, const ggml_tensor * k, const ggml_tensor * v) {
#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
    if (backend_ctx->fa.kernel_flash_attn_f32_f16_bin == nullptr) {
        return false;
    }

    const bool is_mixed = q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16;
    const bool is_q8_0 = q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_Q8_0 && v->type == GGML_TYPE_Q8_0;

    const int n_q = q->ne[1];
    const int dk = q->ne[0];
    const int dv = v->ne[0];

    constexpr bool prefill_only = true;

    return (backend_ctx->gpu_family == GPU_FAMILY::ADRENO &&
            (is_mixed || is_q8_0) && (dk == dv)
            && (dk == 64 || dk == 128 || dk == 256 || dk == 512)
            && (!prefill_only || n_q != 1));
#else
    GGML_UNUSED(backend_ctx);
    GGML_UNUSED(q);
    GGML_UNUSED(k);
    GGML_UNUSED(v);
    return false;
#endif
}

// FA per-(dk,dv) tile tuning table + GGML_OPENCL_FA_TUNE override parsing.
#include "../fa_tune.h"

// FA variant key for the per-(dk,dv,variant) lazy compile cache.
// Kernel built on first dispatch to reduce kernel loading time.
// NB - a warmup run is recommended to get all necessary FA variants compiled
// before actual runs.
enum ggml_opencl_fa_variant {
    FA_VARIANT_PRE      = 0,  // prepass kernels (kv_pad, mask_pad, blk)
    FA_VARIANT_F16      = 1,
    FA_VARIANT_F32      = 2,
    FA_VARIANT_F32_F16  = 3,
    FA_VARIANT_Q8_0     = 4,
    FA_VARIANT_Q4_0     = 5,
    FA_VARIANT_F32_F16_SPLIT = 6,
    FA_VARIANT_Q8_0_SPLIT    = 7,
    FA_VARIANT_Q4_0_SPLIT    = 8,
};

static std::string ggml_opencl_fa_kernel_src(ggml_opencl_fa_variant v) {
#ifdef GGML_OPENCL_EMBED_KERNELS
    switch (v) {
        case FA_VARIANT_F16:
            return std::string{
                #include "flash_attn_f16.cl.h"
            };
        case FA_VARIANT_F32:
            return std::string{
                #include "flash_attn_f32.cl.h"
            };
        case FA_VARIANT_F32_F16:
        case FA_VARIANT_F32_F16_SPLIT:
            return std::string{
                #include "flash_attn_f32_f16.cl.h"
            };
        case FA_VARIANT_PRE:
            return std::string{
                #include "flash_attn_pre_f16.cl.h"
            };
        case FA_VARIANT_Q8_0:
        case FA_VARIANT_Q8_0_SPLIT:
            return std::string{
                #include "flash_attn_f32_q8_0.cl.h"
            };
        case FA_VARIANT_Q4_0:
        case FA_VARIANT_Q4_0_SPLIT:
            return std::string{
                #include "flash_attn_f32_q4_0.cl.h"
            };
    }
    return {};
#else
    switch (v) {
        case FA_VARIANT_F16:           return read_file("flash_attn_f16.cl");
        case FA_VARIANT_F32:           return read_file("flash_attn_f32.cl");
        case FA_VARIANT_F32_F16:
        case FA_VARIANT_F32_F16_SPLIT: return read_file("flash_attn_f32_f16.cl");
        case FA_VARIANT_PRE:           return read_file("flash_attn_pre_f16.cl");
        case FA_VARIANT_Q8_0:
        case FA_VARIANT_Q8_0_SPLIT:    return read_file("flash_attn_f32_q8_0.cl");
        case FA_VARIANT_Q4_0:
        case FA_VARIANT_Q4_0_SPLIT:    return read_file("flash_attn_f32_q4_0.cl");
    }
    return {};
#endif
}

static std::string ggml_opencl_fa_compile_opts(ggml_backend_opencl_context * backend_ctx,
                                                const ggml_opencl_fa_dim * cfg,
                                                ggml_opencl_fa_variant variant) {
    std::string opts = backend_ctx->kernel_compile_opts +
        " -D DK=" + std::to_string(cfg->dk) +
        " -D DV=" + std::to_string(cfg->dv) +
        " -D BLOCK_M=" + std::to_string(cfg->bm) +
        " -D BLOCK_N=" + std::to_string(cfg->bn);

    if (backend_ctx->gpu_family == INTEL) {
        opts += " -D FA_SG=32";
    }

    const bool is_split = variant == FA_VARIANT_F32_F16_SPLIT ||
                          variant == FA_VARIANT_Q8_0_SPLIT    ||
                          variant == FA_VARIANT_Q4_0_SPLIT;
    if (is_split) {
        opts += " -D N_SPLIT=" + std::to_string(cfg->n_split);
    }
    // Shuffle define for the split tile paths AND the cluster-parallel decode
    // kernel (q1_vec_mq_split_c8) in the plain F32_F16 program. Without it the
    // c8 kernel is compiled out (HAS_SUBGROUP_SHUFFLE guard) and dispatch
    // falls back to the baseline mq_split.
    if ((is_split || variant == FA_VARIANT_F32_F16) && backend_ctx->has_subgroup_shuffle) {
        opts += backend_ctx->has_qcom_subgroup_shuffle
            ? " -D cl_qcom_subgroup_shuffle=1"
            : " -D cl_khr_subgroup_shuffle=1";
    }
    // X1E drops the explicit sub-group size pin on the c8 kernels, compiler
    // routes the fp16-heavy kernel to a slow variant with explicit subgroup size
    if (backend_ctx->adreno_gen == ADRENO_GPU_GEN::X1E) {
        opts += " -D FA_C8_NO_SG_PIN";
    }
    // Transposed K tile in local memory: the KV rows the QK loop walks together become
    // adjacent, so a group of them is ONE 128-bit local read instead of several narrow
    // ones. The QK loop is LDS-read-issue-bound (a wrong-math probe that kept every FMA/dp4a
    // but removed the LDS reads ran the kernel ~40% faster), so this is worth up to +26% on
    // fa=1 prefill. Output is bit-identical -- only the layout moves.
    //
    // DK <= 128 only. At DK=256 (gemma-3-4b) it measures 1-2% NEGATIVE and reproduces across
    // rounds; padding the row stride does not recover it, so the cause is not a simple bank
    // conflict and the wider tile does not want this layout.
    //
    // Default on within that gate; GGML_OPENCL_FA_K_LDS_T=0 restores the row-major tile.
    {
        const char * e = getenv("GGML_OPENCL_FA_K_LDS_T");
        if ((e == nullptr || e[0] != '0') && cfg->dk <= 128) {
            opts += " -D FA_K_LDS_T";
        }
    }
    return opts;
}

// only register when the kernel's required dispatch workgroup size is within
// the limit of the device's maximum workgroup size
static bool ggml_opencl_fa_kernel_fits_wg(ggml_backend_opencl_context * backend_ctx,
                                          cl_kernel kernel, size_t required_wg,
                                          const char * name, int dk, int dv) {
    if (kernel == NULL) { return false; }
    const size_t dev_max = backend_ctx->max_workgroup_size;
    if (dev_max < required_wg) {
        GGML_LOG_INFO("ggml_opencl: %s DK=%d DV=%d requires WG %zu > device max %zu; skipping registration (will fall back)\n",
                      name, dk, dv, required_wg, dev_max);
        return false;
    }
    size_t kwg = 0;
    cl_int err = clGetKernelWorkGroupInfo(kernel, backend_ctx->device,
                                          CL_KERNEL_WORK_GROUP_SIZE,
                                          sizeof(kwg), &kwg, NULL);
    if (err != CL_SUCCESS) {
        GGML_LOG_INFO("ggml_opencl: clGetKernelWorkGroupInfo failed for %s DK=%d DV=%d (err=%d); skipping registration\n",
                      name, dk, dv, err);
        return false;
    }
    if (kwg < required_wg) {
        GGML_LOG_INFO("ggml_opencl: %s DK=%d DV=%d per-kernel max %zu < required %zu; skipping registration (will fall back)\n",
                      name, dk, dv, kwg, required_wg);
        return false;
    }
    return true;
}

// Log private memory for an FA kernel. Enable via `GGML_OPENCL_FA_LOG_SPILL=1`.
// On Adreno non-zero private_mem means spilling to global memory due to resource
// constraint and usually causes performance degradation.
// (per-work-item, no cache locality) — a strong signal to pick a config
// with smaller per-thread state (e.g. larger N_SPLIT).
static void ggml_opencl_log_fa_kernel_spill(ggml_backend_opencl_context * backend_ctx,
                                            cl_kernel kernel, const char * name, int dk, int dv) {
    static const bool enabled = []{
        const char * e = std::getenv("GGML_OPENCL_FA_LOG_SPILL");
        return e && e[0] && e[0] != '0';
    }();

    if (!enabled || kernel == nullptr) {
        return;
    }

    cl_ulong priv_mem = 0;
    if (clGetKernelWorkGroupInfo(kernel, backend_ctx->device, CL_KERNEL_PRIVATE_MEM_SIZE,
                                 sizeof(priv_mem), &priv_mem, NULL) == CL_SUCCESS) {
        const char * tag = priv_mem > 0 ? "SPILL" : "ok";
        GGML_LOG_INFO("ggml_opencl: [%s] %s DK=%d DV=%d private_mem=%llu bytes\n",
                tag, name, dk, dv, (unsigned long long) priv_mem);
    }
}

static void ggml_opencl_ensure_fa_pre_kernels(ggml_backend_opencl_context * backend_ctx, int dk, int dv) {
    const std::pair<int, int> dk_dv = {dk, dv};

    const ggml_opencl_fa_dim * cfg = nullptr;
    for (const auto & d : g_opencl_fa_dims) {
        if (d.dk == dk && d.dv == dv) {
            cfg = &d; break;
        }
    }

    if (cfg == nullptr) {
        GGML_ABORT("ggml_opencl: no flash_attn config for DK=%d DV=%d", dk, dv);
    }

    // BM-tile metadata is consumed by the prefill dispatch (n_q_blocks / wg
    // sizing) regardless of whether the prepass kernels are needed for this
    // n_kv — set it unconditionally
    backend_ctx->fa.f32_f16_bm[{dk, dv}]      = cfg->bm;
    backend_ctx->fa.f32_f16_bn[{dk, dv}]      = cfg->bn;
    backend_ctx->fa.f32_f16_wg_size[{dk, dv}] = cfg->bm;
    backend_ctx->fa.bm[{dk, dv}]              = cfg->bm;
    backend_ctx->fa.bn[{dk, dv}]              = cfg->bn;

    if (backend_ctx->fa.kv_pad_f16.count(dk_dv) > 0) { return; }

    GGML_LOG_INFO("ggml_opencl: lazy-compiling flash_attn prepass for DK=%d DV=%d\n", dk, dv);
    cl_int err;
    const std::string src  = ggml_opencl_fa_kernel_src(FA_VARIANT_PRE);
    const std::string opts = ggml_opencl_fa_compile_opts(backend_ctx, cfg, FA_VARIANT_PRE);
    // retry when kernel compile fails
    cl_program prog_pre_f16 = build_program_from_source_ex(
        backend_ctx->context, backend_ctx->device, src.c_str(), opts,
        /*fatal=*/false, "fa prepass f16", backend_ctx->queue);
    if (!prog_pre_f16) { return; }
    cl_kernel k_kv_pad_f16  = clCreateKernel(prog_pre_f16, "flash_attn_kv_pad_f16",   &err);
    if (err != CL_SUCCESS) { clReleaseProgram(prog_pre_f16); return; }
    cl_kernel k_mask_pad_f16 = clCreateKernel(prog_pre_f16, "flash_attn_mask_pad_f16", &err);
    if (err != CL_SUCCESS) { clReleaseKernel(k_kv_pad_f16); clReleaseProgram(prog_pre_f16); return; }
    cl_kernel k_blk_f16     = clCreateKernel(prog_pre_f16, "flash_attn_blk_f16",      &err);
    if (err != CL_SUCCESS) { clReleaseKernel(k_kv_pad_f16); clReleaseKernel(k_mask_pad_f16); clReleaseProgram(prog_pre_f16); return; }
    backend_ctx->fa.kv_pad_f16[{dk, dv}]   = k_kv_pad_f16;
    backend_ctx->fa.mask_pad_f16[{dk, dv}] = k_mask_pad_f16;
    backend_ctx->fa.blk_f16[{dk, dv}]      = k_blk_f16;
    clReleaseProgram(prog_pre_f16);
}

// DK=512 prefill BM-tile
bool ggml_opencl_ensure_fa_f32_f16_prefill_512(ggml_backend_opencl_context * backend_ctx, bool split) {
    const int dk = 512, dv = 512;
    const std::pair<int, int> dk_dv = {dk, dv};
    auto & target = split ? backend_ctx->fa.f32_f16_split : backend_ctx->fa.f32_f16;
    if (target.count(dk_dv) > 0) { return true; }

    static bool failed[2] = { false, false };
    if (failed[split ? 1 : 0]) { return false; }

    const ggml_opencl_fa_dim * cfg = nullptr;
    for (const auto & d : g_opencl_fa_dims) {
        if (d.dk == dk && d.dv == dv) { cfg = &d; break; }
    }
    if (cfg == nullptr) { failed[split ? 1 : 0] = true; return false; }
    if (split && cfg->n_split <= 1) { failed[1] = true; return false; }

    const ggml_opencl_fa_variant variant = split ? FA_VARIANT_F32_F16_SPLIT : FA_VARIANT_F32_F16;
    std::string opts = ggml_opencl_fa_compile_opts(backend_ctx, cfg, variant) + " -D FA_PREFILL_ONLY";
    cl_program prog = build_program_from_source_ex(
        backend_ctx->context, backend_ctx->device,
        ggml_opencl_fa_kernel_src(FA_VARIANT_F32_F16).c_str(), opts,
        /*fatal=*/false, split ? "fa f32_f16 prefill512 split" : "fa f32_f16 prefill512",
        backend_ctx->queue);
    if (!prog) { failed[split ? 1 : 0] = true; return false; }

    cl_int err;
    cl_kernel k = clCreateKernel(prog, "flash_attn_f32_f16", &err);
    if (err != CL_SUCCESS) { clReleaseProgram(prog); failed[split ? 1 : 0] = true; return false; }
    target[dk_dv] = k;
    if (split) {
        backend_ctx->fa.f32_f16_split_wg_size[dk_dv]       = cfg->bm * cfg->n_split;
        backend_ctx->fa.f32_f16_split_nkv_threshold[dk_dv] = cfg->nkv_split_threshold;
    }
    ggml_opencl_log_fa_kernel_spill(backend_ctx, k,
        split ? "flash_attn_f32_f16 (prefill512 split)" : "flash_attn_f32_f16 (prefill512)", dk, dv);
    clReleaseProgram(prog);

    // determine whether to use the K-image variant of the split tile
    static const char * pkimg_build_env = getenv("GGML_OPENCL_FA_PREFILL_K_IMG");
    const bool pkimg_build = (pkimg_build_env != NULL) && (pkimg_build_env[0] != '0');
    if (split && pkimg_build && backend_ctx->fa.f32_f16_split_k_img.count(dk_dv) == 0) {
        std::string opts_img = ggml_opencl_fa_compile_opts(backend_ctx, cfg, variant) +
            " -D FA_PREFILL_ONLY -D FA_K_IMG -D FA_TILE_NAME=flash_attn_f32_f16_k_img";
        cl_program prog_img = build_program_from_source_ex(
            backend_ctx->context, backend_ctx->device,
            ggml_opencl_fa_kernel_src(FA_VARIANT_F32_F16).c_str(), opts_img,
            /*fatal=*/false, "fa f32_f16 prefill512 split k_img", backend_ctx->queue);
        if (prog_img) {
            cl_int err_img;
            cl_kernel k_img = clCreateKernel(prog_img, "flash_attn_f32_f16_k_img", &err_img);
            if (err_img == CL_SUCCESS) {
                backend_ctx->fa.f32_f16_split_k_img[dk_dv] = k_img;
                ggml_opencl_log_fa_kernel_spill(backend_ctx, k_img,
                    "flash_attn_f32_f16 (prefill512 split k_img)", dk, dv);
            }
            clReleaseProgram(prog_img);
        }
    }
    return true;
}

// Compile one (variant, dk, dv); memoised. false = compiler rejected.
static bool ggml_opencl_ensure_fa_variant(ggml_backend_opencl_context * backend_ctx, int dk, int dv, ggml_opencl_fa_variant variant) {
    const std::pair<int, int> dk_dv = {dk, dv};

    const ggml_opencl_fa_dim * cfg = nullptr;
    for (const auto & d : g_opencl_fa_dims) {
        if (d.dk == dk && d.dv == dv) {
            cfg = &d; break;
        }
    }
    if (cfg == nullptr) {
        return false;
    }

    // if a variant has already been compiled
    switch (variant) {
        case FA_VARIANT_F16: {
            if (backend_ctx->fa.f16.count(dk_dv)) {
                return true;
            }
            break;
        }
        case FA_VARIANT_F32: {
            if (backend_ctx->fa.f32.count(dk_dv)) {
                return true;
            }
            break;
        }
        case FA_VARIANT_F32_F16: {
            // The DK=512 decode-only program does not create the f32_f16
            // prefill kernel; check the q1 kernel instead so that repeated
            // calls return a consistent result.
            const bool decode_only = (dk == 512);
            if (decode_only ? (backend_ctx->fa.f32_f16_q1.count(dk_dv) > 0)
                            : (backend_ctx->fa.f32_f16.count(dk_dv)    > 0)) {
                return true;
            }
            break;
        }
        case FA_VARIANT_Q8_0: {
            if (backend_ctx->fa.f32_q8_0.count(dk_dv)) {
                return true;
            }
            break;
        }
        case FA_VARIANT_Q4_0: {
            if (backend_ctx->fa.f32_q4_0.count(dk_dv)) {
                return true;
            }
            break;
        }
        case FA_VARIANT_F32_F16_SPLIT: {
            if (backend_ctx->fa.f32_f16_split.count(dk_dv)) {
                return true;
            }
            break;
        }
        case FA_VARIANT_Q8_0_SPLIT: {
            if (backend_ctx->fa.f32_q8_0_split.count(dk_dv)) {
                return true;
            }
            break;
        }
        case FA_VARIANT_Q4_0_SPLIT: {
            if (backend_ctx->fa.f32_q4_0_split.count(dk_dv)) {
                return true;
            }
            break;
        }
        case FA_VARIANT_PRE: {
            ggml_opencl_ensure_fa_pre_kernels(backend_ctx, dk, dv);
            return true;
        }
    }

    // not registered but attempted - meaning these kernels failed to compile
    const auto attempt_key = std::make_pair(variant, dk_dv);
    if (backend_ctx->fa.variant_attempted.count(attempt_key)) {
        return false;
    }
    backend_ctx->fa.variant_attempted.insert(attempt_key);

    const bool is_split = variant == FA_VARIANT_F32_F16_SPLIT ||
                          variant == FA_VARIANT_Q8_0_SPLIT    ||
                          variant == FA_VARIANT_Q4_0_SPLIT;
    const bool is_quant = variant == FA_VARIANT_Q8_0 || variant == FA_VARIANT_Q8_0_SPLIT ||
                          variant == FA_VARIANT_Q4_0 || variant == FA_VARIANT_Q4_0_SPLIT;
    if (is_quant && (dk % 32 != 0 || dv % 32 != 0)) {
        return false;
    }
    if (is_split && cfg->n_split <= 1) {
        return false;
    }
    if ((variant == FA_VARIANT_Q8_0_SPLIT || variant == FA_VARIANT_Q4_0_SPLIT) &&
        ((dk / 32) % cfg->n_split != 0 || (dv / 4) % cfg->n_split != 0)) {
        return false;
    }

    const std::string src = ggml_opencl_fa_kernel_src(variant);
    if (src.empty()) { return false; }
    std::string opts = ggml_opencl_fa_compile_opts(backend_ctx, cfg, variant);

    // bypass kernels for DK=512
    const bool fa_decode_only = (variant == FA_VARIANT_F32_F16 && dk == 512);
    if (fa_decode_only) {
        opts += " -D FA_DECODE_ONLY -D FA_DECODE_MINIMAL";
    }

    // c8 cluster width (GGML_OPENCL_FA_CL_C overrides): value = GQA4 cluster
    // width (kernel default 8); the g8 programs use 2x the value (default 16).
    // Wider clusters halve per-lane o_acc at the cost of position streams per
    // subgroup
    static const int fa_cl_c_env = []{
        const char * e = std::getenv("GGML_OPENCL_FA_CL_C");
        const int x = (e && e[0]) ? atoi(e) : 0;
        return (x == 8 || x == 16 || x == 32) ? x : 0;   // 0 = per-gen default
    }();
    // X2E needs 16 to keep per-lane o_acc at 128B (the compiler spills the
    // kernel-default width); X1E does not spill, but C=16 is still a measured
    // +28-30% DK128-GQA4 decode win there (X1-85, kv 4096/8192), neutral on
    // DK64 / GQA1 / quant-KV.
    const int fa_cl_c_gqa4 = fa_cl_c_env ? fa_cl_c_env
        : (backend_ctx->adreno_gen == ADRENO_GPU_GEN::X2E ||
           backend_ctx->adreno_gen == ADRENO_GPU_GEN::X1E ? 16 : 0);
    const std::string opts_cl_c_gqa4 = fa_cl_c_gqa4
        ? " -D FA_CL_C=" + std::to_string(fa_cl_c_gqa4) : std::string();
    const std::string fa_cl_c_g8_val = std::to_string(fa_cl_c_gqa4 ? fa_cl_c_gqa4 * 2 : 16);

    const char * tag = nullptr;
    switch (variant) {
        case FA_VARIANT_F16:             tag = "fa f16";             break;
        case FA_VARIANT_F32:             tag = "fa f32";             break;
        case FA_VARIANT_F32_F16:         tag = "fa f32_f16";         break;
        case FA_VARIANT_Q8_0:            tag = "fa q8_0";            break;
        case FA_VARIANT_Q4_0:            tag = "fa q4_0";            break;
        case FA_VARIANT_F32_F16_SPLIT:   tag = "fa f32_f16 split";   break;
        case FA_VARIANT_Q8_0_SPLIT:      tag = "fa q8_0 split";      break;
        case FA_VARIANT_Q4_0_SPLIT:      tag = "fa q4_0 split";      break;
        default: break;
    }
    cl_program prog = build_program_from_source_ex(
        backend_ctx->context, backend_ctx->device, src.c_str(), opts + opts_cl_c_gqa4,
        /*fatal=*/false, tag, backend_ctx->queue);
    if (!prog) { return false; }

    cl_int err;
    switch (variant) {
        case FA_VARIANT_F16: {
            cl_kernel k, kq1;
            CL_CHECK((k   = clCreateKernel(prog, "flash_attn_f16",    &err), err));
            CL_CHECK((kq1 = clCreateKernel(prog, "flash_attn_f16_q1", &err), err));
            backend_ctx->fa.f16[{dk, dv}]    = k;
            backend_ctx->fa.f16_q1[{dk, dv}] = kq1;
            break;
        }
        case FA_VARIANT_F32: {
            cl_kernel k, kq1;
            CL_CHECK((k   = clCreateKernel(prog, "flash_attn_f32",    &err), err));
            CL_CHECK((kq1 = clCreateKernel(prog, "flash_attn_f32_q1", &err), err));
            backend_ctx->fa.f32[{dk, dv}]    = k;
            backend_ctx->fa.f32_q1[{dk, dv}] = kq1;
            break;
        }
        case FA_VARIANT_F32_F16: {
            cl_kernel kq1;
            // BM-tile prefill kernel is excluded from the decode-only (DK=512)
            if (!fa_decode_only) {
                cl_kernel k;
                CL_CHECK((k = clCreateKernel(prog, "flash_attn_f32_f16", &err), err));
                backend_ctx->fa.f32_f16[{dk, dv}] = k;
                ggml_opencl_log_fa_kernel_spill(backend_ctx, k, "flash_attn_f32_f16", dk, dv);
            }
            CL_CHECK((kq1 = clCreateKernel(prog, "flash_attn_f32_f16_q1", &err), err));
            backend_ctx->fa.f32_f16_q1[{dk, dv}] = kq1;
            ggml_opencl_log_fa_kernel_spill(backend_ctx, kq1, "flash_attn_f32_f16_q1", dk, dv);
            cl_kernel k_split = clCreateKernel(prog, "flash_attn_f32_f16_q1_split", &err);
            if (err == CL_SUCCESS) {
                backend_ctx->fa.f32_f16_q1_split[{dk, dv}] = k_split;
                ggml_opencl_log_fa_kernel_spill(backend_ctx, k_split, "flash_attn_f32_f16_q1_split", dk, dv);
            }
            // q1_vec decode kernel (DV-split + subgroup reduce)
            cl_kernel k_q1_vec = clCreateKernel(prog, "flash_attn_f32_f16_q1_vec", &err);
            if (err == CL_SUCCESS) {
                if (ggml_opencl_fa_kernel_fits_wg(backend_ctx, k_q1_vec, 256,
                                                  "flash_attn_f32_f16_q1_vec", dk, dv)) {
                    backend_ctx->fa.f32_f16_q1_vec[{dk, dv}] = k_q1_vec;
                    ggml_opencl_log_fa_kernel_spill(backend_ctx, k_q1_vec, "flash_attn_f32_f16_q1_vec", dk, dv);
                } else {
                    clReleaseKernel(k_q1_vec);
                }
            }
            // KV-head-coalesced vec for high-GQA small models
            cl_kernel k_q1_vec_mq = clCreateKernel(prog, "flash_attn_f32_f16_q1_vec_mq", &err);
            if (err == CL_SUCCESS) {
                if (ggml_opencl_fa_kernel_fits_wg(backend_ctx, k_q1_vec_mq, 256,
                                                  "flash_attn_f32_f16_q1_vec_mq", dk, dv)) {
                    backend_ctx->fa.f32_f16_q1_vec_mq[{dk, dv}] = k_q1_vec_mq;
                    ggml_opencl_log_fa_kernel_spill(backend_ctx, k_q1_vec_mq, "flash_attn_f32_f16_q1_vec_mq", dk, dv);
                } else {
                    clReleaseKernel(k_q1_vec_mq);
                }
            }
            // KV-head-coalesced + flash-decoding split, reuses merge kernel
            cl_kernel k_q1_vec_mq_split = clCreateKernel(prog, "flash_attn_f32_f16_q1_vec_mq_split", &err);
            if (err == CL_SUCCESS) {
                if (ggml_opencl_fa_kernel_fits_wg(backend_ctx, k_q1_vec_mq_split, 256,
                                                  "flash_attn_f32_f16_q1_vec_mq_split", dk, dv)) {
                    backend_ctx->fa.f32_f16_q1_vec_mq_split[{dk, dv}] = k_q1_vec_mq_split;
                    ggml_opencl_log_fa_kernel_spill(backend_ctx, k_q1_vec_mq_split, "flash_attn_f32_f16_q1_vec_mq_split", dk, dv);
                } else {
                    clReleaseKernel(k_q1_vec_mq_split);
                }
            }
            // K-image variant of MQ_GQA=4 split
            cl_kernel k_q1_vec_mq_split_k_img = clCreateKernel(prog, "flash_attn_f32_f16_q1_vec_mq_split_k_img", &err);
            if (err == CL_SUCCESS) {
                if (ggml_opencl_fa_kernel_fits_wg(backend_ctx, k_q1_vec_mq_split_k_img, 256,
                                                  "flash_attn_f32_f16_q1_vec_mq_split_k_img", dk, dv)) {
                    backend_ctx->fa.f32_f16_q1_vec_mq_split_k_img[{dk, dv}] = k_q1_vec_mq_split_k_img;
                    ggml_opencl_log_fa_kernel_spill(backend_ctx, k_q1_vec_mq_split_k_img, "flash_attn_f32_f16_q1_vec_mq_split_k_img", dk, dv);
                } else {
                    clReleaseKernel(k_q1_vec_mq_split_k_img);
                }
            }
            // Cluster-parallel decode variant
            cl_kernel k_q1_vec_mq_split_c8 = clCreateKernel(prog, "flash_attn_f32_f16_q1_vec_mq_split_c8", &err);
            if (err == CL_SUCCESS) {
                if (ggml_opencl_fa_kernel_fits_wg(backend_ctx, k_q1_vec_mq_split_c8, 256,
                                                  "flash_attn_f32_f16_q1_vec_mq_split_c8", dk, dv)) {
                    backend_ctx->fa.f32_f16_q1_vec_mq_split_c8[{dk, dv}] = k_q1_vec_mq_split_c8;
                    ggml_opencl_log_fa_kernel_spill(backend_ctx, k_q1_vec_mq_split_c8, "flash_attn_f32_f16_q1_vec_mq_split_c8", dk, dv);
                } else {
                    clReleaseKernel(k_q1_vec_mq_split_c8);
                }
            }
            cl_kernel k_merge = clCreateKernel(prog, "flash_attn_f32_merge", &err);
            if (err == CL_SUCCESS) {
                backend_ctx->fa.f32_merge[{dk, dv}] = k_merge;
            }
            // local-tile decode variant
            if (dk == 128 && dv == 128) {
                cl_kernel k_lt = clCreateKernel(prog, "flash_attn_f32_f16_q1_local_tile", &err);
                if (err == CL_SUCCESS) {
                    if (ggml_opencl_fa_kernel_fits_wg(backend_ctx, k_lt, 128,
                                                      "flash_attn_f32_f16_q1_local_tile", dk, dv)) {
                        backend_ctx->fa.f32_f16_q1_local_tile[{dk, dv}] = k_lt;
                        ggml_opencl_log_fa_kernel_spill(backend_ctx, k_lt, "flash_attn_f32_f16_q1_local_tile", dk, dv);
                    } else {
                        clReleaseKernel(k_lt);
                    }
                }
                // hybrid local-tile + MQ + FD-split
                cl_kernel k_lmq = clCreateKernel(prog, "flash_attn_f32_f16_q1_local_mq_split", &err);
                if (err == CL_SUCCESS) {
                    if (ggml_opencl_fa_kernel_fits_wg(backend_ctx, k_lmq, 64,
                                                      "flash_attn_f32_f16_q1_local_mq_split", dk, dv)) {
                        backend_ctx->fa.f32_f16_q1_local_mq_split[{dk, dv}] = k_lmq;
                        ggml_opencl_log_fa_kernel_spill(backend_ctx, k_lmq, "flash_attn_f32_f16_q1_local_mq_split", dk, dv);
                    } else {
                        clReleaseKernel(k_lmq);
                    }
                }
            }

            // second compile of the same source with -DMQ_GQA=8.
            // FA_MQ_ONLY keeps only the vec_mq kernels so that the program
            // compiles within the Adreno compiler's memory budget at DK>=256.
            // FA_CL_C for the g8 program: MQ_GQA=8 doubles the c8 kernel's
            // per-lane o_acc, so widen the cluster to keep the register
            // footprint inside the 192-thread WG cap (see fa_cl_c_gqa4 above
            // for the per-gen default).
            const std::string opts_g8 = opts + " -D MQ_GQA=8 -D MQ_NSG=3 -D MQ_NSG_SPLIT=3 -D FA_MQ_ONLY -D FA_CL_C=" + fa_cl_c_g8_val;
            cl_program prog_g8 = fa_decode_only ? nullptr : build_program_from_source_ex(
                backend_ctx->context, backend_ctx->device, src.c_str(), opts_g8,
                /*fatal=*/false, "fa f32_f16 MQ_GQA=8", backend_ctx->queue);
            if (prog_g8) {
                const size_t mq_g8_required_wg = 192;  // Q1_WG_SIZE(64) * MQ_NSG_SPLIT(3)
                cl_kernel k_q1_vec_mq_g8 = clCreateKernel(prog_g8, "flash_attn_f32_f16_q1_vec_mq", &err);
                if (err == CL_SUCCESS) {
                    if (ggml_opencl_fa_kernel_fits_wg(backend_ctx, k_q1_vec_mq_g8, mq_g8_required_wg,
                                                      "flash_attn_f32_f16_q1_vec_mq (g8)", dk, dv)) {
                        backend_ctx->fa.f32_f16_q1_vec_mq_g8[{dk, dv}] = k_q1_vec_mq_g8;
                        ggml_opencl_log_fa_kernel_spill(backend_ctx, k_q1_vec_mq_g8, "flash_attn_f32_f16_q1_vec_mq_g8", dk, dv);
                    } else {
                        clReleaseKernel(k_q1_vec_mq_g8);
                    }
                }
                cl_kernel k_q1_vec_mq_split_g8 = clCreateKernel(prog_g8, "flash_attn_f32_f16_q1_vec_mq_split", &err);
                if (err == CL_SUCCESS) {
                    if (ggml_opencl_fa_kernel_fits_wg(backend_ctx, k_q1_vec_mq_split_g8, mq_g8_required_wg,
                                                      "flash_attn_f32_f16_q1_vec_mq_split (g8)", dk, dv)) {
                        backend_ctx->fa.f32_f16_q1_vec_mq_split_g8[{dk, dv}] = k_q1_vec_mq_split_g8;
                        ggml_opencl_log_fa_kernel_spill(backend_ctx, k_q1_vec_mq_split_g8, "flash_attn_f32_f16_q1_vec_mq_split_g8", dk, dv);
                    } else {
                        clReleaseKernel(k_q1_vec_mq_split_g8);
                    }
                }
                // K-image variant
                cl_kernel k_q1_vec_mq_split_g8_k_img = clCreateKernel(prog_g8, "flash_attn_f32_f16_q1_vec_mq_split_k_img", &err);
                if (err == CL_SUCCESS) {
                    if (ggml_opencl_fa_kernel_fits_wg(backend_ctx, k_q1_vec_mq_split_g8_k_img, mq_g8_required_wg,
                                                      "flash_attn_f32_f16_q1_vec_mq_split_k_img (g8)", dk, dv)) {
                        backend_ctx->fa.f32_f16_q1_vec_mq_split_g8_k_img[{dk, dv}] = k_q1_vec_mq_split_g8_k_img;
                        ggml_opencl_log_fa_kernel_spill(backend_ctx, k_q1_vec_mq_split_g8_k_img, "flash_attn_f32_f16_q1_vec_mq_split_g8_k_img", dk, dv);
                    } else {
                        clReleaseKernel(k_q1_vec_mq_split_g8_k_img);
                    }
                }
                // Cluster-parallel decode, MQ_GQA=8 / FA_CL_C=16 specialization
                cl_kernel k_q1_vec_mq_split_g8_c8 = clCreateKernel(prog_g8, "flash_attn_f32_f16_q1_vec_mq_split_c8", &err);
                if (err == CL_SUCCESS) {
                    if (ggml_opencl_fa_kernel_fits_wg(backend_ctx, k_q1_vec_mq_split_g8_c8, mq_g8_required_wg,
                                                      "flash_attn_f32_f16_q1_vec_mq_split_c8 (g8)", dk, dv)) {
                        backend_ctx->fa.f32_f16_q1_vec_mq_split_g8_c8[{dk, dv}] = k_q1_vec_mq_split_g8_c8;
                        ggml_opencl_log_fa_kernel_spill(backend_ctx, k_q1_vec_mq_split_g8_c8, "flash_attn_f32_f16_q1_vec_mq_split_g8_c8", dk, dv);
                    } else {
                        clReleaseKernel(k_q1_vec_mq_split_g8_c8);
                    }
                }
                // hybrid local-tile + MQ_GQA=8
                if (dk == 128 && dv == 128) {
                    cl_kernel k_lmq_g8 = clCreateKernel(prog_g8, "flash_attn_f32_f16_q1_local_mq_split", &err);
                    if (err == CL_SUCCESS) {
                        if (ggml_opencl_fa_kernel_fits_wg(backend_ctx, k_lmq_g8, 64,
                                                          "flash_attn_f32_f16_q1_local_mq_split (g8)", dk, dv)) {
                            backend_ctx->fa.f32_f16_q1_local_mq_split_g8[{dk, dv}] = k_lmq_g8;
                            ggml_opencl_log_fa_kernel_spill(backend_ctx, k_lmq_g8, "flash_attn_f32_f16_q1_local_mq_split_g8", dk, dv);
                        } else {
                            clReleaseKernel(k_lmq_g8);
                        }
                    }
                }
                clReleaseProgram(prog_g8);
            }
            // NSG_SPLIT=2 programs for the cluster-parallel kernel: its register
            // footprint caps the per-kernel WG at 128 on X2 (< the stock 256/192
            // requirement), so it can never register from the stock programs.
            // With FA_CL_NCL position streams per subgroup, 2 subgroups still
            // carry 16 in-flight rows per WG (baseline WG has 4). FA_MQ_ONLY
            // keeps these compiles minimal; skipped when the stock program c8
            // registered (some other device) or shuffles are absent.
            if (!fa_decode_only && backend_ctx->has_subgroup_shuffle &&
                backend_ctx->fa.f32_f16_q1_vec_mq_split_c8.count({dk, dv}) == 0) {
                const std::string opts_c8_ns2 = opts + " -D FA_MQ_ONLY -D MQ_GQA=4 -D MQ_NSG=2 -D MQ_NSG_SPLIT=2" + opts_cl_c_gqa4;
                cl_program prog_c8 = build_program_from_source_ex(
                    backend_ctx->context, backend_ctx->device, src.c_str(), opts_c8_ns2,
                    /*fatal=*/false, "fa f32_f16 c8 NSG2", backend_ctx->queue);
                if (prog_c8) {
                    cl_kernel k_c8 = clCreateKernel(prog_c8, "flash_attn_f32_f16_q1_vec_mq_split_c8", &err);
                    if (err == CL_SUCCESS) {
                        // WG = MQ_NSG(2) × Q1_WG_SIZE(=FA_SG): 128 Adreno (64), 64 Intel (32).
                        const size_t c8_ns2_wg = backend_ctx->gpu_family == INTEL ? 64 : 128;
                        if (ggml_opencl_fa_kernel_fits_wg(backend_ctx, k_c8, c8_ns2_wg,
                                                          "flash_attn_f32_f16_q1_vec_mq_split_c8 (ns2)", dk, dv)) {
                            backend_ctx->fa.f32_f16_q1_vec_mq_split_c8_ns2[{dk, dv}] = k_c8;
                            ggml_opencl_log_fa_kernel_spill(backend_ctx, k_c8, "flash_attn_f32_f16_q1_vec_mq_split_c8_ns2", dk, dv);
                        } else {
                            clReleaseKernel(k_c8);
                        }
                    }
                    clReleaseProgram(prog_c8);
                }
            }
            // FA_CL_C=32 g8 program for the DK=DV=256 GQA=8
            if (!fa_decode_only && backend_ctx->has_subgroup_shuffle &&
                dk == 256 && dv == 256) {
                const std::string opts_g8_c32 = opts + " -D FA_MQ_ONLY -D MQ_GQA=8 -D MQ_NSG=2 -D MQ_NSG_SPLIT=2 -D FA_CL_C=32";
                cl_program prog_g8_c32 = build_program_from_source_ex(
                    backend_ctx->context, backend_ctx->device, src.c_str(), opts_g8_c32,
                    /*fatal=*/false, "fa f32_f16 c32 g8 d256 NSG2", backend_ctx->queue);
                if (prog_g8_c32) {
                    cl_kernel k_g8_c32 = clCreateKernel(prog_g8_c32, "flash_attn_f32_f16_q1_vec_mq_split_c8", &err);
                    if (err == CL_SUCCESS) {
                        if (ggml_opencl_fa_kernel_fits_wg(backend_ctx, k_g8_c32, 128,
                                                          "flash_attn_f32_f16_q1_vec_mq_split_c8 (g8 c32 d256)", dk, dv)) {
                            backend_ctx->fa.f32_f16_q1_vec_mq_split_g8_c32[{dk, dv}] = k_g8_c32;
                            ggml_opencl_log_fa_kernel_spill(backend_ctx, k_g8_c32, "flash_attn_f32_f16_q1_vec_mq_split_g8_c32", dk, dv);
                        } else {
                            clReleaseKernel(k_g8_c32);
                        }
                    }
                    clReleaseProgram(prog_g8_c32);
                }
            }
            if (!fa_decode_only && backend_ctx->has_subgroup_shuffle &&
                backend_ctx->fa.f32_f16_q1_vec_mq_split_g8_c8.count({dk, dv}) == 0) {
                const std::string opts_g8_c8_ns2 = opts + " -D FA_MQ_ONLY -D MQ_GQA=8 -D MQ_NSG=2 -D MQ_NSG_SPLIT=2 -D FA_CL_C=" + fa_cl_c_g8_val;
                cl_program prog_g8_c8 = build_program_from_source_ex(
                    backend_ctx->context, backend_ctx->device, src.c_str(), opts_g8_c8_ns2,
                    /*fatal=*/false, "fa f32_f16 c8 g8 NSG2", backend_ctx->queue);
                if (prog_g8_c8) {
                    cl_kernel k_g8_c8 = clCreateKernel(prog_g8_c8, "flash_attn_f32_f16_q1_vec_mq_split_c8", &err);
                    if (err == CL_SUCCESS) {
                        if (ggml_opencl_fa_kernel_fits_wg(backend_ctx, k_g8_c8, 128,
                                                          "flash_attn_f32_f16_q1_vec_mq_split_c8 (g8 ns2)", dk, dv)) {
                            backend_ctx->fa.f32_f16_q1_vec_mq_split_g8_c8_ns2[{dk, dv}] = k_g8_c8;
                            ggml_opencl_log_fa_kernel_spill(backend_ctx, k_g8_c8, "flash_attn_f32_f16_q1_vec_mq_split_g8_c8_ns2", dk, dv);
                        } else {
                            clReleaseKernel(k_g8_c8);
                        }
                    }
                    clReleaseProgram(prog_g8_c8);
                }
            }
            break;
        }
        case FA_VARIANT_Q8_0:
        case FA_VARIANT_Q4_0: {
            const bool is_q8 = variant == FA_VARIANT_Q8_0;
            const std::string base = is_q8 ? "flash_attn_f32_q8_0" : "flash_attn_f32_q4_0";
            const std::string name_q1       = base + "_q1";
            const std::string name_q1_split = base + "_q1_split";
            auto & m_q1       = is_q8 ? backend_ctx->fa.f32_q8_0_q1       : backend_ctx->fa.f32_q4_0_q1;
            auto & m_prefill  = is_q8 ? backend_ctx->fa.f32_q8_0          : backend_ctx->fa.f32_q4_0;
            auto & m_q1_split = is_q8 ? backend_ctx->fa.f32_q8_0_q1_split : backend_ctx->fa.f32_q4_0_q1_split;

            cl_kernel k, kq1;
            CL_CHECK((kq1 = clCreateKernel(prog, name_q1.c_str(), &err), err));
            CL_CHECK((k   = clCreateKernel(prog, base.c_str(),    &err), err));
            m_q1[{dk, dv}]      = kq1;
            m_prefill[{dk, dv}] = k;
            ggml_opencl_log_fa_kernel_spill(backend_ctx, kq1, name_q1.c_str(), dk, dv);
            ggml_opencl_log_fa_kernel_spill(backend_ctx, k,   base.c_str(),    dk, dv);
            cl_kernel k_split = clCreateKernel(prog, name_q1_split.c_str(), &err);
            if (err == CL_SUCCESS) {
                m_q1_split[{dk, dv}] = k_split;
                ggml_opencl_log_fa_kernel_spill(backend_ctx, k_split, name_q1_split.c_str(), dk, dv);
            }

            // DV-split decode variant (q1_vec)
            auto & m_q1_vec = is_q8 ? backend_ctx->fa.f32_q8_0_q1_vec : backend_ctx->fa.f32_q4_0_q1_vec;
            const std::string name_q1_vec = name_q1 + "_vec";
            cl_kernel k_q1_vec = clCreateKernel(prog, name_q1_vec.c_str(), &err);
            if (err == CL_SUCCESS) {
                if (ggml_opencl_fa_kernel_fits_wg(backend_ctx, k_q1_vec, 256,
                                                  name_q1_vec.c_str(), dk, dv)) {
                    m_q1_vec[{dk, dv}] = k_q1_vec;
                    ggml_opencl_log_fa_kernel_spill(backend_ctx, k_q1_vec, name_q1_vec.c_str(), dk, dv);
                } else {
                    clReleaseKernel(k_q1_vec);
                }
            }

            // KV-head-coalesced + flash-decoding split
            auto & m_mq_split = is_q8 ? backend_ctx->fa.f32_q8_0_q1_vec_mq_split
                                      : backend_ctx->fa.f32_q4_0_q1_vec_mq_split;
            const std::string name_mq_split = name_q1 + "_vec_mq_split";
            cl_kernel k_q1_vec_mq_split = clCreateKernel(prog, name_mq_split.c_str(), &err);
            if (err == CL_SUCCESS) {
                if (ggml_opencl_fa_kernel_fits_wg(backend_ctx, k_q1_vec_mq_split, 256,
                                                  name_mq_split.c_str(), dk, dv)) {
                    m_mq_split[{dk, dv}] = k_q1_vec_mq_split;
                    ggml_opencl_log_fa_kernel_spill(backend_ctx, k_q1_vec_mq_split, name_mq_split.c_str(), dk, dv);
                } else {
                    clReleaseKernel(k_q1_vec_mq_split);
                }
            }
            if (!backend_ctx->fa.f32_merge.count({dk, dv})) {
                cl_kernel k_merge = clCreateKernel(prog, "flash_attn_f32_merge", &err);
                if (err == CL_SUCCESS) {
                    backend_ctx->fa.f32_merge[{dk, dv}] = k_merge;
                }
            }
            // Second compile with MQ_GQA=8, MQ_NSG=3, MQ_NSG_SPLIT=3
            auto & m_mq_split_g8 = is_q8 ? backend_ctx->fa.f32_q8_0_q1_vec_mq_split_g8
                                         : backend_ctx->fa.f32_q4_0_q1_vec_mq_split_g8;
            const std::string opts_mq_g8 = opts + " -D MQ_GQA=8 -D MQ_NSG=3 -D MQ_NSG_SPLIT=3";
            cl_program prog_mq_g8 = build_program_from_source_ex(
                backend_ctx->context, backend_ctx->device, src.c_str(), opts_mq_g8,
                /*fatal=*/false, is_q8 ? "fa q8_0 MQ_GQA=8" : "fa q4_0 MQ_GQA=8",
                backend_ctx->queue);
            if (prog_mq_g8) {
                const size_t mq_g8_required_wg = 192;
                cl_kernel k_g8 = clCreateKernel(prog_mq_g8, name_mq_split.c_str(), &err);
                if (err == CL_SUCCESS) {
                    if (ggml_opencl_fa_kernel_fits_wg(backend_ctx, k_g8, mq_g8_required_wg,
                                                      name_mq_split.c_str(), dk, dv)) {
                        m_mq_split_g8[{dk, dv}] = k_g8;
                        ggml_opencl_log_fa_kernel_spill(backend_ctx, k_g8, name_mq_split.c_str(), dk, dv);
                    } else {
                        clReleaseKernel(k_g8);
                    }
                }
                clReleaseProgram(prog_mq_g8);
            }
            // GQA=4 cluster-parallel program (NSG_SPLIT=2 / WG=128)
            if (backend_ctx->has_subgroup_shuffle) {
                auto & m_c8_gqa4 = is_q8 ? backend_ctx->fa.f32_q8_0_q1_vec_mq_split_c8
                                         : backend_ctx->fa.f32_q4_0_q1_vec_mq_split_c8;
                const std::string name_c8_gqa4 = name_q1 + "_vec_mq_split_c8";
                const std::string opts_c8_gqa4 = opts + " -D MQ_GQA=4 -D MQ_NSG=2 -D MQ_NSG_SPLIT=2" + opts_cl_c_gqa4;
                cl_program prog_c8_gqa4 = build_program_from_source_ex(
                    backend_ctx->context, backend_ctx->device, src.c_str(), opts_c8_gqa4,
                    /*fatal=*/false, is_q8 ? "fa q8_0 c8 GQA4 NSG2" : "fa q4_0 c8 GQA4 NSG2",
                    backend_ctx->queue);
                if (prog_c8_gqa4) {
                    cl_kernel k_c8_gqa4 = clCreateKernel(prog_c8_gqa4, name_c8_gqa4.c_str(), &err);
                    if (err == CL_SUCCESS) {
                        if (ggml_opencl_fa_kernel_fits_wg(backend_ctx, k_c8_gqa4, 128,
                                                          name_c8_gqa4.c_str(), dk, dv)) {
                            m_c8_gqa4[{dk, dv}] = k_c8_gqa4;
                            ggml_opencl_log_fa_kernel_spill(backend_ctx, k_c8_gqa4, name_c8_gqa4.c_str(), dk, dv);
                        } else {
                            clReleaseKernel(k_c8_gqa4);
                        }
                    }
                    clReleaseProgram(prog_c8_gqa4);
                }
            }
            // Cluster-parallel q4_0 decode kernel
            if (!is_q8 && backend_ctx->has_subgroup_shuffle) {
                const std::string opts_c8 = opts + " -D MQ_GQA=8 -D MQ_NSG=2 -D MQ_NSG_SPLIT=2";
                cl_program prog_c8 = build_program_from_source_ex(
                    backend_ctx->context, backend_ctx->device, src.c_str(), opts_c8,
                    /*fatal=*/false, "fa q4_0 c8 NSG2", backend_ctx->queue);
                if (prog_c8) {
                    cl_kernel k_c8 = clCreateKernel(prog_c8, "flash_attn_f32_q4_0_q1_vec_mq_split_c8", &err);
                    if (err == CL_SUCCESS) {
                        if (ggml_opencl_fa_kernel_fits_wg(backend_ctx, k_c8, 128,
                                                          "flash_attn_f32_q4_0_q1_vec_mq_split_c8 (g8 ns2)", dk, dv)) {
                            backend_ctx->fa.f32_q4_0_q1_vec_mq_split_g8_c8[{dk, dv}] = k_c8;
                            ggml_opencl_log_fa_kernel_spill(backend_ctx, k_c8, "flash_attn_f32_q4_0_q1_vec_mq_split_g8_c8", dk, dv);
                        } else {
                            clReleaseKernel(k_c8);
                        }
                    }
                    clReleaseProgram(prog_c8);
                }
            }
            break;
        }
        case FA_VARIANT_F32_F16_SPLIT: {
            cl_kernel k;
            CL_CHECK((k = clCreateKernel(prog, "flash_attn_f32_f16", &err), err));
            backend_ctx->fa.f32_f16_split[{dk, dv}]               = k;
            backend_ctx->fa.f32_f16_split_wg_size[{dk, dv}]       = cfg->bm * cfg->n_split;
            backend_ctx->fa.f32_f16_split_nkv_threshold[{dk, dv}] = cfg->nkv_split_threshold;
            break;
        }
        case FA_VARIANT_Q8_0_SPLIT:
        case FA_VARIANT_Q4_0_SPLIT: {
            const bool is_q8 = variant == FA_VARIANT_Q8_0_SPLIT;
            cl_kernel k;
            CL_CHECK((k = clCreateKernel(prog, is_q8 ? "flash_attn_f32_q8_0" : "flash_attn_f32_q4_0", &err), err));
            auto & split        = is_q8 ? backend_ctx->fa.f32_q8_0_split               : backend_ctx->fa.f32_q4_0_split;
            auto & split_wg     = is_q8 ? backend_ctx->fa.f32_q8_0_split_wg_size        : backend_ctx->fa.f32_q4_0_split_wg_size;
            auto & split_bm     = is_q8 ? backend_ctx->fa.f32_q8_0_split_bm             : backend_ctx->fa.f32_q4_0_split_bm;
            auto & split_thresh = is_q8 ? backend_ctx->fa.f32_q8_0_split_nkv_threshold  : backend_ctx->fa.f32_q4_0_split_nkv_threshold;
            split[{dk, dv}]        = k;
            split_wg[{dk, dv}]     = cfg->bm * cfg->n_split;
            split_bm[{dk, dv}]     = cfg->bm;
            split_thresh[{dk, dv}] = 0;  // quant prefill: always split
            break;
        }
        default:
            break;
    }
    CL_CHECK(clReleaseProgram(prog));
    return true;
}

// Compile a quant FA split kernel with a hand-picked (BLOCK_M, N_SPLIT) that
// overrides the default fa_dims tuning, for the DK values where the default
// N_SPLIT is degenerate for quant prefill:
//   DK=256: default N_SPLIT=16 leaves DK/32=8 blocks -> 0 blocks/split.
//           Override N_SPLIT=8 (1 block/split), BLOCK_M=16.
//   DK=96 : DK/32 = 3 blocks, not divisible by the default N_SPLIT=2 ->
//           override N_SPLIT=3. BLOCK_M must be 16, not 32: the N_SPLIT=3
//           QK-partial reduction uses sub_group_shuffle, so all 3 split
//           threads of a query must land in one subgroup — WG_SIZE =
//           BLOCK_M*N_SPLIT must be <= the 64-lane Adreno subgroup (16*3=48).
static bool ggml_opencl_ensure_fa_quant_split_override(
        ggml_backend_opencl_context * backend_ctx,
        int dk, int dv, int quant_bm, int quant_n_split, bool is_q8_0
) {
    const std::pair<int, int> dk_dv = {dk, dv};
    if (is_q8_0 && backend_ctx->fa.f32_q8_0_split.count(dk_dv)) {
        return true;
    }
    if (!is_q8_0 && backend_ctx->fa.f32_q4_0_split.count(dk_dv)) {
        return true;
    }

    const ggml_opencl_fa_variant variant = is_q8_0 ? FA_VARIANT_Q8_0_SPLIT : FA_VARIANT_Q4_0_SPLIT;
    const auto attempt_key = std::make_pair(variant, dk_dv);
    if (backend_ctx->fa.variant_attempted.count(attempt_key)) {
        return false;
    }

    backend_ctx->fa.variant_attempted.insert(attempt_key);

    std::string shuffle_opts;
    if (backend_ctx->has_subgroup_shuffle) {
        shuffle_opts = backend_ctx->has_qcom_subgroup_shuffle
            ? " -D cl_qcom_subgroup_shuffle=1"
            : " -D cl_khr_subgroup_shuffle=1";
    }
    const ggml_opencl_fa_dim * cfg = nullptr;
    for (const auto & d : g_opencl_fa_dims) {
        if (d.dk == dk && d.dv == dv) {
            cfg = &d; break;
        }
    }
    if (cfg == nullptr) {
        return false;
    }

    // BLK_PREPASS_BM is the prepass-kernel BLOCK_M, needed so the quant kernel
    // indexes the blk[] classification buffer correctly.
    std::string opts = backend_ctx->kernel_compile_opts + shuffle_opts +
        " -D DK=" + std::to_string(dk) +
        " -D DV=" + std::to_string(dv) +
        " -D BLOCK_M=" + std::to_string(quant_bm) +
        " -D BLOCK_N=" + std::to_string(cfg->bn) +
        " -D N_SPLIT=" + std::to_string(quant_n_split) +
        " -D BLK_PREPASS_BM=" + std::to_string(cfg->bm);

    const std::string src = ggml_opencl_fa_kernel_src(variant);
    if (src.empty()) {
        return false;
    }

    const std::string tag = std::string("fa ") + (is_q8_0 ? "q8_0" : "q4_0") +
        " split DK=" + std::to_string(dk);
    cl_program prog = build_program_from_source_ex(
        backend_ctx->context, backend_ctx->device, src.c_str(), opts,
        /*fatal=*/false, tag.c_str(), backend_ctx->queue);
    if (!prog) { return false; }
    cl_int err;
    cl_kernel k;
    if (is_q8_0) {
        CL_CHECK((k = clCreateKernel(prog, "flash_attn_f32_q8_0", &err), err));
        backend_ctx->fa.f32_q8_0_split[dk_dv]                = k;
        backend_ctx->fa.f32_q8_0_split_wg_size[dk_dv]        = quant_bm * quant_n_split;
        backend_ctx->fa.f32_q8_0_split_bm[dk_dv]             = quant_bm;
        backend_ctx->fa.f32_q8_0_split_nkv_threshold[dk_dv]  = 0;
    } else {
        CL_CHECK((k = clCreateKernel(prog, "flash_attn_f32_q4_0", &err), err));
        backend_ctx->fa.f32_q4_0_split[dk_dv]                = k;
        backend_ctx->fa.f32_q4_0_split_wg_size[dk_dv]        = quant_bm * quant_n_split;
        backend_ctx->fa.f32_q4_0_split_bm[dk_dv]             = quant_bm;
        backend_ctx->fa.f32_q4_0_split_nkv_threshold[dk_dv]  = 0;
    }
    CL_CHECK(clReleaseProgram(prog));
    return true;
}

// Resolve the source buffer + strides for an FA KV tensor: keep the
// caller-supplied AoS buffer if non-NULL, else fall back to tensor->extra.
static void ggml_cl_flash_attn_resolve_src(
        const ggml_tensor * tensor,
        cl_mem &   buf,
        cl_ulong & offset,
        cl_ulong & nb1,
        cl_ulong & nb2,
        cl_ulong & nb3) {
    if (buf != NULL) {
        return;
    }
    ggml_tensor_extra_cl * extra = (ggml_tensor_extra_cl *) tensor->extra;
    GGML_ASSERT(extra && extra->data_device);
    buf    = extra->data_device;
    offset = extra->offset + tensor->view_offs;
    nb1    = tensor->nb[1];
    nb2    = tensor->nb[2];
    nb3    = tensor->nb[3];
}

// Read a (possibly strided-view) tensor from device into a tight host buffer.
// dim 0 is always tight; a strided view is gathered row-by-row.
static void ggml_cl_flash_attn_read_tensor_host(
        ggml_backend_opencl_context * backend_ctx,
        const ggml_tensor *           tensor,
        cl_mem src_buffer, cl_ulong src_offset,
        cl_ulong src_nb1, cl_ulong src_nb2, cl_ulong src_nb3,
        size_t row_bytes, void * dst, size_t total_bytes
) {
    const bool contiguous_layout =
        src_nb1 == row_bytes &&
        src_nb2 == row_bytes * (cl_ulong) tensor->ne[1] &&
        src_nb3 == src_nb2   * (cl_ulong) tensor->ne[2];

    if (contiguous_layout) {
        CL_CHECK(clEnqueueReadBuffer(backend_ctx->queue, src_buffer, CL_TRUE,
                                     src_offset, total_bytes, dst, 0, NULL, NULL));
        return;
    }

    size_t dst_off = 0;
    for (int64_t i3 = 0; i3 < tensor->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < tensor->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < tensor->ne[1]; ++i1) {
                const cl_ulong row_src_off = src_offset +
                    (cl_ulong) i3 * src_nb3 +
                    (cl_ulong) i2 * src_nb2 +
                    (cl_ulong) i1 * src_nb1;
                CL_CHECK(clEnqueueReadBuffer(backend_ctx->queue, src_buffer, CL_TRUE,
                                             row_src_off, row_bytes,
                                             (uint8_t *) dst + dst_off, 0, NULL, NULL));
                dst_off += row_bytes;
            }
        }
    }
    GGML_ASSERT(dst_off == total_bytes);
}

// Rebuild AoS q8_0/q4_0 bytes from a SoA tensor into a temp buffer.
// Returns false if the tensor is not SoA-quantised (already AoS).
static bool ggml_cl_flash_attn_reconstruct_aos(
        ggml_backend_opencl_context *         backend_ctx,
        const ggml_tensor *                   tensor,
        ggml_cl_flash_attn_temp_buffer &      temp,
        cl_mem &                              out_buf,
        cl_ulong &                            out_offset,
        cl_ulong &                            out_nb1,
        cl_ulong &                            out_nb2,
        cl_ulong &                            out_nb3
) {
    if (tensor == nullptr) {
        return false;
    }
    const bool is_q8_0 = tensor->type == GGML_TYPE_Q8_0 && ggml_cl_is_q8_0_soa(tensor);
    const bool is_q4_0 = tensor->type == GGML_TYPE_Q4_0 && ggml_cl_is_q4_0_soa(tensor);
    if (!is_q8_0 && !is_q4_0) {
        return false;
    }

    // For views, SoA extra is on view_src (view->extra is pre-SoA).
    // Noshuffle layout only applies to 2D weights, as determined by `use_adreno_kernels`,
    // where ne2 == 1 and ne3 == 1 -- these are never FA inputs.
    // Therefore, we use `restore_block_qk_0` kernels, not `restore_block_qk_0_noshuffle`.
    const ggml_tensor * soa_src = tensor->view_src ? tensor->view_src : tensor;
    cl_mem extra_q = NULL;
    cl_mem extra_d = NULL;
    if (is_q8_0) {
        auto * e = (ggml_tensor_extra_cl_q8_0 *) soa_src->extra;
        GGML_ASSERT(e && e->q && e->d);
        extra_q = e->q;
        extra_d = e->d;
    } else {
        auto * e = (ggml_tensor_extra_cl_q4_0 *) soa_src->extra;
        GGML_ASSERT(e && e->q && e->d);
        extra_q = e->q;
        extra_d = e->d;
    }

    // Reconstruct the whole parent; view offsets then work naturally.
    const size_t parent_nbytes = ggml_nbytes(soa_src);
    cl_int err;
    temp.data = clCreateBuffer(backend_ctx->context, CL_MEM_READ_WRITE, parent_nbytes, NULL, &err);
    CL_CHECK(err);

    cl_kernel kernel = is_q8_0 ? backend_ctx->repack.kernel_restore_block_q8_0
                               : backend_ctx->repack.kernel_restore_block_q4_0;
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &extra_q));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra_d));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &temp.data));

    const size_t n_blocks = (size_t) ggml_nelements(soa_src) / ggml_blck_size(soa_src->type);
    size_t global_work_size[] = { n_blocks, 1, 1 };
    size_t local_work_size[]  = { 1, 1, 1 };
    CL_CHECK(clEnqueueNDRangeKernel(backend_ctx->queue, kernel, 3, NULL,
                                    global_work_size, local_work_size, 0, NULL, NULL));

    out_buf    = temp.data;
    out_offset = tensor->view_offs;
    out_nb1    = tensor->nb[1];
    out_nb2    = tensor->nb[2];
    out_nb3    = tensor->nb[3];
    return true;
}

// GPU dequant of a contiguous q4_0/q8_0 KV tensor to f16/f32. Caller supplies
// src_buf when reconstructing from SoA. Returns false for non-contig layouts
// (the kernel indexes blocks tightly within ne[0]) so the caller can fall back
// to the host path.
static bool ggml_cl_flash_attn_dequant_kv_gpu(
        ggml_backend_opencl_context *    backend_ctx,
        const ggml_tensor *              tensor,
        ggml_type                        target_type,
        cl_mem                           in_src_buf,
        cl_ulong                         in_src_offset,
        cl_ulong                         in_src_nb1,
        cl_ulong                         in_src_nb2,
        cl_ulong                         in_src_nb3,
        ggml_cl_flash_attn_temp_buffer & temp,
        cl_mem &                         out_buf,
        cl_ulong &                       out_offset,
        cl_ulong &                       out_nb1,
        cl_ulong &                       out_nb2,
        cl_ulong &                       out_nb3
) {
    GGML_ASSERT(tensor->type == GGML_TYPE_Q8_0 || tensor->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(target_type == GGML_TYPE_F16 || target_type == GGML_TYPE_F32);

    const bool is_q8_0 = tensor->type == GGML_TYPE_Q8_0;

    cl_mem   src_buf    = in_src_buf;
    cl_ulong src_offset = in_src_offset;
    cl_ulong src_nb1    = in_src_nb1;
    cl_ulong src_nb2    = in_src_nb2;
    cl_ulong src_nb3    = in_src_nb3;
    ggml_cl_flash_attn_resolve_src(tensor, src_buf, src_offset, src_nb1, src_nb2, src_nb3);

    if (tensor->nb[0] != (cl_ulong) ggml_type_size(tensor->type)) {
        return false;
    }

    const size_t n_blocks = (size_t) ggml_nelements(tensor) / 32; // block size is 32
    const size_t elem_size = ggml_type_size(target_type);
    const size_t out_bytes = n_blocks * 32 * elem_size;
    const cl_int nblk0_arg = (cl_int) (tensor->ne[0] / 32);
    const cl_int ne1_arg   = (cl_int) tensor->ne[1];
    const cl_int ne2_arg   = (cl_int) tensor->ne[2];
    const cl_int ne3_arg   = (cl_int) tensor->ne[3];

    cl_int err;
    temp.data = clCreateBuffer(backend_ctx->context, CL_MEM_READ_WRITE, out_bytes, NULL, &err);
    CL_CHECK(err);

    cl_kernel kernel;
    if (target_type == GGML_TYPE_F16) {
        kernel = is_q8_0 ? backend_ctx->repack.kernel_dequant_q8_0_f16_view_aos
                         : backend_ctx->repack.kernel_dequant_q4_0_f16_view_aos;
    } else {
        kernel = is_q8_0 ? backend_ctx->repack.kernel_dequant_q8_0_f32_view_aos
                         : backend_ctx->repack.kernel_dequant_q4_0_f32_view_aos;
    }

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &src_buf));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &src_offset));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_ulong), &src_nb1));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &src_nb2));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_ulong), &src_nb3));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_int),   &nblk0_arg));
    CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_int),   &ne1_arg));
    CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_int),   &ne2_arg));
    CL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_int),   &ne3_arg));
    CL_CHECK(clSetKernelArg(kernel, 9, sizeof(cl_mem),   &temp.data));

    size_t global_ws[3] = { (size_t) nblk0_arg, (size_t) ne1_arg, (size_t) ne2_arg * (size_t) ne3_arg };
    CL_CHECK(clEnqueueNDRangeKernel(backend_ctx->queue, kernel, 3, NULL,
                                    global_ws, NULL, 0, NULL, NULL));

    out_buf    = temp.data;
    out_offset = 0;
    out_nb1    = (cl_ulong) tensor->ne[0] * elem_size;
    out_nb2    = out_nb1 * (cl_ulong) tensor->ne[1];
    out_nb3    = out_nb2 * (cl_ulong) tensor->ne[2];
    return true;
}

static bool ggml_cl_flash_attn_prepare_quantized_tensor(
        ggml_backend_opencl_context *         backend_ctx,
        const ggml_tensor *                   tensor,
        ggml_type                             target_type,
        ggml_cl_flash_attn_temp_buffer &      temp,
        cl_mem &                              data_device,
        cl_ulong &                            offset,
        cl_ulong &                            nb1,
        cl_ulong &                            nb2,
        cl_ulong &                            nb3
) {
    if (!ggml_is_quantized(tensor->type)) {
        return false;
    }

    // Caller-supplied AoS buffer wins over tensor->extra when present.
    cl_mem   src_buffer = data_device;
    cl_ulong src_offset = offset;
    cl_ulong src_nb1    = nb1;
    cl_ulong src_nb2    = nb2;
    cl_ulong src_nb3    = nb3;
    ggml_cl_flash_attn_resolve_src(tensor, src_buffer, src_offset, src_nb1, src_nb2, src_nb3);

    const int64_t n = ggml_nelements(tensor);
    const size_t  row_bytes = (size_t) (tensor->ne[0] / ggml_blck_size(tensor->type)) * ggml_type_size(tensor->type);
    // tight-packed byte count (ggml_nbytes includes stride gaps).
    const size_t  total_bytes = (size_t) (n / ggml_blck_size(tensor->type)) * ggml_type_size(tensor->type);
    std::vector<uint8_t> host_quant(total_bytes);

    sync_with_other_backends(backend_ctx);
    ggml_cl_flash_attn_read_tensor_host(backend_ctx, tensor, src_buffer, src_offset,
                                        src_nb1, src_nb2, src_nb3,
                                        row_bytes, host_quant.data(), total_bytes);

    std::vector<float> host_f32(n);
    ggml_get_type_traits(tensor->type)->to_float(host_quant.data(), host_f32.data(), n);

    const size_t bytes_per_elem = ggml_type_size(target_type);
    const size_t buffer_size = (size_t) n * bytes_per_elem;

    std::vector<uint8_t> host_linear(buffer_size);
    if (target_type == GGML_TYPE_F32) {
        memcpy(host_linear.data(), host_f32.data(), buffer_size);
    } else {
        GGML_ASSERT(target_type == GGML_TYPE_F16);
        ggml_fp32_to_fp16_row(host_f32.data(), (ggml_fp16_t *) host_linear.data(), n);
    }

    cl_int err;
    temp.data = clCreateBuffer(backend_ctx->context, CL_MEM_READ_WRITE, buffer_size, NULL, &err);
    CL_CHECK(err);
    CL_CHECK(clEnqueueWriteBuffer(backend_ctx->queue, temp.data, CL_TRUE, 0, buffer_size, host_linear.data(), 0, NULL, NULL));

    data_device = temp.data;
    offset = 0;
    nb1 = (cl_ulong) (tensor->ne[0] * bytes_per_elem);
    nb2 = (cl_ulong) (tensor->ne[1] * nb1);
    nb3 = (cl_ulong) (tensor->ne[2] * nb2);

    static bool warned = false;
    if (!warned) {
        GGML_LOG_WARN("ggml_opencl: OpenCL flash attention dequantizes GPU-resident quantized KV cache into temporary linear buffers; performance may be poor\n");
        warned = true;
    }

    return true;
}

// Host-side F16 -> F32 for the asymmetric-KV F32 fallback path.
static bool ggml_cl_flash_attn_convert_f16_to_f32(
        ggml_backend_opencl_context *         backend_ctx,
        const ggml_tensor *                   tensor,
        ggml_cl_flash_attn_temp_buffer &      temp,
        cl_mem &                              data_device,
        cl_ulong &                            offset,
        cl_ulong &                            nb1,
        cl_ulong &                            nb2,
        cl_ulong &                            nb3
) {
    if (tensor->type != GGML_TYPE_F16) {
        return false;
    }

    cl_mem   src_buffer = data_device;
    cl_ulong src_offset = offset;
    cl_ulong src_nb1    = nb1;
    cl_ulong src_nb2    = nb2;
    cl_ulong src_nb3    = nb3;
    ggml_cl_flash_attn_resolve_src(tensor, src_buffer, src_offset, src_nb1, src_nb2, src_nb3);

    const int64_t n = ggml_nelements(tensor);
    const size_t  row_bytes = (size_t) tensor->ne[0] * sizeof(ggml_fp16_t);
    const size_t  total_bytes = (size_t) n * sizeof(ggml_fp16_t);
    std::vector<uint8_t> host_f16(total_bytes);

    sync_with_other_backends(backend_ctx);
    ggml_cl_flash_attn_read_tensor_host(backend_ctx, tensor, src_buffer, src_offset,
                                        src_nb1, src_nb2, src_nb3,
                                        row_bytes, host_f16.data(), total_bytes);

    std::vector<float> host_f32(n);
    ggml_fp16_to_fp32_row((const ggml_fp16_t *) host_f16.data(), host_f32.data(), n);

    const size_t f32_bytes = (size_t) n * sizeof(float);
    cl_int err;
    temp.data = clCreateBuffer(backend_ctx->context, CL_MEM_READ_WRITE, f32_bytes, NULL, &err);
    CL_CHECK(err);
    CL_CHECK(clEnqueueWriteBuffer(backend_ctx->queue, temp.data, CL_TRUE, 0,
                                  f32_bytes, host_f32.data(), 0, NULL, NULL));

    data_device = temp.data;
    offset = 0;
    nb1 = (cl_ulong) (tensor->ne[0] * sizeof(float));
    nb2 = (cl_ulong) (tensor->ne[1] * nb1);
    nb3 = (cl_ulong) (tensor->ne[2] * nb2);

    static bool warned = false;
    if (!warned) {
        GGML_LOG_WARN("ggml_opencl: OpenCL flash attention asymmetric KV converts an F16 cache to F32 host-side; performance may be poor\n");
        warned = true;
    }

    return true;
}

// Flash-Decoding (K-split) dispatch thresholds. FD fires for non-causal
// attention with n_kv >= FD_MIN_N_KV and d_head <= FD_MAX_DK; the KV range is
// split into ~n_kv/FD_KV_PER_SPLIT partials, clamped to [FD_MIN_SPLITS,
// FD_MAX_SPLITS]. Multi-query FD is restricted to small heads
// (d_head <= FD_MAX_DK_MULTI) and capped at FD_MAX_N_Q_MULTI queries.
static constexpr int FD_MIN_N_KV      = 2048;
static constexpr int FD_KV_PER_SPLIT  = 2048;
// f16 KV decode wants more splits than the 2048 default; quantized KV keeps 2048.
static constexpr int FD_KV_PER_SPLIT_F16 = 512;
static constexpr int FD_MIN_SPLITS    = 2;
static constexpr int FD_MAX_SPLITS    = 16;
static constexpr int FD_MAX_DK        = 128;
static constexpr int FD_MAX_DK_MULTI  = 64;
static constexpr int FD_MAX_N_Q_MULTI = 8;
// MQ FD split-groups have few subgroups (MQ_NSG_SPLIT), so use a smaller
// kv_per_split to keep the softmax recurrence short; non-MQ keeps FD_KV_PER_SPLIT.
static constexpr int FD_MQ_KV_PER_SPLIT = 256;
static constexpr int FD_MQ_MAX_SPLITS   = 128;

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
struct ggml_cl_adreno_xmem_attn_schedule {
    int qk_lws0 = 256;
    int qk_lws2 = 1;
    int softmax_reduce_lws0 = 256;
    int softmax_apply_lws0 = 64;
    int softmax_apply_lws2 = 4;
    int pv_lws0 = 64;
    int pv_lws2 = 4;
};

static inline size_t ggml_cl_round_up(size_t x, size_t a) {
    return ((x + a - 1) / a) * a;
}

static inline int ggml_cl_round_up_div(int x, int y) {
    return (x + y - 1) / y;
}

static inline void ggml_cl_set_arg_int4(cl_kernel kernel, cl_uint index, int x, int y, int z, int w) {
    struct { int x, y, z, w; } value { x, y, z, w };
    CL_CHECK(clSetKernelArg(kernel, index, sizeof(value), &value));
}

static cl_mem ggml_cl_make_image2d_half4(cl_context context, cl_mem_flags flags, size_t width, size_t height) {
    cl_int err = CL_SUCCESS;
    cl_image_format format = { CL_RGBA, CL_HALF_FLOAT };
    cl_image_desc desc = {};
    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    desc.image_width = width;
    desc.image_height = height;
    cl_mem image = clCreateImage(context, flags, &format, &desc, nullptr, &err);
    CL_CHECK(err);
    return image;
}

static cl_mem ggml_cl_make_image1d_buffer_half4(cl_context context, cl_mem_flags flags, size_t width, cl_mem backing_buffer) {
    cl_int err = CL_SUCCESS;
    cl_image_format format = { CL_RGBA, CL_HALF_FLOAT };
    cl_image_desc desc = {};
    desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
    desc.image_width = width;
    desc.buffer = backing_buffer;
    cl_mem image = clCreateImage(context, flags, &format, &desc, nullptr, &err);
    CL_CHECK(err);
    return image;
}

static void ggml_cl_release_mem(cl_mem & mem) {
    if (mem != nullptr) {
        CL_CHECK(clReleaseMemObject(mem));
        mem = nullptr;
    }
}

void ggml_cl_adreno_xmem_attn_release_scratch(ggml_backend_opencl_context * backend_ctx) {
    auto & s = backend_ctx->adreno_xmem_attn.scratch;
    ggml_cl_release_mem(s.q_img);
    ggml_cl_release_mem(s.k_img);
    ggml_cl_release_mem(s.v_img);
    ggml_cl_release_mem(s.out_img);
    ggml_cl_release_mem(s.k_transpose_img1d);
    ggml_cl_release_mem(s.k_transpose_buf);
    ggml_cl_release_mem(s.k_packed_buf);
    ggml_cl_release_mem(s.v_packed_buf);
    ggml_cl_release_mem(s.score_img1d);
    ggml_cl_release_mem(s.prob_img1d);
    ggml_cl_release_mem(s.score_buf);
    ggml_cl_release_mem(s.prob_buf);
    ggml_cl_release_mem(s.softmax_stats_img2d);
    ggml_cl_release_mem(s.xmem_qk);
    ggml_cl_release_mem(s.xmem_pv);
    s = {};
}

static ggml_cl_adreno_xmem_attn_schedule ggml_cl_adreno_xmem_attn_select_schedule(
        const ggml_backend_opencl_context * backend_ctx,
        int n_q,
        int n_kv,
        int heads_total,
        int q_width,
        int gqa_ratio) {
    const bool big_h = heads_total >= 8;
    ggml_cl_adreno_xmem_attn_schedule sched;

    if (gqa_ratio == 1) {
        if (n_q >= 512) { sched.qk_lws0 = 512; }
        else if (n_q >= 256) { sched.qk_lws0 = 128; }
        else { sched.qk_lws0 = 64; }
        sched.qk_lws2 = (big_h && n_q >= 512) ? 2 : 1;
    } else {
        if (q_width >= 2048) { sched.qk_lws0 = 512; }
        else if (q_width >= 256) { sched.qk_lws0 = 128; }
        else { sched.qk_lws0 = 64; }
        sched.qk_lws2 = MIN(8, (int) backend_ctx->max_workgroup_size / sched.qk_lws0);
    }

    if (n_kv >= 2048) { sched.softmax_reduce_lws0 = 1024; }
    else if (n_kv >= 512) { sched.softmax_reduce_lws0 = big_h ? 256 : 512; }
    else { sched.softmax_reduce_lws0 = 256; }

    if (n_kv < 256) { sched.softmax_apply_lws0 = 64; }
    else { sched.softmax_apply_lws0 = big_h ? 128 : 64; }
    sched.softmax_apply_lws2 = n_kv >= 512 ? 8 : 4;

    if (n_q < 256) { sched.pv_lws0 = 64; }
    else { sched.pv_lws0 = big_h ? 128 : 64; }
    sched.pv_lws2 = big_h ? 8 : (n_q <= 256 ? 8 : 4);

    const int max_wg = (int) backend_ctx->max_workgroup_size;
    auto fix = [&](int & l0, int & l2) {
        while (l0 * l2 > max_wg) {
            if (l2 > 1) { l2 /= 2; }
            else if (l0 > 32) { l0 /= 2; }
            else { break; }
        }
    };
    fix(sched.qk_lws0, sched.qk_lws2);
    fix(sched.softmax_apply_lws0, sched.softmax_apply_lws2);
    fix(sched.pv_lws0, sched.pv_lws2);
    while (sched.softmax_reduce_lws0 > max_wg) {
        sched.softmax_reduce_lws0 /= 2;
    }

    return sched;
}

static bool ggml_cl_adreno_xmem_attn_prepare(
        ggml_backend_opencl_context * backend_ctx,
        int n_q,
        int n_kv,
        int d_head_q,
        int d_head_v,
        int n_head,
        int n_head_kv,
        int n_batch) {
    auto & s = backend_ctx->adreno_xmem_attn.scratch;
    const int gqa_ratio = n_head / n_head_kv;
    const int q_width = n_q * gqa_ratio;
    const int kv_heads_total = n_head_kv * n_batch;
    const int n_kv_padded = (int) ggml_cl_round_up((size_t) n_kv, 32);
    if (s.q_img != nullptr &&
            s.n_q == n_q &&
            s.n_kv == n_kv &&
            s.n_kv_padded == n_kv_padded &&
            s.d_head_q == d_head_q &&
            s.d_head_v == d_head_v &&
            s.q_width == q_width &&
            s.kv_heads_total == kv_heads_total) {
        return true;
    }

    ggml_cl_adreno_xmem_attn_release_scratch(backend_ctx);

    const int qpack = d_head_q / 4;
    const int vpack = d_head_v / 4;
    const int npack = n_kv_padded / 4;
    const size_t q_img_h = (size_t) kv_heads_total * qpack;
    const size_t v_img_h = (size_t) kv_heads_total * vpack;

    s.q_img = ggml_cl_make_image2d_half4(backend_ctx->context, CL_MEM_READ_WRITE, (size_t) q_width, q_img_h);
    s.k_img = ggml_cl_make_image2d_half4(backend_ctx->context, CL_MEM_READ_WRITE, (size_t) n_kv_padded, q_img_h);
    s.v_img = ggml_cl_make_image2d_half4(backend_ctx->context, CL_MEM_READ_WRITE, (size_t) n_kv_padded, v_img_h);
    s.out_img = ggml_cl_make_image2d_half4(backend_ctx->context, CL_MEM_READ_WRITE, (size_t) q_width, v_img_h);

    const size_t k_transpose_half4_elems = (size_t) npack * kv_heads_total * d_head_q;
    s.k_transpose_buf = clCreateBuffer(backend_ctx->context, CL_MEM_READ_WRITE, k_transpose_half4_elems * sizeof(uint16_t) * 4, nullptr, nullptr);
    GGML_ASSERT(s.k_transpose_buf != nullptr);
    s.k_transpose_img1d = ggml_cl_make_image1d_buffer_half4(backend_ctx->context, CL_MEM_READ_ONLY, k_transpose_half4_elems, s.k_transpose_buf);

    const size_t k_groups16 = (size_t) ggml_cl_round_up_div(kv_heads_total * d_head_q, 16);
    const size_t v_groups16 = (size_t) ggml_cl_round_up_div(kv_heads_total * d_head_v, 16);
    const size_t k_packed_half4_elems = (size_t) n_kv_padded * k_groups16 * 4;
    const size_t v_packed_half4_elems = (size_t) n_kv_padded * v_groups16 * 4;
    s.k_packed_buf = clCreateBuffer(backend_ctx->context, CL_MEM_READ_WRITE, k_packed_half4_elems * sizeof(uint16_t) * 4, nullptr, nullptr);
    s.v_packed_buf = clCreateBuffer(backend_ctx->context, CL_MEM_READ_WRITE, v_packed_half4_elems * sizeof(uint16_t) * 4, nullptr, nullptr);
    GGML_ASSERT(s.k_packed_buf != nullptr && s.v_packed_buf != nullptr);

    const size_t score_half4_elems = (size_t) npack * kv_heads_total * q_width;
    const size_t score_bytes = score_half4_elems * sizeof(uint16_t) * 4;
    s.score_buf = clCreateBuffer(backend_ctx->context, CL_MEM_READ_WRITE, score_bytes, nullptr, nullptr);
    s.prob_buf = clCreateBuffer(backend_ctx->context, CL_MEM_READ_WRITE, score_bytes, nullptr, nullptr);
    GGML_ASSERT(s.score_buf != nullptr && s.prob_buf != nullptr);
    s.score_img1d = ggml_cl_make_image1d_buffer_half4(backend_ctx->context, CL_MEM_READ_ONLY, score_half4_elems, s.score_buf);
    s.prob_img1d = ggml_cl_make_image1d_buffer_half4(backend_ctx->context, CL_MEM_READ_ONLY, score_half4_elems, s.prob_buf);
    s.softmax_stats_img2d = ggml_cl_make_image2d_half4(backend_ctx->context, CL_MEM_READ_WRITE,
                                                       (size_t) q_width, (size_t) kv_heads_total);
    s.xmem_qk = clCreateBuffer(backend_ctx->context, CL_MEM_READ_ONLY, 6144, nullptr, nullptr);
    s.xmem_pv = clCreateBuffer(backend_ctx->context, CL_MEM_READ_ONLY, 6144, nullptr, nullptr);
    GGML_ASSERT(s.softmax_stats_img2d != nullptr && s.xmem_qk != nullptr && s.xmem_pv != nullptr);

    s.n_q = n_q;
    s.n_kv = n_kv;
    s.n_kv_padded = n_kv_padded;
    s.d_head_q = d_head_q;
    s.d_head_v = d_head_v;
    s.q_width = q_width;
    s.kv_heads_total = kv_heads_total;
    return true;
}

static bool ggml_cl_adreno_xmem_attn_can_use(
        const ggml_backend_opencl_context * backend_ctx,
        const ggml_tensor * q,
        const ggml_tensor * k,
        const ggml_tensor * dst) {
    static const char * xmem_sdpa_env = getenv("GGML_OPENCL_XMEM_SDPA");
    if (xmem_sdpa_env == nullptr || xmem_sdpa_env[0] == '0') {
        return false;
    }

    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    if (!backend_ctx->adreno_xmem_attn.compiled || backend_ctx->gpu_family != GPU_FAMILY::ADRENO) {
        return false;
    }
    if (q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
        (k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_F32) ||
        (v->type != GGML_TYPE_F16 && v->type != GGML_TYPE_F32)) {
        return false;
    }
    if (sinks != nullptr) {
        return false;
    }
    if (q->nb[0] != ggml_type_size(q->type) || k->nb[0] != ggml_type_size(k->type) ||
        v->nb[0] != ggml_type_size(v->type) || dst->nb[0] != ggml_type_size(dst->type)) {
        return false;
    }
    if (mask != nullptr && (mask->type != GGML_TYPE_F16 || mask->nb[0] != sizeof(ggml_fp16_t))) {
        return false;
    }

    const int n_q = q->ne[1];
    const int n_kv = k->ne[1];
    const int d_head_q = q->ne[0];
    const int d_head_v = v->ne[0];
    const int n_head = q->ne[2];
    const int n_head_kv = k->ne[2];
    const int n_batch = q->ne[3];

    if (n_q <= 1 || n_kv <= 0 || n_kv > 8192) {
        return false;
    }
    if (d_head_q != k->ne[0] || d_head_v != v->ne[0] || k->ne[1] != v->ne[1] || k->ne[3] != v->ne[3]) {
        return false;
    }
    if (q->ne[3] != k->ne[3]) {
        return false;
    }
    if (n_head_kv <= 0 || n_head % n_head_kv != 0 || k->ne[2] != v->ne[2]) {
        return false;
    }
    if (dst->ne[0] != d_head_v || dst->ne[1] != n_head || dst->ne[2] != n_q || dst->ne[3] != n_batch) {
        return false;
    }
    if ((d_head_q % 8) != 0 || (d_head_v % 32) != 0) {
        return false;
    }
    if (mask != nullptr &&
        (mask->ne[0] < n_kv || mask->ne[1] < n_q || mask->ne[2] <= 0 || mask->ne[3] <= 0)) {
        return false;
    }

    float params[3];
    memcpy(params, dst->op_params, sizeof(params));
    if (params[1] != 0.0f || params[2] != 0.0f) {
        return false;
    }

    const int gqa_ratio = n_head / n_head_kv;
    const int q_width = n_q * gqa_ratio;
    const int kv_heads_total = n_head_kv * n_batch;
    const int n_kv_padded = (int) ggml_cl_round_up((size_t) n_kv, 32);
    const int qpack = d_head_q / 4;
    const int vpack = d_head_v / 4;
    const int npack = n_kv_padded / 4;

    if ((size_t) q_width > backend_ctx->image2d_max_width ||
        (size_t) n_kv_padded > backend_ctx->image2d_max_width) {
        return false;
    }
    if ((size_t) kv_heads_total * (size_t) qpack > backend_ctx->image2d_max_height ||
        (size_t) kv_heads_total * (size_t) vpack > backend_ctx->image2d_max_height) {
        return false;
    }
    if ((size_t) npack * (size_t) kv_heads_total * (size_t) d_head_q > backend_ctx->image_max_buffer_size ||
        (size_t) npack * (size_t) kv_heads_total * (size_t) q_width > backend_ctx->image_max_buffer_size) {
        return false;
    }

    return true;
}

static void ggml_cl_adreno_xmem_attn_run(
        ggml_backend_t backend,
        const ggml_tensor * q,
        const ggml_tensor * k,
        ggml_tensor * dst) {
    ggml_backend_opencl_context * backend_ctx = (ggml_backend_opencl_context *) backend->context;
    auto & xstate = backend_ctx->adreno_xmem_attn;
    auto & s = xstate.scratch;
    if (!xstate.logged) {
        GGML_LOG_INFO("ggml_opencl: using Adreno xmem attention path\n");
        xstate.logged = true;
    }

    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    ggml_tensor_extra_cl * extra_q = (ggml_tensor_extra_cl *) q->extra;
    ggml_tensor_extra_cl * extra_k = (ggml_tensor_extra_cl *) k->extra;
    ggml_tensor_extra_cl * extra_v = (ggml_tensor_extra_cl *) v->extra;
    ggml_tensor_extra_cl * extra_o = (ggml_tensor_extra_cl *) dst->extra;
    ggml_tensor_extra_cl * extra_mask = mask ? (ggml_tensor_extra_cl *) mask->extra : nullptr;

    const cl_ulong offset_q = extra_q->offset + q->view_offs;
    const cl_ulong offset_k = extra_k->offset + k->view_offs;
    const cl_ulong offset_v = extra_v->offset + v->view_offs;
    const cl_ulong offset_o = extra_o->offset + dst->view_offs;
    const cl_ulong offset_mask = extra_mask ? extra_mask->offset + mask->view_offs : 0;

    const int n_q = q->ne[1];
    const int n_kv = k->ne[1];
    const int d_head_q = q->ne[0];
    const int d_head_v = v->ne[0];
    const int n_head = q->ne[2];
    const int n_head_kv = k->ne[2];
    const int n_batch = q->ne[3];
    const int heads_total = n_head * n_batch;
    const int gqa_ratio = n_head / n_head_kv;
    const int q_width = n_q * gqa_ratio;
    const int kv_heads_total = n_head_kv * n_batch;
    const int n_kv_padded = (int) ggml_cl_round_up((size_t) n_kv, 32);
    const int qpack = d_head_q / 4;
    const int opack = d_head_v / 4;
    const int npack = n_kv_padded / 4;
    const float scale = ((const float *) dst->op_params)[0];

    GGML_ASSERT(ggml_cl_adreno_xmem_attn_prepare(
        backend_ctx, n_q, n_kv, d_head_q, d_head_v, n_head, n_head_kv, n_batch));
    const ggml_cl_adreno_xmem_attn_schedule sched =
        ggml_cl_adreno_xmem_attn_select_schedule(
            backend_ctx, n_q, n_kv_padded, heads_total, q_width, gqa_ratio);

    {
        size_t gws[3] = {ggml_cl_round_up((size_t) n_q, 8), (size_t) heads_total, (size_t) qpack};
        size_t lws[3] = {8, 1, (size_t) ((qpack <= 32) ? qpack : 1)};
        cl_kernel kernel = xstate.kernel_q_f32_to_img_scaled;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra_q->data_device));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset_q));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &s.q_img));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(float),    &scale));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int),      &d_head_q));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int),      &n_q));
        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int),      &n_head));
        CL_CHECK(clSetKernelArg(kernel, 7, sizeof(int),      &n_head_kv));
        CL_CHECK(clSetKernelArg(kernel, 8, sizeof(int),      &n_batch));
        CL_CHECK(clSetKernelArg(kernel, 9, sizeof(cl_ulong), &q->nb[1]));
        CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &q->nb[2]));
        CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &q->nb[3]));
        backend_ctx->enqueue_ndrange_kernel(kernel, 3, gws, lws, dst);
    }

    {
        size_t gws[3] = {(size_t) n_kv_padded, (size_t) kv_heads_total, (size_t) qpack};
        size_t lws[3] = {8, 1, (size_t) ((qpack <= 32) ? qpack : 1)};
        cl_kernel kernel = k->type == GGML_TYPE_F16 ?
            xstate.kernel_kv_f16_to_img_gqa : xstate.kernel_kv_f32_to_img_gqa;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra_k->data_device));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset_k));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &s.k_img));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int),      &d_head_q));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int),      &n_kv));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int),      &n_kv_padded));
        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int),      &n_head_kv));
        CL_CHECK(clSetKernelArg(kernel, 7, sizeof(int),      &n_batch));
        CL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_ulong), &k->nb[1]));
        CL_CHECK(clSetKernelArg(kernel, 9, sizeof(cl_ulong), &k->nb[2]));
        CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &k->nb[3]));
        backend_ctx->enqueue_ndrange_kernel(kernel, 3, gws, lws, dst);
    }

    {
        size_t gws[3] = {(size_t) n_kv_padded, (size_t) kv_heads_total, (size_t) opack};
        size_t lws[3] = {8, 1, (size_t) ((opack <= 32) ? opack : 1)};
        cl_kernel kernel = v->type == GGML_TYPE_F16 ?
            xstate.kernel_kv_f16_to_img_gqa : xstate.kernel_kv_f32_to_img_gqa;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra_v->data_device));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset_v));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &s.v_img));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int),      &d_head_v));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int),      &n_kv));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int),      &n_kv_padded));
        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int),      &n_head_kv));
        CL_CHECK(clSetKernelArg(kernel, 7, sizeof(int),      &n_batch));
        CL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_ulong), &v->nb[1]));
        CL_CHECK(clSetKernelArg(kernel, 9, sizeof(cl_ulong), &v->nb[2]));
        CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &v->nb[3]));
        backend_ctx->enqueue_ndrange_kernel(kernel, 3, gws, lws, dst);
    }

    {
        size_t gws[3] = {(size_t) d_head_q, (size_t) kv_heads_total, (size_t) npack};
        size_t lws[3] = {(size_t) MIN(64, d_head_q), (size_t) (kv_heads_total >= 2 ? 2 : 1), (size_t) MIN(8, npack)};
        if (lws[0] * lws[1] * lws[2] > backend_ctx->max_workgroup_size) {
            lws[1] = 1;
        }
        cl_kernel kernel = xstate.kernel_k_gather;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &s.k_transpose_buf));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &s.k_img));
        ggml_cl_set_arg_int4(kernel, 2, n_kv_padded, kv_heads_total, npack, d_head_q);
        ggml_cl_set_arg_int4(kernel, 3, qpack, 0, 0, 0);
        backend_ctx->enqueue_ndrange_kernel(kernel, 3, gws, lws, dst);
    }
    {
        const size_t groups16 = (size_t) ggml_cl_round_up_div(kv_heads_total * d_head_q, 16);
        const size_t packed_linear = (size_t) n_kv_padded * groups16;
        const size_t lws0 = MIN((size_t) 1024, backend_ctx->max_workgroup_size);
        size_t gws[3] = {ggml_cl_round_up(packed_linear, lws0), 1, 1};
        size_t lws[3] = {lws0, 1, 1};
        cl_kernel kernel = xstate.kernel_pack_k;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &s.k_packed_buf));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &s.k_transpose_img1d));
        ggml_cl_set_arg_int4(kernel, 2, 8, (int) packed_linear, qpack, d_head_q);
        ggml_cl_set_arg_int4(kernel, 3, kv_heads_total, kv_heads_total, kv_heads_total, npack);
        ggml_cl_set_arg_int4(kernel, 4, d_head_q, 0, 0, 0);
        backend_ctx->enqueue_ndrange_kernel(kernel, 3, gws, lws, dst);
    }

    {
        size_t lws[3] = {(size_t) sched.qk_lws0, 1, (size_t) sched.qk_lws2};
        const int slices_per_group = sched.qk_lws2 * 8;
        const size_t groups_z = (size_t) ggml_cl_round_up_div(npack, slices_per_group);
        const size_t groups_x = (size_t) ggml_cl_round_up_div(q_width, sched.qk_lws0);
        size_t gws[3] = {
            lws[0] * groups_z,
            groups_x,
            (size_t) kv_heads_total * lws[2],
        };

        cl_kernel kernel = xstate.kernel_qk_gemm;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &s.score_buf));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &s.k_packed_buf));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &s.xmem_qk));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &s.q_img));
        ggml_cl_set_arg_int4(kernel, 4, kv_heads_total, npack, q_width, 32);
        ggml_cl_set_arg_int4(kernel, 5, qpack, 0, 0, kv_heads_total);
        ggml_cl_set_arg_int4(kernel, 6, qpack, 1, 1, 0);
        backend_ctx->enqueue_ndrange_kernel(kernel, 3, gws, lws, dst);
    }
    cl_mem softmax_input_img = s.score_img1d;
    cl_mem softmax_output_buf = s.prob_buf;
    cl_mem pv_prob_img = s.prob_img1d;

    if (mask != nullptr) {
        const cl_ulong mask_nb1 = mask->nb[1];
        const cl_ulong mask_nb2 = mask->nb[2];
        const cl_ulong mask_nb3 = mask->nb[3];
        const int mask_ne2 = mask->ne[2];
        const int mask_ne3 = mask->ne[3];
        size_t lws[3] = {(size_t) sched.softmax_apply_lws0, 1, (size_t) sched.softmax_apply_lws2};
        size_t gws[3] = {
            ggml_cl_round_up((size_t) q_width, lws[0]),
            (size_t) kv_heads_total,
            ggml_cl_round_up((size_t) npack, lws[2]),
        };
        cl_kernel kernel = xstate.kernel_mask_scores;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &s.prob_buf));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &s.score_img1d));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &extra_mask->data_device));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &offset_mask));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int), &q_width));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int), &n_q));
        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int), &n_kv));
        CL_CHECK(clSetKernelArg(kernel, 7, sizeof(int), &n_kv_padded));
        CL_CHECK(clSetKernelArg(kernel, 8, sizeof(int), &kv_heads_total));
        CL_CHECK(clSetKernelArg(kernel, 9, sizeof(int), &n_head));
        CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int), &n_head_kv));
        CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &mask_nb1));
        CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &mask_nb2));
        CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &mask_nb3));
        CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int), &mask_ne2));
        CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int), &mask_ne3));
        backend_ctx->enqueue_ndrange_kernel(kernel, 3, gws, lws, dst);

        softmax_input_img = s.prob_img1d;
        softmax_output_buf = s.score_buf;
        pv_prob_img = s.score_img1d;
    }

    {
        size_t lws[3] = {(size_t) sched.softmax_reduce_lws0, 1, 1};
        size_t gws[3] = {ggml_cl_round_up((size_t) q_width, lws[0]), (size_t) kv_heads_total, 1};
        cl_kernel kernel = xstate.kernel_softmax_reduce_basic;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &softmax_input_img));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &s.softmax_stats_img2d));
        ggml_cl_set_arg_int4(kernel, 2, kv_heads_total, 1, q_width, n_kv);
        ggml_cl_set_arg_int4(kernel, 3, kv_heads_total, q_width, 0, 0);
        backend_ctx->enqueue_ndrange_kernel(kernel, 3, gws, lws, dst);
    }
    {
        size_t lws[3] = {(size_t) sched.softmax_apply_lws0, 1, (size_t) sched.softmax_apply_lws2};
        size_t gws[3] = {
            ggml_cl_round_up((size_t) q_width, lws[0]),
            (size_t) kv_heads_total,
            ggml_cl_round_up((size_t) npack, lws[2]),
        };
        cl_kernel kernel = xstate.kernel_softmax_apply_basic;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &softmax_output_buf));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &softmax_input_img));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &s.softmax_stats_img2d));
        ggml_cl_set_arg_int4(kernel, 3, kv_heads_total, npack, q_width, 1);
        ggml_cl_set_arg_int4(kernel, 4, kv_heads_total, q_width, n_kv, 0);
        backend_ctx->enqueue_ndrange_kernel(kernel, 3, gws, lws, dst);
    }
    {
        const size_t groups16 = (size_t) ggml_cl_round_up_div(kv_heads_total * d_head_v, 16);
        const size_t packed_linear = (size_t) n_kv_padded * groups16;
        const size_t lws0 = MIN((size_t) 1024, backend_ctx->max_workgroup_size);
        size_t gws[3] = {ggml_cl_round_up(packed_linear, lws0), 1, 1};
        size_t lws[3] = {lws0, 1, 1};
        cl_kernel kernel = xstate.kernel_pack_v;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &s.v_packed_buf));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &s.v_img));
        ggml_cl_set_arg_int4(kernel, 2, 8, (int) packed_linear, npack, n_kv_padded);
        ggml_cl_set_arg_int4(kernel, 3, kv_heads_total, kv_heads_total, opack, 0);
        backend_ctx->enqueue_ndrange_kernel(kernel, 3, gws, lws, dst);
    }

    {
        size_t lws[3] = {(size_t) sched.pv_lws0, 1, (size_t) sched.pv_lws2};
        const int blocks = ggml_cl_round_up_div(opack, 8);
        const size_t groups_z = (size_t) ggml_cl_round_up_div(blocks, sched.pv_lws2);
        const size_t groups_x = (size_t) ggml_cl_round_up_div(q_width, sched.pv_lws0);
        size_t gws[3] = {
            lws[0] * groups_z,
            groups_x,
            (size_t) kv_heads_total * lws[2],
        };

        cl_kernel kernel = xstate.kernel_pv_gemm;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &s.v_packed_buf));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &s.xmem_pv));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &pv_prob_img));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &s.out_img));
        ggml_cl_set_arg_int4(kernel, 4, kv_heads_total, opack, q_width, 32);
        ggml_cl_set_arg_int4(kernel, 5, npack, 0, 0, kv_heads_total);
        ggml_cl_set_arg_int4(kernel, 6, kv_heads_total * q_width, npack, q_width, 1);
        ggml_cl_set_arg_int4(kernel, 7, 1, 0, 0, 0);
        backend_ctx->enqueue_ndrange_kernel(kernel, 3, gws, lws, dst);
    }

    {
        size_t gws[3] = {ggml_cl_round_up((size_t) n_q, 8), (size_t) heads_total, (size_t) opack};
        size_t lws[3] = {8, 1, (size_t) ((opack <= 32) ? opack : 1)};
        cl_kernel kernel = xstate.kernel_img_to_f32;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra_o->data_device));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &offset_o));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &s.out_img));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int),      &d_head_v));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int),      &n_q));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int),      &n_head));
        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int),      &n_head_kv));
        CL_CHECK(clSetKernelArg(kernel, 7, sizeof(int),      &n_batch));
        CL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_ulong), &dst->nb[1]));
        CL_CHECK(clSetKernelArg(kernel, 9, sizeof(cl_ulong), &dst->nb[2]));
        CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &dst->nb[3]));
        backend_ctx->enqueue_ndrange_kernel(kernel, 3, gws, lws, dst);
    }
}

#endif // GGML_OPENCL_USE_ADRENO_KERNELS


#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
static void ggml_cl_flash_attn_prefill_bin(ggml_backend_t backend, const ggml_tensor * q, const ggml_tensor * k, ggml_tensor * dst) {
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    GGML_ASSERT(q->extra);
    GGML_ASSERT(k->extra);
    GGML_ASSERT(v->extra);
    GGML_ASSERT(dst->extra);
    if (mask) {
        GGML_ASSERT(mask->extra);
    }
    if (sinks) {
        GGML_ASSERT(sinks->extra);
    }

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;
    cl_context context = backend_ctx->context;

    const int n_q = q->ne[1];
    const int n_kv = k->ne[1];
    const int d_head_q = q->ne[0];
    const int d_head_v = v->ne[0];
    const int n_head = q->ne[2];
    const int n_head_kv = k->ne[2];
    const int n_batch = q->ne[3];

    const std::pair<int, int> dk_dv = {d_head_q, d_head_v};
    cl_kernel kernel = backend_ctx->fa.kernel_flash_attn_f32_f16_bin;
    GGML_ASSERT(kernel != NULL);

    ggml_tensor_extra_cl * extra_q = (ggml_tensor_extra_cl *)q->extra;
    ggml_tensor_extra_cl * extra_k = (ggml_tensor_extra_cl *)k->extra;
    ggml_tensor_extra_cl * extra_v = (ggml_tensor_extra_cl *)v->extra;
    ggml_tensor_extra_cl * extra_o = (ggml_tensor_extra_cl *)dst->extra;
    ggml_tensor_extra_cl * extra_mask = mask ? (ggml_tensor_extra_cl *)mask->extra : NULL;
    ggml_tensor_extra_cl * extra_sinks = sinks ? (ggml_tensor_extra_cl *)sinks->extra : NULL;

    cl_ulong offset_q = extra_q->offset + q->view_offs;
    cl_ulong offset_o = extra_o->offset + dst->view_offs;

    cl_mem   mask_buffer = extra_mask ? extra_mask->data_device : NULL;
    cl_ulong offset_mask = extra_mask ? extra_mask->offset + mask->view_offs : 0;
    cl_mem   sinks_buffer = extra_sinks ? extra_sinks->data_device : NULL;
    cl_ulong offset_sinks = extra_sinks ? extra_sinks->offset + sinks->view_offs : 0;

    const cl_ulong q_nb1 = q->nb[1];
    const cl_ulong q_nb2 = q->nb[2];
    const cl_ulong q_nb3 = q->nb[3];

    cl_mem   k_data_device = extra_k->data_device;
    cl_ulong offset_k = extra_k->offset + k->view_offs;
    cl_ulong k_nb1 = k->nb[1];
    cl_ulong k_nb2 = k->nb[2];
    cl_ulong k_nb3 = k->nb[3];

    cl_mem   v_data_device = extra_v->data_device;
    cl_ulong offset_v = extra_v->offset + v->view_offs;
    cl_ulong v_nb1 = v->nb[1];
    cl_ulong v_nb2 = v->nb[2];
    cl_ulong v_nb3 = v->nb[3];

    const cl_ulong o_nb1 = dst->nb[1];
    const cl_ulong o_nb2 = dst->nb[2];
    const cl_ulong o_nb3 = dst->nb[3];

    const cl_ulong mask_nb1 = mask ? mask->nb[1] : 0;
    const cl_ulong mask_nb2 = mask ? mask->nb[2] : 0;
    const cl_ulong mask_nb3 = mask ? mask->nb[3] : 0;
    const int mask_ne2 = mask ? mask->ne[2] : 0;
    const int mask_ne3 = mask ? mask->ne[3] : 0;

    float * params      = (float *)dst->op_params;
    float scale         = params[0];
    float max_bias      = params[1];
    float logit_softcap = params[2];

    const int is_causal = (mask == NULL && n_q > 1 && n_q == n_kv);     // redundant n_q > 1 check ?

    const int n_head_log2_val = n_head > 0 ? 1u << (int)floorf(log2f((float)n_head)) : 0;
    const float n_head_log2_f = n_head_log2_val > 0 ? (float)n_head_log2_val : 1.0f;
    const float m0 = powf(2.0f, -(max_bias) / n_head_log2_f);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2_f);

    const bool is_q8_0 = q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_Q8_0 && v->type == GGML_TYPE_Q8_0;

    ggml_cl_flash_attn_temp_buffer temp_k;
    ggml_cl_flash_attn_temp_buffer temp_v;
    ggml_cl_flash_attn_temp_buffer temp_k_aos;
    ggml_cl_flash_attn_temp_buffer temp_v_aos;

    if (is_q8_0) {
        ggml_cl_flash_attn_reconstruct_aos(
            backend_ctx, k, temp_k_aos, k_data_device, offset_k, k_nb1, k_nb2, k_nb3);

        ggml_cl_flash_attn_reconstruct_aos(
            backend_ctx, v, temp_v_aos, v_data_device, offset_v, v_nb1, v_nb2, v_nb3);

        bool k_done = ggml_cl_flash_attn_dequant_kv_gpu(
            backend_ctx, k, GGML_TYPE_F16, k_data_device, offset_k, k_nb1, k_nb2, k_nb3,
            temp_k, k_data_device, offset_k, k_nb1, k_nb2, k_nb3);

        bool v_done = ggml_cl_flash_attn_dequant_kv_gpu(
            backend_ctx, v, GGML_TYPE_F16, v_data_device, offset_v, v_nb1, v_nb2, v_nb3,
            temp_v, v_data_device, offset_v, v_nb1, v_nb2, v_nb3);

        GGML_ASSERT(k_done && v_done);
    }

    // Allocate input/output memory buffers
    cl_mem mem_matrixQ;
    cl_mem mem_matrixK;
    cl_mem mem_matrixV;
    cl_mem mem_matrixO;
    cl_buffer_region region;
    cl_int err;

    region.origin = offset_q;
    region.size = ggml_nbytes(q);
    mem_matrixQ = clCreateSubBuffer(extra_q->data_device, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
    CL_CHECK(err);

    region.origin = offset_k;
    region.size = is_q8_0 ? (size_t) k_nb3 * (size_t) k->ne[3] : ggml_nbytes(k);
    mem_matrixK = clCreateSubBuffer(k_data_device, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
    CL_CHECK(err);

    region.origin = offset_v;
    region.size = is_q8_0 ? (size_t) v_nb3 * (size_t) v->ne[3] : ggml_nbytes(v);
    mem_matrixV = clCreateSubBuffer(v_data_device, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
    CL_CHECK(err);

    region.origin = offset_o;
    region.size = ggml_nbytes(dst);
    mem_matrixO = clCreateSubBuffer(extra_o->data_device, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
    CL_CHECK(err);

    cl_image_format img_fmt_1d = { CL_RGBA, CL_FLOAT};
    cl_image_desc img_desc_1d;

    // use image 1d buffer used as fallback when on mask is applied
    cl_mem  mem_tex_mask_fallback_1dbuf;
    img_fmt_1d = { CL_RGBA, CL_HALF_FLOAT};
    memset(&img_desc_1d, 0, sizeof(img_desc_1d));
    img_desc_1d.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
    img_desc_1d.image_width = 1;
    img_desc_1d.buffer = mem_matrixK;
    mem_tex_mask_fallback_1dbuf = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt_1d, &img_desc_1d, NULL, &err);
    CL_CHECK(err);

    cl_mem  mem_tex_matrixO_1dbuf;
    img_fmt_1d = { CL_RGBA, CL_FLOAT};
    memset(&img_desc_1d, 0, sizeof(img_desc_1d));
    img_desc_1d.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
    img_desc_1d.image_width = ggml_nbytes(dst) / 4 / 4;
    img_desc_1d.buffer = mem_matrixO;
    mem_tex_matrixO_1dbuf = clCreateImage(context, CL_MEM_WRITE_ONLY, &img_fmt_1d, &img_desc_1d, NULL, &err);
    CL_CHECK(err);

    // The bin kernel requires 2d (or 3d) buffers packed for data loading/multiplication.
    // These repack kernels launch across all buffers to ensure compatibility
    cl_mem  mem_tex_matrixMask_1dbuf = NULL;
    cl_mem  mem_matrixMask = NULL;
    cl_mem  mem_matrixMask_padded = NULL;
    cl_ulong mask_nb1_padded = mask_nb1, mask_nb2_padded = mask_nb2, mask_nb3_padded = mask_nb3;
    if (extra_mask) {
        // allocate mem_matrixMask w/ new padded size
        size_t n_kv_padded = GGML_PAD(n_kv, 4);
        size_t mask_nb_padded = n_kv_padded * sizeof(cl_half) * mask->ne[1] * mask->ne[2] * mask->ne[3];

        // apply offset and create subBuffer for mask
        region.origin = offset_mask;
        region.size = ggml_nbytes(mask);
        mem_matrixMask = clCreateSubBuffer(extra_mask->data_device, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);

        {
            // create padded mask to contain all data
            mem_matrixMask_padded = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, mask_nb_padded, NULL, &err);
            CL_CHECK(err);

            // pass extra_mask->data_device, mem_matrixMask to kernel for copying/padding
            mask_nb1_padded = (cl_ulong)n_kv_padded * sizeof(cl_half);
            mask_nb2_padded = mask_nb1_padded * (cl_ulong)mask->ne[1];
            mask_nb3_padded = mask_nb2_padded * (cl_ulong)mask->ne[2];

            cl_kernel repack_mask = backend_ctx->fa.kernel_repack_mask_for_wmm;
            CL_CHECK(clSetKernelArg(repack_mask, 0, sizeof(cl_mem),   &mem_matrixMask));
            CL_CHECK(clSetKernelArg(repack_mask, 1, sizeof(cl_ulong), &mask_nb1));
            CL_CHECK(clSetKernelArg(repack_mask, 2, sizeof(cl_ulong), &mask_nb2));
            CL_CHECK(clSetKernelArg(repack_mask, 3, sizeof(cl_ulong), &mask_nb3));
            CL_CHECK(clSetKernelArg(repack_mask, 4, sizeof(int),      &mask_ne2));
            CL_CHECK(clSetKernelArg(repack_mask, 5, sizeof(cl_mem),   &mem_matrixMask_padded));
            CL_CHECK(clSetKernelArg(repack_mask, 6, sizeof(cl_ulong), &mask_nb1_padded));
            CL_CHECK(clSetKernelArg(repack_mask, 7, sizeof(cl_ulong), &mask_nb2_padded));
            CL_CHECK(clSetKernelArg(repack_mask, 8, sizeof(cl_ulong), &mask_nb3_padded));

            size_t repack_mask_gws[3] = {(size_t)n_kv, (size_t)mask->ne[1], (size_t)mask_ne2 * (size_t)mask->ne[3]};
            backend_ctx->enqueue_ndrange_kernel(repack_mask, 3, repack_mask_gws, NULL, dst);
        }

        // use image 1d buffer for matrix Mask (padded row stride)
        cl_image_format img_fmt_mask_1d = { CL_RGBA, CL_HALF_FLOAT};
        cl_image_desc img_desc_mask_1d;
        memset(&img_desc_mask_1d, 0, sizeof(img_desc_mask_1d));
        img_desc_mask_1d.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc_mask_1d.image_width = mask_nb_padded / 2 / 4;
        img_desc_mask_1d.buffer = mem_matrixMask_padded;
        mem_tex_matrixMask_1dbuf = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt_mask_1d, &img_desc_mask_1d, NULL, &err);
        CL_CHECK(err);
    }

    // WMM QK uses repacked 3D images.
    // Q image: rows, heads, packed depth.
    cl_image_format img_fmt_3d = { CL_RGBA, CL_HALF_FLOAT };
    cl_image_desc   img_desc_3d;

    memset(&img_desc_3d, 0, sizeof(img_desc_3d));
    img_desc_3d.image_type   = CL_MEM_OBJECT_IMAGE3D;
    img_desc_3d.image_width  = (size_t)n_q;
    img_desc_3d.image_height = (size_t)n_batch * (size_t)n_head;
    img_desc_3d.image_depth  = (size_t)d_head_q / 4;
    cl_mem img_q_wmm = NULL;
    img_q_wmm = clCreateImage(context, CL_MEM_READ_WRITE, &img_fmt_3d, &img_desc_3d, NULL, &err);
    CL_CHECK(err);

    {
        cl_kernel repack_q = backend_ctx->fa.kernel_repack_q_for_wmm;
        CL_CHECK(clSetKernelArg(repack_q, 0, sizeof(cl_mem),   &mem_matrixQ));
        CL_CHECK(clSetKernelArg(repack_q, 1, sizeof(cl_ulong), &q_nb1));
        CL_CHECK(clSetKernelArg(repack_q, 2, sizeof(cl_ulong), &q_nb2));
        CL_CHECK(clSetKernelArg(repack_q, 3, sizeof(cl_ulong), &q_nb3));
        CL_CHECK(clSetKernelArg(repack_q, 4, sizeof(int),      &n_head));
        CL_CHECK(clSetKernelArg(repack_q, 5, sizeof(cl_mem),   &img_q_wmm));

        size_t repack_q_gws[3] = {(size_t)d_head_q / 4, (size_t)n_q, (size_t)n_batch * (size_t)n_head};
        backend_ctx->enqueue_ndrange_kernel(repack_q, 3, repack_q_gws, NULL, dst);
    }

    // K image: columns, row groups, KV heads.
    const size_t n_kv_row4 = ((size_t)n_kv + 3) / 4;

    memset(&img_desc_3d, 0, sizeof(img_desc_3d));
    img_desc_3d.image_type   = CL_MEM_OBJECT_IMAGE3D;
    img_desc_3d.image_width  = (size_t)d_head_q;
    img_desc_3d.image_height = n_kv_row4;
    img_desc_3d.image_depth  = (size_t)n_batch * (size_t)n_head_kv;
    cl_mem img_k_wmm = NULL;
    img_k_wmm = clCreateImage(context, CL_MEM_READ_WRITE, &img_fmt_3d, &img_desc_3d, NULL, &err);
    CL_CHECK(err);

    {
        cl_kernel repack_k = backend_ctx->fa.kernel_repack_k_for_wmm;
        CL_CHECK(clSetKernelArg(repack_k, 0, sizeof(cl_mem),   &mem_matrixK));
        CL_CHECK(clSetKernelArg(repack_k, 1, sizeof(cl_ulong), &k_nb1));
        CL_CHECK(clSetKernelArg(repack_k, 2, sizeof(cl_ulong), &k_nb2));
        CL_CHECK(clSetKernelArg(repack_k, 3, sizeof(cl_ulong), &k_nb3));
        CL_CHECK(clSetKernelArg(repack_k, 4, sizeof(int),      &n_head_kv));
        CL_CHECK(clSetKernelArg(repack_k, 5, sizeof(int),      &n_kv));
        CL_CHECK(clSetKernelArg(repack_k, 6, sizeof(cl_mem),   &img_k_wmm));

        size_t repack_k_gws[3] = {(size_t)d_head_q, n_kv_row4, (size_t)n_batch * (size_t)n_head_kv};
        backend_ctx->enqueue_ndrange_kernel(repack_k, 3, repack_k_gws, NULL, dst);
    }

    // V image: kv-rows (contracted), packed head-dim groups, KV heads.
    memset(&img_desc_3d, 0, sizeof(img_desc_3d));
    img_desc_3d.image_type   = CL_MEM_OBJECT_IMAGE3D;
    img_desc_3d.image_width  = (size_t)n_kv;
    img_desc_3d.image_height = (size_t)d_head_v / 4;
    img_desc_3d.image_depth  = (size_t)n_batch * (size_t)n_head_kv;
    cl_mem img_v_wmm = NULL;
    img_v_wmm = clCreateImage(context, CL_MEM_READ_WRITE, &img_fmt_3d, &img_desc_3d, NULL, &err);
    CL_CHECK(err);

    {
        cl_kernel repack_v = backend_ctx->fa.kernel_repack_v_for_wmm;
        CL_CHECK(clSetKernelArg(repack_v, 0, sizeof(cl_mem),   &mem_matrixV));
        CL_CHECK(clSetKernelArg(repack_v, 1, sizeof(cl_ulong), &v_nb1));
        CL_CHECK(clSetKernelArg(repack_v, 2, sizeof(cl_ulong), &v_nb2));
        CL_CHECK(clSetKernelArg(repack_v, 3, sizeof(cl_ulong), &v_nb3));
        CL_CHECK(clSetKernelArg(repack_v, 4, sizeof(int),      &n_head_kv));
        CL_CHECK(clSetKernelArg(repack_v, 5, sizeof(cl_mem),   &img_v_wmm));

        size_t repack_v_gws[3] = {(size_t)d_head_v / 4, (size_t)n_kv, (size_t)n_batch * (size_t)n_head_kv};
        backend_ctx->enqueue_ndrange_kernel(repack_v, 3, repack_v_gws, NULL, dst);
    }

    cl_int enable_mask = (extra_mask) ? 1 : 0;
    mask_buffer = extra_mask ? mem_tex_matrixMask_1dbuf : mem_tex_mask_fallback_1dbuf;

    cl_mem mem_sinksBuf = NULL;
    cl_mem mem_tex_sinks_1dbuf = NULL;
    cl_int enable_sinks = (sinks_buffer != NULL) ? 1 : 0;
    if (enable_sinks) {
        region.origin = offset_sinks;
        region.size = ggml_nbytes(sinks);
        mem_sinksBuf = clCreateSubBuffer(extra_sinks->data_device, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        CL_CHECK(err);

        cl_image_format img_fmt_sinks_1d = { CL_R, CL_FLOAT };
        cl_image_desc img_desc_sinks_1d;
        memset(&img_desc_sinks_1d, 0, sizeof(img_desc_sinks_1d));
        img_desc_sinks_1d.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc_sinks_1d.image_width = (size_t)n_head;
        img_desc_sinks_1d.buffer = mem_sinksBuf;
        mem_tex_sinks_1dbuf = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt_sinks_1d, &img_desc_sinks_1d, NULL, &err);
        CL_CHECK(err);
    } else {
        // The image obj cannot be null so we back with buffer of size 1 and use matrixK to back because it always exists
        cl_image_format img_fmt_sinks_fallback = { CL_R, CL_FLOAT };
        cl_image_desc img_desc_sinks_fallback;
        memset(&img_desc_sinks_fallback, 0, sizeof(img_desc_sinks_fallback));
        img_desc_sinks_fallback.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc_sinks_fallback.image_width = 1;
        img_desc_sinks_fallback.buffer = mem_matrixK;
        mem_tex_sinks_1dbuf = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt_sinks_fallback, &img_desc_sinks_fallback, NULL, &err);
        CL_CHECK(err);
    }

    cl_uint arg = 0;

    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem),   &mem_tex_matrixO_1dbuf));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(float),    &scale));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int),      &n_q));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int),      &n_kv));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int),      &is_causal));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int),      &n_head));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &q_nb1));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &q_nb2));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &q_nb3));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &k_nb1));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &k_nb2));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &k_nb3));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &v_nb1));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &v_nb2));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &v_nb3));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &o_nb1));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &o_nb2));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &o_nb3));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(float),    &max_bias));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(float),    &m0));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(float),    &m1));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int),      &n_head_log2_val));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(float),    &logit_softcap));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int),      &n_head_kv));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem),   &mask_buffer));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int),      &enable_mask));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &mask_nb1_padded));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &mask_nb2_padded));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &mask_nb3_padded));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int),      &mask_ne2));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int),      &mask_ne3));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem),   &mem_tex_sinks_1dbuf));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int),      &enable_sinks));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem),   &img_q_wmm));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem),   &img_k_wmm));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem),   &img_v_wmm));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int),      &d_head_q));

    size_t global_work_size[3], local_work_size[3];

    const int n_waves_v = d_head_q / 64;

    local_work_size[0] = 64;
    local_work_size[1] = n_waves_v;
    local_work_size[2] = 1;

    global_work_size[0] = 64;
    global_work_size[1] = ((n_q + 64 - 1) / 64) * n_waves_v;
    global_work_size[2] = n_batch * n_head;

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);

    CL_CHECK(clReleaseMemObject(mem_tex_matrixO_1dbuf));
    CL_CHECK(clReleaseMemObject(img_q_wmm));
    CL_CHECK(clReleaseMemObject(img_k_wmm));
    CL_CHECK(clReleaseMemObject(img_v_wmm));

    if (mem_tex_matrixMask_1dbuf) {
        CL_CHECK(clReleaseMemObject(mem_tex_matrixMask_1dbuf));
    }
    if (mem_matrixMask) {
        CL_CHECK(clReleaseMemObject(mem_matrixMask));
    }
    if (mem_matrixMask_padded) {
        CL_CHECK(clReleaseMemObject(mem_matrixMask_padded));
    }
    if (mem_tex_sinks_1dbuf) {
        CL_CHECK(clReleaseMemObject(mem_tex_sinks_1dbuf));
    }
    if (mem_sinksBuf) {
        CL_CHECK(clReleaseMemObject(mem_sinksBuf));
    }
    CL_CHECK(clReleaseMemObject(mem_matrixQ));
    CL_CHECK(clReleaseMemObject(mem_matrixK));
    CL_CHECK(clReleaseMemObject(mem_matrixV));
    CL_CHECK(clReleaseMemObject(mem_matrixO));
}
#endif // GGML_OPENCL_USE_ADRENO_KERNELS

void ggml_cl_flash_attn(ggml_backend_t backend, const ggml_tensor * q, const ggml_tensor * k, ggml_tensor * dst) {
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    GGML_ASSERT(q->extra);
    GGML_ASSERT(k->extra);
    GGML_ASSERT(v->extra);
    GGML_ASSERT(dst->extra);

    if (mask) {
        GGML_ASSERT(mask->extra);
    }
    if (sinks) {
        GGML_ASSERT(sinks->extra);
    }

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    const int n_q = q->ne[1];
    const int n_kv = k->ne[1];
    const int d_head_q = q->ne[0];
    const int d_head_v = v->ne[0];
    const int n_head = q->ne[2];
    const int n_head_kv = k->ne[2];
    const int n_batch = q->ne[3];

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
    if (ggml_cl_adreno_xmem_attn_can_use(backend_ctx, q, k, dst)) {
        ggml_cl_adreno_xmem_attn_run(backend, q, k, dst);
        return;
    }
#endif

    // DK=512 (Gemma-4 global layers) runs decode-only (q1 / q1_split) on
    // Adreno - it never uses the BM-tile path, and the prepass + split-tile
    // programs OOM the compiler at DK=512; supports_op only admits
    // n_q==1 here and prefill goes to CPU
    const bool fa_decode_only_512 = (d_head_q == 512);

    // per-variant lazy compile for this (dk, dv)
    // DK=512 decode (n_q==1) needs no prepass
    // DK=512 prefill (n_q>1) does, so compile it only when needed
    if (!fa_decode_only_512 || n_q > 1) {
        ggml_opencl_ensure_fa_pre_kernels(backend_ctx, d_head_q, d_head_v);
    }

    cl_kernel kernel = NULL;
    bool use_prefill_k_img = false;  //  K is image1d_buffer_t for DK=512 prefill

    const bool is_f16 = q->type == GGML_TYPE_F16;
    const bool is_mixed = q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16;
    const bool is_q8_0 = q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_Q8_0 && v->type == GGML_TYPE_Q8_0;
    const bool is_q4_0 = q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_Q4_0 && v->type == GGML_TYPE_Q4_0;

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
    if (use_fa_bin_kernels_prefill(backend_ctx, q, k, v)) {
        // We support the prefill path of flash attn with a specialized d_head = 64/128/256
        ggml_cl_flash_attn_prefill_bin(backend, q, k, dst);
        return;
    }
#endif

    if (is_f16) {
        ggml_opencl_ensure_fa_variant(backend_ctx, d_head_q, d_head_v, FA_VARIANT_F16);
    } else if (is_mixed) {
        ggml_opencl_ensure_fa_variant(backend_ctx, d_head_q, d_head_v, FA_VARIANT_F32_F16);
        if (fa_decode_only_512) {
            // DK=512: the BM-tile prefill kernels are specifically compiled from
            // FA_PREFILL_ONLY
            if (n_q > 1) {
                ggml_opencl_ensure_fa_f32_f16_prefill_512(backend_ctx, /*split=*/false);
                ggml_opencl_ensure_fa_f32_f16_prefill_512(backend_ctx, /*split=*/true);
            }
        } else {
            ggml_opencl_ensure_fa_variant(backend_ctx, d_head_q, d_head_v, FA_VARIANT_F32_F16_SPLIT);
        }
    } else if (is_q8_0) {
        ggml_opencl_ensure_fa_variant(backend_ctx, d_head_q, d_head_v, FA_VARIANT_Q8_0);
        if (d_head_q == 96 && d_head_v == 96) {
            ggml_opencl_ensure_fa_quant_split_override(backend_ctx, 96, 96, /*quant_bm=*/16, /*quant_n_split=*/3, /*is_q8_0=*/true);
        } else if (d_head_q == 256 && d_head_v == 256) {
            ggml_opencl_ensure_fa_quant_split_override(backend_ctx, 256, 256, /*quant_bm=*/16, /*quant_n_split=*/8, /*is_q8_0=*/true);
        } else {
            ggml_opencl_ensure_fa_variant(backend_ctx, d_head_q, d_head_v, FA_VARIANT_Q8_0_SPLIT);
        }
    } else if (is_q4_0) {
        ggml_opencl_ensure_fa_variant(backend_ctx, d_head_q, d_head_v, FA_VARIANT_Q4_0);
        if (d_head_q == 96 && d_head_v == 96) {
            ggml_opencl_ensure_fa_quant_split_override(backend_ctx, 96, 96, /*quant_bm=*/16, /*quant_n_split=*/3, /*is_q8_0=*/false);
        } else if (d_head_q == 256 && d_head_v == 256) {
            ggml_opencl_ensure_fa_quant_split_override(backend_ctx, 256, 256, /*quant_bm=*/16, /*quant_n_split=*/8, /*is_q8_0=*/false);
        } else {
            ggml_opencl_ensure_fa_variant(backend_ctx, d_head_q, d_head_v, FA_VARIANT_Q4_0_SPLIT);
        }
    } else {
        ggml_opencl_ensure_fa_variant(backend_ctx, d_head_q, d_head_v, FA_VARIANT_F32);
    }

    const std::pair<int, int> dk_dv = {d_head_q, d_head_v};
    const bool use_native_q8_0_q1 = is_q8_0 && n_q == 1 &&
                                    backend_ctx->fa.f32_q8_0_q1.count(dk_dv) > 0;
    // Native q8_0 prefill - reads q8_0 directly, wg_size = cfg->bm.
    const bool use_native_q8_0 = is_q8_0 && n_q > 1 &&
                                 backend_ctx->fa.f32_q8_0.count(dk_dv) > 0;
    const bool use_native_q4_0_q1 = is_q4_0 && n_q == 1 &&
                                    backend_ctx->fa.f32_q4_0_q1.count(dk_dv) > 0;
    const bool use_native_q4_0    = is_q4_0 && n_q > 1 &&
                                    backend_ctx->fa.f32_q4_0.count(dk_dv) > 0;
    const int block_m = n_q > 1
        ? (is_mixed ? backend_ctx->fa.f32_f16_bm.at(dk_dv) : backend_ctx->fa.bm.at(dk_dv))
        : 0;
    // block_n is only used by the n_q > 1 prefill path; its map is not
    // populated for DK=512 decode, so do not read it for decode.
    const int block_n = (n_q > 1)
        ? (is_mixed ? backend_ctx->fa.f32_f16_bn.at(dk_dv)
                    : backend_ctx->fa.bn.at(dk_dv))
        : 0;
    // Pick split variant only when n_kv crosses the per-(dk,dv) threshold.
    // the N_SPLIT>1 prefill tile reduces DK partials via subgroup shuffle,
    // on Intel it uses the non-split BM tile and does not depend on subgroup size
    const bool use_split_kernel = (n_q > 1 && is_mixed &&
        backend_ctx->gpu_family != INTEL &&
        backend_ctx->fa.f32_f16_split.count(dk_dv) > 0 &&
        n_kv >= backend_ctx->fa.f32_f16_split_nkv_threshold.at(dk_dv));
    const bool use_split_q8_0 = (use_native_q8_0 && backend_ctx->gpu_family != INTEL &&
        backend_ctx->fa.f32_q8_0_split.count(dk_dv) > 0 &&
        n_kv >= backend_ctx->fa.f32_q8_0_split_nkv_threshold.at(dk_dv));
    const bool use_split_q4_0 = (use_native_q4_0 && backend_ctx->gpu_family != INTEL &&
        backend_ctx->fa.f32_q4_0_split.count(dk_dv) > 0 &&
        n_kv >= backend_ctx->fa.f32_q4_0_split_nkv_threshold.at(dk_dv));
    const int wg_size_fa = (n_q > 1 && is_mixed)
        ? (use_split_kernel
            ? backend_ctx->fa.f32_f16_split_wg_size.at(dk_dv)
            : backend_ctx->fa.f32_f16_wg_size.at(dk_dv))
        : block_m;

    ggml_tensor_extra_cl * extra_q = (ggml_tensor_extra_cl *)q->extra;
    ggml_tensor_extra_cl * extra_o = (ggml_tensor_extra_cl *)dst->extra;
    ggml_tensor_extra_cl * extra_mask = mask ? (ggml_tensor_extra_cl *)mask->extra : NULL;
    ggml_tensor_extra_cl * extra_sinks = sinks ? (ggml_tensor_extra_cl *)sinks->extra : NULL;

    // SoA q8_0/q4_0 K/V: data_device aliases the `q` subbuffer; reconstruct
    // AoS into a temp buffer below. AoS tensors use extra_k/v->data_device.
    const bool k_soa = ggml_cl_is_q8_0_soa(k) || ggml_cl_is_q4_0_soa(k);
    const bool v_soa = ggml_cl_is_q8_0_soa(v) || ggml_cl_is_q4_0_soa(v);
    ggml_tensor_extra_cl * extra_k = k_soa ? nullptr : (ggml_tensor_extra_cl *)k->extra;
    ggml_tensor_extra_cl * extra_v = v_soa ? nullptr : (ggml_tensor_extra_cl *)v->extra;

    cl_ulong offset_q = extra_q->offset + q->view_offs;
    cl_ulong offset_k = k_soa ? 0 : extra_k->offset + k->view_offs;
    cl_ulong offset_v = v_soa ? 0 : extra_v->offset + v->view_offs;
    cl_ulong offset_o = extra_o->offset + dst->view_offs;
    cl_mem   mask_buffer = extra_mask ? extra_mask->data_device : NULL;
    cl_ulong offset_mask = extra_mask ? extra_mask->offset + mask->view_offs : 0;
    cl_mem   sinks_buffer = extra_sinks ? extra_sinks->data_device : NULL;
    cl_ulong offset_sinks = extra_sinks ? extra_sinks->offset + sinks->view_offs : 0;

    const cl_ulong q_nb1 = q->nb[1];
    const cl_ulong q_nb2 = q->nb[2];
    const cl_ulong q_nb3 = q->nb[3];

    cl_ulong k_nb1 = k->nb[1];
    cl_ulong k_nb2 = k->nb[2];
    cl_ulong k_nb3 = k->nb[3];

    cl_ulong v_nb1 = v->nb[1];
    cl_ulong v_nb2 = v->nb[2];
    cl_ulong v_nb3 = v->nb[3];

    const cl_ulong o_nb1 = dst->nb[1];
    const cl_ulong o_nb2 = dst->nb[2];
    const cl_ulong o_nb3 = dst->nb[3];

    const cl_ulong mask_nb1 = mask ? mask->nb[1] : 0;
    const cl_ulong mask_nb2 = mask ? mask->nb[2] : 0;
    const cl_ulong mask_nb3 = mask ? mask->nb[3] : 0;
    const int mask_ne2 = mask ? mask->ne[2] : 0;
    const int mask_ne3 = mask ? mask->ne[3] : 0;

    float scale;
    float max_bias;
    float logit_softcap;

    const float * params = (const float *)dst->op_params;
    scale         = params[0];
    max_bias      = params[1];
    logit_softcap = params[2];

    bool use_q1_vec = false;
    bool use_q1_vec_mq = false;
    bool use_local_tile = false;
    // KV-head-coalesced gate: gqa_ratio == compile-time MQ_GQA
    // restricts to DK=DV=256 for now due to local memory size
    const int gqa_ratio_dispatch = n_head_kv > 0 ? (n_head / n_head_kv) : 0;
    if (n_q == 1) {
        if (use_native_q8_0_q1) {
            if (d_head_v >= 256 &&
                backend_ctx->fa.f32_q8_0_q1_vec.count(dk_dv) > 0) {
                kernel = backend_ctx->fa.f32_q8_0_q1_vec.at(dk_dv);
                use_q1_vec = true;
            } else {
                kernel = backend_ctx->fa.f32_q8_0_q1.at(dk_dv);
            }
        } else if (use_native_q4_0_q1) {
            // q4_0 vec kernel uses per-lane dp4a (cl_khr_integer_dot_product)
            const char * q4vec_env = getenv("GGML_OPENCL_FA_Q4_VEC");
            const bool   q4vec_off = (q4vec_env != NULL) && (q4vec_env[0] == '0');
            if (!q4vec_off && d_head_v >= 256 &&
                backend_ctx->fa.f32_q4_0_q1_vec.count(dk_dv) > 0) {
                kernel = backend_ctx->fa.f32_q4_0_q1_vec.at(dk_dv);
                use_q1_vec = true;
            } else {
                kernel = backend_ctx->fa.f32_q4_0_q1.at(dk_dv);
            }
        } else if (is_mixed) {
            static const char * lt_env = getenv("GGML_OPENCL_FA_LOCAL_TILE");
            static const bool   lt_on  = (lt_env != NULL) && (lt_env[0] != '0');
            if (lt_on && d_head_q == 128 && d_head_v == 128 &&
                backend_ctx->fa.f32_f16_q1_local_tile.count(dk_dv) > 0) {
                kernel = backend_ctx->fa.f32_f16_q1_local_tile.at(dk_dv);
                use_local_tile = true;
            } else {
                static const char * f16_vec_dk128_env = getenv("GGML_OPENCL_FA_F16_VEC_DK128");
                static const bool   f16_vec_dk128_off = (f16_vec_dk128_env != NULL) && (f16_vec_dk128_env[0] == '0');
                const int dv_gate = f16_vec_dk128_off ? 256 : 128;
                if (d_head_v >= dv_gate &&
                    backend_ctx->fa.f32_f16_q1_vec.count(dk_dv) > 0) {
                    kernel = backend_ctx->fa.f32_f16_q1_vec.at(dk_dv);
                    use_q1_vec = true;
                } else {
                    kernel = backend_ctx->fa.f32_f16_q1.at(dk_dv);
                }
            }
        } else if (is_f16) {
            kernel = backend_ctx->fa.f16_q1.at(dk_dv);
        } else {
            kernel = backend_ctx->fa.f32_q1.at(dk_dv);
        }
    } else {
        if (use_native_q8_0) {
            kernel = use_split_q8_0
                ? backend_ctx->fa.f32_q8_0_split.at(dk_dv)
                : backend_ctx->fa.f32_q8_0.at(dk_dv);
        } else if (use_native_q4_0) {
            kernel = use_split_q4_0
                ? backend_ctx->fa.f32_q4_0_split.at(dk_dv)
                : backend_ctx->fa.f32_q4_0.at(dk_dv);
        } else if (is_mixed) {
            if (use_split_kernel) {
                // DK=512 prefill: opt-in texture-cache K reads (image1d_buffer_t).
                static const char * pkimg_env = getenv("GGML_OPENCL_FA_PREFILL_K_IMG");
                const bool pkimg_on = (pkimg_env != NULL) && (pkimg_env[0] != '0');
                if (d_head_q == 512 && pkimg_on &&
                    backend_ctx->fa.f32_f16_split_k_img.count(dk_dv) > 0) {
                    kernel = backend_ctx->fa.f32_f16_split_k_img.at(dk_dv);
                    use_prefill_k_img = true;
                } else {
                    kernel = backend_ctx->fa.f32_f16_split.at(dk_dv);
                }
            } else {
                kernel = backend_ctx->fa.f32_f16.at(dk_dv);
            }
        } else if (is_f16) {
            kernel = backend_ctx->fa.f16.at(dk_dv);
        } else {
            kernel = backend_ctx->fa.f32.at(dk_dv);
        }
    }

    // Intel goes to the basic q1 kernel
    if (backend_ctx->gpu_family == INTEL && n_q == 1) {
        use_q1_vec = use_q1_vec_mq = use_local_tile = false;
        if (is_mixed && backend_ctx->fa.f32_f16_q1.count(dk_dv))      { kernel = backend_ctx->fa.f32_f16_q1.at(dk_dv); }
        else if (is_f16 && backend_ctx->fa.f16_q1.count(dk_dv))       { kernel = backend_ctx->fa.f16_q1.at(dk_dv); }
        else if (is_q8_0 && backend_ctx->fa.f32_q8_0_q1.count(dk_dv)) { kernel = backend_ctx->fa.f32_q8_0_q1.at(dk_dv); }
        else if (is_q4_0 && backend_ctx->fa.f32_q4_0_q1.count(dk_dv)) { kernel = backend_ctx->fa.f32_q4_0_q1.at(dk_dv); }
        else if (backend_ctx->fa.f32_q1.count(dk_dv))                 { kernel = backend_ctx->fa.f32_q1.at(dk_dv); }
    }
    GGML_ASSERT(kernel != NULL);

    ggml_cl_flash_attn_temp_buffer temp_k;
    ggml_cl_flash_attn_temp_buffer temp_v;
    ggml_cl_flash_attn_temp_buffer temp_k_pad;
    ggml_cl_flash_attn_temp_buffer temp_v_pad;
    ggml_cl_flash_attn_temp_buffer temp_mask_pad;
    ggml_cl_flash_attn_temp_buffer temp_blk;
    const ggml_type kv_target_type = is_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32;

    cl_mem k_data_device = k_soa ? NULL : extra_k->data_device;
    cl_mem v_data_device = v_soa ? NULL : extra_v->data_device;

    // SoA q8_0/q4_0 -> reconstruct AoS for downstream kernels that expect
    // tight records (no-op when k/v is already AoS).
    ggml_cl_flash_attn_temp_buffer temp_k_aos;
    ggml_cl_flash_attn_temp_buffer temp_v_aos;
    ggml_cl_flash_attn_reconstruct_aos(backend_ctx, k, temp_k_aos,
                                       k_data_device, offset_k, k_nb1, k_nb2, k_nb3);
    ggml_cl_flash_attn_reconstruct_aos(backend_ctx, v, temp_v_aos,
                                       v_data_device, offset_v, v_nb1, v_nb2, v_nb3);

    // currently FA kernels support KV cache with f16, f32, q4_0 and q8_0.
    // there two cases that these kernels cannot cover,
    //   1. KV cache types are q4_0 or q8_0, but the FA kernels fail to compile
    //   2. KV cache types not currently supported by an FA kernel, e.g., q4_1
    // these two cases are supported here by dequantizing to f32/f16 and this
    // causes performance degradation.
    // For q4_0 or q8_0 cases that fail kernel compilation, dequant happens in GPU;
    // for types that do not have FA kernels, dequant happens on host.
    if (!use_native_q8_0_q1 && !use_native_q8_0 &&
        !use_native_q4_0_q1 && !use_native_q4_0) {
        // for q4_0, q8_0 FA kernels that fail to compile
        bool k_done = false;
        bool v_done = false;
        if (k->type == GGML_TYPE_Q8_0 || k->type == GGML_TYPE_Q4_0) {
            k_done = ggml_cl_flash_attn_dequant_kv_gpu(
                backend_ctx, k, kv_target_type, k_data_device, offset_k, k_nb1, k_nb2, k_nb3,
                temp_k, k_data_device, offset_k, k_nb1, k_nb2, k_nb3);
        }
        if (v->type == GGML_TYPE_Q8_0 || v->type == GGML_TYPE_Q4_0) {
            v_done = ggml_cl_flash_attn_dequant_kv_gpu(
                backend_ctx, v, kv_target_type, v_data_device, offset_v, v_nb1, v_nb2, v_nb3,
                temp_v, v_data_device, offset_v, v_nb1, v_nb2, v_nb3);
        }
        if (!k_done) {
            ggml_cl_flash_attn_prepare_quantized_tensor(
                backend_ctx, k, kv_target_type, temp_k, k_data_device, offset_k, k_nb1, k_nb2, k_nb3);
        }
        if (!v_done) {
            ggml_cl_flash_attn_prepare_quantized_tensor(
                backend_ctx, v, kv_target_type, temp_v, v_data_device, offset_v, v_nb1, v_nb2, v_nb3);
        }
        // Asymmetric KV on the F32 fallback path: convert the F16 side to F32
        // too. (Symmetric F16 / mixed paths handle F16 directly.)
        if (kv_target_type == GGML_TYPE_F32 && !is_mixed && !is_f16) {
            ggml_cl_flash_attn_convert_f16_to_f32(backend_ctx, k, temp_k, k_data_device, offset_k, k_nb1, k_nb2, k_nb3);
            ggml_cl_flash_attn_convert_f16_to_f32(backend_ctx, v, temp_v, v_data_device, offset_v, v_nb1, v_nb2, v_nb3);
        }
    }

    cl_mem k_pad_buffer = NULL;
    cl_mem v_pad_buffer = NULL;
    cl_mem mask_pad_buffer = NULL;
    cl_mem blk_buffer = NULL;
    cl_ulong mask_pad_nb1 = 0;
    cl_ulong mask_pad_nb2 = 0;
    cl_ulong mask_pad_nb3 = 0;

    // Flash-Decoding K-split decision. Resolved here, before the prefill
    // prepass, because KV-pad and blk prepass are pure overhead when FD fires.
    // Do not infer causality from tensor shapes: a NULL mask means full
    // (bidirectional) attention, e.g. ViT encoders, where n_q == n_kv as well.
    // Causal attention in llama.cpp always comes with an explicit KQ mask.
    // Inferring is_causal here corrupted mmproj output on OpenCL (see #23800).
    const int is_causal = 0;
    const int fd_max_n_q = (d_head_q <= FD_MAX_DK_MULTI) ? FD_MAX_N_Q_MULTI : 1;
    cl_kernel fd_k_split = NULL;
    bool use_fd_mq = false;
    size_t fd_mq_wg = 256;  // MQ_GQA=4 kernel: Q1_WG_SIZE(64) * MQ_NSG_SPLIT(4)
    bool use_fa_k_img = false;  // K bound as image1d_buffer_t instead of (buf, offset)

    {
        const char * mq_env = getenv("GGML_OPENCL_FA_MQ");
        const bool mq_enabled = (mq_env == NULL) ? true : (mq_env[0] != '0');
        const bool mq_kv_ok   = is_mixed || is_q8_0 || is_q4_0;

        const char * lmq_env = getenv("GGML_OPENCL_FA_LOCAL_MQ_SPLIT");
        const bool   lmq_on  = (lmq_env != NULL) && (lmq_env[0] != '0');

        static const char * vec_nq_env = getenv("GGML_OPENCL_FA_VEC_NQ");
        static const int N_MAX_VEC_NQ  = (vec_nq_env != NULL && vec_nq_env[0] != '\0')
                                           ? atoi(vec_nq_env) : 1;

        const bool nq_in_vec_range = (n_q >= 1) && (n_q <= N_MAX_VEC_NQ);
        const bool nq1_only        = (n_q == 1);

        // Cluster-parallel decode default on for Adreno X2E/X1E
        static const int c8_env_state = []{
            const char * e = getenv("GGML_OPENCL_FA_C8");
            if (e == NULL || e[0] == '\0') { return -1; }
            return (e[0] != '0') ? 1 : 0;
        }();
        const bool c8_default_on = backend_ctx->adreno_gen == ADRENO_GPU_GEN::X2E ||
                                   backend_ctx->adreno_gen == ADRENO_GPU_GEN::X1E;
        const bool c8_f16_on = (c8_env_state >= 0) ? (c8_env_state == 1) : c8_default_on;
        // Quant-KV (q4_0/q8_0) GQA4 c8: default-on X2E + X1E
        const bool c8_quant_on = (c8_env_state >= 0) ? (c8_env_state == 1) : c8_default_on;
        if (mq_enabled && mq_kv_ok && nq_in_vec_range && !is_causal &&
            backend_ctx->gpu_family != INTEL &&
            !use_local_tile &&
            n_kv >= FD_MIN_N_KV &&
            backend_ctx->fa.f32_merge.count(dk_dv) > 0) {
            if (nq1_only && lmq_on && is_mixed && d_head_q == 128 && d_head_v == 128 &&
                gqa_ratio_dispatch == 8 &&
                backend_ctx->fa.f32_f16_q1_local_mq_split_g8.count(dk_dv) > 0) {
                fd_k_split = backend_ctx->fa.f32_f16_q1_local_mq_split_g8.at(dk_dv);
                use_fd_mq  = true;
                fd_mq_wg   = 64;
            } else if (nq1_only && lmq_on && is_mixed && d_head_q == 128 && d_head_v == 128 &&
                gqa_ratio_dispatch == 4 &&
                backend_ctx->fa.f32_f16_q1_local_mq_split.count(dk_dv) > 0) {
                fd_k_split = backend_ctx->fa.f32_f16_q1_local_mq_split.at(dk_dv);
                use_fd_mq  = true;
                fd_mq_wg   = 64;
            } else if (nq1_only && is_mixed && gqa_ratio_dispatch == 4 &&
                ((d_head_q == 256 && d_head_v == 256) ||
                 (d_head_q == 128 && d_head_v == 128)) &&
                backend_ctx->fa.f32_f16_q1_vec_mq_split.count(dk_dv) > 0) {
                const bool k_img_on = d_head_q == 128 && d_head_v == 128 &&
                                      getenv("GGML_OPENCL_FA_K_IMG") != NULL &&
                                      getenv("GGML_OPENCL_FA_K_IMG")[0] != '0' &&
                                      backend_ctx->fa.f32_f16_q1_vec_mq_split_k_img.count(dk_dv) > 0;
                // Cluster-parallel decode
                const bool c8_env = d_head_q == 128 && d_head_v == 128 && c8_f16_on;
                if (c8_env && backend_ctx->fa.f32_f16_q1_vec_mq_split_c8.count(dk_dv) > 0) {
                    fd_k_split = backend_ctx->fa.f32_f16_q1_vec_mq_split_c8.at(dk_dv);
                    use_fd_mq  = true;
                } else if (c8_env && backend_ctx->fa.f32_f16_q1_vec_mq_split_c8_ns2.count(dk_dv) > 0) {
                    fd_k_split = backend_ctx->fa.f32_f16_q1_vec_mq_split_c8_ns2.at(dk_dv);
                    use_fd_mq  = true;
                    fd_mq_wg   = 128;
                } else if (k_img_on) {
                    fd_k_split   = backend_ctx->fa.f32_f16_q1_vec_mq_split_k_img.at(dk_dv);
                    use_fd_mq    = true;
                    use_fa_k_img = true;
                } else {
                    fd_k_split = backend_ctx->fa.f32_f16_q1_vec_mq_split.at(dk_dv);
                    use_fd_mq  = true;
                }
            // Cluster-parallel decode, DK=DV=256 GQA=8
            } else if (nq1_only && is_mixed && gqa_ratio_dispatch == 8 &&
                d_head_q == 256 && d_head_v == 256 &&
                c8_env_state == 1 &&
                backend_ctx->fa.f32_f16_q1_vec_mq_split_g8_c32.count(dk_dv) > 0) {
                fd_k_split = backend_ctx->fa.f32_f16_q1_vec_mq_split_g8_c32.at(dk_dv);
                use_fd_mq  = true;
                fd_mq_wg   = 128;
            // Cluster-parallel decode for the g8
            } else if (is_mixed && gqa_ratio_dispatch == 8 &&
                d_head_q == 128 && d_head_v == 128 &&
                c8_f16_on &&
                (backend_ctx->fa.f32_f16_q1_vec_mq_split_g8_c8.count(dk_dv) > 0 ||
                 backend_ctx->fa.f32_f16_q1_vec_mq_split_g8_c8_ns2.count(dk_dv) > 0)) {
                if (backend_ctx->fa.f32_f16_q1_vec_mq_split_g8_c8.count(dk_dv) > 0) {
                    fd_k_split = backend_ctx->fa.f32_f16_q1_vec_mq_split_g8_c8.at(dk_dv);
                    fd_mq_wg   = 192;
                } else {
                    fd_k_split = backend_ctx->fa.f32_f16_q1_vec_mq_split_g8_c8_ns2.at(dk_dv);
                    fd_mq_wg   = 128;
                }
                use_fd_mq  = true;
            } else if (is_mixed && gqa_ratio_dispatch == 8 &&
                d_head_q == 128 && d_head_v == 128 &&
                getenv("GGML_OPENCL_FA_K_IMG") != NULL &&
                getenv("GGML_OPENCL_FA_K_IMG")[0] != '0' &&
                backend_ctx->fa.f32_f16_q1_vec_mq_split_g8_k_img.count(dk_dv) > 0) {
                fd_k_split   = backend_ctx->fa.f32_f16_q1_vec_mq_split_g8_k_img.at(dk_dv);
                use_fd_mq    = true;
                fd_mq_wg     = 192;
                use_fa_k_img = true;
            } else if (is_mixed && gqa_ratio_dispatch == 8 &&
                d_head_q == 128 && d_head_v == 128 &&
                backend_ctx->fa.f32_f16_q1_vec_mq_split_g8.count(dk_dv) > 0) {
                fd_k_split = backend_ctx->fa.f32_f16_q1_vec_mq_split_g8.at(dk_dv);
                use_fd_mq  = true;
                fd_mq_wg   = 192;
            } else if (nq1_only && is_q8_0 && gqa_ratio_dispatch == 8 &&
                d_head_q == 128 && d_head_v == 128 &&
                backend_ctx->fa.f32_q8_0_q1_vec_mq_split_g8.count(dk_dv) > 0) {
                fd_k_split = backend_ctx->fa.f32_q8_0_q1_vec_mq_split_g8.at(dk_dv);
                use_fd_mq  = true;
                fd_mq_wg   = 192;
            } else if (nq1_only && is_q8_0 && gqa_ratio_dispatch == 4 &&
                d_head_q == 128 && d_head_v == 128 &&
                backend_ctx->fa.f32_q8_0_q1_vec_mq_split.count(dk_dv) > 0) {
                // Cluster-parallel q8_0 GQA4
                if (c8_quant_on &&
                    backend_ctx->fa.f32_q8_0_q1_vec_mq_split_c8.count(dk_dv) > 0) {
                    fd_k_split = backend_ctx->fa.f32_q8_0_q1_vec_mq_split_c8.at(dk_dv);
                    fd_mq_wg   = 128;
                } else {
                    fd_k_split = backend_ctx->fa.f32_q8_0_q1_vec_mq_split.at(dk_dv);
                }
                use_fd_mq  = true;
            } else if (nq1_only && is_q4_0) {
                const char * q4_mq_env = getenv("GGML_OPENCL_FA_Q4_MQ");
                const bool   q4_mq_on  = (q4_mq_env != NULL) && (q4_mq_env[0] != '0');
                // Cluster-parallel q4_0
                const bool q4_c8_on = c8_env_state == 1 &&
                                      backend_ctx->fa.f32_q4_0_q1_vec_mq_split_g8_c8.count(dk_dv) > 0;
                if (q4_c8_on && gqa_ratio_dispatch == 8 &&
                    d_head_q == 64 && d_head_v == 64) {
                    fd_k_split = backend_ctx->fa.f32_q4_0_q1_vec_mq_split_g8_c8.at(dk_dv);
                    use_fd_mq  = true;
                    fd_mq_wg   = 128;
                } else if (q4_mq_on && gqa_ratio_dispatch == 8 &&
                    d_head_q == 128 && d_head_v == 128 &&
                    backend_ctx->fa.f32_q4_0_q1_vec_mq_split_g8.count(dk_dv) > 0) {
                    fd_k_split = backend_ctx->fa.f32_q4_0_q1_vec_mq_split_g8.at(dk_dv);
                    use_fd_mq  = true;
                    fd_mq_wg   = 192;
                } else if (gqa_ratio_dispatch == 4 &&
                    d_head_q == 128 && d_head_v == 128 &&
                    backend_ctx->fa.f32_q4_0_q1_vec_mq_split.count(dk_dv) > 0) {
                    // Cluster-parallel q4_0 GQA4
                    if (c8_quant_on &&
                        backend_ctx->fa.f32_q4_0_q1_vec_mq_split_c8.count(dk_dv) > 0) {
                        fd_k_split = backend_ctx->fa.f32_q4_0_q1_vec_mq_split_c8.at(dk_dv);
                        fd_mq_wg   = 128;
                    } else {
                        fd_k_split = backend_ctx->fa.f32_q4_0_q1_vec_mq_split.at(dk_dv);
                    }
                    use_fd_mq  = true;
                }
            }
        }
    }
    // Intel cluster-parallel decode FA
    if (fd_k_split == NULL && backend_ctx->gpu_family == INTEL && n_q == 1 && !is_causal &&
        is_mixed && gqa_ratio_dispatch == 4 && d_head_q == 128 && d_head_v == 128 &&
        n_kv >= FD_MIN_N_KV &&
        getenv("GGML_OPENCL_FA_C8") != NULL && getenv("GGML_OPENCL_FA_C8")[0] != '0' &&
        backend_ctx->fa.f32_merge.count(dk_dv) > 0) {
        if (backend_ctx->fa.f32_f16_q1_vec_mq_split_c8.count(dk_dv) > 0) {
            fd_k_split = backend_ctx->fa.f32_f16_q1_vec_mq_split_c8.at(dk_dv);
            use_fd_mq  = true;
            fd_mq_wg   = 128;
        } else if (backend_ctx->fa.f32_f16_q1_vec_mq_split_c8_ns2.count(dk_dv) > 0) {
            fd_k_split = backend_ctx->fa.f32_f16_q1_vec_mq_split_c8_ns2.at(dk_dv);
            use_fd_mq  = true;
            fd_mq_wg   = 64;
        }
    }
    if (fd_k_split == NULL &&
        n_q >= 1 && n_q <= fd_max_n_q && n_kv >= FD_MIN_N_KV && !is_causal &&
        d_head_q <= FD_MAX_DK &&
        backend_ctx->fa.f32_merge.count(dk_dv) > 0) {
        if (is_mixed && backend_ctx->fa.f32_f16_q1_split.count(dk_dv) > 0) {
            fd_k_split = backend_ctx->fa.f32_f16_q1_split.at(dk_dv);
        } else if (is_q8_0 && backend_ctx->fa.f32_q8_0_q1_split.count(dk_dv) > 0) {
            fd_k_split = backend_ctx->fa.f32_q8_0_q1_split.at(dk_dv);
        } else if (is_q4_0 && backend_ctx->fa.f32_q4_0_q1_split.count(dk_dv) > 0) {
            fd_k_split = backend_ctx->fa.f32_q4_0_q1_split.at(dk_dv);
        }
    }
    const bool use_fd = (fd_k_split != NULL);

    const int n_q_blocks = n_q > 1 ? (n_q + block_m - 1) / block_m : 0;
    const int n_kv_blocks = (n_kv > 0 && block_n > 0) ? (n_kv + block_n - 1) / block_n : 0;
    // KV pad + blk prepass are pure overhead when FD will fire - skip them.
    const bool use_mixed_prepass = is_mixed && n_q > 1 && !use_fd;
    // make sure prepass kernels are compiled
    const bool have_kv_pad = backend_ctx->fa.kv_pad_f16.count(dk_dv) > 0;
    const bool have_blk    = backend_ctx->fa.blk_f16.count(dk_dv) > 0;
    const bool use_kv_pad = use_mixed_prepass && (n_kv % block_n != 0) && have_kv_pad;
    // blk prepass: per-KV-tile mask class (0=masked, 1=mixed, 2=unmasked).
    // Consumed identically by f32_f16, q8_0 and q4_0 prefill kernels.
    const bool use_quant_prepass = (use_native_q8_0 || use_native_q4_0) && !use_fd;
    const bool use_blk_mask = (use_mixed_prepass || use_quant_prepass) && mask_buffer != NULL && have_blk;

    if (use_kv_pad) {
        cl_int err;

        const size_t k_pad_size = (size_t) k_nb1 * (size_t) block_n * (size_t) n_head_kv * (size_t) n_batch;
        temp_k_pad.data = clCreateBuffer(backend_ctx->context, CL_MEM_READ_WRITE, k_pad_size, NULL, &err);
        CL_CHECK(err);
        k_pad_buffer = temp_k_pad.data;

        const size_t v_pad_size = (size_t) v_nb1 * (size_t) block_n * (size_t) n_head_kv * (size_t) n_batch;
        temp_v_pad.data = clCreateBuffer(backend_ctx->context, CL_MEM_READ_WRITE, v_pad_size, NULL, &err);
        CL_CHECK(err);
        v_pad_buffer = temp_v_pad.data;

        cl_kernel kernel_kv_pad = backend_ctx->fa.kv_pad_f16.at(dk_dv);
        CL_CHECK(clSetKernelArg(kernel_kv_pad, 0, sizeof(cl_mem),    &k_data_device));
        CL_CHECK(clSetKernelArg(kernel_kv_pad, 1, sizeof(cl_ulong),  &offset_k));
        CL_CHECK(clSetKernelArg(kernel_kv_pad, 2, sizeof(cl_mem),    &v_data_device));
        CL_CHECK(clSetKernelArg(kernel_kv_pad, 3, sizeof(cl_ulong),  &offset_v));
        CL_CHECK(clSetKernelArg(kernel_kv_pad, 4, sizeof(cl_mem),    &k_pad_buffer));
        CL_CHECK(clSetKernelArg(kernel_kv_pad, 5, sizeof(cl_mem),    &v_pad_buffer));
        CL_CHECK(clSetKernelArg(kernel_kv_pad, 6, sizeof(int),       &n_kv));
        CL_CHECK(clSetKernelArg(kernel_kv_pad, 7, sizeof(int),       &n_head_kv));
        CL_CHECK(clSetKernelArg(kernel_kv_pad, 8, sizeof(int),       &n_batch));
        CL_CHECK(clSetKernelArg(kernel_kv_pad, 9, sizeof(cl_ulong),  &k_nb1));
        CL_CHECK(clSetKernelArg(kernel_kv_pad, 10, sizeof(cl_ulong), &k_nb2));
        CL_CHECK(clSetKernelArg(kernel_kv_pad, 11, sizeof(cl_ulong), &k_nb3));
        CL_CHECK(clSetKernelArg(kernel_kv_pad, 12, sizeof(cl_ulong), &v_nb1));
        CL_CHECK(clSetKernelArg(kernel_kv_pad, 13, sizeof(cl_ulong), &v_nb2));
        CL_CHECK(clSetKernelArg(kernel_kv_pad, 14, sizeof(cl_ulong), &v_nb3));

        size_t global_work_size[] = { (size_t) block_n, (size_t) n_head_kv, (size_t) n_batch };
        backend_ctx->enqueue_ndrange_kernel(kernel_kv_pad, 3, global_work_size, NULL, dst);

        if (mask_buffer != NULL) {
            mask_pad_nb1 = (cl_ulong) block_n * (cl_ulong) sizeof(ggml_fp16_t);
            mask_pad_nb2 = (cl_ulong) n_q * mask_pad_nb1;
            mask_pad_nb3 = (cl_ulong) mask_ne2 * mask_pad_nb2;

            const size_t mask_pad_size = (size_t) mask_ne3 * (size_t) mask_pad_nb3;
            temp_mask_pad.data = clCreateBuffer(backend_ctx->context, CL_MEM_READ_WRITE, mask_pad_size, NULL, &err);
            CL_CHECK(err);
            mask_pad_buffer = temp_mask_pad.data;

            cl_kernel kernel_mask_pad = backend_ctx->fa.mask_pad_f16.at(dk_dv);
            CL_CHECK(clSetKernelArg(kernel_mask_pad, 0, sizeof(cl_mem),    &mask_buffer));
            CL_CHECK(clSetKernelArg(kernel_mask_pad, 1, sizeof(cl_ulong),  &offset_mask));
            CL_CHECK(clSetKernelArg(kernel_mask_pad, 2, sizeof(cl_mem),    &mask_pad_buffer));
            CL_CHECK(clSetKernelArg(kernel_mask_pad, 3, sizeof(int),       &n_q));
            CL_CHECK(clSetKernelArg(kernel_mask_pad, 4, sizeof(int),       &n_kv));
            CL_CHECK(clSetKernelArg(kernel_mask_pad, 5, sizeof(cl_ulong),  &mask_nb1));
            CL_CHECK(clSetKernelArg(kernel_mask_pad, 6, sizeof(cl_ulong),  &mask_nb2));
            CL_CHECK(clSetKernelArg(kernel_mask_pad, 7, sizeof(cl_ulong),  &mask_nb3));
            CL_CHECK(clSetKernelArg(kernel_mask_pad, 8, sizeof(int),       &mask_ne2));
            CL_CHECK(clSetKernelArg(kernel_mask_pad, 9, sizeof(int),       &mask_ne3));

            size_t global_work_size_mask[] = { (size_t) block_n, (size_t) n_q, (size_t) (mask_ne2 * mask_ne3) };
            backend_ctx->enqueue_ndrange_kernel(kernel_mask_pad, 3, global_work_size_mask, NULL, dst);
        }
    }

    if (use_blk_mask) {
        cl_int err;
        const size_t blk_size = (size_t) n_kv_blocks * (size_t) n_q_blocks * (size_t) mask_ne2 * (size_t) mask_ne3;
        temp_blk.data = clCreateBuffer(backend_ctx->context, CL_MEM_READ_WRITE, blk_size, NULL, &err);
        if (err != CL_SUCCESS) {
            // Flush before retry - reclaim deferred driver deallocations.
            CL_CHECK(clFinish(backend_ctx->queue));
            temp_blk.data = clCreateBuffer(backend_ctx->context, CL_MEM_READ_WRITE, blk_size, NULL, &err);
        }
        CL_CHECK(err);
        blk_buffer = temp_blk.data;

        cl_kernel kernel_blk = backend_ctx->fa.blk_f16.at(dk_dv);
        CL_CHECK(clSetKernelArg(kernel_blk, 0, sizeof(cl_mem),    &mask_buffer));
        CL_CHECK(clSetKernelArg(kernel_blk, 1, sizeof(cl_ulong),  &offset_mask));
        CL_CHECK(clSetKernelArg(kernel_blk, 2, sizeof(cl_mem),    &blk_buffer));
        CL_CHECK(clSetKernelArg(kernel_blk, 3, sizeof(int),       &n_q));
        CL_CHECK(clSetKernelArg(kernel_blk, 4, sizeof(int),       &n_kv));
        CL_CHECK(clSetKernelArg(kernel_blk, 5, sizeof(cl_ulong),  &mask_nb1));
        CL_CHECK(clSetKernelArg(kernel_blk, 6, sizeof(cl_ulong),  &mask_nb2));
        CL_CHECK(clSetKernelArg(kernel_blk, 7, sizeof(cl_ulong),  &mask_nb3));
        CL_CHECK(clSetKernelArg(kernel_blk, 8, sizeof(int),       &mask_ne2));
        CL_CHECK(clSetKernelArg(kernel_blk, 9, sizeof(int),       &mask_ne3));

        size_t global_work_size_blk[] = { (size_t) n_kv_blocks, (size_t) n_q_blocks, (size_t) (mask_ne2 * mask_ne3) };
        backend_ctx->enqueue_ndrange_kernel(kernel_blk, 3, global_work_size_blk, NULL, dst);
    }

    const int n_head_log2_val = n_head > 0 ? 1u << (int)floorf(log2f((float)n_head)) : 0;
    const float n_head_log2_f = n_head_log2_val > 0 ? (float)n_head_log2_val : 1.0f;
    const float m0 = powf(2.0f, -(max_bias) / n_head_log2_f);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2_f);

    if (use_fd) {
        static const int fd_env_kv_per_split = []{
            const char * e = getenv("GGML_OPENCL_FD_KV_PER_SPLIT");
            return (e && e[0]) ? atoi(e) : 0;
        }();
        static const int fd_env_max_splits = []{
            const char * e = getenv("GGML_OPENCL_FD_MAX_SPLITS");
            return (e && e[0]) ? atoi(e) : 0;
        }();

        int fd_kv_per_split = use_fd_mq ? FD_MQ_KV_PER_SPLIT
                                        : (is_mixed ? FD_KV_PER_SPLIT_F16 : FD_KV_PER_SPLIT);
        int fd_max_splits   = use_fd_mq ? FD_MQ_MAX_SPLITS   : FD_MAX_SPLITS;
        if (fd_env_kv_per_split > 0) { fd_kv_per_split = fd_env_kv_per_split; }
        if (fd_env_max_splits   > 0) { fd_max_splits   = fd_env_max_splits; }
        int n_splits = (n_kv + fd_kv_per_split - 1) / fd_kv_per_split;
        if (n_splits < FD_MIN_SPLITS) { n_splits = FD_MIN_SPLITS; }
        if (n_splits > fd_max_splits) { n_splits = fd_max_splits; }
        const int kv_per_split = (n_kv + n_splits - 1) / n_splits;

        const int fa_partial_floats = 2 + d_head_v;
        const size_t partial_size_bytes =
            (size_t) n_batch * n_head * n_q * n_splits * fa_partial_floats * sizeof(float);

        ggml_cl_flash_attn_temp_buffer temp_partial;
        cl_int err;
        temp_partial.data = clCreateBuffer(backend_ctx->context, CL_MEM_READ_WRITE,
                                           partial_size_bytes, NULL, &err);
        if (err != CL_SUCCESS) {
            CL_CHECK(clFinish(backend_ctx->queue));
            temp_partial.data = clCreateBuffer(backend_ctx->context, CL_MEM_READ_WRITE,
                                               partial_size_bytes, NULL, &err);
        }
        CL_CHECK(err);

        cl_kernel k_split = fd_k_split;
        int argi = 0;
        CL_CHECK(clSetKernelArg(k_split, argi++, sizeof(cl_mem),   &extra_q->data_device));
        CL_CHECK(clSetKernelArg(k_split, argi++, sizeof(cl_ulong), &offset_q));
        if (use_fa_k_img) {
            const size_t nb00_bytes  = sizeof(uint16_t);
            const size_t k_bytes_span =
                (size_t)(n_kv > 0 ? n_kv - 1 : 0) * (size_t)k_nb1 +
                (size_t)(n_head_kv > 0 ? n_head_kv - 1 : 0) * (size_t)k_nb2 +
                (size_t)(n_batch > 0 ? n_batch - 1 : 0) * (size_t)k_nb3 +
                (size_t)d_head_q * nb00_bytes;
            const size_t k_bytes  = (k_bytes_span + 7) & ~(size_t)7;
            const size_t k_pixels = k_bytes >> 3;
            cl_mem k_img = nullptr;
            if (k_pixels > 0 && k_pixels <= backend_ctx->image_max_buffer_size) {
                k_img = ggml_cl_img_pool_get_or_create(
                    backend_ctx, backend_ctx->kq_img_pool,
                    k_data_device, offset_k, k_bytes, CL_HALF_FLOAT);
            }

            // if image creation fails, fallback to buffer based kernels
            if (k_img == nullptr) {
                if (gqa_ratio_dispatch == 4 &&
                    backend_ctx->fa.f32_f16_q1_vec_mq_split.count(dk_dv) > 0) {
                    k_split = backend_ctx->fa.f32_f16_q1_vec_mq_split.at(dk_dv);
                } else {
                    k_split = backend_ctx->fa.f32_f16_q1_vec_mq_split_g8.at(dk_dv);
                }
                use_fa_k_img = false;
                CL_CHECK(clSetKernelArg(k_split, argi++, sizeof(cl_mem),   &k_data_device));
                CL_CHECK(clSetKernelArg(k_split, argi++, sizeof(cl_ulong), &offset_k));
            } else {
                CL_CHECK(clSetKernelArg(k_split, argi++, sizeof(cl_mem),   &k_img));
            }
        } else {
            CL_CHECK(clSetKernelArg(k_split, argi++, sizeof(cl_mem),   &k_data_device));
            CL_CHECK(clSetKernelArg(k_split, argi++, sizeof(cl_ulong), &offset_k));
        }
        CL_CHECK(clSetKernelArg(k_split, argi++, sizeof(cl_mem),   &v_data_device));
        CL_CHECK(clSetKernelArg(k_split, argi++, sizeof(cl_ulong), &offset_v));
        CL_CHECK(clSetKernelArg(k_split, argi++, sizeof(float),    &scale));
        CL_CHECK(clSetKernelArg(k_split, argi++, sizeof(int),      &n_q));
        CL_CHECK(clSetKernelArg(k_split, argi++, sizeof(int),      &n_kv));
        CL_CHECK(clSetKernelArg(k_split, argi++, sizeof(int),      &n_head));
        CL_CHECK(clSetKernelArg(k_split, argi++, sizeof(cl_ulong), &q_nb1));
        CL_CHECK(clSetKernelArg(k_split, argi++, sizeof(cl_ulong), &q_nb2));
        CL_CHECK(clSetKernelArg(k_split, argi++, sizeof(cl_ulong), &q_nb3));
        CL_CHECK(clSetKernelArg(k_split, argi++, sizeof(cl_ulong), &k_nb1));
        CL_CHECK(clSetKernelArg(k_split, argi++, sizeof(cl_ulong), &k_nb2));
        CL_CHECK(clSetKernelArg(k_split, argi++, sizeof(cl_ulong), &k_nb3));
        CL_CHECK(clSetKernelArg(k_split, argi++, sizeof(cl_ulong), &v_nb1));
        CL_CHECK(clSetKernelArg(k_split, argi++, sizeof(cl_ulong), &v_nb2));
        CL_CHECK(clSetKernelArg(k_split, argi++, sizeof(cl_ulong), &v_nb3));
        CL_CHECK(clSetKernelArg(k_split, argi++, sizeof(float),    &max_bias));
        CL_CHECK(clSetKernelArg(k_split, argi++, sizeof(float),    &m0));
        CL_CHECK(clSetKernelArg(k_split, argi++, sizeof(float),    &m1));
        CL_CHECK(clSetKernelArg(k_split, argi++, sizeof(int),      &n_head_log2_val));
        CL_CHECK(clSetKernelArg(k_split, argi++, sizeof(float),    &logit_softcap));
        CL_CHECK(clSetKernelArg(k_split, argi++, sizeof(int),      &n_head_kv));
        CL_CHECK(clSetKernelArg(k_split, argi++, sizeof(cl_mem),   &mask_buffer));
        CL_CHECK(clSetKernelArg(k_split, argi++, sizeof(cl_ulong), &offset_mask));
        CL_CHECK(clSetKernelArg(k_split, argi++, sizeof(cl_ulong), &mask_nb1));
        CL_CHECK(clSetKernelArg(k_split, argi++, sizeof(cl_ulong), &mask_nb2));
        CL_CHECK(clSetKernelArg(k_split, argi++, sizeof(cl_ulong), &mask_nb3));
        CL_CHECK(clSetKernelArg(k_split, argi++, sizeof(int),      &mask_ne2));
        CL_CHECK(clSetKernelArg(k_split, argi++, sizeof(int),      &mask_ne3));
        CL_CHECK(clSetKernelArg(k_split, argi++, sizeof(cl_mem),   &temp_partial.data));
        CL_CHECK(clSetKernelArg(k_split, argi++, sizeof(int),      &n_splits));
        CL_CHECK(clSetKernelArg(k_split, argi++, sizeof(int),      &kv_per_split));

        // MQ split kernel uses MQ_NSG_SPLIT subgroups and one WG per (kv_head, batch, split)
        // matches Q1_WG_SIZE * NSG (MQ_GQA=4 -> 256; MQ_GQA=8 -> 192)
        const size_t fd_wg = use_fd_mq ? fd_mq_wg : 64;
        const size_t fd_head_dim = use_fd_mq
            ? (size_t)(n_head_kv * n_batch)
            : (size_t)(n_head     * n_batch);
        size_t fd_lws[3] = { fd_wg, 1, 1 };
        // gid(2) packs q_idx * n_splits + split_idx.
        size_t fd_gws[3] = { fd_wg, fd_head_dim, (size_t)(n_splits * n_q) };
        backend_ctx->enqueue_ndrange_kernel(k_split, 3, fd_gws, fd_lws, dst);

        cl_kernel k_merge = backend_ctx->fa.f32_merge.at(dk_dv);
        argi = 0;
        CL_CHECK(clSetKernelArg(k_merge, argi++, sizeof(cl_mem),   &temp_partial.data));
        CL_CHECK(clSetKernelArg(k_merge, argi++, sizeof(cl_mem),   &extra_o->data_device));
        CL_CHECK(clSetKernelArg(k_merge, argi++, sizeof(cl_ulong), &offset_o));
        CL_CHECK(clSetKernelArg(k_merge, argi++, sizeof(int),      &n_head));
        CL_CHECK(clSetKernelArg(k_merge, argi++, sizeof(int),      &n_splits));
        CL_CHECK(clSetKernelArg(k_merge, argi++, sizeof(cl_ulong), &o_nb1));
        CL_CHECK(clSetKernelArg(k_merge, argi++, sizeof(cl_ulong), &o_nb2));
        CL_CHECK(clSetKernelArg(k_merge, argi++, sizeof(cl_ulong), &o_nb3));
        CL_CHECK(clSetKernelArg(k_merge, argi++, sizeof(cl_mem),   &sinks_buffer));
        CL_CHECK(clSetKernelArg(k_merge, argi++, sizeof(cl_ulong), &offset_sinks));
        CL_CHECK(clSetKernelArg(k_merge, argi++, sizeof(int),      &n_q));

        const size_t merge_wg  = (size_t) (d_head_v / 4); // one lane per float4
        size_t merge_lws[3] = { merge_wg, 1, 1 };
        size_t merge_gws[3] = { merge_wg, (size_t)(n_head * n_batch), (size_t) n_q };
        backend_ctx->enqueue_ndrange_kernel(k_merge, 3, merge_gws, merge_lws, dst);
        return;
    }

    cl_mem prefill_k_img = nullptr;
    if (use_prefill_k_img) {
        const size_t nb00_bytes = sizeof(uint16_t);
        const size_t k_bytes_span =
            (size_t)(n_kv > 0 ? n_kv - 1 : 0) * (size_t)k_nb1 +
            (size_t)(n_head_kv > 0 ? n_head_kv - 1 : 0) * (size_t)k_nb2 +
            (size_t)(n_batch > 0 ? n_batch - 1 : 0) * (size_t)k_nb3 +
            (size_t)d_head_q * nb00_bytes;
        const size_t k_bytes  = (k_bytes_span + 7) & ~(size_t)7;
        const size_t k_pixels = k_bytes >> 3;
        if (k_pixels > 0 && k_pixels <= backend_ctx->image_max_buffer_size) {
            prefill_k_img = ggml_cl_img_pool_get_or_create(
                backend_ctx, backend_ctx->kq_img_pool,
                k_data_device, offset_k, k_bytes, CL_HALF_FLOAT);
        }
        if (prefill_k_img == nullptr) {
            kernel = backend_ctx->fa.f32_f16_split.at(dk_dv);
            use_prefill_k_img = false;
        }
    }

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),    &extra_q->data_device));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_ulong),  &offset_q));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),    use_prefill_k_img ? &prefill_k_img : &k_data_device));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_ulong),  &offset_k));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem),    &v_data_device));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_ulong),  &offset_v));
    CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_mem),    &extra_o->data_device));
    CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_ulong),  &offset_o));
    CL_CHECK(clSetKernelArg(kernel, 8, sizeof(float),     &scale));
    CL_CHECK(clSetKernelArg(kernel, 9, sizeof(int),       &n_q));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &n_kv));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &is_causal));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &n_head));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &q_nb1));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &q_nb2));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(cl_ulong), &q_nb3));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &k_nb1));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong), &k_nb2));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong), &k_nb3));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &v_nb1));
    CL_CHECK(clSetKernelArg(kernel, 20, sizeof(cl_ulong), &v_nb2));
    CL_CHECK(clSetKernelArg(kernel, 21, sizeof(cl_ulong), &v_nb3));
    CL_CHECK(clSetKernelArg(kernel, 22, sizeof(cl_ulong), &o_nb1));
    CL_CHECK(clSetKernelArg(kernel, 23, sizeof(cl_ulong), &o_nb2));
    CL_CHECK(clSetKernelArg(kernel, 24, sizeof(cl_ulong), &o_nb3));
    CL_CHECK(clSetKernelArg(kernel, 25, sizeof(float),    &max_bias));
    CL_CHECK(clSetKernelArg(kernel, 26, sizeof(float),    &m0));
    CL_CHECK(clSetKernelArg(kernel, 27, sizeof(float),    &m1));
    CL_CHECK(clSetKernelArg(kernel, 28, sizeof(int),      &n_head_log2_val));
    CL_CHECK(clSetKernelArg(kernel, 29, sizeof(float),    &logit_softcap));
    CL_CHECK(clSetKernelArg(kernel, 30, sizeof(int),      &n_head_kv));
    CL_CHECK(clSetKernelArg(kernel, 31, sizeof(cl_mem),   &mask_buffer));
    CL_CHECK(clSetKernelArg(kernel, 32, sizeof(cl_ulong), &offset_mask));
    CL_CHECK(clSetKernelArg(kernel, 33, sizeof(cl_ulong), &mask_nb1));
    CL_CHECK(clSetKernelArg(kernel, 34, sizeof(cl_ulong), &mask_nb2));
    CL_CHECK(clSetKernelArg(kernel, 35, sizeof(cl_ulong), &mask_nb3));
    CL_CHECK(clSetKernelArg(kernel, 36, sizeof(int),      &mask_ne2));
    CL_CHECK(clSetKernelArg(kernel, 37, sizeof(int),      &mask_ne3));
    CL_CHECK(clSetKernelArg(kernel, 38, sizeof(cl_mem),   &sinks_buffer));
    CL_CHECK(clSetKernelArg(kernel, 39, sizeof(cl_ulong), &offset_sinks));
    if (n_q > 1 && is_mixed) {
        CL_CHECK(clSetKernelArg(kernel, 40, sizeof(cl_mem),    &k_pad_buffer));
        CL_CHECK(clSetKernelArg(kernel, 41, sizeof(cl_mem),    &v_pad_buffer));
        CL_CHECK(clSetKernelArg(kernel, 42, sizeof(cl_mem),    &mask_pad_buffer));
        CL_CHECK(clSetKernelArg(kernel, 43, sizeof(cl_mem),    &blk_buffer));
        CL_CHECK(clSetKernelArg(kernel, 44, sizeof(int),       &n_kv_blocks));
        CL_CHECK(clSetKernelArg(kernel, 45, sizeof(cl_ulong),  &mask_pad_nb1));
        CL_CHECK(clSetKernelArg(kernel, 46, sizeof(cl_ulong),  &mask_pad_nb2));
        CL_CHECK(clSetKernelArg(kernel, 47, sizeof(cl_ulong),  &mask_pad_nb3));
    } else if (use_native_q8_0 || use_native_q4_0) {
        // arg 40 = blk classification buffer (NULL disables prepass opt).
        CL_CHECK(clSetKernelArg(kernel, 40, sizeof(cl_mem),    &blk_buffer));
    }

    if (n_q == 1) {
        if (use_local_tile) {
            const size_t lt_wg = 128;
            size_t local_work_size[]  = { lt_wg, 1, 1 };
            size_t global_work_size[] = { lt_wg, (size_t) n_head, (size_t) n_batch };
            backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
        } else {
            // q1_vec dispatches with NSG subgroups
            const size_t q1_wg = backend_ctx->gpu_family == INTEL ? 32 : 64;
            const size_t wg_size = use_q1_vec ? 256 : q1_wg;
            const size_t head_dim_global = use_q1_vec_mq
                ? (size_t)(n_head_kv * n_batch)
                : (size_t)(n_head     * n_batch);
            size_t local_work_size[] = { wg_size, 1 };
            size_t global_work_size[] = { wg_size, head_dim_global };
            backend_ctx->enqueue_ndrange_kernel(kernel, 2, global_work_size, local_work_size, dst);
        }
    } else if (use_native_q8_0 || use_native_q4_0) {
        // Native quant prefill. The split variant may override BLOCK_M
        // (e.g. DK=96 quant uses BM=16).
        const bool use_split = use_native_q8_0 ? use_split_q8_0 : use_split_q4_0;
        int    bm;
        size_t wg_size;
        if (use_split) {
            bm      = use_native_q8_0 ? backend_ctx->fa.f32_q8_0_split_bm.at(dk_dv)
                                      : backend_ctx->fa.f32_q4_0_split_bm.at(dk_dv);
            wg_size = use_native_q8_0 ? backend_ctx->fa.f32_q8_0_split_wg_size.at(dk_dv)
                                      : backend_ctx->fa.f32_q4_0_split_wg_size.at(dk_dv);
        } else {
            bm      = backend_ctx->fa.bm.at(dk_dv);
            wg_size = (size_t) bm;
        }
        size_t local_work_size[]  = { wg_size, 1 };
        size_t global_work_size[] = { (size_t)((n_q + bm - 1) / bm) * wg_size, (size_t)(n_head * n_batch) };
        backend_ctx->enqueue_ndrange_kernel(kernel, 2, global_work_size, local_work_size, dst);
    } else {
        const size_t wg_size = (size_t) wg_size_fa;
        size_t local_work_size[] = { wg_size, 1 };
        size_t global_work_size[] = { (size_t)((n_q + block_m - 1) / block_m) * wg_size, (size_t)(n_head * n_batch) };
        backend_ctx->enqueue_ndrange_kernel(kernel, 2, global_work_size, local_work_size, dst);
    }
}
