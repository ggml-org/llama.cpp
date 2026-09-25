#include "../cl-common.h"
#include "../ops.h"

void ggml_cl_load_kernels_mul_mat_adreno(ggml_backend_opencl_context * backend_ctx) {
#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
    cl_int err;
    const std::string & compile_opts = backend_ctx->kernel_compile_opts;
    const auto opencl_c_std =
        std::string("CL") + std::to_string(backend_ctx->opencl_c_version.major) + "." +
        std::to_string(backend_ctx->opencl_c_version.minor);

    // transpose
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "transpose.cl.h"
        };
#else
        const std::string kernel_src = read_file("transpose.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_transpose_32_16 = clCreateKernel(prog, "kernel_transpose_32_16", &err), err));
        CL_CHECK((backend_ctx->mul_mat.kernel_transpose_32    = clCreateKernel(prog, "kernel_transpose_32", &err), err));
        CL_CHECK((backend_ctx->mul_mat.kernel_transpose_16    = clCreateKernel(prog, "kernel_transpose_16", &err), err));
        CL_CHECK((backend_ctx->mul_mat.kernel_transpose_8_buf  = clCreateKernel(prog, "kernel_transpose_8_buf", &err), err));
        CL_CHECK((backend_ctx->mul_mat.kernel_transpose_16_buf = clCreateKernel(prog, "kernel_transpose_16_buf", &err), err));
        CL_CHECK((backend_ctx->mul_mat.kernel_transpose_32_buf = clCreateKernel(prog, "kernel_transpose_32_buf", &err), err));
        CL_CHECK((backend_ctx->mul_mat.kernel_transpose_16_4x1 = clCreateKernel(prog, "kernel_transpose_16_4x1", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemm_noshuffle_q1_0_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemm_noshuffle_q1_0_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemm_noshuffle_q1_0_f32.cl");
#endif
        cl_program prog = build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);
        CL_CHECK((backend_ctx->mul_mat.kernel_gemm_noshuffle_q1_0_f32 = clCreateKernel(prog, "kernel_gemm_noshuffle_q1_0_f32", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemv_noshuffle_q1_0_f32
    {
        std::string CL_gemv_compile_opts = std::string("-cl-std=") + opencl_c_std +
                                       " -cl-mad-enable "
                                       " -DSIMDGROUP_WIDTH=" +
                                       std::to_string(backend_ctx->adreno_wave_size);

#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src_CL_gemv_general {
            #include "gemv_noshuffle_q1_0_f32.cl.h"
        };
#else
        const std::string kernel_src_CL_gemv_general = read_file("gemv_noshuffle_q1_0_f32.cl");
#endif

        cl_program prog = build_program_from_source(backend_ctx, kernel_src_CL_gemv_general.c_str(), CL_gemv_compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_gemv_noshuffle_q1_0_f32 = clCreateKernel(prog, "kernel_gemv_noshuffle_q1_0_f32", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemv_noshuffle_general
    {
        std::string CL_gemv_compile_opts = std::string("-cl-std=") + opencl_c_std +
                                       " -cl-mad-enable "
                                       " -DSIMDGROUP_WIDTH=" +
                                       std::to_string(backend_ctx->adreno_wave_size);
        if (backend_ctx->has_vector_subgroup_broadcast) {
            CL_gemv_compile_opts += " -DVECTOR_SUB_GROUP_BROADCAST ";
        }

#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src_CL_gemv_general {
            #include "gemv_noshuffle_q4_0_f32.cl.h"
        };
#else
        const std::string kernel_src_CL_gemv_general = read_file("gemv_noshuffle_q4_0_f32.cl");
#endif

        cl_program prog = build_program_from_source(backend_ctx, kernel_src_CL_gemv_general.c_str(), CL_gemv_compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_gemv_noshuffle_q4_0_f32 = clCreateKernel(prog, "kernel_gemv_noshuffle_q4_0_f32", &err), err));
        CL_CHECK((backend_ctx->mul_mat.kernel_gemv_noshuffle_q4_0_f32_mc3 = clCreateKernel(prog, "kernel_gemv_noshuffle_q4_0_f32_mc3", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemv_noshuffle
    {
        // Gemv 2048, 16384
        std::string CL_gemv_compile_opts = std::string("-cl-std=") + opencl_c_std +
            " -cl-mad-enable "
            " -DLINE_STRIDE_A=2048 "
            " -DBLOCK_STRIDE_A=16384 "
            " -DSIMDGROUP_WIDTH=" +
            std::to_string(backend_ctx->adreno_wave_size);
        if (backend_ctx->has_vector_subgroup_broadcast) {
            CL_gemv_compile_opts += " -DVECTOR_SUB_GROUP_BROADCAST ";
        }

#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src_CL_gemv {
            #include "gemv_noshuffle_q4_0_f32_spec.cl.h"
        };
#else
        const std::string kernel_src_CL_gemv = read_file("gemv_noshuffle_q4_0_f32_spec.cl");
#endif

        cl_program prog = build_program_from_source(backend_ctx, kernel_src_CL_gemv.c_str(), CL_gemv_compile_opts);
        CL_CHECK((backend_ctx->mul_mat.kernel_gemv_noshuffle_q4_0_f32_4096_1_4096 = clCreateKernel(prog, "kernel_gemv_noshuffle_q4_0_f32", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");

        // Gemv 2048, 16384
        CL_gemv_compile_opts = std::string("-cl-std=") + opencl_c_std +
            " -cl-mad-enable "
            " -DLINE_STRIDE_A=2048 "
            " -DBLOCK_STRIDE_A=16384 "
            " -DSIMDGROUP_WIDTH=" +
            std::to_string(backend_ctx->adreno_wave_size);
        if (backend_ctx->has_vector_subgroup_broadcast) {
            CL_gemv_compile_opts += " -DVECTOR_SUB_GROUP_BROADCAST ";
        }

        prog = build_program_from_source(backend_ctx, kernel_src_CL_gemv.c_str(), CL_gemv_compile_opts);
        CL_CHECK((backend_ctx->mul_mat.kernel_gemv_noshuffle_q4_0_f32_4096_1_11008 = clCreateKernel(prog, "kernel_gemv_noshuffle_q4_0_f32", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");

        // Gemv 5504, 44032
        CL_gemv_compile_opts = std::string("-cl-std=") + opencl_c_std +
            " -cl-mad-enable "
            " -DLINE_STRIDE_A=5504 "
            " -DBLOCK_STRIDE_A=44032 "
            " -DSIMDGROUP_WIDTH=" +
            std::to_string(backend_ctx->adreno_wave_size);
        if (backend_ctx->has_vector_subgroup_broadcast) {
            CL_gemv_compile_opts += " -DVECTOR_SUB_GROUP_BROADCAST ";
        }

        prog = build_program_from_source(backend_ctx, kernel_src_CL_gemv.c_str(), CL_gemv_compile_opts);
        CL_CHECK((backend_ctx->mul_mat.kernel_gemv_noshuffle_q4_0_f32_11008_1_4096 = clCreateKernel(prog, "kernel_gemv_noshuffle_q4_0_f32", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");

        // Gemv 16000, 128000
        CL_gemv_compile_opts = std::string("-cl-std=") + opencl_c_std +
            " -cl-mad-enable "
            " -DLINE_STRIDE_A=16000 "
            " -DBLOCK_STRIDE_A=128000 "
            " -DSIMDGROUP_WIDTH=" +
            std::to_string(backend_ctx->adreno_wave_size);

        if (backend_ctx->has_vector_subgroup_broadcast) {
            CL_gemv_compile_opts += " -DVECTOR_SUB_GROUP_BROADCAST ";
        }

        prog = build_program_from_source(backend_ctx, kernel_src_CL_gemv.c_str(), CL_gemv_compile_opts);
        CL_CHECK((backend_ctx->mul_mat.kernel_gemv_noshuffle_q4_0_f32_32000_1_4096 = clCreateKernel(prog, "kernel_gemv_noshuffle_q4_0_f32", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mat_Ab_Bi_8x4
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src_CL_gemm {
            #include "gemm_noshuffle_q4_0_f32.cl.h"
        };
#else
        const std::string kernel_src_CL_gemm = read_file("gemm_noshuffle_q4_0_f32.cl");
#endif
        cl_program prog = build_program_from_source(backend_ctx, kernel_src_CL_gemm.c_str(), compile_opts);
        CL_CHECK((backend_ctx->mul_mat.kernel_gemm_noshuffle_q4_0_f32 = clCreateKernel(prog, "kernel_gemm_noshuffle_q4_0_f32", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    backend_ctx->mul_mat.kernel_gemv_noshuffle_q4_0_f32_32b_trans = nullptr;
    backend_ctx->mul_mat.kernel_gemm_noshuffle_q4_0_f32_32b_trans_ila_a8_bin = nullptr;
    backend_ctx->mul_mat.kernel_gemm_noshuffle_q4_0_q8_1_dp4a_ila_a8_bin = nullptr;
    if (backend_ctx->adreno_gen == ADRENO_GPU_GEN::X2E) {
        {
            std::string opts = std::string("-cl-std=") + opencl_c_std +
                                           " -cl-mad-enable "
                                           " -DSIMDGROUP_WIDTH=" +
                                           std::to_string(backend_ctx->adreno_wave_size);
#ifdef GGML_OPENCL_EMBED_KERNELS
            const std::string kernel_src {
                #include "gemv_noshuffle_q4_0_f32_32b_trans.cl.h"
            };
#else
            const std::string kernel_src = read_file("gemv_noshuffle_q4_0_f32_32b_trans.cl");
#endif
            cl_program prog = build_program_from_source(backend_ctx, kernel_src.c_str(), opts);
            CL_CHECK((backend_ctx->mul_mat.kernel_gemv_noshuffle_q4_0_f32_32b_trans =
                clCreateKernel(prog, "kernel_gemv_noshuffle_q4_0_f32_32b_trans", &err), err));
            CL_CHECK(clReleaseProgram(prog));
            GGML_LOG_CONT(".");
        }

        if (use_adreno_bin_kernels(backend_ctx)) {
            size_t bin_size = 0;
            const char * kernel_bin = (const char *)backend_ctx->get_adreno_bin_kernel("gemm_noshuffle_q4_0_f32_32b_trans_ila_a8", &bin_size);
            if (kernel_bin && bin_size > 0) {
                cl_program bin_prog =
                    build_program_from_binary(backend_ctx->context, backend_ctx->device, kernel_bin, "", bin_size);

                CL_CHECK((backend_ctx->mul_mat.kernel_gemm_noshuffle_q4_0_f32_32b_trans_ila_a8_bin =
                    clCreateKernel(bin_prog, "kernel_gemm_noshuffle_q4_0_f32_32b_trans_ila_a8", &err), err));
                CL_CHECK(clReleaseProgram(bin_prog));
                GGML_LOG_CONT(".");
            }

            kernel_bin = (const char *)backend_ctx->get_adreno_bin_kernel("gemm_noshuffle_q4_0_q8_1_dp4a_ila_a8", &bin_size);
            if (kernel_bin && bin_size > 0) {
                cl_program bin_prog =
                    build_program_from_binary(backend_ctx->context, backend_ctx->device, kernel_bin, "", bin_size);

                CL_CHECK((backend_ctx->mul_mat.kernel_gemm_noshuffle_q4_0_q8_1_dp4a_ila_a8_bin =
                    clCreateKernel(bin_prog, "kernel_gemm_noshuffle_q4_0_q8_1_dp4a_ila_a8", &err), err));
                CL_CHECK(clReleaseProgram(bin_prog));
                GGML_LOG_CONT(".");
            }
        }
    }

    // gemm_noshuffle_q4_1_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemm_noshuffle_q4_1_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemm_noshuffle_q4_1_f32.cl");
#endif
        cl_program prog = build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);
        CL_CHECK((backend_ctx->mul_mat.kernel_gemm_noshuffle_q4_1_f32 = clCreateKernel(prog, "kernel_gemm_noshuffle_q4_1_f32", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemv_noshuffle_q4_1_f32
    {
        std::string CL_gemv_compile_opts = std::string("-cl-std=") + opencl_c_std +
                                       " -cl-mad-enable ";
        if (backend_ctx->has_vector_subgroup_broadcast) {
            CL_gemv_compile_opts += " -DVECTOR_SUB_GROUP_BROADCAT ";
        }

#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemv_noshuffle_q4_1_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemv_noshuffle_q4_1_f32.cl");
#endif

        cl_program prog = build_program_from_source(backend_ctx, kernel_src.c_str(), CL_gemv_compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_gemv_noshuffle_q4_1_f32 = clCreateKernel(prog, "kernel_gemv_noshuffle_q4_1_f32", &err), err));
        CL_CHECK((backend_ctx->mul_mat.kernel_gemv_noshuffle_q4_1_f32_mc3 = clCreateKernel(prog, "kernel_gemv_noshuffle_q4_1_f32_mc3", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemm_noshuffle_q5_0_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemm_noshuffle_q5_0_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemm_noshuffle_q5_0_f32.cl");
#endif
        cl_program prog = build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);
        CL_CHECK((backend_ctx->mul_mat.kernel_gemm_noshuffle_q5_0_f32 = clCreateKernel(prog, "kernel_gemm_noshuffle_q5_0_f32", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemm_noshuffle_q5_0_q8_1_dp4a (dp4a dense q5_0 prefill GEMM)
    if (backend_ctx->has_integer_dot) {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemm_noshuffle_q5_0_q8_1_dp4a.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemm_noshuffle_q5_0_q8_1_dp4a.cl");
#endif
        cl_program prog = build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);
        CL_CHECK((backend_ctx->mul_mat.kernel_gemm_noshuffle_q5_0_q8_1_dp4a = clCreateKernel(prog, "kernel_gemm_noshuffle_q5_0_q8_1_dp4a", &err), err));
        CL_CHECK((backend_ctx->mul_mat.kernel_gemm_noshuffle_q5_0_q8_1_dp4a_wimg = clCreateKernel(prog, "kernel_gemm_noshuffle_q5_0_q8_1_dp4a_wimg", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemv_noshuffle_q5_0_f32
    {
        std::string CL_gemv_compile_opts = std::string("-cl-std=") + opencl_c_std +
                                       " -cl-mad-enable ";
        if (backend_ctx->has_vector_subgroup_broadcast) {
            CL_gemv_compile_opts += " -DVECTOR_SUB_GROUP_BROADCAST ";
        }

#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemv_noshuffle_q5_0_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemv_noshuffle_q5_0_f32.cl");
#endif
        cl_program prog = build_program_from_source(backend_ctx, kernel_src.c_str(), CL_gemv_compile_opts);
        CL_CHECK((backend_ctx->mul_mat.kernel_gemv_noshuffle_q5_0_f32 = clCreateKernel(prog, "kernel_gemv_noshuffle_q5_0_f32", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemm_noshuffle_q5_1_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemm_noshuffle_q5_1_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemm_noshuffle_q5_1_f32.cl");
#endif
        cl_program prog = build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);
        CL_CHECK((backend_ctx->mul_mat.kernel_gemm_noshuffle_q5_1_f32 = clCreateKernel(prog, "kernel_gemm_noshuffle_q5_1_f32", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemv_noshuffle_q5_1_f32
    {
        std::string CL_gemv_compile_opts = std::string("-cl-std=") + opencl_c_std +
                                       " -cl-mad-enable ";
        if (backend_ctx->has_vector_subgroup_broadcast) {
            CL_gemv_compile_opts += " -DVECTOR_SUB_GROUP_BROADCAST ";
        }

#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemv_noshuffle_q5_1_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemv_noshuffle_q5_1_f32.cl");
#endif
        cl_program prog = build_program_from_source(backend_ctx, kernel_src.c_str(), CL_gemv_compile_opts);
        CL_CHECK((backend_ctx->mul_mat.kernel_gemv_noshuffle_q5_1_f32 = clCreateKernel(prog, "kernel_gemv_noshuffle_q5_1_f32", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemm_noshuffle_iq4_nl_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemm_noshuffle_iq4_nl_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemm_noshuffle_iq4_nl_f32.cl");
#endif
        cl_program prog = build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);
        CL_CHECK((backend_ctx->mul_mat.kernel_gemm_noshuffle_iq4_nl_f32 = clCreateKernel(prog, "kernel_gemm_noshuffle_iq4_nl_f32", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemm_noshuffle_iq4_nl_q8_1_dp4a (dp4a dense IQ4_NL prefill GEMM)
    if (backend_ctx->has_integer_dot) {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemm_noshuffle_iq4_nl_q8_1_dp4a.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemm_noshuffle_iq4_nl_q8_1_dp4a.cl");
#endif
        cl_program prog = build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);
        CL_CHECK((backend_ctx->mul_mat.kernel_gemm_noshuffle_iq4_nl_q8_1_dp4a = clCreateKernel(prog, "kernel_gemm_noshuffle_iq4_nl_q8_1_dp4a", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemm_noshuffle_q4_0_q8_1_dp4a (dp4a dense q4_0 prefill GEMM)
    if (backend_ctx->has_integer_dot) {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemm_noshuffle_q4_0_q8_1_dp4a.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemm_noshuffle_q4_0_q8_1_dp4a.cl");
#endif
        cl_program prog = build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);
        CL_CHECK((backend_ctx->mul_mat.kernel_gemm_noshuffle_q4_0_q8_1_dp4a = clCreateKernel(prog, "kernel_gemm_noshuffle_q4_0_q8_1_dp4a", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemv_noshuffle_iq4_nl_f32
    {
        std::string CL_gemv_compile_opts = std::string("-cl-std=") + opencl_c_std +
                                       " -cl-mad-enable ";
        if (backend_ctx->has_vector_subgroup_broadcast) {
            CL_gemv_compile_opts += " -DVECTOR_SUB_GROUP_BROADCAST ";
        }

#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemv_noshuffle_iq4_nl_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemv_noshuffle_iq4_nl_f32.cl");
#endif

        cl_program prog = build_program_from_source(backend_ctx, kernel_src.c_str(), CL_gemv_compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_gemv_noshuffle_iq4_nl_f32 = clCreateKernel(prog, "kernel_gemv_noshuffle_iq4_nl_f32", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mm_q8_0_f32_8x4
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemm_noshuffle_q8_0_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemm_noshuffle_q8_0_f32.cl");
#endif
        cl_program prog = build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);
        CL_CHECK((backend_ctx->mul_mat.kernel_gemm_noshuffle_q8_0_f32 = clCreateKernel(prog, "kernel_gemm_noshuffle_q8_0_f32", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemm_noshuffle_q8_0_f32_bin
    {
        size_t bin_size = 0;
        backend_ctx->mul_mat.kernel_gemm_noshuffle_q8_0_f32_bin = nullptr;

        if (use_adreno_bin_kernels(backend_ctx)) {
            const char * kernel_bin = (const char *)backend_ctx->get_adreno_bin_kernel("gemm_noshuffle_q8_0_f32_ila", &bin_size);
            if (kernel_bin && bin_size > 0) {
                cl_program prog =
                    build_program_from_binary(backend_ctx->context, backend_ctx->device, kernel_bin, compile_opts, bin_size);

                CL_CHECK((backend_ctx->mul_mat.kernel_gemm_noshuffle_q8_0_f32_bin = clCreateKernel(prog, "kernel_gemm_noshuffle_q8_0_f32_ila", &err), err));
                CL_CHECK(clReleaseProgram(prog));
                GGML_LOG_CONT(".");
            }
        }
    }

    // gemv_noshuffle_general_q8_0_f32
    {
        std::string CL_gemv_compile_opts = std::string("-cl-std=") + opencl_c_std +
                                       " -cl-mad-enable "
                                       " -DSIMDGROUP_WIDTH=" +
                                       std::to_string(backend_ctx->adreno_wave_size);
        if (backend_ctx->has_vector_subgroup_broadcast) {
            CL_gemv_compile_opts += " -DVECTOR_SUB_GROUP_BROADCAT ";
        }

#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src_CL_gemv_general {
            #include "gemv_noshuffle_q8_0_f32.cl.h"
        };
#else
        const std::string kernel_src_CL_gemv_general = read_file("gemv_noshuffle_q8_0_f32.cl");
#endif

        cl_program prog = build_program_from_source(backend_ctx, kernel_src_CL_gemv_general.c_str(), CL_gemv_compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_gemv_noshuffle_q8_0_f32 = clCreateKernel(prog, "kernel_gemv_noshuffle_q8_0_f32", &err), err));
        CL_CHECK((backend_ctx->mul_mat.kernel_gemv_noshuffle_q8_0_f32_splitk = clCreateKernel(prog, "kernel_gemv_noshuffle_q8_0_f32_splitk", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemm_noshuffle_q4_k_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemm_noshuffle_q4_k_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemm_noshuffle_q4_k_f32.cl");
#endif
        cl_program prog = build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);
        CL_CHECK((backend_ctx->mul_mat.kernel_gemm_noshuffle_q4_k_f32 = clCreateKernel(prog, "kernel_gemm_noshuffle_q4_k_f32", &err), err));
        CL_CHECK((backend_ctx->mul_mat.kernel_gemm_noshuffle_q4_k_f32_r1 = clCreateKernel(prog, "kernel_gemm_noshuffle_q4_k_f32_r1", &err), err));
        CL_CHECK((backend_ctx->mul_mat.kernel_gemm_noshuffle_q4_k_f32_kimg = clCreateKernel(prog, "kernel_gemm_noshuffle_q4_k_f32_kimg", &err), err));
        CL_CHECK((backend_ctx->mul_mat.kernel_gemm_noshuffle_q4_k_f32_cok = clCreateKernel(prog, "kernel_gemm_noshuffle_q4_k_f32_cok", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemm_noshuffle_q4_k_q8_1_dp4a (dp4a dense prefill GEMM)
    if (backend_ctx->has_integer_dot) {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemm_noshuffle_q4_k_q8_1_dp4a.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemm_noshuffle_q4_k_q8_1_dp4a.cl");
#endif
        // Per-device dp4a dense tile. The X2-tuned TILESIZE_N=32 over-occupies LDS on
        // X1 (1152 B/WG -> few resident WGs); TILESIZE_N=8 (288 B) lifts occupancy on
        // X1, byte-identical. X2E keeps 32. Env override wins.
        int q4k_dp4a_ts = (backend_ctx->adreno_gen == ADRENO_GPU_GEN::X1E) ? 8 : 32;
        if (const char * e = getenv("GGML_OPENCL_Q4K_DP4A_TS")) { q4k_dp4a_ts = atoi(e); }
        std::string dp4a_opts = compile_opts + " -DTILESIZE_N=" + std::to_string(q4k_dp4a_ts);
        cl_program prog = build_program_from_source(backend_ctx, kernel_src.c_str(), dp4a_opts);
        CL_CHECK((backend_ctx->mul_mat.kernel_gemm_noshuffle_q4_k_q8_1_dp4a = clCreateKernel(prog, "kernel_gemm_noshuffle_q4_k_q8_1_dp4a", &err), err));
        CL_CHECK((backend_ctx->mul_mat.kernel_gemm_noshuffle_q4_k_q8_1_dp4a_wimg = clCreateKernel(prog, "kernel_gemm_noshuffle_q4_k_q8_1_dp4a_wimg", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemm_noshuffle_q8_0_q8_1_dp4a (dp4a dense q8_0 prefill GEMM)
    if (backend_ctx->has_integer_dot) {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemm_noshuffle_q8_0_q8_1_dp4a.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemm_noshuffle_q8_0_q8_1_dp4a.cl");
#endif
        cl_program prog = build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);
        CL_CHECK((backend_ctx->mul_mat.kernel_gemm_noshuffle_q8_0_q8_1_dp4a = clCreateKernel(prog, "kernel_gemm_noshuffle_q8_0_q8_1_dp4a", &err), err));
        CL_CHECK((backend_ctx->mul_mat.kernel_gemm_noshuffle_q8_0_q8_1_dp4a_wimg = clCreateKernel(prog, "kernel_gemm_noshuffle_q8_0_q8_1_dp4a_wimg", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemm_noshuffle_q5_k_q8_1_dp4a (dp4a dense prefill GEMM for q5_K)
    if (backend_ctx->has_integer_dot) {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemm_noshuffle_q5_k_q8_1_dp4a.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemm_noshuffle_q5_k_q8_1_dp4a.cl");
#endif
        cl_program prog = build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);
        CL_CHECK((backend_ctx->mul_mat.kernel_gemm_noshuffle_q5_k_q8_1_dp4a = clCreateKernel(prog, "kernel_gemm_noshuffle_q5_k_q8_1_dp4a", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemm_noshuffle_q6_k_q8_1_dp4a (dp4a dense prefill GEMM for q6_K ffn_down/output)
    if (backend_ctx->has_integer_dot) {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemm_noshuffle_q6_k_q8_1_dp4a.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemm_noshuffle_q6_k_q8_1_dp4a.cl");
#endif
        cl_program prog = build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);
        CL_CHECK((backend_ctx->mul_mat.kernel_gemm_noshuffle_q6_k_q8_1_dp4a = clCreateKernel(prog, "kernel_gemm_noshuffle_q6_k_q8_1_dp4a", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // quant_a_q8_1 (plain activation q8_1 pre-pass for the dense dp4a GEMM)
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "quant_a_q8_1.cl.h"
        };
#else
        const std::string kernel_src = read_file("quant_a_q8_1.cl");
#endif
        cl_program prog = build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);
        CL_CHECK((backend_ctx->mul_mat.kernel_quant_a_q8_1 = clCreateKernel(prog, "kernel_quant_a_q8_1", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemv_noshuffle_q4_k_f32
    {
        std::string CL_gemv_compile_opts = std::string("-cl-std=") + opencl_c_std +
                                       " -cl-mad-enable ";
        if (backend_ctx->has_vector_subgroup_broadcast) {
            CL_gemv_compile_opts += " -DVECTOR_SUB_GROUP_BROADCAST ";
        }
        // Opt-in: dequant-once-per-block mc3 verify GEMV (factors q4_K dequant
        // out of the 3-column loop; byte-identical, lower spill). A/B vs the
        // shipped inline mc3 in the same binary.
        if (getenv("GGML_OPENCL_Q4K_MC3_DQ")) {
            CL_gemv_compile_opts += " -DQ4K_MC3_DEQUANT_ONCE ";
        }
        // Opt-in: LDS-staged dequant mc3 verify GEMV (stages the dequantized
        // q4_K weights in __local instead of private regs that spill to slow
        // global on Adreno; byte-identical). A/B vs inline + dequant-once.
        if (getenv("GGML_OPENCL_Q4K_MC3_LDS")) {
            CL_gemv_compile_opts += " -DQ4K_MC3_DEQUANT_LDS ";
        }

#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemv_noshuffle_q4_k_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemv_noshuffle_q4_k_f32.cl");
#endif

        cl_program prog = build_program_from_source(backend_ctx, kernel_src.c_str(), CL_gemv_compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_gemv_noshuffle_q4_k_f32 = clCreateKernel(prog, "kernel_gemv_noshuffle_q4_k_f32", &err), err));
        CL_CHECK((backend_ctx->mul_mat.kernel_gemv_noshuffle_q4_k_f32_mc3 = clCreateKernel(prog, "kernel_gemv_noshuffle_q4_k_f32_mc3", &err), err));
        CL_CHECK((backend_ctx->mul_mat.kernel_gemv_noshuffle_q4_k_f32_splitk = clCreateKernel(prog, "kernel_gemv_noshuffle_q4_k_f32_splitk", &err), err));
        CL_CHECK((backend_ctx->mul_mat.kernel_gemv_splitk_reduce_f32 = clCreateKernel(prog, "kernel_gemv_splitk_reduce_f32", &err), err));
        CL_CHECK((backend_ctx->mul_mat.kernel_gemv_noshuffle_q4_k_f32_glu = clCreateKernel(prog, "kernel_gemv_noshuffle_q4_k_f32_glu", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    backend_ctx->mul_mat.kernel_gemv_noshuffle_q4_k_f32_32b_trans = nullptr;
    backend_ctx->mul_mat.kernel_gemm_noshuffle_q4_k_f32_32b_trans_ila_a8_bin = nullptr;
    backend_ctx->mul_mat.kernel_gemm_noshuffle_q4_k_q8_1_dp4a_ila_a8_bin = nullptr;
    if (backend_ctx->adreno_gen == ADRENO_GPU_GEN::X2E) {
        {
            std::string opts = std::string("-cl-std=") + opencl_c_std +
                                           " -cl-mad-enable "
                                           " -DSIMDGROUP_WIDTH=" +
                                           std::to_string(backend_ctx->adreno_wave_size);
#ifdef GGML_OPENCL_EMBED_KERNELS
            const std::string kernel_src {
                #include "gemv_noshuffle_q4_k_f32_32b_trans.cl.h"
            };
#else
            const std::string kernel_src = read_file("gemv_noshuffle_q4_k_f32_32b_trans.cl");
#endif
            cl_program prog = build_program_from_source(backend_ctx, kernel_src.c_str(), opts);
            CL_CHECK((backend_ctx->mul_mat.kernel_gemv_noshuffle_q4_k_f32_32b_trans =
                clCreateKernel(prog, "gemv_noshuffle_q4_k_f32_32b_trans", &err), err));
            CL_CHECK(clReleaseProgram(prog));
            GGML_LOG_CONT(".");
        }

        if (use_adreno_bin_kernels(backend_ctx)) {
            size_t bin_size = 0;
            const char * kernel_bin = (const char *)backend_ctx->get_adreno_bin_kernel("gemm_noshuffle_q4_k_f32_32b_trans_ila_a8", &bin_size);
            if (kernel_bin && bin_size > 0) {
                cl_program bin_prog =
                    build_program_from_binary(backend_ctx->context, backend_ctx->device, kernel_bin, "", bin_size);

                CL_CHECK((backend_ctx->mul_mat.kernel_gemm_noshuffle_q4_k_f32_32b_trans_ila_a8_bin =
                    clCreateKernel(bin_prog, "kernel_gemm_noshuffle_q4_k_f32_32b_trans_ila_a8", &err), err));
                CL_CHECK(clReleaseProgram(bin_prog));
                GGML_LOG_CONT(".");
            }

            kernel_bin = (const char *)backend_ctx->get_adreno_bin_kernel("gemm_noshuffle_q4_k_q8_1_dp4a_ila_a8", &bin_size);
            if (kernel_bin && bin_size > 0) {
                cl_program bin_prog =
                    build_program_from_binary(backend_ctx->context, backend_ctx->device, kernel_bin, "", bin_size);

                CL_CHECK((backend_ctx->mul_mat.kernel_gemm_noshuffle_q4_k_q8_1_dp4a_ila_a8_bin =
                    clCreateKernel(bin_prog, "kernel_gemm_noshuffle_q4_k_q8_1_dp4a_ila_a8", &err), err));
                CL_CHECK(clReleaseProgram(bin_prog));
                GGML_LOG_CONT(".");
            }
        }
    }

    std::string CL_moe_compile_opts = std::string("-cl-std=") + opencl_c_std +
            " -cl-mad-enable "
            " -cl-fast-relaxed-math";

    // gemv_noshuffle_q4_k_f32_o4 - 4-output-per-WI variant for the long-vocab
    // q4_K lm_head/embed GEMV (shares one activation read across 4 output rows).
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemv_noshuffle_q4_k_f32_o4.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemv_noshuffle_q4_k_f32_o4.cl");
#endif
        std::string CL_gemv_compile_opts = std::string("-cl-std=") + opencl_c_std + " -cl-mad-enable ";
        if (backend_ctx->has_vector_subgroup_broadcast) {
            CL_gemv_compile_opts += " -DVECTOR_SUB_GROUP_BROADCAST ";
        }
        cl_program prog = build_program_from_source(
            backend_ctx, kernel_src.c_str(), CL_gemv_compile_opts);
        CL_CHECK((backend_ctx->mul_mat.kernel_gemv_noshuffle_q4_k_f32_o4 = clCreateKernel(prog, "kernel_gemv_noshuffle_q4_k_f32_o4", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemv_noshuffle_q4_k_f32_tiled - tiled-wide canonical layout, default ON
    // (opt out: GGML_OPENCL_Q4K_GEMV_TILED=0; separate convert + GEMV; weights via __global).
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemv_noshuffle_q4_k_f32_tiled.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemv_noshuffle_q4_k_f32_tiled.cl");
#endif
        std::string compile_opts = std::string("-cl-std=") + opencl_c_std + " -cl-mad-enable ";
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);
        CL_CHECK((backend_ctx->mul_mat.kernel_gemv_noshuffle_q4_k_f32_tiled =
            clCreateKernel(prog, "kernel_gemv_noshuffle_q4_k_f32_tiled", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemv_moe_q4_1_f32_ns
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemv_moe_q4_1_f32_ns.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemv_moe_q4_1_f32_ns.cl");
#endif
        cl_program prog = build_program_from_source(backend_ctx, kernel_src.c_str(), CL_moe_compile_opts);

        CL_CHECK((backend_ctx->mul_mat_id.kernel_gemv_moe_q4_1_f32_ns = clCreateKernel(prog, "kernel_gemv_moe_q4_1_f32_ns", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemm_moe_q4_1_f32_ns
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemm_moe_q4_1_f32_ns.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemm_moe_q4_1_f32_ns.cl");
#endif
        cl_program prog = build_program_from_source(backend_ctx, kernel_src.c_str(), CL_moe_compile_opts);

        CL_CHECK((backend_ctx->mul_mat_id.kernel_gemm_moe_q4_1_f32_ns = clCreateKernel(prog, "kernel_gemm_moe_q4_1_f32_ns", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemm_moe_q4_1_f32_ns_bin
    {
        size_t bin_size = 0;
        backend_ctx->mul_mat_id.kernel_gemm_moe_q4_1_f32_ns_bin = nullptr;

        if (use_adreno_bin_kernels(backend_ctx)) {
            const char * kernel_bin = (const char *)backend_ctx->get_adreno_bin_kernel("gemm_moe_q4_1_f32_ns_ila", &bin_size);
            if (kernel_bin && bin_size > 0) {
                cl_program prog =
                    build_program_from_binary(backend_ctx->context, backend_ctx->device, kernel_bin, CL_moe_compile_opts, bin_size);

                CL_CHECK((backend_ctx->mul_mat_id.kernel_gemm_moe_q4_1_f32_ns_bin = clCreateKernel(prog, "kernel_gemm_moe_q4_1_f32_ns_ila", &err), err));
                CL_CHECK(clReleaseProgram(prog));
                GGML_LOG_CONT(".");
            }
        }
    }

    // gemv_moe_mxfp4_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemv_moe_mxfp4_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemv_moe_mxfp4_f32.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), CL_moe_compile_opts);

        CL_CHECK((backend_ctx->mul_mat_id.kernel_gemv_moe_mxfp4_f32 = clCreateKernel(prog, "kernel_gemv_moe_mxfp4_f32", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemm_moe_mxfp4_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemm_moe_mxfp4_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemm_moe_mxfp4_f32.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), CL_moe_compile_opts);

        CL_CHECK((backend_ctx->mul_mat_id.kernel_gemm_moe_mxfp4_f32 = clCreateKernel(prog, "kernel_gemm_moe_mxfp4_f32", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemv_moe_q4_0_f32_ns
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemv_moe_q4_0_f32_ns.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemv_moe_q4_0_f32_ns.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), CL_moe_compile_opts);

        CL_CHECK((backend_ctx->mul_mat_id.kernel_gemv_moe_q4_0_f32_ns = clCreateKernel(prog, "kernel_gemv_moe_q4_0_f32_ns", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemm_moe_q4_0_f32_ns
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemm_moe_q4_0_f32_ns.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemm_moe_q4_0_f32_ns.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), CL_moe_compile_opts);

        CL_CHECK((backend_ctx->mul_mat_id.kernel_gemm_moe_q4_0_f32_ns = clCreateKernel(prog, "kernel_gemm_moe_q4_0_f32_ns", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemm_moe_q4_0_f32_ns_bin
    {
        size_t bin_size = 0;
        backend_ctx->mul_mat_id.kernel_gemm_moe_q4_0_f32_ns_bin = nullptr;

        if (use_adreno_bin_kernels(backend_ctx)) {
            const char * kernel_bin = (const char *)backend_ctx->get_adreno_bin_kernel("gemm_moe_q4_0_f32_ns_ila", &bin_size);
            if (kernel_bin && bin_size > 0) {
                cl_program prog =
                    build_program_from_binary(backend_ctx->context, backend_ctx->device, kernel_bin, CL_moe_compile_opts, bin_size);

                CL_CHECK((backend_ctx->mul_mat_id.kernel_gemm_moe_q4_0_f32_ns_bin = clCreateKernel(prog, "kernel_gemm_moe_q4_0_f32_ns_ila", &err), err));
                CL_CHECK(clReleaseProgram(prog));
                GGML_LOG_CONT(".");
            }
        }
    }

    // gemm_moe_q8_0_f32_ns
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemm_moe_q8_0_f32_ns.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemm_moe_q8_0_f32_ns.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), CL_moe_compile_opts);

        CL_CHECK((backend_ctx->mul_mat_id.kernel_gemm_moe_q8_0_f32_ns = clCreateKernel(prog, "kernel_gemm_moe_q8_0_f32_ns", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemv_moe_q5_0_f32_ns
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemv_moe_q5_0_f32_ns.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemv_moe_q5_0_f32_ns.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), CL_moe_compile_opts);

        CL_CHECK((backend_ctx->mul_mat_id.kernel_gemv_moe_q5_0_f32_ns = clCreateKernel(prog, "kernel_gemv_moe_q5_0_f32_ns", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemm_moe_q5_0_f32_ns
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemm_moe_q5_0_f32_ns.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemm_moe_q5_0_f32_ns.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), CL_moe_compile_opts);

        CL_CHECK((backend_ctx->mul_mat_id.kernel_gemm_moe_q5_0_f32_ns = clCreateKernel(prog, "kernel_gemm_moe_q5_0_f32_ns", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemv_moe_q5_1_f32_ns
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemv_moe_q5_1_f32_ns.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemv_moe_q5_1_f32_ns.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), CL_moe_compile_opts);

        CL_CHECK((backend_ctx->mul_mat_id.kernel_gemv_moe_q5_1_f32_ns = clCreateKernel(prog, "kernel_gemv_moe_q5_1_f32_ns", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemm_moe_q5_1_f32_ns
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemm_moe_q5_1_f32_ns.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemm_moe_q5_1_f32_ns.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), CL_moe_compile_opts);

        CL_CHECK((backend_ctx->mul_mat_id.kernel_gemm_moe_q5_1_f32_ns = clCreateKernel(prog, "kernel_gemm_moe_q5_1_f32_ns", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemv_moe_q4_k_f32_ns
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemv_moe_q4_k_f32_ns.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemv_moe_q4_k_f32_ns.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), CL_moe_compile_opts);

        CL_CHECK((backend_ctx->mul_mat_id.kernel_gemv_moe_q4_k_f32_ns = clCreateKernel(prog, "kernel_gemv_moe_q4_k_f32_ns", &err), err));
        CL_CHECK((backend_ctx->mul_mat_id.kernel_gemv_moe_q4_k_f32_ns_wimg = clCreateKernel(prog, "kernel_gemv_moe_q4_k_f32_ns_wimg", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemm_moe_q4_k_f32_ns
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemm_moe_q4_k_f32_ns.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemm_moe_q4_k_f32_ns.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), CL_moe_compile_opts);

        CL_CHECK((backend_ctx->mul_mat_id.kernel_gemm_moe_q4_k_f32_ns = clCreateKernel(prog, "kernel_gemm_moe_q4_k_f32_ns", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemm_moe_q4_k_f32_ns_bin
    {
        size_t bin_size = 0;
        backend_ctx->mul_mat_id.kernel_gemm_moe_q4_k_f32_ns_bin = nullptr;

        if (use_adreno_bin_kernels(backend_ctx)) {
            const char * kernel_bin = (const char *)backend_ctx->get_adreno_bin_kernel("gemm_moe_q4_k_f32_ns_ila", &bin_size);
            if (kernel_bin && bin_size > 0) {
                cl_program prog =
                    build_program_from_binary(backend_ctx->context, backend_ctx->device, kernel_bin, CL_moe_compile_opts, bin_size);

                CL_CHECK((backend_ctx->mul_mat_id.kernel_gemm_moe_q4_k_f32_ns_bin = clCreateKernel(prog, "kernel_gemm_moe_q4_k_f32_ns_ila", &err), err));
                CL_CHECK(clReleaseProgram(prog));
                GGML_LOG_CONT(".");
            }
        }
    }

    // gemm_moe_q4_k_q8_1_dp4a (dp4a prefill GEMM)
    if (backend_ctx->has_integer_dot) {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemm_moe_q4_k_q8_1_dp4a.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemm_moe_q4_k_q8_1_dp4a.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), CL_moe_compile_opts);

        CL_CHECK((backend_ctx->mul_mat_id.kernel_gemm_moe_q4_k_q8_1_dp4a = clCreateKernel(prog, "kernel_gemm_moe_q4_k_q8_1_dp4a", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemm_moe_mxfp4_q8_1_dp4a (dp4a prefill GEMM)
    if (backend_ctx->has_integer_dot) {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemm_moe_mxfp4_q8_1_dp4a.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemm_moe_mxfp4_q8_1_dp4a.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), CL_moe_compile_opts);

        CL_CHECK((backend_ctx->mul_mat_id.kernel_gemm_moe_mxfp4_q8_1_dp4a = clCreateKernel(prog, "kernel_gemm_moe_mxfp4_q8_1_dp4a", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    if (backend_ctx->has_integer_dot) {
        size_t bin_size = 0;

        if (use_adreno_bin_kernels(backend_ctx)) {
            const char * kernel_bin = (const char *) backend_ctx->get_adreno_bin_kernel("gemm_moe_mxfp4_q8_1_dp4a_ila", &bin_size);
            if (kernel_bin && bin_size > 0) {
                cl_program prog = build_program_from_binary(backend_ctx->context, backend_ctx->device, kernel_bin, CL_moe_compile_opts, bin_size);
                CL_CHECK((backend_ctx->mul_mat_id.kernel_gemm_moe_mxfp4_q8_1_dp4a_bin = clCreateKernel(prog, "kernel_gemm_moe_mxfp4_q8_1_dp4a_ila", &err), err));
                CL_CHECK(clReleaseProgram(prog));
                GGML_LOG_CONT(".");
            }
        }
    }

    // gemm_moe_q4_0_q8_1_dp4a (dp4a prefill GEMM)
    if (backend_ctx->has_integer_dot) {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemm_moe_q4_0_q8_1_dp4a.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemm_moe_q4_0_q8_1_dp4a.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), CL_moe_compile_opts);

        CL_CHECK((backend_ctx->mul_mat_id.kernel_gemm_moe_q4_0_q8_1_dp4a = clCreateKernel(prog, "kernel_gemm_moe_q4_0_q8_1_dp4a", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    if (backend_ctx->has_integer_dot) {
        size_t bin_size = 0;

        if (use_adreno_bin_kernels(backend_ctx)) {
            const char * kernel_bin = (const char *) backend_ctx->get_adreno_bin_kernel("gemm_moe_q4_0_q8_1_dp4a_ila", &bin_size);
            if (kernel_bin && bin_size > 0) {
                cl_program prog = build_program_from_binary(backend_ctx->context, backend_ctx->device, kernel_bin, CL_moe_compile_opts, bin_size);
                CL_CHECK((backend_ctx->mul_mat_id.kernel_gemm_moe_q4_0_q8_1_dp4a_bin = clCreateKernel(prog, "kernel_gemm_moe_q4_0_q8_1_dp4a_ila", &err), err));
                CL_CHECK(clReleaseProgram(prog));
                GGML_LOG_CONT(".");
            }
        }
    }

    // gemm_moe_q8_1_dp4a (generic dp4a MoE GEMM; MOE_QT=80 -> q8_0 expert variant)
    if (backend_ctx->has_integer_dot) {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemm_moe_q8_1_dp4a.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemm_moe_q8_1_dp4a.cl");
#endif
        const std::string opts80 = CL_moe_compile_opts + " -DMOE_QT=80";
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), opts80.c_str());
        CL_CHECK((backend_ctx->mul_mat_id.kernel_gemm_moe_q8_1_dp4a_q80 = clCreateKernel(prog, "kernel_gemm_moe_q8_1_dp4a", &err), err));
        CL_CHECK(clReleaseProgram(prog));

        const std::string opts50 = CL_moe_compile_opts + " -DMOE_QT=50";
        cl_program prog50 =
            build_program_from_source(backend_ctx, kernel_src.c_str(), opts50.c_str());
        CL_CHECK((backend_ctx->mul_mat_id.kernel_gemm_moe_q8_1_dp4a_q50 = clCreateKernel(prog50, "kernel_gemm_moe_q8_1_dp4a", &err), err));
        CL_CHECK(clReleaseProgram(prog50));

        const std::string opts5 = CL_moe_compile_opts + " -DMOE_QT=5";
        cl_program prog5 =
            build_program_from_source(backend_ctx, kernel_src.c_str(), opts5.c_str());
        CL_CHECK((backend_ctx->mul_mat_id.kernel_gemm_moe_q8_1_dp4a_q5k = clCreateKernel(prog5, "kernel_gemm_moe_q8_1_dp4a", &err), err));
        CL_CHECK(clReleaseProgram(prog5));
        GGML_LOG_CONT(".");
    }

    // moe_reorder_quant_a_q8_1 (fused reorder + q8_1 quant)
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "moe_reorder_quant_a_q8_1.cl.h"
        };
#else
        const std::string kernel_src = read_file("moe_reorder_quant_a_q8_1.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), CL_moe_compile_opts);

        CL_CHECK((backend_ctx->mul_mat_id.kernel_moe_reorder_quant_a_q8_1 = clCreateKernel(prog, "kernel_moe_reorder_quant_a_q8_1", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemv_moe_q5_k_f32_ns
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemv_moe_q5_k_f32_ns.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemv_moe_q5_k_f32_ns.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), CL_moe_compile_opts);

        CL_CHECK((backend_ctx->mul_mat_id.kernel_gemv_moe_q5_k_f32_ns = clCreateKernel(prog, "kernel_gemv_moe_q5_k_f32_ns", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemm_moe_q5_k_f32_ns
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemm_moe_q5_k_f32_ns.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemm_moe_q5_k_f32_ns.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), CL_moe_compile_opts);

        CL_CHECK((backend_ctx->mul_mat_id.kernel_gemm_moe_q5_k_f32_ns = clCreateKernel(prog, "kernel_gemm_moe_q5_k_f32_ns", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemv_moe_q6_k_f32_ns
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemv_moe_q6_k_f32_ns.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemv_moe_q6_k_f32_ns.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), CL_moe_compile_opts);

        CL_CHECK((backend_ctx->mul_mat_id.kernel_gemv_moe_q6_k_f32_ns = clCreateKernel(prog, "kernel_gemv_moe_q6_k_f32_ns", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemm_moe_q6_k_f32_ns
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemm_moe_q6_k_f32_ns.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemm_moe_q6_k_f32_ns.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), CL_moe_compile_opts);

        CL_CHECK((backend_ctx->mul_mat_id.kernel_gemm_moe_q6_k_f32_ns = clCreateKernel(prog, "kernel_gemm_moe_q6_k_f32_ns", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemm_moe_q6_k_f32_ns_bin
    {
        size_t bin_size = 0;
        backend_ctx->mul_mat_id.kernel_gemm_moe_q6_k_f32_ns_bin = nullptr;

        if (use_adreno_bin_kernels(backend_ctx)) {
            const char * kernel_bin = (const char *)backend_ctx->get_adreno_bin_kernel("gemm_moe_q6_k_f32_ns_ila", &bin_size);
            if (kernel_bin && bin_size > 0) {
                cl_program prog =
                    build_program_from_binary(backend_ctx->context, backend_ctx->device, kernel_bin, CL_moe_compile_opts, bin_size);

                CL_CHECK((backend_ctx->mul_mat_id.kernel_gemm_moe_q6_k_f32_ns_bin = clCreateKernel(prog, "kernel_gemm_moe_q6_k_f32_ns_ila", &err), err));
                CL_CHECK(clReleaseProgram(prog));
                GGML_LOG_CONT(".");
            }
        }
    }

    // gemm_moe_q6_k_q8_1_dp4a (dp4a q6_K MoE prefill GEMM)
    if (backend_ctx->has_integer_dot) {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemm_moe_q6_k_q8_1_dp4a.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemm_moe_q6_k_q8_1_dp4a.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), CL_moe_compile_opts);

        CL_CHECK((backend_ctx->mul_mat_id.kernel_gemm_moe_q6_k_q8_1_dp4a = clCreateKernel(prog, "kernel_gemm_moe_q6_k_q8_1_dp4a", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemv_moe_mxfp4_f32_ns
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemv_moe_mxfp4_f32_ns.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemv_moe_mxfp4_f32_ns.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), CL_moe_compile_opts);

        CL_CHECK((backend_ctx->mul_mat_id.kernel_gemv_moe_mxfp4_f32_ns = clCreateKernel(prog, "kernel_gemv_moe_mxfp4_f32_ns", &err), err));
        CL_CHECK((backend_ctx->mul_mat_id.kernel_gemv_moe_mxfp4_f32_ns_wimg = clCreateKernel(prog, "kernel_gemv_moe_mxfp4_f32_ns_wimg", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemm_moe_mxfp4_f32_ns
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemm_moe_mxfp4_f32_ns.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemm_moe_mxfp4_f32_ns.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), CL_moe_compile_opts);

            CL_CHECK((backend_ctx->mul_mat_id.kernel_gemm_moe_mxfp4_f32_ns = clCreateKernel(prog, "kernel_gemm_moe_mxfp4_f32_ns", &err), err));
            CL_CHECK(clReleaseProgram(prog));
            GGML_LOG_CONT(".");
    }

    // gemm_moe_mxfp4_f32_ns_bin
    {
        size_t bin_size = 0;
        backend_ctx->mul_mat_id.kernel_gemm_moe_mxfp4_f32_ns_bin = nullptr;

        if (use_adreno_bin_kernels(backend_ctx)) {
            const char * kernel_bin = (const char *)backend_ctx->get_adreno_bin_kernel("gemm_moe_mxfp4_f32_ns_ila", &bin_size);
            if (kernel_bin && bin_size > 0) {
                cl_program prog =
                    build_program_from_binary(backend_ctx->context, backend_ctx->device, kernel_bin, CL_moe_compile_opts, bin_size);

                CL_CHECK((backend_ctx->mul_mat_id.kernel_gemm_moe_mxfp4_f32_ns_bin = clCreateKernel(prog, "kernel_gemm_moe_mxfp4_f32_ns_ila", &err), err));
                CL_CHECK(clReleaseProgram(prog));
                GGML_LOG_CONT(".");
            }
        }
    }

    // moe_reorder_b
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "moe_reorder_b.cl.h"
        };
#else
        const std::string kernel_src = read_file("moe_reorder_b.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), CL_moe_compile_opts);

        CL_CHECK((backend_ctx->mul_mat_id.kernel_moe_reorder_b = clCreateKernel(prog, "kernel_moe_reorder_b", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // moe_sort_by_expert
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "moe_sort_by_expert.cl.h"
        };
#else
        const std::string kernel_src = read_file("moe_sort_by_expert.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), CL_moe_compile_opts);

        CL_CHECK((backend_ctx->mul_mat_id.kernel_moe_histogram = clCreateKernel(prog, "kernel_moe_histogram", &err), err));
        CL_CHECK((backend_ctx->mul_mat_id.kernel_moe_scan = clCreateKernel(prog, "kernel_moe_scan", &err), err));
        CL_CHECK((backend_ctx->mul_mat_id.kernel_moe_fill = clCreateKernel(prog, "kernel_moe_fill", &err), err));
        CL_CHECK((backend_ctx->mul_mat_id.kernel_moe_scatter = clCreateKernel(prog, "kernel_moe_scatter", &err), err));
        CL_CHECK((backend_ctx->mul_mat_id.kernel_moe_scatter_stable = clCreateKernel(prog, "kernel_moe_scatter_stable", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    backend_ctx->mul_mat.kernel_gemv_noshuffle_q6_k_f32_32b_trans = nullptr;
    backend_ctx->mul_mat.kernel_gemm_noshuffle_q6_k_f32_32b_trans_ila_a8_bin = nullptr;
    backend_ctx->mul_mat.kernel_gemm_noshuffle_q6_k_q8_1_dp4a_ila_a8_bin = nullptr;
    if (backend_ctx->adreno_gen == ADRENO_GPU_GEN::X2E) {
        {
            std::string opts = std::string("-cl-std=") + opencl_c_std +
                                           " -cl-mad-enable "
                                           " -DSIMDGROUP_WIDTH=" +
                                           std::to_string(backend_ctx->adreno_wave_size);
#ifdef GGML_OPENCL_EMBED_KERNELS
            const std::string kernel_src {
                #include "gemv_noshuffle_q6_k_f32_32b_trans.cl.h"
            };
#else
            const std::string kernel_src = read_file("gemv_noshuffle_q6_k_f32_32b_trans.cl");
#endif
            cl_program prog = build_program_from_source(backend_ctx, kernel_src.c_str(), opts);
            CL_CHECK((backend_ctx->mul_mat.kernel_gemv_noshuffle_q6_k_f32_32b_trans =
                clCreateKernel(prog, "kernel_gemv_noshuffle_q6_k_f32_32b_trans", &err), err));
            CL_CHECK(clReleaseProgram(prog));
            GGML_LOG_CONT(".");
        }

        if (use_adreno_bin_kernels(backend_ctx)) {
            size_t bin_size = 0;
            const char * kernel_bin = (const char *)backend_ctx->get_adreno_bin_kernel("gemm_noshuffle_q6_k_f32_32b_trans_ila_a8", &bin_size);
            if (kernel_bin && bin_size > 0) {
                cl_program bin_prog =
                    build_program_from_binary(backend_ctx->context, backend_ctx->device, kernel_bin, "", bin_size);

                CL_CHECK((backend_ctx->mul_mat.kernel_gemm_noshuffle_q6_k_f32_32b_trans_ila_a8_bin =
                    clCreateKernel(bin_prog, "kernel_gemm_noshuffle_q6_k_f32_32b_trans_ila_a8", &err), err));
                CL_CHECK(clReleaseProgram(bin_prog));
                GGML_LOG_CONT(".");
            }

            kernel_bin = (const char *)backend_ctx->get_adreno_bin_kernel("gemm_noshuffle_q6_k_q8_1_dp4a_ila_a8", &bin_size);
            if (kernel_bin && bin_size > 0) {
                cl_program bin_prog =
                    build_program_from_binary(backend_ctx->context, backend_ctx->device, kernel_bin, "", bin_size);

                CL_CHECK((backend_ctx->mul_mat.kernel_gemm_noshuffle_q6_k_q8_1_dp4a_ila_a8_bin =
                    clCreateKernel(bin_prog, "kernel_gemm_noshuffle_q6_k_q8_1_dp4a_ila_a8", &err), err));
                CL_CHECK(clReleaseProgram(bin_prog));
                GGML_LOG_CONT(".");
            }
        }
    }

    // gemv_noshuffle_q6_k_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemv_noshuffle_q6_k_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemv_noshuffle_q6_k_f32.cl");
#endif

        std::string CL_gemv_compile_opts = std::string("-cl-std=") + opencl_c_std +
                                       " -cl-mad-enable ";
        if (backend_ctx->has_vector_subgroup_broadcast) {
            CL_gemv_compile_opts += " -DVECTOR_SUB_GROUP_BROADCAT ";
        }

        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), CL_gemv_compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_gemv_noshuffle_q6_K_f32 = clCreateKernel(prog, "kernel_gemv_noshuffle_q6_K_f32", &err), err));
        CL_CHECK((backend_ctx->mul_mat.kernel_gemv_noshuffle_q6_K_f32_mc3 = clCreateKernel(prog, "kernel_gemv_noshuffle_q6_K_f32_mc3", &err), err));
        if (getenv("GGML_OPENCL_MC3_PROBE")) {
            cl_ulong pm6 = 0, pm4 = 0; size_t wg6 = 0, wg4 = 0, mult = 0;
            clGetKernelWorkGroupInfo(backend_ctx->mul_mat.kernel_gemv_noshuffle_q6_K_f32_mc3, backend_ctx->device, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(pm6), &pm6, NULL);
            clGetKernelWorkGroupInfo(backend_ctx->mul_mat.kernel_gemv_noshuffle_q6_K_f32_mc3, backend_ctx->device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(wg6), &wg6, NULL);
            clGetKernelWorkGroupInfo(backend_ctx->mul_mat.kernel_gemv_noshuffle_q4_k_f32_mc3, backend_ctx->device, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(pm4), &pm4, NULL);
            clGetKernelWorkGroupInfo(backend_ctx->mul_mat.kernel_gemv_noshuffle_q4_k_f32_mc3, backend_ctx->device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(wg4), &wg4, NULL);
            clGetKernelWorkGroupInfo(backend_ctx->mul_mat.kernel_gemv_noshuffle_q6_K_f32_mc3, backend_ctx->device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(mult), &mult, NULL);
            fprintf(stderr, "[MC3-PROBE] q4K_mc3 private=%llu wg_cap=%zu | q6K_mc3 private=%llu wg_cap=%zu | pref_mult=%zu\n",
                          (unsigned long long)pm4, wg4, (unsigned long long)pm6, wg6, mult);
            fflush(stderr);
        }
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemv_noshuffle_q6_k_f32_o4 - 4-output-per-WI variant, opt-in via
    // GGML_OPENCL_Q6K_GEMV_O4=1 (~3x fewer dispatches on long-vocab lm_head).
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemv_noshuffle_q6_k_f32_o4.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemv_noshuffle_q6_k_f32_o4.cl");
#endif

        std::string CL_gemv_compile_opts = std::string("-cl-std=") + opencl_c_std +
                                       " -cl-mad-enable ";
        if (backend_ctx->has_vector_subgroup_broadcast) {
            CL_gemv_compile_opts += " -DVECTOR_SUB_GROUP_BROADCAT ";
        }

        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), CL_gemv_compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_gemv_noshuffle_q6_K_f32_o4 = clCreateKernel(prog, "kernel_gemv_noshuffle_q6_K_f32_o4", &err), err));
        CL_CHECK(clReleaseProgram(prog));

        // Global-read variant: weights read from __global coalesced instead of
        // image1d_buffer (the texture cache caps the streaming lm_head read
        // bandwidth). Opt-in via GGML_OPENCL_Q6K_GEMV_O4_GLOBAL.
        cl_program prog_g = build_program_from_source(backend_ctx, kernel_src.c_str(), CL_gemv_compile_opts + " -DQ6K_O4_GLOBAL");
        CL_CHECK((backend_ctx->mul_mat.kernel_gemv_noshuffle_q6_K_f32_o4_global =
            clCreateKernel(prog_g, "kernel_gemv_noshuffle_q6_K_f32_o4_global", &err), err));
        CL_CHECK(clReleaseProgram(prog_g));
        GGML_LOG_CONT(".");
    }

    // gemv_noshuffle_q6_k_f32_tiled - tiled-wide canonical layout, default ON
    // (opt out: GGML_OPENCL_Q6K_GEMV_TILED=0; separate convert + GEMV; weights via __global).
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemv_noshuffle_q6_k_f32_tiled.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemv_noshuffle_q6_k_f32_tiled.cl");
#endif
        std::string compile_opts = std::string("-cl-std=") + opencl_c_std + " -cl-mad-enable ";
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);
        CL_CHECK((backend_ctx->mul_mat.kernel_gemv_noshuffle_q6_K_f32_tiled =
            clCreateKernel(prog, "kernel_gemv_noshuffle_q6_K_f32_tiled", &err), err));
        CL_CHECK((backend_ctx->mul_mat.kernel_gemv_noshuffle_q6_K_f32_tiled_mc3 =
            clCreateKernel(prog, "kernel_gemv_noshuffle_q6_K_f32_tiled_mc3", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemm_noshuffle_q6_k_f32_tiled - batched (N>1) GEMM over the same tiled-wide
    // canonical layout, so batched lm_head/embed stays correct + on GPU.
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemm_noshuffle_q6_k_f32_tiled.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemm_noshuffle_q6_k_f32_tiled.cl");
#endif
        std::string compile_opts = std::string("-cl-std=") + opencl_c_std + " -cl-mad-enable ";
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);
        CL_CHECK((backend_ctx->mul_mat.kernel_gemm_noshuffle_q6_K_f32_tiled =
            clCreateKernel(prog, "kernel_gemm_noshuffle_q6_K_f32_tiled", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemm_noshuffle_q6_k_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemm_noshuffle_q6_k_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemm_noshuffle_q6_k_f32.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), CL_moe_compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_gemm_noshuffle_q6_K_f32 = clCreateKernel(prog, "kernel_gemm_noshuffle_q6_K_f32", &err), err));
        CL_CHECK((backend_ctx->mul_mat.kernel_gemm_noshuffle_q6_K_f32_cok = clCreateKernel(prog, "kernel_gemm_noshuffle_q6_K_f32_cok", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemv_noshuffle_q5_k_f32
    {
        std::string CL_gemv_compile_opts = std::string("-cl-std=") + opencl_c_std +
                                       " -cl-mad-enable ";
        if (backend_ctx->has_vector_subgroup_broadcast) {
            CL_gemv_compile_opts += " -DVECTOR_SUB_GROUP_BROADCAST ";
        }

#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemv_noshuffle_q5_k_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemv_noshuffle_q5_k_f32.cl");
#endif

        cl_program prog = build_program_from_source(backend_ctx, kernel_src.c_str(), CL_gemv_compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_gemv_noshuffle_q5_k_f32 = clCreateKernel(prog, "kernel_gemv_noshuffle_q5_k_f32", &err), err));
        CL_CHECK((backend_ctx->mul_mat.kernel_gemv_noshuffle_q5_k_f32_mc3 = clCreateKernel(prog, "kernel_gemv_noshuffle_q5_k_f32_mc3", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemm_noshuffle_q5_k_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemm_noshuffle_q5_k_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemm_noshuffle_q5_k_f32.cl");
#endif
        cl_program prog = build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);
        CL_CHECK((backend_ctx->mul_mat.kernel_gemm_noshuffle_q5_k_f32 = clCreateKernel(prog, "kernel_gemm_noshuffle_q5_k_f32", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // gemm_xmem_f16_f32_os8
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "gemm_xmem_f16_f32_os8.cl.h"
        };
#else
        const std::string kernel_src = read_file("gemm_xmem_f16_f32_os8.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_adreno_xmem_pack_src_f32 =
            clCreateKernel(prog, "adreno_xmem_pack_src_f32", &err), err));
        CL_CHECK((backend_ctx->mul_mat.kernel_adreno_xmem_prepack_weight_f16 =
            clCreateKernel(prog, "adreno_xmem_prepack_weight_f16", &err), err));
        CL_CHECK((backend_ctx->mul_mat.kernel_gemm_xmem_f16_f32_os8 =
            clCreateKernel(prog, "kernel_gemm_xmem_f16_f32_os8", &err), err));
        CL_CHECK((backend_ctx->mul_mat.kernel_adreno_xmem_store_dst_f32 =
            clCreateKernel(prog, "adreno_xmem_store_dst_f32", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mm_f16_f32_attn_adreno
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mm_f16_f32_attn_adreno.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mm_f16_f32_attn_adreno.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        cl_int err_x8gi = CL_SUCCESS;
        backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_x8_gqa4_img =
            clCreateKernel(prog, "kernel_mul_mat_f16_f32_l4_x8_gqa4_img", &err_x8gi);
        if (err_x8gi != CL_SUCCESS) { backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_x8_gqa4_img = nullptr; }

        cl_int err_x8gi_r4 = CL_SUCCESS;
        backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_x8_gqa_r4_img =
            clCreateKernel(prog, "kernel_mul_mat_f16_f32_l4_x8_gqa_r4_img", &err_x8gi_r4);
        if (err_x8gi_r4 != CL_SUCCESS) { backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_x8_gqa_r4_img = nullptr; }

        cl_int err_r2dk256 = CL_SUCCESS;
        backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_x8_gqa_r2_dk256_img =
            clCreateKernel(prog, "kernel_mul_mat_f16_f32_l4_x8_gqa_r2_dk256_img", &err_r2dk256);
        if (err_r2dk256 != CL_SUCCESS) { backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_x8_gqa_r2_dk256_img = nullptr; }

        cl_int err_y8gi = CL_SUCCESS;
        backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_y8_gqa_img =
            clCreateKernel(prog, "kernel_mul_mat_f16_f32_l4_y8_gqa_img", &err_y8gi);
        if (err_y8gi != CL_SUCCESS) { backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_y8_gqa_img = nullptr; }

        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mm_f16_f32_kq_kqv
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mm_f16_f32_kq_kqv.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mm_f16_f32_kq_kqv.cl");
#endif
        cl_program prog_kqv =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts+" -DKQV ");
        cl_program prog_kq =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mm_f16_f32_kqv = clCreateKernel(prog_kqv, "mul_mm_f16_f32_kqv", &err), err));
        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mm_f16_f32_kq = clCreateKernel(prog_kq, "mul_mm_f16_f32_kq", &err), err));
        CL_CHECK(clReleaseProgram(prog_kqv));
        CL_CHECK(clReleaseProgram(prog_kq));
        GGML_LOG_CONT(".");
    }

#else
    GGML_UNUSED(backend_ctx);
#endif // GGML_OPENCL_USE_ADRENO_KERNELS
}

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
bool ggml_cl_can_use_adreno_xmem_gemm_f16_f32(
        const ggml_backend_opencl_context * backend_ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * dst) {
    if (!backend_ctx->adreno_xmem_gemm_enabled) {
        return false;
    }
    if (backend_ctx->gpu_family != GPU_FAMILY::ADRENO) {
        return false;
    }
    if ((src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_BF16) ||
        src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 ||
        src1->ne[2] != 1 || src1->ne[3] != 1 ||
        dst->ne[2]  != 1 || dst->ne[3]  != 1) {
        return false;
    }
    const int K = src0->ne[0];
    const int M = src0->ne[1];
    const int N = src1->ne[1];
    if (src1->ne[0] != K || dst->ne[0] != M || dst->ne[1] != N) {
        return false;
    }
    if (N <= 1 || M < 64 || N < 16 || K < 64) {
        return false;
    }
    if ((K % 8) != 0) {
        return false;
    }
    const int kpack = K / 4;
    const int npack = CEIL_DIV(M, 4);
    if (static_cast<size_t>(N) > backend_ctx->image2d_max_width ||
        static_cast<size_t>(kpack) > backend_ctx->image2d_max_height) {
        return false;
    }
    if (static_cast<size_t>(N) > backend_ctx->image2d_max_width ||
        static_cast<size_t>(npack) > backend_ctx->image2d_max_height) {
        return false;
    }
    return true;
}

void ggml_cl_mul_mat_f16_f32_adreno_xmem(
        ggml_backend_t backend,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {
    ggml_backend_opencl_context * backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    const cl_ulong offset0 = extra0->offset + src0->view_offs;
    const cl_ulong offset1 = extra1->offset + src1->view_offs;
    const cl_ulong offsetd = extrad->offset + dst->view_offs;

    const int K = src0->ne[0];
    const int M = src0->ne[1];
    const int N = src1->ne[1];
    const int kpack = K / 4;
    const int npack = CEIL_DIV(M, 4);
    const int os = 8;

    const size_t xmem_bytes = 6144;
    const size_t weight_bytes = static_cast<size_t>(kpack) * static_cast<size_t>(npack) * 4u * sizeof(cl_half4);

    backend_ctx->prealloc_adreno_xmem_const.allocate(backend_ctx->context, xmem_bytes);

    cl_int err = CL_SUCCESS;
    cl_image_format fmt = {};
    fmt.image_channel_order = CL_RGBA;
    fmt.image_channel_data_type = CL_HALF_FLOAT;

    cl_image_desc desc_src = {};
    desc_src.image_type = CL_MEM_OBJECT_IMAGE2D;
    desc_src.image_width = static_cast<size_t>(N);
    desc_src.image_height = static_cast<size_t>(kpack);
    cl_mem src_img = clCreateImage(backend_ctx->context, CL_MEM_READ_WRITE, &fmt, &desc_src, nullptr, &err);
    CL_CHECK(err);

    cl_image_desc desc_dst = {};
    desc_dst.image_type = CL_MEM_OBJECT_IMAGE2D;
    desc_dst.image_width = static_cast<size_t>(N);
    desc_dst.image_height = static_cast<size_t>(npack);
    cl_mem dst_img = clCreateImage(backend_ctx->context, CL_MEM_READ_WRITE, &fmt, &desc_dst, nullptr, &err);
    CL_CHECK(err);

    cl_mem weights = clCreateBuffer(backend_ctx->context, CL_MEM_READ_WRITE, weight_bytes, nullptr, &err);
    CL_CHECK(err);

    cl_kernel prepack = backend_ctx->mul_mat.kernel_adreno_xmem_prepack_weight_f16;
    CL_CHECK(clSetKernelArg(prepack, 0, sizeof(cl_mem),   &weights));
    CL_CHECK(clSetKernelArg(prepack, 1, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(prepack, 2, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(prepack, 3, sizeof(int),      &K));
    CL_CHECK(clSetKernelArg(prepack, 4, sizeof(int),      &M));
    CL_CHECK(clSetKernelArg(prepack, 5, sizeof(int),      &kpack));
    CL_CHECK(clSetKernelArg(prepack, 6, sizeof(int),      &npack));
    CL_CHECK(clSetKernelArg(prepack, 7, sizeof(int),      &os));
    size_t lws = 256;
    size_t max_wg = backend_ctx->get_kernel_workgroup_size(prepack);
    if (lws > max_wg) {
        lws = max_wg;
    }
    size_t gws = CEIL_DIV(static_cast<size_t>(kpack) * static_cast<size_t>(npack), lws) * lws;
    backend_ctx->enqueue_ndrange_kernel(prepack, 1, &gws, &lws, dst);

    cl_kernel pack_src = backend_ctx->mul_mat.kernel_adreno_xmem_pack_src_f32;
    CL_CHECK(clSetKernelArg(pack_src, 0, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(pack_src, 1, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(pack_src, 2, sizeof(cl_mem),   &src_img));
    CL_CHECK(clSetKernelArg(pack_src, 3, sizeof(int),      &K));
    CL_CHECK(clSetKernelArg(pack_src, 4, sizeof(int),      &N));
    size_t pack_src_lws[2] = { 16, 16 };
    size_t pack_src_gws[2] = {
        CEIL_DIV(static_cast<size_t>(N), pack_src_lws[0])*pack_src_lws[0],
        CEIL_DIV(static_cast<size_t>(kpack), pack_src_lws[1])*pack_src_lws[1]
    };
    backend_ctx->enqueue_ndrange_kernel(pack_src, 2, pack_src_gws, pack_src_lws, dst);

    cl_kernel gemm = backend_ctx->mul_mat.kernel_gemm_xmem_f16_f32_os8;
    CL_CHECK(clSetKernelArg(gemm, 0, sizeof(cl_mem), &weights));
    CL_CHECK(clSetKernelArg(gemm, 1, sizeof(cl_mem), &backend_ctx->prealloc_adreno_xmem_const.buffer));
    CL_CHECK(clSetKernelArg(gemm, 2, sizeof(cl_mem), &src_img));
    CL_CHECK(clSetKernelArg(gemm, 3, sizeof(cl_mem), &dst_img));
    CL_CHECK(clSetKernelArg(gemm, 4, sizeof(int),    &N));
    CL_CHECK(clSetKernelArg(gemm, 5, sizeof(int),    &npack));
    CL_CHECK(clSetKernelArg(gemm, 6, sizeof(int),    &kpack));
    const size_t z_values = CEIL_DIV(static_cast<size_t>(npack), static_cast<size_t>(os));
    size_t gemm_lws[3] = { 64, 1, 1 };
    size_t gemm_gws[3] = {
        z_values*gemm_lws[0],
        CEIL_DIV(static_cast<size_t>(N), gemm_lws[0]),
        1
    };
    backend_ctx->enqueue_ndrange_kernel(gemm, 3, gemm_gws, gemm_lws, dst);

    cl_kernel store_dst = backend_ctx->mul_mat.kernel_adreno_xmem_store_dst_f32;
    CL_CHECK(clSetKernelArg(store_dst, 0, sizeof(cl_mem),   &dst_img));
    CL_CHECK(clSetKernelArg(store_dst, 1, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(store_dst, 2, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(store_dst, 3, sizeof(int),      &M));
    CL_CHECK(clSetKernelArg(store_dst, 4, sizeof(int),      &N));
    size_t store_lws[2] = { 16, 16 };
    size_t store_gws[2] = {
        CEIL_DIV(static_cast<size_t>(N), store_lws[0])*store_lws[0],
        CEIL_DIV(static_cast<size_t>(npack), store_lws[1])*store_lws[1]
    };
    backend_ctx->enqueue_ndrange_kernel(store_dst, 2, store_gws, store_lws, dst);

    CL_CHECK(clReleaseMemObject(weights));
    CL_CHECK(clReleaseMemObject(dst_img));
    CL_CHECK(clReleaseMemObject(src_img));
}
#endif // GGML_OPENCL_USE_ADRENO_KERNELS

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
bool ggml_cl_mul_mat_f16_f32_attn_adreno(
        ggml_backend_t backend,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst
) {
    const enum ggml_type src0t = (src0->type == GGML_TYPE_BF16) ? GGML_TYPE_F16 : src0->type;
    const enum ggml_type src1t = src1->type;

    ggml_backend_opencl_context * backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    const cl_ulong offset0 = extra0->offset + src0->view_offs;
    const cl_ulong offset1 = extra1->offset + src1->view_offs;
    const cl_ulong offsetd = extrad->offset + dst->view_offs;

    GGML_TENSOR_LOCALS(int,      ne0, src0, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb0, src0, nb);
    GGML_TENSOR_LOCALS(int,      ne1, src1, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb1, src1, nb);
    GGML_TENSOR_LOCALS(int,      ne,  dst,  ne);

    const int r2 = ne12 / ne02;
    const int r3 = ne13 / ne03;

    if(src0t == GGML_TYPE_F16 && src1t == GGML_TYPE_F32){
        // Two tiling assumptions these kernels make but nothing enforced:
        //
        //   ne00 % TILESIZE_K(16): the K loop has no tail, so a K that does not
        //   divide folds 1-15 rows of whatever follows the operands into every
        //   output.
        //
        //   ne01 % TILESIZE_M(64): mm_store_c_N guards the n direction with its
        //   `mask` argument but nothing guards m -- the store walks all 64 rows
        //   of the tile at a stride of M. When M does not divide, the last tile
        //   does not run off the end of the buffer, it writes 64 - (M % 64)
        //   values ON TOP OF the next column, so the result is silently wrong.
        //   Reachable on the KQV side for any head size >= 64 that is not a
        //   multiple of it (80, 96, 112).
        //
        // Attention shapes in the graph satisfy both -- head sizes are multiples
        // of 64 and n_kv is padded -- which is why this has stayed latent.
        // Declining leaves the odd shapes on the generic GEMM, which handles them.
        if (ne01 >= 64 && ne1 >= 32 && ne00 >= 16 &&
            (ne00 % 16) == 0 && (ne01 % 64) == 0 && (ne12 % ne02) == 0  &&
            // the KQ/KQV image kernels do not handle dim 3 (multi-stream batches)
            ne03 == 1 && ne13 == 1 &&
            // dst is wrapped with image1d_buffer, the size limit applies, also src0
            (ne0 * ne1 * dst->ne[2] * dst->nb[0] / 4 <= backend_ctx->image_max_buffer_size)) {
            // For KQ.
            //
            // Layout admission, mirroring the KQV arm below. The KQ kernel takes
            // no stride arguments for A or B: it derives them as K*D_A*2 and
            // K*D_B*4, i.e. it assumes both operands pack exactly D heads of K
            // elements per row. Every real KV-cache view and permuted-Q view
            // does, but a view spanning part of a wider allocation does not, and
            // the kernel then walks the wrong rows with nothing to range-check
            // it. Gate on the packed layout itself rather than on the stride
            // ORDERING, which a wider parent satisfies just as well.
            const bool kq_packed_a = (nb01 == (cl_ulong)ne00 * ne02 * ggml_type_size(src0t)) &&
                                     (nb02 == (cl_ulong)ne00 * ggml_type_size(src0t));
            const bool kq_packed_b = (nb11 == (cl_ulong)ne10 * ne12 * ggml_type_size(src1t)) &&
                                     (nb12 == (cl_ulong)ne10 * ggml_type_size(src1t));
            //
            // ggml_is_permuted(src0) stands in for "K is head-major", but it is
            // only a proxy and it COLLAPSES at n_head_kv == 1: with a single
            // head there is no head stride to be out of order, so nb01 == nb02
            // and the view reports itself unpermuted. Such a KQ was declined
            // here and fell through to the generic GEMM (gemma-4 E2B, and any
            // other multi-query model). The packed check above is the contract
            // the kernel actually needs -- it pins both strides exactly -- so
            // require permutedness only where there is more than one head for
            // it to mean anything.
            //
            // Default on; GGML_OPENCL_KQ_NHEAD_KV1=0 restores the old proxy so
            // the two routings can be compared in one binary.
            static const char * kq_nhkv1_env = getenv("GGML_OPENCL_KQ_NHEAD_KV1");
            static const bool   kq_nhkv1_on  =
                (kq_nhkv1_env == nullptr || kq_nhkv1_env[0] != '0');
            if ((ggml_is_permuted(src0) || (ne02 == 1 && kq_nhkv1_on)) && ggml_is_permuted(src1) &&
                kq_packed_a && kq_packed_b &&
                ((nb01 * ne01 / 4)/4 <= backend_ctx->image_max_buffer_size) &&
                nb00 <= nb02 &&
                nb02 <= nb01 &&
                nb01 <= nb03 &&
                nb10 <= nb12 &&
                nb12 <= nb11 &&
                nb11 <= nb13) {
                ggml_cl_mul_mat_kq_kqv_adreno(backend, src0, src1, dst, /*is_kq =*/ true);
                return true;
            }
            // For KQV. Reaching this arm is what makes the op a KQV; the callee
            // is told so explicitly rather than re-deriving it from the strides
            // the arm above has already ruled on.
            if (!ggml_is_contiguous(src0) && ggml_is_contiguous(src1) &&
                ((nb02 * ne02 / 4)/4 <= backend_ctx->image_max_buffer_size)) {
                ggml_cl_mul_mat_kq_kqv_adreno(backend, src0, src1, dst, /*is_kq =*/ false);
                return true;
            }
        }

        static const char * mm_kq_gqa_img_env = getenv("GGML_OPENCL_MM_KQ_GQA_IMG");
        static const bool mm_kq_gqa_img_on = (mm_kq_gqa_img_env == nullptr || mm_kq_gqa_img_env[0] != '0');
        static const char * mm_kq_gqa_r4_img_env = getenv("GGML_OPENCL_MM_KQ_GQA_R4_IMG");
        static const bool mm_kq_gqa_r4_img_on = (mm_kq_gqa_r4_img_env == nullptr || mm_kq_gqa_r4_img_env[0] != '0');
        const bool img_r4_gate =
            mm_kq_gqa_r4_img_on &&
            backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_x8_gqa_r4_img != nullptr &&
            ne11 == 1 && ne01 >= 64 && (ne01 % 16) == 0 && ne00 == 128 &&
            (ne12 % ne02) == 0 && (ne12 / ne02) == 4 && (ne13 / ne03) == 1;
        if (mm_kq_gqa_img_on &&
            backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_x8_gqa4_img != nullptr &&
            ne11 == 1 && ne01 >= 64 && (ne01 % 16) == 0 && ne00 == 128 &&
            (ne12 % ne02) == 0 && (ne12 / ne02) == 8 && (ne13 / ne03) == 1) {
            const size_t nb00_bytes = sizeof(uint16_t);
            const size_t k_bytes_span =
                (size_t)(ne01 > 0 ? ne01 - 1 : 0) * (size_t)nb01 +
                (size_t)(ne02 > 0 ? ne02 - 1 : 0) * (size_t)nb02 +
                (size_t)(ne03 > 0 ? ne03 - 1 : 0) * (size_t)nb03 +
                (size_t)ne00 * nb00_bytes;

            const size_t k_bytes = (k_bytes_span + 15) & ~(size_t)15;
            const size_t k_pixels = k_bytes >> 4;
            if (k_pixels > 0 && k_pixels <= backend_ctx->image_max_buffer_size) {
                cl_kernel kernel = backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_x8_gqa4_img;
                cl_mem K_img = ggml_cl_img_pool_get_or_create(
                    backend_ctx, backend_ctx->kq_img_pool,
                    extra0->data_device, offset0, k_bytes, CL_FLOAT);
                if (K_img != nullptr) {
                    cl_uint k_arg = 0;
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_mem),   &K_img));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_mem),   &extra1->data_device));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_ulong), &offset1));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_mem),   &extrad->data_device));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_ulong), &offsetd));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int),      &ne00));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int),      &ne01));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int),      &ne02));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_ulong), &nb01));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_ulong), &nb02));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_ulong), &nb03));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int),      &ne10));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int),      &ne11));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int),      &ne12));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_ulong), &nb10));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_ulong), &nb11));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_ulong), &nb12));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_ulong), &nb13));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int),      &ne0));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int),      &ne1));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int),      &r2));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int),      &r3));

                    const int nth0_d = 64;
                    const int64_t n_wg_x = ne01 / 16;
                    size_t global_work_size[] = {(size_t)n_wg_x * nth0_d, (size_t)1, (size_t)ne02 * ne13};
                    size_t local_work_size[]  = {(size_t)nth0_d, (size_t)1, 1};
                    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
                    return true;
                }
            }
        }

        // r2=4 specialization
        if (img_r4_gate) {
            const size_t nb00_bytes = sizeof(uint16_t);
            const size_t k_bytes_span =
                (size_t)(ne01 > 0 ? ne01 - 1 : 0) * (size_t)nb01 +
                (size_t)(ne02 > 0 ? ne02 - 1 : 0) * (size_t)nb02 +
                (size_t)(ne03 > 0 ? ne03 - 1 : 0) * (size_t)nb03 +
                (size_t)ne00 * nb00_bytes;
            const size_t k_bytes = (k_bytes_span + 15) & ~(size_t)15;
            const size_t k_pixels = k_bytes >> 4;
            if (k_pixels > 0 && k_pixels <= backend_ctx->image_max_buffer_size) {
                cl_kernel kernel = backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_x8_gqa_r4_img;
                cl_mem K_img = ggml_cl_img_pool_get_or_create(
                    backend_ctx, backend_ctx->kq_img_pool,
                    extra0->data_device, offset0, k_bytes, CL_FLOAT);
                if (K_img != nullptr) {
                    cl_uint k_arg = 0;
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_mem),   &K_img));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_mem),   &extra1->data_device));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_ulong), &offset1));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_mem),   &extrad->data_device));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_ulong), &offsetd));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int),      &ne00));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int),      &ne01));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int),      &ne02));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_ulong), &nb01));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_ulong), &nb02));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_ulong), &nb03));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int),      &ne10));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int),      &ne11));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int),      &ne12));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_ulong), &nb10));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_ulong), &nb11));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_ulong), &nb12));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_ulong), &nb13));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int),      &ne0));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int),      &ne1));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int),      &r2));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int),      &r3));

                    const int nth0_d = 64;
                    const int64_t n_wg_x = ne01 / 16;
                    size_t global_work_size[] = {(size_t)n_wg_x * nth0_d, (size_t)1, (size_t)ne02 * ne13};
                    size_t local_work_size[]  = {(size_t)nth0_d, (size_t)1, 1};
                    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
                    return true;
                }
            }
        }

        // DK=256, r2=2 specialization
        static const char * mm_kq_r2_dk256_env = getenv("GGML_OPENCL_MM_KQ_GQA_R2_DK256_IMG");
        static const bool mm_kq_r2_dk256_on = (mm_kq_r2_dk256_env != nullptr && mm_kq_r2_dk256_env[0] != '0');
        if (mm_kq_r2_dk256_on &&
            backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_x8_gqa_r2_dk256_img != nullptr &&
            ne11 == 1 && ne01 >= 64 && (ne01 % 16) == 0 && ne00 == 256 &&
            (ne12 % ne02) == 0 && (ne12 / ne02) == 2 && (ne13 / ne03) == 1) {
            const size_t nb00_bytes = sizeof(uint16_t);
            const size_t k_bytes_span =
                (size_t)(ne01 > 0 ? ne01 - 1 : 0) * (size_t)nb01 +
                (size_t)(ne02 > 0 ? ne02 - 1 : 0) * (size_t)nb02 +
                (size_t)(ne03 > 0 ? ne03 - 1 : 0) * (size_t)nb03 +
                (size_t)ne00 * nb00_bytes;
            const size_t k_bytes = (k_bytes_span + 15) & ~(size_t)15;
            const size_t k_pixels = k_bytes >> 4;
            if (k_pixels > 0 && k_pixels <= backend_ctx->image_max_buffer_size) {
                cl_kernel kernel = backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_x8_gqa_r2_dk256_img;
                cl_mem K_img = ggml_cl_img_pool_get_or_create(
                    backend_ctx, backend_ctx->kq_img_pool,
                    extra0->data_device, offset0, k_bytes, CL_FLOAT);
                if (K_img != nullptr) {
                    cl_uint k_arg = 0;
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_mem),   &K_img));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_mem),   &extra1->data_device));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_ulong), &offset1));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_mem),   &extrad->data_device));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_ulong), &offsetd));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int),      &ne00));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int),      &ne01));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int),      &ne02));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_ulong), &nb01));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_ulong), &nb02));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_ulong), &nb03));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int),      &ne10));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int),      &ne11));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int),      &ne12));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_ulong), &nb10));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_ulong), &nb11));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_ulong), &nb12));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_ulong), &nb13));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int),      &ne0));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int),      &ne1));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int),      &r2));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int),      &r3));

                    const int nth0_d = 64;
                    const int64_t n_wg_x = ne01 / 16;
                    size_t global_work_size[] = {(size_t)n_wg_x * nth0_d, (size_t)1, (size_t)ne02 * ne13};
                    size_t local_work_size[]  = {(size_t)nth0_d, (size_t)1, 1};
                    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
                    return true;
                }
            }
        }

        // GQA-coalesced KQV for decode using image1d_buffer_t
        static const char * mm_kqv_gqa_img_env = getenv("GGML_OPENCL_MM_KQV_GQA_IMG");
        static const bool mm_kqv_gqa_img_on = (mm_kqv_gqa_img_env != nullptr && mm_kqv_gqa_img_env[0] != '0');
        if (mm_kqv_gqa_img_on &&
            backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_y8_gqa_img != nullptr &&
            ne11 == 1 && ne01 == 128 &&
            (ne12 % ne02) == 0 && (ne12 / ne02) == 8 && (ne13 / ne03) == 1) {
            const size_t nb00_bytes = sizeof(uint16_t);
            const size_t v_bytes_span =
                (size_t)(ne01 > 0 ? ne01 - 1 : 0) * (size_t)nb01 +
                (size_t)(ne02 > 0 ? ne02 - 1 : 0) * (size_t)nb02 +
                (size_t)(ne03 > 0 ? ne03 - 1 : 0) * (size_t)nb03 +
                (size_t)ne00 * nb00_bytes;
            const size_t v_bytes = (v_bytes_span + 7) & ~(size_t)7;
            const size_t v_pixels = v_bytes >> 3;
            if (v_pixels > 0 && v_pixels <= backend_ctx->image_max_buffer_size) {
                cl_kernel kernel = backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_y8_gqa_img;
                cl_mem V_img = ggml_cl_img_pool_get_or_create(
                    backend_ctx, backend_ctx->kqv_img_pool,
                    extra0->data_device, offset0, v_bytes, CL_HALF_FLOAT);
                if (V_img != nullptr) {
                    cl_uint k_arg = 0;
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_mem),   &V_img));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_mem),   &extra1->data_device));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_ulong), &offset1));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_mem),   &extrad->data_device));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_ulong), &offsetd));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int),      &ne00));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int),      &ne01));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int),      &ne02));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_ulong), &nb01));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_ulong), &nb02));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_ulong), &nb03));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int),      &ne10));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int),      &ne11));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int),      &ne12));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_ulong), &nb10));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_ulong), &nb11));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_ulong), &nb12));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_ulong), &nb13));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int),      &ne0));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int),      &ne1));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int),      &r2));
                    CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int),      &r3));

                    const int nth0_d = 64;
                    const int64_t n_wg_x = ne01 / 8;
                    size_t global_work_size[] = {(size_t)n_wg_x * nth0_d, (size_t)1, (size_t)ne02 * ne13};
                    size_t local_work_size[]  = {(size_t)nth0_d, (size_t)1, 1};
                    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
                    return true;
                }
            }
        }
    }

    return false;
}
#endif // GGML_OPENCL_USE_ADRENO_KERNELS

void ggml_cl_mul_mat_kq_kqv_adreno(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, bool is_kq) {
    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    const int  ne00 = src0->ne[0];
    const int  ne01 = src0->ne[1];
    const int  ne02 = src0->ne[2];

    const cl_ulong nb01 = src0->nb[1];
    const cl_ulong nb02 = src0->nb[2];

    const int  ne10 = src1->ne[0];
    const int  ne11 = src1->ne[1];
    const int  ne12 = src1->ne[2];

    const cl_ulong nb10 = src1->nb[0];

    const int  ne0 = dst->ne[0];
    const int  ne1 = dst->ne[1];

    GGML_ASSERT(ne00 == ne10);

    cl_kernel kernel;
    cl_context context = backend_ctx->context;

    cl_int              status;
    cl_image_format     img_fmt_1d;
    cl_image_desc       img_desc_1d;
    cl_buffer_region    region;
    cl_mem              A_image1d;
    cl_mem              A_sub_buffer;
    cl_mem              B_sub_buffer;
    cl_mem              D_image1d;
    cl_mem              D_sub_buffer;

    int M = ne01;
    int N = ne1;
    int K = ne00;

    kernel = is_kq ? backend_ctx->mul_mat.kernel_mul_mm_f16_f32_kq
                   : backend_ctx->mul_mat.kernel_mul_mm_f16_f32_kqv;
    // create sub-buffer for A
    // <--------------------------------------------> //
    extra0 = src0->view_src ? (ggml_tensor_extra_cl *)src0->view_src->extra : (ggml_tensor_extra_cl *)src0->extra;

    region.origin = (extra0->offset + src0->view_offs);
    if (is_kq) {
        // KQ
        region.size = nb01 * ne01;
    } else {
        // KQV
        region.size = nb02 * ne02;
    }

    A_sub_buffer = clCreateSubBuffer((extra0->data_device), 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
    CL_CHECK(status);

    // <--------------------------------------------> //

    // create sub-buffer for B
    // <--------------------------------------------> //
    region.origin = (extra1->offset + src1->view_offs);
    region.size = nb10 * ne10 * ne11 * ne12;
    B_sub_buffer = clCreateSubBuffer((extra1->data_device), 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
    CL_CHECK(status);
    // <--------------------------------------------> //

    img_fmt_1d = {CL_RGBA, CL_FLOAT};
    memset(&img_desc_1d, 0, sizeof(img_desc_1d));
    img_desc_1d.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
    if (is_kq) {
        img_desc_1d.image_width = (nb01 * ne01 / 4)/4;
    }
    else {
        img_desc_1d.image_width = (nb02 * ne02 / 4)/4;
    }
    img_desc_1d.buffer = A_sub_buffer;
    A_image1d = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt_1d, &img_desc_1d, NULL, &status);
    CL_CHECK(status);

    // create sub-buffer for output C
    // <--------------------------------------------> //
    region.origin = (extrad->offset + dst->view_offs);
    region.size = ne0 * ne1 * dst->ne[2] * dst->nb[0]; // size of C in bytes
    D_sub_buffer = clCreateSubBuffer((extrad->data_device), 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
    CL_CHECK(status);
    // <--------------------------------------------> //

    // create image for C output
    // <--------------------------------------------> //
    img_fmt_1d = {CL_R, CL_FLOAT};
    memset(&img_desc_1d, 0, sizeof(img_desc_1d));
    img_desc_1d.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
    img_desc_1d.image_width = ne0 * ne1 * dst->ne[2] * dst->nb[0] / 4;
    img_desc_1d.buffer = D_sub_buffer;
    D_image1d = clCreateImage(context, CL_MEM_WRITE_ONLY, &img_fmt_1d, &img_desc_1d, NULL, &status);
    CL_CHECK(status);
    // <--------------------------------------------> //

    int offset_src0 = 0;
    int offset_src1 = 0;

    // set kernel args
    // <--------------------------------------------> //
    cl_uint k_arg = 0;
    CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(cl_mem), &A_image1d));
    CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(int),    &offset_src0));
    CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(cl_mem), &B_sub_buffer));
    CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(int),    &offset_src1));
    CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(cl_mem), &D_image1d));
    CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(int),    &extrad->offset));
    CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(int),    &M));
    CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(int),    &K));
    CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(int),    &N));
    CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(int),    &ne02));
    CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(int),    &ne12));
    CL_CHECK(clSetKernelArg(kernel,  k_arg++, sizeof(int),    &nb01));

    size_t global_work_size[3] = {64, static_cast<size_t>(((M+63)/64)), static_cast<size_t>(((N+31)/32)*ne12)};
    size_t local_work_size[3] = {64, 1, 2};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);

    // deallocate sub buffers and images
    // <--------------------------------------------> //
    CL_CHECK(clReleaseMemObject(A_image1d));
    CL_CHECK(clReleaseMemObject(D_image1d));
    CL_CHECK(clReleaseMemObject(A_sub_buffer));
    CL_CHECK(clReleaseMemObject(B_sub_buffer));
    CL_CHECK(clReleaseMemObject(D_sub_buffer));
}

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
void ggml_cl_mul_mat_q1_0_f32_adreno(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    GGML_ASSERT(src0->type == GGML_TYPE_Q1_0);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;
    ggml_tensor_extra_cl_q1_0 * extra0_q1_0 = (ggml_tensor_extra_cl_q1_0 *)src0->extra;

    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    GGML_ASSERT(src1->view_offs == 0);
    GGML_ASSERT(dst->view_offs == 0);

    const int  ne00 = src0->ne[0];
    const int  ne01 = src0->ne[1];
    const int  ne02 = src0->ne[2];

    const int  ne10 = src1->ne[0];
    const int  ne12 = src1->ne[2];

    const int  ne0 = dst->ne[0];
    const int  ne1 = dst->ne[1];

    GGML_ASSERT(ne00 == ne10);
    GGML_ASSERT((ne00 % 128) == 0);
    GGML_ASSERT(ne0 == ne01);

    cl_context context = backend_ctx->context;
    cl_kernel kernel;

    cl_int              err;
    cl_image_format     img_fmt;
    cl_image_desc       img_desc;
    cl_buffer_region    region;

    int M = ne01;
    int N = ne1;
    int K = ne00;

    if (ne1 == 1) {
        cl_mem q_img = nullptr;
        cl_mem b_sub_buf = nullptr;
        cl_mem b_img = nullptr;

        // image for q (uint32: each texel packs 32 sign bits)
        img_fmt = { CL_R, CL_UNSIGNED_INT32};
        memset(&img_desc, 0, sizeof(img_desc));
        img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc.image_width = M * K / 32;
        img_desc.buffer = extra0_q1_0->q;
        CL_CHECK((q_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));

        // create a sub_buffer for B
        region.origin = offset1;
        region.size = K * N * sizeof(float);
        CL_CHECK((b_sub_buf = clCreateSubBuffer((extra1->data_device), 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

        // image for activations
        img_fmt = {CL_RGBA, CL_FLOAT};
        memset(&img_desc, 0, sizeof(img_desc));
        img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc.image_width = K * N / 4;
        img_desc.buffer = b_sub_buf;
        CL_CHECK((b_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));

        kernel = backend_ctx->mul_mat.kernel_gemv_noshuffle_q1_0_f32;

        int r2 = 1;
        int r3 = 1;

        CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &q_img));
        CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0_q1_0->d));
        CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &b_img));
        CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &extra1->offset));
        CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
        CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &extrad->offset));
        CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
        CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
        CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne02));
        CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne10));
        CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne12));
        CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne0));
        CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne1));
        CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &r2));
        CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &r3));

        size_t wavesize = backend_ctx->adreno_wave_size;
        size_t local_work_size[]  = { wavesize, 4, 1 };
        size_t global_work_size[] = { CEIL_DIV(M, wavesize)*wavesize, 4, 1 };

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);

        CL_CHECK(clReleaseMemObject(q_img));
        CL_CHECK(clReleaseMemObject(b_img));
        CL_CHECK(clReleaseMemObject(b_sub_buf));
    } else {
        cl_mem b_sub_buf = nullptr;
        cl_mem b_sub_buf_trans = nullptr;
        cl_mem b_img = nullptr;
        cl_mem b_img_trans = nullptr;

        // subbuffer for activations
        region.origin = offset1;
        region.size = K * N * sizeof(float);
        CL_CHECK((b_sub_buf = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

        // image for activations
        img_fmt = {CL_RGBA, CL_FLOAT};
        memset(&img_desc, 0, sizeof(img_desc));
        img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc.image_width = K * N / 4;
        img_desc.buffer = b_sub_buf;
        CL_CHECK((b_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));

        // pad N to multiple of 8
        int extra_elements = N % 8;
        int padding = 0;
        if (extra_elements > 0){
            padding = 8 - extra_elements;
        }

        // subbuffer for transposed activations
        region.origin = 0;
        region.size = K * (N + padding) * sizeof(float)/2;
        backend_ctx->prealloc_act_trans.allocate(context, region.size);
        CL_CHECK((b_sub_buf_trans = clCreateSubBuffer(backend_ctx->prealloc_act_trans.buffer, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

        // image for transposed activations
        img_fmt = {CL_RGBA, CL_HALF_FLOAT};
        memset(&img_desc, 0, sizeof(img_desc));
        img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc.image_width = K * (N + padding) / 4;
        img_desc.buffer = b_sub_buf_trans;
        CL_CHECK((b_img_trans = clCreateImage(context, 0, &img_fmt, &img_desc, NULL, &err), err));

        // transpose activations
        int height_B = N/4;
        if (height_B == 0) {
            height_B = 1;
        }
        int width_B = K/4;
        int padded_height_B = (N + padding)/4;

        kernel = backend_ctx->mul_mat.kernel_transpose_32_16;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &b_img));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_img_trans));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int),    &height_B));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int),    &width_B));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int),    &padded_height_B));

        size_t local_work_size_t[2] = { 1, 16 };
        size_t global_work_size_t[2] = { (size_t)width_B, (size_t)padded_height_B };
        backend_ctx->enqueue_ndrange_kernel(kernel, 2, global_work_size_t, local_work_size_t, dst);

        // gemm
        kernel = backend_ctx->mul_mat.kernel_gemm_noshuffle_q1_0_f32;
        int padded_N = N + padding;

        CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0_q1_0->q));
        CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0_q1_0->d));
        CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &b_img_trans));
        CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_mem),   &extrad->data_device));
        CL_CHECK(clSetKernelArg(kernel,  4, sizeof(int),      &K));
        CL_CHECK(clSetKernelArg(kernel,  5, sizeof(int),      &M));
        CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &padded_N));
        CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &N));
        CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &offsetd));

        size_t global_work_size[] = { (size_t)CEIL_DIV(N, 8), (size_t)CEIL_DIV(M, 4), 1 };
        size_t local_work_size[]  = { 2, 128, 1 };

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);

        CL_CHECK(clReleaseMemObject(b_img_trans));
        CL_CHECK(clReleaseMemObject(b_sub_buf_trans));
        CL_CHECK(clReleaseMemObject(b_img));
        CL_CHECK(clReleaseMemObject(b_sub_buf));
    }
}
#endif

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
static void ggml_cl_mul_mat_q4_0_f32_adreno_ila(ggml_backend_t backend, const ggml_tensor * src0,
                                                const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;
    ggml_tensor_extra_cl_q4_0 * extra0_q4_0 = (ggml_tensor_extra_cl_q4_0 *)src0->extra;

    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];

    const int ne1 = dst->ne[1];

    GGML_ASSERT(ne00 % ggml_blck_size(src0->type) == 0);

    cl_context context = backend_ctx->context;
    cl_kernel kernel;

    cl_int              err;
    cl_image_format     img_fmt;
    cl_image_desc       img_desc;
    cl_buffer_region    region;

    int M = ne01;
    int N = ne1;
    int K = ne00;

    if (ne1 == 1) {
        cl_mem b_sub_buf = nullptr;
        cl_mem b_img     = nullptr;

        region.origin = offset1;
        region.size   = (size_t)K * N * sizeof(float);
        CL_CHECK((b_sub_buf = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

        img_fmt = { CL_RGBA, CL_FLOAT };
        memset(&img_desc, 0, sizeof(img_desc));
        img_desc.image_type  = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc.image_width = (size_t)K * N / 4;
        img_desc.buffer      = b_sub_buf;
        CL_CHECK((b_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));

        kernel = backend_ctx->mul_mat.kernel_gemv_noshuffle_q4_0_f32_32b_trans;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0_q4_0->q_img));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),   &extra0_q4_0->d));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &b_img));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem),   &extrad->data_device));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_ulong), &offsetd));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_int),   &K));
        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_int),   &M));

        size_t wavesize = backend_ctx->adreno_wave_size;
        size_t local_work_size[3]  = { wavesize, 4, 1 };
        size_t global_work_size[3] = { (size_t)CEIL_DIV(M, 64) * 64, 4, 1 };
        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);

        CL_CHECK(clReleaseMemObject(b_sub_buf));
        CL_CHECK(clReleaseMemObject(b_img));
    } else {
        const int gemm_tile_n = 64;
        int N_pad = (N + gemm_tile_n - 1) & ~(gemm_tile_n - 1);

        cl_mem a_img = extra0_q4_0->q_img;
        cl_mem s_img = extra0_q4_0->d_img;
        GGML_ASSERT(a_img && s_img && "ILA Q4_0 weight images missing; set_tensor should have built them");

        static const char * q4_0_bin_dp4a_env = getenv("GGML_OPENCL_Q4_0_BIN_DP4A");
                     bool   q4_0_bin_dp4a_on  = q4_0_bin_dp4a_env
                                                  ? (atoi(q4_0_bin_dp4a_env) != 0)
                                                  : true;
        // dot prod has to be available
        q4_0_bin_dp4a_on = backend_ctx->has_integer_dot && q4_0_bin_dp4a_on;

        if (q4_0_bin_dp4a_on && backend_ctx->mul_mat.kernel_gemm_noshuffle_q4_0_q8_1_dp4a_ila_a8_bin) {
            const int    dp4a_N_pad = CEIL_DIV(N, 32) * 32;
            const size_t n_blocks   = (size_t)dp4a_N_pad * (K / 32);

            backend_ctx->prealloc_moe_qa.allocate(context, (size_t)dp4a_N_pad * K * sizeof(cl_char));
            backend_ctx->prealloc_moe_da.allocate(context, n_blocks * sizeof(cl_half));
            backend_ctx->prealloc_moe_sa.allocate(context, n_blocks * sizeof(cl_half));

            cl_mem b_sub = nullptr;
            region.origin = offset1;
            region.size   = (size_t)K * N * sizeof(float);
            CL_CHECK((b_sub = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

            cl_int    tb = (cl_int)((size_t)N * (K / 32));
            cl_kernel qk = backend_ctx->mul_mat.kernel_quant_a_q8_1;
            CL_CHECK(clSetKernelArg(qk, 0, sizeof(cl_mem), &b_sub));
            CL_CHECK(clSetKernelArg(qk, 1, sizeof(cl_mem), &backend_ctx->prealloc_moe_qa.buffer));
            CL_CHECK(clSetKernelArg(qk, 2, sizeof(cl_mem), &backend_ctx->prealloc_moe_da.buffer));
            CL_CHECK(clSetKernelArg(qk, 3, sizeof(cl_mem), &backend_ctx->prealloc_moe_sa.buffer));
            CL_CHECK(clSetKernelArg(qk, 4, sizeof(cl_int), &tb));
            size_t q_local[1]  = { 64 };
            size_t q_global[1] = { (size_t)CEIL_DIV(tb, 64) * 64 };
            backend_ctx->enqueue_ndrange_kernel(qk, 1, q_global, q_local, dst);

            cl_mem d_sub = nullptr;
            cl_mem d_img = nullptr;
            region.origin = offsetd;
            region.size   = (size_t)M * N * sizeof(float);
            CL_CHECK((d_sub = clCreateSubBuffer(extrad->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

            img_fmt = { CL_R, CL_FLOAT };
            memset(&img_desc, 0, sizeof(img_desc));
            img_desc.image_type  = CL_MEM_OBJECT_IMAGE1D_BUFFER;
            img_desc.image_width = (size_t)M * N;
            img_desc.buffer      = d_sub;
            CL_CHECK((d_img = clCreateImage(context, CL_MEM_WRITE_ONLY, &img_fmt, &img_desc, NULL, &err), err));

            kernel = backend_ctx->mul_mat.kernel_gemm_noshuffle_q4_0_q8_1_dp4a_ila_a8_bin;

            cl_uint k_arg = 0;
            CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_mem), &a_img));
            CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_mem), &extra0_q4_0->d));
            CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_mem), &backend_ctx->prealloc_moe_qa.buffer));
            CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_mem), &backend_ctx->prealloc_moe_da.buffer));
            CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_mem), &d_img));
            CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int),    &K));
            CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int),    &M));
            CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int),    &N));

            size_t local_work_size[3]  = { 64, 1, 1 };
            size_t global_work_size[3] = { 64, (size_t)(M / 64), (size_t)(dp4a_N_pad / 32) };
            backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);

            CL_CHECK(clReleaseMemObject(b_sub));
            CL_CHECK(clReleaseMemObject(d_img));
            CL_CHECK(clReleaseMemObject(d_sub));
            return;
        }

        // Pad B through a zero-filled scratch buffer when N needs
        // padding, since the GEMM kernel always reads a full N-tile.
        const bool need_pad = N_pad > N;
        cl_mem b_sub_buf = nullptr;
        cl_mem b_padded  = nullptr;
        if (need_pad) {
            CL_CHECK((b_padded = clCreateBuffer(context, CL_MEM_READ_WRITE,
                (size_t)K * N_pad * sizeof(float), NULL, &err), err));
            const float zero = 0.0f;
            CL_CHECK(clEnqueueFillBuffer(backend_ctx->queue, b_padded, &zero, sizeof(zero),
                0, (size_t)K * N_pad * sizeof(float), 0, NULL, NULL));
            CL_CHECK(clEnqueueCopyBuffer(backend_ctx->queue, extra1->data_device, b_padded,
                offset1, 0, (size_t)K * N * sizeof(float), 0, NULL, NULL));
        } else {
            region.origin = offset1;
            region.size   = (size_t)K * N * sizeof(float);
            CL_CHECK((b_sub_buf = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));
        }

        img_fmt = { CL_R, CL_FLOAT };
        memset(&img_desc, 0, sizeof(img_desc));
        img_desc.image_type  = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc.image_width = need_pad ? (size_t)K * N_pad : (size_t)K * N;
        img_desc.buffer      = need_pad ? b_padded : b_sub_buf;
        cl_mem b_img;
        CL_CHECK((b_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));

        region.origin = offsetd;
        region.size   = (size_t)M * N * sizeof(float);
        cl_mem d_sub_buf;
        CL_CHECK((d_sub_buf = clCreateSubBuffer(extrad->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

        img_fmt = { CL_R, CL_FLOAT };
        memset(&img_desc, 0, sizeof(img_desc));
        img_desc.image_type  = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc.image_width = (size_t)M * N;
        img_desc.buffer      = d_sub_buf;
        cl_mem d_img;
        CL_CHECK((d_img = clCreateImage(context, CL_MEM_WRITE_ONLY, &img_fmt, &img_desc, NULL, &err), err));

        int line_stride_matrix_A_in_bytes = M * 4;
        int line_stride_matrix_S_in_bytes = M * 2;
        int line_stride_matrix_B_in_bytes = K * 4;
        int line_stride_matrix_C_in_bytes = M * 4;

        int c_offset_for_kernel = 0;
        int b_offset_for_kernel = 0;

        kernel = backend_ctx->mul_mat.kernel_gemm_noshuffle_q4_0_f32_32b_trans_ila_a8_bin;

        cl_uint k_arg = 0;
        CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_mem), &a_img));
        CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_mem), &s_img));
        CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_mem), &b_img));
        CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int),    &b_offset_for_kernel));
        CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_mem), &d_img));
        CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int), &c_offset_for_kernel));
        CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int), &K));
        CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int), &line_stride_matrix_A_in_bytes));
        CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int), &line_stride_matrix_S_in_bytes));
        CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int), &line_stride_matrix_B_in_bytes));
        CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int), &line_stride_matrix_C_in_bytes));
        CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int), &M));
        CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int), &N));

        size_t local_work_size[3]  = { 64, 2, 2 };
        size_t m_tiles = (size_t)CEIL_DIV(M, 64);
        size_t global_work_size[3] = { 64, m_tiles, (size_t)CEIL_DIV(N_pad, gemm_tile_n) };
        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);

        CL_CHECK(clReleaseMemObject(b_img));
        if (b_sub_buf) {
            CL_CHECK(clReleaseMemObject(b_sub_buf));
        }
        if (b_padded) {
            CL_CHECK(clReleaseMemObject(b_padded));
        }
        CL_CHECK(clReleaseMemObject(d_img));
        CL_CHECK(clReleaseMemObject(d_sub_buf));
    }
}
#endif // GGML_OPENCL_USE_ADRENO_KERNELS

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
void ggml_cl_mul_mat_q4_0_f32_adreno(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;
    ggml_tensor_extra_cl_q4_0 * extra0_q4_0 = (ggml_tensor_extra_cl_q4_0 *)src0->extra;

    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];

    const int ne10 = src1->ne[0];
    const int ne12 = src1->ne[2];

    const int ne0 = dst->ne[0];
    const int ne1 = dst->ne[1];

    GGML_ASSERT(ne00 % ggml_blck_size(src0->type) == 0);

    cl_context context = backend_ctx->context;
    cl_kernel kernel;

    cl_int              err;
    cl_image_format     img_fmt;
    cl_image_desc       img_desc;
    cl_buffer_region    region;

    int M = ne01;
    int N = ne1;
    int K = ne00;

    // Multi-column (N=3) verify GEMV for q4_0: route the spec/MTP verify batch
    // (ne1==3) onto the efficient GEMV path instead of the transposed-GEMM dead-
    // zone (gemm_noshuffle_q4_0 is ~50% of MTP decode on a Q4_0 model since q4_0
    // weights have no cok/mc3, unlike q4_K). Reuses the ne1==1 GEMV image setup
    // (activation image already sized by N=ne1). Byte-identical. Opt-in via
    // GGML_OPENCL_Q40_MC3=1. Per-layer only (ne01 < 32768); q4_0 lm_head doesn't
    // occur (token_embd/output stay Q6_K), guard kept for parity with q4_K mc3.
    static const bool q40_mc3 = (getenv("GGML_OPENCL_Q40_MC3") != nullptr);
    const bool use_q40_mc3 = q40_mc3 && (ne1 >= 2 && ne1 <= 4) && (ne01 < 32768);

    const bool use_bin = use_q4_0_bin_kernels(backend_ctx, src0);

    if (use_bin) {
        if (use_q40_mc3) {
            static bool warned = false;
            if (!warned) {
                GGML_LOG_WARN("ggml_opencl: GGML_OPENCL_Q40_MC3 is bypassed by Q4_0 binary kernels\n");
                warned = true;
            }
        }
        ggml_cl_mul_mat_q4_0_f32_adreno_ila(backend, src0, src1, dst);
        return;
    }

    if (ne1 == 1 || use_q40_mc3) {
        cl_mem q_img = nullptr;
        cl_mem b_sub_buf = nullptr;
        cl_mem b_img = nullptr;

        // image for q
        img_fmt = { CL_R, CL_UNSIGNED_INT32};
        memset(&img_desc, 0, sizeof(img_desc));
        img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc.image_width = M * K / 2 / 4;
        img_desc.buffer = extra0_q4_0->q;
        CL_CHECK((q_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));

        // subbuffer for activations
        region.origin = offset1;
        region.size = K * N * sizeof(float);
        CL_CHECK((b_sub_buf = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

        // image for activations
        img_fmt = {CL_RGBA, CL_FLOAT};
        memset(&img_desc, 0, sizeof(img_desc));
        img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc.image_width = K * N / 4;
        img_desc.buffer = b_sub_buf;
        CL_CHECK((b_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));

        if (use_q40_mc3) {
            kernel = backend_ctx->mul_mat.kernel_gemv_noshuffle_q4_0_f32_mc3;
            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &q_img));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),   &extra0_q4_0->d));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &b_img));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int),      &ne01));
            CL_CHECK(clSetKernelArg(kernel, 7, sizeof(int),      &ne1));
        } else {
            kernel = backend_ctx->mul_mat.kernel_gemv_noshuffle_q4_0_f32;
            if (M == 4096 && K == 4096) {
                kernel = backend_ctx->mul_mat.kernel_gemv_noshuffle_q4_0_f32_4096_1_4096;
            } else if (M == 4096 && K == 11008) {
                kernel = backend_ctx->mul_mat.kernel_gemv_noshuffle_q4_0_f32_4096_1_11008;
            } else if (M == 11008 && K == 4096) {
                kernel = backend_ctx->mul_mat.kernel_gemv_noshuffle_q4_0_f32_11008_1_4096;
            } else if (M == 32000 && K == 4096) {
                kernel = backend_ctx->mul_mat.kernel_gemv_noshuffle_q4_0_f32_32000_1_4096;
            }

            int r2 = 1;
            int r3 = 1;

            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &q_img));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0_q4_0->d));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &b_img));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne02));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne10));
            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne12));
            CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne0));
            CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne1));
            CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &r2));
            CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &r3));
        }

        // Small-M mc3 verify is occupancy/latency-bound (too few WGs at small M, so
        // its bandwidth falls well short of the FFN matmuls'). Use 8 subgroups (512-WI WGs, half the
        // per-lane K-walk) for small M. Layout stride is fixed (4 uints/block), so only
        // the K-split count changes; the mc3 kernel reads it via get_local_size(1). The
        // ne1==1 base kernel hardcodes N_SIMDGROUP=4, so it always stays at 4.
        const int mc3_nsg = (use_q40_mc3 && ne01 < 4096) ? 8 : 4;
        size_t local_work_size[3] = {64, (size_t)mc3_nsg, 1};
        size_t global_work_size[3] = {(size_t)CEIL_DIV(ne01/2, 64)*64, (size_t)mc3_nsg, 1};

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);

        CL_CHECK(clReleaseMemObject(q_img));
        CL_CHECK(clReleaseMemObject(b_sub_buf));
        CL_CHECK(clReleaseMemObject(b_img));
    } else {
        // dp4a (int8) dense prefill GEMM, default off
        static const char * q4_0_dense_dp4a_env = getenv("GGML_OPENCL_Q4_0_DENSE_DP4A");
        bool q4_0_dense_dp4a_on = q4_0_dense_dp4a_env
            ? (atoi(q4_0_dense_dp4a_env) != 0)
            : false;
        // dot prod has to be available
        q4_0_dense_dp4a_on = backend_ctx->has_integer_dot && q4_0_dense_dp4a_on;

        if (q4_0_dense_dp4a_on && backend_ctx->mul_mat.kernel_gemm_noshuffle_q4_0_q8_1_dp4a
                && N > 8 && (K % 32 == 0) && (M % 64 == 0)) {
            cl_mem a_sub = nullptr;
            region.origin = offset1;
            region.size   = (size_t)K * N * sizeof(float);
            CL_CHECK((a_sub = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

            const size_t n_blocks = (size_t)N * (K / 32);
            backend_ctx->prealloc_moe_qa.allocate(context, (size_t)N * K * sizeof(cl_char));
            backend_ctx->prealloc_moe_da.allocate(context, n_blocks * sizeof(cl_half));
            backend_ctx->prealloc_moe_sa.allocate(context, n_blocks * sizeof(cl_half));

            cl_int tb = (cl_int)n_blocks;
            cl_kernel qk = backend_ctx->mul_mat.kernel_quant_a_q8_1;
            CL_CHECK(clSetKernelArg(qk, 0, sizeof(cl_mem), &a_sub));
            CL_CHECK(clSetKernelArg(qk, 1, sizeof(cl_mem), &backend_ctx->prealloc_moe_qa.buffer));
            CL_CHECK(clSetKernelArg(qk, 2, sizeof(cl_mem), &backend_ctx->prealloc_moe_da.buffer));
            CL_CHECK(clSetKernelArg(qk, 3, sizeof(cl_mem), &backend_ctx->prealloc_moe_sa.buffer));
            CL_CHECK(clSetKernelArg(qk, 4, sizeof(cl_int), &tb));
            size_t q_local[1]  = { 64 };
            size_t q_global[1] = { (size_t)(((n_blocks + 63) / 64) * 64) };
            backend_ctx->enqueue_ndrange_kernel(qk, 1, q_global, q_local, dst);

            cl_kernel dk = backend_ctx->mul_mat.kernel_gemm_noshuffle_q4_0_q8_1_dp4a;
            int ai = 0;
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem),   &extra0_q4_0->q));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem),   &extra0_q4_0->d));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem),   &backend_ctx->prealloc_moe_qa.buffer));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem),   &backend_ctx->prealloc_moe_da.buffer));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem),   &backend_ctx->prealloc_moe_sa.buffer));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_int),   &M));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_int),   &N));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_int),   &K));
            size_t d_local[3]  = { 64, 1, 1 };
            size_t d_global[3] = { 64, (size_t)(M / 64), (size_t)CEIL_DIV(N, 32) };
            backend_ctx->enqueue_ndrange_kernel(dk, 3, d_global, d_local, dst);

            CL_CHECK(clReleaseMemObject(a_sub));
            return;
        }

        cl_mem b_sub_buf = nullptr;
        cl_mem b_sub_buf_trans = nullptr;
        cl_mem b_img = nullptr;
        cl_mem b_img_trans = nullptr;
        cl_mem d_sub_buf = nullptr;

        // subbuffer for activations
        region.origin = offset1;
        region.size = K * N * sizeof(float);
        CL_CHECK((b_sub_buf = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

        // image for activations
        img_fmt = {CL_RGBA, CL_FLOAT};
        memset(&img_desc, 0, sizeof(img_desc));
        img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc.image_width = K * N / 4;
        img_desc.buffer = b_sub_buf;
        CL_CHECK((b_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));

        // pad N to multiple of 8
        int extra_elements = N % 8;
        int padding = 0;
        if (extra_elements > 0){
            padding = 8 - extra_elements;
        }

        // subbuffer for transposed activations
        region.origin = 0;
        region.size = K * (N + padding) * sizeof(float)/2;
        backend_ctx->prealloc_act_trans.allocate(context, region.size);
        CL_CHECK((b_sub_buf_trans = clCreateSubBuffer(backend_ctx->prealloc_act_trans.buffer, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

        // image for transposed activations
        img_fmt = {CL_RGBA, CL_HALF_FLOAT};
        memset(&img_desc, 0, sizeof(img_desc));
        img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc.image_width = K * (N + padding) / 4;
        img_desc.buffer = b_sub_buf_trans;
        CL_CHECK((b_img_trans = clCreateImage(context, 0, &img_fmt, &img_desc, NULL, &err), err));

        // subbuffer for output
        region.origin = extrad->offset; // Specify the starting offset (in bytes)
        region.size = M * N * sizeof(float); // Specify the size of the sub-buffer
        CL_CHECK((d_sub_buf = clCreateSubBuffer(extrad->data_device, CL_MEM_WRITE_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

        // transpose activations
        int height_B = N/4;
        if (height_B == 0) {
            height_B = 1;
        }
        int width_B = K/4;
        int padded_height_B = (N + padding)/4;

        kernel = backend_ctx->mul_mat.kernel_transpose_32_16;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &b_img));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_img_trans));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int),    &height_B));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int),    &width_B));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int),    &padded_height_B));

        size_t local_work_size_t[2] = { 1, 16 };
        size_t global_work_size_t[2] = { (size_t)width_B, (size_t)padded_height_B };
        if (ne0 == 4096 && ne1 == 128 && ne10 == 4096) {
            local_work_size_t[0]=4;
            local_work_size_t[1]=8;
        } else if (ne0 == 11008 && ne1 == 128 && ne10 == 4096) {
            local_work_size_t[0]=2;
            local_work_size_t[1]=8;
        } else if(ne0 == 4096 && ne1 == 128 && ne10 == 11008) {
            local_work_size_t[0]=1;
            local_work_size_t[1]=8;
        } else if(ne0 == 32000 && ne1 == 128 && ne10 == 4096) {
            local_work_size_t[0]=2;
            local_work_size_t[1]=8;
        }
        backend_ctx->enqueue_ndrange_kernel(kernel, 2, global_work_size_t, local_work_size_t, dst);

        // gemm
        kernel = backend_ctx->mul_mat.kernel_gemm_noshuffle_q4_0_f32;
        int padded_N = N + padding;

        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0_q4_0->q));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),   &extra0_q4_0->d));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &b_img_trans));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem),   &d_sub_buf));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_int),   &ne01));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_int),   &padded_N));
        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_int),   &ne00));
        CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_int),   &ne1));

        size_t global_work_size[3] = {(size_t)CEIL_DIV(ne1, 8), (size_t)CEIL_DIV(ne01, 4), 1};
        size_t local_work_size[3] = {1, 128, 1};
        if (ne0 == 4096 && ne1 == 128 && ne10 == 4096) {
            local_work_size[0] = 1;
            local_work_size[1] = 128;
        } else if (ne0 == 11008 && ne1 == 128 && ne10 == 4096) {
            local_work_size[0] = 2;
            local_work_size[1] = 64;
        } else if (ne0 == 4096 && ne1 == 128 && ne10 == 11008) {
            local_work_size[0] = 2;
            local_work_size[1] = 64;
        } else if (ne0 == 32000 && ne1 == 128 && ne10 == 4096) {
            local_work_size[0] = 2;
            local_work_size[1] = 64;
        }

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);

        CL_CHECK(clReleaseMemObject(b_sub_buf));
        CL_CHECK(clReleaseMemObject(b_sub_buf_trans));
        CL_CHECK(clReleaseMemObject(b_img));
        CL_CHECK(clReleaseMemObject(b_img_trans));
        CL_CHECK(clReleaseMemObject(d_sub_buf));
    }
}
#endif // GGML_OPENCL_USE_ADRENO_KERNELS

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
void ggml_cl_mul_mat_q4_1_f32_adreno(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;
    ggml_tensor_extra_cl_q4_1 * extra0_q4_1 = (ggml_tensor_extra_cl_q4_1 *)src0->extra;

    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    const int  ne00 = src0->ne[0];
    const int  ne01 = src0->ne[1];

    const int  ne1 = dst->ne[1];

    GGML_ASSERT(ne00 % ggml_blck_size(src0->type) == 0);

    cl_context context = backend_ctx->context;
    cl_kernel kernel;

    cl_int              err;
    cl_image_format     img_fmt;
    cl_image_desc       img_desc;
    cl_buffer_region    region;

    int M = ne01;
    int N = ne1;
    int K = ne00;

    // Multi-column (N=3) verify GEMV for q4_1: route the spec/MTP verify batch
    // (ne1==3) onto the efficient GEMV path instead of the transposed-GEMM dead-
    // zone (gemm_noshuffle_q4_1). Reuses the ne1==1 GEMV image setup. Opt-in via
    // GGML_OPENCL_Q41_MC3=1. Per-layer only (ne01 < 32768).
    static const bool q41_mc3 = (getenv("GGML_OPENCL_Q41_MC3") != nullptr);
    const bool use_q41_mc3 = q41_mc3 && (ne1 >= 2 && ne1 <= 4) && (ne01 < 32768);

    if (ne1 == 1 || use_q41_mc3) {
        cl_mem q_img = nullptr;
        cl_mem b_sub_buf = nullptr;
        cl_mem b_img = nullptr;

        // image for q
        img_fmt = { CL_R, CL_UNSIGNED_INT32};
        memset(&img_desc, 0, sizeof(img_desc));
        img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc.image_width = M * K / 2 / 4;
        img_desc.buffer = extra0_q4_1->q;
        CL_CHECK((q_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));

        // subbuffer for activations
        region.origin = offset1;
        region.size = K * N * sizeof(float);
        CL_CHECK((b_sub_buf = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

        // image for activations
        img_fmt = {CL_RGBA, CL_FLOAT};
        memset(&img_desc, 0, sizeof(img_desc));
        img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc.image_width = K * N / 4;
        img_desc.buffer = b_sub_buf;
        CL_CHECK((b_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));

        kernel = use_q41_mc3 ? backend_ctx->mul_mat.kernel_gemv_noshuffle_q4_1_f32_mc3
                             : backend_ctx->mul_mat.kernel_gemv_noshuffle_q4_1_f32;

        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &q_img));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),   &extra0_q4_1->d));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extra0_q4_1->m));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem),   &b_img));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem),   &extrad->data_device));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_ulong), &offsetd));
        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_int),   &ne00));
        CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_int),   &ne01));
        if (use_q41_mc3) {
            CL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_int), &ne1));  // n_cols
        }

        size_t local_work_size[3] = {64, 4, 1};
        size_t global_work_size[3] = {(size_t)CEIL_DIV(ne01/2, 64)*64, 4, 1};

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);

        CL_CHECK(clReleaseMemObject(q_img));
        CL_CHECK(clReleaseMemObject(b_sub_buf));
        CL_CHECK(clReleaseMemObject(b_img));
    } else {
        cl_mem b_sub_buf = nullptr;
        cl_mem b_sub_buf_trans = nullptr;
        cl_mem b_img = nullptr;
        cl_mem b_img_trans = nullptr;

        // subbuffer for activations
        region.origin = offset1;
        region.size = K * N * sizeof(float);
        CL_CHECK((b_sub_buf = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

        // image for activations
        img_fmt = {CL_RGBA, CL_FLOAT};
        memset(&img_desc, 0, sizeof(img_desc));
        img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc.image_width = K * N / 4;
        img_desc.buffer = b_sub_buf;
        CL_CHECK((b_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));

        // pad N to multiple of 8
        int extra_elements = N % 8;
        int padding = 0;
        if (extra_elements > 0){
            padding = 8 - extra_elements;
        }

        // subbuffer for transposed activations
        region.origin = 0;
        region.size = K * (N + padding) * sizeof(float)/2;
        backend_ctx->prealloc_act_trans.allocate(context, region.size);
        CL_CHECK((b_sub_buf_trans = clCreateSubBuffer(backend_ctx->prealloc_act_trans.buffer, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

        // image for transposed activations
        img_fmt = {CL_RGBA, CL_HALF_FLOAT};
        memset(&img_desc, 0, sizeof(img_desc));
        img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc.image_width = K * (N + padding) / 4;
        img_desc.buffer = b_sub_buf_trans;
        CL_CHECK((b_img_trans = clCreateImage(context, 0, &img_fmt, &img_desc, NULL, &err), err));

        // transpose activations
        int height_B = N/4;
        if (height_B == 0) {
            height_B = 1;
        }
        int width_B = K/4;
        int padded_height_B = (N + padding)/4;

        kernel = backend_ctx->mul_mat.kernel_transpose_32_16;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &b_img));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_img_trans));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int),    &height_B));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int),    &width_B));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int),    &padded_height_B));

        size_t local_work_size_t[2] = { 1, 16 };
        size_t global_work_size_t[2] = { (size_t)width_B, (size_t)padded_height_B };
        backend_ctx->enqueue_ndrange_kernel(kernel, 2, global_work_size_t, local_work_size_t, dst);

        // gemm
        kernel = backend_ctx->mul_mat.kernel_gemm_noshuffle_q4_1_f32;
        int padded_N = N + padding;

        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0_q4_1->q));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),   &extra0_q4_1->d));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extra0_q4_1->m));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem),   &b_img_trans));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem),   &extrad->data_device));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_ulong), &offsetd));
        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_int),   &ne01));
        CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_int),   &padded_N));
        CL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_int),   &ne00));
        CL_CHECK(clSetKernelArg(kernel, 9, sizeof(cl_int),   &ne1));

        size_t global_work_size[3] = {(size_t)CEIL_DIV(ne1, 8), (size_t)CEIL_DIV(ne01, 4), 1};
        size_t local_work_size[3] = {1, 128, 1};

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);

        CL_CHECK(clReleaseMemObject(b_sub_buf));
        CL_CHECK(clReleaseMemObject(b_sub_buf_trans));
        CL_CHECK(clReleaseMemObject(b_img));
        CL_CHECK(clReleaseMemObject(b_img_trans));
    }
}
#endif // GGML_OPENCL_USE_ADRENO_KERNELS

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
void ggml_cl_mul_mat_q5_0_f32_adreno(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;
    ggml_tensor_extra_cl_q5_0 * extra0_q5_0 = (ggml_tensor_extra_cl_q5_0 *)src0->extra;

    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];

    const int ne1 = dst->ne[1];

    GGML_ASSERT(ne00 % ggml_blck_size(src0->type) == 0);

    cl_context context = backend_ctx->context;
    cl_kernel kernel;

    cl_int              err;
    cl_image_format     img_fmt;
    cl_image_desc       img_desc;
    cl_buffer_region    region;

    int M = ne01;
    int N = ne1;
    int K = ne00;

    if (ne1 == 1) {
        cl_mem qs_img = nullptr;
        cl_mem b_sub_buf = nullptr;
        cl_mem b_img = nullptr;

        // image for qs
        img_fmt = { CL_R, CL_UNSIGNED_INT32 };
        memset(&img_desc, 0, sizeof(img_desc));
        img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc.image_width = M * K / 2 / 4;
        img_desc.buffer = extra0_q5_0->qs;
        CL_CHECK((qs_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));

        // subbuffer for activations
        region.origin = offset1;
        region.size = K * N * sizeof(float);
        CL_CHECK((b_sub_buf = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

        // image for activations
        img_fmt = {CL_RGBA, CL_FLOAT};
        memset(&img_desc, 0, sizeof(img_desc));
        img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc.image_width = K * N / 4;
        img_desc.buffer = b_sub_buf;
        CL_CHECK((b_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));

        kernel = backend_ctx->mul_mat.kernel_gemv_noshuffle_q5_0_f32;

        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &qs_img));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),   &extra0_q5_0->qh));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extra0_q5_0->d));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem),   &b_img));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem),   &extrad->data_device));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_ulong), &offsetd));
        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_int),   &ne00));
        CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_int),   &ne01));

        size_t local_work_size[3] = {64, 4, 1};
        size_t global_work_size[3] = {(size_t)CEIL_DIV(ne01/2, 64)*64, 4, 1};

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);

        CL_CHECK(clReleaseMemObject(qs_img));
        CL_CHECK(clReleaseMemObject(b_sub_buf));
        CL_CHECK(clReleaseMemObject(b_img));
    } else {
        // dp4a (int8) dense q5_0 prefill GEMM, default off
        static const char * q5_dense_dp4a_env = getenv("GGML_OPENCL_Q5_DENSE_DP4A");
        static const char * q5_dense_wimg_env = getenv("GGML_OPENCL_Q5_DENSE_DP4A_WIMG");
        const bool q5_dense_wimg_on = q5_dense_wimg_env && (atoi(q5_dense_wimg_env) != 0);
              bool q5_dense_dp4a_on = q5_dense_wimg_on
            ? true
            : (q5_dense_dp4a_env && (atoi(q5_dense_dp4a_env) != 0));
        // dot prod has to be available
        q5_dense_dp4a_on = backend_ctx->has_integer_dot && q5_dense_dp4a_on;

        if (q5_dense_dp4a_on && backend_ctx->mul_mat.kernel_gemm_noshuffle_q5_0_q8_1_dp4a
                && N > 8 && (K % 32 == 0) && (M % 64 == 0)) {
            cl_mem a_sub = nullptr;
            region.origin = offset1;
            region.size   = (size_t)K * N * sizeof(float);
            CL_CHECK((a_sub = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

            const size_t n_blocks = (size_t)N * (K / 32);
            backend_ctx->prealloc_moe_qa.allocate(context, (size_t)N * K * sizeof(cl_char));
            backend_ctx->prealloc_moe_da.allocate(context, n_blocks * sizeof(cl_half));
            backend_ctx->prealloc_moe_sa.allocate(context, n_blocks * sizeof(cl_half));

            cl_int tb = (cl_int)n_blocks;
            cl_kernel qk = backend_ctx->mul_mat.kernel_quant_a_q8_1;
            CL_CHECK(clSetKernelArg(qk, 0, sizeof(cl_mem), &a_sub));
            CL_CHECK(clSetKernelArg(qk, 1, sizeof(cl_mem), &backend_ctx->prealloc_moe_qa.buffer));
            CL_CHECK(clSetKernelArg(qk, 2, sizeof(cl_mem), &backend_ctx->prealloc_moe_da.buffer));
            CL_CHECK(clSetKernelArg(qk, 3, sizeof(cl_mem), &backend_ctx->prealloc_moe_sa.buffer));
            CL_CHECK(clSetKernelArg(qk, 4, sizeof(cl_int), &tb));
            size_t q_local[1]  = { 64 };
            size_t q_global[1] = { (size_t)(((n_blocks + 63) / 64) * 64) };
            backend_ctx->enqueue_ndrange_kernel(qk, 1, q_global, q_local, dst);

            // optional qs texture (image1d_buffer over the nibble plane; the same
            // CL_R/UINT32 view, width M*K/8, the GEMV path builds).
            cl_mem q5_qs_img = nullptr;
            bool use_wimg = q5_dense_wimg_on;
            if (use_wimg) {
                const size_t tex = (size_t)M * (size_t)K / 8;  // uint32 texels (2 ushorts/texel)
                if (tex == 0 || tex > backend_ctx->image_max_buffer_size) {
                    use_wimg = false;
                } else {
                    img_fmt = { CL_R, CL_UNSIGNED_INT32 };
                    memset(&img_desc, 0, sizeof(img_desc));
                    img_desc.image_type  = CL_MEM_OBJECT_IMAGE1D_BUFFER;
                    img_desc.image_width = tex;
                    img_desc.buffer      = extra0_q5_0->qs;
                    q5_qs_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err);
                    if (err != CL_SUCCESS || q5_qs_img == nullptr) { use_wimg = false; q5_qs_img = nullptr; }
                }
            }

            cl_kernel dk = use_wimg ? backend_ctx->mul_mat.kernel_gemm_noshuffle_q5_0_q8_1_dp4a_wimg
                                    : backend_ctx->mul_mat.kernel_gemm_noshuffle_q5_0_q8_1_dp4a;
            int ai = 0;
            if (use_wimg) {
                CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem), &q5_qs_img));
            } else {
                CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem), &extra0_q5_0->qs));
            }
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem),   &extra0_q5_0->qh));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem),   &extra0_q5_0->d));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem),   &backend_ctx->prealloc_moe_qa.buffer));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem),   &backend_ctx->prealloc_moe_da.buffer));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem),   &backend_ctx->prealloc_moe_sa.buffer));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_int),   &M));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_int),   &N));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_int),   &K));
            size_t d_local[3]  = { 64, 1, 1 };
            size_t d_global[3] = { 64, (size_t)(M / 64), (size_t)CEIL_DIV(N, 32) };
            backend_ctx->enqueue_ndrange_kernel(dk, 3, d_global, d_local, dst);

            if (q5_qs_img != nullptr) {
                CL_CHECK(clReleaseMemObject(q5_qs_img));
            }
            CL_CHECK(clReleaseMemObject(a_sub));
            return;
        }

        cl_mem b_sub_buf = nullptr;
        cl_mem b_sub_buf_trans = nullptr;
        cl_mem b_img = nullptr;
        cl_mem b_img_trans = nullptr;
        cl_mem d_sub_buf = nullptr;

        // subbuffer for activations
        region.origin = offset1;
        region.size = K * N * sizeof(float);
        CL_CHECK((b_sub_buf = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

        // image for activations
        img_fmt = {CL_RGBA, CL_FLOAT};
        memset(&img_desc, 0, sizeof(img_desc));
        img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc.image_width = K * N / 4;
        img_desc.buffer = b_sub_buf;
        CL_CHECK((b_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));

        // pad N to multiple of 8
        int extra_elements = N % 8;
        int padding = 0;
        if (extra_elements > 0){
            padding = 8 - extra_elements;
        }

        // subbuffer for transposed activations
        region.origin = 0;
        region.size = K * (N + padding) * sizeof(float)/2;
        backend_ctx->prealloc_act_trans.allocate(context, region.size);
        CL_CHECK((b_sub_buf_trans = clCreateSubBuffer(backend_ctx->prealloc_act_trans.buffer, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

        // image for transposed activations
        img_fmt = {CL_RGBA, CL_HALF_FLOAT};
        memset(&img_desc, 0, sizeof(img_desc));
        img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc.image_width = K * (N + padding) / 4;
        img_desc.buffer = b_sub_buf_trans;
        CL_CHECK((b_img_trans = clCreateImage(context, 0, &img_fmt, &img_desc, NULL, &err), err));

        // subbuffer for output
        region.origin = extrad->offset;
        region.size = M * N * sizeof(float);
        CL_CHECK((d_sub_buf = clCreateSubBuffer(extrad->data_device, CL_MEM_WRITE_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

        // transpose activations
        int height_B = N/4;
        if (height_B == 0) {
            height_B = 1;
        }
        int width_B = K/4;
        int padded_height_B = (N + padding)/4;

        kernel = backend_ctx->mul_mat.kernel_transpose_32_16;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &b_img));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_img_trans));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int),    &height_B));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int),    &width_B));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int),    &padded_height_B));

        size_t local_work_size_t[2] = { 1, 16 };
        size_t global_work_size_t[2] = { (size_t)width_B, (size_t)padded_height_B };
        backend_ctx->enqueue_ndrange_kernel(kernel, 2, global_work_size_t, local_work_size_t, dst);

        // gemm
        kernel = backend_ctx->mul_mat.kernel_gemm_noshuffle_q5_0_f32;
        int padded_N = N + padding;

        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0_q5_0->qs));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),   &extra0_q5_0->qh));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extra0_q5_0->d));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem),   &b_img_trans));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem),   &d_sub_buf));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_int),   &ne01));
        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_int),   &padded_N));
        CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_int),   &ne00));
        CL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_int),   &ne1));

        size_t global_work_size[3] = {(size_t)CEIL_DIV(ne1, 8), (size_t)CEIL_DIV(ne01, 4), 1};
        size_t local_work_size[3] = {1, 128, 1};

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);

        CL_CHECK(clReleaseMemObject(b_sub_buf));
        CL_CHECK(clReleaseMemObject(b_sub_buf_trans));
        CL_CHECK(clReleaseMemObject(b_img));
        CL_CHECK(clReleaseMemObject(b_img_trans));
        CL_CHECK(clReleaseMemObject(d_sub_buf));
    }
}
#endif // GGML_OPENCL_USE_ADRENO_KERNELS

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
void ggml_cl_mul_mat_q5_1_f32_adreno(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;
    ggml_tensor_extra_cl_q5_1 * extra0_q5_1 = (ggml_tensor_extra_cl_q5_1 *)src0->extra;

    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];

    const int ne1 = dst->ne[1];

    GGML_ASSERT(ne00 % ggml_blck_size(src0->type) == 0);

    cl_context context = backend_ctx->context;
    cl_kernel kernel;

    cl_int              err;
    cl_image_format     img_fmt;
    cl_image_desc       img_desc;
    cl_buffer_region    region;

    int M = ne01;
    int N = ne1;
    int K = ne00;

    if (ne1 == 1) {
        cl_mem qs_img = nullptr;
        cl_mem b_sub_buf = nullptr;
        cl_mem b_img = nullptr;

        // image for qs
        img_fmt = { CL_R, CL_UNSIGNED_INT32 };
        memset(&img_desc, 0, sizeof(img_desc));
        img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc.image_width = M * K / 2 / 4;
        img_desc.buffer = extra0_q5_1->qs;
        CL_CHECK((qs_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));

        // subbuffer for activations
        region.origin = offset1;
        region.size = K * N * sizeof(float);
        CL_CHECK((b_sub_buf = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

        // image for activations
        img_fmt = {CL_RGBA, CL_FLOAT};
        memset(&img_desc, 0, sizeof(img_desc));
        img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc.image_width = K * N / 4;
        img_desc.buffer = b_sub_buf;
        CL_CHECK((b_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));

        kernel = backend_ctx->mul_mat.kernel_gemv_noshuffle_q5_1_f32;

        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &qs_img));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),   &extra0_q5_1->qh));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extra0_q5_1->d));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem),   &extra0_q5_1->m));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem),   &b_img));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_mem),   &extrad->data_device));
        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_ulong), &offsetd));
        CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_int),   &ne00));
        CL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_int),   &ne01));

        size_t local_work_size[3] = {64, 4, 1};
        size_t global_work_size[3] = {(size_t)CEIL_DIV(ne01/2, 64)*64, 4, 1};

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);

        CL_CHECK(clReleaseMemObject(qs_img));
        CL_CHECK(clReleaseMemObject(b_sub_buf));
        CL_CHECK(clReleaseMemObject(b_img));
    } else {
        cl_mem b_sub_buf = nullptr;
        cl_mem b_sub_buf_trans = nullptr;
        cl_mem b_img = nullptr;
        cl_mem b_img_trans = nullptr;
        cl_mem d_sub_buf = nullptr;

        // subbuffer for activations
        region.origin = offset1;
        region.size = K * N * sizeof(float);
        CL_CHECK((b_sub_buf = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

        // image for activations
        img_fmt = {CL_RGBA, CL_FLOAT};
        memset(&img_desc, 0, sizeof(img_desc));
        img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc.image_width = K * N / 4;
        img_desc.buffer = b_sub_buf;
        CL_CHECK((b_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));

        // pad N to multiple of 8
        int extra_elements = N % 8;
        int padding = 0;
        if (extra_elements > 0){
            padding = 8 - extra_elements;
        }

        // subbuffer for transposed activations
        region.origin = 0;
        region.size = K * (N + padding) * sizeof(float)/2;
        backend_ctx->prealloc_act_trans.allocate(context, region.size);
        CL_CHECK((b_sub_buf_trans = clCreateSubBuffer(backend_ctx->prealloc_act_trans.buffer, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

        // image for transposed activations
        img_fmt = {CL_RGBA, CL_HALF_FLOAT};
        memset(&img_desc, 0, sizeof(img_desc));
        img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc.image_width = K * (N + padding) / 4;
        img_desc.buffer = b_sub_buf_trans;
        CL_CHECK((b_img_trans = clCreateImage(context, 0, &img_fmt, &img_desc, NULL, &err), err));

        // subbuffer for output
        region.origin = extrad->offset;
        region.size = M * N * sizeof(float);
        CL_CHECK((d_sub_buf = clCreateSubBuffer(extrad->data_device, CL_MEM_WRITE_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

        // transpose activations
        int height_B = N/4;
        if (height_B == 0) {
            height_B = 1;
        }
        int width_B = K/4;
        int padded_height_B = (N + padding)/4;

        kernel = backend_ctx->mul_mat.kernel_transpose_32_16;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &b_img));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_img_trans));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int),    &height_B));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int),    &width_B));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int),    &padded_height_B));

        size_t local_work_size_t[2] = { 1, 16 };
        size_t global_work_size_t[2] = { (size_t)width_B, (size_t)padded_height_B };
        backend_ctx->enqueue_ndrange_kernel(kernel, 2, global_work_size_t, local_work_size_t, dst);

        // gemm
        kernel = backend_ctx->mul_mat.kernel_gemm_noshuffle_q5_1_f32;
        int padded_N = N + padding;

        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0_q5_1->qs));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),   &extra0_q5_1->qh));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extra0_q5_1->d));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem),   &extra0_q5_1->m));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem),   &b_img_trans));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_mem),   &d_sub_buf));
        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_int),   &ne01));
        CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_int),   &padded_N));
        CL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_int),   &ne00));
        CL_CHECK(clSetKernelArg(kernel, 9, sizeof(cl_int),   &ne1));

        size_t global_work_size[3] = {(size_t)CEIL_DIV(ne1, 8), (size_t)CEIL_DIV(ne01, 4), 1};
        size_t local_work_size[3] = {1, 128, 1};

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);

        CL_CHECK(clReleaseMemObject(b_sub_buf));
        CL_CHECK(clReleaseMemObject(b_sub_buf_trans));
        CL_CHECK(clReleaseMemObject(b_img));
        CL_CHECK(clReleaseMemObject(b_img_trans));
        CL_CHECK(clReleaseMemObject(d_sub_buf));
    }
}
#endif // GGML_OPENCL_USE_ADRENO_KERNELS

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
void ggml_cl_mul_mat_iq4_nl_f32_adreno(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;
    ggml_tensor_extra_cl_iq4_nl * extra0_iq4_nl = (ggml_tensor_extra_cl_iq4_nl *)src0->extra;

    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    const int  ne00 = src0->ne[0];
    const int  ne01 = src0->ne[1];

    const int  ne1 = dst->ne[1];

    GGML_ASSERT(ne00 % 32 == 0);

    cl_context context = backend_ctx->context;
    cl_kernel kernel;

    cl_int              err;
    cl_image_format     img_fmt;
    cl_image_desc       img_desc;
    cl_buffer_region    region;

    int M = ne01;
    int N = ne1;
    int K = ne00;

    if (ne1 == 1) {
        cl_mem q_img = nullptr;
        cl_mem b_sub_buf = nullptr;
        cl_mem b_img = nullptr;

        // image for q
        img_fmt = { CL_R, CL_UNSIGNED_INT32};
        memset(&img_desc, 0, sizeof(img_desc));
        img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc.image_width = M * K / 2 / 4;
        img_desc.buffer = extra0_iq4_nl->q;
        CL_CHECK((q_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));

        // subbuffer for activations
        region.origin = offset1;
        region.size = K * N * sizeof(float);
        CL_CHECK((b_sub_buf = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

        // image for activations
        img_fmt = {CL_RGBA, CL_FLOAT};
        memset(&img_desc, 0, sizeof(img_desc));
        img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc.image_width = K * N / 4;
        img_desc.buffer = b_sub_buf;
        CL_CHECK((b_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));

        kernel = backend_ctx->mul_mat.kernel_gemv_noshuffle_iq4_nl_f32;

        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &q_img));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),   &extra0_iq4_nl->d));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &b_img));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem),   &extrad->data_device));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_ulong), &offsetd));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_int),   &ne00));
        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_int),   &ne01));

        size_t local_work_size[3] = {64, 4, 1};
        size_t global_work_size[3] = {(size_t)CEIL_DIV(ne01/2, 64)*64, 4, 1};

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);

        CL_CHECK(clReleaseMemObject(q_img));
        CL_CHECK(clReleaseMemObject(b_sub_buf));
        CL_CHECK(clReleaseMemObject(b_img));
    } else {
        // dp4a (int8) dense IQ4_NL prefill GEMM. Quantizes the [N,K] activations to
        // q8_1 and runs the int8 dot instead of the f16 half-dot. Large-batch
        // (ne1>8) only
        static const char * iq4nl_dense_dp4a_env = getenv("GGML_OPENCL_IQ4NL_DENSE_DP4A");
        bool iq4nl_dense_dp4a_on = iq4nl_dense_dp4a_env
            ? (atoi(iq4nl_dense_dp4a_env) != 0)
            : (backend_ctx->adreno_gen == ADRENO_GPU_GEN::X2E);
        // dot prod has to be available
        iq4nl_dense_dp4a_on = backend_ctx->has_integer_dot && iq4nl_dense_dp4a_on;

        if (iq4nl_dense_dp4a_on && backend_ctx->mul_mat.kernel_gemm_noshuffle_iq4_nl_q8_1_dp4a
                && N > 8 && (K % 32 == 0) && (M % 64 == 0)) {
            cl_mem a_sub = nullptr;
            region.origin = offset1;
            region.size   = (size_t)K * N * sizeof(float);
            CL_CHECK((a_sub = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

            const size_t n_blocks = (size_t)N * (K / 32);
            backend_ctx->prealloc_moe_qa.allocate(context, (size_t)N * K * sizeof(cl_char));
            backend_ctx->prealloc_moe_da.allocate(context, n_blocks * sizeof(cl_half));
            backend_ctx->prealloc_moe_sa.allocate(context, n_blocks * sizeof(cl_half));

            cl_int tb = (cl_int)n_blocks;
            cl_kernel qk = backend_ctx->mul_mat.kernel_quant_a_q8_1;
            CL_CHECK(clSetKernelArg(qk, 0, sizeof(cl_mem), &a_sub));
            CL_CHECK(clSetKernelArg(qk, 1, sizeof(cl_mem), &backend_ctx->prealloc_moe_qa.buffer));
            CL_CHECK(clSetKernelArg(qk, 2, sizeof(cl_mem), &backend_ctx->prealloc_moe_da.buffer));
            CL_CHECK(clSetKernelArg(qk, 3, sizeof(cl_mem), &backend_ctx->prealloc_moe_sa.buffer));
            CL_CHECK(clSetKernelArg(qk, 4, sizeof(cl_int), &tb));
            size_t q_local[1]  = { 64 };
            size_t q_global[1] = { (size_t)(((n_blocks + 63) / 64) * 64) };
            backend_ctx->enqueue_ndrange_kernel(qk, 1, q_global, q_local, dst);

            cl_kernel dk = backend_ctx->mul_mat.kernel_gemm_noshuffle_iq4_nl_q8_1_dp4a;
            int ai = 0;
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem),   &extra0_iq4_nl->q));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem),   &extra0_iq4_nl->d));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem),   &backend_ctx->prealloc_moe_qa.buffer));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem),   &backend_ctx->prealloc_moe_da.buffer));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_int),   &M));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_int),   &N));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_int),   &K));
            size_t d_local[3]  = { 64, 1, 1 };
            size_t d_global[3] = { 64, (size_t)(M / 64), (size_t)CEIL_DIV(N, 32) };
            backend_ctx->enqueue_ndrange_kernel(dk, 3, d_global, d_local, dst);

            CL_CHECK(clReleaseMemObject(a_sub));
            return;
        }

        cl_mem b_sub_buf = nullptr;
        cl_mem b_sub_buf_trans = nullptr;
        cl_mem b_img = nullptr;
        cl_mem b_img_trans = nullptr;

        // subbuffer for activations
        region.origin = offset1;
        region.size = K * N * sizeof(float);
        CL_CHECK((b_sub_buf = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

        // image for activations
        img_fmt = {CL_RGBA, CL_FLOAT};
        memset(&img_desc, 0, sizeof(img_desc));
        img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc.image_width = K * N / 4;
        img_desc.buffer = b_sub_buf;
        CL_CHECK((b_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));

        // pad N to multiple of 8
        int extra_elements = N % 8;
        int padding = 0;
        if (extra_elements > 0){
            padding = 8 - extra_elements;
        }

        // subbuffer for transposed activations
        region.origin = 0;
        region.size = K * (N + padding) * sizeof(float)/2;
        backend_ctx->prealloc_act_trans.allocate(context, region.size);
        CL_CHECK((b_sub_buf_trans = clCreateSubBuffer(backend_ctx->prealloc_act_trans.buffer, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

        // image for transposed activations
        img_fmt = {CL_RGBA, CL_HALF_FLOAT};
        memset(&img_desc, 0, sizeof(img_desc));
        img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc.image_width = K * (N + padding) / 4;
        img_desc.buffer = b_sub_buf_trans;
        CL_CHECK((b_img_trans = clCreateImage(context, 0, &img_fmt, &img_desc, NULL, &err), err));

        // transpose activations
        int height_B = N/4;
        if (height_B == 0) {
            height_B = 1;
        }
        int width_B = K/4;
        int padded_height_B = (N + padding)/4;

        kernel = backend_ctx->mul_mat.kernel_transpose_32_16;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &b_img));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_img_trans));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int),    &height_B));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int),    &width_B));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int),    &padded_height_B));

        size_t local_work_size_t[2] = { 1, 16 };
        size_t global_work_size_t[2] = { (size_t)width_B, (size_t)padded_height_B };
        backend_ctx->enqueue_ndrange_kernel(kernel, 2, global_work_size_t, local_work_size_t, dst);

        // gemm
        kernel = backend_ctx->mul_mat.kernel_gemm_noshuffle_iq4_nl_f32;
        int padded_N = N + padding;

        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0_iq4_nl->q));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),   &extra0_iq4_nl->d));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &b_img_trans));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem),   &extrad->data_device));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_ulong), &offsetd));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_int),   &ne01));
        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_int),   &padded_N));
        CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_int),   &ne00));
        CL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_int),   &ne1));

        size_t global_work_size[3] = {(size_t)CEIL_DIV(ne1, 8), (size_t)CEIL_DIV(ne01, 4), 1};
        size_t local_work_size[3] = {1, 128, 1};

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);

        CL_CHECK(clReleaseMemObject(b_sub_buf));
        CL_CHECK(clReleaseMemObject(b_sub_buf_trans));
        CL_CHECK(clReleaseMemObject(b_img));
        CL_CHECK(clReleaseMemObject(b_img_trans));
    }
}
#endif // GGML_OPENCL_USE_ADRENO_KERNELS

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
void ggml_cl_mul_mat_q8_0_f32_adreno(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    GGML_ASSERT(src0->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;
    // SoA extra lives on view_src (view->extra is pre-SoA).
    const ggml_tensor * soa0_src = src0->view_src != nullptr ? src0->view_src : src0;
    ggml_tensor_extra_cl_q8_0 * extra0_q8_0 = (ggml_tensor_extra_cl_q8_0 *)soa0_src->extra;

    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    const int  ne00 = src0->ne[0];
    const int  ne01 = src0->ne[1];
    const int  ne02 = src0->ne[2];

    const int  ne10 = src1->ne[0];
    const int  ne12 = src1->ne[2];

    const int  ne0 = dst->ne[0];
    const int  ne1 = dst->ne[1];

    GGML_ASSERT(ne00 == ne10);
    GGML_ASSERT((ne00 % 32) == 0);
    GGML_ASSERT(ne0 == ne01);

    cl_context context = backend_ctx->context;
    cl_kernel kernel;

    cl_int              err;
    cl_image_format     img_fmt;
    cl_image_desc       img_desc;
    cl_buffer_region    region;

    int M = ne01;
    int N = ne1;
    int K = ne00;

    if (ne1 == 1) {
        cl_mem q_img = nullptr;
        cl_mem b_sub_buf = nullptr;
        cl_mem b_img = nullptr;

        // image for q
        img_fmt = { CL_R, CL_UNSIGNED_INT32};
        memset(&img_desc, 0, sizeof(img_desc));
        img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc.image_width = M * K / 4;
        img_desc.buffer = extra0_q8_0->q;
        CL_CHECK((q_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));

        // create a sub_buffer for B
        region.origin = offset1;
        region.size = K * N * sizeof(float);
        CL_CHECK((b_sub_buf = clCreateSubBuffer((extra1->data_device), 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

        // image for activations
        img_fmt = {CL_RGBA, CL_FLOAT};
        memset(&img_desc, 0, sizeof(img_desc));
        img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc.image_width = K * N / 4;
        img_desc.buffer = b_sub_buf;
        CL_CHECK((b_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));

        // Split-K for small-M decode GEMVs. The base kernel puts one output row
        // per lane and splits K only inside one workgroup, so M is the sole source
        // of workgroup parallelism: gpt-oss's K and V projections are M=512 = 8
        // workgroups on a 16-CU X2, and the kernel measures 48 GB/s where the
        // M=2880/4096 projections in the same decode graph reach 122-123. Mirrors
        // the q4_0/q4_K split-K above and reuses their reduce kernel.
        //
        // Enabled where it is measured to win, like the q4_K gate: X2-90 +2.8%
        // tg32 @d4096 on gpt-oss; Adreno 840 (12 CU) NEUTRAL on Llama-3.2-3B-Q8_0
        // (0.0% @d4096 -- its K/V proj is M=1024 = 16 workgroups, which already
        // fills 12 CUs). Unmeasured on X1E/A7X/A6X and the q4_K split-K measured
        // -0.7% on X1E, so the default is not widened on absence of evidence.
        static const bool q8_splitk_env_set = []{
            const char * e = std::getenv("GGML_OPENCL_Q8_GEMV_SPLITK");
            return e && e[0] != '\0';
        }();
        static const bool q8_splitk_env_on = []{
            const char * e = std::getenv("GGML_OPENCL_Q8_GEMV_SPLITK");
            return !(e && e[0] == '0');
        }();
        const bool q8_splitk_on = q8_splitk_env_set
            ? q8_splitk_env_on
            : (backend_ctx->adreno_gen == ADRENO_GPU_GEN::X2E);
        if (q8_splitk_on && backend_ctx->mul_mat.kernel_gemv_noshuffle_q8_0_f32_splitk &&
            ne01 <= 1024 && ne01 % 64 == 0) {
            const int    nsg    = 8;
            const int    ksplit = 8;                        // -> 8 * M/64 workgroups
            const size_t gx     = (size_t) CEIL_DIV(ne01, 64) * 64;

            backend_ctx->prealloc_splitk_partial.allocate(
                backend_ctx->context, (size_t) ksplit * ne01 * sizeof(float));
            cl_mem partial = backend_ctx->prealloc_splitk_partial.buffer;

            cl_kernel ks = backend_ctx->mul_mat.kernel_gemv_noshuffle_q8_0_f32_splitk;
            CL_CHECK(clSetKernelArg(ks, 0, sizeof(cl_mem), &q_img));
            CL_CHECK(clSetKernelArg(ks, 1, sizeof(cl_mem), &extra0_q8_0->d));
            CL_CHECK(clSetKernelArg(ks, 2, sizeof(cl_mem), &b_img));
            CL_CHECK(clSetKernelArg(ks, 3, sizeof(cl_mem), &partial));
            CL_CHECK(clSetKernelArg(ks, 4, sizeof(cl_int), &ne00));
            CL_CHECK(clSetKernelArg(ks, 5, sizeof(cl_int), &ne01));
            size_t lsk[3] = { 64, (size_t) nsg, 1 };
            size_t gsk[3] = { gx, (size_t) (nsg * ksplit), 1 };
            backend_ctx->enqueue_ndrange_kernel(ks, 3, gsk, lsk, dst);

            cl_kernel kr = backend_ctx->mul_mat.kernel_gemv_splitk_reduce_f32;
            CL_CHECK(clSetKernelArg(kr, 0, sizeof(cl_mem),   &partial));
            CL_CHECK(clSetKernelArg(kr, 1, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kr, 2, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kr, 3, sizeof(cl_int),   &ne01));
            CL_CHECK(clSetKernelArg(kr, 4, sizeof(cl_int),   &ksplit));
            size_t lr[3] = { 64, 1, 1 };
            size_t gr[3] = { (size_t) CEIL_DIV(ne01, 64) * 64, 1, 1 };
            backend_ctx->enqueue_ndrange_kernel(kr, 3, gr, lr, dst);

            CL_CHECK(clReleaseMemObject(q_img));
            CL_CHECK(clReleaseMemObject(b_img));
            CL_CHECK(clReleaseMemObject(b_sub_buf));
            return;
        }

        kernel = backend_ctx->mul_mat.kernel_gemv_noshuffle_q8_0_f32;

        int r2 = 1;
        int r3 = 1;

        CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &q_img));
        CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0_q8_0->d));
        CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &b_img));
        CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
        CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
        CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
        CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
        CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
        CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne02));
        CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne10));
        CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne12));
        CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne0));
        CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne1));
        CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &r2));
        CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &r3));

        size_t wavesize = backend_ctx->adreno_wave_size;
        size_t local_work_size[]  = { wavesize, 4, 1 };
        size_t global_work_size[] = { CEIL_DIV(M, wavesize)*wavesize, 4, 1 };

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);

        CL_CHECK(clReleaseMemObject(q_img));
        CL_CHECK(clReleaseMemObject(b_img));
        CL_CHECK(clReleaseMemObject(b_sub_buf));
    } else {
        // dp4a dense q8_0 prefill GEMM. Quantizes the [N,K] activations to
        // q8_1 and runs the int8 dot instead of the f16 half-dot. Large-batch
        // (ne1>8) only; q8_0 weights are already int8 (no requant) and symmetric
        // (no min term)
        static const char * q8_dense_dp4a_env = getenv("GGML_OPENCL_Q8_DENSE_DP4A");
        static const char * q8_dense_wimg_env = getenv("GGML_OPENCL_Q8_DENSE_DP4A_WIMG");
        const bool q8_dense_wimg_on = q8_dense_wimg_env && (atoi(q8_dense_wimg_env) != 0);

        const bool q8_bin_loaded = (backend_ctx->mul_mat.kernel_gemm_noshuffle_q8_0_f32_bin != nullptr);
        // bin kernel takes precedence
        bool q8_dense_dp4a_on = q8_dense_wimg_on
            ? true
            : q8_dense_dp4a_env
            ? (atoi(q8_dense_dp4a_env) != 0)
            : (backend_ctx->adreno_gen == ADRENO_GPU_GEN::X2E && !q8_bin_loaded);
        // dot prod has to be available
        q8_dense_dp4a_on = backend_ctx->has_integer_dot && q8_dense_dp4a_on;

        if (q8_dense_dp4a_on && backend_ctx->mul_mat.kernel_gemm_noshuffle_q8_0_q8_1_dp4a
                && N > 8 && (K % 32 == 0) && (M % 64 == 0)) {
            cl_mem a_sub = nullptr;
            region.origin = offset1;
            region.size   = (size_t)K * N * sizeof(float);
            CL_CHECK((a_sub = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

            const size_t n_blocks = (size_t)N * (K / 32);
            backend_ctx->prealloc_moe_qa.allocate(context, (size_t)N * K * sizeof(cl_char));
            backend_ctx->prealloc_moe_da.allocate(context, n_blocks * sizeof(cl_half));
            backend_ctx->prealloc_moe_sa.allocate(context, n_blocks * sizeof(cl_half));

            cl_int tb = (cl_int)n_blocks;
            cl_kernel qk = backend_ctx->mul_mat.kernel_quant_a_q8_1;
            CL_CHECK(clSetKernelArg(qk, 0, sizeof(cl_mem), &a_sub));
            CL_CHECK(clSetKernelArg(qk, 1, sizeof(cl_mem), &backend_ctx->prealloc_moe_qa.buffer));
            CL_CHECK(clSetKernelArg(qk, 2, sizeof(cl_mem), &backend_ctx->prealloc_moe_da.buffer));
            CL_CHECK(clSetKernelArg(qk, 3, sizeof(cl_mem), &backend_ctx->prealloc_moe_sa.buffer));
            CL_CHECK(clSetKernelArg(qk, 4, sizeof(cl_int), &tb));
            size_t q_local[1]  = { 64 };
            size_t q_global[1] = { (size_t)(((n_blocks + 63) / 64) * 64) };
            backend_ctx->enqueue_ndrange_kernel(qk, 1, q_global, q_local, dst);

            // optional weight texture, the same CL_R/UINT32 view, width M*K/4
            cl_mem q8_q_img = nullptr;
            bool use_wimg = q8_dense_wimg_on;
            if (use_wimg) {
                const size_t tex = (size_t)M * (size_t)K / 4;  // uint32 texels
                if (tex == 0 || tex > backend_ctx->image_max_buffer_size) {
                    use_wimg = false;
                } else {
                    img_fmt = { CL_R, CL_UNSIGNED_INT32 };
                    memset(&img_desc, 0, sizeof(img_desc));
                    img_desc.image_type  = CL_MEM_OBJECT_IMAGE1D_BUFFER;
                    img_desc.image_width = tex;
                    img_desc.buffer      = extra0_q8_0->q;
                    q8_q_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err);
                    if (err != CL_SUCCESS || q8_q_img == nullptr) { use_wimg = false; q8_q_img = nullptr; }
                }
            }

            cl_kernel dk = use_wimg ? backend_ctx->mul_mat.kernel_gemm_noshuffle_q8_0_q8_1_dp4a_wimg
                                    : backend_ctx->mul_mat.kernel_gemm_noshuffle_q8_0_q8_1_dp4a;
            int ai = 0;
            if (use_wimg) {
                CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem), &q8_q_img));
            } else {
                CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem), &extra0_q8_0->q));
            }
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem),   &extra0_q8_0->d));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem),   &backend_ctx->prealloc_moe_qa.buffer));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem),   &backend_ctx->prealloc_moe_da.buffer));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_int),   &M));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_int),   &N));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_int),   &K));
            size_t d_local[3]  = { 64, 1, 1 };
            size_t d_global[3] = { 64, (size_t)(M / 64), (size_t)CEIL_DIV(N, 32) };
            backend_ctx->enqueue_ndrange_kernel(dk, 3, d_global, d_local, dst);

            if (q8_q_img != nullptr) {
                CL_CHECK(clReleaseMemObject(q8_q_img));
            }
            CL_CHECK(clReleaseMemObject(a_sub));
            return;
        }

        // use bin kernel if available
        if (backend_ctx->mul_mat.kernel_gemm_noshuffle_q8_0_f32_bin) {
            int K_pad = K;

            cl_mem b_sub_buf = nullptr;
            cl_mem d_sub_buf = nullptr;

            cl_mem a_img = nullptr;
            cl_mem s_img = nullptr;
            cl_mem b_img = nullptr;
            cl_mem d_img = nullptr;

            // subbuffer for activations
            region.origin = offset1;
            region.size = K_pad * N * sizeof(float);
            CL_CHECK((b_sub_buf = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

            // Create subbuffer and image1d_buffer for dst
            region.origin = (extrad->offset); // + dst->view_offs;
            region.size = M * N * sizeof(float);
            CL_CHECK((d_sub_buf = clCreateSubBuffer((extrad->data_device), 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

            // create an image for A
            img_fmt = { CL_R, CL_FLOAT};
            memset(&img_desc, 0, sizeof(img_desc));
            img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
            img_desc.image_width = M * K / 4;    // Divide by 4 for char -> float
            img_desc.buffer = extra0_q8_0->q;
            CL_CHECK((a_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));

            // create an image for Scale
            img_fmt = { CL_R, CL_HALF_FLOAT};
            memset(&img_desc, 0, sizeof(img_desc));
            img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
            img_desc.image_width = M * K / 32;    // Block size is 32
            img_desc.buffer = extra0_q8_0->d;
            CL_CHECK((s_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));

            // create an image for B from sub_buffer
            img_fmt = {CL_R, CL_FLOAT};
            memset(&img_desc, 0, sizeof(img_desc));
            img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
            img_desc.image_width = K_pad * N;
            img_desc.buffer = b_sub_buf;
            CL_CHECK((b_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));

            // img for d
            img_fmt = {CL_R, CL_FLOAT};
            memset(&img_desc, 0, sizeof(img_desc));
            img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
            img_desc.image_width = M * N;
            img_desc.buffer = d_sub_buf;
            CL_CHECK((d_img = clCreateImage(context, CL_MEM_WRITE_ONLY, &img_fmt, &img_desc, NULL, &err), err));

            // gemm
            kernel = backend_ctx->mul_mat.kernel_gemm_noshuffle_q8_0_f32_bin;

            bool layoutA_Mfirst = true;
            bool layoutS_Mfirst = true;
            bool layoutB_Nfirst = false;
            bool layoutC_Mfirst = true;

            cl_uint lineStrideMatrixAinBytes = layoutA_Mfirst ? M * 4 : K;                // int8
            cl_uint lineStrideMatrixSinBytes = layoutS_Mfirst ? M * 2 : (K / 32) * 2;     // fp16
            cl_uint lineStrideMatrixBinBytes = layoutB_Nfirst ? N * 4 : K_pad * 4;        // fp32
            cl_uint lineStrideMatrixCinBytes = layoutC_Mfirst ? M * 4 : N * 4;            // fp32

            CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem), &a_img));
            CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem), &s_img));
            CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem), &b_img));
            CL_CHECK(clSetKernelArg(kernel,  3, sizeof(int),    &extra1->offset));
            CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem), &d_img));
            CL_CHECK(clSetKernelArg(kernel,  5, sizeof(int),    &extrad->offset));
            CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),    &K));
            CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),    &lineStrideMatrixAinBytes));
            CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),    &lineStrideMatrixSinBytes));
            CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),    &lineStrideMatrixBinBytes));
            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),    &lineStrideMatrixCinBytes));

            size_t global_work_size[] = { 64, (size_t)CEIL_DIV(M, 64), (size_t)CEIL_DIV(N, 64)};
            size_t local_work_size[]  = { 64, 2, 2 };

            backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);

            CL_CHECK(clReleaseMemObject(b_sub_buf));
            CL_CHECK(clReleaseMemObject(d_sub_buf));
            CL_CHECK(clReleaseMemObject(a_img));
            CL_CHECK(clReleaseMemObject(s_img));
            CL_CHECK(clReleaseMemObject(b_img));
            CL_CHECK(clReleaseMemObject(d_img));
            return;
        }

        cl_mem b_sub_buf = nullptr;
        cl_mem b_sub_buf_trans = nullptr;
        cl_mem b_img = nullptr;
        cl_mem b_img_trans = nullptr;

        // subbuffer for activations
        region.origin = offset1;
        region.size = K * N * sizeof(float);
        CL_CHECK((b_sub_buf = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

        // image for activations
        img_fmt = {CL_RGBA, CL_FLOAT};
        memset(&img_desc, 0, sizeof(img_desc));
        img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc.image_width = K * N / 4;
        img_desc.buffer = b_sub_buf;
        CL_CHECK((b_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));

        // pad N to multiple of 8
        int extra_elements = N % 8;
        int padding = 0;
        if (extra_elements > 0){
            padding = 8 - extra_elements;
        }

        // subbuffer for transposed activations
        region.origin = 0;
        region.size = K * (N + padding) * sizeof(float)/2;
        backend_ctx->prealloc_act_trans.allocate(context, region.size);
        CL_CHECK((b_sub_buf_trans = clCreateSubBuffer(backend_ctx->prealloc_act_trans.buffer, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

        // image for transposed activations
        img_fmt = {CL_RGBA, CL_HALF_FLOAT};
        memset(&img_desc, 0, sizeof(img_desc));
        img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc.image_width = K * (N + padding) / 4;
        img_desc.buffer = b_sub_buf_trans;
        CL_CHECK((b_img_trans = clCreateImage(context, 0, &img_fmt, &img_desc, NULL, &err), err));

        // transpose activations
        int height_B = N/4;
        if (height_B == 0) {
            height_B = 1;
        }
        int width_B = K/4;
        int padded_height_B = (N + padding)/4;

        kernel = backend_ctx->mul_mat.kernel_transpose_32_16;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &b_img));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_img_trans));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int),    &height_B));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int),    &width_B));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int),    &padded_height_B));

        size_t local_work_size_t[2] = { 1, 16 };
        size_t global_work_size_t[2] = { (size_t)width_B, (size_t)padded_height_B };
        backend_ctx->enqueue_ndrange_kernel(kernel, 2, global_work_size_t, local_work_size_t, dst);

        // gemm
        kernel = backend_ctx->mul_mat.kernel_gemm_noshuffle_q8_0_f32;
        int padded_N = N + padding;

        CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0_q8_0->q));
        CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0_q8_0->d));
        CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &b_img_trans));
        CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_mem),   &extrad->data_device));
        CL_CHECK(clSetKernelArg(kernel,  4, sizeof(int),      &K));
        CL_CHECK(clSetKernelArg(kernel,  5, sizeof(int),      &M));
        CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &padded_N));
        CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &N));
        CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &offsetd));

        size_t global_work_size[] = { (size_t)CEIL_DIV(N, 8), (size_t)CEIL_DIV(M, 4), 1 };
        size_t local_work_size[]  = { 2, 128, 1 };

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);

        CL_CHECK(clReleaseMemObject(b_img_trans));
        CL_CHECK(clReleaseMemObject(b_sub_buf_trans));
        CL_CHECK(clReleaseMemObject(b_img));
        CL_CHECK(clReleaseMemObject(b_sub_buf));
    }
}
#endif // GGML_OPENCL_USE_ADRENO_KERNELS

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
static void ggml_cl_mul_mat_q4_k_f32_adreno_ila(ggml_backend_t backend, const ggml_tensor * src0,
                                                const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;
    ggml_tensor_extra_cl_q4_K * extra0_q4_k = (ggml_tensor_extra_cl_q4_K *)src0->extra;

    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];

    const int ne1 = dst->ne[1];

    GGML_ASSERT(ne00 % ggml_blck_size(src0->type) == 0);

    cl_context context = backend_ctx->context;
    cl_kernel kernel;

    cl_int              err;
    cl_image_format     img_fmt;
    cl_image_desc       img_desc;
    cl_buffer_region    region;

    int M = ne01;
    int N = ne1;
    int K = ne00;

    if (ne1 == 1) {
        cl_mem b_sub_buf = nullptr;
        cl_mem b_img     = nullptr;

        region.origin = offset1;
        region.size   = (size_t)K * N * sizeof(float);
        CL_CHECK((b_sub_buf = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

        img_fmt = { CL_RGBA, CL_FLOAT };
        memset(&img_desc, 0, sizeof(img_desc));
        img_desc.image_type  = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc.image_width = (size_t)K * N / 4;
        img_desc.buffer      = b_sub_buf;
        CL_CHECK((b_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));

        kernel = backend_ctx->mul_mat.kernel_gemv_noshuffle_q4_k_f32_32b_trans;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0_q4_k->q_img));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),   &extra0_q4_k->d));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extra0_q4_k->dm));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem),   &extra0_q4_k->s));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem),   &b_img));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_mem),   &extrad->data_device));
        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_ulong), &offsetd));
        CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_int),   &ne00));
        CL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_int),   &ne01));

        size_t local_work_size[3]  = { 64, 8, 1 };
        size_t global_work_size[3] = { (size_t)ne01, 8, 1 };
        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);

        CL_CHECK(clReleaseMemObject(b_sub_buf));
        CL_CHECK(clReleaseMemObject(b_img));
    } else {
        const int gemm_tile_n = 64;
        int N_pad = CEIL_DIV(N, gemm_tile_n) * gemm_tile_n;

        static const char * q4_k_bin_dp4a_env = getenv("GGML_OPENCL_Q4_K_BIN_DP4A");
                     bool   q4_k_bin_dp4a_on  = q4_k_bin_dp4a_env
                                                  ? (atoi(q4_k_bin_dp4a_env) != 0)
                                                  : true;
        // dot prod has to be available
        q4_k_bin_dp4a_on = backend_ctx->has_integer_dot && q4_k_bin_dp4a_on;

        if (q4_k_bin_dp4a_on && backend_ctx->mul_mat.kernel_gemm_noshuffle_q4_k_q8_1_dp4a_ila_a8_bin) {
            const int    dp4a_N_pad = CEIL_DIV(N, 32) * 32;
            const size_t n_blocks   = (size_t)dp4a_N_pad * (K / 32);

            backend_ctx->prealloc_moe_qa.allocate(context, (size_t)dp4a_N_pad * K * sizeof(cl_char));
            backend_ctx->prealloc_moe_da.allocate(context, n_blocks * sizeof(cl_half));
            backend_ctx->prealloc_moe_sa.allocate(context, n_blocks * sizeof(cl_half));

            cl_mem b_sub = nullptr;
            region.origin = offset1;
            region.size   = (size_t)K * N * sizeof(float);
            CL_CHECK((b_sub = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

            cl_int    tb = (cl_int)((size_t)N * (K / 32));
            cl_kernel qk = backend_ctx->mul_mat.kernel_quant_a_q8_1;
            CL_CHECK(clSetKernelArg(qk, 0, sizeof(cl_mem), &b_sub));
            CL_CHECK(clSetKernelArg(qk, 1, sizeof(cl_mem), &backend_ctx->prealloc_moe_qa.buffer));
            CL_CHECK(clSetKernelArg(qk, 2, sizeof(cl_mem), &backend_ctx->prealloc_moe_da.buffer));
            CL_CHECK(clSetKernelArg(qk, 3, sizeof(cl_mem), &backend_ctx->prealloc_moe_sa.buffer));
            CL_CHECK(clSetKernelArg(qk, 4, sizeof(cl_int), &tb));
            size_t q_local[1]  = { 64 };
            size_t q_global[1] = { (size_t)CEIL_DIV(tb, 64) * 64 };
            backend_ctx->enqueue_ndrange_kernel(qk, 1, q_global, q_local, dst);

            cl_mem d_sub = nullptr;
            cl_mem d_img = nullptr;
            region.origin = offsetd;
            region.size   = (size_t)M * N * sizeof(float);
            CL_CHECK((d_sub = clCreateSubBuffer(extrad->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

            img_fmt = { CL_R, CL_FLOAT };
            memset(&img_desc, 0, sizeof(img_desc));
            img_desc.image_type  = CL_MEM_OBJECT_IMAGE1D_BUFFER;
            img_desc.image_width = (size_t)M * N;
            img_desc.buffer      = d_sub;
            CL_CHECK((d_img = clCreateImage(context, CL_MEM_WRITE_ONLY, &img_fmt, &img_desc, NULL, &err), err));

            kernel = backend_ctx->mul_mat.kernel_gemm_noshuffle_q4_k_q8_1_dp4a_ila_a8_bin;

            cl_uint k_arg = 0;
            CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_mem),  &extra0_q4_k->q_img));
            CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_mem),  &extra0_q4_k->d));
            CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_mem),  &extra0_q4_k->dm));
            CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_mem),  &extra0_q4_k->s));
            CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_mem),  &backend_ctx->prealloc_moe_qa.buffer));
            CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_mem),  &backend_ctx->prealloc_moe_da.buffer));
            CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_mem),  &backend_ctx->prealloc_moe_sa.buffer));
            CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_mem),  &d_img));
            CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_uint), &ne00));
            CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_uint), &ne01));
            CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_int),  &N));

            size_t local_work_size[3]  = { 64, 1, 1 };
            size_t global_work_size[3] = { 64, (size_t)(M / 64), (size_t)(dp4a_N_pad / 32) };
            backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);

            CL_CHECK(clReleaseMemObject(b_sub));
            CL_CHECK(clReleaseMemObject(d_img));
            CL_CHECK(clReleaseMemObject(d_sub));
            return;
        }

        cl_mem b_sub_buf = nullptr;
        cl_mem b_padded  = nullptr;
        cl_mem b_buf     = nullptr;
        if (N_pad == N) {
            region.origin = offset1;
            region.size   = (size_t)K * N * sizeof(float);
            CL_CHECK((b_sub_buf = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));
            b_buf = b_sub_buf;
        } else {
            CL_CHECK((b_padded = clCreateBuffer(context, CL_MEM_READ_WRITE, (size_t)K * N_pad * sizeof(float), NULL, &err), err));
            const float zero = 0.0f;
            CL_CHECK(clEnqueueFillBuffer(backend_ctx->queue, b_padded, &zero, sizeof(zero), 0, (size_t)K * N_pad * sizeof(float), 0, NULL, NULL));
            CL_CHECK(clEnqueueCopyBuffer(backend_ctx->queue, extra1->data_device, b_padded, offset1, 0, (size_t)K * N * sizeof(float), 0, NULL, NULL));
            b_buf = b_padded;
        }

        img_fmt = { CL_R, CL_FLOAT };
        memset(&img_desc, 0, sizeof(img_desc));
        img_desc.image_type  = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc.image_width = (size_t)K * N_pad;
        img_desc.buffer      = b_buf;
        cl_mem b_img;
        CL_CHECK((b_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));

        region.origin = offsetd;
        region.size   = (size_t)M * N * sizeof(float);
        cl_mem d_sub_buf;
        CL_CHECK((d_sub_buf = clCreateSubBuffer(extrad->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));
        img_fmt = { CL_R, CL_FLOAT };
        memset(&img_desc, 0, sizeof(img_desc));
        img_desc.image_type  = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc.image_width = (size_t)M * N;
        img_desc.buffer      = d_sub_buf;
        cl_mem d_img;
        CL_CHECK((d_img = clCreateImage(context, CL_MEM_WRITE_ONLY, &img_fmt, &img_desc, NULL, &err), err));

        kernel = backend_ctx->mul_mat.kernel_gemm_noshuffle_q4_k_f32_32b_trans_ila_a8_bin;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),  &extra0_q4_k->q_img));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),  &extra0_q4_k->d));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),  &extra0_q4_k->dm));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem),  &extra0_q4_k->s));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem),  &b_img));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_mem),  &d_img));
        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_uint), &ne00));
        CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_uint), &ne01));
        CL_CHECK(clSetKernelArg(kernel, 8, sizeof(int),     &N));

        size_t local_work_size[3]  = { 64, 2, 2 };
        size_t m_tiles = (size_t)CEIL_DIV(M, 64);
        size_t global_work_size[3] = { 64, m_tiles, (size_t)CEIL_DIV(N_pad, gemm_tile_n) };
        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);

        CL_CHECK(clReleaseMemObject(b_img));
        if (b_sub_buf) {
            CL_CHECK(clReleaseMemObject(b_sub_buf));
        }
        if (b_padded) {
            CL_CHECK(clReleaseMemObject(b_padded));
        }
        CL_CHECK(clReleaseMemObject(d_img));
        CL_CHECK(clReleaseMemObject(d_sub_buf));
    }
}
#endif // GGML_OPENCL_USE_ADRENO_KERNELS

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
void ggml_cl_mul_mat_q4_K_f32_adreno(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;
    ggml_tensor_extra_cl_q4_K * extra0_q4_k = (ggml_tensor_extra_cl_q4_K *)src0->extra;

    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    const int  ne00 = src0->ne[0];
    const int  ne01 = src0->ne[1];

    const int  ne1 = dst->ne[1];

    GGML_ASSERT(ne00 % ggml_blck_size(src0->type) == 0);

    cl_context context = backend_ctx->context;
    cl_kernel kernel;

    cl_int              err;
    cl_image_format     img_fmt;
    cl_image_desc       img_desc;
    cl_buffer_region    region;

    int M = ne01;
    int N = ne1;
    int K = ne00;

    cl_uchar mask_d6 = 0x3F;
    cl_uchar mask_d4 = 0x0F;
    cl_uchar mask_hi2 = 0xC0;

    // Multi-column verify GEMV: route the spec/MTP verify batch (ne1==3 = 2
    // drafts + 1 bonus) onto the efficient GEMV path (subgroup-broadcast, no
    // transpose) instead of the transposed-GEMM dead-zone. Reuses the ne1==1
    // GEMV setup (the activation image is already sized by N=ne1). Byte-
    // identical. Opt-in via GGML_OPENCL_Q4K_MC3=1 while validating.
    static const bool q4k_mc3 = (getenv("GGML_OPENCL_Q4K_MC3") != nullptr);
    // Per-layer only (ne01 < 32768): the batched large-vocab lm_head at ne1==3
    // is left to the existing routing (corrupts on the Adreno GEMV path; x2-
    // unified routes batched Q6_K lm_head to CPU). Per-layer mc3 is byte-identical.
    const bool use_mc3 = q4k_mc3 && (ne1 == 3) && (ne01 < 32768);

    const bool use_bin = use_q4_k_bin_kernels(backend_ctx, src0);

    if (use_bin) {
        if (use_mc3) {
            static bool warned = false;
            if (!warned) {
                GGML_LOG_WARN("ggml_opencl: GGML_OPENCL_Q4K_MC3 is bypassed by Q4_K binary kernels\n");
                warned = true;
            }
        }
        ggml_cl_mul_mat_q4_k_f32_adreno_ila(backend, src0, src1, dst);
        return;
    }

    if (ne1 == 1 || use_mc3) {
        cl_mem q_img = nullptr;
        cl_mem b_sub_buf = nullptr;
        cl_mem b_img = nullptr;

        const bool use_tiled = !use_mc3 && use_q4k_tiled(backend_ctx, src0);

        // image for q (not needed for the tiled path, which reads __global)
        if (!use_tiled) {
            img_fmt = { CL_R, CL_UNSIGNED_INT32};
            memset(&img_desc, 0, sizeof(img_desc));
            img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
            img_desc.image_width = M * K / 2 / 4;
            img_desc.buffer = extra0_q4_k->q;
            CL_CHECK((q_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));
        }

        // subbuffer for activations
        region.origin = offset1;
        region.size = K * N * sizeof(float);
        CL_CHECK((b_sub_buf = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

        // image for activations
        img_fmt = {CL_RGBA, CL_FLOAT};
        memset(&img_desc, 0, sizeof(img_desc));
        img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc.image_width = K * N / 4;
        img_desc.buffer = b_sub_buf;
        CL_CHECK((b_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));

        // 4-output-per-WI o4 variant for the long-vocab lm_head/embed GEMV
        // (ne01 = vocab ~256K on Gemma): shares one activation read across 4
        // output rows. Gated to large ne01 (lm_head/embed). Default on; opt-out
        // GGML_OPENCL_Q4K_GEMV_O4=0. (Skipped when mc3 handles the ne1==3 verify.)
        static const bool q4k_o4_env = []{
            const char * e = std::getenv("GGML_OPENCL_Q4K_GEMV_O4");
            return !e || e[0] == '\0' || e[0] != '0';
        }();
        const bool use_q4k_o4 = !use_tiled && !use_mc3 && q4k_o4_env && (ne01 % 4 == 0) && (ne01 >= 32768);
        // Split-K across workgroups for small-M decode GEMVs. A single-token GEMV
        // makes only CEIL_DIV(M/2,64) workgroups; even with the wide intra-WG split
        // (16 subgroups) those all land on ONE CU, so small-M matmuls under-fill the
        // 16 CUs and their bandwidth falls well short of what the large-M FFN matmuls
        // reach. Adding a `ksplit` second grid dim that spreads K across WGs (+ a
        // reduce pass) fills the CUs. Gate is M<=2560: the tiny M<=1024 ones only
        // break even (the reduce dispatch eats the kernel win), but the big-K M=2560
        // cases (ffn_down, attn_output) make the per-call win dwarf the reduce, and
        // are byte-identical. ffn_gate/up (large M) fill the CUs already and are excluded.
        //
        // DEVICE-GATED. Split-K buys GPU time by spending an extra kernel LAUNCH (the
        // reduce), so it only pays where launches are cheap. That is a per-device
        // property and it does not travel from the X2-90 this was tuned on. Measured
        // with one binary, env A/B (tg32, GGML_OPENCL_Q4K_GEMV_SPLITK=0/1):
        //
        //     X2-90   +3.36%   gemma-4 E4B      (the number this gate was built on)
        //     840      -1.3%   Qwen3.5-4B-Q4_K_M   14.00 -> 13.85
        //     850     -20.0%   Qwen3-1.7B-Q4_K_M    6.97 -> 5.58  (6 interleaved reps)
        //
        // The kernel is not the problem. On the 850 split-K makes the GPU strictly
        // faster -- total busy 537 -> 485 ms, this GEMV 43.7 -> 34.0 us/call (-22%) --
        // and still costs a fifth of decode, because the +3696 reduce dispatches cost
        // ~550 us of HOST round-trip each against 2.7 us of GPU work (~200x; that part
        // is ~95% host-bound at decode). The 840 pays the same tax at ~42 us/dispatch.
        // Break-even needs launch cost below the ~9.7 us/call the split actually saves,
        // so this is not a "the 850 is slow" adjustment that a faster part would fix --
        // the 840 is 13x cheaper per launch and still loses.
        //
        // Enabled where it is measured to win, i.e. X2E only. The X1-85 was measured
        // afterwards and is NOT a win either: Qwen3.5-4B-Q4_K_M tg32, split-K off
        // 17.98/18.10/18.19 vs on 18.03/17.91/17.97 = -0.7%, so X1E stays excluded on
        // evidence rather than on absence of it. Do not widen this without a NEW
        // measurement. The env still forces either way so every device stays measurable.
        static const bool splitk_env_set = []{
            const char * e = std::getenv("GGML_OPENCL_Q4K_GEMV_SPLITK");
            return e && e[0] != '\0';
        }();
        static const bool splitk_env_on = []{
            const char * e = std::getenv("GGML_OPENCL_Q4K_GEMV_SPLITK");
            return !(e && e[0] == '0');
        }();
        const bool splitk_wg_env = splitk_env_set
            ? splitk_env_on
            : (backend_ctx->adreno_gen == ADRENO_GPU_GEN::X2E);
        // Gate: small-M decode GEMVs that under-fill the 16 CUs even with the wide
        // intra-WG split (all 16 subgroups land on one CU). M<=2560 covers Kcur/Vcur
        // (M=1024), Qcur (2048), attn_output + ffn_down (2560). The tiny ones
        // (M<=1024) only break even (reduce dispatch eats the kernel win), but the
        // big-K M=2560 cases (ffn_down K=10240 @182us, attn_output @42us) have a
        // large per-call win that dwarfs the ~5us reduce, so extending to 2560 nets
        // positive end-to-end. ffn_gate/up (M=10240) already fill the CUs -> excluded.
        const bool use_splitk = splitk_wg_env && !use_tiled && !use_q4k_o4 && !use_mc3 && ne01 <= 2560;

        if (use_splitk) {
            const int    nsg    = 8;
            const int    ksplit = (ne01 <= 512) ? 8 : 4;   // -> ~32 total WGs
            const size_t gx     = (size_t)CEIL_DIV(ne01/2, 64) * 64;

            backend_ctx->prealloc_splitk_partial.allocate(
                backend_ctx->context, (size_t)ksplit * ne01 * sizeof(float));
            cl_mem partial = backend_ctx->prealloc_splitk_partial.buffer;

            cl_kernel ks = backend_ctx->mul_mat.kernel_gemv_noshuffle_q4_k_f32_splitk;
            CL_CHECK(clSetKernelArg(ks, 0,  sizeof(cl_mem),   &q_img));
            CL_CHECK(clSetKernelArg(ks, 1,  sizeof(cl_mem),   &extra0_q4_k->d));
            CL_CHECK(clSetKernelArg(ks, 2,  sizeof(cl_mem),   &extra0_q4_k->dm));
            CL_CHECK(clSetKernelArg(ks, 3,  sizeof(cl_mem),   &extra0_q4_k->s));
            CL_CHECK(clSetKernelArg(ks, 4,  sizeof(cl_mem),   &b_img));
            CL_CHECK(clSetKernelArg(ks, 5,  sizeof(cl_mem),   &partial));
            CL_CHECK(clSetKernelArg(ks, 6,  sizeof(cl_int),   &ne00));
            CL_CHECK(clSetKernelArg(ks, 7,  sizeof(cl_int),   &ne01));
            CL_CHECK(clSetKernelArg(ks, 8,  sizeof(cl_uchar), &mask_d6));
            CL_CHECK(clSetKernelArg(ks, 9,  sizeof(cl_uchar), &mask_d4));
            CL_CHECK(clSetKernelArg(ks, 10, sizeof(cl_uchar), &mask_hi2));
            size_t lsk[3] = {64, (size_t)nsg, 1};
            size_t gsk[3] = {gx, (size_t)(nsg * ksplit), 1};
            backend_ctx->enqueue_ndrange_kernel(ks, 3, gsk, lsk, dst);

            cl_kernel kr = backend_ctx->mul_mat.kernel_gemv_splitk_reduce_f32;
            CL_CHECK(clSetKernelArg(kr, 0, sizeof(cl_mem),   &partial));
            CL_CHECK(clSetKernelArg(kr, 1, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kr, 2, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kr, 3, sizeof(cl_int),   &ne01));
            CL_CHECK(clSetKernelArg(kr, 4, sizeof(cl_int),   &ksplit));
            size_t lr[3] = {64, 1, 1};
            size_t gr[3] = {(size_t)CEIL_DIV(ne01, 64) * 64, 1, 1};
            backend_ctx->enqueue_ndrange_kernel(kr, 3, gr, lr, dst);

            if (q_img) CL_CHECK(clReleaseMemObject(q_img));
            CL_CHECK(clReleaseMemObject(b_sub_buf));
            CL_CHECK(clReleaseMemObject(b_img));
            return;
        }

        kernel = use_mc3    ? backend_ctx->mul_mat.kernel_gemv_noshuffle_q4_k_f32_mc3
               : use_tiled  ? backend_ctx->mul_mat.kernel_gemv_noshuffle_q4_k_f32_tiled
               : use_q4k_o4 ? backend_ctx->mul_mat.kernel_gemv_noshuffle_q4_k_f32_o4
                            : backend_ctx->mul_mat.kernel_gemv_noshuffle_q4_k_f32;

        if (use_tiled) {
            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0_q4_k->q));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),   &extra0_q4_k->d));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extra0_q4_k->dm));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem),   &extra0_q4_k->s));
            CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem),   &b_img));
            CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_int),   &ne00));
            CL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_int),   &ne01));
        } else {
            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &q_img));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),   &extra0_q4_k->d));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extra0_q4_k->dm));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem),   &extra0_q4_k->s));
            CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem),   &b_img));
            CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_int),   &ne00));
            CL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_int),   &ne01));
            CL_CHECK(clSetKernelArg(kernel, 9, sizeof(cl_uchar), &mask_d6));
            CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_uchar), &mask_d4));
            CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_uchar), &mask_hi2));
        }

        // Wide K-split for the decode GEMV: the default 4-subgroup K-split leaves
        // each Adreno SP with only ~4 waves, too few to hide LPDDR weight-load
        // latency, so even the large FFN matmuls run well below the achievable
        // bandwidth. Widen to 16 subgroups/WG (= the 1024-lane Adreno WG max) so
        // each SP holds enough in-flight memory requests. Prefill is unaffected (the
        // GEMM path is separate) and coherence-identical (greedy output unchanged).
        // Applies to the plain base
        // GEMV only; tiled/o4/mc3 keep 4 (their reductions are hard-coded to 4).
        // Layout-safe: the base kernel derives its K-split from get_local_size(1)
        // and the packed block stride is a physical constant (independent of it).
        // Opt-out: GGML_OPENCL_Q4K_GEMV_WIDE=0.
        static const bool splitk_wide_env = []{
            const char * e = std::getenv("GGML_OPENCL_Q4K_GEMV_WIDE");
            return !e || e[0] == '\0' || e[0] != '0';
        }();
        const bool   splitk_wide = splitk_wide_env && !use_tiled && !use_q4k_o4 && !use_mc3;
        size_t       nsg_y       = splitk_wide ? 16 : 4;
        // Cap the wide K-split by the kernel's real max WG. X1-class drivers cap
        // this GEMV at 768 (< 64*16 = 1024), so an uncapped lws aborts the
        // dispatch with CL_INVALID_WORK_GROUP_SIZE (-54) and breaks ALL q4_K
        // decode for M>2560. nsg_y is a pure K-split (the base kernel reads it
        // from get_local_size(1); the packed block stride is a physical constant),
        // so halving it stays coherent - just a narrower split. X2 keeps 16
        // (maxwg 1024); X1 falls to 8.
        if (splitk_wide) {
            const size_t maxwg = backend_ctx->get_kernel_workgroup_size(kernel);
            while (nsg_y > 4 && 64 * nsg_y > maxwg) { nsg_y >>= 1; }
        }
        size_t local_work_size[3] = {64, nsg_y, 1};
        size_t global_work_size[3] = {(size_t)CEIL_DIV(use_tiled ? ne01 : (use_q4k_o4 ? ne01/4 : ne01/2), 64)*64, nsg_y, 1};

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);

        if (q_img) CL_CHECK(clReleaseMemObject(q_img));
        CL_CHECK(clReleaseMemObject(b_sub_buf));
        CL_CHECK(clReleaseMemObject(b_img));
    } else {

        cl_mem b_sub_buf = nullptr;
        cl_mem b_sub_buf_trans = nullptr;
        cl_mem b_img = nullptr;
        cl_mem b_img_trans = nullptr;

        // subbuffer for activations
        region.origin = offset1;
        region.size = K * N * sizeof(float);
        CL_CHECK((b_sub_buf = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

        // image for activations
        img_fmt = {CL_RGBA, CL_FLOAT};
        memset(&img_desc, 0, sizeof(img_desc));
        img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc.image_width = K * N / 4;
        img_desc.buffer = b_sub_buf;
        CL_CHECK((b_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));

        // pad N to multiple of 8
        int extra_elements = N % 8;
        int padding = 0;
        if (extra_elements > 0){
            padding = 8 - extra_elements;
        }

        // subbuffer for transposed activations
        region.origin = 0;
        region.size = K * (N + padding) * sizeof(float)/2;
        backend_ctx->prealloc_act_trans.allocate(context, region.size);
        CL_CHECK((b_sub_buf_trans = clCreateSubBuffer(backend_ctx->prealloc_act_trans.buffer, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

        // image for transposed activations
        img_fmt = {CL_RGBA, CL_HALF_FLOAT};
        memset(&img_desc, 0, sizeof(img_desc));
        img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc.image_width = K * (N + padding) / 4;
        img_desc.buffer = b_sub_buf_trans;
        CL_CHECK((b_img_trans = clCreateImage(context, 0, &img_fmt, &img_desc, NULL, &err), err));

        // transpose activations
        int height_B = N/4;
        if (height_B == 0) {
            height_B = 1;
        }
        int width_B = K/4;
        int padded_height_B = (N + padding)/4;

        kernel = backend_ctx->mul_mat.kernel_transpose_32_16;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &b_img));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_img_trans));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int),    &height_B));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int),    &width_B));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int),    &padded_height_B));

        size_t local_work_size_t[2] = { 1, 16 };
        size_t global_work_size_t[2] = { (size_t)width_B, (size_t)padded_height_B };
        backend_ctx->enqueue_ndrange_kernel(kernel, 2, global_work_size_t, local_work_size_t, dst);

        // dp4a (int8) dense prefill GEMM and weight via texture
        static const char * q4k_dense_dp4a_env = getenv("GGML_OPENCL_Q4K_DENSE_DP4A");
        static const char * q4k_dense_wimg_env = getenv("GGML_OPENCL_Q4K_DENSE_DP4A_WIMG");

        const bool q4k_dense_wimg_on = q4k_dense_wimg_env && (atoi(q4k_dense_wimg_env) != 0);
              bool q4k_dense_dp4a_on = q4k_dense_wimg_on
            ? true
            : q4k_dense_dp4a_env
            ? (atoi(q4k_dense_dp4a_env) != 0)
            : (backend_ctx->adreno_gen == ADRENO_GPU_GEN::X2E);

        // dp4 has to be available
        q4k_dense_dp4a_on = backend_ctx->has_integer_dot && q4k_dense_dp4a_on;

        // Min N for the dp4a prefill GEMM, default 9, i.e., ne1 > 8
        static const char * q4k_dp4a_minn_env = getenv("GGML_OPENCL_Q4K_DP4A_MINN");
        const int           q4k_dp4a_minn     = q4k_dp4a_minn_env ? atoi(q4k_dp4a_minn_env) : 9;

        if (q4k_dense_dp4a_on && N >= q4k_dp4a_minn && (K % 32 == 0) && (M % 64 == 0)) {
            const size_t n_blocks = (size_t)N * (K / 32);
            backend_ctx->prealloc_moe_qa.allocate(context, (size_t)N * K * sizeof(cl_char));
            backend_ctx->prealloc_moe_da.allocate(context, n_blocks * sizeof(cl_half));
            backend_ctx->prealloc_moe_sa.allocate(context, n_blocks * sizeof(cl_half));

            cl_int tb = (cl_int)n_blocks;
            cl_kernel qk = backend_ctx->mul_mat.kernel_quant_a_q8_1;
            CL_CHECK(clSetKernelArg(qk, 0, sizeof(cl_mem), &b_sub_buf));
            CL_CHECK(clSetKernelArg(qk, 1, sizeof(cl_mem), &backend_ctx->prealloc_moe_qa.buffer));
            CL_CHECK(clSetKernelArg(qk, 2, sizeof(cl_mem), &backend_ctx->prealloc_moe_da.buffer));
            CL_CHECK(clSetKernelArg(qk, 3, sizeof(cl_mem), &backend_ctx->prealloc_moe_sa.buffer));
            CL_CHECK(clSetKernelArg(qk, 4, sizeof(cl_int), &tb));
            size_t q_local[1]  = { 64 };
            size_t q_global[1] = { (size_t)(((n_blocks + 63) / 64) * 64) };
            backend_ctx->enqueue_ndrange_kernel(qk, 1, q_global, q_local, dst);

            // check if weights go through texture
            cl_mem q4k_q_img = nullptr;
            bool use_wimg = q4k_dense_wimg_on;
            if (use_wimg) {
                const size_t tex = (size_t)M * (size_t)K / 8;  // uint32 texels = bytes/4
                if (tex == 0 || tex > backend_ctx->image_max_buffer_size) {
                    use_wimg = false;
                } else {
                    img_fmt = { CL_R, CL_UNSIGNED_INT32 };
                    memset(&img_desc, 0, sizeof(img_desc));
                    img_desc.image_type  = CL_MEM_OBJECT_IMAGE1D_BUFFER;
                    img_desc.image_width = tex;
                    img_desc.buffer      = extra0_q4_k->q;
                    q4k_q_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err);
                    if (err != CL_SUCCESS || q4k_q_img == nullptr) {
                        use_wimg  = false;
                        q4k_q_img = nullptr;
                    }
                }
            }

            cl_kernel dk = use_wimg ? backend_ctx->mul_mat.kernel_gemm_noshuffle_q4_k_q8_1_dp4a_wimg
                                    : backend_ctx->mul_mat.kernel_gemm_noshuffle_q4_k_q8_1_dp4a;
            int ai = 0;
            if (use_wimg) {
                CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem), &q4k_q_img));
            } else {
                CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem), &extra0_q4_k->q));
            }
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem),   &extra0_q4_k->s));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem),   &extra0_q4_k->d));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem),   &extra0_q4_k->dm));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem),   &backend_ctx->prealloc_moe_qa.buffer));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem),   &backend_ctx->prealloc_moe_da.buffer));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem),   &backend_ctx->prealloc_moe_sa.buffer));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_int),   &M));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_int),   &N));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_int),   &K));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_uchar), &mask_d6));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_uchar), &mask_d4));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_uchar), &mask_hi2));
            // Must match the compile-time TILESIZE_N chosen at program build (per-device,
            // X1E=8 else 32; env override). Same inputs -> same value.
            int q4k_dp4a_ts = (backend_ctx->adreno_gen == ADRENO_GPU_GEN::X1E) ? 8 : 32;
            if (const char * e = getenv("GGML_OPENCL_Q4K_DP4A_TS")) q4k_dp4a_ts = atoi(e);
            size_t d_local[3]  = { 64, 1, 1 };
            size_t d_global[3] = { 64, (size_t)(M / 64), (size_t)CEIL_DIV(N, q4k_dp4a_ts) };
            backend_ctx->enqueue_ndrange_kernel(dk, 3, d_global, d_local, dst);

            if (q4k_q_img != nullptr) {
                CL_CHECK(clReleaseMemObject(q4k_q_img));
            }
            CL_CHECK(clReleaseMemObject(b_sub_buf));
            CL_CHECK(clReleaseMemObject(b_sub_buf_trans));
            CL_CHECK(clReleaseMemObject(b_img));
            CL_CHECK(clReleaseMemObject(b_img_trans));
            return;
        }

        // gemm
        // Small-batch (medium n_q) occupancy fix: at ne1<=8 the 2x8 grid is
        // (1, ceil(M/2)) -> ~M/256 workgroups, which under-occupies the SP and
        // makes the GEMM much slower than the ne1==1 GEMV at the same weight
        // traffic. The _r1 (1-row) kernel doubles the M-axis workgroup count
        // and removes the accumulator spill. Opt-in via env while validating.
        static const bool q4k_gemm_r1   = (getenv("GGML_OPENCL_Q4K_GEMM_R1")   != nullptr);
        static const bool q4k_gemm_kimg = (getenv("GGML_OPENCL_Q4K_GEMM_KIMG") != nullptr);
        // Cooperative-K (intra-WG K-split + reduction) for the small-batch
        // (n_q in [2..8]) path: DEFAULT ON, opt out with GGML_OPENCL_Q4K_GEMM_COK=0.
        // Byte-identical greedy output; large-batch (ne1>8) untouched.
        static const char * q4k_cok_env = getenv("GGML_OPENCL_Q4K_GEMM_COK");
        static const bool q4k_gemm_cok  = (q4k_cok_env == nullptr) || (atoi(q4k_cok_env) != 0);
        const bool use_cok  = q4k_gemm_cok && (ne1 <= 8);
        const bool use_r1   = !use_cok && q4k_gemm_r1 && (ne1 <= 8);
        // Weights-as-image (L1/TPL1) for the small-batch weight-read-bound path.
        const bool use_kimg = !use_cok && !use_r1 && q4k_gemm_kimg && (ne1 <= 8);

        cl_mem q_img = nullptr;
        if (use_kimg) {
            img_fmt = { CL_R, CL_UNSIGNED_INT32 };
            memset(&img_desc, 0, sizeof(img_desc));
            img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
            img_desc.image_width = M * K / 2 / 4;
            img_desc.buffer = extra0_q4_k->q;
            CL_CHECK((q_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));
        }

        kernel = use_cok  ? backend_ctx->mul_mat.kernel_gemm_noshuffle_q4_k_f32_cok
               : use_r1   ? backend_ctx->mul_mat.kernel_gemm_noshuffle_q4_k_f32_r1
               : use_kimg ? backend_ctx->mul_mat.kernel_gemm_noshuffle_q4_k_f32_kimg
                          : backend_ctx->mul_mat.kernel_gemm_noshuffle_q4_k_f32;
        int padded_N = N + padding;

        if (use_kimg) {
            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &q_img));
        } else {
            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &extra0_q4_k->q));
        }
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),   &extra0_q4_k->s));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extra0_q4_k->d));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem),   &extra0_q4_k->dm));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem),   &b_img_trans));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_mem),   &extrad->data_device));
        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_ulong), &offsetd));
        CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_int),   &ne01));
        CL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_int),   &padded_N));
        CL_CHECK(clSetKernelArg(kernel, 9, sizeof(cl_int),   &ne00));
        CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_int),   &ne1));
        CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_uchar), &mask_d6));
        CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_uchar), &mask_d4));
        CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_uchar), &mask_hi2));

        size_t global_work_size[3];
        size_t local_work_size[3];
        if (use_cok) {
            // (COK_SG lanes x COK_NSG subgroups): one row per lane, K split
            // across the COK_NSG subgroups. ne01 is a multiple of 64.
            global_work_size[0] = (size_t)ne01;   // rows
            global_work_size[1] = 8;              // COK_NSG
            global_work_size[2] = 1;
            local_work_size[0] = 64;              // COK_SG
            local_work_size[1] = 8;               // COK_NSG
            local_work_size[2] = 1;
        } else if (use_r1) {
            // 1 row per WI (opt-in occupancy experiment).
            global_work_size[0] = (size_t)CEIL_DIV(ne1, 8);
            global_work_size[1] = (size_t)ne01;
            global_work_size[2] = 1;
            local_work_size[0] = 1;
            local_work_size[1] = 128;
            local_work_size[2] = 1;
        } else if (use_kimg) {
            // kimg is a 2-row tile (opt-in weights-as-image experiment).
            global_work_size[0] = (size_t)CEIL_DIV(ne1, 8);
            global_work_size[1] = (size_t)CEIL_DIV(ne01, 2);
            global_work_size[2] = 1;
            local_work_size[0] = 1;
            local_work_size[1] = 128;
            local_work_size[2] = 1;
        } else {
            // Default: x2-unified base kernel is the 4-row (gx<<2) tile.
            global_work_size[0] = (size_t)CEIL_DIV(ne1, 8);
            global_work_size[1] = (size_t)CEIL_DIV(ne01, 4);
            global_work_size[2] = 1;
            local_work_size[0] = 1;
            local_work_size[1] = 128;
            local_work_size[2] = 1;
        }

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
        if (q_img) CL_CHECK(clReleaseMemObject(q_img));
        CL_CHECK(clReleaseMemObject(b_sub_buf));
        CL_CHECK(clReleaseMemObject(b_sub_buf_trans));
        CL_CHECK(clReleaseMemObject(b_img));
        CL_CHECK(clReleaseMemObject(b_img_trans));
    }
}
#endif // GGML_OPENCL_USE_ADRENO_KERNELS

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
void ggml_cl_mul_mat_q5_K_f32_adreno(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;
    ggml_tensor_extra_cl_q5_K * extra0_q5_k = (ggml_tensor_extra_cl_q5_K *)src0->extra;

    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne1  = dst->ne[1];

    GGML_ASSERT(ne00 % ggml_blck_size(src0->type) == 0);

    cl_context context = backend_ctx->context;
    cl_kernel kernel;

    cl_int           err;
    cl_image_format  img_fmt;
    cl_image_desc    img_desc;
    cl_buffer_region region;

    int M = ne01;
    int N = ne1;
    int K = ne00;

    cl_uchar mask_d6  = 0x3F;
    cl_uchar mask_d4  = 0x0F;
    cl_uchar mask_hi2 = 0xC0;

    // Multi-column (N=3) verify GEMV for q5_K: route the spec/MTP verify batch
    // (ne1==3) onto the efficient GEMV path instead of the transposed-GEMM dead-
    // zone (gemm_noshuffle_q5_k, the #2 chunk of MTP decode on a Q4_0-mix model
    // after q4_0 mc3). Reuses the ne1==1 GEMV image setup (q + qh + activations).
    // Opt-in via GGML_OPENCL_Q5K_MC3=1. Per-layer only (ne01 < 32768).
    static const bool q5k_mc3 = (getenv("GGML_OPENCL_Q5K_MC3") != nullptr);
    const bool use_q5k_mc3 = q5k_mc3 && (ne1 >= 2 && ne1 <= 4) && (ne01 < 32768);

    if (ne1 == 1 || use_q5k_mc3) {
        cl_mem q_img  = nullptr;
        cl_mem qh_img = nullptr;
        cl_mem b_sub_buf = nullptr;
        cl_mem b_img = nullptr;

        // image for q (CL_R, CL_UNSIGNED_INT32): width = M*K/2/4
        img_fmt = {CL_R, CL_UNSIGNED_INT32};
        memset(&img_desc, 0, sizeof(img_desc));
        img_desc.image_type  = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc.image_width = M * K / 2 / 4;
        img_desc.buffer      = extra0_q5_k->q;
        CL_CHECK((q_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));

        // image for qh (CL_R, CL_HALF_FLOAT): width = M*K/16
        img_fmt = {CL_R, CL_HALF_FLOAT};
        memset(&img_desc, 0, sizeof(img_desc));
        img_desc.image_type  = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc.image_width = M * K / 16;
        img_desc.buffer      = extra0_q5_k->qh;
        CL_CHECK((qh_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));

        // subbuffer for activations
        region.origin = offset1;
        region.size   = K * N * sizeof(float);
        CL_CHECK((b_sub_buf = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

        // image for activations (CL_RGBA, CL_FLOAT): width = K*N/4
        img_fmt = {CL_RGBA, CL_FLOAT};
        memset(&img_desc, 0, sizeof(img_desc));
        img_desc.image_type  = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc.image_width = K * N / 4;
        img_desc.buffer      = b_sub_buf;
        CL_CHECK((b_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));

        kernel = use_q5k_mc3 ? backend_ctx->mul_mat.kernel_gemv_noshuffle_q5_k_f32_mc3
                             : backend_ctx->mul_mat.kernel_gemv_noshuffle_q5_k_f32;

        CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &q_img));
        CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &qh_img));
        CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra0_q5_k->d));
        CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_mem),   &extra0_q5_k->dm));
        CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extra0_q5_k->s));
        CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_mem),   &b_img));
        CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_mem),   &extrad->data_device));
        CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &offsetd));
        CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_int),   &ne00));
        CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_int),   &ne01));
        CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_uchar), &mask_d6));
        CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_uchar), &mask_d4));
        CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_uchar), &mask_hi2));
        if (use_q5k_mc3) {
            CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_int), &ne1));  // n_cols
        }

        size_t local_work_size[3]  = {64, 4, 1};
        size_t global_work_size[3] = {(size_t)CEIL_DIV(ne01/2, 64)*64, 4, 1};

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);

        CL_CHECK(clReleaseMemObject(q_img));
        CL_CHECK(clReleaseMemObject(qh_img));
        CL_CHECK(clReleaseMemObject(b_sub_buf));
        CL_CHECK(clReleaseMemObject(b_img));
    } else {
        cl_mem b_sub_buf      = nullptr;
        cl_mem b_sub_buf_trans = nullptr;
        cl_mem b_img          = nullptr;
        cl_mem b_img_trans    = nullptr;

        // subbuffer for activations
        region.origin = offset1;
        region.size   = K * N * sizeof(float);
        CL_CHECK((b_sub_buf = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

        // image for activations
        img_fmt = {CL_RGBA, CL_FLOAT};
        memset(&img_desc, 0, sizeof(img_desc));
        img_desc.image_type  = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc.image_width = K * N / 4;
        img_desc.buffer      = b_sub_buf;
        CL_CHECK((b_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));

        // pad N to multiple of 8
        int extra_elements = N % 8;
        int padding = 0;
        if (extra_elements > 0) {
            padding = 8 - extra_elements;
        }

        // subbuffer for transposed activations
        region.origin = 0;
        region.size   = K * (N + padding) * sizeof(float) / 2;
        backend_ctx->prealloc_act_trans.allocate(context, region.size);
        CL_CHECK((b_sub_buf_trans = clCreateSubBuffer(backend_ctx->prealloc_act_trans.buffer, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

        // image for transposed activations
        img_fmt = {CL_RGBA, CL_HALF_FLOAT};
        memset(&img_desc, 0, sizeof(img_desc));
        img_desc.image_type  = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc.image_width = K * (N + padding) / 4;
        img_desc.buffer      = b_sub_buf_trans;
        CL_CHECK((b_img_trans = clCreateImage(context, 0, &img_fmt, &img_desc, NULL, &err), err));

        // transpose activations
        int height_B       = N / 4;
        if (height_B == 0) height_B = 1;
        int width_B        = K / 4;
        int padded_height_B = (N + padding) / 4;

        kernel = backend_ctx->mul_mat.kernel_transpose_32_16;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &b_img));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_img_trans));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int),    &height_B));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int),    &width_B));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int),    &padded_height_B));

        size_t local_work_size_t[2]  = {1, 16};
        size_t global_work_size_t[2] = {(size_t)width_B, (size_t)padded_height_B};
        backend_ctx->enqueue_ndrange_kernel(kernel, 2, global_work_size_t, local_work_size_t, dst);

        // dp4a (int8) dense q5_K prefill GEMM
        static const char * q5k_dense_dp4a_env = getenv("GGML_OPENCL_Q5K_DENSE_DP4A");
                     bool   q5k_dense_dp4a_on  = q5k_dense_dp4a_env
            ? (atoi(q5k_dense_dp4a_env) != 0)
            : (backend_ctx->adreno_gen == ADRENO_GPU_GEN::X2E);
        // dot prod has to be available
        q5k_dense_dp4a_on = backend_ctx->has_integer_dot && q5k_dense_dp4a_on;

        if (q5k_dense_dp4a_on && ne1 > 8 && (ne00 % 32 == 0) && (ne01 % 64 == 0)) {
            const int Mm = ne01, Nn = ne1, Kk = ne00;
            const size_t n_blocks = (size_t)Nn * (Kk / 32);
            backend_ctx->prealloc_moe_qa.allocate(context, (size_t)Nn * Kk * sizeof(cl_char));
            backend_ctx->prealloc_moe_da.allocate(context, n_blocks * sizeof(cl_half));
            backend_ctx->prealloc_moe_sa.allocate(context, n_blocks * sizeof(cl_half));

            cl_int tb = (cl_int)n_blocks;
            cl_kernel qk = backend_ctx->mul_mat.kernel_quant_a_q8_1;
            CL_CHECK(clSetKernelArg(qk, 0, sizeof(cl_mem), &b_sub_buf));
            CL_CHECK(clSetKernelArg(qk, 1, sizeof(cl_mem), &backend_ctx->prealloc_moe_qa.buffer));
            CL_CHECK(clSetKernelArg(qk, 2, sizeof(cl_mem), &backend_ctx->prealloc_moe_da.buffer));
            CL_CHECK(clSetKernelArg(qk, 3, sizeof(cl_mem), &backend_ctx->prealloc_moe_sa.buffer));
            CL_CHECK(clSetKernelArg(qk, 4, sizeof(cl_int), &tb));
            size_t q_local[1]  = { 64 };
            size_t q_global[1] = { (size_t)(((n_blocks + 63) / 64) * 64) };
            backend_ctx->enqueue_ndrange_kernel(qk, 1, q_global, q_local, dst);

            cl_kernel dk = backend_ctx->mul_mat.kernel_gemm_noshuffle_q5_k_q8_1_dp4a;
            int ai = 0;
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem),   &extra0_q5_k->q));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem),   &extra0_q5_k->qh));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem),   &extra0_q5_k->s));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem),   &extra0_q5_k->d));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem),   &extra0_q5_k->dm));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem),   &backend_ctx->prealloc_moe_qa.buffer));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem),   &backend_ctx->prealloc_moe_da.buffer));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem),   &backend_ctx->prealloc_moe_sa.buffer));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_int),   &Mm));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_int),   &Nn));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_int),   &Kk));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_uchar), &mask_d6));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_uchar), &mask_d4));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_uchar), &mask_hi2));
            size_t d_local[3]  = { 64, 1, 1 };
            size_t d_global[3] = { 64, (size_t)(Mm / 64), (size_t)CEIL_DIV(Nn, 32) };
            backend_ctx->enqueue_ndrange_kernel(dk, 3, d_global, d_local, dst);

            CL_CHECK(clReleaseMemObject(b_sub_buf));
            CL_CHECK(clReleaseMemObject(b_sub_buf_trans));
            CL_CHECK(clReleaseMemObject(b_img));
            CL_CHECK(clReleaseMemObject(b_img_trans));
            return;
        }

        // gemm
        kernel = backend_ctx->mul_mat.kernel_gemm_noshuffle_q5_k_f32;
        int padded_N = N + padding;

        CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0_q5_k->q));
        CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0_q5_k->qh));
        CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra0_q5_k->s));
        CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_mem),   &extra0_q5_k->d));
        CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extra0_q5_k->dm));
        CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_mem),   &b_img_trans));
        CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_mem),   &extrad->data_device));
        CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &offsetd));
        CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_int),   &ne01));
        CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_int),   &padded_N));
        CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_int),   &ne00));
        CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_int),   &ne1));
        CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_uchar), &mask_d6));
        CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_uchar), &mask_d4));
        CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_uchar), &mask_hi2));

        size_t global_work_size[3] = {(size_t)CEIL_DIV(ne1, 8), (size_t)CEIL_DIV(ne01, 4), 1};
        size_t local_work_size[3]  = {1, 128, 1};

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);

        CL_CHECK(clReleaseMemObject(b_sub_buf));
        CL_CHECK(clReleaseMemObject(b_sub_buf_trans));
        CL_CHECK(clReleaseMemObject(b_img));
        CL_CHECK(clReleaseMemObject(b_img_trans));
    }
}
#endif // GGML_OPENCL_USE_ADRENO_KERNELS

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
static void ggml_cl_mul_mat_q6_K_f32_adreno_ila(ggml_backend_t backend, const ggml_tensor * src0,
                                                const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl_q6_K * extra0_q6_K = (ggml_tensor_extra_cl_q6_K *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];

    const int ne1 = dst->ne[1];

    GGML_ASSERT(ne00 % ggml_blck_size(src0->type) == 0);

    cl_context context = backend_ctx->context;
    cl_kernel kernel;

    cl_int           err;
    cl_buffer_region region;
    cl_image_format  img_fmt;
    cl_image_desc    img_desc;

    const int M = ne01;
    const int N = ne1;
    const int K = ne00;

    if (ne1 == 1) {
        cl_mem b_sub_buf  = nullptr;
        cl_mem b_img      = nullptr;

        region.origin = offset1;
        region.size   = (size_t)K * N * sizeof(float);
        CL_CHECK((b_sub_buf = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

        img_fmt = { CL_RGBA, CL_FLOAT };
        memset(&img_desc, 0, sizeof(img_desc));
        img_desc.image_type  = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc.image_width = (size_t)K * N / 4;
        img_desc.buffer      = b_sub_buf;
        CL_CHECK((b_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));

        kernel = backend_ctx->mul_mat.kernel_gemv_noshuffle_q6_k_f32_32b_trans;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &extra0_q6_K->ql_img));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),   &extra0_q6_K->qh_img));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extra0_q6_K->s));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem),   &extra0_q6_K->d));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem),   &b_img));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_mem),   &extrad->data_device));
        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_ulong), &offsetd));
        CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_int),   &ne00));
        CL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_int),   &ne01));

        size_t local_work_size[3]  = { 64, 8, 1 };
        size_t global_work_size[3] = { (size_t)ne01, 8, 1 };
        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);

        CL_CHECK(clReleaseMemObject(b_img));
        CL_CHECK(clReleaseMemObject(b_sub_buf));
    } else {
        const int gemm_tile_n = 64;
        int N_pad = CEIL_DIV(N, gemm_tile_n) * gemm_tile_n;

        static const char * q6_k_bin_dp4a_env = getenv("GGML_OPENCL_Q6_K_BIN_DP4A");
                     bool   q6_k_bin_dp4a_on  = q6_k_bin_dp4a_env
                                                  ? (atoi(q6_k_bin_dp4a_env) != 0)
                                                  : true;
        // dot prod has to be available
        q6_k_bin_dp4a_on = backend_ctx->has_integer_dot && q6_k_bin_dp4a_on;

        if (q6_k_bin_dp4a_on && backend_ctx->mul_mat.kernel_gemm_noshuffle_q6_k_q8_1_dp4a_ila_a8_bin) {
            const int    dp4a_N_pad = CEIL_DIV(N, 32) * 32;
            const size_t n_blocks   = (size_t)dp4a_N_pad * (K / 32);

            backend_ctx->prealloc_moe_qa.allocate(context, (size_t)dp4a_N_pad * K * sizeof(cl_char));
            backend_ctx->prealloc_moe_da.allocate(context, n_blocks * sizeof(cl_half));
            backend_ctx->prealloc_moe_sa.allocate(context, n_blocks * sizeof(cl_half));

            cl_mem b_sub = nullptr;
            region.origin = offset1;
            region.size   = (size_t)K * N * sizeof(float);
            CL_CHECK((b_sub = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

            cl_int    tb = (cl_int)((size_t)N * (K / 32));
            cl_kernel qk = backend_ctx->mul_mat.kernel_quant_a_q8_1;
            CL_CHECK(clSetKernelArg(qk, 0, sizeof(cl_mem), &b_sub));
            CL_CHECK(clSetKernelArg(qk, 1, sizeof(cl_mem), &backend_ctx->prealloc_moe_qa.buffer));
            CL_CHECK(clSetKernelArg(qk, 2, sizeof(cl_mem), &backend_ctx->prealloc_moe_da.buffer));
            CL_CHECK(clSetKernelArg(qk, 3, sizeof(cl_mem), &backend_ctx->prealloc_moe_sa.buffer));
            CL_CHECK(clSetKernelArg(qk, 4, sizeof(cl_int), &tb));
            size_t q_local[1]  = { 64 };
            size_t q_global[1] = { (size_t)CEIL_DIV(tb, 64) * 64 };
            backend_ctx->enqueue_ndrange_kernel(qk, 1, q_global, q_local, dst);

            cl_mem d_sub = nullptr;
            cl_mem d_img = nullptr;
            region.origin = offsetd;
            region.size   = (size_t)M * N * sizeof(float);
            CL_CHECK((d_sub = clCreateSubBuffer(extrad->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

            img_fmt = { CL_R, CL_FLOAT };
            memset(&img_desc, 0, sizeof(img_desc));
            img_desc.image_type  = CL_MEM_OBJECT_IMAGE1D_BUFFER;
            img_desc.image_width = (size_t)M * N;
            img_desc.buffer      = d_sub;
            CL_CHECK((d_img = clCreateImage(context, CL_MEM_WRITE_ONLY, &img_fmt, &img_desc, NULL, &err), err));

            kernel = backend_ctx->mul_mat.kernel_gemm_noshuffle_q6_k_q8_1_dp4a_ila_a8_bin;

            cl_uint k_arg = 0;
            CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_mem), &extra0_q6_K->ql_img));
            CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_mem), &extra0_q6_K->qh));
            CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_mem), &extra0_q6_K->s));
            CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_mem), &extra0_q6_K->d));
            CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_mem), &backend_ctx->prealloc_moe_qa.buffer));
            CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_mem), &backend_ctx->prealloc_moe_da.buffer));
            CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(cl_mem), &d_img));
            CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int),    &K));
            CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int),    &M));
            CL_CHECK(clSetKernelArg(kernel, k_arg++, sizeof(int),    &N));

            size_t local_work_size[3]  = { 64, 1, 1 };
            size_t global_work_size[3] = { 64, (size_t)(M / 64), (size_t)(dp4a_N_pad / 32) };
            backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);

            CL_CHECK(clReleaseMemObject(b_sub));
            CL_CHECK(clReleaseMemObject(d_img));
            CL_CHECK(clReleaseMemObject(d_sub));
            return;
        }

        cl_mem b_sub_buf = nullptr;
        cl_mem b_padded  = nullptr;
        cl_mem b_buf     = nullptr;
        if (N_pad == N) {
            region.origin = offset1;
            region.size   = (size_t)K * N * sizeof(float);
            CL_CHECK((b_sub_buf = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));
            b_buf = b_sub_buf;
        } else {
            CL_CHECK((b_padded = clCreateBuffer(context, CL_MEM_READ_WRITE, (size_t)K * N_pad * sizeof(float), NULL, &err), err));
            const float zero = 0.0f;
            CL_CHECK(clEnqueueFillBuffer(backend_ctx->queue, b_padded, &zero, sizeof(zero), 0, (size_t)K * N_pad * sizeof(float), 0, NULL, NULL));
            CL_CHECK(clEnqueueCopyBuffer(backend_ctx->queue, extra1->data_device, b_padded, offset1, 0, (size_t)K * N * sizeof(float), 0, NULL, NULL));
            b_buf = b_padded;
        }

        img_fmt = { CL_R, CL_FLOAT };
        memset(&img_desc, 0, sizeof(img_desc));
        img_desc.image_type  = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc.image_width = (size_t)K * N_pad;
        img_desc.buffer      = b_buf;
        cl_mem b_img;
        CL_CHECK((b_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));

        region.origin = offsetd;
        region.size   = (size_t)M * N * sizeof(float);
        cl_mem d_sub_buf;
        CL_CHECK((d_sub_buf = clCreateSubBuffer(extrad->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));
        img_fmt = { CL_R, CL_FLOAT };
        memset(&img_desc, 0, sizeof(img_desc));
        img_desc.image_type  = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc.image_width = (size_t)M * N;
        img_desc.buffer      = d_sub_buf;
        cl_mem d_img;
        CL_CHECK((d_img = clCreateImage(context, CL_MEM_WRITE_ONLY, &img_fmt, &img_desc, NULL, &err), err));

        kernel = backend_ctx->mul_mat.kernel_gemm_noshuffle_q6_k_f32_32b_trans_ila_a8_bin;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),  &extra0_q6_K->ql_img));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),  &extra0_q6_K->qh));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),  &extra0_q6_K->s));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem),  &extra0_q6_K->d));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem),  &b_img));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_mem),  &d_img));
        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_uint), &ne00));
        CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_uint), &ne01));
        CL_CHECK(clSetKernelArg(kernel, 8, sizeof(int),     &N));

        size_t local_work_size[3]  = { 64, 2, 2 };
        size_t m_tiles = (size_t)CEIL_DIV(M, 64);
        size_t global_work_size[3] = { 64, m_tiles, (size_t)CEIL_DIV(N_pad, gemm_tile_n) };
        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);

        CL_CHECK(clReleaseMemObject(b_img));
        if (b_sub_buf) {
            CL_CHECK(clReleaseMemObject(b_sub_buf));
        }
        if (b_padded) {
            CL_CHECK(clReleaseMemObject(b_padded));
        }
        CL_CHECK(clReleaseMemObject(d_img));
        CL_CHECK(clReleaseMemObject(d_sub_buf));
    }
}
#endif // GGML_OPENCL_USE_ADRENO_KERNELS

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
void ggml_cl_mul_mat_q6_K_f32_adreno(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl_q6_K * extra0_q6_K = (ggml_tensor_extra_cl_q6_K *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];

    const int ne1 = dst->ne[1];

    GGML_ASSERT(ne00 % ggml_blck_size(src0->type) == 0);

    cl_context context = backend_ctx->context;
    cl_kernel kernel;

    cl_int           err;
    cl_buffer_region region;
    cl_image_format  img_fmt;
    cl_image_desc    img_desc;

    // subbuffer and image for activation
    // Multi-column verify GEMV: route the spec/MTP verify q6_K matmuls (ne1==3)
    // onto the efficient GEMV path instead of the transposed-GEMM dead-zone.
    // Reuses the ne1==1 image setup (activation image sized by N=ne1). Byte-
    // identical. Opt-in via GGML_OPENCL_Q6K_MC3=1 while validating.
    static const bool q6k_mc3 = (getenv("GGML_OPENCL_Q6K_MC3") != nullptr);
    // Per-layer only (ne01 < 32768): batched large-vocab lm_head stays on the
    // existing path (x2-unified routes batched Q6_K lm_head to CPU; the Adreno
    // GEMV corrupts it). Per-layer mc3 is byte-identical.
    const bool use_q6k_mc3 = q6k_mc3 && (ne1 == 3) && (ne01 < 32768);
    // Batched verify lm_head/embed (ne1==3, tiled layout): multi-column tiled
    // GEMV - streams the large lm_head weight once across the 3 verify columns
    // (the #1 MTP bottleneck; mc3 above can't, it reads the noshuffle layout).
    const bool use_q6k_tiled_mc = q6k_mc3 && (ne1 == 3) && (ne01 >= 32768) && use_q6k_tiled(backend_ctx, src0);

    const bool use_bin = use_q6_k_bin_kernels(backend_ctx, src0);

    if (use_bin) {
        if (use_q6k_mc3 || use_q6k_tiled_mc) {
            static bool warned = false;
            if (!warned) {
                GGML_LOG_WARN("ggml_opencl: GGML_OPENCL_Q6K_MC3 is bypassed by Q6_K binary kernels\n");
                warned = true;
            }
        }
        ggml_cl_mul_mat_q6_K_f32_adreno_ila(backend, src0, src1, dst);
        return;
    }

    if (ne1 == 1 || use_q6k_mc3 || use_q6k_tiled_mc) {
        cl_mem ql_img = nullptr;
        cl_mem qh_img = nullptr;
        cl_mem b_sub_buffer = nullptr;
        cl_mem b_img = nullptr;

        // o4 = 4-output-per-WI variant for long-vocab lm_head/embed; gated to
        // ne01 >= 32768 so per-layer q6_K (ne01=hidden 2-8K) keeps the 2-output
        // kernel (o4 regresses there). o4_global reads the weights from __global
        // coalesced instead of image1d_buffer -- the texture cache caps the
        // read-once-per-token lm_head bandwidth, while __global reaches the higher
        // rate the rest of the model gets. Both default ON; opt out via
        // GGML_OPENCL_Q6K_GEMV_O4 / GGML_OPENCL_Q6K_GEMV_O4_GLOBAL = 0.
        static const bool gemv_o4_env = []{
            const char * e = std::getenv("GGML_OPENCL_Q6K_GEMV_O4");
            return !e || e[0] == '\0' || e[0] != '0';
        }();
        static const bool o4_global_env = []{
            const char * e = std::getenv("GGML_OPENCL_Q6K_GEMV_O4_GLOBAL");
            return !e || e[0] == '\0' || e[0] != '0';
        }();
        const bool use_tiled     = !use_q6k_mc3 && use_q6k_tiled(backend_ctx, src0);
        const bool use_o4        = !use_tiled && !use_q6k_mc3 && gemv_o4_env && (ne01 % 4 == 0) && (ne01 >= 32768);
        const bool use_o4_global = use_o4 && o4_global_env;

        // ql/qh image views are only needed when NOT reading weights from global.
        if (!use_o4_global && !use_tiled) {
            // image for ql
            img_fmt.image_channel_order = CL_R;
            img_fmt.image_channel_data_type = CL_FLOAT;
            memset(&img_desc, 0, sizeof(img_desc));
            img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
            img_desc.image_width = ne01 * ne00 / 8;
            img_desc.buffer = extra0_q6_K->ql;
            CL_CHECK((ql_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));

            // image for qh
            img_fmt.image_channel_order = CL_R;
            img_fmt.image_channel_data_type = CL_HALF_FLOAT;
            memset(&img_desc, 0, sizeof(img_desc));
            img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
            img_desc.image_width = ne01 * ne00 / 8;
            img_desc.buffer = extra0_q6_K->qh;
            CL_CHECK((qh_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));
        }

        region.origin = offset1;
        region.size = ne00 * ne1 * sizeof(float);
        CL_CHECK((b_sub_buffer = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

        img_fmt.image_channel_order = CL_RGBA;
        img_fmt.image_channel_data_type = CL_FLOAT;
        memset(&img_desc, 0, sizeof(img_desc));
        img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc.image_width = ne00 * ne1 / 4;
        img_desc.buffer = b_sub_buffer;
        CL_CHECK((b_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));

        kernel = use_q6k_mc3      ? backend_ctx->mul_mat.kernel_gemv_noshuffle_q6_K_f32_mc3
               : use_q6k_tiled_mc ? backend_ctx->mul_mat.kernel_gemv_noshuffle_q6_K_f32_tiled_mc3
               : use_tiled        ? backend_ctx->mul_mat.kernel_gemv_noshuffle_q6_K_f32_tiled
               : use_o4_global ? backend_ctx->mul_mat.kernel_gemv_noshuffle_q6_K_f32_o4_global
               : use_o4        ? backend_ctx->mul_mat.kernel_gemv_noshuffle_q6_K_f32_o4
                               : backend_ctx->mul_mat.kernel_gemv_noshuffle_q6_K_f32;

        if (use_o4_global || use_tiled) {
            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &extra0_q6_K->ql));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra0_q6_K->qh));
        } else {
            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &ql_img));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &qh_img));
        }
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &extra0_q6_K->s));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem),   &extra0_q6_K->d));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem),   &b_img));
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_mem),   &extrad->data_device));
        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_ulong), &offsetd));
        CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_int),   &ne00));
        CL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_int),   &ne01));

        const size_t gws_x = use_tiled
                                 ? (size_t) CEIL_DIV(ne01, 64) * 64
                                 : use_o4
                                 ? (size_t) CEIL_DIV(ne01/4, 64) * 64
                                 : (size_t) CEIL_DIV(ne01/2, 64) * 64;
        size_t local_work_size[3]  = {64, 4, 1};
        size_t global_work_size[3] = {gws_x, 4, 1};

        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);

        if (ql_img) CL_CHECK(clReleaseMemObject(ql_img));
        if (qh_img) CL_CHECK(clReleaseMemObject(qh_img));
        CL_CHECK(clReleaseMemObject(b_sub_buffer));
        CL_CHECK(clReleaseMemObject(b_img));
    } else {
        // Tiled-layout batched GEMM. When the weight was converted to the 64-row
        // tiled canonical layout (use_q6k_tiled - the default for lm_head/embed),
        // the plain noshuffle GEMM below reads it as plain-transposed and produces
        // garbage. Use the batched GEMM that matches the decode tiled GEMV's
        // layout; it reads the f32 activation directly (column-major, no transpose).
        if (use_q6k_tiled(backend_ctx, src0)) {
            cl_mem b_sub_buf_t = nullptr;
            cl_mem b_img_t     = nullptr;

            region.origin = offset1;
            region.size = ne00 * ne1 * sizeof(float);
            CL_CHECK((b_sub_buf_t = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

            img_fmt.image_channel_order = CL_RGBA;
            img_fmt.image_channel_data_type = CL_FLOAT;
            memset(&img_desc, 0, sizeof(img_desc));
            img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
            img_desc.image_width = ne00 * ne1 / 4;
            img_desc.buffer = b_sub_buf_t;
            CL_CHECK((b_img_t = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));

            cl_kernel kt = backend_ctx->mul_mat.kernel_gemm_noshuffle_q6_K_f32_tiled;
            CL_CHECK(clSetKernelArg(kt, 0, sizeof(cl_mem),   &extra0_q6_K->ql));
            CL_CHECK(clSetKernelArg(kt, 1, sizeof(cl_mem),   &extra0_q6_K->qh));
            CL_CHECK(clSetKernelArg(kt, 2, sizeof(cl_mem),   &extra0_q6_K->s));
            CL_CHECK(clSetKernelArg(kt, 3, sizeof(cl_mem),   &extra0_q6_K->d));
            CL_CHECK(clSetKernelArg(kt, 4, sizeof(cl_mem),   &b_img_t));
            CL_CHECK(clSetKernelArg(kt, 5, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(kt, 6, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(kt, 7, sizeof(int),      &ne00));
            CL_CHECK(clSetKernelArg(kt, 8, sizeof(int),      &ne01));
            CL_CHECK(clSetKernelArg(kt, 9, sizeof(int),      &ne1));

            // Must match the kernel: NTILES=4 64-row tiles per work-group (256 rows),
            // BN=8 output columns per work-group.
            const int BN_T  = 16;
            const int WROWS = 4 * 64; // NTILES * TILE_ROWS
            size_t local_work_size[3]  = {64, 4, 1};
            size_t global_work_size[3] = {(size_t)CEIL_DIV(ne01, WROWS) * 64, 4, (size_t)CEIL_DIV(ne1, BN_T)};
            backend_ctx->enqueue_ndrange_kernel(kt, 3, global_work_size, local_work_size, dst);

            CL_CHECK(clReleaseMemObject(b_img_t));
            CL_CHECK(clReleaseMemObject(b_sub_buf_t));
            return;
        }

        cl_mem b_sub_buf;
        cl_mem b_buf_trans;
        cl_mem b_img;
        cl_mem b_img_trans;

        // subbuffer for activation
        region.origin = offset1;
        region.size = ne00 * ne1 * sizeof(float);
        CL_CHECK((b_sub_buf = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

        // dp4a (int8) dense q6_K prefill GEMM
        static const char * q6k_dense_dp4a_env = getenv("GGML_OPENCL_Q6K_DENSE_DP4A");
                     bool   q6k_dense_dp4a_on  = (q6k_dense_dp4a_env != nullptr)
                                                   ? (atoi(q6k_dense_dp4a_env) != 0)
                                                   : (backend_ctx->adreno_gen != ADRENO_GPU_GEN::X1E);
        // dot prod has to be available
        q6k_dense_dp4a_on = backend_ctx->has_integer_dot && q6k_dense_dp4a_on;

        const bool is_output_w_dp4a = strncmp(src0->name, "output", 6) == 0 ||
                                      strncmp(src0->name, "token_embd", 10) == 0;

        if (q6k_dense_dp4a_on && !is_output_w_dp4a && ne1 > 8 && (ne00 % 32 == 0) && (ne01 % 64 == 0)) {
            const int M = ne01, N = ne1, K = ne00;
            const size_t n_blocks = (size_t)N * (K / 32);
            backend_ctx->prealloc_moe_qa.allocate(context, (size_t)N * K * sizeof(cl_char));
            backend_ctx->prealloc_moe_da.allocate(context, n_blocks * sizeof(cl_half));
            backend_ctx->prealloc_moe_sa.allocate(context, n_blocks * sizeof(cl_half));

            cl_int tb = (cl_int)n_blocks;
            cl_kernel qk = backend_ctx->mul_mat.kernel_quant_a_q8_1;
            CL_CHECK(clSetKernelArg(qk, 0, sizeof(cl_mem), &b_sub_buf));
            CL_CHECK(clSetKernelArg(qk, 1, sizeof(cl_mem), &backend_ctx->prealloc_moe_qa.buffer));
            CL_CHECK(clSetKernelArg(qk, 2, sizeof(cl_mem), &backend_ctx->prealloc_moe_da.buffer));
            CL_CHECK(clSetKernelArg(qk, 3, sizeof(cl_mem), &backend_ctx->prealloc_moe_sa.buffer));
            CL_CHECK(clSetKernelArg(qk, 4, sizeof(cl_int), &tb));
            size_t q_local[1]  = { 64 };
            size_t q_global[1] = { (size_t)(((n_blocks + 63) / 64) * 64) };
            backend_ctx->enqueue_ndrange_kernel(qk, 1, q_global, q_local, dst);

            cl_kernel dk = backend_ctx->mul_mat.kernel_gemm_noshuffle_q6_k_q8_1_dp4a;
            int ai = 0;
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem),   &extra0_q6_K->ql));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem),   &extra0_q6_K->qh));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem),   &extra0_q6_K->s));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem),   &extra0_q6_K->d));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem),   &backend_ctx->prealloc_moe_qa.buffer));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem),   &backend_ctx->prealloc_moe_da.buffer));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_mem),   &extrad->data_device));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_ulong), &offsetd));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_int),   &M));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_int),   &N));
            CL_CHECK(clSetKernelArg(dk, ai++, sizeof(cl_int),   &K));
            size_t d_local[3]  = { 64, 1, 1 };
            size_t d_global[3] = { 64, (size_t)(M / 64), (size_t)CEIL_DIV(N, 32) };
            backend_ctx->enqueue_ndrange_kernel(dk, 3, d_global, d_local, dst);

            CL_CHECK(clReleaseMemObject(b_sub_buf));
            return;
        }

        // image for activation
        img_fmt.image_channel_order = CL_RGBA;
        img_fmt.image_channel_data_type = CL_FLOAT;
        memset(&img_desc, 0, sizeof(img_desc));
        img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc.image_width = ne00 * ne1 / 4;
        img_desc.buffer = b_sub_buf;
        CL_CHECK((b_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));

        // pad N to multiple of 8
        int extra_elements = ne1 % 8;
        int padding = 0;
        if (extra_elements > 0){
            padding = 8 - extra_elements;
        }

        // subbuffer for transposed activation
        region.origin = 0;
        region.size = ne00 * (ne1 + padding) * sizeof(float)/2;
        backend_ctx->prealloc_act_trans.allocate(context, region.size);
        CL_CHECK((b_buf_trans = clCreateSubBuffer(backend_ctx->prealloc_act_trans.buffer, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));

        // image for transposed activation
        img_fmt.image_channel_order = CL_RGBA;
        img_fmt.image_channel_data_type = CL_HALF_FLOAT;
        memset(&img_desc, 0, sizeof(img_desc));
        img_desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc.image_width = ne00 * (ne1 + padding) / 4;
        img_desc.buffer = b_buf_trans;
        CL_CHECK((b_img_trans = clCreateImage(context, 0, &img_fmt, &img_desc, NULL, &err), err));

        // transpose activation
        int height_B = ne1/4;
        if (height_B == 0) {
            height_B = 1;
        }
        int width_B = ne00/4;
        int padded_height_B = (ne1 + padding) / 4;

        kernel = backend_ctx->mul_mat.kernel_transpose_32_16;
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &b_img));
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_img_trans));
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int),    &height_B));
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int),    &width_B));
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int),    &padded_height_B));

        size_t local_size_t[2] = { 1, 16 };
        size_t global_size_t[2] = { (size_t)width_B, (size_t)padded_height_B };
        backend_ctx->enqueue_ndrange_kernel(kernel, 2, global_size_t, local_size_t, dst);

        // gemm
        // Cooperative-K small-batch (n_q in [2..8]) path: intra-WG K-split,
        // mirrors the q4_K _cok path (batched serving). OPT-IN
        // (GGML_OPENCL_Q6K_GEMM_COK=1), DEFAULT OFF: q6_K is the tied lm_head/
        // output projection, so the K-reassociation perturbs final logits and
        // greedy is NOT byte-identical (op-tests pass, output coherent, but not
        // bit-exact). It is also NEUTRAL on end-to-end MTP (q4_K cok already
        // captured that; the MTP bottleneck moved off the GEMMs). Keep opt-in
        // for batched serving until PPL-validated on a non-GDN q6_K model.
        static const char * q6k_cok_env = getenv("GGML_OPENCL_Q6K_GEMM_COK");
        static const bool q6k_gemm_cok  = (q6k_cok_env != nullptr) && (atoi(q6k_cok_env) != 0);
        const bool use_q6k_cok = q6k_gemm_cok && (ne1 <= 8);
        kernel = use_q6k_cok ? backend_ctx->mul_mat.kernel_gemm_noshuffle_q6_K_f32_cok
                             : backend_ctx->mul_mat.kernel_gemm_noshuffle_q6_K_f32;
        int padded_N = ne1 + padding;

        cl_ushort mask_f000 = 0xF000;
        cl_uchar  mask_c0   = 0xC0;

        CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0_q6_K->ql));
        CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0_q6_K->qh));
        CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra0_q6_K->s));
        CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_mem),   &extra0_q6_K->d));
        CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &b_img_trans));
        CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_mem),   &extrad->data_device));
        CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_ulong), &offsetd));
        CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
        CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &padded_N));
        CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne00));
        CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne1));
        CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ushort),&mask_f000));
        CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_uchar), &mask_c0));

        size_t global_work_size[3];
        size_t local_work_size[3];
        if (use_q6k_cok) {
            global_work_size[0] = (size_t)ne01;   // rows (1 per lane)
            global_work_size[1] = 8;              // COK_NSG
            global_work_size[2] = 1;
            local_work_size[0] = 64;              // COK_SG
            local_work_size[1] = 8;               // COK_NSG
            local_work_size[2] = 1;
        } else {
            global_work_size[0] = (size_t)CEIL_DIV(ne1, 8);
            global_work_size[1] = (size_t)CEIL_DIV(ne01, 4);
            global_work_size[2] = 1;
            local_work_size[0] = 2;
            local_work_size[1] = 128;
            local_work_size[2] = 1;
        }
        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);

        CL_CHECK(clReleaseMemObject(b_sub_buf));
        CL_CHECK(clReleaseMemObject(b_img));
        CL_CHECK(clReleaseMemObject(b_buf_trans));
        CL_CHECK(clReleaseMemObject(b_img_trans));
    }
}
#endif // GGML_OPENCL_USE_ADRENO_KERNELS

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
void ggml_cl_mul_mat_q4_K_glu_fused(ggml_backend_t backend, ggml_tensor * gate_tensor, ggml_tensor * up_tensor, ggml_tensor * glu_tensor) {
    GGML_ASSERT(gate_tensor && up_tensor && glu_tensor);

    const ggml_tensor * Wg   = gate_tensor->src[0];
    const ggml_tensor * Wu   = up_tensor->src[0];
    const ggml_tensor * src1 = gate_tensor->src[1];   // == up_tensor->src[1]
    const ggml_tensor * dst  = glu_tensor;

    GGML_ASSERT(Wg && Wg->extra);
    GGML_ASSERT(Wu && Wu->extra);
    GGML_ASSERT(src1 && src1->extra);
    GGML_ASSERT(dst && dst->extra);

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl       * extra1  = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl       * extrad  = (ggml_tensor_extra_cl *)dst->extra;
    ggml_tensor_extra_cl_q4_K  * extra_g = (ggml_tensor_extra_cl_q4_K *)Wg->extra;
    ggml_tensor_extra_cl_q4_K  * extra_u = (ggml_tensor_extra_cl_q4_K *)Wu->extra;

    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    const int K = Wg->ne[0];   // ne00
    const int M = Wg->ne[1];   // ne01 (= ffn intermediate width)
    const int N = 1;           // decode GEMV

    const cl_uchar mask_d6 = 0x3F, mask_d4 = 0x0F, mask_hi2 = 0xC0;
    const int glu_op = (int)ggml_get_glu_op(dst);

    cl_context context = backend_ctx->context;
    cl_int           err;
    cl_image_format  img_fmt;
    cl_image_desc    img_desc;
    cl_buffer_region region;

    // q images for the two weight matrices (standard noshuffle layout)
    img_fmt = { CL_R, CL_UNSIGNED_INT32 };
    memset(&img_desc, 0, sizeof(img_desc));
    img_desc.image_type  = CL_MEM_OBJECT_IMAGE1D_BUFFER;
    img_desc.image_width = (size_t)M * K / 2 / 4;
    img_desc.buffer      = extra_g->q;
    cl_mem qg_img = nullptr, qu_img = nullptr;
    CL_CHECK((qg_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));
    img_desc.buffer = extra_u->q;
    CL_CHECK((qu_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));

    // shared activation image (one column at decode)
    region.origin = offset1;
    region.size   = (size_t)K * N * sizeof(float);
    cl_mem b_sub_buf = nullptr, b_img = nullptr;
    CL_CHECK((b_sub_buf = clCreateSubBuffer(extra1->data_device, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err), err));
    img_fmt = { CL_RGBA, CL_FLOAT };
    memset(&img_desc, 0, sizeof(img_desc));
    img_desc.image_type  = CL_MEM_OBJECT_IMAGE1D_BUFFER;
    img_desc.image_width = (size_t)K * N / 4;
    img_desc.buffer      = b_sub_buf;
    CL_CHECK((b_img = clCreateImage(context, CL_MEM_READ_ONLY, &img_fmt, &img_desc, NULL, &err), err));

    cl_kernel kernel = backend_ctx->mul_mat.kernel_gemv_noshuffle_q4_k_f32_glu;
    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &qg_img));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra_g->d));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra_g->dm));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_mem),   &extra_g->s));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &qu_img));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_mem),   &extra_u->d));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_mem),   &extra_u->dm));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_mem),   &extra_u->s));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_mem),   &b_img));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_int),   &K));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_int),   &M));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_int),   &glu_op));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_uchar), &mask_d6));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(cl_uchar), &mask_d4));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_uchar), &mask_hi2));

    // K-split = nsg_y subgroups. HARD-CAP at 8 (512 work-items): the fused
    // kernel's cross-subgroup reduce uses a float4 reduceLM (gate+up packed) =
    // 2x the LDS of the base GEMV's float2 reduce, so 16 co-resident subgroups
    // exceed the per-CU LDS budget on X2 and the WG barrier DEADLOCKS -> GPU TDR
    // (reproduced on upstream gemma-4 E4B decode, K=2560 M=10240). This used to
    // be masked: get_kernel_workgroup_size reported 896 for this kernel (so the
    // cap loop fell to 8), but it now returns 1024 and the Adreno per-kernel WG
    // query is unreliable (over-reports), so cap explicitly instead of trusting
    // it. nsg_y < 16 also means the cross-subgroup accumulation grouping differs
    // from the standalone wide (nsg=16) GEMV, so the output is coherent but NOT
    // byte-identical to the per-op path. Keep the maxwg query as a further floor
    // for any driver that reports < 512.
    size_t maxwg = backend_ctx->get_kernel_workgroup_size(kernel);
    size_t nsg_y = 8;
    while (nsg_y > 1 && 64 * nsg_y > maxwg) { nsg_y >>= 1; }
    size_t local_work_size[3]  = { 64, nsg_y, 1 };
    size_t global_work_size[3] = { (size_t)CEIL_DIV(M / 2, 64) * 64, nsg_y, 1 };

    if (getenv("GGML_OPENCL_FUSE_DEBUG")) {
        static int dbg = 0;
        if (dbg < 3) { fprintf(stderr, "[FUSE_MM_GLU] fired #%d K=%d M=%d glu_op=%d nsg=%zu maxwg=%zu\n", ++dbg, K, M, glu_op, nsg_y, maxwg); fflush(stderr); }
    }

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);

    CL_CHECK(clReleaseMemObject(qg_img));
    CL_CHECK(clReleaseMemObject(qu_img));
    CL_CHECK(clReleaseMemObject(b_img));
    CL_CHECK(clReleaseMemObject(b_sub_buf));
}
#endif // GGML_OPENCL_USE_ADRENO_KERNELS
