#include "../cl-common.h"
#include "../ops.h"

void ggml_cl_load_kernels_mul_mat(ggml_backend_opencl_context * backend_ctx) {
    cl_int err;
    const std::string & compile_opts = backend_ctx->kernel_compile_opts;

    // mul_mv_q4_0_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_q4_0_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_q4_0_f32.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mat_q4_0_f32 = clCreateKernel(prog, "kernel_mul_mat_q4_0_f32", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mv_q4_0_f32_v
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_q4_0_f32_v.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_q4_0_f32_v.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mat_q4_0_f32_v = clCreateKernel(prog, "kernel_mul_mat_q4_0_f32_v", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mv_q4_0_f32_8x_flat
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_q4_0_f32_8x_flat.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_q4_0_f32_8x_flat.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mat_q4_0_f32_8x_flat = clCreateKernel(prog, "kernel_mul_mat_q4_0_f32_8x_flat", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mv_q4_0_f32_1d_8x_flat
    // This kernel does not compiler on Adreno cl compiler 38.01. Skip it for
    // those compiler versions since it is anyway not used for Adreno.
    if (backend_ctx->gpu_family != ADRENO ||
        backend_ctx->adreno_cl_compiler_version.newer_than_or_same(E031, 38, 11, 0) ||
        backend_ctx->adreno_cl_compiler_version.type == E17 ||
        backend_ctx->adreno_cl_compiler_version.type == DX) {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_q4_0_f32_1d_8x_flat.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_q4_0_f32_1d_8x_flat.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mat_q4_0_f32_1d_8x_flat = clCreateKernel(prog, "kernel_mul_mat_q4_0_f32_1d_8x_flat", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mv_q4_0_f32_1d_16x_flat
    // This kernel does not compiler on Adreno cl compiler 38.01. Skip it for
    // those compiler versions since it is anyway not used for Adreno.
    if (backend_ctx->gpu_family != ADRENO ||
        backend_ctx->adreno_cl_compiler_version.newer_than_or_same(E031, 38, 11, 0) ||
    backend_ctx->adreno_cl_compiler_version.type == DX) {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_q4_0_f32_1d_16x_flat.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_q4_0_f32_1d_16x_flat.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mat_q4_0_f32_1d_16x_flat = clCreateKernel(prog, "kernel_mul_mat_q4_0_f32_1d_16x_flat", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mv_q4_1_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_q4_1_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_q4_1_f32.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mv_q4_1_f32 = clCreateKernel(prog, "kernel_mul_mv_q4_1_f32", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mv_q4_1_f32_flat
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_q4_1_f32_flat.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_q4_1_f32_flat.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mv_q4_1_f32_flat = clCreateKernel(prog, "kernel_mul_mv_q4_1_f32_flat", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mv_q4_k_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_q4_k_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_q4_k_f32.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mv_q4_K_f32 = clCreateKernel(prog, "kernel_mul_mv_q4_K_f32", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mv_q4_k_f32_flat
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_q4_k_f32_flat.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_q4_k_f32_flat.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mv_q4_K_f32_flat = clCreateKernel(prog, "kernel_mul_mv_q4_K_f32_flat", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mv_q5_0_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_q5_0_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_q5_0_f32.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mv_q5_0_f32 = clCreateKernel(prog, "kernel_mul_mv_q5_0_f32", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mv_q5_0_f32_flat
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_q5_0_f32_flat.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_q5_0_f32_flat.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mv_q5_0_f32_flat = clCreateKernel(prog, "kernel_mul_mv_q5_0_f32_flat", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mv_q5_1_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_q5_1_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_q5_1_f32.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mv_q5_1_f32 = clCreateKernel(prog, "kernel_mul_mv_q5_1_f32", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mv_q5_1_f32_flat
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_q5_1_f32_flat.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_q5_1_f32_flat.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mv_q5_1_f32_flat = clCreateKernel(prog, "kernel_mul_mv_q5_1_f32_flat", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mv_q5_k_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_q5_k_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_q5_k_f32.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mv_q5_K_f32 = clCreateKernel(prog, "kernel_mul_mv_q5_K_f32", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mv_q5_k_f32_flat
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_q5_k_f32_flat.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_q5_k_f32_flat.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mv_q5_K_f32_flat = clCreateKernel(prog, "kernel_mul_mv_q5_K_f32_flat", &err), err));
        CL_CHECK(clReleaseProgram(prog));
    }

    // mul_mv_q6_k_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_q6_k_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_q6_k_f32.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mv_q6_K_f32 = clCreateKernel(prog, "kernel_mul_mv_q6_K_f32", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mv_q6_k_f32_flat
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_q6_k_f32_flat.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_q6_k_f32_flat.cl");
#endif
        const std::string q6k_opts = backend_ctx->q6_k_flat_old_compiler
            ? compile_opts + " -DADRENO_OLD_COMPILER=1"
            : compile_opts;
        cl_program prog = build_program_from_source(backend_ctx, kernel_src.c_str(), q6k_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mv_q6_K_f32_flat = clCreateKernel(prog, "kernel_mul_mv_q6_K_f32_flat", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mv_q8_0_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_q8_0_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_q8_0_f32.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mv_q8_0_f32 = clCreateKernel(prog, "kernel_mul_mv_q8_0_f32", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mv_q8_0_f32_flat
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_q8_0_f32_flat.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_q8_0_f32_flat.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mv_q8_0_f32_flat = clCreateKernel(prog, "kernel_mul_mv_q8_0_f32_flat", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mv_q1_0_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_q1_0_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_q1_0_f32.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mv_q1_0_f32 = clCreateKernel(prog, "kernel_mul_mv_q1_0_f32", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mv_q1_0_f32_flat
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_q1_0_f32_flat.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_q1_0_f32_flat.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mv_q1_0_f32_flat = clCreateKernel(prog, "kernel_mul_mv_q1_0_f32_flat", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mv_iq4_nl_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_iq4_nl_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_iq4_nl_f32.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mv_iq4_nl_f32 = clCreateKernel(prog, "kernel_mul_mv_iq4_nl_f32", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mv_iq4_nl_f32_flat
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_iq4_nl_f32_flat.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_iq4_nl_f32_flat.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mv_iq4_nl_f32_flat = clCreateKernel(prog, "kernel_mul_mv_iq4_nl_f32_flat", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mv_mxfp4_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_mxfp4_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_mxfp4_f32.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mv_mxfp4_f32 = clCreateKernel(prog, "kernel_mul_mv_mxfp4_f32", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mv_mxfp4_f32_flat
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_mxfp4_f32_flat.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_mxfp4_f32_flat.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mv_mxfp4_f32_flat = clCreateKernel(prog, "kernel_mul_mv_mxfp4_f32_flat", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mv_f16_f16
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_f16_f16.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_f16_f16.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mat_f16_f16 = clCreateKernel(prog, "kernel_mul_mat_f16_f16", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mv_f16_f32_1row
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_f16_f32_1row.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_f16_f32_1row.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mat_f16_f32_1row = clCreateKernel(prog, "kernel_mul_mat_f16_f32_1row", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mv_f16_f32_mrow (multi-row decode GEMV)
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_f16_f32_mrow.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_f16_f32_mrow.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mat_f16_f32_mrow = clCreateKernel(prog, "kernel_mul_mat_f16_f32_mrow", &err), err));
        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mat_f16_f32_mrow_r2 = clCreateKernel(prog, "kernel_mul_mat_f16_f32_mrow_r2", &err), err));
        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mat_f16_f32_mrow_r4 = clCreateKernel(prog, "kernel_mul_mat_f16_f32_mrow_r4", &err), err));
        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mat_f16_f32_mrow_h8 = clCreateKernel(prog, "kernel_mul_mat_f16_f32_mrow_h8", &err), err));
        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mat_f16_f32_mrow_h8r2 = clCreateKernel(prog, "kernel_mul_mat_f16_f32_mrow_h8r2", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mv_f16_f32_l4
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_f16_f32_l4.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_f16_f32_l4.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4   = clCreateKernel(prog, "kernel_mul_mat_f16_f32_l4", &err), err));
        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_dr = clCreateKernel(prog, "kernel_mul_mat_f16_f32_l4_dr", &err), err));
        if (backend_ctx->gpu_family == ADRENO) {
            CL_CHECK((backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_dr_ls = clCreateKernel(prog, "kernel_mul_mat_f16_f32_l4_dr_ls", &err), err));
            CL_CHECK((backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_dr_lq = clCreateKernel(prog, "kernel_mul_mat_f16_f32_l4_dr_lq", &err), err));
        }

        cl_int err_x8 = CL_SUCCESS;
        backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_x8 =
            clCreateKernel(prog, "kernel_mul_mat_f16_f32_l4_x8", &err_x8);
        if (err_x8 != CL_SUCCESS) { backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_x8 = nullptr; }

        cl_int err_x8p = CL_SUCCESS;
        backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_x8_pair =
            clCreateKernel(prog, "kernel_mul_mat_f16_f32_l4_x8_pair", &err_x8p);
        if (err_x8p != CL_SUCCESS) { backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_x8_pair = nullptr; }

        cl_int err_x8g = CL_SUCCESS;
        backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_x8_gqa4 =
            clCreateKernel(prog, "kernel_mul_mat_f16_f32_l4_x8_gqa4", &err_x8g);
        if (err_x8g != CL_SUCCESS) { backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_x8_gqa4 = nullptr; }

        cl_int err_y8 = CL_SUCCESS;
        backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_y8 =
            clCreateKernel(prog, "kernel_mul_mat_f16_f32_l4_y8", &err_y8);
        if (err_y8 != CL_SUCCESS) { backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_y8 = nullptr; }

        cl_int err_y8g = CL_SUCCESS;
        backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_y8_gqa =
            clCreateKernel(prog, "kernel_mul_mat_f16_f32_l4_y8_gqa", &err_y8g);
        if (err_y8g != CL_SUCCESS) { backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_y8_gqa = nullptr; }

        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mv_f16_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_f16_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_f16_f32.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mat_f16_f32 = clCreateKernel(prog, "kernel_mul_mat_f16_f32", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mv_f32_f32
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mv_f32_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mv_f32_f32.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mat_f32_f32 = clCreateKernel(prog, "kernel_mul_mat_f32_f32", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mat_f16_f32_tiled
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mat_f16_f32.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mat_f16_f32.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mat_f16_f32_tiled = clCreateKernel(prog, "mul_mat_f16_f32", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mm_f32_f32_l4_lm
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mm_f32_f32_l4_lm.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mm_f32_f32_l4_lm.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mm_f32_f32_l4_lm = clCreateKernel(prog, "kernel_mul_mm_f32_f32_l4_lm", &err), err));
        CL_CHECK((backend_ctx->mul_mat.kernel_gemv_f32_f32_mc = clCreateKernel(prog, "kernel_gemv_f32_f32_mc", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mm_f16_f32_l4_lm
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mm_f16_f32_l4_lm.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mm_f16_f32_l4_lm.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mm_f16_f32_l4_lm = clCreateKernel(prog, "kernel_mul_mm_f16_f32_l4_lm", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mm_q4_0_f32_l4_lm
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mm_q4_0_f32_l4_lm.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mm_q4_0_f32_l4_lm.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mm_q4_0_f32_l4_lm = clCreateKernel(prog, "kernel_mul_mm_q4_0_f32_l4_lm", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mm_q4_1_f32_l4_lm
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mm_q4_1_f32_l4_lm.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mm_q4_1_f32_l4_lm.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mm_q4_1_f32_l4_lm = clCreateKernel(prog, "kernel_mul_mm_q4_1_f32_l4_lm", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mm_q5_0_f32_l4_lm
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mm_q5_0_f32_l4_lm.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mm_q5_0_f32_l4_lm.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mm_q5_0_f32_l4_lm = clCreateKernel(prog, "kernel_mul_mm_q5_0_f32_l4_lm", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mm_q5_1_f32_l4_lm
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mm_q5_1_f32_l4_lm.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mm_q5_1_f32_l4_lm.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mm_q5_1_f32_l4_lm = clCreateKernel(prog, "kernel_mul_mm_q5_1_f32_l4_lm", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mm_q8_0_f32_l4_lm
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mm_q8_0_f32_l4_lm.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mm_q8_0_f32_l4_lm.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mm_q8_0_f32_l4_lm = clCreateKernel(prog, "kernel_mul_mm_q8_0_f32_l4_lm", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mm_q1_0_f32_l4_lm
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mm_q1_0_f32_l4_lm.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mm_q1_0_f32_l4_lm.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mm_q1_0_f32_l4_lm = clCreateKernel(prog, "kernel_mul_mm_q1_0_f32_l4_lm", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mm_iq4_nl_f32_l4_lm
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mm_iq4_nl_f32_l4_lm.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mm_iq4_nl_f32_l4_lm.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mm_iq4_nl_f32_l4_lm = clCreateKernel(prog, "kernel_mul_mm_iq4_nl_f32_l4_lm", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mm_q4_k_f32_l4_lm
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mm_q4_k_f32_l4_lm.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mm_q4_k_f32_l4_lm.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mm_q4_k_f32_l4_lm = clCreateKernel(prog, "kernel_mul_mm_q4_k_f32_l4_lm", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mm_q6_k_f32_l4_lm
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mm_q6_k_f32_l4_lm.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mm_q6_k_f32_l4_lm.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mm_q6_k_f32_l4_lm = clCreateKernel(prog, "kernel_mul_mm_q6_k_f32_l4_lm", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }

    // mul_mm_q5_k_f32_l4_lm
    {
#ifdef GGML_OPENCL_EMBED_KERNELS
        const std::string kernel_src {
            #include "mul_mm_q5_k_f32_l4_lm.cl.h"
        };
#else
        const std::string kernel_src = read_file("mul_mm_q5_k_f32_l4_lm.cl");
#endif
        cl_program prog =
            build_program_from_source(backend_ctx, kernel_src.c_str(), compile_opts);

        CL_CHECK((backend_ctx->mul_mat.kernel_mul_mm_q5_k_f32_l4_lm = clCreateKernel(prog, "kernel_mul_mm_q5_k_f32_l4_lm", &err), err));
        CL_CHECK(clReleaseProgram(prog));
        GGML_LOG_CONT(".");
    }
}

// Copy a noncontiguous tensor to contiguous tensor. ne[] remains the same but
// nb[] is recalculated such that tensor is contiguous.
static void ggml_cl_copy_to_contiguous(ggml_backend_t backend, const ggml_tensor * src, cl_mem dst,
                                       cl_ulong &nb0, cl_ulong &nb1, cl_ulong &nb2, cl_ulong &nb3) {
    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    const int tensor_type_size = ggml_type_size(src->type);

    const int ne00 = src->ne[0];
    const int ne01 = src->ne[1];
    const int ne02 = src->ne[2];
    const int ne03 = src->ne[3];

    const cl_ulong nb00 = src->nb[0];
    const cl_ulong nb01 = src->nb[1];
    const cl_ulong nb02 = src->nb[2];
    const cl_ulong nb03 = src->nb[3];

    const int ne0 = src->ne[0];
    const int ne1 = src->ne[1];
    const int ne2 = src->ne[2];
    const int ne3 = src->ne[3];

    nb0 = tensor_type_size;
    nb1 = tensor_type_size*ne00;
    nb2 = tensor_type_size*ne00*ne01;
    nb3 = tensor_type_size*ne00*ne01*ne02;

    ggml_tensor_extra_cl * extra = (ggml_tensor_extra_cl *)src->extra;

    cl_ulong offset0 = extra->offset + src->view_offs;
    cl_ulong offsetd = 0;

    cl_kernel kernel;

    switch (src->type) {
        case GGML_TYPE_F32:
            kernel = backend_ctx->cpy.kernel_cpy_f32_f32;
            break;
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16: // stored as f16 on device
            kernel = backend_ctx->cpy.kernel_cpy_f16_f16;
            break;
        default:
            GGML_ASSERT(false && "not implemented");
    }

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &dst));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne02));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne03));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &nb00));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb02));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb03));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne0));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne1));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &ne2));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne3));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nb0));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong), &nb1));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong), &nb2));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &nb3));

    const int nth = MIN(64, ne00);

    size_t global_work_size[] = {(size_t)ne01*nth, (size_t)ne02, (size_t)ne03};
    size_t local_work_size[] = {(size_t)nth, 1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, src);
}

// Dequant a possibly-strided q4_0/q8_0 tensor to tight-packed f16. Returns a
// temp cl_mem the caller must release. SoA inputs are reconstructed into a
// temp AoS buffer reported via *extra_reconstruct (also caller-released).
// this is for quantized K cache without FA.
static cl_mem ggml_cl_mul_mat_dequant_quant_to_f16(
        ggml_backend_opencl_context * backend_ctx,
        const ggml_tensor *           tensor,
        cl_mem *                      extra_reconstruct /* out, may be NULL */
) {
    GGML_ASSERT(tensor->type == GGML_TYPE_Q4_0 || tensor->type == GGML_TYPE_Q8_0);

    if (extra_reconstruct) {
        *extra_reconstruct = NULL;
    }

    cl_mem   src_buf;
    cl_ulong src_offset;
    cl_ulong src_nb1;
    cl_ulong src_nb2;
    cl_ulong src_nb3;

    uintptr_t pool_key_buf = 0;
    cl_ulong  pool_key_off = (cl_ulong) tensor->view_offs;

    const bool is_soa = tensor->type == GGML_TYPE_Q4_0
        ? ggml_cl_is_q4_0_soa(tensor)
        : ggml_cl_is_q8_0_soa(tensor);

    cl_mem aos = nullptr;
    if (is_soa) {
        // Reconstruct full parent AoS; view's own nb[] then index it correctly.
        const ggml_tensor * parent = tensor->view_src ? tensor->view_src : tensor;
        const ggml_tensor * soa_src = parent;
        const size_t block_bytes = (size_t) ggml_type_size(tensor->type);
        const size_t blck_size   = (size_t) ggml_blck_size(tensor->type);
        const size_t parent_row_blocks = (size_t) parent->ne[0] / blck_size;
        const size_t parent_row_bytes  = parent_row_blocks * block_bytes;
        const size_t parent_nbytes = (size_t) ggml_nelements(parent) / blck_size * block_bytes;

        cl_int err;
        aos = clCreateBuffer(backend_ctx->context, CL_MEM_READ_WRITE, parent_nbytes, NULL, &err);
        CL_CHECK(err);

        // large q4_0/q8_0 WEIGHTS are stored transposed and small weights
        // (and the AoS KV-cache, handled in the else branch above) are not.
        // choose a proper restore kernel based on this.
        bool restored = false;
#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        const int p_ne00 = (int) parent->ne[0];
        const int p_ne01 = (int) parent->ne[1];
        if (tensor->type == GGML_TYPE_Q8_0 && enable_adreno_trans_weight(backend_ctx, parent)) {
            auto * extra = (ggml_tensor_extra_cl_q8_0 *) soa_src->extra;
            pool_key_buf = (uintptr_t) extra->q;
            cl_kernel kernel = backend_ctx->repack.kernel_restore_block_q8_0_trans;
            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &extra->q));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->d));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &aos));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_int), &p_ne00));
            CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_int), &p_ne01));
            size_t gws[] = { (size_t)(((p_ne01 + 63) / 64) * 64), 1, 1 };
            size_t lws[] = { 64, 1, 1 };
            CL_CHECK(clEnqueueNDRangeKernel(backend_ctx->queue, kernel, 3, NULL, gws, lws, 0, NULL, NULL));
            restored = true;
        } else if (tensor->type == GGML_TYPE_Q4_0 &&
                   use_adreno_kernels(backend_ctx, parent) &&
                   !use_adreno_moe_kernels(backend_ctx, parent)) {
            auto * extra = (ggml_tensor_extra_cl_q4_0 *) soa_src->extra;
            pool_key_buf = (uintptr_t) extra->q;
            const size_t size_q = (size_t) ggml_nelements(parent) / blck_size * (blck_size / 2);
            const size_t size_d = (size_t) ggml_nelements(parent) / blck_size * sizeof(ggml_fp16_t);

            cl_int err;
            cl_mem buf_tq = clCreateBuffer(backend_ctx->context, CL_MEM_READ_WRITE, size_q, NULL, &err);
            CL_CHECK(err);

            cl_mem buf_td = clCreateBuffer(backend_ctx->context, CL_MEM_READ_WRITE, size_d, NULL, &err);
            CL_CHECK(err);
            transpose_2d_as_16b(backend_ctx, extra->q, buf_tq, size_q, p_ne01, p_ne00 / 4);
            transpose_2d_as_16b(backend_ctx, extra->d, buf_td, size_d, p_ne01, p_ne00 / 32);

            cl_uchar mask_0F = 0x0F;
            cl_uchar mask_F0 = 0xF0;
            cl_kernel kernel = backend_ctx->repack.kernel_restore_block_q4_0_noshuffle;
            CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem),   &buf_tq));
            CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem),   &buf_td));
            CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem),   &aos));
            CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_uchar), &mask_0F));
            CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_uchar), &mask_F0));

            const size_t n_blk = parent_nbytes / block_bytes;
            size_t gws[] = { n_blk, 1, 1 };
            size_t lws[] = { 1, 1, 1 };
            CL_CHECK(clEnqueueNDRangeKernel(backend_ctx->queue, kernel, 3, NULL, gws, lws, 0, NULL, NULL));

            CL_CHECK(clReleaseMemObject(buf_tq));
            CL_CHECK(clReleaseMemObject(buf_td));
            restored = true;
        }
#endif // GGML_OPENCL_USE_ADRENO_KERNELS

        if (!restored) {
            cl_kernel kernel;
            if (tensor->type == GGML_TYPE_Q8_0) {
                auto * extra = (ggml_tensor_extra_cl_q8_0 *) soa_src->extra;
                kernel = backend_ctx->repack.kernel_restore_block_q8_0;
                CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &extra->q));
                CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->d));
                CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &aos));
                pool_key_buf = (uintptr_t) extra->q;
            } else {
                auto * extra = (ggml_tensor_extra_cl_q4_0 *) soa_src->extra;
                kernel = backend_ctx->repack.kernel_restore_block_q4_0;
                CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &extra->q));
                CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &extra->d));
                CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &aos));
                pool_key_buf = (uintptr_t) extra->q;
            }

            const size_t n_blocks = parent_nbytes / block_bytes;
            size_t gws_rec[] = { n_blocks, 1, 1 };
            size_t lws_rec[] = { 1, 1, 1 };
            CL_CHECK(clEnqueueNDRangeKernel(backend_ctx->queue, kernel, 3, NULL, gws_rec, lws_rec, 0, NULL, NULL));
        }

        (void) parent_row_blocks;
        (void) parent_row_bytes;
        src_buf    = aos;
        src_offset = tensor->view_offs;
        src_nb1    = tensor->nb[1];
        src_nb2    = tensor->nb[2];
        src_nb3    = tensor->nb[3];

        if (extra_reconstruct) {
            *extra_reconstruct = aos;
        }
    } else {
        auto * extra = (ggml_tensor_extra_cl *) tensor->extra;
        GGML_ASSERT(extra && extra->data_device);
        src_buf    = extra->data_device;
        src_offset = extra->offset + tensor->view_offs;
        src_nb1    = tensor->nb[1];
        src_nb2    = tensor->nb[2];
        src_nb3    = tensor->nb[3];
        pool_key_buf = (uintptr_t) extra->data_device;
        pool_key_off = (cl_ulong) src_offset;
    }

    const cl_int nblk0 = (cl_int) (tensor->ne[0] / ggml_blck_size(tensor->type));
    const cl_int ne1_  = (cl_int) tensor->ne[1];
    const cl_int ne2_  = (cl_int) tensor->ne[2];
    const cl_int ne3_  = (cl_int) tensor->ne[3];

    const size_t out_bytes = (size_t) ggml_nelements(tensor) * sizeof(ggml_fp16_t);

    // reuse a pooled f16 buffer for this KV-cache view across decode steps instead of
    // allocating new one per attention op
    cl_mem out = nullptr;
    {
        auto & pool = backend_ctx->dequant_f16_pool;
        ggml_backend_opencl_context::ImagePoolKey key{pool_key_buf, (uint64_t) pool_key_off};
        auto it = pool.find(key);
        if (it != pool.end() && it->second.k_bytes >= out_bytes && it->second.image) {
            out = it->second.image;
        } else {
            if (it != pool.end()) {
                if (it->second.image) { CL_CHECK(clReleaseMemObject(it->second.image)); }
                pool.erase(it);
            }
            cl_int err = CL_SUCCESS;
            out = clCreateBuffer(backend_ctx->context, CL_MEM_READ_WRITE, out_bytes, NULL, &err);
            CL_CHECK(err);
            ggml_backend_opencl_context::ImagePoolEntry entry;
            entry.image   = out;
            entry.k_bytes = out_bytes;
            pool[key]     = entry;
        }
    }

    cl_kernel dq_kernel = tensor->type == GGML_TYPE_Q8_0
        ? backend_ctx->repack.kernel_dequant_q8_0_f16_view_aos
        : backend_ctx->repack.kernel_dequant_q4_0_f16_view_aos;

    CL_CHECK(clSetKernelArg(dq_kernel, 0, sizeof(cl_mem),   &src_buf));
    CL_CHECK(clSetKernelArg(dq_kernel, 1, sizeof(cl_ulong), &src_offset));
    CL_CHECK(clSetKernelArg(dq_kernel, 2, sizeof(cl_ulong), &src_nb1));
    CL_CHECK(clSetKernelArg(dq_kernel, 3, sizeof(cl_ulong), &src_nb2));
    CL_CHECK(clSetKernelArg(dq_kernel, 4, sizeof(cl_ulong), &src_nb3));
    CL_CHECK(clSetKernelArg(dq_kernel, 5, sizeof(cl_int),   &nblk0));
    CL_CHECK(clSetKernelArg(dq_kernel, 6, sizeof(cl_int),   &ne1_));
    CL_CHECK(clSetKernelArg(dq_kernel, 7, sizeof(cl_int),   &ne2_));
    CL_CHECK(clSetKernelArg(dq_kernel, 8, sizeof(cl_int),   &ne3_));
    CL_CHECK(clSetKernelArg(dq_kernel, 9, sizeof(cl_mem),   &out));

    size_t gws[3] = { (size_t) nblk0, (size_t) ne1_, (size_t) (ne2_ * ne3_) };
    size_t lws[3] = { 1, 1, 1 };
    CL_CHECK(clEnqueueNDRangeKernel(backend_ctx->queue, dq_kernel, 3, NULL, gws, lws, 0, NULL, NULL));

    // release the reconstructed aos if
    //  1. it was actually reconstructed
    //  2. the caller didn't request it to be returned
    // src_buf may refer to aos, so we should release after this enqueue
    if (aos && !extra_reconstruct) {
        CL_CHECK(clReleaseMemObject(aos));
    }
    return out;
}

static bool ggml_cl_mm_f32_f32(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_backend_opencl_context * backend_ctx = (ggml_backend_opencl_context *) backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *) src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *) src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *) dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    GGML_TENSOR_LOCALS(int,      ne0, src0, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb0, src0, nb);
    GGML_TENSOR_LOCALS(int,      ne1, src1, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb1, src1, nb);
    GGML_TENSOR_LOCALS(int,      ne,  dst,  ne);

    int r2 = ne12/ne02;
    int r3 = ne13/ne03;

    // Small-N f32 GEMV for the spec/MTP verify batch: the tiled GEMM
    // below always computes a full 64x64 tile, so at ne11=3 with a
    // skinny f32 weight (GDN ssm_alpha/ssm_beta, M=32) it launches one
    // under-occupied WG at ~2.3% tile utilization. Route to a per-output
    // (m,n) GEMV (64-thread WG, K-split + __local reduce) instead.
    // Opt-in GGML_OPENCL_F32_MC=1; 2D contiguous, small N + skinny M only.
    static const bool f32_mc = (getenv("GGML_OPENCL_F32_MC") != nullptr);
    if (f32_mc && ne11 >= 2 && ne11 <= 8 && ne01 <= 512 && (ne00 % 4 == 0) &&
        ne02 == 1 && ne12 == 1 && ne13 == 1 &&
        ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        cl_kernel kernel = backend_ctx->mul_mat.kernel_gemv_f32_f32_mc;

        int stride_a = ne00;
        int stride_b = ne00;
        int stride_d = ne01;
        CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
        CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
        CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
        CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
        CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
        CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
        CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
        CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
        CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne11));
        CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &stride_a));
        CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &stride_b));
        CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &stride_d));
        size_t global_work_size[3] = {64, (size_t) ne01 * (size_t) ne11, 1};
        size_t local_work_size[3] = {64, 1, 1};
        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
        return true;
    }

    cl_kernel kernel = backend_ctx->mul_mat.kernel_mul_mm_f32_f32_l4_lm;
    const int nth0 = 128; // calculated as (BM*BN)/(TM*TN)

    int batch_stride_a = ne00*ne01;
    int batch_stride_b = ne10*ne11;
    int batch_stride_d = ne0*ne1;

    cl_mem mem_src0 = extra0->data_device;
    cl_mem mem_src1 = extra1->data_device;

    cl_ulong nb00_cont = nb00;
    cl_ulong nb01_cont = nb01;
    cl_ulong nb02_cont = nb02;
    cl_ulong nb03_cont = nb03;

    cl_ulong nb10_cont = nb10;
    cl_ulong nb11_cont = nb11;
    cl_ulong nb12_cont = nb12;
    cl_ulong nb13_cont = nb13;

    cl_ulong offset0_cont = offset0;
    cl_ulong offset1_cont = offset1;

    if (!ggml_is_contiguous(src0)) {
        backend_ctx->prealloc_src0.allocate(backend_ctx->context, ggml_nbytes(src0));
        ggml_cl_copy_to_contiguous(backend, src0, backend_ctx->prealloc_src0.buffer,
            nb00_cont, nb01_cont, nb02_cont, nb03_cont);
        mem_src0 = backend_ctx->prealloc_src0.buffer;
        offset0_cont = 0;
    }

    if (!ggml_is_contiguous(src1)) {
        backend_ctx->prealloc_src1.allocate(backend_ctx->context, ggml_nbytes(src1));
        ggml_cl_copy_to_contiguous(backend, src1, backend_ctx->prealloc_src1.buffer,
            nb10_cont, nb11_cont, nb12_cont, nb13_cont);
        mem_src1 = backend_ctx->prealloc_src1.buffer;
        offset1_cont = 0;
    }

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &mem_src0));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0_cont));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &mem_src1));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1_cont));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne02));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne11));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne12));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne10)); // stride_a
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne10)); // stride_b
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne01)); // stride_d
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &batch_stride_a));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &batch_stride_b));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &batch_stride_d));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &r2));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &r3));

    // 64 is block tile size BM and BN - change here when BM and BN in the kernel are changed.
    size_t global_work_size[] = {(size_t)(CEIL_DIV(ne01, 64)*nth0), (size_t)(CEIL_DIV(ne11, 64)), (size_t)ne12*ne13};
    size_t local_work_size[] = {(size_t)nth0, 1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
    return true;
}

static bool ggml_cl_mm_f16_f32(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_backend_opencl_context * backend_ctx = (ggml_backend_opencl_context *) backend->context;

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
    if (ggml_cl_can_use_adreno_xmem_gemm_f16_f32(backend_ctx, src0, src1, dst)) {
        ggml_cl_mul_mat_f16_f32_adreno_xmem(backend, src0, src1, dst);
        return true;
    }
#endif

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *) src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *) src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *) dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    GGML_TENSOR_LOCALS(int,      ne0, src0, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb0, src0, nb);
    GGML_TENSOR_LOCALS(int,      ne1, src1, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb1, src1, nb);
    GGML_TENSOR_LOCALS(int,      ne,  dst,  ne);

    int r2 = ne12/ne02;
    int r3 = ne13/ne03;

    cl_kernel kernel = backend_ctx->mul_mat.kernel_mul_mm_f16_f32_l4_lm;
    const int nth0 = 128; // calculated as (BM*BN)/(TM*TN)

    int batch_stride_a = ne00*ne01;
    int batch_stride_b = ne10*ne11;
    int batch_stride_d = ne0*ne1;

    cl_mem mem_src0 = extra0->data_device;
    cl_mem mem_src1 = extra1->data_device;

    cl_ulong nb00_cont = nb00;
    cl_ulong nb01_cont = nb01;
    cl_ulong nb02_cont = nb02;
    cl_ulong nb03_cont = nb03;

    cl_ulong nb10_cont = nb10;
    cl_ulong nb11_cont = nb11;
    cl_ulong nb12_cont = nb12;
    cl_ulong nb13_cont = nb13;

    cl_ulong offset0_cont = offset0;
    cl_ulong offset1_cont = offset1;

    if (!ggml_is_contiguous(src0)) {
        backend_ctx->prealloc_src0.allocate(backend_ctx->context, ggml_nbytes(src0));
        ggml_cl_copy_to_contiguous(backend, src0, backend_ctx->prealloc_src0.buffer,
            nb00_cont, nb01_cont, nb02_cont, nb03_cont);
        mem_src0 = backend_ctx->prealloc_src0.buffer;
        offset0_cont = 0;
    }

    if (!ggml_is_contiguous(src1)) {
        backend_ctx->prealloc_src1.allocate(backend_ctx->context, ggml_nbytes(src1));
        ggml_cl_copy_to_contiguous(backend, src1, backend_ctx->prealloc_src1.buffer,
                nb10_cont, nb11_cont, nb12_cont, nb13_cont);
        mem_src1 = backend_ctx->prealloc_src1.buffer;
        offset1_cont = 0;
    }

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &mem_src0));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0_cont));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &mem_src1));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1_cont));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne02));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne11));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne12));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne10)); // stride_a
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne10)); // stride_b
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne01)); // stride_d
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &batch_stride_a));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &batch_stride_b));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &batch_stride_d));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &r2));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &r3));

    // 64 is block tile size BM and BN - change here when BM and BN in the kernel are changed.
    size_t global_work_size[] = {(size_t)(CEIL_DIV(ne01, 64)*nth0), (size_t)(CEIL_DIV(ne11, 64)), (size_t)ne12*ne13};
    size_t local_work_size[] = {(size_t)nth0, 1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
    return true;
}

static void ggml_cl_mul_mat_f16_f32_tiled(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *)src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    const int M = src0->ne[1];
    const int N = src1->ne[1];
    const int K = src0->ne[0];

    cl_kernel kernel = backend_ctx->mul_mat.kernel_mul_mat_f16_f32_tiled;

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(int),      &M));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(int),      &N));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int),      &K));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_ulong), &offsetd));

    // Tiling parameters. These need to be tuned for optimal performance.
    // They must match the #defines in the kernel mul_mat_f16_f32.cl.
    //
    // OPWM / OPWN: Output tile size per Work-Group. A work-group computes a tile of size OPWM x OPWN.
    // TPWM / TPWN: Threads per Work-group. This is the work-group size.
    // OPTM / OPTN: Output elements per Thread. Each thread computes OPTM x OPTN elements.
    //
    // The following relationships must hold:
    //   OPWM = TPWM * OPTM
    //   OPWN = TPWN * OPTN
    //
    const int OPWM = 64;
    const int OPWN = 64;
    const int TPWM = 16;
    const int TPWN = 8;

    size_t local_work_size[2] = { TPWM, TPWN };
    size_t global_work_size[2] = {
        (size_t) ((M + OPWM - 1) / OPWM) * TPWM,
        (size_t) ((N + OPWN - 1) / OPWN) * TPWN,
    };

    backend_ctx->enqueue_ndrange_kernel(kernel, 2, global_work_size, local_work_size, dst);
}

static bool ggml_cl_mm_q1_0_f32(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_LOCALS(int, ne0, src0, ne);
    GGML_TENSOR_LOCALS(int, ne1, src1, ne);
    GGML_TENSOR_LOCALS(int, ne,  dst,  ne);

    if (ne11 < 32) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
        return false;
    }

    ggml_backend_opencl_context * backend_ctx = (ggml_backend_opencl_context *) backend->context;

    ggml_tensor_extra_cl_q1_0 * extra0 = (ggml_tensor_extra_cl_q1_0 *) src0->extra;
    ggml_tensor_extra_cl *      extra1 = (ggml_tensor_extra_cl *) src1->extra;
    ggml_tensor_extra_cl *      extrad = (ggml_tensor_extra_cl *) dst->extra;

    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    int r2 = ne12/ne02;
    int r3 = ne13/ne03;

    cl_kernel kernel = backend_ctx->mul_mat.kernel_mul_mm_q1_0_f32_l4_lm;
    const int nth0 = 128; // calculated as (BM*BN)/(TM*TN)

    int batch_stride_a = ne00*ne01;
    int batch_stride_b = ne10*ne11;
    int batch_stride_d = ne0*ne1;

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->q));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0->d));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne02));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne11));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne12));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne10)); // stride_a
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne10)); // stride_b
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne01)); // stride_d
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &batch_stride_a));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &batch_stride_b));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &batch_stride_d));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &r2));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &r3));

    // 64 is block tile size BM and BN - change here when BM and BN in the kernel are changed.
    size_t global_work_size[] = {(size_t)(CEIL_DIV(ne01, 64)*nth0), (size_t)(CEIL_DIV(ne11, 64)), (size_t)ne12*ne13};
    size_t local_work_size[] = {(size_t)nth0, 1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
    return true;
}

static bool ggml_cl_mm_q4_0_f32(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_LOCALS(int, ne0, src0, ne);
    GGML_TENSOR_LOCALS(int, ne1, src1, ne);
    GGML_TENSOR_LOCALS(int, ne,  dst,  ne);

    if (ne11 < 32) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
        return false;
    }

    ggml_backend_opencl_context * backend_ctx = (ggml_backend_opencl_context *) backend->context;

    const ggml_tensor * soa0_src = src0->view_src != nullptr ? src0->view_src : src0;
    ggml_tensor_extra_cl_q4_0 * extra0 = (ggml_tensor_extra_cl_q4_0 *) soa0_src->extra;
    ggml_tensor_extra_cl *      extra1 = (ggml_tensor_extra_cl *) src1->extra;
    ggml_tensor_extra_cl *      extrad = (ggml_tensor_extra_cl *) dst->extra;

    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    int r2 = ne12/ne02;
    int r3 = ne13/ne03;

    cl_kernel kernel = backend_ctx->mul_mat.kernel_mul_mm_q4_0_f32_l4_lm;
    const int nth0 = 128; // calculated as (BM*BN)/(TM*TN)

    int batch_stride_a = ne00*ne01;
    int batch_stride_b = ne10*ne11;
    int batch_stride_d = ne0*ne1;

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->q));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0->d));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne02));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne11));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne12));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne10)); // stride_a
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne10)); // stride_b
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne01)); // stride_d
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &batch_stride_a));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &batch_stride_b));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &batch_stride_d));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &r2));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &r3));

    // 64 is block tile size BM and BN - change here when BM and BN in the kernel are changed.
    size_t global_work_size[] = {(size_t)(CEIL_DIV(ne01, 64)*nth0), (size_t)(CEIL_DIV(ne11, 64)), (size_t)ne12*ne13};
    size_t local_work_size[] = {(size_t)nth0, 1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
    return true;
}

static bool ggml_cl_mm_q4_1_f32(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_LOCALS(int, ne0, src0, ne);
    GGML_TENSOR_LOCALS(int, ne1, src1, ne);
    GGML_TENSOR_LOCALS(int, ne,  dst,  ne);

    if (ne11 < 32) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
        return false;
    }

    ggml_backend_opencl_context * backend_ctx = (ggml_backend_opencl_context *) backend->context;

    const ggml_tensor * soa0_src = src0->view_src != nullptr ? src0->view_src : src0;
    ggml_tensor_extra_cl_q4_1 * extra0 = (ggml_tensor_extra_cl_q4_1 *) soa0_src->extra;
    ggml_tensor_extra_cl *      extra1 = (ggml_tensor_extra_cl *) src1->extra;
    ggml_tensor_extra_cl *      extrad = (ggml_tensor_extra_cl *) dst->extra;

    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    int r2 = ne12/ne02;
    int r3 = ne13/ne03;

    cl_kernel kernel = backend_ctx->mul_mat.kernel_mul_mm_q4_1_f32_l4_lm;
    const int nth0 = 128; // calculated as (BM*BN)/(TM*TN)

    int batch_stride_a = ne00*ne01;
    int batch_stride_b = ne10*ne11;
    int batch_stride_d = ne0*ne1;

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->q));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0->d));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra0->m));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne02));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne11));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne12));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne10)); // stride_a
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne10)); // stride_b
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &ne01)); // stride_d
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &batch_stride_a));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &batch_stride_b));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &batch_stride_d));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &r2));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(int),      &r3));

    // 64 is block tile size BM and BN - change here when BM and BN in the kernel are changed.
    size_t global_work_size[] = {(size_t)(CEIL_DIV(ne01, 64)*nth0), (size_t)(CEIL_DIV(ne11, 64)), (size_t)ne12*ne13};
    size_t local_work_size[] = {(size_t)nth0, 1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
    return true;
}

static bool ggml_cl_mm_q5_0_f32(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_LOCALS(int, ne0, src0, ne);
    GGML_TENSOR_LOCALS(int, ne1, src1, ne);
    GGML_TENSOR_LOCALS(int, ne,  dst,  ne);

    if (ne11 < 32) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
        return false;
    }

    ggml_backend_opencl_context * backend_ctx = (ggml_backend_opencl_context *) backend->context;

    const ggml_tensor * soa0_src = src0->view_src != nullptr ? src0->view_src : src0;
    ggml_tensor_extra_cl_q5_0 * extra0 = (ggml_tensor_extra_cl_q5_0 *) soa0_src->extra;
    ggml_tensor_extra_cl *      extra1 = (ggml_tensor_extra_cl *) src1->extra;
    ggml_tensor_extra_cl *      extrad = (ggml_tensor_extra_cl *) dst->extra;

    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    int r2 = ne12/ne02;
    int r3 = ne13/ne03;

    cl_kernel kernel = backend_ctx->mul_mat.kernel_mul_mm_q5_0_f32_l4_lm;
    const int nth0 = 128; // calculated as (BM*BN)/(TM*TN)

    int batch_stride_a = ne00*ne01;
    int batch_stride_b = ne10*ne11;
    int batch_stride_d = ne0*ne1;

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->qs));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0->qh));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra0->d));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne02));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne11));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne12));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne10)); // stride_a
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne10)); // stride_b
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &ne01)); // stride_d
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &batch_stride_a));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &batch_stride_b));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &batch_stride_d));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &r2));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(int),      &r3));

    // 64 is block tile size BM and BN - change here when BM and BN in the kernel are changed.
    size_t global_work_size[] = {(size_t)(CEIL_DIV(ne01, 64)*nth0), (size_t)(CEIL_DIV(ne11, 64)), (size_t)ne12*ne13};
    size_t local_work_size[] = {(size_t)nth0, 1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
    return true;
}

static bool ggml_cl_mm_q5_1_f32(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_LOCALS(int, ne0, src0, ne);
    GGML_TENSOR_LOCALS(int, ne1, src1, ne);
    GGML_TENSOR_LOCALS(int, ne,  dst,  ne);

    if (ne11 < 32) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
        return false;
    }

    ggml_backend_opencl_context * backend_ctx = (ggml_backend_opencl_context *) backend->context;

    const ggml_tensor * soa0_src = src0->view_src != nullptr ? src0->view_src : src0;
    ggml_tensor_extra_cl_q5_1 * extra0 = (ggml_tensor_extra_cl_q5_1 *) soa0_src->extra;
    ggml_tensor_extra_cl *      extra1 = (ggml_tensor_extra_cl *) src1->extra;
    ggml_tensor_extra_cl *      extrad = (ggml_tensor_extra_cl *) dst->extra;

    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    int r2 = ne12/ne02;
    int r3 = ne13/ne03;

    cl_kernel kernel = backend_ctx->mul_mat.kernel_mul_mm_q5_1_f32_l4_lm;
    const int nth0 = 128; // calculated as (BM*BN)/(TM*TN)

    int batch_stride_a = ne00*ne01;
    int batch_stride_b = ne10*ne11;
    int batch_stride_d = ne0*ne1;

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->qs));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0->qh));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra0->d));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_mem),   &extra0->m));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne02));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne11));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne12));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne10)); // stride_a
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &ne10)); // stride_b
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne01)); // stride_d
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &batch_stride_a));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &batch_stride_b));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &batch_stride_d));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(int),      &r2));
    CL_CHECK(clSetKernelArg(kernel, 20, sizeof(int),      &r3));

    // 64 is block tile size BM and BN - change here when BM and BN in the kernel are changed.
    size_t global_work_size[] = {(size_t)(CEIL_DIV(ne01, 64)*nth0), (size_t)(CEIL_DIV(ne11, 64)), (size_t)ne12*ne13};
    size_t local_work_size[] = {(size_t)nth0, 1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
    return true;
}

static bool ggml_cl_mm_q8_0_f32(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_LOCALS(int, ne0, src0, ne);
    GGML_TENSOR_LOCALS(int, ne1, src1, ne);
    GGML_TENSOR_LOCALS(int, ne,  dst,  ne);

    if (ne11 < 32) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
        return false;
    }

    ggml_backend_opencl_context * backend_ctx = (ggml_backend_opencl_context *) backend->context;

    const ggml_tensor * soa0_src = src0->view_src != nullptr ? src0->view_src : src0;
    ggml_tensor_extra_cl_q8_0 * extra0 = (ggml_tensor_extra_cl_q8_0 *) soa0_src->extra;
    ggml_tensor_extra_cl *      extra1 = (ggml_tensor_extra_cl *) src1->extra;
    ggml_tensor_extra_cl *      extrad = (ggml_tensor_extra_cl *) dst->extra;

    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    int r2 = ne12/ne02;
    int r3 = ne13/ne03;

    cl_kernel kernel = backend_ctx->mul_mat.kernel_mul_mm_q8_0_f32_l4_lm;
    const int nth0 = 128; // calculated as (BM*BN)/(TM*TN)

    int batch_stride_a = ne00*ne01;
    int batch_stride_b = ne10*ne11;
    int batch_stride_d = ne0*ne1;

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->q));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0->d));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne02));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne11));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne12));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne10)); // stride_a
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne10)); // stride_b
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne01)); // stride_d
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &batch_stride_a));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &batch_stride_b));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &batch_stride_d));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &r2));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &r3));

    // 64 is block tile size BM and BN - change here when BM and BN in the kernel are changed.
    size_t global_work_size[] = {(size_t)(CEIL_DIV(ne01, 64)*nth0), (size_t)(CEIL_DIV(ne11, 64)), (size_t)ne12*ne13};
    size_t local_work_size[] = {(size_t)nth0, 1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
    return true;
}

static bool ggml_cl_mm_iq4_nl_f32(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_LOCALS(int, ne0, src0, ne);
    GGML_TENSOR_LOCALS(int, ne1, src1, ne);
    GGML_TENSOR_LOCALS(int, ne,  dst,  ne);

    if (ne11 < 32) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
        return false;
    }

    ggml_backend_opencl_context * backend_ctx = (ggml_backend_opencl_context *) backend->context;

    const ggml_tensor * soa0_src = src0->view_src != nullptr ? src0->view_src : src0;
    ggml_tensor_extra_cl_iq4_nl * extra0 = (ggml_tensor_extra_cl_iq4_nl *) soa0_src->extra;
    ggml_tensor_extra_cl *        extra1 = (ggml_tensor_extra_cl *) src1->extra;
    ggml_tensor_extra_cl *        extrad = (ggml_tensor_extra_cl *) dst->extra;

    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    int r2 = ne12/ne02;
    int r3 = ne13/ne03;

    cl_kernel kernel = backend_ctx->mul_mat.kernel_mul_mm_iq4_nl_f32_l4_lm;
    const int nth0 = 128; // calculated as (BM*BN)/(TM*TN)

    int batch_stride_a = ne00*ne01;
    int batch_stride_b = ne10*ne11;
    int batch_stride_d = ne0*ne1;

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->q));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0->d));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne02));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne11));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne12));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne10)); // stride_a
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne10)); // stride_b
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne01)); // stride_d
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &batch_stride_a));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &batch_stride_b));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &batch_stride_d));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &r2));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &r3));

    // 64 is block tile size BM and BN - change here when BM and BN in the kernel are changed.
    size_t global_work_size[] = {(size_t)(CEIL_DIV(ne01, 64)*nth0), (size_t)(CEIL_DIV(ne11, 64)), (size_t)ne12*ne13};
    size_t local_work_size[] = {(size_t)nth0, 1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
    return true;
}

static bool ggml_cl_mm_q4_k_f32(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_LOCALS(int, ne0, src0, ne);
    GGML_TENSOR_LOCALS(int, ne1, src1, ne);
    GGML_TENSOR_LOCALS(int, ne,  dst,  ne);

    if (ne11 < 32) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
        return false;
    }

    ggml_backend_opencl_context * backend_ctx = (ggml_backend_opencl_context *) backend->context;

    const ggml_tensor * soa0_src = src0->view_src != nullptr ? src0->view_src : src0;
    ggml_tensor_extra_cl_q4_K * extra0 = (ggml_tensor_extra_cl_q4_K *) soa0_src->extra;
    ggml_tensor_extra_cl *      extra1 = (ggml_tensor_extra_cl *) src1->extra;
    ggml_tensor_extra_cl *      extrad = (ggml_tensor_extra_cl *) dst->extra;

    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    int r2 = ne12/ne02;
    int r3 = ne13/ne03;

    cl_kernel kernel = backend_ctx->mul_mat.kernel_mul_mm_q4_k_f32_l4_lm;
    // (BM*BN)/(TM*TN): Intel uses an 8x8 microtile (WG=64), others 4x8 (WG=128)
    const int nth0 = (backend_ctx->gpu_family == INTEL) ? 64 : 128;

    int batch_stride_a = ne00*ne01;
    int batch_stride_b = ne10*ne11;
    int batch_stride_d = ne0*ne1;

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->q));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0->s));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra0->d));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_mem),   &extra0->dm));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne02));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne11));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne12));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne10)); // stride_a
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &ne10)); // stride_b
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne01)); // stride_d
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &batch_stride_a));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &batch_stride_b));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &batch_stride_d));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(int),      &r2));
    CL_CHECK(clSetKernelArg(kernel, 20, sizeof(int),      &r3));

    // 64 is block tile size BM and BN - change here when BM and BN in the kernel are changed.
    size_t global_work_size[] = {(size_t)(CEIL_DIV(ne01, 64)*nth0), (size_t)(CEIL_DIV(ne11, 64)), (size_t)ne12*ne13};
    size_t local_work_size[] = {(size_t)nth0, 1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
    return true;
}

static bool ggml_cl_mm_q5_k_f32(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_LOCALS(int, ne0, src0, ne);
    GGML_TENSOR_LOCALS(int, ne1, src1, ne);
    GGML_TENSOR_LOCALS(int, ne,  dst,  ne);

    if (ne11 < 32) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
        return false;
    }

    ggml_backend_opencl_context * backend_ctx = (ggml_backend_opencl_context *) backend->context;

    const ggml_tensor * soa0_src = src0->view_src != nullptr ? src0->view_src : src0;
    ggml_tensor_extra_cl_q5_K * extra0 = (ggml_tensor_extra_cl_q5_K *) soa0_src->extra;
    ggml_tensor_extra_cl *      extra1 = (ggml_tensor_extra_cl *) src1->extra;
    ggml_tensor_extra_cl *      extrad = (ggml_tensor_extra_cl *) dst->extra;

    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    int r2 = ne12/ne02;
    int r3 = ne13/ne03;

    cl_kernel kernel = backend_ctx->mul_mat.kernel_mul_mm_q5_k_f32_l4_lm;
    const int nth0 = (backend_ctx->gpu_family == INTEL) ? 64 : 128; // Intel 8x8 microtile

    int batch_stride_a = ne00*ne01;
    int batch_stride_b = ne10*ne11;
    int batch_stride_d = ne0*ne1;

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->q));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0->qh));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra0->s));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_mem),   &extra0->d));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extra0->dm));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne02));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne11));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne12));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &ne10)); // stride_a
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne10)); // stride_b
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &ne01)); // stride_d
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &batch_stride_a));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &batch_stride_b));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(int),      &batch_stride_d));
    CL_CHECK(clSetKernelArg(kernel, 20, sizeof(int),      &r2));
    CL_CHECK(clSetKernelArg(kernel, 21, sizeof(int),      &r3));

    // 64 is block tile size BM and BN - change here when BM and BN in the kernel are changed.
    size_t global_work_size[] = {(size_t)(CEIL_DIV(ne01, 64)*nth0), (size_t)(CEIL_DIV(ne11, 64)), (size_t)ne12*ne13};
    size_t local_work_size[] = {(size_t)nth0, 1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
    return true;
}

static bool ggml_cl_mm_q6_k_f32(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_LOCALS(int, ne0, src0, ne);
    GGML_TENSOR_LOCALS(int, ne1, src1, ne);
    GGML_TENSOR_LOCALS(int, ne,  dst,  ne);

    if (ne11 < 32) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
        return false;
    }

    ggml_backend_opencl_context * backend_ctx = (ggml_backend_opencl_context *) backend->context;

    const ggml_tensor * soa0_src = src0->view_src != nullptr ? src0->view_src : src0;
    ggml_tensor_extra_cl_q6_K * extra0 = (ggml_tensor_extra_cl_q6_K *) soa0_src->extra;
    ggml_tensor_extra_cl *      extra1 = (ggml_tensor_extra_cl *) src1->extra;
    ggml_tensor_extra_cl *      extrad = (ggml_tensor_extra_cl *) dst->extra;

    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    int r2 = ne12/ne02;
    int r3 = ne13/ne03;

    cl_kernel kernel = backend_ctx->mul_mat.kernel_mul_mm_q6_k_f32_l4_lm;
    const int nth0 = 128; // calculated as (BM*BN)/(TM*TN)

    int batch_stride_a = ne00*ne01;
    int batch_stride_b = ne10*ne11;
    int batch_stride_d = ne0*ne1;

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->ql));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0->qh));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra0->s));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_mem),   &extra0->d));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne02));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne11));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne12));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne10)); // stride_a
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &ne10)); // stride_b
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne01)); // stride_d
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &batch_stride_a));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &batch_stride_b));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &batch_stride_d));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(int),      &r2));
    CL_CHECK(clSetKernelArg(kernel, 20, sizeof(int),      &r3));

    // 64 is block tile size BM and BN - change here when BM and BN in the kernel are changed.
    size_t global_work_size[] = {(size_t)(CEIL_DIV(ne01, 64)*nth0), (size_t)(CEIL_DIV(ne11, 64)), (size_t)ne12*ne13};
    size_t local_work_size[] = {(size_t)nth0, 1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
    return true;
}

static void ggml_cl_mv_f32_f32(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_backend_opencl_context * backend_ctx = (ggml_backend_opencl_context *) backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *) src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *) src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *) dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    GGML_TENSOR_LOCALS(int,      ne0, src0, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb0, src0, nb);
    GGML_TENSOR_LOCALS(int,      ne1, src1, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb1, src1, nb);
    GGML_TENSOR_LOCALS(int,      ne,  dst,  ne);

    int r2 = ne12/ne02;
    int r3 = ne13/ne03;

    int nth0 = 32;
    int nth1 = 1;
    int nrows = 1;
    cl_kernel kernel;

    //GGML_ASSERT(ne02 == ne12);
    kernel = backend_ctx->mul_mat.kernel_mul_mat_f32_f32;
    nrows = 4;

    if (backend_ctx->gpu_family == INTEL) {
        nth0 = 32;
        nth1 = 1;
    } else if (backend_ctx->gpu_family == ADRENO) {
        nth0 = 64;
        nth1 = 1;
    } else {
        GGML_ASSERT(false && "TODO: Unknown GPU");
    }

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne02));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb00));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb02));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb03));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne10));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &ne11));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne12));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nb10));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong), &nb11));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong), &nb12));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &nb13));
    CL_CHECK(clSetKernelArg(kernel, 20, sizeof(int),      &ne0));
    CL_CHECK(clSetKernelArg(kernel, 21, sizeof(int),      &ne1));
    CL_CHECK(clSetKernelArg(kernel, 22, sizeof(int),      &r2));
    CL_CHECK(clSetKernelArg(kernel, 23, sizeof(int),      &r3));

    int64_t ny = (ne11 + nrows - 1)/nrows;

    size_t global_work_size[] = {(size_t)ne01*nth0, (size_t)ny*nth1, (size_t)ne12*ne13};
    size_t local_work_size[] = {(size_t)nth0, (size_t)nth1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

static void ggml_cl_mv_f16_f32(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_backend_opencl_context * backend_ctx = (ggml_backend_opencl_context *) backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *) src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *) src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *) dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    GGML_TENSOR_LOCALS(int,      ne0, src0, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb0, src0, nb);
    GGML_TENSOR_LOCALS(int,      ne1, src1, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb1, src1, nb);
    GGML_TENSOR_LOCALS(int,      ne,  dst,  ne);

    int r2 = ne12/ne02;
    int r3 = ne13/ne03;

    int nth0 = 32;
    int nth1 = 1;
    int nrows = 1;
    cl_kernel kernel;
    bool use_f16_mrow = false;

    //GGML_ASSERT(ne02 == ne12);
    if (backend_ctx->gpu_family == INTEL) {
        nth0 = 32;
        nth1 = 1;
    } else if (backend_ctx->gpu_family == ADRENO) {
        nth0 = 64;
        nth1 = 1;
    } else {
        GGML_ASSERT(false && "TODO: Unknown GPU");
    }

    // heuristic for packing more work for Adreno
    const bool adreno_use_lane_split =
        backend_ctx->gpu_family == ADRENO &&
        ne11 == 1 &&
        ne01 >= 8 &&
        ne00 % 4 == 0 &&
        r3 == 1 && r2 >= 1 && r2 <= 8 &&
        (ne12 % r2) == 0;

    if (ne11 * ne12 < 4) {
        // Decode (single token): the legacy _1row runs one 64-lane
        // subgroup per WG (one output row), under-utilizing BW. Route the
        // wide f16 weight matmuls (attn proj + lm_head) to the multi-row
        // variant: MROW rows per WG -> more loads in flight + activation
        // staged once in __local. ne00<=8192 bounds the LDS. The mrow WG
        // is 64 x MROW = 1024 work-items (> Intel's 512 max) and reduces
        // within a 64-wide subgroup, so skip on Intel.
        if (backend_ctx->f16_mrow && backend_ctx->gpu_family != INTEL &&
            backend_ctx->mul_mat.kernel_mul_mat_f16_f32_mrow != nullptr &&
            ne00 >= 128 && ne01 >= 8 && ne00 % 4 == 0 && ne00 <= 8192) {
            // The register-blocked / half8 variants cast the src0 row pointer to
            // half4 / half8 (8- and 16-byte loads) with no scalar fallback inside
            // the kernel. ne00 % 4 == 0 constrains the element count per row, NOT
            // the byte stride between rows: a permuted or strided src0 (or a view
            // at an odd offset) can leave nb01/nb02/nb03 unaligned. Only take them
            // when every row this dispatch touches is aligned; the base mrow kernel
            // re-checks per row and falls back to its scalar loop.
            const cl_ulong row_addr_bits = offset0 | nb01 | nb02 | nb03;
            const bool aligned8  = (row_addr_bits & 7)  == 0;
            const bool aligned16 = (row_addr_bits & 15) == 0;

            // Register-blocked variants: each subgroup does RPT rows (more
            // weight loads in flight per lane). 8/16 use half8 (128-bit)
            // loads, gated on ne00 % 8 == 0.
            const int rpt = backend_ctx->f16_mrow_rpt;
            if (rpt == 16 && ne00 % 8 == 0 && aligned16 && backend_ctx->mul_mat.kernel_mul_mat_f16_f32_mrow_h8r2 != nullptr) {
                kernel = backend_ctx->mul_mat.kernel_mul_mat_f16_f32_mrow_h8r2;
            } else if (rpt == 8 && ne00 % 8 == 0 && aligned16 && backend_ctx->mul_mat.kernel_mul_mat_f16_f32_mrow_h8 != nullptr) {
                kernel = backend_ctx->mul_mat.kernel_mul_mat_f16_f32_mrow_h8;
            } else if (rpt == 4 && aligned8 && backend_ctx->mul_mat.kernel_mul_mat_f16_f32_mrow_r4 != nullptr) {
                kernel = backend_ctx->mul_mat.kernel_mul_mat_f16_f32_mrow_r4;
            } else if (rpt == 2 && aligned8 && backend_ctx->mul_mat.kernel_mul_mat_f16_f32_mrow_r2 != nullptr) {
                kernel = backend_ctx->mul_mat.kernel_mul_mat_f16_f32_mrow_r2;
            } else {
                kernel = backend_ctx->mul_mat.kernel_mul_mat_f16_f32_mrow;
            }
            use_f16_mrow = true;
        } else {
            kernel = backend_ctx->mul_mat.kernel_mul_mat_f16_f32_1row;
        }
    } else if (adreno_use_lane_split && ne00 >= 64 && ne00 <= 128) {
        kernel = backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_dr_lq;
        nrows  = 1;
    } else if (adreno_use_lane_split && r2 >= 2 && ne00 > 128 && ne00 <= 256) {
        kernel = backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_dr_ls;
        nrows  = 1;
    } else if (ne00 >= 128 && ne01 >= 8 && ne00%4 == 0) {
        // multi-output decode variants when Q is a single row
        static const char * mm_force_l4_env = getenv("GGML_OPENCL_MM_F16_FORCE_L4");
        static const bool mm_force_l4_on = (mm_force_l4_env != nullptr && mm_force_l4_env[0] != '0');
        const bool can_multi_out = !mm_force_l4_on && ne11 == 1 && ne01 >= 64 && ne01 % 8 == 0;
        // paired-K-row variant that doubles per-wave-cycle
        static const char * mm_kq_pair_env = getenv("GGML_OPENCL_MM_KQ_PAIR");
        static const bool mm_kq_pair_on = (mm_kq_pair_env != nullptr && mm_kq_pair_env[0] != '0');
        // GQA-coalesced variant that reads each K-row once and
        // emits gqa_ratio outputs
        static const char * mm_kq_gqa_env = getenv("GGML_OPENCL_MM_KQ_GQA");
        static const bool mm_kq_gqa_on = (mm_kq_gqa_env != nullptr && mm_kq_gqa_env[0] != '0');
        // GQA-coalesced KQV variant (DK=128/r2=8/r3=1) that reads
        // each V slab once per K-head and emits all r2 Q-heads
        static const char * mm_kqv_gqa_env = getenv("GGML_OPENCL_MM_KQV_GQA");
        static const bool mm_kqv_gqa_on = (mm_kqv_gqa_env != nullptr && mm_kqv_gqa_env[0] != '0');
        if (can_multi_out && (ne01 % 16) == 0 && ne00 == 128 && r2 == 8 && r3 == 1 && mm_kq_gqa_on &&
            backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_x8_gqa4 != nullptr) {
            kernel = backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_x8_gqa4;
            nrows = 1;
        } else if (can_multi_out && ne00 <= 256 && mm_kq_pair_on &&
            backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_x8_pair != nullptr) {
            kernel = backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_x8_pair;
            nrows = 1;
        } else if (can_multi_out && ne00 <= 256 &&
            backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_x8 != nullptr) {
            kernel = backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_x8;
            nrows = 1;
        } else if (can_multi_out && ne01 == 128 && r2 == 8 && r3 == 1 && mm_kqv_gqa_on &&
            backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_y8_gqa != nullptr) {
            kernel = backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_y8_gqa;
            nrows = 1;
        } else if (can_multi_out &&
            backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_y8 != nullptr) {
            kernel = backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_y8;
            nrows = 1;
        } else if (ne11 == 1) {
            // Decode shapes that don't satisfy the x8/y8 row
            // constraints (ne01 < 64 or ne01 % 8 != 0) fall back to
            // upstream's 4-output _dr kernel.
            kernel = backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_dr;
            nrows  = 1; // not used by this kernel
        } else {
            kernel = backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4;
            nrows = ne11;
        }
    } else {
        kernel = backend_ctx->mul_mat.kernel_mul_mat_f16_f32;
        nrows = 4;
    }

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne02));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb00));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb02));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb03));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne10));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &ne11));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne12));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nb10));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong), &nb11));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong), &nb12));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &nb13));
    CL_CHECK(clSetKernelArg(kernel, 20, sizeof(int),      &ne0));
    CL_CHECK(clSetKernelArg(kernel, 21, sizeof(int),      &ne1));
    CL_CHECK(clSetKernelArg(kernel, 22, sizeof(int),      &r2));
    CL_CHECK(clSetKernelArg(kernel, 23, sizeof(int),      &r3));
    if (use_f16_mrow) {
        const int MROW = 16; // must match MROW in mul_mv_f16_f32_mrow.cl
        // rows-per-subgroup multiplier for the selected variant:
        //   1/2/4 -> half4 register blocking; 8 -> half8(1 row); 16 -> half8(2 rows)
        const int rpt = backend_ctx->f16_mrow_rpt;
        int rmul;
        if (rpt == 16) {
            rmul = (ne00 % 8 == 0) ? 2 : 1;
        } else if (rpt == 8) {
            rmul = 1;
        } else {
            rmul = rpt; // 1,2,4
        }

        const int rows_per_wg = MROW * rmul;
        // __local activation buffer: ne00 floats, rounded up for float4 access
        CL_CHECK(clSetKernelArg(kernel, 24, sizeof(float) * ((ne00 + 3) / 4 * 4), nullptr));
        size_t mrow_global[] = { (size_t)((ne01 + rows_per_wg - 1) / rows_per_wg) * 64, (size_t)ne11 * MROW, (size_t)ne12 * ne13 };
        size_t mrow_local[]  = { 64, (size_t)MROW, 1 };
        backend_ctx->enqueue_ndrange_kernel(kernel, 3, mrow_global, mrow_local, dst);
        return;
    }

    if (kernel == backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_x8 ||
               kernel == backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_x8_pair ||
               kernel == backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_y8) {
        // multi-output decode variants: each WG processes 8 outputs along ne01, ne11 == 1
        const int64_t n_wg_x = ne01 / 8;
        size_t global_work_size[] = {(size_t)n_wg_x*nth0, (size_t)nth1, (size_t)ne12*ne13};
        size_t local_work_size[]  = {(size_t)nth0, (size_t)nth1, 1};
        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
    } else if (kernel == backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_x8_gqa4) {
        // GQA-coalesced KQ: one WG per K-head emits N_K_ROWS_GQA=16 K-rows * r2 Q-heads
        const int64_t n_wg_x = ne01 / 16;
        size_t global_work_size[] = {(size_t)n_wg_x*nth0, (size_t)nth1, (size_t)ne02*ne13};
        size_t local_work_size[]  = {(size_t)nth0, (size_t)nth1, 1};
        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
    } else if (kernel == backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_y8_gqa) {
        // GQA-coalesced KQV: one WG per K-head emits 8 DV-rows * r2 Q-heads
        const int64_t n_wg_x = ne01 / 8;
        size_t global_work_size[] = {(size_t)n_wg_x*nth0, (size_t)nth1, (size_t)ne02*ne13};
        size_t local_work_size[]  = {(size_t)nth0, (size_t)nth1, 1};
        backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
    } else {
        if (kernel == backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_dr) {
            const int NDST_DR = 4;
            size_t global_work_size[] = {(size_t)CEIL_DIV(ne01, NDST_DR)*nth0, (size_t)nth1, (size_t)ne12*ne13};
            size_t local_work_size[]  = {(size_t)nth0, (size_t)nth1, 1};

            backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
        } else if (kernel == backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_dr_ls) {
            size_t global_work_size[] = {(size_t)CEIL_DIV(ne01, 2)*nth0, (size_t)nth1, (size_t)ne02*ne03};
            size_t local_work_size[]  = {(size_t)nth0, (size_t)nth1, 1};

            backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
        } else if (kernel == backend_ctx->mul_mat.kernel_mul_mat_f16_f32_l4_dr_lq) {
            size_t global_work_size[] = {(size_t)CEIL_DIV(ne01, 4)*nth0, (size_t)nth1, (size_t)ne02*ne03};
            size_t local_work_size[]  = {(size_t)nth0, (size_t)nth1, 1};

            backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
        } else {
            int64_t ny = (ne11 + nrows - 1)/nrows;

            size_t global_work_size[] = {(size_t)ne01*nth0, (size_t)ny*nth1, (size_t)ne12*ne13};
            size_t local_work_size[] = {(size_t)nth0, (size_t)nth1, 1};

            backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
        }
    }

}

static void ggml_cl_mv_f16_f16(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_backend_opencl_context * backend_ctx = (ggml_backend_opencl_context *) backend->context;

    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *) src0->extra;
    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *) src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *) dst->extra;

    cl_ulong offset0 = extra0->offset + src0->view_offs;
    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

    GGML_TENSOR_LOCALS(int,      ne0, src0, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb0, src0, nb);
    GGML_TENSOR_LOCALS(int,      ne1, src1, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb1, src1, nb);
    GGML_TENSOR_LOCALS(int,      ne,  dst,  ne);

    int r2 = ne12/ne02;
    int r3 = ne13/ne03;

    int nth0 = 32;
    int nth1 = 1;
    int nrows = 4;
    cl_kernel kernel;

    //GGML_ASSERT(ne02 == ne12);
    if (backend_ctx->gpu_family == INTEL) {
        nth0 = 32;
        nth1 = 1;
    } else if (backend_ctx->gpu_family == ADRENO) {
        nth0 = 64;
        nth1 = 1;
    } else {
        GGML_ASSERT(false && "TODO: Unknown GPU");
    }

    kernel = backend_ctx->mul_mat.kernel_mul_mat_f16_f16;

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne02));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb00));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb02));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb03));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne10));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &ne11));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne12));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nb10));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong), &nb11));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_ulong), &nb12));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_ulong), &nb13));
    CL_CHECK(clSetKernelArg(kernel, 20, sizeof(int),      &ne0));
    CL_CHECK(clSetKernelArg(kernel, 21, sizeof(int),      &ne1));
    CL_CHECK(clSetKernelArg(kernel, 22, sizeof(int),      &r2));
    CL_CHECK(clSetKernelArg(kernel, 23, sizeof(int),      &r3));

    int64_t ny = (ne11 + nrows - 1)/nrows;

    size_t global_work_size[] = {(size_t)ne01*nth0, (size_t)ny*nth1, (size_t)ne12*ne13};
    size_t local_work_size[] = {(size_t)nth0, (size_t)nth1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

static void ggml_cl_mv_q1_0_f32(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_backend_opencl_context * backend_ctx = (ggml_backend_opencl_context *) backend->context;

#ifndef GGML_OPENCL_SOA_Q
    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *) src0->extra;
    cl_ulong offset0 = extra0->offset + src0->view_offs;
#endif

    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *) src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *) dst->extra;

    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

#ifdef GGML_OPENCL_SOA_Q
    ggml_tensor_extra_cl_q1_0 * extra0_q1_0 = (ggml_tensor_extra_cl_q1_0 *) src0->extra;
#endif

    GGML_TENSOR_LOCALS(int,      ne0, src0, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb0, src0, nb);
    GGML_TENSOR_LOCALS(int,      ne1, src1, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb1, src1, nb);
    GGML_TENSOR_LOCALS(int,      ne,  dst,  ne);

    int r2 = ne12/ne02;
    int r3 = ne13/ne03;

    int nth0 = 32;
    int nth1 = 1;
    int ndst = 4;
    cl_kernel kernel;

#ifdef GGML_OPENCL_SOA_Q
    kernel = backend_ctx->mul_mat.kernel_mul_mv_q1_0_f32_flat;

    // nth0 - subgroup size
    // nth1 - number of subgroups per workgroup
    // ndst - number of output values per workgroup = output per subgroup * number of subgroups
    if (backend_ctx->gpu_family == INTEL) {
        nth0 = 16;
        nth1 = 2;
        ndst = nth1*4;
    } else if (backend_ctx->gpu_family == ADRENO) {
        nth0 = 64;
        nth1 = 2;
        ndst = nth1*4;
    } else {
        GGML_ASSERT(false && "TODO: Unknown GPU");
    }

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0_q1_0->q));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0_q1_0->d));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb02));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb03));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne12));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb11));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &nb12));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &nb13));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne0));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &ne1));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &r2));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &r3));
#else
    kernel = backend_ctx->mul_mat.kernel_mul_mv_q1_0_f32;

    if (backend_ctx->gpu_family == INTEL) {
        nth0 = 16;
        nth1 = 2;
        ndst = nth1*4;
    } else if (backend_ctx->gpu_family == ADRENO) {
        nth0 = 64;
        nth1 = 2;
        ndst = nth1*4;
    } else {
        GGML_ASSERT(false && "TODO: Unknown GPU");
    }

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb02));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb03));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne12));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb11));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &nb12));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &nb13));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne0));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &ne1));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &r2));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &r3));
#endif // GGML_OPENCL_SOA_Q

    // Each subgroup produces N_DST values in the result.
    size_t global_work_size[] = {(size_t)(ne01 + ndst-1)/ndst*nth0, (size_t)ne11*nth1, (size_t)ne12*ne13};
    size_t local_work_size[] = {(size_t)nth0, (size_t)nth1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

static void ggml_cl_mv_q4_0_f32(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_backend_opencl_context * backend_ctx = (ggml_backend_opencl_context *) backend->context;

#ifndef GGML_OPENCL_SOA_Q
    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *) src0->extra;
    cl_ulong offset0 = extra0->offset + src0->view_offs;
#endif

    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *) src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *) dst->extra;

    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

#ifdef GGML_OPENCL_SOA_Q
    const ggml_tensor * soa0_src = src0->view_src != nullptr ? src0->view_src : src0;
    ggml_tensor_extra_cl_q4_0 * extra0_q4_0 = (ggml_tensor_extra_cl_q4_0 *) soa0_src->extra;
#endif

    GGML_TENSOR_LOCALS(int,      ne0, src0, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb0, src0, nb);
    GGML_TENSOR_LOCALS(int,      ne1, src1, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb1, src1, nb);
    GGML_TENSOR_LOCALS(int,      ne,  dst,  ne);

    int r2 = ne12/ne02;
    int r3 = ne13/ne03;

    int nth0 = 32;
    int nth1 = 1;
    int ndst = 4;
    cl_kernel kernel;

    // This should have been satisfied.
    GGML_ASSERT(ne11 == ne1);
    GGML_ASSERT(ne01 == ne0);

#ifdef GGML_OPENCL_SOA_Q
    if (backend_ctx->gpu_family == INTEL) {
        nth0 = 16;
        nth1 = 1;

        kernel = backend_ctx->mul_mat.kernel_mul_mat_q4_0_f32_8x_flat;
        ndst = 8;
    } else if (backend_ctx->gpu_family == ADRENO) {
        nth0 = 64;
        nth1 = 1;

        kernel = backend_ctx->mul_mat.kernel_mul_mat_q4_0_f32_8x_flat;
        ndst =8;
    } else {
        GGML_ASSERT(false && "TODO: Unknown GPU");
    }

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0_q4_0->q));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0_q4_0->d));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
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
#else // GGML_OPENCL_SOA_Q
    if (backend_ctx->gpu_family == INTEL) {
        // Use 1D local size. Each workgroup is a SIMD group. Each SIMD
        // group produces N_DST (4 for Q4_0 kernel) values in the result.
        // The number of workgroups on dim 0 (the leading dimension) is
        // the nearest multiple of 4 that covers ne0 (equals ne01).
        nth0 = 16;
        nth1 = 1;

        kernel = backend_ctx->mul_mat.kernel_mul_mat_q4_0_f32;
        ndst = 4;
    } else if (backend_ctx->gpu_family == ADRENO) {
        nth0 = 64;
        nth1 = 1;

        kernel = backend_ctx->mul_mat.kernel_mul_mat_q4_0_f32_v;
        ndst = 4;
    } else {
        GGML_ASSERT(false && "TODO: Unknown GPU");
    }

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
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
#endif // GGML_OPENCL_SOA_Q

    // Each subgroup produces N_DST values in the result.
    size_t global_work_size[] = {(size_t)(ne01 + ndst-1)/ndst*nth0, (size_t)ne11*nth1, (size_t)ne12*ne13};
    size_t local_work_size[] = {(size_t)nth0, (size_t)nth1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

static void ggml_cl_mv_q4_1_f32(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_backend_opencl_context * backend_ctx = (ggml_backend_opencl_context *) backend->context;

#ifndef GGML_OPENCL_SOA_Q
    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *) src0->extra;
    cl_ulong offset0 = extra0->offset + src0->view_offs;
#endif

    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *) src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *) dst->extra;

    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

#ifdef GGML_OPENCL_SOA_Q
    const ggml_tensor * soa0_src = src0->view_src != nullptr ? src0->view_src : src0;
    ggml_tensor_extra_cl_q4_1 * extra0_q4_1 = (ggml_tensor_extra_cl_q4_1 *) soa0_src->extra;
#endif

    GGML_TENSOR_LOCALS(int,      ne0, src0, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb0, src0, nb);
    GGML_TENSOR_LOCALS(int,      ne1, src1, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb1, src1, nb);
    GGML_TENSOR_LOCALS(int,      ne,  dst,  ne);

    int r2 = ne12/ne02;
    int r3 = ne13/ne03;

    int nth0 = 32;
    int nth1 = 1;
    int ndst = 4;
    cl_kernel kernel;

#ifdef GGML_OPENCL_SOA_Q
    if (backend_ctx->gpu_family == INTEL) {
        nth0 = 16;
        nth1 = 1;
        ndst = 4;
    } else if (backend_ctx->gpu_family == ADRENO) {
        nth0 = 64;
        nth1 = 1;
        ndst = 4;
    } else {
        GGML_ASSERT(false && "TODO: Unknown GPU");
    }

    kernel = backend_ctx->mul_mat.kernel_mul_mv_q4_1_f32_flat;

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0_q4_1->q));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0_q4_1->d));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra0_q4_1->m));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne02));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne10));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne12));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne0));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne1));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &r2));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &r3));
#else
    if (backend_ctx->gpu_family == INTEL) {
        nth0 = 16;
        nth1 = 1;
        ndst = 4;
    } else if (backend_ctx->gpu_family == ADRENO) {
        nth0 = 64;
        nth1 = 1;
        ndst = 4;
    } else {
        GGML_ASSERT(false && "TODO: Unknown GPU");
    }

    kernel = backend_ctx->mul_mat.kernel_mul_mv_q4_1_f32;

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
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
#endif // GGML_OPENCL_SOA_Q

    // Each subgroup produces N_DST values in the result.
    size_t global_work_size[] = {(size_t)(ne01 + ndst-1)/ndst*nth0, (size_t)ne11*nth1, (size_t)ne12*ne13};
    size_t local_work_size[] = {(size_t)nth0, (size_t)nth1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

static void ggml_cl_mv_q5_0_f32(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_backend_opencl_context * backend_ctx = (ggml_backend_opencl_context *) backend->context;

#ifndef GGML_OPENCL_SOA_Q
    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *) src0->extra;
    cl_ulong offset0 = extra0->offset + src0->view_offs;
#endif

    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *) src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *) dst->extra;

    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

#ifdef GGML_OPENCL_SOA_Q
    const ggml_tensor * soa0_src = src0->view_src != nullptr ? src0->view_src : src0;
    ggml_tensor_extra_cl_q5_0 * extra0_q5_0 = (ggml_tensor_extra_cl_q5_0 *) soa0_src->extra;
#endif

    GGML_TENSOR_LOCALS(int,      ne0, src0, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb0, src0, nb);
    GGML_TENSOR_LOCALS(int,      ne1, src1, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb1, src1, nb);
    GGML_TENSOR_LOCALS(int,      ne,  dst,  ne);

    int r2 = ne12/ne02;
    int r3 = ne13/ne03;

    int nth0 = 32;
    int nth1 = 1;
    int ndst = 4;
    cl_kernel kernel;

#ifdef GGML_OPENCL_SOA_Q
    if (backend_ctx->gpu_family == INTEL) {
        nth0 = 16;
        nth1 = 1;
        ndst = 4;
    } else if (backend_ctx->gpu_family == ADRENO) {
        nth0 = 64;
        nth1 = 1;
        ndst = 4;
    } else {
        GGML_ASSERT(false && "TODO: Unknown GPU");
    }

    kernel = backend_ctx->mul_mat.kernel_mul_mv_q5_0_f32_flat;

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0_q5_0->qs));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0_q5_0->qh));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra0_q5_0->d));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne02));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne10));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne12));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne0));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne1));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &r2));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &r3));
#else
    if (backend_ctx->gpu_family == INTEL) {
        nth0 = 16;
        nth1 = 1;
        ndst = 4;
    } else if (backend_ctx->gpu_family == ADRENO) {
        nth0 = 64;
        nth1 = 1;
        ndst = 4;
    } else {
        GGML_ASSERT(false && "TODO: Unknown GPU");
    }

    kernel = backend_ctx->mul_mat.kernel_mul_mv_q5_0_f32;

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
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
#endif // GGML_OPENCL_SOA_Q

    // Each subgroup produces N_DST values in the result.
    size_t global_work_size[] = {(size_t)(ne01 + ndst-1)/ndst*nth0, (size_t)ne11*nth1, (size_t)ne12*ne13};
    size_t local_work_size[] = {(size_t)nth0, (size_t)nth1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

static void ggml_cl_mv_q5_1_f32(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_backend_opencl_context * backend_ctx = (ggml_backend_opencl_context *) backend->context;

#ifndef GGML_OPENCL_SOA_Q
    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *) src0->extra;
    cl_ulong offset0 = extra0->offset + src0->view_offs;
#endif

    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *) src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *) dst->extra;

    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

#ifdef GGML_OPENCL_SOA_Q
    const ggml_tensor * soa0_src = src0->view_src != nullptr ? src0->view_src : src0;
    ggml_tensor_extra_cl_q5_1 * extra0_q5_1 = (ggml_tensor_extra_cl_q5_1 *) soa0_src->extra;
#endif

    GGML_TENSOR_LOCALS(int,      ne0, src0, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb0, src0, nb);
    GGML_TENSOR_LOCALS(int,      ne1, src1, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb1, src1, nb);
    GGML_TENSOR_LOCALS(int,      ne,  dst,  ne);

    int r2 = ne12/ne02;
    int r3 = ne13/ne03;

    int nth0 = 32;
    int nth1 = 1;
    int ndst = 4;
    cl_kernel kernel;

#ifdef GGML_OPENCL_SOA_Q
    if (backend_ctx->gpu_family == INTEL) {
        nth0 = 16;
        nth1 = 1;
        ndst = 4;
    } else if (backend_ctx->gpu_family == ADRENO) {
        nth0 = 64;
        nth1 = 1;
        ndst = 4;
    } else {
        GGML_ASSERT(false && "TODO: Unknown GPU");
    }

    kernel = backend_ctx->mul_mat.kernel_mul_mv_q5_1_f32_flat;

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0_q5_1->qs));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0_q5_1->qh));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra0_q5_1->d));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_mem),   &extra0_q5_1->m));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne02));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne10));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne12));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne0));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &ne1));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &r2));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &r3));
#else
    if (backend_ctx->gpu_family == INTEL) {
        nth0 = 16;
        nth1 = 1;
        ndst = 4;
    } else if (backend_ctx->gpu_family == ADRENO) {
        nth0 = 64;
        nth1 = 1;
        ndst = 4;
    } else {
        GGML_ASSERT(false && "TODO: Unknown GPU");
    }

    kernel = backend_ctx->mul_mat.kernel_mul_mv_q5_1_f32;

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
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
#endif // GGML_OPENCL_SOA_Q

    // Each subgroup produces N_DST values in the result.
    size_t global_work_size[] = {(size_t)(ne01 + ndst-1)/ndst*nth0, (size_t)ne11*nth1, (size_t)ne12*ne13};
    size_t local_work_size[] = {(size_t)nth0, (size_t)nth1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

static void ggml_cl_mv_q8_0_f32(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_backend_opencl_context * backend_ctx = (ggml_backend_opencl_context *) backend->context;

#ifndef GGML_OPENCL_SOA_Q
    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *) src0->extra;
    cl_ulong offset0 = extra0->offset + src0->view_offs;
#endif

    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *) src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *) dst->extra;

    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

#ifdef GGML_OPENCL_SOA_Q
    const ggml_tensor * soa0_src = src0->view_src != nullptr ? src0->view_src : src0;
    ggml_tensor_extra_cl_q8_0 * extra0_q8_0 = (ggml_tensor_extra_cl_q8_0 *) soa0_src->extra;
#endif

    GGML_TENSOR_LOCALS(int,      ne0, src0, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb0, src0, nb);
    GGML_TENSOR_LOCALS(int,      ne1, src1, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb1, src1, nb);
    GGML_TENSOR_LOCALS(int,      ne,  dst,  ne);

    int r2 = ne12/ne02;
    int r3 = ne13/ne03;

    int nth0 = 32;
    int nth1 = 1;
    int ndst = 4;
    cl_kernel kernel;

#ifdef GGML_OPENCL_SOA_Q
    kernel = backend_ctx->mul_mat.kernel_mul_mv_q8_0_f32_flat;

    // nth0 - subgroup size
    // nth1 - number of subgroups per workgroup
    // ndst - number of output values per workgroup = output per subgroup * number of subgroups
    if (backend_ctx->gpu_family == INTEL) {
        nth0 = 16;
        nth1 = 2;
        ndst = nth1*4;
    } else if (backend_ctx->gpu_family == ADRENO) {
        nth0 = 64;
        nth1 = 2;
        ndst = nth1*4;
    } else {
        GGML_ASSERT(false && "TODO: Unknown GPU");
    }

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0_q8_0->q));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0_q8_0->d));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb02));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb03));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne12));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb11));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &nb12));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &nb13));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne0));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &ne1));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &r2));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &r3));
#else
    kernel = backend_ctx->mul_mat.kernel_mul_mv_q8_0_f32;

    // nth0 - subgroup size
    // nth1 - number of subgroups per workgroup
    // ndst - number of output values per workgroup = output per subgroup * number of subgroups
    if (backend_ctx->gpu_family == INTEL) {
        nth0 = 16;
        nth1 = 2;
        ndst = nth1*4;
    } else if (backend_ctx->gpu_family == ADRENO) {
        nth0 = 64;
        nth1 = 2;
        ndst = nth1*4;
    } else {
        GGML_ASSERT(false && "TODO: Unknown GPU");
    }

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb02));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb03));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne12));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb11));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &nb12));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &nb13));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne0));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &ne1));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &r2));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &r3));
#endif // GGML_OPENCL_SOA_Q

    // Each subgroup produces N_DST values in the result.
    size_t global_work_size[] = {(size_t)(ne01 + ndst-1)/ndst*nth0, (size_t)ne11*nth1, (size_t)ne12*ne13};
    size_t local_work_size[] = {(size_t)nth0, (size_t)nth1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

static void ggml_cl_mv_iq4_nl_f32(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_backend_opencl_context * backend_ctx = (ggml_backend_opencl_context *) backend->context;

#ifndef GGML_OPENCL_SOA_Q
    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *) src0->extra;
    cl_ulong offset0 = extra0->offset + src0->view_offs;
#endif

    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *) src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *) dst->extra;

    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

#ifdef GGML_OPENCL_SOA_Q
    const ggml_tensor * soa0_src = src0->view_src != nullptr ? src0->view_src : src0;
    ggml_tensor_extra_cl_iq4_nl * extra0_iq4_nl = (ggml_tensor_extra_cl_iq4_nl *) soa0_src->extra;
#endif

    GGML_TENSOR_LOCALS(int,      ne0, src0, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb0, src0, nb);
    GGML_TENSOR_LOCALS(int,      ne1, src1, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb1, src1, nb);
    GGML_TENSOR_LOCALS(int,      ne,  dst,  ne);

    int r2 = ne12/ne02;
    int r3 = ne13/ne03;

    int nth0 = 32;
    int nth1 = 1;
    int ndst = 4;
    cl_kernel kernel;

#ifdef GGML_OPENCL_SOA_Q
    kernel = backend_ctx->mul_mat.kernel_mul_mv_iq4_nl_f32_flat;

    if (backend_ctx->gpu_family == INTEL) {
        nth0 = 16;
        nth1 = 1;
        ndst = 8;
    } else if (backend_ctx->gpu_family == ADRENO) {
        nth0 = 64;
        nth1 = 1;
        ndst = 8;
    } else {
        GGML_ASSERT(false && "TODO: Unknown GPU");
    }

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0_iq4_nl->q));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0_iq4_nl->d));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
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
#else
    kernel = backend_ctx->mul_mat.kernel_mul_mv_iq4_nl_f32;

    if (backend_ctx->gpu_family == INTEL) {
        nth0 = 16;
        nth1 = 1;
        ndst = 4;
    } else if (backend_ctx->gpu_family == ADRENO) {
        nth0 = 64;
        nth1 = 1;
        ndst = 4;
    } else {
        GGML_ASSERT(false && "TODO: Unknown GPU");
    }

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
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
#endif // GGML_OPENCL_SOA_Q

    // Each subgroup produces N_DST values in the result.
    size_t global_work_size[] = {(size_t)(ne01 + ndst-1)/ndst*nth0, (size_t)ne11*nth1, (size_t)ne12*ne13};
    size_t local_work_size[] = {(size_t)nth0, (size_t)nth1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

static void ggml_cl_mv_q4_k_f32(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_backend_opencl_context * backend_ctx = (ggml_backend_opencl_context *) backend->context;

#ifndef GGML_OPENCL_SOA_Q
    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *) src0->extra;
    cl_ulong offset0 = extra0->offset + src0->view_offs;
#endif

    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *) src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *) dst->extra;

    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

#ifdef GGML_OPENCL_SOA_Q
    const ggml_tensor * soa0_src = src0->view_src != nullptr ? src0->view_src : src0;
    ggml_tensor_extra_cl_q4_K * extra0_q4_K = (ggml_tensor_extra_cl_q4_K *) soa0_src->extra;
#endif

    GGML_TENSOR_LOCALS(int,      ne0, src0, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb0, src0, nb);
    GGML_TENSOR_LOCALS(int,      ne1, src1, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb1, src1, nb);
    GGML_TENSOR_LOCALS(int,      ne,  dst,  ne);

    int r2 = ne12/ne02;
    int r3 = ne13/ne03;

    int nth0 = 32;
    int nth1 = 1;
    int ndst = 4;
    cl_kernel kernel;

#ifdef GGML_OPENCL_SOA_Q
    kernel = backend_ctx->mul_mat.kernel_mul_mv_q4_K_f32_flat;

    if (backend_ctx->gpu_family == INTEL) {
        nth0 = 16;
        nth1 = 1;
        ndst = 16; // 8->16 rows per subgroup - matches N_DST in mul_mv_q4_k_f32_flat.cl (32 spills)
    } else if (backend_ctx->gpu_family == ADRENO) {
        nth0 = 64;
        nth1 = 2;
        ndst = 16;
    } else {
        GGML_ASSERT(false && "TODO: Unknown GPU");
    }

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0_q4_K->q));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0_q4_K->s));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra0_q4_K->d));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_mem),   &extra0_q4_K->dm));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(int),      &offset1));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb02));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb03));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne12));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &nb11));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(cl_ulong), &nb12));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nb13));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &ne0));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &ne1));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(int),      &r2));
    CL_CHECK(clSetKernelArg(kernel, 20, sizeof(int),      &r3));
#else
    kernel = backend_ctx->mul_mat.kernel_mul_mv_q4_K_f32;

    if (backend_ctx->gpu_family == INTEL) {
        nth0 = 16;
        nth1 = 1;
        ndst = 4;
    } else if (backend_ctx->gpu_family == ADRENO) {
        nth0 = 64;
        nth1 = 1;
        ndst = 4;
    } else {
        GGML_ASSERT(false && "TODO: Unknown GPU");
    }

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),     &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(int),        &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),     &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(int),        &offset1));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),     &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(int),        &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),        &ne00));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),        &ne01));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong),   &nb01));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong),   &nb02));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong),   &nb03));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),        &ne12));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong),   &nb11));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong),   &nb12));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong),   &nb13));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),        &ne0));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),        &ne1));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),        &r2));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),        &r3));
#endif // GGML_OPENCL_SOA_Q

    size_t global_work_size[] = {(size_t)(ne01+ndst*nth1-1)/(ndst*nth1)*nth0, (size_t)ne11*nth1, (size_t)ne12*ne13};
    size_t local_work_size[] = {(size_t)nth0, (size_t)nth1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

static void ggml_cl_mv_q5_k_f32(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_backend_opencl_context * backend_ctx = (ggml_backend_opencl_context *) backend->context;

#ifndef GGML_OPENCL_SOA_Q
    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *) src0->extra;
    cl_ulong offset0 = extra0->offset + src0->view_offs;
#endif

    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *) src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *) dst->extra;

    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

#ifdef GGML_OPENCL_SOA_Q
    const ggml_tensor * soa0_src = src0->view_src != nullptr ? src0->view_src : src0;
    ggml_tensor_extra_cl_q5_K * extra0_q5_K = (ggml_tensor_extra_cl_q5_K *) soa0_src->extra;
#endif

    GGML_TENSOR_LOCALS(int,      ne0, src0, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb0, src0, nb);
    GGML_TENSOR_LOCALS(int,      ne1, src1, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb1, src1, nb);
    GGML_TENSOR_LOCALS(int,      ne,  dst,  ne);

    int r2 = ne12/ne02;
    int r3 = ne13/ne03;

    int nth0 = 32;
    int nth1 = 1;
    int ndst = 4;
    cl_kernel kernel;

#ifdef GGML_OPENCL_SOA_Q
        kernel = backend_ctx->mul_mat.kernel_mul_mv_q5_K_f32_flat;

    if (backend_ctx->gpu_family == INTEL) {
        nth0 = 16;
        nth1 = 1;
        ndst = 8; // 4->8 rows per subgroup (2x activation reuse)
    } else if (backend_ctx->gpu_family == ADRENO) {
        nth0 = 64;
        nth1 = 2;
        ndst = 16;
    } else {
        GGML_ASSERT(false && "TODO: Unknown GPU");
    }

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0_q5_K->q));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0_q5_K->qh));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra0_q5_K->s));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_mem),   &extra0_q5_K->d));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extra0_q5_K->dm));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb02));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &nb03));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &ne12));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(cl_ulong), &nb11));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(cl_ulong), &nb12));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_ulong), &nb13));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &ne0));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(int),      &ne1));
    CL_CHECK(clSetKernelArg(kernel, 20, sizeof(int),      &r2));
    CL_CHECK(clSetKernelArg(kernel, 21, sizeof(int),      &r3));
#else
    kernel = backend_ctx->mul_mat.kernel_mul_mv_q5_K_f32;

    if (backend_ctx->gpu_family == INTEL) {
        nth0 = 16;
        nth1 = 1;
        ndst = 4;
    } else if (backend_ctx->gpu_family == ADRENO) {
        nth0 = 64;
        nth1 = 1;
        ndst = 4;
    } else {
        GGML_ASSERT(false && "TODO: Unknown GPU");
    }

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(int),      &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(int),      &offset1));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(int),      &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb02));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_ulong), &nb03));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne12));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb11));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &nb12));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(cl_ulong), &nb13));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne0));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &ne1));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &r2));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(int),      &r3));
#endif // GGML_OPENCL_SOA_Q

    size_t global_work_size[] = {(size_t)(ne01+ndst*nth1-1)/(ndst*nth1)*nth0, (size_t)ne11*nth1, (size_t)ne12*ne13};
    size_t local_work_size[] = {(size_t)nth0, (size_t)nth1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

static void ggml_cl_mv_q6_k_f32(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_backend_opencl_context * backend_ctx = (ggml_backend_opencl_context *) backend->context;

#ifndef GGML_OPENCL_SOA_Q
    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *) src0->extra;
    cl_ulong offset0 = extra0->offset + src0->view_offs;
#endif

    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *) src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *) dst->extra;

    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

#ifdef GGML_OPENCL_SOA_Q
    const ggml_tensor * soa0_src = src0->view_src != nullptr ? src0->view_src : src0;
    ggml_tensor_extra_cl_q6_K * extra0_q6_K = (ggml_tensor_extra_cl_q6_K *) soa0_src->extra;
#endif

    GGML_TENSOR_LOCALS(int,      ne0, src0, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb0, src0, nb);
    GGML_TENSOR_LOCALS(int,      ne1, src1, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb1, src1, nb);
    GGML_TENSOR_LOCALS(int,      ne,  dst,  ne);

    int r2 = ne12/ne02;
    int r3 = ne13/ne03;

    int nth0 = 32;
    int nth1 = 1;
    int ndst = 4;
    cl_kernel kernel;

#ifdef GGML_OPENCL_SOA_Q
    kernel = backend_ctx->mul_mat.kernel_mul_mv_q6_K_f32_flat;

    if (backend_ctx->gpu_family == INTEL) {
        nth0 = 16;
        nth1 = 2;
        ndst = 4;
    } else if (backend_ctx->gpu_family == ADRENO) {
        nth0 = 64;
        nth1 = 2;
        ndst = 16;
    } else {
        GGML_ASSERT(false && "TODO: Unknown GPU");
    }

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0_q6_K->ql));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0_q6_K->qh));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra0_q6_K->s));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_mem),   &extra0_q6_K->d));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(int),      &ne01));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne02));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int),      &ne10));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int),      &ne12));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int),      &ne0));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &ne1));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &r2));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &r3));
    // The optimizer-barrier arg exists only in the ADRENO_OLD_COMPILER build of
    // this kernel; conformant compilers get the original 17-arg signature.
    if (backend_ctx->q6_k_flat_old_compiler) {
        cl_uchar q6k_mask = 0xFF;   // never 0xFE in prod; see the kernel note
        CL_CHECK(clSetKernelArg(kernel, 17, sizeof(cl_uchar), &q6k_mask));
    }
#else
    kernel = backend_ctx->mul_mat.kernel_mul_mv_q6_K_f32;

    if (backend_ctx->gpu_family == INTEL) {
        nth0 = 16;
        nth1 = 2;
        ndst = 1;
    } else if (backend_ctx->gpu_family == ADRENO) {
        nth0 = 64;
        nth1 = 2;
        ndst = 1;
    } else {
        GGML_ASSERT(false && "TODO: Unknown GPU");
    }

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
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
#endif // GGML_OPENCL_SOA_Q

    size_t global_work_size[] = {(size_t)(ne01+ndst*nth1-1)/(ndst*nth1)*nth0, (size_t)ne11*nth1, (size_t)ne12*ne13};
    size_t local_work_size[] = {(size_t)nth0, (size_t)nth1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

static void ggml_cl_mv_mxfp4_f32(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_backend_opencl_context * backend_ctx = (ggml_backend_opencl_context *) backend->context;

#ifndef GGML_OPENCL_SOA_Q
    ggml_tensor_extra_cl * extra0 = (ggml_tensor_extra_cl *) src0->extra;
    cl_ulong offset0 = extra0->offset + src0->view_offs;
#endif

    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *) src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *) dst->extra;

    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

#ifdef GGML_OPENCL_SOA_Q
    const ggml_tensor * soa0_src = src0->view_src != nullptr ? src0->view_src : src0;
    ggml_tensor_extra_cl_mxfp4 * extra0_mxfp4 = (ggml_tensor_extra_cl_mxfp4 *) soa0_src->extra;
#endif

    GGML_TENSOR_LOCALS(int,      ne0, src0, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb0, src0, nb);
    GGML_TENSOR_LOCALS(int,      ne1, src1, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb1, src1, nb);
    GGML_TENSOR_LOCALS(int,      ne,  dst,  ne);

    int r2 = ne12/ne02;
    int r3 = ne13/ne03;

    int nth0 = 32;
    int nth1 = 1;
    int ndst = 4;
    cl_kernel kernel;

#ifdef GGML_OPENCL_SOA_Q
    kernel = backend_ctx->mul_mat.kernel_mul_mv_mxfp4_f32_flat;

    cl_mem q;
    if (backend_ctx->gpu_family == INTEL) {
        nth0 = 16;
        nth1 = 2;
        ndst = nth1*2;

        q = extra0_mxfp4->q;
    } else if (backend_ctx->gpu_family == ADRENO) {
        nth0 = 64;
        nth1 = 2;
        ndst = nth1*2;

        q = extra0_mxfp4->q_img;
    } else {
        GGML_ASSERT(false && "TODO: Unknown GPU");
    }

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &q));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0_mxfp4->e));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &nb02));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb03));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne12));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb11));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb12));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &nb13));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &ne0));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne1));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &r2));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &r3));
#else
    kernel = backend_ctx->mul_mat.kernel_mul_mv_mxfp4_f32;

    if (backend_ctx->gpu_family == INTEL) {
        nth0 = 16;
        nth1 = 2;
        ndst = nth1*2;
    } else if (backend_ctx->gpu_family == ADRENO) {
        nth0 = 64;
        nth1 = 2;
        ndst = nth1*2;
    } else {
        GGML_ASSERT(false && "TODO: Unknown GPU");
    }

    CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0->data_device));
    CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_ulong), &offset0));
    CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
    CL_CHECK(clSetKernelArg(kernel,  3, sizeof(cl_ulong), &offset1));
    CL_CHECK(clSetKernelArg(kernel,  4, sizeof(cl_mem),   &extrad->data_device));
    CL_CHECK(clSetKernelArg(kernel,  5, sizeof(cl_ulong), &offsetd));
    CL_CHECK(clSetKernelArg(kernel,  6, sizeof(int),      &ne00));
    CL_CHECK(clSetKernelArg(kernel,  7, sizeof(cl_ulong), &nb01));
    CL_CHECK(clSetKernelArg(kernel,  8, sizeof(cl_ulong), &nb02));
    CL_CHECK(clSetKernelArg(kernel,  9, sizeof(cl_ulong), &nb03));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int),      &ne12));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_ulong), &nb11));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_ulong), &nb12));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(cl_ulong), &nb13));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(int),      &ne0));
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(int),      &ne1));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(int),      &r2));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int),      &r3));
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(float)*nth0,nullptr));
#endif

    // Each SIMD group produces N_DST values in the result. Assuming each
    // workgroup has N_SIMDGROUP SIMD groups, then each workgroup will
    // produce N_DST*N_SIMDGROUP values in the result.
    size_t global_work_size[] = {(size_t)(ne01 + ndst-1)/ndst*nth0, (size_t)ne11*nth1, (size_t)ne12*ne13};
    size_t local_work_size[] = {(size_t)nth0, (size_t)nth1, 1};

    backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
}

void ggml_cl_mul_mat(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0);
    GGML_ASSERT(src0->extra);
    GGML_ASSERT(src1);
    GGML_ASSERT(src1->extra);
    GGML_ASSERT(dst);
    GGML_ASSERT(dst->extra);

    // bf16 is stored as f16 on device
    const enum ggml_type src0t = (src0->type == GGML_TYPE_BF16) ? GGML_TYPE_F16 : src0->type;
    const enum ggml_type src1t = src1->type;

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    // quant kv without FA
    // used for non-contiguous src0 (the usual head-major permuted K view when n_head_kv>1)
    // AND for the contiguous case that occurs when n_head_kv==1 (e.g. Gemma-4 E2B)
    if ((src0t == GGML_TYPE_Q4_0 || src0t == GGML_TYPE_Q8_0) &&
        (!ggml_is_contiguous(src0) || src1->ne[2] > src0->ne[2])) {
        cl_mem f16_buf = ggml_cl_mul_mat_dequant_quant_to_f16(backend_ctx, src0, nullptr);

        ggml_tensor         fake_src0 = *src0;
        ggml_tensor_extra_cl fake_extra = {};
        fake_extra.data_device = f16_buf;
        fake_extra.offset      = 0;
        fake_src0.type     = GGML_TYPE_F16;
        fake_src0.extra    = &fake_extra;
        fake_src0.view_src = nullptr;
        fake_src0.view_offs = 0;
        fake_src0.nb[0] = sizeof(ggml_fp16_t);
        fake_src0.nb[1] = fake_src0.nb[0] * src0->ne[0];
        fake_src0.nb[2] = fake_src0.nb[1] * src0->ne[1];
        fake_src0.nb[3] = fake_src0.nb[2] * src0->ne[2];

        ggml_cl_mul_mat(backend, &fake_src0, src1, dst);
        return;
    }

    ggml_tensor_extra_cl * extra1 = (ggml_tensor_extra_cl *)src1->extra;
    ggml_tensor_extra_cl * extrad = (ggml_tensor_extra_cl *)dst->extra;

    cl_ulong offset1 = extra1->offset + src1->view_offs;
    cl_ulong offsetd = extrad->offset + dst->view_offs;

#ifdef GGML_OPENCL_SOA_Q
    // view->extra stays pre-SoA; cast to the SoA struct would SIGSEGV.
    // Follow view_src to reach the real SoA extra.
    const ggml_tensor * soa0_src = src0->view_src != nullptr ? src0->view_src : src0;
    ggml_tensor_extra_cl_q4_0 * extra0_q4_0 = (ggml_tensor_extra_cl_q4_0 *)soa0_src->extra;
#endif

    GGML_TENSOR_LOCALS(int,      ne0, src0, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb0, src0, nb);
    GGML_TENSOR_LOCALS(int,      ne1, src1, ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb1, src1, nb);
    GGML_TENSOR_LOCALS(int,      ne,  dst,  ne);
    GGML_TENSOR_LOCALS(cl_ulong, nb,  dst,  nb);

    int r2 = ne12/ne02;
    int r3 = ne13/ne03;

    GGML_ASSERT(ne00 == ne10);

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
    // adreno GEMM/GEMV kernels do not support broadcast, assuming ne2 and ne3 are 1 for src1
    // so we handle broadcast here
    if ((ne12 > 1 || ne13 > 1) && ne02 == 1 && ne03 == 1 &&
        src0t != GGML_TYPE_F16 && src0t != GGML_TYPE_F32) {
        for (int i13 = 0; i13 < ne13; ++i13) {
            for (int i12 = 0; i12 < ne12; ++i12) {
                ggml_tensor s1 = *src1;
                s1.ne[2] = 1; s1.ne[3] = 1;
                s1.view_offs = src1->view_offs + (size_t)i12*nb12 + (size_t)i13*nb13;
                ggml_tensor d = *dst;
                d.ne[2] = 1; d.ne[3] = 1;
                d.view_offs = dst->view_offs + (size_t)i12*nb2 + (size_t)i13*nb3;
                ggml_cl_mul_mat(backend, src0, &s1, &d);
            }
        }
        return;
    }
#endif

    int nth0 = 32;
    int nth1 = 1;

    cl_kernel kernel;

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
    if (ggml_cl_mul_mat_f16_f32_attn_adreno(backend, src0, src1, dst)) {
        return;
    }
#endif

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
    if (ne01 && ne1 && use_adreno_kernels(backend_ctx, src0)) {
        // NOTE: Kernels using image1d_buffer_t (e.g., src0_q) would normally require
        // a limit check, but q4_0 / q4_1 tensors are very unlikely to exceed that
        // limit, so the check is omitted.

        // q1_0 x fp32
        if (src0t == GGML_TYPE_Q1_0 && src1t == GGML_TYPE_F32 &&
            enable_adreno_trans_weight(backend_ctx, src0)) {
                ggml_cl_mul_mat_q1_0_f32_adreno(backend, src0, src1, dst);
                return;
        }

        // q4_0 x fp32
        if(src0t == GGML_TYPE_Q4_0 && src1t == GGML_TYPE_F32) {
            ggml_cl_mul_mat_q4_0_f32_adreno(backend, src0, src1, dst);
            return;
        }

        // q4_1 x fp32
        if (src0t == GGML_TYPE_Q4_1 && src1t == GGML_TYPE_F32) {
            ggml_cl_mul_mat_q4_1_f32_adreno(backend, src0, src1, dst);
            return;
        }

        // q5_0 x fp32
        if (src0t == GGML_TYPE_Q5_0 && src1t == GGML_TYPE_F32) {
            ggml_cl_mul_mat_q5_0_f32_adreno(backend, src0, src1, dst);
            return;
        }

        // q5_1 x fp32
        if (src0t == GGML_TYPE_Q5_1 && src1t == GGML_TYPE_F32) {
            ggml_cl_mul_mat_q5_1_f32_adreno(backend, src0, src1, dst);
            return;
        }

        // iq4_nl x fp32
        if (src0t == GGML_TYPE_IQ4_NL && src1t == GGML_TYPE_F32) {
            ggml_cl_mul_mat_iq4_nl_f32_adreno(backend, src0, src1, dst);
            return;
        }

        // q8_0 x fp32
        if (src0t == GGML_TYPE_Q8_0 && src1t == GGML_TYPE_F32 &&
            enable_adreno_trans_weight(backend_ctx, src0)) {
                ggml_cl_mul_mat_q8_0_f32_adreno(backend, src0, src1, dst);
                return;
        }

        // q4_k x fp32
        if (src0t == GGML_TYPE_Q4_K && src1t == GGML_TYPE_F32 && !use_flat_gemv_for_large_m_q4_K(backend_ctx, src0)) {
            ggml_cl_mul_mat_q4_K_f32_adreno(backend, src0, src1, dst);
            return;
        }

        // q6_K x fp32
        if (src0t == GGML_TYPE_Q6_K && src1t == GGML_TYPE_F32 && !use_flat_gemv_for_large_m_q6_K(backend_ctx, src0)) {
            ggml_cl_mul_mat_q6_K_f32_adreno(backend, src0, src1, dst);
            return;
        }

        // q5_K x fp32
        if (src0t == GGML_TYPE_Q5_K && src1t == GGML_TYPE_F32 &&
            enable_adreno_trans_weight_q5_K(backend_ctx, src0)) {
            ggml_cl_mul_mat_q5_K_f32_adreno(backend, src0, src1, dst);
            return;
        }
    } // if (ne01 && ne1)
#endif // GGML_OPENCL_USE_ADRENO_KERNELS

    // GEMM using local memory
    // Current BK = 16, so ne00 % 16 == 0
    //
    // Certain A7X compiler (E031.41) executes kernel_mul_mm_f32_f32_l4_lm poorly;
    // matrices with ne11 <= 8 appears OK.
    // Fallback to the MV style kernels for A7x and ne11 > 8.
    // Override with GGML_OPENCL_A7X_F32_LM_BYPASS=0.
    static const char * a7x_f32lm_env    = getenv("GGML_OPENCL_A7X_F32_LM_BYPASS");
    static const bool   a7x_f32lm_bypass = (a7x_f32lm_env == nullptr || a7x_f32lm_env[0] != '0');
    if (src1t == GGML_TYPE_F32 &&
        ne00 % 16 == 0 &&
        ne11 > 1 &&
        !(a7x_f32lm_bypass && src0t == GGML_TYPE_F32 && ne11 > 8 &&
          backend_ctx->adreno_gen == ADRENO_GPU_GEN::A7X)) {
        switch(src0t) {
            case GGML_TYPE_F32:
                if (ggml_cl_mm_f32_f32(backend, src0, src1, dst)) {
                    return;
                }
                break;
            case GGML_TYPE_F16:
                if (ggml_cl_mm_f16_f32(backend, src0, src1, dst)) {
                    return;
                }
                break;
            case GGML_TYPE_Q1_0:
                if (ggml_cl_mm_q1_0_f32(backend, src0, src1, dst)) {
                    return;
                }
                break;
            case GGML_TYPE_Q4_0:
                if (ggml_cl_mm_q4_0_f32(backend, src0, src1, dst)) {
                    return;
                }
                break;
            case GGML_TYPE_Q4_1:
                if (ggml_cl_mm_q4_1_f32(backend, src0, src1, dst)) {
                    return;
                }
                break;
            case GGML_TYPE_Q5_0:
                if (ggml_cl_mm_q5_0_f32(backend, src0, src1, dst)) {
                    return;
                }
                break;
            case GGML_TYPE_Q5_1:
                if (ggml_cl_mm_q5_1_f32(backend, src0, src1, dst)) {
                    return;
                }
                break;
            case GGML_TYPE_Q8_0:
                if (ggml_cl_mm_q8_0_f32(backend, src0, src1, dst)) {
                    return;
                }
                break;
            case GGML_TYPE_IQ4_NL:
                if (ggml_cl_mm_iq4_nl_f32(backend, src0, src1, dst)) {
                    return;
                }
                break;
            case GGML_TYPE_Q4_K:
                if (ggml_cl_mm_q4_k_f32(backend, src0, src1, dst)) {
                    return;
                }
                break;
            case GGML_TYPE_Q5_K:
                if (ggml_cl_mm_q5_k_f32(backend, src0, src1, dst)) {
                    return;
                }
                break;
            case GGML_TYPE_Q6_K:
                if (ggml_cl_mm_q6_k_f32(backend, src0, src1, dst)) {
                    return;
                }
                break;
            default:
                break;
        }
    }

    if (src0t == GGML_TYPE_F16 && src1t == GGML_TYPE_F32 &&
        src0->ne[1] > 32 &&   // M > 32
        src1->ne[1] > 32 &&   // N > 32
        src0->ne[0] > 32 &&   // K > 32
        src0->ne[2] == 1 && src0->ne[3] == 1 &&
        src1->ne[2] == 1 && src1->ne[3] == 1 &&
        ggml_is_contiguous(src0) && ggml_is_contiguous(src1) &&
        backend_ctx->mul_mat.kernel_mul_mat_f16_f32_tiled != NULL) {
        ggml_cl_mul_mat_f16_f32_tiled(backend, src0, src1, dst);
        return;
    }

    if (!ggml_is_transposed(src0) &&
        !ggml_is_transposed(src1) &&
        src1t == GGML_TYPE_F32 &&
        ne00%32 == 0 &&
        ne11 > 2) {
#ifdef GGML_OPENCL_SOA_Q
        // Set up kernel.
        switch(src0t) {
            case GGML_TYPE_Q4_0:
                // This should have been satisfied.
                GGML_ASSERT(ne11 == ne1);
                GGML_ASSERT(ne01 == ne0);

                if (backend_ctx->gpu_family == INTEL) {
                    nth0 = 16;
                    nth1 = 1;

                    kernel = backend_ctx->mul_mat.kernel_mul_mat_q4_0_f32_1d_16x_flat;
                } else if (backend_ctx->gpu_family == ADRENO) {
                    nth0 = 64;
                    nth1 = 1;

                    kernel = backend_ctx->mul_mat.kernel_mul_mat_q4_0_f32_1d_8x_flat;
                } else {
                    GGML_ASSERT(false && "TODO: Unknown GPU");
                }

                CL_CHECK(clSetKernelArg(kernel,  0, sizeof(cl_mem),   &extra0_q4_0->q));
                CL_CHECK(clSetKernelArg(kernel,  1, sizeof(cl_mem),   &extra0_q4_0->d));
                CL_CHECK(clSetKernelArg(kernel,  2, sizeof(cl_mem),   &extra1->data_device));
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
                break;
            default:
                break;
        }

        // Launch kernel.
        if (src0t == GGML_TYPE_Q4_0) {
            size_t global_work_size[] = {(size_t)(ne01 + 7)/8*nth0, (size_t)ne11*nth1, (size_t)ne12*ne13};
            size_t local_work_size[] = {(size_t)nth0, (size_t)nth1, 1};

            if (backend_ctx->gpu_family == INTEL) {
                // Set global size for Intel. It uses 16x output values.
                global_work_size[0] = (size_t)(ne01 + 15)/16*nth0;
                global_work_size[1] = (size_t)ne11*nth1;
                global_work_size[2] = (size_t)ne12*ne13;
            }

            backend_ctx->enqueue_ndrange_kernel(kernel, 3, global_work_size, local_work_size, dst);
            return;
        }
#else // GGML_OPENCL_SOA_Q
        // TODO: add block_q4_0 variant.
#endif // GGML_OPENCL_SOA_Q
    }

    // use custom matrix x vector kernel
    switch (src0t) {
        case GGML_TYPE_F32:
            GGML_ASSERT(src1t == GGML_TYPE_F32);
            ggml_cl_mv_f32_f32(backend, src0, src1, dst);
            return;
        case GGML_TYPE_F16:
            if (src1t == GGML_TYPE_F32) {
                ggml_cl_mv_f16_f32(backend, src0, src1, dst);
            } else {
                ggml_cl_mv_f16_f16(backend, src0, src1, dst);
            }
            return;
        case GGML_TYPE_Q1_0:
            ggml_cl_mv_q1_0_f32(backend, src0, src1, dst);
            return;
        case GGML_TYPE_Q4_0:
            ggml_cl_mv_q4_0_f32(backend, src0, src1, dst);
            return;
        case GGML_TYPE_Q4_1:
            ggml_cl_mv_q4_1_f32(backend, src0, src1, dst);
            return;
        case GGML_TYPE_Q5_0:
            ggml_cl_mv_q5_0_f32(backend, src0, src1, dst);
            return;
        case GGML_TYPE_Q5_1:
            ggml_cl_mv_q5_1_f32(backend, src0, src1, dst);
            return;
        case GGML_TYPE_Q8_0:
            ggml_cl_mv_q8_0_f32(backend, src0, src1, dst);
            return;
        case GGML_TYPE_IQ4_NL:
            ggml_cl_mv_iq4_nl_f32(backend, src0, src1, dst);
            return;
        case GGML_TYPE_Q4_K:
            ggml_cl_mv_q4_k_f32(backend, src0, src1, dst);
            return;
        case GGML_TYPE_Q5_K:
            ggml_cl_mv_q5_k_f32(backend, src0, src1, dst);
            return;
        case GGML_TYPE_Q6_K:
            ggml_cl_mv_q6_k_f32(backend, src0, src1, dst);
            return;
        case GGML_TYPE_MXFP4:
            ggml_cl_mv_mxfp4_f32(backend, src0, src1, dst);
            return;
        default:
            GGML_ASSERT(false && "not implemented");
    }
}
