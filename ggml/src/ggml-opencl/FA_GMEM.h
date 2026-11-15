#pragma once

#if defined(GGML_OPENCL_USE_ADRENO_KERNELS) && defined(GGML_OPENCL_USE_ADRENO_BIN_KERNELS) && GGML_OPENCL_TARGET_VERSION >= 300

#include <cstddef>
#include <tuple>
#include <utility>
#include <map>
#include <string>
#include <vector>
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif // CL_TARGET_OPENCL_VERSION
#include <CL/cl.h>
#include <CL/cl_platform.h>
#include "ggml.h"
#include "opencl_profiling_info.h"
#include "cl-recordable-queue.h"


class FA_GMEM
{
public:

    static void cleanup(cl_context context);
    static size_t get_hash(cl_device_id device_id, cl_uint M, cl_uint N0K1, cl_uint K0N1, cl_uint numHeads,
        ggml_type gtAct = GGML_TYPE_F32, // FP16 or FP32
        ggml_type gtKV = GGML_TYPE_F16); // FP16 or FP32

    static std::string mm0_program_name(ggml_type gtKV);
    static std::string mm1_program_name(ggml_type gtKV);

    static FA_GMEM * create(cl_context context,
        cl_device_id device_id,
        cl_command_queue queue,
        ProfilingInfo& profiling_info,
        cl_program program_mm0, cl_program program_softmax, cl_program program_mm1,
        ggml_type gtKV = GGML_TYPE_F16); // FP16 or FP32

    void enqueue(cl_uint M, cl_uint N0K1, cl_uint K0N1, cl_uint numHeads,
        cl_mem mem_matrixQ, cl_ulong offsetQ,
        cl_mem mem_matrixK, cl_ulong offsetK,
        float scale,
        cl_mem mem_matrixV, cl_ulong offsetV,
        cl_mem mem_matrixO, cl_ulong offsetO,
        const ggml_tensor * tensorC);

    ~FA_GMEM() = default;

private:
    FA_GMEM(cl_context context, cl_device_id device_id, ProfilingInfo & profiling_info, ggml_type gtKV);
    static constexpr size_t widthOfTileN = 64;
    static constexpr bool layoutA_Mfirst = false; // not supported

    // online_softmax_max / online_softmax_sum dispatch shape: fusedSoftmaxLocalM
    // lanes walk adjacent m_4 row-groups (the coalesced/vload4 axis), fusedSoftmaxLocalN lanes
    // partition the N reduction (combined via local memory instead of sub_group_reduce) -- see
    // FA_GMEM.cpp's dispatch and kernels/fa_gmem.cl's kernel comments.
    //
    // Ported from a colleague's llama.cpp/FA_Grid_LLM fork 2026-08-09: their version fuses the
    // same scale + max pass but reduces via sub_group_reduce_max over a full wavefront
    // (their WG is {64,1}: one m_4 row-group per WG, no row-group-mixing risk since LOCAL_M==1).
    // TCU keeps its own local-memory reduction instead of adopting sub_group_reduce, since the
    // two are not equivalent once LOCAL_M>1 shares a wave across row-groups (see kernel comment).
    //
    // FastWan's unmasked FP32 attention was re-swept on 2026-09-02. {4,16} beat {2,32}
    // and {1,64}; re-sweep if this kernel body or the production shape changes.
    static constexpr size_t fusedSoftmaxLocalM = 4;
    static constexpr size_t fusedSoftmaxLocalN = 16;

    typedef std::tuple<cl_context, cl_device_id, ggml_type, ggml_type> kernel_map_key;
    struct kernel_group
    {
        cl_kernel kernel_mm0 = nullptr;
        cl_kernel kernel_cp_pad_a0 = nullptr;
        cl_kernel kernel_cp_pad_b0 = nullptr;

        cl_kernel kernel_online_softmax_max = nullptr;
        cl_kernel kernel_online_softmax_sum = nullptr;
        cl_kernel kernel_online_softmax_output = nullptr;
        cl_kernel kernel_rescale_output = nullptr;

        cl_kernel kernel_mm1 = nullptr;
        cl_kernel kernel_cp_pad_b1 = nullptr;
        cl_kernel kernel_cp_xpose_c1 = nullptr;

        // MM0 and MM1 share a kernel entry point but use different programs.
        std::string variant_suffix_mm0;
        std::string variant_suffix_mm1;

        bool init(cl_program program_mm0, cl_program program_softmax, cl_program program_mm1, cl_device_id device_id, ggml_type gtKV);

        ~kernel_group() {
            if (kernel_mm0) { clReleaseKernel(kernel_mm0); }
            if (kernel_cp_pad_a0) { clReleaseKernel(kernel_cp_pad_a0); }
            if (kernel_cp_pad_b0) { clReleaseKernel(kernel_cp_pad_b0); }

            if (kernel_online_softmax_max) { clReleaseKernel(kernel_online_softmax_max); }
            if (kernel_online_softmax_sum) { clReleaseKernel(kernel_online_softmax_sum); }
            if (kernel_online_softmax_output) { clReleaseKernel(kernel_online_softmax_output); }
            if (kernel_rescale_output) { clReleaseKernel(kernel_rescale_output); }

            if (kernel_mm1) { clReleaseKernel(kernel_mm1); }
            if (kernel_cp_pad_b1) { clReleaseKernel(kernel_cp_pad_b1); }
            if (kernel_cp_xpose_c1) { clReleaseKernel(kernel_cp_xpose_c1); }
        }
    };
    static std::map<kernel_map_key, kernel_group *> kernel_map;
    kernel_group * kernels = nullptr;

    cl_context        m_Context;
    cl_device_id      m_DeviceID;
#ifdef GGML_OPENCL_PROFILING
    ProfilingInfo&    m_ProfilingInfo;
#endif // GGML_OPENCL_PROFILING
    const ggml_type gtypeA;
    const ggml_type gtypeB;
    const bool fp16Q;
    const bool fp16KV;
    const size_t sizeof_TQ;
    const size_t sizeof_TKV;

    static void query_gmem_size(cl_device_id device_id);
    static std::map<cl_device_id, size_t> available_onchip_global_memory;

    static std::tuple<cl_uint, cl_uint, cl_uint, cl_uint, cl_uint, cl_uint, cl_uint> compute_stride_bytes_QKV(
        cl_uint blockM, cl_uint blockN0K1, cl_uint blockK0N1,
        ggml_type gtAct, ggml_type gtKV);
    static bool block_dims_fit(cl_device_id device_id,
        cl_uint blockM, cl_uint blockN0K1, cl_uint blockK0N1,
        ggml_type gtAct, ggml_type gtKV);
    static bool determine_block_dims(cl_device_id device_id,
        cl_uint M, cl_uint N0K1, cl_uint K0N1,
        ggml_type gtAct, ggml_type gtKV,
        cl_uint * bm, cl_uint * bn);

    cl_recordable_queue rq;
};
#endif
