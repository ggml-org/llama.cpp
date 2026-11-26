#pragma once

#if defined(GGML_OPENCL_USE_ADRENO_KERNELS) && defined(GGML_OPENCL_USE_ADRENO_BIN_KERNELS) && GGML_OPENCL_TARGET_VERSION >= 300

#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>
#include <map>
#include <string>
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif // CL_TARGET_OPENCL_VERSION
#include <CL/cl.h>
#include <CL/cl_platform.h>
#include "ggml.h"
#include "opencl_profiling_info.h"
#include "cl-recordable-queue.h"


class MM_GMEM {
public:

    // Smaller dimensions use the existing matmul path.
    static constexpr cl_uint min_dim = 128;

    static void cleanup(cl_context context);
    static size_t get_hash(cl_device_id device_id, cl_uint M, cl_uint N, cl_uint K,
        ggml_type gtA = GGML_TYPE_F16,  // FP16, FP32, or Q8_0
        ggml_type gtB = GGML_TYPE_F32,  // FP16 or FP32
        ggml_prec prec = GGML_PREC_UNDEFINED, // Undefined allows FP16 accumulation; F32 requires FP32.
        bool q8_0A_transposed = false);

    static std::string program_name(ggml_type gtA, ggml_type gtB, ggml_prec prec, bool q8_0A_transposed);

    static MM_GMEM * create(cl_context context,
        cl_device_id device_id,
        cl_command_queue queue,
        ProfilingInfo& profiling_info,
        cl_program program,
        ggml_type gtA = GGML_TYPE_F16,  // FP16, FP32, or Q8_0
        ggml_type gtB = GGML_TYPE_F32,  // FP16 or FP32
        ggml_prec prec = GGML_PREC_UNDEFINED, // Undefined allows FP16 accumulation; F32 requires FP32.
        bool q8_0A_transposed = false);

    void enqueue(cl_uint M, cl_uint N, cl_uint K,
        cl_mem mem_matrixA, cl_ulong offsetA,
        cl_mem mem_matrixB, cl_ulong offsetB,
        cl_mem mem_matrixC, cl_ulong offsetC,
        const ggml_tensor * tensorC,
        cl_mem mem_scalesA = nullptr, cl_ulong offsetScalesA = 0);

    // Batched matmul: batch_count 2D matmuls of the same (M, N, K) shape, each
    // shifted from the previous by (strideA_bytes, strideB_bytes, strideC_bytes).
    // Thin wrapper that iterates 'enqueue' per batch; keeps mm_gmem.cl unchanged.
    // Pass strideX_bytes = 0 to broadcast that operand across all batches.
    void enqueue_batched(cl_uint M, cl_uint N, cl_uint K,
        cl_mem mem_matrixA, cl_ulong offsetA_base, cl_ulong strideA_bytes,
        cl_mem mem_matrixB, cl_ulong offsetB_base, cl_ulong strideB_bytes,
        cl_mem mem_matrixC, cl_ulong offsetC_base, cl_ulong strideC_bytes,
        cl_uint batch_count,
        const ggml_tensor * tensorC);

    ~MM_GMEM() = default;

private:
    MM_GMEM(cl_context context, cl_device_id device_id, cl_command_queue queue, ProfilingInfo & profiling_info,
        ggml_type gtA, ggml_type gtB, ggml_prec prec, bool q8_0A_transposed);
    static constexpr size_t sizeof_Tout = 4;
    static constexpr size_t widthOfTileN = 64;
    typedef std::tuple<ggml_type, ggml_type, bool, bool> kernel_key; // (gtA, gtB, fp16_acc, q8_0A_transposed)
    typedef std::tuple<cl_context, cl_device_id, ggml_type, ggml_type, bool, bool> kernel_map_key;
    struct kernel_group
    {
        cl_kernel kernel_mm = nullptr;
        cl_kernel kernel_cp_pad_a = nullptr;
        cl_kernel kernel_cp_pad_b = nullptr;

        bool init(cl_program program, cl_device_id device_id, const kernel_key & kk);

        ~kernel_group() {
            if (kernel_mm) { clReleaseKernel(kernel_mm); }
            if (kernel_cp_pad_a) { clReleaseKernel(kernel_cp_pad_a); }
            if (kernel_cp_pad_b) { clReleaseKernel(kernel_cp_pad_b); }
        }
    };
    static std::map<kernel_map_key, kernel_group *> kernel_map;
    kernel_group * kernels = nullptr;

    cl_context        m_Context;
    cl_device_id      m_DeviceID;
    cl_command_queue  m_CommandQueue;
#ifdef GGML_OPENCL_PROFILING
    ProfilingInfo&    m_ProfilingInfo;
#endif // GGML_OPENCL_PROFILING
    const ggml_type gtypeA;
    const ggml_type gtypeB;
    const ggml_prec prec;
    // Q8_0's Adreno transpose permutes quant words and scale elements separately.
    // The copy stage restores a K-first on-chip tile before MM_GMEM consumes it.
    const bool q8_0A_transposed;
    const bool fp16A;
    const bool fp16B;
    const size_t sizeof_TA;
    const size_t sizeof_TB;

    static void query_gmem_size(cl_device_id device_id);
    static std::map<cl_device_id, size_t> available_onchip_global_memory;

    // Matrix C's per-block image view spans blockN * full_M elements regardless
    // of the enqueue loop nesting, so it also needs the device image-buffer limit.
    static void query_max_image_elements(cl_device_id device_id);
    static std::map<cl_device_id, size_t> max_image_elements;

    // clCreateSubBuffer requires its origin to be a multiple of this (CL_DEVICE_MEM_BASE_ADDR_ALIGN,
    // reported in bits); a non-last blockK boundary is exactly such an origin (see enqueue()'s
    // k_block_offset), so determine_block_dims must constrain blockK to it, not just to kmult.
    static size_t query_mem_base_addr_align_bytes(cl_device_id device_id);
    static std::map<cl_device_id, size_t> mem_base_addr_align_bytes;

    static std::tuple<cl_uint, cl_uint> compute_stride_bytes_AB(
        cl_uint K, ggml_type gtA, ggml_type gtB, bool fp16_acc = false);
    static bool block_dims_fit(cl_device_id device_id,
        cl_uint blockM, cl_uint blockN, cl_uint blockK, cl_uint full_M,
        ggml_type gtA, ggml_type gtB, bool fp16_acc = false);
    static bool determine_block_dims(cl_device_id device_id, cl_uint M, cl_uint N, cl_uint K,
        ggml_type gtA, ggml_type gtB,
        cl_uint * bm, cl_uint * bn, cl_uint * bk, bool fp16_acc = false);

    // BF16 and F32 precision requests use the F32 accumulator program.
    static bool use_fp16_acc(ggml_prec prec) {
        return prec != GGML_PREC_F32 && prec != GGML_PREC_BF16;
    }

    static bool is_q8_0(ggml_type gt) { return gt == GGML_TYPE_Q8_0; }
    // A's copy step must consume a whole number of K-tiles per iteration; Q8_0 A tightens
    // that to a whole quant block (ggml-common.h's QK8_0 == 32, not pulled in here to avoid
    // the extra include) so every blockK boundary is also a scale boundary.
    static cl_uint k_multiple(ggml_type gtA, bool fp16_acc) {
        return is_q8_0(gtA) ? 32 : (fp16_acc ? 16 : 8);
    }

    cl_recordable_queue rq;
};
#endif
