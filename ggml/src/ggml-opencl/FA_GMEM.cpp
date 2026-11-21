#if defined(GGML_OPENCL_USE_ADRENO_KERNELS) && defined(GGML_OPENCL_USE_ADRENO_BIN_KERNELS) && GGML_OPENCL_TARGET_VERSION >= 300

#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <tuple>
#include <vector>
#include <malloc.h>
#include <algorithm>
#include <limits>

#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif // CL_TARGET_OPENCL_VERSION
#include <CL/cl.h>
#include <CL/cl_platform.h>
#include <CL/cl_ext_qcom.h>

#include "FA_GMEM.h"
#include "ggml.h"
#include "ggml-impl.h"
#include "opencl_profiling_info.h"
#include "fa-per-row-data.h"


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// static FA_GMEM
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::map<FA_GMEM::kernel_map_key, FA_GMEM::kernel_group *> FA_GMEM::kernel_map;
void FA_GMEM::cleanup(cl_context context) {
    for (auto it = kernel_map.begin(); it != kernel_map.end();) {
        if (std::get<0>(it->first) == context) {
            delete it->second;
            it = kernel_map.erase(it);
        } else {
            ++it;
        }
    }
}
std::tuple<cl_uint, cl_uint, cl_uint, cl_uint, cl_uint, cl_uint, cl_uint> FA_GMEM::compute_stride_bytes_QKV(
    const cl_uint blockM, const cl_uint blockN0K1, const cl_uint totalK0N1,
    ggml_type gtAct, ggml_type gtKV) {

    // MM0
    cl_uint mA0_paddedK = totalK0N1; // Q, K-first
    cl_uint mB0_paddedK = totalK0N1; // K, K-first
    cl_uint mX0_paddedM = blockM;    // Q*Kt, M-first

    // SM
    cl_uint mSM_paddedN = sizeof(fa_per_row_data) / sizeof(float); // Float elements per four-row group.

    // MM1
    cl_uint mX1_paddedK = blockN0K1; // softmax(Q*Kt), K-first
    cl_uint mB1_paddedK = blockN0K1; // V, K-first (N-first in DDR, but transposed into GMEM)
    cl_uint mC1_paddedM = blockM;    // softmax(Q*Kt)*V, M-first

    const bool fp16Q   = (gtAct == GGML_TYPE_F16);
    const bool fp16KV  = (gtKV == GGML_TYPE_F16);
    const size_t sizeof_TQ   = fp16Q   ? 2 : 4;
    const size_t sizeof_TKV  = fp16KV  ? 2 : 4;
    constexpr size_t sizeof_TScore = sizeof(float);
    // MM1 uses packed-half probabilities with the ordinary FP16-accumulator program, independent of Q's input type.
    const size_t sizeof_TA1_onchip = 2;
    // Source V staging always writes packed half, including native FP32 V.
    const size_t sizeof_TB1_onchip = 2;

    // 512bits is one line in the CCHE (512bits/8bits = 64bytes)
    GGML_ASSERT(!layoutA_Mfirst);
    // independent of Q tensor layout, the block A0 layout is always K-first
    mA0_paddedK += (mA0_paddedK % (128 / sizeof_TQ)) ? 0 : 64 / sizeof_TQ;
    // independent of K tensor layout, the block B0 layout is always K-first
    mB0_paddedK += (mB0_paddedK % (128 / sizeof_TKV))  ? 0 : 64 / sizeof_TKV;
    // QKt is an implementation detail, so pick block X0 layout as M-first, and block X1 as K-first
    mX0_paddedM += (mX0_paddedM % (128 / sizeof_TScore)) ? 0 : 64 / sizeof_TScore;

    mSM_paddedN += (mSM_paddedN % (128 / sizeof(float)))  ? 0 : 64 / sizeof(float);

    mX1_paddedK += (mX1_paddedK % (128 / sizeof_TA1_onchip)) ? 0 : 64 / sizeof_TA1_onchip;
    // independent of V tensor layout, the block B1 layout is always K-first
    mB1_paddedK += (mB1_paddedK % (128 / sizeof_TB1_onchip))  ? 0 : 64 / sizeof_TB1_onchip;
    mC1_paddedM += (mC1_paddedM % (128 / sizeof_TScore)) ? 0 : 64 / sizeof_TScore;


    cl_uint lineStrideMatrixA0inBytes = sizeof_TQ   * mA0_paddedK;
    cl_uint lineStrideMatrixB0inBytes = sizeof_TKV  * mB0_paddedK;
    cl_uint lineStrideMatrixX0inBytes = sizeof_TScore * mX0_paddedM;

    cl_uint lineStrideMatrixSMinBytes = sizeof(float) * mSM_paddedN;

    cl_uint lineStrideMatrixX1inBytes = sizeof_TA1_onchip * mX1_paddedK;
    cl_uint lineStrideMatrixB1inBytes = sizeof_TB1_onchip * mB1_paddedK;
    cl_uint lineStrideMatrixC1inBytes = sizeof_TScore * mC1_paddedM;


    return {lineStrideMatrixA0inBytes, lineStrideMatrixB0inBytes, lineStrideMatrixX0inBytes,
        lineStrideMatrixSMinBytes,
        lineStrideMatrixX1inBytes, lineStrideMatrixB1inBytes, lineStrideMatrixC1inBytes};
}

// blockA0 is only used for MM0, per blockM
// blockX1 is only used for MM1, per blockN, so can't be shared with A0
// blockB will be shared across the two MM, so pick the max size
// blockX0 is only used for MM0 (plus max reduction), per blockN
// blockC1 is only used for MM1, but a longer lifetime (for totalN), so can't be shared with X0
// blockSM aka SoftMax per-row max / sum
// Softmax writes padding for full four-row groups, so A1 must reserve the last group.
#define COMPUTE_STRIDE_BYTES_QKV(blockM, blockN0K1, totalK0N1, gtAct, gtKV) \
auto stridesQKV = compute_stride_bytes_QKV(blockM, blockN0K1, totalK0N1, gtAct, gtKV); \
    const cl_uint lineStrideMatrixA0inBytes = std::get<0>(stridesQKV); \
    const cl_uint lineStrideMatrixB0inBytes = std::get<1>(stridesQKV); \
    const cl_uint lineStrideMatrixC0inBytes = std::get<2>(stridesQKV); \
    const cl_uint lineStrideMatrixSMinBytes = std::get<3>(stridesQKV); \
    const cl_uint lineStrideMatrixA1inBytes = std::get<4>(stridesQKV); \
    const cl_uint lineStrideMatrixB1inBytes = std::get<5>(stridesQKV); \
    const cl_uint lineStrideMatrixC1inBytes = std::get<6>(stridesQKV); \
    \
    const size_t msizeA0_block_bytes = size_t(blockM) * lineStrideMatrixA0inBytes; \
    const size_t _msizeB0_block_bytes = size_t(blockN0K1) * lineStrideMatrixB0inBytes; \
    const size_t _msizeC0_block_bytes = size_t(blockN0K1) * lineStrideMatrixC0inBytes; \
    \
    const size_t msizeSM_block_bytes = ((size_t(blockM) + 3) / 4) * lineStrideMatrixSMinBytes; \
    \
    const size_t _msizeA1_block_bytes = ((size_t(blockM) + 3) & ~size_t(3)) * lineStrideMatrixA1inBytes; \
    const size_t _msizeB1_block_bytes = size_t(totalK0N1) * lineStrideMatrixB1inBytes; \
    const size_t msizeC1_block_bytes = size_t(totalK0N1) * lineStrideMatrixC1inBytes; \
    \
    const size_t msizeB_block_bytes = std::max(_msizeB0_block_bytes, _msizeB1_block_bytes)

bool FA_GMEM::block_dims_fit(cl_device_id device_id,
    const cl_uint blockM, const cl_uint blockN0K1, const cl_uint totalK0N1,
    ggml_type gtAct, ggml_type gtKV) {

    COMPUTE_STRIDE_BYTES_QKV(blockM, blockN0K1, totalK0N1, gtAct, gtKV);

    query_gmem_size(device_id);
    const size_t total_block_bytes = msizeA0_block_bytes + _msizeC0_block_bytes + _msizeA1_block_bytes +
        msizeSM_block_bytes + msizeC1_block_bytes + msizeB_block_bytes;

    if (total_block_bytes > available_onchip_global_memory.at(device_id)) {
        return false;
    }

    return true;
}

bool FA_GMEM::determine_block_dims(cl_device_id device_id,
    const cl_uint M, const cl_uint N0K1, const cl_uint totalK0N1,
    ggml_type gtAct, ggml_type gtKV,
    cl_uint * bm, cl_uint * bn) {

    // kernel assumption
    GGML_ASSERT(totalK0N1 % 8 == 0);

    // Recalibrated 2026-08-02 (end-to-end self-attn FA A/B, interleaved same-session, multiple
    // reps per floor -- see fa-gmem-min-efficient-blockm-recalibrated.md): the original
    // MIN_EFFICIENT_BLOCKM=1024 (2026-07-23) predates MM1's fp16-accumulate/pkmad.f16
    // enablement and a softmax coalescing fix, both of which shifted mm1's efficiency-vs-blockM
    // curve. Real self-attn FA latency (M=8190, N0K1=8192, K0N1=128, NH=24, -Mask) at
    // floor=1024 (blockM=1408) measured ~429ms avg across 4 interleaved reps; floor=1792/2048
    // (both land on blockM=2048, blockN0K1=1088) measured ~405-412ms, a reproducible ~5% win
    // with non-overlapping ranges across every rep; floor=2560 (blockM=2752, matching the OLD
    // pre-recalibration algorithm's answer almost exactly) regressed to ~445ms -- confirming
    // launches(blockM) is still genuinely non-monotonic post-recalibration, not just shifted
    // uniformly higher. Cross-attn (N0K1=512) is unaffected by this whole range: it already
    // lands on blockM=2752, blockN0K1=512 (the whole N0K1 in one block) at any floor <= 2560.
    // Exact values shift with GMEM budget/dtype since they come from a search (see TILE
    // alignment note below), not a fixed target -- this is the floor that empirically won for
    // the two production shapes measured, not a mm1-utilization-plateau constant re-derived
    // from first principles.
    constexpr cl_uint MIN_EFFICIENT_BLOCKM = 2048;

    // mm0/mm1's tiled kernel dispatch (FA_GMEM.cpp:817,965) uses ceil(blockM/TILE) and
    // ceil(blockN0K1/TILE) for the global size of EVERY m_block/n0k1_block, and only narrows
    // that for the true LAST block (FA_GMEM.cpp:818-819,821-822,966-967) -- each non-last
    // block's GMEM buffer is sized to exactly blockM/blockN0K1, not rounded up to a tile
    // multiple. So any non-last block size that isn't a multiple of TILE makes the kernel
    // overrun its own buffer. The old halving search always produced powers of 2 (hence
    // TILE-aligned) and never hit this; a candidate covering the WHOLE dimension in one block
    // is inherently "last" and is exempt (block_dims_fit(..., M or N0K1, ...) below).
    constexpr cl_uint TILE = static_cast<cl_uint>(widthOfTileN);

    // largest blockN0K1 <= N0K1 that fits at this blockM (0 = infeasible even at minimum)
    auto max_bn_for_bm = [&](cl_uint bm_) -> cl_uint {
        if (!block_dims_fit(device_id, bm_, 8, totalK0N1, gtAct, gtKV)) return 0;
        if (block_dims_fit(device_id, bm_, N0K1, totalK0N1, gtAct, gtKV)) return N0K1; // single block: exempt
        cl_uint max_full_tiles = N0K1 / TILE; // largest TILE-multiple <= N0K1
        if (max_full_tiles == 0 || !block_dims_fit(device_id, bm_, TILE, totalK0N1, gtAct, gtKV)) return 0;
        cl_uint lo = 1, hi = max_full_tiles; // binary search in units of TILE
        while (hi - lo > 1) {
            cl_uint mid = lo + (hi - lo) / 2;
            if (block_dims_fit(device_id, bm_, mid * TILE, totalK0N1, gtAct, gtKV)) lo = mid; else hi = mid;
        }
        return lo * TILE;
    };

    // launches(blockM) is not monotonic (e.g. self-attn: 32,35,36,32,40,36,45,34 for
    // blockM=1024..4096), so scan the whole feasible range rather than binary-searching it.
    // Pure arithmetic (block_dims_fit does no device I/O beyond the one-time memoized
    // query_gmem_size), so this is microseconds even for M in the thousands.
    cl_uint best_bm = 0, best_bn = 0, best_launches = std::numeric_limits<cl_uint>::max();

    auto consider = [&](cl_uint cand_bm) {
        cl_uint cand_bn = max_bn_for_bm(cand_bm);
        if (cand_bn == 0) return; // infeasible at this blockM
        cl_uint m_blocks = (M + cand_bm - 1) / cand_bm;
        cl_uint n_blocks = (N0K1 + cand_bn - 1) / cand_bn;
        cl_uint launches = m_blocks * n_blocks;
        if (launches < best_launches) { // strict '<': keeps the SMALLEST blockM among ties,
            best_launches = launches;   // staying inside the empirically-measured range and
            best_bm = cand_bm;          // leaving more GMEM headroom
            best_bn = cand_bn;
        }
    };

    const cl_uint start = std::min(MIN_EFFICIENT_BLOCKM, M);
    const cl_uint scan_start = ((start + TILE - 1) / TILE) * TILE; // non-last m_blocks must be TILE-aligned
    for (cl_uint cand_bm = scan_start; cand_bm < M; cand_bm += TILE) {
        consider(cand_bm);
    }
    consider(M); // single m_block covering all of M: inherently "last", exempt from alignment

    if (best_bm == 0) {
        // Find the largest smaller M tile with a valid N block, not merely one that fits N=8.
        cl_uint cap = std::min(MIN_EFFICIENT_BLOCKM, M);
        cl_uint max_full_tiles = cap / TILE;
        if (max_full_tiles == 0 || max_bn_for_bm(TILE) == 0) return false;
        cl_uint lo = 1, hi = max_full_tiles;
        while (hi - lo > 1) {
            cl_uint mid = lo + (hi - lo) / 2;
            if (max_bn_for_bm(mid * TILE) != 0) lo = mid; else hi = mid;
        }
        best_bm = lo * TILE;
        best_bn = max_bn_for_bm(best_bm);
    }

    if (best_bm == 0 || best_bn == 0) return false;
    *bm = best_bm;
    *bn = best_bn;
    return true;
}

size_t FA_GMEM::get_hash(cl_device_id device_id, const cl_uint M, const cl_uint N0K1, const cl_uint totalK0N1,
    const cl_uint numHeads, ggml_type gtAct, ggml_type gtKV) {

    if (!cl_recordable_queue::supported(device_id)) {
        return 0;
    }
    size_t extension_size = 0;
    CL_CHECK(clGetDeviceInfo(device_id, CL_DEVICE_EXTENSIONS, 0, nullptr, &extension_size));
    std::vector<char> extensions(extension_size);
    CL_CHECK(clGetDeviceInfo(device_id, CL_DEVICE_EXTENSIONS, extensions.size(), extensions.data(), nullptr));
    if (std::string(extensions.data()).find("cl_qcom_onchip_global_memory") == std::string::npos) {
        return 0;
    }
    if (!(gtAct == GGML_TYPE_F16 || gtAct == GGML_TYPE_F32)) {
        return 0LL;
    }
    if (!(gtKV == GGML_TYPE_F16 || gtKV == GGML_TYPE_F32)) {
        return 0LL;
    }
    if (N0K1 > std::numeric_limits<cl_uint>::max() - 15) {
        return 0;
    }
    // grid element size MxN is 512x512, but underlying tile is 54x64
    // kernel is more efficient for larger K
    // N0K1 is rounded up to the next multiple of 16 (MM1's TILESIZE_K, since MM1 is
    // unconditionally FP16ACC/pkmad.f16 -- was 8 before that port, now stale) to support
    // non-aligned n_kv counts without MM1's K-loop overreading into never-written GMEM.
    const cl_uint N0K1_padded = (N0K1 + 15) & ~15u;
    if (N0K1_padded % 4) {
        return 0LL;
    }
    if (M < 64 || N0K1_padded < 64 || totalK0N1 < 64 || totalK0N1 % 8) {
        return 0LL;
    }
    const size_t max_bytes = std::numeric_limits<cl_uint>::max();
    const size_t native_q = gtAct == GGML_TYPE_F16 ? 2 : 4;
    const size_t native_kv = gtKV == GGML_TYPE_F16 ? 2 : 4;
    if (uint64_t(M) * totalK0N1 > max_bytes / native_q ||
        uint64_t(N0K1_padded) * totalK0N1 > max_bytes / native_kv ||
        uint64_t(M) * N0K1_padded > max_bytes / sizeof(float)) {
        return 0;
    }

    // some geometries are too big for GMEM
    cl_uint bm = 0;
    cl_uint bn = 0;
    if (!determine_block_dims(device_id, M, N0K1_padded, totalK0N1, gtAct, gtKV, &bm, &bn)) {
        return 0LL;
    }
    const cl_uint blockN0K1 = bn;
    const cl_uint n0k1_blocks = (N0K1_padded + blockN0K1 - 1) / blockN0K1;
    const cl_uint blockM = bm;
    const cl_uint m_blocks = (M + blockM - 1) / blockM;

    size_t hash = 0LL;
    if (gtAct == GGML_TYPE_F16) { hash |= (1 << 0); }
    if (gtKV == GGML_TYPE_F16) { hash |= (1 << 1); }
    hash |= ((size_t)m_blocks & 0xFFFF) << 2;
    hash |= ((size_t)n0k1_blocks & 0xFFFF) << 18;
    hash |= ((size_t)numHeads & 0xFFFF) << 34;

    return hash;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// FA_GMEM::kernel_group
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool FA_GMEM::kernel_group::init(cl_program program_mm0, cl_program program_softmax, cl_program program_mm1,
    cl_device_id device_id, ggml_type gtKV) {
    variant_suffix_mm0 = gtKV == GGML_TYPE_F16 ? "_af32_bf16" : "_af32_bf32";
    variant_suffix_mm1 = variant_suffix_mm0 + "_f16acc";

    cl_program programs[] = { program_mm0, program_softmax, program_softmax, program_softmax, program_softmax,
        program_softmax, program_mm1, program_softmax, program_softmax, program_softmax };
    const char * names[] = { "kernel_mul_mm_f16_f32_gmem", "copy_block_a_padded", "copy_block_b_padded",
        "online_softmax_max", "online_softmax_sum", "flash_attn_f32_f16_gmem",
        "kernel_mul_mm_f16_f32_gmem", "copy_xpose_block_b_padded", "copy_xpose_permute_block_c", "online_softmax_rescale_output" };
    cl_kernel * handles[] = { &kernel_mm0, &kernel_cp_pad_a0, &kernel_cp_pad_b0, &kernel_online_softmax_max,
        &kernel_online_softmax_sum, &kernel_online_softmax_output, &kernel_mm1, &kernel_cp_pad_b1, &kernel_cp_xpose_c1,
        &kernel_rescale_output };
    const cl_uint required_args[] = { 10, 4, 5, 8, 7, 9, 10, 6, 9, 6 };
    const size_t required_workgroups[] = { 256, 128, 64, fusedSoftmaxLocalM * fusedSoftmaxLocalN,
        fusedSoftmaxLocalM * fusedSoftmaxLocalN, 16, 256, 64, 64, 64 };
    const size_t count = sizeof(handles) / sizeof(handles[0]);
    for (size_t i = 0; i < count; ++i) {
        cl_int status = CL_SUCCESS;
        *handles[i] = clCreateKernel(programs[i], names[i], &status);
        if (status != CL_SUCCESS) { return false; }
        size_t max_workgroup = 0;
        cl_uint num_args = 0;
        if (clGetKernelWorkGroupInfo(*handles[i], device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(max_workgroup), &max_workgroup, nullptr) != CL_SUCCESS ||
            clGetKernelInfo(*handles[i], CL_KERNEL_NUM_ARGS, sizeof(num_args), &num_args, nullptr) != CL_SUCCESS ||
            max_workgroup < required_workgroups[i] || num_args != required_args[i]) {
            return false;
        }
    }
    return true;
}

std::string FA_GMEM::mm0_program_name(ggml_type gtKV) {
    return gtKV == GGML_TYPE_F16 ? "mm_gmem_f32_f16_acc32" : "mm_gmem_f32_f32_acc32";
}

std::string FA_GMEM::mm1_program_name(ggml_type gtKV) {
    return gtKV == GGML_TYPE_F16 ? "mm_gmem_f32_f16_acc16" : "mm_gmem_f32_f32_acc16";
}

FA_GMEM * FA_GMEM::create(cl_context context, cl_device_id device_id, cl_command_queue queue, ProfilingInfo & profiling_info,
    cl_program program_mm0, cl_program program_softmax, cl_program program_mm1, ggml_type gtKV) {
    kernel_map_key kmk(context, device_id, GGML_TYPE_F32, gtKV);
    auto it = kernel_map.find(kmk);
    if (it == kernel_map.end()) {
        auto * group = new kernel_group;
        if (!group->init(program_mm0, program_softmax, program_mm1, device_id, gtKV)) {
            delete group;
            group = nullptr;
        }
        it = kernel_map.emplace(kmk, group).first;
    }
    if (!it->second) { return nullptr; }
    auto * result = new FA_GMEM(context, device_id, profiling_info, gtKV);
    result->kernels = it->second;
    if (!result->rq.init(context, device_id, queue)) {
        delete result;
        return nullptr;
    }
    return result;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// FA_GMEM
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::map<cl_device_id, size_t> FA_GMEM::available_onchip_global_memory;
void FA_GMEM::query_gmem_size(cl_device_id device_id) {
    size_t & gmem_size = available_onchip_global_memory[device_id];
    if (gmem_size == 0) {
        size_t ext_str_size;

        clGetDeviceInfo(device_id, CL_DEVICE_EXTENSIONS, 0, nullptr, &ext_str_size);
        std::string ext_buffer(ext_str_size, '\0');
        clGetDeviceInfo(device_id, CL_DEVICE_EXTENSIONS, ext_str_size, ext_buffer.data(), nullptr);

        // query available GMEM
        if (ext_buffer.find("cl_qcom_onchip_global_memory") == std::string::npos) {
            GGML_LOG_ERROR("device does not support GMEM (cl_qcom_onchip_global_memory)\n");
            GGML_ASSERT(0);
        }
        CL_CHECK(clGetDeviceInfo(device_id, CL_DEVICE_ONCHIP_GLOBAL_MEM_SIZE_QCOM, sizeof(gmem_size),
            &gmem_size, nullptr));

        GGML_ASSERT(gmem_size);
        GGML_LOG_WARN("Available onchip global memory: %zu\n", gmem_size);
    }
}

FA_GMEM::FA_GMEM(cl_context context, cl_device_id device_id, ProfilingInfo& profiling_info,
    ggml_type gtKV)
    :
    m_Context(context),
    m_DeviceID(device_id),
#ifdef GGML_OPENCL_PROFILING
    m_ProfilingInfo(profiling_info),
#endif // GGML_OPENCL_PROFILING
    gtypeA(GGML_TYPE_F32),
    gtypeB(gtKV),
    fp16Q(gtypeA == GGML_TYPE_F16),
    fp16KV(gtypeB == GGML_TYPE_F16),
    sizeof_TQ(fp16Q ? 2 : 4),
    sizeof_TKV(fp16KV ? 2 : 4) {

#ifndef GGML_OPENCL_PROFILING
    GGML_UNUSED(profiling_info);
#endif // !GGML_OPENCL_PROFILING
}

// Divide large workload [C = M x N x K] into a grid of smaller blocks [blockC = blockM x blockN0K1 x blockK0N1]
// For each block, copy a portion of A and B into GMEM for faster access
void FA_GMEM::enqueue(
    cl_uint M, cl_uint N0K1, cl_uint K0N1,
    cl_uint numHeads,
    cl_mem mem_matrixQ, cl_ulong offsetQ,
    cl_mem mem_matrixK, cl_ulong offsetK,
    float scale,
    cl_mem mem_matrixV, cl_ulong offsetV,
    cl_mem mem_matrixO, cl_ulong offsetO,
    const ggml_tensor * tensorC) {

    cl_int status = CL_SUCCESS;
#ifdef GGML_OPENCL_PROFILING
    std::vector<cl_event> evts;
    evts.reserve(1024);
#else
    GGML_UNUSED(tensorC);
#endif // GGML_OPENCL_PROFILING

    const cl_uint totalM = M;
    const cl_uint totalN0K1_actual = N0K1;
    // padded to a multiple of 16 (MM1's TILESIZE_K under its unconditional FP16ACC/pkmad.f16
    // path) for FA_GMEM block sizing -- see get_hash's matching N0K1_padded computation above
    const cl_uint totalN0K1 = (N0K1 + 15) & ~15u;
    const cl_uint totalK0N1 = K0N1;
    constexpr cl_uint localSizeInTilesM = 2; // empirically determined
    constexpr cl_uint localSizeInTilesN = 2;

    // determine (hopefully) optimal block sizes
    cl_uint bm = 0;
    cl_uint bn = 0;
    bool fits = determine_block_dims(m_DeviceID, totalM, totalN0K1, totalK0N1, gtypeA, gtypeB, &bm, &bn);
    GGML_ASSERT(fits);
    const cl_uint blockM = bm;
    const cl_uint blockN0K1 = bn;

    cl_uint read_modify_write_X0 = false; // never tile K in first MM

    COMPUTE_STRIDE_BYTES_QKV(blockM, blockN0K1, totalK0N1, gtypeA, gtypeB);
    const size_t msizeX0_block_bytes = _msizeC0_block_bytes;
    const size_t msizeX1_block_bytes = _msizeA1_block_bytes;

    const cl_uint m_blocks = (totalM + blockM - 1) / blockM;
    const cl_uint blockM_last = (totalM % blockM) == 0 ? blockM : (totalM % blockM);
    const cl_uint n0k1_blocks = (totalN0K1 + blockN0K1 - 1) / blockN0K1;
    const cl_uint blockN0K1_last = (totalN0K1 % blockN0K1) == 0 ? blockN0K1 : (totalN0K1 % blockN0K1);

    const cl_uint lineStrideMatrixA0inElements = lineStrideMatrixA0inBytes / sizeof_TQ;
    const cl_uint lineStrideMatrixB0inElements = lineStrideMatrixB0inBytes / sizeof_TKV;
    const cl_uint lineStrideMatrixC0inElements = lineStrideMatrixC0inBytes / sizeof(float);

    const cl_uint lineStrideMatrixSMinElements = lineStrideMatrixSMinBytes / sizeof(float);

    // A1 (X0's softmax-output role) is always packed half on-chip, regardless of sizeof_TQ --
    // matches flash_attn_f32_f16_gmem's matrixX0_out now being half*, and mm_load_a's FP16ACC
    // read assuming packed-half-width strides.
    const cl_uint lineStrideMatrixA1inElements = lineStrideMatrixA1inBytes / 2;
    // B1 is always packed half on-chip (see sizeof_TB1_onchip in compute_stride_bytes_QKV):
    // copy_xpose_block_b_padded writes it through a half* dest, so its row stride is in
    // 2-byte elements regardless of gtKV.
    const cl_uint lineStrideMatrixB1inElements = lineStrideMatrixB1inBytes / 2;
    const cl_uint lineStrideMatrixC1inElements = lineStrideMatrixC1inBytes / sizeof(float);

    // Set global & local sizes
    const cl_uint globalSizeInTilesM = (blockM + 63) / 64;
    const cl_uint globalSizeInTilesN0K1 = (blockN0K1 + widthOfTileN - 1) / widthOfTileN; // MM0
    const cl_uint globalSizeInTilesK0N1 = (totalK0N1 + widthOfTileN - 1) / widthOfTileN; // MM1

    // matrixA0 aka Q (query)
    size_t gmem_offset = 0;
    cl_mem mem_matrixA0_block = nullptr;
    cl_mem mem_tex_matrixA0_1dbuf_block = nullptr;
    {
        // reusable scratch space in GMEM for matrix A block
        cl_mem_properties q_buffer_properties[5] = {CL_MEM_ONCHIP_GLOBAL_QCOM, 1, CL_MEM_ONCHIP_GLOBAL_OFFSET_QCOM, gmem_offset, 0};
        mem_matrixA0_block = clCreateBufferWithProperties(m_Context, q_buffer_properties,
            CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, msizeA0_block_bytes, nullptr, &status);
        CL_CHECK(status);

        // use image 1d buffer for matrix A block (padded)
        cl_image_format img_fmt_1d = {CL_R, static_cast<cl_channel_type>(fp16Q ? CL_HALF_FLOAT : CL_FLOAT)};
        cl_image_desc img_desc_1d = {};
        img_desc_1d.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc_1d.image_width = msizeA0_block_bytes / sizeof_TQ;
        img_desc_1d.buffer = mem_matrixA0_block;
        mem_tex_matrixA0_1dbuf_block = clCreateImage(m_Context, CL_MEM_READ_WRITE, &img_fmt_1d, &img_desc_1d, nullptr, &status);
        CL_CHECK(status);

        gmem_offset += msizeA0_block_bytes;
    }

    // matrixB aka K (keys) and V (values)
    cl_mem mem_matrixB_block = nullptr;
    cl_mem mem_tex_matrixB_1dbuf_block = nullptr;
    cl_mem mem_tex_matrixB1_1dbuf_block = nullptr;
    {
        // reusable scratch space in GMEM for matrix B block
        cl_mem_properties q_buffer_properties[5] = {CL_MEM_ONCHIP_GLOBAL_QCOM, 1, CL_MEM_ONCHIP_GLOBAL_OFFSET_QCOM, gmem_offset, 0};
        mem_matrixB_block = clCreateBufferWithProperties(m_Context, q_buffer_properties,
            CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, msizeB_block_bytes, nullptr, &status);
        CL_CHECK(status);

        // use image 1d buffer for matrix B block (padded)
        // This view is for mm0's B0 (K), which is fp32-accumulate: one texel per element.
        cl_image_format img_fmt_1d = {CL_R, static_cast<cl_channel_type>(fp16KV ? CL_HALF_FLOAT : CL_FLOAT)};
        cl_image_desc img_desc_1d = {};
        img_desc_1d.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc_1d.image_width = msizeB_block_bytes / sizeof_TKV;
        img_desc_1d.buffer = mem_matrixB_block;
        mem_tex_matrixB_1dbuf_block = clCreateImage(m_Context, CL_MEM_READ_WRITE, &img_fmt_1d, &img_desc_1d, nullptr, &status);
        CL_CHECK(status);

        // Second view on the SAME buffer for mm1's B1 (V). mm1 is unconditionally FP16ACC, so
        // mm_load_b issues packed reads (REG_B_DATATYPE uint8 = 8 words = 16 packed halves) and
        // needs raw 32-bit word texels -- CL_FLOAT sized by word count, NOT CL_HALF_FLOAT sized
        // by element count. Mirrors MM_GMEM.cpp's `(fp16B && !fp16_acc)` format selection and
        // its `bytes / (fp16_acc ? 4 : sizeof_TB)` width. This costs no extra GMEM (it aliases
        // mem_matrixB_block); B0 and B1 cannot share one view because they disagree on texel
        // width. Reusing the B0 view for mm1 yields NaN.
        cl_image_format img_fmt_1d_b1 = {CL_R, CL_FLOAT};
        cl_image_desc img_desc_1d_b1 = {};
        img_desc_1d_b1.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc_1d_b1.image_width = msizeB_block_bytes / 4;
        img_desc_1d_b1.buffer = mem_matrixB_block;
        mem_tex_matrixB1_1dbuf_block = clCreateImage(m_Context, CL_MEM_READ_WRITE, &img_fmt_1d_b1, &img_desc_1d_b1, nullptr, &status);
        CL_CHECK(status);

        gmem_offset += msizeB_block_bytes;
    }

    // matrixX0 aka QKt (intermediate)
    cl_mem mem_matrixX0_block = nullptr;
    cl_mem mem_tex_matrixX0_1dbuf_block = nullptr;
    {
        // reusable scratch space in GMEM for matrix X0 block
        cl_mem_properties q_buffer_properties[5] = {CL_MEM_ONCHIP_GLOBAL_QCOM, 1, CL_MEM_ONCHIP_GLOBAL_OFFSET_QCOM, gmem_offset, 0};
        mem_matrixX0_block = clCreateBufferWithProperties(m_Context, q_buffer_properties,
            CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, msizeX0_block_bytes, nullptr, &status);
        CL_CHECK(status);

        // use image 1d buffer for matrix C block (padded)
        cl_image_format img_fmt_1d = {CL_R, CL_FLOAT};
        cl_image_desc img_desc_1d = {};
        img_desc_1d.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc_1d.image_width = msizeX0_block_bytes / sizeof(float);
        img_desc_1d.buffer = mem_matrixX0_block;
        mem_tex_matrixX0_1dbuf_block = clCreateImage(m_Context, CL_MEM_READ_WRITE, &img_fmt_1d, &img_desc_1d, nullptr, &status);
        CL_CHECK(status);

        gmem_offset += msizeX0_block_bytes;
    }

    // matrixX1 aka softmax (intermediate)
    cl_mem mem_matrixX1_block = nullptr;
    cl_mem mem_tex_matrixX1_1dbuf_block = nullptr;
    {
        // Compact packed-half probability tile for MM1.
        cl_mem_properties q_buffer_properties[5] = {CL_MEM_ONCHIP_GLOBAL_QCOM, 1, CL_MEM_ONCHIP_GLOBAL_OFFSET_QCOM, gmem_offset, 0};
        mem_matrixX1_block = clCreateBufferWithProperties(m_Context, q_buffer_properties,
            CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, msizeX1_block_bytes, nullptr, &status);
        CL_CHECK(status);

        // use image 1d buffer for matrix A block (padded)
        cl_image_format img_fmt_1d = {CL_R, CL_FLOAT};
        cl_image_desc img_desc_1d = {};
        img_desc_1d.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc_1d.image_width = msizeX1_block_bytes / 4;
        img_desc_1d.buffer = mem_matrixX1_block;
        mem_tex_matrixX1_1dbuf_block = clCreateImage(m_Context, CL_MEM_READ_WRITE, &img_fmt_1d, &img_desc_1d, nullptr, &status);
        CL_CHECK(status);

        gmem_offset += msizeX1_block_bytes;
    }

    // matrixSM aka SoftMax per-row max / sum
    cl_mem mem_matrixSM_block = nullptr;
    {
        // reusable scratch space in GMEM for SM block
        cl_mem_properties q_buffer_properties[5] = {CL_MEM_ONCHIP_GLOBAL_QCOM, 1, CL_MEM_ONCHIP_GLOBAL_OFFSET_QCOM, gmem_offset, 0};
        mem_matrixSM_block = clCreateBufferWithProperties(m_Context, q_buffer_properties,
            CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, msizeSM_block_bytes, nullptr, &status);
        CL_CHECK(status);

        gmem_offset += msizeSM_block_bytes;
    }

    // matrixC1 aka Output (intermediate)
    cl_mem mem_matrixC1_block = nullptr;
    cl_mem mem_tex_matrixC1_1dbuf_block = nullptr;
    {
        // reusable scratch space in GMEM for matrix C1 block
        cl_mem_properties q_buffer_properties[5] = {CL_MEM_ONCHIP_GLOBAL_QCOM, 1, CL_MEM_ONCHIP_GLOBAL_OFFSET_QCOM, gmem_offset, 0};
        mem_matrixC1_block = clCreateBufferWithProperties(m_Context, q_buffer_properties,
            CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, msizeC1_block_bytes, nullptr, &status);
        CL_CHECK(status);

        // use image 1d buffer for matrix C block (padded)
        cl_image_format img_fmt_1d = {CL_R, CL_FLOAT};
        cl_image_desc img_desc_1d = {};
        img_desc_1d.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        img_desc_1d.image_width = msizeC1_block_bytes / sizeof(float);
        img_desc_1d.buffer = mem_matrixC1_block;
        mem_tex_matrixC1_1dbuf_block = clCreateImage(m_Context, CL_MEM_READ_WRITE, &img_fmt_1d, &img_desc_1d, nullptr, &status);
        CL_CHECK(status);

        gmem_offset += msizeC1_block_bytes;
        GGML_UNUSED(gmem_offset);
    }

    // matrixO aka dst (output)
    cl_mem mem_matrixO_sub = nullptr;
    {
        cl_buffer_region region;
        region.origin = offsetO;
        region.size = numHeads * totalM * totalK0N1 * sizeof(float);
        mem_matrixO_sub = clCreateSubBuffer(mem_matrixO, CL_MEM_WRITE_ONLY | CL_MEM_HOST_NO_ACCESS, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
        CL_CHECK(status);
    }

    // outer loop over heads
    for (cl_uint headNum = 0; headNum < numHeads; headNum++) {

        // loop over M blocks first to facilitate row-based softmax
        for (cl_uint m_block = 0; m_block < m_blocks; ++m_block)
        {
            cl_uint blockM_loop = (m_block == (m_blocks - 1)) ? blockM_last : blockM;

            // copy (and pad) one block of matrixA aka Q (query)
            cl_mem mem_matrixQ_sub = nullptr;
            {
                // Q is unpadded
                cl_buffer_region region;
                size_t head_offset = headNum * totalM * totalK0N1 * sizeof_TQ;
                size_t m_block_offset = m_block * blockM * totalK0N1 * sizeof_TQ;
                region.origin = offsetQ + head_offset + m_block_offset;
                region.size = (blockM_loop * totalK0N1 * sizeof_TQ);
                mem_matrixQ_sub = clCreateSubBuffer(mem_matrixQ, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                CL_CHECK(status);

                // Set the arguments to our pad kernel
                cl_uint arg_index = 0;
                rq.set_arg(kernels->kernel_cp_pad_a0, arg_index++, sizeof(cl_mem), static_cast<const void*>(&mem_matrixQ_sub));
                rq.set_arg(kernels->kernel_cp_pad_a0, arg_index++, sizeof(cl_mem), static_cast<const void*>(&mem_matrixA0_block));
                rq.set_arg(kernels->kernel_cp_pad_a0, arg_index++, sizeof(cl_uint), &totalK0N1);
                rq.set_arg(kernels->kernel_cp_pad_a0, arg_index++, sizeof(cl_uint), &lineStrideMatrixA0inElements);
                GGML_UNUSED(arg_index);

                GGML_ASSERT(totalK0N1 % 8 == 0);
                std::vector<size_t> global_size = {totalK0N1/8, blockM_loop};
                std::vector<size_t> local_size = {16, 8};
                rq.record(kernels->kernel_cp_pad_a0, global_size, local_size);
            }

            // these are recorded by reference, so need to stay in scope until the recordable queue is submitted
            std::vector<cl_uint> blockN0K1_loop(n0k1_blocks);
            std::vector<cl_uint> blockN0K1_actual_loop(n0k1_blocks);
            std::vector<cl_mem> mem_matrixK_sub(n0k1_blocks, nullptr);
            std::vector<cl_mem> mem_matrixV_sub(n0k1_blocks, nullptr);
            std::vector<cl_uint> init_per_row_data(n0k1_blocks);
            std::vector<cl_uint> read_modify_write_C1(n0k1_blocks);

            // loop over N0K1 blocks next
            for (cl_uint n0k1_block = 0; n0k1_block < n0k1_blocks; ++n0k1_block)
            {
                blockN0K1_loop[n0k1_block] = (n0k1_block == (n0k1_blocks - 1)) ? blockN0K1_last : blockN0K1;
                {
                    int start = (int)n0k1_block * (int)blockN0K1;
                    int actual = (int)totalN0K1_actual - start;
                    blockN0K1_actual_loop[n0k1_block] = (cl_uint)std::min((int)blockN0K1_loop[n0k1_block], std::max(0, actual));
                }

                // copy (and pad) one block of matrixB aka K (keys)
                {
                    // K is unpadded; head stride uses actual n_kv so offset lands in the real K data
                    cl_buffer_region region;
                    size_t head_offset = headNum * totalN0K1_actual * totalK0N1 * sizeof_TKV;
                    size_t n0k1_block_offset = n0k1_block * blockN0K1 * totalK0N1 * sizeof_TKV;
                    region.origin = offsetK + head_offset + n0k1_block_offset;
                    region.size = (blockN0K1_actual_loop[n0k1_block] * totalK0N1 * sizeof_TKV);
                    mem_matrixK_sub[n0k1_block] = clCreateSubBuffer(mem_matrixK, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                    CL_CHECK(status);

                    // Set the arguments to our pad kernel
                    cl_uint arg_index = 0;
                    rq.set_arg(kernels->kernel_cp_pad_b0, arg_index++, sizeof(cl_mem), static_cast<const void*>(&mem_matrixK_sub[n0k1_block]));
                    rq.set_arg(kernels->kernel_cp_pad_b0, arg_index++, sizeof(cl_uint), &totalK0N1);
                    rq.set_arg(kernels->kernel_cp_pad_b0, arg_index++, sizeof(cl_mem), static_cast<void*>(&mem_matrixB_block));
                    rq.set_arg(kernels->kernel_cp_pad_b0, arg_index++, sizeof(cl_uint), &lineStrideMatrixB0inElements);
                    rq.set_arg(kernels->kernel_cp_pad_b0, arg_index++, sizeof(cl_uint), &blockN0K1_actual_loop[n0k1_block]);
                    GGML_UNUSED(arg_index);

                    GGML_ASSERT(totalK0N1 % 8 == 0);
                    // dispatch for full padded row count; kernel zero-fills rows >= actual_rows
                    std::vector<size_t> global_size = {totalK0N1/8, blockN0K1_loop[n0k1_block]};
                    std::vector<size_t> local_size = {16, 4};
                    rq.record(kernels->kernel_cp_pad_b0, global_size, local_size);
                }

                // MM0
                {
                    // Set the arguments to our compute kernel
                    cl_uint arg_index = 0;
                    rq.set_arg(kernels->kernel_mm0, arg_index++, sizeof(cl_mem),  static_cast<const void*>(&mem_tex_matrixA0_1dbuf_block));
                    rq.set_arg(kernels->kernel_mm0, arg_index++, sizeof(cl_mem),  static_cast<const void*>(&mem_tex_matrixB_1dbuf_block));
                    rq.set_arg(kernels->kernel_mm0, arg_index++, sizeof(cl_mem),  static_cast<void*>(&mem_tex_matrixX0_1dbuf_block));
                    rq.set_arg(kernels->kernel_mm0, arg_index++, sizeof(cl_uint), &read_modify_write_X0);
                    rq.set_arg(kernels->kernel_mm0, arg_index++, sizeof(cl_uint), &blockM_loop);
                    rq.set_arg(kernels->kernel_mm0, arg_index++, sizeof(cl_uint), &blockN0K1_loop[n0k1_block]);
                    rq.set_arg(kernels->kernel_mm0, arg_index++, sizeof(cl_uint), &totalK0N1);
                    rq.set_arg(kernels->kernel_mm0, arg_index++, sizeof(cl_uint), &lineStrideMatrixA0inBytes);
                    rq.set_arg(kernels->kernel_mm0, arg_index++, sizeof(cl_uint), &lineStrideMatrixB0inBytes);
                    rq.set_arg(kernels->kernel_mm0, arg_index++, sizeof(cl_uint), &lineStrideMatrixC0inBytes);
                    GGML_UNUSED(arg_index);

                    std::vector<size_t> global_size = {64, globalSizeInTilesM, globalSizeInTilesN0K1}; // 1 block = 64x64 output and each block is 64fibers (1 wave)
                    if (m_block == (m_blocks - 1)) {
                        global_size[1] = (blockM_loop + 63) / 64;
                    }
                    if (n0k1_block == (n0k1_blocks - 1)) {
                        global_size[2] = (blockN0K1_loop[n0k1_block] + widthOfTileN - 1) / widthOfTileN;
                    }
                    std::vector<size_t> local_size = {64, localSizeInTilesM, localSizeInTilesN};
                    rq.record(kernels->kernel_mm0, global_size, local_size);
                }

                // per-row (online) softmax
                {
                    // Initialize at start of row; else rotate at start of block
                    init_per_row_data[n0k1_block] = n0k1_block == 0;

                    // pad global_size(0) up to a fusedSoftmaxLocalM multiple; the kernel guards
                    // padding row-groups (m_4 >= ceil(blockM_loop/4)) via its "valid" check
                    const cl_uint num_row_groups = (blockM_loop + 3) / 4;
                    const size_t global_m = ((num_row_groups + fusedSoftmaxLocalM - 1) / fusedSoftmaxLocalM) * fusedSoftmaxLocalM;
                    std::vector<size_t> global_size = {global_m, fusedSoftmaxLocalN};
                    std::vector<size_t> local_size = {fusedSoftmaxLocalM, fusedSoftmaxLocalN};

                    // X0 -> scale and max reduction -> SM.m_i
                    {
                        cl_uint arg_index = 0;
                        rq.set_arg(kernels->kernel_online_softmax_max, arg_index++, sizeof(cl_mem),  static_cast<const void*>(&mem_matrixX0_block));
                        rq.set_arg(kernels->kernel_online_softmax_max, arg_index++, sizeof(cl_uint), &lineStrideMatrixC0inElements);
                        rq.set_arg(kernels->kernel_online_softmax_max, arg_index++, sizeof(float),   &scale);
                        rq.set_arg(kernels->kernel_online_softmax_max, arg_index++, sizeof(cl_uint), &blockM_loop);
                        rq.set_arg(kernels->kernel_online_softmax_max, arg_index++, sizeof(cl_uint), &blockN0K1_actual_loop[n0k1_block]);
                        rq.set_arg(kernels->kernel_online_softmax_max, arg_index++, sizeof(cl_mem),  static_cast<void*>(&mem_matrixSM_block));
                        rq.set_arg(kernels->kernel_online_softmax_max, arg_index++, sizeof(cl_uint), &lineStrideMatrixSMinElements);
                        rq.set_arg(kernels->kernel_online_softmax_max, arg_index++, sizeof(cl_uint), &init_per_row_data[n0k1_block]);
                        GGML_UNUSED(arg_index);

                        rq.record(kernels->kernel_online_softmax_max, global_size, local_size);
                    }

                    // compute (online) sum per row, given SM.m_i from the fused pass above
                    {
                        cl_uint arg_index = 0;
                        rq.set_arg(kernels->kernel_online_softmax_sum, arg_index++, sizeof(cl_mem),  static_cast<const void*>(&mem_matrixX0_block));
                        rq.set_arg(kernels->kernel_online_softmax_sum, arg_index++, sizeof(cl_uint), &lineStrideMatrixC0inElements);
                        rq.set_arg(kernels->kernel_online_softmax_sum, arg_index++, sizeof(float), &scale);
                        rq.set_arg(kernels->kernel_online_softmax_sum, arg_index++, sizeof(cl_uint), &blockM_loop);
                        rq.set_arg(kernels->kernel_online_softmax_sum, arg_index++, sizeof(cl_uint), &blockN0K1_actual_loop[n0k1_block]);
                        rq.set_arg(kernels->kernel_online_softmax_sum, arg_index++, sizeof(cl_mem),  static_cast<void*>(&mem_matrixSM_block));
                        rq.set_arg(kernels->kernel_online_softmax_sum, arg_index++, sizeof(cl_uint), &lineStrideMatrixSMinElements);
                        GGML_UNUSED(arg_index);

                        rq.record(kernels->kernel_online_softmax_sum, global_size, local_size);
                    }

                    // write (and xpose) softmax to A1
                    {
                        cl_uint arg_index = 0;
                        rq.set_arg(kernels->kernel_online_softmax_output, arg_index++, sizeof(cl_mem),  static_cast<const void*>(&mem_matrixX0_block));
                        rq.set_arg(kernels->kernel_online_softmax_output, arg_index++, sizeof(cl_uint), &lineStrideMatrixC0inElements);
                        rq.set_arg(kernels->kernel_online_softmax_output, arg_index++, sizeof(float), &scale);
                        rq.set_arg(kernels->kernel_online_softmax_output, arg_index++, sizeof(cl_mem),  static_cast<const void*>(&mem_matrixX1_block));
                        rq.set_arg(kernels->kernel_online_softmax_output, arg_index++, sizeof(cl_uint), &lineStrideMatrixA1inElements);
                        rq.set_arg(kernels->kernel_online_softmax_output, arg_index++, sizeof(cl_uint), &blockM_loop);
                        rq.set_arg(kernels->kernel_online_softmax_output, arg_index++, sizeof(cl_uint), &blockN0K1_actual_loop[n0k1_block]);
                        rq.set_arg(kernels->kernel_online_softmax_output, arg_index++, sizeof(cl_mem),  static_cast<const void*>(&mem_matrixSM_block));
                        rq.set_arg(kernels->kernel_online_softmax_output, arg_index++, sizeof(cl_uint), &lineStrideMatrixSMinElements);
                        GGML_UNUSED(arg_index);

                        // MM1 reads 16-wide K chunks; zero-fill probabilities through the padded block boundary.
                        std::vector<size_t> global_size = {(blockM_loop + 3)/4, (blockN0K1_loop[n0k1_block] + 3)/4};
                        std::vector<size_t> local_size = {4, 4};
                        rq.record(kernels->kernel_online_softmax_output, global_size, local_size);
                    }
                }

                // copy (and pad) one block of matrixB aka V (values)
                {
                    // V is unpadded, but N-first; head stride uses actual n_kv so offset lands in the real V data
                    cl_buffer_region region;
                    size_t head_offset = headNum * totalN0K1_actual * totalK0N1 * sizeof_TKV;
                    size_t n0k1_block_offset = n0k1_block * blockN0K1 * totalK0N1 * sizeof_TKV;
                    region.origin = offsetV + head_offset + n0k1_block_offset;
                    region.size = (blockN0K1_actual_loop[n0k1_block] * totalK0N1 * sizeof_TKV);
                    mem_matrixV_sub[n0k1_block] = clCreateSubBuffer(mem_matrixV, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                    CL_CHECK(status);

                    // Set the arguments to our pad kernel
                    cl_uint arg_index = 0;
                    rq.set_arg(kernels->kernel_cp_pad_b1, arg_index++, sizeof(cl_mem), static_cast<const void*>(&mem_matrixV_sub[n0k1_block]));
                    rq.set_arg(kernels->kernel_cp_pad_b1, arg_index++, sizeof(cl_uint), &totalK0N1);
                    rq.set_arg(kernels->kernel_cp_pad_b1, arg_index++, sizeof(cl_mem), static_cast<const void*>(&mem_matrixB_block));
                    rq.set_arg(kernels->kernel_cp_pad_b1, arg_index++, sizeof(cl_uint), &lineStrideMatrixB1inElements);
                    rq.set_arg(kernels->kernel_cp_pad_b1, arg_index++, sizeof(cl_uint), &totalK0N1);
                    rq.set_arg(kernels->kernel_cp_pad_b1, arg_index++, sizeof(cl_uint), &blockN0K1_actual_loop[n0k1_block]);
                    GGML_UNUSED(arg_index);

                    GGML_ASSERT(totalK0N1 % 4 == 0);
                    // MM1 reads 16-wide K chunks; V staging must zero-fill the same tail as the probabilities.
                    std::vector<size_t> global_size = {(blockN0K1_loop[n0k1_block]+3)/4, totalK0N1/4};
                    std::vector<size_t> local_size = {16, 4};
                    rq.record(kernels->kernel_cp_pad_b1, global_size, local_size);
                }

                // Rescale previous output before ordinary MM1 adds the next block.
                if (n0k1_block != 0) {
                    cl_uint arg_index = 0;
                    rq.set_arg(kernels->kernel_rescale_output, arg_index++, sizeof(cl_mem), &mem_matrixC1_block);
                    rq.set_arg(kernels->kernel_rescale_output, arg_index++, sizeof(cl_uint), &lineStrideMatrixC1inElements);
                    rq.set_arg(kernels->kernel_rescale_output, arg_index++, sizeof(cl_mem), &mem_matrixSM_block);
                    rq.set_arg(kernels->kernel_rescale_output, arg_index++, sizeof(cl_uint), &lineStrideMatrixSMinElements);
                    rq.set_arg(kernels->kernel_rescale_output, arg_index++, sizeof(cl_uint), &blockM_loop);
                    rq.set_arg(kernels->kernel_rescale_output, arg_index++, sizeof(cl_uint), &totalK0N1);
                    rq.record(kernels->kernel_rescale_output, {(blockM_loop + 3)/4, totalK0N1}, {16, 4});
                }

                // MM1
                {
                    read_modify_write_C1[n0k1_block] = n0k1_block == 0 ? 0 : 1;

                    // Set the arguments to our compute kernel
                    cl_uint arg_index = 0;
                    rq.set_arg(kernels->kernel_mm1, arg_index++, sizeof(cl_mem),  static_cast<const void*>(&mem_tex_matrixX1_1dbuf_block));
                    rq.set_arg(kernels->kernel_mm1, arg_index++, sizeof(cl_mem),  static_cast<const void*>(&mem_tex_matrixB1_1dbuf_block));
                    rq.set_arg(kernels->kernel_mm1, arg_index++, sizeof(cl_mem),  static_cast<const void*>(&mem_tex_matrixC1_1dbuf_block));
                    rq.set_arg(kernels->kernel_mm1, arg_index++, sizeof(cl_uint), &read_modify_write_C1[n0k1_block]);
                    rq.set_arg(kernels->kernel_mm1, arg_index++, sizeof(cl_uint), &blockM_loop);
                    rq.set_arg(kernels->kernel_mm1, arg_index++, sizeof(cl_uint), &totalK0N1);
                    rq.set_arg(kernels->kernel_mm1, arg_index++, sizeof(cl_uint), &blockN0K1_actual_loop[n0k1_block]);
                    rq.set_arg(kernels->kernel_mm1, arg_index++, sizeof(cl_uint), &lineStrideMatrixA1inBytes);
                    rq.set_arg(kernels->kernel_mm1, arg_index++, sizeof(cl_uint), &lineStrideMatrixB1inBytes);
                    rq.set_arg(kernels->kernel_mm1, arg_index++, sizeof(cl_uint), &lineStrideMatrixC1inBytes);
                    GGML_UNUSED(arg_index);

                    std::vector<size_t> global_size = {64, globalSizeInTilesM, globalSizeInTilesK0N1}; // 1 block = 64x64 output and each block is 64fibers (1 wave)
                    if (m_block == (m_blocks - 1)) {
                        global_size[1] = (blockM_loop + 63) / 64;
                    }
                    std::vector<size_t> local_size = {64, localSizeInTilesM, localSizeInTilesN};
                    rq.record(kernels->kernel_mm1, global_size, local_size);
                }

            } // n_blocks

            // matrixO aka dst (output)
            {
                // copy, xpose, and permute C1 to Output
                cl_uint arg_index = 0;
                rq.set_arg(kernels->kernel_cp_xpose_c1, arg_index++, sizeof(cl_mem),  static_cast<const void*>(&mem_matrixC1_block));
                rq.set_arg(kernels->kernel_cp_xpose_c1, arg_index++, sizeof(cl_uint), &lineStrideMatrixC1inElements);
                rq.set_arg(kernels->kernel_cp_xpose_c1, arg_index++, sizeof(cl_mem),  static_cast<void*>(&mem_matrixO_sub));
                rq.set_arg(kernels->kernel_cp_xpose_c1, arg_index++, sizeof(cl_uint), &totalK0N1);
                rq.set_arg(kernels->kernel_cp_xpose_c1, arg_index++, sizeof(cl_uint), &blockM);
                rq.set_arg(kernels->kernel_cp_xpose_c1, arg_index++, sizeof(cl_uint), &m_block);
                rq.set_arg(kernels->kernel_cp_xpose_c1, arg_index++, sizeof(cl_uint), &blockM_loop);
                rq.set_arg(kernels->kernel_cp_xpose_c1, arg_index++, sizeof(cl_uint), &headNum);
                rq.set_arg(kernels->kernel_cp_xpose_c1, arg_index++, sizeof(cl_uint), &numHeads);
                GGML_UNUSED(arg_index);

                GGML_ASSERT(totalK0N1 % 4 == 0);
                std::vector<size_t> global_size = {(blockM_loop+3)/4, totalK0N1/4};
                std::vector<size_t> local_size = {16, 4};
                rq.record(kernels->kernel_cp_xpose_c1, global_size, local_size);
            }

#ifdef GGML_OPENCL_PROFILING
            cl_event event = nullptr;
            rq.enqueue(&event);
            if (event) {
                evts.push_back(event);
            }
#else
            rq.enqueue(nullptr);
#endif // GGML_OPENCL_PROFILING

            // now that the recordable queue has been submitted, all the referenced
            // pointers can be released
            for (cl_uint i = 0; i < n0k1_blocks; ++i) {
                clReleaseMemObject(mem_matrixV_sub[i]);
                clReleaseMemObject(mem_matrixK_sub[i]);
            }
            clReleaseMemObject(mem_matrixQ_sub);

        } // m_blocks

    } // numHeads

    clReleaseMemObject(mem_matrixO_sub);

    // release GMEM resources
    clReleaseMemObject(mem_tex_matrixC1_1dbuf_block);
    clReleaseMemObject(mem_matrixC1_block);
    clReleaseMemObject(mem_tex_matrixX1_1dbuf_block);
    clReleaseMemObject(mem_matrixX1_block);

    clReleaseMemObject(mem_matrixSM_block);

    clReleaseMemObject(mem_tex_matrixX0_1dbuf_block);
    clReleaseMemObject(mem_matrixX0_block);
    clReleaseMemObject(mem_tex_matrixB1_1dbuf_block);
    clReleaseMemObject(mem_tex_matrixB_1dbuf_block);
    clReleaseMemObject(mem_matrixB_block);
    clReleaseMemObject(mem_tex_matrixA0_1dbuf_block);
    clReleaseMemObject(mem_matrixA0_block);

#ifdef GGML_OPENCL_PROFILING
    cl_uint global_blocks_m = (totalM + 63) / 64;
    cl_uint global_blocks_n = (totalN0K1 + widthOfTileN - 1) / widthOfTileN;
    size_t global_size[3] = {64, global_blocks_m, global_blocks_n};
    size_t local_size[3] = {64, localSizeInTilesM, localSizeInTilesN};
    m_ProfilingInfo.populate(evts, "fa_gmem" + kernels->variant_suffix_mm0 + kernels->variant_suffix_mm1, 3, global_size, local_size, tensorC);
    for (auto *evt : evts) {
        clReleaseEvent(evt);
    }
    evts.clear();
#endif // GGML_OPENCL_PROFILING
}

#endif
