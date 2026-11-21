#if defined(GGML_OPENCL_USE_ADRENO_KERNELS) && defined(GGML_OPENCL_USE_ADRENO_BIN_KERNELS) && GGML_OPENCL_TARGET_VERSION >= 300

#include <algorithm>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif // CL_TARGET_OPENCL_VERSION
#include <CL/cl.h>
#include <CL/cl_platform.h>
#include <CL/cl_ext_qcom.h>

#include "MM_GMEM.h"
#include "ggml.h"
#include "ggml-impl.h"
#include "opencl_profiling_info.h"


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// static MM_GMEM
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// MM_GMEM follows the OpenCL backend's serialized lifetime model; this cache is not thread-safe.
std::map<MM_GMEM::kernel_map_key, MM_GMEM::kernel_group *> MM_GMEM::kernel_map;
void MM_GMEM::cleanup(cl_context context) {
    // Device capability caches are context-independent and remain valid for the device lifetime.
    for (auto it = kernel_map.begin(); it != kernel_map.end(); ) {
        if (std::get<0>(it->first) == context) {
            delete it->second;
            it = kernel_map.erase(it);
        } else {
            ++it;
        }
    }
}

std::tuple<cl_uint, cl_uint> MM_GMEM::compute_stride_bytes_AB(
    const cl_uint K, ggml_type gtA, ggml_type gtB, bool fp16_acc) {

    cl_uint mA_paddedK = K;
    cl_uint mB_paddedK = K;

    // A and B are both narrowed to packed half on-chip under FP16ACC regardless of native
    // storage (see copy_block_a_padded / copy_block_b_padded). Q8_0 A is dequantized to
    // half on-chip unconditionally, so it is 2 either way.
    const bool fp16A = (gtA == GGML_TYPE_F16) || is_q8_0(gtA);
    const bool fp16B = (gtB == GGML_TYPE_F16);
    const size_t sizeof_TA = (fp16A || fp16_acc) ? 2 : 4;
    const size_t sizeof_TB = (fp16B || fp16_acc) ? 2 : 4;

    // 512bits is one line in the CCHE (512bits/8bits = 64bytes)
    // The copy stage always restores a K-first on-chip A tile.
    mA_paddedK += (mA_paddedK % (128 / sizeof_TA)) ? 0 : 64 / sizeof_TA;
    // independent of src1 tensor layout, the block B layout is always K-first
    mB_paddedK += (mB_paddedK % (128 / sizeof_TB)) ? 0 : 64 / sizeof_TB;

    cl_uint lineStrideMatrixAinBytes = sizeof_TA * mA_paddedK;
    cl_uint lineStrideMatrixBinBytes = sizeof_TB * mB_paddedK;

    return {lineStrideMatrixAinBytes, lineStrideMatrixBinBytes};
}

bool MM_GMEM::block_dims_fit(cl_device_id device_id,
    const cl_uint blockM, const cl_uint blockN, const cl_uint blockK, const cl_uint full_M,
    ggml_type gtA, ggml_type gtB, bool fp16_acc) {

    auto stridesAB = compute_stride_bytes_AB(blockK, gtA, gtB, fp16_acc);
    const cl_uint lineStrideMatrixAinBytes = std::get<0>(stridesAB);
    const cl_uint lineStrideMatrixBinBytes = std::get<1>(stridesAB);

    const uint64_t msizeA_block_bytes = uint64_t(blockM) * lineStrideMatrixAinBytes;
    const uint64_t msizeB_block_bytes = uint64_t(blockN) * lineStrideMatrixBinBytes;

    query_gmem_size(device_id);
    if ((msizeA_block_bytes + msizeB_block_bytes) > available_onchip_global_memory.at(device_id)) {
        return false;
    }

    query_max_image_elements(device_id);
    if (uint64_t(blockN) * full_M > max_image_elements.at(device_id)) {
        return false;
    }

    return true;
}

std::map<cl_device_id, size_t> MM_GMEM::mem_base_addr_align_bytes;
size_t MM_GMEM::query_mem_base_addr_align_bytes(cl_device_id device_id) {
    size_t & align_bytes = mem_base_addr_align_bytes[device_id];
    if (align_bytes == 0) {
        cl_uint align_bits = 0;
        CL_CHECK(clGetDeviceInfo(device_id, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(align_bits), &align_bits, nullptr));
        GGML_ASSERT(align_bits % 8 == 0);
        align_bytes = align_bits / 8;
    }
    return align_bytes;
}

bool MM_GMEM::determine_block_dims(cl_device_id device_id,
    const cl_uint M, const cl_uint N, const cl_uint K,
    ggml_type gtA, ggml_type gtB,
    cl_uint * bm, cl_uint * bn, cl_uint * bk, bool fp16_acc) {

    // kernel assumption: the mm K-loop steps by TILESIZE_K (8 for mad.f32, 16 for the
    // packed pkmad.f16 / FP16ACC path) with no partial-K-tile handling, so blockK must
    // be a multiple of that step or the trailing K remainder is silently dropped. Q8_0 A
    // tightens that to a whole quant block (32) so every blockK boundary is also a scale
    // boundary -- see copy_block_a_padded's k>>2 scale index.
    const cl_uint kmult = k_multiple(gtA, fp16_acc);
    GGML_ASSERT(K % kmult == 0);
    constexpr cl_uint TILE = static_cast<cl_uint>(widthOfTileN);

    // Native DDR width per element (NOT the on-chip packed-half width compute_stride_bytes_AB
    // uses) -- this is what's actually streamed from DDR, so it's what the cost model below is
    // built on. Mirrors the sizeof_TA/sizeof_TB member init in the constructor; duplicated here
    // since this function is static.
    const size_t sizeof_TA_native = is_q8_0(gtA) ? 1 : ((gtA == GGML_TYPE_F16) ? 2 : 4);
    const size_t sizeof_TB_native = (gtB == GGML_TYPE_F16) ? 2 : 4;
    const double A_total = double(M) * K * sizeof_TA_native;
    const double B_total = double(N) * K * sizeof_TB_native;

    // Each non-last K block starts an aligned sub-buffer for B and non-Q8 A.
    const size_t align_bytes = query_mem_base_addr_align_bytes(device_id);
    auto gcd_ = [](size_t a, size_t b) { while (b) { size_t t = a % b; a = b; b = t; } return a; };
    auto lcm_ = [&](size_t a, size_t b) { return a / gcd_(a, b) * b; };
    auto k_align_elems = [&](size_t sizeof_X) -> size_t { return align_bytes / gcd_(align_bytes, sizeof_X); };
    size_t kmult_aligned_sz = kmult;
    kmult_aligned_sz = lcm_(kmult_aligned_sz, k_align_elems(sizeof_TB_native));
    if (!is_q8_0(gtA)) {
        kmult_aligned_sz = lcm_(kmult_aligned_sz, k_align_elems(sizeof_TA_native));
    }
    const cl_uint kmult_aligned = static_cast<cl_uint>(kmult_aligned_sz);

    // Full K is faster for this measured FFN-up shape; other shapes use the generic selector.
    if (is_q8_0(gtA) && gtB == GGML_TYPE_F32 && fp16_acc &&
        M == 14336 && N == 8190 && K == 3072) {
        struct FullKCandidate {
            cl_uint bm = 0;
            cl_uint bn = 0;
            uint64_t pairs = UINT64_MAX;
            cl_uint outer_blocks = UINT_MAX;
        } best_full_k;

        auto max_bn_for_full_k = [&](cl_uint bm_) -> cl_uint {
            if (block_dims_fit(device_id, bm_, N, K, M, gtA, gtB, fp16_acc)) return N;
            const cl_uint max_full_tiles = N / TILE;
            if (max_full_tiles == 0 ||
                !block_dims_fit(device_id, bm_, TILE, K, M, gtA, gtB, fp16_acc)) return 0;

            cl_uint lo = 1;
            cl_uint hi = max_full_tiles;
            while (hi - lo > 1) {
                const cl_uint mid = lo + (hi - lo) / 2;
                if (block_dims_fit(device_id, bm_, mid * TILE, K, M, gtA, gtB, fp16_acc)) lo = mid; else hi = mid;
            }
            return lo * TILE;
        };

        auto consider_full_k = [&](cl_uint bm_) {
            const cl_uint bn_ = max_bn_for_full_k(bm_);
            if (bn_ == 0) return;
            const cl_uint m_blocks = (M + bm_ - 1) / bm_;
            const cl_uint n_blocks = (N + bn_ - 1) / bn_;
            const uint64_t pairs = uint64_t(m_blocks) * n_blocks;
            const cl_uint outer_blocks = std::min(m_blocks, n_blocks);
            if (pairs < best_full_k.pairs ||
                (pairs == best_full_k.pairs && outer_blocks < best_full_k.outer_blocks) ||
                (pairs == best_full_k.pairs && outer_blocks == best_full_k.outer_blocks && bn_ > best_full_k.bn)) {
                best_full_k = {bm_, bn_, pairs, outer_blocks};
            }
        };

        for (cl_uint bm_ = TILE; bm_ < M; bm_ += TILE) consider_full_k(bm_);
        consider_full_k(M);

        if (best_full_k.bm != 0) {
            *bm = best_full_k.bm;
            *bn = best_full_k.bn;
            *bk = K;
            return true;
        }
    }

    // The outer operand is copied once; the inner operand is copied once per outer block.
    // Larger K blocks reduce C read/modify/write traffic.

    // Search aligned K blocks. A full-K block does not need an aligned next origin.
    auto max_bk_for = [&](cl_uint bm_, cl_uint bn_) -> cl_uint {
        if (block_dims_fit(device_id, bm_, bn_, K, M, gtA, gtB, fp16_acc)) return K; // whole K: exempt
        if (kmult_aligned > K || !block_dims_fit(device_id, bm_, bn_, kmult_aligned, M, gtA, gtB, fp16_acc)) return 0;
        cl_uint max_full = K / kmult_aligned;
        cl_uint lo = 1, hi = max_full;
        while (hi - lo > 1) {
            cl_uint mid = lo + (hi - lo) / 2;
            if (block_dims_fit(device_id, bm_, bn_, mid * kmult_aligned, M, gtA, gtB, fp16_acc)) lo = mid; else hi = mid;
        }
        return lo * kmult_aligned;
    };

    // Search whole tiles unless the full dimension fits in one block.
    auto max_size_for = [&](cl_uint total, auto && probe) -> cl_uint {
        if (probe(total)) return total;
        cl_uint max_full_tiles = total / TILE;
        if (max_full_tiles == 0 || !probe(TILE)) return 0;
        cl_uint lo = 1, hi = max_full_tiles;
        while (hi - lo > 1) {
            cl_uint mid = lo + (hi - lo) / 2;
            if (probe(mid * TILE)) lo = mid; else hi = mid;
        }
        return lo * TILE;
    };

    struct Candidate { cl_uint bm = 0, bn = 0, bk = 0; double cost = 0.0; bool ok = false; };

    // Smaller inner blocks have poor utilization in the measured MM workloads.
    constexpr cl_uint MIN_EFFICIENT_BLOCK = 512;

    // Grow the outer block first, then K, to reduce repeated copies and C updates.
    auto evaluate = [&](bool m_is_outer, cl_uint inner_floor) -> Candidate {
        const cl_uint outer_total = m_is_outer ? M : N;
        const cl_uint inner_total = m_is_outer ? N : M;
        cl_uint inner_size = std::min(inner_total,
            ((std::min(inner_floor, inner_total) + TILE - 1) / TILE) * TILE);

        auto probe_outer = [&](cl_uint outer_cand) {
            cl_uint bm_ = m_is_outer ? outer_cand : inner_size;
            cl_uint bn_ = m_is_outer ? inner_size : outer_cand;
            return block_dims_fit(device_id, bm_, bn_, kmult, M, gtA, gtB, fp16_acc);
        };
        const cl_uint inner_floor_aligned = inner_size;
        const cl_uint outer_size = max_size_for(outer_total, probe_outer);
        if (outer_size == 0) return Candidate{};

        cl_uint bm_ = m_is_outer ? outer_size : inner_size;
        cl_uint bn_ = m_is_outer ? inner_size : outer_size;
        const cl_uint bk_ = max_bk_for(bm_, bn_);
        if (bk_ == 0) return Candidate{};

        // Use remaining GMEM to grow the inner block and reduce kernel launches.
        auto probe_inner = [&](cl_uint inner_cand) {
            cl_uint bm2 = m_is_outer ? outer_size : inner_cand;
            cl_uint bn2 = m_is_outer ? inner_cand : outer_size;
            return block_dims_fit(device_id, bm2, bn2, bk_, M, gtA, gtB, fp16_acc);
        };
        inner_size = max_size_for(inner_total, probe_inner);

        // Keep split inner blocks in the measured 512-element size range.
        if (inner_size < inner_total) {
            const cl_uint snapped = (inner_size / MIN_EFFICIENT_BLOCK) * MIN_EFFICIENT_BLOCK;
            if (snapped >= inner_floor_aligned) inner_size = snapped;
        }
        bm_ = m_is_outer ? outer_size : inner_size;
        bn_ = m_is_outer ? inner_size : outer_size;

        const cl_uint outer_blocks = (outer_total + outer_size - 1) / outer_size;
        const cl_uint k_blocks = (K + bk_ - 1) / bk_;
        const double C_total = (2.0 * k_blocks - 1.0) * M * N * sizeof_Tout;

        Candidate c;
        c.bm = bm_;
        c.bn = bn_;
        c.bk = bk_;
        c.cost = m_is_outer
            ? (A_total + double(outer_blocks) * B_total + C_total)
            : (B_total + double(outer_blocks) * A_total + C_total);
        c.ok = true;
        return c;
    };

    Candidate best;
    auto consider = [&](const Candidate & c) {
        if (c.ok && (!best.ok || c.cost < best.cost)) best = c;
    };

    // Try both outer assignments at the standard floor first.
    consider(evaluate(true, MIN_EFFICIENT_BLOCK));
    consider(evaluate(false, MIN_EFFICIENT_BLOCK));

    // Neither fits with the inner pinned at the efficiency floor (tiny GMEM budget or an
    // extreme shape) -- relax the floor in TILE steps, same last-resort shape as the old
    // search's fallback to a smaller M/N, until something fits or we're down to one TILE.
    for (cl_uint floor = MIN_EFFICIENT_BLOCK; !best.ok && floor > TILE; ) {
        floor -= TILE;
        consider(evaluate(true, floor));
        consider(evaluate(false, floor));
    }

    if (!best.ok) return false;
    *bm = best.bm;
    *bn = best.bn;
    *bk = best.bk;
    return true;
}

size_t MM_GMEM::get_hash(cl_device_id device_id, const cl_uint M, const cl_uint N, const cl_uint K,
    ggml_type gtA, ggml_type gtB, ggml_prec prec, bool q8_0A_transposed) {

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

    // A may be F16, F32, or Q8_0 (as a structure of arrays: dense int8 quant plane + a
    // separate half scale plane; dequantized to half in copy_block_a_padded).
    if (!(gtA == GGML_TYPE_F16 || gtA == GGML_TYPE_F32 || is_q8_0(gtA))) {
        return 0LL;
    }
    if (q8_0A_transposed && !is_q8_0(gtA)) {
        return 0LL;
    }
    if (!(gtB == GGML_TYPE_F16 || gtB == GGML_TYPE_F32)) {
        return 0LL;
    }

    // grid element size MxN is 512x512
    // kernel is more efficient for larger K
    // FP16ACC (fp16-accumulate) steps K by 16 with no partial-tile handling, so require
    // K % 16 == 0 in that case; otherwise K % 8 == 0. Q8_0 A requires K % 32 == 0 so that
    // every per-block K offset lands on a quant-block (scale) boundary.
    const bool fp16_acc = use_fp16_acc(prec);
    const cl_uint kmult = k_multiple(gtA, fp16_acc);
    if (M < min_dim || N < min_dim || K < min_dim || (K % kmult)) {
        return 0LL;
    }
    const size_t max_bytes = std::numeric_limits<cl_uint>::max();
    const size_t native_a = is_q8_0(gtA) ? 1 : gtA == GGML_TYPE_F16 ? 2 : 4;
    const size_t native_b = gtB == GGML_TYPE_F16 ? 2 : 4;
    if (uint64_t(M) * K > max_bytes / native_a || uint64_t(N) * K > max_bytes / native_b ||
        uint64_t(M) * N > max_bytes / sizeof_Tout) {
        return 0;
    }

    // some geometries are too big for GMEM
    cl_uint bm = 0;
    cl_uint bn = 0;
    cl_uint bk = 0;
    if (!determine_block_dims(device_id, M, N, K, gtA, gtB, &bm, &bn, &bk, fp16_acc)) {
        return 0LL;
    }
    const cl_uint blockM = bm;
    const cl_uint blockN = bn;
    const cl_uint blockK = bk;
    const cl_uint m_blocks = (M + blockM - 1) / blockM;
    const cl_uint n_blocks = (N + blockN - 1) / blockN;
    const cl_uint k_blocks = (K + blockK - 1) / blockK;

    // Layout and block counts define the recording topology.
    size_t hash = 0LL;
    if (gtA == GGML_TYPE_F16) { hash |= (1 << 0); }
    if (gtB == GGML_TYPE_F16) { hash |= (1 << 1); }
    if (fp16_acc) { hash |= (1 << 2); }
    hash |= ((size_t)m_blocks & 0xFFFF) << 3;
    hash |= ((size_t)n_blocks & 0xFFFF) << 19;
    hash |= ((size_t)k_blocks & 0xFFFF) << 35;
    // Q8 copy kernels have a different argument layout from F32 copy kernels.
    if (is_q8_0(gtA)) { hash |= ((size_t)1 << 51); }
    if (q8_0A_transposed) { hash |= ((size_t)1 << 52); }

    return hash;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// MM_GMEM::kernel_group
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool MM_GMEM::kernel_group::init(cl_program program, cl_device_id device_id, const kernel_key & kk) {
    cl_int status = CL_SUCCESS;
    const char * names[] = { "kernel_mul_mm_f16_f32_gmem", "copy_block_a_padded", "copy_block_b_padded" };
    cl_kernel * handles[] = { &kernel_mm, &kernel_cp_pad_a, &kernel_cp_pad_b };
    const size_t required_workgroups[] = { 256, 128, 64 };
    const cl_uint required_args[] = { 10, std::get<3>(kk) ? 8u : is_q8_0(std::get<0>(kk)) ? 7u : 4u, 5 };
    for (size_t i = 0; i < 3; ++i) {
        *handles[i] = clCreateKernel(program, names[i], &status);
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

std::string MM_GMEM::program_name(ggml_type gtA, ggml_type gtB, ggml_prec prec, bool q8_0A_transposed) {
    const char * a = is_q8_0(gtA) ? (q8_0A_transposed ? "q8trans" : "q8") : gtA == GGML_TYPE_F16 ? "f16" : "f32";
    return std::string("mm_gmem_") + a + (gtB == GGML_TYPE_F16 ? "_f16" : "_f32") + (use_fp16_acc(prec) ? "_acc16" : "_acc32");
}

MM_GMEM * MM_GMEM::create(cl_context context, cl_device_id device_id, cl_command_queue queue, ProfilingInfo & profiling_info,
    cl_program program, ggml_type gtA, ggml_type gtB, ggml_prec prec, bool q8_0A_transposed) {
    kernel_key kk(gtA, gtB, use_fp16_acc(prec), q8_0A_transposed);
    kernel_map_key kmk(context, device_id, gtA, gtB, use_fp16_acc(prec), q8_0A_transposed);
    auto it = kernel_map.find(kmk);
    if (it == kernel_map.end()) {
        auto * group = new kernel_group;
        if (!group->init(program, device_id, kk)) {
            delete group;
            group = nullptr;
        }
        it = kernel_map.emplace(kmk, group).first;
    }
    if (!it->second) { return nullptr; }
    auto * result = new MM_GMEM(context, device_id, queue, profiling_info, gtA, gtB, prec, q8_0A_transposed);
    result->kernels = it->second;
    if (!result->rq.init(context, device_id, queue)) {
        delete result;
        return nullptr;
    }
    return result;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// MM_GMEM
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::map<cl_device_id, size_t> MM_GMEM::available_onchip_global_memory;
void MM_GMEM::query_gmem_size(cl_device_id device_id) {
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

std::map<cl_device_id, size_t> MM_GMEM::max_image_elements;
void MM_GMEM::query_max_image_elements(cl_device_id device_id) {
    size_t & max_elements = max_image_elements[device_id];
    if (max_elements == 0) {
        CL_CHECK(clGetDeviceInfo(device_id, CL_DEVICE_IMAGE_MAX_BUFFER_SIZE,
            sizeof(max_elements), &max_elements, nullptr));
        GGML_ASSERT(max_elements);
    }
}

MM_GMEM::MM_GMEM(cl_context context, cl_device_id device_id, cl_command_queue queue, ProfilingInfo& profiling_info,
    ggml_type gtA, ggml_type gtB, ggml_prec prec,
    bool q8_0A_transposed)
    :
    m_Context(context),
    m_DeviceID(device_id),
    m_CommandQueue(queue),
#ifdef GGML_OPENCL_PROFILING
    m_ProfilingInfo(profiling_info),
#endif // GGML_OPENCL_PROFILING
    gtypeA(gtA),
    gtypeB(gtB),
    prec(prec),
    q8_0A_transposed(q8_0A_transposed),
    fp16A(gtypeA == GGML_TYPE_F16 || is_q8_0(gtypeA)),
    fp16B(gtypeB == GGML_TYPE_F16),
    sizeof_TA(is_q8_0(gtypeA) ? 1 : (fp16A ? 2 : 4)),
    sizeof_TB(fp16B ? 2 : 4) {

#ifndef GGML_OPENCL_PROFILING
    GGML_UNUSED(profiling_info);
#endif // !GGML_OPENCL_PROFILING

    GGML_ASSERT(!q8_0A_transposed || is_q8_0(gtypeA));
}

// Divide large workload [C = M x N x K] into a grid of smaller blocks [blockC = blockM x blockN x blockK]
// For each block, copy a portion of A and B into GMEM for faster access
void MM_GMEM::enqueue(
    cl_uint M, cl_uint N, cl_uint K,
    cl_mem mem_matrixA, cl_ulong offsetA,
    cl_mem mem_matrixB, cl_ulong offsetB,
    cl_mem mem_matrixC, cl_ulong offsetC,
    const ggml_tensor * tensorC,
    cl_mem mem_scalesA, cl_ulong offsetScalesA) {

    cl_int status = CL_SUCCESS;
#ifdef GGML_OPENCL_PROFILING
    std::vector<cl_event> evts;
    evts.reserve(1024);
#else
    GGML_UNUSED(tensorC);
#endif // GGML_OPENCL_PROFILING

    const cl_uint totalM = M;
    const cl_uint totalN = N;
    const cl_uint totalK = K;
    constexpr cl_uint localSizeInTilesM = 2; // empirically determined
    constexpr cl_uint localSizeInTilesN = 2;

    const bool q8A = is_q8_0(gtypeA);
    // A's scale plane is mandatory for Q8_0 and meaningless otherwise.
    GGML_ASSERT(q8A == (mem_scalesA != nullptr));
    GGML_ASSERT(!q8A || (K % 32) == 0);
    // half-element offset of the caller's scale plane within mem_scalesA
    const cl_uint scalesA_base_elems = q8A ? (cl_uint)(offsetScalesA / sizeof(cl_half)) : 0;
    const cl_uint scalesA_row_elems  = q8A ? (K / 32) : 0;

    // determine (hopefully) optimal block sizes
    const bool fp16_acc = use_fp16_acc(prec);
    cl_uint bm = 0;
    cl_uint bn = 0;
    cl_uint bk = 0;
    bool fits = determine_block_dims(m_DeviceID, totalM, totalN, totalK, gtypeA, gtypeB, &bm, &bn, &bk, fp16_acc);
    GGML_ASSERT(fits);
    const cl_uint blockM = bm;
    const cl_uint blockN = bn;
    const cl_uint blockK = bk;

    const cl_uint k_blocks = (totalK + blockK - 1) / blockK;
    const cl_uint blockK_last = (totalK % blockK) == 0 ? blockK : (totalK % blockK);
    const cl_uint lineStrideMatrixCinBytes = sizeof_Tout * totalM;
    for (cl_uint k_block = 0; k_block < k_blocks; ++k_block) {
        cl_uint blockK_loop = (k_block == (k_blocks - 1)) ? blockK_last : blockK;
        const cl_uint k_offset = k_block * blockK;
        cl_uint read_modify_write_C = k_block == 0 ? 0 : 1;

        auto stridesAB = compute_stride_bytes_AB(blockK_loop, gtypeA, gtypeB, fp16_acc);
        const cl_uint lineStrideMatrixAinBytes = std::get<0>(stridesAB);
        const cl_uint lineStrideMatrixBinBytes = std::get<1>(stridesAB);
        // sizeof_TA/sizeof_TB (members) are fixed at native DDR width and must stay that way --
        // they're also used below for DDR-side region offset math. On-chip, A and B are both
        // narrowed to packed half under FP16ACC regardless of native width (see
        // copy_block_a_padded / copy_block_b_padded), so the element count implied by the
        // (possibly-narrowed) on-chip stride needs its own width.
        const size_t sizeof_TA_onchip = (fp16A || fp16_acc) ? 2 : 4;
        const size_t sizeof_TB_onchip = (fp16B || fp16_acc) ? 2 : 4;
        const cl_uint lineStrideMatrixAinElements = lineStrideMatrixAinBytes / sizeof_TA_onchip;
        const cl_uint lineStrideMatrixBinElements = lineStrideMatrixBinBytes / sizeof_TB_onchip;
        const cl_uint msizeA_block_bytes = blockM * lineStrideMatrixAinBytes;
        const cl_uint msizeB_block_bytes = blockN * lineStrideMatrixBinBytes;
        query_gmem_size(m_DeviceID);
        GGML_ASSERT((msizeA_block_bytes + msizeB_block_bytes) <= available_onchip_global_memory.at(m_DeviceID));

        const cl_uint m_blocks = (totalM + blockM - 1) / blockM;
        const cl_uint blockM_last = (totalM % blockM) == 0 ? blockM : (totalM % blockM);
        const cl_uint n_blocks = (totalN + blockN - 1) / blockN;
        const cl_uint blockN_last = (totalN % blockN) == 0 ? blockN : (totalN % blockN);

        // Set global & local sizes
        const cl_uint globalSizeInTilesM = (blockM + 63) / 64;
        const cl_uint globalSizeInTilesN = (blockN + widthOfTileN - 1) / widthOfTileN;

        // matrixA aka src0 (weights)
        cl_mem mem_matrixA_block = nullptr;
        cl_mem mem_tex_matrixA_1dbuf_block = nullptr;
        {
            // reusable scratch space in GMEM for matrix A block
            cl_mem_properties q_buffer_properties[5] = {CL_MEM_ONCHIP_GLOBAL_QCOM, 1, CL_MEM_ONCHIP_GLOBAL_OFFSET_QCOM, 0, 0};
            mem_matrixA_block = clCreateBufferWithProperties(m_Context, q_buffer_properties,
                CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, msizeA_block_bytes, nullptr, &status);
            CL_CHECK(status);

            // use image 1d buffer for matrix A block (padded)
            // Under FP16ACC the on-chip buffer always holds packed half A (narrowed at
            // copy_block_a_padded time if A's native storage is fp32 -- see that kernel),
            // so it's read as raw 32-bit word texels (2 packed halves per texel): the image
            // must be CL_FLOAT sized by word-count, NOT CL_HALF_FLOAT.
            cl_image_format img_fmt_1d = {CL_R, static_cast<cl_channel_type>((fp16A && !fp16_acc) ? CL_HALF_FLOAT : CL_FLOAT)};
            cl_image_desc img_desc_1d = {};
            img_desc_1d.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
            img_desc_1d.image_width = msizeA_block_bytes / (fp16_acc ? 4 : sizeof_TA_onchip);
            img_desc_1d.buffer = mem_matrixA_block;
            mem_tex_matrixA_1dbuf_block = clCreateImage(m_Context, CL_MEM_READ_WRITE, &img_fmt_1d, &img_desc_1d, nullptr, &status);
            CL_CHECK(status);
        }

        // matrixB aka src1 (activation)
        cl_mem mem_matrixB_block = nullptr;
        cl_mem mem_tex_matrixB_1dbuf_block = nullptr;
        {
            // reusable scratch space in GMEM for matrix B block
            cl_mem_properties q_buffer_properties[5] = {CL_MEM_ONCHIP_GLOBAL_QCOM, 1, CL_MEM_ONCHIP_GLOBAL_OFFSET_QCOM, msizeA_block_bytes, 0};
            mem_matrixB_block = clCreateBufferWithProperties(m_Context, q_buffer_properties,
                CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, msizeB_block_bytes, nullptr, &status);
            CL_CHECK(status);

            // use image 1d buffer for matrix B block (padded)
            // Under FP16ACC the on-chip buffer always holds packed half B (narrowed at
            // copy_block_b_padded time if B's native storage is fp32 -- see that kernel),
            // so it's read as raw 32-bit word texels (2 packed halves per texel): the image
            // must be CL_FLOAT sized by word-count, NOT CL_HALF_FLOAT.
            cl_image_format img_fmt_1d = {CL_R, static_cast<cl_channel_type>((fp16B && !fp16_acc) ? CL_HALF_FLOAT : CL_FLOAT)};
            cl_image_desc img_desc_1d = {};
            img_desc_1d.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
            img_desc_1d.image_width = msizeB_block_bytes / (fp16_acc ? 4 : sizeof_TB);
            img_desc_1d.buffer = mem_matrixB_block;
            mem_tex_matrixB_1dbuf_block = clCreateImage(m_Context, CL_MEM_READ_WRITE, &img_fmt_1d, &img_desc_1d, nullptr, &status);
            CL_CHECK(status);
        }

        // Whichever of M,N has fewer blocks should be the outer loop: the inner operand's
        // copy/pad kernel is called once per (outer,inner) block pair regardless of nesting
        // (that product is fixed by the geometry), while the outer operand is cached and
        // called only once per outer iteration. Minimizing which operand pays that "+1 per
        // pair" tax means nesting whichever of M,N has fewer blocks on the outside.
        const bool m_outer = m_blocks <= n_blocks;

        if (m_outer) {
            // loop over M blocks first (fewer of them here) to reuse the A copy/pad in GMEM
            for (cl_uint m_block = 0; m_block < m_blocks; ++m_block) {
                cl_uint blockM_loop = (m_block == (m_blocks - 1)) ? blockM_last : blockM;
                const cl_uint m_offset = m_block * blockM;

                // copy (and pad) one block of matrixA aka src0 (weights)
                cl_mem mem_matrixA_sub = nullptr;
                // recorded by reference; must outlive rq.enqueue() below
                cl_uint scalesA_offset_elems = 0;
                cl_uint matrixA_offset_elems8 = 0;
                {
                    size_t m_block_offset = m_block * blockM * totalK * sizeof_TA;
                    size_t k_block_offset = k_block * blockK * sizeof_TA;

                    // Set the arguments to our pad kernel
                    cl_uint arg_index = 0;
                    if (q8A) {
                        if (q8_0A_transposed) {
                            rq.set_arg(kernels->kernel_cp_pad_a, arg_index++, sizeof(cl_mem), static_cast<const void*>(&mem_matrixA));
                            rq.set_arg(kernels->kernel_cp_pad_a, arg_index++, sizeof(cl_mem), static_cast<const void*>(&mem_scalesA));
                            rq.set_arg(kernels->kernel_cp_pad_a, arg_index++, sizeof(cl_mem), static_cast<const void*>(&mem_matrixA_block));
                            rq.set_arg(kernels->kernel_cp_pad_a, arg_index++, sizeof(cl_uint), &totalM);
                            rq.set_arg(kernels->kernel_cp_pad_a, arg_index++, sizeof(cl_uint), &totalK);
                            rq.set_arg(kernels->kernel_cp_pad_a, arg_index++, sizeof(cl_uint), &m_offset);
                            rq.set_arg(kernels->kernel_cp_pad_a, arg_index++, sizeof(cl_uint), &k_offset);
                            rq.set_arg(kernels->kernel_cp_pad_a, arg_index++, sizeof(cl_uint), &lineStrideMatrixAinElements);
                        } else {
                        // mem_matrixA is already a clCreateSubBuffer of ggml's Q8_0 SoA quant
                        // plane (staged at set_tensor time); OpenCL forbids creating a
                        // sub-buffer of a sub-buffer, so pass the whole plane + an element
                        // offset instead, same as the scale plane below.
                        size_t byteOffset = offsetA + m_block_offset + k_block_offset;
                        GGML_ASSERT((byteOffset % 8) == 0);
                        matrixA_offset_elems8 = (cl_uint)(byteOffset / 8);
                        rq.set_arg(kernels->kernel_cp_pad_a, arg_index++, sizeof(cl_mem), static_cast<const void*>(&mem_matrixA));
                        rq.set_arg(kernels->kernel_cp_pad_a, arg_index++, sizeof(cl_uint), &matrixA_offset_elems8);
                        }
                    } else {
                        // A is unpadded
                        cl_buffer_region region;
                        region.origin = offsetA + m_block_offset + k_block_offset;
                        region.size = (blockM_loop * totalK * sizeof_TA) - k_block_offset;
                        mem_matrixA_sub = clCreateSubBuffer(mem_matrixA, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                        CL_CHECK(status);
                        rq.set_arg(kernels->kernel_cp_pad_a, arg_index++, sizeof(cl_mem), static_cast<const void*>(&mem_matrixA_sub));
                    }
                    if (q8A && !q8_0A_transposed) {
                        // The scale plane is passed whole (not sub-buffered like the quant
                        // plane): at 2 bytes per 32 elements a per-block origin would not
                        // reliably satisfy CL_DEVICE_MEM_BASE_ADDR_ALIGN.
                        scalesA_offset_elems = scalesA_base_elems
                            + (cl_uint)(m_block * blockM * scalesA_row_elems)
                            + (cl_uint)(k_block * blockK / 32);
                        rq.set_arg(kernels->kernel_cp_pad_a, arg_index++, sizeof(cl_mem), static_cast<const void*>(&mem_scalesA));
                        rq.set_arg(kernels->kernel_cp_pad_a, arg_index++, sizeof(cl_uint), &scalesA_offset_elems);
                    }
                    if (!q8_0A_transposed) {
                        rq.set_arg(kernels->kernel_cp_pad_a, arg_index++, sizeof(cl_mem), static_cast<const void*>(&mem_matrixA_block));
                        rq.set_arg(kernels->kernel_cp_pad_a, arg_index++, sizeof(cl_uint), &totalK);
                        rq.set_arg(kernels->kernel_cp_pad_a, arg_index++, sizeof(cl_uint), &lineStrideMatrixAinElements);
                    }
                    GGML_UNUSED(arg_index);

                    GGML_ASSERT(blockK_loop % k_multiple(gtypeA, fp16_acc) == 0);
                    std::vector<size_t> global_size = {blockK_loop/8, blockM_loop};
                    std::vector<size_t> local_size = {16, 8};
                    rq.record(kernels->kernel_cp_pad_a, global_size, local_size);
                }

                // these are recorded by reference, so need to stay in scope until the recordable queue is submitted
                std::vector<cl_mem> mem_matrixB_sub(n_blocks);
                std::vector<cl_mem> mem_matrixC_blocks(n_blocks);
                std::vector<cl_mem> mem_tex_matrixC_1dbuf_blocks(n_blocks);
                std::vector<cl_uint> blockN_loop(n_blocks);
                for (cl_uint n_block = 0; n_block < n_blocks; ++n_block) {
                    blockN_loop[n_block] = (n_block == (n_blocks - 1)) ? blockN_last : blockN;

                    // copy (and pad) one block of matrixB aka src1 (activation)
                    {
                        // B is unpadded
                        cl_buffer_region region;
                        size_t k_block_offset = k_block * blockK * sizeof_TB;
                        size_t n_block_offset = n_block * blockN * totalK * sizeof_TB;
                        region.origin = offsetB + k_block_offset + n_block_offset;
                        region.size = (blockN_loop[n_block] * totalK * sizeof_TB) - k_block_offset;
                        mem_matrixB_sub[n_block] = clCreateSubBuffer(mem_matrixB, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                        CL_CHECK(status);

                        // Set the arguments to our pad kernel
                        cl_uint arg_index = 0;
                        rq.set_arg(kernels->kernel_cp_pad_b, arg_index++, sizeof(cl_mem), static_cast<const void*>(&mem_matrixB_sub[n_block]));
                        rq.set_arg(kernels->kernel_cp_pad_b, arg_index++, sizeof(cl_uint), &totalK);
                        rq.set_arg(kernels->kernel_cp_pad_b, arg_index++, sizeof(cl_mem), static_cast<const void*>(&mem_matrixB_block));
                        rq.set_arg(kernels->kernel_cp_pad_b, arg_index++, sizeof(cl_uint), &lineStrideMatrixBinElements);
                        rq.set_arg(kernels->kernel_cp_pad_b, arg_index++, sizeof(cl_uint), &blockN_loop[n_block]);
                        GGML_UNUSED(arg_index);

                        GGML_ASSERT(blockK_loop % (fp16_acc ? 16 : 8) == 0);
                        std::vector<size_t> global_size = {blockK_loop/8, blockN_loop[n_block]};
                        std::vector<size_t> local_size = {16, 4};
                        rq.record(kernels->kernel_cp_pad_b, global_size, local_size);
                    }

                    // matrixC aka dst (output)
                    {
                        cl_buffer_region region;
                        size_t m_block_offset = m_block * blockM * sizeof_Tout;
                        size_t n_block_offset = n_block * blockN * totalM * sizeof_Tout;
                        region.origin = offsetC + m_block_offset + n_block_offset;
                        region.size = (blockN_loop[n_block] * totalM * sizeof_Tout) - m_block_offset;
                        mem_matrixC_blocks[n_block] = clCreateSubBuffer(mem_matrixC, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
                                                             &region, &status);
                        CL_CHECK(status);

                        // use image 1d buffer for matrix C
                        cl_image_format img_fmt_1d = {CL_R, CL_FLOAT};
                        cl_image_desc img_desc_1d = {};
                        img_desc_1d.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
                        img_desc_1d.image_width = region.size / sizeof_Tout;
                        img_desc_1d.buffer = mem_matrixC_blocks[n_block];
                        mem_tex_matrixC_1dbuf_blocks[n_block] = clCreateImage(m_Context, CL_MEM_READ_WRITE, &img_fmt_1d, &img_desc_1d, nullptr, &status);
                        CL_CHECK(status);
                    }

                    // compute one block of the output
                    {
                        // Set the arguments to our compute kernel
                        cl_uint arg_index = 0;
                        rq.set_arg(kernels->kernel_mm, arg_index++, sizeof(cl_mem),  static_cast<const void*>(&mem_tex_matrixA_1dbuf_block));
                        rq.set_arg(kernels->kernel_mm, arg_index++, sizeof(cl_mem),  static_cast<const void*>(&mem_tex_matrixB_1dbuf_block));
                        rq.set_arg(kernels->kernel_mm, arg_index++, sizeof(cl_mem),  static_cast<const void*>(&mem_tex_matrixC_1dbuf_blocks[n_block]));
                        rq.set_arg(kernels->kernel_mm, arg_index++, sizeof(cl_uint), &read_modify_write_C);
                        rq.set_arg(kernels->kernel_mm, arg_index++, sizeof(cl_uint), &blockM_loop);
                        rq.set_arg(kernels->kernel_mm, arg_index++, sizeof(cl_uint), &blockN_loop[n_block]);
                        rq.set_arg(kernels->kernel_mm, arg_index++, sizeof(cl_uint), &blockK_loop);
                        rq.set_arg(kernels->kernel_mm, arg_index++, sizeof(cl_uint), &lineStrideMatrixAinBytes);
                        rq.set_arg(kernels->kernel_mm, arg_index++, sizeof(cl_uint), &lineStrideMatrixBinBytes);
                        rq.set_arg(kernels->kernel_mm, arg_index++, sizeof(cl_uint), &lineStrideMatrixCinBytes);
                        GGML_UNUSED(arg_index);

                        std::vector<size_t> global_size = {64, globalSizeInTilesM, globalSizeInTilesN}; // 1 block = 64x64 output and each block is 64fibers (1 wave)
                        if (m_block == (m_blocks - 1)) {
                            global_size[1] = (blockM_loop + 63) / 64;
                        }
                        if (n_block == (n_blocks - 1)) {
                            global_size[2] = (blockN_loop[n_block] + widthOfTileN - 1) / widthOfTileN;
                        }
                        std::vector<size_t> local_size = {64, localSizeInTilesM, localSizeInTilesN};
                        rq.record(kernels->kernel_mm, global_size, local_size);
                    }
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

                for (cl_uint i = 0; i < n_blocks; ++i) {
                    clReleaseMemObject(mem_tex_matrixC_1dbuf_blocks[i]);
                    clReleaseMemObject(mem_matrixC_blocks[i]);
                    clReleaseMemObject(mem_matrixB_sub[i]);
                }
                if (mem_matrixA_sub) {
                    clReleaseMemObject(mem_matrixA_sub);
                }
            }
        } else {
            // loop over N blocks first to reuse the larger (B) copy/pad in GMEM
            for (cl_uint n_block = 0; n_block < n_blocks; ++n_block) {
                cl_uint blockN_loop = (n_block == (n_blocks - 1)) ? blockN_last : blockN;

                // copy (and pad) one block of matrixB aka src1 (activation)
                cl_mem mem_matrixB_sub = nullptr;
                {
                    // B is unpadded
                    cl_buffer_region region;
                    size_t k_block_offset = k_block * blockK * sizeof_TB;
                    size_t n_block_offset = n_block * blockN * totalK * sizeof_TB;
                    region.origin = offsetB + k_block_offset + n_block_offset;
                    region.size = (blockN_loop * totalK * sizeof_TB) - k_block_offset;
                    mem_matrixB_sub = clCreateSubBuffer(mem_matrixB, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                    CL_CHECK(status);

                    // Set the arguments to our pad kernel
                    cl_uint arg_index = 0;
                    rq.set_arg(kernels->kernel_cp_pad_b, arg_index++, sizeof(cl_mem), static_cast<const void*>(&mem_matrixB_sub));
                    rq.set_arg(kernels->kernel_cp_pad_b, arg_index++, sizeof(cl_uint), &totalK);
                    rq.set_arg(kernels->kernel_cp_pad_b, arg_index++, sizeof(cl_mem), static_cast<const void*>(&mem_matrixB_block));
                    rq.set_arg(kernels->kernel_cp_pad_b, arg_index++, sizeof(cl_uint), &lineStrideMatrixBinElements);
                    rq.set_arg(kernels->kernel_cp_pad_b, arg_index++, sizeof(cl_uint), &blockN_loop);
                    GGML_UNUSED(arg_index);

                    GGML_ASSERT(blockK_loop % (fp16_acc ? 16 : 8) == 0);
                    std::vector<size_t> global_size = {blockK_loop/8, blockN_loop};
                    std::vector<size_t> local_size = {16, 4};
                    rq.record(kernels->kernel_cp_pad_b, global_size, local_size);
                }

                // these are recorded by reference, so need to stay in scope until the recordable queue is submitted
                std::vector<cl_mem> mem_matrixA_sub(m_blocks);
                std::vector<cl_mem> mem_matrixC_blocks(m_blocks);
                std::vector<cl_mem> mem_tex_matrixC_1dbuf_blocks(m_blocks);
                std::vector<cl_uint> blockM_loop(m_blocks);
                std::vector<cl_uint> scalesA_offset_elems(m_blocks);
                std::vector<cl_uint> matrixA_offset_elems8(m_blocks);
                std::vector<cl_uint> m_offsets(m_blocks);
                for (cl_uint m_block = 0; m_block < m_blocks; ++m_block) {
                    blockM_loop[m_block] = (m_block == (m_blocks - 1)) ? blockM_last : blockM;
                    scalesA_offset_elems[m_block] = 0;
                    matrixA_offset_elems8[m_block] = 0;
                    m_offsets[m_block] = m_block * blockM;
                    mem_matrixA_sub[m_block] = nullptr;

                    // copy (and pad) one block of matrixA aka src0 (weights)
                    {
                        size_t m_block_offset = m_block * blockM * totalK * sizeof_TA;
                        size_t k_block_offset = k_block * blockK * sizeof_TA;

                        // Set the arguments to our pad kernel
                        cl_uint arg_index = 0;
                        if (q8A) {
                            if (q8_0A_transposed) {
                                rq.set_arg(kernels->kernel_cp_pad_a, arg_index++, sizeof(cl_mem), static_cast<const void*>(&mem_matrixA));
                                rq.set_arg(kernels->kernel_cp_pad_a, arg_index++, sizeof(cl_mem), static_cast<const void*>(&mem_scalesA));
                                rq.set_arg(kernels->kernel_cp_pad_a, arg_index++, sizeof(cl_mem), static_cast<const void*>(&mem_matrixA_block));
                                rq.set_arg(kernels->kernel_cp_pad_a, arg_index++, sizeof(cl_uint), &totalM);
                                rq.set_arg(kernels->kernel_cp_pad_a, arg_index++, sizeof(cl_uint), &totalK);
                                rq.set_arg(kernels->kernel_cp_pad_a, arg_index++, sizeof(cl_uint), &m_offsets[m_block]);
                                rq.set_arg(kernels->kernel_cp_pad_a, arg_index++, sizeof(cl_uint), &k_offset);
                                rq.set_arg(kernels->kernel_cp_pad_a, arg_index++, sizeof(cl_uint), &lineStrideMatrixAinElements);
                            } else {
                            // see the m_outer branch for why mem_matrixA can't be sub-buffered
                            size_t byteOffset = offsetA + m_block_offset + k_block_offset;
                            GGML_ASSERT((byteOffset % 8) == 0);
                            matrixA_offset_elems8[m_block] = (cl_uint)(byteOffset / 8);
                            rq.set_arg(kernels->kernel_cp_pad_a, arg_index++, sizeof(cl_mem), static_cast<const void*>(&mem_matrixA));
                            rq.set_arg(kernels->kernel_cp_pad_a, arg_index++, sizeof(cl_uint), &matrixA_offset_elems8[m_block]);
                            }
                        } else {
                            // A is unpadded
                            cl_buffer_region region;
                            region.origin = offsetA + m_block_offset + k_block_offset;
                            region.size = (blockM_loop[m_block] * totalK * sizeof_TA) - k_block_offset;
                            mem_matrixA_sub[m_block] = clCreateSubBuffer(mem_matrixA, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &status);
                            CL_CHECK(status);
                            rq.set_arg(kernels->kernel_cp_pad_a, arg_index++, sizeof(cl_mem), static_cast<const void*>(&mem_matrixA_sub[m_block]));
                        }
                        if (q8A && !q8_0A_transposed) {
                            // see the m_outer branch for why the scale plane is not sub-buffered
                            scalesA_offset_elems[m_block] = scalesA_base_elems
                                + (cl_uint)(m_block * blockM * scalesA_row_elems)
                                + (cl_uint)(k_block * blockK / 32);
                            rq.set_arg(kernels->kernel_cp_pad_a, arg_index++, sizeof(cl_mem), static_cast<const void*>(&mem_scalesA));
                            rq.set_arg(kernels->kernel_cp_pad_a, arg_index++, sizeof(cl_uint), &scalesA_offset_elems[m_block]);
                        }
                        if (!q8_0A_transposed) {
                            rq.set_arg(kernels->kernel_cp_pad_a, arg_index++, sizeof(cl_mem), static_cast<const void*>(&mem_matrixA_block));
                            rq.set_arg(kernels->kernel_cp_pad_a, arg_index++, sizeof(cl_uint), &totalK);
                            rq.set_arg(kernels->kernel_cp_pad_a, arg_index++, sizeof(cl_uint), &lineStrideMatrixAinElements);
                        }
                        GGML_UNUSED(arg_index);

                        GGML_ASSERT(blockK_loop % k_multiple(gtypeA, fp16_acc) == 0);
                        std::vector<size_t> global_size = {blockK_loop/8, blockM_loop[m_block]};
                        std::vector<size_t> local_size = {16, 8};
                        rq.record(kernels->kernel_cp_pad_a, global_size, local_size);
                    }

                    // matrixC aka dst (output)
                    {
                        cl_buffer_region region;
                        size_t m_block_offset = m_block * blockM * sizeof_Tout;
                        size_t n_block_offset = n_block * blockN * totalM * sizeof_Tout;
                        region.origin = offsetC + m_block_offset + n_block_offset;
                        region.size = (blockN_loop * totalM * sizeof_Tout) - m_block_offset;
                        mem_matrixC_blocks[m_block] = clCreateSubBuffer(mem_matrixC, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
                                                             &region, &status);
                        CL_CHECK(status);

                        // use image 1d buffer for matrix C
                        cl_image_format img_fmt_1d = {CL_R, CL_FLOAT};
                        cl_image_desc img_desc_1d = {};
                        img_desc_1d.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
                        img_desc_1d.image_width = region.size / sizeof_Tout;
                        img_desc_1d.buffer = mem_matrixC_blocks[m_block];
                        mem_tex_matrixC_1dbuf_blocks[m_block] = clCreateImage(m_Context, CL_MEM_READ_WRITE, &img_fmt_1d, &img_desc_1d, nullptr, &status);
                        CL_CHECK(status);
                    }

                    // compute one block of the output
                    {
                        // Set the arguments to our compute kernel
                        cl_uint arg_index = 0;
                        rq.set_arg(kernels->kernel_mm, arg_index++, sizeof(cl_mem),  static_cast<const void*>(&mem_tex_matrixA_1dbuf_block));
                        rq.set_arg(kernels->kernel_mm, arg_index++, sizeof(cl_mem),  static_cast<const void*>(&mem_tex_matrixB_1dbuf_block));
                        rq.set_arg(kernels->kernel_mm, arg_index++, sizeof(cl_mem),  static_cast<const void*>(&mem_tex_matrixC_1dbuf_blocks[m_block]));
                        rq.set_arg(kernels->kernel_mm, arg_index++, sizeof(cl_uint), &read_modify_write_C);
                        rq.set_arg(kernels->kernel_mm, arg_index++, sizeof(cl_uint), &blockM_loop[m_block]);
                        rq.set_arg(kernels->kernel_mm, arg_index++, sizeof(cl_uint), &blockN_loop);
                        rq.set_arg(kernels->kernel_mm, arg_index++, sizeof(cl_uint), &blockK_loop);
                        rq.set_arg(kernels->kernel_mm, arg_index++, sizeof(cl_uint), &lineStrideMatrixAinBytes);
                        rq.set_arg(kernels->kernel_mm, arg_index++, sizeof(cl_uint), &lineStrideMatrixBinBytes);
                        rq.set_arg(kernels->kernel_mm, arg_index++, sizeof(cl_uint), &lineStrideMatrixCinBytes);
                        GGML_UNUSED(arg_index);

                        std::vector<size_t> global_size = {64, globalSizeInTilesM, globalSizeInTilesN}; // 1 block = 64x64 output and each block is 64fibers (1 wave)
                        if (m_block == (m_blocks - 1)) {
                            global_size[1] = (blockM_loop[m_block] + 63) / 64;
                        }
                        if (n_block == (n_blocks - 1)) {
                            global_size[2] = (blockN_loop + widthOfTileN - 1) / widthOfTileN;
                        }
                        std::vector<size_t> local_size = {64, localSizeInTilesM, localSizeInTilesN};
                        rq.record(kernels->kernel_mm, global_size, local_size);
                    }
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

                for (cl_uint i = 0; i < m_blocks; ++i) {
                    clReleaseMemObject(mem_tex_matrixC_1dbuf_blocks[i]);
                    clReleaseMemObject(mem_matrixC_blocks[i]);
                    if (mem_matrixA_sub[i]) {
                        clReleaseMemObject(mem_matrixA_sub[i]);
                    }
                }
                clReleaseMemObject(mem_matrixB_sub);
            }
        }

        clReleaseMemObject(mem_tex_matrixB_1dbuf_block);
        clReleaseMemObject(mem_matrixB_block);
        clReleaseMemObject(mem_tex_matrixA_1dbuf_block);
        clReleaseMemObject(mem_matrixA_block);
    }

#ifdef GGML_OPENCL_PROFILING
    cl_uint global_blocks_m = (totalM + 63) / 64;
    cl_uint global_blocks_n = (totalN + widthOfTileN - 1) / widthOfTileN;
    size_t global_size[3] = {64, global_blocks_m, global_blocks_n};
    size_t local_size[3] = {64, localSizeInTilesM, localSizeInTilesN};
    for (cl_event event : evts) {
        m_ProfilingInfo.populate(event, kernels->kernel_mm, 3, global_size, local_size, tensorC);
    }
    for (auto *evt : evts) {
        clReleaseEvent(evt);
    }
    evts.clear();
#endif // GGML_OPENCL_PROFILING
}

void MM_GMEM::enqueue_batched(
    cl_uint M, cl_uint N, cl_uint K,
    cl_mem mem_matrixA, cl_ulong offsetA_base, cl_ulong strideA_bytes,
    cl_mem mem_matrixB, cl_ulong offsetB_base, cl_ulong strideB_bytes,
    cl_mem mem_matrixC, cl_ulong offsetC_base, cl_ulong strideC_bytes,
    cl_uint batch_count,
    const ggml_tensor * tensorC) {

    // Q8_0 A would need a per-batch scale-plane stride here (batch_count scale offsets, not
    // one), which no caller needs yet -- reject rather than silently pass a null scale plane.
    GGML_ASSERT(!is_q8_0(gtypeA));

    for (cl_uint b = 0; b < batch_count; ++b) {
        enqueue(M, N, K,
            mem_matrixA, offsetA_base + (cl_ulong)b * strideA_bytes,
            mem_matrixB, offsetB_base + (cl_ulong)b * strideB_bytes,
            mem_matrixC, offsetC_base + (cl_ulong)b * strideC_bytes,
            tensorC);
    }
}

#endif
