#pragma once

#define CL_TARGET_OPENCL_VERSION GGML_OPENCL_TARGET_VERSION
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

// suppress warnings in CL headers for GCC and Clang
#pragma GCC diagnostic ignored "-Woverlength-strings"
#ifdef __clang__
#pragma GCC diagnostic ignored "-Wgnu-anonymous-struct"
#endif

#include "ggml-opencl.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml.h"

#include "cl-program-cache.h"
#include "ops.h"

#ifdef GGML_OPENCL_USE_ADRENO_BIN_KERNELS
#include "libdl.h"
#ifdef _WIN32
#define KERNEL_LIB_NAME "adreno-opencl-kernels.dll"
#else
#define KERNEL_LIB_NAME "libadreno-opencl-kernels.so"
#endif // _WIN32
#endif // GGML_OPENCL_USE_ADRENO_BIN_KERNELS

typedef const void * (*get_adreno_bin_kernel_func_t)(
    const char * name,
    const char * gpu_name,
    const char * compiler_ver,
    size_t     * out_size
);

#include <CL/cl.h>

#include <inttypes.h>
#include <string.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <map>
#include <memory>
#include <charconv>
#include <mutex>
#include <regex>
#include <set>
#include <unordered_set>

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

#define UNUSED(x) (void)(x)

#define CL_CHECK(err)                                               \
    do {                                                            \
        cl_int err_ = (err);                                        \
        if (err_ != CL_SUCCESS) {                                   \
            GGML_LOG_ERROR("ggml_opencl: %s error %d at %s:%d\n",  \
                #err, err_, __FILE__, __LINE__);                    \
            GGML_ASSERT(0);                                         \
        }                                                           \
    } while (0)

//------------------------------------------------------------------------------
// OpenCL
//------------------------------------------------------------------------------


// See https://gmplib.org/~tege/divcnst-pldi94.pdf figure 4.1.
// Precompute mp (m' in the paper) and L such that division
// can be computed using a multiply (high 32b of 64b result)
// and a shift:
//
// n/d = (mulhi(n, mp) + n) >> L;
struct fastdiv_vals {
    uint32_t mp;
    uint32_t L;
    uint32_t d;
    uint32_t pad;
};
static_assert(sizeof(fastdiv_vals) == 16, "fastdiv_vals size incorrect");

static fastdiv_vals init_fastdiv_values(uint64_t d_64) {
    GGML_ASSERT(d_64 != 0);
    GGML_ASSERT(d_64 <= std::numeric_limits<uint32_t>::max());

    uint32_t d = (uint32_t)d_64;

    // compute L = ceil(log2(d));
    uint32_t L = 0;
    while (L < 32 && (uint32_t{ 1 } << L) < d) {
        L++;
    }

    uint32_t mp = (uint32_t) ((uint64_t{ 1 } << 32) * ((uint64_t{ 1 } << L) - d) / d + 1);
    // pack divisor as well to reduce error surface
    return { mp, L, d, 0 };
}

enum GPU_FAMILY {
    ADRENO,
    INTEL,
    UNKNOWN,
};

enum ADRENO_GPU_GEN {
    ADRENO_UNKNOWN,
    A6X,
    A7X,
    A8X,
    X1E,
    X2E,
};

enum ADRENO_CL_COMPILER_TYPE {
    E031,
    E17,
    DX,
};

struct ggml_cl_version {
    cl_uint major = 0;
    cl_uint minor = 0;
};


struct ggml_cl_compiler_version {
    ADRENO_CL_COMPILER_TYPE type;
    int major = -1;
    int minor = -1;
    int patch = -1;

    bool same(ADRENO_CL_COMPILER_TYPE t, int x, int y, int z) const {
        return major == x && minor == y && patch == z && type == t;
    }
    bool newer_than(ADRENO_CL_COMPILER_TYPE t, int x, int y, int z) const {
        return major*10000 + minor*100 + patch > x*10000 + y*100 + z && type == t;
    }
    bool newer_than_or_same(ADRENO_CL_COMPILER_TYPE t, int x, int y, int z) const {
        return same(t, x, y, z) || newer_than(t, x, y, z);
    }
};

static size_t align_to(size_t value, size_t to_alignment) {
    GGML_ASSERT(to_alignment && "Invalid alignment (must be non-zero)");
    GGML_ASSERT((to_alignment & (to_alignment - 1)) == 0 && "to_alignment must be power-of-two");

    return ((value + to_alignment - 1) / to_alignment) * to_alignment;
}


// Parses a version string of form "XX.YY ". On an error returns ggml_cl_version with all zeroes.
static ggml_cl_version parse_cl_version(std::string_view str) {
    size_t major_str_begin = 0;
    size_t major_str_end   = str.find(".", major_str_begin);
    if (major_str_end == std::string::npos) {
        return {};
    }

    size_t minor_str_begin = major_str_end + 1;
    size_t minor_str_end   = str.find(" ", minor_str_begin);
    if (minor_str_end == std::string::npos) {
        return {};
    }

    cl_uint version_major;
    if (std::from_chars(str.data() + major_str_begin, str.data() + major_str_end, version_major).ec != std::errc{}) {
        return {};
    }

    cl_uint version_minor;
    if (std::from_chars(str.data() + minor_str_begin, str.data() + minor_str_end, version_minor).ec != std::errc{}) {
        return {};
    }
    return { version_major, version_minor };
}

// Returns OpenCL platform's version. On an error returns ggml_cl_version with all zeroes.
static ggml_cl_version get_opencl_platform_version(cl_platform_id platform) {
    size_t param_size;
    CL_CHECK(clGetPlatformInfo(platform, CL_PLATFORM_VERSION, 0, nullptr, &param_size));
    std::unique_ptr<char[]> param_storage(new char[param_size]);
    CL_CHECK(clGetPlatformInfo(platform, CL_PLATFORM_VERSION, param_size, param_storage.get(), nullptr));

    auto              param_value    = std::string_view(param_storage.get(), param_size);
    const std::string version_prefix = "OpenCL ";  // Suffix: "XX.YY <platform-specific-info>"
    if (param_value.find(version_prefix) != 0) {
        return {};
    }
    param_value.remove_prefix(version_prefix.length());
    return parse_cl_version(param_value);
}

// Return a version to use in OpenCL C compilation. On an error returns ggml_cl_version with all zeroes.
static ggml_cl_version get_opencl_device_version(cl_device_id device) {
    size_t param_size;
    if (clGetDeviceInfo(device, CL_DEVICE_VERSION, 0, nullptr, &param_size) != CL_SUCCESS || !param_size) {
        return {};
    }
    std::unique_ptr<char[]> param_storage(new char[param_size]);
    if (clGetDeviceInfo(device, CL_DEVICE_VERSION, param_size, param_storage.get(), nullptr) != CL_SUCCESS) {
        return {};
    }

    auto param_value = std::string_view(param_storage.get(), param_size);
    const std::string version_prefix = "OpenCL ";  // "OpenCL <major>.<minor> <device-specific-info>"
    if (param_value.find(version_prefix) != 0) {
        return {};
    }
    param_value.remove_prefix(version_prefix.length());
    return parse_cl_version(param_value);
}

static ggml_cl_version get_opencl_c_version(ggml_cl_version platform_version, cl_device_id device) {
    size_t param_size;

#if CL_TARGET_OPENCL_VERSION >= 300
    // CL_DEVICE_OPENCL_C_ALL_VERSIONS is an OpenCL 3.0 *device* query, so gating it on the
    // *platform* version is not enough: a 3.0 platform can expose 2.0 devices, where the
    // query returns CL_INVALID_VALUE and the old CL_CHECK aborted during backend init.
    // Gate on the device version, and treat a failure as "fall back to the legacy query"
    // rather than fatal -- a device may advertise 3.0 and still refuse the property.
    const ggml_cl_version device_version = get_opencl_device_version(device);
    if (platform_version.major >= 3 && device_version.major >= 3) {
        cl_int err = clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_ALL_VERSIONS, 0, nullptr, &param_size);
        if (err == CL_SUCCESS && param_size) {
            std::unique_ptr<cl_name_version[]> versions(new cl_name_version[param_size]);
            err = clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_ALL_VERSIONS, param_size, versions.get(), nullptr);
            if (err == CL_SUCCESS) {
                unsigned versions_count = param_size / sizeof(cl_name_version);

                cl_version version_max = 0;
                for (unsigned i = 0; i < versions_count; i++) {
                    version_max = std::max<cl_version>(versions[i].version, version_max);
                }

                return { CL_VERSION_MAJOR(version_max), CL_VERSION_MINOR(version_max) };
            }
        }
        // fall through to CL_DEVICE_OPENCL_C_VERSION below
    }
#else
    GGML_UNUSED(platform_version);
#endif  // CL_TARGET_OPENCL_VERSION >= 300

    if (clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, 0, nullptr, &param_size) != CL_SUCCESS || !param_size) {
        return {};
    }

    std::unique_ptr<char[]> param_storage(new char[param_size]);
    if (clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, param_size, param_storage.get(), nullptr) != CL_SUCCESS) {
        return {};
    }
    auto param_value = std::string_view(param_storage.get(), param_size);

    const std::string version_prefix = "OpenCL C ";  // Suffix: "XX.YY <platform-specific-info>"
    if (param_value.find(version_prefix) != 0) {
        return {};
    }
    param_value.remove_prefix(version_prefix.length());

    return parse_cl_version(param_value);
}

static ADRENO_GPU_GEN get_adreno_gpu_gen(const char *device_name) {
    if (strstr(device_name, "610") || strstr(device_name, "612") ||
        strstr(device_name, "613") || strstr(device_name, "615") ||
        strstr(device_name, "616") || strstr(device_name, "618") ||
        strstr(device_name, "619") || strstr(device_name, "620") ||
        strstr(device_name, "630") || strstr(device_name, "640") ||
        strstr(device_name, "642") || strstr(device_name, "643") ||
        strstr(device_name, "644") || strstr(device_name, "650") ||
        strstr(device_name, "660") || strstr(device_name, "663") ||
        strstr(device_name, "680") || strstr(device_name, "685") ||
        strstr(device_name, "690")) {
        return ADRENO_GPU_GEN::A6X;
    }

    if (strstr(device_name, "730") ||
        strstr(device_name, "740") ||
        strstr(device_name, "750")) {
        return ADRENO_GPU_GEN::A7X;
    }

    if (strstr(device_name, "810") ||
        strstr(device_name, "830") ||
        strstr(device_name, "840") ||
        strstr(device_name, "850")) {
        return ADRENO_GPU_GEN::A8X;
    }

    if (strstr(device_name, "X1")) {
        return ADRENO_GPU_GEN::X1E;
    }

    if (strstr(device_name, "X2")) {
        return ADRENO_GPU_GEN::X2E;
    }

    return ADRENO_GPU_GEN::ADRENO_UNKNOWN;
}

static ggml_cl_compiler_version get_adreno_cl_compiler_version(const char *driver_version) {
    std::string driver_ver_str(driver_version);
    ADRENO_CL_COMPILER_TYPE type = ADRENO_CL_COMPILER_TYPE::E031;
    size_t compiler_ver_pos = driver_ver_str.find("E031");
    size_t compiler_ver_len = 13;
    size_t compiler_major_offset = 5;
    size_t compiler_minor_offset = 8;
    size_t compiler_patch_offset = 11;

    if (compiler_ver_pos == std::string::npos) {
        compiler_ver_pos = driver_ver_str.find("E17");
        if (compiler_ver_pos != std::string::npos) {
            type = ADRENO_CL_COMPILER_TYPE::E17;
            compiler_ver_len = 12;
            compiler_major_offset = 4;
            compiler_minor_offset = 7;
            compiler_patch_offset = 10;
        }
    }

    if (compiler_ver_pos == std::string::npos) {
        compiler_ver_pos = driver_ver_str.find("DX");
        if (compiler_ver_pos == std::string::npos) {
            return {};
        }
        type = ADRENO_CL_COMPILER_TYPE::DX;
        compiler_ver_len = 11;
        compiler_major_offset = 3;
        compiler_minor_offset = 6;
        compiler_patch_offset = 9;
    }

    std::string compiler_ver_str = driver_ver_str.substr(compiler_ver_pos, compiler_ver_len);
    int major = std::atoi(compiler_ver_str.substr(compiler_major_offset, 2).c_str());
    int minor = std::atoi(compiler_ver_str.substr(compiler_minor_offset, 2).c_str());
    int patch = std::atoi(compiler_ver_str.substr(compiler_patch_offset, 2).c_str());
    return { type, major, minor, patch };
}

// cl buffer wrapper
struct ggml_cl_buffer {
    cl_mem buffer;
    size_t size;

    ggml_cl_buffer()
        : buffer(nullptr), size(0) {}

    ~ggml_cl_buffer() {
        if (buffer) {
            CL_CHECK(clReleaseMemObject(buffer));
        }
    }

    void allocate(cl_context context, size_t new_size) {
        if (new_size > size) {
            size = new_size;
            if (buffer) {
                CL_CHECK(clReleaseMemObject(buffer));
            }
            cl_int err;
            CL_CHECK((buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &err), err));
        }
    }
};

// Profiling
struct ProfilingInfo {
    std::string op_name;
    std::string kernel_name;

    cl_kernel kernel;
    cl_event evt;

    cl_ulong cmd_queued;
    cl_ulong cmd_submit;
    cl_ulong cmd_start;
    cl_ulong cmd_end;
    cl_ulong overhead_start;
    cl_ulong overhead_end;
    // For the times below, see spec for clGetEventProfilingInfo
    // The time kernel spent in cmd queue - SUBMIT - QUEUED
    cl_ulong cmd_queued_duration_ns;
    // The time kernel spent for submission - START - SUBMIT
    cl_ulong cmd_submit_duration_ns;
    // Kernel execution time in nanoseconds - END - START
    cl_ulong cmd_duration_ns;
    // The time for the kernel to complete - COMPLETE - END
    cl_ulong cmd_complete_duration_ns;
    // Total time to finish the kernel - COMPLETE - QUEUED
    cl_ulong cmd_total_duration_ns;
    // Global and local work sizes.
    size_t global_size[3];
    size_t local_size[3];
    // Op output size.
    size_t output_size[4];
};

static void populateProfilingInfo(
        ProfilingInfo& info, cl_event evt, cl_kernel kernel, cl_uint work_dim,
        size_t global_size[3], size_t local_size[3],
        const ggml_tensor * tensor) {
    info.op_name     = tensor->name;
    info.kernel      = kernel;
    info.evt         = evt;

    // 0 means not specified, e.g., 2D workgroup, or NULL for driver to choose
    info.local_size[0] = 0;
    info.local_size[1] = 0;
    info.local_size[2] = 0;

    info.global_size[0] = 0;
    info.global_size[1] = 0;
    info.global_size[2] = 0;

    if (local_size) {
        for (cl_uint i = 0; i < work_dim; ++i) {
            info.local_size[i] = local_size[i];
        }
    }

    for (cl_uint i = 0; i < work_dim; ++i) {
        info.global_size[i] = global_size[i];
    }

    info.output_size[0] = tensor->ne[0];
    info.output_size[1] = tensor->ne[1];
    info.output_size[2] = tensor->ne[2];
    info.output_size[3] = tensor->ne[3];
}

struct ggml_backend_opencl_context;
#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
void ggml_cl_adreno_xmem_attn_release_scratch(ggml_backend_opencl_context * backend_ctx);
#endif


// backend device context
struct ggml_backend_opencl_device_context {
    cl_platform_id platform;
    std::string platform_name;

    cl_device_id   device;
    std::string    device_name;
    cl_device_type device_type;
    std::string    device_version;

    // Initialized by ggml_cl_init().
    ggml_backend_opencl_context * backend_ctx = nullptr;

    // Initialized by ggml_backend_opencl_device_get_buffer_type()
    ggml_backend_buffer_type buffer_type;

    cl_context context = nullptr;

    GPU_FAMILY     gpu_family = GPU_FAMILY::UNKNOWN;
    ADRENO_GPU_GEN adreno_gen = ADRENO_GPU_GEN::ADRENO_UNKNOWN;

    std::regex *opfilter = nullptr; // regex of ops to not claim
    std::string opfilter_str = ""; // regex string for opfilter
    size_t global_mem_size = 0;
};

// Lazily-compiled flash-attention kernels and their per-(dk,dv) tile metadata.
// One map per (Q/KV dtype, decode/prefill, split) combination; the int maps
// hold tile dims (bm/bn), workgroup sizes and the n_kv split thresholds.
struct ggml_opencl_fa_kernels {
    // FA bin kernels
#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
    cl_kernel kernel_flash_attn_f32_f16_bin;

    cl_kernel kernel_repack_q_for_wmm;
    cl_kernel kernel_repack_k_for_wmm;
    cl_kernel kernel_repack_v_for_wmm;
    cl_kernel kernel_repack_mask_for_wmm;
#endif

    // f16 Q / f16 KV
    std::map<std::pair<int, int>, cl_kernel> f16;
    std::map<std::pair<int, int>, cl_kernel> f16_q1;
    // f32 Q / f32 KV
    std::map<std::pair<int, int>, cl_kernel> f32;
    std::map<std::pair<int, int>, cl_kernel> f32_q1;
    // f32 Q / f16 KV (mixed)
    std::map<std::pair<int, int>, cl_kernel> f32_f16;
    std::map<std::pair<int, int>, cl_kernel> f32_f16_split;          // N_SPLIT>1 variant
    std::map<std::pair<int, int>, cl_kernel> f32_f16_split_k_img;    // DK=512 prefill split, K via image1d_buffer_t
    std::map<std::pair<int, int>, cl_kernel> f32_f16_q1;
    std::map<std::pair<int, int>, cl_kernel> f32_f16_q1_split;       // flash-decoding K-split
    // vec decode
    std::map<std::pair<int, int>, cl_kernel> f32_f16_q1_vec;
    // kv-head-coalesced vec decode
    std::map<std::pair<int, int>, cl_kernel> f32_f16_q1_vec_mq;
    // kv-head-coalesced + flash-decoding split
    std::map<std::pair<int, int>, cl_kernel> f32_f16_q1_vec_mq_split;
    // MQ_GQA=8 specializations
    std::map<std::pair<int, int>, cl_kernel> f32_f16_q1_vec_mq_g8;
    std::map<std::pair<int, int>, cl_kernel> f32_f16_q1_vec_mq_split_g8;
    // k-image variant of MQ_G8 vec_mq_split
    std::map<std::pair<int, int>, cl_kernel> f32_f16_q1_vec_mq_split_g8_k_img;
    // k-image variant of MQ_GQA=4 vec_mq_split
    std::map<std::pair<int, int>, cl_kernel> f32_f16_q1_vec_mq_split_k_img;
    // Cluster-parallel decode
    std::map<std::pair<int, int>, cl_kernel> f32_f16_q1_vec_mq_split_c8;
    std::map<std::pair<int, int>, cl_kernel> f32_f16_q1_vec_mq_split_g8_c8;
    // NSG_SPLIT=2 specializations (WG=128): the c8 kernel's register footprint
    // caps its per-kernel WG at 128 on X2, below the stock 256/192 requirement.
    // 2 subgroups × FA_CL_NCL streams still gives 16 in-flight rows per WG.
    std::map<std::pair<int, int>, cl_kernel> f32_f16_q1_vec_mq_split_c8_ns2;
    std::map<std::pair<int, int>, cl_kernel> f32_f16_q1_vec_mq_split_g8_c8_ns2;
    // FA_CL_C=32 / MQ_GQA=8 / NSG_SPLIT=2 specialization for the DK=DV=256
    // GQA=8 class (Qwen3.5/3.6-35B-A3B: 16 Q heads, 2 KV heads). o_acc =
    // DV_VEC/32 × 8 = 128B/lane (in budget); the baseline fa1 path for this
    // shape has NO MQ/FD at all and pays an 8× KV re-read per Q head.
    std::map<std::pair<int, int>, cl_kernel> f32_f16_q1_vec_mq_split_g8_c32;
    // alternative decode
    std::map<std::pair<int, int>, cl_kernel> f32_f16_q1_local_tile;
    // hybrid local-tile + MQ + FD-split kernel for DK=DV=128 only
    std::map<std::pair<int, int>, cl_kernel> f32_f16_q1_local_mq_split;
    std::map<std::pair<int, int>, cl_kernel> f32_f16_q1_local_mq_split_g8;
    std::map<std::pair<int, int>, int>       f32_f16_bm;
    std::map<std::pair<int, int>, int>       f32_f16_bn;
    std::map<std::pair<int, int>, int>       f32_f16_wg_size;
    std::map<std::pair<int, int>, int>       f32_f16_split_wg_size;
    std::map<std::pair<int, int>, int>       f32_f16_split_nkv_threshold;
    // f32 Q / native q8_0 KV
    std::map<std::pair<int, int>, cl_kernel> f32_q8_0_q1;            // decode
    std::map<std::pair<int, int>, cl_kernel> f32_q8_0_q1_vec;        // DV-split + multi-subgroup decode
    std::map<std::pair<int, int>, cl_kernel> f32_q8_0_q1_split;      // flash-decoding pass 1
    // KV-head-coalesced + flash-decoding split for q8_0 KV
    std::map<std::pair<int, int>, cl_kernel> f32_q8_0_q1_vec_mq_split;
    std::map<std::pair<int, int>, cl_kernel> f32_q8_0_q1_vec_mq_split_g8;
    // Cluster-parallel q8_0 decode
    std::map<std::pair<int, int>, cl_kernel> f32_q8_0_q1_vec_mq_split_c8;
    std::map<std::pair<int, int>, cl_kernel> f32_q8_0;               // prefill (baseline)
    std::map<std::pair<int, int>, cl_kernel> f32_q8_0_split;         // N_SPLIT>1 variant
    std::map<std::pair<int, int>, int>       f32_q8_0_split_wg_size;        // wg_size = bm*n_split
    std::map<std::pair<int, int>, int>       f32_q8_0_split_nkv_threshold;  // use split when n_kv >= this
    std::map<std::pair<int, int>, int>       f32_q8_0_split_bm;             // per-split BLOCK_M
    // f32 Q / native q4_0 KV
    std::map<std::pair<int, int>, cl_kernel> f32_q4_0_q1;
    std::map<std::pair<int, int>, cl_kernel> f32_q4_0_q1_vec;        // DV-split + multi-subgroup decode
    std::map<std::pair<int, int>, cl_kernel> f32_q4_0_q1_split;
    // kv-head-coalesced + flash-decoding split for q4_0 kv (dp4a K dot)
    std::map<std::pair<int, int>, cl_kernel> f32_q4_0_q1_vec_mq_split;
    std::map<std::pair<int, int>, cl_kernel> f32_q4_0_q1_vec_mq_split_g8;
    // Cluster-parallel q4_0 decode
    std::map<std::pair<int, int>, cl_kernel> f32_q4_0_q1_vec_mq_split_g8_c8;
    std::map<std::pair<int, int>, cl_kernel> f32_q4_0_q1_vec_mq_split_c8;
    std::map<std::pair<int, int>, cl_kernel> f32_q4_0;
    std::map<std::pair<int, int>, cl_kernel> f32_q4_0_split;
    std::map<std::pair<int, int>, int>       f32_q4_0_split_wg_size;
    std::map<std::pair<int, int>, int>       f32_q4_0_split_nkv_threshold;
    std::map<std::pair<int, int>, int>       f32_q4_0_split_bm;
    // shared: flash-decoding merge + prefill prepass (kv-pad, mask-pad, blk class)
    std::map<std::pair<int, int>, cl_kernel> f32_merge;
    std::map<std::pair<int, int>, cl_kernel> kv_pad_f16;
    std::map<std::pair<int, int>, cl_kernel> mask_pad_f16;
    std::map<std::pair<int, int>, cl_kernel> blk_f16;
    // generic prefill tile dims (f16 / f32 paths)
    std::map<std::pair<int, int>, int>       bm;
    std::map<std::pair<int, int>, int>       bn;
    // attempted (variant, (dk, dv))
    // all attempted FA kernels appear here, but those not registered failed compilation
    std::set<std::pair<int, std::pair<int, int>>> variant_attempted;
};

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
struct ggml_cl_adreno_xmem_attn_scratch {
    cl_mem q_img = nullptr;
    cl_mem k_img = nullptr;
    cl_mem v_img = nullptr;
    cl_mem out_img = nullptr;
    cl_mem k_transpose_buf = nullptr;
    cl_mem k_transpose_img1d = nullptr;
    cl_mem k_packed_buf = nullptr;
    cl_mem v_packed_buf = nullptr;
    cl_mem score_buf = nullptr;
    cl_mem prob_buf = nullptr;
    cl_mem score_img1d = nullptr;
    cl_mem prob_img1d = nullptr;
    cl_mem softmax_stats_img2d = nullptr;
    cl_mem xmem_qk = nullptr;
    cl_mem xmem_pv = nullptr;

    int n_q = 0;
    int n_kv = 0;
    int n_kv_padded = 0;
    int d_head_q = 0;
    int d_head_v = 0;
    int q_width = 0;
    int kv_heads_total = 0;
};

struct ggml_cl_adreno_xmem_attn_state {
    bool compiled = false;
    bool logged = false;

    cl_kernel kernel_q_f32_to_img_scaled = nullptr;
    cl_kernel kernel_kv_f32_to_img_gqa = nullptr;
    cl_kernel kernel_kv_f16_to_img_gqa = nullptr;
    cl_kernel kernel_img_to_f32 = nullptr;
    cl_kernel kernel_k_gather = nullptr;
    cl_kernel kernel_pack_k = nullptr;
    cl_kernel kernel_qk_gemm = nullptr;
    cl_kernel kernel_softmax_reduce_basic = nullptr;
    cl_kernel kernel_softmax_apply_basic = nullptr;
    cl_kernel kernel_mask_scores = nullptr;
    cl_kernel kernel_pack_v = nullptr;
    cl_kernel kernel_pv_gemm = nullptr;

    ggml_cl_adreno_xmem_attn_scratch scratch;
};
#endif

// backend context
struct ggml_backend_opencl_context {
    int ref_count;

    cl_device_id device;
    std::string device_name;

    ggml_cl_version platform_version;
    ggml_cl_version opencl_c_version;

    // argsort is loaded in supports_op because its availability depends on how
    // many workgroups are allowed, which requires kernel compilation.
    bool kernels_loaded_argsort = false;
    // rest of the kernels are currently always loaded in alloc_buffer.
    bool kernels_loaded = false;

    std::string driver_version;

    GPU_FAMILY gpu_family;
    ADRENO_GPU_GEN adreno_gen;

    cl_int alignment;
    size_t global_mem_size;
    size_t max_alloc_size;
    size_t max_workgroup_size;
    bool fp16_support;
    bool has_vector_subgroup_broadcast;
    bool has_subgroup_shuffle = false;       // cl_khr_subgroup_shuffle or cl_qcom_subgroup_shuffle
    bool has_integer_dot      = false;       // cl_khr_integer_dot_product or cl_qcom_dot_product8
    bool has_qcom_subgroup_shuffle = false;  // specifically cl_qcom_subgroup_shuffle
    bool disable_fusion;
    bool fuse_mm_glu = true;                     // opt-out GGML_OPENCL_FUSE_MM_GLU=0 (byte-identical gate+up GEMV + GLU, q4_K FFN)
    bool fuse_rms_add = true;                    // opt-out GGML_OPENCL_FUSE_RMS_ADD=0 (fused rms_norm*w + residual)
    bool f16_mrow = true;                        // opt-out GGML_OPENCL_F16_MROW=0 (multi-row-per-WG f16 decode GEMV for attn proj + lm_head)
    int  f16_mrow_rpt = 1;                       // GGML_OPENCL_F16_MROW_RPT={1,2,4,8,16} rows-per-subgroup register blocking

    // ragged moe, use int to directly pass to kernel
    cl_uint  adreno_use_moe_ragged;
    cl_uint  adreno_moe_ragged_skip_gran;
    cl_uint  adreno_use_moe_ragged_dp4;

    // whether fuse moe combine
    cl_uint fuse_moe_combine;

    // whether to fold the MoE bias adds into swiglu_oai
    cl_uint fuse_moe_bias_glu;

    // whether to fold the MoE down-projection bias add into the combine
    cl_uint fuse_moe_bias_combine;

    bool adreno_has_large_buffer;
    bool adreno_use_large_buffer;
    bool adreno_use_bin_kernels;
    get_adreno_bin_kernel_func_t get_adreno_bin_kernel_func = nullptr;
    ggml_cl_compiler_version adreno_cl_compiler_version;
    // The q6_K flat mul_mat codegen workarounds are needed by old E031 compilers only.
    bool q6_k_flat_old_compiler;

    std::string kernel_compile_opts;  // cached for lazy-compiled kernels.

    int adreno_wave_size;

    cl_bool non_uniform_workgroups;
    size_t  image_max_buffer_size;
    size_t  image2d_max_width;
    size_t  image2d_max_height;

    cl_device_svm_capabilities svm_caps;

    cl_context context;
    cl_command_queue queue;

    // On-disk compiled-program cache (see GGML_OPENCL_KERNEL_CACHE_DIR).
    cl_program_cache_state program_cache;
    bool program_cache_initialized = false;

    // prealloc buffers for transposing weights and activations
    ggml_cl_buffer prealloc_quant_trans;
    ggml_cl_buffer prealloc_scales_trans;
    ggml_cl_buffer prealloc_act_trans;
    // q8_1-quantized reordered MoE activations for the dp4a prefill GEMM.
    ggml_cl_buffer prealloc_moe_qa;   // int8 quants  [tok_slots * ne00]
    ggml_cl_buffer prealloc_moe_da;   // per-block d  [tok_slots * ne00/32] (half)
    ggml_cl_buffer prealloc_moe_sa;   // per-block s  [tok_slots * ne00/32] (half)
    // scratch copy of the router weights to avoid dst aliasing
    ggml_cl_buffer prealloc_moe_combine_w;
    ggml_cl_buffer prealloc_splitk_partial;  // [ksplit * M] partials for split-K GEMV

    // pool of persistent image1d_buffer views over kv-cache layers, keyed by
    // (parent buffer, offset within parent)
    // used by the img-variant KQ/KQV dispatch paths to avoid per-call
    // clCreateSubBuffer + clCreateImage + pending-release-queue on long-context decode
    struct ImagePoolKey {
        uintptr_t buf;
        uint64_t  offset;
        bool operator<(const ImagePoolKey & o) const {
            if (buf != o.buf) return buf < o.buf;
            return offset < o.offset;
        }
    };
    struct ImagePoolEntry {
        cl_mem sub_buffer = nullptr;
        cl_mem image      = nullptr;
        size_t k_bytes    = 0;
        cl_channel_type channel_data_type = CL_FLOAT;
    };
    std::map<ImagePoolKey, ImagePoolEntry> kq_img_pool;
    std::map<ImagePoolKey, ImagePoolEntry> kqv_img_pool;

    // pool for the on-device f16 buffer for kv-cache with non-FA quantized-K (q8_0/q4_0)
    std::map<ImagePoolKey, ImagePoolEntry> dequant_f16_pool;

    // prealloc buffers for src0 and src1
    ggml_cl_buffer prealloc_src0;
    ggml_cl_buffer prealloc_src1;

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
    ggml_cl_buffer prealloc_adreno_xmem_const;
    bool adreno_xmem_gemm_enabled = false;
#endif

    // prealloc buffers for MoE router table preprocess
    bool toggle_reorder = false;
    ggml_cl_buffer prealloc_post_router;
    ggml_cl_buffer prealloc_emap;
    ggml_cl_buffer prealloc_hist;
    ggml_cl_buffer prealloc_tile_offset;
    ggml_cl_buffer prealloc_total_tiles;
    ggml_cl_buffer prealloc_slot_counter;

    ggml_opencl_add_kernels add = {};
    ggml_opencl_add_id_kernels add_id = {};
    ggml_opencl_mul_kernels mul = {};
    ggml_opencl_div_kernels div = {};
    ggml_opencl_sub_kernels sub = {};
    ggml_opencl_scale_kernels scale = {};
    ggml_opencl_sqr_kernels sqr = {};
    ggml_opencl_sqrt_kernels sqrt = {};
    ggml_opencl_mean_kernels mean = {};
    ggml_opencl_silu_kernels silu = {};
    ggml_opencl_gelu_kernels gelu = {};
    ggml_opencl_relu_kernels relu = {};
    ggml_opencl_sigmoid_kernels sigmoid = {};
    ggml_opencl_tri_kernels tri = {};
    ggml_opencl_fill_kernels fill = {};
    ggml_opencl_clamp_kernels clamp = {};
    ggml_opencl_glu_kernels glu = {};
    ggml_opencl_norm_kernels norm = {};
    ggml_opencl_rms_norm_kernels rms_norm = {};
    ggml_opencl_l2_norm_kernels l2_norm = {};
    ggml_opencl_group_norm_kernels group_norm = {};
    ggml_opencl_diag_mask_inf_kernels diag_mask_inf = {};
    ggml_opencl_diag_kernels diag = {};
    ggml_opencl_soft_max_kernels soft_max = {};
    ggml_opencl_fa_kernels fa;
#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
    ggml_cl_adreno_xmem_attn_state adreno_xmem_attn;
#endif
    ggml_opencl_get_rows_kernels get_rows = {};
    ggml_opencl_set_rows_kernels set_rows = {};
    ggml_opencl_rope_kernels rope = {};
    ggml_opencl_cpy_kernels cpy = {};
    ggml_opencl_repack_kernels repack = {};
    ggml_opencl_mul_mat_kernels mul_mat = {};
    ggml_opencl_mul_mat_id_kernels mul_mat_id = {};
    ggml_opencl_solve_tri_kernels solve_tri = {};
    ggml_opencl_im2col_kernels im2col = {};
    ggml_opencl_argsort_kernels argsort = {};
    ggml_opencl_sum_rows_kernels sum_rows = {};
    ggml_opencl_cumsum_kernels cumsum = {};
    ggml_opencl_repeat_kernels repeat = {};
    ggml_opencl_pad_kernels pad = {};
    ggml_opencl_tanh_kernels tanh = {};
    ggml_opencl_neg_kernels neg = {};
    ggml_opencl_exp_kernels exp = {};
    ggml_opencl_expm1_kernels expm1 = {};
    ggml_opencl_abs_kernels abs = {};
    ggml_opencl_unary_ext_kernels unary_ext = {};
    ggml_opencl_softplus_kernels softplus = {};
    ggml_opencl_upscale_kernels upscale = {};
    ggml_opencl_concat_kernels concat = {};
    ggml_opencl_conv_2d_kernels conv_2d = {};
    ggml_opencl_ssm_conv_kernels ssm_conv = {};
    ggml_opencl_gated_delta_net_kernels gated_delta_net = {};
    ggml_opencl_ssm_scan_kernels ssm_scan = {};
    ggml_opencl_timestep_embedding_kernels timestep_embedding = {};
    std::vector<ProfilingInfo> profiling_info;
    std::vector<ProfilingInfo> profiling_results;

    void flush_profiling_batch() {
        if (profiling_info.empty()) {
            return;
        }

        // Populate profiling info
        for (ProfilingInfo & info : profiling_info) {
            cl_ulong cmd_queued;
            cl_ulong cmd_submit;
            cl_ulong cmd_start;
            cl_ulong cmd_end;
            cl_ulong cmd_complete;

            CL_CHECK(clWaitForEvents(1, &info.evt));
            CL_CHECK(clGetEventProfilingInfo(
                info.evt, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &cmd_queued, NULL));
            CL_CHECK(clGetEventProfilingInfo(
                info.evt, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &cmd_submit, NULL));
            CL_CHECK(clGetEventProfilingInfo(
                info.evt, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &cmd_start, NULL));
            CL_CHECK(clGetEventProfilingInfo(
                info.evt, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &cmd_end, NULL));
            CL_CHECK(clGetEventProfilingInfo(
                info.evt, CL_PROFILING_COMMAND_COMPLETE, sizeof(cl_ulong), &cmd_complete, NULL));
            CL_CHECK(clReleaseEvent(info.evt));
            info.evt = nullptr;

            char kernel_name[512];
            CL_CHECK(clGetKernelInfo(info.kernel, CL_KERNEL_FUNCTION_NAME,
                sizeof(kernel_name), kernel_name, NULL));
            info.kernel_name = kernel_name;

            info.cmd_queued = cmd_queued;
            info.cmd_submit = cmd_submit;
            info.cmd_start  = cmd_start;
            info.cmd_end    = cmd_end;

            info.cmd_queued_duration_ns     = cmd_submit    - cmd_queued;
            info.cmd_submit_duration_ns     = cmd_start     - cmd_submit;
            info.cmd_duration_ns            = cmd_end       - cmd_start;
            info.cmd_complete_duration_ns   = cmd_complete  - cmd_end;
            info.cmd_total_duration_ns      = cmd_complete  - cmd_queued;
        }
        profiling_results.insert(profiling_results.end(),
            std::make_move_iterator(profiling_info.begin()),
            std::make_move_iterator(profiling_info.end()));
        profiling_info.clear();
    }

    void write_profiling_info() {
        if (profiling_results.empty()) {
            return;
        }

        // Dump a csv
        FILE * fperf = fopen("cl_profiling.csv", "w");
        if (!fperf) {
            GGML_LOG_ERROR("Failed to open cl_profiling.csv\n");
            return;
        }

        fprintf(fperf, "op name, kernel name, exec duration (ms), global size, local size, output size\n");
        for (const ProfilingInfo & info : profiling_results) {
            fprintf(fperf, "%s,%s,%f,%zux%zux%zu,%zux%zux%zu,%zux%zux%zux%zu\n",
                info.op_name.c_str(), info.kernel_name.c_str(),
                info.cmd_duration_ns/1.e6f,
                info.global_size[0], info.global_size[1], info.global_size[2],
                info.local_size[0], info.local_size[1], info.local_size[2],
                info.output_size[0], info.output_size[1], info.output_size[2], info.output_size[3]);
        }
        fclose(fperf);

        // Dump a simple chrome trace
        FILE * ftrace = fopen("cl_trace.json", "w");
        if (!ftrace) {
            GGML_LOG_ERROR("Failed to open cl_trace.json\n");
            return;
        }

        fprintf(ftrace, "[\n");
        for (const ProfilingInfo & info : profiling_results) {
            fprintf(ftrace, "{\"name\": \"%s\", \"cat\": \"OpenCL\", \"ph\": \"B\", \"ts\": %" PRIu64 ", \"pid\": \"\", \"tid\": \"Host\"},\n",
                info.kernel_name.c_str(), info.cmd_queued/1000);
            fprintf(ftrace, "{\"name\": \"%s\", \"cat\": \"OpenCL\", \"ph\": \"E\", \"ts\": %" PRIu64 ", \"pid\": \"\", \"tid\": \"Host\"},\n",
                info.kernel_name.c_str(), info.cmd_submit/1000);

            fprintf(ftrace, "{\"name\": \"%s\", \"cat\": \"OpenCL\", \"ph\": \"B\", \"ts\": %" PRIu64 ", \"pid\": \"\", \"tid\": \"Device\"},\n",
                info.kernel_name.c_str(), info.cmd_start/1000);
            fprintf(ftrace, "{\"name\": \"%s\", \"cat\": \"OpenCL\", \"ph\": \"E\", \"ts\": %" PRIu64 ", \"pid\": \"\", \"tid\": \"Device\"},\n",
                info.kernel_name.c_str(), info.cmd_end/1000);
        }
        fprintf(ftrace, "]\n");
        fclose(ftrace);
    }

    size_t get_kernel_workgroup_size(cl_kernel kernel) const {
        size_t workgroup_size = 0;
        size_t ret_size = 0;
        CL_CHECK(
            clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE,
                sizeof(size_t), &workgroup_size, &ret_size));
        GGML_ASSERT(sizeof(size_t) == ret_size);
        return workgroup_size;
    }

    void enqueue_ndrange_kernel(cl_kernel kernel, cl_uint work_dim, size_t *global_work_size, size_t *local_work_size, const ggml_tensor * tensor) {
        // From the spec on clEnqueueNDRangeKernel:
        // If the device associated with command_queue is an OpenCL 2.1 or newer device,
        // and global_work_size is NULL or the value in any passed dimension is zero,
        // then the kernel command will trivially succeed after its event dependencies
        // are satisfied and subsequently update its completion event.
        // So this ensures such cases always return trivially without causing errors in
        // case of an older device.
        for (cl_uint i = 0; i < work_dim; i++) {
            if (global_work_size[i] == 0) {
                return;
            }
        }
#ifdef GGML_OPENCL_PROFILING
        cl_event evt;
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, work_dim, NULL, global_work_size, local_work_size, 0, NULL, &evt));

        profiling_info.emplace_back();
        populateProfilingInfo(profiling_info.back(), evt, kernel, work_dim, global_work_size, local_work_size, tensor);
        if (profiling_info.size() >= 2048) {
            flush_profiling_batch();
        }
#else
        GGML_UNUSED(tensor);
        CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, work_dim, NULL, global_work_size, local_work_size, 0, NULL, NULL));
#endif
    }

    const void * get_adreno_bin_kernel(const std::string &kernel_name, size_t *bin_size) const {
        if (!get_adreno_bin_kernel_func) {
            return nullptr;
        }

        size_t sz;
        const void * kernel_bin = get_adreno_bin_kernel_func(
            kernel_name.c_str(), device_name.c_str(), driver_version.c_str(), &sz);
        if (bin_size) {
            *bin_size = sz;
        }
        return kernel_bin;
    }

    void free() {
        clFinish(queue);

        ref_count--;
        if (ref_count == 0) {
#ifdef GGML_OPENCL_PROFILING
            flush_profiling_batch();
            write_profiling_info();
            profiling_results.clear();
#endif
            // release pooled image1d_buffer views over KV cache layers.
            for (auto & kv : kq_img_pool) {
                if (kv.second.image)      { CL_CHECK(clReleaseMemObject(kv.second.image)); }
                if (kv.second.sub_buffer) { CL_CHECK(clReleaseMemObject(kv.second.sub_buffer)); }
            }
            kq_img_pool.clear();
            for (auto & kv : kqv_img_pool) {
                if (kv.second.image)      { CL_CHECK(clReleaseMemObject(kv.second.image)); }
                if (kv.second.sub_buffer) { CL_CHECK(clReleaseMemObject(kv.second.sub_buffer)); }
            }
            kqv_img_pool.clear();
            for (auto & kv : dequant_f16_pool) {
                if (kv.second.image) { CL_CHECK(clReleaseMemObject(kv.second.image)); }
            }
            dequant_f16_pool.clear();
#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
            ggml_cl_adreno_xmem_attn_release_scratch(this);
#endif
        }
    }
};

struct ggml_tensor_extra_cl {
    // The buffer object that holds the data.
    cl_mem data_device;
    // The offset into the buffer object. This is primarily for scratch buffer
    // and view operation.
    // NB: this offset no longer includes view offset (view_offs). Whenever this
    // offset is used, view_offs should be considered.
    cl_ulong offset;
    // The actual size of the cl_mem object. This is needed when returning the
    // block to the pool.
    size_t actual_size;

    void reset() {
        data_device = nullptr;
        offset = 0;
        actual_size = 0;
    }
};

struct ggml_tensor_extra_cl_q1_0 {
    cl_mem q = nullptr;
    cl_mem q_img = nullptr;

    cl_mem d = nullptr;
    cl_mem d_img = nullptr;

    size_t size_q = 0;
    size_t size_d = 0;

    ~ggml_tensor_extra_cl_q1_0() {
        reset();
    }

    void reset() {
        // q and d are subbuffers into the bigger buffer allocated in ggml_backend_buffer.
        // They must be properly released so that the original buffer can be
        // properly released to avoid memory leak.
        if (q != nullptr) {
            CL_CHECK(clReleaseMemObject(q));
            q = nullptr;
        }
        if (d != nullptr) {
            CL_CHECK(clReleaseMemObject(d));
            d = nullptr;
        }
        q_img = nullptr;
        d_img = nullptr;
        size_q = 0;
        size_d = 0;
    }
};

// Additional tensor extra structs for quantized tensors.
// These tensors are loaded from files and should not be allocated in scratch --
// they should always be allocated from the pool. Hence, they do not have an
// `offset`, which indicate their locations in the scratch buffer.
struct ggml_tensor_extra_cl_q4_0 {
    // Quantized values.
    cl_mem q = nullptr;
    // Quantized values in image1d_buffer_t.
    cl_mem q_img = nullptr;
    // Scales.
    cl_mem d = nullptr;
    // Scales in image1d_buffer_t.
    cl_mem d_img = nullptr;
    // Size of quantized values.
    size_t size_q = 0;
    // Size of scales.
    size_t size_d = 0;

    ~ggml_tensor_extra_cl_q4_0() {
        reset();
    }

    void reset() {
        // q and d are subbuffers into the bigger buffer allocated in ggml_backend_buffer.
        // They must be properly released so that the original buffer can be
        // properly released to avoid memory leak.
        if (q != nullptr) {
            CL_CHECK(clReleaseMemObject(q));
            q = nullptr;
        }
        if (d != nullptr) {
            CL_CHECK(clReleaseMemObject(d));
            d = nullptr;
        }
        if (q_img != nullptr) {
            CL_CHECK(clReleaseMemObject(q_img));
            q_img = nullptr;
        }
        if (d_img != nullptr) {
            CL_CHECK(clReleaseMemObject(d_img));
            d_img = nullptr;
        }
        size_q = 0;
        size_d = 0;
    }
};

struct ggml_tensor_extra_cl_q4_1 {
    // Quantized values.
    cl_mem q = nullptr;
    // Quantized values in image1d_buffer_t.
    cl_mem q_img = nullptr;
    // Scales.
    cl_mem d = nullptr;
    // Scales in image1d_buffer_t.
    cl_mem d_img = nullptr;
    // Min
    cl_mem m = nullptr;
    // Min in image1d_buffer_t.
    cl_mem m_img = nullptr;
    // Size of quantized values.
    size_t size_q = 0;
    // Size of scales.
    size_t size_d = 0;
    // Size of min values.
    size_t size_m = 0;

    ~ggml_tensor_extra_cl_q4_1() {
        reset();
    }

    void reset() {
        // q and d are subbuffers into the bigger buffer allocated in ggml_backend_buffer.
        // They must be properly released so that the original buffer can be
        // properly released to avoid memory leak.
        if (q != nullptr) {
            CL_CHECK(clReleaseMemObject(q));
            q = nullptr;
        }
        if (d != nullptr) {
            CL_CHECK(clReleaseMemObject(d));
            d = nullptr;
        }
        if (m != nullptr) {
            CL_CHECK(clReleaseMemObject(m));
            m = nullptr;
        }
        if (q_img != nullptr) {
            CL_CHECK(clReleaseMemObject(q_img));
            q_img = nullptr;
        }
        // Currently, q_img and d_img are only initialized when SMALL_ALLOC is
        // enabled. They point to the images in ggml_backend_opencl_buffer_context.
        // So, there is no need to release them here.
        // TODO: initialize them for non SMALL_PATH path, or remove them.
        d_img = nullptr;
        m_img = nullptr;
        size_q = 0;
        size_d = 0;
        size_m = 0;
    }
};

struct ggml_tensor_extra_cl_q5_0 {
    // Quantized values.
    cl_mem qs = nullptr;
    // Quantized values in image1d_buffer_t.
    cl_mem qs_img = nullptr;
    // 5-th bit values.
    cl_mem qh = nullptr;
    // 5-th bit values in image1d_buffer_t.
    cl_mem qh_img = nullptr;
    // Scales.
    cl_mem d = nullptr;
    // Scales in image1d_buffer_t.
    cl_mem d_img = nullptr;
    // Uniform per-32-block scale (2/block) + min (1/block, = d*16 for the -16 centering)
    // for the generic dp4a MoE GEMM. Built from d.
    cl_mem scale = nullptr;
    cl_mem min = nullptr;
    // Size of quantized values.
    size_t size_qs = 0;
    // Size of 5-th bit values.
    size_t size_qh = 0;
    // Size of scales.
    size_t size_d = 0;

    ~ggml_tensor_extra_cl_q5_0() {
        reset();
    }

    void reset() {
        if (qs != nullptr) {
            CL_CHECK(clReleaseMemObject(qs));
            qs = nullptr;
        }
        if (qh != nullptr) {
            CL_CHECK(clReleaseMemObject(qh));
            qh = nullptr;
        }
        if (d != nullptr) {
            CL_CHECK(clReleaseMemObject(d));
            d = nullptr;
        }
        if (qs_img != nullptr) {
            CL_CHECK(clReleaseMemObject(qs_img));
            qs_img = nullptr;
        }
        if (scale != nullptr) {
            CL_CHECK(clReleaseMemObject(scale));
            scale = nullptr;
        }
        if (min != nullptr) {
            CL_CHECK(clReleaseMemObject(min));
            min = nullptr;
        }

        qh_img = nullptr;
        d_img = nullptr;
        size_qs = 0;
        size_qh = 0;
        size_d = 0;
    }
};

struct ggml_tensor_extra_cl_q5_1 {
    // Quantized values.
    cl_mem qs = nullptr;
    // Quantized values in image1d_buffer_t.
    cl_mem qs_img = nullptr;
    // 5-th bit values.
    cl_mem qh = nullptr;
    // 5-th bit values in image1d_buffer_t.
    cl_mem qh_img = nullptr;
    // Scales.
    cl_mem d = nullptr;
    // Scales in image1d_buffer_t.
    cl_mem d_img = nullptr;
    // Min
    cl_mem m = nullptr;
    // Min in image1d_buffer_t.
    cl_mem m_img = nullptr;
    // Size of quantized values.
    size_t size_qs = 0;
    // Size of 5-th bit values.
    size_t size_qh = 0;
    // Size of scales.
    size_t size_d = 0;
    // Size of min values.
    size_t size_m = 0;

    ~ggml_tensor_extra_cl_q5_1() {
        reset();
    }

    void reset() {
        // q and d are subbuffers into the bigger buffer allocated in ggml_backend_buffer.
        // They must be properly released so that the original buffer can be
        // properly released to avoid memory leak.
        if (qs != nullptr) {
            CL_CHECK(clReleaseMemObject(qs));
            qs = nullptr;
        }
        if (qh != nullptr) {
            CL_CHECK(clReleaseMemObject(qh));
            qh = nullptr;
        }
        if (d != nullptr) {
            CL_CHECK(clReleaseMemObject(d));
            d = nullptr;
        }
        if (m != nullptr) {
            CL_CHECK(clReleaseMemObject(m));
            m = nullptr;
        }
        if (qs_img != nullptr) {
            CL_CHECK(clReleaseMemObject(qs_img));
            qs_img = nullptr;
        }
        // qh_img, d_img, and m_img are not currently allocated separately.
        // TODO: initialize them for non SMALL_PATH path, or remove them.
        qh_img = nullptr;
        d_img = nullptr;
        m_img = nullptr;
        size_qs = 0;
        size_qh = 0;
        size_d = 0;
        size_m = 0;
    }
};

struct ggml_tensor_extra_cl_mxfp4 {
    // Quantized values.
    cl_mem q = nullptr;
    // Quantized values in image1d_buffer_t.
    cl_mem q_img = nullptr;
    // Scales in E8M0.
    cl_mem e = nullptr;
    // Scales in image1d_buffer_t.
    cl_mem e_img = nullptr;
    // Size of quantized values.
    size_t size_q = 0;
    // Size of scales.
    size_t size_e = 0;

    ~ggml_tensor_extra_cl_mxfp4() {
        reset();
    }

    void reset() {
        // q and d are subbuffers into the bigger buffer allocated in ggml_backend_buffer.
        // They must be properly released so that the original buffer can be
        // properly released to avoid memory leak.
        if (q != nullptr) {
            CL_CHECK(clReleaseMemObject(q));
            q = nullptr;
        }
        if (e != nullptr) {
            CL_CHECK(clReleaseMemObject(e));
            e = nullptr;
        }
        if (q_img != nullptr) {
            CL_CHECK(clReleaseMemObject(q_img));
            q_img = nullptr;
        }
        // Currently, e_img is not used. They can be image1d_buffer_t
        // that wraps around q and d to utilize image access path.
        e_img = nullptr;
        size_q = 0;
        size_e = 0;
    }
};

struct ggml_tensor_extra_cl_q8_0 {
    cl_mem q = nullptr;
    cl_mem q_img = nullptr;

    cl_mem d = nullptr;
    cl_mem d_img = nullptr;

    // Uniform per-16-segment scale (16/superblock) for the generic dp4a MoE GEMM.
    // Expanded from d at set_tensor; the int8 codes are reused from q.
    // q8_0 is symmetric so no min buffer (has_min=0).
    cl_mem scale = nullptr;

    size_t size_q = 0;
    size_t size_d = 0;

    ~ggml_tensor_extra_cl_q8_0() {
        reset();
    }

    void reset() {
        // q and d are subbuffers into the bigger buffer allocated in ggml_backend_buffer.
        // They must be properly released so that the original buffer can be
        // properly released to avoid memory leak.
        if (q != nullptr) {
            CL_CHECK(clReleaseMemObject(q));
            q = nullptr;
        }
        if (d != nullptr) {
            CL_CHECK(clReleaseMemObject(d));
            d = nullptr;
        }
        if (scale != nullptr) {
            CL_CHECK(clReleaseMemObject(scale));
            scale = nullptr;
        }
        // Currently, q_img and d_img are not used. They can be image1d_buffer_t
        // that wraps around q and d to utilize image access path.
        q_img = nullptr;
        d_img = nullptr;
        size_q = 0;
        size_d = 0;
    }
};

struct ggml_tensor_extra_cl_iq4_nl {
    cl_mem q = nullptr;
    cl_mem q_img = nullptr;

    cl_mem d = nullptr;
    cl_mem d_img = nullptr;

    size_t size_q = 0;
    size_t size_d = 0;

    ~ggml_tensor_extra_cl_iq4_nl() {
        reset();
    }

    void reset() {
        if (q != nullptr) { CL_CHECK(clReleaseMemObject(q)); q = nullptr; }
        if (d != nullptr) { CL_CHECK(clReleaseMemObject(d)); d = nullptr; }
        q_img = nullptr;
        d_img = nullptr;
        size_q = 0;
        size_d = 0;
    }
};

struct ggml_tensor_extra_cl_q4_K {
    // Quantized values
    cl_mem q = nullptr;
    // Quantized values in image1d_buffer_t.
    cl_mem q_img = nullptr;
    // Scales for each super block.
    cl_mem s  = nullptr;
    // Scales
    cl_mem d = nullptr;
    // Min
    cl_mem dm  = nullptr;

    ~ggml_tensor_extra_cl_q4_K() {
        reset();
    }

    void reset() {
        if (q != nullptr) {
            CL_CHECK(clReleaseMemObject(q));
            q = nullptr;
        }
        if (s != nullptr) {
            CL_CHECK(clReleaseMemObject(s));
            s = nullptr;
        }
        if (d != nullptr) {
            CL_CHECK(clReleaseMemObject(d));
            d = nullptr;
        }
        if (dm != nullptr) {
            CL_CHECK(clReleaseMemObject(dm));
            dm = nullptr;
        }
        if (q_img != nullptr) {
            CL_CHECK(clReleaseMemObject(q_img));
            q_img = nullptr;
        }
    }
};

struct ggml_tensor_extra_cl_q5_K {
    // Lower 4 bits of quantized weights.
    cl_mem q  = nullptr;
    // Quantized values in image1d_buffer_t.
    cl_mem q_img = nullptr;
    // Upper 1 bit of quantized weights.
    cl_mem qh = nullptr;
    // Scales for each block.
    cl_mem s  = nullptr;
    // Scales for each super block.
    cl_mem d  = nullptr;
    // Min for each super block.
    cl_mem dm = nullptr;
    // Uniform per-32-block scale (2/block) + min (1/block, = dm*mn) decoded from the
    // 6-bit packed s[] for the generic dp4a MoE GEMM kernel_gemm_moe_q8_1_dp4a.
    // Built from s/d/dm at set_tensor; q/qh are reused as-is.
    cl_mem scale = nullptr;
    cl_mem min   = nullptr;

    size_t size_q  = 0;
    size_t size_qh = 0;
    size_t size_s  = 0;
    size_t size_d  = 0;
    size_t size_dm = 0;

    ~ggml_tensor_extra_cl_q5_K() {
        reset();
    }

    void reset() {
        if (q != nullptr) {
            CL_CHECK(clReleaseMemObject(q));
            q = nullptr;
        }
        if (qh != nullptr) {
            CL_CHECK(clReleaseMemObject(qh));
            qh = nullptr;
        }
        if (s != nullptr) {
            CL_CHECK(clReleaseMemObject(s));
            s = nullptr;
        }
        if (d != nullptr) {
            CL_CHECK(clReleaseMemObject(d));
            d = nullptr;
        }
        if (dm != nullptr) {
            CL_CHECK(clReleaseMemObject(dm));
            dm = nullptr;
        }
        if (q_img != nullptr) {
            CL_CHECK(clReleaseMemObject(q_img));
            q_img = nullptr;
        }
        if (scale != nullptr) {
            CL_CHECK(clReleaseMemObject(scale));
            scale = nullptr;
        }
        if (min != nullptr) {
            CL_CHECK(clReleaseMemObject(min));
            min = nullptr;
        }

        size_q  = 0;
        size_qh = 0;
        size_s  = 0;
        size_d  = 0;
        size_dm = 0;
    }
};

struct ggml_tensor_extra_cl_q6_K {
    // Lower 4 bits of quantized weights.
    cl_mem ql = nullptr;
    // Lower 4 bits as image1d_buffer_t
    cl_mem ql_img = nullptr;
    // Upper 2 bits of quantized weights.
    cl_mem qh = nullptr;
    // Upper 2 bits as image1d_buffer_t
    cl_mem qh_img = nullptr;
    // Scales for each block.
    cl_mem s  = nullptr;
    // Scales for each super block.
    cl_mem d  = nullptr;

    size_t size_ql = 0;
    size_t size_qh = 0;
    size_t size_s  = 0;
    size_t size_d  = 0;

    ~ggml_tensor_extra_cl_q6_K() {
        reset();
    }

    void reset() {
        if (ql != nullptr) {
            CL_CHECK(clReleaseMemObject(ql));
            ql = nullptr;
        }
        if (qh != nullptr) {
            CL_CHECK(clReleaseMemObject(qh));
            qh = nullptr;
        }
        if (qh_img != nullptr) {
            CL_CHECK(clReleaseMemObject(qh_img));
            qh_img = nullptr;
        }
        if (s != nullptr) {
            CL_CHECK(clReleaseMemObject(s));
            s = nullptr;
        }
        if (d != nullptr) {
            CL_CHECK(clReleaseMemObject(d));
            d = nullptr;
        }
        if (ql_img != nullptr) {
            CL_CHECK(clReleaseMemObject(ql_img));
            ql_img = nullptr;
        }

        size_ql = 0;
        size_qh = 0;
        size_s  = 0;
        size_d  = 0;
    }
};
struct ggml_backend_opencl_buffer_context {
    // A buffer context can hold multiple cl_mem objects. This is for flattening
    // quantized weights and should be used with GGML_OPENCL_SMALL_ALLOC where
    // each tensor is allocated a separate buffer. When flattening is enabled
    // with small allocation, each tensor is backed by two cl_mem objects (for
    // quants and scales) packed into a backend_opencl_buffer.
    ggml_backend_opencl_buffer_context(cl_mem buf)
        : name("OpenCL") {
        buffer.push_back(buf);
    }

    ~ggml_backend_opencl_buffer_context() {
        for (cl_mem buf : buffer) {
            CL_CHECK(clReleaseMemObject(buf));
        }
        for (cl_mem im : img) {
            CL_CHECK(clReleaseMemObject(im));
        }

        // Delete all extras to trigger their destructors
        for (ggml_tensor_extra_cl * e : temp_tensor_extras) {
            delete e;
        }
        for (ggml_tensor_extra_cl * e : temp_tensor_extras_in_use) {
            delete e;
        }
        for (ggml_tensor_extra_cl_q4_0 * e : temp_tensor_extras_q4_0) {
            delete e;
        }
        for (ggml_tensor_extra_cl_q4_0 * e : temp_tensor_extras_q4_0_in_use) {
            delete e;
        }
        for (ggml_tensor_extra_cl_q4_1 * e : temp_tensor_extras_q4_1) {
            delete e;
        }
        for (ggml_tensor_extra_cl_q4_1 * e : temp_tensor_extras_q4_1_in_use) {
            delete e;
        }
        for (ggml_tensor_extra_cl_q5_0 * e : temp_tensor_extras_q5_0) {
            delete e;
        }
        for (ggml_tensor_extra_cl_q5_0 * e : temp_tensor_extras_q5_0_in_use) {
            delete e;
        }
        for (ggml_tensor_extra_cl_q5_1 * e : temp_tensor_extras_q5_1) {
            delete e;
        }
        for (ggml_tensor_extra_cl_q5_1 * e : temp_tensor_extras_q5_1_in_use) {
            delete e;
        }
        for (ggml_tensor_extra_cl_mxfp4 * e : temp_tensor_extras_mxfp4) {
            delete e;
        }
        for (ggml_tensor_extra_cl_mxfp4 * e : temp_tensor_extras_mxfp4_in_use) {
            delete e;
        }
        for (ggml_tensor_extra_cl_q8_0 * e : temp_tensor_extras_q8_0) {
            delete e;
        }
        for (ggml_tensor_extra_cl_q8_0 * e : temp_tensor_extras_q8_0_in_use) {
            delete e;
        }
        for (ggml_tensor_extra_cl_q1_0 * e : temp_tensor_extras_q1_0) {
            delete e;
        }
        for (ggml_tensor_extra_cl_q1_0 * e : temp_tensor_extras_q1_0_in_use) {
            delete e;
        }
        for (ggml_tensor_extra_cl_iq4_nl * e : temp_tensor_extras_iq4_nl) {
            delete e;
        }
        for (ggml_tensor_extra_cl_iq4_nl * e : temp_tensor_extras_iq4_nl_in_use) {
            delete e;
        }
        for (ggml_tensor_extra_cl_q4_K * e : temp_tensor_extras_q4_K) {
            delete e;
        }
        for (ggml_tensor_extra_cl_q4_K * e : temp_tensor_extras_q4_K_in_use) {
            delete e;
        }
        for (ggml_tensor_extra_cl_q6_K * e : temp_tensor_extras_q6_K) {
            delete e;
        }
        for (ggml_tensor_extra_cl_q6_K * e : temp_tensor_extras_q6_K_in_use) {
            delete e;
        }
        for (ggml_tensor_extra_cl_q5_K * e : temp_tensor_extras_q5_K) {
            delete e;
        }
        for (ggml_tensor_extra_cl_q5_K * e : temp_tensor_extras_q5_K_in_use) {
            delete e;
        }
    }

    ggml_tensor_extra_cl * ggml_opencl_alloc_temp_tensor_extra() {
        ggml_tensor_extra_cl * extra;
        if (temp_tensor_extras.empty()) {
            extra = new ggml_tensor_extra_cl();
        } else {
            extra = temp_tensor_extras.back();
            temp_tensor_extras.pop_back();
        }

        temp_tensor_extras_in_use.push_back(extra);

        extra->reset();
        return extra;
    }

    ggml_tensor_extra_cl_q1_0 * ggml_opencl_alloc_temp_tensor_extra_q1_0() {
        ggml_tensor_extra_cl_q1_0 * extra;
        if (temp_tensor_extras_q1_0.empty()) {
            extra = new ggml_tensor_extra_cl_q1_0();
        } else {
            extra = temp_tensor_extras_q1_0.back();
            temp_tensor_extras_q1_0.pop_back();
        }

        temp_tensor_extras_q1_0_in_use.push_back(extra);

        extra->reset();
        return extra;
    }

    ggml_tensor_extra_cl_q4_0 * ggml_opencl_alloc_temp_tensor_extra_q4_0() {
        ggml_tensor_extra_cl_q4_0 * extra;
        if (temp_tensor_extras_q4_0.empty()) {
            extra = new ggml_tensor_extra_cl_q4_0();
        } else {
            extra = temp_tensor_extras_q4_0.back();
            temp_tensor_extras_q4_0.pop_back();
        }

        temp_tensor_extras_q4_0_in_use.push_back(extra);

        extra->reset();
        return extra;
    }

    ggml_tensor_extra_cl_q4_1 * ggml_opencl_alloc_temp_tensor_extra_q4_1() {
        ggml_tensor_extra_cl_q4_1 * extra;
        if (temp_tensor_extras_q4_1.empty()) {
            extra = new ggml_tensor_extra_cl_q4_1();
        } else {
            extra = temp_tensor_extras_q4_1.back();
            temp_tensor_extras_q4_1.pop_back();
        }

        temp_tensor_extras_q4_1_in_use.push_back(extra);

        extra->reset();
        return extra;
    }

    ggml_tensor_extra_cl_q5_0 * ggml_opencl_alloc_temp_tensor_extra_q5_0() {
        ggml_tensor_extra_cl_q5_0 * extra;
        if (temp_tensor_extras_q5_0.empty()) {
            extra = new ggml_tensor_extra_cl_q5_0();
        } else {
            extra = temp_tensor_extras_q5_0.back();
            temp_tensor_extras_q5_0.pop_back();
        }

        temp_tensor_extras_q5_0_in_use.push_back(extra);

        extra->reset();
        return extra;
    }

    ggml_tensor_extra_cl_q5_1 * ggml_opencl_alloc_temp_tensor_extra_q5_1() {
        ggml_tensor_extra_cl_q5_1 * extra;
        if (temp_tensor_extras_q5_1.empty()) {
            extra = new ggml_tensor_extra_cl_q5_1();
        } else {
            extra = temp_tensor_extras_q5_1.back();
            temp_tensor_extras_q5_1.pop_back();
        }

        temp_tensor_extras_q5_1_in_use.push_back(extra);

        extra->reset();
        return extra;
    }

    ggml_tensor_extra_cl_mxfp4 * ggml_opencl_alloc_temp_tensor_extra_mxfp4() {
        ggml_tensor_extra_cl_mxfp4 * extra;
        if (temp_tensor_extras_mxfp4.empty()) {
            extra = new ggml_tensor_extra_cl_mxfp4();
        } else {
            extra = temp_tensor_extras_mxfp4.back();
            temp_tensor_extras_mxfp4.pop_back();
        }

        temp_tensor_extras_mxfp4_in_use.push_back(extra);

        extra->reset();
        return extra;
    }

    ggml_tensor_extra_cl_q8_0 * ggml_opencl_alloc_temp_tensor_extra_q8_0() {
        ggml_tensor_extra_cl_q8_0 * extra;
        if (temp_tensor_extras_q8_0.empty()) {
            extra = new ggml_tensor_extra_cl_q8_0();
        } else {
            extra = temp_tensor_extras_q8_0.back();
            temp_tensor_extras_q8_0.pop_back();
        }

        temp_tensor_extras_q8_0_in_use.push_back(extra);

        extra->reset();
        return extra;
    }

    ggml_tensor_extra_cl_iq4_nl * ggml_opencl_alloc_temp_tensor_extra_iq4_nl() {
        ggml_tensor_extra_cl_iq4_nl * extra;
        if (temp_tensor_extras_iq4_nl.empty()) {
            extra = new ggml_tensor_extra_cl_iq4_nl();
        } else {
            extra = temp_tensor_extras_iq4_nl.back();
            temp_tensor_extras_iq4_nl.pop_back();
        }

        temp_tensor_extras_iq4_nl_in_use.push_back(extra);

        extra->reset();
        return extra;
    }

    ggml_tensor_extra_cl_q4_K * ggml_opencl_alloc_temp_tensor_extra_q4_K() {
        ggml_tensor_extra_cl_q4_K * extra;
        if (temp_tensor_extras_q4_K.empty()) {
            extra = new ggml_tensor_extra_cl_q4_K();
        } else {
            extra = temp_tensor_extras_q4_K.back();
            temp_tensor_extras_q4_K.pop_back();
        }

        temp_tensor_extras_q4_K_in_use.push_back(extra);

        extra->reset();
        return extra;
    }

    ggml_tensor_extra_cl_q5_K * ggml_opencl_alloc_temp_tensor_extra_q5_K() {
        ggml_tensor_extra_cl_q5_K * extra;
        if (temp_tensor_extras_q5_K.empty()) {
            extra = new ggml_tensor_extra_cl_q5_K();
        } else {
            extra = temp_tensor_extras_q5_K.back();
            temp_tensor_extras_q5_K.pop_back();
        }

        temp_tensor_extras_q5_K_in_use.push_back(extra);

        extra->reset();
        return extra;
    }

    ggml_tensor_extra_cl_q6_K * ggml_opencl_alloc_temp_tensor_extra_q6_K() {
        ggml_tensor_extra_cl_q6_K * extra;
        if (temp_tensor_extras_q6_K.empty()) {
            extra = new ggml_tensor_extra_cl_q6_K();
        } else {
            extra = temp_tensor_extras_q6_K.back();
            temp_tensor_extras_q6_K.pop_back();
        }

        temp_tensor_extras_q6_K_in_use.push_back(extra);

        extra->reset();
        return extra;
    }

    void reset() {
        for (ggml_tensor_extra_cl * e : temp_tensor_extras_in_use) {
            temp_tensor_extras.push_back(e);
        }
        temp_tensor_extras_in_use.clear();

        for (ggml_tensor_extra_cl_q1_0 * e : temp_tensor_extras_q1_0_in_use) {
            temp_tensor_extras_q1_0.push_back(e);
        }
        temp_tensor_extras_q1_0_in_use.clear();

        for (ggml_tensor_extra_cl_q4_0 * e : temp_tensor_extras_q4_0_in_use) {
            temp_tensor_extras_q4_0.push_back(e);
        }
        temp_tensor_extras_q4_0_in_use.clear();

        for (ggml_tensor_extra_cl_q4_1 * e : temp_tensor_extras_q4_1_in_use) {
            temp_tensor_extras_q4_1.push_back(e);
        }
        temp_tensor_extras_q4_1_in_use.clear();

        for (ggml_tensor_extra_cl_q5_0 * e : temp_tensor_extras_q5_0_in_use) {
            temp_tensor_extras_q5_0.push_back(e);
        }
        temp_tensor_extras_q5_0_in_use.clear();

        for (ggml_tensor_extra_cl_q5_1 * e : temp_tensor_extras_q5_1_in_use) {
            temp_tensor_extras_q5_1.push_back(e);
        }
        temp_tensor_extras_q5_1_in_use.clear();

        for (ggml_tensor_extra_cl_mxfp4 * e : temp_tensor_extras_mxfp4_in_use) {
            temp_tensor_extras_mxfp4.push_back(e);
        }
        temp_tensor_extras_mxfp4_in_use.clear();

        for (ggml_tensor_extra_cl_q8_0 * e : temp_tensor_extras_q8_0_in_use) {
            temp_tensor_extras_q8_0.push_back(e);
        }
        temp_tensor_extras_q8_0_in_use.clear();

        for (ggml_tensor_extra_cl_iq4_nl * e : temp_tensor_extras_iq4_nl_in_use) {
            temp_tensor_extras_iq4_nl.push_back(e);
        }
        temp_tensor_extras_iq4_nl_in_use.clear();

        for (ggml_tensor_extra_cl_q4_K * e : temp_tensor_extras_q4_K_in_use) {
            temp_tensor_extras_q4_K.push_back(e);
        }
        temp_tensor_extras_q4_K_in_use.clear();

        for (ggml_tensor_extra_cl_q5_K * e : temp_tensor_extras_q5_K_in_use) {
            temp_tensor_extras_q5_K.push_back(e);
        }
        temp_tensor_extras_q5_K_in_use.clear();

        for (ggml_tensor_extra_cl_q6_K * e : temp_tensor_extras_q6_K_in_use) {
            temp_tensor_extras_q6_K.push_back(e);
        }
        temp_tensor_extras_q6_K_in_use.clear();

        q8_0_soa_tensors.clear();
        q4_0_soa_tensors.clear();
    }

    // Pools for extras. Available extras are in `temp_tensor_extras`. Extras
    // being used are in `temp_tensor_extras_in_use`. At the first run, new
    // extras get created and put in `in_use`. When the buffer is reset via
    // the `reset` callback, all extras in `in_use` get moved to available extras
    // for reuse.
    std::vector<ggml_tensor_extra_cl *> temp_tensor_extras;
    std::vector<ggml_tensor_extra_cl *> temp_tensor_extras_in_use;
    std::vector<ggml_tensor_extra_cl_q1_0 *> temp_tensor_extras_q1_0;
    std::vector<ggml_tensor_extra_cl_q1_0 *> temp_tensor_extras_q1_0_in_use;
    std::vector<ggml_tensor_extra_cl_q4_0 *> temp_tensor_extras_q4_0;
    std::vector<ggml_tensor_extra_cl_q4_0 *> temp_tensor_extras_q4_0_in_use;
    std::vector<ggml_tensor_extra_cl_q4_1 *> temp_tensor_extras_q4_1;
    std::vector<ggml_tensor_extra_cl_q4_1 *> temp_tensor_extras_q4_1_in_use;
    std::vector<ggml_tensor_extra_cl_q5_0 *> temp_tensor_extras_q5_0;
    std::vector<ggml_tensor_extra_cl_q5_0 *> temp_tensor_extras_q5_0_in_use;
    std::vector<ggml_tensor_extra_cl_q5_1 *> temp_tensor_extras_q5_1;
    std::vector<ggml_tensor_extra_cl_q5_1 *> temp_tensor_extras_q5_1_in_use;
    std::vector<ggml_tensor_extra_cl_mxfp4 *> temp_tensor_extras_mxfp4;
    std::vector<ggml_tensor_extra_cl_mxfp4 *> temp_tensor_extras_mxfp4_in_use;
    std::vector<ggml_tensor_extra_cl_q8_0 *> temp_tensor_extras_q8_0;
    std::vector<ggml_tensor_extra_cl_q8_0 *> temp_tensor_extras_q8_0_in_use;
    std::vector<ggml_tensor_extra_cl_iq4_nl *> temp_tensor_extras_iq4_nl;
    std::vector<ggml_tensor_extra_cl_iq4_nl *> temp_tensor_extras_iq4_nl_in_use;
    std::vector<ggml_tensor_extra_cl_q4_K *> temp_tensor_extras_q4_K;
    std::vector<ggml_tensor_extra_cl_q4_K *> temp_tensor_extras_q4_K_in_use;
    std::vector<ggml_tensor_extra_cl_q5_K *> temp_tensor_extras_q5_K;
    std::vector<ggml_tensor_extra_cl_q5_K *> temp_tensor_extras_q5_K_in_use;
    std::vector<ggml_tensor_extra_cl_q6_K *> temp_tensor_extras_q6_K;
    std::vector<ggml_tensor_extra_cl_q6_K *> temp_tensor_extras_q6_K_in_use;

    // q8_0 tensors with AoS->SoA layout conversion installed by set_tensor.
    // Two types of tensors get SOA'ed - normal weights and MoE weights.
    // In Q8_0's case, we only have normal weights. If we ever have Q8_0 as MoE
    // weights, they need to be added to this set in `set_tensors`.
    std::unordered_set<const ggml_tensor *> q8_0_soa_tensors;

    // Same for q4_0. KV-cache q4_0 tensors are allocated but never pass
    // through set_tensor, so they stay AoS and aren't in this set.
    // In Q4_0's case, in addition to normal weights, we have MoE weights.
    std::unordered_set<const ggml_tensor *> q4_0_soa_tensors;

    // The buffer_context is initially created by ggml_backend_buft_alloc_buffer
    // before any tensor is initialized (at the beginning of alloc_tensor_range).
    // Hence, there is always a buffer object in this vector. When each tensor is
    // being initialized, this original buffer object will be released if both
    // flattening and small allocation are enabled, and additional buffer
    // objects will be created in init_tensor to represent flattened quantized
    // weights.
    std::vector<cl_mem> buffer;
    // These are image1d_buffer_t objects that wrap around the quants and scales.
    // For Q4_0 quantization, there should be two of them - one for quants and
    // one for scales. They should be populated only when flattening and small
    // allocation are enabled.
    std::vector<cl_mem> img;
    std::string name;
};


std::string read_file(const std::string & path);
cl_program build_program_from_source_ex(
        cl_context ctx, cl_device_id dev, const char * program_buffer,
        const std::string & compile_opts, bool fatal,
        const char * tag = nullptr, cl_command_queue retry_queue = nullptr);
cl_program build_program_from_source(
        ggml_backend_opencl_context * backend_ctx, const char * program_buffer,
        const std::string & compile_opts);
cl_program build_program_from_binary(
        cl_context ctx, cl_device_id dev, const char * program_buffer,
        const std::string & compile_opts, size_t bin_size = 0);
bool use_adreno_bin_kernels(ggml_backend_opencl_context * backend_ctx);

void load_cl_kernels(ggml_backend_opencl_context * backend_ctx);
void sync_with_other_backends(ggml_backend_opencl_context * backend_ctx);
void sync_with_other_backends(ggml_backend_t backend);
void transpose_2d_as_8b(ggml_backend_opencl_context * backend_ctx, cl_mem src, cl_mem dst,
                        size_t size, cl_int stride, cl_int rows,
                        bool blocking = true, bool rows_soa = false);
void transpose_2d_as_16b(ggml_backend_opencl_context * backend_ctx, cl_mem src, cl_mem dst,
                         size_t size, cl_int stride, cl_int rows, bool blocking = true);
void transpose_2d_as_32b(ggml_backend_opencl_context * backend_ctx, cl_mem src, cl_mem dst,
                         size_t size, cl_int stride, cl_int rows, bool blocking = true);
bool use_q4_0_bin_kernels(const ggml_backend_opencl_context * backend_ctx, const ggml_tensor * tensor);
bool use_q4_k_bin_kernels(const ggml_backend_opencl_context * backend_ctx, const ggml_tensor * tensor);
bool use_q6_k_bin_kernels(const ggml_backend_opencl_context * backend_ctx, const ggml_tensor * tensor);
bool use_adreno_kernels(const ggml_backend_opencl_context * backend_ctx, const ggml_tensor * tensor);
bool adreno_e17_compiler_quirks(const ggml_backend_opencl_context * backend_ctx);
bool use_adreno_moe_kernels(const ggml_backend_opencl_context * backend_ctx, const ggml_tensor * tensor);
bool enable_adreno_trans_weight(const ggml_backend_opencl_context * backend_ctx, const ggml_tensor * tensor);
bool use_q6k_tiled(const ggml_backend_opencl_context * backend_ctx, const ggml_tensor * tensor);
bool use_q4k_tiled(const ggml_backend_opencl_context * backend_ctx, const ggml_tensor * tensor);
bool enable_adreno_trans_weight_q5_K(const ggml_backend_opencl_context * backend_ctx, const ggml_tensor * tensor);
bool use_flat_gemv_for_large_m_q4_K(const ggml_backend_opencl_context * backend_ctx, const ggml_tensor * tensor);
bool use_flat_gemv_for_large_m_q6_K(const ggml_backend_opencl_context * backend_ctx, const ggml_tensor * tensor);
bool ggml_cl_is_q4_0_soa(const ggml_tensor * tensor);
bool ggml_cl_is_q8_0_soa(const ggml_tensor * tensor);
cl_mem ggml_cl_img_pool_get_or_create(
        ggml_backend_opencl_context * backend_ctx,
        std::map<ggml_backend_opencl_context::ImagePoolKey,
                 ggml_backend_opencl_context::ImagePoolEntry> & pool,
        cl_mem data_device, cl_ulong offset0, size_t required_bytes,
        cl_channel_type channel_data_type);

void ggml_backend_opencl_buffer_set_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor,
        const void * data, size_t offset, size_t size);
void ggml_backend_opencl_buffer_get_tensor(
        ggml_backend_buffer_t buffer, const ggml_tensor * tensor,
        void * data, size_t offset, size_t size);

extern std::vector<ggml_backend_device> g_ggml_backend_opencl_devices;
extern std::vector<std::unique_ptr<ggml_backend_opencl_device_context>> g_ggml_backend_opencl_dev_ctxs;
extern struct ggml_backend_device_i ggml_backend_opencl_device_i;

bool ggml_cl_compute_forward(ggml_backend_t backend, struct ggml_tensor * tensor);
void ggml_cl_load_kernels_repack(ggml_backend_opencl_context * backend_ctx);
void ggml_cl_load_kernels_unpack(ggml_backend_opencl_context * backend_ctx);
