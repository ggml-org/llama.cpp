#pragma once

#include "ggml-vulkan.h"

#include <vulkan/vulkan_core.h>

#if defined(GGML_VULKAN_RUN_TESTS) || defined(GGML_VULKAN_CHECK_RESULTS)
#include <chrono>
#include "ggml-cpu.h"
#endif

#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1

#if VK_HEADER_VERSION >= 301
namespace vk::detail { class DispatchLoaderDynamic; }
using vk::detail::DispatchLoaderDynamic;
#else
namespace vk { class DispatchLoaderDynamic; }
using vk::DispatchLoaderDynamic;
#endif

DispatchLoaderDynamic & ggml_vk_default_dispatcher();

#define VULKAN_HPP_DEFAULT_DISPATCHER ggml_vk_default_dispatcher()

#include <vulkan/vulkan.hpp>

#ifndef VK_NV_cooperative_matrix_decode_vector
#define VK_NV_cooperative_matrix_decode_vector 1
#define VK_NV_COOPERATIVE_MATRIX_DECODE_VECTOR_EXTENSION_NAME "VK_NV_cooperative_matrix_decode_vector"
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_DECODE_VECTOR_FEATURES_NV ((VkStructureType)1000689000)
typedef struct VkPhysicalDeviceCooperativeMatrixDecodeVectorFeaturesNV {
    VkStructureType    sType;
    void*              pNext;
    VkBool32           cooperativeMatrixDecodeVector;
} VkPhysicalDeviceCooperativeMatrixDecodeVectorFeaturesNV;
#endif

#if __has_include(<spirv/unified1/spirv.hpp>)
#    include <spirv/unified1/spirv.hpp>
#elif __has_include(<spirv-headers/spirv.hpp>)
#    include <spirv-headers/spirv.hpp>
#elif __has_include(<spirv.hpp>)
#    include <spirv.hpp>
#else
     // Fallback to let the compiler throw a standard "file not found" error
#    include <spirv/unified1/spirv.hpp>
#endif

#include <algorithm>

#include <cmath>

#include <iomanip>

#include <iostream>

#include <tuple>

#include <vector>

#include <deque>

#include <sstream>

#include <utility>

#include <memory>

#include <limits>

#include <map>

#include <set>

#include <unordered_map>

#include <shared_mutex>

#include <mutex>

#include <future>

#include <condition_variable>

#include <thread>

#if defined(_MSC_VER)
# define NOMINMAX 1
# include <windows.h>
# define YIELD() YieldProcessor()
#elif defined(__clang__) || defined(__GNUC__)
# if defined(__x86_64__) ||defined(__i386__)
#  include <immintrin.h>
#  define YIELD() _mm_pause()
# elif defined(__arm__) || defined(__aarch64__)
#  if defined(__clang__)
#   include <arm_acle.h>
#   define YIELD() __yield()
#  else
#   define YIELD() asm volatile("yield")
#  endif
# endif
#endif

#if !defined(YIELD)
#define YIELD()
#endif

#include "ggml-impl.h"

#include "ggml-backend-impl.h"

#include "ggml-vulkan-shaders.hpp"

#if !defined(VK_KHR_shader_bfloat16)

#define VK_KHR_shader_bfloat16 1
#define VK_KHR_SHADER_BFLOAT16_SPEC_VERSION                          1
#define VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME                        "VK_KHR_shader_bfloat16"
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR ((VkStructureType)1000141000)
#define VK_COMPONENT_TYPE_BFLOAT16_KHR                               ((VkComponentTypeKHR)1000141000)

typedef struct VkPhysicalDeviceShaderBfloat16FeaturesKHR {
    VkStructureType                       sType;
    void*                                 pNext;
    VkBool32                              shaderBFloat16Type;
    VkBool32                              shaderBFloat16DotProduct;
    VkBool32                              shaderBFloat16CooperativeMatrix;
} VkPhysicalDeviceShaderBfloat16FeaturesKHR;
#endif

#if !defined(VK_VALVE_shader_mixed_float_dot_product)
#define VK_VALVE_shader_mixed_float_dot_product 1
#define VK_VALVE_SHADER_MIXED_FLOAT_DOT_PRODUCT_SPEC_VERSION 1
#define VK_VALVE_SHADER_MIXED_FLOAT_DOT_PRODUCT_EXTENSION_NAME "VK_VALVE_shader_mixed_float_dot_product"
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MIXED_FLOAT_DOT_PRODUCT_FEATURES_VALVE ((VkStructureType)1000673000)
typedef struct VkPhysicalDeviceShaderMixedFloatDotProductFeaturesVALVE {
    VkStructureType    sType;
    void*              pNext;
    VkBool32           shaderMixedFloatDotProductFloat16AccFloat32;
    VkBool32           shaderMixedFloatDotProductFloat16AccFloat16;
    VkBool32           shaderMixedFloatDotProductBFloat16Acc;
    VkBool32           shaderMixedFloatDotProductFloat8AccFloat32;
} VkPhysicalDeviceShaderMixedFloatDotProductFeaturesVALVE;
#endif

#define ROUNDUP_POW2(M, N) (((M) + (N) - 1) & ~((N) - 1))

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

static bool is_pow2(uint32_t x) { return x > 1 && (x & (x-1)) == 0; }

#define VK_VENDOR_ID_AMD 0x1002

#define VK_VENDOR_ID_APPLE 0x106b

#define VK_VENDOR_ID_INTEL 0x8086

#define VK_VENDOR_ID_NVIDIA 0x10de

#define VK_VENDOR_ID_QUALCOMM 0x5143

#define VK_DEVICE_DESCRIPTOR_POOL_SIZE 256

#define VK_CHECK(err, msg)                                          \
    do {                                                            \
        vk::Result err_ = (err);                                    \
        if (err_ != vk::Result::eSuccess) {                         \
            fprintf(stderr, "ggml_vulkan: %s error %s at %s:%d\n",  \
                #err, to_string(err_).c_str(), __FILE__, __LINE__); \
            exit(1);                                                \
        }                                                           \
    } while (0)

#ifdef GGML_VULKAN_DEBUG
#define VK_LOG_DEBUG(msg) std::cerr << msg << std::endl
#else
#define VK_LOG_DEBUG(msg) ((void) 0)
#endif // GGML_VULKAN_DEBUG

#define MAX_PARAMETER_COUNT 12

#define MAX_FUSED_ADDS (MAX_PARAMETER_COUNT - 3)

struct vk_pipeline_struct;

typedef std::shared_ptr<struct vk_pipeline_struct> vk_pipeline;

struct vk_pipeline_struct {
    std::string name;
    vk::ShaderModule shader_module;
    vk::PipelineLayout layout;
    vk::Pipeline pipeline;
    uint32_t push_constant_size;
    uint32_t parameter_count;
    std::array<uint32_t, 3> wg_denoms;
    uint32_t align;
    // true if fields have been set by ggml_vk_create_pipeline
    bool initialized {};
    // true while a compile is in flight, used to dedupe concurrent claims.
    // Protected by device->compile_mutex.
    bool compile_pending {};
    // set to true when the shader has been compiled
    std::atomic<bool> compiled {};
    // number of registers used, extracted from pipeline executable properties
    uint32_t register_count {};

#if defined(VK_EXT_shader_64bit_indexing)
    bool is_64b_indexing {};
#endif
    // linked list of pipelines for multiple compilation variants.
    // currently only used to compile a 64-bit indexing variant.
    vk_pipeline next;
};

typedef std::weak_ptr<vk_pipeline_struct> vk_pipeline_ref;

struct vk_matmul_pipeline_struct {
    vk_pipeline l, m, s;
    vk_pipeline a_l, a_m, a_s;
    // Returns true when all unaligned pipelines are null.
    // We only check for unaligned variants since one of the unaligned pipelines must exist
    // while aligned pipelines are optional
    bool is_empty() const {
        return l == nullptr && m == nullptr && s == nullptr;
    }
};

typedef std::shared_ptr<vk_matmul_pipeline_struct> vk_matmul_pipeline;

struct vk_matmul_pipeline2 {
    vk_matmul_pipeline2() {
        f16acc = std::make_shared<vk_matmul_pipeline_struct>();
        f32acc = std::make_shared<vk_matmul_pipeline_struct>();
    }
    vk_matmul_pipeline f32acc;
    vk_matmul_pipeline f16acc;
};

struct vk_device_struct;

typedef std::shared_ptr<vk_device_struct> vk_device;

typedef std::weak_ptr<vk_device_struct> vk_device_ref;

struct vk_buffer_struct;

typedef std::shared_ptr<vk_buffer_struct> vk_buffer;

typedef std::weak_ptr<vk_buffer_struct> vk_buffer_ref;

struct ggml_backend_vk_buffer_type_context {
    std::string name;
    vk_device device;
};

struct vk_command_buffer {
    vk::CommandBuffer buf;
    uint64_t use_counter = 0;
    bool in_use = false;
};

struct vk_queue;

struct vk_command_pool {
    void init(vk_device& device, vk_queue *q_);
    void destroy(vk::Device& device);

    vk::CommandPool pool;
    // Using deque so the pointers to command buffers
    // remain valid even if we add more
    std::deque<vk_command_buffer> cmd_buffers;

    vk_queue *q;

    size_t buffers_in_use() const {
        return std::count_if(cmd_buffers.begin(), cmd_buffers.end(),
            [](const auto& cb) { return cb.in_use; });
    }
};

struct vk_queue {
    uint32_t queue_family_index;
    vk::Queue queue;

    vk_command_pool cmd_pool;

    vk::PipelineStageFlags stage_flags;

    bool transfer_only;

    // copy everything except the cmd_pool
    void copyFrom(vk_queue &other) {
        queue_family_index = other.queue_family_index;
        queue = other.queue;
        stage_flags = other.stage_flags;
        transfer_only = other.transfer_only;
    }
};

static constexpr uint32_t mul_mat_vec_max_cols = 8;

static constexpr uint32_t p021_max_gqa_ratio = 8;

enum vk_device_architecture {
    OTHER,
    AMD_GCN,
    AMD_RDNA1,
    AMD_RDNA2,
    AMD_RDNA3,
    INTEL_XE2,
    NVIDIA_PRE_TURING,
    NVIDIA_TURING,
};

enum vk_conv_shapes {
    CONV_SHAPE_128x128,
    CONV_SHAPE_64x32,
    CONV_SHAPE_32x256,
    CONV_SHAPE_64x128,
    CONV_SHAPE_COUNT,
};

struct vk_conv_block_size {
    uint32_t K;
    uint32_t NPQ;
    uint32_t CRS;
};

inline vk_conv_block_size vk_conv_block_sizes[CONV_SHAPE_COUNT] = {
    // K   NPQ  CRS
    { 128, 128, 16 }, // CONV_SHAPE_128x128
    {  64,  32, 32 }, // CONV_SHAPE_64x32
    {  32, 256, 16 }, // CONV_SHAPE_32x256
    {  64, 128, 16 }, // CONV_SHAPE_64x128
};

enum dmmv_wg_sizes {
    DMMV_WG_SIZE_SUBGROUP,
    DMMV_WG_SIZE_LARGE,
    DMMV_WG_SIZE_COUNT,
};

enum FaCodePath {
    FA_SCALAR,
    FA_COOPMAT1,
    FA_COOPMAT2,
};

struct vk_fa_pipeline_state {
    uint32_t HSK, HSV;
    uint32_t Br, Bc;
    uint32_t D_split, row_split;
    bool shmem_staging;
    FaCodePath path;
    uint32_t workgroup_size, subgroup_size;
    bool aligned;
    bool f32acc;
    uint32_t flags;
    uint32_t limit_occupancy_shmem;
    ggml_type k_type;
    ggml_type v_type;

    bool operator<(const vk_fa_pipeline_state &b) const {
        return std::tie(HSK, HSV, Br, Bc, D_split, row_split, shmem_staging, path, workgroup_size, subgroup_size, aligned, f32acc, flags, limit_occupancy_shmem, k_type, v_type) <
               std::tie(b.HSK, b.HSV, b.Br, b.Bc, b.D_split, b.row_split, b.shmem_staging, b.path, b.workgroup_size, b.subgroup_size, b.aligned, b.f32acc, b.flags, b.limit_occupancy_shmem, b.k_type, b.v_type);
    }
};

struct vk_conv2d_pipeline_state {
    vk_conv2d_pipeline_state(uint32_t s0, uint32_t s1, uint32_t p0, uint32_t p1, uint32_t d0, uint32_t d1, uint32_t KW, uint32_t KH, uint32_t aligned)
        : s0(s0), s1(s1), p0(p0), p1(p1), d0(d0), d1(d1), KW(KW), KH(KH), aligned(aligned) {}

    uint32_t s0, s1, p0, p1, d0, d1, KW, KH;
    // when set, shader can skip K/CRS/NPQ bounds checks and address clamps
    uint32_t aligned;

    bool operator<(const vk_conv2d_pipeline_state &b) const {
        return std::tie(s0, s1, p0, p1, d0, d1, KW, KH, aligned) <
               std::tie(b.s0, b.s1, b.p0, b.p1, b.d0, b.d1, b.KW, b.KH, b.aligned);
    }
};

struct vk_solve_tri_pipeline_state {
    vk_solve_tri_pipeline_state(uint32_t N, uint32_t K)
        : N(N), K(K) {}

    uint32_t N, K;

    bool operator<(const vk_solve_tri_pipeline_state &b) const {
        return std::tie(N, K) <
               std::tie(b.N, b.K);
    }
};

enum shader_reduction_mode {
    SHADER_REDUCTION_MODE_SHMEM,
    SHADER_REDUCTION_MODE_HYBRID,
    SHADER_REDUCTION_MODE_SUBGROUP,
    SHADER_REDUCTION_MODE_COUNT,
};

static constexpr uint32_t num_argsort_pipelines = 11;

static constexpr uint32_t num_topk_moe_pipelines = 10;

static constexpr uint32_t num_topk_pipelines = 11;

static constexpr std::initializer_list<ggml_op> topk_moe_early_softmax_norm{ GGML_OP_SOFT_MAX, GGML_OP_RESHAPE,  GGML_OP_ARGSORT,
                                                                             GGML_OP_VIEW,     GGML_OP_GET_ROWS, GGML_OP_RESHAPE,
                                                                             GGML_OP_SUM_ROWS, GGML_OP_CLAMP,    GGML_OP_DIV,
                                                                             GGML_OP_RESHAPE };

static constexpr std::initializer_list<ggml_op> topk_moe_sigmoid_norm_bias{ GGML_OP_UNARY,    GGML_OP_RESHAPE,  GGML_OP_ADD,
                                                                            GGML_OP_ARGSORT,  GGML_OP_VIEW,     GGML_OP_GET_ROWS,
                                                                            GGML_OP_RESHAPE,  GGML_OP_SUM_ROWS, GGML_OP_CLAMP,
                                                                            GGML_OP_DIV,      GGML_OP_RESHAPE };

static constexpr std::initializer_list<ggml_op> topk_moe_early_softmax     { GGML_OP_SOFT_MAX, GGML_OP_RESHAPE,  GGML_OP_ARGSORT,
                                                                             GGML_OP_VIEW,     GGML_OP_GET_ROWS };

static constexpr std::initializer_list<ggml_op> topk_moe_late_softmax      { GGML_OP_ARGSORT,  GGML_OP_VIEW,
                                                                             GGML_OP_GET_ROWS, GGML_OP_RESHAPE,
                                                                             GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

static constexpr std::initializer_list<ggml_op> snake_pattern              { GGML_OP_MUL,      GGML_OP_SIN,
                                                                             GGML_OP_SQR,      GGML_OP_MUL,
                                                                             GGML_OP_ADD };

static constexpr std::initializer_list<std::array<int, 3>> topk_moe_early_softmax_norm_edges {
    { 1, 0, 0 }, // reshape->src[0]  == softmax
    { 2, 0, 0 }, // argsort->src[0]  == softmax
    { 3, 0, 2 }, // view->src[0]     == argsort
    { 4, 0, 1 }, // get_rows->src[0] == reshape
    { 4, 1, 3 }, // get_rows->src[1] == view
    { 5, 0, 4 }, // reshape->src[0]  == get_rows
    { 6, 0, 5 }, // sum_rows->src[0] == reshape
    { 7, 0, 6 }, // clamp->src[0]    == sum_rows
    { 8, 0, 5 }, // div->src[0]      == reshape
    { 8, 1, 7 }, // div->src[1]      == clamp
    { 9, 0, 8 }, // reshape->src[0]  == div
};

static constexpr std::initializer_list<std::array<int, 3>> topk_moe_sigmoid_norm_bias_edges {
    { 1, 0, 0 }, // reshape->src[0]  == sigmoid
    { 2, 0, 0 }, // add->src[0]      == sigmoid
    { 3, 0, 2 }, // argsort->src[0]  == add
    { 4, 0, 3 }, // view->src[0]     == argsort
    { 5, 0, 1 }, // get_rows->src[0] == reshape
    { 5, 1, 4 }, // get_rows->src[1] == view
    { 6, 0, 5 }, // reshape->src[0]  == get_rows
    { 7, 0, 6 }, // sum_rows->src[0] == reshape
    { 8, 0, 7 }, // clamp->src[0]    == sum_rows
    { 9, 0, 6 }, // div->src[0]      == reshape
    { 9, 1, 8 }, // div->src[1]      == clamp
    {10, 0, 9 }, // reshape->src[0]  == div
};

static constexpr std::initializer_list<std::array<int, 3>> topk_moe_early_softmax_edges {
    { 1, 0, 0 }, // reshape->src[0]  == softmax
    { 2, 0, 0 }, // argsort->src[0]  == softmax
    { 3, 0, 2 }, // view->src[0]     == argsort
    { 4, 0, 1 }, // get_rows->src[0] == reshape
    { 4, 1, 3 }, // get_rows->src[1] == view
};

static constexpr std::initializer_list<std::array<int, 3>> topk_moe_late_softmax_edges {
    { 1, 0, 0 }, // view->src[0]     == argsort
    { 2, 1, 1 }, // get_rows->src[1] == view
    { 3, 0, 2 }, // reshape->src[0]  == get_rows
    { 4, 0, 3 }, // soft_max->src[0] == reshape
    { 5, 0, 4 }, // reshape->src[0]  == soft_max
};

enum topk_moe_mode {
    TOPK_MOE_EARLY_SOFTMAX,
    TOPK_MOE_EARLY_SOFTMAX_NORM,
    TOPK_MOE_LATE_SOFTMAX,
    TOPK_MOE_SIGMOID_NORM_BIAS,
    TOPK_MOE_COUNT,
};

static constexpr std::initializer_list<std::array<int, 3>> rope_view_set_rows_edges {
    { 1, 0, 0 }, // view->src[0]     == rope
    { 2, 0, 1 }, // set_rows->src[0] == view
};

static constexpr std::initializer_list<std::array<int, 3>> rms_norm_mul_rope_view_set_rows_edges {
    { 1, 0, 0 }, // mul->src[0]      == rms
    { 2, 0, 1 }, // rope->src[0]     == mul
    { 3, 0, 2 }, // view->src[0]     == rope
    { 4, 0, 3 }, // set_rows->src[0] == view
};

class vk_memory_logger;

void ggml_vk_destroy_buffer(vk_buffer& buf);
void ggml_vk_destroy_pipeline(vk::Device& device, vk_pipeline& pipeline);

struct vk_device_struct {
    std::recursive_mutex mutex;
    mutable std::shared_mutex pinned_memory_mutex;

    // Guards compile_pending, all_pipelines, and the dynamic pipeline maps
    // (flash_attn, fa_mask_opt, solve_tri, conv2d, etc). The actual compile
    // runs with no lock held, so different pipelines can compile in parallel.
    // Lock order is device->mutex -> compile_mutex, never the reverse.
    std::mutex compile_mutex;
    std::condition_variable compile_cv;

    vk::PhysicalDevice physical_device;
    vk::PhysicalDeviceProperties properties;
    std::string name;
    uint64_t max_memory_allocation_size;
    uint64_t max_buffer_size;
    uint64_t suballocation_block_size;
    uint64_t min_imported_host_pointer_alignment;
    bool external_memory_host {};
    bool fp16;
    bool bf16;
    bool pipeline_robustness;
    bool memory_priority;
    vk::Device device;
    uint32_t vendor_id;
    vk::DriverId driver_id;
    vk_device_architecture architecture;
    vk_queue compute_queue;
    vk_queue transfer_queue;
    bool single_queue;
    bool support_async;
    bool async_use_transfer_queue;
    uint32_t subgroup_size;
    uint32_t subgroup_size_log2;
    uint32_t shader_core_count;
    bool uma;
    bool prefer_host_memory;
    bool float_controls_rte_fp16;
    bool subgroup_basic;
    bool subgroup_arithmetic;
    bool subgroup_shuffle;
    bool subgroup_ballot;
    bool subgroup_clustered;
    bool subgroup_vote;
    bool multi_add;
    bool shader_int64;
    bool buffer_device_address;
    bool vulkan_memory_model;

    bool add_rms_fusion;
    uint32_t partials_binding_alignment;

    bool shader_64b_indexing;

    bool integer_dot_product;
    // 0: default, 1: force mmvq, -1: disable mmvq
    int32_t mmvq_mode;

    bool subgroup_size_control;
    uint32_t subgroup_min_size;
    uint32_t subgroup_max_size;
    bool subgroup_require_full_support;

    // floor(log2(maxComputeWorkGroupInvocations))
    uint32_t max_workgroup_size_log2 {};

    bool coopmat_support;
    bool coopmat_acc_f32_support {};
    bool coopmat_acc_f16_support {};
    bool coopmat_bf16_support {};
    bool coopmat_support_16x16x16_f16acc {};
    bool coopmat_support_16x16x16_f32acc {};
    bool coopmat1_fa_support {};
    uint32_t coopmat_m;
    uint32_t coopmat_n;
    uint32_t coopmat_k;

    bool coopmat_int_support;
    uint32_t coopmat_int_m;
    uint32_t coopmat_int_n;
    uint32_t coopmat_int_k;

    bool coopmat2;
    bool coopmat2_bf16_support {};
    bool coopmat2_decode_vector;

    bool dot2_f16 {};

    bool pipeline_executable_properties_support {};

    size_t idx;

    bool mul_mat_l[GGML_TYPE_COUNT];
    bool mul_mat_m[GGML_TYPE_COUNT];
    bool mul_mat_s[GGML_TYPE_COUNT];
    bool mul_mat_id_l[GGML_TYPE_COUNT];
    bool mul_mat_id_m[GGML_TYPE_COUNT];
    bool mul_mat_id_s[GGML_TYPE_COUNT];

    // Separate flags for the q8_1 (integer dot) mmq path, whose shader uses
    // a different shared-memory layout than the float matmul shaders.
    bool mul_mat_l_int[GGML_TYPE_COUNT];
    bool mul_mat_m_int[GGML_TYPE_COUNT];
    bool mul_mat_s_int[GGML_TYPE_COUNT];
    bool mul_mat_id_l_int[GGML_TYPE_COUNT];
    bool mul_mat_id_m_int[GGML_TYPE_COUNT];
    bool mul_mat_id_s_int[GGML_TYPE_COUNT];

    vk::DescriptorSetLayout dsl;

    vk_matmul_pipeline pipeline_matmul_f32 {};
    vk_matmul_pipeline pipeline_matmul_f32_f16 {};
    vk_matmul_pipeline pipeline_matmul_bf16 {};
    vk_matmul_pipeline2 pipeline_matmul_f16;
    vk_matmul_pipeline2 pipeline_matmul_f16_f32;

    vk_matmul_pipeline2 pipeline_dequant_mul_mat_mat[GGML_TYPE_COUNT];
    vk_matmul_pipeline2 pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_COUNT];
    vk_matmul_pipeline2 pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_COUNT];

    vk_matmul_pipeline pipeline_matmul_id_f32 {};
    vk_matmul_pipeline pipeline_matmul_id_bf16 {};
    vk_matmul_pipeline2 pipeline_matmul_id_f16;
    vk_matmul_pipeline2 pipeline_matmul_id_f16_f32;

    vk_matmul_pipeline2 pipeline_dequant_mul_mat_mat_id[GGML_TYPE_COUNT];
    vk_matmul_pipeline2 pipeline_dequant_mul_mat_mat_id_q8_1[GGML_TYPE_COUNT];

    vk_pipeline pipeline_matmul_split_k_reduce;
    vk_pipeline pipeline_quantize_q8_1_x4;

    vk_pipeline pipeline_dequant[GGML_TYPE_COUNT];
    vk_pipeline pipeline_dequant_mul_mat_vec_f32_f32[DMMV_WG_SIZE_COUNT][GGML_TYPE_COUNT][mul_mat_vec_max_cols];
    vk_pipeline pipeline_dequant_mul_mat_vec_f16_f32[DMMV_WG_SIZE_COUNT][GGML_TYPE_COUNT][mul_mat_vec_max_cols];
    vk_pipeline pipeline_dequant_mul_mat_vec_id_f32[DMMV_WG_SIZE_COUNT][GGML_TYPE_COUNT];

    vk_pipeline pipeline_dequant_mul_mat_vec_q8_1_f32[DMMV_WG_SIZE_COUNT][GGML_TYPE_COUNT][mul_mat_vec_max_cols];
    vk_pipeline pipeline_dequant_mul_mat_vec_id_q8_1_f32[DMMV_WG_SIZE_COUNT][GGML_TYPE_COUNT];

    vk_pipeline pipeline_mul_mat_vec_p021_f16_f32[p021_max_gqa_ratio];
    vk_pipeline pipeline_mul_mat_vec_nc_f16_f32;
    vk_pipeline pipeline_get_rows[GGML_TYPE_COUNT];
    vk_pipeline pipeline_get_rows_f32[GGML_TYPE_COUNT];
    vk_pipeline pipeline_acc_f32;
    vk_pipeline pipeline_set_f32;

    // [src0 0=fp32,1=fp16][src1 0=fp32,1=fp16][dst 0=fp32,1=fp16]
    vk_pipeline pipeline_add[2][2][2];
    vk_pipeline pipeline_add_norepeat[2][2][2];
    vk_pipeline pipeline_sub[2][2][2];
    vk_pipeline pipeline_sub_norepeat[2][2][2];
    vk_pipeline pipeline_mul[2][2][2];
    vk_pipeline pipeline_mul_norepeat[2][2][2];
    vk_pipeline pipeline_div[2][2][2];
    vk_pipeline pipeline_div_norepeat[2][2][2];
    vk_pipeline pipeline_add_rms[2][2][2];
    vk_pipeline pipeline_add_rms_norepeat[2][2][2];

    // indexed by num_additional_fused_ops == num_adds - 1
    vk_pipeline pipeline_multi_add[MAX_FUSED_ADDS];
    vk_pipeline pipeline_multi_add_rms[MAX_FUSED_ADDS];

    vk_pipeline pipeline_add_id_f32;

    vk_pipeline pipeline_concat_i8, pipeline_concat_i16, pipeline_concat_i32, pipeline_concat_i64;
    vk_pipeline pipeline_upscale_nearest_f32, pipeline_upscale_bilinear_f32, pipeline_upscale_bicubic_f32, pipeline_upscale_bilinear_antialias_f32;
    vk_pipeline pipeline_scale_f32;
    vk_pipeline pipeline_sqr_f32;
    vk_pipeline pipeline_sqrt_f32;
    vk_pipeline pipeline_sin_f32;
    vk_pipeline pipeline_cos_f32;
    vk_pipeline pipeline_log[2];
    vk_pipeline pipeline_tri[2];
    vk_pipeline pipeline_diag[2];
    vk_pipeline pipeline_clamp_f32;
    vk_pipeline pipeline_pad_f32;
    vk_pipeline pipeline_roll_f32;
    vk_pipeline pipeline_repeat_i32, pipeline_repeat_back_f32;
    vk_pipeline pipeline_repeat_i16;
    vk_pipeline pipeline_cpy_f32_f32, pipeline_cpy_f32_f16, pipeline_cpy_f16_f16, pipeline_cpy_f16_f32, pipeline_cpy_f32_bf16, pipeline_cpy_bf16_f32, pipeline_cpy_f32_i32, pipeline_cpy_i32_f32;
    vk_pipeline pipeline_contig_cpy_f32_f32, pipeline_contig_cpy_f32_f16, pipeline_contig_cpy_f16_f16, pipeline_contig_cpy_f16_f32, pipeline_contig_cpy_f32_bf16, pipeline_contig_cpy_bf16_f32, pipeline_contig_cpy_f32_i32, pipeline_contig_cpy_i32_f32;
    vk_pipeline pipeline_cpy_f32_quant[GGML_TYPE_COUNT];
    vk_pipeline pipeline_cpy_quant_f32[GGML_TYPE_COUNT];
    vk_pipeline pipeline_cpy_transpose_16, pipeline_cpy_transpose_32;
    vk_pipeline pipeline_set_rows_i32[GGML_TYPE_COUNT];
    vk_pipeline pipeline_set_rows_i64[GGML_TYPE_COUNT];
    vk_pipeline pipeline_norm_f32;
    vk_pipeline pipeline_group_norm_f32;
    vk_pipeline pipeline_rms_norm_f32;
    vk_pipeline pipeline_rms_norm_mul_f32;
    vk_pipeline pipeline_rms_norm_partials_f32;
    vk_pipeline pipeline_rms_norm_mul_partials_f32;
    vk_pipeline pipeline_rms_norm_mul_rope_f32_f32;
    vk_pipeline pipeline_rms_norm_mul_rope_f32_f16;
    vk_pipeline pipeline_rms_norm_back_f32;
    vk_pipeline pipeline_l2_norm_f32;

    // [src/dst 0=fp32,1=fp16]
    vk_pipeline pipeline_exp[2];
    vk_pipeline pipeline_expm1[2];
    vk_pipeline pipeline_elu[2];
    vk_pipeline pipeline_gelu[2];
    vk_pipeline pipeline_gelu_erf[2];
    vk_pipeline pipeline_gelu_quick[2];
    vk_pipeline pipeline_silu[2];
    vk_pipeline pipeline_relu[2];
    vk_pipeline pipeline_xielu[2];
    vk_pipeline pipeline_neg[2];
    vk_pipeline pipeline_tanh[2];
    vk_pipeline pipeline_sigmoid[2];
    vk_pipeline pipeline_hardsigmoid[2];
    vk_pipeline pipeline_hardswish[2];
    vk_pipeline pipeline_abs[2];
    vk_pipeline pipeline_softplus[2];
    vk_pipeline pipeline_step[2];
    vk_pipeline pipeline_round[2];
    vk_pipeline pipeline_ceil[2];
    vk_pipeline pipeline_floor[2];
    vk_pipeline pipeline_trunc[2];
    vk_pipeline pipeline_sgn[2];

    vk_pipeline pipeline_add1_f16_f16;
    vk_pipeline pipeline_add1_f16_f32;
    vk_pipeline pipeline_add1_f32_f32;

    vk_pipeline pipeline_arange_f32;

    vk_pipeline pipeline_fill_f32;
    vk_pipeline pipeline_fill_f16;

    vk_pipeline pipeline_geglu[2];
    vk_pipeline pipeline_reglu[2];
    vk_pipeline pipeline_swiglu[2];
    vk_pipeline pipeline_swiglu_oai[2];
    vk_pipeline pipeline_geglu_erf[2];
    vk_pipeline pipeline_geglu_quick[2];

    vk_pipeline pipeline_leaky_relu_f32;
    vk_pipeline pipeline_silu_back_f32;
    vk_pipeline pipeline_diag_mask_inf_f32;
    vk_pipeline pipeline_soft_max_f32, pipeline_soft_max_f32_f16;
    vk_pipeline pipeline_soft_max_f32_wg512, pipeline_soft_max_f32_f16_wg512;
    vk_pipeline pipeline_soft_max_back_f32;

    vk_pipeline pipeline_soft_max_large1_f32, pipeline_soft_max_large1_f32_f16;
    vk_pipeline pipeline_soft_max_large2_f32, pipeline_soft_max_large2_f32_f16;
    vk_pipeline pipeline_soft_max_large3_f32, pipeline_soft_max_large3_f32_f16;

    vk_pipeline pipeline_rope_norm_f32, pipeline_rope_norm_f16, pipeline_rope_norm_f32_f16;
    vk_pipeline pipeline_rope_neox_f32, pipeline_rope_neox_f16, pipeline_rope_neox_f32_f16;
    vk_pipeline pipeline_rope_multi_f32, pipeline_rope_multi_f16, pipeline_rope_multi_f32_f16;
    vk_pipeline pipeline_rope_vision_f32, pipeline_rope_vision_f16;
    vk_pipeline pipeline_argsort_f32[num_argsort_pipelines];
    vk_pipeline pipeline_argsort_large_f32[num_argsort_pipelines];
    vk_pipeline pipeline_topk_f32[num_topk_pipelines];
    vk_pipeline pipeline_sum_rows_f32;
    vk_pipeline pipeline_fwht_f32[4];
    vk_pipeline pipeline_cumsum_f32;
    vk_pipeline pipeline_cumsum_small_f32;
    vk_pipeline pipeline_cumsum_multipass1_f32;
    vk_pipeline pipeline_cumsum_multipass2_f32;
    vk_pipeline pipeline_argmax_f32;
    vk_pipeline pipeline_count_equal_i32;
    std::map<vk_solve_tri_pipeline_state, vk_pipeline> pipeline_solve_tri_f32;
    vk_pipeline pipeline_im2col_f32, pipeline_im2col_f32_f16;
    vk_pipeline pipeline_im2col_3d_f32, pipeline_im2col_3d_f32_f16;
    vk_pipeline pipeline_timestep_embedding_f32;
    vk_pipeline pipeline_conv_transpose_1d_f32;
    vk_pipeline pipeline_col2im_1d_f32;
    vk_pipeline pipeline_col2im_1d_f16;
    vk_pipeline pipeline_col2im_1d_bf16;
    vk_pipeline pipeline_snake_f32;
    vk_pipeline pipeline_snake_f16;
    vk_pipeline pipeline_snake_bf16;
    vk_pipeline pipeline_pool2d_f32;
    vk_pipeline pipeline_rwkv_wkv6_f32;
    vk_pipeline pipeline_rwkv_wkv7_f32;
    // [size_idx][kda] where size_idx: 0=d16, 1=d32, 2=d64, 3=d128
    vk_pipeline pipeline_gated_delta_net[4][2];
    vk_pipeline pipeline_ssm_scan_f32_d128;
    vk_pipeline pipeline_ssm_scan_f32_d256;
    vk_pipeline pipeline_ssm_conv_f32;
    vk_pipeline pipeline_ssm_conv_silu_f32;
    vk_pipeline pipeline_ssm_conv_bias_silu_f32;
    vk_pipeline pipeline_opt_step_adamw_f32;
    vk_pipeline pipeline_opt_step_sgd_f32;
    std::map<vk_conv2d_pipeline_state, vk_pipeline> pipeline_conv2d_f32[CONV_SHAPE_COUNT];
    std::map<vk_conv2d_pipeline_state, vk_pipeline> pipeline_conv2d_f16_f32[CONV_SHAPE_COUNT];
    std::map<vk_conv2d_pipeline_state, vk_pipeline> pipeline_conv_transpose_2d_f32[CONV_SHAPE_COUNT];
    std::map<vk_conv2d_pipeline_state, vk_pipeline> pipeline_conv_transpose_2d_f16_f32[CONV_SHAPE_COUNT];
    vk_pipeline pipeline_conv2d_dw_whcn_f32, pipeline_conv2d_dw_whcn_f16_f32;
    vk_pipeline pipeline_conv2d_dw_cwhn_f32, pipeline_conv2d_dw_cwhn_f16_f32;

    std::map<vk_fa_pipeline_state, vk_pipeline> pipeline_flash_attn_f32_f16;

    std::map<std::pair<uint32_t, uint32_t>, vk_pipeline> pipeline_fa_mask_opt;

    vk_pipeline pipeline_flash_attn_split_k_reduce;
    vk_pipeline pipeline_count_experts;

    // [2] is for whether to take n_experts from spec constant (0) or push constant (1)
    vk_pipeline pipeline_topk_moe[num_topk_moe_pipelines][2];

    std::vector<vk_pipeline_ref> all_pipelines;

    std::vector<std::tuple<void*, size_t, vk_buffer>> pinned_memory;

    vk::Fence fence;
    vk_buffer sync_staging;

    ggml_backend_buffer_type buffer_type;

    bool disable_fusion;
    bool disable_host_visible_vidmem;
    bool allow_sysmem_fallback;
    bool disable_graph_optimize;

    std::unique_ptr<vk_memory_logger> memory_logger;

    ~vk_device_struct() {
        VK_LOG_DEBUG("destroy device " << name);

        device.destroyFence(fence);

        ggml_vk_destroy_buffer(sync_staging);

        compute_queue.cmd_pool.destroy(device);
        transfer_queue.cmd_pool.destroy(device);

        for (auto& pipeline : all_pipelines) {
            if (pipeline.expired()) {
                continue;
            }

            vk_pipeline pl = pipeline.lock();
            ggml_vk_destroy_pipeline(device, pl);
        }
        all_pipelines.clear();

        device.destroyDescriptorSetLayout(dsl);

        device.destroy();
    }
};

inline void vk_command_pool::init(vk_device& device, vk_queue *q_) {
    cmd_buffers.clear();
    q = q_;

    vk::CommandPoolCreateInfo command_pool_create_info(
        vk::CommandPoolCreateFlags(VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT),
        q->queue_family_index);
    pool = device->device.createCommandPool(command_pool_create_info);
}

inline void vk_command_pool::destroy(vk::Device& device) {
    device.destroyCommandPool(pool);
    pool = nullptr;
    cmd_buffers.clear();
}

struct vk_buffer_struct {
    vk::Buffer buffer = VK_NULL_HANDLE;
    vk::DeviceMemory device_memory = VK_NULL_HANDLE;
    vk::MemoryPropertyFlags memory_property_flags;
    void * ptr;
    size_t size = 0;
    vk::DeviceAddress bda_addr {};

    vk_device device;

    ~vk_buffer_struct() {
        if (size == 0) {
            return;
        }
        VK_LOG_DEBUG("~vk_buffer_struct(" << buffer << ", " << size << ")");

        device->device.freeMemory(device_memory);
        device->device.destroyBuffer(buffer);
    }
};

struct vk_subbuffer {
    vk_buffer buffer;
    uint64_t offset;
    uint64_t size;

    operator vk::DescriptorBufferInfo() const {
        return { buffer->buffer, offset, size };
    }
};

struct vk_semaphore {
    vk::Semaphore s;
    uint64_t value;
};

struct vk_event {
    std::vector<vk::Event> events_free; // Events available for reuse
    std::vector<vk::Event> events_submitted; // Events that are fully submitted and can be reused on next synchronize
    vk::Event event;
    bool has_event;

    vk_semaphore tl_semaphore;
    vk_command_buffer* cmd_buffer = nullptr;
    uint64_t cmd_buffer_use_counter = 0;
};

struct vk_submission {
    vk_command_buffer* buffer = nullptr;
    std::vector<vk_semaphore> wait_semaphores;
    std::vector<vk_semaphore> signal_semaphores;
};

typedef std::vector<vk_submission> vk_sequence;

#define MAT_VEC_FUSION_FLAGS_BIAS0 0x1

#define MAT_VEC_FUSION_FLAGS_BIAS1 0x2

#define MAT_VEC_FUSION_FLAGS_SCALE0 0x4

#define MAT_VEC_FUSION_FLAGS_SCALE1 0x8

struct vk_staging_memcpy {
    vk_staging_memcpy(void * _dst, const void * _src, size_t _n) : dst(_dst), src(_src), n(_n) {}

    void * dst;
    const void * src;
    size_t n;
};

struct vk_staging_memset {
    vk_staging_memset(void * _dst, uint32_t _val, size_t _n) : dst(_dst), val(_val), n(_n) {}

    void * dst;
    uint32_t val;
    size_t n;
};

struct vk_context_struct {
    vk_submission * s;
    std::vector<vk_sequence> seqs;

    int exit_tensor_idx;

    std::vector<vk_staging_memcpy> in_memcpys;
    std::vector<vk_staging_memcpy> out_memcpys;
    std::vector<vk_staging_memset> memsets;

    vk_command_pool * p {};
};

typedef std::shared_ptr<vk_context_struct> vk_context;

typedef std::weak_ptr<vk_context_struct> vk_context_ref;

struct ggml_vk_garbage_collector {
    std::vector<vk_semaphore> tl_semaphores;
    std::vector<vk_semaphore> semaphores;
    std::vector<vk::Event> events;
    std::vector<vk_context> contexts;
};

#define VK_LOG_MEMORY(msg) if (vk_memory_logger_enabled) { std::cerr << "ggml_vulkan memory: " << msg << std::endl; }

static std::string format_size(size_t size) {
    const size_t kib = 1024;
    const size_t mib = kib * 1024;
    const size_t gib = mib * 1024;

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);

    if (size >= gib) {
        oss << static_cast<double>(size) / gib << " GiB";
    } else if (size >= mib) {
        oss << static_cast<double>(size) / mib << " MiB";
    } else if (size >= kib) {
        oss << static_cast<double>(size) / kib << " KiB";
    } else {
        oss << size << " B";
    }

    return oss.str();
}

class vk_memory_logger {
public:
    vk_memory_logger(): total_device(0), total_host(0) {}
    void log_allocation(vk_buffer_ref buf_ref, size_t size);
    void log_deallocation(vk_buffer_ref buf_ref);

private:
    std::map<vk::Buffer, size_t> allocations; // Track allocations
    size_t total_device;
    size_t total_host;
    static std::mutex log_mutex;
};

inline std::mutex vk_memory_logger::log_mutex;

extern uint32_t vk_perf_logger_frequency;

class vk_perf_logger {
  public:
    void print_timings(bool force = false) {
        if (timings.empty()) {
            return;
        }
        print_count++;
        if ((print_count % vk_perf_logger_frequency) != 0 && !force) {
            return;
        }
        print_count = 0;
        uint64_t total_all_op_times = 0;
        std::cerr << "----------------\nVulkan Timings:" << std::endl;
        for (const auto & t : timings) {
            uint64_t total_op_times = 0;
            for (const auto & time : t.second) {
                total_op_times += time;
            }
            std::cerr << t.first << ": " << t.second.size() << " x " << (total_op_times / t.second.size() / 1000.0)
                      << " us = " << (total_op_times / 1000.0) << " us";

            // If we have as many flops entries as timing entries for the op, then compute and log the flops/S.
            auto it = flops.find(t.first);
            if (it != flops.end() && (it->second).size() == t.second.size()) {
                uint64_t total_op_flops = 0;
                for (const auto & elem : it->second) {
                    total_op_flops += elem;
                }
                std::cerr << " ("
                          << (double(total_op_flops) / (1000.0 * 1000.0 * 1000.0)) /
                                 (double(total_op_times) / (1000.0 * 1000.0 * 1000.0))
                          << " GFLOPS/s)";
            }

            total_all_op_times += total_op_times;

            std::cerr << std::endl;
        }

        if (timings.size() > 0) {
            std::cerr << "Total time: " << total_all_op_times / 1000.0 << " us." << std::endl;
        }

        timings.clear();
        flops.clear();
    }

    std::string get_node_fusion_name(const ggml_tensor * node, const char *fusion_name, uint64_t *n_flops) {
        *n_flops = 0;
        std::string fusion_str;
        if (fusion_name) {
            fusion_str = fusion_name + std::string(" ");
        }
        if (node->op == GGML_OP_UNARY) {
            return fusion_str + ggml_unary_op_name(ggml_get_unary_op(node));
        }
        if (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID) {
            const uint64_t m     = node->ne[0];
            const uint64_t n     = node->ne[1];
            const uint64_t k     = node->src[1]->ne[0];
            const uint64_t batch = node->ne[2] * node->ne[3];
            std::string    name  = ggml_op_name(node->op);
            if ((node->op == GGML_OP_MUL_MAT && n <= mul_mat_vec_max_cols) ||
                (node->op == GGML_OP_MUL_MAT_ID && node->src[2]->ne[1] == 1)) {
                name += "_VEC";
            }
            name += " ";
            name += ggml_type_name(node->src[0]->type);
            name += " m=" + std::to_string(m) + " n=" + std::to_string(n) + " k=" + std::to_string(k);
            if (node->op == GGML_OP_MUL_MAT_ID) {
                name += " n_expert=" + std::to_string(node->src[0]->ne[2]);
            }
            if (batch > 1) {
                name += " batch=" + std::to_string(batch);
            }
            name = fusion_str + name;
            *n_flops = m * n * (k + (k - 1)) * batch;
            return name;
        }
        if (node->op == GGML_OP_CONV_2D || node->op == GGML_OP_CONV_TRANSPOSE_2D) {
            std::string   name    = ggml_op_name(node->op);
            ggml_tensor * knl     = node->src[0];
            uint64_t      OW      = node->ne[0];
            uint64_t      OH      = node->ne[1];
            uint64_t      N       = node->ne[3];
            uint64_t      Cout    = node->ne[2];
            uint64_t      KW      = knl->ne[0];
            uint64_t      KH      = knl->ne[1];
            uint64_t      Cin     = node->src[1]->ne[2];
            // KxCRS @ CRSxNPQ = KxNPQ -> M=K, K=CRS, N=NPQ
            uint64_t      size_M  = Cout;
            uint64_t      size_K  = Cin * KW * KH;
            uint64_t      size_N  = N * OW * OH;
            *n_flops = size_M * size_N * (size_K + (size_K - 1));
            name += " M=Cout=" + std::to_string(size_M) + ", K=Cin*KW*KH=" + std::to_string(size_K) +
                    ", N=N*OW*OH=" + std::to_string(size_N);
            name = fusion_str + name;
            return name;
        }
        if (node->op == GGML_OP_RMS_NORM) {
            std::string   name    = ggml_op_name(node->op);
            name += "(" + std::to_string(node->ne[0]) + "," + std::to_string(node->ne[1]) + "," + std::to_string(node->ne[2]) + "," + std::to_string(node->ne[3]) + ")";
            name = fusion_str + name;
            return name;
        }
        if (node->op == GGML_OP_FLASH_ATTN_EXT) {
            const ggml_tensor * dst = node;
            const ggml_tensor * q = node->src[0];
            const ggml_tensor * k = node->src[1];
            const ggml_tensor * v = node->src[2];
            const ggml_tensor * m = node->src[3];
            std::stringstream name;
            name << fusion_str;
            name << ggml_op_name(node->op) <<
                " dst(" << dst->ne[0] << "," << dst->ne[1] << "," << dst->ne[2] << "," << dst->ne[3] << "), " <<
                " q(" << q->ne[0] << "," << q->ne[1] << "," << q->ne[2] << "," << q->ne[3] << "), " <<
                " k(" << k->ne[0] << "," << k->ne[1] << "," << k->ne[2] << "," << k->ne[3] << "), " <<
                " v(" << v->ne[0] << "," << v->ne[1] << "," << v->ne[2] << "," << v->ne[3] << "), " <<
                " m(" << (m?m->ne[0]:0) << "," << (m?m->ne[1]:0) << "," << (m?m->ne[2]:0) << "," << (m?m->ne[3]:0) << ")";
            *n_flops = 2ull * q->ne[1] * q->ne[2] * (k->ne[0] + v->ne[0]) * k->ne[1] * q->ne[3];
            return name.str();
        }
        if (node->op == GGML_OP_TOP_K) {
            std::stringstream name;
            name << fusion_str;
            name << ggml_op_name(node->op) <<
                " K=" << node->ne[0] <<
                " (" << node->src[0]->ne[0] << "," << node->src[0]->ne[1] << "," << node->src[0]->ne[2] << "," << node->src[0]->ne[3] << ")";
            return name.str();
        }
        return fusion_str + ggml_op_name(node->op);
    }

    void log_timing(const ggml_tensor * node, const char *fusion_name, uint64_t time) {
        uint64_t n_flops;
        std::string name = get_node_fusion_name(node, fusion_name, &n_flops);
        if (n_flops) {
            flops[name].push_back(n_flops);
        }
        timings[name].push_back(time);
    }

    void log_timing(const std::vector<ggml_tensor *> &nodes, const std::vector<const char *> &names, uint64_t time) {
        uint64_t total_flops = 0;
        std::string name;
        for (size_t n = 0; n < nodes.size(); ++n) {
            uint64_t n_flops = 0;
            name += get_node_fusion_name(nodes[n], names[n], &n_flops);
            total_flops += n_flops;

            if (n != nodes.size() - 1) {
                name += ", ";
            }
        }
        if (total_flops) {
            flops[name].push_back(total_flops);
        }
        timings[name].push_back(time);
    }

  private:
    std::map<std::string, std::vector<uint64_t>> timings;
    std::map<std::string, std::vector<uint64_t>> flops;
    uint32_t print_count {};
};

struct ggml_backend_vk_context {
    std::string name;

    vk_device device;

    size_t semaphore_idx, event_idx;
    ggml_vk_garbage_collector gc;
    size_t prealloc_size_x, prealloc_size_y, prealloc_size_split_k, prealloc_size_add_rms_partials, prealloc_size_add_rms_partials_offset;
    vk_buffer prealloc_x, prealloc_y, prealloc_split_k, prealloc_add_rms_partials, sync_staging;
    vk::Fence fence, almost_ready_fence;
    bool submit_pending {};
    bool almost_ready_fence_pending {};
    // Set before op_add and unset after op_rms_norm to indicate that the add should
    // write partial sums to accumulate the square of the vector components
    bool do_add_rms_partials_offset_calculation;
    bool do_add_rms_partials;

    uint64_t last_total_mul_mat_bytes {};

    // Cache most recent tensor that was converted into prealloc_y, and what pipeline it used to convert.
    vk_pipeline_struct * prealloc_y_last_pipeline_used {};
    const ggml_tensor * prealloc_y_last_tensor_used {};
    // True when prealloc_y holds the padded fp16 layout used by the coopmat2 B decode-vector callback.
    // If false, then it's contiguous.
    bool prealloc_y_last_decode_vector_staging {};

    // Track which nodes have been used since the last sync, and whether they were written to
    std::vector<const ggml_tensor *> unsynced_nodes_written;
    std::vector<const ggml_tensor *> unsynced_nodes_read;
    // Track which prealloc buffers have pending reads that need to be synchronized.
    // These are checked before writing to the buffer (and call ggml_vk_sync_buffers if set),
    // and set to true after the buffer contents are consumed.
    bool prealloc_x_need_sync, prealloc_y_need_sync, prealloc_split_k_need_sync;

    vk_context_ref compute_ctx;

    vk_context_ref transfer_ctx;
    vk_semaphore transfer_semaphore;
    uint64_t transfer_semaphore_last_submitted {};

    std::vector<vk_context_ref> tensor_ctxs;

    std::vector<vk::DescriptorPool> descriptor_pools;
    std::vector<vk::DescriptorSet> descriptor_sets;
    uint32_t descriptor_set_idx {};
    uint32_t pipeline_descriptor_set_requirements {};

    vk_command_pool compute_cmd_pool;
    vk_command_pool transfer_cmd_pool;

    // number of additional consecutive nodes that are being fused with the
    // node currently being processed
    int num_additional_fused_ops {};
    // Bitmask of which fused ops need to write an intermediate value to memory.
    // Bit 'i' means nodes[start_of_fusion + i] writes to memory.
    // If there's no fusion, bit 0 is still set.
    int fused_ops_write_mask {};
    topk_moe_mode fused_topk_moe_mode {};
    bool fused_topk_moe_scale {};

    // for GGML_VK_PERF_LOGGER
    std::unique_ptr<vk_perf_logger> perf_logger;
    vk::QueryPool query_pool;
    std::vector<const char *> query_fusion_names;
    std::vector<int> query_fusion_node_count;
    std::vector<ggml_tensor *> query_nodes;
    std::vector<int> query_node_idx;
    int32_t num_queries {};
    int32_t query_idx {};
};

struct ggml_backend_vk_buffer_context {
    vk_device_ref device;
    vk_buffer dev_buffer;
    std::string name;

    ggml_backend_vk_buffer_context(vk_device_ref device, vk_buffer&& dev_buffer, std::string& name) :
        device(device),
        dev_buffer(dev_buffer),
        name(name) {
    }

    ~ggml_backend_vk_buffer_context() {
        ggml_vk_destroy_buffer(dev_buffer);
    }
};

struct vk_instance_t {
    vk::Instance instance;

    bool debug_utils_support = false;  // VK_EXT_debug_utils enabled
    PFN_vkSetDebugUtilsObjectNameEXT pfn_vkSetDebugUtilsObjectNameEXT = {};
    PFN_vkQueueBeginDebugUtilsLabelEXT pfn_vkQueueBeginDebugUtilsLabelEXT = {};
    PFN_vkQueueEndDebugUtilsLabelEXT   pfn_vkQueueEndDebugUtilsLabelEXT   = {};
    PFN_vkCmdBeginDebugUtilsLabelEXT   pfn_vkCmdBeginDebugUtilsLabelEXT   = {};
    PFN_vkCmdEndDebugUtilsLabelEXT pfn_vkCmdEndDebugUtilsLabelEXT = {};
    PFN_vkCmdInsertDebugUtilsLabelEXT  pfn_vkCmdInsertDebugUtilsLabelEXT  = {};

    std::vector<size_t> device_indices;
    std::vector<bool>   device_supports_membudget;
    vk_device devices[GGML_VK_MAX_DEVICES];
};

typedef void (*ggml_vk_func_t)(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

static constexpr uint32_t kSpvOpCooperativeMatrixLoadTensorNV = 5367;

static constexpr uint32_t kSpvCapabilityCooperativeMatrixDecodeVectorNV = 5447;

static constexpr uint32_t kSpvTensorAddressingDecodeVectorFuncBit = 0x4;

struct vk_fa_tuning_params {
    FaCodePath path;
    uint32_t workgroup_size;
    uint32_t subgroup_size;
    uint32_t block_rows;
    uint32_t block_cols;
    uint32_t d_split;
    uint32_t row_split;
    bool shmem_staging;
    bool disable_subgroups;
    uint32_t limit_occupancy_shmem;

    void print() const {
        std::cerr << "path=" << path << " workgroup_size=" << workgroup_size << " subgroup_size=" << subgroup_size <<
                     " block_rows=" << block_rows << " block_cols=" << block_cols << " d_split=" << d_split <<
                     " row_split=" << row_split << " shmem_staging=" << shmem_staging << " disable_subgroups=" << disable_subgroups <<
                     " limit_occupancy_shmem=" << limit_occupancy_shmem << std::endl;
    }
};

struct GpuPipelineConfig {
    // GPU architecture identifier.
    // Example: vk_device_architecture::AMD_GCN
    vk_device_architecture arch;

    // Mapping of pipeline names to their specific subgroup sizes.
    // Example: {"soft_max_f32", 64}
    std::unordered_map<std::string, uint32_t> pipelines;

    // Default subgroup size for this GPU.
    // Defaults to 0 if not explicitly provided.
    uint32_t default_subgroup_size = 0;
};

static constexpr uint32_t RDNA_DEFAULT_SUBGROUP_SIZE = 32;

struct CompileTask {
    vk_pipeline pipeline;
    size_t spv_size;
    const void * spv_data;
    std::string entrypoint;
    uint32_t parameter_count;
    std::array<uint32_t, 3> wg_denoms;
    std::vector<uint32_t> specialization_constants;
    bool disable_robustness;
    bool require_full_subgroups;
    uint32_t required_subgroup_size;
};

#define UNUSED GGML_UNUSED

struct ggml_backend_vk_device_context {
    size_t device;
    std::string name;
    std::string description;
    bool is_integrated_gpu;
    std::string pci_bus_id;
    int op_offload_min_batch_size;
};

