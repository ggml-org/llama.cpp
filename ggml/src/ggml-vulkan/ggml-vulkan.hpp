#ifndef GGML_VULKAN_HPP
#define GGML_VULKAN_HPP

#include <vulkan/vulkan.hpp>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "ggml.h"
#include "ggml-backend-impl.h"

#include "ggml-vulkan-matmul.hpp"

#define ROUNDUP_POW2(M, N) (((M) + (N) - 1) & ~((N) - 1))
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

#define VK_VENDOR_ID_AMD 0x1002
#define VK_VENDOR_ID_APPLE 0x106b
#define VK_VENDOR_ID_INTEL 0x8086
#define VK_VENDOR_ID_NVIDIA 0x10de

#define VK_DEVICE_DESCRIPTOR_POOL_SIZE 32

#define GGML_VK_MAX_NODES 8192

#define MAX_VK_BUFFERS 256

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

struct ggml_backend_vk_context;

struct vk_queue {
    uint32_t queue_family_index;
    vk::Queue queue;
    vk::CommandPool pool;
    uint32_t cmd_buffer_idx;
    std::vector<vk::CommandBuffer> cmd_buffers;

    vk::PipelineStageFlags stage_flags;

    bool transfer_only;
};

struct vk_pipeline_struct {
    std::string name;
    vk::ShaderModule shader_module;
    vk::DescriptorSetLayout dsl;
    std::vector<vk::DescriptorPool> descriptor_pools;
    std::vector<vk::DescriptorSet> descriptor_sets;
    uint32_t descriptor_set_idx;
    vk::PipelineLayout layout;
    vk::Pipeline pipeline;
    uint32_t push_constant_size;
    uint32_t parameter_count;
    std::array<uint32_t, 3> wg_denoms;
    uint32_t align;
    // set to true to request the pipeline is compiled after the dryrun
    bool needed {};
    // set to true when the shader has been compiled
    bool compiled {};
};

typedef std::shared_ptr<vk_pipeline_struct> vk_pipeline;
typedef std::weak_ptr<vk_pipeline_struct> vk_pipeline_ref;

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

enum vk_device_architecture {
    OTHER,
    AMD_GCN,
    AMD_RDNA1,
    AMD_RDNA2,
    AMD_RDNA3,
};

static constexpr uint32_t mul_mat_vec_max_cols = 8;

struct vk_device_struct {
    std::mutex mutex;

    vk::PhysicalDevice physical_device;
    vk::PhysicalDeviceProperties properties;
    std::string name;
    uint64_t max_memory_allocation_size;
    uint64_t suballocation_block_size;
    bool fp16;
    bool pipeline_robustness;
    vk::Device device;
    uint32_t vendor_id;
    vk_device_architecture architecture;
    vk_queue compute_queue;
    vk_queue transfer_queue;
    bool single_queue;
    uint32_t subgroup_size;
    uint32_t shader_core_count;
    bool uma;
    bool prefer_host_memory;
    bool float_controls_rte_fp16;

    bool subgroup_size_control;
    uint32_t subgroup_min_size;
    uint32_t subgroup_max_size;
    bool subgroup_require_full_support;

    bool coopmat_support;
    bool coopmat_acc_f32_support;
    bool coopmat_acc_f16_support;
    uint32_t coopmat_m;
    uint32_t coopmat_n;
    uint32_t coopmat_k;
    bool coopmat2;

    size_t idx;

    bool mul_mat_l[GGML_TYPE_COUNT];
    bool mul_mat_m[GGML_TYPE_COUNT];
    bool mul_mat_s[GGML_TYPE_COUNT];
    bool mul_mat_id_l[GGML_TYPE_COUNT];
    bool mul_mat_id_m[GGML_TYPE_COUNT];
    bool mul_mat_id_s[GGML_TYPE_COUNT];

    // set to true to indicate that some shaders need to be compiled after the dryrun
    bool need_compiles {};

    vk_matmul_pipeline pipeline_matmul_f32 {};
    vk_matmul_pipeline pipeline_matmul_f32_f16 {};
    vk_matmul_pipeline2 pipeline_matmul_f16;
    vk_matmul_pipeline2 pipeline_matmul_f16_f32;
    vk_pipeline pipeline_matmul_split_k_reduce;

    vk_matmul_pipeline2 pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_COUNT];
    vk_matmul_pipeline2 pipeline_dequant_mul_mat_mat[GGML_TYPE_COUNT];

    vk_matmul_pipeline pipeline_matmul_id_f32 {};
    vk_matmul_pipeline2 pipeline_matmul_id_f16;
    vk_matmul_pipeline2 pipeline_matmul_id_f16_f32;

    vk_matmul_pipeline2 pipeline_dequant_mul_mat_mat_id[GGML_TYPE_COUNT];

    vk_pipeline pipeline_dequant[GGML_TYPE_COUNT];
    vk_pipeline pipeline_dequant_mul_mat_vec_f32_f32[GGML_TYPE_COUNT][mul_mat_vec_max_cols];
    vk_pipeline pipeline_dequant_mul_mat_vec_f16_f32[GGML_TYPE_COUNT][mul_mat_vec_max_cols];
    vk_pipeline pipeline_dequant_mul_mat_vec_id_f32[GGML_TYPE_COUNT];

    vk_pipeline pipeline_mul_mat_vec_p021_f16_f32;
    vk_pipeline pipeline_mul_mat_vec_nc_f16_f32;
    vk_pipeline pipeline_get_rows[GGML_TYPE_COUNT];
    vk_pipeline pipeline_get_rows_f32[GGML_TYPE_COUNT];
    vk_pipeline pipeline_acc_f32;
    vk_pipeline pipeline_add_f32, pipeline_add_f32_norepeat;
    vk_pipeline pipeline_add_f16_f32_f16, pipeline_add_f16_f32_f16_norepeat;
    vk_pipeline pipeline_sub_f32, pipeline_sub_f32_norepeat;
    vk_pipeline pipeline_mul_f32, pipeline_mul_f32_norepeat;
    vk_pipeline pipeline_div_f32, pipeline_div_f32_norepeat;
    vk_pipeline pipeline_concat_f32, pipeline_concat_f16, pipeline_concat_i32;
    vk_pipeline pipeline_upscale_f32;
    vk_pipeline pipeline_scale_f32;
    vk_pipeline pipeline_sqr_f32;
    vk_pipeline pipeline_sin_f32;
    vk_pipeline pipeline_cos_f32;
    vk_pipeline pipeline_clamp_f32;
    vk_pipeline pipeline_pad_f32;
    vk_pipeline pipeline_repeat_f32, pipeline_repeat_back_f32;
    vk_pipeline pipeline_cpy_f32_f32, pipeline_cpy_f32_f16, pipeline_cpy_f16_f16;
    vk_pipeline pipeline_contig_cpy_f32_f32, pipeline_contig_cpy_f32_f16, pipeline_contig_cpy_f16_f16;
    vk_pipeline pipeline_cpy_f32_quant[GGML_TYPE_COUNT];
    vk_pipeline pipeline_cpy_quant_f32[GGML_TYPE_COUNT];
    vk_pipeline pipeline_norm_f32;
    vk_pipeline pipeline_group_norm_f32;
    vk_pipeline pipeline_rms_norm_f32;
    vk_pipeline pipeline_rms_norm_back_f32;
    vk_pipeline pipeline_l2_norm_f32;
    vk_pipeline pipeline_gelu_f32;
    vk_pipeline pipeline_gelu_quick_f32;
    vk_pipeline pipeline_silu_f32;
    vk_pipeline pipeline_silu_back_f32;
    vk_pipeline pipeline_relu_f32;
    vk_pipeline pipeline_leaky_relu_f32;
    vk_pipeline pipeline_tanh_f32;
    vk_pipeline pipeline_sigmoid_f32;
    vk_pipeline pipeline_diag_mask_inf_f32;
    vk_pipeline pipeline_soft_max_f32, pipeline_soft_max_f32_f16;
    vk_pipeline pipeline_soft_max_f32_wg512, pipeline_soft_max_f32_f16_wg512;
    vk_pipeline pipeline_soft_max_back_f32;
    vk_pipeline pipeline_rope_norm_f32, pipeline_rope_norm_f16;
    vk_pipeline pipeline_rope_neox_f32, pipeline_rope_neox_f16;
    vk_pipeline pipeline_rope_multi_f32, pipeline_rope_multi_f16;
    vk_pipeline pipeline_rope_vision_f32, pipeline_rope_vision_f16;
    vk_pipeline pipeline_argsort_f32;
    vk_pipeline pipeline_sum_rows_f32;
    vk_pipeline pipeline_argmax_f32;
    vk_pipeline pipeline_count_equal_i32;
    vk_pipeline pipeline_im2col_f32, pipeline_im2col_f32_f16;
    vk_pipeline pipeline_timestep_embedding_f32;
    vk_pipeline pipeline_pool2d_f32;
    vk_pipeline pipeline_rwkv_wkv6_f32;
    vk_pipeline pipeline_rwkv_wkv7_f32;
    vk_pipeline pipeline_opt_step_adamw_f32;

    // [2][2][2] is for {f16acc,f32acc}x{large,small_rows}x{unaligned, aligned}
    vk_pipeline pipeline_flash_attn_f32_f16_D64[GGML_TYPE_COUNT][2][2][2];
    vk_pipeline pipeline_flash_attn_f32_f16_D80[GGML_TYPE_COUNT][2][2][2];
    vk_pipeline pipeline_flash_attn_f32_f16_D96[GGML_TYPE_COUNT][2][2][2];
    vk_pipeline pipeline_flash_attn_f32_f16_D112[GGML_TYPE_COUNT][2][2][2];
    vk_pipeline pipeline_flash_attn_f32_f16_D128[GGML_TYPE_COUNT][2][2][2];
    vk_pipeline pipeline_flash_attn_f32_f16_D256[GGML_TYPE_COUNT][2][2][2];

    std::unordered_map<std::string, vk_pipeline_ref> pipelines;
    std::unordered_map<std::string, uint64_t> pipeline_descriptor_set_requirements;

    std::vector<std::tuple<void*, size_t, vk_buffer>> pinned_memory;

    vk::Fence fence;
    vk_buffer sync_staging;

    ggml_backend_buffer_type buffer_type;

#ifdef GGML_VULKAN_MEMORY_DEBUG
    std::unique_ptr<vk_memory_logger> memory_logger;
#endif
#ifdef GGML_VULKAN_PERF
    std::unique_ptr<vk_perf_logger> perf_logger;
#endif

    ~vk_device_struct();
};

struct vk_buffer_struct {
    vk::Buffer buffer = VK_NULL_HANDLE;
    vk::DeviceMemory device_memory = VK_NULL_HANDLE;
    vk::MemoryPropertyFlags memory_property_flags;
    void * ptr;
    size_t size = 0;

    vk_device device;

    ~vk_buffer_struct();
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

struct vk_submission {
    vk::CommandBuffer buffer;
    std::vector<vk_semaphore> wait_semaphores;
    std::vector<vk_semaphore> signal_semaphores;
};

typedef std::vector<vk_submission> vk_sequence;

struct vk_mat_mat_push_constants {
    uint32_t M; uint32_t N; uint32_t K;
    uint32_t stride_a; uint32_t stride_b; uint32_t stride_d;
    uint32_t batch_stride_a; uint32_t batch_stride_b; uint32_t batch_stride_d;
    uint32_t k_split;
    uint32_t ne02; uint32_t ne12; uint32_t broadcast2; uint32_t broadcast3;
    uint32_t padded_N;
};
struct vk_mat_vec_push_constants {
    uint32_t ncols; uint32_t stride_a; uint32_t stride_b; uint32_t stride_d;
    uint32_t batch_stride_a; uint32_t batch_stride_b; uint32_t batch_stride_d;
    uint32_t ne02; uint32_t ne12; uint32_t broadcast2; uint32_t broadcast3;
};

struct vk_mat_mat_id_push_constants {
    uint32_t M; uint32_t N; uint32_t K;
    uint32_t stride_a; uint32_t stride_b; uint32_t stride_d;
    uint32_t batch_stride_a; uint32_t batch_stride_b; uint32_t batch_stride_d;
    uint32_t nei0; uint32_t nei1; uint32_t nbi1; uint32_t ne11;
    uint32_t padded_N;
};
struct vk_mat_vec_id_push_constants {
    uint32_t ncols; uint32_t stride_a; uint32_t stride_b; uint32_t stride_d;
    uint32_t batch_stride_a; uint32_t batch_stride_b; uint32_t batch_stride_d;
    uint32_t nei0; uint32_t ne11;
};

struct vk_flash_attn_push_constants {
    uint32_t N;
    uint32_t KV;

    uint32_t ne1;
    uint32_t ne2;
    uint32_t ne3;

    uint32_t neq2;
    uint32_t neq3;
    uint32_t nek2;
    uint32_t nek3;
    uint32_t nev2;
    uint32_t nev3;
    uint32_t nem1;

    uint32_t nb01;
    uint32_t nb02;
    uint32_t nb03;
    uint32_t nb11;
    uint32_t nb12;
    uint32_t nb13;
    uint32_t nb21;
    uint32_t nb22;
    uint32_t nb23;
    uint32_t nb31;

    float scale;
    float max_bias;
    float logit_softcap;

    uint32_t mask;
    uint32_t n_head_log2;
    float m0;
    float m1;
};

struct vk_op_push_constants {
    uint32_t KX;
    uint32_t KY;
    float param1;
    float param2;
};

struct vk_op_unary_push_constants {
    uint32_t ne;
    uint32_t ne00; uint32_t ne01; uint32_t ne02; uint32_t ne03; uint32_t nb00; uint32_t nb01; uint32_t nb02; uint32_t nb03;
    uint32_t ne10; uint32_t ne11; uint32_t ne12; uint32_t ne13; uint32_t nb10; uint32_t nb11; uint32_t nb12; uint32_t nb13;
    uint32_t misalign_offsets;
    float param1; float param2;
    uint32_t ne0_012mp; uint32_t ne0_012L;
    uint32_t ne0_01mp;  uint32_t ne0_01L;
    uint32_t ne0_0mp;   uint32_t ne0_0L;
    uint32_t ne1_012mp; uint32_t ne1_012L;
    uint32_t ne1_01mp;  uint32_t ne1_01L;
    uint32_t ne1_0mp;   uint32_t ne1_0L;
};
static_assert(sizeof(vk_op_unary_push_constants) <= 128, "sizeof(vk_op_unary_push_constants) must be <= 128");

struct vk_op_binary_push_constants {
    uint32_t ne;
    uint32_t ne00; uint32_t ne01; uint32_t ne02; uint32_t ne03; uint32_t nb00; uint32_t nb01; uint32_t nb02; uint32_t nb03;
    uint32_t ne10; uint32_t ne11; uint32_t ne12; uint32_t ne13; uint32_t nb10; uint32_t nb11; uint32_t nb12; uint32_t nb13;
    uint32_t ne20; uint32_t ne21; uint32_t ne22; uint32_t ne23; uint32_t nb20; uint32_t nb21; uint32_t nb22; uint32_t nb23;
    uint32_t misalign_offsets;
    float param1; float param2; int32_t param3;
};

struct vk_op_diag_mask_push_constants {
    uint32_t ncols;
    uint32_t rows_per_channel;
    int32_t n_past;
};

struct vk_op_rope_push_constants {
    uint32_t ncols;
    uint32_t n_dims;
    float freq_scale;
    uint32_t p_delta_rows;
    float freq_base;
    float ext_factor;
    float attn_factor;
    float corr_dims[2];
    float theta_scale;
    uint32_t has_ff;
    uint32_t ne02;
    uint32_t s1;
    uint32_t s2;
    int32_t sections[4];
    uint32_t is_back;
};

struct vk_op_soft_max_push_constants {
    uint32_t KX;
    uint32_t KY;
    float scale;
    float max_bias;
    float m0;
    float m1;
    uint32_t n_head_log2;
    uint32_t nrows_x;
};

struct vk_op_argsort_push_constants {
    uint32_t ncols;
    uint32_t ncols_pad;
    int32_t order;
};

struct vk_op_im2col_push_constants {
    uint32_t batch_offset; uint32_t offset_delta;
    uint32_t IC;
    uint32_t IW; uint32_t IH;
    uint32_t OW; uint32_t OH;
    uint32_t KW; uint32_t KH;
    uint32_t pelements;
    uint32_t CHW;
    int32_t s0; int32_t s1;
    int32_t p0; int32_t p1;
    int32_t d0; int32_t d1;
};

struct vk_op_timestep_embedding_push_constants {
    uint32_t nb1;
    uint32_t dim;
    uint32_t max_period;
};

struct vk_op_pool2d_push_constants {
    uint32_t IW; uint32_t IH;
    uint32_t OW; uint32_t OH;
    uint32_t OC;
    uint32_t pelements;
    uint32_t op;
    int32_t k0; int32_t k1;
    int32_t s0; int32_t s1;
    int32_t p0; int32_t p1;
};

struct vk_op_rwkv_wkv6_push_constants {
    uint32_t B;
    uint32_t T;
    uint32_t C;
    uint32_t H;
};

struct vk_op_rwkv_wkv7_push_constants {
    uint32_t B;
    uint32_t T;
    uint32_t C;
    uint32_t H;
};

// Allow pre-recording command buffers
struct vk_staging_memcpy {
    vk_staging_memcpy(void * _dst, const void * _src, size_t _n) : dst(_dst), src(_src), n(_n) {}

    void * dst;
    const void * src;
    size_t n;
};

struct vk_op_upscale_push_constants {
    uint32_t ne; uint32_t a_offset; uint32_t d_offset;
    uint32_t nb00; uint32_t nb01; uint32_t nb02; uint32_t nb03;
    uint32_t ne10; uint32_t ne11; uint32_t ne12; uint32_t ne13;
    float sf0; float sf1; float sf2; float sf3;
};

struct vk_context_struct {
    vk_submission * s;
    std::vector<vk_sequence> seqs;

    int exit_tensor_idx;

    std::vector<vk_staging_memcpy> in_memcpys;
    std::vector<vk_staging_memcpy> out_memcpys;

    vk_queue * q;
};
typedef std::shared_ptr<vk_context_struct> vk_context;
typedef std::weak_ptr<vk_context_struct> vk_context_ref;

struct ggml_vk_garbage_collector {
    std::vector<vk_semaphore> tl_semaphores;
    std::vector<vk_semaphore> semaphores;
    std::vector<vk::Event> events;
    std::vector<vk_buffer> temp_buffers;
    std::vector<vk_context> contexts;
};

#if defined(GGML_VULKAN_MEMORY_DEBUG) || defined(GGML_VULKAN_DEBUG)
#define VK_LOG_MEMORY(msg) std::cerr << "ggml_vulkan memory: " << msg << std::endl

static std::mutex log_mutex;

class vk_memory_logger {
public:
    vk_memory_logger(): total_device(0), total_host(0) {}
    void log_allocation(vk_buffer_ref buf_ref, size_t size);
    void log_deallocation(vk_buffer_ref buf_ref);

private:
    std::map<vk::Buffer, size_t> allocations; // Track allocations
    size_t total_device;
    size_t total_host;
};
#else
#define VK_LOG_MEMORY(msg) ((void) 0)
#endif // GGML_VULKAN_MEMORY_DEBUG

#if defined(GGML_VULKAN_PERF)

class vk_perf_logger {
public:
    void print_timings();

    void log_timing(const ggml_tensor * node, uint64_t time);
private:
    std::map<std::string, std::vector<uint64_t>> timings;
};
#endif // GGML_VULKAN_PERF

struct ggml_backend_vk_context {
    std::string name;

    vk_device device;

    size_t semaphore_idx, event_idx;
    ggml_vk_garbage_collector gc;
    size_t prealloc_size_x, prealloc_size_y, prealloc_size_split_k;
    vk_buffer prealloc_x, prealloc_y, prealloc_split_k;
    vk::Fence fence;

    vk_buffer buffer_pool[MAX_VK_BUFFERS];

    vk_context_ref compute_ctx;
    vk_context_ref transfer_ctx;

    std::vector<vk_context_ref> tensor_ctxs;
};

static void * const vk_ptr_base = (void *)(uintptr_t) 0x1000;  // NOLINT

static uint64_t vk_tensor_offset(const ggml_tensor * tensor);

struct ggml_backend_vk_buffer_context {
    vk_device_ref device;
    vk_buffer dev_buffer;
    std::string name;

    ggml_backend_vk_buffer_context(vk_device_ref device, vk_buffer&& dev_buffer, std::string& name);

    ~ggml_backend_vk_buffer_context();
};

struct vk_instance_t {
    vk::Instance instance;

    std::vector<size_t> device_indices;
    vk_device devices[GGML_VK_MAX_DEVICES];
};

#endif // GGML_VULKAN_HPP
