#include "ggml-vulkan.h"
#include <vulkan/vulkan_core.h>
#ifdef GGML_VULKAN_RUN_TESTS
#include <chrono>
#endif

#include <vulkan/vulkan.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <vector>
#include <sstream>
#include <utility>
#include <memory>
#include <limits>
#include <map>
#include <memory>
#include <mutex>

#include "ggml.h"
#include "ggml-backend-impl.h"

#include "ggml-vulkan-shaders.hpp"

#define VK_API_VERSION VK_API_VERSION_1_2

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

#define VK_VENDOR_ID_AMD 0x1002
#define VK_VENDOR_ID_APPLE 0x106b
#define VK_VENDOR_ID_INTEL 0x8086
#define VK_VENDOR_ID_NVIDIA 0x10de

#define VK_DEVICE_DESCRIPTOR_POOL_MODE_UNKNOWN 0
#define VK_DEVICE_DESCRIPTOR_POOL_MODE_MULTI 1
#define VK_DEVICE_DESCRIPTOR_POOL_MODE_SINGLE 2

#define GGML_VK_MAX_NODES 8192

#define MAX_VK_BUFFERS 256

#ifndef K_QUANTS_PER_ITERATION
#define K_QUANTS_PER_ITERATION 1
#else
static_assert(K_QUANTS_PER_ITERATION == 1 || K_QUANTS_PER_ITERATION == 2, "K_QUANTS_PER_ITERATION must be 1 or 2");
#endif

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
};

typedef std::shared_ptr<vk_pipeline_struct> vk_pipeline;
typedef std::weak_ptr<vk_pipeline_struct> vk_pipeline_ref;

static void ggml_vk_destroy_pipeline(vk::Device& device, vk_pipeline& pipeline);

struct vk_matmul_pipeline_struct {
    vk_pipeline l, m, s;
    vk_pipeline a_l, a_m, a_s;
};

typedef std::shared_ptr<vk_matmul_pipeline_struct> vk_matmul_pipeline;

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

GGML_CALL static const char * ggml_backend_vk_buffer_type_name(ggml_backend_buffer_type_t buft);
GGML_CALL static ggml_backend_buffer_t ggml_backend_vk_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size);
GGML_CALL static size_t ggml_backend_vk_buffer_type_get_alignment(ggml_backend_buffer_type_t buft);
GGML_CALL static size_t ggml_backend_vk_buffer_type_get_max_size(ggml_backend_buffer_type_t buft);
GGML_CALL static size_t ggml_backend_vk_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor);
static ggml_backend_buffer_type_i ggml_backend_vk_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_vk_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_vk_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_vk_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_vk_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_vk_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

#ifdef GGML_VULKAN_MEMORY_DEBUG
class vk_memory_logger;
#endif
static void ggml_vk_destroy_buffer(vk_buffer& buf);

struct vk_device_struct {
    std::mutex mutex;

    vk::PhysicalDevice physical_device;
    vk::PhysicalDeviceProperties properties;
    std::string name;
    uint64_t max_memory_allocation_size;
    bool fp16;
    vk::Device device;
    uint32_t vendor_id;
    vk_queue compute_queue;
    vk_queue transfer_queue;
    bool single_queue;
    uint32_t descriptor_set_mode;
    uint32_t subgroup_size;
    bool uma;

    size_t idx;

    vk_matmul_pipeline pipeline_matmul_f32;
    vk_matmul_pipeline pipeline_matmul_f32_f16;
    vk_matmul_pipeline pipeline_matmul_f16;
    vk_matmul_pipeline pipeline_matmul_f16_f32;
    vk_pipeline pipeline_matmul_split_k_reduce;

    vk_matmul_pipeline pipeline_dequant_mul_mat_mat[GGML_TYPE_COUNT];

    vk_matmul_pipeline pipeline_matmul_id_f32;
    vk_matmul_pipeline pipeline_matmul_id_f16;
    vk_matmul_pipeline pipeline_matmul_id_f16_f32;

    vk_matmul_pipeline pipeline_dequant_mul_mat_mat_id[GGML_TYPE_COUNT];

    vk_pipeline pipeline_dequant[GGML_TYPE_COUNT];
    vk_pipeline pipeline_dequant_mul_mat_vec_f32_f32[GGML_TYPE_COUNT];
    vk_pipeline pipeline_dequant_mul_mat_vec_f16_f32[GGML_TYPE_COUNT];
    vk_pipeline pipeline_dequant_mul_mat_vec_id_f32[GGML_TYPE_COUNT];

    vk_pipeline pipeline_mul_mat_vec_p021_f16_f32;
    vk_pipeline pipeline_mul_mat_vec_nc_f16_f32;
    vk_pipeline pipeline_get_rows[GGML_TYPE_COUNT];
    vk_pipeline pipeline_get_rows_f32[GGML_TYPE_COUNT];
    vk_pipeline pipeline_add_f32, pipeline_add_f16_f32_f16;
    vk_pipeline pipeline_mul_f32;
    vk_pipeline pipeline_div_f32;
    vk_pipeline pipeline_concat_f32, pipeline_concat_f16, pipeline_concat_i32;
    vk_pipeline pipeline_upscale_f32;
    vk_pipeline pipeline_scale_f32;
    vk_pipeline pipeline_sqr_f32;
    vk_pipeline pipeline_clamp_f32;
    vk_pipeline pipeline_pad_f32;
    vk_pipeline pipeline_cpy_f32_f32, pipeline_cpy_f32_f16, pipeline_cpy_f16_f16;
    vk_pipeline pipeline_norm_f32;
    vk_pipeline pipeline_group_norm_f32;
    vk_pipeline pipeline_rms_norm_f32;
    vk_pipeline pipeline_gelu_f32;
    vk_pipeline pipeline_gelu_quick_f32;
    vk_pipeline pipeline_silu_f32;
    vk_pipeline pipeline_relu_f32;
    vk_pipeline pipeline_leaky_relu_f32;
    vk_pipeline pipeline_tanh_f32;
    vk_pipeline pipeline_diag_mask_inf_f32;
    vk_pipeline pipeline_soft_max_f32, pipeline_soft_max_f32_f16;
    vk_pipeline pipeline_rope_norm_f32, pipeline_rope_norm_f16;
    vk_pipeline pipeline_rope_neox_f32, pipeline_rope_neox_f16;
    vk_pipeline pipeline_argsort_f32;
    vk_pipeline pipeline_sum_rows_f32;
    vk_pipeline pipeline_im2col_f32, pipeline_im2col_f32_f16;
    vk_pipeline pipeline_timestep_embedding_f32;

    std::vector<vk_pipeline_ref> pipelines;

    std::vector<std::tuple<void*, size_t, vk_buffer>> pinned_memory;

    vk::Fence fence;
    vk_buffer sync_staging;

    ggml_backend_buffer_type buffer_type;

#ifdef GGML_VULKAN_MEMORY_DEBUG
    std::unique_ptr<vk_memory_logger> memory_logger;
#endif

    ~vk_device_struct() {
        VK_LOG_DEBUG("destroy device " << name);

        device.destroyFence(fence);

        ggml_vk_destroy_buffer(sync_staging);

        device.destroyCommandPool(compute_queue.pool);
        if (!single_queue) {
            device.destroyCommandPool(transfer_queue.pool);
        }

        for (auto& pipeline : pipelines) {
            if (pipeline.expired()) {
                continue;
            }

            vk_pipeline pl = pipeline.lock();
            ggml_vk_destroy_pipeline(device, pl);
        }
        pipelines.clear();

        device.destroy();
    }
};

struct vk_buffer_struct {
    vk::Buffer buffer = VK_NULL_HANDLE;
    vk::DeviceMemory device_memory = VK_NULL_HANDLE;
    vk::MemoryPropertyFlags memory_property_flags;
    void * ptr;
    size_t size = 0;

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
};
struct vk_mat_vec_id_push_constants {
    uint32_t ncols; uint32_t stride_a; uint32_t stride_b; uint32_t stride_d;
    uint32_t batch_stride_a; uint32_t batch_stride_b; uint32_t batch_stride_d;
    uint32_t nei0; uint32_t ne11;
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
    uint32_t d_offset;
    float param1; float param2;
};

struct vk_op_binary_push_constants {
    uint32_t ne;
    uint32_t ne00; uint32_t ne01; uint32_t ne02; uint32_t ne03; uint32_t nb00; uint32_t nb01; uint32_t nb02; uint32_t nb03;
    uint32_t ne10; uint32_t ne11; uint32_t ne12; uint32_t ne13; uint32_t nb10; uint32_t nb11; uint32_t nb12; uint32_t nb13;
    uint32_t ne20; uint32_t ne21; uint32_t ne22; uint32_t ne23; uint32_t nb20; uint32_t nb21; uint32_t nb22; uint32_t nb23;
    uint32_t d_offset;
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
};

struct vk_op_soft_max_push_constants {
    uint32_t KX;
    uint32_t KY;
    float scale;
    float max_bias;
    float m0;
    float m1;
    uint32_t n_head_log2;
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

// Allow pre-recording command buffers
struct vk_staging_memcpy {
    vk_staging_memcpy(void * _dst, const void * _src, size_t _n) : dst(_dst), src(_src), n(_n) {}

    void * dst;
    const void * src;
    size_t n;
};

struct vk_op_upscale_push_constants {
    uint32_t ne; uint32_t d_offset;
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

struct ggml_tensor_extra_gpu {
    vk_buffer_ref buffer_gpu;
    uint64_t offset;

    void reset() {
        buffer_gpu.reset();
        offset = 0;
    }
};

struct ggml_vk_garbage_collector {
    std::vector<vk_semaphore> tl_semaphores;
    std::vector<vk_semaphore> semaphores;
    std::vector<vk::Event> events;
    std::vector<vk_buffer> temp_buffers;
    std::vector<vk_context> contexts;
};

#if defined(GGML_VULKAN_MEMORY_DEBUG) || defined(GGML_VULKAN_DEBUG)
#define VK_LOG_MEMORY(msg) std::cerr << "ggml_vulkan memory: " << msg << std::endl

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

struct ggml_backend_vk_context {
    std::string name;

    vk_device device;

    size_t semaphore_idx, event_idx;
    ggml_vk_garbage_collector gc;
    size_t prealloc_size_x, prealloc_size_y, prealloc_size_split_k;
    vk_buffer prealloc_x, prealloc_y, prealloc_split_k;
    vk::Fence fence;
    vk_buffer staging;
    size_t staging_size;
    size_t staging_offset;

    vk_buffer buffer_pool[MAX_VK_BUFFERS];

    vk_context_ref compute_ctx;
    vk_context_ref transfer_ctx;

    std::vector<vk_context_ref> tensor_ctxs;
};

#ifdef GGML_VULKAN_MEMORY_DEBUG
void vk_memory_logger::log_allocation(vk_buffer_ref buf_ref, size_t size) {
    std::lock_guard<std::mutex> guard(log_mutex);
    vk_buffer buf = buf_ref.lock();
    const bool device = bool(buf->memory_property_flags & vk::MemoryPropertyFlagBits::eDeviceLocal);
    const std::string type = device ? "device" : "host";
    allocations[buf->buffer] = size;
    total_device += device ? size : 0;
    total_host += device ? 0 : size;
    VK_LOG_MEMORY(buf->device->name << ": +" << format_size(size) << " " << type << " at " << buf->buffer << ". Total device: " << format_size(total_device) << ", total host: " << format_size(total_host));
}

void vk_memory_logger::log_deallocation(vk_buffer_ref buf_ref) {
    if (buf_ref.expired() || buf_ref.lock()->size == 0) {
        return;
    }

    std::lock_guard<std::mutex> guard(log_mutex);
    vk_buffer buf = buf_ref.lock();
    const bool device = bool(buf->memory_property_flags & vk::MemoryPropertyFlagBits::eDeviceLocal);
    std::string type = device ? "device" : "host";
    auto it = allocations.find(buf->buffer);
    total_device -= device ? it->second : 0;
    total_host -= device ? 0 : it->second;
    if (it != allocations.end()) {
        VK_LOG_MEMORY(buf->device->name << ": -" << format_size(it->second) << " " << type << " at " << buf->buffer << ". Total device: " << format_size(total_device) << ", total host: " << format_size(total_host));
        allocations.erase(it);
    } else {
        VK_LOG_MEMORY("ERROR " << buf->device->name << ": Attempted to deallocate unknown " << type << " memory at " << buf->buffer);
    }
}
#endif // GGML_VULKAN_MEMORY_DEBUG

struct vk_instance_t {
    vk::Instance instance;

    std::vector<size_t> device_indices;
    vk_device devices[GGML_VK_MAX_DEVICES];
};

static bool vk_instance_initialized = false;
static vk_instance_t vk_instance;

#ifdef GGML_VULKAN_CHECK_RESULTS
static size_t vk_skip_checks;
static size_t vk_output_tensor;

static void ggml_vk_print_tensor(const ggml_tensor * tensor, const char * name);
static void ggml_vk_check_results_0(ggml_tensor * tensor);
static void ggml_vk_check_results_1(ggml_tensor * tensor);
#endif

typedef void (*ggml_vk_func_t)(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

GGML_CALL static void ggml_backend_vk_free(ggml_backend_t backend);

static void ggml_vk_create_pipeline(vk_device& device, vk_pipeline& pipeline, const std::string& name, size_t spv_size, const void* spv_data, const std::string& entrypoint, uint32_t parameter_count, uint32_t push_constant_size, std::array<uint32_t, 3> wg_denoms, std::vector<uint32_t>&& specialization_constants, uint32_t align) {
    VK_LOG_DEBUG("ggml_vk_create_pipeline(" << device->name << ", " << name << ", " << entrypoint << ", " << parameter_count << ", " << push_constant_size << ", (" << wg_denoms[0] << "," << wg_denoms[1] << "," << wg_denoms[2] << "), specialization_constants, " << align << ")");
    GGML_ASSERT(parameter_count > 0);
    GGML_ASSERT(wg_denoms[0] > 0 && wg_denoms[1] > 0 && wg_denoms[2] > 0); // NOLINT

    std::lock_guard<std::mutex> guard(device->mutex);

    pipeline = std::make_shared<vk_pipeline_struct>();
    pipeline->name = name;
    pipeline->parameter_count = parameter_count;
    pipeline->push_constant_size = push_constant_size;
    pipeline->wg_denoms = wg_denoms;
    pipeline->align = align;

    vk::ShaderModuleCreateInfo shader_module_create_info({}, spv_size, reinterpret_cast<const uint32_t *>(spv_data));
    pipeline->shader_module = device->device.createShaderModule(shader_module_create_info);

    std::vector<vk::DescriptorSetLayoutBinding> dsl_binding;
    std::vector<vk::DescriptorBindingFlags> dsl_binding_flags;
    for (uint32_t i = 0; i < parameter_count; i++) {
        dsl_binding.push_back({i, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute});
        dsl_binding_flags.push_back({});
    }

    vk::DescriptorSetLayoutBindingFlagsCreateInfo dslbfci = { dsl_binding_flags };

    vk::PushConstantRange pcr(
        vk::ShaderStageFlagBits::eCompute,
        0,
        pipeline->push_constant_size
    );

    vk::DescriptorSetLayoutCreateInfo descriptor_set_layout_create_info(
        {},
        dsl_binding);
    descriptor_set_layout_create_info.setPNext(&dslbfci);
    pipeline->dsl = device->device.createDescriptorSetLayout(descriptor_set_layout_create_info);

    // Check if device supports multiple descriptors per pool
    if (device->descriptor_set_mode == VK_DEVICE_DESCRIPTOR_POOL_MODE_UNKNOWN) {
        const uint32_t alloc_count = 2;

        // Try allocating multiple sets from one pool
        // This fails on AMD for some reason, so add a fall back to allocating one pool per set
        vk::DescriptorPoolSize descriptor_pool_size(vk::DescriptorType::eStorageBuffer, pipeline->parameter_count);
        vk::DescriptorPoolCreateInfo descriptor_pool_create_info({}, alloc_count, descriptor_pool_size);
        vk::DescriptorPool pool = device->device.createDescriptorPool(descriptor_pool_create_info);

        std::vector<vk::DescriptorSetLayout> layouts(alloc_count);
        for (uint32_t i = 0; i < alloc_count; i++) {
            layouts[i] = pipeline->dsl;
        }
        try {
            vk::DescriptorSetAllocateInfo descriptor_set_alloc_info(pool, alloc_count, layouts.data());
            std::vector<vk::DescriptorSet> sets = device->device.allocateDescriptorSets(descriptor_set_alloc_info);
        } catch(vk::OutOfPoolMemoryError const&) {
            device->descriptor_set_mode = VK_DEVICE_DESCRIPTOR_POOL_MODE_SINGLE;
        }

        device->device.destroyDescriptorPool(pool);
    }

    if (device->descriptor_set_mode == VK_DEVICE_DESCRIPTOR_POOL_MODE_MULTI) {
        vk::DescriptorPoolSize descriptor_pool_size(vk::DescriptorType::eStorageBuffer, pipeline->parameter_count);
        vk::DescriptorPoolCreateInfo descriptor_pool_create_info({}, 128, descriptor_pool_size);
        pipeline->descriptor_pools.push_back(device->device.createDescriptorPool(descriptor_pool_create_info));
    }

    pipeline->descriptor_set_idx = 0;

    vk::PipelineLayoutCreateInfo pipeline_layout_create_info(vk::PipelineLayoutCreateFlags(), pipeline->dsl, pcr);
    pipeline->layout = device->device.createPipelineLayout(pipeline_layout_create_info);

    std::vector<vk::SpecializationMapEntry> specialization_entries(specialization_constants.size());

    for (size_t i = 0; i < specialization_constants.size(); i++) {
        specialization_entries[i].constantID = i;
        specialization_entries[i].offset = i * sizeof(uint32_t);
        specialization_entries[i].size = sizeof(uint32_t);
    }

    vk::SpecializationInfo specialization_info(
        specialization_entries.size(),
        specialization_entries.data(),
        specialization_constants.size() * sizeof(uint32_t),
        specialization_constants.data()
    );

    vk::PipelineShaderStageCreateInfo pipeline_shader_create_info(
            vk::PipelineShaderStageCreateFlags(),
            vk::ShaderStageFlagBits::eCompute,
            pipeline->shader_module,
            entrypoint.c_str(),
            &specialization_info);
    vk::ComputePipelineCreateInfo compute_pipeline_create_info(
        vk::PipelineCreateFlags(),
        pipeline_shader_create_info,
        pipeline->layout);
    pipeline->pipeline = device->device.createComputePipeline(VK_NULL_HANDLE, compute_pipeline_create_info).value;

    device->pipelines.push_back(pipeline);
}

static void ggml_vk_destroy_pipeline(vk::Device& device, vk_pipeline& pipeline) {
    VK_LOG_DEBUG("ggml_pipeline_destroy_pipeline(" << pipeline->name << ")");
    for (auto& pool : pipeline->descriptor_pools) {
        device.destroyDescriptorPool(pool);
    }
    pipeline->descriptor_pools.clear();
    pipeline->descriptor_sets.clear();
    pipeline->descriptor_set_idx = 0;

    device.destroyDescriptorSetLayout(pipeline->dsl);

    device.destroyPipelineLayout(pipeline->layout);

    device.destroyShaderModule(pipeline->shader_module);

    device.destroyPipeline(pipeline->pipeline);
}

static void ggml_pipeline_allocate_descriptor_sets(vk_device& device, vk_pipeline& pipeline, uint32_t n) {
    VK_LOG_DEBUG("ggml_pipeline_allocate_descriptor_sets(" << pipeline->name << ", " << n << ")");
    if (pipeline->descriptor_sets.size() >= pipeline->descriptor_set_idx + n) {
        // Enough descriptors are available
        return;
    }

    std::lock_guard<std::mutex> guard(device->mutex);

    if (device->descriptor_set_mode == VK_DEVICE_DESCRIPTOR_POOL_MODE_MULTI) {
        const uint32_t alloc_count = pipeline->descriptor_set_idx + n - pipeline->descriptor_sets.size();

        std::vector<vk::DescriptorSetLayout> layouts(alloc_count);
        for (uint32_t i = 0; i < alloc_count; i++) {
            layouts[i] = pipeline->dsl;
        }
        vk::DescriptorSetAllocateInfo descriptor_set_alloc_info(pipeline->descriptor_pools[0], alloc_count, layouts.data());
        std::vector<vk::DescriptorSet> sets = device->device.allocateDescriptorSets(descriptor_set_alloc_info);
        pipeline->descriptor_sets.insert(pipeline->descriptor_sets.end(), sets.begin(), sets.end());
    } else {
        for (uint32_t i = pipeline->descriptor_sets.size(); i < pipeline->descriptor_set_idx + n; i++) {
            vk::DescriptorPoolSize descriptor_pool_size(vk::DescriptorType::eStorageBuffer, pipeline->parameter_count);
            vk::DescriptorPoolCreateInfo descriptor_pool_create_info({}, 1, descriptor_pool_size);
            pipeline->descriptor_pools.push_back(device->device.createDescriptorPool(descriptor_pool_create_info));

            vk::DescriptorSetAllocateInfo descriptor_set_alloc_info(pipeline->descriptor_pools[i], 1, &pipeline->dsl);
            std::vector<vk::DescriptorSet> sets = device->device.allocateDescriptorSets(descriptor_set_alloc_info);
            pipeline->descriptor_sets.push_back(sets[0]);
        }
    }
}

static void ggml_pipeline_cleanup(vk_pipeline& pipeline) {
    VK_LOG_DEBUG("ggml_pipeline_cleanup(" << pipeline->name << ")");
    pipeline->descriptor_set_idx = 0;
}

static vk::CommandBuffer ggml_vk_create_cmd_buffer(vk_device& device, vk_queue& q) {
    VK_LOG_DEBUG("ggml_vk_create_cmd_buffer()");
    std::lock_guard<std::mutex> guard(device->mutex);

    if (q.cmd_buffers.size() > q.cmd_buffer_idx) {
        // Reuse command buffer
        return q.cmd_buffers[q.cmd_buffer_idx++];
    }

    vk::CommandBufferAllocateInfo command_buffer_alloc_info(
        q.pool,
        vk::CommandBufferLevel::ePrimary,
        1);
    const std::vector<vk::CommandBuffer> cmd_buffers = device->device.allocateCommandBuffers(command_buffer_alloc_info);
    auto buf = cmd_buffers.front();

    q.cmd_buffers.push_back(buf);
    q.cmd_buffer_idx++;

    return buf;
}

static vk_submission ggml_vk_create_submission(vk_device& device, vk_queue& q, std::vector<vk_semaphore> wait_semaphores, std::vector<vk_semaphore> signal_semaphores) {
    VK_LOG_DEBUG("ggml_vk_create_submission()");
    vk_submission s;
    s.buffer = ggml_vk_create_cmd_buffer(device, q);
    s.wait_semaphores = std::move(wait_semaphores);
    s.signal_semaphores = std::move(signal_semaphores);
    return s;
}

static void ggml_vk_submit(vk_context& ctx, vk::Fence fence) {
    if (ctx->seqs.empty()) {
        return;
    }
    VK_LOG_DEBUG("ggml_vk_submit(" << ctx << ", " << fence << ")");

    std::vector<std::vector<uint64_t>> tl_wait_vals;
    std::vector<std::vector<uint64_t>> tl_signal_vals;
    std::vector<std::vector<vk::Semaphore>> tl_wait_semaphores;
    std::vector<std::vector<vk::Semaphore>> tl_signal_semaphores;
    std::vector<vk::TimelineSemaphoreSubmitInfo> tl_submit_infos;
    std::vector<vk::SubmitInfo> submit_infos;
    int idx = -1;
    std::vector<std::vector<vk::PipelineStageFlags>> stage_flags;

    size_t reserve = 0;

    for (const auto& sequence : ctx->seqs) {
        reserve += sequence.size();
    }

    // Pre-reserve vectors to prevent reallocation, which invalidates pointers
    tl_wait_semaphores.reserve(reserve);
    tl_wait_vals.reserve(reserve);
    tl_signal_semaphores.reserve(reserve);
    tl_signal_vals.reserve(reserve);
    tl_submit_infos.reserve(reserve);
    submit_infos.reserve(reserve);
    stage_flags.reserve(reserve);

    for (const auto& sequence : ctx->seqs) {
        for (const auto& submission : sequence) {
            stage_flags.push_back({});
            idx++;
            tl_wait_vals.push_back({});
            tl_wait_semaphores.push_back({});
            tl_signal_vals.push_back({});
            tl_signal_semaphores.push_back({});
            for (size_t i = 0; i < submission.wait_semaphores.size(); i++) {
                stage_flags[idx].push_back(ctx->q->stage_flags);
                tl_wait_vals[idx].push_back(submission.wait_semaphores[i].value);
                tl_wait_semaphores[idx].push_back(submission.wait_semaphores[i].s);
            }
            for (size_t i = 0; i < submission.signal_semaphores.size(); i++) {
                tl_signal_vals[idx].push_back(submission.signal_semaphores[i].value);
                tl_signal_semaphores[idx].push_back(submission.signal_semaphores[i].s);
            }
            tl_submit_infos.push_back({
                (uint32_t) submission.wait_semaphores.size(),
                tl_wait_vals[idx].data(),
                (uint32_t) submission.signal_semaphores.size(),
                tl_signal_vals[idx].data(),
            });
            tl_submit_infos[idx].sType = vk::StructureType::eTimelineSemaphoreSubmitInfo;
            tl_submit_infos[idx].pNext = nullptr;
            vk::SubmitInfo si{
                (uint32_t) submission.wait_semaphores.size(),
                tl_wait_semaphores[idx].data(),
                stage_flags[idx].data(),
                1,
                &submission.buffer,
                (uint32_t) submission.signal_semaphores.size(),
                tl_signal_semaphores[idx].data(),
            };
            si.setPNext(&tl_submit_infos[idx]);
            submit_infos.push_back(si);
        }
    }

    ctx->q->queue.submit(submit_infos, fence);

    ctx->seqs.clear();
}

static uint32_t ggml_vk_find_queue_family_index(std::vector<vk::QueueFamilyProperties>& queue_family_props, const vk::QueueFlags& required, const vk::QueueFlags& avoid, int32_t compute_index, uint32_t min_num_queues) {
    VK_LOG_DEBUG("ggml_vk_find_queue_family_index()");
    const uint32_t qfsize = queue_family_props.size();

    // Try with avoid preferences first
    for (uint32_t i = 0; i < qfsize; i++) {
        if (queue_family_props[i].queueCount >= min_num_queues && (compute_index < 0 || i != (uint32_t) compute_index) && queue_family_props[i].queueFlags & required && !(queue_family_props[i].queueFlags & avoid)) {
            return i;
        }
    }

    // Fall back to only required
    for (size_t i = 0; i < qfsize; i++) {
        if (queue_family_props[i].queueCount >= min_num_queues && (compute_index < 0 || i != (uint32_t) compute_index) && queue_family_props[i].queueFlags & required) {
            return i;
        }
    }

    // Fall back to reusing compute queue
    for (size_t i = 0; i < qfsize; i++) {
        if (queue_family_props[i].queueCount >= min_num_queues && queue_family_props[i].queueFlags & required) {
            return i;
        }
    }

    // Fall back to ignoring min_num_queries
    for (size_t i = 0; i < qfsize; i++) {
        if (queue_family_props[i].queueFlags & required) {
            return i;
        }
    }

    // All commands that are allowed on a queue that supports transfer operations are also allowed on a queue that supports either graphics or compute operations.
    // Thus, if the capabilities of a queue family include VK_QUEUE_GRAPHICS_BIT or VK_QUEUE_COMPUTE_BIT, then reporting the VK_QUEUE_TRANSFER_BIT capability separately for that queue family is optional.
    if (compute_index >= 0) {
        return compute_index;
    }

    std::cerr << "ggml_vulkan: No suitable queue family index found." << std::endl;

    for(auto &q_family : queue_family_props) {
        std::cerr << "Queue number: "  + std::to_string(q_family.queueCount) << " flags: " + to_string(q_family.queueFlags) << std::endl;
    }
    abort();
}

static void ggml_vk_create_queue(vk_device& device, vk_queue& q, uint32_t queue_family_index, uint32_t queue_index, vk::PipelineStageFlags&& stage_flags) {
    VK_LOG_DEBUG("ggml_vk_create_queue()");
    std::lock_guard<std::mutex> guard(device->mutex);

    q.queue_family_index = queue_family_index;

    vk::CommandPoolCreateInfo command_pool_create_info_compute(vk::CommandPoolCreateFlags(VK_COMMAND_POOL_CREATE_TRANSIENT_BIT), queue_family_index);
    q.pool = device->device.createCommandPool(command_pool_create_info_compute);

    q.cmd_buffer_idx = 0;

    q.queue = device->device.getQueue(queue_family_index, queue_index);

    q.stage_flags = stage_flags;
}

static vk_context ggml_vk_create_context(ggml_backend_vk_context * ctx, vk_queue& q) {
    vk_context result = std::make_shared<vk_context_struct>();
    VK_LOG_DEBUG("ggml_vk_create_context(" << result << ")");
    ctx->gc.contexts.emplace_back(result);
    result->q = &q;
    return result;
}

static vk_context ggml_vk_create_temporary_context(vk_queue& q) {
    vk_context result = std::make_shared<vk_context_struct>();
    VK_LOG_DEBUG("ggml_vk_create_temporary_context(" << result << ")");
    result->q = &q;
    return result;
}

static vk_semaphore * ggml_vk_create_binary_semaphore(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_create_timeline_semaphore()");
    vk::SemaphoreTypeCreateInfo tci{ vk::SemaphoreType::eBinary, 0 };
    vk::SemaphoreCreateInfo ci{};
    ci.setPNext(&tci);
    vk::Semaphore semaphore = ctx->device->device.createSemaphore(ci);
    ctx->gc.semaphores.push_back({ semaphore, 0 });
    return &ctx->gc.semaphores[ctx->gc.semaphores.size() - 1];
}

static vk_semaphore * ggml_vk_create_timeline_semaphore(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_create_timeline_semaphore()");
    if (ctx->semaphore_idx >= ctx->gc.tl_semaphores.size()) {
        vk::SemaphoreTypeCreateInfo tci{ vk::SemaphoreType::eTimeline, 0 };
        vk::SemaphoreCreateInfo ci{};
        ci.setPNext(&tci);
        vk::Semaphore semaphore = ctx->device->device.createSemaphore(ci);
        ctx->gc.tl_semaphores.push_back({ semaphore, 0 });
    }
    return &ctx->gc.tl_semaphores[ctx->semaphore_idx++];
}

static vk::Event ggml_vk_create_event(ggml_backend_vk_context * ctx) {
    if (ctx->event_idx >= ctx->gc.events.size()) {
        ctx->gc.events.push_back(ctx->device->device.createEvent({}));
    }
    return ctx->gc.events[ctx->event_idx++];
}

static void ggml_vk_queue_cleanup(vk_device& device, vk_queue& q) {
    VK_LOG_DEBUG("ggml_vk_queue_cleanup()");
    std::lock_guard<std::mutex> guard(device->mutex);

    // Requires command buffers to be done
    device->device.resetCommandPool(q.pool);
    q.cmd_buffer_idx = 0;
}

static uint32_t find_properties(const vk::PhysicalDeviceMemoryProperties* mem_props, vk::MemoryRequirements* mem_req, vk::MemoryPropertyFlags flags) {
    for (uint32_t i = 0; i < mem_props->memoryTypeCount; ++i) {
        vk::MemoryType memory_type = mem_props->memoryTypes[i];
        if ((mem_req->memoryTypeBits & ((uint64_t)1 << i)) &&
            (flags & memory_type.propertyFlags) == flags &&
            mem_props->memoryHeaps[memory_type.heapIndex].size >= mem_req->size) {
            return static_cast<int32_t>(i);
        }
    }
    return UINT32_MAX;
}

static vk_buffer ggml_vk_create_buffer(vk_device& device, size_t size, vk::MemoryPropertyFlags req_flags, vk::MemoryPropertyFlags fallback_flags = vk::MemoryPropertyFlags(0)) {
    VK_LOG_DEBUG("ggml_vk_create_buffer(" << device->name << ", " << size << ", " << to_string(req_flags) << ", " << to_string(fallback_flags) << ")");
    if (size > device->max_memory_allocation_size) {
        throw vk::OutOfDeviceMemoryError("Requested buffer size exceeds device memory allocation limit");
    }

    std::lock_guard<std::mutex> guard(device->mutex);

    vk_buffer buf = std::make_shared<vk_buffer_struct>();

    if (size == 0) {
        buf->size = 0;
        return buf;
    }

    buf->size = size;
    vk::BufferCreateInfo buffer_create_info{
        vk::BufferCreateFlags(),
        size,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        vk::SharingMode::eExclusive,
        0,
        nullptr,
    };

    buf->buffer = device->device.createBuffer(buffer_create_info);

    vk::MemoryRequirements mem_req = device->device.getBufferMemoryRequirements(buf->buffer);

    vk::PhysicalDeviceMemoryProperties mem_props = device->physical_device.getMemoryProperties();

    uint32_t memory_type_index = UINT32_MAX;

    memory_type_index = find_properties(&mem_props, &mem_req, req_flags);
    buf->memory_property_flags = req_flags;

    if (memory_type_index == UINT32_MAX && fallback_flags) {
        memory_type_index = find_properties(&mem_props, &mem_req, fallback_flags);
        buf->memory_property_flags = fallback_flags;
    }

    if (memory_type_index == UINT32_MAX) {
        device->device.destroyBuffer(buf->buffer);
        buf->size = 0;
        throw vk::OutOfDeviceMemoryError("No suitable memory type found");
    }

    try {
        buf->device_memory = device->device.allocateMemory({ mem_req.size, memory_type_index });
    } catch (const vk::SystemError& e) {
        // Out of Host/Device memory, clean up buffer
        device->device.destroyBuffer(buf->buffer);
        buf->size = 0;
        throw e;
    }
    buf->ptr = nullptr;

    if (buf->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
        buf->ptr = device->device.mapMemory(buf->device_memory, 0, VK_WHOLE_SIZE);
    }

    device->device.bindBufferMemory(buf->buffer, buf->device_memory, 0);

    buf->device = device;

#ifdef GGML_VULKAN_MEMORY_DEBUG
    device->memory_logger->log_allocation(buf, size);
#endif

    return buf;
}

static vk_buffer ggml_vk_create_buffer_check(vk_device& device, size_t size, vk::MemoryPropertyFlags req_flags, vk::MemoryPropertyFlags fallback_flags = vk::MemoryPropertyFlags(0)) {
    try {
        return ggml_vk_create_buffer(device, size, req_flags, fallback_flags);
    } catch (const vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Memory allocation of size " << size << " failed." << std::endl;
        std::cerr << "ggml_vulkan: " << e.what() << std::endl;
        throw e;
    }
}

static vk_buffer ggml_vk_create_buffer_device(vk_device& device, size_t size) {
    vk_buffer buf;
    try {
        if (device->uma) {
            // Fall back to host memory type
            buf = ggml_vk_create_buffer(device, size, vk::MemoryPropertyFlagBits::eDeviceLocal, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        } else {
            buf = ggml_vk_create_buffer(device, size, vk::MemoryPropertyFlagBits::eDeviceLocal);
        }
    } catch (const vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Device memory allocation of size " << size << " failed." << std::endl;
        std::cerr << "ggml_vulkan: " << e.what() << std::endl;
        throw e;
    }

    return buf;
}

static void ggml_vk_destroy_buffer(vk_buffer& buf) {
    if (buf == nullptr) {
        return;
    }

#ifdef GGML_VULKAN_MEMORY_DEBUG
    if (buf->device != nullptr) {
        buf->device->memory_logger->log_deallocation(buf);
    }
#endif

    buf.reset();
}

static vk_subbuffer ggml_vk_subbuffer(vk_buffer& buf) {
    return { buf, 0, VK_WHOLE_SIZE };
}

static void ggml_vk_sync_buffers(vk_context& ctx) {
    VK_LOG_DEBUG("ggml_vk_sync_buffers()");
    const std::vector<vk::MemoryBarrier> mem_barriers{ { { vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite }, { vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite } } };

    ctx->s->buffer.pipelineBarrier(
        ctx->q->stage_flags,
        ctx->q->stage_flags,
        {},
        mem_barriers,
        {},
        {}
    );
}

static void ggml_vk_wait_events(vk_context& ctx, std::vector<vk::Event>&& events) {
    VK_LOG_DEBUG("ggml_vk_wait_events()");
    if (events.empty()) {
        return;
    }

    ctx->s->buffer.waitEvents(
        events,
        ctx->q->stage_flags,
        ctx->q->stage_flags,
        {},
        {},
        {}
    );
}

static void ggml_vk_load_shaders(vk_device& device) {
    VK_LOG_DEBUG("ggml_vk_load_shaders(" << device->name << ")");

    // mulmat
    std::initializer_list<uint32_t> warptile_l = { 128, 128, 128, 16, device->subgroup_size * 2, 64, 2, 4, 4, device->subgroup_size };
    std::initializer_list<uint32_t> warptile_m = { 128,  64,  64, 16, device->subgroup_size, 32, 2, 4, 2, device->subgroup_size };
    std::initializer_list<uint32_t> warptile_s = { device->subgroup_size,  32,  32, 16, 32, 32, 2, 2, 2, device->subgroup_size };

    std::initializer_list<uint32_t> warptile_mmq_l = { 128, 128, 128, 32, device->subgroup_size * 2, 64, 2, 4, 4, device->subgroup_size };
    std::initializer_list<uint32_t> warptile_mmq_m = { 128,  64,  64, 32, device->subgroup_size, 32, 2, 4, 2, device->subgroup_size };
    std::initializer_list<uint32_t> warptile_mmq_s = { device->subgroup_size,  32,  32, 32, 32, 32, 2, 2, 2, device->subgroup_size };

    std::array<uint32_t, 3> l_wg_denoms = {128, 128, 1 };
    std::array<uint32_t, 3> m_wg_denoms = { 64,  64, 1 };
    std::array<uint32_t, 3> s_wg_denoms = { 32,  32, 1 };

    uint32_t l_align = 128;
    uint32_t m_align =  64;
    uint32_t s_align =  32;

    device->pipeline_matmul_f32 = std::make_shared<vk_matmul_pipeline_struct>();
    device->pipeline_matmul_f32_f16 = std::make_shared<vk_matmul_pipeline_struct>();
    device->pipeline_matmul_f16_f32 = std::make_shared<vk_matmul_pipeline_struct>();
    device->pipeline_matmul_f16 = std::make_shared<vk_matmul_pipeline_struct>();
    device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_0] = std::make_shared<vk_matmul_pipeline_struct>();
    device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_1] = std::make_shared<vk_matmul_pipeline_struct>();
    device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_0] = std::make_shared<vk_matmul_pipeline_struct>();
    device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_1] = std::make_shared<vk_matmul_pipeline_struct>();
    device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q8_0] = std::make_shared<vk_matmul_pipeline_struct>();
    device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q2_K] = std::make_shared<vk_matmul_pipeline_struct>();
    device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q3_K] = std::make_shared<vk_matmul_pipeline_struct>();
    device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_K] = std::make_shared<vk_matmul_pipeline_struct>();
    device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_K] = std::make_shared<vk_matmul_pipeline_struct>();
    device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q6_K] = std::make_shared<vk_matmul_pipeline_struct>();
    device->pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ4_NL] = std::make_shared<vk_matmul_pipeline_struct>();

    device->pipeline_matmul_id_f32 = std::make_shared<vk_matmul_pipeline_struct>();
    device->pipeline_matmul_id_f16_f32 = std::make_shared<vk_matmul_pipeline_struct>();
    device->pipeline_matmul_id_f16 = std::make_shared<vk_matmul_pipeline_struct>();
    device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_0] = std::make_shared<vk_matmul_pipeline_struct>();
    device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_1] = std::make_shared<vk_matmul_pipeline_struct>();
    device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_0] = std::make_shared<vk_matmul_pipeline_struct>();
    device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_1] = std::make_shared<vk_matmul_pipeline_struct>();
    device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q8_0] = std::make_shared<vk_matmul_pipeline_struct>();
    device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q2_K] = std::make_shared<vk_matmul_pipeline_struct>();
    device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q3_K] = std::make_shared<vk_matmul_pipeline_struct>();
    device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_K] = std::make_shared<vk_matmul_pipeline_struct>();
    device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_K] = std::make_shared<vk_matmul_pipeline_struct>();
    device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q6_K] = std::make_shared<vk_matmul_pipeline_struct>();
    device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ4_NL] = std::make_shared<vk_matmul_pipeline_struct>();

    if (device->fp16) {
        ggml_vk_create_pipeline(device, device->pipeline_matmul_f32->l, "matmul_f32_l", matmul_f32_f32_len, matmul_f32_f32_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_l, 1);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_f32->m, "matmul_f32_m", matmul_f32_f32_len, matmul_f32_f32_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_m, 1);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_f32->s, "matmul_f32_s", matmul_f32_f32_len, matmul_f32_f32_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_s, 1);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_f32->a_l, "matmul_f32_aligned_l", matmul_f32_f32_aligned_len, matmul_f32_f32_aligned_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_f32->a_m, "matmul_f32_aligned_m", matmul_f32_f32_aligned_len, matmul_f32_f32_aligned_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_f32->a_s, "matmul_f32_aligned_s", matmul_f32_f32_aligned_len, matmul_f32_f32_aligned_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_matmul_f32_f16->l, "matmul_f32_f16_l", matmul_f32_f16_len, matmul_f32_f16_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_l, 1);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_f32_f16->m, "matmul_f32_f16_m", matmul_f32_f16_len, matmul_f32_f16_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_m, 1);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_f32_f16->s, "matmul_f32_f16_s", matmul_f32_f16_len, matmul_f32_f16_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_s, 1);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_f32_f16->a_l, "matmul_f32_f16_aligned_l", matmul_f32_f16_aligned_len, matmul_f32_f16_aligned_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_f32_f16->a_m, "matmul_f32_f16_aligned_m", matmul_f32_f16_aligned_len, matmul_f32_f16_aligned_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_f32_f16->a_s, "matmul_f32_f16_aligned_s", matmul_f32_f16_aligned_len, matmul_f32_f16_aligned_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_matmul_f16->l, "matmul_f16_l", matmul_f16_len, matmul_f16_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_l, 1);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_f16->m, "matmul_f16_m", matmul_f16_len, matmul_f16_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_m, 1);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_f16->s, "matmul_f16_s", matmul_f16_len, matmul_f16_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_s, 1);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_f16->a_l, "matmul_f16_aligned_l", matmul_f16_aligned_len, matmul_f16_aligned_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_f16->a_m, "matmul_f16_aligned_m", matmul_f16_aligned_len, matmul_f16_aligned_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_f16->a_s, "matmul_f16_aligned_s", matmul_f16_aligned_len, matmul_f16_aligned_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_matmul_f16_f32->l, "matmul_f16_f32_l", matmul_f16_f32_len, matmul_f16_f32_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_l, 1);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_f16_f32->m, "matmul_f16_f32_m", matmul_f16_f32_len, matmul_f16_f32_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_m, 1);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_f16_f32->s, "matmul_f16_f32_s", matmul_f16_f32_len, matmul_f16_f32_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_s, 1);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_f16_f32->a_l, "matmul_f16_f32_aligned_l", matmul_f16_f32_aligned_len, matmul_f16_f32_aligned_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_f16_f32->a_m, "matmul_f16_f32_aligned_m", matmul_f16_f32_aligned_len, matmul_f16_f32_aligned_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_f16_f32->a_s, "matmul_f16_f32_aligned_s", matmul_f16_f32_aligned_len, matmul_f16_f32_aligned_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_0]->l, "matmul_q4_0_f32_l", matmul_q4_0_f32_len, matmul_q4_0_f32_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_0]->m, "matmul_q4_0_f32_m", matmul_q4_0_f32_len, matmul_q4_0_f32_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_0]->s, "matmul_q4_0_f32_s", matmul_q4_0_f32_len, matmul_q4_0_f32_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_mmq_s, s_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_0]->a_l, "matmul_q4_0_f32_aligned_l", matmul_q4_0_f32_aligned_len, matmul_q4_0_f32_aligned_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_0]->a_m, "matmul_q4_0_f32_aligned_m", matmul_q4_0_f32_aligned_len, matmul_q4_0_f32_aligned_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_0]->a_s, "matmul_q4_0_f32_aligned_s", matmul_q4_0_f32_aligned_len, matmul_q4_0_f32_aligned_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_mmq_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_1]->l, "matmul_q4_1_f32_l", matmul_q4_1_f32_len, matmul_q4_1_f32_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_1]->m, "matmul_q4_1_f32_m", matmul_q4_1_f32_len, matmul_q4_1_f32_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_1]->s, "matmul_q4_1_f32_s", matmul_q4_1_f32_len, matmul_q4_1_f32_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_mmq_s, s_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_1]->a_l, "matmul_q4_1_f32_aligned_l", matmul_q4_1_f32_aligned_len, matmul_q4_1_f32_aligned_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_1]->a_m, "matmul_q4_1_f32_aligned_m", matmul_q4_1_f32_aligned_len, matmul_q4_1_f32_aligned_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_1]->a_s, "matmul_q4_1_f32_aligned_s", matmul_q4_1_f32_aligned_len, matmul_q4_1_f32_aligned_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_mmq_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_0]->l, "matmul_q5_0_f32_l", matmul_q5_0_f32_len, matmul_q5_0_f32_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_0]->m, "matmul_q5_0_f32_m", matmul_q5_0_f32_len, matmul_q5_0_f32_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_0]->s, "matmul_q5_0_f32_s", matmul_q5_0_f32_len, matmul_q5_0_f32_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_mmq_s, s_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_0]->a_l, "matmul_q5_0_f32_aligned_l", matmul_q5_0_f32_aligned_len, matmul_q5_0_f32_aligned_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_0]->a_m, "matmul_q5_0_f32_aligned_m", matmul_q5_0_f32_aligned_len, matmul_q5_0_f32_aligned_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_0]->a_s, "matmul_q5_0_f32_aligned_s", matmul_q5_0_f32_aligned_len, matmul_q5_0_f32_aligned_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_mmq_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_1]->l, "matmul_q5_1_f32_l", matmul_q5_1_f32_len, matmul_q5_1_f32_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_1]->m, "matmul_q5_1_f32_m", matmul_q5_1_f32_len, matmul_q5_1_f32_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_1]->s, "matmul_q5_1_f32_s", matmul_q5_1_f32_len, matmul_q5_1_f32_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_mmq_s, s_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_1]->a_l, "matmul_q5_1_f32_aligned_l", matmul_q5_1_f32_aligned_len, matmul_q5_1_f32_aligned_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_1]->a_m, "matmul_q5_1_f32_aligned_m", matmul_q5_1_f32_aligned_len, matmul_q5_1_f32_aligned_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_1]->a_s, "matmul_q5_1_f32_aligned_s", matmul_q5_1_f32_aligned_len, matmul_q5_1_f32_aligned_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_mmq_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q8_0]->l, "matmul_q8_0_f32_l", matmul_q8_0_f32_len, matmul_q8_0_f32_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q8_0]->m, "matmul_q8_0_f32_m", matmul_q8_0_f32_len, matmul_q8_0_f32_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q8_0]->s, "matmul_q8_0_f32_s", matmul_q8_0_f32_len, matmul_q8_0_f32_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_mmq_s, s_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q8_0]->a_l, "matmul_q8_0_f32_aligned_l", matmul_q8_0_f32_aligned_len, matmul_q8_0_f32_aligned_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q8_0]->a_m, "matmul_q8_0_f32_aligned_m", matmul_q8_0_f32_aligned_len, matmul_q8_0_f32_aligned_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q8_0]->a_s, "matmul_q8_0_f32_aligned_s", matmul_q8_0_f32_aligned_len, matmul_q8_0_f32_aligned_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_mmq_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q2_K]->l, "matmul_q2_k_f32_l", matmul_q2_k_f32_len, matmul_q2_k_f32_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q2_K]->m, "matmul_q2_k_f32_m", matmul_q2_k_f32_len, matmul_q2_k_f32_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q2_K]->s, "matmul_q2_k_f32_s", matmul_q2_k_f32_len, matmul_q2_k_f32_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_mmq_s, s_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q2_K]->a_l, "matmul_q2_k_f32_aligned_l", matmul_q2_k_f32_aligned_len, matmul_q2_k_f32_aligned_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q2_K]->a_m, "matmul_q2_k_f32_aligned_m", matmul_q2_k_f32_aligned_len, matmul_q2_k_f32_aligned_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q2_K]->a_s, "matmul_q2_k_f32_aligned_s", matmul_q2_k_f32_aligned_len, matmul_q2_k_f32_aligned_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_mmq_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q3_K]->l, "matmul_q3_k_f32_l", matmul_q3_k_f32_len, matmul_q3_k_f32_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q3_K]->m, "matmul_q3_k_f32_m", matmul_q3_k_f32_len, matmul_q3_k_f32_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q3_K]->s, "matmul_q3_k_f32_s", matmul_q3_k_f32_len, matmul_q3_k_f32_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_mmq_s, s_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q3_K]->a_l, "matmul_q3_k_f32_aligned_l", matmul_q3_k_f32_aligned_len, matmul_q3_k_f32_aligned_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q3_K]->a_m, "matmul_q3_k_f32_aligned_m", matmul_q3_k_f32_aligned_len, matmul_q3_k_f32_aligned_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q3_K]->a_s, "matmul_q3_k_f32_aligned_s", matmul_q3_k_f32_aligned_len, matmul_q3_k_f32_aligned_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_mmq_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_K]->l, "matmul_q4_k_f32_l", matmul_q4_k_f32_len, matmul_q4_k_f32_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_K]->m, "matmul_q4_k_f32_m", matmul_q4_k_f32_len, matmul_q4_k_f32_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_K]->s, "matmul_q4_k_f32_s", matmul_q4_k_f32_len, matmul_q4_k_f32_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_mmq_s, s_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_K]->a_l, "matmul_q4_k_f32_aligned_l", matmul_q4_k_f32_aligned_len, matmul_q4_k_f32_aligned_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_K]->a_m, "matmul_q4_k_f32_aligned_m", matmul_q4_k_f32_aligned_len, matmul_q4_k_f32_aligned_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_K]->a_s, "matmul_q4_k_f32_aligned_s", matmul_q4_k_f32_aligned_len, matmul_q4_k_f32_aligned_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_mmq_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_K]->l, "matmul_q5_k_f32_l", matmul_q5_k_f32_len, matmul_q5_k_f32_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_K]->m, "matmul_q5_k_f32_m", matmul_q5_k_f32_len, matmul_q5_k_f32_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_K]->s, "matmul_q5_k_f32_s", matmul_q5_k_f32_len, matmul_q5_k_f32_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_mmq_s, s_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_K]->a_l, "matmul_q5_k_f32_aligned_l", matmul_q5_k_f32_aligned_len, matmul_q5_k_f32_aligned_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_K]->a_m, "matmul_q5_k_f32_aligned_m", matmul_q5_k_f32_aligned_len, matmul_q5_k_f32_aligned_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_K]->a_s, "matmul_q5_k_f32_aligned_s", matmul_q5_k_f32_aligned_len, matmul_q5_k_f32_aligned_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_mmq_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q6_K]->l, "matmul_q6_k_f32_l", matmul_q6_k_f32_len, matmul_q6_k_f32_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q6_K]->m, "matmul_q6_k_f32_m", matmul_q6_k_f32_len, matmul_q6_k_f32_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q6_K]->s, "matmul_q6_k_f32_s", matmul_q6_k_f32_len, matmul_q6_k_f32_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_mmq_s, s_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q6_K]->a_l, "matmul_q6_k_f32_aligned_l", matmul_q6_k_f32_aligned_len, matmul_q6_k_f32_aligned_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q6_K]->a_m, "matmul_q6_k_f32_aligned_m", matmul_q6_k_f32_aligned_len, matmul_q6_k_f32_aligned_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q6_K]->a_s, "matmul_q6_k_f32_aligned_s", matmul_q6_k_f32_aligned_len, matmul_q6_k_f32_aligned_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_mmq_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ4_NL]->l, "matmul_iq4_nl_f32_l", matmul_iq4_nl_f32_len, matmul_iq4_nl_f32_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ4_NL]->m, "matmul_iq4_nl_f32_m", matmul_iq4_nl_f32_len, matmul_iq4_nl_f32_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ4_NL]->s, "matmul_iq4_nl_f32_s", matmul_iq4_nl_f32_len, matmul_iq4_nl_f32_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_mmq_s, s_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ4_NL]->a_l, "matmul_iq4_nl_f32_aligned_l", matmul_iq4_nl_f32_aligned_len, matmul_iq4_nl_f32_aligned_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ4_NL]->a_m, "matmul_iq4_nl_f32_aligned_m", matmul_iq4_nl_f32_aligned_len, matmul_iq4_nl_f32_aligned_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ4_NL]->a_s, "matmul_iq4_nl_f32_aligned_s", matmul_iq4_nl_f32_aligned_len, matmul_iq4_nl_f32_aligned_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_mmq_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_matmul_id_f32->l, "matmul_id_f32_l", matmul_id_f32_f32_len, matmul_id_f32_f32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_l, 1);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_id_f32->m, "matmul_id_f32_m", matmul_id_f32_f32_len, matmul_id_f32_f32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_m, 1);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_id_f32->s, "matmul_id_f32_s", matmul_id_f32_f32_len, matmul_id_f32_f32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_s, 1);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_id_f32->a_l, "matmul_id_f32_aligned_l", matmul_id_f32_f32_aligned_len, matmul_id_f32_f32_aligned_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_id_f32->a_m, "matmul_id_f32_aligned_m", matmul_id_f32_f32_aligned_len, matmul_id_f32_f32_aligned_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_id_f32->a_s, "matmul_id_f32_aligned_s", matmul_id_f32_f32_aligned_len, matmul_id_f32_f32_aligned_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_matmul_id_f16->l, "matmul_id_f16_l", matmul_id_f16_len, matmul_id_f16_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_l, 1);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_id_f16->m, "matmul_id_f16_m", matmul_id_f16_len, matmul_id_f16_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_m, 1);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_id_f16->s, "matmul_id_f16_s", matmul_id_f16_len, matmul_id_f16_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_s, 1);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_id_f16->a_l, "matmul_id_f16_aligned_l", matmul_id_f16_aligned_len, matmul_id_f16_aligned_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_id_f16->a_m, "matmul_id_f16_aligned_m", matmul_id_f16_aligned_len, matmul_id_f16_aligned_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_id_f16->a_s, "matmul_id_f16_aligned_s", matmul_id_f16_aligned_len, matmul_id_f16_aligned_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_matmul_id_f16_f32->l, "matmul_id_f16_f32_l", matmul_id_f16_f32_len, matmul_id_f16_f32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_l, 1);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_id_f16_f32->m, "matmul_id_f16_f32_m", matmul_id_f16_f32_len, matmul_id_f16_f32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_m, 1);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_id_f16_f32->s, "matmul_id_f16_f32_s", matmul_id_f16_f32_len, matmul_id_f16_f32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_s, 1);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_id_f16_f32->a_l, "matmul_id_f16_f32_aligned_l", matmul_id_f16_f32_aligned_len, matmul_id_f16_f32_aligned_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_id_f16_f32->a_m, "matmul_id_f16_f32_aligned_m", matmul_id_f16_f32_aligned_len, matmul_id_f16_f32_aligned_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_id_f16_f32->a_s, "matmul_id_f16_f32_aligned_s", matmul_id_f16_f32_aligned_len, matmul_id_f16_f32_aligned_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_0]->l, "matmul_id_q4_0_f32_l", matmul_id_q4_0_f32_len, matmul_id_q4_0_f32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_0]->m, "matmul_id_q4_0_f32_m", matmul_id_q4_0_f32_len, matmul_id_q4_0_f32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_0]->s, "matmul_id_q4_0_f32_s", matmul_id_q4_0_f32_len, matmul_id_q4_0_f32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_mmq_s, s_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_0]->a_l, "matmul_id_q4_0_f32_aligned_l", matmul_id_q4_0_f32_aligned_len, matmul_id_q4_0_f32_aligned_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_0]->a_m, "matmul_id_q4_0_f32_aligned_m", matmul_id_q4_0_f32_aligned_len, matmul_id_q4_0_f32_aligned_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_0]->a_s, "matmul_id_q4_0_f32_aligned_s", matmul_id_q4_0_f32_aligned_len, matmul_id_q4_0_f32_aligned_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_mmq_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_1]->l, "matmul_id_q4_1_f32_l", matmul_id_q4_1_f32_len, matmul_id_q4_1_f32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_1]->m, "matmul_id_q4_1_f32_m", matmul_id_q4_1_f32_len, matmul_id_q4_1_f32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_1]->s, "matmul_id_q4_1_f32_s", matmul_id_q4_1_f32_len, matmul_id_q4_1_f32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_mmq_s, s_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_1]->a_l, "matmul_id_q4_1_f32_aligned_l", matmul_id_q4_1_f32_aligned_len, matmul_id_q4_1_f32_aligned_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_1]->a_m, "matmul_id_q4_1_f32_aligned_m", matmul_id_q4_1_f32_aligned_len, matmul_id_q4_1_f32_aligned_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_1]->a_s, "matmul_id_q4_1_f32_aligned_s", matmul_id_q4_1_f32_aligned_len, matmul_id_q4_1_f32_aligned_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_mmq_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_0]->l, "matmul_id_q5_0_f32_l", matmul_id_q5_0_f32_len, matmul_id_q5_0_f32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_0]->m, "matmul_id_q5_0_f32_m", matmul_id_q5_0_f32_len, matmul_id_q5_0_f32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_0]->s, "matmul_id_q5_0_f32_s", matmul_id_q5_0_f32_len, matmul_id_q5_0_f32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_mmq_s, s_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_0]->a_l, "matmul_id_q5_0_f32_aligned_l", matmul_id_q5_0_f32_aligned_len, matmul_id_q5_0_f32_aligned_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_0]->a_m, "matmul_id_q5_0_f32_aligned_m", matmul_id_q5_0_f32_aligned_len, matmul_id_q5_0_f32_aligned_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_0]->a_s, "matmul_id_q5_0_f32_aligned_s", matmul_id_q5_0_f32_aligned_len, matmul_id_q5_0_f32_aligned_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_mmq_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_1]->l, "matmul_id_q5_1_f32_l", matmul_id_q5_1_f32_len, matmul_id_q5_1_f32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_1]->m, "matmul_id_q5_1_f32_m", matmul_id_q5_1_f32_len, matmul_id_q5_1_f32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_1]->s, "matmul_id_q5_1_f32_s", matmul_id_q5_1_f32_len, matmul_id_q5_1_f32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_mmq_s, s_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_1]->a_l, "matmul_id_q5_1_f32_aligned_l", matmul_id_q5_1_f32_aligned_len, matmul_id_q5_1_f32_aligned_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_1]->a_m, "matmul_id_q5_1_f32_aligned_m", matmul_id_q5_1_f32_aligned_len, matmul_id_q5_1_f32_aligned_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_1]->a_s, "matmul_id_q5_1_f32_aligned_s", matmul_id_q5_1_f32_aligned_len, matmul_id_q5_1_f32_aligned_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_mmq_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q8_0]->l, "matmul_id_q8_0_f32_l", matmul_id_q8_0_f32_len, matmul_id_q8_0_f32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q8_0]->m, "matmul_id_q8_0_f32_m", matmul_id_q8_0_f32_len, matmul_id_q8_0_f32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q8_0]->s, "matmul_id_q8_0_f32_s", matmul_id_q8_0_f32_len, matmul_id_q8_0_f32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_mmq_s, s_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q8_0]->a_l, "matmul_id_q8_0_f32_aligned_l", matmul_id_q8_0_f32_aligned_len, matmul_id_q8_0_f32_aligned_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q8_0]->a_m, "matmul_id_q8_0_f32_aligned_m", matmul_id_q8_0_f32_aligned_len, matmul_id_q8_0_f32_aligned_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q8_0]->a_s, "matmul_id_q8_0_f32_aligned_s", matmul_id_q8_0_f32_aligned_len, matmul_id_q8_0_f32_aligned_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_mmq_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q2_K]->l, "matmul_id_q2_k_f32_l", matmul_id_q2_k_f32_len, matmul_id_q2_k_f32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q2_K]->m, "matmul_id_q2_k_f32_m", matmul_id_q2_k_f32_len, matmul_id_q2_k_f32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q2_K]->s, "matmul_id_q2_k_f32_s", matmul_id_q2_k_f32_len, matmul_id_q2_k_f32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_mmq_s, s_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q2_K]->a_l, "matmul_id_q2_k_f32_aligned_l", matmul_id_q2_k_f32_aligned_len, matmul_id_q2_k_f32_aligned_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q2_K]->a_m, "matmul_id_q2_k_f32_aligned_m", matmul_id_q2_k_f32_aligned_len, matmul_id_q2_k_f32_aligned_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q2_K]->a_s, "matmul_id_q2_k_f32_aligned_s", matmul_id_q2_k_f32_aligned_len, matmul_id_q2_k_f32_aligned_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_mmq_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q3_K]->l, "matmul_id_q3_k_f32_l", matmul_id_q3_k_f32_len, matmul_id_q3_k_f32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q3_K]->m, "matmul_id_q3_k_f32_m", matmul_id_q3_k_f32_len, matmul_id_q3_k_f32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q3_K]->s, "matmul_id_q3_k_f32_s", matmul_id_q3_k_f32_len, matmul_id_q3_k_f32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_mmq_s, s_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q3_K]->a_l, "matmul_id_q3_k_f32_aligned_l", matmul_id_q3_k_f32_aligned_len, matmul_id_q3_k_f32_aligned_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q3_K]->a_m, "matmul_id_q3_k_f32_aligned_m", matmul_id_q3_k_f32_aligned_len, matmul_id_q3_k_f32_aligned_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q3_K]->a_s, "matmul_id_q3_k_f32_aligned_s", matmul_id_q3_k_f32_aligned_len, matmul_id_q3_k_f32_aligned_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_mmq_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_K]->l, "matmul_id_q4_k_f32_l", matmul_id_q4_k_f32_len, matmul_id_q4_k_f32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_K]->m, "matmul_id_q4_k_f32_m", matmul_id_q4_k_f32_len, matmul_id_q4_k_f32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_K]->s, "matmul_id_q4_k_f32_s", matmul_id_q4_k_f32_len, matmul_id_q4_k_f32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_mmq_s, s_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_K]->a_l, "matmul_id_q4_k_f32_aligned_l", matmul_id_q4_k_f32_aligned_len, matmul_id_q4_k_f32_aligned_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_K]->a_m, "matmul_id_q4_k_f32_aligned_m", matmul_id_q4_k_f32_aligned_len, matmul_id_q4_k_f32_aligned_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_K]->a_s, "matmul_id_q4_k_f32_aligned_s", matmul_id_q4_k_f32_aligned_len, matmul_id_q4_k_f32_aligned_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_mmq_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_K]->l, "matmul_id_q5_k_f32_l", matmul_id_q5_k_f32_len, matmul_id_q5_k_f32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_K]->m, "matmul_id_q5_k_f32_m", matmul_id_q5_k_f32_len, matmul_id_q5_k_f32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_K]->s, "matmul_id_q5_k_f32_s", matmul_id_q5_k_f32_len, matmul_id_q5_k_f32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_mmq_s, s_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_K]->a_l, "matmul_id_q5_k_f32_aligned_l", matmul_id_q5_k_f32_aligned_len, matmul_id_q5_k_f32_aligned_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_K]->a_m, "matmul_id_q5_k_f32_aligned_m", matmul_id_q5_k_f32_aligned_len, matmul_id_q5_k_f32_aligned_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_K]->a_s, "matmul_id_q5_k_f32_aligned_s", matmul_id_q5_k_f32_aligned_len, matmul_id_q5_k_f32_aligned_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_mmq_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q6_K]->l, "matmul_id_q6_k_f32_l", matmul_id_q6_k_f32_len, matmul_id_q6_k_f32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q6_K]->m, "matmul_id_q6_k_f32_m", matmul_id_q6_k_f32_len, matmul_id_q6_k_f32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q6_K]->s, "matmul_id_q6_k_f32_s", matmul_id_q6_k_f32_len, matmul_id_q6_k_f32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_mmq_s, s_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q6_K]->a_l, "matmul_id_q6_k_f32_aligned_l", matmul_id_q6_k_f32_aligned_len, matmul_id_q6_k_f32_aligned_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q6_K]->a_m, "matmul_id_q6_k_f32_aligned_m", matmul_id_q6_k_f32_aligned_len, matmul_id_q6_k_f32_aligned_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q6_K]->a_s, "matmul_id_q6_k_f32_aligned_s", matmul_id_q6_k_f32_aligned_len, matmul_id_q6_k_f32_aligned_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_mmq_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ4_NL]->l, "matmul_id_iq4_nl_f32_l", matmul_id_iq4_nl_f32_len, matmul_id_iq4_nl_f32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ4_NL]->m, "matmul_id_iq4_nl_f32_m", matmul_id_iq4_nl_f32_len, matmul_id_iq4_nl_f32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ4_NL]->s, "matmul_id_iq4_nl_f32_s", matmul_id_iq4_nl_f32_len, matmul_id_iq4_nl_f32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_mmq_s, s_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ4_NL]->a_l, "matmul_id_iq4_nl_f32_aligned_l", matmul_id_iq4_nl_f32_aligned_len, matmul_id_iq4_nl_f32_aligned_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ4_NL]->a_m, "matmul_id_iq4_nl_f32_aligned_m", matmul_id_iq4_nl_f32_aligned_len, matmul_id_iq4_nl_f32_aligned_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ4_NL]->a_s, "matmul_id_iq4_nl_f32_aligned_s", matmul_id_iq4_nl_f32_aligned_len, matmul_id_iq4_nl_f32_aligned_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_mmq_s, s_align);
    } else {
        ggml_vk_create_pipeline(device, device->pipeline_matmul_f32->l, "matmul_f32_l", matmul_f32_f32_fp32_len, matmul_f32_f32_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_l, 1);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_f32->m, "matmul_f32_m", matmul_f32_f32_fp32_len, matmul_f32_f32_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_m, 1);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_f32->s, "matmul_f32_s", matmul_f32_f32_fp32_len, matmul_f32_f32_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_s, 1);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_f32->a_l, "matmul_f32_aligned_l", matmul_f32_f32_aligned_fp32_len, matmul_f32_f32_aligned_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_f32->a_m, "matmul_f32_aligned_m", matmul_f32_f32_aligned_fp32_len, matmul_f32_f32_aligned_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_f32->a_s, "matmul_f32_aligned_s", matmul_f32_f32_aligned_fp32_len, matmul_f32_f32_aligned_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_matmul_f32_f16->l, "matmul_f32_f16_l", matmul_f32_f16_fp32_len, matmul_f32_f16_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_l, 1);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_f32_f16->m, "matmul_f32_f16_m", matmul_f32_f16_fp32_len, matmul_f32_f16_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_m, 1);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_f32_f16->s, "matmul_f32_f16_s", matmul_f32_f16_fp32_len, matmul_f32_f16_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_s, 1);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_f32_f16->a_l, "matmul_f32_f16_aligned_l", matmul_f32_f16_aligned_fp32_len, matmul_f32_f16_aligned_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_f32_f16->a_m, "matmul_f32_f16_aligned_m", matmul_f32_f16_aligned_fp32_len, matmul_f32_f16_aligned_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_f32_f16->a_s, "matmul_f32_f16_aligned_s", matmul_f32_f16_aligned_fp32_len, matmul_f32_f16_aligned_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_matmul_f16->l, "matmul_f16_l", matmul_f16_fp32_len, matmul_f16_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_l, 1);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_f16->m, "matmul_f16_m", matmul_f16_fp32_len, matmul_f16_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_m, 1);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_f16->s, "matmul_f16_s", matmul_f16_fp32_len, matmul_f16_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_s, 1);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_f16->a_l, "matmul_f16_aligned_l", matmul_f16_aligned_fp32_len, matmul_f16_aligned_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_f16->a_m, "matmul_f16_aligned_m", matmul_f16_aligned_fp32_len, matmul_f16_aligned_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_f16->a_s, "matmul_f16_aligned_s", matmul_f16_aligned_fp32_len, matmul_f16_aligned_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_matmul_f16_f32->l, "matmul_f16_f32_l", matmul_f16_f32_fp32_len, matmul_f16_f32_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_l, 1);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_f16_f32->m, "matmul_f16_f32_m", matmul_f16_f32_fp32_len, matmul_f16_f32_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_m, 1);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_f16_f32->s, "matmul_f16_f32_s", matmul_f16_f32_fp32_len, matmul_f16_f32_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_s, 1);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_f16_f32->a_l, "matmul_f16_f32_aligned_l", matmul_f16_f32_aligned_fp32_len, matmul_f16_f32_aligned_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_f16_f32->a_m, "matmul_f16_f32_aligned_m", matmul_f16_f32_aligned_fp32_len, matmul_f16_f32_aligned_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_f16_f32->a_s, "matmul_f16_f32_aligned_s", matmul_f16_f32_aligned_fp32_len, matmul_f16_f32_aligned_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_0]->l, "matmul_q4_0_f32_l", matmul_q4_0_f32_fp32_len, matmul_q4_0_f32_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_0]->m, "matmul_q4_0_f32_m", matmul_q4_0_f32_fp32_len, matmul_q4_0_f32_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_0]->s, "matmul_q4_0_f32_s", matmul_q4_0_f32_fp32_len, matmul_q4_0_f32_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_mmq_s, s_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_0]->a_l, "matmul_q4_0_f32_aligned_l", matmul_q4_0_f32_aligned_fp32_len, matmul_q4_0_f32_aligned_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_0]->a_m, "matmul_q4_0_f32_aligned_m", matmul_q4_0_f32_aligned_fp32_len, matmul_q4_0_f32_aligned_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_0]->a_s, "matmul_q4_0_f32_aligned_s", matmul_q4_0_f32_aligned_fp32_len, matmul_q4_0_f32_aligned_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_mmq_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_1]->l, "matmul_q4_1_f32_l", matmul_q4_1_f32_fp32_len, matmul_q4_1_f32_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_1]->m, "matmul_q4_1_f32_m", matmul_q4_1_f32_fp32_len, matmul_q4_1_f32_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_1]->s, "matmul_q4_1_f32_s", matmul_q4_1_f32_fp32_len, matmul_q4_1_f32_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_mmq_s, s_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_1]->a_l, "matmul_q4_1_f32_aligned_l", matmul_q4_1_f32_aligned_fp32_len, matmul_q4_1_f32_aligned_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_1]->a_m, "matmul_q4_1_f32_aligned_m", matmul_q4_1_f32_aligned_fp32_len, matmul_q4_1_f32_aligned_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_1]->a_s, "matmul_q4_1_f32_aligned_s", matmul_q4_1_f32_aligned_fp32_len, matmul_q4_1_f32_aligned_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_mmq_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_0]->l, "matmul_q5_0_f32_l", matmul_q5_0_f32_fp32_len, matmul_q5_0_f32_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_0]->m, "matmul_q5_0_f32_m", matmul_q5_0_f32_fp32_len, matmul_q5_0_f32_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_0]->s, "matmul_q5_0_f32_s", matmul_q5_0_f32_fp32_len, matmul_q5_0_f32_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_mmq_s, s_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_0]->a_l, "matmul_q5_0_f32_aligned_l", matmul_q5_0_f32_aligned_fp32_len, matmul_q5_0_f32_aligned_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_0]->a_m, "matmul_q5_0_f32_aligned_m", matmul_q5_0_f32_aligned_fp32_len, matmul_q5_0_f32_aligned_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_0]->a_s, "matmul_q5_0_f32_aligned_s", matmul_q5_0_f32_aligned_fp32_len, matmul_q5_0_f32_aligned_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_mmq_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_1]->l, "matmul_q5_1_f32_l", matmul_q5_1_f32_fp32_len, matmul_q5_1_f32_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_1]->m, "matmul_q5_1_f32_m", matmul_q5_1_f32_fp32_len, matmul_q5_1_f32_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_1]->s, "matmul_q5_1_f32_s", matmul_q5_1_f32_fp32_len, matmul_q5_1_f32_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_mmq_s, s_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_1]->a_l, "matmul_q5_1_f32_aligned_l", matmul_q5_1_f32_aligned_fp32_len, matmul_q5_1_f32_aligned_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_1]->a_m, "matmul_q5_1_f32_aligned_m", matmul_q5_1_f32_aligned_fp32_len, matmul_q5_1_f32_aligned_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_1]->a_s, "matmul_q5_1_f32_aligned_s", matmul_q5_1_f32_aligned_fp32_len, matmul_q5_1_f32_aligned_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_mmq_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q8_0]->l, "matmul_q8_0_f32_l", matmul_q8_0_f32_fp32_len, matmul_q8_0_f32_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q8_0]->m, "matmul_q8_0_f32_m", matmul_q8_0_f32_fp32_len, matmul_q8_0_f32_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q8_0]->s, "matmul_q8_0_f32_s", matmul_q8_0_f32_fp32_len, matmul_q8_0_f32_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_mmq_s, s_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q8_0]->a_l, "matmul_q8_0_f32_aligned_l", matmul_q8_0_f32_aligned_fp32_len, matmul_q8_0_f32_aligned_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q8_0]->a_m, "matmul_q8_0_f32_aligned_m", matmul_q8_0_f32_aligned_fp32_len, matmul_q8_0_f32_aligned_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q8_0]->a_s, "matmul_q8_0_f32_aligned_s", matmul_q8_0_f32_aligned_fp32_len, matmul_q8_0_f32_aligned_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_mmq_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q2_K]->l, "matmul_q2_k_f32_l", matmul_q2_k_f32_fp32_len, matmul_q2_k_f32_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q2_K]->m, "matmul_q2_k_f32_m", matmul_q2_k_f32_fp32_len, matmul_q2_k_f32_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q2_K]->s, "matmul_q2_k_f32_s", matmul_q2_k_f32_fp32_len, matmul_q2_k_f32_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_mmq_s, s_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q2_K]->a_l, "matmul_q2_k_f32_aligned_l", matmul_q2_k_f32_aligned_fp32_len, matmul_q2_k_f32_aligned_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q2_K]->a_m, "matmul_q2_k_f32_aligned_m", matmul_q2_k_f32_aligned_fp32_len, matmul_q2_k_f32_aligned_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q2_K]->a_s, "matmul_q2_k_f32_aligned_s", matmul_q2_k_f32_aligned_fp32_len, matmul_q2_k_f32_aligned_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_mmq_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q3_K]->l, "matmul_q3_k_f32_l", matmul_q3_k_f32_fp32_len, matmul_q3_k_f32_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q3_K]->m, "matmul_q3_k_f32_m", matmul_q3_k_f32_fp32_len, matmul_q3_k_f32_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q3_K]->s, "matmul_q3_k_f32_s", matmul_q3_k_f32_fp32_len, matmul_q3_k_f32_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_mmq_s, s_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q3_K]->a_l, "matmul_q3_k_f32_aligned_l", matmul_q3_k_f32_aligned_fp32_len, matmul_q3_k_f32_aligned_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q3_K]->a_m, "matmul_q3_k_f32_aligned_m", matmul_q3_k_f32_aligned_fp32_len, matmul_q3_k_f32_aligned_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q3_K]->a_s, "matmul_q3_k_f32_aligned_s", matmul_q3_k_f32_aligned_fp32_len, matmul_q3_k_f32_aligned_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_mmq_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_K]->l, "matmul_q4_k_f32_l", matmul_q4_k_f32_fp32_len, matmul_q4_k_f32_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_K]->m, "matmul_q4_k_f32_m", matmul_q4_k_f32_fp32_len, matmul_q4_k_f32_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_K]->s, "matmul_q4_k_f32_s", matmul_q4_k_f32_fp32_len, matmul_q4_k_f32_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_mmq_s, s_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_K]->a_l, "matmul_q4_k_f32_aligned_l", matmul_q4_k_f32_aligned_fp32_len, matmul_q4_k_f32_aligned_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_K]->a_m, "matmul_q4_k_f32_aligned_m", matmul_q4_k_f32_aligned_fp32_len, matmul_q4_k_f32_aligned_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_K]->a_s, "matmul_q4_k_f32_aligned_s", matmul_q4_k_f32_aligned_fp32_len, matmul_q4_k_f32_aligned_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_mmq_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_K]->l, "matmul_q5_k_f32_l", matmul_q5_k_f32_fp32_len, matmul_q5_k_f32_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_K]->m, "matmul_q5_k_f32_m", matmul_q5_k_f32_fp32_len, matmul_q5_k_f32_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_K]->s, "matmul_q5_k_f32_s", matmul_q5_k_f32_fp32_len, matmul_q5_k_f32_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_mmq_s, s_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_K]->a_l, "matmul_q5_k_f32_aligned_l", matmul_q5_k_f32_aligned_fp32_len, matmul_q5_k_f32_aligned_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_K]->a_m, "matmul_q5_k_f32_aligned_m", matmul_q5_k_f32_aligned_fp32_len, matmul_q5_k_f32_aligned_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_K]->a_s, "matmul_q5_k_f32_aligned_s", matmul_q5_k_f32_aligned_fp32_len, matmul_q5_k_f32_aligned_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_mmq_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q6_K]->l, "matmul_q6_k_f32_l", matmul_q6_k_f32_fp32_len, matmul_q6_k_f32_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q6_K]->m, "matmul_q6_k_f32_m", matmul_q6_k_f32_fp32_len, matmul_q6_k_f32_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q6_K]->s, "matmul_q6_k_f32_s", matmul_q6_k_f32_fp32_len, matmul_q6_k_f32_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_mmq_s, s_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q6_K]->a_l, "matmul_q6_k_f32_aligned_l", matmul_q6_k_f32_aligned_fp32_len, matmul_q6_k_f32_aligned_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q6_K]->a_m, "matmul_q6_k_f32_aligned_m", matmul_q6_k_f32_aligned_fp32_len, matmul_q6_k_f32_aligned_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_Q6_K]->a_s, "matmul_q6_k_f32_aligned_s", matmul_q6_k_f32_aligned_fp32_len, matmul_q6_k_f32_aligned_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_mmq_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ4_NL]->l, "matmul_iq4_nl_f32_l", matmul_iq4_nl_f32_fp32_len, matmul_iq4_nl_f32_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ4_NL]->m, "matmul_iq4_nl_f32_m", matmul_iq4_nl_f32_fp32_len, matmul_iq4_nl_f32_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ4_NL]->s, "matmul_iq4_nl_f32_s", matmul_iq4_nl_f32_fp32_len, matmul_iq4_nl_f32_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_mmq_s, s_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ4_NL]->a_l, "matmul_iq4_nl_f32_aligned_l", matmul_iq4_nl_f32_aligned_fp32_len, matmul_iq4_nl_f32_aligned_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ4_NL]->a_m, "matmul_iq4_nl_f32_aligned_m", matmul_iq4_nl_f32_aligned_fp32_len, matmul_iq4_nl_f32_aligned_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ4_NL]->a_s, "matmul_iq4_nl_f32_aligned_s", matmul_iq4_nl_f32_aligned_fp32_len, matmul_iq4_nl_f32_aligned_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants), s_wg_denoms, warptile_mmq_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_matmul_id_f32->l, "matmul_id_f32_l", matmul_id_f32_f32_fp32_len, matmul_id_f32_f32_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_l, 1);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_id_f32->m, "matmul_id_f32_m", matmul_id_f32_f32_fp32_len, matmul_id_f32_f32_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_m, 1);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_id_f32->s, "matmul_id_f32_s", matmul_id_f32_f32_fp32_len, matmul_id_f32_f32_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_s, 1);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_id_f32->a_l, "matmul_id_f32_aligned_l", matmul_id_f32_f32_aligned_fp32_len, matmul_id_f32_f32_aligned_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_id_f32->a_m, "matmul_id_f32_aligned_m", matmul_id_f32_f32_aligned_fp32_len, matmul_id_f32_f32_aligned_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_id_f32->a_s, "matmul_id_f32_aligned_s", matmul_id_f32_f32_aligned_fp32_len, matmul_id_f32_f32_aligned_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_matmul_id_f16->l, "matmul_id_f16_l", matmul_id_f16_fp32_len, matmul_id_f16_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_l, 1);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_id_f16->m, "matmul_id_f16_m", matmul_id_f16_fp32_len, matmul_id_f16_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_m, 1);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_id_f16->s, "matmul_id_f16_s", matmul_id_f16_fp32_len, matmul_id_f16_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_s, 1);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_id_f16->a_l, "matmul_id_f16_aligned_l", matmul_id_f16_aligned_fp32_len, matmul_id_f16_aligned_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_id_f16->a_m, "matmul_id_f16_aligned_m", matmul_id_f16_aligned_fp32_len, matmul_id_f16_aligned_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_id_f16->a_s, "matmul_id_f16_aligned_s", matmul_id_f16_aligned_fp32_len, matmul_id_f16_aligned_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_matmul_id_f16_f32->l, "matmul_id_f16_f32_l", matmul_id_f16_f32_fp32_len, matmul_id_f16_f32_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_l, 1);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_id_f16_f32->m, "matmul_id_f16_f32_m", matmul_id_f16_f32_fp32_len, matmul_id_f16_f32_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_m, 1);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_id_f16_f32->s, "matmul_id_f16_f32_s", matmul_id_f16_f32_fp32_len, matmul_id_f16_f32_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_s, 1);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_id_f16_f32->a_l, "matmul_id_f16_f32_aligned_l", matmul_id_f16_f32_aligned_fp32_len, matmul_id_f16_f32_aligned_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_id_f16_f32->a_m, "matmul_id_f16_f32_aligned_m", matmul_id_f16_f32_aligned_fp32_len, matmul_id_f16_f32_aligned_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_matmul_id_f16_f32->a_s, "matmul_id_f16_f32_aligned_s", matmul_id_f16_f32_aligned_fp32_len, matmul_id_f16_f32_aligned_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_0]->l, "matmul_id_q4_0_f32_l", matmul_id_q4_0_f32_fp32_len, matmul_id_q4_0_f32_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_0]->m, "matmul_id_q4_0_f32_m", matmul_id_q4_0_f32_fp32_len, matmul_id_q4_0_f32_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_0]->s, "matmul_id_q4_0_f32_s", matmul_id_q4_0_f32_fp32_len, matmul_id_q4_0_f32_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_mmq_s, s_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_0]->a_l, "matmul_id_q4_0_f32_aligned_l", matmul_id_q4_0_f32_aligned_fp32_len, matmul_id_q4_0_f32_aligned_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_0]->a_m, "matmul_id_q4_0_f32_aligned_m", matmul_id_q4_0_f32_aligned_fp32_len, matmul_id_q4_0_f32_aligned_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_0]->a_s, "matmul_id_q4_0_f32_aligned_s", matmul_id_q4_0_f32_aligned_fp32_len, matmul_id_q4_0_f32_aligned_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_mmq_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_1]->l, "matmul_id_q4_1_f32_l", matmul_id_q4_1_f32_fp32_len, matmul_id_q4_1_f32_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_1]->m, "matmul_id_q4_1_f32_m", matmul_id_q4_1_f32_fp32_len, matmul_id_q4_1_f32_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_1]->s, "matmul_id_q4_1_f32_s", matmul_id_q4_1_f32_fp32_len, matmul_id_q4_1_f32_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_mmq_s, s_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_1]->a_l, "matmul_id_q4_1_f32_aligned_l", matmul_id_q4_1_f32_aligned_fp32_len, matmul_id_q4_1_f32_aligned_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_1]->a_m, "matmul_id_q4_1_f32_aligned_m", matmul_id_q4_1_f32_aligned_fp32_len, matmul_id_q4_1_f32_aligned_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_1]->a_s, "matmul_id_q4_1_f32_aligned_s", matmul_id_q4_1_f32_aligned_fp32_len, matmul_id_q4_1_f32_aligned_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_mmq_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_0]->l, "matmul_id_q5_0_f32_l", matmul_id_q5_0_f32_fp32_len, matmul_id_q5_0_f32_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_0]->m, "matmul_id_q5_0_f32_m", matmul_id_q5_0_f32_fp32_len, matmul_id_q5_0_f32_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_0]->s, "matmul_id_q5_0_f32_s", matmul_id_q5_0_f32_fp32_len, matmul_id_q5_0_f32_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_mmq_s, s_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_0]->a_l, "matmul_id_q5_0_f32_aligned_l", matmul_id_q5_0_f32_aligned_fp32_len, matmul_id_q5_0_f32_aligned_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_0]->a_m, "matmul_id_q5_0_f32_aligned_m", matmul_id_q5_0_f32_aligned_fp32_len, matmul_id_q5_0_f32_aligned_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_0]->a_s, "matmul_id_q5_0_f32_aligned_s", matmul_id_q5_0_f32_aligned_fp32_len, matmul_id_q5_0_f32_aligned_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_mmq_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_1]->l, "matmul_id_q5_1_f32_l", matmul_id_q5_1_f32_fp32_len, matmul_id_q5_1_f32_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_1]->m, "matmul_id_q5_1_f32_m", matmul_id_q5_1_f32_fp32_len, matmul_id_q5_1_f32_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_1]->s, "matmul_id_q5_1_f32_s", matmul_id_q5_1_f32_fp32_len, matmul_id_q5_1_f32_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_mmq_s, s_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_1]->a_l, "matmul_id_q5_1_f32_aligned_l", matmul_id_q5_1_f32_aligned_fp32_len, matmul_id_q5_1_f32_aligned_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_1]->a_m, "matmul_id_q5_1_f32_aligned_m", matmul_id_q5_1_f32_aligned_fp32_len, matmul_id_q5_1_f32_aligned_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_1]->a_s, "matmul_id_q5_1_f32_aligned_s", matmul_id_q5_1_f32_aligned_fp32_len, matmul_id_q5_1_f32_aligned_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_mmq_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q8_0]->l, "matmul_id_q8_0_f32_l", matmul_id_q8_0_f32_fp32_len, matmul_id_q8_0_f32_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q8_0]->m, "matmul_id_q8_0_f32_m", matmul_id_q8_0_f32_fp32_len, matmul_id_q8_0_f32_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q8_0]->s, "matmul_id_q8_0_f32_s", matmul_id_q8_0_f32_fp32_len, matmul_id_q8_0_f32_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_mmq_s, s_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q8_0]->a_l, "matmul_id_q8_0_f32_aligned_l", matmul_id_q8_0_f32_aligned_fp32_len, matmul_id_q8_0_f32_aligned_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q8_0]->a_m, "matmul_id_q8_0_f32_aligned_m", matmul_id_q8_0_f32_aligned_fp32_len, matmul_id_q8_0_f32_aligned_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q8_0]->a_s, "matmul_id_q8_0_f32_aligned_s", matmul_id_q8_0_f32_aligned_fp32_len, matmul_id_q8_0_f32_aligned_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_mmq_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q2_K]->l, "matmul_id_q2_k_f32_l", matmul_id_q2_k_f32_fp32_len, matmul_id_q2_k_f32_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q2_K]->m, "matmul_id_q2_k_f32_m", matmul_id_q2_k_f32_fp32_len, matmul_id_q2_k_f32_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q2_K]->s, "matmul_id_q2_k_f32_s", matmul_id_q2_k_f32_fp32_len, matmul_id_q2_k_f32_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_mmq_s, s_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q2_K]->a_l, "matmul_id_q2_k_f32_aligned_l", matmul_id_q2_k_f32_aligned_fp32_len, matmul_id_q2_k_f32_aligned_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q2_K]->a_m, "matmul_id_q2_k_f32_aligned_m", matmul_id_q2_k_f32_aligned_fp32_len, matmul_id_q2_k_f32_aligned_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q2_K]->a_s, "matmul_id_q2_k_f32_aligned_s", matmul_id_q2_k_f32_aligned_fp32_len, matmul_id_q2_k_f32_aligned_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_mmq_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q3_K]->l, "matmul_id_q3_k_f32_l", matmul_id_q3_k_f32_fp32_len, matmul_id_q3_k_f32_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q3_K]->m, "matmul_id_q3_k_f32_m", matmul_id_q3_k_f32_fp32_len, matmul_id_q3_k_f32_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q3_K]->s, "matmul_id_q3_k_f32_s", matmul_id_q3_k_f32_fp32_len, matmul_id_q3_k_f32_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_mmq_s, s_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q3_K]->a_l, "matmul_id_q3_k_f32_aligned_l", matmul_id_q3_k_f32_aligned_fp32_len, matmul_id_q3_k_f32_aligned_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q3_K]->a_m, "matmul_id_q3_k_f32_aligned_m", matmul_id_q3_k_f32_aligned_fp32_len, matmul_id_q3_k_f32_aligned_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q3_K]->a_s, "matmul_id_q3_k_f32_aligned_s", matmul_id_q3_k_f32_aligned_fp32_len, matmul_id_q3_k_f32_aligned_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_mmq_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_K]->l, "matmul_id_q4_k_f32_l", matmul_id_q4_k_f32_fp32_len, matmul_id_q4_k_f32_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_K]->m, "matmul_id_q4_k_f32_m", matmul_id_q4_k_f32_fp32_len, matmul_id_q4_k_f32_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_K]->s, "matmul_id_q4_k_f32_s", matmul_id_q4_k_f32_fp32_len, matmul_id_q4_k_f32_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_mmq_s, s_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_K]->a_l, "matmul_id_q4_k_f32_aligned_l", matmul_id_q4_k_f32_aligned_fp32_len, matmul_id_q4_k_f32_aligned_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_K]->a_m, "matmul_id_q4_k_f32_aligned_m", matmul_id_q4_k_f32_aligned_fp32_len, matmul_id_q4_k_f32_aligned_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_K]->a_s, "matmul_id_q4_k_f32_aligned_s", matmul_id_q4_k_f32_aligned_fp32_len, matmul_id_q4_k_f32_aligned_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_mmq_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_K]->l, "matmul_id_q5_k_f32_l", matmul_id_q5_k_f32_fp32_len, matmul_id_q5_k_f32_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_K]->m, "matmul_id_q5_k_f32_m", matmul_id_q5_k_f32_fp32_len, matmul_id_q5_k_f32_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_K]->s, "matmul_id_q5_k_f32_s", matmul_id_q5_k_f32_fp32_len, matmul_id_q5_k_f32_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_mmq_s, s_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_K]->a_l, "matmul_id_q5_k_f32_aligned_l", matmul_id_q5_k_f32_aligned_fp32_len, matmul_id_q5_k_f32_aligned_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_K]->a_m, "matmul_id_q5_k_f32_aligned_m", matmul_id_q5_k_f32_aligned_fp32_len, matmul_id_q5_k_f32_aligned_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_K]->a_s, "matmul_id_q5_k_f32_aligned_s", matmul_id_q5_k_f32_aligned_fp32_len, matmul_id_q5_k_f32_aligned_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_mmq_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q6_K]->l, "matmul_id_q6_k_f32_l", matmul_id_q6_k_f32_fp32_len, matmul_id_q6_k_f32_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q6_K]->m, "matmul_id_q6_k_f32_m", matmul_id_q6_k_f32_fp32_len, matmul_id_q6_k_f32_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q6_K]->s, "matmul_id_q6_k_f32_s", matmul_id_q6_k_f32_fp32_len, matmul_id_q6_k_f32_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_mmq_s, s_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q6_K]->a_l, "matmul_id_q6_k_f32_aligned_l", matmul_id_q6_k_f32_aligned_fp32_len, matmul_id_q6_k_f32_aligned_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q6_K]->a_m, "matmul_id_q6_k_f32_aligned_m", matmul_id_q6_k_f32_aligned_fp32_len, matmul_id_q6_k_f32_aligned_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q6_K]->a_s, "matmul_id_q6_k_f32_aligned_s", matmul_id_q6_k_f32_aligned_fp32_len, matmul_id_q6_k_f32_aligned_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_mmq_s, s_align);

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ4_NL]->l, "matmul_id_iq4_nl_f32_l", matmul_id_iq4_nl_f32_fp32_len, matmul_id_iq4_nl_f32_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ4_NL]->m, "matmul_id_iq4_nl_f32_m", matmul_id_iq4_nl_f32_fp32_len, matmul_id_iq4_nl_f32_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ4_NL]->s, "matmul_id_iq4_nl_f32_s", matmul_id_iq4_nl_f32_fp32_len, matmul_id_iq4_nl_f32_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_mmq_s, s_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ4_NL]->a_l, "matmul_id_iq4_nl_f32_aligned_l", matmul_id_iq4_nl_f32_aligned_fp32_len, matmul_id_iq4_nl_f32_aligned_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), l_wg_denoms, warptile_mmq_l, l_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ4_NL]->a_m, "matmul_id_iq4_nl_f32_aligned_m", matmul_id_iq4_nl_f32_aligned_fp32_len, matmul_id_iq4_nl_f32_aligned_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), m_wg_denoms, warptile_mmq_m, m_align);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ4_NL]->a_s, "matmul_id_iq4_nl_f32_aligned_s", matmul_id_iq4_nl_f32_aligned_fp32_len, matmul_id_iq4_nl_f32_aligned_fp32_data, "main", 4, sizeof(vk_mat_mat_id_push_constants), s_wg_denoms, warptile_mmq_s, s_align);
    }

    // mul mat vec
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[GGML_TYPE_F32 ], "mul_mat_vec_f32_f32_f32",  mul_mat_vec_f32_f32_f32_len,  mul_mat_vec_f32_f32_f32_data,  "main", 3, sizeof(vk_mat_vec_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[GGML_TYPE_F16 ], "mul_mat_vec_f16_f32_f32",  mul_mat_vec_f16_f32_f32_len,  mul_mat_vec_f16_f32_f32_data,  "main", 3, sizeof(vk_mat_vec_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[GGML_TYPE_Q4_0], "mul_mat_vec_q4_0_f32_f32", mul_mat_vec_q4_0_f32_f32_len, mul_mat_vec_q4_0_f32_f32_data, "main", 3, sizeof(vk_mat_vec_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[GGML_TYPE_Q4_1], "mul_mat_vec_q4_1_f32_f32", mul_mat_vec_q4_1_f32_f32_len, mul_mat_vec_q4_1_f32_f32_data, "main", 3, sizeof(vk_mat_vec_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[GGML_TYPE_Q5_0], "mul_mat_vec_q5_0_f32_f32", mul_mat_vec_q5_0_f32_f32_len, mul_mat_vec_q5_0_f32_f32_data, "main", 3, sizeof(vk_mat_vec_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[GGML_TYPE_Q5_1], "mul_mat_vec_q5_1_f32_f32", mul_mat_vec_q5_1_f32_f32_len, mul_mat_vec_q5_1_f32_f32_data, "main", 3, sizeof(vk_mat_vec_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[GGML_TYPE_Q8_0], "mul_mat_vec_q8_0_f32_f32", mul_mat_vec_q8_0_f32_f32_len, mul_mat_vec_q8_0_f32_f32_data, "main", 3, sizeof(vk_mat_vec_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[GGML_TYPE_Q2_K], "mul_mat_vec_q2_k_f32_f32", mul_mat_vec_q2_k_f32_f32_len, mul_mat_vec_q2_k_f32_f32_data, "main", 3, sizeof(vk_mat_vec_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[GGML_TYPE_Q3_K], "mul_mat_vec_q3_k_f32_f32", mul_mat_vec_q3_k_f32_f32_len, mul_mat_vec_q3_k_f32_f32_data, "main", 3, sizeof(vk_mat_vec_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[GGML_TYPE_Q4_K], "mul_mat_vec_q4_k_f32_f32", mul_mat_vec_q4_k_f32_f32_len, mul_mat_vec_q4_k_f32_f32_data, "main", 3, sizeof(vk_mat_vec_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[GGML_TYPE_Q5_K], "mul_mat_vec_q5_k_f32_f32", mul_mat_vec_q5_k_f32_f32_len, mul_mat_vec_q5_k_f32_f32_data, "main", 3, sizeof(vk_mat_vec_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[GGML_TYPE_Q6_K], "mul_mat_vec_q6_k_f32_f32", mul_mat_vec_q6_k_f32_f32_len, mul_mat_vec_q6_k_f32_f32_data, "main", 3, sizeof(vk_mat_vec_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[GGML_TYPE_IQ4_NL], "mul_mat_vec_iq4_nl_f32_f32", mul_mat_vec_iq4_nl_f32_f32_len, mul_mat_vec_iq4_nl_f32_f32_data, "main", 3, sizeof(vk_mat_vec_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);

    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[GGML_TYPE_F32 ], "mul_mat_vec_f32_f16_f32",  mul_mat_vec_f32_f16_f32_len,  mul_mat_vec_f32_f16_f32_data,  "main", 3, sizeof(vk_mat_vec_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[GGML_TYPE_F16 ], "mul_mat_vec_f16_f16_f32",  mul_mat_vec_f16_f16_f32_len,  mul_mat_vec_f16_f16_f32_data,  "main", 3, sizeof(vk_mat_vec_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[GGML_TYPE_Q4_0], "mul_mat_vec_q4_0_f16_f32", mul_mat_vec_q4_0_f16_f32_len, mul_mat_vec_q4_0_f16_f32_data, "main", 3, sizeof(vk_mat_vec_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[GGML_TYPE_Q4_1], "mul_mat_vec_q4_1_f16_f32", mul_mat_vec_q4_1_f16_f32_len, mul_mat_vec_q4_1_f16_f32_data, "main", 3, sizeof(vk_mat_vec_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[GGML_TYPE_Q5_0], "mul_mat_vec_q5_0_f16_f32", mul_mat_vec_q5_0_f16_f32_len, mul_mat_vec_q5_0_f16_f32_data, "main", 3, sizeof(vk_mat_vec_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[GGML_TYPE_Q5_1], "mul_mat_vec_q5_1_f16_f32", mul_mat_vec_q5_1_f16_f32_len, mul_mat_vec_q5_1_f16_f32_data, "main", 3, sizeof(vk_mat_vec_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[GGML_TYPE_Q8_0], "mul_mat_vec_q8_0_f16_f32", mul_mat_vec_q8_0_f16_f32_len, mul_mat_vec_q8_0_f16_f32_data, "main", 3, sizeof(vk_mat_vec_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[GGML_TYPE_Q2_K], "mul_mat_vec_q2_k_f16_f32", mul_mat_vec_q2_k_f16_f32_len, mul_mat_vec_q2_k_f16_f32_data, "main", 3, sizeof(vk_mat_vec_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[GGML_TYPE_Q3_K], "mul_mat_vec_q3_k_f16_f32", mul_mat_vec_q3_k_f16_f32_len, mul_mat_vec_q3_k_f16_f32_data, "main", 3, sizeof(vk_mat_vec_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[GGML_TYPE_Q4_K], "mul_mat_vec_q4_k_f16_f32", mul_mat_vec_q4_k_f16_f32_len, mul_mat_vec_q4_k_f16_f32_data, "main", 3, sizeof(vk_mat_vec_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[GGML_TYPE_Q5_K], "mul_mat_vec_q5_k_f16_f32", mul_mat_vec_q5_k_f16_f32_len, mul_mat_vec_q5_k_f16_f32_data, "main", 3, sizeof(vk_mat_vec_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[GGML_TYPE_Q6_K], "mul_mat_vec_q6_k_f16_f32", mul_mat_vec_q6_k_f16_f32_len, mul_mat_vec_q6_k_f16_f32_data, "main", 3, sizeof(vk_mat_vec_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[GGML_TYPE_IQ4_NL], "mul_mat_vec_iq4_nl_f16_f32", mul_mat_vec_iq4_nl_f16_f32_len, mul_mat_vec_iq4_nl_f16_f32_data, "main", 3, sizeof(vk_mat_vec_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);

    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[GGML_TYPE_F32 ], "mul_mat_vec_id_f32_f32",  mul_mat_vec_id_f32_f32_len,  mul_mat_vec_id_f32_f32_data,  "main", 4, sizeof(vk_mat_vec_id_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[GGML_TYPE_F16 ], "mul_mat_vec_id_f16_f32",  mul_mat_vec_id_f16_f32_len,  mul_mat_vec_id_f16_f32_data,  "main", 4, sizeof(vk_mat_vec_id_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[GGML_TYPE_Q4_0], "mul_mat_vec_id_q4_0_f32", mul_mat_vec_id_q4_0_f32_len, mul_mat_vec_id_q4_0_f32_data, "main", 4, sizeof(vk_mat_vec_id_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[GGML_TYPE_Q4_1], "mul_mat_vec_id_q4_1_f32", mul_mat_vec_id_q4_1_f32_len, mul_mat_vec_id_q4_1_f32_data, "main", 4, sizeof(vk_mat_vec_id_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[GGML_TYPE_Q5_0], "mul_mat_vec_id_q5_0_f32", mul_mat_vec_id_q5_0_f32_len, mul_mat_vec_id_q5_0_f32_data, "main", 4, sizeof(vk_mat_vec_id_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[GGML_TYPE_Q5_1], "mul_mat_vec_id_q5_1_f32", mul_mat_vec_id_q5_1_f32_len, mul_mat_vec_id_q5_1_f32_data, "main", 4, sizeof(vk_mat_vec_id_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[GGML_TYPE_Q8_0], "mul_mat_vec_id_q8_0_f32", mul_mat_vec_id_q8_0_f32_len, mul_mat_vec_id_q8_0_f32_data, "main", 4, sizeof(vk_mat_vec_id_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[GGML_TYPE_Q2_K], "mul_mat_vec_id_q2_k_f32", mul_mat_vec_id_q2_k_f32_len, mul_mat_vec_id_q2_k_f32_data, "main", 4, sizeof(vk_mat_vec_id_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[GGML_TYPE_Q3_K], "mul_mat_vec_id_q3_k_f32", mul_mat_vec_id_q3_k_f32_len, mul_mat_vec_id_q3_k_f32_data, "main", 4, sizeof(vk_mat_vec_id_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[GGML_TYPE_Q4_K], "mul_mat_vec_id_q4_k_f32", mul_mat_vec_id_q4_k_f32_len, mul_mat_vec_id_q4_k_f32_data, "main", 4, sizeof(vk_mat_vec_id_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[GGML_TYPE_Q5_K], "mul_mat_vec_id_q5_k_f32", mul_mat_vec_id_q5_k_f32_len, mul_mat_vec_id_q5_k_f32_data, "main", 4, sizeof(vk_mat_vec_id_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[GGML_TYPE_Q6_K], "mul_mat_vec_id_q6_k_f32", mul_mat_vec_id_q6_k_f32_len, mul_mat_vec_id_q6_k_f32_data, "main", 4, sizeof(vk_mat_vec_id_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[GGML_TYPE_IQ4_NL], "mul_mat_vec_id_iq4_nl_f32", mul_mat_vec_id_iq4_nl_f32_len, mul_mat_vec_id_iq4_nl_f32_data, "main", 4, sizeof(vk_mat_vec_id_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);

    // dequant shaders
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_F32 ], "f32_to_f16",   dequant_f32_len,  dequant_f32_data,  "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q4_0], "dequant_q4_0", dequant_q4_0_len, dequant_q4_0_data, "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q4_1], "dequant_q4_1", dequant_q4_1_len, dequant_q4_1_data, "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q5_0], "dequant_q5_0", dequant_q5_0_len, dequant_q5_0_data, "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q5_1], "dequant_q5_1", dequant_q5_1_len, dequant_q5_1_data, "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q8_0], "dequant_q8_0", dequant_q8_0_len, dequant_q8_0_data, "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q2_K], "dequant_q2_k", dequant_q2_k_len, dequant_q2_k_data, "main", 2, 5 * sizeof(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q3_K], "dequant_q3_k", dequant_q3_k_len, dequant_q3_k_data, "main", 2, 5 * sizeof(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q4_K], "dequant_q4_k", dequant_q4_k_len, dequant_q4_k_data, "main", 2, 5 * sizeof(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q5_K], "dequant_q5_k", dequant_q5_k_len, dequant_q5_k_data, "main", 2, 5 * sizeof(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q6_K], "dequant_q6_k", dequant_q6_k_len, dequant_q6_k_data, "main", 2, 5 * sizeof(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ4_NL], "dequant_iq4_nl", dequant_iq4_nl_len, dequant_iq4_nl_data, "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);

    // get_rows
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_F32 ], "get_rows_f32",  get_rows_f32_len,  get_rows_f32_data,  "main", 3, sizeof(vk_op_binary_push_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_F16 ], "get_rows_f16",  get_rows_f16_len,  get_rows_f16_data,  "main", 3, sizeof(vk_op_binary_push_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q4_0], "get_rows_q4_0", get_rows_q4_0_len, get_rows_q4_0_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q4_1], "get_rows_q4_1", get_rows_q4_1_len, get_rows_q4_1_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q5_0], "get_rows_q5_0", get_rows_q5_0_len, get_rows_q5_0_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q5_1], "get_rows_q5_1", get_rows_q5_1_len, get_rows_q5_1_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q8_0], "get_rows_q8_0", get_rows_q8_0_len, get_rows_q8_0_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ4_NL], "get_rows_iq4_nl", get_rows_iq4_nl_len, get_rows_iq4_nl_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_F32 ], "get_rows_f32_f32",  get_rows_f32_f32_len,  get_rows_f32_f32_data,  "main", 3, sizeof(vk_op_binary_push_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_F16 ], "get_rows_f16_f32",  get_rows_f16_f32_len,  get_rows_f16_f32_data,  "main", 3, sizeof(vk_op_binary_push_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q4_0], "get_rows_q4_0_f32", get_rows_q4_0_f32_len, get_rows_q4_0_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q4_1], "get_rows_q4_1_f32", get_rows_q4_1_f32_len, get_rows_q4_1_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q5_0], "get_rows_q5_0_f32", get_rows_q5_0_f32_len, get_rows_q5_0_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q5_1], "get_rows_q5_1_f32", get_rows_q5_1_f32_len, get_rows_q5_1_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q8_0], "get_rows_q8_0_f32", get_rows_q8_0_f32_len, get_rows_q8_0_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ4_NL], "get_rows_iq4_nl_f32", get_rows_iq4_nl_f32_len, get_rows_iq4_nl_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_matmul_split_k_reduce, "split_k_reduce", split_k_reduce_len, split_k_reduce_data, "main", 2, 2 * sizeof(uint32_t), {256, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_mul_mat_vec_p021_f16_f32, "mul_mat_vec_p021_f16_f32", mul_mat_vec_p021_f16_f32_len, mul_mat_vec_p021_f16_f32_data, "main", 3, 6 * sizeof(uint32_t), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_mul_mat_vec_nc_f16_f32, "mul_mat_vec_nc_f16_f32", mul_mat_vec_nc_f16_f32_len, mul_mat_vec_nc_f16_f32_data, "main", 3, 7 * sizeof(uint32_t), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_norm_f32, "norm_f32", norm_f32_len, norm_f32_data, "main", 2, sizeof(vk_op_push_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_group_norm_f32, "group_norm_f32", group_norm_f32_len, group_norm_f32_data, "main", 2, sizeof(vk_op_push_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_f32, "rms_norm_f32", rms_norm_f32_len, rms_norm_f32_data, "main", 2, sizeof(vk_op_push_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_f32, "cpy_f32_f32", cpy_f32_f32_len, cpy_f32_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_f16, "cpy_f32_f16", cpy_f32_f16_len, cpy_f32_f16_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f16_f16, "cpy_f16_f16", cpy_f16_f16_len, cpy_f16_f16_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_add_f32, "add_f32", add_f32_len, add_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_add_f16_f32_f16, "add_f16_f32_f16", add_f16_f32_f16_len, add_f16_f32_f16_data, "main", 3, sizeof(vk_op_binary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_mul_f32, "mul_f32", mul_f32_len, mul_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_div_f32, "div_f32", div_f32_len, div_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_concat_f32, "concat_f32", concat_f32_len, concat_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_concat_f16, "concat_f16", concat_f16_len, concat_f16_data, "main", 3, sizeof(vk_op_binary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_concat_i32, "concat_i32", concat_i32_len, concat_i32_data, "main", 3, sizeof(vk_op_binary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_upscale_f32, "upscale_f32", upscale_f32_len, upscale_f32_data, "main", 2, sizeof(vk_op_upscale_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_scale_f32, "scale_f32", scale_f32_len, scale_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_sqr_f32, "sqr_f32", sqr_f32_len, sqr_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_clamp_f32, "clamp_f32", clamp_f32_len, clamp_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_pad_f32, "pad_f32", pad_f32_len, pad_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_gelu_f32, "gelu_f32", gelu_f32_len, gelu_f32_data, "main", 2, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_gelu_quick_f32, "gelu_quick_f32", gelu_quick_f32_len, gelu_quick_f32_data, "main", 2, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_silu_f32, "silu_f32", silu_f32_len, silu_f32_data, "main", 2, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_relu_f32, "relu_f32", relu_f32_len, relu_f32_data, "main", 2, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_leaky_relu_f32, "leaky_relu_f32", leaky_relu_f32_len, leaky_relu_f32_data, "main", 2, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_tanh_f32, "tanh_f32", tanh_f32_len, tanh_f32_data, "main", 2, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_diag_mask_inf_f32, "diag_mask_inf_f32", diag_mask_inf_f32_len, diag_mask_inf_f32_data, "main", 2, sizeof(vk_op_diag_mask_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32, "soft_max_f32", soft_max_f32_len, soft_max_f32_data, "main", 3, sizeof(vk_op_soft_max_push_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32_f16, "soft_max_f32_f16", soft_max_f32_f16_len, soft_max_f32_f16_data, "main", 3, sizeof(vk_op_soft_max_push_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rope_norm_f32, "rope_norm_f32", rope_norm_f32_len, rope_norm_f32_data, "main", 4, sizeof(vk_op_rope_push_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_norm_f16, "rope_norm_f16", rope_norm_f16_len, rope_norm_f16_data, "main", 4, sizeof(vk_op_rope_push_constants), {1, 512, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rope_neox_f32, "rope_neox_f32", rope_neox_f32_len, rope_neox_f32_data, "main", 4, sizeof(vk_op_rope_push_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_neox_f16, "rope_neox_f16", rope_neox_f16_len, rope_neox_f16_data, "main", 4, sizeof(vk_op_rope_push_constants), {1, 512, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_argsort_f32, "argsort_f32", argsort_f32_len, argsort_f32_data, "main", 2, sizeof(vk_op_argsort_push_constants), {1024, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_sum_rows_f32, "sum_rows_f32", sum_rows_f32_len, sum_rows_f32_data, "main", 2, sizeof(vk_op_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);

    ggml_vk_create_pipeline(device, device->pipeline_im2col_f32, "im2col_f32", im2col_f32_len, im2col_f32_data, "main", 2, sizeof(vk_op_im2col_push_constants), {256, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_im2col_f32_f16, "im2col_f32_f16", im2col_f32_f16_len, im2col_f32_f16_data, "main", 2, sizeof(vk_op_im2col_push_constants), {256, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_timestep_embedding_f32, "timestep_embedding_f32", timestep_embedding_f32_len, timestep_embedding_f32_data, "main", 2, sizeof(vk_op_timestep_embedding_push_constants), {256, 1, 1}, {}, 1);
}

static vk_device ggml_vk_get_device(size_t idx) {
    VK_LOG_DEBUG("ggml_vk_get_device(" << idx << ")");

    if (vk_instance.devices[idx] == nullptr) {
        VK_LOG_DEBUG("Initializing new vk_device");
        vk_device device = std::make_shared<vk_device_struct>();
        vk_instance.devices[idx] = device;

#ifdef GGML_VULKAN_MEMORY_DEBUG
        device->memory_logger = std::unique_ptr<vk_memory_logger>(new vk_memory_logger());
#endif

        size_t dev_num = vk_instance.device_indices[idx];

        std::vector<vk::PhysicalDevice> physical_devices = vk_instance.instance.enumeratePhysicalDevices();

        if (dev_num >= physical_devices.size()) {
            std::cerr << "ggml_vulkan: Device with index " << dev_num << " does not exist." << std::endl;
            throw std::runtime_error("Device not found");
        }

        device->physical_device = physical_devices[dev_num];
        const std::vector<vk::ExtensionProperties> ext_props = device->physical_device.enumerateDeviceExtensionProperties();

        bool maintenance4_support = false;

        // Check if maintenance4 is supported
        for (const auto& properties : ext_props) {
            if (strcmp("VK_KHR_maintenance4", properties.extensionName) == 0) {
                maintenance4_support = true;
            }
        }

        vk::PhysicalDeviceProperties2 props2;
        vk::PhysicalDeviceMaintenance3Properties props3;
        vk::PhysicalDeviceMaintenance4Properties props4;
        vk::PhysicalDeviceSubgroupProperties subgroup_props;
        props2.pNext = &props3;
        props3.pNext = &subgroup_props;
        if (maintenance4_support) {
            subgroup_props.pNext = &props4;
        }
        device->physical_device.getProperties2(&props2);
        device->properties = props2.properties;

        const char* GGML_VK_FORCE_MAX_ALLOCATION_SIZE = getenv("GGML_VK_FORCE_MAX_ALLOCATION_SIZE");

        if (GGML_VK_FORCE_MAX_ALLOCATION_SIZE != nullptr) {
            device->max_memory_allocation_size = std::stoi(GGML_VK_FORCE_MAX_ALLOCATION_SIZE);
        } else if (maintenance4_support) {
            device->max_memory_allocation_size = std::min(props3.maxMemoryAllocationSize, props4.maxBufferSize);
        } else {
            device->max_memory_allocation_size = props3.maxMemoryAllocationSize;
        }

        device->vendor_id = device->properties.vendorID;
        device->subgroup_size = subgroup_props.subgroupSize;
        device->uma = device->properties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu;

        bool fp16_storage = false;
        bool fp16_compute = false;

        for (const auto& properties : ext_props) {
            if (strcmp("VK_KHR_16bit_storage", properties.extensionName) == 0) {
                fp16_storage = true;
            } else if (strcmp("VK_KHR_shader_float16_int8", properties.extensionName) == 0) {
                fp16_compute = true;
            }
        }

        const char* GGML_VK_DISABLE_F16 = getenv("GGML_VK_DISABLE_F16");
        const bool force_disable_f16 = GGML_VK_DISABLE_F16 != nullptr;

        device->fp16 = !force_disable_f16 && fp16_storage && fp16_compute;

        std::vector<vk::QueueFamilyProperties> queue_family_props = device->physical_device.getQueueFamilyProperties();

        // Try to find a non-graphics compute queue and transfer-focused queues
        const uint32_t compute_queue_family_index = ggml_vk_find_queue_family_index(queue_family_props, vk::QueueFlagBits::eCompute, vk::QueueFlagBits::eGraphics, -1, 1);
        const uint32_t transfer_queue_family_index = ggml_vk_find_queue_family_index(queue_family_props, vk::QueueFlagBits::eTransfer, vk::QueueFlagBits::eCompute | vk::QueueFlagBits::eGraphics, compute_queue_family_index, 1);

        const float priorities[] = { 1.0f, 1.0f };
        device->single_queue = compute_queue_family_index == transfer_queue_family_index && queue_family_props[compute_queue_family_index].queueCount == 1;

        std::vector<vk::DeviceQueueCreateInfo> device_queue_create_infos;
        if (compute_queue_family_index != transfer_queue_family_index) {
            device_queue_create_infos.push_back({vk::DeviceQueueCreateFlags(), compute_queue_family_index, 1, priorities});
            device_queue_create_infos.push_back({vk::DeviceQueueCreateFlags(), transfer_queue_family_index, 1, priorities + 1});
        } else if(!device->single_queue) {
            device_queue_create_infos.push_back({vk::DeviceQueueCreateFlags(), compute_queue_family_index, 2, priorities});
        } else {
            device_queue_create_infos.push_back({vk::DeviceQueueCreateFlags(), compute_queue_family_index, 1, priorities});
        }
        vk::DeviceCreateInfo device_create_info;
        std::vector<const char *> device_extensions;
        vk::PhysicalDeviceFeatures device_features = device->physical_device.getFeatures();

        VkPhysicalDeviceFeatures2 device_features2;
        device_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        device_features2.pNext = nullptr;
        device_features2.features = (VkPhysicalDeviceFeatures)device_features;

        VkPhysicalDeviceVulkan11Features vk11_features;
        vk11_features.pNext = nullptr;
        vk11_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        device_features2.pNext = &vk11_features;

        VkPhysicalDeviceVulkan12Features vk12_features;
        vk12_features.pNext = nullptr;
        vk12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        vk11_features.pNext = &vk12_features;

        vkGetPhysicalDeviceFeatures2(device->physical_device, &device_features2);

        device->fp16 = device->fp16 && vk12_features.shaderFloat16;

        if (!vk11_features.storageBuffer16BitAccess) {
            std::cerr << "ggml_vulkan: device " << GGML_VK_NAME << idx << " does not support 16-bit storage." << std::endl;
            throw std::runtime_error("Unsupported device");
        }

        device_extensions.push_back("VK_KHR_16bit_storage");

#ifdef GGML_VULKAN_VALIDATE
        device_extensions.push_back("VK_KHR_shader_non_semantic_info");
#endif

        if (device->fp16) {
            device_extensions.push_back("VK_KHR_shader_float16_int8");
        }
        device->name = device->properties.deviceName.data();

        device_create_info = {
            vk::DeviceCreateFlags(),
            device_queue_create_infos,
            {},
            device_extensions
        };
        device_create_info.setPNext(&device_features2);
        device->device = device->physical_device.createDevice(device_create_info);

        device->descriptor_set_mode = VK_DEVICE_DESCRIPTOR_POOL_MODE_UNKNOWN;

        // Queues
        ggml_vk_create_queue(device, device->compute_queue, compute_queue_family_index, 0, { vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer });

        // Shaders
        ggml_vk_load_shaders(device);

        if (!device->single_queue) {
            const uint32_t transfer_queue_index = compute_queue_family_index == transfer_queue_family_index ? 1 : 0;
            ggml_vk_create_queue(device, device->transfer_queue, transfer_queue_family_index, transfer_queue_index, { vk::PipelineStageFlagBits::eTransfer });
        } else {
            // TODO: Use pointer or reference to avoid copy
            device->transfer_queue = device->compute_queue;
        }

        device->buffer_type = {
            /* .iface    = */ ggml_backend_vk_buffer_type_interface,
            /* .context  = */ new ggml_backend_vk_buffer_type_context{ device->name, device },
        };

        device->fence = device->device.createFence({});

        device->idx = idx;

        return device;
    }

    return vk_instance.devices[idx];
}


static void ggml_vk_print_gpu_info(size_t idx) {
    GGML_ASSERT(idx < vk_instance.device_indices.size());
    size_t dev_num = vk_instance.device_indices[idx];
    VK_LOG_DEBUG("ggml_vk_print_gpu_info(" << dev_num << ")");
    GGML_ASSERT(vk_instance_initialized);

    std::vector<vk::PhysicalDevice> devices = vk_instance.instance.enumeratePhysicalDevices();

    if (dev_num >= devices.size()) {
        std::cerr << "ggml_vulkan: Device with index " << dev_num << " does not exist." << std::endl;
        throw std::runtime_error("Device not found");
    }

    vk::PhysicalDevice physical_device = devices[dev_num];
    std::vector<vk::ExtensionProperties> ext_props = physical_device.enumerateDeviceExtensionProperties();

    vk::PhysicalDeviceProperties2 props2;
    vk::PhysicalDeviceMaintenance3Properties props3;
    vk::PhysicalDeviceSubgroupProperties subgroup_props;
    vk::PhysicalDeviceDriverProperties driver_props;
    props2.pNext = &props3;
    props3.pNext = &subgroup_props;
    subgroup_props.pNext = &driver_props;
    physical_device.getProperties2(&props2);

    const size_t subgroup_size = subgroup_props.subgroupSize;
    const bool uma = props2.properties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu;

    bool fp16_storage = false;
    bool fp16_compute = false;

    for (auto properties : ext_props) {
        if (strcmp("VK_KHR_16bit_storage", properties.extensionName) == 0) {
            fp16_storage = true;
        } else if (strcmp("VK_KHR_shader_float16_int8", properties.extensionName) == 0) {
            fp16_compute = true;
        }
    }

    const char* GGML_VK_DISABLE_F16 = getenv("GGML_VK_DISABLE_F16");
    bool force_disable_f16 = GGML_VK_DISABLE_F16 != nullptr;

    bool fp16 = !force_disable_f16 && fp16_storage && fp16_compute;

    vk::PhysicalDeviceFeatures device_features = physical_device.getFeatures();

    VkPhysicalDeviceFeatures2 device_features2;
    device_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    device_features2.pNext = nullptr;
    device_features2.features = (VkPhysicalDeviceFeatures)device_features;

    VkPhysicalDeviceVulkan11Features vk11_features;
    vk11_features.pNext = nullptr;
    vk11_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    device_features2.pNext = &vk11_features;

    VkPhysicalDeviceVulkan12Features vk12_features;
    vk12_features.pNext = nullptr;
    vk12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vk11_features.pNext = &vk12_features;

    vkGetPhysicalDeviceFeatures2(physical_device, &device_features2);

    fp16 = fp16 && vk12_features.shaderFloat16;

    std::string device_name = props2.properties.deviceName.data();
    std::cerr << GGML_VK_NAME << idx << ": " << device_name << " (" << driver_props.driverName << ") | uma: " << uma << " | fp16: " << fp16 << " | warp size: " << subgroup_size << std::endl;

    if (props2.properties.deviceType == vk::PhysicalDeviceType::eCpu) {
        std::cerr << "ggml_vulkan: Warning: Device type is CPU. This is probably not the device you want." << std::endl;
    }
}

static bool ggml_vk_instance_validation_ext_available(const std::vector<vk::ExtensionProperties>& instance_extensions);
static bool ggml_vk_instance_portability_enumeration_ext_available(const std::vector<vk::ExtensionProperties>& instance_extensions);

void ggml_vk_instance_init() {
    if (vk_instance_initialized) {
        return;
    }
    VK_LOG_DEBUG("ggml_vk_instance_init()");

    vk_instance_initialized = true;

    vk::ApplicationInfo app_info{ "ggml-vulkan", 1, nullptr, 0, VK_API_VERSION };

    const std::vector<vk::ExtensionProperties> instance_extensions = vk::enumerateInstanceExtensionProperties();
    const bool validation_ext = ggml_vk_instance_validation_ext_available(instance_extensions);
#ifdef __APPLE__
    const bool portability_enumeration_ext = ggml_vk_instance_portability_enumeration_ext_available(instance_extensions);
#endif

    std::vector<const char*> layers;

    if (validation_ext) {
        layers.push_back("VK_LAYER_KHRONOS_validation");
    }
    std::vector<const char*> extensions;
    if (validation_ext) {
        extensions.push_back("VK_EXT_validation_features");
    }
#ifdef __APPLE__
    if (portability_enumeration_ext) {
        extensions.push_back("VK_KHR_portability_enumeration");
    }
#endif
    vk::InstanceCreateInfo instance_create_info(vk::InstanceCreateFlags{}, &app_info, layers, extensions);
#ifdef __APPLE__
    if (portability_enumeration_ext) {
        instance_create_info.flags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
    }
#endif

    std::vector<vk::ValidationFeatureEnableEXT> features_enable;
    vk::ValidationFeaturesEXT validation_features;

    if (validation_ext) {
        features_enable = { vk::ValidationFeatureEnableEXT::eBestPractices };
        validation_features = {
            features_enable,
            {},
        };
        validation_features.setPNext(nullptr);
        instance_create_info.setPNext(&validation_features);

        std::cerr << "ggml_vulkan: Validation layers enabled" << std::endl;
    }
    vk_instance.instance = vk::createInstance(instance_create_info);

    size_t num_available_devices = vk_instance.instance.enumeratePhysicalDevices().size();

    // Emulate behavior of CUDA_VISIBLE_DEVICES for Vulkan
    char * devices_env = getenv("GGML_VK_VISIBLE_DEVICES");
    if (devices_env != nullptr) {
        std::string devices(devices_env);
        std::replace(devices.begin(), devices.end(), ',', ' ');

        std::stringstream ss(devices);
        size_t tmp;
        while (ss >> tmp) {
            if(tmp >= num_available_devices) {
                std::cerr << "ggml_vulkan: Invalid device index " << tmp << " in GGML_VK_VISIBLE_DEVICES." << std::endl;
                throw std::runtime_error("Invalid Vulkan device index");
            }
            vk_instance.device_indices.push_back(tmp);
        }
    } else {
        std::vector<vk::PhysicalDevice> devices = vk_instance.instance.enumeratePhysicalDevices();

        // Make sure at least one device exists
        if (devices.empty()) {
            std::cerr << "ggml_vulkan: Error: No devices found." << std::endl;
            GGML_ABORT("fatal error");
        }

        // Default to using all dedicated GPUs
        for (size_t i = 0; i < devices.size(); i++) {
            vk::PhysicalDeviceProperties2 new_props;
            vk::PhysicalDeviceDriverProperties new_driver;
            vk::PhysicalDeviceIDProperties new_id;
            new_props.pNext = &new_driver;
            new_driver.pNext = &new_id;
            devices[i].getProperties2(&new_props);

            if (new_props.properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
                // Check if there are two physical devices corresponding to the same GPU
                auto old_device = std::find_if(
                    vk_instance.device_indices.begin(),
                    vk_instance.device_indices.end(),
                    [&devices, &new_id](const size_t k){
                        vk::PhysicalDeviceProperties2 old_props;
                        vk::PhysicalDeviceIDProperties old_id;
                        old_props.pNext = &old_id;
                        devices[k].getProperties2(&old_props);
                        return std::equal(std::begin(old_id.deviceUUID), std::end(old_id.deviceUUID), std::begin(new_id.deviceUUID));
                    }
                );
                if (old_device == vk_instance.device_indices.end()) {
                    vk_instance.device_indices.push_back(i);
                } else {
                    // There can be two physical devices corresponding to the same GPU if there are 2 different drivers
                    // This can cause error when splitting layers aross the devices, need to keep only 1
                    VK_LOG_DEBUG("Device " << i << " and device " << *old_device << " have the same deviceUUID");

                    vk::PhysicalDeviceProperties2 old_props;
                    vk::PhysicalDeviceDriverProperties old_driver;
                    old_props.pNext = &old_driver;
                    devices[*old_device].getProperties2(&old_props);

                    std::map<vk::DriverId, int> driver_priorities {};
                    int old_priority = std::numeric_limits<int>::max();
                    int new_priority = std::numeric_limits<int>::max();

                    // Check https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkDriverId.html for the list of driver id
                    // Smaller number -> higher priority
                    switch (old_props.properties.vendorID) {
                        case VK_VENDOR_ID_AMD:
                            driver_priorities[vk::DriverId::eMesaRadv] = 1;
                            driver_priorities[vk::DriverId::eAmdOpenSource] = 2;
                            driver_priorities[vk::DriverId::eAmdProprietary] = 3;
                            break;
                        case VK_VENDOR_ID_INTEL:
                            driver_priorities[vk::DriverId::eIntelOpenSourceMESA] = 1;
                            driver_priorities[vk::DriverId::eIntelProprietaryWindows] = 2;
                            break;
                        case VK_VENDOR_ID_NVIDIA:
                            driver_priorities[vk::DriverId::eNvidiaProprietary] = 1;
#if defined(VK_API_VERSION_1_3) && VK_HEADER_VERSION >= 235
                            driver_priorities[vk::DriverId::eMesaNvk] = 2;
#endif
                            break;
                    }

                    if (driver_priorities.count(old_driver.driverID)) {
                        old_priority = driver_priorities[old_driver.driverID];
                    }
                    if (driver_priorities.count(new_driver.driverID)) {
                        new_priority = driver_priorities[new_driver.driverID];
                    }

                    if (new_priority < old_priority) {
                        auto r = std::remove(vk_instance.device_indices.begin(), vk_instance.device_indices.end(), *old_device);
                        vk_instance.device_indices.erase(r, vk_instance.device_indices.end());
                        vk_instance.device_indices.push_back(i);

                        VK_LOG_DEBUG("Prioritize device " << i << " driver " << new_driver.driverName << " over device " << *old_device << " driver " << old_driver.driverName);
                    }
                    else {
                        VK_LOG_DEBUG("Prioritize device " << *old_device << " driver " << old_driver.driverName << " over device " << i << " driver " << new_driver.driverName << std::endl);
                    }
                }
            }
        }

        // If no dedicated GPUs found, fall back to GPU 0
        if (vk_instance.device_indices.empty()) {
            vk_instance.device_indices.push_back(0);
        }
    }

    std::cerr << "ggml_vulkan: Found " << vk_instance.device_indices.size() << " Vulkan devices:" << std::endl;

    for (size_t i = 0; i < vk_instance.device_indices.size(); i++) {
        ggml_vk_print_gpu_info(i);
    }
}

static void ggml_vk_init(ggml_backend_vk_context * ctx, size_t idx) {
    VK_LOG_DEBUG("ggml_vk_init(" << ctx->name << ", " << idx << ")");
    ggml_vk_instance_init();
    GGML_ASSERT(idx < vk_instance.device_indices.size());

    ctx->name = GGML_VK_NAME + std::to_string(idx);

    ctx->device = ggml_vk_get_device(idx);

    ctx->semaphore_idx = 0;
    ctx->event_idx = 0;

    ctx->prealloc_size_x = 0;
    ctx->prealloc_size_y = 0;
    ctx->prealloc_size_split_k = 0;

    ctx->fence = ctx->device->device.createFence({});

    ctx->staging_size = 0;
    ctx->staging_offset = 0;

#ifdef GGML_VULKAN_CHECK_RESULTS
    const char* skip_checks = getenv("GGML_VULKAN_SKIP_CHECKS");
    vk_skip_checks = (skip_checks == NULL ? 0 : atoi(skip_checks));
    const char* output_tensor = getenv("GGML_VULKAN_OUTPUT_TENSOR");
    vk_output_tensor = (output_tensor == NULL ? 0 : atoi(output_tensor));
#endif
}

static vk_pipeline ggml_vk_get_to_fp16(ggml_backend_vk_context * ctx, ggml_type type) {
    VK_LOG_DEBUG("ggml_vk_get_to_fp16()");
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ4_NL:
            break;
        default:
            return nullptr;
    }

    return ctx->device->pipeline_dequant[type];
}

static vk_matmul_pipeline ggml_vk_get_mul_mat_mat_pipeline(ggml_backend_vk_context * ctx, ggml_type src0_type, ggml_type src1_type) {
    VK_LOG_DEBUG("ggml_vk_get_mul_mat_mat_pipeline(" << ggml_type_name(src0_type) << ", " << ggml_type_name(src1_type) << ")");
    if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
        return ctx->device->pipeline_matmul_f32;
    }
    if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F16) {
        return ctx->device->pipeline_matmul_f32_f16;
    }
    if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F32) {
        return ctx->device->pipeline_matmul_f16_f32;
    }
    if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F16) {
        return ctx->device->pipeline_matmul_f16;
    }

    if (src1_type != GGML_TYPE_F32) {
        return nullptr;
    }

    switch (src0_type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ4_NL:
            break;
        default:
            return nullptr;
    }

    return ctx->device->pipeline_dequant_mul_mat_mat[src0_type];
}

static vk_pipeline ggml_vk_get_dequantize_mul_mat_vec(ggml_backend_vk_context * ctx, ggml_type a_type, ggml_type b_type) {
    VK_LOG_DEBUG("ggml_vk_get_dequantize_mul_mat_vec()");
    GGML_ASSERT(b_type == GGML_TYPE_F32 || b_type == GGML_TYPE_F16);

    switch (a_type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ4_NL:
            break;
        default:
            return nullptr;
    }

    return b_type == GGML_TYPE_F32 ? ctx->device->pipeline_dequant_mul_mat_vec_f32_f32[a_type] : ctx->device->pipeline_dequant_mul_mat_vec_f16_f32[a_type];
}

static vk_matmul_pipeline ggml_vk_get_mul_mat_mat_id_pipeline(ggml_backend_vk_context * ctx, ggml_type src0_type, ggml_type src1_type) {
    VK_LOG_DEBUG("ggml_vk_get_mul_mat_mat_id_pipeline()");
    if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
        return ctx->device->pipeline_matmul_id_f32;
    }
    if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F32) {
        return ctx->device->pipeline_matmul_id_f16_f32;
    }
    if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F16) {
        return ctx->device->pipeline_matmul_id_f16;
    }

    GGML_ASSERT(src1_type == GGML_TYPE_F32);

    switch (src0_type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ4_NL:
            break;
        default:
            return nullptr;
    }

    return ctx->device->pipeline_dequant_mul_mat_mat_id[src0_type];
}

static vk_pipeline ggml_vk_get_dequantize_mul_mat_vec_id(ggml_backend_vk_context * ctx, ggml_type a_type, ggml_type b_type) {
    VK_LOG_DEBUG("ggml_vk_get_dequantize_mul_mat_vec()");
    GGML_ASSERT(b_type == GGML_TYPE_F32);

    switch (a_type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ4_NL:
            break;
        default:
            return nullptr;
    }

    return ctx->device->pipeline_dequant_mul_mat_vec_id_f32[a_type];
}

static vk_buffer ggml_vk_pool_malloc(ggml_backend_vk_context * ctx, size_t size) {
    VK_LOG_DEBUG("ggml_vk_pool_malloc(" << size << ")");
    VK_LOG_MEMORY("ggml_vk_pool_malloc");

    int best_i = -1;
    size_t best_size = std::numeric_limits<size_t>::max(); //smallest unused buffer that fits our needs
    int worst_i = -1;
    size_t worst_size = 0; //largest unused buffer seen so far
    for (int i = 0; i < MAX_VK_BUFFERS; ++i) {
        vk_buffer &b = ctx->buffer_pool[i];
        if (b != nullptr && b->size >= size && b->size < best_size) {
            best_i = i;
            best_size = b->size;
        }
        if (b != nullptr && b->size > worst_size) {
            worst_i = i;
            worst_size = b->size;
        }
    }
    if(best_i != -1) {
        //found the smallest buffer that fits our needs
        vk_buffer b = ctx->buffer_pool[best_i];
        ctx->buffer_pool[best_i].reset();
        return b;
    }
    if(worst_i != -1) {
        //no buffer that fits our needs, resize largest one to save memory
        vk_buffer& b = ctx->buffer_pool[worst_i];
        ggml_vk_destroy_buffer(b);
    }

    return ggml_vk_create_buffer_device(ctx->device, size);
}

static void ggml_vk_pool_free(ggml_backend_vk_context * ctx, vk_buffer& buffer) {
    VK_LOG_DEBUG("ggml_vk_pool_free(" << buffer->size << ")");
    for (int i = 0; i < MAX_VK_BUFFERS; ++i) {
        vk_buffer& b = ctx->buffer_pool[i];
        if (b == nullptr) {
            b = buffer;
            return;
        }
    }
    std::cerr << "ggml_vulkan: WARNING: vk buffer pool full, increase MAX_VK_BUFFERS" << std::endl;
    ggml_vk_destroy_buffer(buffer);
}

// Returns an available temporary buffer that may only be used temporarily, it will be reused
static vk_buffer ggml_vk_create_buffer_temp(ggml_backend_vk_context * ctx, size_t size) {
    // Try to find existing temp buffer with enough capacity
    for (auto& buffer : ctx->gc.temp_buffers) {
        if (buffer->size >= size) {
            return buffer;
        }
    }

    VK_LOG_MEMORY("ggml_vk_create_buffer_temp(" << size << ")");

    // Otherwise create new buffer
    vk_buffer buf = ggml_vk_pool_malloc(ctx, size);
    ctx->gc.temp_buffers.push_back(buf);

    return buf;
}

static void * ggml_vk_host_malloc(vk_device& device, size_t size) {
    VK_LOG_MEMORY("ggml_vk_host_malloc(" << size << ")");
    vk_buffer buf = ggml_vk_create_buffer(device, size,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    if(!(buf->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible)) {
        fprintf(stderr, "WARNING: failed to allocate %.2f MB of pinned memory\n",
            size/1024.0/1024.0);
        device->device.freeMemory(buf->device_memory);
        device->device.destroyBuffer(buf->buffer);
        return nullptr;
    }

    device->pinned_memory.push_back(std::make_tuple(buf->ptr, size, buf));

    return buf->ptr;
}

static void ggml_vk_host_free(vk_device& device, void* ptr) {
    if (ptr == nullptr) {
        return;
    }
    VK_LOG_MEMORY("ggml_vk_host_free(" << ptr << ")");
    vk_buffer buf;
    size_t index;
    for (size_t i = 0; i < device->pinned_memory.size(); i++) {
        const uint8_t* addr = (const uint8_t*) std::get<0>(device->pinned_memory[i]);
        const uint8_t* endr = addr + std::get<1>(device->pinned_memory[i]);
        if (ptr >= addr && ptr < endr) {
            buf = std::get<2>(device->pinned_memory[i]);
            index = i;
            break;
        }
    }
    if (buf == nullptr) {
        fprintf(stderr, "WARNING: failed to free pinned memory: memory not in map\n");
        return;
    }

    ggml_vk_destroy_buffer(buf);

    device->pinned_memory.erase(device->pinned_memory.begin() + index);
}

static void ggml_vk_host_get(vk_device& device, const void * ptr, vk_buffer& buf, size_t& buf_offset) {
    buf = nullptr;
    buf_offset = 0;
    for (size_t i = 0; i < device->pinned_memory.size(); i++) {
        const uint8_t* addr = (const uint8_t*) std::get<0>(device->pinned_memory[i]);
        const uint8_t* endr = addr + std::get<1>(device->pinned_memory[i]);
        if (ptr >= addr && ptr < endr) {
            buf = std::get<2>(device->pinned_memory[i]);
            buf_offset = ((const uint8_t *)ptr) - addr;
            break;
        }
    }
}

static vk_submission ggml_vk_begin_submission(vk_device& device, vk_queue& q, bool one_time = true) {
    vk_submission s;
    s.buffer = ggml_vk_create_cmd_buffer(device, q);
    if (one_time) {
        s.buffer.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
    } else {
        s.buffer.begin({ vk::CommandBufferUsageFlags{} });
    }

    return s;
}

static void ggml_vk_dispatch_pipeline(ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline& pipeline, std::vector<vk_subbuffer>&& buffers, size_t push_constant_size, const void* push_constants, std::array<uint32_t, 3> elements) {
    const uint32_t wg0 = CEIL_DIV(elements[0], pipeline->wg_denoms[0]);
    const uint32_t wg1 = CEIL_DIV(elements[1], pipeline->wg_denoms[1]);
    const uint32_t wg2 = CEIL_DIV(elements[2], pipeline->wg_denoms[2]);
    VK_LOG_DEBUG("ggml_vk_dispatch_pipeline(" << pipeline->name << ", {";
    for (auto& buffer : buffers) {
        std::cerr << "(" << buffer.buffer << ", " << buffer.offset << ", " << buffer.size << "), ";
    }
    std::cerr << "}, (" << wg0 << "," << wg1 << "," << wg2 << "))");
    std::vector<vk::DescriptorBufferInfo> descriptor_buffer_infos;
    std::vector<vk::WriteDescriptorSet> write_descriptor_sets;
    GGML_ASSERT(pipeline->descriptor_set_idx < pipeline->descriptor_sets.size());
    GGML_ASSERT(buffers.size() == pipeline->parameter_count);
    vk::DescriptorSet& descriptor_set = pipeline->descriptor_sets[pipeline->descriptor_set_idx++];
    for (uint32_t i = 0; i < pipeline->parameter_count; i++) {
        descriptor_buffer_infos.push_back({buffers[i].buffer->buffer, buffers[i].offset, buffers[i].size});
    }
    for (uint32_t i = 0; i < pipeline->parameter_count; i++) {
        write_descriptor_sets.push_back({descriptor_set, i, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &descriptor_buffer_infos[i]});
    }

    ctx->device->device.updateDescriptorSets(write_descriptor_sets, {});

    subctx->s->buffer.pushConstants(pipeline->layout, vk::ShaderStageFlagBits::eCompute, 0, push_constant_size, push_constants);
    subctx->s->buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->pipeline);
    subctx->s->buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                pipeline->layout,
                                0,
                                { descriptor_set },
                                {});
    subctx->s->buffer.dispatch(wg0, wg1, wg2);
}

static void ggml_vk_end_submission(vk_submission& s, std::vector<vk_semaphore> wait_semaphores, std::vector<vk_semaphore> signal_semaphores) {
    s.buffer.end();

    s.wait_semaphores = std::move(wait_semaphores);
    s.signal_semaphores = std::move(signal_semaphores);
}

static void ggml_vk_ctx_end(vk_context& ctx) {
    VK_LOG_DEBUG("ggml_vk_ctx_end(" << ctx << ", " << ctx->seqs.size() << ")");
    if (ctx->s == nullptr) {
        return;
    }

    ctx->s->buffer.end();
    ctx->s = nullptr;
}

static void ggml_vk_ctx_begin(vk_device& device, vk_context& subctx) {
    VK_LOG_DEBUG("ggml_vk_ctx_begin(" << device->name << ")");
    if (subctx->s != nullptr) {
        ggml_vk_ctx_end(subctx);
    }

    subctx->seqs.push_back({ ggml_vk_begin_submission(device, *subctx->q) });
    subctx->s = subctx->seqs[subctx->seqs.size() - 1].data();
}

static size_t ggml_vk_align_size(size_t width, size_t align) {
    VK_LOG_DEBUG("ggml_vk_align_size(" << width << ", " << align << ")");
    return CEIL_DIV(width, align) * align;
}

static void deferred_memcpy(void * dst, const void * src, size_t size, std::vector<vk_staging_memcpy>* memcpys = nullptr) {
    if (memcpys == nullptr) {
        memcpy(dst, src, size);
    } else {
        memcpys->emplace_back(dst, src, size);
    }
}

static void ggml_vk_ensure_sync_staging_buffer(vk_device& device, size_t size) {
    if (device->sync_staging == nullptr || device->sync_staging->size < size) {
        VK_LOG_MEMORY("ggml_vk_ensure_sync_staging_buffer(" << size << ")");
        ggml_vk_destroy_buffer(device->sync_staging);
        device->sync_staging = ggml_vk_create_buffer_check(device, size,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    }
}

static void ggml_vk_buffer_write_nc_async(ggml_backend_vk_context * ctx, vk_context& subctx, vk_buffer& dst, size_t offset, const ggml_tensor * tensor, bool sync_staging = false) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_nc_async(" << tensor << ")");
    GGML_ASSERT(!ggml_is_contiguous(tensor));
    // Buffer is already mapped
    if(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
        std::cerr << "ggml_vulkan: buffer_write_nc_async dst buffer is host_visible. Use synchronous write." << std::endl;
        GGML_ABORT("fatal error");
    }
    // Check if src is pinned memory
    vk_buffer buf;
    size_t buf_offset;
    ggml_vk_host_get(ctx->device, tensor->data, buf, buf_offset);

    const uint64_t ne0 = tensor->ne[0];
    const uint64_t ne1 = tensor->ne[1];
    const uint64_t ne2 = tensor->ne[2];
    const uint64_t ne3 = tensor->ne[3];
    const uint64_t nb0 = tensor->nb[0];
    const uint64_t nb1 = tensor->nb[1];
    const uint64_t nb2 = tensor->nb[2];
    const uint64_t nb3 = tensor->nb[3];
    const ggml_type type = tensor->type;
    const uint64_t ts = ggml_type_size(type);
    const uint64_t bs = ggml_blck_size(type);

    const uint64_t dstnb0 = ts;
    const uint64_t dstnb1 = dstnb0*(ne0/bs);
    const uint64_t dstnb2 = dstnb1*ne1;
    const uint64_t dstnb3 = dstnb2*ne2;

    const uint64_t ne = ggml_nelements(tensor);

    if (buf != nullptr) {
        // Memory is pinned, use as staging buffer
        std::vector<vk::BufferCopy> slices;

        for (uint64_t i3 = 0; i3 < ne3; i3++) {
            for (uint64_t i2 = 0; i2 < ne2; i2++) {
                // Find longest contiguous slice
                if (ne1*nb1 == dstnb2) {
                    slices.push_back({ buf_offset + i3*nb3 + i2*nb2, offset + i3*dstnb3 + i2*dstnb2, dstnb2 });
                } else {
                    for (uint64_t i1 = 0; i1 < ne1; i1++) {
                        if (ne0*nb0/bs == dstnb1) {
                            slices.push_back({ buf_offset + i3*nb3 + i2*nb2 + i1*nb1, offset + i3*dstnb3 + i2*dstnb2 + i1*dstnb1, dstnb1 });
                        } else {
                            const uint64_t s_off = buf_offset + i3*nb3 + i2*nb2 + i1*nb1;
                            const uint64_t d_off = offset + i3*dstnb3 + i2*dstnb2 + i1*dstnb1;
                            for (uint64_t i0 = 0; i0 < ne0; i0++) {
                                slices.push_back({ s_off + i1*nb0, d_off + i0*dstnb0, dstnb0 });
                            }
                        }
                    }
                }
            }
        }

        ggml_vk_sync_buffers(subctx);
        subctx->s->buffer.copyBuffer(buf->buffer, dst->buffer, slices);
        return;
    }

    // Staging buffer required
    vk_buffer staging = ctx->staging;
    size_t staging_offset = ctx->staging_offset;
    const size_t copy_size = ts*ne/bs;
    if (ctx->staging->size < ctx->staging_offset + copy_size) {
        if (sync_staging) {
            // Create temporary larger buffer
            ggml_vk_ensure_sync_staging_buffer(ctx->device, copy_size);

            staging = ctx->device->sync_staging;
            staging_offset = 0;
        } else {
            GGML_ABORT("fatal error");
        }
    }

    VkBufferCopy buf_copy{ staging_offset, offset, copy_size };

    ggml_vk_sync_buffers(subctx);
    vkCmdCopyBuffer(subctx->s->buffer, staging->buffer, dst->buffer, 1, &buf_copy);

    for (uint64_t i3 = 0; i3 < ne3; i3++) {
        for (uint64_t i2 = 0; i2 < ne2; i2++) {
            // Find longest contiguous slice
            if (ne1*nb1 == dstnb2) {
                deferred_memcpy((uint8_t *)staging->ptr + staging_offset + i3*dstnb3 + i2*dstnb2, (const uint8_t *) tensor->data + buf_offset + i3*nb3 + i2*nb2, dstnb2, &subctx->in_memcpys);
            } else {
                for (uint64_t i1 = 0; i1 < ne1; i1++) {
                    if (ne0*nb0/bs == dstnb1) {
                        deferred_memcpy((uint8_t *)staging->ptr + staging_offset + i3*dstnb3 + i2*dstnb2 + i1*dstnb1, (const uint8_t *) tensor->data + buf_offset + i3*nb3 + i2*nb2 + i1*nb1, dstnb1, &subctx->in_memcpys);
                    } else {
                        const uint64_t s_off = buf_offset + i3*nb3 + i2*nb2 + i1*nb1;
                        const uint64_t d_off = staging_offset + i3*dstnb3 + i2*dstnb2 + i1*dstnb1;
                        for (uint64_t i0 = 0; i0 < ne0; i0++) {
                            deferred_memcpy((uint8_t *)staging->ptr + d_off + i0*dstnb0, (const uint8_t *) tensor->data + s_off + i0*nb0, dstnb0, &subctx->in_memcpys);
                        }
                    }
                }
            }
        }
    }
}

static void ggml_vk_buffer_write_2d_async(vk_context subctx, vk_buffer& dst, size_t offset, const void * src, size_t spitch, size_t width, size_t height, vk_buffer staging_buffer, size_t staging_offset, bool sync_staging = false) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_2d_async(" << width << ", " << height << ")");
    // Buffer is already mapped
    if(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
        std::cerr << "ggml_vulkan: buffer_write_async dst buffer is host_visible. Use synchronous write." << std::endl;
        GGML_ABORT("fatal error");
    }
    // Check if src is pinned memory
    vk_buffer buf = nullptr;
    size_t buf_offset;
    ggml_vk_host_get(dst->device, src, buf, buf_offset);

    if (buf != nullptr) {
        // Memory is pinned, use as staging buffer
        std::vector<vk::BufferCopy> slices(1);
        if (width == spitch) {
            // Only do single write if stride is equal
            slices[0].srcOffset = buf_offset;
            slices[0].dstOffset = offset;
            slices[0].size = width * height;
        } else {
            slices.resize(height);
            for (size_t i = 0; i < height; i++) {
                slices[i].srcOffset = buf_offset + i * spitch;
                slices[i].dstOffset = offset + i * width;
                slices[i].size = width;
            }
        }

        ggml_vk_sync_buffers(subctx);
        subctx->s->buffer.copyBuffer(buf->buffer, dst->buffer, slices);
        return;
    }
    VK_LOG_DEBUG("STAGING");

    // Staging buffer required
    const size_t copy_size = width*height;
    if (staging_buffer == nullptr || staging_buffer->size < staging_offset + copy_size) {
        if (sync_staging) {
            ggml_vk_ensure_sync_staging_buffer(dst->device, copy_size);

            staging_buffer = dst->device->sync_staging;
            staging_offset = 0;
        } else {
            GGML_ABORT("fatal error");
        }
    }

    VkBufferCopy buf_copy = {
        staging_offset,
        offset,
        copy_size};

    ggml_vk_sync_buffers(subctx);
    vkCmdCopyBuffer(subctx->s->buffer, staging_buffer->buffer, dst->buffer, 1, &buf_copy);

    if (width == spitch) {
        deferred_memcpy((uint8_t *)staging_buffer->ptr + staging_offset, src, width * height, &subctx->in_memcpys);
    } else {
        for (size_t i = 0; i < height; i++) {
            deferred_memcpy((uint8_t *)staging_buffer->ptr + staging_offset + i * width, (const uint8_t *) src + i * spitch, width, &subctx->in_memcpys);
        }
    }
}

static void ggml_vk_buffer_write_async(vk_context subctx, vk_buffer& dst, size_t offset, const void * src, size_t size, vk_buffer staging_buffer, size_t staging_offset, bool sync_staging = false) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_async(" << size << ")");
    return ggml_vk_buffer_write_2d_async(subctx, dst, offset, src, size, size, 1, staging_buffer, staging_offset, sync_staging);
}

static void ggml_vk_buffer_write_2d(vk_buffer& dst, size_t offset, const void * src, size_t spitch, size_t width, size_t height) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_2d(" << width << ", " << height << ")");
    // Buffer is already mapped
    if(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
        GGML_ASSERT(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCoherent);

        for (size_t i = 0; i < height; i++) {
            memcpy((uint8_t *)dst->ptr + offset + i * width, (const uint8_t *) src + i * spitch, width);
        }
    } else {
        vk_context subctx = ggml_vk_create_temporary_context(dst->device->transfer_queue);
        ggml_vk_ctx_begin(dst->device, subctx);
        ggml_vk_buffer_write_2d_async(subctx, dst, offset, src, spitch, width, height, nullptr, 0, true);
        ggml_vk_ctx_end(subctx);

        for (auto& cpy : subctx->in_memcpys) {
            memcpy(cpy.dst, cpy.src, cpy.n);
        }

        ggml_vk_submit(subctx, dst->device->fence);
        VK_CHECK(dst->device->device.waitForFences({ dst->device->fence }, true, UINT64_MAX), "vk_buffer_write_2d waitForFences");
        dst->device->device.resetFences({ dst->device->fence });
    }
}

static void ggml_vk_buffer_write(vk_buffer& dst, size_t offset, const void * src, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_write(" << size << ")");
    ggml_vk_buffer_write_2d(dst, offset, src, 0, size, 1);
}

static void ggml_vk_buffer_read_2d_async(vk_context subctx, vk_buffer& src, size_t offset, void * dst, size_t spitch, size_t dpitch, size_t width, size_t height, vk_buffer staging_buffer, size_t staging_offset, bool sync_staging = false) {
    VK_LOG_DEBUG("ggml_vk_buffer_read_2d_async(offset=" << offset << ", width=" << width << ", height=" << height << ")");
    GGML_ASSERT(width > 0);
    GGML_ASSERT(height > 0);
    GGML_ASSERT(src != nullptr);

    // TODO: staging_offset is not used

    // Check if dst is pinned memory
    vk_buffer buf = nullptr;
    size_t buf_offset;
    ggml_vk_host_get(src->device, dst, buf, buf_offset);

    std::vector<vk::BufferCopy> slices(1);
    if (width == spitch && width == dpitch) {
        // Only do single write if stride is equal
        slices[0].srcOffset = offset;
        slices[0].dstOffset = buf_offset;
        slices[0].size = width * height;
    } else {
        slices.resize(height);
        for (size_t i = 0; i < height; i++) {
            slices[i].srcOffset = offset + i * spitch;
            slices[i].dstOffset = buf_offset + i * dpitch;
            slices[i].size = width;
        }
    }

    if (buf != nullptr) {
        // Memory is pinned, use as staging buffer
        ggml_vk_sync_buffers(subctx);
        subctx->s->buffer.copyBuffer(src->buffer, buf->buffer, slices);

        return;
    }
    VK_LOG_DEBUG("STAGING");

    // Fall back to staging buffer
    const size_t copy_size = dpitch * height;
    if (staging_buffer == nullptr || staging_buffer->size < staging_offset + copy_size) {
        if (sync_staging) {
            // Create temporary larger buffer
            ggml_vk_ensure_sync_staging_buffer(src->device, copy_size);

            staging_buffer = src->device->sync_staging;
        } else {
            GGML_ABORT("fatal error");
        }
    }

    ggml_vk_sync_buffers(subctx);
    subctx->s->buffer.copyBuffer(src->buffer, staging_buffer->buffer, slices);

    deferred_memcpy(dst, staging_buffer->ptr, copy_size, &subctx->out_memcpys);
}

static void ggml_vk_buffer_read_async(vk_context subctx, vk_buffer& src, size_t offset, void * dst, size_t size, vk_buffer staging_buffer, size_t staging_offset, bool sync_staging = false) {
    return ggml_vk_buffer_read_2d_async(subctx, src, offset, dst, size, size, size, 1, staging_buffer, staging_offset, sync_staging);
}

static void ggml_vk_buffer_read(vk_buffer& src, size_t offset, void * dst, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_read(" << src->buffer << ", " << offset << ", " << size << ")");
    if(src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
        GGML_ASSERT(src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCoherent);

        memcpy(dst, (uint8_t *) src->ptr + offset, size);
    } else {
        vk_context subctx = ggml_vk_create_temporary_context(src->device->transfer_queue);
        ggml_vk_ctx_begin(src->device, subctx);
        ggml_vk_buffer_read_async(subctx, src, offset, dst, size, nullptr, 0, true);
        ggml_vk_ctx_end(subctx);

        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, true, UINT64_MAX), "vk_buffer_read waitForFences");
        src->device->device.resetFences({ src->device->fence });

        for (auto& cpy : subctx->out_memcpys) {
            memcpy(cpy.dst, cpy.src, cpy.n);
        }
    }
}

static void ggml_vk_buffer_copy_async(vk_context& ctx, vk_buffer& dst, size_t dst_offset, vk_buffer& src, size_t src_offset, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_copy_async(" << size << ")");
    // Make sure both buffers are on same device
    GGML_ASSERT(src->device == dst->device);

    VkBufferCopy bc{ src_offset, dst_offset, size };

    vkCmdCopyBuffer(ctx->s->buffer, src->buffer, dst->buffer, 1, &bc);
}

static void ggml_vk_buffer_copy(vk_buffer& dst, size_t dst_offset, vk_buffer& src, size_t src_offset, size_t size) {
    if (src->device == dst->device) {
        VK_LOG_DEBUG("ggml_vk_buffer_copy(SINGLE_DEVICE, " << size << ")");
        // Copy within the device
        vk_context subctx = ggml_vk_create_temporary_context(src->device->transfer_queue);
        ggml_vk_ctx_begin(src->device, subctx);
        ggml_vk_buffer_copy_async(subctx, dst, dst_offset, src, src_offset, size);
        ggml_vk_ctx_end(subctx);
        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, true, UINT64_MAX), "vk_buffer_copy waitForFences");
        src->device->device.resetFences({ src->device->fence });
    } else {
        VK_LOG_DEBUG("ggml_vk_buffer_copy(MULTI_DEVICE, " << size << ")");
        // Copy device to device
        ggml_vk_ensure_sync_staging_buffer(src->device, size);
        ggml_vk_ensure_sync_staging_buffer(dst->device, size);

        // Copy to src staging buffer
        ggml_vk_buffer_copy(src->device->sync_staging, 0, src, src_offset, size);
        // memcpy to dst staging buffer
        memcpy(dst->device->sync_staging->ptr, src->device->sync_staging->ptr, size);
        // Copy to dst buffer
        ggml_vk_buffer_copy(dst, dst_offset, dst->device->sync_staging, 0, size);
    }
}

static void ggml_vk_buffer_memset(vk_buffer& dst, size_t offset, uint32_t c, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_memset(" << offset << ", " << c << ", " << size << ")");

    vk_context subctx = ggml_vk_create_temporary_context(dst->device->transfer_queue);
    ggml_vk_ctx_begin(dst->device, subctx);
    subctx->s->buffer.fillBuffer(dst->buffer, offset, size, c);
    ggml_vk_ctx_end(subctx);

    ggml_vk_submit(subctx, dst->device->fence);
    VK_CHECK(dst->device->device.waitForFences({ dst->device->fence }, true, UINT64_MAX), "vk_memset waitForFences");
    dst->device->device.resetFences({ dst->device->fence });
}

static uint32_t ggml_vk_guess_split_k(int m, int n, int k) {
    VK_LOG_DEBUG("ggml_vk_guess_split_k(" << m << ", " << n << ", " << k << ")");
    // if (k > 128 && (m < 128 || n < 128) && m > 2 && n > 2) {
    //     return 4;
    // }

    return 1;

    GGML_UNUSED(m); GGML_UNUSED(n); GGML_UNUSED(k);
}

static vk_pipeline ggml_vk_guess_matmul_pipeline_amd(ggml_backend_vk_context * ctx, vk_matmul_pipeline& mmp, int m, int n, bool aligned) {
    if (m <= 32 || n <= 32) {
        return aligned ? mmp->a_s : mmp->s;
    }
    return aligned ? mmp->a_m : mmp->m;

    GGML_UNUSED(ctx);
}

static vk_pipeline ggml_vk_guess_matmul_pipeline_apple(ggml_backend_vk_context * ctx, vk_matmul_pipeline& mmp, bool aligned) {
    return aligned ? mmp->a_m : mmp->m;

    GGML_UNUSED(ctx);
}

static vk_pipeline ggml_vk_guess_matmul_pipeline_intel(ggml_backend_vk_context * ctx, vk_matmul_pipeline& mmp, bool aligned) {
    return aligned ? mmp->a_s : mmp->s;

    GGML_UNUSED(ctx);
}

static vk_pipeline ggml_vk_guess_matmul_pipeline(ggml_backend_vk_context * ctx, vk_matmul_pipeline& mmp, int m, int n, bool aligned) {
    VK_LOG_DEBUG("ggml_vk_guess_matmul_pipeline(" << m << ", " << n << ", " << aligned << ")");
    switch (ctx->device->vendor_id) {
    case VK_VENDOR_ID_AMD:
        return ggml_vk_guess_matmul_pipeline_amd(ctx, mmp, m, n, aligned);
    case VK_VENDOR_ID_APPLE:
        return ggml_vk_guess_matmul_pipeline_apple(ctx, mmp, aligned);
    case VK_VENDOR_ID_INTEL:
        return ggml_vk_guess_matmul_pipeline_intel(ctx, mmp, aligned);
    default:
        break;
    }

    if (m <= 32 || n <= 32) {
        return aligned ? mmp->a_s : mmp->s;
    }
    if (m <= 64 || n <= 64) {
        return aligned ? mmp->a_m : mmp->m;
    }
    return aligned ? mmp->a_l : mmp->l;
}

static uint32_t ggml_vk_guess_matmul_pipeline_align(ggml_backend_vk_context * ctx, vk_matmul_pipeline& mmp, int m, int n) {
    VK_LOG_DEBUG("ggml_vk_guess_matmul_pipeline_align(" << m << ", " << n << ")");
    return ggml_vk_guess_matmul_pipeline(ctx, mmp, m, n, true)->align;
}

static void ggml_vk_matmul(
        ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline& pipeline,
        vk_subbuffer&& a, vk_subbuffer&& b, vk_subbuffer&& d, vk_subbuffer&& split_k_buffer,
        uint32_t m, uint32_t n, uint32_t k, uint32_t stride_a, uint32_t stride_b, uint32_t stride_d,
        uint32_t batch_stride_a, uint32_t batch_stride_b, uint32_t batch_stride_d,
        uint32_t split_k, uint32_t batch, uint32_t ne02, uint32_t ne12, uint32_t broadcast2, uint32_t broadcast3) {
        VK_LOG_DEBUG("ggml_vk_matmul(a: (" << a.buffer->buffer << ", " << a.offset << ", " << a.size << "), b: (" << b.buffer->buffer << ", " << b.offset << ", " << b.size << "), d: (" << d.buffer->buffer << ", " << d.offset << ", " << d.size << "), split_k: (" << (split_k_buffer.buffer != nullptr ? split_k_buffer.buffer->buffer : VK_NULL_HANDLE) << ", " << split_k_buffer.offset << ", " << split_k_buffer.size << "), m: " << m << ", n: " << n << ", k: " << k << ", stride_a: " << stride_a << ", stride_b: " << stride_b << ", stride_d: " << stride_d << ", batch_stride_a: " << batch_stride_a << ", batch_stride_b: " << batch_stride_b << ", batch_stride_d: " << batch_stride_d << ", split_k: " << split_k << ", batch: " << batch << ", ne02: " << ne02 << ", ne12: " << ne12 << ", broadcast2: " << broadcast2 << ", broadcast3: " << broadcast3 << ")");
    ggml_vk_sync_buffers(subctx);
    if (split_k == 1) {
        const vk_mat_mat_push_constants pc = { m, n, k, stride_a, stride_b, stride_d, batch_stride_a, batch_stride_b, batch_stride_d, k, ne02, ne12, broadcast2, broadcast3 };
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { a, b, d }, sizeof(vk_mat_mat_push_constants), &pc, { m, n, batch });
        return;
    }

    GGML_ASSERT(batch_stride_d == m * n);

    const vk_mat_mat_push_constants pc1 = { m, n, k, stride_a, stride_b, stride_d, batch_stride_a, batch_stride_b, batch_stride_d, CEIL_DIV(k, split_k), ne02, ne12, broadcast2, broadcast3 };
    // Make sure enough workgroups get assigned for split k to work
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { a, b, split_k_buffer }, sizeof(vk_mat_mat_push_constants), &pc1, { (CEIL_DIV(m, pipeline->wg_denoms[0]) * pipeline->wg_denoms[0]) * split_k, n, batch });
    ggml_vk_sync_buffers(subctx);
    const std::array<uint32_t, 2> pc2 = { (uint32_t)(m * n * batch), split_k };
    ggml_vk_dispatch_pipeline(ctx, subctx, ctx->device->pipeline_matmul_split_k_reduce, { split_k_buffer, d }, pc2.size() * sizeof(uint32_t), pc2.data(), { m * n * batch, 1, 1 });
}

static void ggml_vk_matmul_id(
        ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline& pipeline,
        vk_subbuffer&& a, vk_subbuffer&& b, vk_subbuffer&& d, vk_subbuffer&& ids,
        uint32_t m, uint32_t n, uint32_t k, uint32_t stride_a, uint32_t stride_b, uint32_t stride_d,
        uint32_t batch_stride_a, uint32_t batch_stride_b, uint32_t batch_stride_d,
        uint32_t n_as, uint32_t nei0, uint32_t nei1, uint32_t nbi1, uint32_t ne11) {
    VK_LOG_DEBUG("ggml_vk_matmul_id(a: (" << a.buffer->buffer << ", " << a.offset << ", " << a.size << "), b: (" << b.buffer->buffer << ", " << b.offset << ", " << b.size << "), d: (" << d.buffer->buffer << ", " << d.offset << ", " << d.size << "), ids: (" << ids.buffer->buffer << ", " << ids.offset << ", " << ids.size << "), " <<
        "m: " << m << ", n: " << n << ", k: " << k << ", stride_a: " << stride_a << ", stride_b: " << stride_b << ", stride_d: " << stride_d << ", " <<
        "batch_stride_a: " << batch_stride_a << ", batch_stride_b: " << batch_stride_b << ", batch_stride_d: " << batch_stride_d << ", " <<
        "n_as: " << n_as << ", nei0: " << nei0 << ", nei1: " << nei1 << ", nbi1: " << nbi1 << ", ne11: " << ne11 << ")");
    ggml_vk_sync_buffers(subctx);
    const vk_mat_mat_id_push_constants pc = { m, n, k, stride_a, stride_b, stride_d, batch_stride_a, batch_stride_b, batch_stride_d,
                                              nei0, nei1, nbi1, ne11 };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { a, b, d, ids }, sizeof(vk_mat_mat_id_push_constants), &pc, { m, nei1, n_as });
}

static bool ggml_vk_dim01_contiguous(const ggml_tensor * tensor) {
    return
        tensor->nb[0] == ggml_type_size(tensor->type) &&
        tensor->nb[1] == (tensor->nb[0]*tensor->ne[0])/ggml_blck_size(tensor->type) &&
        tensor->nb[3] == tensor->nb[2]*tensor->ne[2];
}

static vk_pipeline ggml_vk_get_cpy_pipeline(ggml_backend_vk_context * ctx, ggml_type from, ggml_type to) {
    if (from == GGML_TYPE_F32 && to == GGML_TYPE_F32) {
        return ctx->device->pipeline_cpy_f32_f32;
    }
    if (from == GGML_TYPE_F32 && to == GGML_TYPE_F16) {
        return ctx->device->pipeline_cpy_f32_f16;
    }
    if (from == GGML_TYPE_F16 && to == GGML_TYPE_F16) {
        return ctx->device->pipeline_cpy_f16_f16;
    }

    std::cerr << "Missing CPY op for types: " << ggml_type_name(from) << " " << ggml_type_name(to) << std::endl;
    GGML_ABORT("fatal error");
}

static void ggml_vk_cpy_to_contiguous(ggml_backend_vk_context * ctx, vk_context& subctx, vk_pipeline pipeline, const ggml_tensor * tensor, vk_subbuffer&& in, vk_subbuffer&& out) {
    VK_LOG_DEBUG("ggml_vk_cpy_to_contiguous((" << tensor << ", type=" << tensor->type << ", ne0=" << tensor->ne[0] << ", ne1=" << tensor->ne[1] << ", ne2=" << tensor->ne[2] << ", ne3=" << tensor->ne[3] << ", nb0=" << tensor->nb[0] << ", nb1=" << tensor->nb[1] << ", nb2=" << tensor->nb[2] << ", nb3=" << tensor->nb[3] << "), ";
    std::cerr << "buffer in size=" << in.buffer->size << ", buffer out size=" << out.buffer->size << ")");
    const int tensor_type_size = ggml_type_size(tensor->type);

    const uint32_t ne = ggml_nelements(tensor);

    const vk_op_unary_push_constants pc = {
        (uint32_t)ne,
        (uint32_t)tensor->ne[0], (uint32_t)tensor->ne[1], (uint32_t)tensor->ne[2], (uint32_t)tensor->ne[3], (uint32_t)tensor->nb[0] / tensor_type_size, (uint32_t)tensor->nb[1] / tensor_type_size, (uint32_t)tensor->nb[2] / tensor_type_size, (uint32_t)tensor->nb[3] / tensor_type_size,
        (uint32_t)tensor->ne[0], (uint32_t)tensor->ne[1], (uint32_t)tensor->ne[2], (uint32_t)tensor->ne[3],                       1                   , (uint32_t)tensor->ne[0]                   , (uint32_t)(tensor->ne[0] * tensor->ne[1]) , (uint32_t)(tensor->ne[0] * tensor->ne[1] * tensor->ne[2]),
        0,
        0.0f, 0.0f,
    };
    ggml_vk_sync_buffers(subctx);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { in, out }, sizeof(vk_op_unary_push_constants), &pc, { ne, 1, 1 });
}

static void ggml_vk_mul_mat_q_f16(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_mul_mat_q_f16((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3] << "),)");
    GGML_ASSERT(ggml_vk_dim01_contiguous(src0) || src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);  // NOLINT
    GGML_ASSERT(ggml_vk_dim01_contiguous(src1) || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);  // NOLINT

    const uint64_t ne00 = src0->ne[0];
    const uint64_t ne01 = src0->ne[1];
    const uint64_t ne02 = src0->ne[2];
    const uint64_t ne03 = src0->ne[3];

    const uint64_t ne10 = src1->ne[0];
    const uint64_t ne11 = src1->ne[1];
    const uint64_t ne12 = src1->ne[2];
    const uint64_t ne13 = src1->ne[3];

    const uint64_t ne20 = dst->ne[0];
    const uint64_t ne21 = dst->ne[1];

    const uint64_t r2 = ne12 / ne02;
    const uint64_t r3 = ne13 / ne03;

    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) dst->extra;
    ggml_tensor_extra_gpu * extra_src0 = (ggml_tensor_extra_gpu *) src0->extra;
    ggml_tensor_extra_gpu * extra_src1 = (ggml_tensor_extra_gpu *) src1->extra;

    vk_buffer d_Qx;
    size_t qx_buf_offset = 0;
    vk_buffer d_Qy;
    size_t qy_buf_offset = 0;

    bool src0_uma = false;
    bool src1_uma = false;

    if (ctx->device->uma) {
        ggml_vk_host_get(ctx->device, src0->data, d_Qx, qx_buf_offset);
        ggml_vk_host_get(ctx->device, src1->data, d_Qy, qy_buf_offset);
        src0_uma = d_Qx != nullptr;
        src1_uma = d_Qy != nullptr;
    }

    const bool x_non_contig = !ggml_vk_dim01_contiguous(src0);
    const bool y_non_contig = !ggml_vk_dim01_contiguous(src1);

    const bool y_f32_kernel = src1->type == GGML_TYPE_F32 && !y_non_contig;

    vk_matmul_pipeline mmp = ggml_vk_get_mul_mat_mat_pipeline(ctx, src0->type, y_non_contig ? GGML_TYPE_F16 : src1->type);

    const bool qx_needs_dequant = mmp == nullptr || x_non_contig;
    const bool qy_needs_dequant = (src1->type != GGML_TYPE_F16 && !y_f32_kernel) || y_non_contig;

    if (mmp == nullptr) {
        // Fall back to dequant + f16 mulmat
        mmp = ggml_vk_get_mul_mat_mat_pipeline(ctx, GGML_TYPE_F16, y_f32_kernel ? GGML_TYPE_F32 : GGML_TYPE_F16);
    }

    // Not implemented
    GGML_ASSERT(y_non_contig || !qy_needs_dequant);  // NOLINT

    const int x_ne = ne01 * ne00;
    const int y_ne = ne11 * ne10;
    const int d_ne = ne11 * ne01;

    const uint32_t kpad = ggml_vk_align_size(ne10, ggml_vk_guess_matmul_pipeline_align(ctx, mmp, ne01, ne11));
    const bool aligned = ne10 == kpad && ne01 > 8 && ne11 > 8;

    const uint32_t split_k = ggml_vk_guess_split_k(ne01, ne11, ne10);

    vk_pipeline pipeline = ggml_vk_guess_matmul_pipeline(ctx, mmp, ne01, ne11, aligned);

    const uint64_t qx_sz = ggml_type_size(src0->type) * x_ne / ggml_blck_size(src0->type);
    const uint64_t qy_sz = ggml_type_size(src1->type) * y_ne / ggml_blck_size(src1->type);
    const uint64_t x_sz = !qx_needs_dequant ? qx_sz : sizeof(ggml_fp16_t) * x_ne;
    const uint64_t y_sz = y_f32_kernel ? sizeof(float) * y_ne : sizeof(ggml_fp16_t) * y_ne;
    const uint64_t d_sz = sizeof(float) * d_ne;

    vk_buffer d_D = extra->buffer_gpu.lock();
    const uint64_t d_buf_offset = extra->offset + dst->view_offs;
    GGML_ASSERT(d_D != nullptr);
    GGML_ASSERT(d_D->size >= d_buf_offset + d_sz * ne02 * ne03);
    vk_buffer d_X;
    uint64_t x_buf_offset = 0;
    vk_buffer d_Y;
    uint64_t y_buf_offset = 0;
    if (!src0_uma) {
        d_Qx = extra_src0->buffer_gpu.lock();
        qx_buf_offset = extra_src0->offset + src0->view_offs;
        GGML_ASSERT(d_Qx != nullptr);
    }
    if (!src1_uma) {
        d_Qy = extra_src1->buffer_gpu.lock();
        qy_buf_offset = extra_src1->offset + src1->view_offs;
        GGML_ASSERT(d_Qy != nullptr);
    }
    if (qx_needs_dequant) {
        d_X = ctx->prealloc_x;
        GGML_ASSERT(d_X->size >= x_sz * ne02 * ne03);
    } else {
        d_X = d_Qx;
        x_buf_offset = qx_buf_offset;
        GGML_ASSERT(qx_sz == x_sz);
    }
    if (qy_needs_dequant) {
        d_Y = ctx->prealloc_y;
        GGML_ASSERT(d_Y->size >= y_sz * ne02 * ne03);
    } else {
        d_Y = d_Qy;
        y_buf_offset = qy_buf_offset;
        GGML_ASSERT(qy_sz == y_sz);
    }

    vk_pipeline to_fp16_vk_0 = nullptr;
    vk_pipeline to_fp16_vk_1 = nullptr;

    if (x_non_contig) {
        to_fp16_vk_0 = ggml_vk_get_cpy_pipeline(ctx, src0->type, GGML_TYPE_F16);
    } else {
        to_fp16_vk_0 = ggml_vk_get_to_fp16(ctx, src0->type);
    }
    if (y_non_contig) {
        to_fp16_vk_1 = ggml_vk_get_cpy_pipeline(ctx, src1->type, GGML_TYPE_F16);
    } else {
        to_fp16_vk_1 = ggml_vk_get_to_fp16(ctx, src1->type);
    }
    GGML_ASSERT(!qx_needs_dequant || to_fp16_vk_0 != nullptr);  // NOLINT
    GGML_ASSERT(!qy_needs_dequant || to_fp16_vk_1 != nullptr);  // NOLINT

    // Allocate descriptor sets
    ggml_pipeline_allocate_descriptor_sets(ctx->device, pipeline, 1);
    if (qx_needs_dequant) {
        ggml_pipeline_allocate_descriptor_sets(ctx->device, to_fp16_vk_0, 1);
    }
    if (qy_needs_dequant) {
        ggml_pipeline_allocate_descriptor_sets(ctx->device, to_fp16_vk_1, 1);
    }
    if (split_k > 1) {
        ggml_pipeline_allocate_descriptor_sets(ctx->device, ctx->device->pipeline_matmul_split_k_reduce, 1);
    }

    if (x_non_contig) {
        ggml_vk_cpy_to_contiguous(ctx, subctx, to_fp16_vk_0, src0, { d_Qx, qx_buf_offset, VK_WHOLE_SIZE }, { d_X, 0, VK_WHOLE_SIZE });
    } else if (qx_needs_dequant) {
        const std::vector<uint32_t> pc = { (uint32_t)ne01, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)(ggml_nelements(src0)) };
        ggml_vk_sync_buffers(subctx);
        ggml_vk_dispatch_pipeline(ctx, subctx, to_fp16_vk_0, { { d_Qx, qx_buf_offset, qx_sz * ne02 * ne03 }, { d_X, 0, x_sz * ne02 * ne03 } }, pc.size() * sizeof(uint32_t), pc.data(), { (uint32_t)(x_ne * ne02 * ne03), 1, 1});
    }
    if (y_non_contig) {
        ggml_vk_cpy_to_contiguous(ctx, subctx, to_fp16_vk_1, src1, { d_Qy, qy_buf_offset, VK_WHOLE_SIZE }, { d_Y, 0, VK_WHOLE_SIZE });
    }

    uint32_t stride_batch_x = ne00*ne01;
    uint32_t stride_batch_y = ne10*ne11;

    if (!ggml_vk_dim01_contiguous(src0) && !qx_needs_dequant) {
        stride_batch_x = src0->nb[0] / ggml_type_size(src0->type);
    }

    if (!ggml_vk_dim01_contiguous(src1) && !qy_needs_dequant) {
        stride_batch_y = src1->nb[0] / ggml_type_size(src1->type);
    }

    // compute
    ggml_vk_matmul(
        ctx, subctx, pipeline,
        { d_X, x_buf_offset, x_sz * ne02 * ne03 }, { d_Y, y_buf_offset, y_sz * ne12 * ne13 },
        { d_D, d_buf_offset, d_sz * ne12 * ne13 }, { ctx->prealloc_split_k, 0, d_sz * ne12 * ne13 * split_k },
        ne01, ne11, ne10,
        ne10, ne10, ne01, stride_batch_x, stride_batch_y, ne20*ne21,
        split_k, ne12*ne13, ne02, ne12, r2, r3
    );  // NOLINT
}

static void ggml_vk_mul_mat_vec_q_f16(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_mul_mat_vec_q_f16((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3] << "),)");
    GGML_ASSERT(ggml_vk_dim01_contiguous(src0) || src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);  // NOLINT
    GGML_ASSERT(ggml_vk_dim01_contiguous(src1) || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);  // NOLINT

    const uint64_t ne00 = src0->ne[0];
    const uint64_t ne01 = src0->ne[1];
    const uint64_t ne02 = src0->ne[2];
    const uint64_t ne03 = src0->ne[3];

    const uint64_t ne10 = src1->ne[0];
    const uint64_t ne11 = src1->ne[1];
    const uint64_t ne12 = src1->ne[2];
    const uint64_t ne13 = src1->ne[3];

    GGML_ASSERT(ne11 == 1);

    const uint64_t ne20 = dst->ne[0];
    const uint64_t ne21 = dst->ne[1];
    const uint64_t ne22 = dst->ne[2];
    const uint64_t ne23 = dst->ne[3];

    const uint64_t r2 = ne12 / ne02;
    const uint64_t r3 = ne13 / ne03;

    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) dst->extra;
    ggml_tensor_extra_gpu * extra_src0 = (ggml_tensor_extra_gpu *) src0->extra;
    ggml_tensor_extra_gpu * extra_src1 = (ggml_tensor_extra_gpu *) src1->extra;

    vk_buffer d_Qx;
    size_t qx_buf_offset = 0;
    vk_buffer d_Qy;
    size_t qy_buf_offset = 0;

    bool src0_uma = false;
    bool src1_uma = false;

    if (ctx->device->uma) {
        ggml_vk_host_get(ctx->device, src0->data, d_Qx, qx_buf_offset);
        ggml_vk_host_get(ctx->device, src1->data, d_Qy, qy_buf_offset);
        src0_uma = d_Qx != nullptr;
        src1_uma = d_Qy != nullptr;
    }

    const bool x_non_contig = !ggml_vk_dim01_contiguous(src0);
    const bool y_non_contig = !ggml_vk_dim01_contiguous(src1);

    const bool f16_f32_kernel = src1->type == GGML_TYPE_F32;

    const bool qx_needs_dequant = x_non_contig;
    const bool qy_needs_dequant = (src1->type != GGML_TYPE_F16 && !f16_f32_kernel) || y_non_contig;

    // Not implemented
    GGML_ASSERT(y_non_contig || !qy_needs_dequant);  // NOLINT

    const uint64_t x_ne = ne01 * ne00;
    const uint64_t y_ne = ne11 * ne10;
    const uint64_t d_ne = ne11 * ne01;

    const uint64_t qx_sz = ggml_vk_align_size(ggml_type_size(src0->type) * x_ne / ggml_blck_size(src0->type), ctx->device->properties.limits.minStorageBufferOffsetAlignment);
    const uint64_t qy_sz = ggml_type_size(src1->type) * y_ne / ggml_blck_size(src1->type);
    const uint64_t x_sz = x_non_contig ? ggml_vk_align_size(ggml_type_size(src0->type) * x_ne, ctx->device->properties.limits.minStorageBufferOffsetAlignment) : qx_sz;
    const uint64_t y_sz = f16_f32_kernel ? sizeof(float) * y_ne : sizeof(ggml_fp16_t) * y_ne;
    const uint64_t d_sz = sizeof(float) * d_ne;

    vk_buffer d_D = extra->buffer_gpu.lock();
    const uint64_t d_buf_offset = extra->offset + dst->view_offs;
    GGML_ASSERT(d_D != nullptr);
    vk_buffer d_X;
    uint64_t x_buf_offset = 0;
    vk_buffer d_Y;
    uint64_t y_buf_offset = 0;
    if(!src0_uma) {
        d_Qx = extra_src0->buffer_gpu.lock();
        qx_buf_offset = extra_src0->offset + src0->view_offs;
        GGML_ASSERT(d_Qx != nullptr);
    }
    if(!src1_uma) {
        d_Qy = extra_src1->buffer_gpu.lock();
        qy_buf_offset = extra_src1->offset + src1->view_offs;
        GGML_ASSERT(d_Qy != nullptr);
    }
    if (qx_needs_dequant) {
        d_X = ctx->prealloc_x;
    } else {
        d_X = d_Qx;
        x_buf_offset = qx_buf_offset;
        GGML_ASSERT(qx_sz == x_sz);
    }
    if (qy_needs_dequant) {
        d_Y = ctx->prealloc_y;
    } else {
        d_Y = d_Qy;
        y_buf_offset = qy_buf_offset;
        GGML_ASSERT(qy_sz == y_sz);
    }

    vk_pipeline to_fp16_vk_0 = nullptr;
    vk_pipeline to_fp16_vk_1 = nullptr;
    if (x_non_contig) {
        to_fp16_vk_0 = ggml_vk_get_cpy_pipeline(ctx, src0->type, src0->type);
    }
    if (y_non_contig) {
        to_fp16_vk_1 = ggml_vk_get_cpy_pipeline(ctx, src1->type, src1->type);
    } else {
        to_fp16_vk_1 = ggml_vk_get_to_fp16(ctx, src1->type);
    }
    vk_pipeline dmmv = ggml_vk_get_dequantize_mul_mat_vec(ctx, src0->type, src1->type);
    GGML_ASSERT(!qx_needs_dequant || to_fp16_vk_0 != nullptr);  // NOLINT
    GGML_ASSERT(!qy_needs_dequant || to_fp16_vk_1 != nullptr);  // NOLINT
    GGML_ASSERT(dmmv != nullptr);

    // Allocate descriptor sets
    if (qx_needs_dequant) {
        ggml_pipeline_allocate_descriptor_sets(ctx->device, to_fp16_vk_0, 1);
    }
    if (qy_needs_dequant) {
        ggml_pipeline_allocate_descriptor_sets(ctx->device, to_fp16_vk_1, y_non_contig ? 1 : ne12 * ne13);
    }
    ggml_pipeline_allocate_descriptor_sets(ctx->device, dmmv, ne12 * ne13);

    if (x_non_contig) {
        GGML_ASSERT(x_sz == ggml_vk_align_size(ggml_type_size(src0->type) * x_ne, ctx->device->properties.limits.minStorageBufferOffsetAlignment));
        ggml_vk_cpy_to_contiguous(ctx, subctx, to_fp16_vk_0, src0, { d_Qx, qx_buf_offset, VK_WHOLE_SIZE }, { d_X, 0, VK_WHOLE_SIZE });
    }
    if (y_non_contig) {
        GGML_ASSERT(y_sz == ggml_type_size(src1->type) * y_ne);
        ggml_vk_cpy_to_contiguous(ctx, subctx, to_fp16_vk_1, src1, { d_Qy, qy_buf_offset, VK_WHOLE_SIZE }, { d_Y, 0, VK_WHOLE_SIZE });
    }

    uint32_t stride_batch_x = ne00*ne01;
    uint32_t stride_batch_y = ne10*ne11;

    if (!ggml_vk_dim01_contiguous(src0) && !qx_needs_dequant) {
        stride_batch_x = src0->nb[0] / ggml_type_size(src0->type);
    }

    if (!ggml_vk_dim01_contiguous(src1) && !qy_needs_dequant) {
        stride_batch_y = src1->nb[0] / ggml_type_size(src1->type);
    }

    const uint32_t max_groups_x = ctx->device->properties.limits.maxComputeWorkGroupCount[0];

    uint32_t groups_x = ne01;
    uint32_t groups_z = 1;

    if (ne01 > max_groups_x) {
        groups_z = 64;
        groups_x /= groups_z;
    }

    // compute
    const vk_mat_vec_push_constants pc = {
        (uint32_t)ne00, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)ne01,
        stride_batch_x, stride_batch_y, (uint32_t)(ne20*ne21),
        (uint32_t)ne02, (uint32_t)ne12, (uint32_t)r2, (uint32_t)r3,
    };
    ggml_vk_sync_buffers(subctx);
    ggml_vk_dispatch_pipeline(ctx, subctx, dmmv,
                              { { d_X, x_buf_offset, x_sz * ne02 * ne03 }, { d_Y, y_buf_offset, y_sz * ne12 * ne13 }, { d_D, d_buf_offset, d_sz * ne22 * ne23} },
                              sizeof(vk_mat_vec_push_constants), &pc, { groups_x, (uint32_t)(ne12 * ne13), groups_z });
}

static void ggml_vk_mul_mat_vec_p021_f16_f32(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_mul_mat_p021_f16_f32(" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3] << "),)");
    GGML_ASSERT(ggml_is_permuted(src0) && ggml_is_permuted(src1));
    GGML_ASSERT(src0->nb[0] <= src0->nb[1] && src0->nb[2] <= src0->nb[3]);  // NOLINT
    GGML_ASSERT(src1->nb[0] <= src1->nb[1] && src1->nb[2] <= src1->nb[3]);  // NOLINT
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const uint64_t ne00 = src0->ne[0];
    const uint64_t ne01 = src0->ne[1];
    const uint64_t ne02 = src0->ne[2];
    // const uint64_t ne03 = src0->ne[3];

    const uint64_t ne10 = src1->ne[0];
    const uint64_t ne11 = src1->ne[1];
    const uint64_t ne12 = src1->ne[2];
    // const uint64_t ne13 = src1->ne[3];

    GGML_ASSERT(ne11 == 1);

    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) dst->extra;
    ggml_tensor_extra_gpu * extra_src0 = (ggml_tensor_extra_gpu *) src0->extra;
    ggml_tensor_extra_gpu * extra_src1 = (ggml_tensor_extra_gpu *) src1->extra;

    vk_buffer d_Qy;
    size_t qy_buf_offset = 0;

    bool src1_uma = false;

    if (ctx->device->uma) {
        ggml_vk_host_get(ctx->device, src1->data, d_Qy, qy_buf_offset);
        src1_uma = d_Qy != nullptr;
    }

    const uint64_t x_ne = ne00 * ne01 * ne02;
    const uint64_t y_ne = ne10 * ne11 * ne12;
    const uint64_t d_ne = ne01 * ne11 * ne12;

    const uint64_t qx_sz = ggml_vk_align_size(ggml_type_size(src0->type) * x_ne / ggml_blck_size(src0->type), ctx->device->properties.limits.minStorageBufferOffsetAlignment);
    const uint64_t qy_sz = ggml_type_size(src1->type) * y_ne / ggml_blck_size(src1->type);
    const uint64_t d_sz = sizeof(float) * d_ne;

    vk_buffer d_D = extra->buffer_gpu.lock();
    const uint64_t d_buf_offset = extra->offset + dst->view_offs;
    GGML_ASSERT(d_D != nullptr);
    vk_buffer d_Qx = extra_src0->buffer_gpu.lock();
    const uint64_t qx_buf_offset = extra_src0->offset + src0->view_offs;
    GGML_ASSERT(d_Qx != nullptr);
    if (!src1_uma) {
        d_Qy = extra_src1->buffer_gpu.lock();
        qy_buf_offset = extra_src1->offset + src1->view_offs;
        GGML_ASSERT(d_Qx != nullptr);
    }

    // Allocate descriptor sets
    ggml_pipeline_allocate_descriptor_sets(ctx->device, ctx->device->pipeline_mul_mat_vec_p021_f16_f32, 1);

    const uint64_t qy_buffer_offset = (qy_buf_offset / ctx->device->properties.limits.minStorageBufferOffsetAlignment) * ctx->device->properties.limits.minStorageBufferOffsetAlignment;
    const uint64_t qy_shader_offset = qy_buf_offset - qy_buffer_offset;

    const uint64_t d_buffer_offset = (d_buf_offset / ctx->device->properties.limits.minStorageBufferOffsetAlignment) * ctx->device->properties.limits.minStorageBufferOffsetAlignment;
    const uint64_t d_shader_offset = d_buf_offset - d_buffer_offset;

    // compute
    const std::array<uint32_t, 6> pc = { (uint32_t)ne00, (uint32_t)ne01, (uint32_t)ne02, (uint32_t)ne12, (uint32_t)(qy_shader_offset / ggml_type_size(src1->type)), (uint32_t)(d_shader_offset / ggml_type_size(dst->type)) };
    ggml_vk_sync_buffers(subctx);
    ggml_vk_dispatch_pipeline(ctx, subctx, ctx->device->pipeline_mul_mat_vec_p021_f16_f32, { { d_Qx, qx_buf_offset, qx_sz }, { d_Qy, qy_buffer_offset, qy_sz + qy_shader_offset }, { d_D, d_buffer_offset, d_sz + d_shader_offset } }, 6 * sizeof(uint32_t), &pc, { 1, (uint32_t)ne01, (uint32_t)ne12 });
}

static void ggml_vk_mul_mat_vec_nc_f16_f32(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_mul_mat_nc_f16_f32((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3] << "),)");
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_is_permuted(src0));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const uint64_t ne00 = src0->ne[0];
    const uint64_t ne01 = src0->ne[1];
    const uint64_t ne02 = src0->ne[2];
    // const uint64_t ne03 = src0->ne[3];

    const uint64_t nb01 = src0->nb[1];
    const uint64_t nb02 = src0->nb[2];

    // const uint64_t ne10 = src1->ne[0];
    const uint64_t ne11 = src1->ne[1];
    const uint64_t ne12 = src1->ne[2];
    // const uint64_t ne13 = src1->ne[3];

    GGML_ASSERT(ne11 == 1);

    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) dst->extra;
    ggml_tensor_extra_gpu * extra_src0 = (ggml_tensor_extra_gpu *) src0->extra;
    ggml_tensor_extra_gpu * extra_src1 = (ggml_tensor_extra_gpu *) src1->extra;

    vk_buffer d_Qy = nullptr;
    size_t qy_buf_offset = 0;

    bool src1_uma = false;

    if (ctx->device->uma) {
        ggml_vk_host_get(ctx->device, src1->data, d_Qy, qy_buf_offset);
        src1_uma = d_Qy != nullptr;
    }

    const uint64_t d_ne = ne01 * ne11 * ne12;

    const uint32_t row_stride_x = nb01 / sizeof(ggml_fp16_t);
    const uint32_t channel_stride_x = nb02 / sizeof(ggml_fp16_t);

    const uint64_t qx_sz = ggml_nbytes(src0);
    const uint64_t qy_sz = ggml_nbytes(src1);
    const uint64_t d_sz = sizeof(float) * d_ne;

    vk_buffer d_D = extra->buffer_gpu.lock();
    const uint64_t d_buf_offset = extra->offset + dst->view_offs;
    GGML_ASSERT(d_D != nullptr);
    vk_buffer d_Qx = extra_src0->buffer_gpu.lock();
    const uint64_t qx_buf_offset = extra_src0->offset + src0->view_offs;
    GGML_ASSERT(d_Qx != nullptr);
    if (!src1_uma) {
        d_Qy = extra_src1->buffer_gpu.lock();
        qy_buf_offset = extra_src1->offset + src1->view_offs;
        GGML_ASSERT(d_Qx != nullptr);
    }

    // Allocate descriptor sets
    ggml_pipeline_allocate_descriptor_sets(ctx->device, ctx->device->pipeline_mul_mat_vec_nc_f16_f32, 1);

    const uint64_t qy_buffer_offset = (qy_buf_offset / ctx->device->properties.limits.minStorageBufferOffsetAlignment) * ctx->device->properties.limits.minStorageBufferOffsetAlignment;
    const uint64_t qy_shader_offset = qy_buf_offset - qy_buffer_offset;

    const uint64_t d_buffer_offset = (d_buf_offset / ctx->device->properties.limits.minStorageBufferOffsetAlignment) * ctx->device->properties.limits.minStorageBufferOffsetAlignment;
    const uint64_t d_shader_offset = d_buf_offset - d_buffer_offset;

    // compute
    const std::array<uint32_t, 7> pc = { (uint32_t)ne00, (uint32_t)ne01, row_stride_x, channel_stride_x, (uint32_t)(ne12 / ne02), (uint32_t)(qy_shader_offset / ggml_type_size(src1->type)), (uint32_t)(d_shader_offset / ggml_type_size(dst->type)) };
    ggml_vk_sync_buffers(subctx);
    ggml_vk_dispatch_pipeline(ctx, subctx, ctx->device->pipeline_mul_mat_vec_nc_f16_f32, { { d_Qx, qx_buf_offset, qx_sz }, { d_Qy, qy_buffer_offset, qy_sz + qy_shader_offset }, { d_D, d_buffer_offset, d_sz + d_shader_offset } }, 7 * sizeof(uint32_t), &pc, { 1, (uint32_t)ne01, (uint32_t)ne12 });
}

static void ggml_vk_mul_mat(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_mul_mat(" << src0 << ", " << src1 << ", " << dst << ")");
    if (src0->type == GGML_TYPE_F16 && ggml_is_permuted(src0) && ggml_is_permuted(src1) && dst->ne[1] == 1) {
        ggml_vk_mul_mat_vec_p021_f16_f32(ctx, subctx, src0, src1, dst);
    } else if (src0->type == GGML_TYPE_F16 && !ggml_is_contiguous(src0) && !ggml_is_transposed(src1) && dst->ne[1] == 1) {
        ggml_vk_mul_mat_vec_nc_f16_f32(ctx, subctx, src0, src1, dst);
    } else if (dst->ne[1] == 1 && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type))) {
        ggml_vk_mul_mat_vec_q_f16(ctx, subctx, src0, src1, dst);
    } else {
        ggml_vk_mul_mat_q_f16(ctx, subctx, src0, src1, dst);
    }
}

static void ggml_vk_mul_mat_id_q_f16(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_mul_mat_id_q_f16((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << ids << ", name=" << ids->name << ", type=" << ids->type << ", ne0=" << ids->ne[0] << ", ne1=" << ids->ne[1] << ", ne2=" << ids->ne[2] << ", ne3=" << ids->ne[3] << ", nb0=" << ids->nb[0] << ", nb1=" << ids->nb[1] << ", nb2=" << ids->nb[2] << ", nb3=" << ids->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3] << "),)");
    GGML_ASSERT(ggml_vk_dim01_contiguous(src1) || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);  // NOLINT
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    const uint64_t ne00 = src0->ne[0];
    const uint64_t ne01 = src0->ne[1];
    const uint64_t ne02 = src0->ne[2];
    const uint64_t ne03 = src0->ne[3];

    const uint64_t ne10 = src1->ne[0];
    const uint64_t ne11 = src1->ne[1];
    const uint64_t ne12 = src1->ne[2];
    const uint64_t ne13 = src1->ne[3];

    const uint64_t nei0 = ids->ne[0];
    const uint64_t nei1 = ids->ne[1];
    GGML_ASSERT(nei0 * nei1 <= 3072);

    const uint32_t nbi1 = ids->nb[1];
    const uint32_t nbi2 = ids->nb[2];

    const uint64_t ne20 = dst->ne[0];
    const uint64_t ne21 = dst->ne[1];
    const uint64_t ne22 = dst->ne[2];
    const uint64_t ne23 = dst->ne[3];

    const uint64_t n_as = ne02;

    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) dst->extra;
    ggml_tensor_extra_gpu * extra_src0 = (ggml_tensor_extra_gpu *) src0->extra;
    ggml_tensor_extra_gpu * extra_src1 = (ggml_tensor_extra_gpu *) src1->extra;
    ggml_tensor_extra_gpu * extra_ids = (ggml_tensor_extra_gpu *) ids->extra;

    vk_buffer d_Qx;
    size_t qx_buf_offset = 0;
    vk_buffer d_Qy;
    size_t qy_buf_offset = 0;
    vk_buffer d_ids;
    size_t ids_buf_offset = 0;

    bool src0_uma = false;
    bool src1_uma = false;
    bool ids_uma = false;

    if (ctx->device->uma) {
        ggml_vk_host_get(ctx->device, src0->data, d_Qx, qx_buf_offset);
        ggml_vk_host_get(ctx->device, src1->data, d_Qy, qy_buf_offset);
        ggml_vk_host_get(ctx->device, ids->data, d_ids, ids_buf_offset);
        src0_uma = d_Qx != nullptr;
        src1_uma = d_Qy != nullptr;
        ids_uma = d_ids != nullptr;
    }

    const bool x_non_contig = !ggml_vk_dim01_contiguous(src0);
    const bool y_non_contig = !ggml_vk_dim01_contiguous(src1);

    const bool y_f32_kernel = src1->type == GGML_TYPE_F32 && !y_non_contig;

    vk_matmul_pipeline mmp = ggml_vk_get_mul_mat_mat_id_pipeline(ctx, src0->type, y_non_contig ? GGML_TYPE_F16 : src1->type);

    const bool qx_needs_dequant = mmp == nullptr || x_non_contig;
    const bool qy_needs_dequant = (src1->type != GGML_TYPE_F16 && !y_f32_kernel) || y_non_contig;

    if (mmp == nullptr) {
        GGML_ABORT("fatal error");
    }

    // Not implemented
    GGML_ASSERT(y_non_contig || !qy_needs_dequant);  // NOLINT

    const uint64_t x_ne = ne01 * ne00;
    const uint64_t y_ne = ne11 * ne10;
    const uint64_t d_ne = ne21 * ne20;

    const uint32_t kpad = ggml_vk_align_size(ne10, ggml_vk_guess_matmul_pipeline_align(ctx, mmp, ne01, nei1));
    const bool aligned = ne10 == kpad && ne01 > 8 && nei1 > 8;

    vk_pipeline pipeline = ggml_vk_guess_matmul_pipeline(ctx, mmp, ne01, nei1, aligned);

    const uint64_t qx_sz = ggml_type_size(src0->type) * x_ne / ggml_blck_size(src0->type);
    const uint64_t qy_sz = ggml_type_size(src1->type) * y_ne / ggml_blck_size(src1->type);
    const uint64_t x_sz = !qx_needs_dequant ? qx_sz : sizeof(ggml_fp16_t) * x_ne;
    const uint64_t y_sz = y_f32_kernel ? sizeof(float) * y_ne : sizeof(ggml_fp16_t) * y_ne;
    const uint64_t ids_sz = nbi2;
    const uint64_t d_sz = sizeof(float) * d_ne;

    vk_buffer d_D = extra->buffer_gpu.lock();
    const uint64_t d_buf_offset = extra->offset + dst->view_offs;
    GGML_ASSERT(d_D != nullptr);
    vk_buffer d_X;
    uint64_t x_buf_offset = 0;
    vk_buffer d_Y;
    uint64_t y_buf_offset = 0;
    if (!src0_uma) {
        d_Qx = extra_src0->buffer_gpu.lock();
        qx_buf_offset = extra_src0->offset + src0->view_offs;
        GGML_ASSERT(d_Qx != nullptr);
    }
    if (!src1_uma) {
        d_Qy = extra_src1->buffer_gpu.lock();
        qy_buf_offset = extra_src1->offset + src1->view_offs;
        GGML_ASSERT(d_Qy != nullptr);
    }
    if (!ids_uma) {
        d_ids = extra_ids->buffer_gpu.lock();
        ids_buf_offset = extra_ids->offset + ids->view_offs;
        GGML_ASSERT(d_ids != nullptr);
    }
    if (qx_needs_dequant) {
        d_X = ctx->prealloc_x;
        GGML_ASSERT(d_X->size >= x_sz * ne02 * ne03);
    } else {
        d_X = d_Qx;
        x_buf_offset = qx_buf_offset;
        GGML_ASSERT(qx_sz == x_sz);
    }
    if (qy_needs_dequant) {
        d_Y = ctx->prealloc_y;
        GGML_ASSERT(d_Y->size >= y_sz * ne02 * ne03);
    } else {
        d_Y = d_Qy;
        y_buf_offset = qy_buf_offset;
        GGML_ASSERT(qy_sz == y_sz);
    }

    vk_pipeline to_fp16_vk_0 = nullptr;
    vk_pipeline to_fp16_vk_1 = nullptr;

    if (x_non_contig) {
        to_fp16_vk_0 = ggml_vk_get_cpy_pipeline(ctx, src0->type, GGML_TYPE_F16);
    } else {
        to_fp16_vk_0 = ggml_vk_get_to_fp16(ctx, src0->type);
    }
    if (y_non_contig) {
        to_fp16_vk_1 = ggml_vk_get_cpy_pipeline(ctx, src1->type, GGML_TYPE_F16);
    } else {
        to_fp16_vk_1 = ggml_vk_get_to_fp16(ctx, src1->type);
    }
    GGML_ASSERT(!qx_needs_dequant || to_fp16_vk_0 != nullptr);  // NOLINT
    GGML_ASSERT(!qy_needs_dequant || to_fp16_vk_1 != nullptr);  // NOLINT

    // Allocate descriptor sets
    ggml_pipeline_allocate_descriptor_sets(ctx->device, pipeline, 1);
    if (qx_needs_dequant) {
        ggml_pipeline_allocate_descriptor_sets(ctx->device, to_fp16_vk_0, 1);
    }
    if (qy_needs_dequant) {
        ggml_pipeline_allocate_descriptor_sets(ctx->device, to_fp16_vk_1, 1);
    }

    if (x_non_contig) {
        ggml_vk_cpy_to_contiguous(ctx, subctx, to_fp16_vk_0, src0, { d_Qx, qx_buf_offset, VK_WHOLE_SIZE }, { d_X, 0, VK_WHOLE_SIZE });
    } else if (qx_needs_dequant) {
        const std::vector<uint32_t> pc = { (uint32_t)ne01, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)(ggml_nelements(src0)) };
        ggml_vk_sync_buffers(subctx);
        ggml_vk_dispatch_pipeline(ctx, subctx, to_fp16_vk_0, { { d_Qx, qx_buf_offset, qx_sz * ne02 * ne03 }, { d_X, 0, x_sz * ne02 * ne03 } }, pc.size() * sizeof(uint32_t), pc.data(), { (uint32_t)(x_ne * ne02 * ne03), 1, 1});
    }
    if (y_non_contig) {
        ggml_vk_cpy_to_contiguous(ctx, subctx, to_fp16_vk_1, src1, { d_Qy, qy_buf_offset, VK_WHOLE_SIZE }, { d_Y, 0, VK_WHOLE_SIZE });
    }

    uint32_t stride_batch_x = ne00*ne01;
    uint32_t stride_batch_y = ne10*ne11;

    if (!ggml_vk_dim01_contiguous(src0) && !qx_needs_dequant) {
        stride_batch_x = src0->nb[0] / ggml_type_size(src0->type);
    }

    if (!ggml_vk_dim01_contiguous(src1) && !qy_needs_dequant) {
        stride_batch_y = src1->nb[0] / ggml_type_size(src1->type);
    }

    // compute
    ggml_vk_matmul_id(
        ctx, subctx, pipeline,
        { d_X, x_buf_offset, x_sz * ne02 * ne03 }, { d_Y, y_buf_offset, y_sz * ne12 * ne13 },
        { d_D, d_buf_offset, d_sz * ne22 * ne23 }, { d_ids, ids_buf_offset, ids_sz },
        ne01, ne21, ne10, ne10, ne10, ne01,
        stride_batch_x, stride_batch_y, ne20*ne21,
        n_as, nei0, nei1, nbi1 / ggml_type_size(ids->type), ne11
    );  // NOLINT
}

static void ggml_vk_mul_mat_vec_id_q_f16(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_mul_mat_vec_id_q_f16((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    std::cerr << "), (" << ids << ", name=" << ids->name << ", type=" << ids->type << ", ne0=" << ids->ne[0] << ", ne1=" << ids->ne[1] << ", ne2=" << ids->ne[2] << ", ne3=" << ids->ne[3] << ", nb0=" << ids->nb[0] << ", nb1=" << ids->nb[1] << ", nb2=" << ids->nb[2] << ", nb3=" << ids->nb[3];
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3] << "),)");
    GGML_ASSERT(ggml_vk_dim01_contiguous(src0) || src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);  // NOLINT
    GGML_ASSERT(ggml_vk_dim01_contiguous(src1) || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);  // NOLINT
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    const uint64_t ne00 = src0->ne[0];
    const uint64_t ne01 = src0->ne[1];
    const uint64_t ne02 = src0->ne[2];
    const uint64_t ne03 = src0->ne[3];

    const uint64_t ne10 = src1->ne[0];
    const uint64_t ne11 = src1->ne[1];
    const uint64_t ne12 = src1->ne[2];
    const uint64_t ne13 = src1->ne[3];

    const uint64_t nei0 = ids->ne[0];
    const uint64_t nei1 = ids->ne[1];

    const uint64_t nbi2 = ids->nb[2];

    GGML_ASSERT(nei1 == 1);

    const uint64_t ne20 = dst->ne[0];
    const uint64_t ne21 = dst->ne[1];
    const uint64_t ne22 = dst->ne[2];
    const uint64_t ne23 = dst->ne[3];

    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) dst->extra;
    ggml_tensor_extra_gpu * extra_src0 = (ggml_tensor_extra_gpu *) src0->extra;
    ggml_tensor_extra_gpu * extra_src1 = (ggml_tensor_extra_gpu *) src1->extra;
    ggml_tensor_extra_gpu * extra_ids = (ggml_tensor_extra_gpu *) ids->extra;

    vk_buffer d_Qx;
    size_t qx_buf_offset = 0;
    vk_buffer d_Qy;
    size_t qy_buf_offset = 0;
    vk_buffer d_ids;
    size_t ids_buf_offset = 0;

    bool src0_uma = false;
    bool src1_uma = false;
    bool ids_uma = false;

    if (ctx->device->uma) {
        ggml_vk_host_get(ctx->device, src0->data, d_Qx, qx_buf_offset);
        ggml_vk_host_get(ctx->device, src1->data, d_Qy, qy_buf_offset);
        ggml_vk_host_get(ctx->device, ids->data, d_ids, ids_buf_offset);
        src0_uma = d_Qx != nullptr;
        src1_uma = d_Qy != nullptr;
        ids_uma = d_ids != nullptr;
    }

    const bool x_non_contig = !ggml_vk_dim01_contiguous(src0);
    const bool y_non_contig = !ggml_vk_dim01_contiguous(src1);

    const bool f16_f32_kernel = src1->type == GGML_TYPE_F32;

    const bool qx_needs_dequant = x_non_contig;
    const bool qy_needs_dequant = (src1->type != GGML_TYPE_F16 && !f16_f32_kernel) || y_non_contig;

    // Not implemented
    GGML_ASSERT(y_non_contig || !qy_needs_dequant);  // NOLINT

    const uint64_t x_ne = ne01 * ne00;
    const uint64_t y_ne = ne11 * ne10;
    const uint64_t d_ne = ne21 * ne20;

    const uint64_t qx_sz = ggml_vk_align_size(ggml_type_size(src0->type) * x_ne / ggml_blck_size(src0->type), ctx->device->properties.limits.minStorageBufferOffsetAlignment);
    const uint64_t qy_sz = ggml_type_size(src1->type) * y_ne / ggml_blck_size(src1->type);
    const uint64_t x_sz = x_non_contig ? ggml_vk_align_size(ggml_type_size(src0->type) * x_ne, ctx->device->properties.limits.minStorageBufferOffsetAlignment) : qx_sz;
    const uint64_t y_sz = f16_f32_kernel ? sizeof(float) * y_ne : sizeof(ggml_fp16_t) * y_ne;
    const uint64_t ids_sz = nbi2;
    const uint64_t d_sz = sizeof(float) * d_ne;

    vk_buffer d_D = extra->buffer_gpu.lock();
    const uint64_t d_buf_offset = extra->offset + dst->view_offs;
    GGML_ASSERT(d_D != nullptr);
    vk_buffer d_X;
    uint64_t x_buf_offset = 0;
    vk_buffer d_Y;
    uint64_t y_buf_offset = 0;
    if(!src0_uma) {
        d_Qx = extra_src0->buffer_gpu.lock();
        qx_buf_offset = extra_src0->offset + src0->view_offs;
        GGML_ASSERT(d_Qx != nullptr);
    }
    if(!src1_uma) {
        d_Qy = extra_src1->buffer_gpu.lock();
        qy_buf_offset = extra_src1->offset + src1->view_offs;
        GGML_ASSERT(d_Qy != nullptr);
    }
    if(!ids_uma) {
        d_ids = extra_ids->buffer_gpu.lock();
        ids_buf_offset = extra_ids->offset + ids->view_offs;
        GGML_ASSERT(d_ids != nullptr);
    }
    if (qx_needs_dequant) {
        d_X = ctx->prealloc_x;
    } else {
        d_X = d_Qx;
        x_buf_offset = qx_buf_offset;
        GGML_ASSERT(qx_sz == x_sz);
    }
    if (qy_needs_dequant) {
        d_Y = ctx->prealloc_y;
    } else {
        d_Y = d_Qy;
        y_buf_offset = qy_buf_offset;
        GGML_ASSERT(qy_sz == y_sz);
    }

    vk_pipeline to_fp16_vk_0 = nullptr;
    vk_pipeline to_fp16_vk_1 = nullptr;
    if (x_non_contig) {
        to_fp16_vk_0 = ggml_vk_get_cpy_pipeline(ctx, src0->type, src0->type);
    }
    if (y_non_contig) {
        to_fp16_vk_1 = ggml_vk_get_cpy_pipeline(ctx, src1->type, src1->type);
    } else {
        to_fp16_vk_1 = ggml_vk_get_to_fp16(ctx, src1->type);
    }
    vk_pipeline dmmv = ggml_vk_get_dequantize_mul_mat_vec_id(ctx, src0->type, src1->type);
    GGML_ASSERT(!qx_needs_dequant || to_fp16_vk_0 != nullptr);  // NOLINT
    GGML_ASSERT(!qy_needs_dequant || to_fp16_vk_1 != nullptr);  // NOLINT
    GGML_ASSERT(dmmv != nullptr);

    // Allocate descriptor sets
    if (qx_needs_dequant) {
        ggml_pipeline_allocate_descriptor_sets(ctx->device, to_fp16_vk_0, 1);
    }
    if (qy_needs_dequant) {
        ggml_pipeline_allocate_descriptor_sets(ctx->device, to_fp16_vk_1, y_non_contig ? 1 : ne12 * ne13);
    }
    ggml_pipeline_allocate_descriptor_sets(ctx->device, dmmv, ne12 * ne13);

    if (x_non_contig) {
        GGML_ASSERT(x_sz == ggml_vk_align_size(ggml_type_size(src0->type) * x_ne, ctx->device->properties.limits.minStorageBufferOffsetAlignment));
        ggml_vk_cpy_to_contiguous(ctx, subctx, to_fp16_vk_0, src0, { d_Qx, qx_buf_offset, VK_WHOLE_SIZE }, { d_X, 0, VK_WHOLE_SIZE });
    }
    if (y_non_contig) {
        GGML_ASSERT(y_sz == ggml_type_size(src1->type) * y_ne);
        ggml_vk_cpy_to_contiguous(ctx, subctx, to_fp16_vk_1, src1, { d_Qy, qy_buf_offset, VK_WHOLE_SIZE }, { d_Y, 0, VK_WHOLE_SIZE });
    }

    uint32_t stride_batch_y = ne10*ne11;

    if (!ggml_vk_dim01_contiguous(src1) && !qy_needs_dequant) {
        stride_batch_y = src1->nb[0] / ggml_type_size(src1->type);
    }

    const uint32_t max_groups_x = ctx->device->properties.limits.maxComputeWorkGroupCount[0];

    uint32_t groups_x = ne01;
    uint32_t groups_z = 1;

    if (ne01 > max_groups_x) {
        groups_z = 64;
        groups_x /= groups_z;
    }

    // compute
    const vk_mat_vec_id_push_constants pc = {
        (uint32_t)ne00, (uint32_t)ne10, (uint32_t)ne10, (uint32_t)ne01,
        (uint32_t)x_ne, stride_batch_y, (uint32_t)(ne20*ne21),
        (uint32_t)nei0, (uint32_t)ne11,
    };
    ggml_vk_sync_buffers(subctx);
    ggml_vk_dispatch_pipeline(ctx, subctx, dmmv,
        { { d_X, x_buf_offset, x_sz * ne02 * ne03 }, { d_Y, y_buf_offset, y_sz * ne12 * ne13 }, { d_D, d_buf_offset, d_sz * ne22 * ne23}, { d_ids, ids_buf_offset, ids_sz } },
        sizeof(vk_mat_vec_id_push_constants), &pc, { groups_x, (uint32_t)nei0, groups_z });
}

static void ggml_vk_mul_mat_id(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_mul_mat_id(" << src0 << ", " << src1 << ", " << src2 << ", " << dst << ")");
    if (src2->ne[1] == 1 && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type))) {
        ggml_vk_mul_mat_vec_id_q_f16(ctx, subctx, src0, src1, src2, dst);
    } else {
        ggml_vk_mul_mat_id_q_f16(ctx, subctx, src0, src1, src2, dst);
    }
}

static void ggml_vk_op_repeat(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_vk_op_repeat(" << src0 << ", " << src1 << ", " << dst << ")");
    const uint64_t ne0 = dst->ne[0];
    const uint64_t ne1 = dst->ne[1];
    const uint64_t ne2 = dst->ne[2];
    const uint64_t ne3 = dst->ne[3];

    const uint64_t ne00 = src0->ne[0];
    const uint64_t ne01 = src0->ne[1];
    const uint64_t ne02 = src0->ne[2];
    const uint64_t ne03 = src0->ne[3];

    const uint64_t nb0 = dst->nb[0];
    const uint64_t nb1 = dst->nb[1];
    const uint64_t nb2 = dst->nb[2];
    const uint64_t nb3 = dst->nb[3];

    const uint64_t nb00 = src0->nb[0];
    const uint64_t nb01 = src0->nb[1];
    const uint64_t nb02 = src0->nb[2];
    const uint64_t nb03 = src0->nb[3];

    // guaranteed to be an integer due to the check in ggml_can_repeat
    const uint64_t nr0 = ne0/ne00;
    const uint64_t nr1 = ne1/ne01;
    const uint64_t nr2 = ne2/ne02;
    const uint64_t nr3 = ne3/ne03;

    // TODO: support for transposed / permuted tensors
    GGML_ASSERT(nb0  == sizeof(float));
    GGML_ASSERT(nb00 == sizeof(float));

    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) dst->extra;
    ggml_tensor_extra_gpu * extra_src0 = (ggml_tensor_extra_gpu *) src0->extra;

    const vk_buffer src_buf = extra_src0->buffer_gpu.lock();
    const uint64_t src_offset = extra_src0->offset + src0->view_offs;
    vk_buffer dst_buf = extra->buffer_gpu.lock();
    const uint64_t dst_offset = extra->offset + dst->view_offs;

    std::vector<vk::BufferCopy> copies;

    for                         (uint64_t i3 = 0; i3 < nr3;  i3++) {
        for                     (uint64_t k3 = 0; k3 < ne03; k3++) {
            for                 (uint64_t i2 = 0; i2 < nr2;  i2++) {
                for             (uint64_t k2 = 0; k2 < ne02; k2++) {
                    for         (uint64_t i1 = 0; i1 < nr1;  i1++) {
                        for     (uint64_t k1 = 0; k1 < ne01; k1++) {
                            for (uint64_t i0 = 0; i0 < nr0;  i0++) {
                                copies.push_back({
                                    src_offset + (          k3)*nb03 + (          k2)*nb02 + (          k1)*nb01,
                                    dst_offset + (i3*ne03 + k3)*nb3  + (i2*ne02 + k2)*nb2  + (i1*ne01 + k1)*nb1 + (i0*ne00)*nb0,
                                    ne00*nb0,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    ggml_vk_sync_buffers(subctx);
    subctx->s->buffer.copyBuffer(src_buf->buffer, dst_buf->buffer, copies);

    GGML_UNUSED(ctx);
    GGML_UNUSED(src1);
}


static vk_pipeline ggml_vk_op_get_pipeline(ggml_backend_vk_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, ggml_tensor * dst, ggml_op op) {
    switch (op) {
    case GGML_OP_GET_ROWS:
        GGML_ASSERT(src1->type == GGML_TYPE_I32);
        if (dst->type == GGML_TYPE_F16) {
            return ctx->device->pipeline_get_rows[src0->type];
        }
        if (dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_get_rows_f32[src0->type];
        }
        return nullptr;
    case GGML_OP_ADD:
        if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_add_f32;
        }
        if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
            return ctx->device->pipeline_add_f16_f32_f16;
        }
        return nullptr;
    case GGML_OP_MUL:
        if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_mul_f32;
        }
        return nullptr;
    case GGML_OP_DIV:
        if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_div_f32;
        }
        return nullptr;
    case GGML_OP_CONCAT:
        if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_concat_f32;
        }
        if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
            return ctx->device->pipeline_concat_f16;
        }
        if (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_I32) {
            return ctx->device->pipeline_concat_i32;
        }
        return nullptr;
    case GGML_OP_UPSCALE:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_upscale_f32;
        }
        return nullptr;
    case GGML_OP_SCALE:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_scale_f32;
        }
        return nullptr;
    case GGML_OP_SQR:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_sqr_f32;
        }
        return nullptr;
    case GGML_OP_CLAMP:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_clamp_f32;
        }
        return nullptr;
    case GGML_OP_PAD:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_pad_f32;
        }
        return nullptr;
    case GGML_OP_CPY:
    case GGML_OP_CONT:
    case GGML_OP_DUP:
        return ggml_vk_get_cpy_pipeline(ctx, src0->type, dst->type);
    case GGML_OP_NORM:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_norm_f32;
        }
        return nullptr;
    case GGML_OP_GROUP_NORM:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_group_norm_f32;
        }
        return nullptr;
    case GGML_OP_RMS_NORM:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_rms_norm_f32;
        }
        return nullptr;
    case GGML_OP_UNARY:
        switch (ggml_get_unary_op(dst)) {
            case GGML_UNARY_OP_SILU:
                if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    return ctx->device->pipeline_silu_f32;
                }
                break;
            case GGML_UNARY_OP_GELU:
                if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    return ctx->device->pipeline_gelu_f32;
                }
                break;
            case GGML_UNARY_OP_GELU_QUICK:
                if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    return ctx->device->pipeline_gelu_quick_f32;
                }
                break;
            case GGML_UNARY_OP_RELU:
                if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    return ctx->device->pipeline_relu_f32;
                }
                break;
            case GGML_UNARY_OP_TANH:
                if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    return ctx->device->pipeline_tanh_f32;
                }
                break;
            default:
                break;
        }
        return nullptr;
    case GGML_OP_DIAG_MASK_INF:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_diag_mask_inf_f32;
        }
        return nullptr;
    case GGML_OP_SOFT_MAX:
        GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);

        if (src0->type == GGML_TYPE_F32 && (src1 == nullptr || src1->type == GGML_TYPE_F32) && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_soft_max_f32;
        }
        if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_soft_max_f32_f16;
        }
        return nullptr;
    case GGML_OP_ROPE:
        {
            const int mode = ((const int32_t *) dst->op_params)[2];
            const bool is_neox = mode & 2;

            if (is_neox) {
                if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    return ctx->device->pipeline_rope_neox_f32;
                }
                if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
                    return ctx->device->pipeline_rope_neox_f16;
                }
            } else {
                if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
                    return ctx->device->pipeline_rope_norm_f32;
                }
                if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
                    return ctx->device->pipeline_rope_norm_f16;
                }
            }
            return nullptr;
        }
    case GGML_OP_ARGSORT:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_I32) {
            return ctx->device->pipeline_argsort_f32;
        }
        return nullptr;
    case GGML_OP_SUM_ROWS:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_sum_rows_f32;
        }
        return nullptr;
    case GGML_OP_IM2COL:
        if (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_im2col_f32;
        }
        if (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
            return ctx->device->pipeline_im2col_f32_f16;
        }
        return nullptr;
    case GGML_OP_TIMESTEP_EMBEDDING:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_timestep_embedding_f32;
        }
        return nullptr;
    case GGML_OP_LEAKY_RELU:
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            return ctx->device->pipeline_leaky_relu_f32;
        }
        return nullptr;
    default:
        return nullptr;
    }

    GGML_UNUSED(src2);
}

static ggml_vk_func_t ggml_vk_op_get_func(ggml_op op) {
    switch(op) {
    case GGML_OP_REPEAT:
        return ggml_vk_op_repeat;
    default:
        return nullptr;
    }
}

static bool ggml_vk_op_supports_incontiguous(ggml_op op) {
    switch (op) {
    case GGML_OP_CPY:
    case GGML_OP_GET_ROWS:
    case GGML_OP_ADD:
    case GGML_OP_MUL:
    case GGML_OP_DIV:
    case GGML_OP_CONCAT:
    case GGML_OP_UPSCALE:
    case GGML_OP_SCALE:
    case GGML_OP_SQR:
    case GGML_OP_CLAMP:
    case GGML_OP_PAD:
        return true;
    default:
        return false;
    }
}

template<typename PC>
static void ggml_vk_op_f32(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, ggml_tensor * dst, ggml_op op, const PC&& pc) {
    VK_LOG_DEBUG("ggml_vk_op_f32((" << src0 << ", name=" << src0->name << ", type=" << src0->type << ", ne0=" << src0->ne[0] << ", ne1=" << src0->ne[1] << ", ne2=" << src0->ne[2] << ", ne3=" << src0->ne[3] << ", nb0=" << src0->nb[0] << ", nb1=" << src0->nb[1] << ", nb2=" << src0->nb[2] << ", nb3=" << src0->nb[3];
    if (src1 != nullptr) {
        std::cerr << "), (" << src1 << ", name=" << src1->name << ", type=" << src1->type << ", ne0=" << src1->ne[0] << ", ne1=" << src1->ne[1] << ", ne2=" << src1->ne[2] << ", ne3=" << src1->ne[3] << ", nb0=" << src1->nb[0] << ", nb1=" << src1->nb[1] << ", nb2=" << src1->nb[2] << ", nb3=" << src1->nb[3];
    }
    if (src2 != nullptr) {
        std::cerr << "), (" << src2 << ", name=" << src2->name << ", type=" << src2->type << ", ne0=" << src2->ne[0] << ", ne1=" << src2->ne[1] << ", ne2=" << src2->ne[2] << ", ne3=" << src2->ne[3] << ", nb0=" << src2->nb[0] << ", nb1=" << src2->nb[1] << ", nb2=" << src2->nb[2] << ", nb3=" << src2->nb[3];
    }
    std::cerr << "), (" << dst << ", name=" << dst->name << ", type=" << dst->type << ", ne0=" << dst->ne[0] << ", ne1=" << dst->ne[1] << ", ne2=" << dst->ne[2] << ", ne3=" << dst->ne[3] << ", nb0=" << dst->nb[0] << ", nb1=" << dst->nb[1] << ", nb2=" << dst->nb[2] << ", nb3=" << dst->nb[3] << "), " << ggml_op_name(op) << ")");
    GGML_ASSERT(op == GGML_OP_GET_ROWS || (!ggml_is_quantized(src0->type) && (src1 == nullptr || !ggml_is_quantized(src1->type))));  // NOLINT
    GGML_ASSERT(ggml_vk_op_supports_incontiguous(op) || ggml_vk_dim01_contiguous(src0));  // NOLINT
    GGML_ASSERT(dst->extra != nullptr);
    const uint64_t ne00 = src0->ne[0];
    const uint64_t ne01 = src0->ne[1];
    const uint64_t ne02 = src0->ne[2];
    const uint64_t ne03 = src0->ne[3];
    const uint64_t ne0 = ne00 * ne01;

    const bool use_src1 = src1 != nullptr;
    const uint64_t ne10 = use_src1 ? src1->ne[0] : 0;
    const uint64_t ne11 = use_src1 ? src1->ne[1] : 0;
    const uint64_t ne12 = use_src1 ? src1->ne[2] : 0;
    const uint64_t ne13 = use_src1 ? src1->ne[3] : 0;
    const uint64_t ne1 = ne10 * ne11;
    // const uint64_t nb10 = use_src1 ? src1->nb[0] : 0;

    const bool use_src2 = src2 != nullptr;
    const uint64_t ne20 = use_src2 ? src2->ne[0] : 0;
    const uint64_t ne21 = use_src2 ? src2->ne[1] : 0;
    const uint64_t ne22 = use_src2 ? src2->ne[2] : 0;
    const uint64_t ne23 = use_src2 ? src2->ne[3] : 0;
    const uint64_t ne2 = ne20 * ne21;

    const uint64_t ned0 = dst->ne[0];
    const uint64_t ned1 = dst->ne[1];
    const uint64_t ned2 = dst->ne[2];
    const uint64_t ned3 = dst->ne[3];
    const uint64_t ned = ned0 * ned1;

    vk_pipeline pipeline = ggml_vk_op_get_pipeline(ctx, src0, src1, src2, dst, op);
    ggml_vk_func_t op_func;

    if (pipeline == nullptr) {
        op_func = ggml_vk_op_get_func(op);
        if (op_func == nullptr) {
            std::cerr << "ggml_vulkan: Error: Missing op: " << ggml_op_name(op) << " for " << ggml_type_name(src0->type);
            if (src1 != nullptr) {
                std::cerr << " and " << ggml_type_name(src1->type);
            }
            std::cerr << " to " << ggml_type_name(dst->type) << std::endl;
            GGML_ABORT("fatal error");
        }

        op_func(ctx, subctx, src0, src1, dst);
        return;
    }

    const bool op_supports_incontiguous = ggml_vk_op_supports_incontiguous(op);

    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) dst->extra;
    ggml_tensor_extra_gpu * extra_src0 = (ggml_tensor_extra_gpu *) src0->extra;
    ggml_tensor_extra_gpu * extra_src1 = use_src1 ? (ggml_tensor_extra_gpu *) src1->extra : nullptr;
    ggml_tensor_extra_gpu * extra_src2 = use_src2 ? (ggml_tensor_extra_gpu *) src2->extra : nullptr;

    vk_buffer d_X = nullptr;
    size_t x_buf_offset = 0;
    vk_buffer d_Y = nullptr;
    size_t y_buf_offset = 0;
    vk_buffer d_Z = nullptr;
    size_t z_buf_offset = 0;

    bool src0_uma = false;
    bool src1_uma = false;
    bool src2_uma = false;

    if (ctx->device->uma) {
        ggml_vk_host_get(ctx->device, src0->data, d_X, x_buf_offset);
        src0_uma = d_X != nullptr;
        if (use_src1) {
            ggml_vk_host_get(ctx->device, src1->data, d_Y, y_buf_offset);
            src1_uma = d_Y != nullptr;
        }
        if (use_src2) {
            ggml_vk_host_get(ctx->device, src2->data, d_Z, z_buf_offset);
            src2_uma = d_Z != nullptr;
        }
    }

    uint64_t x_sz = ggml_type_size(src0->type)/ggml_blck_size(src0->type) * ne0;
    uint64_t y_sz = use_src1 ? ggml_type_size(src1->type) * ne1 : 0;
    uint64_t z_sz = use_src2 ? ggml_type_size(src2->type) * ne2 : 0;
    uint64_t d_sz = ggml_type_size(dst->type) * ned;

    vk_buffer d_D = extra->buffer_gpu.lock();

    // Workaround for tiny tensor inputs on ROPE
    if (op == GGML_OP_ROPE && use_src1 && y_sz > d_D->size) {
        y_sz = VK_WHOLE_SIZE;
    }

    GGML_ASSERT(d_D != nullptr);
    uint64_t d_buf_offset = ((extra->offset + dst->view_offs) / ctx->device->properties.limits.minStorageBufferOffsetAlignment) * ctx->device->properties.limits.minStorageBufferOffsetAlignment;
    GGML_ASSERT(d_buf_offset == extra->offset || op == GGML_OP_CPY);  // NOLINT
    if(!src0_uma) {
        d_X = extra_src0->buffer_gpu.lock();
        x_buf_offset = extra_src0->offset + src0->view_offs;
        GGML_ASSERT(d_X != nullptr);
    }
    if (use_src1 && !src1_uma) {
        d_Y = extra_src1->buffer_gpu.lock();
        y_buf_offset = extra_src1->offset + src1->view_offs;
        GGML_ASSERT(d_Y != nullptr);
    }
    if (use_src2 && !src2_uma) {
        d_Z = extra_src2->buffer_gpu.lock();
        z_buf_offset = extra_src2->offset + src2->view_offs;
        GGML_ASSERT(d_Z != nullptr);
    }

    if (op_supports_incontiguous) {
        x_sz = ggml_nbytes(src0);
        y_sz = use_src1 ? ggml_nbytes(src1) : 0;
        z_sz = use_src2 ? ggml_nbytes(src2) : 0;
        d_sz = ggml_nbytes(dst);

        if (x_buf_offset + x_sz >= d_X->size) {
            x_sz = VK_WHOLE_SIZE;
        }
        if (use_src1 && y_buf_offset + y_sz >= d_Y->size) {
            y_sz = VK_WHOLE_SIZE;
        }
        if (use_src2 && z_buf_offset + z_sz >= d_Z->size) {
            z_sz = VK_WHOLE_SIZE;
        }
        if (d_buf_offset + d_sz >= d_D->size) {
            d_sz = VK_WHOLE_SIZE;
        }
    }

    std::array<uint32_t, 3> elements;

    // Single call if dimension 2 is contiguous
    if (op_supports_incontiguous || (ggml_is_contiguous(src0) && (src1 == nullptr || ggml_is_contiguous(src1)))) {
        ggml_pipeline_allocate_descriptor_sets(ctx->device, pipeline, 1);

        switch (op) {
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_SUM_ROWS:
            {
                const uint32_t nr = ggml_nrows(src0);
                if (nr > 262144) {
                    elements = { 512, 512, CEIL_DIV(nr, 262144) };
                } else if (nr > 512) {
                    elements = { 512, CEIL_DIV(nr, 512), 1 };
                } else {
                    elements = { nr, 1, 1 };
                }
            } break;
        case GGML_OP_GROUP_NORM:
            {
                const uint32_t num_groups = dst->op_params[0];
                elements = { num_groups * (uint32_t)src0->ne[3], 1, 1 };
            } break;
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_ROPE:
            elements = { (uint32_t)ggml_nrows(src0), (uint32_t)ne00, 1 };
            break;
        case GGML_OP_GET_ROWS:
            elements = { (uint32_t)ne00, (uint32_t)ne10, (uint32_t)(ne11 * ne12) };
            break;
        case GGML_OP_ARGSORT:
            elements = { (uint32_t)ne00, (uint32_t)ggml_nrows(src0), 1 };
            break;
        case GGML_OP_IM2COL:
            {
                const bool is_2D = dst->op_params[6] == 1;

                const uint32_t IC = src1->ne[is_2D ? 2 : 1];

                const uint32_t KH = is_2D ? src0->ne[1] : 1;
                const uint32_t KW =         src0->ne[0];

                const uint32_t OH = is_2D ? dst->ne[2] : 1;
                const uint32_t OW =         dst->ne[1];

                const uint32_t batch = src1->ne[3];

                elements = { OW * KW * KH, OH, batch * IC };
            } break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            {
                const uint32_t dim = dst->op_params[0];
                uint32_t half_ceil = (dim + 1) / 2;
                elements = { half_ceil, (uint32_t)src0->ne[0], 1 };
            } break;
        case GGML_OP_ADD:
        case GGML_OP_DIV:
        case GGML_OP_MUL:
        case GGML_OP_SCALE:
        case GGML_OP_SQR:
        case GGML_OP_CLAMP:
        case GGML_OP_PAD:
        case GGML_OP_CPY:
        case GGML_OP_CONCAT:
        case GGML_OP_UPSCALE:
        case GGML_OP_UNARY:
            {
                const uint32_t ne = ggml_nelements(dst);
                if (ne > 262144) {
                    elements = { 512, 512, CEIL_DIV(ne, 262144) };
                } else if (ne > 512) {
                    elements = { 512, CEIL_DIV(ne, 512), 1 };
                } else {
                    elements = { ne, 1, 1 };
                }
            } break;
        default:
            elements = { (uint32_t)ggml_nelements(src0), 1, 1 };
            break;
        }

        if (!op_supports_incontiguous) {
            if (x_sz != VK_WHOLE_SIZE) {
                x_sz *= ne02 * ne03;
            }
            if (use_src1 && y_sz != VK_WHOLE_SIZE) {
                y_sz *= ne12 * ne13;
            }
            if (use_src2 && z_sz != VK_WHOLE_SIZE) {
                z_sz *= ne22 * ne23;
            }
            if (d_sz != VK_WHOLE_SIZE) {
                d_sz *= ned2 * ned3;
            }
        }

        if (op == GGML_OP_SOFT_MAX) {
            // Empty src1 is possible in soft_max, but the shader needs a buffer
            vk_subbuffer subbuf_y;
            if (use_src1) {
                subbuf_y = { d_Y, y_buf_offset, y_sz };
            } else {
                subbuf_y = { d_X, 0, x_sz };
            }

            ggml_vk_sync_buffers(subctx);
            ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { { d_X, x_buf_offset, x_sz }, subbuf_y, { d_D, d_buf_offset, d_sz } }, sizeof(PC), &pc, elements);
        } else if (op == GGML_OP_ROPE) {
            // Empty src2 is possible in rope, but the shader needs a buffer
            vk_subbuffer subbuf_z;
            if (use_src2) {
                subbuf_z = { d_Z, z_buf_offset, z_sz };
            } else {
                subbuf_z = { d_X, 0, x_sz };
            }

            ggml_vk_sync_buffers(subctx);
            ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { { d_X, x_buf_offset, x_sz }, { d_Y, y_buf_offset, y_sz }, subbuf_z, { d_D, d_buf_offset, d_sz } }, sizeof(PC), &pc, elements);
        } else if (op == GGML_OP_IM2COL) {
            // im2col uses only src1 and dst buffers
            ggml_vk_sync_buffers(subctx);
            ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { { d_Y, y_buf_offset, y_sz }, { d_D, d_buf_offset, d_sz } }, sizeof(PC), &pc, elements);
        } else if (use_src2) {
            ggml_vk_sync_buffers(subctx);
            ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { { d_X, x_buf_offset, x_sz }, { d_Y, y_buf_offset, y_sz }, { d_Z, z_buf_offset, z_sz }, { d_D, d_buf_offset, d_sz } }, sizeof(PC), &pc, elements);
        } else if (use_src1) {
            ggml_vk_sync_buffers(subctx);
            ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { { d_X, x_buf_offset, x_sz }, { d_Y, y_buf_offset, y_sz }, { d_D, d_buf_offset, d_sz } }, sizeof(PC), &pc, elements);
        } else {
            ggml_vk_sync_buffers(subctx);
            ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { { d_X, x_buf_offset, x_sz }, { d_D, d_buf_offset, d_sz } }, sizeof(PC), &pc, elements);
        }
    } else {
        GGML_ASSERT(op != GGML_OP_SOFT_MAX);
        GGML_ASSERT(op != GGML_OP_ARGSORT);
        GGML_ASSERT(!use_src2);

        ggml_pipeline_allocate_descriptor_sets(ctx->device, pipeline, ne02 * ne03);

        switch (op) {
        case GGML_OP_NORM:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_RMS_NORM:
            elements = { (uint32_t)ne01, 1, 1 };
            break;
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_ROPE:
            elements = { (uint32_t)ne01, (uint32_t)ne00, 1 };
            break;
        case GGML_OP_GET_ROWS:
            elements = {  (uint32_t)ne00, (uint32_t)ne10, (uint32_t)(ne11 * ne12) };
            break;
        default:
            elements = { (uint32_t)ne0, 1, 1 };
            break;
        }

        for (uint64_t i03 = 0; i03 < ne03; i03++) {
            for (uint64_t i02 = 0; i02 < ne02; i02++) {
                const uint32_t it_idx0 = (i03 * ne02 + i02);
                const uint32_t it_idx1 = use_src1 ? ((i03 % ne13) * ne12 + (i02 % ne12)) : 0;
                const uint32_t x_offset = x_sz * it_idx0;
                const uint32_t y_offset = y_sz * it_idx1;
                const uint32_t d_offset = d_sz * it_idx0;

                if (use_src1) {
                    ggml_vk_sync_buffers(subctx);
                    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { { d_X, x_buf_offset + x_offset, x_sz }, { d_Y, y_buf_offset + y_offset, y_sz }, { d_D, d_buf_offset + d_offset, d_sz } }, sizeof(PC), &pc, elements);
                } else {
                    ggml_vk_sync_buffers(subctx);
                    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { { d_X, x_buf_offset + x_offset, x_sz }, { d_D, d_buf_offset + d_offset, d_sz } }, sizeof(PC), &pc, elements);
                }
            }
        }
    }
}

static void ggml_vk_repeat(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32<vk_op_push_constants>(ctx, subctx, src0, nullptr, nullptr, dst, GGML_OP_REPEAT, {});
}

static void ggml_vk_get_rows(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const uint32_t src0_type_size = ggml_type_size(src0->type);
    const uint32_t src1_type_size = ggml_type_size(src1->type);
    const uint32_t dst_type_size = ggml_type_size(dst->type);

    ggml_vk_op_f32<vk_op_binary_push_constants>(ctx, subctx, src0, src1, nullptr, dst, GGML_OP_GET_ROWS, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_size, (uint32_t)src0->nb[1] / src0_type_size, (uint32_t)src0->nb[2] / src0_type_size, (uint32_t)src0->nb[3] / src0_type_size,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_size, (uint32_t)src1->nb[1] / src1_type_size, (uint32_t)src1->nb[2] / src1_type_size, (uint32_t)src1->nb[3] / src1_type_size,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_size, (uint32_t) dst->nb[1] /  dst_type_size, (uint32_t) dst->nb[2] /  dst_type_size, (uint32_t) dst->nb[3] /  dst_type_size,
        0,
        0.0f, 0.0f, 0,
    });
}

static void ggml_vk_add(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const uint32_t src0_type_size = ggml_type_size(src0->type);
    const uint32_t src1_type_size = ggml_type_size(src1->type);
    const uint32_t dst_type_size = ggml_type_size(dst->type);

    ggml_vk_op_f32<vk_op_binary_push_constants>(ctx, subctx, src0, src1, nullptr, dst, GGML_OP_ADD, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_size, (uint32_t)src0->nb[1] / src0_type_size, (uint32_t)src0->nb[2] / src0_type_size, (uint32_t)src0->nb[3] / src0_type_size,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_size, (uint32_t)src1->nb[1] / src1_type_size, (uint32_t)src1->nb[2] / src1_type_size, (uint32_t)src1->nb[3] / src1_type_size,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_size, (uint32_t) dst->nb[1] /  dst_type_size, (uint32_t) dst->nb[2] /  dst_type_size, (uint32_t) dst->nb[3] /  dst_type_size,
        0,
        0.0f, 0.0f, 0,
    });
}

static void ggml_vk_mul(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const uint32_t src0_type_size = ggml_type_size(src0->type);
    const uint32_t src1_type_size = ggml_type_size(src1->type);
    const uint32_t dst_type_size = ggml_type_size(dst->type);

    ggml_vk_op_f32<vk_op_binary_push_constants>(ctx, subctx, src0, src1, nullptr, dst, GGML_OP_MUL, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_size, (uint32_t)src0->nb[1] / src0_type_size, (uint32_t)src0->nb[2] / src0_type_size, (uint32_t)src0->nb[3] / src0_type_size,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_size, (uint32_t)src1->nb[1] / src1_type_size, (uint32_t)src1->nb[2] / src1_type_size, (uint32_t)src1->nb[3] / src1_type_size,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_size, (uint32_t) dst->nb[1] /  dst_type_size, (uint32_t) dst->nb[2] /  dst_type_size, (uint32_t) dst->nb[3] /  dst_type_size,
        0,
        0.0f, 0.0f, 0,
    });
}

static void ggml_vk_div(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const uint32_t src0_type_size = ggml_type_size(src0->type);
    const uint32_t src1_type_size = ggml_type_size(src1->type);
    const uint32_t dst_type_size = ggml_type_size(dst->type);

    ggml_vk_op_f32<vk_op_binary_push_constants>(ctx, subctx, src0, src1, nullptr, dst, GGML_OP_DIV, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_size, (uint32_t)src0->nb[1] / src0_type_size, (uint32_t)src0->nb[2] / src0_type_size, (uint32_t)src0->nb[3] / src0_type_size,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_size, (uint32_t)src1->nb[1] / src1_type_size, (uint32_t)src1->nb[2] / src1_type_size, (uint32_t)src1->nb[3] / src1_type_size,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_size, (uint32_t) dst->nb[1] /  dst_type_size, (uint32_t) dst->nb[2] /  dst_type_size, (uint32_t) dst->nb[3] /  dst_type_size,
        0,
        0.0f, 0.0f, 0,
    });
}

static void ggml_vk_concat(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    int * op_params = (int *)dst->op_params;

    const uint32_t src0_type_size = ggml_type_size(src0->type);
    const uint32_t src1_type_size = ggml_type_size(src1->type);
    const uint32_t dst_type_size = ggml_type_size(dst->type);

    ggml_vk_op_f32<vk_op_binary_push_constants>(ctx, subctx, src0, src1, nullptr, dst, GGML_OP_CONCAT, {
        (uint32_t)ggml_nelements(dst),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2],(uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_size, (uint32_t)src0->nb[1] / src0_type_size, (uint32_t)src0->nb[2] / src0_type_size, (uint32_t)src0->nb[3] / src0_type_size,
        (uint32_t)src1->ne[0], (uint32_t)src1->ne[1], (uint32_t)src1->ne[2],(uint32_t)src1->ne[3], (uint32_t)src1->nb[0] / src1_type_size, (uint32_t)src1->nb[1] / src1_type_size, (uint32_t)src1->nb[2] / src1_type_size, (uint32_t)src1->nb[3] / src1_type_size,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2],(uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_size, (uint32_t) dst->nb[1] /  dst_type_size, (uint32_t) dst->nb[2] /  dst_type_size, (uint32_t) dst->nb[3] /  dst_type_size,
        0,
        0.0f, 0.0f, op_params[0],
    });
}

static void ggml_vk_upscale(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    const uint32_t src0_type_size = ggml_type_size(src0->type);

    const float sf0 = (float)dst->ne[0] / src0->ne[0];
    const float sf1 = (float)dst->ne[1] / src0->ne[1];
    const float sf2 = (float)dst->ne[2] / src0->ne[2];
    const float sf3 = (float)dst->ne[3] / src0->ne[3];

    ggml_vk_op_f32<vk_op_upscale_push_constants>(ctx, subctx, src0, nullptr, nullptr, dst, GGML_OP_UPSCALE, {
        (uint32_t)ggml_nelements(dst), 0,
        (uint32_t)src0->nb[0] / src0_type_size, (uint32_t)src0->nb[1] / src0_type_size, (uint32_t)src0->nb[2] / src0_type_size, (uint32_t)src0->nb[3] / src0_type_size,
        (uint32_t)dst->ne[0], (uint32_t)dst->ne[1], (uint32_t)dst->ne[2],(uint32_t)dst->ne[3],
        sf0, sf1, sf2, sf3,
    });
}

static void ggml_vk_scale(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    float * op_params = (float *)dst->op_params;
    const uint32_t src0_type_size = ggml_type_size(src0->type);
    const uint32_t dst_type_size = ggml_type_size(dst->type);

    ggml_vk_op_f32<vk_op_unary_push_constants>(ctx, subctx, src0, nullptr, nullptr, dst, GGML_OP_SCALE, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2], (uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_size, (uint32_t)src0->nb[1] / src0_type_size, (uint32_t)src0->nb[2] / src0_type_size, (uint32_t)src0->nb[3] / src0_type_size,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2], (uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_size, (uint32_t) dst->nb[1] /  dst_type_size, (uint32_t) dst->nb[2] /  dst_type_size, (uint32_t) dst->nb[3] /  dst_type_size,
        0,
        op_params[0], 0.0f
    });
}

static void ggml_vk_sqr(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    const uint32_t src0_type_size = ggml_type_size(src0->type);
    const uint32_t dst_type_size = ggml_type_size(dst->type);

    ggml_vk_op_f32<vk_op_unary_push_constants>(ctx, subctx, src0, nullptr, nullptr, dst, GGML_OP_SQR, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2], (uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_size, (uint32_t)src0->nb[1] / src0_type_size, (uint32_t)src0->nb[2] / src0_type_size, (uint32_t)src0->nb[3] / src0_type_size,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2], (uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_size, (uint32_t) dst->nb[1] /  dst_type_size, (uint32_t) dst->nb[2] /  dst_type_size, (uint32_t) dst->nb[3] /  dst_type_size,
        0,
        0.0f, 0.0f,
    });
}

static void ggml_vk_clamp(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    float * op_params = (float *)dst->op_params;
    const uint32_t src0_type_size = ggml_type_size(src0->type);
    const uint32_t dst_type_size = ggml_type_size(dst->type);

    ggml_vk_op_f32<vk_op_unary_push_constants>(ctx, subctx, src0, nullptr, nullptr, dst, GGML_OP_CLAMP, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2], (uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_size, (uint32_t)src0->nb[1] / src0_type_size, (uint32_t)src0->nb[2] / src0_type_size, (uint32_t)src0->nb[3] / src0_type_size,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2], (uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_size, (uint32_t) dst->nb[1] /  dst_type_size, (uint32_t) dst->nb[2] /  dst_type_size, (uint32_t) dst->nb[3] /  dst_type_size,
        0,
        op_params[0], op_params[1],
    });
}

static void ggml_vk_pad(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    const uint32_t src0_type_size = ggml_type_size(src0->type);
    const uint32_t dst_type_size = ggml_type_size(dst->type);

    ggml_vk_op_f32<vk_op_unary_push_constants>(ctx, subctx, src0, nullptr, nullptr, dst, GGML_OP_PAD, {
        (uint32_t)ggml_nelements(dst),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2], (uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_size, (uint32_t)src0->nb[1] / src0_type_size, (uint32_t)src0->nb[2] / src0_type_size, (uint32_t)src0->nb[3] / src0_type_size,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2], (uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_size, (uint32_t) dst->nb[1] /  dst_type_size, (uint32_t) dst->nb[2] /  dst_type_size, (uint32_t) dst->nb[3] /  dst_type_size,
        0,
        0.0f, 0.0f,
    });
}

static void ggml_vk_cpy(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) dst->extra;
    const uint32_t src0_type_size = ggml_type_size(src0->type);
    const uint32_t dst_type_size = ggml_type_size(dst->type);
    const uint32_t d_offset = ((extra->offset + dst->view_offs) % ctx->device->properties.limits.minStorageBufferOffsetAlignment) / dst_type_size;

    ggml_vk_op_f32<vk_op_unary_push_constants>(ctx, subctx, src0, nullptr, nullptr, dst, GGML_OP_CPY, {
        (uint32_t)ggml_nelements(src0),
        (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], (uint32_t)src0->ne[2], (uint32_t)src0->ne[3], (uint32_t)src0->nb[0] / src0_type_size, (uint32_t)src0->nb[1] / src0_type_size, (uint32_t)src0->nb[2] / src0_type_size, (uint32_t)src0->nb[3] / src0_type_size,
        (uint32_t) dst->ne[0], (uint32_t) dst->ne[1], (uint32_t) dst->ne[2], (uint32_t) dst->ne[3], (uint32_t) dst->nb[0] /  dst_type_size, (uint32_t) dst->nb[1] /  dst_type_size, (uint32_t) dst->nb[2] /  dst_type_size, (uint32_t) dst->nb[3] /  dst_type_size,
        d_offset,
        0.0f, 0.0f,
    });
}

static void ggml_vk_norm(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    float * op_params = (float *)dst->op_params;

    ggml_vk_op_f32<vk_op_push_constants>(ctx, subctx, src0, nullptr, nullptr, dst, GGML_OP_NORM, { (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], op_params[0], 0.0f });
}

static void ggml_vk_group_norm(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    int * op_params = (int *)dst->op_params;

    uint32_t num_groups = op_params[0];
    uint32_t group_size = src0->ne[0] * src0->ne[1] * ((src0->ne[2] + num_groups - 1) / num_groups);
    static const float eps = 1e-6f;

    ggml_vk_op_f32<vk_op_push_constants>(ctx, subctx, src0, nullptr, nullptr, dst, GGML_OP_GROUP_NORM, { group_size, 0, eps, 0.0f });
}

static void ggml_vk_rms_norm(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    float * op_params = (float *)dst->op_params;
    ggml_vk_op_f32<vk_op_push_constants>(ctx, subctx, src0, nullptr, nullptr, dst, GGML_OP_RMS_NORM, { (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], op_params[0], 0.0f });
}

static void ggml_vk_unary(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32<vk_op_push_constants>(ctx, subctx, src0, nullptr, nullptr, dst, GGML_OP_UNARY, { (uint32_t)ggml_nelements(src0), 0, 0.0f, 0.0f });
}

static void ggml_vk_diag_mask_inf(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    int32_t * op_params = (int32_t *)dst->op_params;
    ggml_vk_op_f32<vk_op_diag_mask_push_constants>(ctx, subctx, src0, nullptr, nullptr, dst, GGML_OP_DIAG_MASK_INF, { (uint32_t)src0->ne[0], (uint32_t)src0->ne[1], op_params[0] });
}

static void ggml_vk_soft_max(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    float * op_params = (float *)dst->op_params;

    float scale = op_params[0];
    float max_bias = op_params[1];

    const uint32_t ncols =   (uint32_t)src0->ne[0];
    const uint32_t nrows_x = (uint32_t)ggml_nrows(src0);
    const uint32_t nrows_y = (uint32_t)src0->ne[1];

    const uint32_t n_head_kv   = nrows_x/nrows_y;
    const uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head_kv));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    ggml_vk_op_f32<vk_op_soft_max_push_constants>(ctx, subctx, src0, src1, nullptr, dst, GGML_OP_SOFT_MAX, {
        ncols,
        src1 != nullptr ? nrows_y : (uint32_t)0,
        scale, max_bias,
        m0, m1,
        n_head_log2,
    });
}

static void ggml_vk_rope(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * src2, ggml_tensor * dst) {
    const int n_dims        = ((int32_t *) dst->op_params)[1];
    // const int mode          = ((int32_t *) dst->op_params)[2];
    // const int n_ctx         = ((int32_t *) dst->op_params)[3];
    const int n_ctx_orig    = ((int32_t *) dst->op_params)[4];
    const float freq_base   = ((float *)   dst->op_params)[5];
    const float freq_scale  = ((float *)   dst->op_params)[6];
    const float ext_factor  = ((float *)   dst->op_params)[7];
    const float attn_factor = ((float *)   dst->op_params)[8];
    const float beta_fast   = ((float *)   dst->op_params)[9];
    const float beta_slow   = ((float *)   dst->op_params)[10];

    float corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    ggml_vk_op_f32<vk_op_rope_push_constants>(ctx, subctx, src0, src1, src2, dst, GGML_OP_ROPE, {
        (uint32_t)src0->ne[0], (uint32_t)n_dims, freq_scale, (uint32_t)src0->ne[1],
        freq_base, ext_factor, attn_factor, {corr_dims[0], corr_dims[1]}, theta_scale,
        src2 != nullptr,
    });
}

static void ggml_vk_argsort(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    int32_t * op_params = (int32_t *)dst->op_params;

    uint32_t ncols = src0->ne[0];

    uint32_t ncols_pad = 1;
    while (ncols_pad < ncols) {
        ncols_pad *= 2;
    }

    GGML_ASSERT(ncols_pad <= 1024);

    ggml_vk_op_f32<vk_op_argsort_push_constants>(ctx, subctx, src0, nullptr, nullptr, dst, GGML_OP_ARGSORT, {
        ncols,
        ncols_pad,
        op_params[0],
    });
}

static void ggml_vk_sum_rows(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    ggml_vk_op_f32<vk_op_push_constants>(ctx, subctx, src0, nullptr, nullptr, dst, GGML_OP_SUM_ROWS, { (uint32_t)src0->ne[0], 0, 0.0f, 0.0f });
}

static void ggml_vk_im2col(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int32_t s0 = dst->op_params[0];
    const int32_t s1 = dst->op_params[1];
    const int32_t p0 = dst->op_params[2];
    const int32_t p1 = dst->op_params[3];
    const int32_t d0 = dst->op_params[4];
    const int32_t d1 = dst->op_params[5];

    const bool is_2D = dst->op_params[6] == 1;

    const uint32_t IC = src1->ne[is_2D ? 2 : 1];
    const uint32_t IH = is_2D ? src1->ne[1] : 1;
    const uint32_t IW =         src1->ne[0];

    const uint32_t KH = is_2D ? src0->ne[1] : 1;
    const uint32_t KW =         src0->ne[0];

    const uint32_t OH = is_2D ? dst->ne[2] : 1;
    const uint32_t OW =         dst->ne[1];

    const uint32_t offset_delta = src1->nb[is_2D ? 2 : 1] / 4; // nb is byte offset, src is type float32
    const uint32_t batch_offset = src1->nb[3] / 4; // nb is byte offset, src is type float32

    const uint32_t pelements = OW * KW * KH;

    ggml_vk_op_f32<vk_op_im2col_push_constants>(ctx, subctx, src0, src1, nullptr, dst, GGML_OP_IM2COL, {
        batch_offset, offset_delta,
        IC, IW, IH, OW, OH, KW, KH,
        pelements,
        IC * KH * KW,
        s0, s1, p0, p1, d0, d1,
    });
}

static void ggml_vk_timestep_embedding(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    const uint32_t dim = dst->op_params[0];
    const uint32_t max_period = dst->op_params[1];
    const uint32_t nb1 = dst->nb[1] / ggml_type_size(dst->type);

    ggml_vk_op_f32<vk_op_timestep_embedding_push_constants>(ctx, subctx, src0, nullptr, nullptr, dst, GGML_OP_TIMESTEP_EMBEDDING, {
        nb1, dim, max_period,
    });
}

static void ggml_vk_leaky_relu(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src0, ggml_tensor * dst) {
    const float * op_params = (const float *)dst->op_params;
    ggml_vk_op_f32<vk_op_push_constants>(ctx, subctx, src0, nullptr, nullptr, dst, GGML_OP_LEAKY_RELU, { (uint32_t)ggml_nelements(src0), 0, op_params[0], 0.0f });
}

#ifdef GGML_VULKAN_RUN_TESTS
static void ggml_vk_print_matrix_area(const void * data, ggml_type type, int ne0, int ne1, int i0, int i1, int i2) {
    if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16) {
        return;
    }
    i0 = std::max(i0, 5);
    i1 = std::max(i1, 5);
    i2 = std::max(i2, 0);
    fprintf(stderr, "         ");
    for (int idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
        fprintf(stderr, "%7d ", idx1);
    }
    fprintf(stderr, "\n");
    for (int idx0 = i0 - 5; idx0 < i0 + 5; idx0++) {
        fprintf(stderr, "%7d: ", idx0);
        for (int idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
            if (idx0 >= 0 && idx0 < ne0 && idx1 >= 0 && idx1 < ne1) {
                float val;
                if (type == GGML_TYPE_F32) {
                    val = *((const float *) data + i2*ne1*ne0 + idx1*ne0 + idx0);
                } else if (type == GGML_TYPE_F16) {
                    val = ggml_fp16_to_fp32(*((const ggml_fp16_t *) data + i2*ne1*ne0 + idx1*ne0 + idx0));
                } else {
                    GGML_ABORT("fatal error");
                }
                fprintf(stderr, "% 7.2f ", val);
            } else {
                fprintf(stderr, "        ");
            }
        }
        fprintf(stderr, "\n");
    }
}

template <typename X_TYPE, typename Y_TYPE>
static void ggml_vk_test_matmul(ggml_backend_vk_context * ctx, size_t m, size_t n, size_t k, size_t batch, size_t num_it, int split_k, int shader_size) {
    VK_LOG_DEBUG("ggml_vk_test_matmul(" << m << ", " << n << ", " << k << ", " << batch << ", " << num_it << ", " << split_k << ", " << shader_size << ")");
    const size_t x_ne = m * k * batch;
    const size_t y_ne = k * n * batch;
    const size_t d_ne = m * n * batch;

    vk_pipeline p;
    std::string shname;
    if (shader_size == 0) {
        if (std::is_same<float, X_TYPE>() && std::is_same<float, Y_TYPE>()) {
            p = ctx->device->pipeline_matmul_f32->a_s;
            shname = "F32_ALIGNED_S";
        } else if (std::is_same<float, X_TYPE>() && std::is_same<ggml_fp16_t, Y_TYPE>()) {
            p = ctx->device->pipeline_matmul_f32_f16->a_s;
            shname = "F32_F16_ALIGNED_S";
        } else if (std::is_same<ggml_fp16_t, X_TYPE>() && std::is_same<float, Y_TYPE>()) {
            p = ctx->device->pipeline_matmul_f16_f32->a_s;
            shname = "F16_F32_ALIGNED_S";
        } else if (std::is_same<ggml_fp16_t, X_TYPE>() && std::is_same<ggml_fp16_t, Y_TYPE>()) {
            p = ctx->device->pipeline_matmul_f16->a_s;
            shname = "F16_ALIGNED_S";
        } else {
            GGML_ABORT("fatal error");
        }
    } else if (shader_size == 1) {
        if (std::is_same<float, X_TYPE>() && std::is_same<float, Y_TYPE>()) {
            p = ctx->device->pipeline_matmul_f32->a_m;
            shname = "F32_ALIGNED_M";
        } else if (std::is_same<float, X_TYPE>() && std::is_same<ggml_fp16_t, Y_TYPE>()) {
            p = ctx->device->pipeline_matmul_f32_f16->a_m;
            shname = "F32_F16_ALIGNED_M";
        } else if (std::is_same<ggml_fp16_t, X_TYPE>() && std::is_same<float, Y_TYPE>()) {
            p = ctx->device->pipeline_matmul_f16_f32->a_m;
            shname = "F16_F32_ALIGNED_M";
        } else if (std::is_same<ggml_fp16_t, X_TYPE>() && std::is_same<ggml_fp16_t, Y_TYPE>()) {
            p = ctx->device->pipeline_matmul_f16->a_m;
            shname = "F16_ALIGNED_M";
        } else {
            GGML_ABORT("fatal error");
        }
    } else if (shader_size == 2) {
        if (std::is_same<float, X_TYPE>() && std::is_same<float, Y_TYPE>()) {
            p = ctx->device->pipeline_matmul_f32->a_l;
            shname = "F32_ALIGNED_L";
        } else if (std::is_same<float, X_TYPE>() && std::is_same<ggml_fp16_t, Y_TYPE>()) {
            p = ctx->device->pipeline_matmul_f32_f16->a_l;
            shname = "F32_F16_ALIGNED_L";
        } else if (std::is_same<ggml_fp16_t, X_TYPE>() && std::is_same<float, Y_TYPE>()) {
            p = ctx->device->pipeline_matmul_f16_f32->a_l;
            shname = "F16_F32_ALIGNED_L";
        } else if (std::is_same<ggml_fp16_t, X_TYPE>() && std::is_same<ggml_fp16_t, Y_TYPE>()) {
            p = ctx->device->pipeline_matmul_f16->a_l;
            shname = "F16_ALIGNED_L";
        } else {
            GGML_ABORT("fatal error");
        }
    } else {
        GGML_ASSERT(0);
    }

    const size_t kpad = ggml_vk_align_size(k, p->align);

    if (k != kpad) {
        if (shader_size == 0) {
            if (std::is_same<float, X_TYPE>() && std::is_same<float, Y_TYPE>()) {
                p = ctx->device->pipeline_matmul_f32->s;
                shname = "F32_S";
            } else if (std::is_same<float, X_TYPE>() && std::is_same<ggml_fp16_t, Y_TYPE>()) {
                p = ctx->device->pipeline_matmul_f32_f16->s;
                shname = "F32_F16_S";
            } else if (std::is_same<ggml_fp16_t, X_TYPE>() && std::is_same<float, Y_TYPE>()) {
                p = ctx->device->pipeline_matmul_f16_f32->s;
                shname = "F16_F32_S";
            } else if (std::is_same<ggml_fp16_t, X_TYPE>() && std::is_same<ggml_fp16_t, Y_TYPE>()) {
                p = ctx->device->pipeline_matmul_f16->s;
                shname = "F16_S";
            }
        } else if (shader_size == 1) {
            if (std::is_same<float, X_TYPE>() && std::is_same<float, Y_TYPE>()) {
                p = ctx->device->pipeline_matmul_f32->m;
                shname = "F32_M";
            } else if (std::is_same<float, X_TYPE>() && std::is_same<ggml_fp16_t, Y_TYPE>()) {
                p = ctx->device->pipeline_matmul_f32_f16->m;
                shname = "F32_F16_M";
            } else if (std::is_same<ggml_fp16_t, X_TYPE>() && std::is_same<float, Y_TYPE>()) {
                p = ctx->device->pipeline_matmul_f16_f32->m;
                shname = "F16_F32_M";
            } else if (std::is_same<ggml_fp16_t, X_TYPE>() && std::is_same<ggml_fp16_t, Y_TYPE>()) {
                p = ctx->device->pipeline_matmul_f16->m;
                shname = "F16_M";
            }
        } else if (shader_size == 2) {
            if (std::is_same<float, X_TYPE>() && std::is_same<float, Y_TYPE>()) {
                p = ctx->device->pipeline_matmul_f32->l;
                shname = "F32_L";
            } else if (std::is_same<float, X_TYPE>() && std::is_same<ggml_fp16_t, Y_TYPE>()) {
                p = ctx->device->pipeline_matmul_f32_f16->l;
                shname = "F32_F16_L";
            } else if (std::is_same<ggml_fp16_t, X_TYPE>() && std::is_same<float, Y_TYPE>()) {
                p = ctx->device->pipeline_matmul_f16_f32->l;
                shname = "F16_F32_L";
            } else if (std::is_same<ggml_fp16_t, X_TYPE>() && std::is_same<ggml_fp16_t, Y_TYPE>()) {
                p = ctx->device->pipeline_matmul_f16->l;
                shname = "F16_L";
            }
        }
    }

    ggml_pipeline_allocate_descriptor_sets(ctx->device, p, num_it);
    if (split_k > 1) {
        ggml_pipeline_allocate_descriptor_sets(ctx->device, ctx->device->pipeline_matmul_split_k_reduce, num_it);

        if (ctx->prealloc_split_k == nullptr || ctx->prealloc_split_k->size < sizeof(float) * d_ne * split_k) {
            // Resize buffer
            if (ctx->prealloc_split_k != nullptr) {
                ggml_vk_destroy_buffer(ctx->prealloc_split_k);
            }
            ctx->prealloc_split_k = ggml_vk_create_buffer_check(ctx->device, sizeof(float) * d_ne * split_k, vk::MemoryPropertyFlagBits::eDeviceLocal);
        }
    }

    vk_buffer d_X = ggml_vk_create_buffer_check(ctx->device, sizeof(X_TYPE) * x_ne, vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk_buffer d_Y = ggml_vk_create_buffer_check(ctx->device, sizeof(Y_TYPE) * y_ne, vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk_buffer d_D = ggml_vk_create_buffer_check(ctx->device, sizeof(float) * d_ne, vk::MemoryPropertyFlagBits::eDeviceLocal);

    X_TYPE* x = (X_TYPE *) malloc(sizeof(X_TYPE) * x_ne);
    Y_TYPE* y = (Y_TYPE *) malloc(sizeof(Y_TYPE) * y_ne);
    float* d = (float *) malloc(sizeof(float) * d_ne);

    for (size_t i = 0; i < x_ne; i++) {
        if (std::is_same<float, X_TYPE>()) {
            x[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        } else if (std::is_same<ggml_fp16_t, X_TYPE>()) {
            x[i] = ggml_fp32_to_fp16((rand() / (float)RAND_MAX) * 2.0f - 1.0f);
        } else {
            GGML_ABORT("fatal error");
        }
    }
    for (size_t i = 0; i < y_ne; i++) {
        if (std::is_same<float, Y_TYPE>()) {
            // y[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
            y[i] = (i % k == i / k) ? 1.0f : 0.0f;
        } else if (std::is_same<ggml_fp16_t, Y_TYPE>()) {
            // y[i] = ggml_fp32_to_fp16((rand() / (float)RAND_MAX) * 2.0f - 1.0f);
            y[i] = ggml_fp32_to_fp16((i % k == i / k) ? 1.0f : 0.0f);
        } else {
            GGML_ABORT("fatal error");
        }
    }

    ggml_vk_buffer_write(d_X, 0, x, sizeof(X_TYPE) * k * m * batch);
    ggml_vk_buffer_write(d_Y, 0, y, sizeof(Y_TYPE) * k * n * batch);

    vk_context subctx = ggml_vk_create_context(ctx, ctx->device->compute_queue);
    for (size_t i = 0; i < num_it; i++) {
        ggml_vk_ctx_begin(ctx->device, subctx);
        ggml_vk_matmul(
            ctx, subctx, p, ggml_vk_subbuffer(d_X), ggml_vk_subbuffer(d_Y), ggml_vk_subbuffer(d_D), ggml_vk_subbuffer(ctx->prealloc_split_k),
            m, n, k,
            k, k, m, k*m, k*n, m*n,
            split_k, batch, batch, batch, 1, 1
        );
        ggml_vk_ctx_end(subctx);
    }

    auto begin = std::chrono::high_resolution_clock::now();
    ggml_vk_submit(subctx, ctx->fence);
    VK_CHECK(ctx->device->device.waitForFences({ ctx->fence }, true, UINT64_MAX), "ggml_vk_test_matmul waitForFences");
    ctx->device->device.resetFences({ ctx->fence });

    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0;

    // copy dst to host
    ggml_vk_buffer_read(d_D, 0, d, sizeof(float) * d_ne);

    float * d_chk = (float *) malloc(sizeof(float) * d_ne);

    ggml_init_params iparams = {
        /*.mem_size   =*/ 1024*1024*1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    ggml_context * ggml_ctx = ggml_init(iparams);

    ggml_type src0_type;
    ggml_type src1_type;

    if (std::is_same<float, X_TYPE>()) {
        src0_type = GGML_TYPE_F32;
    } else if (std::is_same<ggml_fp16_t, X_TYPE>()) {
        src0_type = GGML_TYPE_F16;
    } else {
        GGML_ABORT("fatal error");
    }
    if (std::is_same<float, Y_TYPE>()) {
        src1_type = GGML_TYPE_F32;
    } else if (std::is_same<ggml_fp16_t, Y_TYPE>()) {
        src1_type = GGML_TYPE_F16;
    } else {
        GGML_ABORT("fatal error");
    }

    ggml_tensor * src0_ggml = ggml_new_tensor_3d(ggml_ctx, src0_type, k, m, batch);
    ggml_tensor * src1_ggml = ggml_new_tensor_3d(ggml_ctx, src1_type, k, n, batch);
    ggml_tensor * tensor_ggml = ggml_mul_mat(ggml_ctx, src0_ggml, src1_ggml);

    src0_ggml->data = x;
    src1_ggml->data = y;
    tensor_ggml->data = d_chk;

    ggml_cgraph * cgraph = ggml_new_graph(ggml_ctx);
    ggml_build_forward_expand(cgraph, tensor_ggml);

    ggml_graph_compute_with_ctx(ggml_ctx, cgraph, 1);

    ggml_free(ggml_ctx);

    double avg_err = 0.0;
    int first_err_n = -1;
    int first_err_m = -1;
    int first_err_b = -1;

    for (size_t i = 0; i < m*n*batch; i++) {
        double err = std::fabs(d[i] - d_chk[i]);
        avg_err += err;

        if (err > 0.05f && first_err_n == -1) {
            first_err_b = i / (m * n);
            first_err_n = (i % (m * n)) / m;
            first_err_m = (i % (m * n)) % m;
        }
    }

    avg_err /= m * n;

    std::cerr << "TEST " << shname << " m=" << m << " n=" << n << " k=" << k << " batch=" << batch << " split_k=" << split_k << " matmul " << time / num_it << "ms avg_err=" << avg_err << std::endl;

    if (avg_err > 0.1) {
        std::cerr << "m = " << first_err_m << " n = " << first_err_n << " b = " << first_err_b << std::endl;
        std::cerr << "Actual result: " << std::endl << std::endl;
        ggml_vk_print_matrix_area(d, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);
        std::cerr << std::endl;
        ggml_vk_print_matrix_area(d, GGML_TYPE_F32, m, n, first_err_m, first_err_n + 15, first_err_b);
        std::cerr << "Expected result: " << std::endl << std::endl;
        ggml_vk_print_matrix_area(d_chk, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

        if (split_k > 1) {
            float * split_k_buf = (float *) malloc(sizeof(float) * d_ne * split_k);
            ggml_vk_buffer_read(ctx->prealloc_split_k, 0, split_k_buf, sizeof(float) * d_ne * split_k);

            std::cerr << "d_buf0: " << std::endl << std::endl;
            ggml_vk_print_matrix_area(split_k_buf, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            std::cerr << "d_buf1: " << std::endl << std::endl;
            ggml_vk_print_matrix_area(split_k_buf + d_ne, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            std::cerr << "d_buf2: " << std::endl << std::endl;
            ggml_vk_print_matrix_area(split_k_buf + 2 * d_ne, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            std::cerr << "d_buf3: " << std::endl << std::endl;
            ggml_vk_print_matrix_area(split_k_buf + 3 * d_ne, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            free(split_k_buf);
        }
    }

    free(d_chk);

    ggml_vk_queue_cleanup(ctx->device, ctx->device->transfer_queue);
    ggml_vk_queue_cleanup(ctx->device, ctx->device->compute_queue);

    ggml_vk_destroy_buffer(d_X);
    ggml_vk_destroy_buffer(d_Y);
    ggml_vk_destroy_buffer(d_D);

    ggml_pipeline_cleanup(p);
    ggml_pipeline_cleanup(ctx->device->pipeline_matmul_split_k_reduce);

    free(x);
    free(y);
    free(d);
}

static void ggml_vk_print_tensor_area(const ggml_tensor * tensor, int i0, int i1, int i2, int i3) {
    if (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16) {
        return;
    }
    i0 = std::max(i0, 5);
    i1 = std::max(i1, 5);
    i2 = std::max(i2, 0);
    i3 = std::max(i3, 0);
    fprintf(stderr, "         ");
    for (int idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
        fprintf(stderr, "%7d ", idx1);
    }
    fprintf(stderr, "\n");
    for (int idx0 = i0 - 5; idx0 < i0 + 5; idx0++) {
        fprintf(stderr, "%7d: ", idx0);
        for (int idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
            if (idx0 >= 0 && idx0 < tensor->ne[0] && idx1 >= 0 && idx1 < tensor->ne[1] && i2 >= 0 && i2 < tensor->ne[2] && i3 >= 0 && i3 < tensor->ne[3]) {
                float val;
                if (tensor->type == GGML_TYPE_F32) {
                    val = *(float *) ((char *) tensor->data + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]);
                } else if (tensor->type == GGML_TYPE_F16) {
                    val = ggml_fp16_to_fp32(*(ggml_fp16_t *) ((char *) tensor->data + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]));
                } else {
                    GGML_ABORT("fatal error");
                }
                fprintf(stderr, "% 7.2f ", val);
            } else {
                fprintf(stderr, "        ");
            }
        }
        fprintf(stderr, "\n");
    }
}

static void ggml_vk_quantize_data(const float * from, void * to, size_t ne, ggml_type quant) {
    ggml_quantize_chunk(quant, from, to, 0, 1, ne, nullptr);
}

static void ggml_vk_dequantize_data(const void * from, float * to, size_t ne, ggml_type quant) {
    if (quant == GGML_TYPE_F32) {
        memcpy(to, from, sizeof(float) * ne);
        return;
    }

    ggml_type_traits_t tt = ggml_internal_get_type_traits(quant);

    ggml_to_float_t dequant_fn = tt.to_float;

    dequant_fn(from, to, ne);
}

static void ggml_vk_test_dequant(ggml_backend_vk_context * ctx, size_t ne, ggml_type quant) {
    VK_LOG_DEBUG("ggml_vk_test_dequant(" << ne << ")");
    const size_t x_sz = sizeof(float) * ne;
    const size_t x_sz_f16 = sizeof(ggml_fp16_t) * ne;
    const size_t qx_sz = ne * ggml_type_size(quant)/ggml_blck_size(quant);
    float * x = (float *) malloc(x_sz);
    void * qx = malloc(qx_sz);
    vk_buffer qx_buf = ggml_vk_create_buffer_check(ctx->device, qx_sz, vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk_buffer x_buf = ggml_vk_create_buffer_check(ctx->device, x_sz_f16, vk::MemoryPropertyFlagBits::eDeviceLocal);
    float * x_ref = (float *) malloc(x_sz);
    ggml_fp16_t * x_chk = (ggml_fp16_t *) malloc(x_sz_f16);

    for (size_t i = 0; i < ne; i++) {
        x[i] = rand() / (float)RAND_MAX;
    }

    vk_pipeline p = ggml_vk_get_to_fp16(ctx, quant);

    ggml_vk_quantize_data(x, qx, ne, quant);
    ggml_vk_dequantize_data(qx, x_ref, ne, quant);

    ggml_pipeline_allocate_descriptor_sets(ctx->device, p, 1);

    ggml_vk_buffer_write(qx_buf, 0, qx, qx_sz);

    vk_context subctx = ggml_vk_create_context(ctx, ctx->device->compute_queue);
    ggml_vk_ctx_begin(ctx->device, subctx);
    const std::vector<uint32_t> pc = { 1, (uint32_t)ne, (uint32_t)ne, (uint32_t)ne, (uint32_t)ne };
    ggml_vk_dispatch_pipeline(ctx, subctx, p, { { qx_buf, 0, qx_sz }, { x_buf, 0, x_sz_f16 } }, pc.size() * sizeof(int), pc.data(), { (uint32_t)ne, 1, 1});
    ggml_vk_ctx_end(subctx);

    auto begin = std::chrono::high_resolution_clock::now();

    ggml_vk_submit(subctx, ctx->fence);
    VK_CHECK(ctx->device->device.waitForFences({ ctx->fence }, true, UINT64_MAX), "ggml_vk_test_dequant waitForFences");
    ctx->device->device.resetFences({ ctx->fence });

    auto end = std::chrono::high_resolution_clock::now();

    double ms_dequant = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0;
    ggml_vk_buffer_read(x_buf, 0, x_chk, x_sz_f16);

    int first_err = -1;

    double avg_err = 0.0;
    for (size_t i = 0; i < ne; i++) {
        double error = std::fabs(x_ref[i] - ggml_fp16_to_fp32(x_chk[i]));
        avg_err += error;

        if (first_err < 0 && error > 0.05) {
            first_err = i;
        }
    }

    avg_err /= ne;

    std::cerr << "TEST DEQUANT " << ggml_type_name(quant) << " time=" << ms_dequant << "ms avg_err=" << avg_err << std::endl;

    if (avg_err > 0.1) {
        std::cerr << "first_error = " << first_err << std::endl;
        std::cerr << "Actual result: " << std::endl << std::endl;
        for (int i = std::max(0, first_err - 5); i < std::min((int)ne, first_err + 5); i++) {
            std::cerr << ggml_fp16_to_fp32(x_chk[i]) << ", ";
        }
        std::cerr << std::endl << "Expected result: " << std::endl << std::endl;
        for (int i = std::max(0, first_err - 5); i < std::min((int)ne, first_err + 5); i++) {
            std::cerr << x_ref[i] << ", ";
        }
        std::cerr << std::endl;
    }

    ggml_vk_destroy_buffer(x_buf);
    ggml_vk_destroy_buffer(qx_buf);

    free(x);
    free(qx);
    free(x_ref);
    free(x_chk);
}

static void ggml_vk_test_dequant_matmul(ggml_backend_vk_context * ctx, size_t m, size_t n, size_t k, size_t batch, size_t num_it, size_t split_k, size_t shader_size, ggml_type quant) {
    VK_LOG_DEBUG("ggml_vk_test_dequant_matmul(" << m << ", " << n << ", " << k << ", " << batch << ", " << num_it << ", " << split_k << ", " << ggml_type_name(quant) << ")");
    const size_t x_ne = m * k * batch;
    const size_t y_ne = k * n * batch;
    const size_t d_ne = m * n * batch;

    vk_pipeline p;
    std::string shname;
    if (shader_size == 0) {
        p = ctx->device->pipeline_dequant_mul_mat_mat[quant]->a_s;
        shname = std::string(ggml_type_name(quant)) + "_ALIGNED_S";
    } else if (shader_size == 1) {
        p = ctx->device->pipeline_dequant_mul_mat_mat[quant]->a_m;
        shname = std::string(ggml_type_name(quant)) + "_ALIGNED_M";
    } else if (shader_size == 2) {
        p = ctx->device->pipeline_dequant_mul_mat_mat[quant]->a_l;
        shname = std::string(ggml_type_name(quant)) + "_ALIGNED_L";
    } else {
        GGML_ASSERT(0);
    }

    const size_t kpad = ggml_vk_align_size(k, p->align);

    if (k != kpad) {
        if (shader_size == 0) {
            p = ctx->device->pipeline_dequant_mul_mat_mat[quant]->s;
            shname = std::string(ggml_type_name(quant)) + "_S";
        } else if (shader_size == 1) {
            p = ctx->device->pipeline_dequant_mul_mat_mat[quant]->m;
            shname = std::string(ggml_type_name(quant)) + "_M";
        } else if (shader_size == 2) {
            p = ctx->device->pipeline_dequant_mul_mat_mat[quant]->l;
            shname = std::string(ggml_type_name(quant)) + "_L";
        } else {
            GGML_ASSERT(0);
        }
    }

    const size_t x_sz = sizeof(float) * x_ne;
    const size_t y_sz = sizeof(float) * y_ne;
    const size_t qx_sz = x_ne * ggml_type_size(quant)/ggml_blck_size(quant);
    const size_t d_sz = sizeof(float) * d_ne;
    float * x = (float *) malloc(x_sz);
    float * y = (float *) malloc(y_sz);
    void * qx = malloc(qx_sz);
    vk_buffer qx_buf = ggml_vk_create_buffer_check(ctx->device, qx_sz, vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk_buffer y_buf = ggml_vk_create_buffer_check(ctx->device, y_sz, vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk_buffer d_buf = ggml_vk_create_buffer_check(ctx->device, d_sz, vk::MemoryPropertyFlagBits::eDeviceLocal);
    float * d = (float *) malloc(d_sz);
    float * d_chk = (float *) malloc(d_sz);

    for (size_t i = 0; i < x_ne; i++) {
        x[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }

    ggml_vk_quantize_data(x, qx, x_ne, quant);

    for (size_t i = 0; i < y_ne; i++) {
        // y[i] = rand() / (float)RAND_MAX;
        y[i] = (i % k == i / k) ? 1.0f : 0.0f;
    }

    ggml_pipeline_allocate_descriptor_sets(ctx->device, p, num_it);
    if (split_k > 1) {
        ggml_pipeline_allocate_descriptor_sets(ctx->device, ctx->device->pipeline_matmul_split_k_reduce, num_it);

        if (ctx->prealloc_split_k == nullptr || ctx->prealloc_split_k->size < sizeof(float) * d_ne * split_k) {
            // Resize buffer
            if (ctx->prealloc_split_k != nullptr) {
                ggml_vk_destroy_buffer(ctx->prealloc_split_k);
            }
            ctx->prealloc_split_k = ggml_vk_create_buffer_check(ctx->device, sizeof(float) * d_ne * split_k, vk::MemoryPropertyFlagBits::eDeviceLocal);
        }
    }

    ggml_vk_buffer_write(qx_buf, 0, qx, qx_sz);
    ggml_vk_buffer_write(y_buf, 0, y, y_sz);

    vk_context subctx = ggml_vk_create_context(ctx, ctx->device->compute_queue);
    for (size_t i = 0; i < num_it; i++) {
        ggml_vk_ctx_begin(ctx->device, subctx);
        ggml_vk_matmul(
            ctx, subctx, p, ggml_vk_subbuffer(qx_buf), ggml_vk_subbuffer(y_buf), ggml_vk_subbuffer(d_buf), ggml_vk_subbuffer(ctx->prealloc_split_k),
            m, n, k,
            k, k, m, k*m, k*n, m*n,
            split_k, batch, batch, batch, 1, 1
        );
        ggml_vk_ctx_end(subctx);
    }

    auto begin = std::chrono::high_resolution_clock::now();

    ggml_vk_submit(subctx, ctx->fence);
    VK_CHECK(ctx->device->device.waitForFences({ ctx->fence }, true, UINT64_MAX), "ggml_vk_test_dequant waitForFences");
    ctx->device->device.resetFences({ ctx->fence });

    auto end = std::chrono::high_resolution_clock::now();

    double time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0;
    ggml_vk_buffer_read(d_buf, 0, d, d_sz);

    ggml_init_params iparams = {
        /*.mem_size   =*/ 1024*1024*1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    ggml_context * ggml_ctx = ggml_init(iparams);

    ggml_tensor * src0_ggml = ggml_new_tensor_3d(ggml_ctx, quant, k, m, batch);
    ggml_tensor * src1_ggml = ggml_new_tensor_3d(ggml_ctx, GGML_TYPE_F32, k, n, batch);
    ggml_tensor * tensor_ggml = ggml_mul_mat(ggml_ctx, src0_ggml, src1_ggml);

    src0_ggml->data = qx;
    src1_ggml->data = y;
    tensor_ggml->data = d_chk;

    ggml_cgraph * cgraph = ggml_new_graph(ggml_ctx);
    ggml_build_forward_expand(cgraph, tensor_ggml);

    ggml_graph_compute_with_ctx(ggml_ctx, cgraph, 1);

    ggml_free(ggml_ctx);

    double avg_err = 0.0;
    int first_err_n = -1;
    int first_err_m = -1;
    int first_err_b = -1;

    for (size_t i = 0; i < m*n*batch; i++) {
        double err = std::fabs(d[i] - d_chk[i]);
        avg_err += err;

        if ((err > 0.05f || std::isnan(err)) && first_err_n == -1) {
            first_err_b = i / (m * n);
            first_err_n = (i % (m * n)) / m;
            first_err_m = (i % (m * n)) % m;
        }
    }

    avg_err /= m * n;

    std::cerr << "TEST MMQ " << shname << " m=" << m << " n=" << n << " k=" << k << " batch=" << batch << " split_k=" << split_k << " matmul " << time_ms / num_it << "ms avg_err=" << avg_err << std::endl;

    if (avg_err > 0.01 || std::isnan(avg_err)) {
        std::cerr << "m = " << first_err_m << " n = " << first_err_n << " b = " << first_err_b << std::endl;
        std::cerr << "Actual result: " << std::endl << std::endl;
        ggml_vk_print_matrix_area(d, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);
        std::cerr << std::endl;
        std::cerr << "Expected result: " << std::endl << std::endl;
        ggml_vk_print_matrix_area(d_chk, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

        if (split_k > 1) {
            float * split_k_buf = (float *) malloc(sizeof(float) * d_ne * split_k);
            ggml_vk_buffer_read(ctx->prealloc_split_k, 0, split_k_buf, sizeof(float) * d_ne * split_k);

            std::cerr << "d_buf0: " << std::endl << std::endl;
            ggml_vk_print_matrix_area(split_k_buf, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            std::cerr << "d_buf1: " << std::endl << std::endl;
            ggml_vk_print_matrix_area(split_k_buf + d_ne, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            std::cerr << "d_buf2: " << std::endl << std::endl;
            ggml_vk_print_matrix_area(split_k_buf + 2 * d_ne, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            std::cerr << "d_buf3: " << std::endl << std::endl;
            ggml_vk_print_matrix_area(split_k_buf + 3 * d_ne, GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            free(split_k_buf);
        }
    }

    ggml_vk_destroy_buffer(qx_buf);
    ggml_vk_destroy_buffer(y_buf);
    ggml_vk_destroy_buffer(d_buf);

    free(x);
    free(qx);
    free(y);
    free(d);
    free(d_chk);
}
#endif

static ggml_tensor_extra_gpu * ggml_vk_tensor_create_extra(ggml_tensor * tensor) {
    VK_LOG_DEBUG("ggml_vk_create_extra(" << tensor << " (" << tensor->name << ", " << ggml_op_name(tensor->op) << "))");
    ggml_tensor_extra_gpu * extra = new ggml_tensor_extra_gpu;
    extra->reset();
    tensor->extra = extra;
    return extra;
}

static void ggml_vk_preallocate_buffers_graph(ggml_backend_vk_context * ctx, ggml_tensor * node){
    VK_LOG_DEBUG("ggml_vk_preallocate_buffers_graph(" << node << ")");
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) node->extra;

    if (extra == nullptr) {
        return;
    }

    ggml_tensor * src0 = node->src[0];
    ggml_tensor * src1 = node->src[1];

    const bool use_src0 = src0 != nullptr;
    const int64_t ne00 = use_src0 ? src0->ne[0] : 0;
    const int64_t ne01 = use_src0 ? src0->ne[1] : 0;
    const int64_t ne02 = use_src0 ? src0->ne[2] : 0;
    const int64_t ne03 = use_src0 ? src0->ne[3] : 0;
    const bool use_src1 = src1 != nullptr && node->op != GGML_OP_CPY && node->op != GGML_OP_CONT && node->op != GGML_OP_DUP;
    const int64_t ne10 = use_src1 ? src1->ne[0] : 0;
    const int64_t ne11 = use_src1 ? src1->ne[1] : 0;
    const int64_t ne12 = use_src1 ? src1->ne[2] : 0;
    const int64_t ne13 = use_src1 ? src1->ne[3] : 0;
    const int64_t ne20 = node->ne[0];
    const int64_t ne21 = node->ne[1];
    const int64_t ne22 = node->ne[2];
    const int64_t ne23 = node->ne[3];

    const ggml_type src0_type = (use_src0 && src0->type == GGML_TYPE_F32) ? src0->type : GGML_TYPE_F16;
    const ggml_type src1_type = (use_src1 && src1->type == GGML_TYPE_F32) ? src1->type : GGML_TYPE_F16;

    const bool x_non_contig = use_src0 && !ggml_vk_dim01_contiguous(src0);
    const bool y_non_contig = use_src1 && !ggml_vk_dim01_contiguous(src1);

    const bool y_f32_kernel = use_src1 && src1->type == GGML_TYPE_F32 && !y_non_contig;

    bool mmp = (use_src0 && use_src1 && (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID)) ? ggml_vk_get_mul_mat_mat_pipeline(ctx, src0->type, y_non_contig ? GGML_TYPE_F16 : src1->type) != nullptr : false;

    const bool qx_needs_dequant = use_src0 && (!mmp || x_non_contig);
    const bool qy_needs_dequant = use_src1 && ((src1->type != GGML_TYPE_F16 && !y_f32_kernel) || y_non_contig);

    int split_k;
    if (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID) {
        split_k = ggml_vk_guess_split_k(ne01, ne11, ne10);
    } else {
        split_k = 1;
    }
    const uint32_t x_ne = ne00 * ne01;
    const uint32_t y_ne = ne10 * ne11;
    const uint32_t d_ne = ne20 * ne21;

    const uint64_t x_sz = (use_src0 && qx_needs_dequant) ? ggml_vk_align_size(sizeof(src0_type) * x_ne, ctx->device->properties.limits.minStorageBufferOffsetAlignment) * ne02 * ne03 : 0;
    const uint64_t y_sz = (use_src1 && qy_needs_dequant) ? ggml_vk_align_size(sizeof(src1_type) * y_ne, ctx->device->properties.limits.minStorageBufferOffsetAlignment) * ne12 * ne13 : 0;
    uint64_t d_sz = ggml_vk_align_size(ggml_type_size(node->type) * d_ne, ctx->device->properties.limits.minStorageBufferOffsetAlignment) * ne22 * ne23;
    const uint64_t split_k_size = split_k > 1 ? d_sz * 4 : 0;

    if (extra->buffer_gpu.expired()) {
        // Workaround for CPU backend BLAS matmul calls
        extra->buffer_gpu = ggml_vk_create_buffer_temp(ctx, d_sz);
    }

    switch (node->op) {
    case GGML_OP_REPEAT:
    case GGML_OP_GET_ROWS:
    case GGML_OP_RESHAPE:
    case GGML_OP_VIEW:
    case GGML_OP_PERMUTE:
    case GGML_OP_TRANSPOSE:
    case GGML_OP_ADD:
    case GGML_OP_SCALE:
    case GGML_OP_SQR:
    case GGML_OP_CLAMP:
    case GGML_OP_PAD:
    case GGML_OP_CPY:
    case GGML_OP_CONT:
    case GGML_OP_DUP:
    case GGML_OP_MUL:
    case GGML_OP_DIV:
    case GGML_OP_CONCAT:
    case GGML_OP_UPSCALE:
    case GGML_OP_NORM:
    case GGML_OP_GROUP_NORM:
    case GGML_OP_RMS_NORM:
    case GGML_OP_DIAG_MASK_INF:
    case GGML_OP_SOFT_MAX:
    case GGML_OP_ROPE:
    case GGML_OP_ARGSORT:
    case GGML_OP_SUM_ROWS:
    case GGML_OP_IM2COL:
    case GGML_OP_TIMESTEP_EMBEDDING:
    case GGML_OP_LEAKY_RELU:
        break;
    case GGML_OP_UNARY:
        switch (ggml_get_unary_op(node)) {
        case GGML_UNARY_OP_SILU:
        case GGML_UNARY_OP_GELU:
        case GGML_UNARY_OP_GELU_QUICK:
        case GGML_UNARY_OP_RELU:
        case GGML_UNARY_OP_TANH:
            break;
        default:
            return;
        }
        break;
    case GGML_OP_MUL_MAT:
    case GGML_OP_MUL_MAT_ID:
        if (
                x_sz > ctx->device->max_memory_allocation_size ||
                y_sz > ctx->device->max_memory_allocation_size ||
                d_sz > ctx->device->max_memory_allocation_size ||
                split_k_size > ctx->device->max_memory_allocation_size) {
            GGML_ABORT("Requested preallocation size is too large");
        }
        if (ctx->prealloc_size_x < x_sz) {
            ctx->prealloc_size_x = x_sz;
        }
        if (ctx->prealloc_size_y < y_sz) {
            ctx->prealloc_size_y = y_sz;
        }
        if (ctx->prealloc_size_split_k < split_k_size) {
            ctx->prealloc_size_split_k = split_k_size;
        }
        if (ctx->staging_size < x_sz + y_sz) {
            ctx->staging_size = x_sz + y_sz;
        }
        break;
    default:
        return;
    }
}

static void ggml_vk_preallocate_buffers(ggml_backend_vk_context * ctx) {
#if defined(GGML_VULKAN_RUN_TESTS)
    ctx->staging = ggml_vk_create_buffer_check(ctx->device, 100ul * 1024ul * 1024ul,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    ggml_vk_test_dequant(ctx, 7680, GGML_TYPE_F32);
    ggml_vk_test_dequant(ctx, 7680, GGML_TYPE_Q4_0);
    ggml_vk_test_dequant(ctx, 7680, GGML_TYPE_Q4_1);
    ggml_vk_test_dequant(ctx, 7680, GGML_TYPE_Q5_0);
    ggml_vk_test_dequant(ctx, 7680, GGML_TYPE_Q5_1);
    ggml_vk_test_dequant(ctx, 7680, GGML_TYPE_Q8_0);
    ggml_vk_test_dequant(ctx, 7680, GGML_TYPE_Q2_K);
    ggml_vk_test_dequant(ctx, 7680, GGML_TYPE_Q3_K);
    ggml_vk_test_dequant(ctx, 7680, GGML_TYPE_Q4_K);
    ggml_vk_test_dequant(ctx, 7680, GGML_TYPE_Q5_K);
    ggml_vk_test_dequant(ctx, 7680, GGML_TYPE_Q6_K);
    ggml_vk_test_dequant(ctx, 7680, GGML_TYPE_IQ4_NL);

    ggml_vk_test_matmul<ggml_fp16_t, ggml_fp16_t>(ctx, 512, 512, 100, 32, 100, 1, 2);

    ggml_vk_test_matmul<float, float>(ctx, 128, 512, 512, 2, 100, 1, 0);
    ggml_vk_test_matmul<float, float>(ctx, 128, 512, 512, 2, 100, 1, 1);
    ggml_vk_test_matmul<float, float>(ctx, 128, 512, 512, 2, 100, 1, 2);
    // ggml_vk_test_matmul<float, float>(ctx, 128, 512, 512, 2, 100, 4, 0);
    // ggml_vk_test_matmul<float, float>(ctx, 128, 512, 512, 2, 100, 4, 1);
    // ggml_vk_test_matmul<float, float>(ctx, 128, 512, 512, 2, 100, 4, 2);

    ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 1, 0, GGML_TYPE_Q4_0);
    ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 1, 1, GGML_TYPE_Q4_0);
    ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 1, 2, GGML_TYPE_Q4_0);
    // ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 4, 0, GGML_TYPE_Q4_0);
    // ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 4, 1, GGML_TYPE_Q4_0);
    // ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 4, 2, GGML_TYPE_Q4_0);

    ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 1, 0, GGML_TYPE_Q4_1);
    ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 1, 1, GGML_TYPE_Q4_1);
    ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 1, 2, GGML_TYPE_Q4_1);
    // ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 4, 0, GGML_TYPE_Q4_1);
    // ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 4, 1, GGML_TYPE_Q4_1);
    // ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 4, 2, GGML_TYPE_Q4_1);

    ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 1, 0, GGML_TYPE_Q5_0);
    ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 1, 1, GGML_TYPE_Q5_0);
    ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 1, 2, GGML_TYPE_Q5_0);
    // ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 4, 0, GGML_TYPE_Q5_0);
    // ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 4, 1, GGML_TYPE_Q5_0);
    // ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 4, 2, GGML_TYPE_Q5_0);

    ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 1, 0, GGML_TYPE_Q5_1);
    ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 1, 1, GGML_TYPE_Q5_1);
    ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 1, 2, GGML_TYPE_Q5_1);
    // ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 4, 0, GGML_TYPE_Q5_1);
    // ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 4, 1, GGML_TYPE_Q5_1);
    // ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 4, 2, GGML_TYPE_Q5_1);

    ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 1, 0, GGML_TYPE_Q8_0);
    ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 1, 1, GGML_TYPE_Q8_0);
    ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 1, 2, GGML_TYPE_Q8_0);
    // ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 4, 0, GGML_TYPE_Q8_0);
    // ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 4, 1, GGML_TYPE_Q8_0);
    // ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 4, 2, GGML_TYPE_Q8_0);

    ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 1, 0, GGML_TYPE_Q2_K);
    ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 1, 1, GGML_TYPE_Q2_K);
    ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 1, 2, GGML_TYPE_Q2_K);
    // ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 4, 0, GGML_TYPE_Q2_K);
    // ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 4, 1, GGML_TYPE_Q2_K);
    // ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 4, 2, GGML_TYPE_Q2_K);

    ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 1, 0, GGML_TYPE_Q3_K);
    ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 1, 1, GGML_TYPE_Q3_K);
    ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 1, 2, GGML_TYPE_Q3_K);
    // ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 4, 0, GGML_TYPE_Q3_K);
    // ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 4, 1, GGML_TYPE_Q3_K);
    // ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 4, 2, GGML_TYPE_Q3_K);

    ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 1, 0, GGML_TYPE_Q4_K);
    ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 1, 1, GGML_TYPE_Q4_K);
    ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 1, 2, GGML_TYPE_Q4_K);
    // ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 4, 0, GGML_TYPE_Q4_K);
    // ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 4, 1, GGML_TYPE_Q4_K);
    // ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 4, 2, GGML_TYPE_Q4_K);

    ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 1, 0, GGML_TYPE_Q5_K);
    ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 1, 1, GGML_TYPE_Q5_K);
    ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 1, 2, GGML_TYPE_Q5_K);
    // ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 4, 0, GGML_TYPE_Q5_K);
    // ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 4, 1, GGML_TYPE_Q5_K);
    // ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 4, 2, GGML_TYPE_Q5_K);

    ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 1, 0, GGML_TYPE_Q6_K);
    ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 1, 1, GGML_TYPE_Q6_K);
    ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 1, 2, GGML_TYPE_Q6_K);
    // ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 4, 0, GGML_TYPE_Q6_K);
    // ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 4, 1, GGML_TYPE_Q6_K);
    // ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 4, 2, GGML_TYPE_Q6_K);

    ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 1, 0, GGML_TYPE_IQ4_NL);
    ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 1, 1, GGML_TYPE_IQ4_NL);
    ggml_vk_test_dequant_matmul(ctx, 128, 512, 512, 2, 100, 1, 2, GGML_TYPE_IQ4_NL);

    std::cerr << std::endl;

    const std::vector<size_t> vals {
        8, 8, 8,
        100, 46, 576,
        623, 111, 128,
        100, 46, 558,
        512, 1, 256,
        128, 110, 622,
        511, 511, 127,
        511, 511, 7,
        511, 511, 17,
        49, 49, 128,
        128, 49, 49,
        4096, 49, 4096,
        11008, 49, 4096,
        4096, 49, 11008,
        32000, 49, 4096,
        512, 512, 128,
        128, 512, 512,
        4096, 512, 4096,
        11008, 512, 4096,
        4096, 512, 11008,
        32000, 512, 4096,
    };
    const size_t num_it = 1;
    for (size_t i = 0; i < vals.size(); i += 3) {
        ggml_vk_test_matmul<ggml_fp16_t, float>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 1, 0);
        ggml_vk_test_matmul<ggml_fp16_t, float>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 1, 1);
        ggml_vk_test_matmul<ggml_fp16_t, float>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 1, 2);
        // ggml_vk_test_matmul<ggml_fp16_t, float>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 4, 0);
        // ggml_vk_test_matmul<ggml_fp16_t, float>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 4, 1);
        // ggml_vk_test_matmul<ggml_fp16_t, float>(ctx, vals[i], vals[i + 1], vals[i + 2], 2, num_it, 4, 2);
        std::cerr << std::endl;
    }

    GGML_ABORT("fatal error");
#endif

    if (ctx->prealloc_x == nullptr || (ctx->prealloc_size_x > 0 && ctx->prealloc_x->size < ctx->prealloc_size_x)) {
        VK_LOG_MEMORY("ggml_vk_preallocate_buffers(x_size: " << ctx->prealloc_size_x << ")");
        // Resize buffer
        if (ctx->prealloc_x != nullptr) {
            ggml_vk_destroy_buffer(ctx->prealloc_x);
        }
        ctx->prealloc_x = ggml_vk_create_buffer_device(ctx->device, ctx->prealloc_size_x);
    }
    if (ctx->prealloc_y == nullptr || (ctx->prealloc_size_y > 0 && ctx->prealloc_y->size < ctx->prealloc_size_y)) {
        VK_LOG_MEMORY("ggml_vk_preallocate_buffers(y_size: " << ctx->prealloc_size_y << ")");
        // Resize buffer
        if (ctx->prealloc_y != nullptr) {
            ggml_vk_destroy_buffer(ctx->prealloc_y);
        }
        ctx->prealloc_y = ggml_vk_create_buffer_device(ctx->device, ctx->prealloc_size_y);
    }
    if (ctx->prealloc_split_k == nullptr || (ctx->prealloc_size_split_k > 0 && ctx->prealloc_split_k->size < ctx->prealloc_size_split_k)) {
        VK_LOG_MEMORY("ggml_vk_preallocate_buffers(split_k_size: " << ctx->prealloc_size_split_k << ")");
        // Resize buffer
        if (ctx->prealloc_split_k != nullptr) {
            ggml_vk_destroy_buffer(ctx->prealloc_split_k);
        }
        ctx->prealloc_split_k = ggml_vk_create_buffer_device(ctx->device, ctx->prealloc_size_split_k);
    }
    if (ctx->staging == nullptr || (ctx->staging_size > 0 && ctx->staging->size < ctx->staging_size)) {
        VK_LOG_MEMORY("ggml_vk_preallocate_buffers(staging_size: " << ctx->staging_size << ")");
        // Resize buffer
        if (ctx->staging != nullptr) {
            ggml_vk_destroy_buffer(ctx->staging);
        }
        ctx->staging = ggml_vk_create_buffer_check(ctx->device, ctx->staging_size,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    }
}

static void ggml_vk_build_graph(ggml_backend_vk_context * ctx, ggml_tensor * node, int node_idx, bool last_node){
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) node->extra;

    if (ggml_is_empty(node) || extra == nullptr) {
        return;
    }

    VK_LOG_DEBUG("ggml_vk_build_graph(" << node << ", " << ggml_op_name(node->op) << ")");
    ctx->semaphore_idx = 0;
    ctx->staging_offset = 0;

    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];
    const ggml_tensor * src2 = node->src[2];

    switch (node->op) {
    // Return on empty ops to avoid generating a compute_ctx and setting exit_tensor
    case GGML_OP_RESHAPE:
    case GGML_OP_VIEW:
    case GGML_OP_PERMUTE:
    case GGML_OP_TRANSPOSE:
    case GGML_OP_NONE:
        return;
    case GGML_OP_UNARY:
        switch (ggml_get_unary_op(node)) {
        case GGML_UNARY_OP_SILU:
        case GGML_UNARY_OP_GELU:
        case GGML_UNARY_OP_GELU_QUICK:
        case GGML_UNARY_OP_RELU:
        case GGML_UNARY_OP_TANH:
            break;
        default:
            return;
        }
        break;
    case GGML_OP_REPEAT:
    case GGML_OP_GET_ROWS:
    case GGML_OP_ADD:
    case GGML_OP_MUL:
    case GGML_OP_DIV:
    case GGML_OP_CONCAT:
    case GGML_OP_UPSCALE:
    case GGML_OP_SCALE:
    case GGML_OP_SQR:
    case GGML_OP_CLAMP:
    case GGML_OP_PAD:
    case GGML_OP_CPY:
    case GGML_OP_CONT:
    case GGML_OP_DUP:
    case GGML_OP_NORM:
    case GGML_OP_GROUP_NORM:
    case GGML_OP_RMS_NORM:
    case GGML_OP_DIAG_MASK_INF:
    case GGML_OP_SOFT_MAX:
    case GGML_OP_ROPE:
    case GGML_OP_MUL_MAT:
    case GGML_OP_MUL_MAT_ID:
    case GGML_OP_ARGSORT:
    case GGML_OP_SUM_ROWS:
    case GGML_OP_IM2COL:
    case GGML_OP_TIMESTEP_EMBEDDING:
    case GGML_OP_LEAKY_RELU:
        break;
    default:
        std::cerr << "ggml_vulkan: Error: Missing op: " << ggml_op_name(node->op) << std::endl;
        GGML_ABORT("fatal error");
        return;
    }

    vk_context compute_ctx;

    if (ctx->compute_ctx.expired()) {
        compute_ctx = ggml_vk_create_context(ctx, ctx->device->compute_queue);
        ctx->compute_ctx = compute_ctx;
        ggml_vk_ctx_begin(ctx->device, compute_ctx);
    } else {
        compute_ctx = ctx->compute_ctx.lock();
    }

    switch (node->op) {
    case GGML_OP_REPEAT:
        ggml_vk_repeat(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_GET_ROWS:
        ggml_vk_get_rows(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_ADD:
        ggml_vk_add(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_MUL:
        ggml_vk_mul(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_DIV:
        ggml_vk_div(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_CONCAT:
        ggml_vk_concat(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_UPSCALE:
        ggml_vk_upscale(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_SCALE:
        ggml_vk_scale(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_SQR:
        ggml_vk_sqr(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_CLAMP:
        ggml_vk_clamp(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_PAD:
        ggml_vk_pad(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_CPY:
    case GGML_OP_CONT:
    case GGML_OP_DUP:
        ggml_vk_cpy(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_NORM:
        ggml_vk_norm(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_GROUP_NORM:
        ggml_vk_group_norm(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_RMS_NORM:
        ggml_vk_rms_norm(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_UNARY:
        switch (ggml_get_unary_op(node)) {
        case GGML_UNARY_OP_SILU:
        case GGML_UNARY_OP_GELU:
        case GGML_UNARY_OP_GELU_QUICK:
        case GGML_UNARY_OP_RELU:
        case GGML_UNARY_OP_TANH:
            ggml_vk_unary(ctx, compute_ctx, src0, node);
            break;
        default:
            return;
        }
        break;
    case GGML_OP_DIAG_MASK_INF:
        ggml_vk_diag_mask_inf(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_SOFT_MAX:
        ggml_vk_soft_max(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_ROPE:
        ggml_vk_rope(ctx, compute_ctx, src0, src1, src2, node);

        break;
    case GGML_OP_ARGSORT:
        ggml_vk_argsort(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_SUM_ROWS:
        ggml_vk_sum_rows(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_IM2COL:
        ggml_vk_im2col(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_TIMESTEP_EMBEDDING:
        ggml_vk_timestep_embedding(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_LEAKY_RELU:
        ggml_vk_leaky_relu(ctx, compute_ctx, src0, node);

        break;
    case GGML_OP_MUL_MAT:
        ggml_vk_mul_mat(ctx, compute_ctx, src0, src1, node);

        break;
    case GGML_OP_MUL_MAT_ID:
        ggml_vk_mul_mat_id(ctx, compute_ctx, src0, src1, src2, node);

        break;
    default:
        return;
    }

    ctx->tensor_ctxs[node_idx] = compute_ctx;

#ifdef GGML_VULKAN_CHECK_RESULTS
    // Force context reset on each node so that each tensor ends up in its own context
    // and can be run and compared to its CPU equivalent separately
    last_node = true;
#endif

    if (last_node) {
        ggml_vk_ctx_end(compute_ctx);
        compute_ctx->exit_tensor_idx = node_idx;
        ctx->compute_ctx.reset();
    }
}

static bool ggml_vk_compute_forward(ggml_backend_vk_context * ctx, ggml_tensor * tensor, int tensor_idx){
    ggml_tensor_extra_gpu * extra = nullptr;

    switch (tensor->op) {
    case GGML_OP_ADD:
    case GGML_OP_GET_ROWS:
    case GGML_OP_MUL:
    case GGML_OP_DIV:
    case GGML_OP_CONCAT:
    case GGML_OP_UPSCALE:
    case GGML_OP_SCALE:
    case GGML_OP_SQR:
    case GGML_OP_CLAMP:
    case GGML_OP_PAD:
    case GGML_OP_CPY:
    case GGML_OP_CONT:
    case GGML_OP_DUP:
    case GGML_OP_NORM:
    case GGML_OP_GROUP_NORM:
    case GGML_OP_RMS_NORM:
    case GGML_OP_DIAG_MASK_INF:
    case GGML_OP_SOFT_MAX:
    case GGML_OP_ROPE:
    case GGML_OP_RESHAPE:
    case GGML_OP_VIEW:
    case GGML_OP_PERMUTE:
    case GGML_OP_TRANSPOSE:
    case GGML_OP_NONE:
    case GGML_OP_ARGSORT:
    case GGML_OP_SUM_ROWS:
    case GGML_OP_IM2COL:
    case GGML_OP_TIMESTEP_EMBEDDING:
    case GGML_OP_LEAKY_RELU:
    case GGML_OP_REPEAT:
        extra = (ggml_tensor_extra_gpu *) tensor->extra;

        break;
    case GGML_OP_UNARY:
        switch (ggml_get_unary_op(tensor)) {
        case GGML_UNARY_OP_SILU:
        case GGML_UNARY_OP_GELU:
        case GGML_UNARY_OP_GELU_QUICK:
        case GGML_UNARY_OP_RELU:
        case GGML_UNARY_OP_TANH:
            extra = (ggml_tensor_extra_gpu *) tensor->extra;
            break;
        default:
            return false;
        }
        break;
    case GGML_OP_MUL_MAT:
    case GGML_OP_MUL_MAT_ID:
        extra = (ggml_tensor_extra_gpu *) tensor->extra;

        break;
    default:
        return false;
    }

    if (extra == nullptr) {
        return false;
    }

    VK_LOG_DEBUG("ggml_vk_compute_forward(" << tensor << ", name=" << tensor->name << ", op=" << ggml_op_name(tensor->op) << ", type=" << tensor->type << ", ne0=" << tensor->ne[0] << ", ne1=" << tensor->ne[1] << ", ne2=" << tensor->ne[2] << ", ne3=" << tensor->ne[3] << ", nb0=" << tensor->nb[0] << ", nb1=" << tensor->nb[1] << ", nb2=" << tensor->nb[2] << ", nb3=" << tensor->nb[3] << ", view_src=" << tensor->view_src << ", view_offs=" << tensor->view_offs << ")");

#ifdef GGML_VULKAN_CHECK_RESULTS
    ggml_vk_check_results_0(tensor);
#endif

    vk_context subctx = ctx->tensor_ctxs[tensor_idx].lock();

    // Only run if ctx hasn't been submitted yet
    if (!subctx->seqs.empty()) {
        // Do staging buffer copies
        for (auto& cpy : subctx->in_memcpys) {
            memcpy(cpy.dst, cpy.src, cpy.n);
        }

        ggml_vk_submit(subctx, ctx->fence);
    }

    if (tensor_idx == subctx->exit_tensor_idx) {
        VK_CHECK(ctx->device->device.waitForFences({ ctx->fence }, true, UINT64_MAX), "ggml_vk_compute_forward waitForFences");
        ctx->device->device.resetFences({ ctx->fence });

        // Do staging buffer copies
        for (auto& cpy : subctx->out_memcpys) {
            memcpy(cpy.dst, cpy.src, cpy.n);
        }
        subctx->in_memcpys.clear();
        subctx->out_memcpys.clear();
    }

    return true;
}

// Clean up after graph processing is done
static void ggml_vk_graph_cleanup(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_graph_cleanup()");
    for (auto& buffer : ctx->gc.temp_buffers) {
        ggml_vk_pool_free(ctx, buffer);
    }
    ctx->gc.temp_buffers.clear();

    for (auto& pipeline : ctx->device->pipelines) {
        if (pipeline.expired()) {
            continue;
        }

        vk_pipeline pl = pipeline.lock();
        ggml_pipeline_cleanup(pl);
    }

    ggml_vk_queue_cleanup(ctx->device, ctx->device->compute_queue);
    ggml_vk_queue_cleanup(ctx->device, ctx->device->transfer_queue);

    for (size_t i = 0; i < ctx->gc.semaphores.size(); i++) {
        ctx->device->device.destroySemaphore({ ctx->gc.semaphores[i].s });
    }
    ctx->gc.semaphores.clear();

    for (size_t i = 0; i < ctx->gc.tl_semaphores.size(); i++) {
        ctx->device->device.destroySemaphore({ ctx->gc.tl_semaphores[i].s });
    }
    ctx->gc.tl_semaphores.clear();
    ctx->semaphore_idx = 0;

    ctx->event_idx = 0;

    for (auto& event : ctx->gc.events) {
        ctx->device->device.resetEvent(event);
    }

    ctx->staging_offset = 0;

    ctx->tensor_ctxs.clear();
    ctx->gc.contexts.clear();
}

// Clean up on backend free
static void ggml_vk_cleanup(ggml_backend_vk_context * ctx) {
    VK_LOG_DEBUG("ggml_vk_cleanup(" << ctx->name << ")");
    ggml_vk_graph_cleanup(ctx);

    ggml_vk_destroy_buffer(ctx->prealloc_x);
    ggml_vk_destroy_buffer(ctx->prealloc_y);
    ggml_vk_destroy_buffer(ctx->prealloc_split_k);
    ggml_vk_destroy_buffer(ctx->staging);

    for (auto& buffer : ctx->buffer_pool) {
        ggml_vk_destroy_buffer(buffer);
    }

    ctx->prealloc_size_x = 0;
    ctx->prealloc_size_y = 0;
    ctx->prealloc_size_split_k = 0;
    ctx->staging_size = 0;

    for (auto& event : ctx->gc.events) {
        ctx->device->device.destroyEvent(event);
    }
    ctx->gc.events.clear();

    ctx->device->device.destroyFence(ctx->fence);
}

GGML_CALL static int ggml_vk_get_device_count() {
    ggml_vk_instance_init();

    return vk_instance.device_indices.size();
}

GGML_CALL static void ggml_vk_get_device_description(int device, char * description, size_t description_size) {
    ggml_vk_instance_init();

    std::vector<vk::PhysicalDevice> devices = vk_instance.instance.enumeratePhysicalDevices();

    vk::PhysicalDeviceProperties props;
    devices[device].getProperties(&props);

    snprintf(description, description_size, "%s", props.deviceName.data());
}

// backend interface

#define UNUSED GGML_UNUSED

// device backend

static void * const vk_ptr_base = (void *)(uintptr_t) 0x1000;  // NOLINT

struct ggml_backend_vk_buffer_context {
    vk_device_ref device;
    vk_buffer dev_buffer;
    ggml_tensor_extra_gpu * temp_tensor_extras = nullptr;
    size_t temp_tensor_extra_index = 0;
    std::string name;

    ggml_backend_vk_buffer_context(vk_device_ref device, vk_buffer&& dev_buffer, std::string& name) :
        device(device),
        dev_buffer(dev_buffer),
        name(name) {
    }

    ~ggml_backend_vk_buffer_context() {
        ggml_vk_destroy_buffer(dev_buffer);
        if (temp_tensor_extras != nullptr) {
            delete[] temp_tensor_extras;
        }
    }

    ggml_tensor_extra_gpu * ggml_vk_alloc_temp_tensor_extra() {
        if (temp_tensor_extras == nullptr) {
            temp_tensor_extras = new ggml_tensor_extra_gpu[GGML_VK_MAX_NODES];
        }

        size_t alloc_index = temp_tensor_extra_index;
        temp_tensor_extra_index = (temp_tensor_extra_index + 1) % GGML_VK_MAX_NODES;
        ggml_tensor_extra_gpu * extra = &temp_tensor_extras[alloc_index];
        extra->reset();

        return extra;
    }
};

GGML_CALL static const char * ggml_backend_vk_buffer_get_name(ggml_backend_buffer_t buffer) {
    ggml_backend_vk_buffer_context * ctx = (ggml_backend_vk_buffer_context *)buffer->context;
    return ctx->name.c_str();
}

GGML_CALL static bool ggml_backend_buffer_is_vk(ggml_backend_buffer_t buffer) {
    return buffer->iface.get_name == ggml_backend_vk_buffer_get_name;
}

GGML_CALL static void ggml_backend_vk_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    VK_LOG_MEMORY("ggml_backend_vk_buffer_free_buffer()");
    ggml_backend_vk_buffer_context * ctx = (ggml_backend_vk_buffer_context *)buffer->context;
    ggml_vk_destroy_buffer(ctx->dev_buffer);
    delete ctx;
}

GGML_CALL static void * ggml_backend_vk_buffer_get_base(ggml_backend_buffer_t buffer) {
    return vk_ptr_base;

    UNUSED(buffer);
}

GGML_CALL static void ggml_backend_vk_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_init_tensor(" << buffer << " (" << buffer->context << "), " << tensor << ")");
    ggml_backend_vk_buffer_context * ctx = (ggml_backend_vk_buffer_context *)buffer->context;

    if (tensor->view_src != nullptr) {
        GGML_ASSERT(tensor->view_src->buffer->buft == buffer->buft);
        GGML_ASSERT(tensor->view_src->extra != nullptr);
        tensor->extra = tensor->view_src->extra;
    } else {
        ggml_tensor_extra_gpu * extra = ctx->ggml_vk_alloc_temp_tensor_extra();
        extra->buffer_gpu = ctx->dev_buffer;
        extra->offset = (uint8_t *) tensor->data - (uint8_t *) vk_ptr_base;
        tensor->extra = extra;
    }
}

GGML_CALL static void ggml_backend_vk_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_set_tensor(" << buffer << ", " << tensor << ", " << data << ", " << offset << ", " << size << ")");
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) tensor->extra;

    vk_buffer buf = extra->buffer_gpu.lock();

    ggml_vk_buffer_write(buf, extra->offset + tensor->view_offs + offset, data, size);

    GGML_UNUSED(buffer);
}

GGML_CALL static void ggml_backend_vk_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_buffer_get_tensor(" << buffer << ", " << tensor << ", " << data << ", " << offset << ", " << size << ")");
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) tensor->extra;

    vk_buffer buf = extra->buffer_gpu.lock();

    ggml_vk_buffer_read(buf, extra->offset + tensor->view_offs + offset, data, size);

    GGML_UNUSED(buffer);
}

GGML_CALL static bool ggml_backend_vk_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_vk(src->buffer)) {
        ggml_tensor_extra_gpu * src_extra = (ggml_tensor_extra_gpu *) src->extra;
        ggml_tensor_extra_gpu * dst_extra = (ggml_tensor_extra_gpu *) dst->extra;

        vk_buffer src_buf = src_extra->buffer_gpu.lock();
        vk_buffer dst_buf = dst_extra->buffer_gpu.lock();

        ggml_vk_buffer_copy(dst_buf, dst_extra->offset + dst->view_offs, src_buf, src_extra->offset + src->view_offs, ggml_nbytes(src));

        return true;
    }
    return false;

    UNUSED(buffer);
}

GGML_CALL static void ggml_backend_vk_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_vk_buffer_context * ctx = (ggml_backend_vk_buffer_context *)buffer->context;

    ggml_vk_buffer_memset(ctx->dev_buffer, 0, value, buffer->size);
}

static ggml_backend_buffer_i ggml_backend_vk_buffer_interface = {
    /* .get_name        = */ ggml_backend_vk_buffer_get_name,
    /* .free_buffer     = */ ggml_backend_vk_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_vk_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_vk_buffer_init_tensor,
    /* .set_tensor      = */ ggml_backend_vk_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_vk_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_vk_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_vk_buffer_clear,
    /* .reset           = */ NULL,
};

// vk buffer type
GGML_CALL static const char * ggml_backend_vk_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_vk_buffer_type_context * ctx = (ggml_backend_vk_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

GGML_CALL static ggml_backend_buffer_t ggml_backend_vk_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    VK_LOG_MEMORY("ggml_backend_vk_buffer_type_alloc_buffer(" << size << ")");
    ggml_backend_vk_buffer_type_context * ctx = (ggml_backend_vk_buffer_type_context *) buft->context;

    vk_buffer dev_buffer = nullptr;
    try {
        dev_buffer = ggml_vk_create_buffer_device(ctx->device, size);
    } catch (const vk::SystemError& e) {
        return nullptr;
    }

    ggml_backend_vk_buffer_context * bufctx = new ggml_backend_vk_buffer_context(ctx->device, std::move(dev_buffer), ctx->name);

    return ggml_backend_buffer_init(buft, ggml_backend_vk_buffer_interface, bufctx, size);
}

GGML_CALL static size_t ggml_backend_vk_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ggml_backend_vk_buffer_type_context * ctx = (ggml_backend_vk_buffer_type_context *) buft->context;
    return ctx->device->properties.limits.minStorageBufferOffsetAlignment;
}

GGML_CALL static size_t ggml_backend_vk_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_vk_buffer_type_context * ctx = (ggml_backend_vk_buffer_type_context *) buft->context;
    return ctx->device->max_memory_allocation_size;
}

GGML_CALL static size_t ggml_backend_vk_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    return ggml_nbytes(tensor);

    UNUSED(buft);
}

GGML_CALL ggml_backend_buffer_type_t ggml_backend_vk_buffer_type(size_t dev_num) {
    ggml_vk_instance_init();

    VK_LOG_DEBUG("ggml_backend_vk_buffer_type(" << dev_num << ")");

    vk_device dev = ggml_vk_get_device(dev_num);

    return &dev->buffer_type;
}

// host buffer type

GGML_CALL static const char * ggml_backend_vk_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_VK_NAME "_Host";

    UNUSED(buft);
}

GGML_CALL static const char * ggml_backend_vk_host_buffer_name(ggml_backend_buffer_t buffer) {
    return GGML_VK_NAME "_Host";

    UNUSED(buffer);
}

GGML_CALL static void ggml_backend_vk_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    VK_LOG_MEMORY("ggml_backend_vk_host_buffer_free_buffer()");
    ggml_vk_host_free(vk_instance.devices[0], buffer->context);
}

GGML_CALL static ggml_backend_buffer_t ggml_backend_vk_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    VK_LOG_MEMORY("ggml_backend_vk_host_buffer_type_alloc_buffer(" << size << ")");

    size += 32;  // Behave like the CPU buffer type
    void * ptr = nullptr;
    try {
        ptr = ggml_vk_host_malloc(vk_instance.devices[0], size);
    } catch (vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Failed to allocate pinned memory." << std::endl;
        std::cerr << "ggml_vulkan: " << e.what() << std::endl;
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.get_name = ggml_backend_vk_host_buffer_name;
    buffer->iface.free_buffer = ggml_backend_vk_host_buffer_free_buffer;

    return buffer;

    UNUSED(buft);
}

GGML_CALL static size_t ggml_backend_vk_host_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return vk_instance.devices[0]->properties.limits.minMemoryMapAlignment;

    UNUSED(buft);
}

// Should be changed to return device-specific host buffer type
// but that probably requires changes in llama.cpp
GGML_CALL ggml_backend_buffer_type_t ggml_backend_vk_host_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_vk_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name         = */ ggml_backend_vk_host_buffer_type_name,
            /* .alloc_buffer     = */ ggml_backend_vk_host_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_vk_host_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .is_host          = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .context  = */ nullptr,
    };

    // Make sure device 0 is initialized
    ggml_vk_instance_init();
    ggml_vk_get_device(0);

    return &ggml_backend_vk_buffer_type_host;
}


// backend

GGML_CALL static const char * ggml_backend_vk_name(ggml_backend_t backend) {
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    return ctx->name.c_str();
}

GGML_CALL static void ggml_backend_vk_free(ggml_backend_t backend) {
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    VK_LOG_DEBUG("ggml_backend_vk_free(" << ctx->name << ")");

    ggml_vk_cleanup(ctx);

    delete ctx;
    delete backend;
}

GGML_CALL static ggml_backend_buffer_type_t ggml_backend_vk_get_default_buffer_type(ggml_backend_t backend) {
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    return &ctx->device->buffer_type;
}

GGML_CALL static void ggml_backend_vk_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_set_tensor_async(" << size << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    GGML_ASSERT((tensor->buffer->buft == ggml_backend_vk_get_default_buffer_type(backend) || tensor->buffer->buft == ggml_backend_vk_host_buffer_type()) && "unsupported buffer type");

    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) tensor->extra;

    vk_context transfer_ctx;

    if (ctx->transfer_ctx.expired()) {
        // Initialize new transfer context
        transfer_ctx = ggml_vk_create_context(ctx, ctx->device->transfer_queue);
        ctx->transfer_ctx = transfer_ctx;
        ggml_vk_ctx_begin(ctx->device, transfer_ctx);
    } else {
        transfer_ctx = ctx->transfer_ctx.lock();
    }

    vk_buffer buf = extra->buffer_gpu.lock();

    ggml_vk_buffer_write_async(transfer_ctx, buf, extra->offset + tensor->view_offs + offset, data, size, ctx->staging, ctx->staging_offset);
}

GGML_CALL static void ggml_backend_vk_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    VK_LOG_DEBUG("ggml_backend_vk_get_tensor_async(" << size << ")");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    GGML_ASSERT((tensor->buffer->buft == ggml_backend_vk_get_default_buffer_type(backend) || tensor->buffer->buft == ggml_backend_vk_host_buffer_type()) && "unsupported buffer type");

    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) tensor->extra;

    vk_context transfer_ctx;

    if (ctx->transfer_ctx.expired()) {
        // Initialize new transfer context
        transfer_ctx = ggml_vk_create_context(ctx, ctx->device->transfer_queue);
        ctx->transfer_ctx = transfer_ctx;
        ggml_vk_ctx_begin(ctx->device, transfer_ctx);
    } else {
        transfer_ctx = ctx->transfer_ctx.lock();
    }

    vk_buffer buf = extra->buffer_gpu.lock();

    ggml_vk_buffer_read_async(transfer_ctx, buf, extra->offset + tensor->view_offs + offset, data, size, ctx->staging, ctx->staging_offset);
}

GGML_CALL static bool ggml_backend_vk_cpy_tensor_async(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst) {
    VK_LOG_DEBUG("ggml_backend_vk_cpy_tensor_async()");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    if ((dst->buffer->buft == ggml_backend_vk_get_default_buffer_type(backend) || dst->buffer->buft == ggml_backend_vk_host_buffer_type()) && ggml_backend_buffer_is_vk(src->buffer)) {
        ggml_tensor_extra_gpu * src_extra = (ggml_tensor_extra_gpu *) src->extra;
        ggml_tensor_extra_gpu * dst_extra = (ggml_tensor_extra_gpu *) dst->extra;

        vk_context transfer_ctx;

        if (ctx->transfer_ctx.expired()) {
            // Initialize new transfer context
            transfer_ctx = ggml_vk_create_context(ctx, ctx->device->transfer_queue);
            ctx->transfer_ctx = transfer_ctx;
            ggml_vk_ctx_begin(ctx->device, transfer_ctx);
        } else {
            transfer_ctx = ctx->transfer_ctx.lock();
        }

        vk_buffer src_buf = src_extra->buffer_gpu.lock();
        vk_buffer dst_buf = dst_extra->buffer_gpu.lock();

        ggml_vk_buffer_copy_async(transfer_ctx, dst_buf, dst_extra->offset + dst->view_offs, src_buf, src_extra->offset + src->view_offs, ggml_nbytes(src));
        return true;
    }

    return false;
}

GGML_CALL static void ggml_backend_vk_synchronize(ggml_backend_t backend) {
    VK_LOG_DEBUG("ggml_backend_vk_synchronize()");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;
    if(ctx->transfer_ctx.expired()) {
        return;
    }

    vk_context transfer_ctx = ctx->transfer_ctx.lock();

    ggml_vk_ctx_end(transfer_ctx);

    for (auto& cpy : transfer_ctx->in_memcpys) {
        memcpy(cpy.dst, cpy.src, cpy.n);
    }

    ggml_vk_submit(transfer_ctx, ctx->fence);
    VK_CHECK(ctx->device->device.waitForFences({ ctx->fence }, true, UINT64_MAX), "ggml_backend_vk_synchronize waitForFences");
    ctx->device->device.resetFences({ ctx->fence });

    for (auto& cpy : transfer_ctx->out_memcpys) {
        memcpy(cpy.dst, cpy.src, cpy.n);
    }

    ctx->transfer_ctx.reset();
}

static bool ggml_vk_is_empty(ggml_tensor * node) {
    return ggml_is_empty(node) || node->op == GGML_OP_NONE || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE;
}

GGML_CALL static ggml_status ggml_backend_vk_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    VK_LOG_DEBUG("ggml_backend_vk_graph_compute(" << cgraph->n_nodes << " nodes)");
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_vk_preallocate_buffers_graph(ctx, cgraph->nodes[i]);
    }
    ggml_vk_preallocate_buffers(ctx);

    int last_node = cgraph->n_nodes - 1;

    // If the last op in the cgraph isn't backend GPU, the command buffer doesn't get closed properly
    while (last_node > 0 && ggml_vk_is_empty(cgraph->nodes[last_node])) {
        last_node -= 1;
    }

    // Reserve tensor context space for all nodes
    ctx->tensor_ctxs.resize(cgraph->n_nodes);

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_vk_build_graph(ctx, cgraph->nodes[i], i, i == last_node);
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if (ggml_vk_is_empty(node)) {
            continue;
        }

        bool ok = ggml_vk_compute_forward(ctx, node, i);
        if (!ok) {
            if (node->op == GGML_OP_UNARY) {
                std::cerr << __func__ << ": error: op not supported UNARY " << node->name << " (" << ggml_unary_op_name(static_cast<ggml_unary_op>(node->op_params[0])) << ")" << std::endl;
            } else {
                std::cerr << __func__ << ": error: op not supported " << node->name << " (" << ggml_op_name(node->op) << ")" << std::endl;
            }
        }
#ifdef GGML_VULKAN_CHECK_RESULTS
        else {
            ggml_vk_check_results_1(node);
        }
#endif
        GGML_ASSERT(ok);
    }

    ggml_vk_graph_cleanup(ctx);

    return GGML_STATUS_SUCCESS;

    UNUSED(backend);
}

GGML_CALL static bool ggml_backend_vk_supports_op(ggml_backend_t backend, const ggml_tensor * op) {
    // ggml_backend_vk_context * ctx = (ggml_backend_vk_context *) backend->context;

    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_TANH:
                    return ggml_is_contiguous(op->src[0]);
                default:
                    return false;
            }
            break;
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_IQ4_NL:
                        break;
                    default:
                        return false;
                }
                struct ggml_tensor * a;
                struct ggml_tensor * b;
                if (op->op == GGML_OP_MUL_MAT) {
                    a = op->src[0];
                    b = op->src[1];
                } else {
                    a = op->src[2];
                    b = op->src[1];
                }
                if (a->ne[3] != b->ne[3]) {
                    return false;
                }
                return true;
            } break;
        case GGML_OP_GET_ROWS:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_IQ4_NL:
                        return true;
                    default:
                        return false;
                }
            } break;
        case GGML_OP_CPY:
        case GGML_OP_DUP:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1] != nullptr ? op->src[1]->type : src0_type;
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F16) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F16) {
                    return true;
                }
                return false;
            } break;
        case GGML_OP_REPEAT:
            {
                ggml_type src0_type = op->src[0]->type;
                return src0_type != GGML_TYPE_I32 && src0_type != GGML_TYPE_I16;
            } break;
        case GGML_OP_ROPE:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_NORM:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_CONCAT:
        case GGML_OP_UPSCALE:
        case GGML_OP_SCALE:
        case GGML_OP_SQR:
        case GGML_OP_CLAMP:
        case GGML_OP_PAD:
        case GGML_OP_CONT:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_ARGSORT:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_IM2COL:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_LEAKY_RELU:
            return true;
        default:
            return false;
    }

    UNUSED(backend);
}

GGML_CALL static bool ggml_backend_vk_offload_op(ggml_backend_t backend, const ggml_tensor * op) {
    const int min_batch_size = 32;

    return (op->ne[1] >= min_batch_size && op->op != GGML_OP_GET_ROWS) ||
           (op->ne[2] >= min_batch_size && op->op == GGML_OP_MUL_MAT_ID);

    UNUSED(backend);
}

GGML_CALL static bool ggml_backend_vk_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    if (buft->iface.get_name != ggml_backend_vk_buffer_type_name) {
        return false;
    }

    ggml_backend_vk_buffer_type_context * buft_ctx = (ggml_backend_vk_buffer_type_context *)buft->context;
    ggml_backend_vk_context * ctx = (ggml_backend_vk_context *)backend->context;

    return buft_ctx->device == ctx->device;
}

// TODO: enable async and synchronize
static ggml_backend_i ggml_backend_vk_interface = {
    /* .get_name                = */ ggml_backend_vk_name,
    /* .free                    = */ ggml_backend_vk_free,
    /* .get_default_buffer_type = */ ggml_backend_vk_get_default_buffer_type,
    /* .set_tensor_async        = */ NULL,  // ggml_backend_vk_set_tensor_async,
    /* .get_tensor_async        = */ NULL,  // ggml_backend_vk_get_tensor_async,
    /* .cpy_tensor_async        = */ NULL,  // ggml_backend_vk_cpy_tensor_async,
    /* .synchronize             = */ NULL,  // ggml_backend_vk_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_vk_graph_compute,
    /* .supports_op             = */ ggml_backend_vk_supports_op,
    /* .supports_buft           = */ ggml_backend_vk_supports_buft,
    /* .offload_op              = */ ggml_backend_vk_offload_op,
    /* .event_new               = */ NULL,
    /* .event_free              = */ NULL,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .event_synchronize       = */ NULL,
};

static ggml_guid_t ggml_backend_vk_guid() {
    static ggml_guid guid = { 0xb8, 0xf7, 0x4f, 0x86, 0x40, 0x3c, 0xe1, 0x02, 0x91, 0xc8, 0xdd, 0xe9, 0x02, 0x3f, 0xc0, 0x2b };
    return &guid;
}

GGML_CALL ggml_backend_t ggml_backend_vk_init(size_t dev_num) {
    VK_LOG_DEBUG("ggml_backend_vk_init(" << dev_num << ")");

    ggml_backend_vk_context * ctx = new ggml_backend_vk_context;
    ggml_vk_init(ctx, dev_num);

    ggml_backend_t vk_backend = new ggml_backend {
        /* .guid      = */ ggml_backend_vk_guid(),
        /* .interface = */ ggml_backend_vk_interface,
        /* .context   = */ ctx,
    };

    return vk_backend;
}

GGML_CALL bool ggml_backend_is_vk(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_vk_guid());
}

GGML_CALL int ggml_backend_vk_get_device_count() {
    return ggml_vk_get_device_count();
}

GGML_CALL void ggml_backend_vk_get_device_description(int device, char * description, size_t description_size) {
    ggml_vk_get_device_description(device, description, description_size);
}

GGML_CALL void ggml_backend_vk_get_device_memory(int device, size_t * free, size_t * total) {
    GGML_ASSERT(device < (int) vk_instance.device_indices.size());

    vk::PhysicalDevice vkdev = vk_instance.instance.enumeratePhysicalDevices()[vk_instance.device_indices[device]];

    vk::PhysicalDeviceMemoryProperties memprops = vkdev.getMemoryProperties();

    for (const vk::MemoryHeap& heap : memprops.memoryHeaps) {
        if (heap.flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
            *total = heap.size;
            *free = heap.size;
            break;
        }
    }
}

// backend registry
GGML_CALL static ggml_backend_t ggml_backend_reg_vk_init(const char * params, void * user_data) {
    ggml_backend_t vk_backend = ggml_backend_vk_init((int) (intptr_t) user_data);
    return vk_backend;

    UNUSED(params);
}

extern "C" GGML_CALL int ggml_backend_vk_reg_devices();

GGML_CALL int ggml_backend_vk_reg_devices() {
    ggml_vk_instance_init();

    for (size_t i = 0; i < vk_instance.device_indices.size(); i++) {
        char name[128];
        snprintf(name, sizeof(name), "%s%ld", GGML_VK_NAME, i);
        ggml_backend_register(name, ggml_backend_reg_vk_init, ggml_backend_vk_buffer_type(i), (void *) (intptr_t) i);  // NOLINT
    }
    return vk_instance.device_indices.size();
}

// Extension availability
static bool ggml_vk_instance_validation_ext_available(const std::vector<vk::ExtensionProperties>& instance_extensions) {
#ifdef GGML_VULKAN_VALIDATE
    bool portability_enumeration_ext = false;
    // Check for portability enumeration extension for MoltenVK support
    for (const auto& properties : instance_extensions) {
        if (strcmp("VK_KHR_portability_enumeration", properties.extensionName) == 0) {
            return true;
        }
    }
    if (!portability_enumeration_ext) {
        std::cerr << "ggml_vulkan: WARNING: Instance extension VK_KHR_portability_enumeration not found." << std::endl;
    }
#endif
    return false;

    UNUSED(instance_extensions);
}
static bool ggml_vk_instance_portability_enumeration_ext_available(const std::vector<vk::ExtensionProperties>& instance_extensions) {
#ifdef __APPLE__
    bool portability_enumeration_ext = false;
    // Check for portability enumeration extension for MoltenVK support
    for (const auto& properties : instance_extensions) {
        if (strcmp("VK_KHR_portability_enumeration", properties.extensionName) == 0) {
            return true;
        }
    }
    if (!portability_enumeration_ext) {
        std::cerr << "ggml_vulkan: WARNING: Instance extension VK_KHR_portability_enumeration not found." << std::endl;
    }
#endif
    return false;

    UNUSED(instance_extensions);
}

// checks

#ifdef GGML_VULKAN_CHECK_RESULTS
static void ggml_vk_print_graph_origin(const ggml_tensor * tensor, std::vector<const ggml_tensor *>& done, int level = 0) {
    if (std::find(done.begin(), done.end(), tensor) != done.end() || level > 10) {
        return;
    }
    for (int j = 0; j < level; j++) {
        std::cerr << " ";
    }
    std::cerr << ggml_op_name(tensor->op) << " gpu=" << (tensor->extra != nullptr) << std::endl;

    done.push_back(tensor);

    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (tensor->src[i] != nullptr) {
            ggml_vk_print_graph_origin(tensor->src[i], done, level + 1);
        }
    }
}

static void ggml_vk_print_tensor_area(const ggml_tensor * tensor, const void * data, int i0, int i1, int i2, int i3) {
    if (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16 && tensor->type != GGML_TYPE_I32) {
        return;
    }
    i0 = std::max(i0, 5);
    i1 = std::max(i1, 5);
    i2 = std::max(i2, 0);
    i3 = std::max(i3, 0);
    fprintf(stderr, "         ");
    for (int idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
        fprintf(stderr, "%7d ", idx1);
    }
    fprintf(stderr, "\n");
    for (int idx0 = i0 - 5; idx0 < i0 + 5; idx0++) {
        fprintf(stderr, "%7d: ", idx0);
        for (int idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
            if (idx0 >= 0 && idx0 < tensor->ne[0] && idx1 >= 0 && idx1 < tensor->ne[1] && i2 >= 0 && i2 < tensor->ne[2] && i3 >= 0 && i3 < tensor->ne[3]) {
                float val;
                if (tensor->type == GGML_TYPE_F32) {
                    val = *(const float *) ((const char *) data + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]);
                } else if (tensor->type == GGML_TYPE_F16) {
                    val = ggml_fp16_to_fp32(*(const ggml_fp16_t *) ((const char *) data + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]));
                } else if (tensor->type == GGML_TYPE_I32) {
                    val = *(const int32_t *) ((const char *) data + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]);
                } else {
                    GGML_ABORT("fatal error");
                }
                fprintf(stderr, "% 7.2f ", val);
            } else {
                fprintf(stderr, "        ");
            }
        }
        fprintf(stderr, "\n");
    }
}

static void ggml_vk_print_tensor(const ggml_tensor * tensor, const char * name) {
    void * tensor_data = tensor->data;

    const bool is_gpu = tensor->buffer != nullptr && ggml_backend_buffer_is_vk(tensor->buffer);

    if (is_gpu) {
        const size_t tensor_size = ggml_nbytes(tensor);
        tensor_data = malloc(tensor_size);

        ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) tensor->extra;

        vk_buffer buffer_gpu = extra->buffer_gpu.lock();
        ggml_vk_buffer_read(buffer_gpu, extra->offset + tensor->view_offs, tensor_data, tensor_size);
    }

    std::cerr << "TENSOR CHECK " << name << " (" << tensor->name << "): " << ggml_op_name(tensor->op) << std::endl;
    std::cerr << "tensor=" << tensor << " tensor->type: " << ggml_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] << " nb3=" << tensor->nb[3] << std::endl;
    if (tensor->src[0] != nullptr) {
        std::cerr << "tensor->src[0]=" << tensor->src[0] << " name=" << tensor->src[0]->name << " op=" << ggml_op_name(tensor->src[0]->op) << " type=" << ggml_type_name(tensor->src[0]->type) << " ne0=" << tensor->src[0]->ne[0] << " nb0=" << tensor->src[0]->nb[0] << " ne1=" << tensor->src[0]->ne[1] << " nb1=" << tensor->src[0]->nb[1] << " ne2=" << tensor->src[0]->ne[2] << " nb2=" << tensor->src[0]->nb[2] << " ne3=" << tensor->src[0]->ne[3] << " nb3=" << tensor->src[0]->nb[3] << std::endl;
    }
    if (tensor->src[1] != nullptr) {
        std::cerr << "tensor->src[1]=" << tensor->src[1] << " name=" << tensor->src[1]->name << " op=" << ggml_op_name(tensor->src[1]->op) << " type=" << ggml_type_name(tensor->src[1]->type) << " ne0=" << tensor->src[1]->ne[0] << " nb0=" << tensor->src[1]->nb[0] << " ne1=" << tensor->src[1]->ne[1] << " nb1=" << tensor->src[1]->nb[1] << " ne2=" << tensor->src[1]->ne[2] << " nb2=" << tensor->src[1]->nb[2] << " ne3=" << tensor->src[1]->ne[3] << " nb3=" << tensor->src[1]->nb[3] << std::endl;
    }
    std::cerr << std::endl << "Result:" << std::endl;
    ggml_vk_print_tensor_area(tensor, tensor_data, 5, 5, 0, 0);
    std::cerr << std::endl;
    std::vector<const ggml_tensor *> done;
    ggml_vk_print_graph_origin(tensor, done);

    if (is_gpu) {
        free(tensor_data);
    }
}

void * comp_result;
size_t comp_size;
size_t comp_nb[GGML_MAX_DIMS];
size_t check_counter = 0;
static void ggml_vk_check_results_0(ggml_tensor * tensor) {
    if (tensor->op == GGML_OP_TRANSPOSE) {
        return;
    }

    check_counter++;
    if (!(vk_output_tensor > 0 && vk_output_tensor == check_counter) && check_counter <= vk_skip_checks) {
        return;
    }

    VK_LOG_DEBUG("ggml_vk_check_results_0(" << tensor->name << ")");

    ggml_tensor * src0 = tensor->src[0];
    ggml_tensor * src1 = tensor->src[1];
    ggml_tensor * src2 = tensor->src[2];

    struct ggml_init_params iparams = {
        /*.mem_size   =*/ 2ul*1024ul*1024ul*1024ul,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ggml_ctx = ggml_init(iparams);

    struct ggml_tensor * src0_clone = nullptr;
    struct ggml_tensor * src1_clone = nullptr;
    struct ggml_tensor * src2_clone = nullptr;
    struct ggml_tensor * tensor_clone = nullptr;

    size_t src0_size;
    size_t src1_size;
    size_t src2_size;

    void * src0_buffer = nullptr;
    void * src1_buffer = nullptr;
    void * src2_buffer = nullptr;

    if (src0 != nullptr) {
        src0_clone = ggml_dup_tensor(ggml_ctx, src0);

        src0_size = ggml_nbytes(src0);

        src0_buffer = malloc(src0_size);
        src0_clone->data = src0_buffer;
        if (ggml_backend_buffer_is_host(src0->buffer)) {
            memcpy(src0_clone->data, src0->data, src0_size);
            memcpy(src0_clone->nb, src0->nb, sizeof(size_t) * GGML_MAX_DIMS);
        } else if (ggml_backend_buffer_is_vk(src0->buffer)) {
            ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) src0->extra;
            vk_buffer buffer_gpu = extra->buffer_gpu.lock();
            uint64_t offset = extra->offset + src0->view_offs;
            if (!ggml_is_contiguous(src0) && ggml_vk_dim01_contiguous(src0)) {
                for (int i3 = 0; i3 < src0->ne[3]; i3++) {
                    for (int i2 = 0; i2 < src0->ne[2]; i2++) {
                        const int idx = i3*src0->ne[2] + i2;
                        ggml_vk_buffer_read(buffer_gpu, offset + idx * src0->nb[2], ((char *)src0_clone->data + idx * src0_clone->nb[2]), src0->ne[1] * src0->nb[1]);
                    }
                }

                src0_clone->nb[0] = src0->nb[0];
                src0_clone->nb[1] = src0->nb[1];
                for (int i = 2; i < GGML_MAX_DIMS; i++) {
                    src0_clone->nb[i] = src0_clone->nb[i - 1]*src0_clone->ne[i - 1];
                }
            } else {
                if (offset + src0_size >= buffer_gpu->size) {
                    src0_size = buffer_gpu->size - offset;
                }
                ggml_vk_buffer_read(buffer_gpu, offset, src0_clone->data, src0_size);
                memcpy(src0_clone->nb, src0->nb, sizeof(size_t) * GGML_MAX_DIMS);
            }
        } else {
            GGML_ABORT("fatal error");
        }

        if (vk_output_tensor > 0 && vk_output_tensor == check_counter) {
            ggml_vk_print_tensor(src0, "src0");
        }
    }
    if (src1 != nullptr) {
        src1_clone = ggml_dup_tensor(ggml_ctx, src1);

        src1_size = ggml_nbytes(src1);

        src1_buffer = malloc(src1_size);
        src1_clone->data = src1_buffer;
        if (ggml_backend_buffer_is_host(src1->buffer)) {
            memcpy(src1_clone->data, src1->data, src1_size);
            memcpy(src1_clone->nb, src1->nb, sizeof(size_t) * GGML_MAX_DIMS);
        } else if (ggml_backend_buffer_is_vk(src1->buffer)) {
            ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) src1->extra;
            vk_buffer buffer_gpu = extra->buffer_gpu.lock();
            uint64_t offset = extra->offset + src1->view_offs;
            if (!ggml_is_contiguous(src1) && ggml_vk_dim01_contiguous(src1)) {
                for (int i3 = 0; i3 < src1->ne[3]; i3++) {
                    for (int i2 = 0; i2 < src1->ne[2]; i2++) {
                        const int idx = i3*src1->ne[2] + i2;
                        ggml_vk_buffer_read(buffer_gpu, offset + idx * src1->nb[2], ((char *)src1_clone->data + idx * src1_clone->nb[2]), src1->ne[1] * src1->nb[1]);
                    }
                }

                src1_clone->nb[0] = src1->nb[0];
                src1_clone->nb[1] = src1->nb[1];
                for (int i = 2; i < GGML_MAX_DIMS; i++) {
                    src1_clone->nb[i] = src1_clone->nb[i - 1]*src1_clone->ne[i - 1];
                }
            } else {
                if (offset + src1_size >= buffer_gpu->size) {
                    src1_size = buffer_gpu->size - offset;
                }
                ggml_vk_buffer_read(buffer_gpu, offset, src1_clone->data, src1_size);
                memcpy(src1_clone->nb, src1->nb, sizeof(size_t) * GGML_MAX_DIMS);
            }
        } else {
            GGML_ABORT("fatal error");
        }

        if (vk_output_tensor > 0 && vk_output_tensor == check_counter) {
            ggml_vk_print_tensor(src1, "src1");
        }
    }
    if (src2 != nullptr) {
        src2_clone = ggml_dup_tensor(ggml_ctx, src2);

        src2_size = ggml_nbytes(src2);

        src2_buffer = malloc(src2_size);
        src2_clone->data = src2_buffer;
        if (ggml_backend_buffer_is_host(src2->buffer)) {
            memcpy(src2_clone->data, src2->data, src2_size);
            memcpy(src2_clone->nb, src2->nb, sizeof(size_t) * GGML_MAX_DIMS);
        } else if (ggml_backend_buffer_is_vk(src2->buffer)) {
            ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) src2->extra;
            vk_buffer buffer_gpu = extra->buffer_gpu.lock();
            uint64_t offset = extra->offset + src2->view_offs;
            if (!ggml_is_contiguous(src2) && ggml_vk_dim01_contiguous(src2)) {
                for (int i3 = 0; i3 < src2->ne[3]; i3++) {
                    for (int i2 = 0; i2 < src2->ne[2]; i2++) {
                        const int idx = i3*src2->ne[2] + i2;
                        ggml_vk_buffer_read(buffer_gpu, offset + idx * src2->nb[2], ((char *)src2_clone->data + idx * src2_clone->nb[2]), src2->ne[1] * src2->nb[1]);
                    }
                }

                src2_clone->nb[0] = src2->nb[0];
                src2_clone->nb[1] = src2->nb[1];
                for (int i = 2; i < GGML_MAX_DIMS; i++) {
                    src2_clone->nb[i] = src2_clone->nb[i - 1]*src2_clone->ne[i - 1];
                }
            } else {
                if (offset + src2_size >= buffer_gpu->size) {
                    src2_size = buffer_gpu->size - offset;
                }
                ggml_vk_buffer_read(buffer_gpu, offset, src2_clone->data, src2_size);
                memcpy(src2_clone->nb, src2->nb, sizeof(size_t) * GGML_MAX_DIMS);
            }
        } else {
            GGML_ABORT("fatal error");
        }

        if (vk_output_tensor > 0 && vk_output_tensor == check_counter) {
            ggml_vk_print_tensor(src2, "src2");
        }
    }

    if (tensor->op == GGML_OP_MUL_MAT) {
        tensor_clone = ggml_mul_mat(ggml_ctx, src0_clone, src1_clone);
    } else if (tensor->op == GGML_OP_MUL_MAT_ID) {
        tensor_clone = ggml_mul_mat_id(ggml_ctx, src0_clone, src1_clone, src2_clone);
    } else if (tensor->op == GGML_OP_MUL) {
        tensor_clone = ggml_mul(ggml_ctx, src0_clone, src1_clone);
    } else if (tensor->op == GGML_OP_DIV) {
        tensor_clone = ggml_div(ggml_ctx, src0_clone, src1_clone);
    } else if (tensor->op == GGML_OP_CONCAT) {
        tensor_clone = ggml_concat(ggml_ctx, src0_clone, src1_clone, *(int *)tensor->op_params);
    } else if (tensor->op == GGML_OP_UPSCALE) {
        tensor_clone = ggml_upscale_ext(ggml_ctx, src0_clone, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    } else if (tensor->op == GGML_OP_SCALE) {
        tensor_clone = ggml_scale(ggml_ctx, src0_clone, ((float *)tensor->op_params)[0]);
    } else if (tensor->op == GGML_OP_SQR) {
        tensor_clone = ggml_sqr(ggml_ctx, src0_clone);
    } else if (tensor->op == GGML_OP_CLAMP) {
        tensor_clone = ggml_clamp(ggml_ctx, src0_clone, ((float *)tensor->op_params)[0], ((float *)tensor->op_params)[1]);
    } else if (tensor->op == GGML_OP_PAD) {
        tensor_clone = ggml_pad(ggml_ctx, src0_clone, tensor->ne[0] - src0_clone->ne[0], tensor->ne[1] - src0_clone->ne[1], tensor->ne[2] - src0_clone->ne[2], tensor->ne[3] - src0_clone->ne[3]);
    } else if (tensor->op == GGML_OP_ADD) {
        tensor_clone = ggml_add(ggml_ctx, src0_clone, src1_clone);
    } else if (tensor->op == GGML_OP_NORM) {
        tensor_clone = ggml_norm(ggml_ctx, src0_clone, *(float *)tensor->op_params);
    } else if (tensor->op == GGML_OP_GROUP_NORM) {
        tensor_clone = ggml_group_norm(ggml_ctx, src0_clone, *(int *)tensor->op_params);
    } else if (tensor->op == GGML_OP_RMS_NORM) {
        tensor_clone = ggml_rms_norm(ggml_ctx, src0_clone, *(float *)tensor->op_params);
    } else if (tensor->op == GGML_OP_SOFT_MAX) {
        if (src1 != nullptr) {
            tensor_clone = ggml_soft_max_ext(ggml_ctx, src0_clone, src1_clone, ((float *)tensor->op_params)[0], ((float *)tensor->op_params)[1]);
        } else {
            tensor_clone = ggml_soft_max(ggml_ctx, src0_clone);
        }
    } else if (tensor->op == GGML_OP_DIAG_MASK_INF) {
        tensor_clone = ggml_diag_mask_inf(ggml_ctx, src0_clone, *(int *)tensor->op_params);
    } else if (tensor->op == GGML_OP_ROPE) {
        const int n_dims      = ((int32_t *) tensor->op_params)[1];
        const int mode        = ((int32_t *) tensor->op_params)[2];
        //const int n_ctx_ggml       = ((int32_t *) tensor->op_params)[3];
        const int n_ctx_orig_ggml  = ((int32_t *) tensor->op_params)[4];
        const float freq_base       = ((float *) tensor->op_params)[5];
        const float freq_scale      = ((float *) tensor->op_params)[6];
        const float ext_factor      = ((float *) tensor->op_params)[7];
        const float attn_factor     = ((float *) tensor->op_params)[8];
        const float beta_fast       = ((float *) tensor->op_params)[9];
        const float beta_slow       = ((float *) tensor->op_params)[10];
        tensor_clone = ggml_rope_ext(ggml_ctx, src0_clone, src1_clone, src2_clone, n_dims, mode, n_ctx_orig_ggml, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    } else if (tensor->op == GGML_OP_UNARY) {
        switch (ggml_get_unary_op(tensor)) {
        case GGML_UNARY_OP_SILU:
            tensor_clone = ggml_silu(ggml_ctx, src0_clone);
            break;
        case GGML_UNARY_OP_GELU:
            tensor_clone = ggml_gelu(ggml_ctx, src0_clone);
            break;
        case GGML_UNARY_OP_GELU_QUICK:
            tensor_clone = ggml_gelu_quick(ggml_ctx, src0_clone);
            break;
        case GGML_UNARY_OP_RELU:
            tensor_clone = ggml_relu(ggml_ctx, src0_clone);
            break;
        case GGML_UNARY_OP_TANH:
            tensor_clone = ggml_tanh(ggml_ctx, src0_clone);
            break;
        default:
            std::cerr << "Missing vk_check_results OP: " << ggml_op_name(tensor->op) << std::endl;
            GGML_ABORT("fatal error");
        }
    } else if (tensor->op == GGML_OP_CPY || tensor->op == GGML_OP_DUP) {
        if (src1 == nullptr) {
            tensor_clone = ggml_dup(ggml_ctx, src0_clone);
            tensor_clone->type = tensor->type;
        } else {
            tensor_clone = ggml_cpy(ggml_ctx, src0_clone, src1_clone);
        }
    } else if (tensor->op == GGML_OP_CONT) {
        tensor_clone = ggml_cont_4d(ggml_ctx, src0_clone, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    } else if (tensor->op == GGML_OP_RESHAPE) {
        tensor_clone = ggml_reshape_4d(ggml_ctx, src0_clone, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    } else if (tensor->op == GGML_OP_VIEW) {
        tensor_clone = ggml_view_4d(ggml_ctx, src0_clone, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], tensor->nb[1], tensor->nb[2], tensor->nb[3], ((int32_t *) tensor->op_params)[0]);
    } else if (tensor->op == GGML_OP_PERMUTE) {
        int32_t * params = (int32_t *)tensor->op_params;
        tensor_clone = ggml_permute(ggml_ctx, src0_clone, params[0], params[1], params[2], params[3]);
    } else if (tensor->op == GGML_OP_TRANSPOSE) {
        tensor_clone = ggml_transpose(ggml_ctx, src0_clone);
    } else if (tensor->op == GGML_OP_GET_ROWS) {
        tensor_clone = ggml_get_rows(ggml_ctx, src0_clone, src1_clone);
    } else if (tensor->op == GGML_OP_ARGSORT) {
        tensor_clone = ggml_argsort(ggml_ctx, src0_clone, (ggml_sort_order) *(int *)tensor->op_params);
    } else if (tensor->op == GGML_OP_SUM_ROWS) {
        tensor_clone = ggml_sum_rows(ggml_ctx, src0_clone);
    } else if (tensor->op == GGML_OP_IM2COL) {
        const int32_t s0 = tensor->op_params[0];
        const int32_t s1 = tensor->op_params[1];
        const int32_t p0 = tensor->op_params[2];
        const int32_t p1 = tensor->op_params[3];
        const int32_t d0 = tensor->op_params[4];
        const int32_t d1 = tensor->op_params[5];

        const bool is_2D = tensor->op_params[6] == 1;
        tensor_clone = ggml_im2col(ggml_ctx, src0_clone, src1_clone, s0, s1, p0, p1, d0, d1, is_2D, tensor->type);
    } else if (tensor->op == GGML_OP_TIMESTEP_EMBEDDING) {
        const int32_t dim = tensor->op_params[0];
        const int32_t max_period = tensor->op_params[1];
        tensor_clone = ggml_timestep_embedding(ggml_ctx, src0_clone, dim, max_period);
    } else if (tensor->op == GGML_OP_LEAKY_RELU) {
        const float * op_params = (const float *)tensor->op_params;
        tensor_clone = ggml_leaky_relu(ggml_ctx, src0_clone, op_params[0], false);
    } else {
        std::cerr << "Missing vk_check_results OP: " << ggml_op_name(tensor->op) << std::endl;
        GGML_ABORT("fatal error");
    }

    ggml_cgraph * cgraph = ggml_new_graph(ggml_ctx);
    ggml_build_forward_expand(cgraph, tensor_clone);

    ggml_graph_compute_with_ctx(ggml_ctx, cgraph, 8);

    if (vk_output_tensor > 0 && vk_output_tensor == check_counter) {
        ggml_vk_print_tensor(tensor_clone, "tensor_clone");
    }

    comp_size = ggml_nbytes(tensor_clone);

    comp_result = malloc(comp_size);
    memcpy(comp_result, tensor_clone->data, comp_size);
    memcpy(comp_nb, tensor_clone->nb, sizeof(size_t) * GGML_MAX_DIMS);

    if (src0 != nullptr) {
        free(src0_buffer);
    }
    if (src1 != nullptr) {
        free(src1_buffer);
    }

    ggml_free(ggml_ctx);

    VK_LOG_DEBUG("END ggml_vk_check_results_0(" << tensor->name << ")");
}

static void ggml_vk_check_results_1(ggml_tensor * tensor) {
    if (tensor->op == GGML_OP_TRANSPOSE) {
        return;
    }
    if (!(vk_output_tensor > 0 && vk_output_tensor == check_counter) && check_counter <= vk_skip_checks) {
        return;
    }

    VK_LOG_DEBUG("ggml_vk_check_results_1(" << tensor->name << ")");

    ggml_tensor * src0 = tensor->src[0];
    ggml_tensor * src1 = tensor->src[1];
    ggml_tensor * src2 = tensor->src[2];

    void * tensor_data = tensor->data;

    if (ggml_backend_buffer_is_vk(tensor->buffer)) {
        size_t tensor_size = ggml_nbytes(tensor);
        tensor_data = malloc(tensor_size);

        ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) tensor->extra;

        vk_buffer buffer_gpu = extra->buffer_gpu.lock();
        if (extra->offset + tensor->view_offs + tensor_size >= buffer_gpu->size) {
            tensor_size = buffer_gpu->size - (extra->offset + tensor->view_offs);
        }

        ggml_vk_buffer_read(buffer_gpu, extra->offset + tensor->view_offs, tensor_data, tensor_size);
    }

    float first_error_result = -1.0f;
    float first_error_correct = -1.0f;
    std::array<int, 4> first_error = { -1, -1, -1, -1 };
    double avg_err = 0.0;
    size_t counter = 0;

    for (int i3 = 0; i3 < tensor->ne[3]; i3++) {
        for (int i2 = 0; i2 < tensor->ne[2]; i2++) {
            for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                    const bool buffer_size_fit = i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0] < comp_size;
                    float correct = 0.0f;
                    float result = 0.0f;

                    if (buffer_size_fit) {
                        if (tensor->type == GGML_TYPE_F32) {
                            correct = *(float *) ((char *) comp_result + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]);
                            result  = *(float *) ((char *) tensor_data + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]);
                        } else if (tensor->type == GGML_TYPE_F16) {
                            correct = ggml_fp16_to_fp32(*(ggml_fp16_t *) ((char *) comp_result + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]));
                            result  = ggml_fp16_to_fp32(*(ggml_fp16_t *) ((char *) tensor_data + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]));
                        } else if (tensor->type == GGML_TYPE_I32) {
                            correct = *(int32_t *) ((char *) comp_result + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]);
                            result  = *(int32_t *) ((char *) tensor_data + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]);
                        } else {
                            std::cerr << "Results check not implemented for type " << ggml_type_name(tensor->type) << std::endl;
                        }
                    } else {
                        std::cerr << "Missing debug code for type " << ggml_type_name(tensor->type) << std::endl;
                        GGML_ABORT("fatal error");
                    }

                    if ((std::isnan(correct) != std::isnan(result)) || (std::isinf(correct) != std::isinf(result)) || !buffer_size_fit) {
                        std::cerr << "ERROR: Invalid value in " << ggml_op_name(tensor->op) << " i3=" << i3 << " i2=" << i2 << " i1=" << i1 << " i0=" << i0 << " result=" << result << " correct=" << correct << " avg_err=" << (avg_err / counter) << std::endl;
                        std::cerr << "tensor=" << tensor << " tensor->name=" << tensor->name << " tensor->type: " << ggml_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] << " nb3=" << tensor->nb[3] << " offset=" << tensor->view_offs << std::endl;
                        if (src0 != nullptr) {
                            std::cerr << "src0=" << src0 << " src0->name=" << src0->name << " op=" << ggml_op_name(src0->op) << " type=" << ggml_type_name(src0->type) << " ne0=" << src0->ne[0] << " nb0=" << src0->nb[0] << " ne1=" << src0->ne[1] << " nb1=" << src0->nb[1] << " ne2=" << src0->ne[2] << " nb2=" << src0->nb[2] << " ne3=" << src0->ne[3] << " nb3=" << src0->nb[3] << " offset=" << src0->view_offs << std::endl;
                        }
                        if (src1 != nullptr) {
                            std::cerr << "src1=" << src1 << " src1->name=" << src1->name << " op=" << ggml_op_name(src1->op) << " type=" << ggml_type_name(src1->type) << " ne0=" << src1->ne[0] << " nb0=" << src1->nb[0] << " ne1=" << src1->ne[1] << " nb1=" << src1->nb[1] << " ne2=" << src1->ne[2] << " nb2=" << src1->nb[2] << " ne3=" << src1->ne[3] << " nb3=" << src1->nb[3] << " offset=" << src1->view_offs << std::endl;
                        }
                        if (src2 != nullptr) {
                            std::cerr << "src2=" << src2 << " src2->name=" << src2->name << " op=" << ggml_op_name(src2->op) << " type=" << ggml_type_name(src2->type) << " ne0=" << src2->ne[0] << " nb0=" << src2->nb[0] << " ne1=" << src2->ne[1] << " nb1=" << src2->nb[1] << " ne2=" << src2->ne[2] << " nb2=" << src2->nb[2] << " ne3=" << src2->ne[3] << " nb3=" << src2->nb[3] << " offset=" << src2->view_offs << std::endl;
                        }
                        std::cerr << "First error: result=" << first_error_result << " correct=" << first_error_correct  << " i3=" << first_error[3] << " i2=" << first_error[2] << " i1=" << first_error[1] << " i0=" << first_error[0] << std::endl;
                        std::cerr << std::endl << "Result:" << std::endl;
                        ggml_vk_print_tensor_area(tensor, tensor_data, i0, i1, i2, i3);
                        std::cerr << std::endl << "Correct:" << std::endl;
                        ggml_vk_print_tensor_area(tensor, comp_result, i0, i1, i2, i3);
                        std::cerr << std::endl;
                        std::vector<const ggml_tensor *> done;
                        ggml_vk_print_graph_origin(tensor, done);
                        GGML_ABORT("fatal error");
                    }
                    if (first_error[0] == -1 && std::fabs(correct - result) > 0.1f) {
                        first_error[0] = i0;
                        first_error[1] = i1;
                        first_error[2] = i2;
                        first_error[3] = i3;
                        first_error_result = result;
                        first_error_correct = correct;
                    }

                    // Special case, value is infinite, avoid NaN result in avg_err
                    // NaN also appears in results, if both are nan error is 0
                    if (!std::isinf(correct) && !std::isinf(result) && !std::isnan(correct) && !std::isnan(result)) {
                        avg_err += std::fabs(correct - result);
                    }
                    counter++;
                }
            }
        }
    }

    avg_err /= counter;

    if (vk_output_tensor > 0 && vk_output_tensor == check_counter) {
        std::cerr << "TENSOR CHECK: avg_err=" << avg_err << " in " << ggml_op_name(tensor->op) << " (check " << check_counter << ")" << std::endl;
        std::cerr << "tensor=" << tensor << " tensor->name=" << tensor->name << " tensor->type: " << ggml_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] << " nb3=" << tensor->nb[3] << " offset=" << tensor->view_offs << std::endl;
        if (src0 != nullptr) {
            std::cerr << "src0=" << src0 << " op=" << ggml_op_name(src0->op) << " type=" << ggml_type_name(src0->type) << " ne0=" << src0->ne[0] << " nb0=" << src0->nb[0] << " ne1=" << src0->ne[1] << " nb1=" << src0->nb[1] << " ne2=" << src0->ne[2] << " nb2=" << src0->nb[2] << " ne3=" << src0->ne[3] << " nb3=" << src0->nb[3] << " offset=" << src0->view_offs << std::endl;
        }
        if (src1 != nullptr) {
            std::cerr << "src1=" << src1 << " op=" << ggml_op_name(src1->op) << " type=" << ggml_type_name(src1->type) << " ne0=" << src1->ne[0] << " nb0=" << src1->nb[0] << " ne1=" << src1->ne[1] << " nb1=" << src1->nb[1] << " ne2=" << src1->ne[2] << " nb2=" << src1->nb[2] << " ne3=" << src1->ne[3] << " nb3=" << src1->nb[3] << " offset=" << src1->view_offs << std::endl;
        }
        if (src2 != nullptr) {
            std::cerr << "src2=" << src2 << " op=" << ggml_op_name(src2->op) << " type=" << ggml_type_name(src2->type) << " ne0=" << src2->ne[0] << " nb0=" << src2->nb[0] << " ne1=" << src2->ne[1] << " nb1=" << src2->nb[1] << " ne2=" << src2->ne[2] << " nb2=" << src2->nb[2] << " ne3=" << src2->ne[3] << " nb3=" << src2->nb[3] << " offset=" << src2->view_offs << std::endl;
        }
        std::cerr << "First error: result=" << first_error_result << " correct=" << first_error_correct  << " i3=" << first_error[3] << " i2=" << first_error[2] << " i1=" << first_error[1] << " i0=" << first_error[0] << std::endl;
        std::cerr << std::endl << "Result:" << std::endl;
        ggml_vk_print_tensor_area(tensor, tensor_data, 5, 5, 0, 0);
        std::cerr << std::endl << "Correct:" << std::endl;
        ggml_vk_print_tensor_area(tensor, comp_result, 5, 5, 0, 0);
        std::cerr << std::endl;
        std::vector<const ggml_tensor *> done;
        ggml_vk_print_graph_origin(tensor, done);
    }

    if (avg_err > 0.05 || std::isnan(avg_err)) {
        std::cerr << "ERROR: avg_err=" << avg_err << " in " << ggml_op_name(tensor->op) << " (check " << check_counter << ")" << std::endl;
        std::cerr << "tensor=" << tensor << " tensor->name=" << tensor->name << " tensor->type: " << ggml_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] << " nb3=" << tensor->nb[3] << " offset=" << tensor->view_offs << std::endl;
        if (src0 != nullptr) {
            std::cerr << "src0=" << src0 << " op=" << ggml_op_name(src0->op) << " type=" << ggml_type_name(src0->type) << " ne0=" << src0->ne[0] << " nb0=" << src0->nb[0] << " ne1=" << src0->ne[1] << " nb1=" << src0->nb[1] << " ne2=" << src0->ne[2] << " nb2=" << src0->nb[2] << " ne3=" << src0->ne[3] << " nb3=" << src0->nb[3] << " offset=" << src0->view_offs << std::endl;
        }
        if (src1 != nullptr) {
            std::cerr << "src1=" << src1 << " op=" << ggml_op_name(src1->op) << " type=" << ggml_type_name(src1->type) << " ne0=" << src1->ne[0] << " nb0=" << src1->nb[0] << " ne1=" << src1->ne[1] << " nb1=" << src1->nb[1] << " ne2=" << src1->ne[2] << " nb2=" << src1->nb[2] << " ne3=" << src1->ne[3] << " nb3=" << src1->nb[3] << " offset=" << src1->view_offs << std::endl;
        }
        if (src2 != nullptr) {
            std::cerr << "src2=" << src2 << " op=" << ggml_op_name(src2->op) << " type=" << ggml_type_name(src2->type) << " ne0=" << src2->ne[0] << " nb0=" << src2->nb[0] << " ne1=" << src2->ne[1] << " nb1=" << src2->nb[1] << " ne2=" << src2->ne[2] << " nb2=" << src2->nb[2] << " ne3=" << src2->ne[3] << " nb3=" << src2->nb[3] << " offset=" << src2->view_offs << std::endl;
        }
        std::cerr << "First error: result=" << first_error_result << " correct=" << first_error_correct  << " i3=" << first_error[3] << " i2=" << first_error[2] << " i1=" << first_error[1] << " i0=" << first_error[0] << std::endl;
        std::cerr << std::endl << "Result:" << std::endl;
        ggml_vk_print_tensor_area(tensor, tensor_data, first_error[0], first_error[1], first_error[2], first_error[3]);
        std::cerr << std::endl << "Correct:" << std::endl;
        ggml_vk_print_tensor_area(tensor, comp_result, first_error[0], first_error[1], first_error[2], first_error[3]);
        std::cerr << std::endl;
        std::vector<const ggml_tensor *> done;
        ggml_vk_print_graph_origin(tensor, done);
        GGML_ABORT("fatal error");
    } else {
        std::cerr << check_counter << " " << tensor->name << " op=" << ggml_op_name(tensor->op) << " avg_err=" << avg_err << std::endl;
    }

    free(comp_result);
    comp_result = nullptr;
    comp_size = 0;

    if (ggml_backend_buffer_is_vk(tensor->buffer)) {
        free(tensor_data);
    }

    VK_LOG_DEBUG("END ggml_vk_check_results_1(" << tensor->name << ")");
}
#endif
